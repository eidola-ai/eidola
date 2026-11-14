//! The engine thread and the bridge to it.
//!
//! The executor seam is synchronous (one step in, one result out), so one dedicated OS
//! thread owns the [`Engine`] and runs its step loop; nothing else touches it. HTTP tasks
//! talk to it through channels:
//!
//! * **Commands** in (`Submit`, `Cancel`) on an unbounded channel. Between steps the
//!   thread drains every pending command, so a submission waits at most one step.
//! * **Events** out, one bounded channel per request ([`EVENT_BUFFER`] events, each one
//!   step's few tokens). The thread never blocks on a reader: a reader that has gone
//!   away, or that has stopped reading for so long that its buffer is full, is cut off
//!   at the next send (the request is cancelled in the engine and its channel closed,
//!   so the reader sees an error once it drains what was buffered). A stalled consumer
//!   therefore costs at most [`EVENT_BUFFER`] events and stops costing compute. An SSE
//!   stream drains the buffer as fast as its socket accepts data and the kernel's socket
//!   buffers absorb far more than one step, so only a reader that has truly stopped fills
//!   it; a non-streaming response drains it in process.
//!
//! **Admission is bounded** by [`Admission`]: an HTTP task takes a permit before it
//! renders or tokenizes anything. The permit is shared (`Arc`) between the engine's
//! record of the request and the [`RequestGuard`], which lives exactly as long as the
//! response holding the request's receiver, so a slot is released only when the request
//! has left the engine **and** its buffered output has been drained or dropped. With
//! every permit taken a request is refused at once with `overloaded`; nothing queues
//! without bound in front of the engine, and buffered output is bounded by
//! `EIDOLA_ENGINE_MAX_REQUESTS × EVENT_BUFFER` events.
//!
//! **Cancellation**: dropping a request's [`RequestGuard`] (the HTTP response future or
//! stream going away, for example on a client disconnect) sends `Cancel`, and the engine
//! releases the sequence before its next step.
//!
//! When idle the thread sleeps until a command arrives or the prefix-cache sweep is due,
//! and sweeps once it is due, so expired KV is zeroed without waiting for traffic. The
//! sweep keeps an absolute deadline, so idle-leaving commands arriving faster than the
//! interval cannot postpone it (while busy, each engine step sweeps on its own schedule). An executor failure
//! is fatal: every in-flight request gets an error, the health check turns unhealthy, and
//! the thread exits (the process then exits; see `main.rs`).
//!
//! **A step that never returns is fatal too.** The engine thread records when each call
//! that reaches the executor (a step, a sweep, a flush) starts and clears it when the
//! call returns; a watchdog on its own OS thread, which shares nothing with the engine
//! thread but atomics and the map of in-flight requests' channels, checks it. A call
//! running past [`Watchdog::limit`] is a wedged engine, which nothing in Rust can
//! interrupt, so the watchdog does without the engine thread what an executor failure
//! does: health turns unhealthy, every in-flight request gets an error, and then, after
//! [`Watchdog::grace`] for the streams to write it, [`Watchdog::on_stall`] runs (for the
//! node, [`exit_stalled`]). The engine thread never holds that map's lock across an
//! executor call, so the watchdog can take it while the thread is stuck. The exit waits
//! on no output either: the log line saying why is written by a thread of its own, since
//! standard output may be full or held by a stuck thread.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use eidola_engine::engine::{
    Engine, FinishReason, Request, RequestId, SchedulerConfig, SubmitError,
};
use eidola_engine::executor::Executor;
use tokio::sync::{mpsc, oneshot};

/// Events buffered per request before a reader that stopped reading is cut off.
pub const EVENT_BUFFER: usize = 256;

/// How long one call into the engine that reaches the executor (a step, a sweep, a
/// flush) may run before the node treats the engine as wedged and exits.
///
/// The longest legitimate call on a pinned deployment is a step at the step budget
/// (8,192 tokens) of prefill chunks at the end of a context of 2^20 positions, whose
/// global attention costs that many queries over a million keys in each of the nine
/// global layers. Scaled linearly from the 2,048-token prefill after 4,096 positions
/// measured on the B300 (16.78 ms of attention, `eidola-engine-cuda`'s `AGENTS.md`), that
/// is about 14 s, and the rest of such a step (every expert reached, the sampler) well
/// under a second; the floor at the GPU's peak rate is under 2 s. Maintenance moves at
/// most the device pools and the host tier once (tens of GiB at device or host-link
/// bandwidth, a second or two), and the decode graphs are captured at boot, before the
/// engine thread's loop. This is about eight times the slowest of those (the CPU
/// executor, for development, is not what it is sized for), and still a fifteenth of
/// the gateway's own deadline on a request.
pub const STEP_STALL_LIMIT: Duration = Duration::from_secs(120);

/// How long a wedged node waits, after ending its in-flight requests with an error,
/// before it exits: time for each stream's next frame (one every 20 ms) to carry it.
pub const STALL_EXIT_GRACE: Duration = Duration::from_secs(1);

/// The exit status of a node whose engine wedged (`sysexits`' `EX_SOFTWARE`).
pub const STALL_EXIT_STATUS: i32 = 70;

/// What the engine thread's watchdog does: the bound on one executor-reaching call, and
/// what follows a call that outlives it.
pub struct Watchdog {
    /// The longest a call may run ([`STEP_STALL_LIMIT`] for the node).
    pub limit: Duration,
    /// The wait between failing the in-flight requests and `on_stall`
    /// ([`STALL_EXIT_GRACE`] for the node).
    pub grace: Duration,
    /// Runs once, last ([`exit_stalled`] for the node).
    pub on_stall: Box<dyn FnOnce() + Send>,
}

impl Watchdog {
    /// The node's own.
    pub fn process() -> Self {
        Watchdog {
            limit: STEP_STALL_LIMIT,
            grace: STALL_EXIT_GRACE,
            on_stall: Box::new(|| exit_stalled()),
        }
    }
}

/// Ends the process at once with [`STALL_EXIT_STATUS`]. `_exit`, not `exit`: the latter
/// runs `atexit` handlers and library destructors (the GPU driver's among them), which
/// may wait on what the stuck thread holds; nothing here waits on that thread.
pub fn exit_stalled() -> ! {
    // SAFETY: `_exit` takes no arguments but the status and does not return; it ends the
    // process without running any Rust or C code after it, so no destructor that soundness
    // depends on is skipped while the process lives on.
    unsafe { libc::_exit(STALL_EXIT_STATUS) }
}

/// The engine thread has stopped; nothing can be submitted.
#[derive(Debug)]
pub struct EngineStopped;

/// What a request's channel carries.
#[derive(Debug)]
pub enum Output {
    /// The engine accepted the request.
    Accepted,
    /// The engine refused it at submission.
    Rejected(SubmitError),
    /// A request with a required cached prefix was admitted: its requirement held, and
    /// nothing more will refuse it. Sent once, after `Accepted` and before its first
    /// tokens; a request whose requirement did not hold gets instead a `Tokens` finishing
    /// with [`FinishReason::CachedPrefixMissing`]. Requests without one never get it.
    Admitted,
    /// Tokens produced by one step, the cached-prompt count once known (with the first
    /// tokens, and again with the finish), and the finish if it finished.
    Tokens {
        tokens: Vec<u32>,
        cached_prompt_tokens: Option<u32>,
        finish: Option<FinishReason>,
    },
    /// The engine failed; the request will produce nothing more.
    Failed,
}

/// Content-free counters (tests and diagnostics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Requests the engine accepted.
    pub submitted: u64,
    /// Requests that finished on their own (EOS or length).
    pub finished: u64,
    /// Requests cancelled (disconnects, stop sequences, and readers cut off).
    pub cancelled: u64,
    /// Of those, readers cut off because they stopped draining their buffer.
    pub cut_off: u64,
    /// Requests the engine refused at submission.
    pub rejected: u64,
    /// Requests the engine refused at first admission for a short prefix-cache hit.
    pub prefix_refused: u64,
    /// Requests in the engine now (waiting or running).
    pub in_engine: u64,
    /// Idle-KV keys the engine holds (a count): zero whenever no request is in the
    /// engine, since the last request's maintenance runs as soon as it ends.
    pub kv_keys_held: u64,
    /// The engine's own counters.
    pub engine_steps: u64,
    pub preemptions: u64,
    pub drafted: u64,
    pub accepted_drafts: u64,
}

enum Command {
    Submit {
        request: Request,
        events: mpsc::Sender<Output>,
        permit: Arc<Permit>,
    },
    Cancel(RequestId),
}

/// A counting bound with non-blocking acquisition: admission (requests being prepared,
/// queued or running in the engine) and, in `http`, the read bound in front of it.
#[derive(Debug)]
pub struct Admission {
    in_flight: AtomicU32,
    limit: u32,
}

impl Admission {
    pub fn new(limit: u32) -> Arc<Self> {
        Arc::new(Admission {
            in_flight: AtomicU32::new(0),
            limit,
        })
    }

    /// A permit, or `None` when every one is taken.
    pub fn try_acquire(self: &Arc<Self>) -> Option<Permit> {
        self.in_flight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.limit).then_some(n + 1)
            })
            .ok()
            .map(|_| Permit(self.clone()))
    }

    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Acquire)
    }
}

/// One admission slot; released on drop.
#[derive(Debug)]
pub struct Permit(Arc<Admission>);

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The HTTP side's handle on the engine thread.
#[derive(Clone)]
pub struct EngineHandle {
    commands: std_mpsc::Sender<Command>,
    next_id: Arc<AtomicU64>,
    stats: Arc<Mutex<Stats>>,
    healthy: Arc<AtomicBool>,
}

impl std::fmt::Debug for EngineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineHandle").finish_non_exhaustive()
    }
}

impl EngineHandle {
    /// A fresh request id.
    pub fn next_id(&self) -> RequestId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Submits a request. The returned guard cancels it when dropped (unless it
    /// finished); the receiver yields [`Output::Accepted`] or [`Output::Rejected`] first.
    ///
    /// The permit is held by both the engine's record and the guard; keep the guard
    /// together with the receiver (the slot is released when both are gone).
    pub fn submit(
        &self,
        request: Request,
        permit: Permit,
    ) -> Result<(RequestGuard, mpsc::Receiver<Output>), EngineStopped> {
        // Unhealthy is stopped, or a wedged engine whose thread cannot take it.
        if !self.is_healthy() {
            return Err(EngineStopped);
        }
        let id = request.id;
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let permit = Arc::new(permit);
        self.commands
            .send(Command::Submit {
                request,
                events: tx,
                permit: permit.clone(),
            })
            .map_err(|_| EngineStopped)?;
        Ok((
            RequestGuard {
                id,
                commands: self.commands.clone(),
                done: false,
                _permit: permit,
            },
            rx,
        ))
    }

    pub fn stats(&self) -> Stats {
        *self.stats.lock().expect("stats lock")
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

/// Cancels its request in the engine when dropped, unless marked done.
#[derive(Debug)]
pub struct RequestGuard {
    id: RequestId,
    commands: std_mpsc::Sender<Command>,
    done: bool,
    _permit: Arc<Permit>,
}

impl RequestGuard {
    /// The request finished in the engine; nothing to cancel.
    pub fn finished(&mut self) {
        self.done = true;
    }

    /// Cancels now (a stop sequence matched).
    pub fn cancel(&mut self) {
        if !self.done {
            self.done = true;
            let _ = self.commands.send(Command::Cancel(self.id));
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Starts the engine thread over the executor `make_executor` builds (on that thread),
/// under the node's watchdog ([`Watchdog::process`]). Returns once the engine is
/// constructed, or with its refusal.
pub fn spawn<E, F>(
    make_executor: F,
    scheduler: SchedulerConfig,
    sweep_interval: Duration,
    stopped: oneshot::Sender<()>,
) -> Result<EngineHandle, String>
where
    E: Executor + 'static,
    F: FnOnce() -> Result<E, String> + Send + 'static,
{
    spawn_watched(
        make_executor,
        scheduler,
        sweep_interval,
        stopped,
        Watchdog::process(),
    )
}

/// [`spawn`] under `watchdog`.
pub fn spawn_watched<E, F>(
    make_executor: F,
    scheduler: SchedulerConfig,
    sweep_interval: Duration,
    stopped: oneshot::Sender<()>,
    watchdog: Watchdog,
) -> Result<EngineHandle, String>
where
    E: Executor + 'static,
    F: FnOnce() -> Result<E, String> + Send + 'static,
{
    let (commands, inbox) = std_mpsc::channel();
    let stats = Arc::new(Mutex::new(Stats::default()));
    let healthy = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
    let thread_stats = stats.clone();
    let shared = Arc::new(Shared {
        busy: Busy::new(),
        live: Mutex::new(HashMap::new()),
    });
    // The engine thread holds the sender for its whole life: the watchdog stops when it
    // disconnects.
    let (alive, engine_gone) = std_mpsc::channel::<()>();
    {
        let shared = shared.clone();
        let healthy = healthy.clone();
        std::thread::Builder::new()
            .name("engine-watchdog".into())
            .spawn(move || watch(&shared, &healthy, watchdog, &engine_gone))
            .map_err(|e| format!("cannot start the engine watchdog: {e}"))?;
    }
    let exit = ExitSignal {
        healthy: healthy.clone(),
        stopped: Some(stopped),
        _alive: alive,
    };
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            // Owned by the thread for its whole life: however the thread ends (a return,
            // an executor error, a panic unwinding through here), dropping it marks the
            // node unhealthy and signals `stopped`.
            let exit = exit;
            let engine = make_executor()
                .and_then(|exec| Engine::new(exec, scheduler).map_err(|e| e.to_string()));
            let engine = match engine {
                Ok(engine) => {
                    exit.healthy.store(true, Ordering::Release);
                    let _ = ready_tx.send(Ok(()));
                    engine
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            Worker {
                engine,
                inbox,
                shared,
                stats: thread_stats,
                counters: Stats::default(),
                started: Instant::now(),
                sweep_interval,
            }
            .run();
            drop(exit);
        })
        .map_err(|e| format!("cannot start the engine thread: {e}"))?;
    ready_rx
        .recv()
        .map_err(|_| "the engine thread exited during startup".to_string())??;
    Ok(EngineHandle {
        commands,
        next_id: Arc::new(AtomicU64::new(1)),
        stats,
        healthy,
    })
}

/// Marks the engine stopped when dropped: health turns unhealthy, the `stopped`
/// receiver resolves, and the watchdog stops.
struct ExitSignal {
    healthy: Arc<AtomicBool>,
    stopped: Option<oneshot::Sender<()>>,
    _alive: std_mpsc::Sender<()>,
}

impl Drop for ExitSignal {
    fn drop(&mut self) {
        self.healthy.store(false, Ordering::Release);
        if let Some(tx) = self.stopped.take() {
            let _ = tx.send(());
        }
    }
}

struct Live {
    events: mpsc::Sender<Output>,
    _permit: Arc<Permit>,
    reported_cached: bool,
    /// A required cached prefix whose admission has not been announced yet.
    awaiting_admission: bool,
}

/// What the engine thread shares with its watchdog.
struct Shared {
    busy: Busy,
    /// The in-flight requests. Never locked across an executor call.
    live: Mutex<HashMap<RequestId, Live>>,
}

fn lock(live: &Mutex<HashMap<RequestId, Live>>) -> MutexGuard<'_, HashMap<RequestId, Live>> {
    live.lock().unwrap_or_else(|p| p.into_inner())
}

/// When the executor-reaching call now running started, if one is.
struct Busy {
    base: Instant,
    /// Nanoseconds since `base`, plus one; 0 while no call runs.
    since: AtomicU64,
    /// The watchdog found the engine wedged.
    tripped: AtomicBool,
}

impl Busy {
    fn new() -> Self {
        Busy {
            base: Instant::now(),
            since: AtomicU64::new(0),
            tripped: AtomicBool::new(false),
        }
    }

    fn enter(&self) -> BusyGuard<'_> {
        let at = self.base.elapsed().as_nanos() as u64 + 1;
        self.since.store(at, Ordering::Release);
        BusyGuard(self)
    }

    fn running_for(&self) -> Option<Duration> {
        let since = self.since.load(Ordering::Acquire);
        (since != 0).then(|| {
            self.base
                .elapsed()
                .saturating_sub(Duration::from_nanos(since - 1))
        })
    }
}

struct BusyGuard<'a>(&'a Busy);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.since.store(0, Ordering::Release);
    }
}

/// The watchdog's thread: wakes a few times per `limit` until the engine thread ends,
/// and acts once if a call outlives `limit`. It waits on nothing the engine thread
/// holds: the in-flight map is taken with `try_lock` for a bounded time, and the exit
/// follows whether or not it was.
fn watch(
    shared: &Shared,
    healthy: &AtomicBool,
    watchdog: Watchdog,
    engine_gone: &std_mpsc::Receiver<()>,
) {
    let tick = (watchdog.limit / 16).clamp(Duration::from_millis(1), Duration::from_secs(1));
    loop {
        match engine_gone.recv_timeout(tick) {
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            _ => return,
        }
        if shared
            .busy
            .running_for()
            .is_some_and(|d| d > watchdog.limit)
        {
            break;
        }
    }
    shared.busy.tripped.store(true, Ordering::Release);
    healthy.store(false, Ordering::Release);
    let failed = fail_in_flight(&shared.live);
    report_stall(watchdog.limit, failed);
    std::thread::sleep(watchdog.grace);
    (watchdog.on_stall)();
}

/// Logs why the node is exiting, from a thread of its own that nothing waits for: the
/// log's writer blocks while standard output is full or locked, and the exit must not.
fn report_stall(limit: Duration, requests_failed: bool) {
    let _ = std::thread::Builder::new()
        .name("engine-stall-report".to_string())
        .spawn(move || {
            use std::io::Write;
            tracing::error!(
                limit_secs = limit.as_secs_f64(),
                requests_failed,
                "an engine step outlived its limit; the engine is wedged and the node is exiting"
            );
            let _ = std::io::stdout().flush();
        });
}

/// Ends every in-flight request with [`Output::Failed`] (its channel closing after it
/// when its buffer is full), as [`Worker::fail`] does. False if the map could not be
/// taken within a second, which the engine thread's lock discipline rules out.
fn fail_in_flight(live: &Mutex<HashMap<RequestId, Live>>) -> bool {
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut map = loop {
        match live.try_lock() {
            Ok(map) => break map,
            Err(TryLockError::Poisoned(p)) => break p.into_inner(),
            Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(TryLockError::WouldBlock) => return false,
        }
    };
    for (_, live) in map.drain() {
        let _ = live.events.try_send(Output::Failed);
    }
    true
}

struct Worker<E: Executor> {
    engine: Engine<E>,
    inbox: std_mpsc::Receiver<Command>,
    shared: Arc<Shared>,
    stats: Arc<Mutex<Stats>>,
    counters: Stats,
    started: Instant,
    sweep_interval: Duration,
}

impl<E: Executor> Worker<E> {
    fn now(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// Runs `f`, a call that reaches the executor, under the watchdog's eye. `None` when
    /// the watchdog has found the engine wedged: it has failed every request already, and
    /// the thread must stop.
    fn watched<T>(&mut self, f: impl FnOnce(&mut Engine<E>) -> T) -> Option<T> {
        let out = {
            let _busy = self.shared.busy.enter();
            f(&mut self.engine)
        };
        (!self.shared.busy.tripped.load(Ordering::Acquire)).then_some(out)
    }

    fn run(mut self) {
        // An absolute deadline, not a fresh timeout per wait: commands that leave the
        // engine idle (refused submissions, cancels of finished requests) can arrive
        // faster than the interval without ever postponing the sweep.
        let mut next_sweep = Instant::now() + self.sweep_interval;
        loop {
            // Idle: wait for work until the sweep is due, and sweep whenever it is.
            if self.engine.unfinished() == 0 {
                let now = Instant::now();
                if now >= next_sweep {
                    let at = self.now();
                    let Some(swept) = self.watched(|engine| engine.sweep(at)) else {
                        return;
                    };
                    if swept.is_err() {
                        return self.fail();
                    }
                    self.publish();
                    next_sweep = now + self.sweep_interval;
                    continue;
                }
                match self.inbox.recv_timeout(next_sweep - now) {
                    Ok(cmd) => self.handle(cmd),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                    // Every handle is gone: the process is shutting down.
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
            while let Ok(cmd) = self.inbox.try_recv() {
                self.handle(cmd);
            }
            self.publish();
            if self.engine.unfinished() > 0 {
                let now = self.now();
                let Some(stepped) = self.watched(|engine| engine.step(now)) else {
                    return;
                };
                match stepped {
                    Ok(events) => {
                        self.announce_admissions();
                        for ev in events {
                            self.deliver(ev, now);
                        }
                    }
                    Err(_) => return self.fail(),
                }
            }
            // The last request just ended (finished, or cancelled while running): run the
            // maintenance it left now rather than at the next sweep, so the blocks it left
            // cached are encrypted, the ones it freed zeroed, and its key dropped while the
            // engine would otherwise sit idle.
            if self.engine.unfinished() == 0 && self.engine.has_pending_maintenance() {
                match self.watched(Engine::flush) {
                    None => return,
                    Some(Err(_)) => return self.fail(),
                    Some(Ok(())) => {}
                }
            }
            self.publish();
        }
    }

    fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Submit {
                request,
                events,
                permit,
            } => {
                let id = request.id;
                let awaiting_admission = request.required_cached_tokens > 0;
                match self.engine.submit(request) {
                    Ok(()) => {
                        self.counters.submitted += 1;
                        let _ = events.try_send(Output::Accepted);
                        lock(&self.shared.live).insert(
                            id,
                            Live {
                                events,
                                _permit: permit,
                                reported_cached: false,
                                awaiting_admission,
                            },
                        );
                    }
                    Err(e) => {
                        self.counters.rejected += 1;
                        let _ = events.try_send(Output::Rejected(e));
                    }
                }
            }
            Command::Cancel(id) => {
                if lock(&self.shared.live).remove(&id).is_some() {
                    let now = self.now();
                    self.engine.cancel(id, now);
                    self.counters.cancelled += 1;
                }
            }
        }
    }

    /// Sends [`Output::Admitted`] to every request with a required cached prefix that the
    /// engine has admitted since the last step. (One admitted and finished within a single
    /// step is announced by [`Worker::deliver`] instead.)
    fn announce_admissions(&mut self) {
        let mut map = lock(&self.shared.live);
        for (&id, live) in map.iter_mut().filter(|(_, l)| l.awaiting_admission) {
            if self.engine.cached_prompt_tokens(id).is_some() {
                live.awaiting_admission = false;
                let _ = live.events.try_send(Output::Admitted);
            }
        }
    }

    fn deliver(&mut self, ev: eidola_engine::engine::Event, now: u64) {
        let mut map = lock(&self.shared.live);
        let Some(live) = map.get_mut(&ev.id) else {
            return;
        };
        let refused = ev.finish == Some(FinishReason::CachedPrefixMissing);
        if live.awaiting_admission && !refused {
            live.awaiting_admission = false;
            let _ = live.events.try_send(Output::Admitted);
        }
        let cached = if ev.finish.is_some() {
            Some(ev.cached_prompt_tokens)
        } else if !live.reported_cached {
            self.engine.cached_prompt_tokens(ev.id)
        } else {
            None
        };
        live.reported_cached |= cached.is_some();
        let sent = live.events.try_send(Output::Tokens {
            tokens: ev.tokens,
            cached_prompt_tokens: cached,
            finish: ev.finish,
        });
        if ev.finish.is_some() {
            // Finished either way; a full buffer just means the reader misses the end
            // (its channel closes now, so it sees an error after draining).
            map.remove(&ev.id);
            if refused {
                self.counters.prefix_refused += 1;
            } else {
                self.counters.finished += 1;
            }
        } else if let Err(e) = sent {
            // The reader is gone (its guard's cancel may still be in flight), or has
            // stopped reading: cut it off, which also closes its channel.
            if matches!(e, mpsc::error::TrySendError::Full(_)) {
                self.counters.cut_off += 1;
            }
            map.remove(&ev.id);
            self.engine.cancel(ev.id, now);
            self.counters.cancelled += 1;
        }
    }

    fn fail(mut self) {
        tracing::error!("executor failure; the engine has stopped");
        for (_, live) in lock(&self.shared.live).drain() {
            let _ = live.events.try_send(Output::Failed);
        }
        self.publish();
    }

    fn publish(&mut self) {
        let s = self.engine.stats();
        self.counters.in_engine = self.engine.unfinished() as u64;
        self.counters.kv_keys_held = self.engine.kv_keys_held() as u64;
        self.counters.engine_steps = s.steps;
        self.counters.preemptions = s.preemptions;
        self.counters.drafted = s.drafted;
        self.counters.accepted_drafts = s.accepted;
        *self.stats.lock().expect("stats lock") = self.counters;
    }
}
