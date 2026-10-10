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

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eidola_engine::engine::{
    Engine, FinishReason, Request, RequestId, SchedulerConfig, SubmitError,
};
use eidola_engine::executor::Executor;
use tokio::sync::{mpsc, oneshot};

/// Events buffered per request before a reader that stopped reading is cut off.
pub const EVENT_BUFFER: usize = 256;

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
    /// Requests in the engine now (waiting or running).
    pub in_engine: u64,
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
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
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

/// Starts the engine thread over the executor `make_executor` builds (on that thread).
/// Returns once the engine is constructed, or with its refusal.
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
    let (commands, inbox) = std_mpsc::channel();
    let stats = Arc::new(Mutex::new(Stats::default()));
    let healthy = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
    let thread_stats = stats.clone();
    let exit = ExitSignal {
        healthy: healthy.clone(),
        stopped: Some(stopped),
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
                live: HashMap::new(),
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

/// Marks the engine stopped when dropped: health turns unhealthy and the `stopped`
/// receiver resolves.
struct ExitSignal {
    healthy: Arc<AtomicBool>,
    stopped: Option<oneshot::Sender<()>>,
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
}

struct Worker<E: Executor> {
    engine: Engine<E>,
    inbox: std_mpsc::Receiver<Command>,
    live: HashMap<RequestId, Live>,
    stats: Arc<Mutex<Stats>>,
    counters: Stats,
    started: Instant,
    sweep_interval: Duration,
}

impl<E: Executor> Worker<E> {
    fn now(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
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
                    if self.engine.sweep(self.now()).is_err() {
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
                match self.engine.step(now) {
                    Ok(events) => {
                        for ev in events {
                            self.deliver(ev, now);
                        }
                    }
                    Err(_) => return self.fail(),
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
                match self.engine.submit(request) {
                    Ok(()) => {
                        self.counters.submitted += 1;
                        let _ = events.try_send(Output::Accepted);
                        self.live.insert(
                            id,
                            Live {
                                events,
                                _permit: permit,
                                reported_cached: false,
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
                if self.live.remove(&id).is_some() {
                    let now = self.now();
                    self.engine.cancel(id, now);
                    self.counters.cancelled += 1;
                }
            }
        }
    }

    fn deliver(&mut self, ev: eidola_engine::engine::Event, now: u64) {
        let Some(live) = self.live.get_mut(&ev.id) else {
            return;
        };
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
            self.live.remove(&ev.id);
            self.counters.finished += 1;
        } else if let Err(e) = sent {
            // The reader is gone (its guard's cancel may still be in flight), or has
            // stopped reading: cut it off, which also closes its channel.
            if matches!(e, mpsc::error::TrySendError::Full(_)) {
                self.counters.cut_off += 1;
            }
            self.live.remove(&ev.id);
            self.engine.cancel(ev.id, now);
            self.counters.cancelled += 1;
        }
    }

    fn fail(mut self) {
        tracing::error!("executor failure; the engine has stopped");
        for (_, live) in self.live.drain() {
            let _ = live.events.try_send(Output::Failed);
        }
        self.publish();
    }

    fn publish(&mut self) {
        let s = self.engine.stats();
        self.counters.in_engine = self.engine.unfinished() as u64;
        self.counters.engine_steps = s.steps;
        self.counters.preemptions = s.preemptions;
        self.counters.drafted = s.drafted;
        self.counters.accepted_drafts = s.accepted;
        *self.stats.lock().expect("stats lock") = self.counters;
    }
}
