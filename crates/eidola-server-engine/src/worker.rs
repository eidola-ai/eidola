//! The engine thread and the bridge to it.
//!
//! The executor seam is synchronous (one step in, one result out), so one dedicated OS
//! thread owns the [`Engine`] and runs its step loop; nothing else touches it. HTTP tasks
//! talk to it through channels:
//!
//! * **Commands** in (`Submit`, `Cancel`) on an unbounded channel. Between steps the
//!   thread drains every pending command, so a submission waits at most one step.
//! * **Events** out, one unbounded channel per request. The thread never blocks on a
//!   slow reader: a request's output is bounded by its `max_tokens`, and a reader that
//!   has gone away is noticed on the next send, which cancels the request in the engine.
//!
//! **Admission is bounded** by [`Admission`]: an HTTP task takes a permit before it
//! renders or tokenizes anything, and the permit travels with the submission and is
//! released when the request leaves the engine (finished, cancelled or refused). With
//! every permit taken a request is refused at once with `overloaded`; nothing queues
//! without bound in front of the engine.
//!
//! **Cancellation**: dropping a request's [`RequestGuard`] (the HTTP response future or
//! stream going away, for example on a client disconnect) sends `Cancel`, and the engine
//! releases the sequence before its next step.
//!
//! When idle the thread sleeps until a command arrives or the prefix-cache sweep is due,
//! and sweeps, so expired KV is zeroed without waiting for traffic. An executor failure
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
    /// Requests cancelled (disconnects and stop sequences).
    pub cancelled: u64,
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
        events: mpsc::UnboundedSender<Output>,
        permit: Permit,
    },
    Cancel(RequestId),
}

/// The bound on requests in the engine (running or queued).
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
    pub fn submit(
        &self,
        request: Request,
        permit: Permit,
    ) -> Result<(RequestGuard, mpsc::UnboundedReceiver<Output>), Permit> {
        let id = request.id;
        let (tx, rx) = mpsc::unbounded_channel();
        match self.commands.send(Command::Submit {
            request,
            events: tx,
            permit,
        }) {
            Ok(()) => Ok((
                RequestGuard {
                    id,
                    commands: self.commands.clone(),
                    done: false,
                },
                rx,
            )),
            Err(std_mpsc::SendError(Command::Submit { permit, .. })) => Err(permit),
            Err(_) => unreachable!("sent a Submit"),
        }
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
    on_fatal: oneshot::Sender<()>,
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
    let thread_healthy = healthy.clone();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let engine = make_executor()
                .and_then(|exec| Engine::new(exec, scheduler).map_err(|e| e.to_string()));
            let engine = match engine {
                Ok(engine) => {
                    thread_healthy.store(true, Ordering::Release);
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
            thread_healthy.store(false, Ordering::Release);
            let _ = on_fatal.send(());
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

struct Live {
    events: mpsc::UnboundedSender<Output>,
    _permit: Permit,
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
        loop {
            // Idle: wait for work, sweeping the prefix cache on schedule.
            if self.engine.unfinished() == 0 {
                match self.inbox.recv_timeout(self.sweep_interval) {
                    Ok(cmd) => self.handle(cmd),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {
                        if self.engine.sweep(self.now()).is_err() {
                            return self.fail();
                        }
                        self.publish();
                        continue;
                    }
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
                        let _ = events.send(Output::Accepted);
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
                        let _ = events.send(Output::Rejected(e));
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
        let sent = live.events.send(Output::Tokens {
            tokens: ev.tokens,
            cached_prompt_tokens: cached,
            finish: ev.finish,
        });
        if ev.finish.is_some() {
            self.live.remove(&ev.id);
            self.counters.finished += 1;
        } else if sent.is_err() {
            // The reader is gone (its guard's cancel may still be in flight).
            self.live.remove(&ev.id);
            self.engine.cancel(ev.id, now);
            self.counters.cancelled += 1;
        }
    }

    fn fail(mut self) {
        tracing::error!("executor failure; the engine has stopped");
        for (_, live) in self.live.drain() {
            let _ = live.events.send(Output::Failed);
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
