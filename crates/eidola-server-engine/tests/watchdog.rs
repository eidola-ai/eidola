//! A step that never returns is fatal: the watchdog ends the in-flight stream with an
//! error, turns health unhealthy and runs its exit hook, without the stuck engine
//! thread. A slow step under the limit, and an idle engine, never trip it.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc as std_mpsc};
use std::time::{Duration, Instant};

use eidola_engine::engine::{CacheScope, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::mock::{MockConfig, MockExecutor, mimo_like_spec};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::SaltDeriver;
use eidola_engine::spec::ModelSpec;
use eidola_server_engine::auth::GatewayToken;
use eidola_server_engine::http::{self, AppState};
use eidola_server_engine::worker::{self, Admission, EngineHandle, Output, Watchdog};

/// Holds a step until the test releases it (or a minute passes, so a failing test
/// cannot leave the thread behind for good).
#[derive(Default)]
struct Gate {
    released: Mutex<bool>,
    opened: Condvar,
}

impl Gate {
    fn wait(&self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut released = self.released.lock().unwrap();
        while !*released && Instant::now() < deadline {
            released = self
                .opened
                .wait_timeout(released, Duration::from_millis(100))
                .unwrap()
                .0;
        }
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.opened.notify_all();
    }
}

/// The mock executor, with every step that runs a row first held at the gate (when
/// there is one) or slowed by `delay`.
struct Held {
    inner: MockExecutor,
    gate: Option<Arc<Gate>>,
    delay: Duration,
    steps: Arc<AtomicU64>,
}

impl Executor for Held {
    fn spec(&self) -> &ModelSpec {
        self.inner.spec()
    }
    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        if !step.seqs.is_empty() {
            if let Some(gate) = &self.gate {
                gate.wait();
            }
            std::thread::sleep(self.delay);
        }
        self.steps.fetch_add(1, Ordering::SeqCst);
        self.inner.execute(step)
    }
}

/// A watchdog whose exit hook reports to the returned receiver instead of exiting.
fn watchdog(limit: Duration) -> (Watchdog, std_mpsc::Receiver<Instant>) {
    let (tx, rx) = std_mpsc::channel();
    (
        Watchdog {
            limit,
            grace: Duration::from_millis(100),
            on_stall: Box::new(move || {
                let _ = tx.send(Instant::now());
            }),
        },
        rx,
    )
}

fn engine(
    vocab: u32,
    gate: Option<Arc<Gate>>,
    delay: Duration,
    sweep: Duration,
    watchdog: Watchdog,
) -> (EngineHandle, Arc<AtomicU64>) {
    let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
    // Nothing here serves: the receiver only has to outlive the test.
    std::mem::forget(stopped_rx);
    let steps = Arc::new(AtomicU64::new(0));
    let executor_steps = steps.clone();
    let spec = mimo_like_spec(vocab, 16, 128, 128, 256, 4, 0);
    let handle = worker::spawn_watched(
        move || {
            Ok(Held {
                inner: MockExecutor::new(spec, MockConfig::default()),
                gate,
                delay,
                steps: executor_steps,
            })
        },
        SchedulerConfig {
            eos_token_ids: Vec::new(),
            speculative: false,
            sweep_interval_ms: sweep.as_millis() as u64,
            ..SchedulerConfig::default()
        },
        sweep,
        stopped_tx,
        watchdog,
    )
    .unwrap();
    (handle, steps)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_step_ends_the_stream_with_an_error_and_exits() {
    let limit = Duration::from_millis(400);
    let (watchdog, exited) = watchdog(limit);
    let gate = Arc::new(Gate::default());
    let model = common::loaded();
    let vocab = model.tokenizer().vocab_size() as u32;
    let (handle, _) = engine(
        vocab,
        Some(gate.clone()),
        Duration::ZERO,
        Duration::from_secs(1),
        watchdog,
    );
    let admission = Admission::new(4);
    let state = Arc::new(AppState {
        model_id: common::MODEL_ID.into(),
        executor: "cpu",
        device: None,
        max_model_len: 1024,
        model,
        token: GatewayToken::verify(common::TOKEN.into(), common::token_hash()).unwrap(),
        salts: SaltDeriver::new(),
        engine: handle.clone(),
        admission: admission.clone(),
        reading: Admission::new(4),
    });
    let node = common::TestNode::serve(eidola_server_engine::Node {
        router: http::router(state),
        engine: handle.clone(),
        admission,
        weights_hash: common::weights_hash().into(),
        engine_stopped: tokio::sync::oneshot::channel().1,
    })
    .await;

    let mut body = common::request("hello", 8);
    body["stream"] = true.into();
    let start = Instant::now();
    let events = tokio::time::timeout(Duration::from_secs(20), node.chat_stream(&body))
        .await
        .expect("the stalled stream ended");
    let took = start.elapsed();
    let last = events.last().expect("the stream sent its role chunk");
    assert!(
        last.starts_with("error: "),
        "it ended with an error: {events:?}"
    );
    assert!(
        !events.iter().any(|e| e == "[DONE]"),
        "and not as a finished answer"
    );
    assert!(took >= limit, "the error came after the limit ({took:?})");
    assert!(
        took < limit + Duration::from_secs(3),
        "the error came {took:?} after the request"
    );

    assert!(!handle.is_healthy());
    let health = node
        .client
        .get(format!("{}/healthz", node.base))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status().as_u16(), 503);
    let exit = exited
        .recv_timeout(Duration::from_secs(5))
        .expect("the exit hook ran");
    assert!(exit.duration_since(start) >= limit);
    // A request after the trip is refused rather than queued behind the stuck step.
    let (status, _) = tokio::time::timeout(
        Duration::from_secs(5),
        node.chat_json(&common::request("again", 4)),
    )
    .await
    .expect("refused at once");
    assert_eq!(status, 503);

    gate.release();
    assert!(
        exited.recv_timeout(Duration::from_millis(500)).is_err(),
        "the hook runs once"
    );
}

/// Runs one request through `handle` to its end; returns whether it finished.
fn run_one(handle: &EngineHandle, max_tokens: u32) -> bool {
    let admission = Admission::new(1);
    let (_guard, mut events) = handle
        .submit(
            Request {
                id: handle.next_id(),
                prompt: vec![1, 2, 3],
                sampling: SamplingParams::greedy(),
                max_tokens,
                required_cached_tokens: 0,
                stop_token_ids: Vec::new(),
                cache: CacheScope::Private,
            },
            admission.try_acquire().unwrap(),
        )
        .unwrap();
    while let Some(out) = events.blocking_recv() {
        match out {
            Output::Tokens {
                finish: Some(_), ..
            } => return true,
            Output::Failed | Output::Rejected(_) => return false,
            _ => {}
        }
    }
    false
}

/// Every step under the limit, the request as a whole far past it: the limit is per call.
#[test]
fn slow_steps_under_the_limit_do_not_trip_it() {
    let limit = Duration::from_millis(500);
    let (watchdog, exited) = watchdog(limit);
    let (handle, steps) = engine(
        32,
        None,
        Duration::from_millis(150),
        Duration::from_secs(1),
        watchdog,
    );
    let start = Instant::now();
    assert!(run_one(&handle, 8), "the request finished");
    assert!(steps.load(Ordering::SeqCst) >= 8);
    assert!(start.elapsed() > 2 * limit, "it ran past the limit in all");
    assert!(handle.is_healthy());
    assert!(
        exited.recv_timeout(limit * 2).is_err(),
        "the watchdog stayed quiet"
    );
}

/// An idle engine trips nothing however long it waits: the limit covers a call while it
/// runs, not the time since one. Its sweeps (each a call) come at three times the limit.
#[test]
fn an_idle_engine_does_not_trip_it() {
    let limit = Duration::from_millis(100);
    let (watchdog, exited) = watchdog(limit);
    let (handle, _) = engine(
        32,
        None,
        Duration::ZERO,
        Duration::from_millis(300),
        watchdog,
    );
    assert!(run_one(&handle, 2));
    assert!(
        exited.recv_timeout(Duration::from_secs(1)).is_err(),
        "the watchdog stayed quiet while idle"
    );
    assert!(handle.is_healthy());
    assert!(run_one(&handle, 2), "and the engine still serves");
}

const CHILD_ENV: &str = "EIDOLA_WATCHDOG_TEST_CHILD";
/// Set for a child that logs to standard output and holds it locked throughout.
const CHILD_STDOUT_HELD_ENV: &str = "EIDOLA_WATCHDOG_TEST_CHILD_STDOUT_HELD";

/// The child half of the test below (a no-op unless run as that child): an engine whose
/// first step never returns, under a short limit and the node's own exit.
#[test]
fn child_with_a_wedged_engine() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    if std::env::var_os(CHILD_STDOUT_HELD_ENV).is_some() {
        // The node's own logging, then standard output locked for good by a thread
        // that never lets go: nothing can be written to it again.
        tracing_subscriber::fmt().init();
        let (held_tx, held_rx) = std_mpsc::channel();
        std::thread::spawn(move || {
            let _lock = std::io::stdout().lock();
            held_tx.send(()).unwrap();
            loop {
                std::thread::park();
            }
        });
        held_rx.recv().unwrap();
    }
    let watchdog = Watchdog {
        limit: Duration::from_millis(300),
        grace: Duration::from_millis(50),
        on_stall: Box::new(|| worker::exit_stalled()),
    };
    let (handle, _) = engine(
        32,
        Some(Arc::new(Gate::default())),
        Duration::ZERO,
        Duration::from_secs(1),
        watchdog,
    );
    assert!(!run_one(&handle, 4), "the request failed");
    // The watchdog ends the process after its grace; outliving this is the failure.
    std::thread::sleep(Duration::from_secs(30));
    std::process::exit(0);
}

#[test]
fn a_wedged_engine_ends_the_process_with_the_stall_status() {
    assert_eq!(wedged_child_status(false), Some(worker::STALL_EXIT_STATUS));
}

/// The exit waits on no output: a node whose standard output can never be written to
/// again (its log goes there) still ends, with the same status.
#[test]
fn a_wedged_engine_ends_the_process_though_its_log_cannot_be_written() {
    assert_eq!(wedged_child_status(true), Some(worker::STALL_EXIT_STATUS));
}

/// Runs the child above and returns its exit status.
fn wedged_child_status(stdout_held: bool) -> Option<i32> {
    use std::process::{Command, Stdio};
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "child_with_a_wedged_engine",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if stdout_held {
        command.env(CHILD_STDOUT_HELD_ENV, "1");
    }
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(20) {
            child.kill().unwrap();
            panic!("the wedged child kept running");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    status.code()
}
