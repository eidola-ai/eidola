//! Any exit of the engine thread is fatal to the node: an injected executor panic turns
//! health unhealthy, fails the in-flight request, and ends serving with an error (which
//! `main` turns into a non-zero exit) promptly, even with a request stalled mid-upload.

use std::time::Duration;

use eidola_engine::engine::{CacheScope, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::mock::{MockConfig, MockExecutor, mimo_like_spec};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::ModelSpec;
use eidola_server_engine::worker::{self, Admission, Output};

/// The mock executor, panicking on its first step.
struct Panicking(MockExecutor);

impl Executor for Panicking {
    fn spec(&self) -> &ModelSpec {
        self.0.spec()
    }
    fn execute(&mut self, _: &StepInput) -> Result<StepOutput, ExecutorError> {
        panic!("injected executor panic");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_engine_thread_panic_stops_the_node() {
    let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
    let spec = mimo_like_spec(32, 4, 6, 10, 64, 8, 3);
    let handle = worker::spawn(
        move || Ok(Panicking(MockExecutor::new(spec, MockConfig::default()))),
        SchedulerConfig::default(),
        Duration::from_secs(1),
        stopped_tx,
    )
    .unwrap();
    assert!(handle.is_healthy());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server = tokio::spawn(eidola_server_engine::serve(
        listener,
        axum::Router::new(),
        stopped_rx,
        std::future::pending(),
    ));

    let admission = Admission::new(1);
    let (guard, mut events) = handle
        .submit(
            Request {
                id: handle.next_id(),
                prompt: vec![1, 2, 3],
                sampling: SamplingParams::greedy(),
                max_tokens: 4,
                stop_token_ids: Vec::new(),
                cache: CacheScope::Private,
            },
            admission.try_acquire().unwrap(),
        )
        .unwrap();
    assert!(matches!(events.recv().await, Some(Output::Accepted)));
    // The panic drops the request's channel: no more output.
    assert!(events.recv().await.is_none());

    let result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("serve ends when the engine stops")
        .unwrap();
    assert!(result.is_err(), "an engine stop is an error exit");
    assert!(!handle.is_healthy());
    // The slot is released once the response side lets go of the request too.
    assert_eq!(admission.in_flight(), 1);
    drop((guard, events));
    assert_eq!(
        admission.in_flight(),
        0,
        "the request's permit was released"
    );
}

const CHILD_ENV: &str = "EIDOLA_ENGINE_EXIT_TEST_CHILD";

fn request(id: u64) -> Request {
    Request {
        id,
        prompt: vec![1, 2, 3],
        sampling: SamplingParams::greedy(),
        max_tokens: 4,
        stop_token_ids: Vec::new(),
        cache: CacheScope::Private,
    }
}

/// The child half of the test below (a no-op unless run as that child): a node whose
/// engine panics on its first step, which is submitted when a line arrives on stdin.
#[test]
fn child_node_with_a_panicking_engine() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
    let spec = mimo_like_spec(32, 4, 6, 10, 64, 8, 3);
    let handle = worker::spawn(
        move || Ok(Panicking(MockExecutor::new(spec, MockConfig::default()))),
        SchedulerConfig::default(),
        Duration::from_secs(1),
        stopped_tx,
    )
    .unwrap();
    std::thread::spawn(move || {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let admission = Admission::new(1);
        let submitted = handle.submit(request(handle.next_id()), admission.try_acquire().unwrap());
        // Keep the request alive while the engine dies.
        std::thread::sleep(Duration::from_secs(60));
        drop(submitted);
    });
    let router = axum::Router::new().route(
        "/upload",
        axum::routing::post(|body: axum::body::Bytes| async move { body.len().to_string() }),
    );
    let result = eidola_server_engine::run(
        "127.0.0.1:0".parse().unwrap(),
        router,
        stopped_rx,
        std::future::pending(),
        |addr| {
            use std::io::Write;
            println!("LISTENING {addr}");
            std::io::stdout().flush().unwrap();
        },
    );
    std::process::exit(if result.is_err() { 3 } else { 0 });
}

#[test]
fn a_fatal_engine_stop_ends_the_process_despite_a_stalled_upload() {
    use std::io::{BufRead, Write};
    use std::process::{Command, Stdio};
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_node_with_a_panicking_engine",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let addr = loop {
        let line = lines
            .next()
            .expect("the child printed its address")
            .unwrap();
        // libtest may print the test's name on the same line first.
        if let Some((_, a)) = line.split_once("LISTENING ") {
            break a.trim().to_string();
        }
    };
    // A request whose body never finishes arriving.
    let mut conn = std::net::TcpStream::connect(&addr).unwrap();
    conn.write_all(b"POST /upload HTTP/1.1\r\nhost: x\r\ncontent-length: 1000000\r\n\r\npartial")
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    // Now the engine dies.
    let stdin = child.stdin.as_mut().unwrap();
    stdin.write_all(b"go\n").unwrap();
    stdin.flush().unwrap();
    let start = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(10) {
            child.kill().unwrap();
            panic!("the node kept running after its engine stopped");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(3), "a non-zero (error) exit");
    assert!(start.elapsed() < eidola_server_engine::TEARDOWN_LIMIT + Duration::from_secs(3));
    drop(conn);
}
