//! Any exit of the engine thread is fatal to the node: an injected executor panic turns
//! health unhealthy, fails the in-flight request, and ends `serve` with an error (which
//! `main` turns into a non-zero exit).

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
    let (_guard, mut events) = handle
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
    assert_eq!(
        admission.in_flight(),
        0,
        "the request's permit was released"
    );
}
