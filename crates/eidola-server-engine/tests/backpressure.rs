//! A reader that stops reading cannot make the node buffer without bound: its request is
//! cut off once [`EVENT_BUFFER`] events wait unread, and its admission slot stays taken
//! until the buffered output is drained or dropped.

use std::time::{Duration, Instant};

use eidola_engine::engine::{CacheScope, Request, SchedulerConfig};
use eidola_engine::mock::{MockConfig, MockExecutor, mimo_like_spec};
use eidola_engine::sampling::SamplingParams;
use eidola_server_engine::worker::{self, Admission, EVENT_BUFFER, Output, Stats};

fn wait_for(handle: &worker::EngineHandle, pred: impl Fn(&Stats) -> bool) -> Stats {
    let start = Instant::now();
    loop {
        let s = handle.stats();
        if pred(&s) {
            return s;
        }
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "timed out: {s:?}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_stalled_reader_is_cut_off_and_its_buffer_stays_admitted() {
    let (stopped_tx, _stopped_rx) = tokio::sync::oneshot::channel();
    let spec = mimo_like_spec(32, 4, 6, 10, 2048, 8, 3);
    let handle = worker::spawn(
        move || Ok(MockExecutor::new(spec, MockConfig::default())),
        SchedulerConfig {
            eos_token_ids: Vec::new(),
            ..SchedulerConfig::default()
        },
        Duration::from_secs(1),
        stopped_tx,
    )
    .unwrap();
    let admission = Admission::new(1);
    let (guard, mut events) = handle
        .submit(
            Request {
                id: handle.next_id(),
                prompt: vec![1, 2, 3],
                sampling: SamplingParams::greedy(),
                // Far more steps than the buffer holds.
                max_tokens: 4000,
                stop_token_ids: Vec::new(),
                cache: CacheScope::Private,
            },
            admission.try_acquire().unwrap(),
        )
        .unwrap();

    // Nobody reads: the request is cut off once the buffer is full.
    let s = wait_for(&handle, |s| s.in_engine == 0 && s.submitted == 1);
    assert_eq!((s.cut_off, s.cancelled, s.finished), (1, 1, 0), "{s:?}");
    let steps = s.engine_steps;
    assert!(
        steps < 1000,
        "it stopped long before its 4000 tokens: {s:?}"
    );
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        handle.stats().engine_steps,
        steps,
        "no compute after the cut-off"
    );

    // The buffered output still holds the slot.
    assert_eq!(admission.in_flight(), 1);
    assert!(admission.try_acquire().is_none());

    // What was buffered is bounded, then the channel ends.
    let mut buffered = 0;
    while let Some(out) = events.blocking_recv() {
        assert!(matches!(
            out,
            Output::Accepted | Output::Tokens { finish: None, .. }
        ));
        buffered += 1;
    }
    assert_eq!(buffered, EVENT_BUFFER);

    // Dropping the response's guard and receiver releases the slot.
    drop((guard, events));
    assert_eq!(admission.in_flight(), 0);
}
