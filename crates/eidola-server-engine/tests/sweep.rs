//! The idle engine thread sweeps the prefix cache on schedule however often commands
//! arrive: a stream of commands that leave the engine idle cannot postpone expiry.

use std::time::{Duration, Instant};

use eidola_engine::engine::{CacheScope, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::mock::{MockConfig, MockExecutor, mimo_like_spec};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::EngineSalt;
use eidola_server_engine::worker::{self, Admission, Output, Stats};

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

fn request(handle: &worker::EngineHandle, prompt: Vec<u32>, cache: CacheScope) -> Request {
    Request {
        id: handle.next_id(),
        prompt,
        sampling: SamplingParams::greedy(),
        max_tokens: 1,
        stop_token_ids: Vec::new(),
        cache,
    }
}

#[test]
fn commands_faster_than_the_sweep_interval_do_not_postpone_it() {
    const SWEEP: Duration = Duration::from_millis(100);
    const IDLE_TTL_MS: u64 = 600;
    let (stopped_tx, _stopped_rx) = tokio::sync::oneshot::channel();
    let spec = mimo_like_spec(32, 4, 6, 10, 256, 8, 3);
    let handle = worker::spawn(
        move || Ok(MockExecutor::new(spec, MockConfig::default())),
        SchedulerConfig {
            eos_token_ids: Vec::new(),
            speculative: false,
            cache: CachePolicy {
                enabled: true,
                idle_ttl_ms: IDLE_TTL_MS,
                max_age_ms: 60_000,
            },
            // The engine's own in-step sweep never comes due here: only the idle sweep
            // can evict.
            sweep_interval_ms: 3_600_000,
            ..SchedulerConfig::default()
        },
        SWEEP,
        stopped_tx,
    )
    .unwrap();
    let admission = Admission::new(64);

    // A keyed request leaves whole prompt blocks in the prefix cache.
    let (guard, mut events) = handle
        .submit(
            request(
                &handle,
                (1..=17).collect(),
                CacheScope::Keyed(EngineSalt::from_bytes([7; 32])),
            ),
            admission.try_acquire().unwrap(),
        )
        .unwrap();
    while let Some(out) = events.blocking_recv() {
        if matches!(
            out,
            Output::Tokens {
                finish: Some(_),
                ..
            }
        ) {
            break;
        }
    }
    drop((guard, events));
    let cached_at = Instant::now();
    wait_for(&handle, |s| s.finished == 1 && s.in_engine == 0);
    // Let the sweeps due before expiry run (nothing has expired yet), so the baseline
    // includes any maintenance the finish queued.
    std::thread::sleep(SWEEP * 3);
    let baseline = handle.stats().engine_steps;

    // Commands every 10 ms, each refused (an empty prompt), so the engine stays idle,
    // until well after the cached entry expired.
    let mut evicted = None;
    while cached_at.elapsed() < Duration::from_millis(IDLE_TTL_MS) + SWEEP * 10 {
        let (_guard, mut events) = handle
            .submit(
                request(&handle, Vec::new(), CacheScope::Private),
                admission.try_acquire().unwrap(),
            )
            .unwrap();
        assert!(matches!(events.blocking_recv(), Some(Output::Rejected(_))));
        let s = handle.stats();
        if s.engine_steps > baseline {
            evicted = Some(s);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let s = evicted.expect("no sweep ran while commands kept arriving");
    assert!(
        cached_at.elapsed() >= Duration::from_millis(IDLE_TTL_MS),
        "swept before expiry: {s:?}"
    );
}
