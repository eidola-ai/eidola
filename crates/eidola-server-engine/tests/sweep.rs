//! The idle engine thread sweeps the prefix cache on schedule however often commands
//! arrive: a stream of commands that leave the engine idle cannot postpone expiry.

use std::time::{Duration, Instant};

use eidola_engine::engine::{CacheScope, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::mock::{MockConfig, MockExecutor, mimo_like_spec};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
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

fn lineage(n: u8) -> Lineage {
    SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32])).lineage(&CacheKey::from_bytes([n; 32]))
}

fn request(handle: &worker::EngineHandle, prompt: Vec<u32>, cache: CacheScope) -> Request {
    Request {
        id: handle.next_id(),
        prompt,
        sampling: SamplingParams::greedy(),
        max_tokens: 1,
        required_cached_tokens: 0,
        stop_token_ids: Vec::new(),
        cache,
    }
}

/// When the last request ends, the engine runs the maintenance it left (encrypting the
/// blocks it left cached, zeroing the ones it freed) at once, not at the next sweep: with
/// the sweep an hour away, the finish is followed by exactly one maintenance-only step,
/// and the engine holds no key.
#[test]
fn the_last_request_leaves_no_maintenance_and_no_key() {
    let (stopped_tx, _stopped_rx) = tokio::sync::oneshot::channel();
    let spec = mimo_like_spec(32, 4, 6, 10, 256, 8, 3);
    let handle = worker::spawn(
        move || Ok(MockExecutor::new(spec, MockConfig::default())),
        SchedulerConfig {
            eos_token_ids: Vec::new(),
            speculative: false,
            sweep_interval_ms: 3_600_000,
            ..SchedulerConfig::default()
        },
        Duration::from_secs(3600),
        stopped_tx,
    )
    .unwrap();
    let admission = Admission::new(4);
    let (guard, mut events) = handle
        .submit(
            request(&handle, (1..=17).collect(), CacheScope::Keyed(lineage(7))),
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
    // One step prefills the 17 tokens and samples the one output token; the second is
    // the maintenance its release queued.
    let s = wait_for(&handle, |s| s.finished == 1 && s.in_engine == 0);
    assert_eq!((s.engine_steps, s.kv_keys_held), (2, 0), "{s:?}");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(handle.stats().engine_steps, 2, "nothing else runs");
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

    // A keyed request leaves whole prompt blocks in the prefix cache, stamped with the
    // engine's clock at the step that finishes it. That stamp is not observable here, and
    // the finish reaches this thread an unbounded scheduling delay after it, so expiry is
    // measured from the submission, which precedes the stamp.
    let submitted_at = Instant::now();
    let (guard, mut events) = handle
        .submit(
            request(&handle, (1..=17).collect(), CacheScope::Keyed(lineage(7))),
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

    // Commands every 10 ms, each refused (an empty prompt), so the engine stays idle,
    // until a step beyond the request's own two (the one that prefilled and sampled, and
    // the maintenance its release queued, as the test above pins) shows the sweep evicted
    // the entry. Counting from that fixed number rather than a baseline read later keeps
    // an eviction that lands before this thread looks from going unseen.
    const REQUEST_STEPS: u64 = 2;
    let s = loop {
        let (_guard, mut events) = handle
            .submit(
                request(&handle, Vec::new(), CacheScope::Private),
                admission.try_acquire().unwrap(),
            )
            .unwrap();
        assert!(matches!(events.blocking_recv(), Some(Output::Rejected(_))));
        let s = handle.stats();
        if s.engine_steps > REQUEST_STEPS {
            break s;
        }
        assert!(
            submitted_at.elapsed() < Duration::from_millis(IDLE_TTL_MS) + SWEEP * 10,
            "no sweep ran while commands kept arriving: {s:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    // The engine's clock counts whole milliseconds since it started, so an entry stamped
    // at the end of one millisecond and swept at the start of another may have been idle
    // up to a millisecond less than the TTL by this thread's clock.
    assert!(
        submitted_at.elapsed() >= Duration::from_millis(IDLE_TTL_MS - 1),
        "swept before expiry: {s:?}"
    );
}
