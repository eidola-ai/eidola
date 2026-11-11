//! What the executor leaves on the device between requests, on the truncated
//! real checkpoint (`real_flash.rs` for the environment; the drafted cases
//! need its MTP layers): once the engine goes idle every buffer a step with
//! rows wrote (scratch, sampler, drafting buffers and tables, decode graphs'
//! step table and read-back, pad blocks) reads back as zeros, having held data
//! just before; the scrub runs once per idle transition, not on every
//! maintenance-only step; it changes no token; and `scrub_device`, what drop
//! runs, zeroes live KV too. Undrafted and drafted (three depths), graphs off
//! and on.

mod common;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use common::setup;
use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, SaltDeriver};
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, CudaGraphs, KvBlocks, MtpHidden};
use eidola_engine_model::safetensors::WeightSet;

const KEEP: [usize; 4] = [0, 1, 2, 5];
const SAMPLEABLE: u32 = 151_675;

struct Env {
    store: Arc<WeightSet>,
    text: Vec<u32>,
    mtp: bool,
}

fn env() -> Option<&'static Env> {
    static ENV: OnceLock<Option<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let (Some(dir), Some(golden)) = (
            std::env::var_os("EIDOLA_MIMO_DIR"),
            std::env::var_os("EIDOLA_MIMO_GOLDEN"),
        ) else {
            eprintln!("skipping: EIDOLA_MIMO_DIR / EIDOLA_MIMO_GOLDEN not set");
            return None;
        };
        let g = WeightSet::open_files(&[PathBuf::from(golden)]).unwrap();
        let text = g
            .get("tokens")
            .unwrap()
            .to_i64("tokens")
            .unwrap()
            .into_iter()
            .map(|x| u32::try_from(x).unwrap())
            .collect();
        let store = Arc::new(WeightSet::open_dir(&PathBuf::from(dir)).unwrap());
        let mtp = store.contains("model.mtp.layers.2.eh_proj.weight");
        Some(Env { store, text, mtp })
    })
    .as_ref()
}

/// The executor behind the engine, shared so a test can reach it mutably.
struct Shared {
    spec: ModelSpec,
    ex: Rc<RefCell<CudaExecutor>>,
}

impl Executor for Shared {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        self.ex.borrow_mut().execute(step)
    }
}

fn engine(depths: u32, graphs: CudaGraphs) -> Option<(Engine<Shared>, Rc<RefCell<CudaExecutor>>)> {
    let env = env()?;
    if depths > 0 && !env.mtp {
        eprintln!("skipping drafted case: EIDOLA_MIMO_DIR holds no MTP layers");
        return None;
    }
    let su = setup()?;
    let cfg = CudaExecutorConfig {
        block_size: 16,
        host_blocks: KvBlocks::default(),
        num_blocks: KvBlocks {
            global: 64,
            sliding: 64,
            drafter: if depths > 0 { 64 } else { 0 },
        },
        num_state_slots: 8,
        max_model_len: 1024,
        buckets: vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 8,
                max_tokens: 256,
            },
        ],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs,
        draft_tokens: depths,
        mtp_hidden: MtpHidden::Normed,
    };
    let ex = CudaExecutor::new(su.gpu, &su.dir, env.store.clone(), Some(&KEEP), cfg).unwrap();
    let ex = Rc::new(RefCell::new(ex));
    let shared = Shared {
        spec: ex.borrow().spec().clone(),
        ex: ex.clone(),
    };
    let sched = SchedulerConfig {
        max_batched_tokens: 256,
        max_seqs: 8,
        // Two prefill chunks, then decode.
        max_prefill_chunk: 48,
        eos_token_ids: Vec::new(),
        speculative: depths > 0,
        cache: CachePolicy::default(),
        sweep_interval_ms: u64::MAX,
    };
    Some((Engine::new(shared, sched).expect("valid configuration"), ex))
}

fn request(id: u64, prompt: Vec<u32>) -> Request {
    let lineage = SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32]))
        .lineage(&CacheKey::from_bytes([7; 32]));
    Request {
        id,
        prompt,
        sampling: SamplingParams::greedy(),
        max_tokens: 12,
        stop_token_ids: Vec::new(),
        cache: CacheScope::Keyed(lineage),
    }
}

/// Run `req` to its end without the engine's idle flush; its tokens.
fn run(eng: &mut Engine<Shared>, req: Request, now: u64) -> Vec<u32> {
    eng.submit(req).unwrap();
    let mut out = Vec::new();
    let mut guard = 0;
    while eng.unfinished() > 0 {
        for e in eng.step(now).unwrap() {
            out.extend(e.tokens);
        }
        guard += 1;
        assert!(guard < 1000, "no progress");
    }
    out
}

/// The buffers a run must have left data in (the positive baseline), by
/// configuration.
fn expected(depths: u32, graphs: CudaGraphs) -> Vec<&'static str> {
    let mut v = vec![
        "model.h",
        "model.x",
        "model.qkv",
        "model.attn",
        "model.logits",
    ];
    if depths == 0 {
        // A drafted step's forwards read their token ids and positions from
        // its step table instead.
        v.extend([
            "model.tokens",
            "model.positions",
            "sampler.probs",
            "sampler.tokens",
        ]);
        if graphs == CudaGraphs::On {
            v.extend(["decode.table", "decode.readback", "kv.pad_block"]);
        }
    } else {
        v.extend([
            "draft.levels",
            "draft.target",
            "draft.probs",
            "draft.readback",
            "draft.eager_table",
        ]);
        if graphs == CudaGraphs::On {
            v.extend(["draft.graph_table", "kv.pad_block"]);
        }
    }
    v
}

fn idle_scrub_case(depths: u32, graphs: CudaGraphs) {
    let Some((mut eng, ex)) = engine(depths, graphs) else {
        return;
    };
    let prompt = env().unwrap().text[..90].to_vec();
    let first = run(&mut eng, request(1, prompt.clone()), 0);
    assert!(
        eng.has_pending_maintenance(),
        "the release left maintenance"
    );

    // Baseline: the run left data in every buffer the configuration uses.
    let held = ex.borrow().residue_nonzero().unwrap();
    for name in expected(depths, graphs) {
        assert!(
            held.contains(&name),
            "{name} holds no data before the idle step (held: {held:?})"
        );
    }
    let scrubs = ex.borrow().idle_scrubs();

    // The engine goes idle: its flush is a step with no rows.
    eng.flush().unwrap();
    assert_eq!(ex.borrow().idle_scrubs(), scrubs + 1);
    let left = ex.borrow().residue_nonzero().unwrap();
    assert!(left.is_empty(), "data left after the idle step: {left:?}");

    // Another maintenance-only step while idle (expiry zeroing the cache)
    // does not scrub again.
    let later = 3 * 60 * 60 * 1000;
    let steps = eng.stats().steps;
    eng.sweep(later).unwrap();
    assert_eq!(
        eng.stats().steps,
        steps + 1,
        "the sweep ran a maintenance-only step"
    );
    assert_eq!(ex.borrow().idle_scrubs(), scrubs + 1);

    // The same request again (now recomputing: its cache expired) produces
    // the same tokens, and the next idle transition scrubs once more.
    let second = run(&mut eng, request(2, prompt), later);
    assert_eq!(first, second, "the scrub changed a token");
    eng.flush().unwrap();
    assert_eq!(ex.borrow().idle_scrubs(), scrubs + 2);
    let left = ex.borrow().residue_nonzero().unwrap();
    assert!(
        left.is_empty(),
        "data left after the second idle step: {left:?}"
    );
}

#[test]
fn idle_scrub_undrafted_eager() {
    idle_scrub_case(0, CudaGraphs::Off);
}

#[test]
fn idle_scrub_undrafted_graphs() {
    idle_scrub_case(0, CudaGraphs::On);
}

#[test]
fn idle_scrub_drafted_eager() {
    idle_scrub_case(3, CudaGraphs::Off);
}

#[test]
fn idle_scrub_drafted_graphs() {
    idle_scrub_case(3, CudaGraphs::On);
}

/// What drop runs: mid-request, every block a sequence maps (plaintext KV)
/// and every residue buffer read back as zeros afterwards, having held data.
#[test]
fn scrub_device_zeroes_live_kv() {
    let Some((mut eng, ex)) = engine(0, CudaGraphs::Off) else {
        return;
    };
    let prompt = env().unwrap().text[..90].to_vec();
    eng.submit(request(1, prompt)).unwrap();
    for _ in 0..4 {
        eng.step(0).unwrap();
    }
    assert!(eng.unfinished() > 0, "still running");
    let live = eng.kv().sequence_blocks();
    assert!(!live.is_empty());
    let nonzero = |ex: &CudaExecutor| {
        live.iter()
            .filter(|&&(g, b)| !ex.block_is_zero(g, b).unwrap())
            .count()
    };
    assert!(nonzero(&ex.borrow()) > 0, "live blocks hold KV");
    assert!(!ex.borrow().residue_nonzero().unwrap().is_empty());
    ex.borrow_mut().scrub_device().unwrap();
    assert_eq!(nonzero(&ex.borrow()), 0, "live KV left after the scrub");
    let left = ex.borrow().residue_nonzero().unwrap();
    assert!(left.is_empty(), "data left after the scrub: {left:?}");
}
