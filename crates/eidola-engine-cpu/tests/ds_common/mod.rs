//! Shared harness: the serving core's engine over the DeepSeek-V4.1 CPU executor on the
//! model crate's committed synthetic fixture (eight layers split four and four, window
//! 48, a pooled and a per-position global bank with pruning top-ks, two Engram layers,
//! a three-block drafter of four slots), with every row checked against the dense
//! oracle.

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::executor::Executor;
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream, mix64};
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cpu::deepseek::oracle::ds_check_generation;
use eidola_engine_cpu::deepseek::{
    DsCpuExecutor, DsCpuExecutorConfig, DsDenseOracle, DsDenseRun, DsRowRecord, StepTransfer,
};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::{DsLoadOptions, DsModel, load_reference};

/// Deterministic test RNG (SplitMix64).
pub struct TestRng(pub u64);

impl TestRng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
    pub fn var_tokens(&mut self, min: usize, extra: u64) -> Vec<u32> {
        let n = min + self.below(extra) as usize;
        self.tokens(n)
    }
    pub fn tokens(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.below(VOCAB as u64) as u32).collect()
    }
}

/// See the MiMo harness: raises the soft memory-lock limit for host-tier pools and
/// serializes the tests that build them.
pub fn memlock_guard() -> std::sync::MutexGuard<'static, ()> {
    use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let l = getrlimit(Resource::Memlock);
    if l.current.is_some() && l.current != l.maximum {
        let _ = setrlimit(
            Resource::Memlock,
            Rlimit {
                current: l.maximum,
                maximum: l.maximum,
            },
        );
    }
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The lineage of client key `[n; 32]` under a fixed boot key.
pub fn salt(n: u8) -> Lineage {
    SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32])).lineage(&CacheKey::from_bytes([n; 32]))
}

pub fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../eidola-engine-model/tests/fixtures/deepseek_tiny")
}

/// The fixture's Engram compressed-vocabulary map, kept with its goldens.
pub fn token_map() -> Vec<u32> {
    let golden = WeightSet::open_files(&[fixture_dir().join("golden.safetensors")]).unwrap();
    let name = "engram.token_map";
    let ids = golden.get(name).unwrap().to_i64(name).unwrap();
    ids.into_iter().map(|t| t as u32).collect()
}

pub fn load_fixture() -> DsModel {
    load_reference(
        &fixture_dir(),
        Ok,
        &DsLoadOptions::default(),
        Some(token_map()),
    )
    .unwrap()
}

pub fn fixture() -> Arc<DsModel> {
    static MODEL: OnceLock<Arc<DsModel>> = OnceLock::new();
    MODEL.get_or_init(|| Arc::new(load_fixture())).clone()
}

/// Token ids a request may hold or produce: the head has 96 rows, and id 95 is the image
/// token, which the text model refuses, so it is padding here.
pub const VOCAB: u32 = 95;

/// The chain's window, and how far behind a target its top and bottom layers start when
/// the drafter is on.
pub const WINDOW: u32 = 48;
pub const TOP: u32 = 48;
pub const REACH: u32 = 48 + 3 * 47;

pub fn exec_config(blocks: u32, draft_width: u32) -> DsCpuExecutorConfig {
    DsCpuExecutorConfig {
        block_size: 8,
        num_blocks: blocks,
        host_blocks: 0,
        num_state_slots: 16,
        max_model_len: 1024,
        buckets: vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 8,
                max_tokens: 64,
            },
            Bucket {
                max_seqs: 16,
                max_tokens: 512,
            },
        ],
        draft_width,
        sampleable_vocab_size: VOCAB,
        record: true,
    }
}

pub fn sched(speculative: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: 512,
        max_seqs: 16,
        max_prefill_chunk: 512,
        eos_token_ids: Vec::new(),
        speculative,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    }
}

pub fn chunked(chunk: u32, speculative: bool) -> SchedulerConfig {
    let mut cfg = sched(speculative);
    cfg.max_prefill_chunk = chunk;
    cfg
}

pub fn request(
    id: u64,
    prompt: Vec<u32>,
    params: SamplingParams,
    max_tokens: u32,
    cache: CacheScope,
) -> Request {
    Request {
        id,
        prompt,
        sampling: params,
        max_tokens,
        required_cached_tokens: 0,
        stop_token_ids: Vec::new(),
        cache,
    }
}

pub fn params(i: u64) -> SamplingParams {
    match i % 3 {
        0 => SamplingParams::greedy(),
        1 => SamplingParams::random(1.0, 1000 + i).unwrap(),
        _ => SamplingParams::new(0.7, 8, 0.9, 0.05, 2000 + i).unwrap(),
    }
}

pub fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Dense runs keyed by token sequence; any extension of a record's tokens serves it, since
/// every logit and every drafter KV depends only on its prefix.
#[derive(Default)]
pub struct DenseCache {
    runs: Vec<(Vec<u32>, Arc<DsDenseRun>)>,
    pub computed: usize,
}

impl DenseCache {
    pub fn get(&mut self, oracle: &DsDenseOracle, tokens: &[u32]) -> Arc<DsDenseRun> {
        if let Some((_, r)) = self.runs.iter().find(|(t, _)| t.starts_with(tokens)) {
            return r.clone();
        }
        let r = Arc::new(oracle.run(tokens).unwrap());
        self.computed += 1;
        self.runs.push((tokens.to_vec(), r.clone()));
        r
    }
}

/// Counters from checked records.
#[derive(Clone, Copy, Debug, Default)]
pub struct Checked {
    pub records: usize,
    pub target_rows: usize,
    pub draft_rows: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Rows whose chain started after the row did (a triangle, not every layer).
    pub triangles: usize,
}

/// Checks one record bit for bit against the dense oracle over the record's own tokens
/// (its prefix, host tokens and drafts, rejected ones included): every target logit
/// row, every draft slot's logits, every draft draw, and the produced tokens.
pub fn check_record(
    oracle: &DsDenseOracle,
    cache: &mut DenseCache,
    r: &DsRowRecord,
    sum: &mut Checked,
) {
    let dense = cache.get(oracle, &r.tokens);
    let ctx = format!(
        "slot {} context {} n {} chain start {}",
        r.slot, r.context_len, r.num_tokens, r.chain_start
    );
    for (pos, l) in &r.target_logits {
        assert_eq!(
            bits(l),
            bits(dense.target(*pos)),
            "{ctx}: target logits at {pos}"
        );
        sum.target_rows += 1;
    }
    sum.records += 1;
    sum.triangles += usize::from(r.chain_start > r.context_len);
    if r.produced.is_empty() {
        return;
    }
    let p = r.context_len + r.num_tokens - 1;
    let params = &r.sampling;
    let sampleable = oracle.sampleable_vocab_size;
    fn logits(row: &[f32], sampleable: u32) -> Logits<'_> {
        Logits::new(row, sampleable)
    }
    let expected = if r.drafts.is_empty() {
        vec![sampling::sample(
            logits(dense.target(p), sampleable),
            params,
            p as u64 + 1,
        )]
    } else {
        let k = r.drafts.len();
        let core = oracle.draft(&dense, p).unwrap();
        let mut previous = r.tokens[p as usize];
        let mut q_rows = Vec::new();
        for (i, &drafted) in r.drafts.iter().enumerate() {
            let row = oracle.model.draft_logits(&core, i, previous);
            assert_eq!(
                bits(&r.draft_logits[i]),
                bits(&row),
                "{ctx}: draft slot {i} logits"
            );
            sum.draft_rows += 1;
            let view = logits(&row, sampleable);
            let pos = p as u64 + 1 + i as u64;
            let (d, q) = if params.is_greedy() {
                let d = sampling::argmax(view);
                let mut q = vec![0.0; view.len()];
                q[d as usize] = 1.0;
                (d, q)
            } else {
                let q = sampling::processed_probs(view, params);
                (
                    sampling::sample_from(&q, sampling::uniform(params.seed(), pos, Stream::Draft)),
                    q,
                )
            };
            assert_eq!(d, drafted, "{ctx}: draft {i}");
            q_rows.push(q);
            previous = d;
        }
        let p_rows: Vec<Vec<f64>> = (0..=k as u32)
            .map(|i| sampling::processed_probs(logits(dense.target(p + i), sampleable), params))
            .collect();
        let out = sampling::chain_accept(&p_rows, &q_rows, &r.drafts, params, p as u64 + 1);
        sum.drafted += k;
        sum.accepted += out.len() - 1;
        out
    };
    assert_eq!(r.produced, expected, "{ctx}: produced tokens");
}

pub struct Harness {
    pub eng: Engine<DsCpuExecutor>,
    pub spec: ModelSpec,
    pub model: Arc<DsModel>,
    pub now: u64,
    pub outputs: HashMap<u64, Vec<u32>>,
    pub finished: HashMap<u64, FinishReason>,
    pub cached: HashMap<u64, u32>,
    pub prompts: HashMap<u64, (Vec<u32>, SamplingParams)>,
    pub records: Vec<DsRowRecord>,
    /// Run the full KV invariant scan after every step.
    pub checks: bool,
}

impl Harness {
    pub fn new(model: Arc<DsModel>, cfg: DsCpuExecutorConfig, sched: SchedulerConfig) -> Self {
        let exec = DsCpuExecutor::new(model.clone(), cfg);
        let spec = exec.spec().clone();
        Self {
            eng: Engine::new(exec, sched).expect("valid configuration"),
            spec,
            model,
            now: 0,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
            prompts: HashMap::new(),
            records: Vec::new(),
            checks: true,
        }
    }

    pub fn fixture(blocks: u32, draft_width: u32, sched: SchedulerConfig) -> Self {
        Self::new(fixture(), exec_config(blocks, draft_width), sched)
    }

    pub fn oracle(&self) -> DsDenseOracle<'_> {
        DsDenseOracle {
            model: &self.model,
            sampleable_vocab_size: self.spec.sampleable_vocab_size,
        }
    }

    pub fn submit(&mut self, req: Request) {
        self.outputs.insert(req.id, Vec::new());
        self.prompts
            .insert(req.id, (req.prompt.clone(), req.sampling));
        self.eng.submit(req).expect("submit");
    }

    pub fn check(&self) {
        if !self.checks {
            return;
        }
        let kv = self.eng.kv();
        let x = self.eng.executor();
        kv.check_invariants();
        kv.check_no_reservations();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    x.block_is_zero(g, b),
                    "free block {g}/{b} holds data with no zero pending"
                );
            }
        }
        let resets = kv.pending_slot_resets();
        for slot in kv.free_slot_set() {
            if !resets.contains(&slot) {
                assert!(
                    x.slot_state_is_zero(slot),
                    "free slot {slot} holds model state with no reset pending"
                );
            }
        }
        let idle = kv.idle_blocks();
        let crypts = kv.pending_crypts();
        let transfers = kv.pending_transfers();
        let landing: BTreeSet<(u32, u32)> = transfers
            .iter()
            .filter(|t| !t.3)
            .map(|t| (t.0, t.1))
            .collect();
        let demoting: BTreeSet<(u32, u32)> = transfers
            .iter()
            .filter(|t| t.3)
            .map(|t| (t.0, t.2))
            .collect();
        let host_zeros = kv.pending_host_zeros();
        for (g, hb) in kv.free_host_set() {
            if !host_zeros.contains(&(g, hb)) {
                assert!(
                    x.host_block_is_zero(g, hb),
                    "free host block {g}/{hb} holds data with no zero pending"
                );
            }
        }
        for (g, hb) in kv.host_cache_blocks() {
            if !demoting.contains(&(g, hb)) {
                assert!(
                    x.host_block_is_encrypted(g, hb),
                    "held host block {g}/{hb} holds no ciphertext"
                );
            }
        }
        for (g, b) in kv.cache_blocks().union(&kv.sequence_blocks()) {
            if landing.contains(&(*g, *b)) {
                continue;
            }
            let want = idle.contains(&(*g, *b));
            let device = x.block_is_encrypted(*g, *b);
            match crypts.get(&(*g, *b)) {
                Some(&encrypt) => assert!(
                    encrypt == want && device != want,
                    "block {g}/{b}: queued crypt disagrees with its state"
                ),
                None => assert_eq!(device, want, "block {g}/{b}: encryption != idle"),
            }
        }
    }

    /// Runs pending maintenance (as a caller does when the last request ends) and checks.
    pub fn flush(&mut self) {
        self.eng.flush().expect("flush");
        self.check();
    }

    pub fn step(&mut self) {
        let events = self.eng.step(self.now).expect("step");
        for e in events {
            self.outputs.entry(e.id).or_default().extend(&e.tokens);
            if let Some(f) = e.finish {
                assert!(self.finished.insert(e.id, f).is_none(), "finished twice");
                self.cached.insert(e.id, e.cached_prompt_tokens);
            }
        }
        self.records.extend(self.eng.executor().take_records());
        self.check();
    }

    pub fn run(&mut self) {
        let mut guard = 0;
        while self.eng.unfinished() > 0 {
            self.step();
            self.now += 1;
            guard += 1;
            assert!(guard < 1_000_000, "no progress");
        }
    }

    /// Submits a request, steps once, returns its hit, and runs it to the end.
    pub fn run_one(
        &mut self,
        id: u64,
        prompt: &[u32],
        p: SamplingParams,
        max: u32,
        cache: CacheScope,
    ) -> u32 {
        self.submit(request(id, prompt.to_vec(), p, max, cache));
        self.step();
        let cached = self
            .eng
            .cached_prompt_tokens(id)
            .or_else(|| self.cached.get(&id).copied())
            .expect("admitted in its first step");
        self.run();
        self.flush();
        cached
    }

    /// Checks every finished output against dense generation (exactly when speculation
    /// is off or the request is greedy), seeding the cache with every full transcript,
    /// then every recorded row against the dense oracle, bit for bit.
    pub fn check_all(&mut self, spec_on: bool) -> Checked {
        let mut cache = DenseCache::default();
        let records = std::mem::take(&mut self.records);
        let oracle = self.oracle();
        let mut ids: Vec<u64> = self.finished.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let (prompt, p) = &self.prompts[&id];
            let out = &self.outputs[&id];
            let mut all = prompt.clone();
            all.extend(out);
            let dense = cache.get(&oracle, &all);
            let exact = !spec_on || p.is_greedy();
            if exact
                && let Err(pos) =
                    ds_check_generation(&dense, oracle.sampleable_vocab_size, prompt.len(), out, p)
            {
                panic!("request {id}: output diverges from dense generation at {pos}");
            }
        }
        let mut sum = Checked::default();
        for r in &records {
            check_record(&oracle, &mut cache, r, &mut sum);
        }
        sum
    }

    pub fn take_transfers(&mut self) -> Vec<StepTransfer> {
        // The engine owns the executor; the transfers are read through a shared borrow.
        self.eng.executor().transfers()
    }
}
