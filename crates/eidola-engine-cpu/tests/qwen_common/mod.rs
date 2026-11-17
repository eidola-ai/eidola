//! Shared harness: the serving core's engine over the Qwen3.8 CPU executor on the model
//! crate's committed synthetic fixture, with every row, every draft draw and every stored
//! state checked against the dense oracle, and the memory checked after every step.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream, mix64};
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cpu::qwen::{
    QwenCpuExecutor, QwenCpuExecutorConfig, QwenDenseOracle, QwenDenseRun, QwenRowRecord,
    StoredState, qwen_check_generation,
};
use eidola_engine_model::qwen::{QwenModel, load_reference};

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
    pub fn var_tokens(&mut self, min: usize, extra: u64, vocab: u32) -> Vec<u32> {
        let n = min + self.below(extra) as usize;
        self.tokens(n, vocab)
    }
    pub fn tokens(&mut self, n: usize, vocab: u32) -> Vec<u32> {
        (0..n).map(|_| self.below(vocab as u64) as u32).collect()
    }
}

/// Raises the soft memory-lock limit to the hard one, so the host tier's locked pools
/// fit; never lowers it.
pub fn allow_memlock() {
    use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
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
}

/// Serializes the tests of one binary that build executors with a host tier.
pub fn memlock_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    allow_memlock();
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The lineage of client key `[n; 32]` under a fixed boot key.
pub fn salt(n: u8) -> Lineage {
    SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32])).lineage(&CacheKey::from_bytes([n; 32]))
}

/// The committed synthetic fixture: three DeltaNet layers and one gated attention layer
/// at hidden size 128, one MTP layer, a vocabulary of 512 (500 is end-of-sequence, 508
/// on vision placeholders).
pub fn fixture() -> Arc<QwenModel> {
    static MODEL: OnceLock<Arc<QwenModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../eidola-engine-model/tests/fixtures/qwen_tiny");
            Arc::new(load_reference(&dir, Ok).unwrap())
        })
        .clone()
}

/// Token ids prompts are drawn from.
pub const VOCAB: u32 = 500;
/// Token ids the executor may emit: every id but the vision placeholders.
pub const SAMPLEABLE: u32 = 508;

/// Block 4, `blocks` KV blocks per group, `buffers` state buffers, 32 slots, `k` drafts.
pub fn exec_config(blocks: u32, buffers: u32, k: u32) -> QwenCpuExecutorConfig {
    QwenCpuExecutorConfig {
        block_size: 4,
        num_blocks: blocks,
        host_blocks: 0,
        state_buffers: buffers,
        host_state_buffers: 0,
        num_state_slots: 32,
        max_model_len: 512,
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
                max_seqs: 32,
                max_tokens: 256,
            },
        ],
        draft_tokens: k,
        sampleable_vocab_size: SAMPLEABLE,
        record: true,
    }
}

pub fn sched(speculative: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: 256,
        max_seqs: 32,
        max_prefill_chunk: 256,
        eos_token_ids: Vec::new(),
        speculative,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    }
}

pub fn chunked(chunk: u32, speculative: bool) -> SchedulerConfig {
    let mut s = sched(speculative);
    s.max_prefill_chunk = chunk;
    s
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

pub fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
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

/// Dense runs keyed by token sequence; any extension of a record's tokens serves it.
#[derive(Default)]
pub struct DenseCache {
    runs: Vec<(Vec<u32>, Arc<QwenDenseRun>)>,
    pub computed: usize,
}

impl DenseCache {
    pub fn get(&mut self, oracle: &QwenDenseOracle, tokens: &[u32]) -> Arc<QwenDenseRun> {
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
    pub drafter_rows: usize,
    pub chain_rows: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Drafting rows by how many drafts they accepted (0 ..= 4).
    pub accepted_counts: [usize; 5],
    pub states: usize,
}

/// Checks one record bit for bit against the dense oracle: every target logit row, every
/// drafter row, every chain row, every draft draw and the produced tokens.
pub fn check_record(
    oracle: &QwenDenseOracle,
    cache: &mut DenseCache,
    r: &QwenRowRecord,
    sum: &mut Checked,
) {
    let dense = cache.get(oracle, &r.tokens);
    let ctx = format!(
        "slot {} context {} n {}",
        r.slot, r.context_len, r.num_tokens
    );
    for (pos, l) in &r.target_logits {
        assert_eq!(
            bits(l),
            bits(dense.target(*pos)),
            "{ctx}: target logits at {pos}"
        );
        sum.target_rows += 1;
    }
    for (s, l) in &r.drafter_logits {
        let exp = dense
            .drafter(*s)
            .unwrap_or_else(|| panic!("{ctx}: no dense drafter row at {s}"));
        assert_eq!(bits(l), bits(exp), "{ctx}: drafter logits at {s}");
        sum.drafter_rows += 1;
    }
    sum.records += 1;
    let p = r.context_len + r.num_tokens - 1;
    let k = r.drafts.len();
    if k > 0 {
        let chain = oracle
            .chain(&dense, &r.tokens, p, &r.drafts[..k - 1])
            .unwrap();
        assert_eq!(r.chain_logits.len(), k, "{ctx}: one chain row per draft");
        for (i, (got, want)) in r.chain_logits.iter().zip(&chain).enumerate() {
            assert_eq!(bits(got), bits(want), "{ctx}: chain step {i}");
            sum.chain_rows += 1;
        }
    }
    if r.produced.is_empty() {
        return;
    }
    let params = &r.sampling;
    let logits = |row| Logits::new(row, oracle.sampleable_vocab_size);
    let expected = if k == 0 {
        vec![sampling::sample(
            logits(dense.target(p)),
            params,
            p as u64 + 1,
        )]
    } else {
        let mut q_rows = Vec::new();
        for (i, &drafted) in r.drafts.iter().enumerate() {
            let drafter = logits(&r.chain_logits[i]);
            let pos = p as u64 + 1 + i as u64;
            let (d, q) = if params.is_greedy() {
                let d = sampling::argmax(drafter);
                let mut q = vec![0.0; drafter.len()];
                q[d as usize] = 1.0;
                (d, q)
            } else {
                let q = sampling::processed_probs(drafter, params);
                (
                    sampling::sample_from(&q, sampling::uniform(params.seed(), pos, Stream::Draft)),
                    q,
                )
            };
            assert_eq!(d, drafted, "{ctx}: draft {i}");
            q_rows.push(q);
        }
        let p_rows: Vec<Vec<f64>> = (0..=k as u32)
            .map(|i| sampling::processed_probs(logits(dense.target(p + i)), params))
            .collect();
        let out = sampling::chain_accept(&p_rows, &q_rows, &r.drafts, params, p as u64 + 1);
        sum.drafted += k;
        sum.accepted += out.len() - 1;
        sum.accepted_counts[out.len() - 1] += 1;
        out
    };
    assert_eq!(r.produced, expected, "{ctx}: produced tokens");
}

/// Checks a stored state against the dense forward's states at its boundary, bit for bit.
pub fn check_stored(oracle: &QwenDenseOracle, s: &StoredState) {
    let want = oracle.states_at(&s.tokens, s.len as usize).unwrap();
    assert_eq!(s.states.len(), want.len());
    for (l, (got, want)) in s.states.iter().zip(&want).enumerate() {
        assert_eq!(
            bits(&got.conv),
            bits(&want.conv),
            "slot {}: layer {l} convolution history at {}",
            s.slot,
            s.len
        );
        assert_eq!(
            bits(&got.recurrent),
            bits(&want.recurrent),
            "slot {}: layer {l} recurrent state at {}",
            s.slot,
            s.len
        );
    }
}

pub struct Harness {
    pub eng: Engine<QwenCpuExecutor>,
    pub spec: ModelSpec,
    pub model: Arc<QwenModel>,
    pub drafting: bool,
    pub now: u64,
    pub outputs: HashMap<u64, Vec<u32>>,
    pub finished: HashMap<u64, FinishReason>,
    pub cached: HashMap<u64, u32>,
    pub prompts: HashMap<u64, (Vec<u32>, SamplingParams)>,
    pub records: Vec<QwenRowRecord>,
    pub stored: Vec<StoredState>,
    /// Run the full memory scan after every step.
    pub checks: bool,
}

impl Harness {
    pub fn new(model: Arc<QwenModel>, cfg: QwenCpuExecutorConfig, sched: SchedulerConfig) -> Self {
        let drafting = cfg.draft_tokens > 0;
        let exec = QwenCpuExecutor::new(model.clone(), cfg);
        let spec = eidola_engine::executor::Executor::spec(&exec).clone();
        Self {
            eng: Engine::new(exec, sched).expect("valid configuration"),
            spec,
            model,
            drafting,
            now: 0,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
            prompts: HashMap::new(),
            records: Vec::new(),
            stored: Vec::new(),
            checks: true,
        }
    }

    pub fn fixture(blocks: u32, buffers: u32, k: u32, sched: SchedulerConfig) -> Self {
        Self::new(fixture(), exec_config(blocks, buffers, k), sched)
    }

    pub fn oracle(&self) -> QwenDenseOracle<'_> {
        QwenDenseOracle {
            model: &self.model,
            drafting: self.drafting,
            sampleable_vocab_size: self.spec.sampleable_vocab_size,
        }
    }

    pub fn submit(&mut self, req: Request) {
        self.outputs.insert(req.id, Vec::new());
        self.prompts
            .insert(req.id, (req.prompt.clone(), req.sampling));
        self.eng.submit(req).expect("submit");
    }

    /// The memory scan: invariants, zero on free (blocks and state buffers alike), reset
    /// slots holding nothing, idle blocks and snapshots ciphertext, the host tier.
    pub fn check(&self) {
        if !self.checks {
            return;
        }
        let kv = self.eng.kv();
        let exec = self.eng.executor();
        kv.check_invariants();
        kv.check_no_reservations();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    exec.block_is_zero(g, b),
                    "free block {g}/{b} holds data with no zero pending"
                );
            }
        }
        let resets = kv.pending_slot_resets();
        for slot in kv.free_slot_set() {
            if !resets.contains(&slot) {
                assert!(
                    exec.slot_state_is_zero(slot),
                    "free slot {slot} holds model state with no reset pending"
                );
            }
        }
        let idle = kv.idle_blocks();
        let crypts = kv.pending_crypts();
        let transfers = kv.pending_transfers();
        let landing: std::collections::BTreeSet<(u32, u32)> = transfers
            .iter()
            .filter(|t| !t.3)
            .map(|t| (t.0, t.1))
            .collect();
        let demoting: std::collections::BTreeSet<(u32, u32)> = transfers
            .iter()
            .filter(|t| t.3)
            .map(|t| (t.0, t.2))
            .collect();
        let host_zeros = kv.pending_host_zeros();
        for (g, hb) in kv.free_host_set() {
            if !host_zeros.contains(&(g, hb)) {
                assert!(
                    exec.host_block_is_zero(g, hb),
                    "free host block {g}/{hb} holds data with no zero pending"
                );
            }
        }
        for (g, hb) in kv.host_cache_blocks() {
            if !demoting.contains(&(g, hb)) {
                assert!(
                    exec.host_block_is_encrypted(g, hb),
                    "held host block {g}/{hb} holds no ciphertext"
                );
            }
        }
        for (g, b) in kv.cache_blocks().union(&kv.sequence_blocks()) {
            if landing.contains(&(*g, *b)) {
                continue;
            }
            let want = idle.contains(&(*g, *b));
            let device = exec.block_is_encrypted(*g, *b);
            match crypts.get(&(*g, *b)) {
                Some(&encrypt) => assert!(
                    encrypt == want && device != want,
                    "block {g}/{b}: queued crypt disagrees with its state"
                ),
                None => assert_eq!(device, want, "block {g}/{b}: encryption != idle"),
            }
        }
    }

    pub fn flush(&mut self) {
        self.eng.flush().expect("flush");
        self.stored.extend(self.eng.executor().take_stored());
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
        self.stored.extend(self.eng.executor().take_stored());
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

    /// Runs one request to completion, flushes, and returns its output.
    pub fn run_one(&mut self, req: Request) -> Vec<u32> {
        let id = req.id;
        self.submit(req);
        self.run();
        self.flush();
        self.outputs[&id].clone()
    }

    /// Checks every finished output against dense generation (exactly when speculation is
    /// off or the request is greedy), every recorded row and every stored state against
    /// the dense oracle, bit for bit, and returns the counters.
    pub fn check_all(&mut self, spec_on: bool) -> Checked {
        let mut cache = DenseCache::default();
        let oracle = self.oracle();
        let mut ids: Vec<u64> = self.finished.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            if self.finished[&id] == FinishReason::Cancelled {
                continue;
            }
            let (prompt, p) = &self.prompts[&id];
            let out = &self.outputs[&id];
            let mut all = prompt.clone();
            all.extend(out);
            let dense = cache.get(&oracle, &all);
            if (!spec_on || p.is_greedy())
                && let Err(pos) = qwen_check_generation(
                    &dense,
                    oracle.sampleable_vocab_size,
                    prompt.len(),
                    out,
                    p,
                )
            {
                panic!("request {id}: output diverges from dense generation at {pos}");
            }
        }
        let mut sum = Checked::default();
        for r in &self.records {
            check_record(&oracle, &mut cache, r, &mut sum);
        }
        for s in &self.stored {
            check_stored(&oracle, s);
            sum.states += 1;
        }
        self.records.clear();
        self.stored.clear();
        sum
    }
}
