//! Shared harness: the serving core's engine over the CPU executor on the committed
//! synthetic MiMo fixture, with every row checked against the dense oracle.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::{self, SamplingParams, Stream, mix64};
use eidola_engine::secret::EngineSalt;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cpu::oracle::check_generation;
use eidola_engine_cpu::{
    CpuExecutor, CpuExecutorConfig, DenseOracle, DenseRun, MtpHidden, RowRecord,
};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{LoadOptions, ModelConfig, ModelWeights, ReferenceModel};

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

pub fn salt(n: u8) -> EngineSalt {
    EngineSalt::from_bytes([n; 32])
}

pub fn load_fixture(opts: &LoadOptions) -> ReferenceModel {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../eidola-engine-model/tests/fixtures/tiny");
    let config = ModelConfig::from_file(&dir.join("config.json")).unwrap();
    let store = Arc::new(WeightSet::open_files(&[dir.join("model.safetensors")]).unwrap());
    ReferenceModel::new(ModelWeights::load(store, config, opts).unwrap())
}

/// The committed synthetic fixture: 4 layers (global dense, two sliding MoE, global MoE),
/// sliding window 8, two MTP layers, vocab 256.
pub fn fixture() -> Arc<ReferenceModel> {
    static MODEL: OnceLock<Arc<ReferenceModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| Arc::new(load_fixture(&LoadOptions::default())))
        .clone()
}

pub const VOCAB: u32 = 256;

pub fn exec_config(blocks: u32, mtp_depths: Vec<usize>) -> CpuExecutorConfig {
    CpuExecutorConfig {
        block_size: 4,
        num_blocks: blocks,
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
        mtp_depths,
        mtp_hidden: MtpHidden::Normed,
        pad_batches: false,
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
        stop_token_ids: Vec::new(),
        cache,
    }
}

pub fn params(i: u64) -> SamplingParams {
    match i % 3 {
        0 => SamplingParams::greedy(),
        1 => SamplingParams::random(1.0, 1000 + i),
        _ => SamplingParams {
            temperature: 0.7,
            top_k: 8,
            top_p: 0.9,
            min_p: 0.05,
            seed: 2000 + i,
        },
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Dense runs keyed by token sequence; any extension of a record's tokens serves it, since
/// every logit depends only on its prefix.
#[derive(Default)]
pub struct DenseCache {
    runs: Vec<(Vec<u32>, Arc<DenseRun>)>,
    pub computed: usize,
}

impl DenseCache {
    pub fn get(&mut self, oracle: &DenseOracle, tokens: &[u32]) -> Arc<DenseRun> {
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
    pub drafted: usize,
    pub accepted: usize,
    /// Per draft depth: (proposed, accepted).
    pub per_depth: [(usize, usize); 4],
}

/// Checks one record bit for bit against the dense oracle: every target logit row, every
/// drafter logit row, every draft draw, and the produced tokens.
pub fn check_record(
    oracle: &DenseOracle,
    cache: &mut DenseCache,
    r: &RowRecord,
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
    for (d, s, l) in &r.drafter_logits {
        let exp = dense
            .drafter(*d, *s)
            .unwrap_or_else(|| panic!("{ctx}: no dense drafter row {d} at slot {s}"));
        assert_eq!(
            bits(l),
            bits(exp),
            "{ctx}: drafter depth {d} logits at slot {s}"
        );
        sum.drafter_rows += 1;
    }
    sum.records += 1;
    if r.produced.is_empty() {
        return;
    }
    let p = r.context_len + r.num_tokens - 1;
    let params = &r.sampling;
    let expected = if r.drafts.is_empty() {
        vec![sampling::sample(dense.target(p), params, p as u64 + 1)]
    } else {
        let k = r.drafts.len();
        let mut q_rows = Vec::new();
        for i in 0..k {
            let logits = dense.drafter(i, p + i as u32).expect("draft row");
            let pos = p as u64 + 1 + i as u64;
            let (d, q) = if params.is_greedy() {
                let d = sampling::argmax(logits);
                let mut q = vec![0.0; logits.len()];
                q[d as usize] = 1.0;
                (d, q)
            } else {
                let q = sampling::processed_probs(logits, params);
                (
                    sampling::sample_from(&q, sampling::uniform(params.seed, pos, Stream::Draft)),
                    q,
                )
            };
            assert_eq!(d, r.drafts[i], "{ctx}: draft {i}");
            q_rows.push(q);
        }
        let p_rows: Vec<Vec<f64>> = (0..=k as u32)
            .map(|i| sampling::processed_probs(dense.target(p + i), params))
            .collect();
        let out = sampling::chain_accept(&p_rows, &q_rows, &r.drafts, params, p as u64 + 1);
        sum.drafted += k;
        sum.accepted += out.len() - 1;
        for i in 0..k {
            sum.per_depth[i].0 += 1;
            if i < out.len() - 1 {
                sum.per_depth[i].1 += 1;
            }
        }
        out
    };
    assert_eq!(r.produced, expected, "{ctx}: produced tokens");
}

pub struct Harness {
    pub eng: Engine<CpuExecutor>,
    pub spec: ModelSpec,
    pub model: Arc<ReferenceModel>,
    pub mtp_depths: Vec<usize>,
    pub mtp_hidden: MtpHidden,
    pub now: u64,
    pub outputs: HashMap<u64, Vec<u32>>,
    pub finished: HashMap<u64, FinishReason>,
    pub cached: HashMap<u64, u32>,
    pub prompts: HashMap<u64, (Vec<u32>, SamplingParams)>,
    pub records: Vec<RowRecord>,
    /// Run the full KV invariant scan after every step.
    pub checks: bool,
}

impl Harness {
    pub fn new(model: Arc<ReferenceModel>, cfg: CpuExecutorConfig, sched: SchedulerConfig) -> Self {
        let mtp_depths = cfg.mtp_depths.clone();
        let mtp_hidden = cfg.mtp_hidden;
        let exec = CpuExecutor::new(model.clone(), cfg);
        let spec = eidola_engine::executor::Executor::spec(&exec).clone();
        Self {
            eng: Engine::new(exec, sched),
            spec,
            model,
            mtp_depths,
            mtp_hidden,
            now: 0,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
            prompts: HashMap::new(),
            records: Vec::new(),
            checks: true,
        }
    }

    pub fn fixture(blocks: u32, mtp_depths: Vec<usize>, sched: SchedulerConfig) -> Self {
        let m = fixture();
        let cfg = exec_config(blocks, mtp_depths);
        Self::new(m, cfg, sched)
    }

    pub fn oracle(&self) -> DenseOracle<'_> {
        DenseOracle {
            model: &self.model,
            mtp_depths: &self.mtp_depths,
            mtp_hidden: self.mtp_hidden,
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
        kv.check_invariants();
        kv.check_no_reservations();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    self.eng.executor().block_is_zero(g, b),
                    "free block {g}/{b} holds data with no zero pending"
                );
            }
        }
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

    /// Checks every finished request's output against dense generation (exactly when
    /// speculation is off or the request is greedy) and returns the dense cache seeded
    /// with every full transcript.
    pub fn check_outputs(&self, spec_on: bool, cache: &mut DenseCache) {
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
            let dense = cache.get(&oracle, &all[..all.len().max(1)]);
            if (!spec_on || p.is_greedy())
                && let Err(pos) = check_generation(&dense, prompt.len(), out, p)
            {
                panic!("request {id}: output diverges from dense generation at {pos}");
            }
        }
    }
}

impl Harness {
    /// Checks every finished output and every recorded row against the dense oracle, bit
    /// for bit, and returns the counters.
    pub fn check_all(&mut self, spec_on: bool) -> Checked {
        let mut cache = DenseCache::default();
        self.check_outputs(spec_on, &mut cache);
        let records = std::mem::take(&mut self.records);
        let oracle = self.oracle();
        let mut sum = Checked::default();
        for r in &records {
            check_record(&oracle, &mut cache, r, &mut sum);
        }
        sum
    }
}

/// The fixture cut down to its first `vocab` token ids (embedding and head rows), with the
/// head scaled by `head_scale`, and only the listed layers: a model whose distributions
/// have few enough outcomes for statistics and drafts that often agree with the target.
pub fn small_vocab_fixture(vocab: usize, head_scale: f32, layers: &[usize]) -> Arc<ReferenceModel> {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../eidola-engine-model/tests/fixtures/tiny");
    let config = ModelConfig::from_file(&dir.join("config.json"))
        .unwrap()
        .truncated(layers)
        .unwrap();
    let store = Arc::new(WeightSet::open_files(&[dir.join("model.safetensors")]).unwrap());
    let mut w = ModelWeights::load(store, config, &LoadOptions::default()).unwrap();
    let rows: Vec<usize> = (0..vocab).collect();
    w.embed = w.embed.select_rows(&rows);
    w.lm_head = w.lm_head.select_rows(&rows);
    for x in &mut w.lm_head.data {
        *x *= head_scale;
    }
    w.config.vocab_size = vocab;
    Arc::new(ReferenceModel::new(w))
}

/// [`small_vocab_fixture`] whose MTP layers echo the hidden state they continue from:
/// `eh_proj = [0 | I]`, unit `hnorm` and `final_layernorm`, and the attention and FFN
/// residual branches scaled by `branch_scale` (still computed, through the paged drafter
/// KV, but small). Each draft then repeats the target's last prediction, which a greedy
/// target that settles into a repeating token accepts, so whole chains get accepted.
pub fn echo_drafter_fixture(
    vocab: usize,
    layers: &[usize],
    branch_scale: f32,
) -> Arc<ReferenceModel> {
    let m = Arc::try_unwrap(small_vocab_fixture(vocab, 1.0, layers))
        .ok()
        .unwrap();
    let mut w = m.weights;
    let h = w.config.hidden_size;
    for mw in &mut w.mtp {
        mw.eh_proj.data.fill(0.0);
        for i in 0..h {
            mw.eh_proj.data[i * 2 * h + h + i] = 1.0;
        }
        mw.hnorm.fill(1.0);
        mw.final_norm.fill(1.0);
        for x in &mut mw.attention.o_proj.data {
            *x *= branch_scale;
        }
        for x in &mut mw.ffn.down.data {
            *x *= branch_scale;
        }
    }
    Arc::new(ReferenceModel::new(w))
}
