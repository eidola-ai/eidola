//! The CUDA executor beside the CPU reference executor, driven by the real engine: every
//! step goes to both, and must agree in its tokens, in every produced token's logits bit
//! for bit, in every state buffer's image byte for byte (plaintext and ciphertext alike:
//! both executors lay a state image out the same way) and in which blocks and buffers are
//! encrypted. After every step the memory scan runs on the device executor: free blocks
//! and buffers zero, free slots empty, idle blocks ciphertext and mapped ones plaintext,
//! the host tier holding only ciphertext.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cpu::qwen::{QwenCpuExecutor, QwenCpuExecutorConfig};
use eidola_engine_cuda::qwen::{
    CudaGraphs, QwenCudaExecutor, QwenCudaExecutorConfig, QwenDevice, StateStrategy,
};
use eidola_engine_model::qwen::{QwenModel, load_reference};

pub const VOCAB: u32 = 500;
pub const SAMPLEABLE: u32 = 508;

/// The model crate's committed synthetic fixture: `qwen_tiny`, or with
/// [`super::with_fixture_256`] `qwen_tiny256` (the released attention geometry, which the
/// fast profile's attention kernels are compiled for).
pub fn fixture() -> Arc<QwenModel> {
    static MODEL: OnceLock<Arc<QwenModel>> = OnceLock::new();
    static MODEL_256: OnceLock<Arc<QwenModel>> = OnceLock::new();
    let (cell, name) = if super::fixture_256() {
        (&MODEL_256, "qwen_tiny256")
    } else {
        (&MODEL, "qwen_tiny")
    };
    cell.get_or_init(|| Arc::new(load_reference(&fixture_dir(name), Ok).expect("the fixture")))
        .clone()
}

pub fn fixture_dir(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../eidola-engine-model/tests/fixtures")
        .join(name)
}

/// Deterministic test RNG (SplitMix64).
pub struct TestRng(pub u64);

impl TestRng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        eidola_engine::sampling::mix64(self.0)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn tokens(&mut self, n: usize, vocab: u32) -> Vec<u32> {
        (0..n)
            .map(|_| u32::try_from(self.below(u64::from(vocab))).expect("a token"))
            .collect()
    }
}

pub struct Geometry {
    pub blocks: u32,
    pub buffers: u32,
    pub host: u32,
    pub host_buffers: u32,
    pub k: u32,
    pub strategy: StateStrategy,
    pub graphs: CudaGraphs,
}

impl Geometry {
    pub fn plain(k: u32) -> Self {
        Geometry {
            blocks: 512,
            buffers: 128,
            host: 0,
            host_buffers: 0,
            k,
            strategy: StateStrategy::Live,
            graphs: super::graphs_now(),
        }
    }
}

fn buckets() -> Vec<Bucket> {
    vec![
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
    ]
}

pub fn cpu_config(g: &Geometry) -> QwenCpuExecutorConfig {
    QwenCpuExecutorConfig {
        block_size: super::block_size_now(),
        num_blocks: g.blocks,
        host_blocks: g.host,
        state_buffers: g.buffers,
        host_state_buffers: g.host_buffers,
        num_state_slots: 32,
        max_model_len: 512,
        buckets: buckets(),
        draft_tokens: g.k,
        sampleable_vocab_size: SAMPLEABLE,
        record: false,
    }
}

pub fn cuda_config(g: &Geometry) -> QwenCudaExecutorConfig {
    QwenCudaExecutorConfig {
        block_size: super::block_size_now(),
        num_blocks: g.blocks,
        host_blocks: g.host,
        state_buffers: g.buffers,
        host_state_buffers: g.host_buffers,
        num_state_slots: 32,
        max_model_len: 512,
        buckets: buckets(),
        draft_tokens: g.k,
        sampleable_vocab_size: SAMPLEABLE,
        strategy: g.strategy,
        graphs: g.graphs,
    }
}

pub fn sched(chunk: u32, speculative: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: 256,
        max_seqs: 32,
        max_prefill_chunk: chunk,
        eos_token_ids: Vec::new(),
        speculative,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    }
}

pub fn salt(n: u8) -> Lineage {
    SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32])).lineage(&CacheKey::from_bytes([n; 32]))
}

pub fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

pub fn request(
    id: u64,
    prompt: Vec<u32>,
    sampling: SamplingParams,
    max_tokens: u32,
    cache: CacheScope,
) -> Request {
    Request {
        id,
        prompt,
        sampling,
        max_tokens,
        required_cached_tokens: 0,
        stop_token_ids: Vec::new(),
        cache,
    }
}

pub fn params(i: u64) -> SamplingParams {
    match i % 3 {
        0 => SamplingParams::greedy(),
        1 => SamplingParams::random(1.0, 1000 + i).expect("params"),
        _ => SamplingParams::new(0.7, 8, 0.9, 0.05, 2000 + i).expect("params"),
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Both executors, step for step.
pub struct Pair<D: QwenDevice> {
    pub cpu: QwenCpuExecutor,
    pub cuda: QwenCudaExecutor<D>,
    pub steps: u64,
    pub compared_logits: u64,
    pub compared_images: u64,
    /// Drafting rows by the tokens they produced (index 1 to `k + 1`).
    pub accepted: [u64; 8],
}

impl<D: QwenDevice> Executor for Pair<D> {
    fn spec(&self) -> &ModelSpec {
        self.cpu.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        self.steps += 1;
        let mut s = step.clone();
        s.return_logits = true;
        let a = self.cpu.execute(&s)?;
        let before = self.cuda.decode_stats();
        let b = self.cuda.execute(&s)?;
        let after = self.cuda.decode_stats();
        super::note_rung_steps(after.direct + after.replay - before.direct - before.replay);
        let n = self.steps;
        assert_eq!(a.num_tokens, b.num_tokens, "step {n}: token counts");
        assert_eq!(a.tokens, b.tokens, "step {n}: tokens");
        for (e, &m) in step.seqs.iter().zip(&a.num_tokens) {
            if e.sample && e.num_drafts > 0 {
                self.accepted[m as usize] += 1;
            }
        }
        let (la, lb) = (
            a.logits.as_ref().expect("logits"),
            b.logits.as_ref().expect("logits"),
        );
        for (r, (x, y)) in la.iter().zip(lb).enumerate() {
            for (j, (u, v)) in x.iter().zip(y).enumerate() {
                assert_eq!(
                    bits(u),
                    bits(v),
                    "step {n} row {r}: logits of produced token {j}"
                );
                self.compared_logits += 1;
            }
        }
        let pool = self.cuda.state_pool_index();
        for buf in 1..self.cpu.spec().state.as_ref().expect("state").num_buffers {
            assert_eq!(
                self.cpu.block_is_encrypted(pool, buf),
                self.cuda.block_is_encrypted(pool, buf),
                "step {n}: state buffer {buf} encryption"
            );
            let (ia, ib) = (self.cpu.block_image(pool, buf), self.cuda.state_image(buf));
            if ia.iter().any(|&w| w != 0) || ib.iter().any(|&w| w != 0) {
                assert!(
                    ia == ib,
                    "step {n}: state buffer {buf} image differs from the reference executor's"
                );
                self.compared_images += 1;
            }
        }
        for g in 0..u32::try_from(self.cpu.spec().kv_groups.len()).expect("groups") {
            for b in 1..self.cpu.spec().kv_groups[g as usize].num_blocks {
                assert_eq!(
                    self.cpu.block_is_encrypted(g, b),
                    self.cuda.block_is_encrypted(g, b),
                    "step {n}: group {g} block {b} encryption"
                );
            }
        }
        let mut out = a;
        if !step.return_logits {
            out.logits = None;
        }
        Ok(out)
    }
}

pub struct Harness<D: QwenDevice> {
    pub eng: Engine<Pair<D>>,
    pub now: u64,
    pub outputs: HashMap<u64, Vec<u32>>,
    pub finished: HashMap<u64, FinishReason>,
    /// Prompt positions each finished request was served from the cache.
    pub cached: HashMap<u64, u32>,
}

impl<D: QwenDevice> Harness<D> {
    pub fn new(dev: D, g: &Geometry, sched: SchedulerConfig) -> Self {
        Self::with_configs(dev, cpu_config(g), cuda_config(g), sched)
    }

    /// Both executors from configurations of their own, which must describe one geometry.
    pub fn with_configs(
        dev: D,
        cpu: QwenCpuExecutorConfig,
        cuda: QwenCudaExecutorConfig,
        sched: SchedulerConfig,
    ) -> Self {
        let model = fixture();
        let cpu = QwenCpuExecutor::new(model.clone(), cpu);
        let mut cuda = QwenCudaExecutor::new(dev, &model, cuda).expect("the executor");
        super::apply_path(&mut cuda);
        assert_eq!(
            cpu.spec(),
            cuda.spec(),
            "both executors give the engine one spec"
        );
        let pair = Pair {
            cpu,
            cuda,
            steps: 0,
            compared_logits: 0,
            compared_images: 0,
            accepted: [0; 8],
        };
        Harness {
            eng: Engine::new(pair, sched).expect("a valid configuration"),
            now: 0,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
        }
    }

    pub fn submit(&mut self, req: Request) {
        self.outputs.insert(req.id, Vec::new());
        self.eng.submit(req).expect("submit");
    }

    /// The memory scan, on the device executor.
    pub fn check(&self) {
        let kv = self.eng.kv();
        let ex = &self.eng.executor().cuda;
        kv.check_invariants();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    ex.block_is_zero(g, b),
                    "free block {g}/{b} holds data with no zero pending"
                );
            }
        }
        let resets = kv.pending_slot_resets();
        for slot in kv.free_slot_set() {
            if !resets.contains(&slot) {
                assert!(
                    ex.slot_state_is_zero(slot),
                    "free slot {slot} holds state with no reset pending"
                );
            }
        }
        let host_zeros = kv.pending_host_zeros();
        for (g, hb) in kv.free_host_set() {
            if !host_zeros.contains(&(g, hb)) {
                assert!(
                    ex.host_block_is_zero(g, hb),
                    "free host block {g}/{hb} holds data"
                );
            }
        }
        let demoting: std::collections::BTreeSet<(u32, u32)> = kv
            .pending_transfers()
            .iter()
            .filter(|t| t.3)
            .map(|t| (t.0, t.2))
            .collect();
        for (g, hb) in kv.host_cache_blocks() {
            if !demoting.contains(&(g, hb)) {
                assert!(
                    ex.host_block_is_encrypted(g, hb),
                    "held host block {g}/{hb} holds no ciphertext"
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
        self.check();
    }

    pub fn run(&mut self) {
        let mut guard = 0;
        while self.eng.unfinished() > 0 {
            self.step();
            self.now += 1;
            guard += 1;
            assert!(guard < 100_000, "no progress");
        }
    }

    pub fn flush(&mut self) {
        self.eng.flush().expect("flush");
        self.check();
    }

    pub fn run_one(&mut self, req: Request) -> Vec<u32> {
        let id = req.id;
        self.submit(req);
        self.run();
        self.flush();
        self.outputs[&id].clone()
    }

    pub fn pair(&self) -> &Pair<D> {
        self.eng.executor()
    }
}
