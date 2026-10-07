//! Shared harness for the simulation tests.

#![allow(dead_code)]

use std::collections::HashMap;

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::mock::{MockConfig, MockExecutor, mimo_like_spec, reference_generate};
use eidola_engine::sampling::{SamplingParams, mix64};
use eidola_engine::secret::EngineSalt;
use eidola_engine::spec::ModelSpec;

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

pub struct Harness {
    pub eng: Engine<MockExecutor>,
    pub spec: ModelSpec,
    pub cfg: MockConfig,
    pub now: u64,
    pub outputs: HashMap<u64, Vec<u32>>,
    pub finished: HashMap<u64, FinishReason>,
    pub cached: HashMap<u64, u32>,
    pub eos: Vec<u32>,
    /// Run the full invariant scan after every step.
    pub checks: bool,
}

impl Harness {
    pub fn new(spec: ModelSpec, sched: SchedulerConfig, cfg: MockConfig) -> Self {
        let eos = sched.eos_token_ids.clone();
        Self {
            eng: Engine::new(MockExecutor::new(spec.clone(), cfg), sched)
                .expect("valid configuration"),
            spec,
            cfg,
            now: 0,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
            eos,
            checks: true,
        }
    }

    pub fn submit(&mut self, req: Request) {
        self.outputs.insert(req.id, Vec::new());
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

    pub fn expected(
        &self,
        prompt: &[u32],
        params: &SamplingParams,
        max_tokens: u32,
        stop: &[u32],
    ) -> Vec<u32> {
        let mut all_stop = stop.to_vec();
        all_stop.extend(&self.eos);
        reference_generate(&self.spec, &self.cfg, prompt, params, max_tokens, &all_stop)
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

/// vocab 32, block 4, SWA window 6, drafter window 10, `blocks` per group, 32 slots, k = 3.
pub fn small_spec(blocks: u32) -> ModelSpec {
    mimo_like_spec(32, 4, 6, 10, blocks, 32, 3)
}
