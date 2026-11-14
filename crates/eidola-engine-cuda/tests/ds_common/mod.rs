//! Shared harness of the DeepSeek-V4.1 executor's tests: the serving core's engine over
//! [`Paired`], which gives every step to the executor under test and to the CPU
//! reference executor and holds them equal, and the scenarios both the host run
//! (`ds_host.rs`) and the device run (`ds_tiny.rs`) drive.
//!
//! Equal means: the same tokens; the same logits, bit for bit, at every sampling row
//! (every step asks for them); the same bytes in every plaintext block of every group,
//! compared in the reference executor's cell order; the same blocks held encrypted; the
//! same set crossing between the two halves; and the same chain work. The CPU suites
//! hold the reference executor to the dense oracle, so nothing here repeats that.

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::{SamplingParams, mix64};
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cpu::deepseek::{DsCpuExecutor, DsCpuExecutorConfig};
use eidola_engine_cuda::ds::half::Device;
use eidola_engine_cuda::ds::{DsCudaExecutor, DsCudaExecutorConfig, StepStats};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::{DsConfig, DsLoadOptions, DsModel, config_of, load_reference};

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
        let n = min + usize::try_from(self.below(extra)).unwrap();
        self.tokens(n)
    }
    pub fn tokens(&mut self, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| u32::try_from(self.below(u64::from(VOCAB))).unwrap())
            .collect()
    }
}

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

/// Token ids a request may hold or produce: the head's last row is the image token.
pub const VOCAB: u32 = 95;
/// The committed fixture's window, and the wide configuration's.
pub const WINDOW: u32 = 48;
pub const WIDE_WINDOW: u32 = 80;

/// A model under test: the configuration, the stored weights the executor loads, the
/// Engram map, and the same model as the reference executor holds it.
pub struct Model {
    pub name: &'static str,
    pub config: DsConfig,
    pub store: Arc<WeightSet>,
    pub token_map: Vec<u32>,
    pub reference: Arc<DsModel>,
    pub window: u32,
}

/// The committed fixture, or (with `wide`) its weights under a window of 80, which no
/// longer fits one 64-slot attention block.
pub fn model(wide: bool) -> Model {
    let dir = fixture_dir();
    let golden = WeightSet::open_files(&[dir.join("golden.safetensors")]).unwrap();
    let name = "engram.token_map";
    let token_map: Vec<u32> = golden
        .get(name)
        .unwrap()
        .to_i64(name)
        .unwrap()
        .into_iter()
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    let window = if wide { WIDE_WINDOW } else { WINDOW };
    let widen = move |mut c: DsConfig| {
        c.attention.window = window as usize;
        Ok(c)
    };
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let config = widen(config_of(&store).unwrap()).unwrap();
    let reference = load_reference(
        &dir,
        widen,
        &DsLoadOptions {
            skip_drafter: true,
            ..DsLoadOptions::default()
        },
        Some(token_map.clone()),
    )
    .unwrap();
    Model {
        name: if wide { "wide" } else { "tiny" },
        config,
        store,
        token_map,
        reference: Arc::new(reference),
        window,
    }
}

pub fn buckets() -> Vec<Bucket> {
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
            max_seqs: 16,
            max_tokens: 512,
        },
    ]
}

pub fn sched() -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: 512,
        max_seqs: 16,
        max_prefill_chunk: 512,
        eos_token_ids: Vec::new(),
        speculative: false,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    }
}

pub fn chunked(chunk: u32) -> SchedulerConfig {
    SchedulerConfig {
        max_prefill_chunk: chunk,
        ..sched()
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

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// What a run compared.
#[derive(Clone, Copy, Debug, Default)]
pub struct Compared {
    pub steps: u64,
    /// Positions computed.
    pub rows: u64,
    /// Logit rows equal bit for bit.
    pub logit_rows: u64,
    /// Plaintext blocks equal byte for byte, summed over steps.
    pub blocks: u64,
    /// Rows whose chain started after the row did.
    pub triangles: u64,
    pub chain_work: u64,
}

/// The executor under test beside the CPU reference executor, given the same steps.
pub struct Paired {
    pub dev: DsCudaExecutor,
    pub cpu: DsCpuExecutor,
    pub compared: Compared,
    /// Every step's timings on the executor under test.
    pub stats: Vec<StepStats>,
    /// The device's pools after the last step: per group, per block, its cells.
    pools: Vec<Vec<Vec<u32>>>,
    /// Per sampling row since the last take: its position and its logits' bits.
    pub sampled: Vec<(u32, Vec<u32>)>,
}

impl Paired {
    pub fn new(m: &Model, device: &dyn Fn() -> Box<dyn Device>, blocks: u32) -> Paired {
        let cfg = DsCudaExecutorConfig {
            block_size: 8,
            num_blocks: blocks,
            num_state_slots: 16,
            max_model_len: 1024,
            buckets: buckets(),
            sampleable_vocab_size: VOCAB,
            devices: [0, 0],
            record: true,
        };
        let dev = DsCudaExecutor::with_devices(
            device(),
            device(),
            &m.store,
            &m.config,
            Some(m.token_map.clone()),
            &cfg,
        )
        .expect("the executor under test");
        let cpu = DsCpuExecutor::new(
            m.reference.clone(),
            DsCpuExecutorConfig {
                block_size: cfg.block_size,
                num_blocks: blocks,
                host_blocks: 0,
                num_state_slots: cfg.num_state_slots,
                max_model_len: cfg.max_model_len,
                buckets: buckets(),
                draft_width: 0,
                sampleable_vocab_size: VOCAB,
                record: true,
            },
        );
        assert_eq!(dev.spec(), cpu.spec(), "the two executors report one model");
        let groups = cpu.spec().kv_groups.len();
        Paired {
            dev,
            cpu,
            compared: Compared::default(),
            stats: Vec::new(),
            pools: vec![Vec::new(); groups],
            sampled: Vec::new(),
        }
    }

    /// Whether a block held only zeros on the device after the last step.
    pub fn block_is_zero(&self, group: u32, block: u32) -> bool {
        // Before the first step nothing has been written.
        self.pools[group as usize]
            .get(block as usize)
            .is_none_or(|cells| cells.iter().all(|w| *w == 0))
    }

    fn compare_memory(&mut self) {
        let groups = self.cpu.spec().kv_groups.clone();
        for (g, spec) in groups.iter().enumerate() {
            let g32 = u32::try_from(g).unwrap();
            let cells = self.dev.pool_cells(g32).expect("read the pool");
            let words = self.cpu.block_bytes(g32) / 4;
            for block in 0..spec.num_blocks {
                let encrypted = self.cpu.block_is_encrypted(g32, block);
                assert_eq!(
                    self.dev.block_is_encrypted(g32, block),
                    encrypted,
                    "group {} block {block}: held encrypted on one executor only",
                    spec.name
                );
                if encrypted {
                    continue;
                }
                let want = &self.cpu.block_image(g32, block)[..words];
                assert_eq!(
                    cells[block as usize], want,
                    "group {} block {block}: bytes differ from the reference executor's",
                    spec.name
                );
                self.compared.blocks += 1;
            }
            self.pools[g] = cells;
        }
    }
}

impl Executor for Paired {
    fn spec(&self) -> &ModelSpec {
        self.cpu.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        let mut asked = step.clone();
        asked.return_logits = true;
        let work = (
            self.dev.chain_layer_positions(),
            self.cpu.chain_layer_positions(),
        );
        let got = self.dev.execute(&asked)?;
        let want = self.cpu.execute(&asked)?;
        assert_eq!(got.tokens, want.tokens, "tokens");
        assert_eq!(got.num_tokens, want.num_tokens, "token counts");
        assert_eq!(got.stride, want.stride);
        let (a, b) = (got.logits.as_ref().unwrap(), want.logits.as_ref().unwrap());
        assert_eq!(a.len(), b.len());
        for (row, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.len(), b.len(), "row {row}: logit rows");
            for (a, b) in a.iter().zip(b) {
                let e = &step.seqs[row];
                assert_eq!(
                    bits(a),
                    bits(b),
                    "slot {} context {} n {}: logits differ from the reference executor's",
                    e.slot,
                    e.context_len,
                    e.num_tokens
                );
                self.compared.logit_rows += 1;
                self.sampled
                    .push((e.context_len + e.num_tokens - 1, bits(a)));
            }
        }
        assert_eq!(
            self.dev.chain_layer_positions() - work.0,
            self.cpu.chain_layer_positions() - work.1,
            "the chain's work"
        );
        let ours = self.dev.take_transfers();
        // The reference executor records a (blank) transfer for a rowless step too.
        let mut theirs = self.cpu.transfers();
        if step.seqs.is_empty() {
            theirs.retain(|t| !t.bank_entries.is_empty());
        }
        assert_eq!(ours.len(), theirs.len(), "steps with rows");
        for (o, t) in ours.iter().zip(&theirs) {
            assert_eq!(o.gathered, t.gathered, "gathered Engram rows");
            assert_eq!(o.bank_entries, t.bank_entries, "bank entries");
            assert_eq!(o.bank_entry_bytes, t.bank_entry_bytes);
            assert_eq!(o.boundary_rows, t.boundary_rows, "boundary rows");
            assert!(t.drafts.is_empty() && t.taps.is_empty());
        }
        self.cpu.take_records();
        self.compare_memory();
        self.compared.steps += 1;
        self.compared.rows += step
            .seqs
            .iter()
            .map(|e| u64::from(e.num_tokens))
            .sum::<u64>();
        self.compared.triangles += step
            .seqs
            .iter()
            .zip(&step.aux.chain_start)
            .filter(|(e, s)| **s > e.context_len)
            .count() as u64;
        self.compared.chain_work = self.dev.chain_layer_positions();
        self.stats.extend(self.dev.take_stats());
        Ok(StepOutput {
            logits: step.return_logits.then(|| got.logits.clone().unwrap()),
            ..got
        })
    }
}

pub struct Harness {
    pub eng: Engine<Paired>,
    pub spec: ModelSpec,
    pub now: u64,
    pub outputs: HashMap<u64, Vec<u32>>,
    pub finished: HashMap<u64, FinishReason>,
    pub cached: HashMap<u64, u32>,
}

impl Harness {
    pub fn new(
        m: &Model,
        device: &dyn Fn() -> Box<dyn Device>,
        blocks: u32,
        sched: SchedulerConfig,
    ) -> Self {
        let exec = Paired::new(m, device, blocks);
        let spec = exec.spec().clone();
        Self {
            eng: Engine::new(exec, sched).expect("valid configuration"),
            spec,
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

    /// The engine's invariants, zero-on-free on the device's bytes, and a scrubbed
    /// state for every free slot.
    pub fn check(&self) {
        let kv = self.eng.kv();
        let x = self.eng.executor();
        kv.check_invariants();
        kv.check_no_reservations();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    x.block_is_zero(g, b) && x.cpu.block_is_zero(g, b),
                    "free block {g}/{b} holds data with no zero pending"
                );
            }
        }
        let resets = kv.pending_slot_resets();
        for slot in kv.free_slot_set() {
            if !resets.contains(&slot) {
                assert!(
                    x.dev.slot_state_is_zero(slot).unwrap(),
                    "free slot {slot} holds model state with no reset pending"
                );
            }
        }
    }

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

    pub fn compared(&self) -> Compared {
        self.eng.executor().compared
    }
}

/// One scenario's result, for the summary.
pub struct Outcome {
    pub name: String,
    pub compared: Compared,
    pub seconds: f64,
    pub stats: Vec<StepStats>,
}

/// The scenarios, over a model and a device.
pub struct Run<'a> {
    pub model: &'a Model,
    pub device: &'a dyn Fn() -> Box<dyn Device>,
    pub outcomes: Vec<Outcome>,
}

impl Run<'_> {
    fn harness(&self, blocks: u32, sched: SchedulerConfig) -> Harness {
        Harness::new(self.model, self.device, blocks, sched)
    }

    fn record(&mut self, name: &str, started: Instant, harnesses: Vec<Harness>) {
        let mut compared = Compared::default();
        let mut stats = Vec::new();
        for h in harnesses {
            let x = h.eng.executor();
            let c = x.compared;
            compared.steps += c.steps;
            compared.rows += c.rows;
            compared.logit_rows += c.logit_rows;
            compared.blocks += c.blocks;
            compared.triangles += c.triangles;
            compared.chain_work += c.chain_work;
            stats.extend(x.stats.iter().copied());
        }
        let seconds = started.elapsed().as_secs_f64();
        eprintln!("{} {name}: {compared:?} in {seconds:.1} s", self.model.name);
        self.outcomes.push(Outcome {
            name: format!("{}/{name}", self.model.name),
            compared,
            seconds,
            stats,
        });
    }

    /// How far before a target the chain's top and bottom layers start (no drafter:
    /// the top is the window less the query's own position).
    fn top(&self) -> u32 {
        self.model.window - 1
    }

    fn reach(&self) -> u32 {
        4 * (self.model.window - 1)
    }

    /// Private requests of mixed lengths and sampling parameters, batched together.
    pub fn single_requests(&mut self) {
        let at = Instant::now();
        let mut h = self.harness(256, sched());
        let mut rng = TestRng(1);
        for id in 0..8 {
            let prompt = rng.var_tokens(1, 60);
            let max = 1 + u32::try_from(rng.below(12)).unwrap();
            h.submit(request(id, prompt, params(id), max, CacheScope::Private));
        }
        h.run();
        for id in 0..8 {
            assert_eq!(h.finished[&id], FinishReason::Length);
        }
        assert!(h.compared().logit_rows > 0);
        self.record("single_requests", at, vec![h]);
    }

    /// Prompts below, around and past the chain's reach, keyed and private, at chunk
    /// sizes that cut the triangles everywhere: outputs and the chain's work do not
    /// depend on the chunking.
    pub fn chunked_prefill(&mut self, chunks: &[u32]) {
        let at = Instant::now();
        let mut rng = TestRng(2);
        let (w, reach) = (self.model.window as usize, self.reach() as usize);
        let lens = [
            1,
            2,
            w - 1,
            w + 1,
            2 * w + 1,
            reach,
            reach + 1,
            reach + 2,
            reach + 75,
        ];
        let prompts: Vec<Vec<u32>> = lens.iter().map(|&n| rng.tokens(n)).collect();
        let mut results = Vec::new();
        let mut done = Vec::new();
        for &chunk in chunks {
            let mut h = self.harness(512, chunked(chunk));
            for (i, p) in prompts.iter().enumerate() {
                let cache = if i % 2 == 0 {
                    CacheScope::Private
                } else {
                    keyed(u8::try_from(i).unwrap())
                };
                h.submit(request(i as u64, p.clone(), params(i as u64), 5, cache));
            }
            h.run();
            h.flush();
            let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
                .map(|i| h.outputs[&i].clone())
                .collect();
            assert!(h.compared().triangles > 0);
            results.push((outs, h.compared().chain_work));
            done.push(h);
        }
        for (outs, work) in &results[1..] {
            assert_eq!(outs, &results[0].0, "outputs depend on chunking");
            assert_eq!(*work, results[0].1, "the chain's work depends on chunking");
        }
        self.record("chunked_prefill", at, done);
    }

    /// A prompt far past the chain's reach runs the decoder over one triangle, whatever
    /// its length and chunking.
    pub fn one_triangle(&mut self) {
        let at = Instant::now();
        let (top, w) = (u64::from(self.top()), u64::from(self.model.window));
        let mut done = Vec::new();
        for (len, chunk) in [(400usize, 1000u32), (400, 37), (350, 5)] {
            let mut h = self.harness(512, chunked(chunk));
            h.submit(request(
                0,
                TestRng(3).tokens(len),
                SamplingParams::greedy(),
                4,
                CacheScope::Private,
            ));
            h.run();
            let triangle: u64 = (0..4).map(|j| top + (3 - j) * (w - 1) + 1).sum();
            assert_eq!(h.compared().chain_work, triangle + 3 * 4);
            done.push(h);
        }
        self.record("one_triangle", at, done);
    }

    /// Hits at regeneration, continuation and branch points, a short turn after a hit,
    /// and a required prefix admitted and refused: every hit maps and decrypts cached
    /// blocks of every group.
    pub fn hits(&mut self) {
        let at = Instant::now();
        let g = SamplingParams::greedy();
        let mut done = Vec::new();
        for chunk in [1000, 41] {
            let mut h = self.harness(512, chunked(chunk));
            let mut rng = TestRng(5);
            let prompt = rng.tokens(330);
            assert_eq!(h.run_one(1, &prompt, g, 6, keyed(1)), 0);
            let regenerated = h.run_one(2, &prompt, g, 6, keyed(1));
            assert_eq!(regenerated, 328, "the regeneration point");
            assert_eq!(h.outputs[&2], h.outputs[&1]);
            let mut cont = prompt.clone();
            cont.extend(&h.outputs[&1]);
            cont.extend(rng.tokens(9));
            let continued = h.run_one(3, &cont, g, 6, keyed(1));
            assert!(continued >= 328, "a continuation hits: {continued}");
            // A short turn after a hit: every layer runs on from the hit's windows.
            let mut turn = cont[..344].to_vec();
            turn.extend(rng.tokens(3));
            assert_eq!(h.run_one(4, &turn, g, 2, keyed(1)), 344);

            let base = rng.tokens(320);
            let turn = |rng: &mut TestRng, first: u32| {
                let mut t = base[..280].to_vec();
                t.push(first);
                t.extend(rng.tokens(30));
                t
            };
            assert_eq!(h.run_one(5, &base, g, 6, keyed(2)), 0);
            // The first divergence finds banks without windows at the branch,
            // recomputes, and keeps it; the next one resumes there.
            let first = turn(&mut rng, (base[280] + 1) % VOCAB);
            assert_eq!(h.run_one(6, &first, g, 6, keyed(2)), 0);
            let second = turn(&mut rng, (base[280] + 2) % VOCAB);
            assert_eq!(h.run_one(7, &second, g, 6, keyed(2)), 280);

            // A required prefix: exactly the hit is admitted; one token more is refused
            // before anything is mapped, decrypted or computed.
            let looked = h.eng.kv().lookup(&salt(2), &second, h.now);
            assert!(looked > 280, "its own regeneration point: {looked}");
            let steps = h.compared().steps;
            let required = |id, n| Request {
                required_cached_tokens: n,
                ..request(id, second.clone(), g, 4, keyed(2))
            };
            h.submit(required(9, looked + 1));
            h.step();
            assert_eq!(h.finished[&9], FinishReason::CachedPrefixMissing);
            assert_eq!(h.compared().steps, steps, "the refused request ran a step");
            h.submit(required(10, looked));
            h.run();
            h.flush();
            assert_eq!(h.cached[&10], looked);
            done.push(h);
        }
        self.record("hits", at, done);
    }

    /// A sequence cancelled in the middle of its triangle retains no hit point; one
    /// cancelled after it does.
    pub fn cancelled_mid_triangle(&mut self) {
        let at = Instant::now();
        let g = SamplingParams::greedy();
        let len = self.reach() as usize + 72;
        let prompt = TestRng(6).tokens(len);
        let mut done = Vec::new();
        let late = u32::try_from(len / 8 + 8).unwrap();
        for (steps, hits) in [(10, false), (late, true)] {
            let mut h = self.harness(512, chunked(8));
            h.submit(request(1, prompt.clone(), g, 40, keyed(1)));
            for _ in 0..steps {
                h.step();
            }
            assert!(h.eng.cancel(1, h.now).is_some());
            h.flush();
            let hit = h.run_one(2, &prompt, g, 4, keyed(1));
            assert_eq!(hit > 0, hits, "steps {steps}: hit {hit}");
            done.push(h);
        }
        self.record("cancelled_mid_triangle", at, done);
    }

    /// Recompute preemption on a pool far too small for the workload, in whole prompts
    /// and in chunks that preempt sequences in the middle of their triangles.
    pub fn preemption(&mut self) {
        let at = Instant::now();
        let mut done = Vec::new();
        for (chunk, extra) in [(512, 60), (11, 60), (17, 260)] {
            let mut h = self.harness(if extra > 100 { 96 } else { 40 }, chunked(chunk));
            let mut rng = TestRng(8);
            for id in 0..12 {
                let prompt = rng.var_tokens(20, extra);
                h.submit(request(
                    id,
                    prompt,
                    params(id),
                    24,
                    keyed(u8::try_from(id).unwrap()),
                ));
            }
            h.run();
            h.flush();
            let stats = h.eng.stats();
            assert!(stats.preemptions > 0, "workload must force preemption");
            done.push(h);
        }
        assert!(
            done.iter().any(|h| h.compared().triangles > 0),
            "a preempted sequence resumed into a triangle"
        );
        self.record("preemption", at, done);
    }

    /// Under KV pressure a lone sequence sheds its prefill chunk and its retention, and
    /// only then ends for lack of memory; with a few blocks more it finishes.
    pub fn shedding(&mut self) {
        let at = Instant::now();
        let g = SamplingParams::greedy();
        let prompt = TestRng(9).tokens(44);
        let mut endings = Vec::new();
        let mut done = Vec::new();
        for blocks in [8u32, 9, 10, 14] {
            let mut h = self.harness(blocks, chunked(64));
            h.submit(request(1, prompt.clone(), g, 24, keyed(1)));
            h.run();
            h.flush();
            endings.push((h.finished[&1], h.outputs[&1].len()));
            done.push(h);
        }
        assert_eq!(endings[0].0, FinishReason::KvExhausted, "{endings:?}");
        assert_eq!(endings[3], (FinishReason::Length, 24), "{endings:?}");
        self.record("shedding", at, done);
    }

    /// A request's tokens and logits are the same alone and beside others in other
    /// slots, at another chunking.
    pub fn batch_invariance(&mut self) {
        let at = Instant::now();
        let prompt = TestRng(12).tokens(self.reach() as usize + 15);
        let p = SamplingParams::random(0.9, 4242).unwrap();
        let mut alone = self.harness(512, chunked(64));
        alone.submit(request(0, prompt.clone(), p, 12, CacheScope::Private));
        alone.run();
        let want: Vec<(u32, Vec<u32>)> = alone.eng.executor().sampled.clone();
        assert_eq!(want.len(), 12);

        let mut rng = TestRng(13);
        let mut busy = self.harness(512, chunked(29));
        for id in 1..=4 {
            busy.submit(request(
                id,
                rng.var_tokens(30, 200),
                params(id),
                30,
                CacheScope::Private,
            ));
        }
        for _ in 0..3 {
            busy.step();
        }
        busy.submit(request(0, prompt.clone(), p, 12, CacheScope::Private));
        busy.run();
        assert_eq!(busy.outputs[&0], alone.outputs[&0]);
        let got = &busy.eng.executor().sampled;
        for row in &want {
            assert!(got.contains(row), "logits at {} depend on the batch", row.0);
        }
        self.record("batch_invariance", at, vec![alone, busy]);
    }

    /// Randomized interleavings: multi-turn keyed conversations, private requests,
    /// cancellations, expiry, small pools; then a drain that leaves every block and
    /// every slot zero on the device.
    pub fn randomized(&mut self, seeds: std::ops::Range<u64>) {
        let at = Instant::now();
        let (mut preempted, mut hits, mut triangles) = (0, 0, 0);
        let mut done = Vec::new();
        for seed in seeds {
            let mut rng = TestRng(700 + seed);
            let blocks = 64 + u32::try_from(rng.below(48)).unwrap();
            let mut cfg = sched();
            cfg.max_prefill_chunk = 5 + u32::try_from(rng.below(90)).unwrap();
            cfg.max_batched_tokens = 64 + u32::try_from(rng.below(200)).unwrap();
            cfg.cache.idle_ttl_ms = 500;
            cfg.cache.max_age_ms = 2000;
            cfg.sweep_interval_ms = 50;
            let mut h = self.harness(blocks, cfg);
            let mut convo: HashMap<u8, Vec<u32>> = HashMap::new();
            let mut live: BTreeSet<u64> = BTreeSet::new();
            let mut next_id = 0u64;
            for _round in 0..110 {
                if rng.chance(0.3) && live.len() < 10 {
                    let id = next_id;
                    next_id += 1;
                    let (prompt, cache) = if rng.chance(0.25) {
                        (rng.var_tokens(1, 260), CacheScope::Private)
                    } else {
                        let s = u8::try_from(rng.below(3)).unwrap();
                        let prompt = match convo.get(&s) {
                            Some(t) if rng.chance(0.3) => t.clone(),
                            Some(t) if rng.chance(0.3) => {
                                let keep = 1 + usize::try_from(rng.below(t.len() as u64)).unwrap();
                                let mut t = t[..keep].to_vec();
                                t.extend(rng.var_tokens(1, 20));
                                t
                            }
                            Some(t) => {
                                let mut t = t.clone();
                                t.extend(rng.var_tokens(1, 20));
                                t
                            }
                            None => rng.var_tokens(20, 230),
                        };
                        (prompt, keyed(s + 1))
                    };
                    if prompt.len() > (blocks * 8 / 2) as usize {
                        continue;
                    }
                    if let CacheScope::Keyed(_) = cache {
                        convo.insert(u8::try_from(id % 3).unwrap(), prompt.clone());
                    }
                    live.insert(id);
                    let max = 1 + u32::try_from(rng.below(16)).unwrap();
                    h.submit(request(id, prompt, params(id), max, cache));
                }
                if rng.chance(0.03) && !live.is_empty() {
                    let nth = usize::try_from(rng.below(live.len() as u64)).unwrap();
                    let victim = *live.iter().nth(nth).unwrap();
                    if h.eng.cancel(victim, h.now).is_some() {
                        live.remove(&victim);
                    }
                }
                h.step();
                live.retain(|id| !h.finished.contains_key(id));
                h.now += if rng.chance(0.02) {
                    600
                } else {
                    1 + rng.below(5)
                };
            }
            h.run();
            hits += h.cached.values().filter(|c| **c > 0).count();
            preempted += h.eng.stats().preemptions;
            triangles += h.compared().triangles;
            // Drain: everything expires, and every block of every group and every slot
            // is zero on the device.
            h.eng.sweep(h.now + 10_000).unwrap();
            h.check();
            assert_eq!(h.eng.kv().cache_entries(), 0);
            for (g, gs) in h.spec.kv_groups.iter().enumerate() {
                assert_eq!(h.eng.kv().free_blocks(g), gs.num_blocks as usize - 1);
                for b in 1..gs.num_blocks {
                    assert!(
                        h.eng.executor().block_is_zero(u32::try_from(g).unwrap(), b),
                        "seed {seed}: group {g} block {b} holds bytes after the drain"
                    );
                }
            }
            for slot in 0..h.spec.num_state_slots {
                assert!(h.eng.executor().dev.slot_state_is_zero(slot).unwrap());
            }
            done.push(h);
        }
        assert!(
            preempted > 0 && hits > 0 && triangles > 0,
            "{preempted} {hits} {triangles}"
        );
        self.record("randomized", at, done);
    }
}

/// Every scenario, in the order the device run takes them.
pub fn all_scenarios(run: &mut Run<'_>, wide: bool) {
    if wide {
        run.chunked_prefill(&[7, 1000]);
        run.hits();
        run.cancelled_mid_triangle();
        run.preemption();
        run.batch_invariance();
        return;
    }
    run.single_requests();
    run.chunked_prefill(&[1, 7, 48, 97, 1000]);
    run.one_triangle();
    run.hits();
    run.cancelled_mid_triangle();
    run.preemption();
    run.shedding();
    run.batch_invariance();
    run.randomized(0..3);
}

/// The summary of a run: per scenario what was compared and how long each step took on
/// the executor under test, part by part.
pub fn summary(device: &str, outcomes: &[Outcome]) -> serde_json::Value {
    let scenario = |o: &Outcome| {
        let with_rows: Vec<&StepStats> = o.stats.iter().filter(|s| s.rows > 0).collect();
        let mean = |f: &dyn Fn(&StepStats) -> f64| match with_rows.len() {
            0 => 0.0,
            n => with_rows.iter().map(|s| f(s)).sum::<f64>() / n as f64,
        };
        let max = with_rows.iter().map(|s| s.total_ms).fold(0.0, f64::max);
        serde_json::json!({
            "name": o.name,
            "seconds": o.seconds,
            "steps": o.compared.steps,
            "positions": o.compared.rows,
            "logit_rows_equal": o.compared.logit_rows,
            "blocks_equal": o.compared.blocks,
            "triangles": o.compared.triangles,
            "chain_layer_positions": o.compared.chain_work,
            "step_ms": {
                "mean": mean(&|s| s.total_ms),
                "max": max,
                "maintenance": mean(&|s| s.maintenance_ms),
                "gather": mean(&|s| s.gather_ms),
                "encoder": mean(&|s| s.encoder_ms),
                "crossing": mean(&|s| s.crossing_ms),
                "decoder": mean(&|s| s.decoder_ms),
            },
            "per_step": with_rows.iter().map(|s| serde_json::json!({
                "positions": s.rows,
                "chain_layer_positions": s.chain_work,
                "sampled": s.sampled,
                "maintenance_ms": s.maintenance_ms,
                "gather_ms": s.gather_ms,
                "encoder_ms": s.encoder_ms,
                "crossing_ms": s.crossing_ms,
                "decoder_ms": s.decoder_ms,
                "total_ms": s.total_ms,
            })).collect::<Vec<_>>(),
        })
    };
    serde_json::json!({
        "device": device,
        "equal_to_cpu_reference": true,
        "scenarios": outcomes.iter().map(scenario).collect::<Vec<_>>(),
    })
}
