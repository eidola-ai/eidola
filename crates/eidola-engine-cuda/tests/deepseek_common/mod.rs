//! Shared harness of the DeepSeek-V4.1 executor's tests: the serving core's engine over
//! [`Paired`], which gives every step to the executor under test and to the CPU
//! reference executor and holds them equal, and the scenarios both the host run
//! (`deepseek_host.rs`) and the device run (`deepseek_tiny.rs`) drive.
//!
//! Equal means: the same tokens; the same logits, bit for bit, at every sampling row
//! (every step asks for them); the same bytes in every plaintext block of every group,
//! compared in the reference executor's cell order; the same blocks held encrypted; the
//! same set crossing between the two halves; and the same chain work. The CPU suites
//! hold the reference executor to the dense oracle, so nothing here repeats that.
//!
//! Ciphertext is not comparable between the two (their block images order the keystream
//! differently), so the memory scenarios hold the executor under test to itself: idle KV
//! is ciphertext, a host block is the ciphertext of the block demoted into it byte for
//! byte, and a recomputed block seals to the same bytes where its group's coverage says
//! it must.

#![allow(dead_code)]

pub mod compiled;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::{SamplingParams, mix64};
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::{Bucket, Coverage, ModelSpec};
use eidola_engine_cpu::deepseek::{DsCpuExecutor, DsCpuExecutorConfig};
use eidola_engine_cuda::deepseek::gemm::GemmForm;
use eidola_engine_cuda::deepseek::half::Device;
use eidola_engine_cuda::deepseek::weights::Half;
use eidola_engine_cuda::deepseek::{DsCudaExecutor, DsCudaExecutorConfig, StepStats};
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

/// Raises the soft memory-lock limit for the reference executor's host-tier pools and
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
    /// The same with the drafter's Markov term removed.
    pub plain_markov: Arc<DsModel>,
    pub window: u32,
    /// The drafter's slots.
    pub slots: u32,
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
        if wide {
            // The released drafter's five slots.
            c.drafter.as_mut().expect("the fixture has a drafter").slots = 5;
        }
        Ok(c)
    };
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let config = widen(config_of(&store).unwrap()).unwrap();
    let mut plain_markov = load_reference(
        &dir,
        widen,
        &DsLoadOptions::default(),
        Some(token_map.clone()),
    )
    .unwrap();
    plain_markov
        .weights
        .drafter
        .as_mut()
        .unwrap()
        .markov_head
        .data
        .fill(0.0);
    let reference = load_reference(
        &dir,
        widen,
        &DsLoadOptions {
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
        plain_markov: Arc::new(plain_markov),
        window,
        slots: if wide { 5 } else { 4 },
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

/// How a pair drafts.
#[derive(Clone, Copy, Debug)]
pub struct Drafting {
    pub width: u32,
    pub sampleable: u32,
    /// Whether the drafter keeps its Markov term.
    pub markov: bool,
    /// Blocks per group of a host tier; 0 for none.
    pub host_blocks: u32,
    /// Longest sequence.
    pub max_model_len: u32,
    /// Blocks per window group, when not the banks' count.
    pub window_blocks: Option<u32>,
}

impl Default for Drafting {
    fn default() -> Self {
        Drafting {
            width: 0,
            sampleable: VOCAB,
            markov: true,
            host_blocks: 0,
            max_model_len: 1024,
            window_blocks: None,
        }
    }
}

thread_local! {
    /// With the staged-experts feature: when set, the executor under test is built with
    /// the backbone's routed experts staged (`deepseek::staged`) from host memory read this
    /// way, with caches of these many experts (the encoder's, the decoder's).
    #[cfg(feature = "dev-staged-experts")]
    pub static STAGED: std::cell::Cell<Option<(eidola_engine_cuda::deepseek::staged::HostMode, [u32; 2])>> =
        const { std::cell::Cell::new(None) };
}

/// The executor under test with its experts staged.
#[cfg(feature = "dev-staged-experts")]
fn staged(
    m: &Model,
    device: &dyn Fn() -> Box<dyn Device>,
    cfg: &DsCudaExecutorConfig,
    mode: &eidola_engine_cuda::deepseek::staged::HostMode,
    capacity: &[u32; 2],
) -> DsCudaExecutor {
    use eidola_engine_cuda::deepseek::staged::{DevOnly, HostExperts};
    use eidola_engine_cuda::deepseek::support::check_supported;
    let token = DevOnly::for_tests_and_dev_tools();
    let shape = check_supported(&m.config, cfg.draft_width).unwrap();
    let host = Arc::new(HostExperts::new(&m.store, &m.config, &shape, *mode, token).unwrap());
    DsCudaExecutor::staged_with_devices(
        device(),
        device(),
        &m.store,
        &m.config,
        Some(m.token_map.clone()),
        cfg,
        &host,
        *capacity,
        token,
    )
    .expect("the staged executor under test")
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
    /// Drafts made, and drafts accepted.
    pub drafted: u64,
    pub accepted: u64,
    /// Draft slots' logit rows equal bit for bit.
    pub draft_rows: u64,
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
    /// Per drafting row: the drafts accepted.
    pub accepted_counts: Vec<u32>,
    /// Drafting rows whose pass stood at or past the length limit.
    pub short_passes: u64,
    /// Per draft slot of every drafting row: the row's last host position, the drafts
    /// before the slot, and the slot's logits' bits.
    pub drafted_rows: Vec<(u32, Vec<u32>, Vec<u32>)>,
}

impl Paired {
    pub fn new(m: &Model, device: &dyn Fn() -> Box<dyn Device>, blocks: u32) -> Paired {
        Paired::drafting(m, device, blocks, Drafting::default())
    }

    /// The pair under a draft width, a sampleable vocabulary and, optionally, with the
    /// drafter's Markov term removed on both executors.
    pub fn drafting(
        m: &Model,
        device: &dyn Fn() -> Box<dyn Device>,
        blocks: u32,
        how: Drafting,
    ) -> Paired {
        let cfg = DsCudaExecutorConfig {
            block_size: 8,
            num_blocks: blocks,
            window_blocks: how.window_blocks.unwrap_or(blocks),
            num_state_slots: 16,
            max_model_len: how.max_model_len,
            buckets: buckets(),
            sampleable_vocab_size: how.sampleable,
            devices: [0, 0],
            draft_width: how.width,
            host_blocks: how.host_blocks,
            position_tags: true,
            record: true,
            gemm: GemmForm::from_env().expect("the GEMM form"),
        };
        #[cfg(feature = "dev-staged-experts")]
        let dev = match STAGED.get() {
            Some((mode, capacity)) => staged(m, device, &cfg, &mode, &capacity),
            None => DsCudaExecutor::with_devices(
                device(),
                device(),
                &m.store,
                &m.config,
                Some(m.token_map.clone()),
                &cfg,
            )
            .expect("the executor under test"),
        };
        #[cfg(not(feature = "dev-staged-experts"))]
        let dev = DsCudaExecutor::with_devices(
            device(),
            device(),
            &m.store,
            &m.config,
            Some(m.token_map.clone()),
            &cfg,
        )
        .expect("the executor under test");
        if !how.markov && how.width > 0 {
            // The Markov head's weight, zeroed on the device as in `plain_markov`.
            let n = dev.half(Half::Decoder);
            let w = n.w.drafter.as_ref().expect("a drafting decoder");
            let c = &m.config;
            let bytes = c.vocab_size * c.drafter.as_ref().unwrap().markov_rank * 2;
            // SAFETY: the Markov head's own allocation, `[vocab, rank]` BF16.
            unsafe { n.dev.zero(w.markov_head, bytes) }.unwrap();
        }
        let cpu = DsCpuExecutor::new(
            match how.markov {
                true => m.reference.clone(),
                false => m.plain_markov.clone(),
            },
            DsCpuExecutorConfig {
                block_size: cfg.block_size,
                num_blocks: blocks,
                window_blocks: how.window_blocks.unwrap_or(blocks),
                host_blocks: how.host_blocks,
                num_state_slots: cfg.num_state_slots,
                max_model_len: cfg.max_model_len,
                buckets: buckets(),
                draft_width: how.width,
                sampleable_vocab_size: how.sampleable,
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
            accepted_counts: Vec::new(),
            short_passes: 0,
            drafted_rows: Vec::new(),
        }
    }

    /// Whether a host-tier block of the executor under test is zero.
    pub fn host_block_is_zero(&self, group: u32, host: u32) -> bool {
        let bytes = self
            .dev
            .host_block_bytes(group, host)
            .expect("a host block");
        bytes.iter().all(|b| *b == 0)
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
            }
            for (j, a) in a.iter().enumerate() {
                let e = &step.seqs[row];
                let at = e.context_len + e.num_tokens - 1 + u32::try_from(j).unwrap();
                self.sampled.push((at, bits(a)));
            }
        }
        assert_eq!(
            self.dev.chain_layer_positions() - work.0,
            self.cpu.chain_layer_positions() - work.1,
            "the chain's work"
        );
        // Each draft slot's logits, given the token drafted before it.
        let ours = self.dev.take_draft_logits();
        let records = self.cpu.take_records();
        let theirs: Vec<_> = records.iter().filter(|r| !r.drafts.is_empty()).collect();
        assert_eq!(ours.len(), theirs.len(), "drafting rows");
        for ((slot, ours), t) in ours.iter().zip(theirs) {
            assert_eq!(*slot, t.slot);
            assert_eq!(ours.len(), t.draft_logits.len(), "slot {slot}: draft slots");
            for (i, (a, b)) in ours.iter().zip(&t.draft_logits).enumerate() {
                assert_eq!(
                    bits(a),
                    bits(b),
                    "slot {slot} draft {i}: draft logits differ from the reference executor's"
                );
                self.compared.draft_rows += 1;
                let at = t.context_len + t.num_tokens - 1;
                self.drafted_rows
                    .push((at, t.drafts[..i].to_vec(), bits(a)));
            }
        }
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
            assert_eq!(o.drafts, t.drafts, "drafted tokens");
            assert_eq!(o.taps, t.taps, "boundary rows with drafter taps");
        }
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
        for (e, produced) in step.seqs.iter().zip(&got.num_tokens) {
            if e.sample && e.num_drafts > 0 {
                let last = e.context_len + e.num_tokens - 1;
                let slots = self.dev.half(Half::Decoder).drafting().unwrap().1.slots;
                if last + slots > self.cpu.spec().max_model_len {
                    self.short_passes += 1;
                }
                self.compared.drafted += u64::from(e.num_drafts);
                self.compared.accepted += u64::from(*produced - 1);
                self.accepted_counts.push(*produced - 1);
            }
        }
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
        Harness::drafting(m, device, blocks, sched, Drafting::default())
    }

    pub fn drafting(
        m: &Model,
        device: &dyn Fn() -> Box<dyn Device>,
        blocks: u32,
        mut sched: SchedulerConfig,
        how: Drafting,
    ) -> Self {
        sched.speculative = how.width > 0;
        let exec = Paired::drafting(m, device, blocks, how);
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
        // Idle blocks are ciphertext and mapped ones plaintext; a free host block is
        // zero and a held one ciphertext, on both executors.
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
                    x.cpu.host_block_is_zero(g, hb) && x.host_block_is_zero(g, hb),
                    "free host block {g}/{hb} holds data with no zero pending"
                );
            }
        }
        for (g, hb) in kv.host_cache_blocks() {
            if !demoting.contains(&(g, hb)) {
                assert!(
                    x.cpu.host_block_is_encrypted(g, hb) && x.dev.host_block_is_encrypted(g, hb),
                    "held host block {g}/{hb} holds no ciphertext"
                );
            }
        }
        for (g, b) in kv.cache_blocks().union(&kv.sequence_blocks()) {
            if landing.contains(&(*g, *b)) {
                continue;
            }
            let want = idle.contains(&(*g, *b));
            let device = x.dev.block_is_encrypted(*g, *b);
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
    /// Drafted tokens per speculative step in every scenario; 0 for none.
    pub width: u32,
}

impl Run<'_> {
    fn harness(&self, blocks: u32, sched: SchedulerConfig) -> Harness {
        let how = Drafting {
            width: self.width,
            ..Drafting::default()
        };
        Harness::drafting(self.model, self.device, blocks, sched, how)
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
            compared.drafted += c.drafted;
            compared.accepted += c.accepted;
            compared.draft_rows += c.draft_rows;
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
    /// depend on the chunking. With drafts on the steps that draft depend on it, so the
    /// chain's work does and a sampled row's draws do: every request is greedy then.
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
                let sampling = match self.width {
                    0 => params(i as u64),
                    _ => SamplingParams::greedy(),
                };
                h.submit(request(i as u64, p.clone(), sampling, 5, cache));
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
            if self.width == 0 {
                assert_eq!(*work, results[0].1, "the chain's work depends on chunking");
            }
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
            match self.width {
                0 => assert_eq!(h.compared().chain_work, triangle + 3 * 4),
                // Drafted positions run the chain too.
                _ => assert!(h.compared().chain_work >= triangle + 3 * 4),
            }
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
        // Past the fixture's reach; the wide window's reach is past these prompts.
        assert!(
            self.model.window != WINDOW || done.iter().any(|h| h.compared().triangles > 0),
            "a preempted sequence resumed into a triangle"
        );
        self.record("preemption", at, done);
    }

    /// Window pools sized apart from the banks (512 bank blocks; block 8 and window 48,
    /// so a live window is at most seven blocks): one conversation of 700-position
    /// prompts through 38 window blocks, the node's rule for one seat and a 64-token step,
    /// resuming at its continuation and regeneration points with no preemption; then six
    /// concurrent sequences whose windows do not fit 17 window blocks at once, which the
    /// window pools preempt as any pool does.
    pub fn window_pools(&mut self) {
        let at = Instant::now();
        let g = SamplingParams::greedy();
        let windowed = |run: &Self, window_blocks, sched| {
            let how = Drafting {
                width: run.width,
                window_blocks: Some(window_blocks),
                ..Drafting::default()
            };
            Harness::drafting(run.model, run.device, 512, sched, how)
        };
        let mut long = windowed(self, 38, chunked(64));
        let mut rng = TestRng(60);
        let prompt = rng.tokens(700);
        assert_eq!(long.run_one(1, &prompt, g, 40, keyed(1)), 0);
        let mut cont = prompt.clone();
        cont.extend(&long.outputs[&1]);
        cont.extend(rng.tokens(150));
        // 700 + 40 - 1 positions entered KV: the last sealed boundary is 736.
        assert_eq!(long.run_one(2, &cont, g, 40, keyed(1)), 736);
        assert_eq!(long.run_one(3, &prompt, g, 40, keyed(1)), 696);
        for id in 1..=3 {
            assert_eq!(long.finished[&id], FinishReason::Length, "request {id}");
        }
        assert_eq!(long.eng.stats().preemptions, 0);

        let mut pressed = windowed(self, 17, chunked(11));
        for id in 0..6 {
            let prompt = rng.var_tokens(150, 100);
            let key = keyed(u8::try_from(id).unwrap());
            pressed.submit(request(id, prompt, params(id), 24, key));
        }
        pressed.run();
        pressed.flush();
        assert!(
            pressed.eng.stats().preemptions > 0,
            "the window pools must force preemption"
        );
        for id in 0..6 {
            assert_eq!(pressed.finished[&id], FinishReason::Length, "request {id}");
        }
        self.record("window_pools", at, vec![long, pressed]);
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

    /// A request's logits are the same bits alone and beside others: in another slot,
    /// in larger buckets (alone, its decode steps run in the bucket of one sequence),
    /// its prompt cut into other chunks, other rows before and after it in the step.
    /// That is every target row it samples from or verifies a draft with and, with
    /// drafts on, every draft slot's logits.
    pub fn batch_invariance(&mut self) {
        let at = Instant::now();
        let prompt = TestRng(12).tokens(self.reach() as usize + 15);
        let p = SamplingParams::random(0.9, 4242).unwrap();
        let mut alone = self.harness(512, chunked(64));
        alone.submit(request(0, prompt.clone(), p, 12, CacheScope::Private));
        alone.run();
        let x = alone.eng.executor();
        let want: Vec<(u32, Vec<u32>)> = x.sampled.clone();
        let want_drafts = x.drafted_rows.clone();
        assert!(want.len() >= 12);
        assert_eq!(want_drafts.is_empty(), self.width == 0);

        let mut done = vec![alone];
        // Beside four others that started first, then beside eleven that arrive with
        // it and after it.
        for (others, early, chunk) in [(4u64, true, 29), (11, false, 61)] {
            let mut rng = TestRng(13 + others);
            let mut busy = self.harness(512, chunked(chunk));
            let mut ids = 1..=others;
            let mut other = |busy: &mut Harness, id| {
                let prompt = rng.var_tokens(30, 200);
                busy.submit(request(id, prompt, params(id), 30, CacheScope::Private));
            };
            if early {
                for id in ids.by_ref() {
                    other(&mut busy, id);
                }
                for _ in 0..3 {
                    busy.step();
                }
            } else {
                for id in ids.by_ref().take(5) {
                    other(&mut busy, id);
                }
            }
            busy.submit(request(0, prompt.clone(), p, 12, CacheScope::Private));
            for id in ids {
                busy.step();
                other(&mut busy, id);
            }
            busy.run();
            assert_eq!(busy.outputs[&0], done[0].outputs[&0]);
            let x = busy.eng.executor();
            for row in &want {
                assert!(
                    x.sampled.contains(row),
                    "logits at {} depend on the batch",
                    row.0
                );
            }
            for row in &want_drafts {
                assert!(
                    x.drafted_rows.contains(row),
                    "draft logits at {} after {:?} depend on the batch",
                    row.0,
                    row.1
                );
            }
            done.push(busy);
        }
        self.record("batch_invariance", at, done);
    }

    /// Greedy speculation never changes a token, at every draft width the drafter has
    /// and across acceptance patterns from every draft rejected to every draft
    /// accepted: the patterns come from narrowing the sampleable vocabulary and from a
    /// drafter whose Markov term is removed.
    pub fn greedy_speculation(&mut self) {
        let at = Instant::now();
        let g = SamplingParams::greedy();
        let slots = self.model.slots;
        let mut seen = vec![0usize; slots as usize + 1];
        let mut done = Vec::new();
        for (sampleable, markov) in [
            (VOCAB, true),
            (VOCAB, false),
            (3, true),
            (2, false),
            (1, true),
        ] {
            let mut rng = TestRng(10);
            let prompts: Vec<Vec<u32>> = (0..5)
                .map(|_| {
                    let n = 2 + usize::try_from(rng.below(70)).unwrap();
                    (0..n)
                        .map(|_| u32::try_from(rng.below(u64::from(sampleable))).unwrap())
                        .collect()
                })
                .collect();
            let mut plain = None;
            for width in 0..=slots {
                let how = Drafting {
                    width,
                    sampleable,
                    markov,
                    ..Drafting::default()
                };
                let mut h = Harness::drafting(self.model, self.device, 256, chunked(64), how);
                for (i, p) in prompts.iter().enumerate() {
                    h.submit(request(i as u64, p.clone(), g, 20, CacheScope::Private));
                }
                h.run();
                h.flush();
                let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
                    .map(|i| h.outputs[&i].clone())
                    .collect();
                assert!(outs.iter().flatten().all(|t| *t < sampleable));
                assert_eq!(plain.get_or_insert(outs.clone()), &outs, "width {width}");
                let x = h.eng.executor();
                assert_eq!(x.compared.drafted > 0, width > 0, "width {width}");
                if width == slots {
                    for a in &x.accepted_counts {
                        seen[*a as usize] += 1;
                    }
                }
                done.push(h);
            }
        }
        assert!(
            seen.iter().all(|n| *n > 0),
            "accepted-count coverage {seen:?}"
        );
        self.record("greedy_speculation", at, done);
    }

    /// Sequences run to the length limit with drafts on. The engine asks for fewer
    /// drafts than the drafter has slots as a sequence nears the limit, and the pass
    /// still computes every slot, at positions up to `slots - 2` past the last one a
    /// sequence can hold.
    pub fn length_limit(&mut self) {
        let at = Instant::now();
        let limit = 64;
        let how = Drafting {
            width: self.model.slots,
            max_model_len: limit,
            ..Drafting::default()
        };
        let mut h = Harness::drafting(self.model, self.device, 256, sched(), how);
        let mut rng = TestRng(15);
        for id in 0..6u64 {
            let prompt = rng.var_tokens(40, 20);
            let room = limit - u32::try_from(prompt.len()).unwrap();
            h.submit(request(id, prompt, params(id), room, CacheScope::Private));
        }
        h.run();
        h.flush();
        for id in 0..6 {
            assert!(!h.outputs[&id].is_empty());
        }
        // A drafting step whose pass reached past the limit: fewer drafts than slots.
        let c = h.compared();
        assert!(c.drafted > 0);
        assert!(
            h.eng.executor().short_passes > 0,
            "no pass ran with fewer drafts than slots at the limit"
        );
        self.record("length_limit", at, vec![h]);
    }

    /// Seeded sampled rows with drafting on: the draft chain on the draft stream, each
    /// slot conditioned on the draft before it, and chain acceptance over the
    /// conditional distributions.
    pub fn sampled_speculation(&mut self) {
        let at = Instant::now();
        let mut done = Vec::new();
        for width in [2, self.model.slots] {
            let how = Drafting {
                width,
                ..Drafting::default()
            };
            let mut h = Harness::drafting(self.model, self.device, 256, sched(), how);
            let mut rng = TestRng(11);
            for id in 0..6u64 {
                let k = usize::try_from(id % 3).unwrap();
                let p = SamplingParams::new(
                    0.8 + 0.1 * k as f32,
                    [0, 5, 40][k],
                    [1.0, 0.95, 0.8][k],
                    0.0,
                    77 + id,
                )
                .unwrap();
                h.submit(request(
                    id,
                    rng.var_tokens(3, 50),
                    p,
                    14,
                    CacheScope::Private,
                ));
            }
            h.run();
            h.flush();
            let c = h.compared();
            assert!(c.drafted > 0 && c.accepted < c.drafted && c.draft_rows > 0);
            done.push(h);
        }
        self.record("sampled_speculation", at, done);
    }

    /// A harness with `width` drafts and a host tier of `host_blocks`.
    fn memory_harness(
        &self,
        blocks: u32,
        width: u32,
        host_blocks: u32,
        sched: SchedulerConfig,
    ) -> Harness {
        let how = Drafting {
            width,
            host_blocks,
            ..Drafting::default()
        };
        Harness::drafting(self.model, self.device, blocks, sched, how)
    }

    /// Idle KV of every group is ciphertext over the block's image on the device, and a
    /// hit decrypts it into the bits the reference executor computes, with drafts off
    /// and on.
    pub fn idle_encryption(&mut self) {
        let at = Instant::now();
        let mut done = Vec::new();
        for width in [0, self.model.slots] {
            let mut h = self.memory_harness(512, width, 0, sched());
            let groups: BTreeSet<u32> =
                (0..u32::try_from(h.spec.kv_groups.len()).unwrap()).collect();
            let prompt = TestRng(40).tokens(210);
            let g = SamplingParams::greedy();
            h.submit(request(1, prompt.clone(), g, 6, keyed(1)));
            let mut seen = Images::new();
            run_recording(&mut h, &mut seen);
            let cached = h.eng.kv().cache_blocks();
            h.flush();
            assert_eq!(h.eng.kv_keys_held(), 0);
            let mut held = BTreeSet::new();
            for (gr, b) in &cached {
                let x = &h.eng.executor().dev;
                assert!(x.block_is_encrypted(*gr, *b));
                let c = x.block_bytes(*gr, *b).unwrap();
                let p = &seen[&(*gr, *b)];
                assert!(
                    same_bytes(&c, p) * 20 < c.len(),
                    "group {gr} block {b}: idle KV is not ciphertext"
                );
                held.insert(*gr);
            }
            assert_eq!(held, groups, "every group is retained");

            // Another lineage, same prompt: a miss.
            assert_eq!(h.run_one(2, &prompt, g, 6, keyed(2)), 0);
            // The first lineage again: a hit, decrypted in every group, bit-exact.
            let mark = h.eng.executor().cpu.crypt_log().len();
            h.submit(request(3, prompt.clone(), g, 6, keyed(1)));
            h.step();
            assert_eq!(h.eng.cached_prompt_tokens(3), Some(208));
            let decrypted: BTreeSet<u32> = h.eng.executor().cpu.crypt_log()[mark..]
                .iter()
                .filter(|e| !e.3)
                .map(|e| e.1)
                .collect();
            assert_eq!(decrypted, groups);
            h.run();
            assert_eq!(h.outputs[&3], h.outputs[&1]);
            assert_eq!(h.outputs[&2], h.outputs[&1]);
            h.flush();
            assert_eq!(h.eng.kv_keys_held(), 0);
            done.push(h);
        }
        self.record("idle_encryption", at, done);
    }

    /// A block of a group whose coverage is `All` is sealed under a nonce fixed by its
    /// hash and group, so its bytes must be the same every time it is sealed: one
    /// lineage's request recomputed beside another sequence, and in other prefill
    /// chunks, seals every such block to the same ciphertext byte for byte on the
    /// device. The chain-covered groups' blocks seal under a nonce per attachment.
    pub fn resealing(&mut self) {
        let at = Instant::now();
        let width = self.model.slots;
        let prompt = TestRng(41).tokens(210);
        let other = TestRng(42).tokens(150);
        let g = SamplingParams::greedy();
        let mine = |id| request(id, prompt.clone(), g, 6, keyed(1));
        let beside = |id| request(id, other.clone(), g, 9, CacheScope::Private);

        let mut h = self.memory_harness(512, width, 0, sched());
        h.submit(mine(1));
        h.run();
        h.flush();
        let first = sealed(&h);
        assert_eq!(
            first.len(),
            h.spec.kv_groups.len(),
            "every group is retained"
        );
        let check = |h: &Harness, what: &str, same_engine: bool| {
            let again = sealed(h);
            for (group, spec) in h.spec.kv_groups.iter().enumerate() {
                let g = u32::try_from(group).unwrap();
                assert_eq!(again[&g].len(), first[&g].len(), "{what}: group {g}");
                match spec.coverage {
                    Coverage::All => assert!(
                        again[&g] == first[&g],
                        "{what}: a block of group {g} sealed to other ciphertext"
                    ),
                    Coverage::Chain if same_engine => assert!(
                        again[&g].iter().all(|c| !first[&g].contains(c)),
                        "{what}: two attachments of group {g} share a keystream"
                    ),
                    Coverage::Chain => {}
                }
            }
        };

        // Expired and zeroed, then recomputed beside a private request that started
        // first.
        h.now += CachePolicy::default().max_age_ms + 1;
        h.eng.sweep(h.now).unwrap();
        h.check();
        assert!(h.eng.kv().cache_blocks().is_empty());
        h.submit(beside(2));
        h.step();
        h.now += 1;
        h.submit(mine(3));
        h.run();
        h.flush();
        assert_eq!(h.cached[&3], 0, "a recompute, not a hit");
        assert_eq!(h.outputs[&3], h.outputs[&1]);
        check(&h, "beside another sequence", true);

        // Another engine, prefilling 13 tokens a step, the other request beside it.
        let mut c = self.memory_harness(512, width, 0, chunked(13));
        c.submit(mine(1));
        c.submit(beside(2));
        c.run();
        c.flush();
        assert_eq!(c.outputs[&1], h.outputs[&1]);
        check(&c, "in other chunks", false);
        self.record("resealing", at, vec![h, c]);
    }

    /// Device pressure demotes a lineage's idle entries to the host tier as the
    /// ciphertext they are, in every group, byte for byte; the hit that promotes them
    /// back is the reference executor's.
    pub fn host_tier(&mut self) {
        let at = Instant::now();
        let _guard = memlock_guard();
        let mut h = self.memory_harness(48, self.model.slots, 256, sched());
        let groups: BTreeSet<u32> = (0..u32::try_from(h.spec.kv_groups.len()).unwrap()).collect();
        let mut rng = TestRng(43);
        let g = SamplingParams::greedy();
        let prompt = rng.tokens(120);
        assert_eq!(h.run_one(1, &prompt, g, 6, keyed(1)), 0);
        let first = h.eng.kv().cache_blocks();
        let ciphertext = images(&h, &first);
        let mark = h.eng.executor().cpu.transfer_log().len();
        for i in 0..4u8 {
            let other = rng.tokens(120);
            h.run_one(10 + u64::from(i), &other, g, 5, keyed(100 + i));
        }
        let demoted: Vec<(u32, u32, u32)> = h.eng.executor().cpu.transfer_log()[mark..]
            .iter()
            .filter(|t| t.4)
            .map(|t| (t.1, t.2, t.3))
            .collect();
        let blocks: BTreeSet<(u32, u32)> = demoted.iter().map(|d| (d.0, d.1)).collect();
        assert!(first.iter().all(|b| blocks.contains(b)), "all demoted");
        assert_eq!(blocks.iter().map(|b| b.0).collect::<BTreeSet<_>>(), groups);
        // The first entry's blocks arrived on the host byte for byte.
        let mut arrived = BTreeSet::new();
        for (g, b, host) in &demoted {
            let x = &h.eng.executor().dev;
            if let Some(c) = ciphertext.get(&(*g, *b))
                && h.eng.kv().host_cache_blocks().contains(&(*g, *host))
                && x.host_block_bytes(*g, *host).unwrap() == *c
            {
                arrived.insert(*g);
            }
        }
        assert_eq!(
            arrived, groups,
            "a demoted block of every group is on the host"
        );
        let stats = h.eng.kv().tier_stats();
        assert_eq!((stats.host_drops, stats.device_drops), (0, 0));

        let mark = h.eng.executor().cpu.transfer_log().len();
        assert_eq!(h.run_one(2, &prompt, g, 6, keyed(1)), 112);
        let promoted: BTreeSet<u32> = h.eng.executor().cpu.transfer_log()[mark..]
            .iter()
            .filter(|t| !t.4)
            .map(|t| t.1)
            .collect();
        assert_eq!(promoted, groups);
        assert_eq!(h.outputs[&2], h.outputs[&1]);
        self.record("host_tier", at, vec![h]);
    }

    /// Randomized interleavings: multi-turn keyed conversations, private requests,
    /// cancellations, expiry, small pools, on odd seeds a host tier; then a drain that
    /// leaves every block and every slot zero on the device.
    pub fn randomized(&mut self, seeds: std::ops::Range<u64>) {
        let at = Instant::now();
        let _guard = memlock_guard();
        let (mut preempted, mut hits, mut triangles) = (0, 0, 0);
        let (mut demoted, mut promoted) = (0, 0);
        let mut tiered = false;
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
            let host_blocks = if seed % 2 == 1 { 2 * blocks } else { 0 };
            tiered |= host_blocks > 0;
            let mut h = self.memory_harness(blocks, self.width, host_blocks, cfg);
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
            let tier = h.eng.kv().tier_stats();
            demoted += tier.demoted_blocks;
            promoted += tier.promoted_blocks;
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
        assert!(
            !tiered || (demoted > 0 && promoted > 0),
            "the host tier was not used: {demoted} demoted, {promoted} promoted"
        );
        self.record("randomized", at, done);
    }
}

/// Device bytes of blocks, by `(group, block)`.
pub type Images = BTreeMap<(u32, u32), Vec<u8>>;

fn images<'a>(h: &Harness, blocks: impl IntoIterator<Item = &'a (u32, u32)>) -> Images {
    let x = &h.eng.executor().dev;
    blocks
        .into_iter()
        .map(|&(g, b)| ((g, b), x.block_bytes(g, b).expect("a block")))
        .collect()
}

/// Steps until every request has finished, keeping the latest plaintext image of every
/// block: after a step, a block is plaintext in the pool if a sequence maps it or its
/// encryption is still queued.
fn run_recording(h: &mut Harness, plain: &mut Images) {
    while h.eng.unfinished() > 0 {
        h.step();
        h.now += 1;
        let kv = h.eng.kv();
        let crypts = kv.pending_crypts();
        let mut blocks: Vec<(u32, u32)> = kv
            .sequence_blocks()
            .into_iter()
            .filter(|b| !crypts.contains_key(b))
            .collect();
        blocks.extend(crypts.iter().filter(|(_, e)| **e).map(|(b, _)| *b));
        let fresh = images(h, &blocks);
        plain.extend(fresh);
    }
}

fn same_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

/// Every cached block's ciphertext on the device, per group, in a canonical order.
fn sealed(h: &Harness) -> BTreeMap<u32, Vec<Vec<u8>>> {
    let x = &h.eng.executor().dev;
    let mut out: BTreeMap<u32, Vec<Vec<u8>>> = BTreeMap::new();
    for (g, b) in h.eng.kv().cache_blocks() {
        assert!(x.block_is_encrypted(g, b));
        out.entry(g).or_default().push(x.block_bytes(g, b).unwrap());
    }
    for images in out.values_mut() {
        images.sort();
    }
    out
}

/// The memory scenarios, on the fixture: idle encryption, resealing, the host tier's
/// round trip, and randomized traffic with and without a host tier.
pub fn memory_scenarios(run: &mut Run<'_>, wide: bool) {
    if wide {
        return;
    }
    run.idle_encryption();
    run.resealing();
    run.host_tier();
    run.randomized(2..4);
}

/// The drafting scenarios: every width and acceptance pattern and seeded sampling, then
/// scenarios of the plain run with every slot drafted (`run.width` the model's slots).
pub fn drafting_scenarios(run: &mut Run<'_>, wide: bool) {
    assert_eq!(run.width, run.model.slots);
    run.greedy_speculation();
    run.sampled_speculation();
    run.length_limit();
    run.hits();
    run.batch_invariance();
    if wide {
        return;
    }
    run.single_requests();
    run.chunked_prefill(&[7, 1000]);
    run.cancelled_mid_triangle();
    run.preemption();
    run.shedding();
    run.randomized(0..2);
}

/// Every scenario, in the order the device run takes them, with `run.width` drafts.
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
    run.window_pools();
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
            "drafted": o.compared.drafted,
            "accepted": o.compared.accepted,
            "draft_logit_rows_equal": o.compared.draft_rows,
            "step_ms": {
                "mean": mean(&|s| s.total_ms),
                "max": max,
                "maintenance": mean(&|s| s.maintenance_ms),
                "draft": mean(&|s| s.draft_ms),
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
                "draft_ms": s.draft_ms,
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
