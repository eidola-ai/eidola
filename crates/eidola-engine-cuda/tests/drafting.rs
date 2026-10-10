//! MTP drafting on the device, on the truncated real checkpoint with its MTP
//! layers (checkpoint layers 0, 1, 2 and 5, and the three MTP layers: the model
//! crate's `golden/fetch_truncated.py` output; `EIDOLA_MIMO_DIR` must name a
//! directory holding them, `EIDOLA_MIMO_GOLDEN` the golden for real token ids):
//!
//! - against the CPU reference executor (`eidola-engine-cpu`, the oracle for
//!   drafting): both engines run the same workload in lockstep at every draft
//!   width; while their outputs agree every step's drafts are compared row by
//!   row, and a greedy row's draft that differs must be within [`MARGIN`]
//!   logits of the CPU drafter's argmax at that depth and position (the GPU's
//!   quantization tolerance; a seeded draft is a draw, so its trail is
//!   printed apart and not bounded); every greedy output is within
//!   [`MARGIN`] of the f32 reference's argmax, and both acceptance rates are
//!   printed;
//! - greedy speculation never changes greedy output: drafted and undrafted
//!   runs on the GPU produce exactly the same tokens at every width (the
//!   executor is batch-invariant, so a verify run computes each position's
//!   logits bit for bit as a decode step does);
//! - seeded sampling with drafts is reproducible, and every token sampleable;
//! - with graphs on, uniform drafted decode steps give bit-identical tokens
//!   and logits replayed, launched directly and run eagerly, and an engine
//!   workload the same tokens with graphs on as off;
//! - after every engine step the KV invariants hold and every free block is
//!   zero on the device or queued (drafter blocks and their taps included),
//!   through prefix hits (the drafter resuming from boundary taps) and
//!   preemption.
//!
//! Prints acceptance per depth, conditional on the width each row drafted
//! (`common/draft_tally.rs`, the definition `examples/eval.rs` reports); run
//! with `--nocapture`.

mod common;
#[path = "common/draft_tally.rs"]
mod draft_tally;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use common::setup;
use draft_tally::Tallying;
use eidola_engine::engine::{CacheScope, Engine, Event, FinishReason, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, SeqEntry, StepInput, TableUpdate};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::EngineSalt;
use eidola_engine::spec::Bucket;
use eidola_engine_cpu::{CpuExecutor, CpuExecutorConfig};
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, DecodePath, KvBlocks, MtpHidden,
};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{ForwardOptions, LoadOptions, LogitsAt, ModelWeights, ReferenceModel};

const KEEP: [usize; 4] = [0, 1, 2, 5];
const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;
/// A produced greedy token (or a greedy draft) may trail the reference's
/// argmax by at most this many logits: the GPU's measured max |Δlogit|
/// against the reference on this truncation is 1.18 (`engine.rs`).
const MARGIN: f32 = 1.5;

struct Env {
    store: Arc<WeightSet>,
    reference: Arc<ReferenceModel>,
    text: Vec<u32>,
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
        let store = Arc::new(WeightSet::open_dir(&PathBuf::from(dir)).unwrap());
        if !store.contains("model.mtp.layers.2.eh_proj.weight") {
            eprintln!("skipping: EIDOLA_MIMO_DIR holds no MTP layers (fetch_truncated.py without --no-mtp)");
            return None;
        }
        let g = WeightSet::open_files(&[PathBuf::from(golden)]).unwrap();
        let text: Vec<u32> = g
            .get("tokens")
            .unwrap()
            .to_i64("tokens")
            .unwrap()
            .into_iter()
            .map(|x| u32::try_from(x).unwrap())
            .collect();
        let config = store.model_config().unwrap().truncated(&KEEP).unwrap();
        let reference = Arc::new(ReferenceModel::new(
            ModelWeights::load(store.clone(), config, &LoadOptions::default()).unwrap(),
        ));
        assert!(
            text.len() >= MIN_TEXT,
            "the golden holds {} tokens; the tests need {MIN_TEXT}",
            text.len()
        );
        Some(Env {
            store,
            reference,
            text,
        })
    })
    .as_ref()
}

fn buckets() -> Vec<Bucket> {
    vec![
        Bucket {
            max_seqs: 1,
            max_tokens: 16,
        },
        Bucket {
            max_seqs: 8,
            max_tokens: 256,
        },
    ]
}

/// State slots of every executor here: the most any test seats at once
/// (`graphs_and_eager_agree_with_drafting`'s three copies of three rows).
const SLOTS: u32 = 9;

fn gpu_config(depths: u32, blocks: u32, graphs: CudaGraphs) -> CudaExecutorConfig {
    CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: blocks,
            sliding: blocks,
            drafter: if depths > 0 { blocks } else { 0 },
        },
        num_state_slots: SLOTS,
        max_model_len: 1024,
        buckets: buckets(),
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs,
        draft_tokens: depths,
        mtp_hidden: MtpHidden::Normed,
    }
}

fn gpu_executor(depths: u32, blocks: u32, graphs: CudaGraphs) -> Option<CudaExecutor> {
    let env = env()?;
    let su = setup()?;
    let cfg = gpu_config(depths, blocks, graphs);
    let t0 = std::time::Instant::now();
    let mut ex = CudaExecutor::new(su.gpu, &su.dir, env.store.clone(), Some(&KEEP), cfg).unwrap();
    ex.record_drafts = true;
    if let Some(g) = ex.draft_graphs() {
        println!(
            "D {depths}: loaded and captured in {:.1?}, rungs (width, rows) {:?}, capture bytes {:?}",
            t0.elapsed(),
            g.rungs(),
            g.capture_bytes()
        );
    }
    Some(ex)
}

fn cpu_executor(depths: u32, blocks: u32) -> CpuExecutor {
    let env = env().unwrap();
    let mut cfg = CpuExecutorConfig::for_model(&env.reference, SAMPLEABLE);
    cfg.block_size = BS;
    cfg.num_blocks = blocks;
    cfg.num_state_slots = SLOTS;
    cfg.max_model_len = 1024;
    cfg.buckets = buckets();
    cfg.mtp_depths = (0..depths as usize).collect();
    cfg.record = true;
    CpuExecutor::new(env.reference.clone(), cfg)
}

fn sched(chunk: u32, speculative: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: 256,
        max_seqs: 8,
        max_prefill_chunk: chunk,
        eos_token_ids: Vec::new(),
        speculative,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    }
}

fn salt(n: u8) -> EngineSalt {
    EngineSalt::from_bytes([n; 32])
}

/// One request of a workload, submitted at step `at`; `key` names a salt
/// (keyed requests share cached prefixes), `None` a private request.
#[derive(Clone, Debug)]
struct Job {
    at: u64,
    id: u64,
    prompt: Vec<u32>,
    sampling: SamplingParams,
    max: u32,
    key: Option<u8>,
}

impl Job {
    fn request(&self) -> Request {
        Request {
            id: self.id,
            prompt: self.prompt.clone(),
            sampling: self.sampling,
            max_tokens: self.max,
            stop_token_ids: Vec::new(),
            cache: self
                .key
                .map_or(CacheScope::Private, |k| CacheScope::Keyed(salt(k))),
        }
    }
}

fn job(
    at: u64,
    id: u64,
    prompt: &[u32],
    sampling: SamplingParams,
    max: u32,
    key: Option<u8>,
) -> Job {
    Job {
        at,
        id,
        prompt: prompt.to_vec(),
        sampling,
        max,
        key,
    }
}

/// The workload: greedy and seeded rows, a prompt sharing a keyed prefix
/// with a later one (which resumes from a cached block, its drafter from
/// the block's tap), prompts of every length class. The other prompts are
/// disjoint spans spread over whatever text the golden holds (at least
/// [`MIN_TEXT`] tokens).
fn workload(text: &[u32]) -> Vec<Job> {
    let seeded = |id: u64| SamplingParams::new(0.8, 40, 0.95, 0.0, 1000 + id).unwrap();
    let g = SamplingParams::greedy;
    // Spans after the shared prefix's 70 tokens, separated by equal gaps.
    let lens = [17usize, 3, 65, 33];
    let gap = text.len().saturating_sub(70 + lens.iter().sum::<usize>()) / lens.len();
    let mut start = 70;
    let mut span = |len: usize| {
        start += gap;
        let s = &text[start..start + len];
        start += len;
        s
    };
    let (a, b, c, d) = (span(lens[0]), span(lens[1]), span(lens[2]), span(lens[3]));
    vec![
        job(0, 0, &text[..40], g(), 24, Some(1)),
        job(0, 1, a, seeded(1), 16, None),
        job(0, 2, b, g(), 20, None),
        job(6, 3, &text[..70], g(), 16, Some(1)),
        job(6, 4, c, seeded(4), 12, None),
        job(8, 5, d, g(), 18, None),
    ]
}

/// The fewest golden tokens the tests' prompts need: the pressure prompts'
/// 225 (the workload needs 188, the graph test 165).
const MIN_TEXT: usize = 225;

/// `drafting_under_kv_pressure_preempts_and_resumes`' prompts: six greedy
/// 40-token spans, 37 apart.
fn pressure_prompts(text: &[u32]) -> HashMap<u64, (Vec<u32>, SamplingParams)> {
    (0..6u64)
        .map(|id| {
            let start = usize::try_from(id * 37).unwrap();
            (
                id,
                (text[start..start + 40].to_vec(), SamplingParams::greedy()),
            )
        })
        .collect()
}

/// `graphs_and_eager_agree_with_drafting`'s rows: contexts across a block
/// boundary and past the drafter's window, three copies of each.
const GRAPH_CONTEXTS: [u32; 3] = [21, 32, 150];

/// Row `slot`'s prefix and next token in the graph test.
fn graph_row_tokens(text: &[u32], slot: u32) -> Vec<u32> {
    let c = GRAPH_CONTEXTS[(slot % 3) as usize] as usize;
    text[(slot % 3) as usize * 7..][..c + 1].to_vec()
}

/// The GPU tests' host-side setup, run without a device: the prompts fit a
/// text of [`MIN_TEXT`] tokens, the graph test's rows fit the executors'
/// slots and its text. A setup bug then fails here, not on the GPU host.
#[test]
fn setup_fits_without_a_device() {
    let text: Vec<u32> = (0..u32::try_from(MIN_TEXT).unwrap()).collect();
    let w = workload(&text);
    assert_eq!(w.len(), 6);
    // Disjoint but for the shared prefix.
    let mut seen = std::collections::HashSet::new();
    for j in w.iter().filter(|j| j.id != 3) {
        for &t in &j.prompt {
            assert!(seen.insert(t), "job {} overlaps", j.id);
        }
    }
    assert_eq!(w[3].prompt[..40], w[0].prompt[..]);
    assert_eq!(pressure_prompts(&text).len(), 6);
    let cfg = gpu_config(3, 64, CudaGraphs::On);
    assert_eq!(cfg.num_state_slots, SLOTS);
    for slot in 0..3 * 3u32 {
        assert!(slot < cfg.num_state_slots, "graph test slot {slot}");
        graph_row_tokens(&text, slot);
    }
}

/// Runs an engine over `workload`, submitting each request at its step.
struct Run<E: Executor> {
    eng: Engine<Tallying<E>>,
    step: u64,
    pending: Vec<Job>,
    outputs: HashMap<u64, Vec<u32>>,
    finished: HashMap<u64, FinishReason>,
    cached: HashMap<u64, u32>,
}

impl<E: Executor> Run<E> {
    fn new(ex: E, chunk: u32, speculative: bool, mut pending: Vec<Job>) -> Run<E> {
        pending.reverse();
        Run {
            eng: Engine::new(Tallying::new(ex), sched(chunk, speculative))
                .expect("a valid configuration"),
            step: 0,
            pending,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
        }
    }

    /// The executor under the tally.
    fn ex(&self) -> &E {
        self.eng.executor().inner()
    }

    /// Conditional acceptance per depth (`draft_tally`), checked against
    /// the engine's own accepted/drafted totals.
    fn acceptance(&self) -> Vec<f64> {
        let (t, s) = (self.eng.executor().tally(), self.eng.stats());
        assert_eq!((t.accepted_total, t.drafted_total), (s.accepted, s.drafted));
        t.rates()
    }

    fn done(&self) -> bool {
        self.pending.is_empty() && self.eng.unfinished() == 0
    }

    fn step(&mut self) -> Vec<Event> {
        while self.pending.last().is_some_and(|j| j.at <= self.step) {
            let job = self.pending.pop().unwrap();
            self.outputs.insert(job.id, Vec::new());
            self.eng.submit(job.request()).expect("submit");
        }
        let events = self.eng.step(self.step).expect("step");
        self.step += 1;
        for e in &events {
            self.outputs.entry(e.id).or_default().extend(&e.tokens);
            if let Some(f) = e.finish {
                assert!(self.finished.insert(e.id, f).is_none(), "finished twice");
                self.cached.insert(e.id, e.cached_prompt_tokens);
            }
        }
        events
    }
}

impl Run<CudaExecutor> {
    /// The KV manager's invariants, and every free block zero on the device
    /// or queued for zeroing (drafter blocks' taps included).
    fn check_kv(&self) -> usize {
        let kv = self.eng.kv();
        kv.check_invariants();
        kv.check_no_reservations();
        let pending = kv.pending_zeros();
        let mut checked = 0;
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    self.ex().block_is_zero(g, b).unwrap(),
                    "free block {g}/{b} holds data with no zero pending"
                );
                checked += 1;
            }
        }
        checked
    }
}

/// Every greedy output within [`MARGIN`] of the f32 reference's argmax over
/// the same tokens; every output sampleable. Returns (tokens checked, of
/// which the reference's own argmax, worst trail).
fn check_outputs(
    prompts: &HashMap<u64, (Vec<u32>, SamplingParams)>,
    outputs: &HashMap<u64, Vec<u32>>,
) -> (usize, usize, f32) {
    let env = env().unwrap();
    let (mut n, mut exact, mut worst) = (0, 0, 0f32);
    let mut ids: Vec<_> = outputs.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let out = &outputs[&id];
        assert!(
            out.iter().all(|&t| t < SAMPLEABLE),
            "request {id}: padded id"
        );
        let (prompt, params) = &prompts[&id];
        if !params.is_greedy() || out.is_empty() {
            continue;
        }
        let mut tokens = prompt.clone();
        tokens.extend(&out[..out.len() - 1]);
        let positions: Vec<usize> = (prompt.len() - 1..tokens.len()).collect();
        let r = env
            .reference
            .forward(
                &tokens,
                &ForwardOptions {
                    logits: LogitsAt::Positions(positions),
                    capture_layers: false,
                },
            )
            .unwrap();
        for (i, &t) in out.iter().enumerate() {
            let row = &r.logits.row(i)[..SAMPLEABLE as usize];
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let trail = max - row[t as usize];
            worst = worst.max(trail);
            exact += usize::from(trail == 0.0);
            n += 1;
            assert!(
                trail <= MARGIN,
                "request {id} token {i}: {t} trails the reference argmax by {trail}"
            );
        }
    }
    (n, exact, worst)
}

fn prompts_of(w: &[Job]) -> HashMap<u64, (Vec<u32>, SamplingParams)> {
    w.iter()
        .map(|j| (j.id, (j.prompt.clone(), j.sampling)))
        .collect()
}

#[test]
fn drafted_steps_match_the_cpu_executor() {
    let Some(env) = env() else { return };
    if setup().is_none() {
        return;
    }
    for depths in 1..=3u32 {
        let gpu = gpu_executor(depths, 64, CudaGraphs::Off).unwrap();
        let cpu = cpu_executor(depths, 64);
        let w = workload(&env.text);
        let mut g = Run::new(gpu, 16, true, w.clone());
        let mut c = Run::new(cpu, 16, true, w.clone());
        let (mut lockstep, mut compared, mut agreed) = (true, 0usize, 0usize);
        // The worst trail of a differing draft, over greedy rows (asserted
        // within `MARGIN`) and seeded rows (a seeded draft is a draw, so its
        // trail is reported, not bounded) apart.
        let (mut worst_greedy, mut worst_seeded) = (0f32, 0f32);
        let mut zero_checks = 0;
        while !g.done() || !c.done() {
            let ge = if g.done() { Vec::new() } else { g.step() };
            let ce = if c.done() { Vec::new() } else { c.step() };
            zero_checks += g.check_kv();
            let gd = g.ex().take_drafts();
            let cr = c.ex().take_records();
            if lockstep {
                assert_eq!(gd.len(), cr.len(), "D {depths}: lockstep steps' rows");
                for (gr, crr) in gd.iter().zip(&cr) {
                    assert_eq!((gr.slot, gr.context_len), (crr.slot, crr.context_len));
                    assert_eq!(gr.drafts.len(), crr.drafts.len(), "slot {}", gr.slot);
                    for (i, (&gt, &ct)) in gr.drafts.iter().zip(&crr.drafts).enumerate() {
                        compared += 1;
                        if gt == ct {
                            agreed += 1;
                            continue;
                        }
                        // Depth i's prediction at p + i, as the CPU computed
                        // it over the same tokens.
                        let p = crr.context_len + crr.num_tokens - 1;
                        let logits = &crr
                            .drafter_logits
                            .iter()
                            .find(|(d, s, _)| *d == i && *s == p + u32::try_from(i).unwrap())
                            .expect("the CPU's drafter logits")
                            .2[..SAMPLEABLE as usize];
                        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                        let trail = max - logits[gt as usize];
                        if crr.sampling.is_greedy() {
                            worst_greedy = worst_greedy.max(trail);
                            assert!(
                                trail <= MARGIN,
                                "D {depths} slot {} at {p}: draft {} trails the CPU's by {trail}",
                                gr.slot,
                                i + 1
                            );
                        } else {
                            worst_seeded = worst_seeded.max(trail);
                        }
                        // Later drafts continue from different tokens.
                        break;
                    }
                }
                lockstep = ge == ce;
            }
        }
        let prompts = prompts_of(&w);
        let (n, exact, trail) = check_outputs(&prompts, &g.outputs);
        // Request 3 resumes from request 0's sealed blocks, its drafter from
        // a boundary tap.
        assert!(g.cached[&3] >= BS, "prefix hit: {}", g.cached[&3]);
        assert_eq!(g.cached[&3], c.cached[&3], "the same hit on both executors");
        let (gs, cs) = (g.eng.stats(), c.eng.stats());
        println!(
            "D {depths}: drafts agreeing with the CPU's {agreed}/{compared} (worst trail: greedy {worst_greedy:.3}, seeded {worst_seeded:.3}); \
             {n} greedy tokens, {exact} the reference argmax, worst trail {trail:.3}; \
             GPU accepted {}/{} drafts, CPU {}/{}; GPU acceptance per depth {:?}, CPU {:?}; \
             {zero_checks} zero checks",
            gs.accepted,
            gs.drafted,
            cs.accepted,
            cs.drafted,
            g.acceptance(),
            c.acceptance(),
        );
    }
}

#[test]
fn greedy_drafting_never_changes_greedy_output() {
    let Some(env) = env() else { return };
    if setup().is_none() {
        return;
    }
    let greedy_only: Vec<Job> = workload(&env.text)
        .into_iter()
        .map(|j| Job {
            sampling: SamplingParams::greedy(),
            ..j
        })
        .collect();
    let mut plain = Run::new(
        gpu_executor(0, 64, CudaGraphs::Off).unwrap(),
        16,
        false,
        greedy_only.clone(),
    );
    while !plain.done() {
        plain.step();
    }
    for depths in 1..=3u32 {
        let mut drafted = Run::new(
            gpu_executor(depths, 64, CudaGraphs::Off).unwrap(),
            16,
            true,
            greedy_only.clone(),
        );
        while !drafted.done() {
            drafted.step();
        }
        for r in &greedy_only {
            let (a, b) = (&plain.outputs[&r.id], &drafted.outputs[&r.id]);
            assert_eq!(
                a,
                b,
                "D {depths} request {}: drafting changed greedy output (first difference at \
                 token {:?})",
                r.id,
                a.iter().zip(b).position(|(x, y)| x != y)
            );
        }
        println!(
            "D {depths}: {} requests identical drafted and undrafted; acceptance per depth {:?}",
            greedy_only.len(),
            drafted.acceptance()
        );
    }
}

#[test]
fn seeded_drafting_is_reproducible() {
    let Some(env) = env() else { return };
    if setup().is_none() {
        return;
    }
    let seeded: Vec<Job> = workload(&env.text)
        .into_iter()
        .map(|j| Job {
            sampling: SamplingParams::new(0.9, 0, 0.95, 0.02, 7 + j.id).unwrap(),
            ..j
        })
        .collect();
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut r = Run::new(
            gpu_executor(3, 64, CudaGraphs::Off).unwrap(),
            16,
            true,
            seeded.clone(),
        );
        while !r.done() {
            r.step();
            r.check_kv();
        }
        let s = r.eng.stats();
        println!(
            "seeded, D 3: accepted {}/{} drafts, acceptance per depth {:?}",
            s.accepted,
            s.drafted,
            r.acceptance()
        );
        check_outputs(&prompts_of(&seeded), &r.outputs);
        runs.push(r.outputs);
    }
    assert_eq!(
        runs[0], runs[1],
        "a seeded drafted workload is reproducible"
    );
}

#[test]
fn drafting_under_kv_pressure_preempts_and_resumes() {
    let Some(env) = env() else { return };
    if setup().is_none() {
        return;
    }
    let prompts = pressure_prompts(&env.text);
    // Few blocks: rows shed their drafts, then preempt each other.
    let mut jobs: Vec<Job> = prompts
        .iter()
        .map(|(&id, (p, s))| job(0, id, p, *s, 14, None))
        .collect();
    jobs.sort_by_key(|j| j.id);
    let mut r = Run::new(
        gpu_executor(3, 16, CudaGraphs::Off).unwrap(),
        32,
        true,
        jobs,
    );
    let mut zero_checks = 0;
    while !r.done() {
        r.step();
        zero_checks += r.check_kv();
    }
    let (n, exact, worst) = check_outputs(&prompts, &r.outputs);
    r.eng.sweep(r.step + 1).unwrap();
    r.check_kv();
    println!(
        "pressure, D 3: {:?}; {n} greedy tokens, {exact} the reference argmax, worst trail {worst:.3}; \
         {zero_checks} zero checks",
        r.eng.stats()
    );
}

/// Uniform drafted decode steps bit-identical across replay, direct launch
/// and the eager path: three copies of the same rows (each prefilled alone,
/// so identical), one decoded per path, twice at the full width (the second
/// step loading the state the first left) and then at width 0 (a step that
/// ran short of KV; the executor captures both widths); then an
/// engine workload with graphs on and off.
#[test]
fn graphs_and_eager_agree_with_drafting() {
    let Some(env) = env() else { return };
    if setup().is_none() {
        return;
    }
    let depths = 3u32;
    let mut ex = gpu_executor(depths, 64, CudaGraphs::On).unwrap();
    // Three rows per copy, at contexts across a block boundary and past the
    // drafter's window; slot = copy * 3 + row.
    let contexts = GRAPH_CONTEXTS;
    let groups = 3u32;
    let mut updates = Vec::new();
    let mut next = 1u32;
    for slot in 0..9u32 {
        let c = contexts[(slot % 3) as usize];
        // Room for four decode steps of at most `1 + depths` tokens.
        for index in 0..=(c + 4 * (depths + 1)) / BS {
            for group in 0..groups {
                updates.push(TableUpdate {
                    slot,
                    group,
                    index,
                    block: next,
                });
            }
            next += 1;
        }
    }
    let row_tokens = |slot: u32| graph_row_tokens(&env.text, slot);
    for slot in 0..9u32 {
        let toks = row_tokens(slot);
        let c = u32::try_from(toks.len() - 1).unwrap();
        ex.set_decode_path(DecodePath::Eager);
        let step = StepInput {
            bucket: buckets()[1],
            maintenance: vec![],
            table_updates: if slot == 0 { updates.clone() } else { vec![] },
            seqs: vec![SeqEntry {
                slot,
                token_start: 0,
                num_tokens: c,
                context_len: 0,
                num_drafts: 0,
                sample: false,
                sampling: SamplingParams::greedy(),
            }],
            token_ids: toks[..c as usize].to_vec(),
            positions: (0..c).collect(),
            return_logits: false,
        };
        ex.execute(&step).unwrap();
    }
    // The prefill steps' (empty) draft records are not the compared steps'.
    ex.take_drafts();
    // By row, so the three copies sample alike: rows 0 and 2 greedy, row 1
    // seeded.
    let sampling = |slot: u32| {
        if (slot % 3).is_multiple_of(2) {
            SamplingParams::greedy()
        } else {
            SamplingParams::new(0.8, 40, 0.95, 0.0, 99).unwrap()
        }
    };
    let mut next_tok: Vec<u32> = (0..9u32).map(|s| *row_tokens(s).last().unwrap()).collect();
    let mut ctx: Vec<u32> = (0..9u32).map(|s| contexts[(s % 3) as usize]).collect();
    // Two rounds at the full width (the second loading the state the first
    // left), then the width a step that ran short of KV is given.
    for (round, width) in [depths, depths, 0].into_iter().enumerate() {
        let mut results = Vec::new();
        for (copy, path) in [DecodePath::Eager, DecodePath::Direct, DecodePath::Replay]
            .into_iter()
            .enumerate()
        {
            let slots: Vec<u32> = (0..3)
                .map(|r| u32::try_from(copy).unwrap() * 3 + r)
                .collect();
            ex.set_decode_path(path);
            let step = StepInput {
                bucket: buckets()[1],
                maintenance: vec![],
                table_updates: vec![],
                seqs: slots
                    .iter()
                    .enumerate()
                    .map(|(i, &slot)| SeqEntry {
                        slot,
                        token_start: u32::try_from(i).unwrap(),
                        num_tokens: 1,
                        context_len: ctx[slot as usize],
                        num_drafts: width,
                        sample: true,
                        sampling: sampling(slot),
                    })
                    .collect(),
                token_ids: slots.iter().map(|&s| next_tok[s as usize]).collect(),
                positions: slots.iter().map(|&s| ctx[s as usize]).collect(),
                return_logits: true,
            };
            let before = ex.decode_stats();
            let out = ex.execute(&step).unwrap();
            let after = ex.decode_stats();
            match path {
                DecodePath::Eager => assert_eq!(after.eager, before.eager + 1),
                DecodePath::Direct => assert_eq!(after.direct, before.direct + 1),
                DecodePath::Replay => assert_eq!(after.replayed, before.replayed + 1),
            }
            for (i, &slot) in slots.iter().enumerate() {
                let row = out.row(i);
                ctx[slot as usize] += u32::try_from(row.len()).unwrap();
                next_tok[slot as usize] = *row.last().unwrap();
            }
            let drafts: Vec<Vec<u32>> = ex.take_drafts().into_iter().map(|d| d.drafts).collect();
            results.push((
                (0..3).map(|i| out.row(i).to_vec()).collect::<Vec<_>>(),
                out.logits.unwrap(),
                drafts,
            ));
        }
        for (name, other) in [("direct", &results[1]), ("replay", &results[2])] {
            assert_eq!(results[0].0, other.0, "round {round}: {name} tokens");
            assert_eq!(results[0].2, other.2, "round {round}: {name} drafts");
            let bits = |l: &Vec<Vec<Vec<f32>>>| -> Vec<u32> {
                l.iter().flatten().flatten().map(|x| x.to_bits()).collect()
            };
            assert_eq!(
                bits(&results[0].1),
                bits(&other.1),
                "round {round}: {name} logits"
            );
        }
        println!(
            "round {round}: tokens {:?}, drafts {:?}",
            results[0].0, results[0].2
        );
    }
    println!("{:?}", ex.decode_stats());
    drop(ex);

    // The engine workload, graphs on and off: the same tokens.
    let mut outs = Vec::new();
    for graphs in [CudaGraphs::Off, CudaGraphs::On] {
        let mut r = Run::new(
            gpu_executor(depths, 64, graphs).unwrap(),
            16,
            true,
            workload(&env.text),
        );
        while !r.done() {
            r.step();
            r.check_kv();
        }
        println!("graphs {graphs:?}: {:?}", r.ex().decode_stats());
        if graphs == CudaGraphs::On {
            assert!(r.ex().decode_stats().replayed > 0, "some steps replayed");
        }
        outs.push(r.outputs);
    }
    assert_eq!(outs[0], outs[1], "graphs on and off give the same tokens");
}
