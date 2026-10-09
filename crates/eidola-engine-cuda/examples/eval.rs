//! Evaluation driver for the CUDA executor on a MiMo-V2.6 checkpoint, built so
//! the same token ids can be fed to another engine (an SGLang oracle) and both
//! outputs scored by the same code.
//!
//! ```text
//! eval render   <model_dir> <tasks.jsonl> <prompts.jsonl>
//! eval generate <kernels_dir> <model_dir> <prompts.jsonl> <outputs.jsonl> <max_tokens> [--draft-tokens K]
//! eval logprobs <kernels_dir> <model_dir> <prompts.jsonl> <out.jsonl> [--reference] [--full <out.f32>]
//! eval score    <model_dir> <tasks.jsonl> <outputs.jsonl>
//! eval compare  <a.jsonl> <b.jsonl> [--full <a.f32> <b.f32>] [--from <prompts.jsonl>]
//! ```
//!
//! - `tasks.jsonl`: `{"id", "messages", "tools"?, "enable_thinking"?, "kind",
//!   "answer"? | "expect"?}` — `kind` is `"gsm8k"` (with a numeric `answer`) or
//!   `"tool"` (with `expect`: `{"name", "arguments", "free"?}`, or `null` for
//!   "no call"). A tool task passes only with exactly one call of that name
//!   whose arguments are exactly `arguments` plus each `free` key (free text,
//!   value not scored), and nothing else.
//! - `prompts.jsonl`: `{"id", "prompt_ids"}` (the chat template rendered with
//!   the generation prompt, then encoded).
//! - `outputs.jsonl`: `{"id", "output_ids"}` (greedy, stopped at EOS).
//! - `logprobs`: per prompt, the 20 most likely next tokens at every position
//!   with their log-probabilities over the sampleable vocabulary, from the GPU
//!   executor or (`--reference`) the f32 reference forward. `--full` also
//!   writes every position's whole log-probability row (little-endian f32,
//!   sampleable vocabulary wide, prompts in order).
//! - `compare`: top-1 agreement and top-20 overlap always. With both sides'
//!   full rows (`--full`), KL(a ‖ b) itself. With only top-20 lists (another
//!   engine's API), never KL but a labelled lower bound on it: the KL of both
//!   distributions coarse-grained to the tokens both lists hold plus one
//!   bucket for all other mass, which by the data-processing inequality can
//!   only understate KL. `--from` restricts both to positions from each
//!   prompt's `start` on. Ids and per-id position counts must match exactly.
//!
//! The executor replays decode graphs when `EIDOLA_ENGINE_CUDA_GRAPHS` is `on`
//! (`off`, eager, when unset), so `generate` runs with each setting give the
//! outputs to compare. `generate --draft-tokens K` drafts `K` tokens a step
//! with the checkpoint's MTP layers (greedy speculation: the outputs are the
//! undrafted run's but for near-ties) and prints the acceptance of each draft
//! depth over the prompts, conditional: draft `d` accepted among the rows
//! that drafted at least `d` tokens in a step and had their first `d - 1`
//! accepted. Rows are counted from what the executor was asked to draft, so a
//! row that drafted nothing (a prefill, a step narrowed to width 0) or fewer
//! than `d` tokens (narrowed by the serving core to stay in the masked expert
//! layout) is outside depth `d`'s denominator, not a rejection there. A second
//! line gives each depth's counts and the width distribution of sampled
//! decode rows (with sampled prefill rows apart), so narrowing is visible; the
//! first line's totals are the engine's own accepted/drafted.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{
    Executor, ExecutorError, SeqEntry, StepInput, StepOutput, TableUpdate,
};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_chat::json::Json;
use eidola_engine_chat::tool_call::parse_complete;
use eidola_engine_chat::{ChatInput, ChatTemplate, MimoTokenizer, RenderOptions, ToolSchemas};
use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, CudaGraphs, Gpu, KvBlocks, MtpHidden};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{ForwardOptions, LoadOptions, LogitsAt, ModelWeights, ReferenceModel};
use serde_json::{Value, json};

const TOP: usize = 20;

/// Tokens per executor step (the largest bucket's): longer prompts are
/// prefilled in chunks of this many.
const STEP_TOKENS: u32 = 2048;

fn read_jsonl(path: &str) -> Vec<Value> {
    let f = std::io::BufReader::new(
        std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")),
    );
    f.lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect()
}

fn write_jsonl(path: &str, rows: &[Value]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for r in rows {
        writeln!(f, "{r}").unwrap();
    }
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .unwrap_or_else(|| panic!("token id {x} is not a u32"))
        })
        .collect()
}

fn executor(
    kernels: &str,
    model: &str,
    max_tokens: u32,
    max_seqs: u32,
    depths: u32,
) -> (CudaExecutor, u32) {
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let gpu = Gpu::open(0).unwrap();
    let dir = KernelDir::new(kernels);
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let cfg = CudaExecutorConfig {
        block_size: 16,
        num_blocks: KvBlocks {
            global: 16_384,
            sliding: 4_096,
            drafter: if depths > 0 { 4_096 } else { 0 },
        },
        num_state_slots: max_seqs,
        max_model_len: 16_384,
        buckets: vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 16,
                max_tokens: 256,
            },
            Bucket {
                max_seqs,
                max_tokens,
            },
        ],
        sampleable_vocab_size: u32::try_from(tok.vocab_size()).unwrap(),
        image: None,
        graphs: std::env::var("EIDOLA_ENGINE_CUDA_GRAPHS").map_or(CudaGraphs::Off, |v| {
            CudaGraphs::parse(&v).unwrap_or_else(|e| panic!("EIDOLA_ENGINE_CUDA_GRAPHS: {e}"))
        }),
        draft_tokens: depths,
        mtp_hidden: MtpHidden::Normed,
    };
    let t0 = Instant::now();
    let ex = CudaExecutor::new(gpu, &dir, store, None, cfg).unwrap();
    eprintln!("executor loaded in {:.1?}", t0.elapsed());
    (ex, u32::try_from(tok.vocab_size()).unwrap())
}

fn render(model: &str, tasks: &str, out: &str) {
    let template = ChatTemplate::from_model_dir(Path::new(model)).unwrap();
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let rows: Vec<Value> = read_jsonl(tasks)
        .iter()
        .map(|t| {
            let tools = t
                .get("tools")
                .filter(|v| !v.is_null())
                .map(|v| v.to_string());
            let mut input =
                ChatInput::from_json(&t["messages"].to_string(), tools.as_deref()).unwrap();
            input.normalize_tool_call_arguments().unwrap();
            let text = template
                .render(
                    &input,
                    RenderOptions {
                        add_generation_prompt: true,
                        enable_thinking: t.get("enable_thinking").and_then(Value::as_bool),
                    },
                )
                .unwrap();
            json!({"id": t["id"], "prompt_ids": tok.encode(&text).unwrap()})
        })
        .collect();
    write_jsonl(out, &rows);
    eprintln!("rendered {} prompts", rows.len());
}

fn generate(kernels: &str, model: &str, prompts: &str, out: &str, max_new: u32, depths: u32) {
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let (ex, _) = executor(kernels, model, 2048, 64, depths);
    let ex = Tallying {
        inner: ex,
        tally: std::cell::RefCell::new(DraftTally::new(depths as usize)),
    };
    let sched = SchedulerConfig {
        max_batched_tokens: 2048,
        max_seqs: 64,
        max_prefill_chunk: 2048,
        eos_token_ids: tok.eos_token_ids().to_vec(),
        speculative: depths > 0,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    };
    let mut eng = Engine::new(ex, sched).unwrap();
    let rows = read_jsonl(prompts);
    by_id(&rows, "prompts");
    let mut index = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        index.insert(i as u64, r["id"].clone());
        eng.submit(Request {
            id: i as u64,
            prompt: ids(&r["prompt_ids"]),
            sampling: SamplingParams::greedy(),
            max_tokens: max_new,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
        })
        .unwrap();
    }
    let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut finish: HashMap<u64, String> = HashMap::new();
    let (t0, mut steps, mut produced) = (Instant::now(), 0u64, 0usize);
    // Steps in which a request produced tokens, over every request.
    let mut row_steps = 0usize;
    while eng.unfinished() > 0 {
        for e in eng.step(steps).unwrap() {
            produced += e.tokens.len();
            if !e.tokens.is_empty() {
                row_steps += 1;
            }
            outputs.entry(e.id).or_default().extend(&e.tokens);
            if let Some(f) = e.finish {
                finish.insert(e.id, format!("{f:?}"));
            }
        }
        steps += 1;
        if steps % 200 == 0 {
            eprintln!(
                "step {steps}: {produced} tokens in {:.1?}, {} unfinished",
                t0.elapsed(),
                eng.unfinished()
            );
        }
    }
    eprintln!(
        "{} requests, {produced} tokens, {steps} steps in {:.1?} ({:?})",
        rows.len(),
        t0.elapsed(),
        eng.stats()
    );
    if depths > 0 {
        let s = eng.stats();
        let rate = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        let t = eng.executor().tally.borrow();
        assert_eq!(
            (t.accepted_total, t.drafted_total),
            (s.accepted, s.drafted),
            "the tally disagrees with the engine's own counts"
        );
        let depth: Vec<String> = t
            .rates()
            .iter()
            .enumerate()
            .map(|(i, r)| format!("draft {}: {:.3}", i + 1, r))
            .collect();
        eprintln!(
            "drafting {depths}: accepted {}/{} drafts ({:.3}), {:.3} tokens per step; {}",
            s.accepted,
            s.drafted,
            rate(s.accepted, s.drafted),
            produced as f64 / row_steps.max(1) as f64,
            depth.join(", ")
        );
        let counts: Vec<String> = t
            .accepted
            .iter()
            .zip(&t.reached)
            .enumerate()
            .map(|(i, (a, r))| format!("draft {}: {a}/{r}", i + 1))
            .collect();
        let widths: Vec<String> = t
            .widths
            .iter()
            .enumerate()
            .map(|(w, n)| format!("w{w} {n}"))
            .collect();
        eprintln!(
            "drafting {depths} counts: {}; decode rows by width: {}; sampled prefill rows: {}",
            counts.join(", "),
            widths.join(", "),
            t.prefill
        );
    }
    let out_rows: Vec<Value> = (0..rows.len() as u64)
        .map(|i| json!({"id": index[&i], "output_ids": outputs.get(&i).cloned().unwrap_or_default(), "finish": finish[&i]}))
        .collect();
    write_jsonl(out, &out_rows);
}

/// Per-row drafting outcomes of every step, tallied from what the executor
/// was asked to draft and what it produced (before the engine cuts a row at
/// EOS, so the totals are the engine's `Stats::drafted` and `accepted`).
#[derive(Debug, Default, PartialEq)]
struct DraftTally {
    /// Sampled decode rows (one host token) by draft width, `0 ..= k`.
    widths: Vec<u64>,
    /// Sampled rows with more than one host token (prefill ends): never drafted.
    prefill: u64,
    /// Per depth `d` (index `d - 1`): rows that drafted at least `d` tokens
    /// and had their first `d - 1` accepted.
    reached: Vec<u64>,
    /// Per depth `d`: those of `reached` whose draft `d` was accepted.
    accepted: Vec<u64>,
    /// Drafts proposed and accepted, over every drafting row.
    drafted_total: u64,
    accepted_total: u64,
}

impl DraftTally {
    fn new(depths: usize) -> Self {
        Self {
            widths: vec![0; depths + 1],
            reached: vec![0; depths],
            accepted: vec![0; depths],
            ..Self::default()
        }
    }

    /// One row of a step: its entry, and the tokens it produced.
    fn record(&mut self, e: &SeqEntry, produced: usize) {
        if !e.sample {
            return;
        }
        if e.num_tokens > 1 {
            self.prefill += 1;
            return;
        }
        let width = e.num_drafts as usize;
        self.widths[width] += 1;
        let accepted = produced.saturating_sub(1);
        self.drafted_total += width as u64;
        self.accepted_total += accepted as u64;
        for d in 1..=width {
            if accepted < d - 1 {
                break;
            }
            self.reached[d - 1] += 1;
            if accepted >= d {
                self.accepted[d - 1] += 1;
            }
        }
    }

    /// Conditional acceptance of each depth (0 where no row reached it).
    fn rates(&self) -> Vec<f64> {
        self.reached
            .iter()
            .zip(&self.accepted)
            .map(|(&r, &a)| if r == 0 { 0.0 } else { a as f64 / r as f64 })
            .collect()
    }
}

/// An executor that tallies every step's drafting rows (`DraftTally`).
struct Tallying<E> {
    inner: E,
    tally: std::cell::RefCell<DraftTally>,
}

impl<E: Executor> Executor for Tallying<E> {
    fn spec(&self) -> &ModelSpec {
        self.inner.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        let out = self.inner.execute(step)?;
        let tally = self.tally.get_mut();
        for (i, e) in step.seqs.iter().enumerate() {
            tally.record(e, out.num_tokens[i] as usize);
        }
        Ok(out)
    }
}

/// Log-probabilities of one logit row over its first `n` ids.
fn log_softmax(row: &[f32], n: usize) -> Vec<f32> {
    let row = &row[..n];
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let lse = max
        + row
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln();
    // Log-probabilities are kept at f32, the precision the logits had.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 log-probabilities stored as f32"
    )]
    row.iter().map(|&v| (v as f64 - lse) as f32).collect()
}

/// Top-`TOP` (id, log-probability) of one log-probability row.
fn top(lp: &[f32]) -> Vec<(u32, f64)> {
    let mut idx: Vec<u32> = (0..u32::try_from(lp.len()).unwrap()).collect();
    // One total order (log-probability descending, then id ascending) for
    // both the cut and the sort: which of several tied tokens make the list
    // never depends on the selection algorithm.
    let order = |&a: &u32, &b: &u32| lp[b as usize].total_cmp(&lp[a as usize]).then(a.cmp(&b));
    if idx.len() > TOP {
        idx.select_nth_unstable_by(TOP, order);
        idx.truncate(TOP);
    }
    idx.sort_by(order);
    idx.iter().map(|&i| (i, lp[i as usize] as f64)).collect()
}

/// Writes rows' top lists, and their whole rows when `full` is open.
struct LogprobSink {
    full: Option<std::io::BufWriter<std::fs::File>>,
}

impl LogprobSink {
    fn row(&mut self, logits: &[f32], n: usize) -> Vec<(u32, f64)> {
        let lp = log_softmax(logits, n);
        if let Some(f) = &mut self.full {
            for v in &lp {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        top(&lp)
    }
}

fn logprobs(
    kernels: &str,
    model: &str,
    prompts: &str,
    out: &str,
    reference: bool,
    full: Option<&str>,
) {
    let rows = read_jsonl(prompts);
    by_id(&rows, "prompts");
    let vocab = 152_576usize;
    let mut sink = LogprobSink {
        full: full.map(|p| std::io::BufWriter::new(std::fs::File::create(p).unwrap())),
    };
    let mut result = Vec::new();
    if reference {
        let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
        let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
        let config = store.model_config().unwrap();
        let opts = LoadOptions {
            load_mtp: false,
            ..LoadOptions::default()
        };
        let m = ReferenceModel::new(ModelWeights::load(store, config, &opts).unwrap());
        for r in &rows {
            let p = ids(&r["prompt_ids"]);
            let t0 = Instant::now();
            let f = m
                .forward(
                    &p,
                    &ForwardOptions {
                        logits: LogitsAt::All,
                        capture_layers: false,
                    },
                )
                .unwrap();
            eprintln!("reference: {} positions in {:.1?}", p.len(), t0.elapsed());
            let tops: Vec<_> = (0..p.len())
                .map(|i| sink.row(f.logits.row(i), tok.vocab_size()))
                .collect();
            result.push(json!({"id": r["id"], "top": tops}));
        }
    } else {
        let (mut ex, n) = executor(kernels, model, STEP_TOKENS, 16, 0);
        ex.record_all_logits = true;
        let max_len = ex.config().max_model_len;
        for r in &rows {
            let p = ids(&r["prompt_ids"]);
            // Prefilled in chunks of the executor's step budget, as serving
            // does; a prompt past the model length is refused by name.
            let len = u32::try_from(p.len())
                .ok()
                .filter(|&l| l <= max_len)
                .unwrap_or_else(|| {
                    panic!(
                        "prompt {}: {} tokens, past the executor's {max_len}-token model length",
                        r["id"],
                        p.len()
                    )
                });
            let blocks = len.div_ceil(16);
            let mut tops = Vec::with_capacity(p.len());
            let mut start = 0u32;
            while start < len {
                let chunk = (len - start).min(STEP_TOKENS);
                // The first chunk maps and zeroes every block the prompt needs.
                let (maintenance, table_updates) = if start == 0 {
                    let mut m = vec![eidola_engine::executor::Maintenance::ResetSlot { slot: 0 }];
                    for g in 0..2 {
                        for b in 1..=blocks {
                            m.push(eidola_engine::executor::Maintenance::Zero {
                                group: g,
                                block: b,
                            });
                        }
                    }
                    let u: Vec<TableUpdate> = (0..2)
                        .flat_map(|group| {
                            (0..blocks).map(move |i| TableUpdate {
                                slot: 0,
                                group,
                                index: i,
                                block: 1 + i,
                            })
                        })
                        .collect();
                    (m, u)
                } else {
                    (Vec::new(), Vec::new())
                };
                let step = StepInput {
                    bucket: *ex.config().buckets.last().expect("a bucket"),
                    maintenance,
                    table_updates,
                    seqs: vec![SeqEntry {
                        slot: 0,
                        token_start: 0,
                        num_tokens: chunk,
                        context_len: start,
                        num_drafts: 0,
                        sample: true,
                        sampling: SamplingParams::greedy(),
                    }],
                    token_ids: p[start as usize..(start + chunk) as usize].to_vec(),
                    positions: (start..start + chunk).collect(),
                    return_logits: false,
                };
                ex.execute(&step).unwrap();
                let all = ex.take_all_logits().unwrap();
                tops.extend(
                    (0..chunk as usize)
                        .map(|i| sink.row(&all[i * vocab..(i + 1) * vocab], n as usize)),
                );
                start += chunk;
            }
            assert_eq!(tops.len(), p.len());
            result.push(json!({"id": r["id"], "top": tops}));
        }
    }
    write_jsonl(out, &result);
    if let Some(mut f) = sink.full {
        f.flush().unwrap();
    }
}

/// One position's top list.
fn top_list(v: &Value) -> Vec<(u64, f64)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|e| (e[0].as_u64().unwrap(), e[1].as_f64().unwrap()))
        .collect()
}

/// A lower bound on KL(a ‖ b) from top lists alone: both distributions
/// coarse-grained to the tokens both lists hold plus one bucket for all
/// other mass. Coarse-graining never increases KL, so this never overstates
/// it; a token in only one list contributes only through the bucket.
fn kl_lower_bound(ta: &[(u64, f64)], tb: &[(u64, f64)]) -> f64 {
    let lb: HashMap<u64, f64> = tb.iter().copied().collect();
    let (mut kl, mut sa, mut sb) = (0f64, 0f64, 0f64);
    for &(t, la) in ta {
        if let Some(&lbt) = lb.get(&t) {
            kl += la.exp() * (la - lbt);
            sa += la.exp();
            sb += lbt.exp();
        }
    }
    let (ra, rb) = ((1.0 - sa).max(0.0), (1.0 - sb).max(f64::MIN_POSITIVE));
    if ra > 0.0 {
        kl += ra * (ra / rb).ln();
    }
    kl
}

/// Rows by id, refusing a duplicate id: a repeated id would otherwise
/// shadow or double-count a prompt.
fn by_id<'a>(rows: &'a [Value], what: &str) -> HashMap<String, &'a Value> {
    let mut m = HashMap::with_capacity(rows.len());
    for r in rows {
        let id = r["id"].to_string();
        assert!(m.insert(id.clone(), r).is_none(), "{what}: id {id} twice");
    }
    m
}

/// The whole log-probability rows of a `--full` file, found by prompt id
/// through the order of the top-list file written beside it.
struct FullRows {
    file: std::fs::File,
    width: usize,
    /// Each id's first row.
    first: HashMap<String, usize>,
}

impl FullRows {
    fn open(path: &str, tops: &[Value]) -> FullRows {
        let mut first = HashMap::new();
        let mut rows = 0;
        for r in tops {
            first.insert(r["id"].to_string(), rows);
            rows += r["top"].as_array().unwrap().len();
        }
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let bytes = usize::try_from(file.metadata().unwrap().len()).unwrap();
        assert!(
            rows > 0 && bytes.is_multiple_of(rows * 4),
            "{path}: {bytes} bytes is not {rows} whole rows"
        );
        FullRows {
            file,
            width: bytes / (rows * 4),
            first,
        }
    }

    /// Row `pos` of prompt `id`.
    fn row(&mut self, id: &str, pos: usize) -> Vec<f64> {
        use std::io::{Read, Seek, SeekFrom};
        let row = self.first[id] + pos;
        let mut buf = vec![0u8; self.width * 4];
        self.file
            .seek(SeekFrom::Start((row * self.width * 4) as u64))
            .unwrap();
        self.file.read_exact(&mut buf).unwrap();
        buf.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c) as f64)
            .collect()
    }
}

/// What `compare` reports.
#[derive(Debug, Default, PartialEq)]
struct Metrics {
    positions: usize,
    top1: usize,
    /// Tokens in both top lists, summed over positions.
    overlap: usize,
    lower_bound_sum: f64,
    lower_bound_max: f64,
    /// KL(a ‖ b) from whole rows (`--full` only).
    kl: Option<(f64, f64)>,
}

/// Compare two log-probability files, pairing every position by prompt id
/// and position, never by file order.
fn compare_rows(
    a: &[Value],
    b: &[Value],
    full: Option<(&str, &str)>,
    starts: Option<&HashMap<String, usize>>,
) -> Metrics {
    let (ia, ib) = (by_id(a, "a"), by_id(b, "b"));
    assert_eq!(ia.len(), ib.len(), "the files hold different prompt sets");
    // Every id in both files, with the same number of positions: a
    // shortened side would otherwise drop out of every metric unseen.
    for (id, ra) in &ia {
        let rb = ib
            .get(id)
            .unwrap_or_else(|| panic!("id {id} is missing from b"));
        let (na, nb) = (
            ra["top"].as_array().unwrap().len(),
            rb["top"].as_array().unwrap().len(),
        );
        assert_eq!(na, nb, "id {id}: {na} positions in a, {nb} in b");
        // A region must start inside the prompt: one starting at or past
        // its end would drop the prompt from every metric unseen.
        if let Some(starts) = starts {
            let start = *starts
                .get(id)
                .unwrap_or_else(|| panic!("id {id} has no start"));
            assert!(
                start < na,
                "id {id}: start {start} is not inside its {na} positions"
            );
        }
    }
    if let Some(starts) = starts {
        assert_eq!(
            starts.len(),
            ia.len(),
            "the region file names other prompts"
        );
    }
    let mut full = full.map(|(fa, fb)| {
        let (fa, fb) = (FullRows::open(fa, a), FullRows::open(fb, b));
        assert_eq!(fa.width, fb.width, "full rows of different widths");
        (fa, fb)
    });
    let mut m = Metrics::default();
    let (mut kl_sum, mut kl_max) = (0f64, 0f64);
    for ra in a {
        let id = ra["id"].to_string();
        let rb = ib[&id];
        let start = starts.map_or(0, |s| {
            *s.get(&id).unwrap_or_else(|| panic!("id {id} has no start"))
        });
        let (pa, pb) = (ra["top"].as_array().unwrap(), rb["top"].as_array().unwrap());
        for pos in start..pa.len() {
            let (ta, tb) = (top_list(&pa[pos]), top_list(&pb[pos]));
            m.positions += 1;
            m.top1 += (ta[0].0 == tb[0].0) as usize;
            m.overlap += ta
                .iter()
                .filter(|(t, _)| tb.iter().any(|(u, _)| u == t))
                .count();
            let lb = kl_lower_bound(&ta, &tb);
            m.lower_bound_sum += lb;
            m.lower_bound_max = m.lower_bound_max.max(lb);
            if let Some((fa, fb)) = &mut full {
                let (la, lb) = (fa.row(&id, pos), fb.row(&id, pos));
                let kl: f64 = la
                    .iter()
                    .zip(&lb)
                    .map(|(&x, &y)| {
                        if x == f64::NEG_INFINITY {
                            0.0
                        } else {
                            x.exp() * (x - y)
                        }
                    })
                    .sum();
                kl_sum += kl;
                kl_max = kl_max.max(kl);
            }
        }
    }
    if full.is_some() {
        m.kl = Some((kl_sum, kl_max));
    }
    m
}

fn compare(a: &str, b: &str, full: Option<(&str, &str)>, from: Option<&str>) {
    let (a, b) = (read_jsonl(a), read_jsonl(b));
    let starts: Option<HashMap<String, usize>> = from.map(|p| {
        let rows = read_jsonl(p);
        by_id(&rows, "prompts")
            .into_iter()
            .map(|(id, r)| (id, usize::try_from(r["start"].as_u64().unwrap()).unwrap()))
            .collect()
    });
    let m = compare_rows(&a, &b, full, starts.as_ref());
    let n = m.positions as f64;
    print!(
        "positions {}: top-1 {}/{} ({:.2}%), top-{TOP} overlap {:.2}%, KL lower bound (shared top-{TOP} + remainder) mean {:.3e} max {:.3e}",
        m.positions,
        m.top1,
        m.positions,
        100.0 * m.top1 as f64 / n,
        100.0 * m.overlap as f64 / (n * TOP as f64),
        m.lower_bound_sum / n,
        m.lower_bound_max
    );
    if let Some((sum, max)) = m.kl {
        print!(", KL mean {:.3e} max {max:.3e}", sum / n);
    }
    println!();
}

fn number(s: &str) -> Option<f64> {
    let t: String = s
        .chars()
        .filter(|c| !matches!(c, ',' | '$' | ' '))
        .collect();
    t.trim_end_matches('.').parse().ok()
}

/// The final answer of a GSM8K response: the last `\boxed{…}`, else the last
/// number in the text.
fn gsm8k_answer(text: &str) -> Option<f64> {
    if let Some(p) = text.rfind("\\boxed{") {
        let rest = &text[p + 7..];
        if let Some(e) = rest.find('}')
            && let Some(x) = number(&rest[..e])
        {
            return Some(x);
        }
    }
    let mut last = None;
    let mut cur = String::new();
    for c in text.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit()
            || c == '.'
            || (c == ',' && !cur.is_empty())
            || (c == '-' && cur.is_empty())
        {
            cur.push(c);
        } else {
            if let Some(x) = number(&cur) {
                last = Some(x);
            }
            cur.clear();
        }
    }
    last
}

/// An integral JSON number, exactly: integers as parsed (64-bit, never
/// through f64), and a float only when it is a whole number f64 holds
/// exactly (`5.0`).
fn integral(n: &serde_json::Number) -> Option<i128> {
    if let Some(i) = n.as_i64() {
        return Some(i128::from(i));
    }
    if let Some(u) = n.as_u64() {
        return Some(i128::from(u));
    }
    let f = n.as_f64()?;
    // Below 2^53 every whole f64 is the integer it prints as.
    (f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0).then(|| {
        #[expect(clippy::cast_possible_truncation, reason = "a whole number below 2^53")]
        let i = f as i64;
        i128::from(i)
    })
}

/// Two JSON numbers are one value: integers compared exactly (`5` and `5.0`
/// are one; `2^53 + 1` and `2^53` are not), other numbers as parsed.
fn numbers_match(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    match (integral(a), integral(b)) {
        (Some(x), Some(y)) => x == y,
        (None, None) => a.as_f64() == b.as_f64(),
        _ => false,
    }
}

/// Exact equality of JSON values: objects with exactly the same keys,
/// arrays element for element, numbers by value ([`numbers_match`]),
/// everything else by type and value.
fn values_match(want: &Value, got: &Value) -> bool {
    match (want, got) {
        (Value::Number(a), Value::Number(b)) => numbers_match(a, b),
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|g| values_match(v, g)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_match(x, y))
        }
        (a, b) => a == b,
    }
}

/// Whether a call's arguments are exactly what `expect` names: its
/// `arguments`, value for value, plus each `free` key (free text, such as a
/// search query, whose wording is not scored), and no other key.
fn arguments_match(expect: &Value, got: &Value) -> bool {
    let (Some(want), Some(got)) = (expect["arguments"].as_object(), got.as_object()) else {
        return false;
    };
    let free: Vec<&str> = expect
        .get("free")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |f| {
            f.iter().map(|k| k.as_str().unwrap()).collect()
        });
    assert!(
        free.iter().all(|k| !want.contains_key(*k)),
        "a free key with an expected value: {expect}"
    );
    got.len() == want.len() + free.len()
        && free.iter().all(|k| got.contains_key(*k))
        && want
            .iter()
            .all(|(k, v)| got.get(k).is_some_and(|g| values_match(v, g)))
}

fn score(model: &str, tasks: &str, outputs: &str) {
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let (outputs, tasks) = (read_jsonl(outputs), read_jsonl(tasks));
    // One output per task, paired by id: a missing, repeated or extra output
    // would otherwise go uncounted.
    let outs = by_id(&outputs, "outputs");
    assert_eq!(
        by_id(&tasks, "tasks").len(),
        outs.len(),
        "outputs and tasks differ"
    );
    let (mut gsm, mut gsm_ok, mut tool, mut tool_ok, mut truncated) = (0, 0, 0, 0, 0);
    for t in &tasks {
        let o = outs
            .get(&t["id"].to_string())
            .unwrap_or_else(|| panic!("no output for {}", t["id"]));
        let out_ids = ids(&o["output_ids"]);
        if !out_ids.last().is_some_and(|&x| tok.is_eos(x)) {
            truncated += 1;
        }
        let text = tok.decode(&out_ids, true).unwrap();
        let thinking = t.get("enable_thinking").and_then(Value::as_bool) != Some(false);
        let content = eidola_engine_chat::reasoning::split_reasoning(&text, thinking).content;
        match t["kind"].as_str().unwrap() {
            "gsm8k" => {
                gsm += 1;
                let want = t["answer"].as_f64().unwrap();
                let ok = gsm8k_answer(&content).is_some_and(|x| (x - want).abs() < 1e-6);
                gsm_ok += ok as usize;
                println!("{} gsm8k {}", t["id"], if ok { "ok" } else { "WRONG" });
            }
            "tool" => {
                tool += 1;
                let schemas = ToolSchemas::from_tools(&Json::from_serde(&t["tools"]));
                let parsed = parse_complete(&content, &schemas);
                let ok = match &t["expect"] {
                    Value::Null => parsed.calls.is_empty(),
                    // Exactly the one expected call: a further call would
                    // run an unintended tool.
                    e => matches!(&parsed.calls[..], [(name, args)]
                        if name == e["name"].as_str().unwrap()
                            && serde_json::from_str::<Value>(args)
                                .is_ok_and(|a| arguments_match(e, &a))),
                };
                tool_ok += ok as usize;
                println!(
                    "{} tool {} {:?}",
                    t["id"],
                    if ok { "ok" } else { "WRONG" },
                    parsed.calls
                );
            }
            k => panic!("task kind {k}"),
        }
    }
    println!(
        "gsm8k {gsm_ok}/{gsm}  tool calls {tool_ok}/{tool}  (outputs without EOS: {truncated})"
    );
}

/// The `count` values after `name` in `args`, if it is there.
fn flag<'a>(args: &'a [String], name: &str, count: usize) -> Option<Vec<&'a str>> {
    let i = args.iter().position(|x| x == name)?;
    let v: Vec<&str> = args[i + 1..]
        .iter()
        .take(count)
        .map(String::as_str)
        .collect();
    assert_eq!(v.len(), count, "{name} takes {count} values");
    Some(v)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("render") => render(&a[2], &a[3], &a[4]),
        Some("generate") => generate(
            &a[2],
            &a[3],
            &a[4],
            &a[5],
            a[6].parse().unwrap(),
            flag(&a[7..], "--draft-tokens", 1).map_or(0, |v| v[0].parse().unwrap()),
        ),
        Some("logprobs") => logprobs(
            &a[2],
            &a[3],
            &a[4],
            &a[5],
            a[6..].iter().any(|x| x == "--reference"),
            flag(&a[6..], "--full", 1).map(|v| v[0]),
        ),
        Some("score") => score(&a[2], &a[3], &a[4]),
        Some("compare") => compare(
            &a[2],
            &a[3],
            flag(&a[4..], "--full", 2).map(|v| (v[0], v[1])),
            flag(&a[4..], "--from", 1).map(|v| v[0]),
        ),
        _ => eprintln!("usage: see the module docs"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ties are cut by id inside the selection itself: with more than `TOP`
    /// tokens tied (at the maximum, or at the cut), the lowest ids are kept,
    /// in id order.
    #[test]
    fn top_lists_break_ties_by_id() {
        let tied = vec![-3.0f32; 64];
        let want: Vec<u32> = (0..u32::try_from(TOP).unwrap()).collect();
        assert_eq!(top(&tied).iter().map(|t| t.0).collect::<Vec<_>>(), want);
        // Two maxima at high ids, then 40 tokens tied at the cut.
        let mut lp = vec![-9.0f32; 100];
        lp[90] = -0.5;
        lp[70] = -0.5;
        for v in &mut lp[20..60] {
            *v = -2.0;
        }
        let got: Vec<u32> = top(&lp).iter().map(|t| t.0).collect();
        let mut want = vec![70, 90];
        want.extend(20..38);
        assert_eq!(got, want);
    }

    /// Integers compare exactly, past f64's 2^53 too; a whole float equals
    /// its integer.
    #[test]
    fn numbers_compare_exactly() {
        let n = |s: &str| serde_json::from_str::<Value>(s).unwrap();
        for (a, b, same) in [
            ("9007199254740993", "9007199254740992", false),
            ("9007199254740993", "9007199254740993", true),
            ("18446744073709551615", "18446744073709551614", false),
            ("-9007199254740993", "-9007199254740992", false),
            ("5", "5.0", true),
            ("9007199254740992", "9007199254740992.0", false),
            ("5", "5.5", false),
            ("0.5", "0.50", true),
            ("-0", "0", true),
        ] {
            assert_eq!(values_match(&n(a), &n(b)), same, "{a} vs {b}");
            assert_eq!(values_match(&n(b), &n(a)), same, "{b} vs {a}");
        }
    }

    #[test]
    fn arguments_match_exactly() {
        let e = json!({"name": "f", "arguments": {"city": "Paris", "opts": {"a": [1, 2]}}});
        let ok = json!({"city": "Paris", "opts": {"a": [1.0, 2]}});
        assert!(arguments_match(&e, &ok));
        for bad in [
            json!({"city": "Paris", "opts": {"a": [1, 2]}, "unit": "c"}),
            json!({"city": "Paris"}),
            json!({"city": "paris", "opts": {"a": [1, 2]}}),
            json!({"city": "Paris", "opts": {"a": [1, 2], "b": 0}}),
            json!({"city": "Paris", "opts": {"a": [1, 2, 3]}}),
            json!({"city": "Paris", "opts": {"a": ["1", 2]}}),
            json!(["Paris"]),
        ] {
            assert!(!arguments_match(&e, &bad), "{bad}");
        }
        let free =
            json!({"name": "web_search", "arguments": {"max_results": 3}, "free": ["query"]});
        assert!(arguments_match(
            &free,
            &json!({"query": "anything", "max_results": 3})
        ));
        assert!(!arguments_match(&free, &json!({"max_results": 3})));
        assert!(!arguments_match(
            &free,
            &json!({"query": "x", "max_results": 10})
        ));
        assert!(!arguments_match(
            &free,
            &json!({"query": "x", "max_results": 3, "lang": "en"})
        ));
    }

    fn write_full(path: &std::path::Path, rows: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for r in rows {
            for v in r {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    /// Positions pair by prompt id in the top lists and in the whole rows,
    /// whatever order each file holds the prompts in.
    #[test]
    fn rows_pair_by_id_in_any_order() {
        let dir = std::env::temp_dir().join(format!("eidola-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Prompt p0 has 2 positions, p1 has 3; every row a distinct
        // distribution over 24 ids.
        let row = |seed: usize| -> Vec<f32> {
            let logits: Vec<f32> = (0..24)
                .map(|i| ((i * 7 + seed * 5) % 11) as f32 * 0.3)
                .collect();
            log_softmax(&logits, 24)
        };
        let prompts = [
            ("p0", vec![row(1), row(2)]),
            ("p1", vec![row(3), row(4), row(5)]),
        ];
        let file = |order: &[usize], name: &str| {
            let tops: Vec<Value> = order
                .iter()
                .map(|&i| json!({"id": prompts[i].0, "top": prompts[i].1.iter().map(|r| top(r)).collect::<Vec<_>>()}))
                .collect();
            let rows: Vec<Vec<f32>> = order.iter().flat_map(|&i| prompts[i].1.clone()).collect();
            let path = dir.join(name);
            write_full(&path, &rows);
            (tops, path)
        };
        let (ta, fa) = file(&[0, 1], "a.f32");
        let (tb, fb) = file(&[1, 0], "b.f32");
        let m = compare_rows(
            &ta,
            &tb,
            Some((fa.to_str().unwrap(), fb.to_str().unwrap())),
            None,
        );
        assert_eq!((m.positions, m.top1, m.overlap), (5, 5, 5 * TOP));
        assert_eq!(m.kl, Some((0.0, 0.0)));
        assert_eq!((m.lower_bound_sum, m.lower_bound_max), (0.0, 0.0));
        let starts = HashMap::from([(json!("p0").to_string(), 1), (json!("p1").to_string(), 2)]);
        assert_eq!(compare_rows(&ta, &tb, None, Some(&starts)).positions, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[should_panic(expected = "positions in a")]
    fn position_counts_must_match() {
        let a = vec![json!({"id": "p", "top": [[[0, 0.0]], [[0, 0.0]]]})];
        let b = vec![json!({"id": "p", "top": [[[0, 0.0]]]})];
        compare_rows(&a, &b, None, None);
    }

    #[test]
    #[should_panic(expected = "start 2 is not inside its 2 positions")]
    fn region_starts_must_be_inside() {
        let r = json!({"id": "p", "top": [[[0, 0.0]], [[0, 0.0]]]});
        let starts = HashMap::from([(json!("p").to_string(), 2)]);
        compare_rows(
            std::slice::from_ref(&r),
            std::slice::from_ref(&r),
            None,
            Some(&starts),
        );
    }

    #[test]
    #[should_panic(expected = "names other prompts")]
    fn region_files_name_the_same_prompts() {
        let r = json!({"id": "p", "top": [[[0, 0.0]], [[0, 0.0]]]});
        let starts = HashMap::from([(json!("p").to_string(), 1), (json!("q").to_string(), 0)]);
        compare_rows(
            std::slice::from_ref(&r),
            std::slice::from_ref(&r),
            None,
            Some(&starts),
        );
    }

    #[test]
    #[should_panic(expected = "twice")]
    fn ids_must_be_unique() {
        let r = json!({"id": "p", "top": [[[0, 0.0]]]});
        compare_rows(&[r.clone(), r.clone()], &[r.clone(), r], None, None);
    }

    fn row(num_tokens: u32, num_drafts: u32, sample: bool) -> SeqEntry {
        SeqEntry {
            slot: 0,
            token_start: 0,
            num_tokens,
            context_len: 16,
            num_drafts,
            sample,
            sampling: SamplingParams::greedy(),
        }
    }

    /// Depth `d`'s denominator is the rows that drafted at least `d` tokens
    /// and had the first `d - 1` accepted: rows that drafted nothing, or were
    /// narrowed below `d`, are not rejections at `d`.
    #[test]
    fn acceptance_is_conditional_on_the_drafted_width() {
        let mut t = DraftTally::new(3);
        // Full width: all accepted, two accepted, none accepted.
        t.record(&row(1, 3, true), 4);
        t.record(&row(1, 3, true), 3);
        t.record(&row(1, 3, true), 1);
        // Narrowed to width 1 (accepted) and width 0, a prefill end, and a
        // prefill chunk that does not sample.
        t.record(&row(1, 1, true), 2);
        t.record(&row(1, 0, true), 1);
        t.record(&row(64, 0, true), 1);
        t.record(&row(64, 0, false), 0);
        assert_eq!(t.widths, vec![1, 1, 0, 3]);
        assert_eq!(t.prefill, 1);
        assert_eq!(t.reached, vec![4, 2, 2]);
        assert_eq!(t.accepted, vec![3, 2, 1]);
        assert_eq!((t.accepted_total, t.drafted_total), (6, 10));
        assert_eq!(t.rates(), vec![0.75, 1.0, 0.5]);
        // No row reached a depth: 0, not NaN.
        assert_eq!(DraftTally::new(2).rates(), vec![0.0, 0.0]);
    }
}
