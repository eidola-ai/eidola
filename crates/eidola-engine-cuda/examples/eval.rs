//! Evaluation driver for the CUDA executor on a MiMo-V2.6 checkpoint, built so
//! the same token ids can be fed to another engine (an SGLang oracle) and both
//! outputs scored by the same code.
//!
//! ```text
//! eval render   <model_dir> <tasks.jsonl> <prompts.jsonl>
//! eval generate <kernels_dir> <model_dir> <prompts.jsonl> <outputs.jsonl> <max_tokens>
//! eval logprobs <kernels_dir> <model_dir> <prompts.jsonl> <out.jsonl> [--reference] [--full <out.f32>]
//! eval score    <model_dir> <tasks.jsonl> <outputs.jsonl>
//! eval compare  <a.jsonl> <b.jsonl> [--full <a.f32> <b.f32>] [--from <prompts.jsonl>]
//! ```
//!
//! - `tasks.jsonl`: `{"id", "messages", "tools"?, "enable_thinking"?, "kind",
//!   "answer"? | "expect"?}` — `kind` is `"gsm8k"` (with a numeric `answer`) or
//!   `"tool"` (with `expect`: `{"name", "arguments"}`, or `null` for "no call").
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

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, SeqEntry, StepInput, TableUpdate};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_chat::json::Json;
use eidola_engine_chat::tool_call::parse_complete;
use eidola_engine_chat::{ChatInput, ChatTemplate, MimoTokenizer, RenderOptions, ToolSchemas};
use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, Gpu, KvBlocks};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{ForwardOptions, LoadOptions, LogitsAt, ModelWeights, ReferenceModel};
use serde_json::{Value, json};

const TOP: usize = 20;

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
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn executor(kernels: &str, model: &str, max_tokens: u32, max_seqs: u32) -> (CudaExecutor, u32) {
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let gpu = Gpu::open(0).unwrap();
    let dir = KernelDir::new(kernels);
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let cfg = CudaExecutorConfig {
        block_size: 16,
        num_blocks: KvBlocks {
            global: 16_384,
            sliding: 4_096,
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
        sampleable_vocab_size: tok.vocab_size() as u32,
        image: None,
    };
    let t0 = Instant::now();
    let ex = CudaExecutor::new(gpu, &dir, store, None, cfg).unwrap();
    eprintln!("executor loaded in {:.1?}", t0.elapsed());
    (ex, tok.vocab_size() as u32)
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

fn generate(kernels: &str, model: &str, prompts: &str, out: &str, max_new: u32) {
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let (ex, _) = executor(kernels, model, 2048, 64);
    let sched = SchedulerConfig {
        max_batched_tokens: 2048,
        max_seqs: 64,
        max_prefill_chunk: 2048,
        eos_token_ids: tok.eos_token_ids().to_vec(),
        speculative: false,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    };
    let mut eng = Engine::new(ex, sched).unwrap();
    let rows = read_jsonl(prompts);
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
    while eng.unfinished() > 0 {
        for e in eng.step(steps).unwrap() {
            produced += e.tokens.len();
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
    let out_rows: Vec<Value> = (0..rows.len() as u64)
        .map(|i| json!({"id": index[&i], "output_ids": outputs.get(&i).cloned().unwrap_or_default(), "finish": finish.get(&i)}))
        .collect();
    write_jsonl(out, &out_rows);
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
    row.iter().map(|&v| (v as f64 - lse) as f32).collect()
}

/// Top-`TOP` (id, log-probability) of one log-probability row.
fn top(lp: &[f32]) -> Vec<(u32, f64)> {
    let mut idx: Vec<u32> = (0..lp.len() as u32).collect();
    idx.select_nth_unstable_by(TOP, |&a, &b| lp[b as usize].total_cmp(&lp[a as usize]));
    idx.truncate(TOP);
    idx.sort_by(|&a, &b| lp[b as usize].total_cmp(&lp[a as usize]).then(a.cmp(&b)));
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
        let (mut ex, n) = executor(kernels, model, 2048, 16);
        ex.record_all_logits = true;
        for r in &rows {
            let p = ids(&r["prompt_ids"]);
            let len = p.len() as u32;
            let blocks = len.div_ceil(16);
            let updates: Vec<TableUpdate> = (0..2)
                .flat_map(|group| {
                    (0..blocks).map(move |i| TableUpdate {
                        slot: 0,
                        group,
                        index: i,
                        block: 1 + i,
                    })
                })
                .collect();
            let mut maintenance = vec![eidola_engine::executor::Maintenance::ResetSlot { slot: 0 }];
            for g in 0..2 {
                for b in 1..=blocks {
                    maintenance
                        .push(eidola_engine::executor::Maintenance::Zero { group: g, block: b });
                }
            }
            let step = StepInput {
                bucket: Bucket {
                    max_seqs: 4,
                    max_tokens: 2048,
                },
                maintenance,
                table_updates: updates,
                seqs: vec![SeqEntry {
                    slot: 0,
                    token_start: 0,
                    num_tokens: len,
                    context_len: 0,
                    num_drafts: 0,
                    sample: true,
                    sampling: SamplingParams::greedy(),
                }],
                token_ids: p.clone(),
                positions: (0..len).collect(),
                return_logits: false,
            };
            ex.execute(&step).unwrap();
            let all = ex.take_all_logits().unwrap();
            let tops: Vec<_> = (0..p.len())
                .map(|i| sink.row(&all[i * vocab..(i + 1) * vocab], n as usize))
                .collect();
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

/// Reads whole log-probability rows of `width` from a `--full` file.
struct FullRows {
    file: std::io::BufReader<std::fs::File>,
    width: usize,
}

impl FullRows {
    fn open(path: &str, rows: usize) -> FullRows {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let bytes = file.metadata().unwrap().len() as usize;
        assert!(
            rows > 0 && bytes.is_multiple_of(rows * 4),
            "{path}: {bytes} bytes is not {rows} whole rows"
        );
        FullRows {
            file: std::io::BufReader::new(file),
            width: bytes / (rows * 4),
        }
    }

    fn next(&mut self) -> Vec<f64> {
        use std::io::Read;
        let mut buf = vec![0u8; self.width * 4];
        self.file.read_exact(&mut buf).unwrap();
        buf.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c) as f64)
            .collect()
    }
}

fn compare(a: &str, b: &str, full: Option<(&str, &str)>, from: Option<&str>) {
    let (a, b) = (read_jsonl(a), read_jsonl(b));
    let key = |r: &Value| r["id"].to_string();
    let by_id: HashMap<String, &Value> = b.iter().map(|r| (key(r), r)).collect();
    assert_eq!(a.len(), b.len(), "the files hold different prompt sets");
    let starts: HashMap<String, usize> = from.map_or_else(HashMap::new, |p| {
        read_jsonl(p)
            .iter()
            .map(|r| (key(r), r["start"].as_u64().unwrap() as usize))
            .collect()
    });
    // Every id in both files, with the same number of positions: a
    // shortened side would otherwise drop out of every metric unseen.
    let mut lens = Vec::with_capacity(a.len());
    for ra in &a {
        let rb = by_id
            .get(&key(ra))
            .unwrap_or_else(|| panic!("id {} is missing from b", key(ra)));
        let (na, nb) = (
            ra["top"].as_array().unwrap().len(),
            rb["top"].as_array().unwrap().len(),
        );
        assert_eq!(na, nb, "id {}: {na} positions in a, {nb} in b", key(ra));
        lens.push(na);
    }
    let total: usize = lens.iter().sum();
    let mut full = full.map(|(fa, fb)| {
        let (fa, fb) = (FullRows::open(fa, total), FullRows::open(fb, total));
        assert_eq!(fa.width, fb.width, "full rows of different widths");
        (fa, fb)
    });
    let (mut n, mut top1, mut overlap) = (0usize, 0usize, 0usize);
    let (mut lb_sum, mut lb_max, mut kl_sum, mut kl_max) = (0f64, 0f64, 0f64, 0f64);
    for ra in &a {
        let rb = by_id[&key(ra)];
        let start = if from.is_some() {
            *starts
                .get(&key(ra))
                .unwrap_or_else(|| panic!("id {} has no start", key(ra)))
        } else {
            0
        };
        for (pos, (pa, pb)) in ra["top"]
            .as_array()
            .unwrap()
            .iter()
            .zip(rb["top"].as_array().unwrap())
            .enumerate()
        {
            let rows = full.as_mut().map(|(fa, fb)| (fa.next(), fb.next()));
            if pos < start {
                continue;
            }
            let (ta, tb) = (top_list(pa), top_list(pb));
            n += 1;
            top1 += (ta[0].0 == tb[0].0) as usize;
            overlap += ta
                .iter()
                .filter(|(t, _)| tb.iter().any(|(u, _)| u == t))
                .count();
            let lb = kl_lower_bound(&ta, &tb);
            lb_sum += lb;
            lb_max = lb_max.max(lb);
            if let Some((la, lb)) = rows {
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
    let nf = n as f64;
    print!(
        "positions {n}: top-1 {top1}/{n} ({:.2}%), top-{TOP} overlap {:.2}%, KL lower bound (shared top-{TOP} + remainder) mean {:.3e} max {:.3e}",
        100.0 * top1 as f64 / nf,
        100.0 * overlap as f64 / (nf * TOP as f64),
        lb_sum / nf,
        lb_max
    );
    if full.is_some() {
        print!(", KL mean {:.3e} max {:.3e}", kl_sum / nf, kl_max);
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

fn values_match(want: &Value, got: &Value) -> bool {
    match (want, got) {
        (Value::String(a), Value::String(b)) => a.trim().eq_ignore_ascii_case(b.trim()),
        (Value::Number(a), Value::Number(b)) => {
            (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-9
        }
        (Value::Number(a), Value::String(b)) | (Value::String(b), Value::Number(a)) => {
            number(b).is_some_and(|x| (x - a.as_f64().unwrap()).abs() < 1e-9)
        }
        (Value::Object(a), Value::Object(b)) => a
            .iter()
            .all(|(k, v)| b.get(k).is_some_and(|g| values_match(v, g))),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_match(x, y))
        }
        (a, b) => a == b,
    }
}

fn score(model: &str, tasks: &str, outputs: &str) {
    let tok = MimoTokenizer::from_model_dir(Path::new(model)).unwrap();
    let outs: HashMap<String, Value> = read_jsonl(outputs)
        .into_iter()
        .map(|r| (r["id"].to_string(), r))
        .collect();
    let (mut gsm, mut gsm_ok, mut tool, mut tool_ok, mut truncated) = (0, 0, 0, 0, 0);
    for t in read_jsonl(tasks) {
        let o = &outs[&t["id"].to_string()];
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
                                .is_ok_and(|a| values_match(&e["arguments"], &a))),
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
        Some("generate") => generate(&a[2], &a[3], &a[4], &a[5], a[6].parse().unwrap()),
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
