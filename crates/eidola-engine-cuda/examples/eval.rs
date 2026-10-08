//! Evaluation driver for the CUDA executor on a MiMo-V2.6 checkpoint, built so
//! the same token ids can be fed to another engine (an SGLang oracle) and both
//! outputs scored by the same code.
//!
//! ```text
//! eval render   <model_dir> <tasks.jsonl> <prompts.jsonl>
//! eval generate <kernels_dir> <model_dir> <prompts.jsonl> <outputs.jsonl> <max_tokens>
//! eval logprobs <kernels_dir> <model_dir> <prompts.jsonl> <out.jsonl> [--reference]
//! eval score    <model_dir> <tasks.jsonl> <outputs.jsonl>
//! eval compare  <a.jsonl> <b.jsonl>
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
//!   executor or (`--reference`) the f32 reference forward.

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
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, Gpu};
use eidola_engine_kernels::{ArtifactDir, Manifest};
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
    let dir = ArtifactDir::new(kernels, Manifest::embedded());
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let cfg = CudaExecutorConfig {
        block_size: 16,
        num_blocks: vec![16_384, 4_096],
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

/// Top-`TOP` (id, log-probability) of one logit row over `n` ids.
fn top(row: &[f32], n: usize) -> Vec<(u32, f64)> {
    let row = &row[..n];
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let lse = max
        + row
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.select_nth_unstable_by(TOP, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    idx.truncate(TOP);
    idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
    idx.iter()
        .map(|&i| (i, row[i as usize] as f64 - lse))
        .collect()
}

fn logprobs(kernels: &str, model: &str, prompts: &str, out: &str, reference: bool) {
    let rows = read_jsonl(prompts);
    let vocab = 152_576usize;
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
                .map(|i| top(f.logits.row(i), tok.vocab_size()))
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
                .map(|i| top(&all[i * vocab..(i + 1) * vocab], n as usize))
                .collect();
            result.push(json!({"id": r["id"], "top": tops}));
        }
    }
    write_jsonl(out, &result);
}

fn compare(a: &str, b: &str) {
    let (a, b) = (read_jsonl(a), read_jsonl(b));
    let by_id: HashMap<String, &Value> = b.iter().map(|r| (r["id"].to_string(), r)).collect();
    let (mut n, mut top1, mut kl_sum, mut kl_max) = (0usize, 0usize, 0f64, 0f64);
    for ra in &a {
        let rb = by_id[&ra["id"].to_string()];
        for (pa, pb) in ra["top"]
            .as_array()
            .unwrap()
            .iter()
            .zip(rb["top"].as_array().unwrap())
        {
            let parse = |v: &Value| -> Vec<(u64, f64)> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|e| (e[0].as_u64().unwrap(), e[1].as_f64().unwrap()))
                    .collect()
            };
            let (ta, tb) = (parse(pa), parse(pb));
            n += 1;
            top1 += (ta[0].0 == tb[0].0) as usize;
            // KL(a || b) over a's top tokens; a token missing from b's list is
            // charged b's smallest listed log-probability (an upper bound on
            // its true value, so this underestimates nothing it can see).
            let floor = tb.last().unwrap().1;
            let lb: HashMap<u64, f64> = tb.iter().copied().collect();
            let kl: f64 = ta
                .iter()
                .map(|&(t, la)| la.exp() * (la - lb.get(&t).copied().unwrap_or(floor)))
                .sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
        }
    }
    println!(
        "positions {n}: top-1 {top1}/{n} ({:.2}%), top-{TOP} KL mean {:.3e} max {:.3e}",
        100.0 * top1 as f64 / n as f64,
        kl_sum / n as f64,
        kl_max
    );
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
                    e => parsed.calls.first().is_some_and(|(name, args)| {
                        name == e["name"].as_str().unwrap()
                            && serde_json::from_str::<Value>(args)
                                .is_ok_and(|a| values_match(&e["arguments"], &a))
                    }),
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
            a.get(6).map(String::as_str) == Some("--reference"),
        ),
        Some("score") => score(&a[2], &a[3], &a[4]),
        Some("compare") => compare(&a[2], &a[3]),
        _ => eprintln!("usage: see the module docs"),
    }
}
