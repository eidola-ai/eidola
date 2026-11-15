//! The DeepSeek-V4.1 generation runner (`main.rs` is its command line; doctrine in this
//! crate's `AGENTS.md` → Generating from the released checkpoint on the CPU).
//!
//! It drives the node's own path without its HTTP side: `LoadedModel::load` (the weights
//! hash checked, the reference model and the Engram map built as the node builds them),
//! `api::parse_request` and `pipeline::prepare` (the format's encoder and tokenizer), the
//! node's executor and scheduler configuration (`deepseek_cpu_config`,
//! `scheduler_config`) over `DsCpuExecutor` and the engine core, and `pipeline::Decoder`
//! (detokenizer, reasoning and tool-call parsers). The one difference from a node is the
//! executor's row records, kept to count draft acceptance per slot; they change no
//! computed value.
//!
//! Every request is checkpointed when it finishes, so a rerun over the same output
//! directory skips the finished ones.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::Engine;
use eidola_engine::secret::SaltDeriver;
use eidola_engine_chat::ChatDelta;
use eidola_engine_cpu::deepseek::DsCpuExecutor;
use eidola_server_engine::config::{CacheConfig, ExecutorKind, Sizing, WeightsStorage};
use eidola_server_engine::model::{LoadedModel, Served};
use eidola_server_engine::pipeline::{self, Decoder, Prepared};
use eidola_server_engine::{api, deepseek_cpu_config, scheduler_config};
use serde_json::{Value, json};

/// The model id the runner's requests name (the node checks it; nothing else reads it).
const MODEL_ID: &str = "deepseek-v4.1-flash";

/// What a run is given.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub weights_dir: PathBuf,
    /// The expected weights hash (lowercase hex), checked by the node's loader.
    pub weights_sha256: String,
    pub out_dir: PathBuf,
    pub prompts: Vec<PromptSpec>,
    /// New tokens per request, at most.
    pub max_new_tokens: u32,
    /// The draft width of the main pass; `None` is the drafter's slot count.
    pub draft_width: Option<u32>,
    /// A run whose estimated time exceeds this is refused.
    pub time_limit_secs: f64,
    /// Positions per prefill step.
    pub prefill_chunk: u32,
    pub block_size: u32,
    /// On Linux, refuse to load unless `MemAvailable` is at least this.
    pub min_available_bytes: u64,
}

/// One chat request: OpenAI messages, optionally one or more tools.
#[derive(Clone, Debug)]
pub struct PromptSpec {
    pub name: String,
    pub messages: Value,
    pub tools: Option<Value>,
    /// Also run at draft width 0 and compare the greedy tokens.
    pub check_width0: bool,
}

impl PromptSpec {
    /// A prompts file: a JSON array of `{"name", "messages", "tools"?, "check_width0"?}`.
    pub fn parse_file(bytes: &[u8]) -> Result<Vec<PromptSpec>, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("prompts: {e}"))?;
        let items = v.as_array().ok_or("prompts: not a JSON array")?;
        let mut out = Vec::new();
        for item in items {
            let name = item["name"]
                .as_str()
                .ok_or("prompts: an entry has no name")?;
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            {
                return Err(format!("prompts: name {name:?} is not [A-Za-z0-9_-]+"));
            }
            if out.iter().any(|p: &PromptSpec| p.name == name) {
                return Err(format!("prompts: {name} appears twice"));
            }
            if let Some(key) = item.as_object().and_then(|o| {
                o.keys()
                    .find(|k| !["name", "messages", "tools", "check_width0"].contains(&k.as_str()))
            }) {
                return Err(format!("prompts: {name}: unknown key {key}"));
            }
            out.push(PromptSpec {
                name: name.to_string(),
                messages: item
                    .get("messages")
                    .filter(|m| m.is_array())
                    .cloned()
                    .ok_or_else(|| format!("prompts: {name} has no messages array"))?,
                tools: item.get("tools").cloned(),
                check_width0: item["check_width0"].as_bool().unwrap_or(false),
            });
        }
        if out.is_empty() {
            return Err("prompts: no entries".into());
        }
        Ok(out)
    }

    /// The request body the node would be sent: greedy, a stream with usage, as the
    /// gateway forwards one.
    fn body(&self, max_new_tokens: u32) -> Value {
        let mut body = json!({
            "model": MODEL_ID,
            "messages": self.messages,
            "max_completion_tokens": max_new_tokens,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if let Some(tools) = &self.tools {
            body["tools"] = tools.clone();
        }
        body
    }
}

/// How a run ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every request has a result; `mismatches` width-0 checks differed.
    Complete { mismatches: usize },
    /// Refused before (or while calibrating) because the estimate exceeds the limit.
    Refused(String),
}

/// A UTC timestamp, `YYYY-MM-DDTHH:MM:SSZ`.
pub fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

pub fn log(msg: impl std::fmt::Display) {
    eprintln!("{} {msg}", timestamp());
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn hms(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    format!("{}h{:02}m{:02}s", s / 3600, s % 3600 / 60, s % 60)
}

/// The process's peak resident set, in bytes.
fn peak_rss() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `getrusage` writes the struct it is given and nothing else.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    let max = usage.ru_maxrss.max(0) as u64;
    // Linux reports kilobytes; macOS, bytes.
    if cfg!(target_os = "macos") {
        max
    } else {
        max * 1024
    }
}

/// A `kB` field of a `/proc` file, in bytes (Linux only).
fn proc_kb(file: &str, key: &str) -> Option<u64> {
    let text = std::fs::read_to_string(file).ok()?;
    text.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.strip_prefix(':')?;
        Some(
            rest.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()?
                * 1024,
        )
    })
}

/// The process's memory: peak resident, and on Linux the current resident set split
/// into anonymous memory (weights decoded at load, KV, activations) and file-backed
/// pages (the mapped checkpoint, reclaimable page cache shared with the kernel).
fn memory() -> Value {
    json!({
        "peak_rss_bytes": peak_rss(),
        "rss_bytes": proc_kb("/proc/self/status", "VmRSS"),
        "rss_anon_bytes": proc_kb("/proc/self/status", "RssAnon"),
        "rss_file_bytes": proc_kb("/proc/self/status", "RssFile"),
        "mem_available_bytes": proc_kb("/proc/meminfo", "MemAvailable"),
    })
}

fn memory_line() -> String {
    let anon = proc_kb("/proc/self/status", "RssAnon");
    let file = proc_kb("/proc/self/status", "RssFile");
    match (anon, file) {
        (Some(a), Some(f)) => format!(
            "peak rss {}, anon {}, file-backed {}",
            gb(peak_rss()),
            gb(a),
            gb(f)
        ),
        _ => format!("peak rss {}", gb(peak_rss())),
    }
}

/// Writes `bytes` to `path` through a temporary file and a rename, so a killed run never
/// leaves a partial checkpoint.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("partial");
    std::fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))
}

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    write_atomic(
        path,
        serde_json::to_string_pretty(v).expect("JSON").as_bytes(),
    )
}

/// Measured costs, kept across reruns in `calibration.json`.
#[derive(Clone, Debug, Default)]
struct Calibration {
    /// Seconds per prompt position of a prefill step.
    prefill_secs_per_position: Option<f64>,
    /// Seconds per decode step, by draft width.
    step_secs: BTreeMap<u32, f64>,
}

impl Calibration {
    fn load(path: &Path) -> Calibration {
        let Ok(bytes) = std::fs::read(path) else {
            return Calibration::default();
        };
        let Ok(v) = serde_json::from_slice::<Value>(&bytes) else {
            return Calibration::default();
        };
        Calibration {
            prefill_secs_per_position: v["prefill_secs_per_position"].as_f64(),
            step_secs: v["step_secs_by_width"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, v)| Some((k.parse().ok()?, v.as_f64()?)))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "prefill_secs_per_position": self.prefill_secs_per_position,
            "step_secs_by_width": self
                .step_secs
                .iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect::<serde_json::Map<_, _>>(),
        })
    }

    fn calibrated(&self) -> bool {
        self.prefill_secs_per_position.is_some() && !self.step_secs.is_empty()
    }

    /// A decode step's cost at `width`: measured, else the most expensive width measured
    /// (a drafting step computes more positions than a plain one, so this bounds a
    /// width not yet measured from above only when the measured width is the wider).
    fn step(&self, width: u32) -> Option<f64> {
        self.step_secs
            .get(&width)
            .copied()
            .or_else(|| self.step_secs.values().copied().reduce(f64::max))
    }

    /// The worst-case time of `tasks`: every prompt position at the calibrated prefill
    /// rate, and every new token its own decode step (no draft accepted).
    fn estimate(&self, tasks: &[&Task], max_new: u32) -> Option<f64> {
        let spp = self.prefill_secs_per_position?;
        tasks.iter().try_fold(0.0, |acc, t| {
            Some(acc + t.prompt_tokens as f64 * spp + max_new as f64 * self.step(t.width)?)
        })
    }
}

/// One request at one draft width.
struct Task {
    spec: PromptSpec,
    width: u32,
    prompt_tokens: u32,
}

impl Task {
    fn key(&self) -> String {
        format!("{}.w{}", self.spec.name, self.width)
    }
}

/// The task's request, prepared by the node's pipeline under `max_model_len`.
fn prepare(
    model: &LoadedModel,
    salts: &SaltDeriver,
    max_model_len: u32,
    id: u64,
    spec: &PromptSpec,
    max_new: u32,
) -> Result<Prepared, String> {
    let body = serde_json::to_vec(&spec.body(max_new)).expect("JSON");
    let refused = |e: eidola_server_engine::error::ApiError| {
        format!("{}: refused: {}", spec.name, e.to_body())
    };
    let valid = api::parse_request(&body, MODEL_ID).map_err(refused)?;
    pipeline::prepare(model, salts, max_model_len, id, valid).map_err(refused)
}

/// Draft acceptance by slot: slot `i` is proposed whenever a row drafts more than `i`
/// tokens, reached when every earlier slot of the row was accepted, and accepted when
/// it was too.
#[derive(Clone, Debug, Default)]
struct SlotCounts {
    proposed: Vec<u64>,
    reached: Vec<u64>,
    accepted: Vec<u64>,
}

impl SlotCounts {
    fn add_row(&mut self, drafts: usize, accepted: usize) {
        if self.proposed.len() < drafts {
            self.proposed.resize(drafts, 0);
            self.reached.resize(drafts, 0);
            self.accepted.resize(drafts, 0);
        }
        for i in 0..drafts {
            self.proposed[i] += 1;
            if accepted >= i {
                self.reached[i] += 1;
            }
            if accepted > i {
                self.accepted[i] += 1;
            }
        }
    }

    fn merge(&mut self, other: &SlotCounts) {
        for _ in self.proposed.len()..other.proposed.len() {
            self.proposed.push(0);
            self.reached.push(0);
            self.accepted.push(0);
        }
        for i in 0..other.proposed.len() {
            self.proposed[i] += other.proposed[i];
            self.reached[i] += other.reached[i];
            self.accepted[i] += other.accepted[i];
        }
    }

    fn from_json(v: &Value) -> SlotCounts {
        let list = |k: &str| -> Vec<u64> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default()
        };
        SlotCounts {
            proposed: list("proposed"),
            reached: list("reached"),
            accepted: list("accepted"),
        }
    }

    fn to_json(&self) -> Value {
        let ratio = |a: &[u64], b: &[u64]| -> Vec<Option<f64>> {
            a.iter()
                .zip(b)
                .map(|(a, b)| (*b > 0).then(|| *a as f64 / *b as f64))
                .collect()
        };
        json!({
            "proposed": self.proposed,
            "reached": self.reached,
            "accepted": self.accepted,
            "conditional_acceptance": ratio(&self.accepted, &self.reached),
            "unconditional_acceptance": ratio(&self.accepted, &self.proposed),
            "total_proposed": self.proposed.iter().sum::<u64>(),
            "total_accepted": self.accepted.iter().sum::<u64>(),
        })
    }
}

/// Calls back after the first decode step, with the measured prefill seconds per
/// position and the decode step's seconds; an error ends the request.
type OnCalibrated<'a> = &'a mut dyn FnMut(f64, f64) -> Result<(), String>;

/// Runs one prepared request alone through `engine` and the node's decoder.
fn run_task(
    engine: &mut Engine<DsCpuExecutor>,
    model: &Arc<LoadedModel>,
    task: &Task,
    prepared: Prepared,
    clock: &Instant,
    mut on_calibrated: Option<OnCalibrated<'_>>,
) -> Result<Value, String> {
    let key = task.key();
    let started = Instant::now();
    let stats_before = engine.stats();
    let Prepared {
        request,
        output,
        stop,
        window_bound,
        ..
    } = prepared;
    let mut decoder =
        Decoder::new(model.clone(), output, stop, "call_".into()).window_bound(window_bound);
    let prompt = request.prompt.clone();
    let max_tokens = request.max_tokens;
    engine
        .submit(request)
        .map_err(|e| format!("{key}: the engine refused the request: {e:?}"))?;

    let mut out = ChatDelta::default();
    let mut tokens: Vec<u32> = Vec::new();
    let mut slots = SlotCounts::default();
    let mut step_log = Vec::new();
    let (mut prefill_steps, mut prefill_secs, mut prefill_positions) = (0u64, 0.0f64, 0u64);
    let (mut decode_steps, mut decode_secs, mut decode_tokens) = (0u64, 0.0f64, 0u64);
    let mut idle_steps = 0u64;
    let (mut engine_finish, mut chat_finish) = (None, None);
    while engine_finish.is_none() {
        let t = Instant::now();
        let events = engine
            .step(clock.elapsed().as_millis() as u64)
            .map_err(|e| format!("{key}: the executor failed: {}", e.0))?;
        let secs = t.elapsed().as_secs_f64();
        let records = engine.executor().take_records();
        // Kept only for record mode's cross-device checks, which nothing here reads.
        drop(engine.executor().transfers());
        let positions: u64 = records
            .iter()
            .map(|r| u64::from(r.num_tokens) + r.drafts.len() as u64)
            .sum();
        let (mut drafted, mut accepted) = (0usize, 0usize);
        for r in records.iter().filter(|r| !r.drafts.is_empty()) {
            let a = r.produced.len().saturating_sub(1);
            slots.add_row(r.drafts.len(), a);
            drafted += r.drafts.len();
            accepted += a;
        }
        let first_output = tokens.is_empty();
        let mut produced = 0usize;
        for ev in events {
            produced += ev.tokens.len();
            tokens.extend(&ev.tokens);
            let step = decoder
                .push(&ev.tokens, ev.finish)
                .map_err(|e| format!("{key}: the decoder refused: {}", e.to_body()))?;
            out.reasoning_content
                .push_str(&step.delta.reasoning_content);
            out.content.push_str(&step.delta.content);
            out.tool_calls.extend(step.delta.tool_calls);
            if step.finish.is_some() {
                chat_finish = step.finish;
            }
            if ev.finish.is_some() {
                engine_finish = ev.finish;
            }
        }
        if records.is_empty() {
            idle_steps += 1;
            continue;
        }
        let phase = if first_output { "prefill" } else { "decode" };
        if first_output {
            prefill_steps += 1;
            prefill_secs += secs;
            prefill_positions += positions;
        } else {
            decode_steps += 1;
            decode_secs += secs;
            decode_tokens += produced as u64;
        }
        log(format_args!(
            "{key}: {phase} step {}: {positions} positions, {produced} tokens{}, {secs:.1} s; \
             {} of {max_tokens} tokens; {}",
            prefill_steps + decode_steps,
            if drafted > 0 {
                format!(" ({accepted} of {drafted} drafts accepted)")
            } else {
                String::new()
            },
            tokens.len(),
            memory_line()
        ));
        step_log.push(json!({
            "phase": phase,
            "secs": secs,
            "positions": positions,
            "tokens": produced,
            "drafts": drafted,
            "accepted": accepted,
        }));
        if !first_output
            && decode_steps == 1
            && prefill_positions > 0
            && let Some(f) = on_calibrated.as_mut()
        {
            f(prefill_secs / prefill_positions as f64, secs)?;
        }
    }
    // As the node's engine thread does when its last request ends.
    engine
        .flush()
        .map_err(|e| format!("{key}: the executor failed: {}", e.0))?;
    let stats = engine.stats();
    let tokenizer = model.tokenizer();
    let prompt_text = tokenizer
        .decode(&prompt, false)
        .map_err(|e| format!("{key}: {e}"))?;
    Ok(json!({
        "name": task.spec.name,
        "draft_width": task.width,
        "prompt_tokens": prompt.len(),
        "max_tokens": max_tokens,
        "generated_tokens": tokens.len(),
        "engine_finish": format!("{:?}", engine_finish.expect("finished")),
        "finish_reason": chat_finish.map(|f| format!("{f:?}")),
        "reasoning_content": out.reasoning_content,
        "content": out.content,
        "tool_calls": out.tool_calls.iter().map(|c| json!({
            "id": c.id, "name": c.name, "arguments": c.arguments,
        })).collect::<Vec<_>>(),
        "token_ids": tokens,
        "prompt_text": prompt_text,
        "messages": task.spec.messages,
        "tools": task.spec.tools,
        "wall_secs": started.elapsed().as_secs_f64(),
        "prefill": {
            "steps": prefill_steps,
            "secs": prefill_secs,
            "positions": prefill_positions,
            "secs_per_position": (prefill_positions > 0)
                .then(|| prefill_secs / prefill_positions as f64),
        },
        "decode": {
            "steps": decode_steps,
            "secs": decode_secs,
            "tokens": decode_tokens,
            "tokens_per_step": (decode_steps > 0).then(|| decode_tokens as f64 / decode_steps as f64),
            "secs_per_step": (decode_steps > 0).then(|| decode_secs / decode_steps as f64),
        },
        "maintenance_only_steps": idle_steps,
        "drafts_by_slot": slots.to_json(),
        "engine_stats": {
            "steps": stats.steps - stats_before.steps,
            "drafted": stats.drafted - stats_before.drafted,
            "accepted": stats.accepted - stats_before.accepted,
            "preemptions": stats.preemptions - stats_before.preemptions,
        },
        "step_log": step_log,
        "memory": memory(),
    }))
}

fn transcript(r: &Value) -> String {
    let mut t = format!(
        "== {} (draft width {}): {} prompt tokens, {} generated, {} / {}\n\n== prompt\n{}\n\n",
        r["name"].as_str().unwrap_or(""),
        r["draft_width"],
        r["prompt_tokens"],
        r["generated_tokens"],
        r["engine_finish"].as_str().unwrap_or(""),
        r["finish_reason"].as_str().unwrap_or(""),
        r["prompt_text"].as_str().unwrap_or(""),
    );
    t.push_str(&format!(
        "== reasoning\n{}\n\n== content\n{}\n",
        r["reasoning_content"].as_str().unwrap_or(""),
        r["content"].as_str().unwrap_or("")
    ));
    for c in r["tool_calls"].as_array().into_iter().flatten() {
        t.push_str(&format!(
            "\n== tool call {}\n{}\n",
            c["name"].as_str().unwrap_or(""),
            c["arguments"].as_str().unwrap_or("")
        ));
    }
    t
}

/// The engine sizing for every task: one sequence at a time, its pools sized from the
/// longest prompt rather than the model's positions.
struct Plan {
    sizing: Sizing,
    kv_blocks: u32,
}

fn plan(cfg: &RunConfig, longest_prompt: u32, width: u32, max_positions: u32) -> Plan {
    let bs = cfg.block_size;
    let reach = longest_prompt + cfg.max_new_tokens + width + 1;
    let max_model_len = (reach + bs)
        .div_ceil(bs)
        .saturating_mul(bs)
        .min(max_positions);
    let step = cfg.prefill_chunk.max(1 + width);
    Plan {
        sizing: Sizing {
            kv_block_size: bs,
            max_model_len,
            max_seqs: 1,
            max_batched_tokens: step,
            max_prefill_chunk: step,
            draft_tokens: width,
            max_requests: 1,
        },
        // Every group can hold the whole sequence twice over, with the null block.
        kv_blocks: 2 * max_model_len.div_ceil(bs) + 32,
    }
}

fn build_engine(
    model: &Arc<LoadedModel>,
    plan: &Plan,
    width: u32,
) -> Result<Engine<DsCpuExecutor>, String> {
    let Served::DeepSeek {
        config,
        model: Some(reference),
    } = model.served()
    else {
        return Err("the weights are not DeepSeek-V4.1 loaded for the cpu executor".into());
    };
    let sizing = Sizing {
        draft_tokens: width,
        ..plan.sizing
    };
    let mut exec = deepseek_cpu_config(
        config,
        &sizing,
        plan.kv_blocks,
        0,
        model.sampleable_vocab_size(),
    )
    .map_err(|e| e.0)?;
    exec.record = true;
    let cache = CacheConfig {
        enabled: true,
        idle_ttl_secs: 900,
        max_age_secs: 7200,
    };
    let t = Instant::now();
    let executor = DsCpuExecutor::new(reference.clone(), exec);
    log(format_args!(
        "built the executor (draft width {width}, {} blocks of {} positions per group) in {:.1} s; {}",
        plan.kv_blocks,
        plan.sizing.kv_block_size,
        t.elapsed().as_secs_f64(),
        memory_line()
    ));
    Engine::new(executor, scheduler_config(&sizing, &cache, model))
        .map_err(|e| format!("the engine refused its configuration: {e:?}"))
}

/// Runs every request not already finished under `cfg.out_dir`, then writes the summary.
pub fn run(cfg: &RunConfig) -> Result<Outcome, String> {
    let clock = Instant::now();
    let requests_dir = cfg.out_dir.join("requests");
    let transcripts_dir = cfg.out_dir.join("transcripts");
    for d in [&cfg.out_dir, &requests_dir, &transcripts_dir] {
        std::fs::create_dir_all(d).map_err(|e| format!("create {}: {e}", d.display()))?;
    }
    if let Some(avail) = proc_kb("/proc/meminfo", "MemAvailable")
        && avail < cfg.min_available_bytes
    {
        return Err(format!(
            "only {} of memory is available; the run needs at least {} (set by \
             EIDOLA_DSGEN_MIN_AVAILABLE_GB)",
            gb(avail),
            gb(cfg.min_available_bytes)
        ));
    }

    log(format_args!(
        "loading {} through the node's loader: every covered file is hashed and the \
         weights hash checked first",
        cfg.weights_dir.display()
    ));
    let t = Instant::now();
    let model = Arc::new(
        LoadedModel::load(
            &cfg.weights_dir,
            &cfg.weights_sha256,
            WeightsStorage::DevWritable,
            ExecutorKind::Cpu,
        )
        .map_err(|e| e.to_string())?,
    );
    log(format_args!(
        "loaded and verified (weights hash {}) in {:.1} s; {}",
        model.weights_hash(),
        t.elapsed().as_secs_f64(),
        memory_line()
    ));
    let Served::DeepSeek { config, .. } = model.served() else {
        return Err("the weights are not DeepSeek-V4.1".into());
    };
    let slots = config.drafter.as_ref().map_or(0, |d| d.slots as u32);
    let width = cfg.draft_width.unwrap_or(slots);

    // Every request rendered and tokenized once, for its length.
    let salts = SaltDeriver::new();
    let mut tasks = Vec::new();
    for spec in &cfg.prompts {
        let p = prepare(
            &model,
            &salts,
            model.max_positions(),
            0,
            spec,
            cfg.max_new_tokens,
        )?;
        tasks.push(Task {
            spec: spec.clone(),
            width,
            prompt_tokens: p.prompt_tokens,
        });
    }
    for spec in cfg.prompts.iter().filter(|p| p.check_width0 && width > 0) {
        let prompt_tokens = tasks
            .iter()
            .find(|t| t.spec.name == spec.name)
            .expect("listed")
            .prompt_tokens;
        tasks.push(Task {
            spec: spec.clone(),
            width: 0,
            prompt_tokens,
        });
    }
    let longest = tasks.iter().map(|t| t.prompt_tokens).max().unwrap_or(1);
    let plan = plan(cfg, longest, width, model.max_positions());
    log(format_args!(
        "{} requests ({} at draft width {width}, {} at width 0); prompts of {} tokens; \
         context {} positions, prefill steps of {}",
        tasks.len(),
        cfg.prompts.len(),
        tasks.len() - cfg.prompts.len(),
        tasks
            .iter()
            .filter(|t| t.width == width)
            .map(|t| t.prompt_tokens.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        plan.sizing.max_model_len,
        plan.sizing.max_batched_tokens
    ));

    let done = |t: &Task| requests_dir.join(format!("{}.json", t.key())).exists();
    let cal_path = cfg.out_dir.join("calibration.json");
    let mut cal = Calibration::load(&cal_path);
    let pending: Vec<&Task> = tasks.iter().filter(|t| !done(t)).collect();
    for t in tasks.iter().filter(|t| done(t)) {
        log(format_args!(
            "{}: finished in an earlier run; skipped",
            t.key()
        ));
    }
    if !pending.is_empty() {
        match cal.estimate(&pending, cfg.max_new_tokens) {
            Some(est) if est > cfg.time_limit_secs => {
                return Ok(Outcome::Refused(format!(
                    "estimated {} for {} requests (calibrated earlier, worst case: no draft \
                     accepted), over the limit of {}",
                    hms(est),
                    pending.len(),
                    hms(cfg.time_limit_secs)
                )));
            }
            Some(est) => log(format_args!(
                "estimated at most {} for {} requests (calibration from an earlier run; \
                 limit {})",
                hms(est),
                pending.len(),
                hms(cfg.time_limit_secs)
            )),
            None => {
                let positions: u64 = pending.iter().map(|t| u64::from(t.prompt_tokens)).sum();
                log(format_args!(
                    "not calibrated yet: the first request's prefill and first decode step \
                     calibrate, and the run is refused there if its estimate exceeds {}. \
                     Prior, at the model crate's measured 1.3 s per position: {} prompt \
                     positions take about {} to prefill",
                    hms(cfg.time_limit_secs),
                    positions,
                    hms(positions as f64 * 1.3)
                ));
            }
        }
    }

    let mut engine: Option<(u32, Engine<DsCpuExecutor>)> = None;
    let mut next_id = 1u64;
    for (i, task) in tasks.iter().enumerate() {
        if done(task) {
            continue;
        }
        if engine.as_ref().is_none_or(|(w, _)| *w != task.width) {
            // One executor at a time: the previous one's pools are freed first.
            drop(engine.take());
            engine = Some((task.width, build_engine(&model, &plan, task.width)?));
        }
        let (_, eng) = engine.as_mut().expect("built");
        let prepared = prepare(
            &model,
            &salts,
            plan.sizing.max_model_len,
            next_id,
            &task.spec,
            cfg.max_new_tokens,
        )?;
        next_id += 1;
        log(format_args!(
            "{}: start ({} prompt tokens, up to {} new)",
            task.key(),
            prepared.prompt_tokens,
            prepared.request.max_tokens
        ));
        let remaining: Vec<&Task> = tasks[i..].iter().filter(|t| !done(t)).collect();
        let needs_calibration = !cal.calibrated() || !cal.step_secs.contains_key(&task.width);
        let mut refusal = None;
        let mut on_cal = |spp: f64, step: f64| -> Result<(), String> {
            let mut c = cal.clone();
            c.prefill_secs_per_position.get_or_insert(spp);
            c.step_secs.entry(task.width).or_insert(step);
            write_json(&cal_path, &c.to_json())?;
            let est = c
                .estimate(&remaining, cfg.max_new_tokens)
                .expect("calibrated");
            log(format_args!(
                "calibrated: {spp:.2} s per prefill position, {step:.1} s per decode step at \
                 width {}; the {} remaining requests take at most {} (no draft accepted; \
                 limit {})",
                task.width,
                remaining.len(),
                hms(est),
                hms(cfg.time_limit_secs)
            ));
            if est > cfg.time_limit_secs {
                let msg = format!(
                    "estimated {} for {} requests (worst case: no draft accepted), over the \
                     limit of {}",
                    hms(est),
                    remaining.len(),
                    hms(cfg.time_limit_secs)
                );
                refusal = Some(msg.clone());
                return Err(msg);
            }
            Ok(())
        };
        let result = run_task(
            eng,
            &model,
            task,
            prepared,
            &clock,
            needs_calibration.then_some(&mut on_cal as OnCalibrated<'_>),
        );
        if let Some(msg) = refusal {
            return Ok(Outcome::Refused(msg));
        }
        let result = result?;
        // The calibration follows the measured averages of finished requests.
        if let Some(spp) = result["prefill"]["secs_per_position"].as_f64() {
            cal.prefill_secs_per_position = Some(spp);
        }
        if let Some(s) = result["decode"]["secs_per_step"].as_f64() {
            cal.step_secs.insert(task.width, s);
        }
        write_json(&cal_path, &cal.to_json())?;
        std::fs::write(
            transcripts_dir.join(format!("{}.txt", task.key())),
            transcript(&result),
        )
        .map_err(|e| format!("write transcript: {e}"))?;
        write_json(&requests_dir.join(format!("{}.json", task.key())), &result)?;
        log(format_args!(
            "{}: done: {} tokens in {} decode steps ({:.2} tokens a step), {:.1} s, {}",
            task.key(),
            result["generated_tokens"],
            result["decode"]["steps"],
            result["decode"]["tokens_per_step"].as_f64().unwrap_or(0.0),
            result["wall_secs"].as_f64().unwrap_or(0.0),
            result["finish_reason"].as_str().unwrap_or("")
        ));
        eprintln!("{}", transcript(&result));
        let rest: Vec<&Task> = tasks[i + 1..].iter().filter(|t| !done(t)).collect();
        if let Some(est) = cal.estimate(&rest, cfg.max_new_tokens)
            && !rest.is_empty()
        {
            log(format_args!(
                "{} requests remain, at most {} (no draft accepted)",
                rest.len(),
                hms(est)
            ));
            if est > cfg.time_limit_secs {
                return Ok(Outcome::Refused(format!(
                    "the {} remaining requests are estimated at {}, over the limit of {}",
                    rest.len(),
                    hms(est),
                    hms(cfg.time_limit_secs)
                )));
            }
        }
    }
    drop(engine);
    summarize(cfg, &model, &tasks, &plan, width, &cal)
}

/// Writes `summary.json` from every finished request's checkpoint.
fn summarize(
    cfg: &RunConfig,
    model: &LoadedModel,
    tasks: &[Task],
    plan: &Plan,
    width: u32,
    cal: &Calibration,
) -> Result<Outcome, String> {
    let mut results = Vec::new();
    for t in tasks {
        let path = cfg
            .out_dir
            .join("requests")
            .join(format!("{}.json", t.key()));
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let v: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        results.push((t, v));
    }
    let mut totals = serde_json::Map::new();
    for w in [width, 0]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let of_width: Vec<&Value> = results
            .iter()
            .filter(|(t, _)| t.width == w)
            .map(|(_, v)| v)
            .collect();
        if of_width.is_empty() {
            continue;
        }
        let sum = |f: &dyn Fn(&Value) -> f64| of_width.iter().map(|v| f(v)).sum::<f64>();
        let mut slots = SlotCounts::default();
        for v in &of_width {
            slots.merge(&SlotCounts::from_json(&v["drafts_by_slot"]));
        }
        let (tokens, decode_steps, decode_secs, decode_tokens) = (
            sum(&|v| v["generated_tokens"].as_f64().unwrap_or(0.0)),
            sum(&|v| v["decode"]["steps"].as_f64().unwrap_or(0.0)),
            sum(&|v| v["decode"]["secs"].as_f64().unwrap_or(0.0)),
            sum(&|v| v["decode"]["tokens"].as_f64().unwrap_or(0.0)),
        );
        let (prefill_secs, prefill_positions) = (
            sum(&|v| v["prefill"]["secs"].as_f64().unwrap_or(0.0)),
            sum(&|v| v["prefill"]["positions"].as_f64().unwrap_or(0.0)),
        );
        totals.insert(
            format!("width_{w}"),
            json!({
                "requests": of_width.len(),
                "generated_tokens": tokens,
                "prefill_positions": prefill_positions,
                "prefill_secs": prefill_secs,
                "prefill_secs_per_position": (prefill_positions > 0.0)
                    .then(|| prefill_secs / prefill_positions),
                "decode_steps": decode_steps,
                "decode_secs": decode_secs,
                "decode_tokens_per_step": (decode_steps > 0.0).then(|| decode_tokens / decode_steps),
                "decode_secs_per_step": (decode_steps > 0.0).then(|| decode_secs / decode_steps),
                "drafts_by_slot": slots.to_json(),
                "peak_rss_bytes": of_width
                    .iter()
                    .filter_map(|v| v["memory"]["peak_rss_bytes"].as_u64())
                    .max(),
            }),
        );
    }
    let mut checks = Vec::new();
    let mut mismatches = 0;
    for (t, v0) in results.iter().filter(|(t, _)| t.width == 0 && width > 0) {
        let Some((_, vk)) = results
            .iter()
            .find(|(u, _)| u.width == width && u.spec.name == t.spec.name)
        else {
            continue;
        };
        let (a, b) = (v0["token_ids"].as_array(), vk["token_ids"].as_array());
        let (a, b) = (
            a.cloned().unwrap_or_default(),
            b.cloned().unwrap_or_default(),
        );
        let first_difference = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x != y)
            .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())));
        if first_difference.is_some() {
            mismatches += 1;
        }
        log(format_args!(
            "{}: greedy tokens at width 0 and width {width} {}",
            t.spec.name,
            match first_difference {
                None => "are identical".to_string(),
                Some(i) => format!("DIFFER from token {i}"),
            }
        ));
        checks.push(json!({
            "name": t.spec.name,
            "identical": first_difference.is_none(),
            "first_difference": first_difference,
            "tokens": [a.len(), b.len()],
        }));
    }
    let summary = json!({
        "generated_at": timestamp(),
        "model": "DeepSeek-V4.1-Flash (text), CPU reference executor (DsCpuExecutor)",
        "weights_dir": cfg.weights_dir.display().to_string(),
        "weights_sha256": model.weights_hash(),
        "weights_hash_verified": true,
        "sampling": "greedy",
        "max_new_tokens": cfg.max_new_tokens,
        "draft_width": width,
        "sizing": {
            "kv_block_size": plan.sizing.kv_block_size,
            "kv_blocks_per_group": plan.kv_blocks,
            "max_model_len": plan.sizing.max_model_len,
            "max_batched_tokens": plan.sizing.max_batched_tokens,
            "max_prefill_chunk": plan.sizing.max_prefill_chunk,
        },
        "totals": totals,
        "width0_checks": checks,
        "calibration": cal.to_json(),
        "requests": results.iter().map(|(_, v)| {
            let mut v = v.clone();
            if let Some(o) = v.as_object_mut() {
                o.remove("step_log");
                o.remove("prompt_text");
            }
            v
        }).collect::<Vec<_>>(),
        "memory": memory(),
    });
    write_json(&cfg.out_dir.join("summary.json"), &summary)?;
    log(format_args!(
        "wrote {}; {}",
        cfg.out_dir.join("summary.json").display(),
        memory_line()
    ));
    Ok(Outcome::Complete { mismatches })
}
