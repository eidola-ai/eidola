//! The released DeepSeek-V4.1 checkpoint on two GPUs, every weight resident: the encoder
//! half on one device and the decoder half (with its Engram table and, for generation,
//! the drafter) on the other, as a node runs it. Dev-only.
//!
//! Phases, run in the order given (`EIDOLA_DS_TWO_PHASES`); each loads what it needs:
//!
//! - `tokens` (no device): the sequences the other phases and the session's yardstick
//!   run, tokenized with the checkpoint's tokenizer and the chat crate's encoder, into
//!   `sequences.json`: the health check's `hello` (thinking off, with the publicly reported
//!   reply, where the reply starts, and the near-tie position with its three reported
//!   tokens), `story` (the story as the reply to a chat request) and `passage` (the
//!   passage after the beginning-of-sequence token), from the texts file the session
//!   extracts from the golden tooling; `ds_generate`'s chat prompts as the node renders
//!   them (thinking on, effort `high`); and the throughput prompts (a fixed input length,
//!   each a different window of the story and the passage). Also `passage.safetensors`,
//!   the passage's tokens and the Engram map in the form `ds_halves` reads
//!   (`EIDOLA_DS_HALVES_GOLDEN`).
//! - `compare`, `score` (one load): both halves through the executor's own step path
//!   (`ds_both`), the Engram rows gathered from the table on the decoder's device.
//!   `compare` runs the text and steps the one-half-at-a-time measurement recorded
//!   (`ds_halves`) and compares every layer's streams, the boundary, the bank entries and
//!   the logits with its recordings bit for bit. `score` runs each health sequence in a
//!   slot of its own with logits at every position: next-token accuracy from where its
//!   reply starts (floors 0.55 for `story` and `passage`), and for `hello` the reported
//!   reply as the most likely token at every position but the near tie, where it must be
//!   one of the three reported. When both ran, `passage` in its own slot must also equal
//!   the comparison's logits bit for bit.
//! - `generate`, `bench` (one load): the executor itself (`DsCudaExecutor::new` with
//!   devices `EIDOLA_DS_TWO_DEVICES`) under the serving core's engine. `generate`:
//!   greedy generation of `ds_generate`'s prompts at each width of `EIDOLA_DS_TWO_WIDTHS`;
//!   the tokens must be the same at every width and, where `EIDOLA_DS_TWO_CPU` names a
//!   `ds_generate` run on the CPU reference executor, the same as its tokens for the same
//!   prompt tokens. `bench`: each width of `EIDOLA_DS_TWO_BENCH_WIDTHS` at each
//!   concurrency of `EIDOLA_DS_TWO_CONCURRENCY`, the throughput prompts with a fixed
//!   output length (no end-of-sequence token ends them): per request the time to the first
//!   token and the decode rate after it, per run the output tokens per second and the
//!   drafts accepted.
//!
//! The GEMMs' form is `EIDOLA_DS_GEMM_FORM` (`GemmForm::from_env`) on both devices. Before
//! loading, each loading phase lays its configuration out on two devices of one B300's
//! memory with no memory behind them (`ds::dry`) and runs its launches there, so whatever
//! does not fit or is refused fails before anything is read.
//!
//! ```sh
//! EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> EIDOLA_DEEPSEEK_FULL_DIR=<checkpoint> \
//! EIDOLA_DS_TWO_OUT=<dir> EIDOLA_DS_TWO_TEXTS=<texts.json> EIDOLA_DS_GEMM_FORM=banded \
//!     cargo run --release -p eidola-engine-cuda --example ds_two
//! ```
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `EIDOLA_DS_TWO_PHASES` | of `tokens`, `compare`, `score`, `generate`, `bench` | all five |
//! | `EIDOLA_DS_TWO_OUT` | results | `ds_two` in the working directory |
//! | `EIDOLA_DS_TWO_TEXTS` | the health texts (`tokens`) | |
//! | `EIDOLA_DS_TWO_RECORDED` | the `ds_halves` output (`compare`) | |
//! | `EIDOLA_DS_TWO_CPU` | a `ds_generate` output on the CPU (`generate`) | none |
//! | `EIDOLA_DS_TWO_DEVICES` | the encoder's and the decoder's GPU | `0,1` |
//! | `EIDOLA_DS_TWO_PROMPTS` | a prompts file in `ds_generate`'s form | `ds_generate`'s |
//! | `EIDOLA_DS_TWO_MAX_NEW` | new tokens per prompt (`generate`) | 96 |
//! | `EIDOLA_DS_TWO_WIDTHS` | draft widths of `generate` | `0,5` |
//! | `EIDOLA_DS_TWO_BENCH_WIDTHS` | draft widths of `bench` | `0,5` |
//! | `EIDOLA_DS_TWO_CONCURRENCY` | concurrencies of `bench` | `1,8,32` |
//! | `EIDOLA_DS_TWO_BENCH_IN`, `_OUT` | prompt and output tokens of `bench` | 1024, 256 |
//! | `EIDOLA_DS_TWO_CHUNK` | positions per step (prefill chunk, step budget) | 2048 |
//!
//! Results: `sequences.json`, `compare.json`, `score.json`, `generate.json`, `bench.json`,
//! and `ds_two.json` (every phase's outcome and time). Exit status 1 when anything
//! compared differs or a health check fails, 2 on a failure.

#[path = "ds_both/mod.rs"]
mod ds_both;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ds_both::{Layout, Recording, Unwatched, fail, memory_json, n32, us};
use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cuda::ds::dry::DryDevice;
use eidola_engine_cuda::ds::gemm::GemmForm;
use eidola_engine_cuda::ds::half::{CudaDevice, Device};
use eidola_engine_cuda::ds::model::HalfNet;
use eidola_engine_cuda::ds::weights::Half;
use eidola_engine_cuda::ds::{DsCudaExecutor, DsCudaExecutorConfig};
use eidola_engine_cuda::{KernelDir, Result};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::{DsConfig, config_of};
use serde_json::{Value, json};

/// One B300's memory, 275,040 MiB: what the dry run lays each half out within.
const B300_BYTES: usize = 275_040 << 20;
/// Next-token accuracy floors of the health check.
const STORY_FLOOR: f64 = 0.55;
const PASSAGE_FLOOR: f64 = 0.55;

/// Opens device `d` of the two (0 the encoder's, 1 the decoder's).
type Open<'a> = &'a dyn Fn(usize) -> Result<Box<dyn Device>>;

// ---------------------------------------------------------------------------------------
// tokens

/// The health check's texts, as the golden tooling holds them.
#[derive(Debug, Clone)]
struct Texts {
    system: String,
    hello_user: String,
    hello_reply: String,
    hello_fork: String,
    hello_fork_top3: Vec<String>,
    story_user: String,
    story: String,
    passage: String,
}

impl Texts {
    fn parse(v: &Value) -> std::result::Result<Texts, String> {
        let s = |k: &str| {
            v[k].as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("the texts have no {k}"))
        };
        Ok(Texts {
            system: s("system")?,
            hello_user: s("hello_user")?,
            hello_reply: s("hello_reply")?,
            hello_fork: s("hello_fork")?,
            hello_fork_top3: v["hello_fork_top3"]
                .as_array()
                .ok_or("the texts have no hello_fork_top3")?
                .iter()
                .map(|t| t.as_str().map(str::to_owned).ok_or("a non-string token"))
                .collect::<std::result::Result<_, _>>()?,
            story_user: s("story_user")?,
            story: s("story")?,
            passage: s("passage")?,
        })
    }
}

/// A chat request without thinking (the health check's `thinking_mode="chat"`) followed
/// by `reply`: its tokens and where the reply starts.
fn chat_and_reply(
    tokenizer: &eidola_engine_chat::Tokenizer,
    system: &str,
    user: &str,
    reply: &str,
) -> std::result::Result<(Vec<u32>, usize), String> {
    use eidola_engine_chat::ChatInput;
    use eidola_engine_chat::deepseek_v4_1::{self, EncodeOptions};
    let messages = json!([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]);
    let input = ChatInput::from_json(&messages.to_string(), None).map_err(|e| e.to_string())?;
    let options = EncodeOptions::from_request(Some(false), None).map_err(|e| e.to_string())?;
    let prompt = deepseek_v4_1::encode(&input, options).map_err(|e| e.to_string())?;
    let mut tokens = tokenizer
        .encode_prompt(&prompt)
        .map_err(|e| e.to_string())?;
    let start = tokens.len();
    tokens.extend(tokenizer.encode(reply).map_err(|e| e.to_string())?);
    Ok((tokens, start))
}

/// The `tokens` phase's output: `sequences.json`'s value and the passage golden's bytes.
fn tokens_phase(
    dir: &Path,
    texts: &Texts,
    prompts_file: &Path,
    bench_in: usize,
    bench_count: usize,
) -> std::result::Result<(Value, Vec<u8>), String> {
    let (prompts, eos, tokenizer, map) = ds_both::prompts(dir, prompts_file)?;
    let enc = |t: &str| tokenizer.encode(t).map_err(|e| e.to_string());
    let (hello, hello_start) = chat_and_reply(
        &tokenizer,
        &texts.system,
        &texts.hello_user,
        &texts.hello_reply,
    )?;
    let bos = hello[0];
    if hello.iter().filter(|&&t| t == bos).count() != 1 {
        return Err(
            "the rendered prompt does not start with one beginning-of-sequence token".into(),
        );
    }
    let fork = hello_start - 1 + enc(&texts.hello_fork)?.len();
    let mut fork_tokens = Vec::new();
    for t in &texts.hello_fork_top3 {
        match enc(t)?.as_slice() {
            [one] => fork_tokens.push(*one),
            other => return Err(format!("{t:?} is {} tokens, not one", other.len())),
        }
    }
    if hello.get(fork + 1) != fork_tokens.first() {
        return Err("the reply's token after the fork is not the first reported token".into());
    }
    let (story, story_start) =
        chat_and_reply(&tokenizer, &texts.system, &texts.story_user, &texts.story)?;
    let mut passage = vec![bos];
    passage.extend(enc(&texts.passage)?);

    // The throughput prompts: windows of the story and the passage, each its own.
    let mut stream = enc(&texts.story)?;
    stream.extend(enc(&texts.passage)?);
    let bench: Vec<Vec<u32>> = (0..bench_count)
        .map(|i| {
            let mut p = vec![bos];
            p.extend((0..bench_in.saturating_sub(1)).map(|j| stream[(i * 101 + j) % stream.len()]));
            p
        })
        .collect();

    let golden = safetensors_i64(&[
        ("tokens", passage.iter().map(|&t| i64::from(t)).collect()),
        (
            "engram.token_map",
            map.iter().map(|&t| i64::from(t)).collect(),
        ),
    ]);
    let v = json!({
        "bos": bos,
        "eos": eos,
        "health": {
            "hello": {"tokens": hello, "start": hello_start, "fork": fork, "fork_tokens": fork_tokens},
            "story": {"tokens": story, "start": story_start},
            "passage": {"tokens": passage, "start": 1},
        },
        "prompts": prompts.iter().map(|p| json!({"name": p.name, "tokens": p.tokens})).collect::<Vec<_>>(),
        "bench": {"input_len": bench_in, "prompts": bench},
    });
    Ok((v, golden))
}

/// A safetensors file of one-dimensional I64 tensors.
fn safetensors_i64(tensors: &[(&str, Vec<i64>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut at = 0usize;
    for (name, data) in tensors {
        let bytes = data.len() * 8;
        header.insert(
            (*name).into(),
            json!({"dtype": "I64", "shape": [data.len()], "data_offsets": [at, at + bytes]}),
        );
        at += bytes;
    }
    let mut text = Value::Object(header).to_string().into_bytes();
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + text.len() + at);
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&text);
    for (_, data) in tensors {
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// One health sequence of `sequences.json`.
#[derive(Debug, Clone)]
struct Health {
    name: String,
    tokens: Vec<u32>,
    start: usize,
    fork: Option<(usize, Vec<u32>)>,
}

fn u32_list(v: &Value) -> std::result::Result<Vec<u32>, String> {
    v.as_array()
        .ok_or("not a list")?
        .iter()
        .map(|t| {
            t.as_u64()
                .and_then(|t| u32::try_from(t).ok())
                .ok_or_else(|| "not a token id".to_string())
        })
        .collect()
}

fn health_of(seqs: &Value) -> std::result::Result<Vec<Health>, String> {
    let mut out = Vec::new();
    for name in ["hello", "story", "passage"] {
        let h = &seqs["health"][name];
        let index = |v: &Value| v.as_u64().and_then(|v| usize::try_from(v).ok());
        let start = index(&h["start"]).ok_or("no start")?;
        let fork = match h.get("fork") {
            Some(f) => Some((index(f).ok_or("no fork")?, u32_list(&h["fork_tokens"])?)),
            None => None,
        };
        out.push(Health {
            name: name.into(),
            tokens: u32_list(&h["tokens"])?,
            start,
            fork,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// compare, score

/// The index of the largest value (the first of equal ones).
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    n32(best)
}

/// The `k` largest log-probabilities of a row of logits, with their ids.
fn top_logprobs(row: &[f32], k: usize) -> Vec<(u32, f64)> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|v| f64::from(v - max).exp()).sum();
    let log_z = f64::from(max) + sum.ln();
    let mut ids: Vec<usize> = (0..row.len()).collect();
    ids.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    ids.into_iter()
        .take(k)
        .map(|i| (n32(i), f64::from(row[i]) - log_z))
        .collect()
}

/// A health sequence scored from its logits (`[positions, vocab]`): next-token accuracy
/// from where its reply starts, and for `hello` the reported reply.
fn score_one(h: &Health, logits: &[f32], vocab: usize) -> (Value, bool) {
    let n = h.tokens.len();
    let positions: Vec<usize> = (h.start.max(1) - 1..n - 1).collect();
    let argmax: Vec<u32> = positions
        .iter()
        .map(|&i| argmax(&logits[i * vocab..(i + 1) * vocab]))
        .collect();
    let hits = positions
        .iter()
        .zip(&argmax)
        .filter(|(i, a)| **a == h.tokens[**i + 1])
        .count();
    let accuracy = hits as f64 / positions.len().max(1) as f64;
    let mut v = json!({
        "tokens": n, "start": h.start, "scored": positions.len(),
        "accuracy": accuracy, "argmax": argmax,
    });
    let healthy = match &h.fork {
        Some((fork, reported)) => {
            let wrong: Vec<usize> = positions
                .iter()
                .zip(&argmax)
                .filter(|(i, a)| **i != *fork && **a != h.tokens[**i + 1])
                .map(|(i, _)| *i)
                .collect();
            let top = top_logprobs(&logits[fork * vocab..(fork + 1) * vocab], 3);
            let fork_ok = reported.contains(&top[0].0);
            v["wrong_positions"] = json!(wrong);
            v["fork"] = json!(fork);
            v["fork_top3"] = json!(top.iter().map(|(t, l)| json!([t, l])).collect::<Vec<_>>());
            v["fork_ok"] = json!(fork_ok);
            wrong.is_empty() && fork_ok
        }
        None => {
            let floor = if h.name == "story" {
                STORY_FLOOR
            } else {
                PASSAGE_FLOOR
            };
            v["floor"] = json!(floor);
            accuracy >= floor
        }
    };
    v["healthy"] = json!(healthy);
    (v, healthy)
}

/// What `compare` and `score` run in one load.
struct Resident {
    /// The recorded text and its step (`compare`), with the recording's directory.
    recorded: Option<(Vec<u32>, PathBuf)>,
    /// The health sequences (`score`).
    health: Vec<Health>,
}

/// Both halves resident on devices from `open`, then each sequence of `r` in its own
/// slot: the comparison with the recording, and the scores. Writes the recording of the
/// comparison under `out` when given. Returns `compare.json`'s and `score.json`'s values
/// and whether everything held.
fn resident(
    l: &Layout,
    r: &Resident,
    open: Open<'_>,
    out: Option<&Path>,
) -> Result<(Value, Value, bool)> {
    let began = Instant::now();
    let enc_dev = open(0)?;
    let w = l.resident(Half::Encoder, enc_dev.as_ref())?;
    let mut enc: HalfNet = l.net(Half::Encoder, enc_dev, w)?;
    let dec_dev = open(1)?;
    let w = l.resident(Half::Decoder, dec_dev.as_ref())?;
    let mut dec = l.net(Half::Decoder, dec_dev, w)?;
    enc.dev.synchronize()?;
    dec.dev.synchronize()?;
    let load = began.elapsed().as_secs_f64();
    let memory = json!([memory_json(&enc.memory()), memory_json(&dec.memory())]);
    eprintln!("both halves resident in {load:.0} s; bytes {memory}");
    let mut all = true;
    let mut compared_logits = None;
    let compare = match &r.recorded {
        Some((tokens, dir)) => {
            let rec: Recording = l.seq(tokens, 0).run(&mut enc, &mut dec, &Unwatched)?;
            if let Some(out) = out {
                rec.write(out)?;
            }
            let (checks, equal) = rec.compare(l, tokens.len(), dir)?;
            all &= equal;
            compared_logits = Some((tokens.clone(), rec.logits.clone()));
            json!({
                "load_seconds": load, "device_bytes": memory, "tokens": tokens.len(),
                "chunk": l.chunk, "steps": rec.steps, "recorded": dir.display().to_string(),
                "equal": equal, "checks": checks,
            })
        }
        None => Value::Null,
    };
    let mut scores = serde_json::Map::new();
    let vocab = l.c.vocab_size;
    for (i, h) in r.health.iter().enumerate() {
        let slot = n32(i) + u32::from(r.recorded.is_some());
        let at = Instant::now();
        let rec = l.seq(&h.tokens, slot).run(&mut enc, &mut dec, &Unwatched)?;
        let (mut v, healthy) = score_one(h, &rec.logits, vocab);
        all &= healthy;
        v["seconds"] = json!(at.elapsed().as_secs_f64());
        if let Some((tokens, logits)) = &compared_logits
            && *tokens == h.tokens
        {
            let same = *logits == rec.logits;
            v["equals_the_comparisons_logits"] = json!(same);
            all &= same;
        }
        eprintln!(
            "{}: accuracy {:.4} over {} positions; healthy {healthy}",
            h.name, v["accuracy"], v["scored"]
        );
        scores.insert(h.name.clone(), v);
    }
    let score = json!({
        "load_seconds": load, "device_bytes": memory, "sequences": scores,
        "healthy": scores.values().all(|v| v["healthy"] == json!(true)),
    });
    Ok((compare, score, all))
}

// ---------------------------------------------------------------------------------------
// generate, bench

/// The executor under the engine with a narrower draft width: the engine asks a row for
/// at most `spec.max_draft_tokens` drafts, and the executor drafts what a row asks for.
struct Narrow<'a> {
    x: &'a mut DsCudaExecutor,
    spec: ModelSpec,
}

impl Executor for Narrow<'_> {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> std::result::Result<StepOutput, ExecutorError> {
        self.x.execute(step)
    }
}

/// The settings of `generate` and `bench`.
#[derive(Clone, Debug)]
struct Serving {
    widths: Vec<u32>,
    max_new: u32,
    bench_widths: Vec<u32>,
    concurrency: Vec<u32>,
    bench_out: u32,
    chunk: u32,
}

/// The executor configuration both phases share: the widest width and concurrency, a
/// sequence holding the longest prompt and its output.
fn serving_config(
    c: &DsConfig,
    s: &Serving,
    longest: usize,
    sampleable: u32,
    devices: [usize; 2],
) -> DsCudaExecutorConfig {
    let block = 64;
    let width = s
        .widths
        .iter()
        .chain(&s.bench_widths)
        .copied()
        .max()
        .unwrap_or(0);
    let seqs = s.concurrency.iter().copied().max().unwrap_or(1).max(1);
    let max_new = s.max_new.max(s.bench_out);
    let max_model_len = (n32(longest) + max_new + width + 1).next_multiple_of(block);
    let max_model_len = c
        .max_positions
        .map_or(max_model_len, |m| max_model_len.min(n32(m)));
    DsCudaExecutorConfig {
        block_size: block,
        // Every seat's sequence, the null block, and a little room.
        num_blocks: seqs * (max_model_len / block) + 8,
        num_state_slots: seqs,
        max_model_len,
        buckets: vec![Bucket {
            max_seqs: seqs,
            max_tokens: s.chunk.max(seqs * (width + 1)),
        }],
        sampleable_vocab_size: sampleable,
        devices,
        draft_width: width,
        host_blocks: 0,
        position_tags: false,
        record: false,
        gemm: GemmForm::Exact,
    }
}

/// An engine over `x` at draft `width`, seating up to `seqs` sequences.
fn engine<'a>(
    x: &'a mut DsCudaExecutor,
    width: u32,
    seqs: u32,
    chunk: u32,
    eos: &[u32],
) -> Result<Engine<Narrow<'a>>> {
    let most = x.spec().max_draft_tokens;
    if width > most {
        return Err(fail(format!(
            "draft width {width}: the executor drafts at most {most}"
        )));
    }
    let mut spec = x.spec().clone();
    if width > 0 {
        spec.max_draft_tokens = width;
    }
    let sched = SchedulerConfig {
        max_batched_tokens: chunk.max(seqs * (width + 1)),
        max_seqs: seqs,
        max_prefill_chunk: chunk,
        eos_token_ids: eos.to_vec(),
        speculative: width > 0,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    };
    Engine::new(Narrow { x, spec }, sched).map_err(|e| fail(format!("{e:?}")))
}

/// One request's course through a run.
#[derive(Clone, Debug, Default)]
struct Course {
    tokens: Vec<u32>,
    submitted: f64,
    first: Option<f64>,
    finished: Option<f64>,
}

impl Course {
    fn json(&self) -> Value {
        let ttft = self.first.map(|f| f - self.submitted);
        let decode = match (self.first, self.finished) {
            (Some(f), Some(e)) if self.tokens.len() > 1 && e > f => {
                Some((self.tokens.len() - 1) as f64 / (e - f))
            }
            _ => None,
        };
        json!({
            "tokens": self.tokens.len(), "ttft_seconds": ttft,
            "seconds": self.finished.map(|e| e - self.submitted),
            "decode_tokens_per_second": decode,
        })
    }
}

/// Runs `prompts` through `eng`, at most `concurrency` in flight, each up to `max_new`
/// tokens, greedily. Returns each request's course and the run's wall time.
fn drive(
    eng: &mut Engine<Narrow<'_>>,
    prompts: &[Vec<u32>],
    max_new: u32,
    concurrency: usize,
) -> Result<(Vec<Course>, f64)> {
    let clock = Instant::now();
    let mut courses = vec![Course::default(); prompts.len()];
    let mut waiting: VecDeque<usize> = (0..prompts.len()).collect();
    let mut running = 0usize;
    loop {
        while running < concurrency.max(1)
            && let Some(i) = waiting.pop_front()
        {
            courses[i].submitted = clock.elapsed().as_secs_f64();
            eng.submit(Request {
                id: i as u64,
                prompt: prompts[i].clone(),
                sampling: SamplingParams::greedy(),
                max_tokens: max_new,
                required_cached_tokens: 0,
                stop_token_ids: Vec::new(),
                cache: CacheScope::Private,
            })
            .map_err(|e| fail(format!("request {i}: {e:?}")))?;
            running += 1;
        }
        if running == 0 {
            break;
        }
        let now = u64::try_from(clock.elapsed().as_millis()).unwrap_or(u64::MAX);
        for e in eng.step(now).map_err(fail)? {
            let at = clock.elapsed().as_secs_f64();
            let c = &mut courses[usize::try_from(e.id).map_err(fail)?];
            if !e.tokens.is_empty() && c.first.is_none() {
                c.first = Some(at);
            }
            c.tokens.extend(&e.tokens);
            if e.finish.is_some() {
                c.finished = Some(at);
                running -= 1;
            }
        }
    }
    eng.flush().map_err(fail)?;
    Ok((courses, clock.elapsed().as_secs_f64()))
}

fn mean(xs: &[f64]) -> Option<f64> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

fn median(xs: &[f64]) -> Option<f64> {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    (!v.is_empty()).then(|| v[v.len() / 2])
}

/// The CPU reference executor's tokens for a prompt, from a `ds_generate` output: its
/// prompt's tokens and its generated tokens, from the result of any width.
fn cpu_tokens(dir: &Path, name: &str) -> Option<(Vec<u32>, Vec<u32>, u64)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join("requests"))
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with(&format!("{name}.w")) && f.ends_with(".json"))
        })
        .collect();
    files.sort();
    let v: Value = serde_json::from_slice(&std::fs::read(files.first()?).ok()?).ok()?;
    Some((
        u32_list(&v["prompt_token_ids"]).ok()?,
        u32_list(&v["token_ids"]).ok()?,
        v["max_tokens"].as_u64()?,
    ))
}

/// Greedy generation of every prompt at each width; the tokens must agree across widths
/// and with the CPU's where given. Returns `generate.json`'s value and whether all agree.
fn generate(
    x: &mut DsCudaExecutor,
    prompts: &[(String, Vec<u32>)],
    s: &Serving,
    eos: &[u32],
    cpu: Option<&Path>,
) -> Result<(Value, bool)> {
    let mut outputs: Vec<Vec<Vec<u32>>> = vec![Vec::new(); prompts.len()];
    let mut results = Vec::new();
    for &width in &s.widths {
        let mut eng = engine(x, width, 1, s.chunk, eos)?;
        for (i, (name, tokens)) in prompts.iter().enumerate() {
            let before = eng.stats();
            let (courses, _) = drive(&mut eng, std::slice::from_ref(tokens), s.max_new, 1)?;
            let after = eng.stats();
            let c = &courses[0];
            eprintln!("{name} at width {width}: {}", c.json());
            let mut r = c.json();
            r["prompt"] = json!(name);
            r["width"] = json!(width);
            r["prompt_tokens"] = json!(tokens.len());
            r["token_ids"] = json!(c.tokens);
            r["drafted"] = json!(after.drafted - before.drafted);
            r["accepted"] = json!(after.accepted - before.accepted);
            results.push(r);
            outputs[i].push(c.tokens.clone());
        }
    }
    let mut agree = true;
    let mut checks = Vec::new();
    for ((name, tokens), outs) in prompts.iter().zip(&outputs) {
        let same = outs.windows(2).all(|w| w[0] == w[1]);
        agree &= same;
        let mut check = json!({"prompt": name, "same_tokens_at_every_width": same});
        if let Some(dir) = cpu {
            match cpu_tokens(dir, name) {
                Some((prompt, theirs, max)) => {
                    let ours = &outs[0];
                    let n = ours.len().min(theirs.len());
                    let prefix = ours.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
                    let same_prompt = prompt == *tokens;
                    // Equal over the shorter run; and equal lengths when both ran to the
                    // same limit.
                    let equal = same_prompt
                        && prefix == n
                        && (u64::from(s.max_new) != max || ours.len() == theirs.len());
                    agree &= equal;
                    check["cpu"] = json!({
                        "same_prompt_tokens": same_prompt, "cpu_tokens": theirs.len(),
                        "cpu_max_tokens": max, "common_prefix": prefix, "equal": equal,
                    });
                }
                None => check["cpu"] = json!("no CPU result for this prompt"),
            }
        }
        checks.push(check);
    }
    Ok((
        json!({"results": results, "checks": checks, "equal": agree}),
        agree,
    ))
}

/// Every width at every concurrency over the throughput prompts. At concurrency 1 three
/// requests run one after another; otherwise one request per seat, all at once.
fn bench(x: &mut DsCudaExecutor, prompts: &[Vec<u32>], s: &Serving) -> Result<Value> {
    let mut runs = Vec::new();
    for &width in &s.bench_widths {
        for &c in &s.concurrency {
            let n = if c == 1 { 3 } else { us(c) };
            if n > prompts.len() {
                return Err(fail(format!("{n} requests, {} prompts", prompts.len())));
            }
            let mut eng = engine(x, width, c, s.chunk, &[])?;
            let (courses, wall) = drive(&mut eng, &prompts[..n], s.bench_out, us(c))?;
            let stats = eng.stats();
            let ttft: Vec<f64> = courses
                .iter()
                .filter_map(|c| c.first.map(|f| f - c.submitted))
                .collect();
            let decode: Vec<f64> = courses
                .iter()
                .filter_map(|c| c.json()["decode_tokens_per_second"].as_f64())
                .collect();
            let tokens: usize = courses.iter().map(|c| c.tokens.len()).sum();
            let run = json!({
                "width": width, "concurrency": c, "requests": n,
                "input_len": prompts[0].len(), "output_len": s.bench_out,
                "wall_seconds": wall, "output_tokens": tokens,
                "output_tokens_per_second": tokens as f64 / wall,
                "mean_ttft_seconds": mean(&ttft), "median_ttft_seconds": median(&ttft),
                "mean_decode_tokens_per_second": mean(&decode),
                "median_decode_tokens_per_second": median(&decode),
                "drafted": stats.drafted, "accepted": stats.accepted,
                "requests_detail": courses.iter().map(Course::json).collect::<Vec<_>>(),
            });
            eprintln!("bench {}", {
                let mut short = run.clone();
                short.as_object_mut().unwrap().remove("requests_detail");
                short
            });
            runs.push(run);
        }
    }
    Ok(json!({"runs": runs}))
}

// ---------------------------------------------------------------------------------------
// main

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> T {
    var(name).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{name} ({v}) is not a number"))
    })
}

fn list(name: &str, default: &str) -> Vec<u32> {
    var(name)
        .unwrap_or_else(|| default.into())
        .split(',')
        .map(|w| {
            w.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name}: {w} is not a number"))
        })
        .collect()
}

fn read_json(path: &Path) -> std::result::Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(v).map_err(fail)?).map_err(fail)
}

/// The recorded text and step of a `ds_halves` output: its tokens (from the golden file
/// it ran over) and its chunk.
fn recorded_text(recorded: &Path, golden: &Path) -> Result<(Vec<u32>, u32)> {
    let halves = read_json(&recorded.join("ds_halves.json")).map_err(fail)?;
    let count = halves["tokens"]
        .as_u64()
        .ok_or_else(|| fail("the recording's token count"))?;
    let chunk = halves["chunk"]
        .as_u64()
        .ok_or_else(|| fail("the recording's chunk"))?;
    let g = WeightSet::open_files(&[golden.to_path_buf()]).map_err(fail)?;
    let v = g
        .get("tokens")
        .map_err(fail)?
        .to_i64("tokens")
        .map_err(fail)?;
    let mut tokens: Vec<u32> = v
        .into_iter()
        .map(|t| u32::try_from(t).map_err(fail))
        .collect::<Result<_>>()?;
    tokens.truncate(usize::try_from(count).map_err(fail)?);
    Ok((tokens, u32::try_from(chunk).map_err(fail)?))
}

fn main() {
    let dir = PathBuf::from(
        var("EIDOLA_DEEPSEEK_FULL_DIR").expect("EIDOLA_DEEPSEEK_FULL_DIR names the checkpoint"),
    );
    let out = var("EIDOLA_DS_TWO_OUT").map_or_else(|| PathBuf::from("ds_two"), PathBuf::from);
    std::fs::create_dir_all(&out).expect("the output directory");
    let phases: Vec<String> = var("EIDOLA_DS_TWO_PHASES")
        .unwrap_or_else(|| "tokens,compare,score,generate,bench".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let devices = list("EIDOLA_DS_TWO_DEVICES", "0,1");
    let devices = [us(devices[0]), us(devices[1])];
    let serving = Serving {
        widths: list("EIDOLA_DS_TWO_WIDTHS", "0,5"),
        max_new: number("EIDOLA_DS_TWO_MAX_NEW", 96),
        bench_widths: list("EIDOLA_DS_TWO_BENCH_WIDTHS", "0,5"),
        concurrency: list("EIDOLA_DS_TWO_CONCURRENCY", "1,8,32"),
        bench_out: number("EIDOLA_DS_TWO_BENCH_OUT", 256),
        chunk: number("EIDOLA_DS_TWO_CHUNK", 2048),
    };
    let bench_in: usize = number("EIDOLA_DS_TWO_BENCH_IN", 1024);
    let prompts_file =
        var("EIDOLA_DS_TWO_PROMPTS").map_or_else(ds_both::default_prompts_file, PathBuf::from);
    let summary_path = out.join("ds_two.json");
    let mut summary = read_json(&summary_path).unwrap_or_else(|_| json!({}));
    summary["gemm"] = json!(format!(
        "{:?}",
        GemmForm::from_env().expect("the GEMM form")
    ));
    summary["devices"] = json!(devices);
    let save = |summary: &Value| write_json(&summary_path, summary).expect("write the summary");
    let exit = |code: i32, what: &str| -> ! {
        eprintln!("{what}");
        std::process::exit(code)
    };
    let kernels = || -> Result<KernelDir> {
        let k = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR")
            .ok_or_else(|| fail("EIDOLA_ENGINE_KERNELS_DIR is not set"))?;
        Ok(KernelDir::new(k))
    };
    let sequences_path = out.join("sequences.json");
    let golden_path = out.join("passage.safetensors");
    let mut good = true;
    let has = |p: &str| phases.iter().any(|q| q == p);

    if has("tokens") {
        let began = Instant::now();
        let texts = var("EIDOLA_DS_TWO_TEXTS")
            .ok_or_else(|| "EIDOLA_DS_TWO_TEXTS names the health texts".to_string())
            .and_then(|p| read_json(Path::new(&p)))
            .and_then(|v| Texts::parse(&v));
        let made = texts.and_then(|t| {
            let count = serving
                .concurrency
                .iter()
                .map(|&c| us(c))
                .max()
                .unwrap_or(1)
                .max(3);
            tokens_phase(&dir, &t, &prompts_file, bench_in, count)
        });
        match made {
            Ok((v, golden)) => {
                write_json(&sequences_path, &v).unwrap();
                std::fs::write(&golden_path, golden).expect("write the passage golden");
                let lengths: Vec<usize> = ["hello", "story", "passage"]
                    .iter()
                    .map(|n| v["health"][n]["tokens"].as_array().map_or(0, Vec::len))
                    .collect();
                summary["tokens"] = json!({
                    "seconds": began.elapsed().as_secs_f64(),
                    "lengths": lengths,
                });
                save(&summary);
            }
            Err(e) => exit(2, &format!("FAILED in tokens: {e}")),
        }
    }

    let need = |p: &str| has(p);
    if need("compare") || need("score") || need("generate") || need("bench") {
        let store = Arc::new(WeightSet::open_dir(&dir).expect("the checkpoint"));
        let c = config_of(&store).expect("the checkpoint's config");
        let seqs = read_json(&sequences_path).unwrap_or_else(|e| exit(2, &format!("FAILED: {e}")));
        let token_map: Vec<u32> = {
            let g = WeightSet::open_files(std::slice::from_ref(&golden_path))
                .expect("the passage golden");
            g.get("engram.token_map")
                .and_then(|t| t.to_i64("engram.token_map"))
                .expect("the Engram map")
                .into_iter()
                .map(|t| u32::try_from(t).unwrap())
                .collect()
        };

        if need("compare") || need("score") {
            let began = Instant::now();
            let result = (|| -> Result<(Value, Value, bool)> {
                let recorded = if need("compare") {
                    let rec = PathBuf::from(var("EIDOLA_DS_TWO_RECORDED").ok_or_else(|| {
                        fail("EIDOLA_DS_TWO_RECORDED names the ds_halves output")
                    })?);
                    let golden = var("EIDOLA_DS_HALVES_GOLDEN")
                        .map_or_else(|| golden_path.clone(), PathBuf::from);
                    Some((rec, golden))
                } else {
                    None
                };
                let (recorded, chunk) = match &recorded {
                    Some((rec, golden)) => {
                        let (tokens, chunk) = recorded_text(rec, golden)?;
                        (Some((tokens, rec.clone())), chunk)
                    }
                    None => (None, 96),
                };
                let health = if need("score") {
                    health_of(&seqs).map_err(fail)?
                } else {
                    Vec::new()
                };
                let r = Resident { recorded, health };
                let longest = r
                    .health
                    .iter()
                    .map(|h| h.tokens.len())
                    .chain(r.recorded.iter().map(|(t, _)| t.len()))
                    .max()
                    .unwrap_or(1);
                let slots = n32(r.health.len()) + u32::from(r.recorded.is_some());
                let dry = Layout::new(None, c.clone(), token_map.clone(), longest, slots, chunk)?;
                let open_dry = |_: usize| -> Result<Box<dyn Device>> {
                    Ok(Box::new(DryDevice::new(Some(B300_BYTES))))
                };
                let dry_run = Resident {
                    recorded: None,
                    health: r.health.clone(),
                };
                resident(&dry, &dry_run, &open_dry, None)?;
                eprintln!("dry run: both halves fit a B300 each and every launch is accepted");
                let l = Layout::new(
                    Some(store.clone()),
                    c.clone(),
                    token_map.clone(),
                    longest,
                    slots,
                    chunk,
                )?;
                let k = kernels()?;
                let open = |d: usize| -> Result<Box<dyn Device>> {
                    let mut dev = CudaDevice::open(devices[d], &k)?;
                    dev.select_gemm(GemmForm::from_env()?)?;
                    Ok(Box::new(dev))
                };
                resident(&l, &r, &open, Some(&out.join("recording")))
            })();
            match result {
                Ok((compare, score, all)) => {
                    good &= all;
                    if !compare.is_null() {
                        write_json(&out.join("compare.json"), &compare).unwrap();
                        summary["compare"] = json!({"equal": compare["equal"]});
                    }
                    if need("score") {
                        write_json(&out.join("score.json"), &score).unwrap();
                        summary["score"] = json!({"healthy": score["healthy"]});
                    }
                    summary["resident_seconds"] = json!(began.elapsed().as_secs_f64());
                    save(&summary);
                }
                Err(e) => {
                    summary["resident_error"] = json!(e.to_string());
                    save(&summary);
                    exit(2, &format!("FAILED in compare/score: {e}"));
                }
            }
        }

        if need("generate") || need("bench") {
            let began = Instant::now();
            let result = (|| -> Result<(Value, Value, bool)> {
                let prompts: Vec<(String, Vec<u32>)> = seqs["prompts"]
                    .as_array()
                    .ok_or_else(|| fail("no prompts"))?
                    .iter()
                    .map(|p| {
                        Ok((
                            p["name"].as_str().unwrap_or("?").to_owned(),
                            u32_list(&p["tokens"]).map_err(fail)?,
                        ))
                    })
                    .collect::<Result<_>>()?;
                let bench_prompts: Vec<Vec<u32>> = seqs["bench"]["prompts"]
                    .as_array()
                    .ok_or_else(|| fail("no throughput prompts"))?
                    .iter()
                    .map(|p| u32_list(p).map_err(fail))
                    .collect::<Result<_>>()?;
                let eos = u32_list(&seqs["eos"]).map_err(fail)?;
                let longest = prompts
                    .iter()
                    .map(|p| p.1.len())
                    .chain(bench_prompts.iter().map(Vec::len))
                    .max()
                    .unwrap_or(1);
                let vocab = n32(c.vocab_size);
                let sampleable = c.image_token_id.map_or(vocab, |i| i.min(vocab));
                let mut cfg = serving_config(&c, &serving, longest, sampleable, devices);
                cfg.gemm = GemmForm::from_env()?;
                eprintln!("{cfg:?}");
                let dry = DsCudaExecutor::layout(
                    Box::new(DryDevice::new(Some(B300_BYTES))),
                    Box::new(DryDevice::new(Some(B300_BYTES))),
                    &c,
                    Some(token_map.clone()),
                    &cfg,
                )?;
                let dry_bytes = dry.memory().map(|m| memory_json(&m));
                drop(dry);
                eprintln!("dry layout: both halves fit a B300 each: {dry_bytes:?}");
                let at = Instant::now();
                let mut x =
                    DsCudaExecutor::new(&kernels()?, &store, &c, Some(token_map.clone()), &cfg)?;
                let load = at.elapsed().as_secs_f64();
                let memory = x.memory().map(|m| memory_json(&m));
                eprintln!("the executor loaded in {load:.0} s; bytes {memory:?}");
                let cpu = var("EIDOLA_DS_TWO_CPU").map(PathBuf::from);
                let (made, agree) = if need("generate") {
                    let (mut v, agree) =
                        generate(&mut x, &prompts, &serving, &eos, cpu.as_deref())?;
                    v["load_seconds"] = json!(load);
                    v["device_bytes"] = json!(memory);
                    v["config"] = json!(format!("{cfg:?}"));
                    (v, agree)
                } else {
                    (Value::Null, true)
                };
                let b = if need("bench") {
                    let mut v = bench(&mut x, &bench_prompts, &serving)?;
                    v["config"] = json!(format!("{cfg:?}"));
                    v
                } else {
                    Value::Null
                };
                Ok((made, b, agree))
            })();
            match result {
                Ok((made, b, agree)) => {
                    good &= agree;
                    if !made.is_null() {
                        write_json(&out.join("generate.json"), &made).unwrap();
                        summary["generate"] = json!({"equal": made["equal"]});
                    }
                    if !b.is_null() {
                        write_json(&out.join("bench.json"), &b).unwrap();
                        summary["bench"] =
                            json!({"runs": b["runs"].as_array().map_or(0, Vec::len)});
                    }
                    summary["serving_seconds"] = json!(began.elapsed().as_secs_f64());
                    save(&summary);
                }
                Err(e) => {
                    summary["serving_error"] = json!(e.to_string());
                    save(&summary);
                    exit(2, &format!("FAILED in generate/bench: {e}"));
                }
            }
        }
    }
    println!("summary: {}", summary_path.display());
    if !good {
        exit(
            1,
            "FAILED: something compared differs or a health check failed (see the summary)",
        );
    }
}

#[cfg(test)]
mod tests {
    use eidola_engine_cuda::ds::host::HostDevice;

    use super::*;

    fn fixture() -> (Arc<WeightSet>, DsConfig, Vec<u32>) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../eidola-engine-model/tests/fixtures/deepseek_tiny");
        let golden = WeightSet::open_files(&[dir.join("golden.safetensors")]).unwrap();
        let name = "engram.token_map";
        let map = golden.get(name).unwrap().to_i64(name).unwrap();
        let map: Vec<u32> = map.into_iter().map(|t| u32::try_from(t).unwrap()).collect();
        let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
        let c = config_of(&store).unwrap();
        (store, c, map)
    }

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("ds_two_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The released checkpoint's chat files: its config and the pinned tokenizer.
    fn released_dir() -> PathBuf {
        use std::io::Read;
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../eidola-engine-chat/tests/fixtures/deepseek_v4_1");
        let dir = temp("released");
        std::fs::copy(fixtures.join("config.json"), dir.join("config.json")).unwrap();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(
            std::fs::File::open(fixtures.join("tokenizer.json.gz")).unwrap(),
        )
        .read_to_end(&mut json)
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), json).unwrap();
        dir
    }

    fn texts() -> Texts {
        Texts {
            system: "You are a helpful assistant.".into(),
            hello_user: "hello".into(),
            hello_reply: "Hello! How can\n can I help you today?".into(),
            hello_fork: "Hello! How can".into(),
            hello_fork_top3: vec!["\n".into(), "#".into(), "<".into()],
            story_user: "Write a short story.".into(),
            story: "The ferry ran twice a day in winter.".into(),
            passage: "The keeper of the northern light kept two logbooks.".into(),
        }
    }

    /// With the released tokenizer: the hello sequence's reply, start and near tie as the
    /// health check makes them, the passage after one beginning-of-sequence token, the
    /// node's chat prompts, distinct throughput prompts of the asked length, and a golden
    /// file `ds_halves` reads back.
    #[test]
    fn the_sequences_are_the_health_checks() {
        let dir = released_dir();
        let (v, golden) =
            tokens_phase(&dir, &texts(), &ds_both::default_prompts_file(), 300, 5).unwrap();
        let h = health_of(&v).unwrap();
        let hello = &h[0];
        let bos = u32::try_from(v["bos"].as_u64().unwrap()).unwrap();
        assert_eq!(hello.tokens[0], bos);
        let (fork, reported) = hello.fork.clone().unwrap();
        assert_eq!(reported.len(), 3);
        assert_eq!(hello.tokens[fork + 1], reported[0]);
        // "Hello! How can" ends right before the fork's prediction.
        assert!(fork >= hello.start && fork < hello.tokens.len() - 1);
        assert_eq!(h[2].tokens[0], bos);
        assert_eq!(h[2].start, 1);
        assert_eq!(v["prompts"].as_array().unwrap().len(), 4);
        let bench = v["bench"]["prompts"].as_array().unwrap();
        assert_eq!(bench.len(), 5);
        let bench: Vec<Vec<u32>> = bench.iter().map(|p| u32_list(p).unwrap()).collect();
        assert!(bench.iter().all(|p| p.len() == 300 && p[0] == bos));
        assert_ne!(bench[0], bench[1]);

        let path = dir.join("passage.safetensors");
        std::fs::write(&path, golden).unwrap();
        let g = WeightSet::open_files(&[path]).unwrap();
        let tokens: Vec<u32> = g
            .get("tokens")
            .unwrap()
            .to_i64("tokens")
            .unwrap()
            .into_iter()
            .map(|t| u32::try_from(t).unwrap())
            .collect();
        assert_eq!(tokens, h[2].tokens);
        let map = g
            .get("engram.token_map")
            .unwrap()
            .to_i64("engram.token_map")
            .unwrap();
        assert_eq!(map.len(), 129_280);
    }

    #[test]
    fn scoring_counts_the_reply_and_the_near_tie() {
        let vocab = 4;
        // Tokens 0 1 2 3 1; the reply starts at 2; the tie is position 2 (predicting 3).
        let tokens = vec![0, 1, 2, 3, 1];
        let row = |best: usize| {
            let mut r = vec![0.0f32; vocab];
            r[best] = 5.0;
            r
        };
        let logits: Vec<f32> = [row(1), row(2), row(0), row(1), row(0)].concat();
        let h = Health {
            name: "hello".into(),
            tokens: tokens.clone(),
            start: 2,
            fork: Some((2, vec![3, 0, 2])),
        };
        // Positions 1, 2, 3 are scored: 2 right, the tie wrong but among the reported.
        let (v, healthy) = score_one(&h, &logits, vocab);
        assert!(healthy, "{v}");
        assert_eq!(v["scored"], 3);
        assert!((v["accuracy"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-12);
        let h = Health {
            fork: Some((2, vec![3, 1, 2])),
            ..h
        };
        assert!(!score_one(&h, &logits, vocab).1);
        // Story: accuracy under the floor.
        let h = Health {
            name: "story".into(),
            tokens,
            start: 1,
            fork: None,
        };
        let (v, healthy) = score_one(&h, &logits, vocab);
        assert_eq!(v["scored"], 4);
        assert_eq!(healthy, v["accuracy"].as_f64().unwrap() >= STORY_FLOOR);
        let top = top_logprobs(&row(2), 2);
        assert_eq!(top[0].0, 2);
        assert!(top[0].1 < 0.0 && top[0].1 > top[1].1);
    }

    /// Both halves resident on host devices: a recording of the text is compared equal
    /// with itself made earlier, the passage in a slot of its own equals the comparison's
    /// logits, and the scores are written; a flipped recorded bit is found.
    #[test]
    fn the_fixture_resident_equals_its_recording_and_scores() {
        let (store, c, map) = fixture();
        let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37 + 11) % 95).collect();
        let base = temp("resident");
        let open = |_: usize| -> Result<Box<dyn Device>> { Ok(Box::new(HostDevice)) };
        let health = vec![
            Health {
                name: "hello".into(),
                tokens: tokens[..20].to_vec(),
                start: 10,
                fork: Some((12, vec![1, 2, 3])),
            },
            Health {
                name: "passage".into(),
                tokens: tokens.clone(),
                start: 1,
                fork: None,
            },
        ];
        let l = Layout::new(Some(store.clone()), c.clone(), map.clone(), 70, 3, 16).unwrap();
        // The first run records.
        let first = Resident {
            recorded: None,
            health: Vec::new(),
        };
        let r = l.seq(&tokens, 0);
        let enc_dev = open(0).unwrap();
        let w = l.resident(Half::Encoder, enc_dev.as_ref()).unwrap();
        let mut enc = l.net(Half::Encoder, enc_dev, w).unwrap();
        let dec_dev = open(1).unwrap();
        let w = l.resident(Half::Decoder, dec_dev.as_ref()).unwrap();
        let mut dec = l.net(Half::Decoder, dec_dev, w).unwrap();
        r.run(&mut enc, &mut dec, &Unwatched)
            .unwrap()
            .write(&base.join("recorded"))
            .unwrap();
        drop((enc, dec, first));

        let run = Resident {
            recorded: Some((tokens.clone(), base.join("recorded"))),
            health,
        };
        let (compare, score, _) = resident(&l, &run, &open, Some(&base.join("again"))).unwrap();
        assert_eq!(compare["equal"], true, "{compare:#}");
        assert_eq!(compare["checks"].as_array().unwrap().len(), 4 + 3 + 4 + 1);
        let passage = &score["sequences"]["passage"];
        assert_eq!(passage["equals_the_comparisons_logits"], true);
        assert_eq!(passage["scored"], 69);
        assert_eq!(score["sequences"]["hello"]["scored"], 10);

        let path = base.join("recorded/logits.bin");
        let mut logits = std::fs::read(&path).unwrap();
        logits[4 * (5 * c.vocab_size + 7)] ^= 1;
        std::fs::write(&path, logits).unwrap();
        let (compare, _, all) = resident(&l, &run, &open, None).unwrap();
        assert!(!all);
        let bad: Vec<&Value> = compare["checks"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["equal"] == json!(false))
            .collect();
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0]["what"], "logits");
        assert_eq!(bad[0]["first"]["position"], 5);
        assert_eq!(bad[0]["first"]["element"], 7);
    }

    /// The executor on host devices under the engine: the same tokens at every width, as
    /// the CPU recording says (a planted difference is found), and the throughput runs.
    #[test]
    fn the_fixture_generates_and_benches() {
        let (store, c, map) = fixture();
        let s = Serving {
            widths: vec![0, 2, 4],
            max_new: 12,
            bench_widths: vec![0, 4],
            concurrency: vec![1, 3],
            bench_out: 6,
            chunk: 16,
        };
        let prompts = vec![
            (
                "a".to_string(),
                (0..40u32).map(|i| (i * 7 + 3) % 95).collect::<Vec<_>>(),
            ),
            (
                "b".to_string(),
                (0..9u32).map(|i| (i * 11 + 5) % 95).collect(),
            ),
        ];
        let bench_prompts: Vec<Vec<u32>> = (0..3u32)
            .map(|k| (0..20u32).map(|i| (i * 13 + k * 5 + 1) % 95).collect())
            .collect();
        let cfg = serving_config(&c, &s, 40, 95, [0, 0]);
        let mut x = DsCudaExecutor::with_devices(
            Box::new(HostDevice),
            Box::new(HostDevice),
            &store,
            &c,
            Some(map),
            &cfg,
        )
        .unwrap();
        let (v, agree) = generate(&mut x, &prompts, &s, &[], None).unwrap();
        assert!(agree, "{v:#}");
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 6);
        let a0 = u32_list(&results[0]["token_ids"]).unwrap();
        assert_eq!(a0.len(), 12);
        assert!(results.iter().any(|r| r["drafted"].as_u64().unwrap() > 0));

        // A CPU result agreeing for `a`, and one differing in its last token for `b`.
        let cpu = temp("cpu");
        std::fs::create_dir_all(cpu.join("requests")).unwrap();
        let b0 = u32_list(&results[1]["token_ids"]).unwrap();
        let mut planted = b0.clone();
        *planted.last_mut().unwrap() ^= 1;
        for (name, prompt, tokens) in [("a", &prompts[0].1, a0), ("b", &prompts[1].1, planted)] {
            let r = json!({"prompt_token_ids": prompt, "token_ids": tokens, "max_tokens": 12});
            write_json(&cpu.join(format!("requests/{name}.w5.json")), &r).unwrap();
        }
        let (v, agree) = generate(&mut x, &prompts, &s, &[], Some(&cpu)).unwrap();
        assert!(!agree);
        assert_eq!(v["checks"][0]["cpu"]["equal"], true, "{v:#}");
        assert_eq!(v["checks"][1]["cpu"]["equal"], false);
        assert_eq!(v["checks"][1]["cpu"]["common_prefix"], 11);

        let b = bench(&mut x, &bench_prompts, &s).unwrap();
        let runs = b["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 4);
        for r in runs {
            let n = r["requests"].as_u64().unwrap();
            assert_eq!(r["output_tokens"].as_u64().unwrap(), n * 6, "{r:#}");
            assert!(r["mean_ttft_seconds"].as_f64().unwrap() > 0.0);
        }
        assert!(
            runs.iter()
                .any(|r| r["width"] == 4 && r["drafted"].as_u64().unwrap() > 0)
        );
    }

    /// The released configuration as the session runs it, laid out with no checkpoint on
    /// two devices of one B300's memory: the resident halves for the comparison and the
    /// scores, and the executor for generation and the throughput runs, each fit.
    #[test]
    fn the_session_s_configurations_fit_two_b300s() {
        let c = DsConfig::from_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../eidola-engine-chat/tests/fixtures/deepseek_v4_1/config.json"),
        )
        .unwrap();
        let e = c.engram.clone().unwrap();
        let map: Vec<u32> = (0..c.vocab_size)
            .map(|i| n32(i % e.compressed_vocab))
            .collect();
        let open = |_: usize| -> Result<Box<dyn Device>> {
            Ok(Box::new(DryDevice::new(Some(B300_BYTES))))
        };
        let token = |i: u32| (i * 7919 + 13) % 129_000;
        let health = vec![
            Health {
                name: "story".into(),
                tokens: (0..900).map(token).collect(),
                start: 30,
                fork: None,
            },
            Health {
                name: "passage".into(),
                tokens: (0..555).map(token).collect(),
                start: 1,
                fork: None,
            },
        ];
        let l = Layout::new(None, c.clone(), map.clone(), 900, 4, 96).unwrap();
        let r = Resident {
            recorded: None,
            health,
        };
        let (_, score, _) = resident(&l, &r, &open, None).unwrap();
        for half in 0..2 {
            let total = score["device_bytes"][half]["total"].as_u64().unwrap();
            assert!(total < B300_BYTES as u64, "{score:#}");
        }
        let s = Serving {
            widths: vec![0, 5],
            max_new: 96,
            bench_widths: vec![0, 5],
            concurrency: vec![1, 8, 32],
            bench_out: 256,
            chunk: 2048,
        };
        let cfg = serving_config(&c, &s, 1024, 129_264, [0, 1]);
        let x = DsCudaExecutor::layout(open(0).unwrap(), open(1).unwrap(), &c, Some(map), &cfg)
            .unwrap();
        let [enc, dec] = x.memory();
        eprintln!("{enc:?}\n{dec:?}");
        assert!(enc.total() < B300_BYTES as u64 && dec.total() < B300_BYTES as u64);
        // The decoder holds its Engram table and the drafter.
        assert!(dec.weights > 250_527_512_880);
    }
}
