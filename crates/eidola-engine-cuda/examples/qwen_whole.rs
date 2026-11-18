//! The released Qwen3.8 checkpoint on one GPU (dev-only): the GPU session's real-weights
//! phases.
//!
//!     qwen_whole <phase> <kernel dir> <checkpoint dir> <out dir>
//!
//! The checkpoint directory holds the hash-verified files and `golden/health.safetensors`
//! (the token sequences `eidola-engine-model`'s `golden/qwen/health_check.py --tokens-only`
//! writes). Phases, each writing `<out>/<phase>.json` and failing loudly:
//!
//! - `compare`: the goldens' passage through the model crate's forward on the CPU, then
//!   through the executor's exact profile (every logit equal bit for bit, required) and its
//!   fast profile (top-1 agreement, units in the last place, KL, recorded);
//! - `health`: both profiles: next-token accuracy over the licence text (floor 0.95) and
//!   the greedy reply to the capital question (it must contain the answer's tokens);
//! - `generate`: the fast profile through the engine, greedily, at draft widths 0 to 3:
//!   the same tokens at every width (required);
//! - `bench`: the fast profile through the engine, one engine per draft width (0 and 3):
//!   the yardstick workloads of `scripts/qwen_workloads.json`, which `scripts/qwen_vllm.py`
//!   runs against vLLM the same way (B: random-id prompts, greedy, `2c` requests and at
//!   least 4 at concurrency `c`; C: conversations whose every turn is the last turn's
//!   prompt, its reply and new tokens, sampled, one a lane and at least 2), at most `c`
//!   requests in flight, each submitted when a lane frees as a client's would be; then a
//!   prefill alone at 1k and 8k tokens. Per run: output and input tokens per second, the
//!   median time to the first token (from submission) and per-stream decode rate, and
//!   the share of prompt tokens served from the prefix cache.
//!
//! `EIDOLA_QWEN_STRATEGY` (`live`, the default, or `lagging`) chooses the state strategy.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use eidola_engine_cuda::qwen::{
    CudaGraphs, QwenCudaExecutor, QwenCudaExecutorConfig, QwenProfile, StateStrategy,
};
use eidola_engine_model::WeightSet;
use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions, QwenModel, QwenWeights, config_of};
use serde_json::json;

fn fail(what: impl std::fmt::Display) -> ! {
    eprintln!("FAILED: {what}");
    std::process::exit(1)
}

fn ids(set: &WeightSet, name: &str) -> Vec<u32> {
    set.get(name)
        .unwrap_or_else(|e| fail(format!("{name}: {e}")))
        .to_i64(name)
        .unwrap_or_else(|e| fail(format!("{name}: {e}")))
        .into_iter()
        .map(|t| u32::try_from(t).unwrap_or_else(|_| fail("a token id")))
        .collect()
}

/// Sequences in flight at most: the workloads' largest concurrency.
const MAX_SEQS: u32 = 128;

/// The yardstick workloads, which `scripts/qwen_vllm.py` reads too.
const WORKLOADS: &str = include_str!("../scripts/qwen_workloads.json");

fn config(k: u32, strategy: StateStrategy) -> QwenCudaExecutorConfig {
    QwenCudaExecutorConfig {
        block_size: 16,
        num_blocks: 16384,
        host_blocks: 0,
        state_buffers: 200,
        host_state_buffers: 0,
        num_state_slots: MAX_SEQS,
        max_model_len: 16384,
        buckets: vec![Bucket {
            max_seqs: MAX_SEQS,
            max_tokens: 2048,
        }],
        draft_tokens: k,
        sampleable_vocab_size: 248_077,
        strategy,
        graphs: match std::env::var("EIDOLA_QWEN_GRAPHS").as_deref() {
            Ok("on") => CudaGraphs::On,
            Ok("off") | Err(_) => CudaGraphs::Off,
            Ok(_) => fail("EIDOLA_QWEN_GRAPHS is on or off"),
        },
    }
}

fn executor(
    dir: &Path,
    model: &QwenModel,
    profile: QwenProfile,
    k: u32,
) -> QwenCudaExecutor<CudaDevice> {
    let strategy = std::env::var("EIDOLA_QWEN_STRATEGY")
        .ok()
        .map(|s| {
            StateStrategy::parse(&s)
                .unwrap_or_else(|| fail("EIDOLA_QWEN_STRATEGY is live or lagging"))
        })
        .unwrap_or_default();
    let t = Instant::now();
    let dev = CudaDevice::open(0, &KernelDir::new(dir), profile).unwrap_or_else(|e| fail(e));
    let ex = QwenCudaExecutor::new(dev, model, config(k, strategy)).unwrap_or_else(|e| fail(e));
    eprintln!(
        "{} executor loaded in {:.0} s",
        profile.name(),
        t.elapsed().as_secs_f64()
    );
    ex
}

fn argmax(row: &[f32], n: usize) -> usize {
    (0..n)
        .max_by(|&i, &j| row[i].total_cmp(&row[j]).then(j.cmp(&i)))
        .unwrap_or(0)
}

fn accuracy(logits: &[Vec<f32>], tokens: &[u32], n: usize) -> f64 {
    let hits = (0..tokens.len() - 1)
        .filter(|&q| argmax(&logits[q], n) == tokens[q + 1] as usize)
        .count();
    hits as f64 / (tokens.len() - 1) as f64
}

fn ord16(x: f32) -> i32 {
    let b = i32::try_from(x.to_bits() >> 16).unwrap_or(0);
    if b & 0x8000 != 0 { -(b & 0x7fff) } else { b }
}

/// Fast against exact: top-1 agreement, BF16 ulp histogram, largest difference, mean KL.
fn deviation(a: &[Vec<f32>], b: &[Vec<f32>], n: usize) -> serde_json::Value {
    let mut hist = [0u64; 6];
    let (mut top, mut kl, mut worst) = (0usize, 0f64, 0f32);
    for (x, y) in a.iter().zip(b) {
        if argmax(x, n) == argmax(y, n) {
            top += 1;
        }
        let lse = |v: &[f32]| {
            let m = v[..n].iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
            m + v[..n]
                .iter()
                .map(|&x| (x as f64 - m).exp())
                .sum::<f64>()
                .ln()
        };
        let (la, lb) = (lse(x), lse(y));
        kl += (0..n)
            .map(|i| {
                let pb = (y[i] as f64 - lb).exp();
                pb * ((y[i] as f64 - lb) - (x[i] as f64 - la))
            })
            .sum::<f64>();
        for i in 0..n {
            let d = (ord16(x[i]) - ord16(y[i])).unsigned_abs() as usize;
            hist[d.min(5)] += 1;
            worst = worst.max((x[i] - y[i]).abs());
        }
    }
    json!({
        "positions": a.len(),
        "top1_agree": top,
        "kl_mean": kl / a.len() as f64,
        "ulp_hist_0_to_5plus": hist,
        "largest_abs_diff": worst,
    })
}

fn engine(
    ex: QwenCudaExecutor<CudaDevice>,
    speculative: bool,
) -> Engine<QwenCudaExecutor<CudaDevice>> {
    Engine::new(
        ex,
        SchedulerConfig {
            max_batched_tokens: 2048,
            max_seqs: MAX_SEQS,
            max_prefill_chunk: eidola_engine_cuda::qwen::SERVING_PREFILL_CHUNK,
            eos_token_ids: Vec::new(),
            speculative,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        },
    )
    .unwrap_or_else(|e| fail(format!("{e:?}")))
}

/// Outputs by id, per step its wall time and tokens, and each request's first token time.
type Driven = (HashMap<u64, Vec<u32>>, Vec<(f64, usize)>, HashMap<u64, f64>);

/// Runs every request to its end; outputs by id, and per step its wall time and tokens.
fn drive(eng: &mut Engine<QwenCudaExecutor<CudaDevice>>, reqs: Vec<Request>) -> Driven {
    let mut out: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut first: HashMap<u64, f64> = HashMap::new();
    let t0 = Instant::now();
    for r in reqs {
        out.insert(r.id, Vec::new());
        eng.submit(r).unwrap_or_else(|e| fail(format!("{e:?}")));
    }
    let mut steps = Vec::new();
    let mut now = 0;
    while eng.unfinished() > 0 {
        let t = Instant::now();
        let ev = eng.step(now).unwrap_or_else(|e| fail(e.0));
        let n: usize = ev.iter().map(|e| e.tokens.len()).sum();
        steps.push((t.elapsed().as_secs_f64(), n));
        for e in ev {
            if !e.tokens.is_empty() {
                first
                    .entry(e.id)
                    .or_insert_with(|| t0.elapsed().as_secs_f64());
            }
            out.get_mut(&e.id).expect("a request").extend(e.tokens);
        }
        now += 1;
    }
    eng.flush().unwrap_or_else(|e| fail(e.0));
    (out, steps, first)
}

/// One conversation of a workload: its first prompt, then each later turn's new tokens.
struct Conversation {
    first: Vec<u32>,
    later: Vec<Vec<u32>>,
}

/// A request served under a workload.
struct Served {
    ttft: f64,
    decode: Option<f64>,
    prompt: usize,
    cached: u32,
    tokens: usize,
}

/// An engine kept across a draft width's runs, as one server is.
struct Bench {
    eng: Engine<QwenCudaExecutor<CudaDevice>>,
    clock: Instant,
    next_id: u64,
    next_key: u64,
}

struct Active {
    conv: usize,
    turn: usize,
    prompt: Vec<u32>,
    reply: Vec<u32>,
    submitted: f64,
    first: Option<f64>,
    last: f64,
}

impl Bench {
    fn new(eng: Engine<QwenCudaExecutor<CudaDevice>>) -> Self {
        Bench {
            eng,
            clock: Instant::now(),
            next_id: 0,
            next_key: 0,
        }
    }

    fn now(&self) -> f64 {
        self.clock.elapsed().as_secs_f64()
    }

    /// Runs `convs` with at most `c` in flight, as a client would: a conversation's next
    /// turn (its last prompt, the reply and the turn's new tokens) is submitted when its
    /// last turn finishes, and the next conversation when one ends. Each conversation is a
    /// cache lineage of its own when `keyed`. Every request must produce `out` tokens.
    fn converse(
        &mut self,
        convs: &[Conversation],
        c: usize,
        out: u32,
        sampling: &dyn Fn(usize, usize) -> SamplingParams,
        keyed: bool,
    ) -> (Vec<Served>, f64) {
        let deriver = SaltDeriver::with_boot_key(BootKey::from_bytes([7; 32]));
        let lineages: Vec<Option<Lineage>> = convs
            .iter()
            .map(|_| {
                self.next_key += 1;
                let mut key = [0u8; 32];
                key[..8].copy_from_slice(&self.next_key.to_le_bytes());
                keyed.then(|| deriver.lineage(&CacheKey::from_bytes(key)))
            })
            .collect();
        let mut active: HashMap<u64, Active> = HashMap::new();
        let mut served = Vec::new();
        let began = self.now();
        let start = |b: &mut Bench,
                     active: &mut HashMap<u64, Active>,
                     conv: usize,
                     turn: usize,
                     prompt: Vec<u32>| {
            b.next_id += 1;
            let id = b.next_id;
            b.eng
                .submit(Request {
                    id,
                    prompt: prompt.clone(),
                    sampling: sampling(conv, turn),
                    max_tokens: out,
                    required_cached_tokens: 0,
                    stop_token_ids: Vec::new(),
                    cache: lineages[conv]
                        .clone()
                        .map_or(CacheScope::Private, CacheScope::Keyed),
                })
                .unwrap_or_else(|e| fail(format!("{e:?}")));
            let submitted = b.now();
            active.insert(
                id,
                Active {
                    conv,
                    turn,
                    prompt,
                    reply: Vec::new(),
                    submitted,
                    first: None,
                    last: submitted,
                },
            );
        };
        let mut next = 0;
        while next < convs.len().min(c) {
            start(self, &mut active, next, 0, convs[next].first.clone());
            next += 1;
        }
        while self.eng.unfinished() > 0 {
            let now = u64::try_from(self.clock.elapsed().as_millis()).unwrap_or(u64::MAX);
            let ev = self.eng.step(now).unwrap_or_else(|e| fail(e.0));
            let at = self.now();
            for e in ev {
                let a = active.get_mut(&e.id).expect("a request");
                if !e.tokens.is_empty() {
                    a.first.get_or_insert(at);
                    a.last = at;
                }
                a.reply.extend(e.tokens);
                if e.finish.is_none() {
                    continue;
                }
                let a = active.remove(&e.id).expect("a request");
                if a.reply.len() != out as usize {
                    fail(format!(
                        "a request produced {} of {out} tokens",
                        a.reply.len()
                    ));
                }
                let first = a.first.unwrap_or(a.last);
                served.push(Served {
                    ttft: first - a.submitted,
                    decode: (a.last > first).then(|| (out - 1) as f64 / (a.last - first)),
                    prompt: a.prompt.len(),
                    cached: e.cached_prompt_tokens,
                    tokens: a.reply.len(),
                });
                let conv = &convs[a.conv];
                if a.turn < conv.later.len() {
                    let mut prompt = a.prompt;
                    prompt.extend(&a.reply);
                    prompt.extend(&conv.later[a.turn]);
                    start(self, &mut active, a.conv, a.turn + 1, prompt);
                } else if next < convs.len() {
                    start(self, &mut active, next, 0, convs[next].first.clone());
                    next += 1;
                }
            }
        }
        self.eng.flush().unwrap_or_else(|e| fail(e.0));
        (served, self.now() - began)
    }
}

/// One run's row: the fields `scripts/qwen_vllm.py`'s bench writes for vLLM's.
fn workload_row(
    workload: &str,
    k: u32,
    c: usize,
    served: &[Served],
    wall: f64,
) -> serde_json::Value {
    let tokens: usize = served.iter().map(|r| r.tokens).sum();
    let inputs: usize = served.iter().map(|r| r.prompt).sum();
    let cached: u64 = served.iter().map(|r| u64::from(r.cached)).sum();
    let ttft: Vec<f64> = served.iter().map(|r| r.ttft).collect();
    let decode: Vec<f64> = served.iter().filter_map(|r| r.decode).collect();
    json!({
        "workload": workload, "k": k, "concurrency": c, "requests": served.len(),
        "wall_s": wall, "output_tokens": tokens,
        "output_tok_per_s": tokens as f64 / wall,
        "input_tok_per_s": inputs as f64 / wall,
        "mean_input_len": inputs as f64 / served.len() as f64,
        "ttft_median_s": median(&ttft),
        "decode_median": median(&decode),
        "prefix_hit_fraction": cached as f64 / inputs as f64,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        fail(
            "usage: qwen_whole <compare|health|generate|bench> <kernel dir> <checkpoint dir> <out dir>",
        );
    }
    let (phase, kdir, ckpt, out) = (
        &args[1],
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
        PathBuf::from(&args[4]),
    );
    std::fs::create_dir_all(&out).unwrap_or_else(|e| fail(e));
    let health = WeightSet::open_files(&[ckpt.join("golden/health.safetensors")])
        .unwrap_or_else(|e| fail(e));
    let t = Instant::now();
    let store = Arc::new(WeightSet::open_dir(&ckpt).unwrap_or_else(|e| fail(e)));
    let cfg = config_of(&store).unwrap_or_else(|e| fail(e));
    let model = QwenModel::new(QwenWeights::load(store, cfg).unwrap_or_else(|e| fail(e)));
    eprintln!("checkpoint opened in {:.0} s", t.elapsed().as_secs_f64());
    let sampleable = 248_077usize;
    let report = match phase.as_str() {
        "compare" => {
            let tokens = ids(&health, "passage.tokens");
            let t = Instant::now();
            let all = QwenForwardOptions {
                logits: LogitRows::All,
                keep_streams: false,
            };
            let reference = model
                .forward(&mut model.new_state(), &tokens, &all)
                .unwrap_or_else(|e| fail(e));
            let cpu_s = t.elapsed().as_secs_f64();
            let want: Vec<Vec<f32>> = (0..tokens.len())
                .map(|q| reference.logits.row(q).to_vec())
                .collect();
            let mut ex = executor(&kdir, &model, QwenProfile::Exact, 0);
            let t = Instant::now();
            let exact = ex.dense_logits(0, &tokens).unwrap_or_else(|e| fail(e));
            let exact_s = t.elapsed().as_secs_f64();
            drop(ex);
            let differ = exact
                .iter()
                .zip(&want)
                .filter(|(a, b)| {
                    a.iter()
                        .zip(b.iter())
                        .any(|(x, y)| x.to_bits() != y.to_bits())
                })
                .count();
            let mut ex = executor(&kdir, &model, QwenProfile::Fast, 0);
            let t = Instant::now();
            let fast = ex.dense_logits(0, &tokens).unwrap_or_else(|e| fail(e));
            let fast_s = t.elapsed().as_secs_f64();
            let r = json!({
                "tokens": tokens.len(),
                "cpu_seconds": cpu_s,
                "exact_seconds": exact_s,
                "fast_seconds": fast_s,
                "exact_positions_differing": differ,
                "fast_against_exact": deviation(&fast, &exact, sampleable),
            });
            if differ != 0 {
                std::fs::write(out.join("compare.json"), r.to_string()).ok();
                fail(format!(
                    "the exact profile differs from the model crate at {differ} positions"
                ));
            }
            r
        }
        "health" => {
            let lic = ids(&health, "license.tokens");
            let prompt = ids(&health, "capital.prompt");
            let stops = ids(&health, "capital.stop");
            let answer = ids(&health, "capital.answer");
            let mut res = serde_json::Map::new();
            for profile in [QwenProfile::Exact, QwenProfile::Fast] {
                let mut ex = executor(&kdir, &model, profile, 0);
                let acc = accuracy(
                    &ex.dense_logits(0, &lic).unwrap_or_else(|e| fail(e)),
                    &lic,
                    sampleable,
                );
                let mut eng = engine(ex, false);
                let mut req = Request {
                    id: 1,
                    prompt: prompt.clone(),
                    sampling: SamplingParams::greedy(),
                    max_tokens: 16,
                    required_cached_tokens: 0,
                    stop_token_ids: stops.clone(),
                    cache: CacheScope::Private,
                };
                req.stop_token_ids.sort_unstable();
                let (o, _, _) = drive(&mut eng, vec![req]);
                let reply = o[&1].clone();
                let paris = reply.windows(answer.len()).any(|w| w == answer.as_slice());
                eprintln!(
                    "{}: licence accuracy {acc:.3}, reply {reply:?}",
                    profile.name()
                );
                if acc < 0.95 || !paris {
                    fail(format!(
                        "{} profile unhealthy: accuracy {acc:.3}, reply {reply:?}",
                        profile.name()
                    ));
                }
                res.insert(
                    profile.name().into(),
                    json!({"licence_accuracy": acc, "reply": reply}),
                );
            }
            serde_json::Value::Object(res)
        }
        "generate" => {
            let prompt = ids(&health, "capital.prompt");
            let passage = ids(&health, "passage.tokens");
            let mut first: Option<HashMap<u64, Vec<u32>>> = None;
            let mut res = serde_json::Map::new();
            for k in 0..=3u32 {
                let ex = executor(&kdir, &model, QwenProfile::Fast, k);
                let mut eng = engine(ex, k > 0);
                let reqs = vec![
                    Request {
                        id: 1,
                        prompt: prompt.clone(),
                        sampling: SamplingParams::greedy(),
                        max_tokens: 48,
                        required_cached_tokens: 0,
                        stop_token_ids: Vec::new(),
                        cache: CacheScope::Private,
                    },
                    Request {
                        id: 2,
                        prompt: passage[..200].to_vec(),
                        sampling: SamplingParams::greedy(),
                        max_tokens: 96,
                        required_cached_tokens: 0,
                        stop_token_ids: Vec::new(),
                        cache: CacheScope::Private,
                    },
                ];
                let (o, steps, _) = drive(&mut eng, reqs);
                if let Some(f) = &first {
                    if f != &o {
                        fail(format!(
                            "greedy tokens at draft width {k} differ from width 0"
                        ));
                    }
                } else {
                    first = Some(o.clone());
                }
                res.insert(format!("k{k}"), json!({"steps": steps.len(), "outputs": o}));
            }
            serde_json::Value::Object(res)
        }
        "bench" => {
            let mut rows = Vec::new();
            let mut seed = 1u64;
            let mut tokens = |n: usize| -> Vec<u32> {
                (0..n)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        u32::try_from((seed >> 33) % 240_000).unwrap_or(0)
                    })
                    .collect()
            };
            let w: serde_json::Value = serde_json::from_str(WORKLOADS).unwrap_or_else(|e| fail(e));
            let get = |path: &[&str]| -> &serde_json::Value {
                let v = path.iter().fold(&w, |v, k| &v[*k]);
                if !v.is_number() {
                    fail(format!("qwen_workloads.json: no number at {path:?}"));
                }
                v
            };
            let count = |path: &[&str]| -> usize {
                get(path)
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or_else(|| fail(format!("qwen_workloads.json: {path:?} is a count")))
            };
            // Sampling parameters, which f32 holds as the file writes them.
            #[allow(clippy::cast_possible_truncation)]
            let real = |path: &[&str]| -> f32 { get(path).as_f64().unwrap_or(f64::NAN) as f32 };
            let lanes = |workload: &str| -> Vec<usize> {
                w[workload]["concurrency"]
                    .as_array()
                    .unwrap_or_else(|| fail("qwen_workloads.json: no concurrency"))
                    .iter()
                    .map(|c| {
                        c.as_u64()
                            .filter(|&c| (1..=u64::from(MAX_SEQS)).contains(&c))
                            .and_then(|c| usize::try_from(c).ok())
                            .unwrap_or_else(|| fail(format!("a concurrency of 1 to {MAX_SEQS}")))
                    })
                    .collect()
            };
            let (b_in, b_out) = (
                count(&["B", "prompt_tokens"]),
                count(&["B", "output_tokens"]),
            );
            let b_temperature = real(&["B", "temperature"]);
            let (turns, step) = (count(&["C", "turns"]), count(&["C", "turn_tokens"]));
            let c_out = count(&["C", "output_tokens"]);
            let c_sampling = (
                real(&["C", "temperature"]),
                u32::try_from(count(&["C", "top_k"])).unwrap_or(0),
                real(&["C", "top_p"]),
            );
            if turns == 0 || step <= c_out {
                fail("qwen_workloads.json: a conversation's turns must each add more than a reply");
            }
            let out_of = |n: usize| u32::try_from(n).unwrap_or_else(|_| fail("an output length"));
            for k in [0u32, 3] {
                let ex = executor(&kdir, &model, QwenProfile::Fast, k);
                let mut bench = Bench::new(engine(ex, k > 0));
                for c in lanes("B") {
                    let n = count(&["B", "requests_at_least"])
                        .max(count(&["B", "requests_per_concurrent"]) * c);
                    let convs: Vec<Conversation> = (0..n)
                        .map(|_| Conversation {
                            first: tokens(b_in),
                            later: Vec::new(),
                        })
                        .collect();
                    let greedy = |_: usize, _: usize| {
                        SamplingParams::new(b_temperature, 0, 1.0, 0.0, 0)
                            .unwrap_or_else(|e| fail(format!("{e:?}")))
                    };
                    let (served, wall) = bench.converse(&convs, c, out_of(b_out), &greedy, false);
                    let r = workload_row("B", k, c, &served, wall);
                    eprintln!("{r}");
                    rows.push(r);
                }
                for c in lanes("C") {
                    let n = count(&["C", "conversations_at_least"])
                        .max(count(&["C", "conversations_per_concurrent"]) * c);
                    let convs: Vec<Conversation> = (0..n)
                        .map(|_| Conversation {
                            first: tokens(step),
                            later: (1..turns).map(|_| tokens(step - c_out)).collect(),
                        })
                        .collect();
                    let sampled = |conv: usize, turn: usize| {
                        let seed = (7919 * c + 31 * conv + turn) as u64;
                        SamplingParams::new(c_sampling.0, c_sampling.1, c_sampling.2, 0.0, seed)
                            .unwrap_or_else(|e| fail(format!("{e:?}")))
                    };
                    let (served, wall) = bench.converse(&convs, c, out_of(c_out), &sampled, true);
                    let mut r = workload_row("C", k, c, &served, wall);
                    r["conversations"] = json!(n);
                    eprintln!("{r}");
                    rows.push(r);
                }
            }
            for n in [1024usize, 8192] {
                let ex = executor(&kdir, &model, QwenProfile::Fast, 0);
                let mut eng = engine(ex, false);
                let req = Request {
                    id: 0,
                    prompt: tokens(n),
                    sampling: SamplingParams::greedy(),
                    max_tokens: 1,
                    required_cached_tokens: 0,
                    stop_token_ids: Vec::new(),
                    cache: CacheScope::Private,
                };
                let t = Instant::now();
                drive(&mut eng, vec![req]);
                let r = json!({"prefill_tokens": n, "ttft_s": t.elapsed().as_secs_f64()});
                eprintln!("{r}");
                rows.push(r);
            }
            json!(rows)
        }
        other => fail(format!("unknown phase {other}")),
    };
    let path = out.join(format!("{phase}.json"));
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    )
    .unwrap_or_else(|e| fail(e));
    println!("{phase}: ok, {}", path.display());
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}
