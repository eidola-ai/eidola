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
//! - `bench`: the fast profile through the engine: decode at concurrency 1, 8, 32 and 64
//!   with draft widths 0 and 3 over 1,024-token prompts, and a prefill alone at 1k and 8k
//!   tokens: wall time per step, output tokens per second, time to the first token.
//!
//! `EIDOLA_QWEN_STRATEGY` (`live`, the default, or `lagging`) chooses the state strategy.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
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

fn config(k: u32, strategy: StateStrategy) -> QwenCudaExecutorConfig {
    QwenCudaExecutorConfig {
        block_size: 16,
        num_blocks: 8192,
        host_blocks: 0,
        state_buffers: 200,
        host_state_buffers: 0,
        num_state_slots: 64,
        max_model_len: 16384,
        buckets: vec![Bucket {
            max_seqs: 64,
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
            max_seqs: 64,
            max_prefill_chunk: 2048,
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
            for k in [0u32, 3] {
                for c in [1usize, 8, 32, 64] {
                    let ex = executor(&kdir, &model, QwenProfile::Fast, k);
                    let mut eng = engine(ex, k > 0);
                    let reqs: Vec<Request> = (0..c)
                        .map(|i| Request {
                            id: i as u64,
                            prompt: tokens(1024),
                            sampling: SamplingParams::greedy(),
                            max_tokens: 64,
                            required_cached_tokens: 0,
                            stop_token_ids: Vec::new(),
                            cache: CacheScope::Private,
                        })
                        .collect();
                    let t = Instant::now();
                    let (o, steps, first) = drive(&mut eng, reqs);
                    let wall = t.elapsed().as_secs_f64();
                    let produced: usize = o.values().map(Vec::len).sum();
                    let decode: Vec<f64> = steps.iter().filter(|s| s.1 > 0).map(|s| s.0).collect();
                    let mut ttft: Vec<f64> = first.values().copied().collect();
                    ttft.sort_by(f64::total_cmp);
                    let r = json!({
                        "k": k, "concurrency": c, "wall_s": wall, "output_tokens": produced,
                        "output_tok_per_s": produced as f64 / wall,
                        "median_step_ms": 1000.0 * median(&decode),
                        "ttft_median_s": median(&ttft),
                    });
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
