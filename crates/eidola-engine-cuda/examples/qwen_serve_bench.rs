//! The released Qwen3.8 checkpoint served through the engine on one GPU (dev-only): the
//! throughput and latency workloads a reference serving stack is measured on.
//!
//!     qwen_serve_bench <kernel dir> <checkpoint dir> <out.jsonl> <B|C>
//!         [--concurrency 1,8,32,64,128] [--draft K] [--profile fast|exact]
//!         [--graphs on|off] [--prompts prompts.json]
//!
//! - **B**: requests of 1,024 prompt tokens and 256 output tokens, greedy, `max(4, 2c)`
//!   requests at concurrency `c`, every prompt unique.
//! - **C**: conversations of 7 turns, `max(2, c)` at concurrency `c` (one turn of each in
//!   flight at a time): turn 1 is 1,760 tokens; each later turn is the previous prompt,
//!   its 256 generated tokens and 1,504 new tokens, so 75 % of the input is a previous
//!   turn's prompt and output (the prefix cache's to hit). Sampled at temperature 1,
//!   top-p 0.95, top-k 20, a seed per turn.
//!
//! Prompts come from `--prompts` (`{"b": [[ids]], "conversations": [{"first": [ids],
//! "later": [[ids]]}]}`, text tokenised as the reference's runs were: acceptance of drafts
//! depends on the text), else seeded random ids. One executor and engine serve every
//! concurrency in turn. Each run is one JSON line: output and input tokens per second,
//! the time to the first token, the decode rate per stream, the drafted tokens per step,
//! and the steps each path ran.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, SaltDeriver};
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use eidola_engine_cuda::qwen::{
    CudaGraphs, QwenCudaExecutor, QwenCudaExecutorConfig, QwenProfile, StateStrategy,
};
use eidola_engine_model::WeightSet;
use eidola_engine_model::qwen::{QwenModel, QwenWeights, config_of};
use serde_json::json;

const B_IN: usize = 1024;
const OUT: u32 = 256;
const TURNS: usize = 7;
const STEP: usize = 1760;

fn fail(what: impl std::fmt::Display) -> ! {
    eprintln!("FAILED: {what}");
    std::process::exit(1)
}

struct Lcg(u64);

impl Lcg {
    fn ids(&mut self, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                u32::try_from((self.0 >> 33) % 240_000).unwrap_or(0)
            })
            .collect()
    }
}

/// A turn waiting to be submitted: its conversation, its turn index and its prompt.
struct Pending {
    conv: usize,
    turn: usize,
    prompt: Vec<u32>,
}

struct Live {
    conv: usize,
    turn: usize,
    prompt: Vec<u32>,
    submitted: f64,
    first: Option<f64>,
    last: f64,
    tokens: Vec<u32>,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i = ((q * v.len() as f64) as usize).min(v.len() - 1);
    v[i]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        fail(
            "usage: qwen_serve_bench <kernels> <checkpoint> <out.jsonl> <B|C> [--concurrency L] \
             [--draft K] [--profile fast|exact] [--graphs on|off] [--prompts F]",
        );
    }
    let (kdir, ckpt, out, workload) = (
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
        args[4].clone(),
    );
    let mut opt: HashMap<String, String> = HashMap::new();
    let mut it = args[5..].iter();
    while let Some(k) = it.next() {
        let v = it
            .next()
            .unwrap_or_else(|| fail(format!("{k} needs a value")));
        opt.insert(k.trim_start_matches("--").to_string(), v.clone());
    }
    let sampled = match workload.as_str() {
        "B" => false,
        "C" => true,
        w => fail(format!("workload {w}: B or C")),
    };
    let conc: Vec<usize> = opt
        .get("concurrency")
        .map_or(
            if sampled {
                "1,8,32,64"
            } else {
                "1,8,32,64,128"
            },
            String::as_str,
        )
        .split(',')
        .map(|c| c.parse().unwrap_or_else(|_| fail("concurrency")))
        .collect();
    let k: u32 = opt
        .get("draft")
        .map_or(Ok(0), |s| s.parse())
        .unwrap_or_else(|_| fail("draft"));
    let profile = QwenProfile::parse(opt.get("profile").map_or("fast", String::as_str))
        .unwrap_or_else(|| fail("profile: fast or exact"));
    let graphs = CudaGraphs::parse(opt.get("graphs").map_or("on", String::as_str))
        .unwrap_or_else(|e| fail(e));
    let prompts: Option<serde_json::Value> = opt.get("prompts").map(|p| {
        serde_json::from_slice(&std::fs::read(p).unwrap_or_else(|e| fail(e)))
            .unwrap_or_else(|e| fail(e))
    });
    let ids_of = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap_or_else(|| fail("prompts: arrays of ids"))
            .iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .unwrap_or_else(|| fail("prompts: a token id"))
            })
            .collect()
    };

    let t = Instant::now();
    let store = Arc::new(WeightSet::open_dir(&ckpt).unwrap_or_else(|e| fail(e)));
    let cfg = config_of(&store).unwrap_or_else(|e| fail(e));
    let model = QwenModel::new(QwenWeights::load(store, cfg).unwrap_or_else(|e| fail(e)));
    let max_c = conc.iter().copied().max().unwrap_or(1);
    let seats = u32::try_from(max_c.max(2)).unwrap_or_else(|_| fail("concurrency"));
    let ex_cfg = QwenCudaExecutorConfig {
        block_size: 16,
        num_blocks: if sampled { 49_152 } else { 12_288 },
        host_blocks: 0,
        state_buffers: if sampled { 3 * seats + 8 } else { seats + 8 },
        host_state_buffers: 0,
        num_state_slots: seats,
        max_model_len: if sampled { 16_384 } else { 2048 },
        buckets: vec![Bucket {
            max_seqs: seats,
            max_tokens: 2048,
        }],
        draft_tokens: k,
        sampleable_vocab_size: 248_077,
        strategy: StateStrategy::Live,
        graphs,
    };
    let dev = CudaDevice::open(0, &KernelDir::new(&kdir), profile).unwrap_or_else(|e| fail(e));
    let ex = QwenCudaExecutor::new(dev, &model, ex_cfg).unwrap_or_else(|e| fail(e));
    eprintln!(
        "{} executor, graphs {}, draft {k}: loaded in {:.0} s",
        profile.name(),
        graphs.as_str(),
        t.elapsed().as_secs_f64()
    );
    let mut eng = Engine::new(
        ex,
        SchedulerConfig {
            max_batched_tokens: 2048,
            max_seqs: seats,
            max_prefill_chunk: 2048,
            eos_token_ids: Vec::new(),
            speculative: k > 0,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        },
    )
    .unwrap_or_else(|e| fail(format!("{e:?}")));
    let deriver = SaltDeriver::with_boot_key(BootKey::from_bytes([7; 32]));

    let mut rng = Lcg(12_345);
    let mut b_cursor = 0usize;
    let mut c_cursor = 0usize;
    let mut lines = Vec::new();
    let mut now = 0u64;
    let mut next_id = 0u64;
    for &c in &conc {
        // The run's turns: B's requests are one-turn conversations.
        let n = if sampled { c.max(2) } else { (2 * c).max(4) };
        let mut later: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n);
        let mut queue: VecDeque<Pending> = VecDeque::new();
        for conv in 0..n {
            let (first, rest) = match (&prompts, sampled) {
                (Some(p), false) => {
                    let b = p["b"]
                        .get(b_cursor)
                        .unwrap_or_else(|| fail("prompts: too few B prompts"));
                    b_cursor += 1;
                    (ids_of(b), Vec::new())
                }
                (Some(p), true) => {
                    let cv = p["conversations"]
                        .get(c_cursor)
                        .unwrap_or_else(|| fail("prompts: too few conversations"));
                    c_cursor += 1;
                    let rest = cv["later"]
                        .as_array()
                        .unwrap_or_else(|| fail("prompts: later turns"))
                        .iter()
                        .map(&ids_of)
                        .collect();
                    (ids_of(&cv["first"]), rest)
                }
                (None, false) => (rng.ids(B_IN), Vec::new()),
                (None, true) => (
                    rng.ids(STEP),
                    (1..TURNS).map(|_| rng.ids(STEP - OUT as usize)).collect(),
                ),
            };
            later.push(rest);
            queue.push_back(Pending {
                conv,
                turn: 0,
                prompt: first,
            });
        }
        let turns = if sampled { TURNS } else { 1 };
        let stats0 = eng.executor().decode_stats();
        let t0 = Instant::now();
        let mut live: HashMap<u64, Live> = HashMap::new();
        let mut done: Vec<Live> = Vec::new();
        let (mut steps, mut step_s, mut produced_steps) = (0u64, Vec::new(), 0u64);
        while !queue.is_empty() || !live.is_empty() {
            while live.len() < c
                && let Some(p) = queue.pop_front()
            {
                let id = next_id;
                next_id += 1;
                let seed = 7919 * c as u64 + 31 * p.conv as u64 + p.turn as u64;
                let sampling = if sampled {
                    SamplingParams::new(1.0, 20, 0.95, 0.0, seed).unwrap_or_else(|e| fail(e.0))
                } else {
                    SamplingParams::greedy()
                };
                let mut kb = [1u8; 32];
                kb[..8].copy_from_slice(&(p.conv as u64 + 1000 * c as u64).to_le_bytes());
                let key = deriver.lineage(&CacheKey::from_bytes(kb));
                eng.submit(Request {
                    id,
                    prompt: p.prompt.clone(),
                    sampling,
                    max_tokens: OUT,
                    required_cached_tokens: 0,
                    stop_token_ids: Vec::new(),
                    cache: CacheScope::Keyed(key),
                })
                .unwrap_or_else(|e| fail(format!("{e:?}")));
                live.insert(
                    id,
                    Live {
                        conv: p.conv,
                        turn: p.turn,
                        prompt: p.prompt,
                        submitted: t0.elapsed().as_secs_f64(),
                        first: None,
                        last: 0.0,
                        tokens: Vec::new(),
                    },
                );
            }
            let ts = Instant::now();
            let events = eng.step(now).unwrap_or_else(|e| fail(e.0));
            now += 1;
            steps += 1;
            step_s.push(ts.elapsed().as_secs_f64());
            let at = t0.elapsed().as_secs_f64();
            let mut any = false;
            for e in events {
                let l = live
                    .get_mut(&e.id)
                    .unwrap_or_else(|| fail("an unknown request"));
                if !e.tokens.is_empty() {
                    any = true;
                    l.first.get_or_insert(at);
                    l.last = at;
                    l.tokens.extend(&e.tokens);
                }
                if e.finish.is_some() {
                    let l = live.remove(&e.id).unwrap_or_else(|| fail("a request"));
                    if l.turn + 1 < turns {
                        let mut prompt = l.prompt.clone();
                        prompt.extend(&l.tokens);
                        prompt.extend(&later[l.conv][l.turn]);
                        queue.push_back(Pending {
                            conv: l.conv,
                            turn: l.turn + 1,
                            prompt,
                        });
                    }
                    done.push(l);
                }
            }
            produced_steps += u64::from(any);
        }
        let wall = t0.elapsed().as_secs_f64();
        let stats = eng.executor().decode_stats();
        let output: usize = done.iter().map(|l| l.tokens.len()).sum();
        let input: usize = done.iter().map(|l| l.prompt.len()).sum();
        let mut ttft: Vec<f64> = done
            .iter()
            .map(|l| l.first.unwrap_or(wall) - l.submitted)
            .collect();
        let mut decode: Vec<f64> = done
            .iter()
            .filter_map(|l| {
                let f = l.first?;
                (l.last > f).then(|| (l.tokens.len() - 1) as f64 / (l.last - f))
            })
            .collect();
        let line = json!({
            "workload": workload, "profile": profile.name(), "graphs": graphs.as_str(),
            "draft": k, "concurrency": c, "requests": done.len(),
            "wall_seconds": wall, "output_tokens": output, "input_tokens": input,
            "output_tokens_per_second": output as f64 / wall,
            "input_tokens_per_second": input as f64 / wall,
            "ttft_median": pct(&mut ttft, 0.5), "ttft_p90": pct(&mut ttft, 0.9),
            "decode_median": pct(&mut decode, 0.5), "decode_p10": pct(&mut decode, 0.1),
            "ms_per_token_c1": if c == 1 { 1000.0 / pct(&mut decode, 0.5).max(1e-9) } else { 0.0 },
            "steps": steps, "median_step_ms": 1000.0 * pct(&mut step_s, 0.5),
            "tokens_per_producing_step": output as f64 / produced_steps.max(1) as f64,
            "eager_steps": stats.eager - stats0.eager,
            "direct_steps": stats.direct - stats0.direct,
            "replayed_steps": stats.replay - stats0.replay,
        });
        println!("{line}");
        lines.push(line.to_string());
        std::fs::write(&out, lines.join("\n") + "\n").unwrap_or_else(|e| fail(e));
    }
    eng.flush().unwrap_or_else(|e| fail(e.0));
}
