//! The released DeepSeek-V4.1 checkpoint whole on one GPU, on the exact executor with
//! the backbone's routed experts staged from host memory (`deepseek::staged`). Dev-only: built
//! only with the `dev-staged-experts` feature, which no node build enables.
//!
//! Two phases, run in order (`EIDOLA_DS_WHOLE_PHASES`):
//!
//! - `compare`: both halves on the device at once, over the text and the steps the
//!   one-half-at-a-time measurement (`examples/deepseek_halves.rs`) recorded: the encoder half
//!   through `DsCudaExecutor::encode_half`, the Engram rows of the decoder's table
//!   gathered on the decoder's half, the crossing, the decoder half through
//!   `decode_half` with logits at every position. Every layer's streams, the boundary,
//!   the bank entries and the logits are written as `deepseek_halves` writes them and
//!   compared bit for bit with its recordings (which it held equal to the model crate's
//!   forward), the first difference named by array, position and element.
//! - `generate`: greedy generation through the serving core's engine over the staged
//!   executor, for the chat prompts of `eidola-server-engine`'s `deepseek_generate` example,
//!   at each draft width of `EIDOLA_DS_WHOLE_WIDTHS` (0 and 5 by default; one executor
//!   built for the widest, each run's engine asking a row for at most its own). The tokens
//!   must be the same at every width; each step's time (by part, and the cache's
//!   read-backs and copies) and each drafting row's acceptance are recorded.
//!
//! ```sh
//! EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> EIDOLA_DEEPSEEK_FULL_DIR=<checkpoint> \
//! EIDOLA_DS_WHOLE_RECORDED=<ds_halves output> EIDOLA_DS_WHOLE_OUT=<dir> \
//!     cargo run --release -p eidola-engine-cuda --features dev-staged-experts --example deepseek_whole
//! ```
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `EIDOLA_DS_WHOLE_PHASES` | of `compare`, `generate`, in order | both |
//! | `EIDOLA_DS_WHOLE_RECORDED` | the `deepseek_halves` output directory (`compare`) | |
//! | `EIDOLA_DS_WHOLE_OUT` | recordings and `ds_whole.json` | `ds_whole` in the working directory |
//! | `EIDOLA_DS_WHOLE_HOST` | `read` (each expert read once into host memory) or `mapped` | `read` |
//! | `EIDOLA_DS_WHOLE_THREADS` | threads reading the experts | the machine's |
//! | `EIDOLA_DS_WHOLE_RESERVE_GB` | device memory left free beside the caches | 4 |
//! | `EIDOLA_DS_WHOLE_PROMPTS` | a prompts file in `deepseek_generate`'s form | `deepseek_generate`'s |
//! | `EIDOLA_DS_WHOLE_MAX_NEW` | new tokens per prompt, at most | 64 |
//! | `EIDOLA_DS_WHOLE_WIDTHS` | draft widths, comma-separated | `0,5` |
//! | `EIDOLA_DS_WHOLE_CHUNK` | positions per prefill step (`generate`) | 96 |
//! | `EIDOLA_DS_HALVES_GOLDEN` | as `deepseek_halves` (the text and the Engram map, `compare`) | as `deepseek_halves` |
//!
//! Before a device phase the same phase runs over a device with no memory (`deepseek::dry`),
//! so whatever arithmetic refuses is refused before anything is loaded. The process
//! exits 1 when anything compared differs and 2 on a failure.

#[path = "deepseek_both/mod.rs"]
mod deepseek_both;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use deepseek_both::{Layout, Prompt, Watch, fail, golden, n32, prompts, us};
use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cuda::deepseek::dry::DryDevice;
use eidola_engine_cuda::deepseek::gemm::GemmForm;
use eidola_engine_cuda::deepseek::half::{CudaDevice, Device};
use eidola_engine_cuda::deepseek::model::HalfNet;
use eidola_engine_cuda::deepseek::staged::{
    self, Capacity, DevOnly, HostExperts, HostMode, StagedStats,
};
use eidola_engine_cuda::deepseek::weights::Half;
use eidola_engine_cuda::deepseek::{DsCudaExecutor, DsCudaExecutorConfig};
use eidola_engine_cuda::{KernelDir, Result};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::{DsConfig, config_of};
use serde_json::{Value, json};

const TOKEN: fn() -> DevOnly = DevOnly::for_tests_and_dev_tools;

/// Slots of the comparison: the text's alone.
const SLOTS: u32 = 1;

/// Opens the device a half runs on.
type Open<'a> = &'a dyn Fn() -> Result<Box<dyn Device>>;
/// How many experts each half's cache holds, asked once both halves are loaded.
type Size<'a> = &'a dyn Fn(usize) -> Result<[u32; 2]>;

/// What each step of the comparison did to the halves' caches.
struct Caches;

impl Watch for Caches {
    type Snap = [StagedStats; 2];
    fn snap(&self, enc: &HalfNet, dec: &HalfNet) -> [StagedStats; 2] {
        [enc, dec].map(|n| n.staged.as_ref().map(|c| c.stats()).unwrap_or_default())
    }
    fn delta(&self, a: &[StagedStats; 2], b: &[StagedStats; 2]) -> Value {
        json!([stats_delta(&a[0], &b[0]), stats_delta(&a[1], &b[1])])
    }
}

/// One half, staged, with every block of the sequence mapped.
fn staged_half(l: &Layout, half: Half, dev: Box<dyn Device>) -> Result<HalfNet> {
    let w = staged::half_weights(
        dev.as_ref(),
        l.store.as_deref(),
        &l.c,
        &l.shape,
        half,
        l.spec.max_model_len,
        TOKEN(),
    )?;
    l.net(half, dev, w)
}

/// Both halves loaded staged on devices from `open`, over every chunk of `tokens`; the
/// recordings written to `out`, and compared with `recorded`'s when given.
fn compare(
    l: &Layout,
    tokens: &[u32],
    out: &Path,
    open: Open<'_>,
    size: Size<'_>,
    host: &Arc<HostExperts>,
    recorded: Option<&Path>,
) -> Result<(Value, bool)> {
    let began = Instant::now();
    let mut enc = staged_half(l, Half::Encoder, open()?)?;
    let mut dec = staged_half(l, Half::Decoder, open()?)?;
    enc.dev.synchronize()?;
    dec.dev.synchronize()?;
    let load = began.elapsed().as_secs_f64();
    let caps = size(host.expert_bytes())?;
    staged::attach(&mut enc, &l.shape, host, caps[0], TOKEN())?;
    staged::attach(&mut dec, &l.shape, host, caps[1], TOKEN())?;
    let memory = json!([enc.memory().total(), dec.memory().total()]);
    eprintln!("both halves loaded in {load:.0} s; caches of {caps:?} experts; bytes {memory}");
    let r = l.seq(tokens, 0).run(&mut enc, &mut dec, &Caches)?;
    r.write(out)?;
    let (checks, equal) = match recorded {
        Some(dir) => r.compare(l, tokens.len(), dir)?,
        None => (Vec::new(), true),
    };
    let cache = [&enc, &dec].map(|n| n.staged.as_ref().map(|c| stats_json(&c.stats())));
    Ok((
        json!({
            "load_seconds": load, "capacity": caps, "device_bytes": memory,
            "steps": r.steps, "cache": cache,
            "recorded": recorded.map(|p| p.display().to_string()),
            "equal": equal, "checks": checks,
        }),
        equal,
    ))
}

fn stats_json(s: &StagedStats) -> Value {
    json!({
        "layers": s.layers, "rounds": s.rounds, "needed": s.needed, "hits": s.hits,
        "loads": s.loads, "evictions": s.evictions, "bytes_loaded": s.bytes_loaded,
        "readback_seconds": s.readback_seconds, "load_seconds": s.load_seconds,
        "hit_rate": if s.needed > 0 { s.hits as f64 / s.needed as f64 } else { 0.0 },
    })
}

fn stats_delta(a: &StagedStats, b: &StagedStats) -> Value {
    stats_json(&StagedStats {
        layers: b.layers - a.layers,
        rounds: b.rounds - a.rounds,
        needed: b.needed - a.needed,
        hits: b.hits - a.hits,
        loads: b.loads - a.loads,
        evictions: b.evictions - a.evictions,
        bytes_loaded: b.bytes_loaded - a.bytes_loaded,
        readback_seconds: b.readback_seconds - a.readback_seconds,
        load_seconds: b.load_seconds - a.load_seconds,
    })
}

/// The staged executor under the engine, recording each step: its time by part, the
/// caches' work, and each drafting row's drafts and acceptance.
struct Probe<'a> {
    x: &'a mut DsCudaExecutor,
    /// The executor's spec with the run's draft width: the engine asks a row for at
    /// most that many drafts, and the executor drafts what a row asks for.
    spec: ModelSpec,
    steps: Vec<Value>,
    /// Per draft slot: rows that proposed it, and rows that accepted it.
    proposed: Vec<u64>,
    accepted: Vec<u64>,
}

impl Executor for Probe<'_> {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> std::result::Result<StepOutput, ExecutorError> {
        let before = self.x.staged_stats();
        let at = Instant::now();
        let out = self.x.execute(step)?;
        let ms = at.elapsed().as_secs_f64() * 1e3;
        let after = self.x.staged_stats();
        let parts = self.x.take_stats();
        self.x.take_transfers();
        self.x.take_draft_logits();
        let mut rows = Vec::new();
        for (i, e) in step.seqs.iter().enumerate() {
            if !e.sample {
                continue;
            }
            let produced = out.num_tokens[i];
            let drafts = e.num_drafts;
            let accepted = produced.saturating_sub(1);
            for d in 0..us(drafts) {
                if self.proposed.len() <= d {
                    self.proposed.resize(d + 1, 0);
                    self.accepted.resize(d + 1, 0);
                }
                self.proposed[d] += 1;
                self.accepted[d] += u64::from(d < us(accepted));
            }
            rows.push(json!({"drafts": drafts, "produced": produced}));
        }
        let part = parts.first().copied().unwrap_or_default();
        let tokens: u32 = step.seqs.iter().map(|e| e.num_tokens).sum();
        self.steps.push(json!({
            "positions": tokens, "ms": ms, "rows": rows,
            "parts_ms": {
                "maintenance": part.maintenance_ms, "draft": part.draft_ms,
                "gather": part.gather_ms, "encoder": part.encoder_ms,
                "crossing": part.crossing_ms, "decoder": part.decoder_ms,
            },
            "cache": [stats_delta(&before[0], &after[0]), stats_delta(&before[1], &after[1])],
        }));
        Ok(out)
    }
}

/// Greedy generation of every prompt at each width, through the engine over `x`; the
/// tokens must agree across widths. Returns the results and whether they agree.
fn generate(
    x: &mut DsCudaExecutor,
    prompts: &[Prompt],
    widths: &[u32],
    max_new: u32,
    chunk: u32,
    eos: &[u32],
    decode: &dyn Fn(&[u32]) -> String,
) -> Result<(Value, bool)> {
    let mut results = Vec::new();
    let mut outputs: Vec<Vec<Vec<u32>>> = vec![Vec::new(); prompts.len()];
    for &width in widths {
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
        let probe = Probe {
            spec,
            x: &mut *x,
            steps: Vec::new(),
            proposed: Vec::new(),
            accepted: Vec::new(),
        };
        let sched = SchedulerConfig {
            max_batched_tokens: chunk,
            max_seqs: 1,
            max_prefill_chunk: chunk,
            eos_token_ids: eos.to_vec(),
            speculative: width > 0,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        };
        let mut eng = Engine::new(probe, sched).map_err(|e| fail(format!("{e:?}")))?;
        for (i, p) in prompts.iter().enumerate() {
            let id = u64::try_from(i).expect("a few prompts");
            let at = Instant::now();
            let first = eng.executor().steps.len();
            eng.submit(Request {
                id,
                prompt: p.tokens.clone(),
                sampling: SamplingParams::greedy(),
                max_tokens: max_new,
                required_cached_tokens: 0,
                stop_token_ids: Vec::new(),
                cache: CacheScope::Private,
            })
            .map_err(|e| fail(format!("{}: {e:?}", p.name)))?;
            let mut out = Vec::new();
            let mut now = 0;
            while eng.unfinished() > 0 {
                for e in eng.step(now).map_err(fail)? {
                    out.extend(&e.tokens);
                }
                now += 1;
            }
            eng.flush().map_err(fail)?;
            let seconds = at.elapsed().as_secs_f64();
            let steps: Vec<Value> = eng.executor().steps[first..].to_vec();
            let decode_steps = steps
                .iter()
                .filter(|s| s["rows"].as_array().is_some_and(|r| !r.is_empty()))
                .count();
            eprintln!(
                "{} at width {width}: {} tokens in {seconds:.1} s, {} steps: {:?}",
                p.name,
                out.len(),
                steps.len(),
                decode(&out)
            );
            results.push(json!({
                "prompt": p.name, "width": width, "prompt_tokens": p.tokens.len(),
                "tokens": out, "text": decode(&out), "seconds": seconds,
                "sampling_steps": decode_steps,
                "tokens_per_second": out.len() as f64 / seconds.max(f64::MIN_POSITIVE),
                "steps": steps,
            }));
            outputs[i].push(out);
        }
        let probe = eng.executor();
        let stats = eng.stats();
        results.push(json!({
            "width": width,
            "acceptance": {
                "drafted": stats.drafted, "accepted": stats.accepted,
                "per_slot_proposed": probe.proposed, "per_slot_accepted": probe.accepted,
            },
        }));
    }
    let mut agree = true;
    let mut checks = Vec::new();
    for (p, outs) in prompts.iter().zip(&outputs) {
        let same = outs.windows(2).all(|w| w[0] == w[1]);
        agree &= same;
        checks.push(json!({"prompt": p.name, "same_tokens_at_every_width": same}));
    }
    Ok((
        json!({"results": results, "checks": checks, "equal": agree}),
        agree,
    ))
}

/// The executor configuration for generating `prompts` up to `max_new` tokens each.
fn generation_config(
    c: &DsConfig,
    prompts: &[Prompt],
    max_new: u32,
    chunk: u32,
    width: u32,
    sampleable: u32,
) -> DsCudaExecutorConfig {
    let block = 32;
    let longest = prompts
        .iter()
        .map(|p| n32(p.tokens.len()))
        .max()
        .unwrap_or(1);
    let max_model_len = (longest + max_new + width + 1).next_multiple_of(block);
    let max_model_len = c
        .max_positions
        .map_or(max_model_len, |m| max_model_len.min(n32(m)));
    DsCudaExecutorConfig {
        block_size: block,
        // The sequence twice over, the null block, and a little room.
        num_blocks: 2 * (max_model_len / block) + 8,
        window_blocks: 2 * (max_model_len / block) + 8,
        num_state_slots: 2,
        max_model_len,
        buckets: vec![Bucket {
            max_seqs: 1,
            max_tokens: chunk,
        }],
        sampleable_vocab_size: sampleable,
        devices: [0, 0],
        draft_width: width,
        host_blocks: 0,
        position_tags: false,
        record: true,
        gemm: GemmForm::Exact,
    }
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> T {
    var(name).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{name} ({v}) is not a number"))
    })
}

fn main() {
    let dir = PathBuf::from(
        var("EIDOLA_DEEPSEEK_FULL_DIR").expect("EIDOLA_DEEPSEEK_FULL_DIR names the checkpoint"),
    );
    let out = var("EIDOLA_DS_WHOLE_OUT").map_or_else(|| PathBuf::from("ds_whole"), PathBuf::from);
    let phases: Vec<String> = var("EIDOLA_DS_WHOLE_PHASES")
        .unwrap_or_else(|| "compare,generate".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let threads = number(
        "EIDOLA_DS_WHOLE_THREADS",
        std::thread::available_parallelism().map_or(8, |n| n.get()),
    );
    let mode = match var("EIDOLA_DS_WHOLE_HOST").as_deref() {
        None | Some("read") => HostMode::Read { threads },
        Some("mapped") => HostMode::Mapped,
        Some(other) => panic!("EIDOLA_DS_WHOLE_HOST: {other} is neither read nor mapped"),
    };
    let reserve = number("EIDOLA_DS_WHOLE_RESERVE_GB", 4usize) << 30;
    std::fs::create_dir_all(&out).expect("the output directory");
    let summary_path = out.join("ds_whole.json");
    let mut summary = json!({"phases": phases});
    let save = |summary: &Value| {
        std::fs::write(
            &summary_path,
            serde_json::to_string_pretty(summary).unwrap(),
        )
        .expect("write the summary");
    };
    let exit = |code: i32, what: &str| -> ! {
        eprintln!("{what}");
        std::process::exit(code)
    };
    let kernels = || -> Result<KernelDir> {
        let k = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR")
            .ok_or_else(|| fail("EIDOLA_ENGINE_KERNELS_DIR is not set"))?;
        Ok(KernelDir::new(k))
    };

    let store = Arc::new(WeightSet::open_dir(&dir).expect("the checkpoint"));
    let c = config_of(&store).expect("the checkpoint's config");
    let shape =
        eidola_engine_cuda::deepseek::support::check_supported(&c, 0).expect("a supported model");
    let at = Instant::now();
    eprintln!("reading the routed experts ({mode:?})");
    let host = Arc::new(
        HostExperts::new(&store, &c, &shape, mode, TOKEN())
            .unwrap_or_else(|e| exit(2, &format!("FAILED: {e}"))),
    );
    summary["host"] = json!({
        "mode": format!("{mode:?}"), "seconds": at.elapsed().as_secs_f64(),
        "read_bytes": host.read_bytes, "read_seconds": host.read_seconds,
        "read_gb_per_second": host.read_bytes as f64 / 1e9 / host.read_seconds.max(f64::MIN_POSITIVE),
        "expert_bytes": host.expert_bytes(),
    });
    eprintln!("{}", summary["host"]);
    save(&summary);
    let mut equal = true;
    let fill = |_: usize| -> Result<[u32; 2]> {
        let free = staged::free_device_bytes(0)?;
        staged::capacities(
            Capacity::Fill { reserve },
            [free, free],
            true,
            host.expert_bytes(),
        )
    };

    for phase in &phases {
        let began = Instant::now();
        let result = match phase.as_str() {
            "compare" => (|| -> Result<(Value, bool)> {
                let recorded =
                    PathBuf::from(var("EIDOLA_DS_WHOLE_RECORDED").ok_or_else(|| {
                        fail("EIDOLA_DS_WHOLE_RECORDED names the ds_halves output")
                    })?);
                let halves: Value = serde_json::from_slice(
                    &std::fs::read(recorded.join("ds_halves.json")).map_err(fail)?,
                )
                .map_err(fail)?;
                let count = usize::try_from(
                    halves["tokens"]
                        .as_u64()
                        .ok_or_else(|| fail("the recording's token count"))?,
                )
                .map_err(fail)?;
                let chunk = u32::try_from(
                    halves["chunk"]
                        .as_u64()
                        .ok_or_else(|| fail("the recording's chunk"))?,
                )
                .map_err(fail)?;
                let g = WeightSet::open_files(&[golden(&dir)]).map_err(fail)?;
                let ints = |name: &str| -> Result<Vec<u32>> {
                    let v = g.get(name).map_err(fail)?.to_i64(name).map_err(fail)?;
                    v.into_iter()
                        .map(|t| u32::try_from(t).map_err(fail))
                        .collect()
                };
                let mut tokens = ints("tokens")?;
                tokens.truncate(count);
                let map = ints("engram.token_map")?;
                let dry = Layout::new(None, c.clone(), map.clone(), count, SLOTS, chunk)?;
                let open_dry = || -> Result<Box<dyn Device>> { Ok(Box::new(DryDevice::new(None))) };
                compare(
                    &dry,
                    &tokens,
                    &out.join("dry"),
                    &open_dry,
                    &|_| Ok([1600, 1600]),
                    &host,
                    None,
                )?;
                eprintln!("dry run: every launch of both halves is accepted");
                let run = Layout::new(Some(store.clone()), c.clone(), map, count, SLOTS, chunk)?;
                let k = kernels()?;
                let open = || -> Result<Box<dyn Device>> {
                    let mut d = CudaDevice::open(0, &k)?;
                    d.select_gemm(GemmForm::from_env()?)?;
                    Ok(Box::new(d))
                };
                let (mut v, same) =
                    compare(&run, &tokens, &out, &open, &fill, &host, Some(&recorded))?;
                v["tokens"] = json!(count);
                v["chunk"] = json!(chunk);
                Ok((v, same))
            })(),
            "generate" => (|| -> Result<(Value, bool)> {
                let file = var("EIDOLA_DS_WHOLE_PROMPTS")
                    .map_or_else(deepseek_both::default_prompts_file, PathBuf::from);
                let (prompts, eos, tokenizer, map) = prompts(&dir, &file).map_err(fail)?;
                let widths: Vec<u32> = var("EIDOLA_DS_WHOLE_WIDTHS")
                    .unwrap_or_else(|| "0,5".into())
                    .split(',')
                    .map(|w| w.trim().parse().map_err(fail))
                    .collect::<Result<_>>()?;
                let max_new = number("EIDOLA_DS_WHOLE_MAX_NEW", 64u32);
                let chunk = number("EIDOLA_DS_WHOLE_CHUNK", 96u32);
                let width = widths.iter().copied().max().unwrap_or(0);
                let vocab = n32(tokenizer.vocab_size());
                let sampleable = c.image_token_id.map_or(vocab, |i| i.min(vocab));
                let cfg = generation_config(&c, &prompts, max_new, chunk, width, sampleable);
                eprintln!(
                    "{} prompts of {:?} tokens; {cfg:?}",
                    prompts.len(),
                    prompts.iter().map(|p| p.tokens.len()).collect::<Vec<_>>()
                );
                let at = Instant::now();
                let mut x = DsCudaExecutor::staged(
                    &kernels()?,
                    &store,
                    &c,
                    Some(map),
                    &cfg,
                    &host,
                    Capacity::Fill { reserve },
                    TOKEN(),
                )?;
                let load = at.elapsed().as_secs_f64();
                eprintln!(
                    "the staged executor loaded in {load:.0} s; caches of {:?} experts",
                    x.staged_capacity()
                );
                let decode = |ids: &[u32]| {
                    tokenizer
                        .decode(ids, false)
                        .unwrap_or_else(|e| format!("<{e}>"))
                };
                let (mut v, same) =
                    generate(&mut x, &prompts, &widths, max_new, chunk, &eos, &decode)?;
                v["load_seconds"] = json!(load);
                v["capacity"] = json!(x.staged_capacity());
                v["device_bytes"] = json!(x.memory().map(|m| m.total()));
                v["cache"] = json!(x.staged_stats().map(|s| stats_json(&s)));
                Ok((v, same))
            })(),
            other => exit(2, &format!("unknown phase {other}")),
        };
        match result {
            Ok((v, same)) => {
                equal &= same;
                summary[phase.as_str()] = v;
            }
            Err(e) => {
                summary[phase.as_str()] = json!({"error": e.to_string()});
                save(&summary);
                exit(2, &format!("FAILED in {phase}: {e}"));
            }
        }
        summary[format!("{phase}_seconds")] = json!(began.elapsed().as_secs_f64());
        save(&summary);
        eprintln!("{phase}: {:.0} s", began.elapsed().as_secs_f64());
    }
    println!("summary: {}", summary_path.display());
    if !equal {
        exit(1, "FAILED: something compared differs (see the summary)");
    }
}

#[cfg(test)]
mod tests {
    use eidola_engine_cuda::deepseek::host::HostDevice;

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

    /// The comparison on the fixture over the host device: recorded with a cache of every
    /// expert, then run again with caches of one row's experts and compared equal; a
    /// flipped recorded bit is named by array, position and element.
    #[test]
    fn the_fixture_staged_equals_its_recording() {
        let (store, c, map) = fixture();
        let shape = eidola_engine_cuda::deepseek::support::check_supported(&c, 0).unwrap();
        let host = Arc::new(
            HostExperts::new(&store, &c, &shape, HostMode::Read { threads: 2 }, TOKEN()).unwrap(),
        );
        let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37 + 11) % 95).collect();
        let base = std::env::temp_dir().join(format!("ds_whole_fixture_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let open = || -> Result<Box<dyn Device>> { Ok(Box::new(HostDevice)) };
        let l = Layout::new(Some(store.clone()), c.clone(), map, tokens.len(), SLOTS, 16).unwrap();
        let (v, _) = compare(
            &l,
            &tokens,
            &base.join("all"),
            &open,
            &|_| Ok([32, 32]),
            &host,
            None,
        )
        .unwrap();
        assert_eq!(v["steps"].as_array().unwrap().len(), 5);
        assert_eq!(v["cache"][0]["evictions"], 0);
        let all = base.join("all");
        let (v, equal) = compare(
            &l,
            &tokens,
            &base.join("two"),
            &open,
            &|_| Ok([2, 2]),
            &host,
            Some(&all),
        )
        .unwrap();
        assert!(equal, "{v:#}");
        // 4 encoder layers, the boundary's three arrays, 4 decoder layers, the logits.
        assert_eq!(v["checks"].as_array().unwrap().len(), 4 + 3 + 4 + 1);
        assert!(v["cache"][0]["evictions"].as_u64().unwrap() > 0);
        assert!(
            v["cache"][1]["rounds"].as_u64().unwrap() > v["cache"][1]["layers"].as_u64().unwrap()
        );

        let path = base.join("all/decoder_layers.bin");
        let mut layers = std::fs::read(&path).unwrap();
        let width = l.width();
        // Position 3 of decoder layer 5 (the second of the decoder), element 9.
        let at = 2 * (70 * width + 3 * width + 9);
        layers[at] ^= 1;
        std::fs::write(&path, layers).unwrap();
        let (v, equal) = compare(
            &l,
            &tokens,
            &base.join("two"),
            &open,
            &|_| Ok([3, 2]),
            &host,
            Some(&all),
        )
        .unwrap();
        assert!(!equal);
        let bad: Vec<&Value> = v["checks"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["equal"] == json!(false))
            .collect();
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0]["what"], "streams after decoder layer 5");
        assert_eq!(bad[0]["first"]["position"], 3);
        assert_eq!(bad[0]["first"]["element"], 9);
    }

    /// Generation through the engine on the fixture: the same tokens at widths 0 and 4,
    /// with drafts proposed and some accepted, per-step records of the caches' work.
    #[test]
    fn the_fixture_generates_the_same_tokens_at_every_width() {
        let (store, c, map) = fixture();
        let shape = eidola_engine_cuda::deepseek::support::check_supported(&c, 4).unwrap();
        let host =
            Arc::new(HostExperts::new(&store, &c, &shape, HostMode::Mapped, TOKEN()).unwrap());
        let prompts = vec![
            Prompt {
                name: "a".into(),
                tokens: (0..40u32).map(|i| (i * 7 + 3) % 95).collect(),
            },
            Prompt {
                name: "b".into(),
                tokens: (0..9u32).map(|i| (i * 11 + 5) % 95).collect(),
            },
        ];
        let cfg = generation_config(&c, &prompts, 24, 16, 4, 95);
        let mut x = DsCudaExecutor::staged_with_devices(
            Box::new(HostDevice),
            Box::new(HostDevice),
            &store,
            &c,
            Some(map),
            &cfg,
            &host,
            [3, 2],
            TOKEN(),
        )
        .unwrap();
        let decode = |ids: &[u32]| format!("{ids:?}");
        let (v, same) = generate(&mut x, &prompts, &[0, 2, 4], 24, 16, &[], &decode).unwrap();
        assert!(same, "{v:#}");
        let results = v["results"].as_array().unwrap();
        let a0 = results
            .iter()
            .find(|r| r["prompt"] == "a" && r["width"] == 0)
            .unwrap();
        assert_eq!(a0["tokens"].as_array().unwrap().len(), 24);
        let acceptance = &results
            .iter()
            .find(|r| r["width"] == 4 && r.get("acceptance").is_some())
            .unwrap()["acceptance"];
        assert!(acceptance["drafted"].as_u64().unwrap() > 0);
        // Each width drafts at most its own number a step, and the widest reaches it.
        for width in [0u64, 2, 4] {
            let drafts: Vec<u64> = results
                .iter()
                .filter(|r| r["width"] == width && r.get("steps").is_some())
                .flat_map(|r| r["steps"].as_array().unwrap().iter())
                .flat_map(|s| s["rows"].as_array().unwrap().iter())
                .map(|row| row["drafts"].as_u64().unwrap())
                .collect();
            assert!(!drafts.is_empty());
            assert!(
                drafts.iter().all(|d| *d <= width),
                "width {width}: {drafts:?}"
            );
            assert_eq!(drafts.iter().max(), Some(&width), "width {width}");
        }
        let step = &a0["steps"][0];
        assert!(step["cache"][0]["needed"].as_u64().unwrap() > 0);
        assert!(x.staged_stats()[1].evictions > 0);
    }
}
