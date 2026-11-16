//! The released DeepSeek-V4.1 checkpoint whole on one GPU, on the exact executor with
//! the backbone's routed experts staged from host memory (`ds::staged`). Dev-only: built
//! only with the `dev-staged-experts` feature, which no node build enables.
//!
//! Two phases, run in order (`EIDOLA_DS_WHOLE_PHASES`):
//!
//! - `compare`: both halves on the device at once, over the text and the steps the
//!   one-half-at-a-time measurement (`examples/ds_halves.rs`) recorded: the encoder half
//!   through `DsCudaExecutor::encode_half`, the Engram rows of the decoder's table
//!   gathered on the decoder's half, the crossing, the decoder half through
//!   `decode_half` with logits at every position. Every layer's streams, the boundary,
//!   the bank entries and the logits are written as `ds_halves` writes them and
//!   compared bit for bit with its recordings (which it held equal to the model crate's
//!   forward), the first difference named by array, position and element.
//! - `generate`: greedy generation through the serving core's engine over the staged
//!   executor, for the chat prompts of `eidola-server-engine`'s `ds_generate` example,
//!   at each draft width of `EIDOLA_DS_WHOLE_WIDTHS` (0 and 5 by default; one executor
//!   built for the widest, each run's engine asking a row for at most its own). The tokens
//!   must be the same at every width; each step's time (by part, and the cache's
//!   read-backs and copies) and each drafting row's acceptance are recorded.
//!
//! ```sh
//! EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> EIDOLA_DEEPSEEK_FULL_DIR=<checkpoint> \
//! EIDOLA_DS_WHOLE_RECORDED=<ds_halves output> EIDOLA_DS_WHOLE_OUT=<dir> \
//!     cargo run --release -p eidola-engine-cuda --features dev-staged-experts --example ds_whole
//! ```
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `EIDOLA_DS_WHOLE_PHASES` | of `compare`, `generate`, in order | both |
//! | `EIDOLA_DS_WHOLE_RECORDED` | the `ds_halves` output directory (`compare`) | |
//! | `EIDOLA_DS_WHOLE_OUT` | recordings and `ds_whole.json` | `ds_whole` in the working directory |
//! | `EIDOLA_DS_WHOLE_HOST` | `read` (each expert read once into host memory) or `mapped` | `read` |
//! | `EIDOLA_DS_WHOLE_THREADS` | threads reading the experts | the machine's |
//! | `EIDOLA_DS_WHOLE_RESERVE_GB` | device memory left free beside the caches | 4 |
//! | `EIDOLA_DS_WHOLE_PROMPTS` | a prompts file in `ds_generate`'s form | `ds_generate`'s |
//! | `EIDOLA_DS_WHOLE_MAX_NEW` | new tokens per prompt, at most | 64 |
//! | `EIDOLA_DS_WHOLE_WIDTHS` | draft widths, comma-separated | `0,5` |
//! | `EIDOLA_DS_WHOLE_CHUNK` | positions per prefill step (`generate`) | 96 |
//! | `EIDOLA_DS_HALVES_GOLDEN` | as `ds_halves` (the text and the Engram map, `compare`) | as `ds_halves` |
//!
//! Before a device phase the same phase runs over a device with no memory (`ds::dry`),
//! so whatever arithmetic refuses is refused before anything is loaded. The process
//! exits 1 when anything compared differs and 2 on a failure.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, Slot, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cuda::ds::abi::EngramGatherParams;
use eidola_engine_cuda::ds::boundary::{GatheredRows, ToDecoder, ToEncoder};
use eidola_engine_cuda::ds::dry::DryDevice;
use eidola_engine_cuda::ds::executor::geometry;
use eidola_engine_cuda::ds::half::{CudaDevice, Device, bytes_of};
use eidola_engine_cuda::ds::kv::KvStore;
use eidola_engine_cuda::ds::model::HalfNet;
use eidola_engine_cuda::ds::plan::{PlanRow, StepPlan};
use eidola_engine_cuda::ds::staged::{self, Capacity, DevOnly, HostExperts, HostMode, StagedStats};
use eidola_engine_cuda::ds::support::DsShape;
use eidola_engine_cuda::ds::weights::Half;
use eidola_engine_cuda::ds::{DsCudaExecutor, DsCudaExecutorConfig};
use eidola_engine_cuda::{CudaError, KernelDir, Result};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::engram::EngramHasher;
use eidola_engine_model::deepseek::{DsConfig, config_of};
use serde_json::{Value, json};

fn fail(e: impl std::fmt::Display) -> CudaError {
    CudaError::new(e.to_string())
}

fn us(x: u32) -> usize {
    x as usize
}

fn n32(x: usize) -> u32 {
    u32::try_from(x).expect("a count of this run")
}

const TOKEN: fn() -> DevOnly = DevOnly::for_tests_and_dev_tools;

/// Slots of the comparison: the text's alone.
const SLOTS: u32 = 1;
/// Positions per KV block of the comparison, as `ds_halves`.
const BLOCK: u32 = 16;

/// Opens the device a half runs on.
type Open<'a> = &'a dyn Fn() -> Result<Box<dyn Device>>;
/// How many experts each half's cache holds, asked once both halves are loaded.
type Size<'a> = &'a dyn Fn(usize) -> Result<[u32; 2]>;

/// The comparison: the text, its steps, and the two halves together.
struct Compare {
    store: Option<Arc<WeightSet>>,
    c: DsConfig,
    shape: DsShape,
    spec: ModelSpec,
    tokens: Vec<u32>,
    hasher: Option<EngramHasher>,
    chunk: u32,
    out: PathBuf,
}

/// What one step of both halves left.
struct Step {
    items: Vec<(Slot, usize)>,
    encoder: Vec<Vec<u16>>,
    crossing: ToDecoder,
    decoder: Vec<Vec<u16>>,
    logits: Vec<f32>,
    seconds: [f64; 4],
}

impl Compare {
    fn new(
        store: Option<Arc<WeightSet>>,
        c: DsConfig,
        tokens: Vec<u32>,
        token_map: Vec<u32>,
        chunk: u32,
        out: PathBuf,
    ) -> Result<Compare> {
        let positions = n32(tokens.len()).next_multiple_of(BLOCK);
        let cfg = DsCudaExecutorConfig {
            block_size: BLOCK,
            num_blocks: positions / BLOCK + 2,
            window_blocks: positions / BLOCK + 2,
            num_state_slots: SLOTS,
            max_model_len: positions,
            buckets: vec![Bucket {
                max_seqs: chunk,
                max_tokens: chunk,
            }],
            sampleable_vocab_size: n32(c.vocab_size),
            devices: [0, 0],
            draft_width: 0,
            host_blocks: 0,
            position_tags: false,
            record: false,
        };
        let (shape, spec) = DsCudaExecutor::preflight(&c, &cfg)?;
        std::fs::create_dir_all(&out).map_err(fail)?;
        let hasher = match &c.engram {
            Some(e) => Some(EngramHasher::new(e, c.vocab_size, token_map).map_err(fail)?),
            None => None,
        };
        Ok(Compare {
            store,
            c,
            shape,
            spec,
            tokens,
            hasher,
            chunk,
            out,
        })
    }

    fn width(&self) -> usize {
        self.c.hc_mult * self.c.dim
    }

    fn entry_bytes(&self) -> usize {
        let (w, i) = (self.c.attention.head_dim, self.c.indexer.head_dim);
        w / 2 + w / 16 + i / 2 + i / 32
    }

    fn chunks(&self) -> Vec<(usize, usize)> {
        (0..self.tokens.len())
            .step_by(us(self.chunk))
            .map(|from| (from, (from + us(self.chunk)).min(self.tokens.len())))
            .collect()
    }

    /// One half, staged, with every block of the sequence mapped.
    fn half(&self, half: Half, dev: Box<dyn Device>) -> Result<HalfNet> {
        let positions = self.spec.max_model_len;
        let w = staged::half_weights(
            dev.as_ref(),
            self.store.as_deref(),
            &self.c,
            &self.shape,
            half,
            positions,
            TOKEN(),
        )?;
        let kv = KvStore::new(
            dev.as_ref(),
            geometry(&self.c, &self.spec, half)?,
            SLOTS,
            self.spec.max_blocks_per_seq(),
        )?;
        let mut n = HalfNet::new(
            half,
            dev,
            w,
            kv,
            &self.c,
            &self.shape,
            positions,
            self.chunk,
            self.chunk,
            self.chunk,
            self.spec.sampleable_vocab_size,
            self.shape.open_cap(),
        )?;
        for g in 0..n.kv.groups.len() {
            for index in 0..self.spec.max_blocks_per_seq() {
                n.kv.set_table(g, 0, index, index + 1);
            }
        }
        n.kv.flush_tables(n.dev.as_ref())?;
        Ok(n)
    }

    fn plan(&self, from: usize, to: usize) -> (Vec<PlanRow>, StepPlan) {
        let rows = vec![PlanRow {
            slot: 0,
            c: n32(from),
            p: n32(to - 1),
            sample: false,
            chain_start: n32(from),
            drafts: 0,
        }];
        let chain = self.spec.window_chain.expect("a window chain");
        let plan = StepPlan::new(&rows, chain.depth, chain.window);
        (rows, plan)
    }

    /// Per Engram layer of the config, the table rows of `items`' positions.
    fn engram_ids(&self, items: &[(Slot, usize)]) -> Result<Vec<Vec<u32>>> {
        let (Some(e), Some(hasher)) = (&self.c.engram, &self.hasher) else {
            return Ok(Vec::new());
        };
        let mut out = vec![Vec::new(); e.layers.len()];
        for (slot, spec) in e.layers.iter().enumerate() {
            if self.shape.engram_layer[slot].is_none() {
                continue;
            }
            for &(_, pos) in items {
                let lookback = &self.tokens[(pos + 1).saturating_sub(e.max_ngram)..=pos];
                for row in hasher.rows(spec, lookback, pos) {
                    out[slot].push(u32::try_from(row).map_err(fail)?);
                }
            }
        }
        Ok(out)
    }

    /// The decoder half's part of the Engram lookups, gathered from the tables on its
    /// device (as the executor's step gathers them).
    fn gather(&self, n: &HalfNet, ids: &[Vec<u32>], rows: u32) -> Result<ToEncoder> {
        let dev = n.dev.as_ref();
        let mut gathered = vec![None; ids.len()];
        for (slot, ids) in ids.iter().enumerate() {
            let Some(table) = n.w.tables.get(slot).copied().flatten() else {
                continue;
            };
            n.s.eids.write(dev, 0, ids)?;
            let (codes, scales) = &n.s.gathered[slot];
            // SAFETY: the table, and `rows` rows of ids and of gathered rows.
            unsafe {
                dev.kernels().engram_gather(&EngramGatherParams {
                    codes: table.codes,
                    scales: table.scales,
                    ids: n.s.eids.addr,
                    out_codes: codes.addr,
                    out_scales: scales.addr,
                    status: n.s.status.addr,
                    table_rows: table.rows,
                    rows,
                    ids_per_row: n.d.lookups,
                    width: n.d.engram_width / n.d.lookups,
                    reserved: 0,
                })?;
            }
            let width = us(n.d.engram_width);
            gathered[slot] = Some(GatheredRows {
                codes: codes.read(dev, 0, us(rows) * width)?,
                scales: scales.read(dev, 0, us(rows) * width / 32)?,
            });
        }
        Ok(ToEncoder { gathered })
    }

    /// Positions `from .. to` through both halves.
    fn step(&self, enc: &mut HalfNet, dec: &mut HalfNet, from: usize, to: usize) -> Result<Step> {
        let began = Instant::now();
        let (plan_rows, plan) = self.plan(from, to);
        let items: Vec<(Slot, usize)> = plan.items.iter().map(|it| (it.slot, us(it.pos))).collect();
        let tokens: Vec<u32> = items.iter().map(|(_, p)| self.tokens[*p]).collect();
        let (count, width) = (items.len(), self.width());
        let ids = self.engram_ids(&items)?;
        let to_encoder = self.gather(dec, &ids, n32(count))?;
        let gather = began.elapsed().as_secs_f64();
        let at = Instant::now();
        let mut encoder = vec![Vec::new(); self.shape.split];
        let crossing = DsCudaExecutor::encode_half(
            enc,
            &self.c,
            &self.shape,
            &plan,
            &plan_rows,
            &tokens,
            &ids,
            &to_encoder,
            &mut |l, n: &HalfNet| {
                encoder[l] = n.s.x.read::<u16>(n.dev.as_ref(), 0, count * width)?;
                Ok(())
            },
        )?;
        assert_eq!(us(crossing.rows), count, "every position's row crosses");
        let encode = at.elapsed().as_secs_f64();

        // The decoder reads every position's logits.
        let at = Instant::now();
        let (_, mut dplan) = self.plan(from, to);
        for it in &mut dplan.items {
            it.top = true;
        }
        dplan.top_rows = n32(count);
        assert_eq!(
            dplan
                .items
                .iter()
                .map(|it| (it.slot, it.pos))
                .collect::<Vec<_>>(),
            crossing.positions,
            "the head's rows keep the crossing's order"
        );
        crossing.check(&dplan, width, self.c.hc_mult, 0)?;
        let split = self.shape.split;
        let mut decoder = vec![Vec::new(); self.shape.depth()];
        let top = DsCudaExecutor::decode_half(
            dec,
            &self.c,
            &self.shape,
            &dplan,
            &crossing,
            &mut |l, n: &HalfNet| {
                decoder[l - split] = n.s.x.read::<u16>(n.dev.as_ref(), 0, count * width)?;
                Ok(())
            },
        )?;
        assert_eq!(us(top), count);
        let logits = dec
            .s
            .logits
            .read::<f32>(dec.dev.as_ref(), 0, count * self.c.vocab_size)?;
        let status = dec.status()?;
        if status != 0 {
            return Err(fail(format!("the decoder raised status {status:#x}")));
        }
        let decode = at.elapsed().as_secs_f64();
        Ok(Step {
            items,
            encoder,
            crossing,
            decoder,
            logits,
            seconds: [gather, encode, decode, began.elapsed().as_secs_f64()],
        })
    }

    /// Both halves loaded staged on devices from `open`, over every chunk; the
    /// recordings written, and compared with `recorded`'s when given.
    fn run(
        &self,
        open: Open<'_>,
        size: Size<'_>,
        host: &Arc<HostExperts>,
        recorded: Option<&Path>,
    ) -> Result<(Value, bool)> {
        let began = Instant::now();
        let mut enc = self.half(Half::Encoder, open()?)?;
        let mut dec = self.half(Half::Decoder, open()?)?;
        enc.dev.synchronize()?;
        dec.dev.synchronize()?;
        let load = began.elapsed().as_secs_f64();
        let caps = size(host.expert_bytes())?;
        staged::attach(&mut enc, &self.shape, host, caps[0], TOKEN())?;
        staged::attach(&mut dec, &self.shape, host, caps[1], TOKEN())?;
        let memory = json!([enc.memory().total(), dec.memory().total()]);
        eprintln!("both halves loaded in {load:.0} s; caches of {caps:?} experts; bytes {memory}");
        let (split, depth) = (self.shape.split, self.shape.depth());
        let mut encoder: Vec<Vec<u16>> = vec![Vec::new(); split];
        let mut decoder: Vec<Vec<u16>> = vec![Vec::new(); depth];
        let (mut entries, mut streams, mut pre, mut logits) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut steps = Vec::new();
        for (from, to) in self.chunks() {
            let before =
                [&enc, &dec].map(|n| n.staged.as_ref().map(|c| c.stats()).unwrap_or_default());
            let s = self.step(&mut enc, &mut dec, from, to)?;
            assert!(s.items.iter().all(|(slot, _)| *slot == 0));
            for (all, layer) in encoder.iter_mut().zip(&s.encoder) {
                all.extend_from_slice(layer);
            }
            for (all, layer) in decoder.iter_mut().zip(&s.decoder) {
                all.extend_from_slice(layer);
            }
            entries.extend_from_slice(&s.crossing.entries);
            streams.extend_from_slice(&s.crossing.streams);
            pre.extend_from_slice(&s.crossing.pre_mix);
            logits.extend_from_slice(&s.logits);
            let after =
                [&enc, &dec].map(|n| n.staged.as_ref().map(|c| c.stats()).unwrap_or_default());
            eprintln!("{from}..{to}: {:.2} s", s.seconds[3]);
            steps.push(json!({
                "positions": to - from,
                "seconds": s.seconds[3], "gather_seconds": s.seconds[0],
                "encoder_seconds": s.seconds[1], "decoder_seconds": s.seconds[2],
                "cache": [stats_delta(&before[0], &after[0]), stats_delta(&before[1], &after[1])],
            }));
        }
        let write =
            |name: &str, bytes: &[u8]| std::fs::write(self.out.join(name), bytes).map_err(fail);
        write("encoder_layers.bin", bytes_of(&encoder.concat()))?;
        write("boundary_entries.bin", &entries)?;
        write("boundary_streams.bin", bytes_of(&streams))?;
        write("boundary_pre.bin", bytes_of(&pre))?;
        write("decoder_layers.bin", bytes_of(&decoder.concat()))?;
        write("logits.bin", bytes_of(&logits))?;
        let mut checks = Vec::new();
        let mut equal = true;
        if let Some(dir) = recorded {
            let read = |name: &str| {
                std::fs::read(dir.join(name)).map_err(|e| fail(format!("{name}: {e}")))
            };
            let (n, width, eb, vocab) = (
                self.tokens.len(),
                self.width(),
                self.entry_bytes(),
                self.c.vocab_size,
            );
            let mut check = |name: String, got: Vec<u32>, want: Vec<u32>, row: usize| {
                let c = first_difference(&name, &got, &want, row);
                equal &= c["equal"] == json!(true);
                eprintln!("{c}");
                checks.push(c);
            };
            let layers = u16s(&read("encoder_layers.bin")?);
            for (i, got) in encoder.iter().enumerate() {
                let want = layers
                    .get(i * n * width..(i + 1) * n * width)
                    .unwrap_or(&[]);
                check(
                    format!("streams after encoder layer {i}"),
                    bf16_bits(got),
                    bf16_bits(want),
                    width,
                );
            }
            check(
                "bank entries".into(),
                entries.iter().map(|b| u32::from(*b)).collect(),
                read("boundary_entries.bin")?
                    .iter()
                    .map(|b| u32::from(*b))
                    .collect(),
                eb,
            );
            check(
                "boundary streams".into(),
                bf16_bits(&streams),
                bf16_bits(&u16s(&read("boundary_streams.bin")?)),
                width,
            );
            check(
                "boundary collapse weights".into(),
                u32s(bytes_of(&pre)),
                u32s(&read("boundary_pre.bin")?),
                self.c.hc_mult,
            );
            let layers = u16s(&read("decoder_layers.bin")?);
            for (j, got) in decoder.iter().enumerate() {
                let want = layers
                    .get(j * n * width..(j + 1) * n * width)
                    .unwrap_or(&[]);
                check(
                    format!("streams after decoder layer {}", split + j),
                    bf16_bits(got),
                    bf16_bits(want),
                    width,
                );
            }
            check(
                "logits".into(),
                u32s(bytes_of(&logits)),
                u32s(&read("logits.bin")?),
                vocab,
            );
        }
        let cache = [&enc, &dec].map(|n| n.staged.as_ref().map(|c| stats_json(&c.stats())));
        Ok((
            json!({
                "load_seconds": load, "capacity": caps, "device_bytes": memory,
                "steps": steps, "cache": cache,
                "recorded": recorded.map(|p| p.display().to_string()),
                "equal": equal, "checks": checks,
            }),
            equal,
        ))
    }
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bf16_bits(bits: &[u16]) -> Vec<u32> {
    bits.iter().map(|b| u32::from(*b) << 16).collect()
}

/// Whether two arrays of rows of `row` elements are equal bit for bit, and where they
/// first differ.
fn first_difference(name: &str, got: &[u32], want: &[u32], row: usize) -> Value {
    if got.len() != want.len() {
        return json!({"what": name, "equal": false, "lengths": [got.len(), want.len()]});
    }
    let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let first = got.iter().zip(want).position(|(a, b)| a != b);
    json!({
        "what": name,
        "equal": differing == 0,
        "elements": got.len(),
        "differing": differing,
        "first": first.map(|at| json!({
            "position": at / row, "element": at % row,
            "got": format!("{:#010x}", got[at]), "want": format!("{:#010x}", want[at]),
        })),
    })
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

/// One prompt to generate from: its name and tokens.
struct Prompt {
    name: String,
    tokens: Vec<u32>,
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
    }
}

/// The text and the Engram map's file: the one named, else the checkpoint's pinned
/// golden, else any of its reference goldens (as `ds_halves`).
fn golden(dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("EIDOLA_DS_HALVES_GOLDEN") {
        return PathBuf::from(path);
    }
    let golden = dir.join("golden");
    let pinned = golden.join("reference-pinned.safetensors");
    if pinned.exists() {
        return pinned;
    }
    let mut others: Vec<PathBuf> = std::fs::read_dir(&golden)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("reference-") && f.ends_with(".safetensors"))
        })
        .collect();
    others.sort();
    others.into_iter().next().unwrap_or(pinned)
}

/// The rendered prompts, the end-of-sequence ids, the tokenizer and the Engram map.
type Rendered = (
    Vec<Prompt>,
    Vec<u32>,
    eidola_engine_chat::Tokenizer,
    Vec<u32>,
);

/// The prompts of `ds_generate`'s file, rendered and tokenized as the node does.
fn prompts(dir: &Path, file: &Path) -> std::result::Result<Rendered, String> {
    use eidola_engine_chat::deepseek_v4_1::{self, EncodeOptions};
    use eidola_engine_chat::{ChatFormat, ChatInput, Tokenizer};
    let tokenizer = Tokenizer::for_format_from_model_dir(ChatFormat::DeepSeekV4_1, dir)
        .map_err(|e| e.to_string())?;
    let map = deepseek_v4_1::engram::compressed_vocab(&tokenizer)
        .map_err(|e| e.to_string())?
        .map;
    let text = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let specs: Vec<Value> = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
    let options = EncodeOptions::from_request(None, None).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for s in specs {
        let name = s["name"]
            .as_str()
            .ok_or("a prompt without a name")?
            .to_owned();
        let messages = serde_json::to_string(&s["messages"]).map_err(|e| e.to_string())?;
        let tools = s
            .get("tools")
            .map(|t| serde_json::to_string(t).expect("JSON"));
        let input = ChatInput::from_json(&messages, tools.as_deref()).map_err(|e| e.to_string())?;
        let prompt = deepseek_v4_1::encode(&input, options).map_err(|e| e.to_string())?;
        let tokens = tokenizer
            .encode_prompt(&prompt)
            .map_err(|e| e.to_string())?;
        out.push(Prompt { name, tokens });
    }
    let eos = tokenizer.eos_token_ids().to_vec();
    Ok((out, eos, tokenizer, map))
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
    let shape = eidola_engine_cuda::ds::support::check_supported(&c, 0).expect("a supported model");
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
                let dry = Compare::new(
                    None,
                    c.clone(),
                    tokens.clone(),
                    map.clone(),
                    chunk,
                    out.join("dry"),
                )?;
                let open_dry = || -> Result<Box<dyn Device>> { Ok(Box::new(DryDevice::new(None))) };
                dry.run(&open_dry, &|_| Ok([1600, 1600]), &host, None)?;
                eprintln!("dry run: every launch of both halves is accepted");
                let run = Compare::new(
                    Some(store.clone()),
                    c.clone(),
                    tokens,
                    map,
                    chunk,
                    out.clone(),
                )?;
                let k = kernels()?;
                let open = || -> Result<Box<dyn Device>> { Ok(Box::new(CudaDevice::open(0, &k)?)) };
                let (mut v, same) = run.run(&open, &fill, &host, Some(&recorded))?;
                v["tokens"] = json!(count);
                v["chunk"] = json!(chunk);
                Ok((v, same))
            })(),
            "generate" => (|| -> Result<(Value, bool)> {
                let file = var("EIDOLA_DS_WHOLE_PROMPTS").map_or_else(
                    || {
                        PathBuf::from(concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/../eidola-server-engine/examples/ds_generate/prompts.json"
                        ))
                    },
                    PathBuf::from,
                );
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

    /// The comparison on the fixture over the host device: recorded with a cache of every
    /// expert, then run again with caches of one row's experts and compared equal; a
    /// flipped recorded bit is named by array, position and element.
    #[test]
    fn the_fixture_staged_equals_its_recording() {
        let (store, c, map) = fixture();
        let shape = eidola_engine_cuda::ds::support::check_supported(&c, 0).unwrap();
        let host = Arc::new(
            HostExperts::new(&store, &c, &shape, HostMode::Read { threads: 2 }, TOKEN()).unwrap(),
        );
        let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37 + 11) % 95).collect();
        let base = std::env::temp_dir().join(format!("ds_whole_fixture_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let open = || -> Result<Box<dyn Device>> { Ok(Box::new(HostDevice)) };
        let first = Compare::new(
            Some(store.clone()),
            c.clone(),
            tokens.clone(),
            map.clone(),
            16,
            base.join("all"),
        )
        .unwrap();
        let (v, _) = first.run(&open, &|_| Ok([32, 32]), &host, None).unwrap();
        assert_eq!(v["steps"].as_array().unwrap().len(), 5);
        assert_eq!(v["cache"][0]["evictions"], 0);
        let again = Compare::new(
            Some(store.clone()),
            c.clone(),
            tokens,
            map,
            16,
            base.join("two"),
        )
        .unwrap();
        let (v, equal) = again
            .run(&open, &|_| Ok([2, 2]), &host, Some(&base.join("all")))
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
        let width = first.width();
        // Position 3 of decoder layer 5 (the second of the decoder), element 9.
        let at = 2 * (70 * width + 3 * width + 9);
        layers[at] ^= 1;
        std::fs::write(&path, layers).unwrap();
        let (v, equal) = again
            .run(&open, &|_| Ok([3, 2]), &host, Some(&base.join("all")))
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
        let shape = eidola_engine_cuda::ds::support::check_supported(&c, 4).unwrap();
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
