//! The released DeepSeek-V4.1 checkpoint on the exact executor, one half at a time on
//! one GPU, against the model crate's forward on the CPU. Dev-only.
//!
//! A device holds one half of the released model, not both. So the halves run in turn
//! through the executor's own step path (`DsCudaExecutor::encode_half` /
//! `decode_half`): the encoder half over real text in a few chunks, with every position
//! in the chain so that its whole boundary is recorded (the split layer's bank entry and
//! the boundary row of every position) beside each layer's residual copies; then, the
//! encoder freed, the decoder half teacher-forced from that recording, with logits at
//! every position. The Engram table the decoder's device would hold stays off it: its
//! rows are gathered on the host from the mapped checkpoint, as the boundary carries
//! them. The reference is `DsModel` over the same tokens: the encoder layers and the
//! bank stub, then the decoder from the recorded boundary. Everything is compared bit
//! for bit, and the first difference is named by layer, position and element.
//!
//! Each device phase ends with an invariance pass: the text's first positions run again
//! in two other slots, in other blocks, side by side in one step with the rows in the
//! other order, and cut into steps at other places. Every layer's streams, the boundary
//! and the logits of both must equal the first run's bit for bit: a position's bits do
//! not depend on its slot, its blocks, its row or the rows beside it.
//!
//! ```sh
//! EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> EIDOLA_DEEPSEEK_FULL_DIR=<checkpoint> \
//! EIDOLA_DS_HALVES_OUT=<dir> \
//!     cargo run --release -p eidola-engine-cuda --example ds_halves
//! ```
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `EIDOLA_DS_HALVES_TOKENS` | tokens of the text to run | 256 |
//! | `EIDOLA_DS_HALVES_CHUNK` | positions per step | 96 |
//! | `EIDOLA_DS_HALVES_INVARIANCE` | positions the invariance pass repeats; 0 for none | twice the chunk |
//! | `EIDOLA_DS_HALVES_PHASES` | of `encoder`, `decoder`, `reference`, in order | all three |
//! | `EIDOLA_DS_HALVES_GOLDEN` | a safetensors file holding `tokens` and `engram.token_map` | `<checkpoint>/golden/reference-pinned.safetensors`, else the first `reference-*.safetensors` there |
//! | `EIDOLA_DS_HALVES_OUT` | recordings and `ds_halves.json` | `ds_halves` in the working directory |
//!
//! Before a device phase the same phases run over a device with no memory (`ds::dry`),
//! so anything arithmetic can refuse is refused before a half is loaded. Each phase
//! reads what the phases before it left in the output directory, so they can
//! run in separate invocations (the device phases while a device is free, the reference
//! on the CPU afterwards). The process exits non-zero when anything compared differs.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::Slot;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cuda::ds::boundary::{GatheredRows, ToDecoder, ToEncoder};
use eidola_engine_cuda::ds::dry::DryDevice;
use eidola_engine_cuda::ds::executor::geometry;
use eidola_engine_cuda::ds::gemm::GemmForm;
use eidola_engine_cuda::ds::half::{CudaDevice, Device, bytes_of};
use eidola_engine_cuda::ds::kv::KvStore;
use eidola_engine_cuda::ds::model::HalfNet;
use eidola_engine_cuda::ds::plan::{PlanRow, StepPlan};
use eidola_engine_cuda::ds::support::DsShape;
use eidola_engine_cuda::ds::weights::{Form, Half, HalfWeights, weight_bytes};
use eidola_engine_cuda::ds::{DsCudaExecutor, DsCudaExecutorConfig};
use eidola_engine_cuda::{CudaError, KernelDir, Result};
use eidola_engine_model::WeightSet;
use eidola_engine_model::compare::compare_logits;
use eidola_engine_model::deepseek::engram::EngramHasher;
use eidola_engine_model::deepseek::forward::{
    Boundary, BoundaryBank, DsForwardOptions, GlobalBank, Rows,
};
use eidola_engine_model::deepseek::numeric::bf16_round;
use eidola_engine_model::deepseek::{DsConfig, DsLoadOptions, DsModel, DsWeights, config_of};
use eidola_engine_model::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};
use eidola_engine_model::tensor::Matrix;
use serde_json::{Value, json};

/// What a run is over.
struct Run {
    /// The checkpoint; none lays the halves out without it (the dry run).
    store: Option<Arc<WeightSet>>,
    c: DsConfig,
    shape: DsShape,
    spec: ModelSpec,
    tokens: Vec<u32>,
    token_map: Vec<u32>,
    hasher: Option<EngramHasher>,
    chunk: u32,
    /// Positions the invariance pass runs again, in two other slots.
    again: usize,
    out: PathBuf,
}

/// Slots of a half: the run's, then the invariance pass's two.
const SLOTS: u32 = 3;
/// Positions per KV block.
const BLOCK: u32 = 16;

/// One step's rows as `(slot, from, to)`.
type StepRows = [(Slot, usize, usize)];

/// What an encoder step left.
struct Encoded {
    /// The step's positions in device row order.
    items: Vec<(Slot, usize)>,
    /// Per encoder layer: the residual copies leaving it, per item.
    layers: Vec<Vec<u16>>,
    crossing: ToDecoder,
    gather_seconds: f64,
    compute_seconds: f64,
    readback_seconds: f64,
    seconds: f64,
}

/// What a decoder step left.
struct Decoded {
    items: Vec<(Slot, usize)>,
    layers: Vec<Vec<u16>>,
    logits: Vec<f32>,
    compute_seconds: f64,
    readback_seconds: f64,
    seconds: f64,
}

/// Rows of `width` elements placed by position.
#[derive(Clone)]
struct ByPosition<T> {
    width: usize,
    data: Vec<T>,
}

impl<T: Copy + Default> ByPosition<T> {
    fn new(positions: usize, width: usize) -> Self {
        ByPosition {
            width,
            data: vec![T::default(); positions * width],
        }
    }

    /// Places the rows of `rows` (one per item) belonging to `slot`.
    fn place(&mut self, items: &[(Slot, usize)], slot: Slot, rows: &[T]) {
        for (i, (s, pos)) in items.iter().enumerate() {
            if *s == slot {
                self.data[pos * self.width..(pos + 1) * self.width]
                    .copy_from_slice(&rows[i * self.width..(i + 1) * self.width]);
            }
        }
    }
}

/// Threads the host gather of Engram rows reads the mapped checkpoint on.
const GATHER_THREADS: usize = 32;

fn fail(e: impl std::fmt::Display) -> CudaError {
    CudaError::new(e.to_string())
}

fn us(x: u32) -> usize {
    x as usize
}

fn n32(x: usize) -> u32 {
    u32::try_from(x).expect("a count of this run")
}

impl Run {
    fn new(
        store: Option<Arc<WeightSet>>,
        c: DsConfig,
        tokens: Vec<u32>,
        token_map: Vec<u32>,
        chunk: u32,
        again: usize,
        out: PathBuf,
    ) -> Result<Run> {
        let block_size = BLOCK;
        let positions = n32(tokens.len()).next_multiple_of(block_size);
        let again = again.min(tokens.len());
        // The sequence's blocks, the null block, and the invariance pass's.
        let blocks = positions / block_size + 2 + 2 * n32(again).div_ceil(block_size);
        let cfg = DsCudaExecutorConfig {
            block_size,
            num_blocks: blocks,
            window_blocks: blocks,
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
            gemm: GemmForm::from_env().map_err(fail)?,
        };
        let (shape, spec) = DsCudaExecutor::preflight(&c, &cfg)?;
        std::fs::create_dir_all(&out).map_err(fail)?;
        let hasher = match &c.engram {
            Some(e) => Some(EngramHasher::new(e, c.vocab_size, token_map.clone()).map_err(fail)?),
            None => None,
        };
        Ok(Run {
            store,
            c,
            shape,
            spec,
            tokens,
            token_map,
            hasher,
            chunk,
            again,
            out,
        })
    }

    fn n(&self) -> usize {
        self.tokens.len()
    }

    /// Elements of a position's residual copies.
    fn width(&self) -> usize {
        self.c.hc_mult * self.c.dim
    }

    fn entry_bytes(&self) -> usize {
        let (w, i) = (self.c.attention.head_dim, self.c.indexer.head_dim);
        w / 2 + w / 16 + i / 2 + i / 32
    }

    fn write<T: eidola_engine_cuda::ds::half::Plain>(&self, name: &str, data: &[T]) -> Result<()> {
        std::fs::write(self.out.join(name), bytes_of(data)).map_err(fail)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        std::fs::read(self.out.join(name)).map_err(|e| fail(format!("{name}: {e}")))
    }

    fn read_u16(&self, name: &str) -> Result<Vec<u16>> {
        Ok(self
            .read(name)?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self
            .read(name)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// One half on `dev`: its weights, pools with every block of the sequence mapped,
    /// and scratch for a chunk. Returns the half and the seconds its weights took.
    fn half(&self, half: Half, dev: Box<dyn Device>) -> Result<(HalfNet, f64)> {
        let began = Instant::now();
        let positions = self.spec.max_model_len;
        // The Engram tables of the decoder's device are gathered on the host here.
        let w = HalfWeights::load(
            dev.as_ref(),
            self.store.as_deref(),
            &self.c,
            &self.shape,
            half,
            positions,
            half == Half::Encoder,
        )?;
        dev.synchronize()?;
        let load = began.elapsed().as_secs_f64();
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
        // Slot 0 holds the sequence; the invariance pass's slots hold its first
        // positions in other blocks, slot 1's in descending order.
        let (all, again) = (
            self.spec.max_blocks_per_seq(),
            n32(self.again).div_ceil(BLOCK),
        );
        for g in 0..n.kv.groups.len() {
            for index in 0..all {
                n.kv.set_table(g, 0, index, index + 1);
            }
            for index in 0..again {
                n.kv.set_table(g, 1, index, all + again - index);
                n.kv.set_table(g, 2, index, all + again + 1 + index);
            }
        }
        n.kv.flush_tables(n.dev.as_ref())?;
        Ok((n, load))
    }

    /// The step of `rows`: every chain layer at every position of each.
    fn plan(&self, rows: &StepRows) -> (Vec<PlanRow>, StepPlan) {
        let rows: Vec<PlanRow> = rows
            .iter()
            .map(|&(slot, from, to)| PlanRow {
                slot,
                c: n32(from),
                p: n32(to - 1),
                sample: false,
                chain_start: n32(from),
                drafts: 0,
            })
            .collect();
        let chain = self.spec.window_chain.expect("a window chain");
        let plan = StepPlan::new(&rows, chain.depth, chain.window);
        (rows, plan)
    }

    fn chunks(&self) -> Vec<(usize, usize)> {
        (0..self.n())
            .step_by(us(self.chunk))
            .map(|from| (from, (from + us(self.chunk)).min(self.n())))
            .collect()
    }

    /// The invariance pass's steps: the first `again` positions once more in slots 1
    /// and 2, cut at other places than the run's chunks and at different places in each,
    /// slot 2's row listed first, the two side by side until the faster one ends.
    fn again_steps(&self) -> Vec<Vec<(Slot, usize, usize)>> {
        let chunk = us(self.chunk);
        let short = (chunk / 3).max(1);
        let long = chunk - short;
        let (mut one, mut two) = (0, 0);
        let mut steps = Vec::new();
        while one < self.again || (long > 0 && two < self.again) {
            let mut rows = Vec::new();
            if long > 0 && two < self.again {
                let to = (two + long).min(self.again);
                rows.push((2, two, to));
                two = to;
            }
            if one < self.again {
                let to = (one + short).min(self.again);
                rows.push((1, one, to));
                one = to;
            }
            steps.push(rows);
        }
        steps
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

    /// The rows of the Engram tables the decoder's device would hold, gathered from the
    /// mapped checkpoint as stored. The rows are scattered over a table of a hundred
    /// gigabytes that is mostly not in memory, so they are read on several threads.
    fn gathered(&self, ids: &[Vec<u32>]) -> Result<ToEncoder> {
        let mut gathered = vec![None; ids.len()];
        let Some(e) = &self.c.engram else {
            return Ok(ToEncoder { gathered });
        };
        for (slot, ids) in ids.iter().enumerate() {
            let (Some(layer), 1) = (
                self.shape.engram_layer[slot],
                self.shape.engram_device[slot],
            ) else {
                continue;
            };
            let (w, sw) = (e.head_dim, e.head_dim / 32);
            let mut rows = GatheredRows {
                codes: vec![0; ids.len() * w],
                scales: vec![0; ids.len() * sw],
            };
            if let Some(store) = &self.store {
                let p = format!("layers.{}.engram.embed", self.c.layers[layer].source_index);
                let codes = store.get(&format!("{p}.weight")).map_err(fail)?.data;
                let scales = store.get(&format!("{p}.scale")).map_err(fail)?.data;
                let per = ids.len().div_ceil(GATHER_THREADS).max(1);
                std::thread::scope(|scope| {
                    let parts = ids
                        .chunks(per)
                        .zip(rows.codes.chunks_mut(per * w))
                        .zip(rows.scales.chunks_mut(per * sw));
                    for ((ids, to_codes), to_scales) in parts {
                        scope.spawn(move || {
                            for (i, id) in ids.iter().enumerate() {
                                let id = us(*id);
                                to_codes[i * w..(i + 1) * w]
                                    .copy_from_slice(&codes[id * w..(id + 1) * w]);
                                to_scales[i * sw..(i + 1) * sw]
                                    .copy_from_slice(&scales[id * sw..(id + 1) * sw]);
                            }
                        });
                    }
                });
            }
            gathered[slot] = Some(rows);
        }
        Ok(ToEncoder { gathered })
    }

    fn memory(&self, n: &HalfNet, load_seconds: f64) -> Value {
        let m = n.memory();
        let w = n.w.bytes;
        let checkpoint = weight_bytes(&self.c, &self.shape, n.half, Form::Checkpoint);
        json!({
            "load_seconds": load_seconds,
            "bytes": {
                "weights": m.weights, "kv": m.kv, "tables": m.tables, "state": m.state,
                "scratch": m.scratch, "total": m.total(),
            },
            "weights": {
                "embed": w.embed, "experts": w.experts, "fp8": w.fp8, "bf16": w.bf16,
                "f32": w.f32, "engram_tables": w.engram_tables,
                "engram_modules": w.engram_modules, "head": w.head, "rope": w.rope,
            },
            "checkpoint_bytes_of_this_half": checkpoint.total(),
            "checkpoint_engram_tables_of_this_half": checkpoint.engram_tables,
            "device_capacity": n.dev.capacity(),
        })
    }

    /// One encoder step over `rows`. The device's queue is drained before each layer's
    /// streams are read, so the time the layers take and the time the read-backs take
    /// are told apart.
    fn encode(&self, n: &mut HalfNet, rows: &StepRows) -> Result<Encoded> {
        let began = Instant::now();
        let (plan_rows, plan) = self.plan(rows);
        let items: Vec<(Slot, usize)> = plan.items.iter().map(|it| (it.slot, us(it.pos))).collect();
        let tokens: Vec<u32> = items.iter().map(|(_, pos)| self.tokens[*pos]).collect();
        let ids = self.engram_ids(&items)?;
        let to_encoder = self.gathered(&ids)?;
        let gather_seconds = began.elapsed().as_secs_f64();
        let (count, width) = (items.len(), self.width());
        let mut layers: Vec<Vec<u16>> = vec![Vec::new(); self.shape.split];
        let (mut compute_seconds, mut readback_seconds) = (0.0, 0.0);
        let crossing = DsCudaExecutor::encode_half(
            n,
            &self.c,
            &self.shape,
            &plan,
            &plan_rows,
            &tokens,
            &ids,
            &to_encoder,
            &mut |l, n: &HalfNet| {
                let at = Instant::now();
                n.dev.synchronize()?;
                compute_seconds += at.elapsed().as_secs_f64();
                let at = Instant::now();
                layers[l] = n.s.x.read::<u16>(n.dev.as_ref(), 0, count * width)?;
                readback_seconds += at.elapsed().as_secs_f64();
                Ok(())
            },
        )?;
        assert_eq!(us(crossing.rows), count, "every position's row crosses");
        assert_eq!(crossing.entry_bytes, self.entry_bytes());
        Ok(Encoded {
            items,
            layers,
            crossing,
            gather_seconds,
            compute_seconds,
            readback_seconds,
            seconds: began.elapsed().as_secs_f64(),
        })
    }

    /// The encoder half over the tokens: records the boundary and each layer's streams,
    /// then runs the invariance pass against them.
    fn encoder(&self, dev: Box<dyn Device>) -> Result<(Value, bool)> {
        let (mut n, load) = self.half(Half::Encoder, dev)?;
        let memory = self.memory(&n, load);
        eprintln!("encoder half loaded in {load:.0} s: {memory}");
        let (width, split, eb, copies) = (
            self.width(),
            self.shape.split,
            self.entry_bytes(),
            self.c.hc_mult,
        );
        let mut layers: Vec<Vec<u16>> = vec![Vec::new(); split];
        let (mut entries, mut streams, mut pre) = (Vec::new(), Vec::new(), Vec::new());
        let mut steps = Vec::new();
        for (from, to) in self.chunks() {
            let step = self.encode(&mut n, &[(0, from, to)])?;
            for (all, layer) in layers.iter_mut().zip(&step.layers) {
                all.extend_from_slice(layer);
            }
            entries.extend_from_slice(&step.crossing.entries);
            streams.extend_from_slice(&step.crossing.streams);
            pre.extend_from_slice(&step.crossing.pre_mix);
            eprintln!("encoder {from}..{to}: {:.2} s", step.seconds);
            steps.push(json!({
                "positions": to - from, "seconds": step.seconds,
                "host_gather_seconds": step.gather_seconds,
                "compute_seconds": step.compute_seconds,
                "tap_readback_seconds": step.readback_seconds,
            }));
        }

        // The first positions again, in other slots, other blocks, other steps.
        let m = self.again;
        let mut checks = Vec::new();
        let mut equal = true;
        let schedule = self.again_steps();
        let mut got: Vec<_> = (1..SLOTS)
            .map(|_| {
                (
                    vec![ByPosition::<u16>::new(m, width); split],
                    ByPosition::<u8>::new(m, eb),
                    ByPosition::<u16>::new(m, width),
                    ByPosition::<f32>::new(m, copies),
                )
            })
            .collect();
        for rows in &schedule {
            let step = self.encode(&mut n, rows)?;
            for (slot, (l, e, s, p)) in (1..SLOTS).zip(&mut got) {
                for (to, from) in l.iter_mut().zip(&step.layers) {
                    to.place(&step.items, slot, from);
                }
                e.place(&step.items, slot, &step.crossing.entries);
                s.place(&step.items, slot, &step.crossing.streams);
                p.place(&step.items, slot, &step.crossing.pre_mix);
            }
        }
        for (slot, (l, e, s, p)) in (1..SLOTS).zip(&got) {
            if schedule.iter().flatten().all(|r| r.0 != slot) {
                continue;
            }
            let mut check = |name: String, got: Vec<u32>, want: Vec<u32>, row: usize| {
                let c = first_difference(&name, &got, &want, row);
                equal &= c["equal"] == json!(true);
                eprintln!("{c}");
                checks.push(c);
            };
            for (i, layer) in l.iter().enumerate() {
                check(
                    format!("slot {slot}: streams after encoder layer {i}"),
                    bf16_bits(&layer.data),
                    bf16_bits(&layers[i][..m * width]),
                    width,
                );
            }
            check(
                format!("slot {slot}: bank entries"),
                e.data.iter().map(|b| u32::from(*b)).collect(),
                entries[..m * eb].iter().map(|b| u32::from(*b)).collect(),
                eb,
            );
            check(
                format!("slot {slot}: boundary streams"),
                bf16_bits(&s.data),
                bf16_bits(&streams[..m * width]),
                width,
            );
            check(
                format!("slot {slot}: boundary collapse weights"),
                f32_bits(&p.data),
                f32_bits(&pre[..m * copies]),
                copies,
            );
        }
        self.write("boundary_entries.bin", &entries)?;
        self.write("boundary_streams.bin", &streams)?;
        self.write("boundary_pre.bin", &pre)?;
        self.write("encoder_layers.bin", &layers.concat())?;
        Ok((
            json!({
                "memory": memory, "steps": steps, "tokens_per_second": rate(&steps),
                "invariance": {
                    "positions": m, "steps": schedule, "equal": equal, "checks": checks,
                },
            }),
            equal,
        ))
    }

    /// One decoder step over `rows`, teacher-forced from the recorded boundary, with
    /// logits at every position.
    fn decode(
        &self,
        n: &mut HalfNet,
        rows: &StepRows,
        entries: &[u8],
        streams: &[u16],
        pre: &[f32],
    ) -> Result<Decoded> {
        let began = Instant::now();
        let (width, copies, eb) = (self.width(), self.c.hc_mult, self.entry_bytes());
        let (_, mut plan) = self.plan(rows);
        // The head reads every position.
        for it in &mut plan.items {
            it.top = true;
        }
        let items: Vec<(Slot, usize)> = plan.items.iter().map(|it| (it.slot, us(it.pos))).collect();
        let count = items.len();
        plan.top_rows = n32(count);
        let gather = |from: &[u8], w: usize| -> Vec<u8> {
            items
                .iter()
                .flat_map(|(_, pos)| from[pos * w..(pos + 1) * w].iter().copied())
                .collect()
        };
        let crossing = ToDecoder {
            positions: plan.items.iter().map(|it| (it.slot, it.pos)).collect(),
            entry_bytes: eb,
            entries: gather(entries, eb),
            rows: n32(count),
            streams: items
                .iter()
                .flat_map(|(_, pos)| streams[pos * width..(pos + 1) * width].iter().copied())
                .collect(),
            pre_mix: items
                .iter()
                .flat_map(|(_, pos)| pre[pos * copies..(pos + 1) * copies].iter().copied())
                .collect(),
            tap_rows: 0,
            taps: Vec::new(),
        };
        crossing.check(&plan, width, copies, 0)?;
        let split = self.shape.split;
        let mut layers: Vec<Vec<u16>> = vec![Vec::new(); self.shape.depth()];
        let (mut compute_seconds, mut readback_seconds) = (0.0, 0.0);
        let top = DsCudaExecutor::decode_half(
            n,
            &self.c,
            &self.shape,
            &plan,
            &crossing,
            &mut |l, n: &HalfNet| {
                let at = Instant::now();
                n.dev.synchronize()?;
                compute_seconds += at.elapsed().as_secs_f64();
                let at = Instant::now();
                layers[l - split] = n.s.x.read::<u16>(n.dev.as_ref(), 0, count * width)?;
                readback_seconds += at.elapsed().as_secs_f64();
                Ok(())
            },
        )?;
        assert_eq!(us(top), count);
        let at = Instant::now();
        n.dev.synchronize()?;
        compute_seconds += at.elapsed().as_secs_f64();
        let at = Instant::now();
        let logits =
            n.s.logits
                .read::<f32>(n.dev.as_ref(), 0, count * self.c.vocab_size)?;
        readback_seconds += at.elapsed().as_secs_f64();
        let status = n.status()?;
        if status != 0 {
            return Err(fail(format!("the decoder raised status {status:#x}")));
        }
        Ok(Decoded {
            items,
            layers,
            logits,
            compute_seconds,
            readback_seconds,
            seconds: began.elapsed().as_secs_f64(),
        })
    }

    /// The decoder half, teacher-forced from the recorded boundary: logits everywhere,
    /// then the invariance pass against them.
    fn decoder(&self, dev: Box<dyn Device>) -> Result<(Value, bool)> {
        let entries = self.read("boundary_entries.bin")?;
        let streams = self.read_u16("boundary_streams.bin")?;
        let pre = self.read_f32("boundary_pre.bin")?;
        let (width, eb) = (self.width(), self.entry_bytes());
        if entries.len() != self.n() * eb || streams.len() != self.n() * width {
            return Err(fail("the recorded boundary is not this run's"));
        }
        let (mut n, load) = self.half(Half::Decoder, dev)?;
        let memory = self.memory(&n, load);
        eprintln!("decoder half loaded in {load:.0} s: {memory}");
        let depth = self.shape.depth();
        let vocab = self.c.vocab_size;
        let mut layers: Vec<Vec<u16>> = vec![Vec::new(); depth];
        let mut logits: Vec<f32> = Vec::new();
        let mut steps = Vec::new();
        for (from, to) in self.chunks() {
            let step = self.decode(&mut n, &[(0, from, to)], &entries, &streams, &pre)?;
            for (all, layer) in layers.iter_mut().zip(&step.layers) {
                all.extend_from_slice(layer);
            }
            logits.extend_from_slice(&step.logits);
            eprintln!("decoder {from}..{to}: {:.2} s", step.seconds);
            steps.push(json!({
                "positions": to - from, "seconds": step.seconds,
                "compute_seconds": step.compute_seconds,
                "readback_seconds": step.readback_seconds,
            }));
        }

        let m = self.again;
        let mut checks = Vec::new();
        let mut equal = true;
        let schedule = self.again_steps();
        let mut got: Vec<_> = (1..SLOTS)
            .map(|_| {
                (
                    vec![ByPosition::<u16>::new(m, width); depth],
                    ByPosition::<f32>::new(m, vocab),
                )
            })
            .collect();
        for rows in &schedule {
            let step = self.decode(&mut n, rows, &entries, &streams, &pre)?;
            for (slot, (l, lg)) in (1..SLOTS).zip(&mut got) {
                for (to, from) in l.iter_mut().zip(&step.layers) {
                    to.place(&step.items, slot, from);
                }
                lg.place(&step.items, slot, &step.logits);
            }
        }
        for (slot, (l, lg)) in (1..SLOTS).zip(&got) {
            if schedule.iter().flatten().all(|r| r.0 != slot) {
                continue;
            }
            let mut check = |name: String, got: Vec<u32>, want: Vec<u32>, row: usize| {
                let c = first_difference(&name, &got, &want, row);
                equal &= c["equal"] == json!(true);
                eprintln!("{c}");
                checks.push(c);
            };
            for (j, layer) in l.iter().enumerate() {
                check(
                    format!(
                        "slot {slot}: streams after decoder layer {}",
                        self.shape.split + j
                    ),
                    bf16_bits(&layer.data),
                    bf16_bits(&layers[j][..m * width]),
                    width,
                );
            }
            check(
                format!("slot {slot}: logits"),
                f32_bits(&lg.data),
                f32_bits(&logits[..m * vocab]),
                vocab,
            );
        }
        self.write("decoder_layers.bin", &layers.concat())?;
        self.write("logits.bin", &logits)?;
        Ok((
            json!({
                "memory": memory, "steps": steps, "tokens_per_second": rate(&steps),
                "invariance": {
                    "positions": m, "steps": schedule, "equal": equal, "checks": checks,
                },
            }),
            equal,
        ))
    }

    /// The model crate's forward over the same tokens against the recordings: the
    /// encoder's layers, its boundary and bank, then the decoder from the recorded
    /// boundary and its logits. Returns the comparison and whether everything is equal.
    fn reference(&self) -> Result<(Value, bool)> {
        let began = Instant::now();
        let store = self.store.clone().ok_or_else(|| fail("no checkpoint"))?;
        let weights = DsWeights::load(
            store,
            self.c.clone(),
            &DsLoadOptions {
                skip_drafter: true,
                ..DsLoadOptions::default()
            },
        )
        .map_err(fail)?;
        let model = DsModel::new(weights, Some(self.token_map.clone())).map_err(fail)?;
        let load = began.elapsed().as_secs_f64();
        let (n, width, split) = (self.n(), self.width(), self.shape.split);
        let opts = DsForwardOptions {
            taps: true,
            no_logits: true,
            ..DsForwardOptions::default()
        };
        let mut checks = Vec::new();
        let mut equal = true;
        let mut check = |name: String, got: Vec<u32>, want: Vec<u32>, row: usize| {
            let c = first_difference(&name, &got, &want, row);
            equal &= c["equal"] == json!(true);
            eprintln!("{c}");
            checks.push(c);
        };

        // The encoder: every layer's streams, the boundary rows, the bank.
        let at = Instant::now();
        let mut state = model.new_state();
        let rows = model.embed(&mut state, &self.tokens).map_err(fail)?;
        let enc = model
            .forward_layers(&mut state, rows, 0..split, &opts)
            .map_err(fail)?;
        let bank = model
            .bank_stub(split, &mut state, &enc.rows)
            .map_err(fail)?;
        let encoder_seconds = at.elapsed().as_secs_f64();
        let got = self.read_u16("encoder_layers.bin")?;
        for (l, tap) in enc.taps.iter().enumerate() {
            check(
                format!("streams after encoder layer {l}"),
                bf16_bits(&got[l * n * width..(l + 1) * n * width]),
                f32_bits(&tap.data),
                width,
            );
        }
        let streams = self.read_u16("boundary_streams.bin")?;
        let pre = self.read_f32("boundary_pre.bin")?;
        check(
            "boundary streams".into(),
            bf16_bits(&streams),
            f32_bits(&enc.rows.streams.data),
            width,
        );
        check(
            "boundary collapse weights".into(),
            f32_bits(&pre),
            f32_bits(&enc.rows.pre_mix.data),
            self.c.hc_mult,
        );
        let recorded = self.recorded_bank()?;
        let (hd, ihd) = (self.c.attention.head_dim, self.c.indexer.head_dim);
        check(
            "bank KV".into(),
            f32_bits(&recorded.kv),
            f32_bits(&bank.kv),
            hd,
        );
        check(
            "bank index keys".into(),
            f32_bits(&recorded.index_keys),
            f32_bits(&bank.index_keys),
            ihd,
        );

        // The decoder, from the recorded boundary.
        let at = Instant::now();
        let boundary = Boundary {
            rows: Rows {
                start: 0,
                streams: Matrix::from_vec(n, width, widen(&streams)),
                pre_mix: Matrix::from_vec(n, self.c.hc_mult, pre),
                engram_rows: enc.rows.engram_rows.clone(),
                selection: None,
                candidates: None,
            },
            bank: Some(BoundaryBank {
                layer: split,
                bank: recorded,
            }),
        };
        let dec = model
            .decode(
                &mut model.new_state(),
                &boundary,
                split,
                &DsForwardOptions {
                    taps: true,
                    ..DsForwardOptions::default()
                },
            )
            .map_err(fail)?;
        let decoder_seconds = at.elapsed().as_secs_f64();
        let got = self.read_u16("decoder_layers.bin")?;
        for (j, tap) in dec.taps.iter().enumerate() {
            check(
                format!("streams after decoder layer {}", split + j),
                bf16_bits(&got[j * n * width..(j + 1) * n * width]),
                f32_bits(&tap.data),
                width,
            );
        }
        let logits = self.read_f32("logits.bin")?;
        let vocab = self.c.vocab_size;
        check(
            "logits".into(),
            f32_bits(&logits),
            f32_bits(&dec.logits.data),
            vocab,
        );
        let agreement = match compare_logits(&dec.logits.data, &logits, vocab) {
            Ok(a) => a.to_string(),
            Err(e) => format!("not comparable: {e}"),
        };
        eprintln!("logits against the reference: {agreement}");
        Ok((
            json!({
                "equal": equal,
                "checks": checks,
                "logits_agreement": agreement,
                "cpu_seconds": {
                    "load": load, "encoder": encoder_seconds, "decoder": decoder_seconds,
                },
            }),
            equal,
        ))
    }

    /// The recorded bank entries as the values the reference holds.
    fn recorded_bank(&self) -> Result<GlobalBank> {
        let entries = self.read("boundary_entries.bin")?;
        let (hd, ihd) = (self.c.attention.head_dim, self.c.indexer.head_dim);
        let mut bank = GlobalBank {
            entries: self.n(),
            ..GlobalBank::default()
        };
        for e in entries.chunks_exact(self.entry_bytes()) {
            let (kv, key) = e.split_at(hd / 2 + hd / 16);
            for j in 0..hd {
                bank.kv.push(bf16_round(
                    nibble(kv, j) * fp8_e4m3_to_f32(kv[hd / 2 + j / 16]),
                ));
            }
            for j in 0..ihd {
                bank.index_keys.push(bf16_round(
                    nibble(key, j) * e8m0_to_f32(key[ihd / 2 + j / 32]),
                ));
            }
        }
        Ok(bank)
    }
}

fn nibble(bytes: &[u8], i: usize) -> f32 {
    E2M1_VALUES[usize::from(bytes[i / 2] >> (4 * (i % 2)) & 0xf)]
}

fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|b| f32::from_bits(u32::from(*b) << 16))
        .collect()
}

fn bf16_bits(bits: &[u16]) -> Vec<u32> {
    bits.iter().map(|b| u32::from(*b) << 16).collect()
}

fn f32_bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Positions per second over the steps.
fn rate(steps: &[Value]) -> f64 {
    let sum = |key: &str| steps.iter().filter_map(|s| s[key].as_f64()).sum::<f64>();
    sum("positions") / sum("seconds").max(f64::MIN_POSITIVE)
}

/// Whether two arrays of rows of `row` elements are equal bit for bit, and where they
/// first differ.
fn first_difference(name: &str, got: &[u32], want: &[u32], row: usize) -> Value {
    if got.len() != want.len() {
        return json!({
            "what": name, "equal": false,
            "lengths": [got.len(), want.len()],
        });
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

/// Runs `phases` in order; `device` opens the device a half runs on. Returns the
/// summary and whether every comparison made was equal.
fn run(
    run: &Run,
    phases: &[String],
    device: &dyn Fn() -> Result<Box<dyn Device>>,
) -> Result<(Value, bool)> {
    let path = run.out.join("ds_halves.json");
    let mut summary: Value = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}));
    summary["tokens"] = json!(run.n());
    summary["chunk"] = json!(run.chunk);
    summary["invariance_positions"] = json!(run.again);
    let mut equal = true;
    for phase in phases {
        let began = Instant::now();
        match phase.as_str() {
            "encoder" | "decoder" => {
                let (result, same) = match phase.as_str() {
                    "encoder" => run.encoder(device()?)?,
                    _ => run.decoder(device()?)?,
                };
                summary[phase.as_str()] = result;
                equal &= same;
            }
            "reference" => {
                let (result, same) = run.reference()?;
                summary["comparison"] = result;
                equal &= same;
            }
            other => return Err(fail(format!("unknown phase {other}"))),
        }
        eprintln!("{phase}: {:.0} s", began.elapsed().as_secs_f64());
        let text = serde_json::to_string_pretty(&summary).map_err(fail)?;
        std::fs::write(&path, text).map_err(fail)?;
    }
    Ok((summary, equal))
}

/// The file the tokens and the Engram map come from: the one named, else the
/// checkpoint's pinned golden, else any of its reference goldens.
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

fn main() {
    let var = |name: &str| std::env::var(name).ok();
    let dir = PathBuf::from(
        var("EIDOLA_DEEPSEEK_FULL_DIR").expect("EIDOLA_DEEPSEEK_FULL_DIR names the checkpoint"),
    );
    let count: usize = var("EIDOLA_DS_HALVES_TOKENS").map_or(256, |n| n.parse().expect("a count"));
    let chunk: u32 = var("EIDOLA_DS_HALVES_CHUNK").map_or(96, |n| n.parse().expect("a count"));
    let phases: Vec<String> = var("EIDOLA_DS_HALVES_PHASES")
        .unwrap_or_else(|| "encoder,decoder,reference".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let out = var("EIDOLA_DS_HALVES_OUT").map_or_else(|| PathBuf::from("ds_halves"), PathBuf::from);
    let again: usize = var("EIDOLA_DS_HALVES_INVARIANCE")
        .map_or(2 * chunk as usize, |n| n.parse().expect("a count"));

    let golden = WeightSet::open_files(&[golden(&dir)]).expect("the tokens and the Engram map");
    let ints = |name: &str| -> Vec<u32> {
        let v = golden.get(name).unwrap().to_i64(name).unwrap();
        v.into_iter().map(|t| u32::try_from(t).unwrap()).collect()
    };
    let mut tokens = ints("tokens");
    assert!(
        count <= tokens.len(),
        "the text has {} tokens",
        tokens.len()
    );
    tokens.truncate(count);
    let store = Arc::new(WeightSet::open_dir(&dir).expect("the checkpoint"));
    let c = config_of(&store).expect("the checkpoint's config");
    let map = ints("engram.token_map");
    // Before a half is loaded: the same phases with no memory behind them, so a count
    // that does not fit or a launch its wrapper refuses fails in a second.
    if phases.iter().any(|p| p != "reference") {
        let dry = Run::new(
            None,
            c.clone(),
            tokens.clone(),
            map.clone(),
            chunk,
            again,
            out.join("dry"),
        );
        let device = || -> Result<Box<dyn Device>> { Ok(Box::new(DryDevice::new(None))) };
        let both: Vec<String> = ["encoder", "decoder"].map(String::from).to_vec();
        if let Err(e) = dry.and_then(|dry| run(&dry, &both, &device)) {
            eprintln!("FAILED before any device was touched: {e}");
            std::process::exit(2);
        }
        eprintln!("dry run: every launch of both halves is accepted");
    }
    let setup = Run::new(Some(store), c, tokens, map, chunk, again, out).unwrap();
    let device = || -> Result<Box<dyn Device>> {
        let kernels = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR")
            .ok_or_else(|| fail("EIDOLA_ENGINE_KERNELS_DIR is not set"))?;
        let mut d = CudaDevice::open(0, &KernelDir::new(kernels))?;
        d.select_gemm(GemmForm::from_env()?)?;
        Ok(Box::new(d))
    };
    match run(&setup, &phases, &device) {
        Ok((summary, equal)) => {
            println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            println!("summary: {}", setup.out.join("ds_halves.json").display());
            if !equal {
                eprintln!("FAILED: something compared differs (see the summary)");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use eidola_engine_cuda::ds::host::HostDevice;

    use super::*;

    /// The whole flow on the committed fixture over the host device: both halves in
    /// turn, the recordings, and the reference equal bit for bit; and a flipped bit in
    /// a recording is named by layer, position and element.
    #[test]
    fn the_fixtures_halves_equal_the_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../eidola-engine-model/tests/fixtures/deepseek_tiny");
        let golden = WeightSet::open_files(&[dir.join("golden.safetensors")]).unwrap();
        let name = "engram.token_map";
        let map = golden.get(name).unwrap().to_i64(name).unwrap();
        let map: Vec<u32> = map.into_iter().map(|t| u32::try_from(t).unwrap()).collect();
        let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
        let c = config_of(&store).unwrap();
        let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37 + 11) % 95).collect();
        let out = std::env::temp_dir().join(format!("ds_halves_fixture_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let setup = Run::new(Some(store), c, tokens, map, 16, 40, out).unwrap();
        let device = || -> Result<Box<dyn Device>> { Ok(Box::new(HostDevice)) };
        let phases: Vec<String> = ["encoder", "decoder", "reference"]
            .map(String::from)
            .to_vec();
        let (summary, equal) = run(&setup, &phases, &device).unwrap();
        assert!(equal, "{summary:#}");
        let checks = summary["comparison"]["checks"].as_array().unwrap();
        // 4 encoder layers, the boundary's two arrays, the bank's two, 4 decoder
        // layers, the logits.
        assert_eq!(checks.len(), 4 + 2 + 2 + 4 + 1);
        assert_eq!(summary["encoder"]["steps"].as_array().unwrap().len(), 5);
        // The invariance pass: 40 positions again in two slots, cut every 5 and every
        // 11, each array of each slot equal to the first run's.
        for (half, arrays) in [("encoder", 4 + 3), ("decoder", 4 + 1)] {
            let again = &summary[half]["invariance"];
            assert_eq!(again["equal"], json!(true));
            assert_eq!(again["positions"], 40);
            assert_eq!(again["steps"].as_array().unwrap().len(), 8);
            assert_eq!(again["steps"][0], json!([[2, 0, 11], [1, 0, 5]]));
            let checks = again["checks"].as_array().unwrap();
            assert_eq!(checks.len(), 2 * arrays);
            assert!(checks.iter().all(|c| c["elements"].as_u64().unwrap() > 0));
        }
        assert!(
            summary["decoder"]["memory"]["bytes"]["weights"]
                .as_u64()
                .unwrap()
                > 0
        );

        // A flipped bit in the recorded streams after decoder layer 5, position 3.
        let width = setup.width();
        let mut layers = setup.read_u16("decoder_layers.bin").unwrap();
        layers[70 * width + 3 * width + 9] ^= 1;
        setup.write("decoder_layers.bin", &layers).unwrap();
        let (result, equal) = setup.reference().unwrap();
        assert!(!equal);
        let bad: Vec<&Value> = result["checks"]
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

    /// The released configuration, laid out with no checkpoint on a device of one
    /// B300's memory that only checks its launches: each half fits beside its scratch
    /// and pools, every count fits the kernels' integers, and every launch of both
    /// halves over the measurement's 555 tokens is one its wrapper accepts.
    #[test]
    fn the_released_halves_fit_one_device_and_every_launch_is_accepted() {
        let c = DsConfig::from_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../eidola-engine-chat/tests/fixtures/deepseek_v4_1/config.json"),
        )
        .unwrap();
        let spec = c.engram.clone().unwrap();
        let map: Vec<u32> = (0..c.vocab_size)
            .map(|i| n32(i % spec.compressed_vocab))
            .collect();
        let tokens: Vec<u32> = (0..555u32).map(|i| (i * 7919 + 13) % 129_280).collect();
        let out = std::env::temp_dir().join(format!("ds_halves_dry_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let setup = Run::new(None, c, tokens, map, 96, 192, out).unwrap();
        // 275,040 MiB.
        let capacity = 275_040usize << 20;
        let device = || -> Result<Box<dyn Device>> { Ok(Box::new(DryDevice::new(Some(capacity)))) };
        let phases: Vec<String> = ["encoder", "decoder"].map(String::from).to_vec();
        let (summary, _) = run(&setup, &phases, &device).unwrap();
        let bytes =
            |half: &str, kind: &str| summary[half]["memory"]["bytes"][kind].as_u64().unwrap();
        let checkpoint = |half: &str| {
            summary[half]["memory"]["checkpoint_bytes_of_this_half"]
                .as_u64()
                .unwrap()
        };
        assert_eq!(checkpoint("encoder"), 250_855_098_528);
        // The decoder's side of the checkpoint holds the drafter.
        assert_eq!(checkpoint("decoder"), 250_527_512_880 + 7_932_874_632);
        eprintln!("{:#}", summary["encoder"]["memory"]);
        eprintln!("{:#}", summary["decoder"]["memory"]);
        // The encoder: its checkpoint bytes less what is not read, plus widened norms,
        // the stub and RoPE tables. The decoder holds no Engram table here.
        assert!(bytes("encoder", "weights").abs_diff(250_855_098_528) < 16 << 20);
        assert!(bytes("decoder", "weights").abs_diff(250_527_512_880 - 101_380_404_048) < 16 << 20);
        for half in ["encoder", "decoder"] {
            assert!(bytes(half, "total") < capacity as u64);
            assert!(bytes(half, "scratch") < 2 << 30, "{half} scratch");
            assert_eq!(summary[half]["steps"].as_array().unwrap().len(), 6);
        }
    }
}
