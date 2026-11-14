//! [`DsCudaExecutor`]: the serving core's [`Executor`] for DeepSeek-V4.1 on two logical
//! devices, eager (every step is launched afresh; no graph is captured).
//!
//! It reports the model as the CPU reference executor does without drafts (the same
//! four groups, ratios, devices and coverage, the window chain, the token history) and
//! runs a step in that executor's order: maintenance and table updates; the decoder's
//! device gathers the Engram rows of the tables it holds; the encoder runs every layer
//! before the split at every position, then the split layer's bank stub; the crossing
//! is checked; the decoder stores the bank entries, runs each chain layer over the rows
//! the row rule gives it, the head over the sampling rows, and samples on its device.
//!
//! Not supported, and refused before a step takes any effect: drafts (a row asking for
//! them is a host bug and panics, as at the seam), and the host tier (`Demote`,
//! `Promote`, `ZeroHost` fail the step).

use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::{Executor, ExecutorError, Maintenance, Slot, StepInput, StepOutput};
use eidola_engine::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, WindowChain,
};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::engram::EngramHasher;

use super::abi::{
    DT_BF16, DT_F32, EmbedParams, EngramGatherParams, KV_FP4_E4M3, KV_FP4_E8M0, ROT_ROTATE,
    STATUS_BAD_INDEX, STATUS_BAD_TOKEN,
};
use super::boundary::{GatheredRows, StepTransfer, ToDecoder, ToEncoder};
use super::half::{CudaDevice, Device, SampleArgs};
use super::kv::{Cell, GroupGeometry, KvStore};
use super::model::{BANK, Dst, HalfNet, LayerRun, MemoryBytes, WINDOW};
use super::plan::{PlanRow, StepPlan, pool_plan, window_slots};
use super::support::{DsShape, check_supported};
use super::weights::{Half, HalfWeights};
use crate::module::KernelDir;
use crate::sampler::{STATUS_NON_FINITE, SampleRow};
use crate::{CudaError, Result, narrow};

/// Executor configuration: memory geometry and devices. The model fixes the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct DsCudaExecutorConfig {
    /// Positions per KV block (every group); a multiple of the pooling ratio.
    pub block_size: u32,
    /// Physical blocks per group, including the null block 0.
    pub num_blocks: u32,
    /// Per-sequence state rows.
    pub num_state_slots: u32,
    /// Longest sequence.
    pub max_model_len: u32,
    /// Step shapes, ascending.
    pub buckets: Vec<Bucket>,
    /// Token ids the model may emit; the head's rows past it take no part in sampling.
    pub sampleable_vocab_size: u32,
    /// The device ordinal of each half (encoder, decoder). The same ordinal twice puts
    /// both logical devices on one GPU.
    pub devices: [usize; 2],
    /// Testing aid: keep each step's [`StepTransfer`] and [`StepStats`].
    pub record: bool,
}

/// Wall-clock time of one step's parts, each waited for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    /// Positions of the step.
    pub rows: u32,
    /// Layer-positions the chain computed.
    pub chain_work: u64,
    pub sampled: u32,
    pub maintenance_ms: f64,
    /// The decoder device's Engram gather and its crossing.
    pub gather_ms: f64,
    pub encoder_ms: f64,
    /// The crossing to the decoder: read-back, check, upload.
    pub crossing_ms: f64,
    pub decoder_ms: f64,
    pub total_ms: f64,
}

/// The DeepSeek-V4.1 executor. See the module documentation.
pub struct DsCudaExecutor {
    spec: ModelSpec,
    c: DsConfig,
    shape: DsShape,
    hasher: Option<EngramHasher>,
    encoder: HalfNet,
    decoder: HalfNet,
    record: bool,
    transfers: Vec<StepTransfer>,
    stats: Vec<StepStats>,
    chain_work: u64,
    steps: u64,
    had_rows: bool,
    idle_scrubs: u64,
}

fn err(e: CudaError) -> ExecutorError {
    ExecutorError(e.to_string())
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// The seam's description of the model under `cfg`: the CPU reference executor's
/// without drafts.
fn model_spec(c: &DsConfig, shape: &DsShape, cfg: &DsCudaExecutorConfig) -> Result<ModelSpec> {
    let window: u32 = narrow(c.attention.window, "window")?;
    let hd: u32 = narrow(c.attention.head_dim, "head dim")?;
    let group =
        |name: &str, attention, layers: usize, ratio, device, coverage| -> Result<KvGroupSpec> {
            Ok(KvGroupSpec {
                name: name.into(),
                role: KvRole::Target,
                attention,
                num_layers: narrow(layers, "layers of a group")?,
                num_kv_heads: 1,
                head_dim_qk: hd,
                head_dim_v: hd,
                num_blocks: cfg.num_blocks,
                host_blocks: 0,
                ratio,
                device,
                coverage,
            })
        };
    let sliding = AttentionKind::Sliding { window };
    let all = Coverage::All;
    let spec = ModelSpec {
        vocab_size: narrow(c.vocab_size, "vocabulary")?,
        sampleable_vocab_size: cfg.sampleable_vocab_size,
        block_size: cfg.block_size,
        max_model_len: cfg.max_model_len,
        kv_groups: vec![
            group(
                "encoder-bank",
                AttentionKind::Full,
                shape.sources.len(),
                shape.ratio,
                0,
                all,
            )?,
            group("encoder-window", sliding, shape.split, 1, 0, all)?,
            group("decoder-bank", AttentionKind::Full, 1, 1, 1, all)?,
            group(
                "decoder-window",
                sliding,
                shape.depth(),
                1,
                1,
                Coverage::Chain,
            )?,
        ],
        max_draft_tokens: 0,
        num_state_slots: cfg.num_state_slots,
        buckets: cfg.buckets.clone(),
        window_chain: Some(WindowChain {
            depth: narrow(shape.depth(), "chain depth")?,
            window,
        }),
        token_history: narrow(shape.history, "token history")?,
    };
    spec.validate().map_err(CudaError::new)?;
    if !cfg.block_size.is_multiple_of(shape.ratio) {
        return Err(CudaError::new(
            "the block size is not a multiple of the pooling ratio",
        ));
    }
    Ok(spec)
}

/// The block layouts of `half`'s groups: its bank, then its windows.
pub fn geometry(c: &DsConfig, spec: &ModelSpec, half: Half) -> Result<Vec<GroupGeometry>> {
    let width: u32 = narrow(c.attention.head_dim, "head dim")?;
    let index_width: u32 = narrow(c.indexer.head_dim, "index head dim")?;
    Ok(spec
        .kv_groups
        .iter()
        .enumerate()
        .filter(|(_, g)| g.device == half.device())
        .map(|(global, g)| GroupGeometry {
            global: u32::try_from(global).expect("a handful of groups"),
            cell: match g.attention {
                AttentionKind::Full => Cell::Bank,
                AttentionKind::Sliding { .. } => Cell::Window,
            },
            layers: g.num_layers,
            ratio: g.ratio,
            block_size: spec.block_size,
            width,
            index_width,
            num_blocks: g.num_blocks,
        })
        .collect())
}

impl DsCudaExecutor {
    /// Configuration checks only: the shape, the spec and the block layouts, with no
    /// device. `new` runs exactly these first.
    pub fn preflight(c: &DsConfig, cfg: &DsCudaExecutorConfig) -> Result<(DsShape, ModelSpec)> {
        let shape = check_supported(c, 0)?;
        let spec = model_spec(c, &shape, cfg)?;
        if c.max_positions
            .is_some_and(|m| cfg.max_model_len as usize > m)
        {
            return Err(CudaError::new(
                "max_model_len is past the positions the checkpoint declares",
            ));
        }
        if cfg.sampleable_vocab_size > 1 << 20 {
            return Err(CudaError::new("the sampler takes at most 2^20 ids"));
        }
        for half in [Half::Encoder, Half::Decoder] {
            for g in geometry(c, &spec, half)? {
                narrow::<u32, usize>(g.block_bytes(), "bytes of a block")?;
            }
        }
        Ok((shape, spec))
    }

    /// An executor on CUDA devices, its two halves opened on `cfg.devices`.
    pub fn new(
        dir: &KernelDir,
        store: &Arc<WeightSet>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
    ) -> Result<DsCudaExecutor> {
        Self::preflight(c, cfg)?;
        let encoder = Box::new(CudaDevice::open(cfg.devices[0], dir)?);
        let decoder = Box::new(CudaDevice::open(cfg.devices[1], dir)?);
        Self::with_devices(encoder, decoder, store, c, token_map, cfg)
    }

    /// An executor over two devices: the encoder's and the decoder's.
    pub fn with_devices(
        encoder: Box<dyn Device>,
        decoder: Box<dyn Device>,
        store: &Arc<WeightSet>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
    ) -> Result<DsCudaExecutor> {
        let (shape, spec) = Self::preflight(c, cfg)?;
        let hasher = match (&c.engram, token_map) {
            (Some(e), Some(map)) => Some(
                EngramHasher::new(e, c.vocab_size, map)
                    .map_err(|e| CudaError::new(e.to_string()))?,
            ),
            (Some(_), None) if shape.engram_layer.iter().any(|l| l.is_some()) => {
                return Err(CudaError::new(
                    "this model has Engram layers and needs the compressed-vocabulary map",
                ));
            }
            _ => None,
        };
        let max_rows = cfg.buckets.iter().map(|b| b.max_tokens).max().unwrap_or(0);
        let samples = cfg.buckets.iter().map(|b| b.max_seqs).max().unwrap_or(0);
        // The open entry after any accepted length: the pooling ratio, the draft width
        // (0) and one.
        let cap = shape.ratio + 1;
        let mut halves = Vec::new();
        for (half, dev) in [(Half::Encoder, encoder), (Half::Decoder, decoder)] {
            let w = HalfWeights::load(dev.as_ref(), store, c, &shape, half, cfg.max_model_len)?;
            let kv = KvStore::new(
                dev.as_ref(),
                geometry(c, &spec, half)?,
                cfg.num_state_slots,
                spec.max_blocks_per_seq(),
            )?;
            halves.push(HalfNet::new(
                half,
                dev,
                w,
                kv,
                c,
                &shape,
                cfg.max_model_len,
                max_rows,
                samples,
                cfg.sampleable_vocab_size,
                cap,
            )?);
        }
        let decoder = halves.pop().expect("two halves");
        let encoder = halves.pop().expect("two halves");
        Ok(DsCudaExecutor {
            spec,
            c: c.clone(),
            shape,
            hasher,
            encoder,
            decoder,
            record: cfg.record,
            transfers: Vec::new(),
            stats: Vec::new(),
            chain_work: 0,
            steps: 0,
            had_rows: false,
            idle_scrubs: 0,
        })
    }

    /// Each half's memory by kind: the encoder's, then the decoder's.
    pub fn memory(&self) -> [MemoryBytes; 2] {
        [self.encoder.memory(), self.decoder.memory()]
    }

    /// A half (tests).
    pub fn half(&self, half: Half) -> &HalfNet {
        match half {
            Half::Encoder => &self.encoder,
            Half::Decoder => &self.decoder,
        }
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Layer-positions the window chain has computed.
    pub fn chain_layer_positions(&self) -> u64 {
        self.chain_work
    }

    /// Rowless steps that scrubbed the scratch.
    pub fn idle_scrubs(&self) -> u64 {
        self.idle_scrubs
    }

    /// What crossed between the halves in each step since the last call (with
    /// `record`).
    pub fn take_transfers(&mut self) -> Vec<StepTransfer> {
        std::mem::take(&mut self.transfers)
    }

    /// Each step's timings since the last call (with `record`).
    pub fn take_stats(&mut self) -> Vec<StepStats> {
        std::mem::take(&mut self.stats)
    }

    fn net(&self, group: u32) -> (&HalfNet, usize) {
        [&self.encoder, &self.decoder]
            .into_iter()
            .find_map(|n| n.kv.local(group).map(|g| (n, g)))
            .unwrap_or_else(|| panic!("no group {group}"))
    }

    fn net_mut(&mut self, group: u32) -> (&mut HalfNet, usize) {
        match self.encoder.kv.local(group) {
            Some(g) => (&mut self.encoder, g),
            None => {
                let g = self
                    .decoder
                    .kv
                    .local(group)
                    .unwrap_or_else(|| panic!("no group {group}"));
                (&mut self.decoder, g)
            }
        }
    }

    /// A block's bytes as the device holds them.
    pub fn block_bytes(&self, group: u32, block: u32) -> Result<Vec<u8>> {
        let (n, g) = self.net(group);
        n.kv.read_block(n.dev.as_ref(), g, block)
    }

    /// Every block of a group in the reference executor's cell order, as the words of
    /// its image's first range.
    pub fn pool_cells(&self, group: u32) -> Result<Vec<Vec<u32>>> {
        let (n, g) = self.net(group);
        let geom = &n.kv.groups[g].geom;
        let pool = n.kv.read_pool(n.dev.as_ref(), g)?;
        Ok(pool
            .chunks_exact(geom.block_bytes())
            .map(|b| geom.cells(b))
            .collect())
    }

    pub fn block_is_zero(&self, group: u32, block: u32) -> Result<bool> {
        Ok(self.block_bytes(group, block)?.iter().all(|b| *b == 0))
    }

    pub fn block_is_encrypted(&self, group: u32, block: u32) -> bool {
        let (n, g) = self.net(group);
        n.kv.is_encrypted(g, block)
    }

    /// Whether a slot's per-sequence model state (the pooled sources' open entries) is
    /// scrubbed.
    pub fn slot_state_is_zero(&self, slot: Slot) -> Result<bool> {
        self.encoder.state_is_zero(slot)
    }

    /// Refuses what this executor does not run, before any of the step takes effect.
    fn check_maintenance(&self, step: &StepInput) -> Result<()> {
        let groups = u32::try_from(self.spec.kv_groups.len()).expect("a handful of groups");
        for m in &step.maintenance {
            let (group, key) = match *m {
                Maintenance::Zero { group, .. } | Maintenance::Copy { group, .. } => (group, None),
                Maintenance::ResetSlot { .. } => continue,
                Maintenance::Encrypt { group, key, .. }
                | Maintenance::Decrypt { group, key, .. } => (group, Some(key)),
                Maintenance::Demote { .. }
                | Maintenance::Promote { .. }
                | Maintenance::ZeroHost { .. } => {
                    return Err(CudaError::new(format!(
                        "unsupported maintenance {m:?}: this executor has no host tier"
                    )));
                }
            };
            if group >= groups {
                return Err(CudaError::new(format!("maintenance names group {group}")));
            }
            if key.is_some_and(|k| k as usize >= step.kv_keys.len()) {
                return Err(CudaError::new("maintenance names a key not in the step"));
            }
        }
        Ok(())
    }

    fn maintain(&mut self, step: &StepInput) -> Result<()> {
        for m in &step.maintenance {
            match *m {
                Maintenance::Zero { group, block } => {
                    let (n, g) = self.net_mut(group);
                    n.kv.zero(n.dev.as_ref(), g, block)?;
                }
                Maintenance::Copy { group, src, dst } => {
                    let (n, g) = self.net_mut(group);
                    n.kv.copy(n.dev.as_ref(), g, src, dst)?;
                }
                Maintenance::ResetSlot { slot } => {
                    self.encoder.kv.reset_slot(slot);
                    self.decoder.kv.reset_slot(slot);
                    self.encoder.reset_state(slot)?;
                }
                Maintenance::Encrypt {
                    group,
                    block,
                    key,
                    nonce,
                }
                | Maintenance::Decrypt {
                    group,
                    block,
                    key,
                    nonce,
                } => {
                    let encrypt = matches!(m, Maintenance::Encrypt { .. });
                    let key = &step.kv_keys[key as usize];
                    let (n, g) = self.net_mut(group);
                    n.kv.crypt(n.dev.as_ref(), g, block, key, &nonce, encrypt)?;
                }
                Maintenance::Demote { .. }
                | Maintenance::Promote { .. }
                | Maintenance::ZeroHost { .. } => unreachable!("refused before the step"),
            }
        }
        for u in &step.table_updates {
            let (n, g) = self.net_mut(u.group);
            n.kv.set_table(g, u.slot, u.index, u.block);
        }
        for n in [&mut self.encoder, &mut self.decoder] {
            n.kv.flush_tables(n.dev.as_ref())?;
        }
        Ok(())
    }

    /// The rows of a step, checked as the reference executor checks them, and each
    /// row's known tokens (the history it was given, then its host tokens).
    fn rows(&self, step: &StepInput) -> (Vec<PlanRow>, Vec<(usize, Vec<u32>)>) {
        let h = self.shape.history;
        let mut rows = Vec::with_capacity(step.seqs.len());
        let mut known = Vec::with_capacity(step.seqs.len());
        for (ri, e) in step.seqs.iter().enumerate() {
            assert!(e.num_tokens >= 1, "a row needs at least one host token");
            let start = e.token_start as usize;
            let end = start + e.num_tokens as usize;
            for (i, &pos) in step.positions[start..end].iter().enumerate() {
                assert_eq!(
                    pos as usize,
                    e.context_len as usize + i,
                    "positions disagree with context_len"
                );
            }
            let k = if e.sample { e.num_drafts } else { 0 };
            assert!(k == 0, "{k} drafts requested; the executor drafts 0");
            let p = e.context_len + e.num_tokens - 1;
            let chain_start = *step
                .aux
                .chain_start
                .get(ri)
                .expect("a row of a model with a window chain carries its chain_start");
            assert!(
                !e.sample || chain_start <= p,
                "slot {}: sampling at {p}, where the chain's top layer does not run",
                e.slot
            );
            let given = step
                .aux
                .history_ids
                .get(ri * h..(ri + 1) * h)
                .expect("a row carries its token history");
            let history = h.min(e.context_len as usize);
            let mut tokens = given[h - history..].to_vec();
            tokens.extend_from_slice(&step.token_ids[start..end]);
            rows.push(PlanRow {
                slot: e.slot,
                c: e.context_len,
                p,
                sample: e.sample,
                chain_start,
            });
            known.push((history, tokens));
        }
        (rows, known)
    }

    /// Per Engram layer of the config, the table rows of every position in device row
    /// order (`[rows, lookups]`); empty for a layer the model does not keep.
    fn engram_ids(
        &self,
        plan: &StepPlan,
        rows: &[PlanRow],
        known: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<u32>>> {
        let Some(e) = &self.c.engram else {
            return Ok(Vec::new());
        };
        let mut out = vec![Vec::new(); e.layers.len()];
        for (slot, spec) in e.layers.iter().enumerate() {
            if self.shape.engram_layer[slot].is_none() {
                continue;
            }
            let hasher = self
                .hasher
                .as_ref()
                .ok_or_else(|| CudaError::new("no Engram hasher"))?;
            for it in &plan.items {
                let (history, tokens) = &known[it.row as usize];
                let at = history + (it.pos - rows[it.row as usize].c) as usize;
                let lookback = &tokens[(at + 1).saturating_sub(e.max_ngram)..=at];
                for row in hasher.rows(spec, lookback, it.pos as usize) {
                    out[slot].push(narrow(row, "Engram table row")?);
                }
            }
        }
        Ok(out)
    }

    /// The decoder device's part of the Engram lookups: the rows of the tables it
    /// holds, read back for the encoder.
    fn gather(&mut self, ids: &[Vec<u32>], rows: u32) -> Result<ToEncoder> {
        let n = &mut self.decoder;
        let dev = n.dev.as_ref();
        let mut gathered = vec![None; ids.len()];
        for (slot, ids) in ids.iter().enumerate() {
            let Some(table) = n.w.tables[slot] else {
                continue;
            };
            n.s.eids.write(dev, 0, ids)?;
            let (codes, scales) = &n.s.gathered[slot];
            // SAFETY: the table and `rows` rows of ids and of gathered rows.
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
            let width = n.d.engram_width as usize;
            gathered[slot] = Some(GatheredRows {
                codes: codes.read(dev, 0, rows as usize * width)?,
                scales: scales.read(dev, 0, rows as usize * width / 32)?,
            });
        }
        Ok(ToEncoder { gathered })
    }

    /// The step's index arrays on one half: positions, bank visibility, table rows,
    /// the window targets of the first `writes` rows and the window slot lists of the
    /// first `queries` rows. Every read and write is checked against the tables here.
    fn step_arrays(n: &HalfNet, plan: &StepPlan, writes: u32, queries: u32) -> Result<()> {
        let dev = n.dev.as_ref();
        let ratio = n.kv.groups[BANK].geom.ratio;
        let items = &plan.items;
        let positions: Vec<u32> = items.iter().map(|it| it.pos).collect();
        let visible: Vec<u32> = items.iter().map(|it| (it.pos + 1) / ratio).collect();
        let table_row: Vec<u32> = items.iter().map(|it| it.slot).collect();
        let targets: Vec<u32> = items[..writes as usize]
            .iter()
            .map(|it| n.kv.target(WINDOW, it.slot, it.pos))
            .collect();
        let queries = &items[..queries as usize];
        let slots = window_slots(queries, n.d.window, n.d.slot_stride, |slot, pos| {
            n.kv.entry(WINDOW, slot, pos)
        });
        // A query scores every visible entry of its bank: each block holding one is
        // mapped.
        for it in queries {
            let entries = (it.pos + 1) / ratio;
            for first in (0..entries * ratio).step_by(n.kv.block_size as usize) {
                n.kv.entry(BANK, it.slot, first);
            }
        }
        n.s.positions.write(dev, 0, &positions)?;
        n.s.visible.write(dev, 0, &visible)?;
        n.s.table_row.write(dev, 0, &table_row)?;
        n.s.wtargets.write(dev, 0, &targets)?;
        n.s.slots.write(dev, 0, &slots)?;
        n.s.status.write(dev, 0, &[0u32])
    }

    /// The encoder over every position of the step, then the split layer's bank stub;
    /// returns what crosses to the decoder.
    fn encode(
        &mut self,
        plan: &StepPlan,
        rows: &[PlanRow],
        tokens: &[u32],
        ids: &[Vec<u32>],
        from_decoder: &ToEncoder,
    ) -> Result<ToDecoder> {
        let (c, shape) = (&self.c, &self.shape);
        let n = &mut self.encoder;
        let count = plan.rows();
        Self::step_arrays(n, plan, count, count)?;
        let d = n.d.clone();
        {
            let dev = n.dev.as_ref();
            n.s.tokens.write(dev, 0, tokens)?;
            // The first sub-layer collapses with the one-hot weights.
            let mut pre = vec![0.0f32; count as usize * d.mix_words as usize];
            for row in pre.chunks_exact_mut(d.mix_words as usize) {
                row[0] = 1.0;
            }
            n.s.mix_b.write(dev, 0, &pre)?;
            // SAFETY: `count` rows of tokens and of the stream.
            unsafe {
                dev.kernels().embed(&EmbedParams {
                    table: n.w.embed.expect("the encoder holds the embedding"),
                    tokens: n.s.tokens.addr,
                    out: n.s.x.addr,
                    status: n.s.status.addr,
                    rows: count,
                    dim: d.dim,
                    copies: d.copies,
                    vocab: d.vocab,
                })?;
            }
            for (slot, ids) in ids.iter().enumerate() {
                let (codes, scales) = &n.s.gathered[slot];
                if let Some(table) = n.w.tables[slot] {
                    n.s.eids.write(dev, 0, ids)?;
                    // SAFETY: as the decoder's gather.
                    unsafe {
                        dev.kernels().engram_gather(&EngramGatherParams {
                            codes: table.codes,
                            scales: table.scales,
                            ids: n.s.eids.addr,
                            out_codes: codes.addr,
                            out_scales: scales.addr,
                            status: n.s.status.addr,
                            table_rows: table.rows,
                            rows: count,
                            ids_per_row: d.lookups,
                            width: d.engram_width / d.lookups,
                            reserved: 0,
                        })?;
                    }
                } else if let Some(g) = from_decoder.gathered.get(slot).and_then(|g| g.as_ref()) {
                    codes.write(dev, 0, &g.codes)?;
                    scales.write(dev, 0, &g.scales)?;
                }
            }
        }
        let pool = pool_plan(plan, rows, n.cap);
        for l in 0..shape.split {
            let spec = &c.layers[l];
            let source = |s: usize| {
                narrow::<u32, usize>(
                    shape.source_index(s).expect("checked: an encoder source"),
                    "bank layer",
                )
            };
            let run = LayerRun {
                active: count,
                emit: count,
                window_layer: narrow(l, "layer")?,
                bank_layer: spec.kv_source.map(source).transpose()?,
                build: match spec.is_kv_source {
                    true => Some((source(l)?, &pool)),
                    false => None,
                },
            };
            n.layer(l, &run)?;
        }

        // The stub: the split layer's attention input, compressed to its bank entry.
        let stub = n.w.stub.expect("the encoder holds the stub");
        let cw = stub.compressor;
        let cis = n.w.rope(c.layers[shape.split].rope);
        n.collapse(&n.s.mix_b, count)?;
        // SAFETY: `count` rows of the collapsed stream, normed in place.
        unsafe {
            n.dev.kernels().rms_norm(&super::abi::RmsNormParams {
                x: n.s.u.addr,
                weight: stub.attn_norm,
                out: n.s.u.addr,
                rows: count,
                width: d.dim,
                eps: c.norm_eps,
                reserved: 0,
            })?;
        }
        n.linear(
            &n.s.u,
            cw.wkv,
            &n.s.pooled,
            count,
            d.head_dim,
            d.dim,
            DT_BF16,
        )?;
        let norm = |x: &super::half::Buffer, weight: u64, width: u32| {
            // SAFETY: `count` rows of `width`, normed in place.
            unsafe {
                n.dev.kernels().rms_norm(&super::abi::RmsNormParams {
                    x: x.addr,
                    weight,
                    out: x.addr,
                    rows: count,
                    width,
                    eps: c.norm_eps,
                    reserved: 0,
                })
            }
        };
        norm(&n.s.pooled, cw.norm, d.head_dim)?;
        n.linear(
            &n.s.pooled,
            cw.wk,
            &n.s.key,
            count,
            d.index_dim,
            d.head_dim,
            DT_BF16,
        )?;
        norm(&n.s.key, cw.k_norm, d.index_dim)?;
        n.rot_quant(
            &n.s.key,
            cis,
            &n.s.positions,
            Dst::Dense(&n.s.bkey_codes, &n.s.bkey_scales),
            count,
            1,
            d.index_dim,
            ROT_ROTATE,
            KV_FP4_E8M0,
        )?;
        n.rot_quant(
            &n.s.pooled,
            cis,
            &n.s.positions,
            Dst::Dense(&n.s.bkv_codes, &n.s.bkv_scales),
            count,
            1,
            d.head_dim,
            ROT_ROTATE,
            KV_FP4_E4M3,
        )?;

        // What crosses, read back into host memory.
        let dev = n.dev.as_ref();
        let rows_n = count as usize;
        let sections = [
            (&n.s.bkv_codes, d.head_dim as usize / 2),
            (&n.s.bkv_scales, d.head_dim as usize / 16),
            (&n.s.bkey_codes, d.index_dim as usize / 2),
            (&n.s.bkey_scales, d.index_dim as usize / 32),
        ];
        let entry_bytes: usize = sections.iter().map(|s| s.1).sum();
        let mut entries = vec![0u8; rows_n * entry_bytes];
        let mut at = 0;
        for (buf, bytes) in sections {
            let data: Vec<u8> = buf.read(dev, 0, rows_n * bytes)?;
            for (row, src) in data.chunks_exact(bytes).enumerate() {
                entries[row * entry_bytes + at..][..bytes].copy_from_slice(src);
            }
            at += bytes;
        }
        let boundary = plan.boundary_rows();
        let width = (d.copies * d.dim) as usize;
        let streams: Vec<u16> = n.s.x.read(dev, 0, boundary as usize * width)?;
        let mixes: Vec<f32> =
            n.s.mix_b
                .read(dev, 0, boundary as usize * d.mix_words as usize)?;
        let pre_mix = mixes
            .chunks_exact(d.mix_words as usize)
            .flat_map(|row| row[..d.copies as usize].iter().copied())
            .collect();
        check_status(n.status()?)?;
        Ok(ToDecoder {
            positions: plan.items.iter().map(|it| (it.slot, it.pos)).collect(),
            entry_bytes,
            entries,
            rows: boundary,
            streams,
            pre_mix,
        })
    }

    /// The decoder: the bank entries stored, the chain by the row rule, the head and
    /// the sampler. Returns each sampling row's token and, when asked, its logits.
    fn decode(
        &mut self,
        plan: &StepPlan,
        step: &StepInput,
        crossing: &ToDecoder,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let (c, shape) = (&self.c, &self.shape);
        let n = &mut self.decoder;
        let d = n.d.clone();
        let boundary = plan.boundary_rows();

        // The bank entries of every position, section by section.
        {
            let dev = n.dev.as_ref();
            let gs = &n.kv.groups[BANK];
            let (entries, block_bytes) = (gs.geom.entries(), gs.geom.block_bytes());
            for (row, (slot, pos)) in crossing.positions.iter().enumerate() {
                let target = n.kv.target(BANK, *slot, *pos);
                let block = (target / entries) as usize * block_bytes;
                let cell = &crossing.entries[row * crossing.entry_bytes..][..crossing.entry_bytes];
                let mut from = 0;
                for (at, bytes) in gs.geom.runs(0, target % entries) {
                    gs.pool.write(dev, block + at, &cell[from..from + bytes])?;
                    from += bytes;
                }
            }
        }
        Self::step_arrays(n, plan, boundary, plan.emit_rows(0))?;
        {
            let dev = n.dev.as_ref();
            n.s.x.write(dev, 0, &crossing.streams)?;
            let mut pre = vec![0.0f32; boundary as usize * d.mix_words as usize];
            for (row, src) in pre
                .chunks_exact_mut(d.mix_words as usize)
                .zip(crossing.pre_mix.chunks_exact(d.copies as usize))
            {
                row[..d.copies as usize].copy_from_slice(src);
            }
            n.s.mix_b.write(dev, 0, &pre)?;
        }
        for j in 0..shape.depth() {
            let l = shape.split + j;
            let run = LayerRun {
                active: plan.chain_rows[j],
                emit: plan.emit_rows(j),
                window_layer: narrow(j, "chain layer")?,
                bank_layer: c.layers[l].kv_source.map(|_| 0),
                build: None,
            };
            n.layer(l, &run)?;
        }

        let top = plan.top_rows;
        let samples: Vec<SampleRow> = step
            .seqs
            .iter()
            .zip(&plan.last_row)
            .filter(|(e, _)| e.sample)
            .map(|(e, row)| SampleRow::new(&e.sampling, e.context_len + e.num_tokens, *row))
            .collect();
        if top == 0 {
            check_status(n.status()?)?;
            return Ok((Vec::new(), Vec::new()));
        }
        let (head, final_norm) = n.w.head.expect("the decoder holds the head");
        n.collapse(&n.s.mix_b, top)?;
        let dev = n.dev.as_ref();
        // SAFETY: `top` rows, each within the scratch; the sampler's buffers hold a row
        // per sampling row of the largest step.
        unsafe {
            dev.kernels().rms_norm(&super::abi::RmsNormParams {
                x: n.s.u.addr,
                weight: final_norm,
                out: n.s.u.addr,
                rows: top,
                width: d.dim,
                eps: c.norm_eps,
                reserved: 0,
            })?;
        }
        n.linear(&n.s.u, head, &n.s.logits, top, d.vocab, d.dim, DT_F32)?;
        // SAFETY: as above.
        unsafe {
            dev.sample(&SampleArgs {
                logits: n.s.logits.addr,
                logits_stride: d.vocab,
                logit_rows: top,
                n: self.spec.sampleable_vocab_size,
                rows: &samples,
                rows_dev: n.s.srows.addr,
                probs: n.s.probs.addr,
                tokens: n.s.stokens.addr,
                status: n.s.status.addr,
            })?;
        }
        let tokens: Vec<u32> = n.s.stokens.read(dev, 0, samples.len())?;
        check_status(n.status()?)?;
        let mut logits = Vec::new();
        if step.return_logits {
            let all: Vec<f32> = n.s.logits.read(dev, 0, top as usize * d.vocab as usize)?;
            for s in &samples {
                let at = s.logit_row as usize * d.vocab as usize;
                logits.push(all[at..at + d.vocab as usize].to_vec());
            }
        }
        Ok((tokens, logits))
    }

    fn run(&mut self, step: &StepInput) -> Result<StepOutput> {
        let started = Instant::now();
        if narrow::<u32, usize>(step.seqs.len(), "rows")? > step.bucket.max_seqs
            || step.query_tokens() > step.bucket.max_tokens
        {
            return Err(CudaError::new("batch exceeds its bucket"));
        }
        if !self.spec.buckets.contains(&step.bucket) {
            return Err(CudaError::new("the step's bucket is not a configured one"));
        }
        self.check_maintenance(step)?;
        let (rows, known) = self.rows(step);
        self.maintain(step)?;
        let mut stats = StepStats::default();
        if self.record {
            self.encoder.dev.synchronize()?;
            self.decoder.dev.synchronize()?;
        }
        stats.maintenance_ms = ms(started);

        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; rows.len() * stride as usize],
            stride,
            num_tokens: vec![0; rows.len()],
            logits: step.return_logits.then(Vec::new),
        };
        if rows.is_empty() {
            // The first rowless step after one with rows leaves no step data behind.
            if std::mem::take(&mut self.had_rows) {
                for n in [&self.encoder, &self.decoder] {
                    n.scrub_scratch()?;
                    n.dev.synchronize()?;
                }
                self.idle_scrubs += 1;
            }
            return Ok(out);
        }
        self.had_rows = true;

        let depth = self.spec.window_chain.expect("a window chain");
        let plan = StepPlan::new(&rows, depth.depth, depth.window);
        let tokens: Vec<u32> = plan
            .items
            .iter()
            .map(|it| {
                let (history, tokens) = &known[it.row as usize];
                tokens[history + (it.pos - rows[it.row as usize].c) as usize]
            })
            .collect();
        for &t in &tokens {
            if t as usize >= self.c.vocab_size || self.c.image_token_id == Some(t) {
                return Err(CudaError::new(format!(
                    "token {t} cannot be embedded (vocabulary {}, image token {:?})",
                    self.c.vocab_size, self.c.image_token_id
                )));
            }
        }
        let ids = self.engram_ids(&plan, &rows, &known)?;

        let at = Instant::now();
        let to_encoder = self.gather(&ids, plan.rows())?;
        check_status(self.decoder.status()?)?;
        stats.gather_ms = ms(at);

        let at = Instant::now();
        let crossing = self.encode(&plan, &rows, &tokens, &ids, &to_encoder)?;
        stats.encoder_ms = ms(at);

        let at = Instant::now();
        let d = &self.encoder.d;
        crossing.check(&plan, (d.copies * d.dim) as usize, d.copies as usize)?;
        stats.crossing_ms = ms(at);

        let at = Instant::now();
        let (sampled, logits) = self.decode(&plan, step, &crossing)?;
        stats.decoder_ms = ms(at);

        let mut next = 0;
        for (i, e) in step.seqs.iter().enumerate() {
            if !e.sample {
                if let Some(l) = out.logits.as_mut() {
                    l.push(Vec::new());
                }
                continue;
            }
            let token = sampled[next];
            if token >= self.spec.sampleable_vocab_size {
                return Err(CudaError::new(format!(
                    "sampled id {token} is not sampleable"
                )));
            }
            out.tokens[i * stride as usize] = token;
            out.num_tokens[i] = 1;
            if let Some(l) = out.logits.as_mut() {
                l.push(vec![logits[next].clone()]);
            }
            next += 1;
        }
        self.chain_work += plan.chain_work();
        if self.record {
            self.transfers
                .push(StepTransfer::of(&plan, &to_encoder, &crossing));
            stats.rows = plan.rows();
            stats.chain_work = plan.chain_work();
            stats.sampled = narrow(sampled.len(), "sampled rows")?;
            stats.total_ms = ms(started);
            self.stats.push(stats);
        }
        Ok(out)
    }
}

fn check_status(status: u32) -> Result<()> {
    if status & STATUS_NON_FINITE != 0 {
        return Err(CudaError::new("non-finite logits"));
    }
    if status & (STATUS_BAD_TOKEN | crate::sampler::STATUS_BAD_TOKEN) != 0 {
        return Err(CudaError::new("a kernel met a token id out of range"));
    }
    if status & (STATUS_BAD_INDEX | crate::engine_ops::STATUS_BAD_INDEX) != 0 {
        return Err(CudaError::new("a kernel met an index out of range"));
    }
    Ok(())
}

impl Executor for DsCudaExecutor {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> std::result::Result<StepOutput, ExecutorError> {
        self.steps += 1;
        self.run(step).map_err(err)
    }
}

impl Drop for DsCudaExecutor {
    /// Zeroes every pool, table, state and scratch buffer before the memory is freed:
    /// best effort, since on a failed device the writes fail too.
    fn drop(&mut self) {
        for n in [&self.encoder, &self.decoder] {
            let _ = n.scrub_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use eidola_engine::executor::{SeqEntry, StepAux, TableUpdate};
    use eidola_engine::sampling::SamplingParams;
    use eidola_engine_model::deepseek::config_of;

    use super::super::host::HostDevice;
    use super::super::support::tests::fixture_dir;
    use super::super::weights::{Form, weight_bytes};
    use super::*;

    fn config() -> DsCudaExecutorConfig {
        DsCudaExecutorConfig {
            block_size: 8,
            num_blocks: 12,
            num_state_slots: 4,
            max_model_len: 64,
            buckets: vec![Bucket {
                max_seqs: 2,
                max_tokens: 16,
            }],
            sampleable_vocab_size: 95,
            devices: [0, 0],
            record: true,
        }
    }

    fn executor() -> DsCudaExecutor {
        let dir = fixture_dir();
        let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
        let golden = WeightSet::open_files(&[dir.join("golden.safetensors")]).unwrap();
        let name = "engram.token_map";
        let map = golden.get(name).unwrap().to_i64(name).unwrap();
        let map = map.into_iter().map(|t| u32::try_from(t).unwrap()).collect();
        let c = config_of(&store).unwrap();
        DsCudaExecutor::with_devices(
            Box::new(HostDevice),
            Box::new(HostDevice),
            &store,
            &c,
            Some(map),
            &config(),
        )
        .unwrap()
    }

    fn step(x: &DsCudaExecutor) -> StepInput {
        StepInput {
            bucket: x.spec.buckets[0],
            maintenance: Vec::new(),
            kv_keys: Vec::new(),
            table_updates: Vec::new(),
            seqs: Vec::new(),
            token_ids: Vec::new(),
            positions: Vec::new(),
            return_logits: false,
            aux: StepAux::default(),
        }
    }

    /// A step of `tokens` at positions `0 ..` of `slot`, every group's first block
    /// mapped to `block`.
    fn first_row(x: &DsCudaExecutor, slot: Slot, block: u32, tokens: &[u32]) -> StepInput {
        let n = u32::try_from(tokens.len()).unwrap();
        StepInput {
            table_updates: (0..4)
                .map(|group| TableUpdate {
                    slot,
                    group,
                    index: 0,
                    block,
                })
                .collect(),
            seqs: vec![SeqEntry {
                slot,
                token_start: 0,
                num_tokens: n,
                context_len: 0,
                num_drafts: 0,
                sample: true,
                sampling: SamplingParams::greedy(),
            }],
            token_ids: tokens.to_vec(),
            positions: (0..n).collect(),
            aux: StepAux {
                chain_start: vec![0],
                history_ids: vec![0; 3],
            },
            ..step(x)
        }
    }

    /// What the loader put on each device is the arithmetic over the configuration, and
    /// each half's memory by kind is its layouts'.
    #[test]
    fn memory_by_kind_is_the_layouts_arithmetic() {
        let x = executor();
        let cfg = config();
        for (half, mem) in [Half::Encoder, Half::Decoder].into_iter().zip(x.memory()) {
            let n = x.half(half);
            let want = weight_bytes(
                &x.c,
                &x.shape,
                half,
                Form::Device {
                    positions: cfg.max_model_len,
                },
            );
            assert_eq!(n.w.bytes, want, "{half:?}");
            assert_eq!(mem.weights, want.total());
            let geometry = geometry(&x.c, &x.spec, half).unwrap();
            let pools: usize = geometry.iter().map(|g| g.pool_bytes()).sum();
            assert_eq!(mem.kv, pools as u64);
            // A table of 8 block words for each of 4 slots, per group.
            assert_eq!(mem.tables, 2 * 4 * 8 * 4);
            assert!(mem.scratch > 0);
            assert_eq!(
                mem.total(),
                mem.weights + mem.kv + mem.tables + mem.state + mem.scratch
            );
        }
        let [enc, dec] = x.memory();
        // One pooled source, three positions kept, two f32 projections of 32 a slot.
        assert_eq!((enc.state, dec.state), (2 * 4 * 3 * 32 * 4, 0));
        // The fixture's blocks: a bank layer of 4 entries of 35 bytes; 4 window layers
        // of 8 entries of 33; a bank layer of 8 entries of 35.
        let block = |half, g: usize| x.half(half).kv.groups[g].geom.block_bytes();
        assert_eq!(block(Half::Encoder, BANK), 144);
        assert_eq!(block(Half::Encoder, WINDOW), 1056);
        assert_eq!(block(Half::Decoder, BANK), 288);
        assert_eq!(block(Half::Decoder, WINDOW), 1056);
    }

    #[test]
    fn the_host_tier_is_refused_before_the_step_takes_effect() {
        let mut x = executor();
        let mut s = first_row(&x, 1, 3, &[5, 6, 7]);
        s.maintenance = vec![
            Maintenance::Zero { group: 0, block: 2 },
            Maintenance::Demote {
                group: 0,
                block: 2,
                host: 1,
            },
        ];
        let e = x.execute(&s).unwrap_err();
        assert!(e.0.contains("no host tier"), "{e}");
        for op in [
            Maintenance::Promote {
                group: 1,
                host: 1,
                block: 2,
            },
            Maintenance::ZeroHost { group: 1, host: 1 },
        ] {
            s.maintenance = vec![op];
            assert!(x.execute(&s).unwrap_err().0.contains("no host tier"));
        }
        // Nothing was mapped: the same rows now run from scratch.
        let n = x.half(Half::Encoder);
        assert!(
            n.kv.read_table(n.dev.as_ref(), WINDOW, 1)
                .unwrap()
                .iter()
                .all(|w| *w == -1)
        );
        s.maintenance.clear();
        let out = x.execute(&s).unwrap();
        assert_eq!(out.num_tokens, vec![1]);
        assert_eq!(x.take_transfers().len(), 1);
    }

    #[test]
    #[should_panic(expected = "drafts requested; the executor drafts 0")]
    fn drafts_are_a_host_bug() {
        let mut x = executor();
        let mut s = first_row(&x, 0, 1, &[5, 6]);
        s.seqs[0].num_drafts = 1;
        let _ = x.execute(&s);
    }

    #[test]
    fn malformed_steps_are_refused() {
        let mut x = executor();
        let mut s = first_row(&x, 0, 1, &[5, 6]);
        s.bucket = Bucket {
            max_seqs: 1,
            max_tokens: 8,
        };
        assert!(x.execute(&s).unwrap_err().0.contains("bucket"));
        let s = first_row(&x, 0, 1, &[5, 95]);
        assert!(x.execute(&s).unwrap_err().0.contains("cannot be embedded"));
        let mut cfg = config();
        cfg.block_size = 7;
        assert!(DsCudaExecutor::preflight(&x.c, &cfg).is_err());
        let mut cfg = config();
        cfg.max_model_len = 4096;
        assert!(DsCudaExecutor::preflight(&x.c, &cfg).is_err());
    }

    /// A slot's open entry is state the reset scrubs, and only that slot's; the first
    /// rowless step after rows scrubs the scratch, once.
    #[test]
    fn resets_and_idle_steps_scrub() {
        let mut x = executor();
        x.execute(&first_row(&x, 1, 3, &[5, 6, 7])).unwrap();
        x.execute(&first_row(&x, 2, 4, &[9])).unwrap();
        assert!(!x.slot_state_is_zero(1).unwrap());
        assert!(!x.slot_state_is_zero(2).unwrap());
        assert!(x.slot_state_is_zero(0).unwrap());
        let scratch_is_zero = |x: &DsCudaExecutor| {
            let n = x.half(Half::Decoder);
            let bytes: Vec<u8> = n.s.x.read(n.dev.as_ref(), 0, n.s.x.bytes).unwrap();
            bytes.iter().all(|b| *b == 0)
        };
        assert!(!scratch_is_zero(&x));

        let mut idle = step(&x);
        idle.maintenance = vec![Maintenance::ResetSlot { slot: 1 }];
        x.execute(&idle).unwrap();
        assert!(x.slot_state_is_zero(1).unwrap());
        assert!(!x.slot_state_is_zero(2).unwrap(), "another slot's state");
        assert!(scratch_is_zero(&x));
        assert_eq!(x.idle_scrubs(), 1);
        x.execute(&step(&x)).unwrap();
        assert_eq!(x.idle_scrubs(), 1);

        // A zeroed block is zero, and only it.
        assert!(!x.block_is_zero(1, 4).unwrap());
        let mut zero = step(&x);
        zero.maintenance = vec![Maintenance::Zero { group: 1, block: 3 }];
        x.execute(&zero).unwrap();
        assert!(x.block_is_zero(1, 3).unwrap());
        assert!(!x.block_is_zero(1, 4).unwrap());
    }
}
