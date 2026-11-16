//! [`DsCudaExecutor`]: the serving core's [`Executor`] for DeepSeek-V4.1 on two logical
//! devices, eager (every step is launched afresh; no graph is captured).
//!
//! It reports the model as the CPU reference executor does (the same groups, ratios,
//! devices and coverage, the window chain, the token history, and with `draft_width` > 0
//! the drafter's group) and runs a step in that executor's order: maintenance (the host
//! tier's `Demote`, `Promote` and `ZeroHost` among it, with `host_blocks` > 0) and table
//! updates; the draft pass of the rows that draft; the decoder's device gathers the
//! Engram rows of the tables it holds; the encoder runs every layer before the split at
//! every position, then the split layer's bank stub; the crossing is checked; the
//! decoder stores the bank entries, runs each chain layer over the rows the row rule
//! gives it, the head over the sampling rows and drafted positions, and samples on its
//! device, or accepts a drafting row's drafts there.
//!
//! Not supported: decode graphs, and the drafter's confidence head. A row asking for more
//! drafts than `draft_width` is a host bug and panics, as at the seam.

use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::{Executor, ExecutorError, Maintenance, Slot, StepInput, StepOutput};
use eidola_engine::sampling::Stream;
use eidola_engine::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK, WindowChain,
};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::engram::EngramHasher;

use super::abi::{
    DT_BF16, DT_F32, EmbedParams, EngramGatherParams, KV_FP4_E4M3, KV_FP4_E8M0, LinearParams,
    ROT_ROTATE, STATUS_BAD_INDEX, STATUS_BAD_TOKEN,
};
use super::boundary::{GatheredRows, StepTransfer, ToDecoder, ToEncoder};
use super::gemm::GemmForm;
use super::half::{AcceptArgs, CudaDevice, Device, SampleArgs};
use super::kv::{Cell, GroupGeometry, KvStore};
use super::model::{BANK, Dst, HalfNet, LayerRun, MemoryBytes, WINDOW};
use super::plan::{DraftPlan, PlanRow, StepPlan, pool_plan, window_slots};
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
    /// Physical blocks per bank group (full attention), including the null block 0.
    pub num_blocks: u32,
    /// Physical blocks per window group (sliding: the encoder's, the decoder's and the
    /// drafter's windows), including the null block 0.
    pub window_blocks: u32,
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
    /// Drafted tokens per speculative step, at most the drafter's slots; 0 disables
    /// drafting and the drafter's KV group.
    pub draft_width: u32,
    /// Blocks per group of the host tier (page-locked host memory holding demoted
    /// ciphertext), including the null block 0; 0 for no host tier.
    pub host_blocks: u32,
    /// Keep a position tag per KV entry on the host and refuse a step that reads KV no
    /// step computed (`ds::kv`).
    pub position_tags: bool,
    /// Testing aid: keep each step's [`StepTransfer`] and [`StepStats`].
    pub record: bool,
    /// The form of the quantised GEMMs (`lin8`, `lin4`) on both halves
    /// ([`super::gemm`]): [`GemmForm::Exact`] (the reference) or [`GemmForm::Banded`]
    /// (tensor cores, the same bits); [`GemmForm::Tolerance`] only on a CUDA device.
    pub gemm: GemmForm,
}

/// Wall-clock time of one step's parts, each waited for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    /// Positions of the step.
    pub rows: u32,
    /// Layer-positions the chain computed.
    pub chain_work: u64,
    /// Tokens produced.
    pub sampled: u32,
    pub maintenance_ms: f64,
    /// The draft pass, its sampling and the read-back of the drafts.
    pub draft_ms: f64,
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
    draft_logits: Vec<(Slot, Vec<Vec<f32>>)>,
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

/// The seam's description of the model under `cfg`: the CPU reference executor's, the
/// drafter's group among it when drafting.
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
                num_blocks: match attention {
                    AttentionKind::Full => cfg.num_blocks,
                    AttentionKind::Sliding { .. } => cfg.window_blocks,
                },
                host_blocks: cfg.host_blocks,
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
        max_draft_tokens: cfg.draft_width,
        num_state_slots: cfg.num_state_slots,
        buckets: cfg.buckets.clone(),
        window_chain: Some(WindowChain {
            depth: narrow(shape.depth(), "chain depth")?,
            window,
        }),
        token_history: narrow(shape.history, "token history")?,
    };
    let mut spec = spec;
    if shape.drafting() {
        // A drafting row reads the `window` positions before its own.
        let mut drafter = group(
            "drafter",
            AttentionKind::Sliding { window: window + 1 },
            shape.drafter_blocks,
            1,
            1,
            Coverage::Chain,
        )?;
        drafter.role = KvRole::Drafter;
        spec.kv_groups.push(drafter);
    }
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
        let shape = check_supported(c, cfg.draft_width)?;
        let spec = model_spec(c, &shape, cfg)?;
        if c.max_positions
            .is_some_and(|m| cfg.max_model_len as usize > m)
        {
            return Err(CudaError::new(
                "max_model_len is past the positions the checkpoint declares",
            ));
        }
        // A launch's grid rows: a step's positions, and its routed pairs.
        let rows = cfg.buckets.iter().map(|b| b.max_tokens).max().unwrap_or(0);
        if u64::from(rows) * c.moe.n_active as u64 > 65_535 {
            return Err(CudaError::new(
                "a bucket's routed pairs pass the 65,535 rows of one launch",
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
        Self::build(encoder, decoder, Some(store.as_ref()), c, token_map, cfg)
    }

    /// An executor laid out over two devices without a checkpoint: every allocation at
    /// its size, no weight uploaded. With [`super::dry::DryDevice`] it answers what a
    /// configuration puts on each device and whether its launches are accepted.
    pub fn layout(
        encoder: Box<dyn Device>,
        decoder: Box<dyn Device>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
    ) -> Result<DsCudaExecutor> {
        Self::build(encoder, decoder, None, c, token_map, cfg)
    }

    fn build(
        encoder: Box<dyn Device>,
        decoder: Box<dyn Device>,
        store: Option<&WeightSet>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
    ) -> Result<DsCudaExecutor> {
        Self::build_with(encoder, decoder, store, c, token_map, cfg, true)
    }

    /// As `build`; without `routed` the backbone's routed experts stay off the devices
    /// (`HalfWeights::load_with`).
    pub(crate) fn build_with(
        encoder: Box<dyn Device>,
        decoder: Box<dyn Device>,
        store: Option<&WeightSet>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
        routed: bool,
    ) -> Result<DsCudaExecutor> {
        let (shape, spec) = Self::preflight(c, cfg)?;
        let (mut encoder, mut decoder) = (encoder, decoder);
        encoder.select_gemm(cfg.gemm)?;
        decoder.select_gemm(cfg.gemm)?;
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
        let seqs = cfg.buckets.iter().map(|b| b.max_seqs).max().unwrap_or(0);
        // A step's positions, or a draft pass's rows: every slot of every drafting row.
        let pass = seqs * narrow::<u32, usize>(shape.draft_slots, "draft slots")?;
        let max_rows = cfg
            .buckets
            .iter()
            .map(|b| b.max_tokens)
            .max()
            .unwrap_or(0)
            .max(pass);
        // The head's rows: a sampling row's last host position and its drafts.
        let samples = seqs * (cfg.draft_width + 1);
        let cap = shape.open_cap();
        let mut halves = Vec::new();
        for (half, dev) in [(Half::Encoder, encoder), (Half::Decoder, decoder)] {
            let w = HalfWeights::load_with(
                dev.as_ref(),
                store,
                c,
                &shape,
                half,
                cfg.max_model_len,
                true,
                routed,
            )?;
            let mut kv = KvStore::new(
                dev.as_ref(),
                geometry(c, &spec, half)?,
                cfg.num_state_slots,
                spec.max_blocks_per_seq(),
            )?;
            if cfg.host_blocks > 0 {
                kv.set_host_tier(dev.as_ref(), cfg.host_blocks)?;
            }
            if cfg.position_tags {
                kv.enable_tags();
            }
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
                seqs,
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
            draft_logits: Vec::new(),
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

    /// Both halves, the encoder's first.
    #[cfg(feature = "dev-staged-experts")]
    pub(crate) fn halves_mut(&mut self) -> [&mut HalfNet; 2] {
        [&mut self.encoder, &mut self.decoder]
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

    /// Per drafting row since the last call, in step order: its slot and each draft
    /// slot's logits, given the token before it in the chain (with `record`).
    pub fn take_draft_logits(&mut self) -> Vec<(Slot, Vec<Vec<f32>>)> {
        std::mem::take(&mut self.draft_logits)
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

    /// A host-tier block's bytes.
    pub fn host_block_bytes(&self, group: u32, host: u32) -> Result<Vec<u8>> {
        let (n, g) = self.net(group);
        n.kv.read_host_block(n.dev.as_ref(), g, host)
    }

    /// Whether a host-tier block holds ciphertext.
    pub fn host_block_is_encrypted(&self, group: u32, host: u32) -> bool {
        let (n, g) = self.net(group);
        n.kv.host_is_encrypted(g, host)
    }

    /// Whether every scratch buffer of both halves (the step's part of the buffers that
    /// also hold per-slot state) is zero.
    pub fn scratch_is_zero(&self) -> Result<bool> {
        Ok(self.encoder.scratch_is_zero()? && self.decoder.scratch_is_zero()?)
    }

    /// Whether a slot's per-sequence model state (the pooled sources' open entries) is
    /// scrubbed.
    pub fn slot_state_is_zero(&self, slot: Slot) -> Result<bool> {
        self.encoder.state_is_zero(slot)
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<()> {
        match tokens
            .iter()
            .find(|t| **t as usize >= self.c.vocab_size || self.c.image_token_id == Some(**t))
        {
            Some(t) => Err(CudaError::new(format!(
                "token {t} cannot be embedded (vocabulary {}, image token {:?})",
                self.c.vocab_size, self.c.image_token_id
            ))),
            None => Ok(()),
        }
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
                Maintenance::Demote { group, host, .. }
                | Maintenance::Promote { group, host, .. }
                | Maintenance::ZeroHost { group, host } => {
                    let blocks = self.encoder.kv.host_blocks();
                    if host == NULL_BLOCK || host >= blocks {
                        return Err(CudaError::new(format!(
                            "maintenance {m:?} names a host block outside a host tier of \
                             {blocks} blocks"
                        )));
                    }
                    (group, None)
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
                Maintenance::Demote { group, block, host } => {
                    let (n, g) = self.net_mut(group);
                    n.kv.demote(n.dev.as_ref(), g, block, host)?;
                }
                Maintenance::Promote { group, host, block } => {
                    let (n, g) = self.net_mut(group);
                    n.kv.promote(n.dev.as_ref(), g, host, block)?;
                }
                Maintenance::ZeroHost { group, host } => {
                    let (n, g) = self.net_mut(group);
                    n.kv.zero_host(g, host);
                }
            }
        }
        for u in &step.table_updates {
            let (n, g) = self.net_mut(u.group);
            n.kv.set_table(g, u.slot, u.index, u.block);
        }
        for n in [&mut self.encoder, &mut self.decoder] {
            n.kv.flush_tables(n.dev.as_ref())?;
            // A host block is zero before the step returns.
            n.kv.finish_host(n.dev.as_ref())?;
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
            assert!(
                k <= self.shape.draft_width,
                "{k} drafts requested; the executor drafts {}",
                self.shape.draft_width
            );
            assert!(
                e.context_len + e.num_tokens > 1 || k == 0,
                "slot {}: {k} drafts requested at position 0",
                e.slot
            );
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
                drafts: k,
            });
            assert!(
                k == 0 || chain_start == e.context_len,
                "slot {}: a drafting row runs every chain layer at every position",
                e.slot
            );
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
    fn step_arrays(n: &mut HalfNet, plan: &StepPlan, writes: u32, queries: u32) -> Result<()> {
        n.items = plan.items.iter().map(|it| (it.slot, it.pos)).collect();
        let n = &*n;
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
    /// Runs `after` on a half once layer `l` has been queued (a tap for measurements).
    fn no_tap(_: usize, _: &HalfNet) -> Result<()> {
        Ok(())
    }

    /// The encoder half over every position of the step, then the split layer's bank
    /// stub; returns what crosses to the decoder. `tap` runs after each layer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_half(
        n: &mut HalfNet,
        c: &DsConfig,
        shape: &DsShape,
        plan: &StepPlan,
        rows: &[PlanRow],
        tokens: &[u32],
        ids: &[Vec<u32>],
        from_decoder: &ToEncoder,
        tap: &mut dyn FnMut(usize, &HalfNet) -> Result<()>,
    ) -> Result<ToDecoder> {
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
                tap: match shape.encoder_targets.iter().position(|t| *t == l) {
                    Some(at) => Some(narrow(at, "target layer")?),
                    None => None,
                },
                build: match spec.is_kv_source {
                    true => Some((source(l)?, &pool)),
                    false => None,
                },
            };
            n.layer(l, &run)?;
            tap(l, n)?;
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
        // The drafter's taps of the encoder's target layers, where its context is written.
        let (mut tap_rows, mut taps) = (0, Vec::new());
        if !shape.encoder_targets.is_empty() {
            let (ds, draft) = n.drafting()?;
            tap_rows = plan.tap_rows();
            taps = ds
                .hidden
                .read(dev, 0, (tap_rows * draft.targets * d.dim) as usize)?;
        }
        check_status(n.status()?)?;
        Ok(ToDecoder {
            positions: plan.items.iter().map(|it| (it.slot, it.pos)).collect(),
            entry_bytes,
            entries,
            rows: boundary,
            streams,
            pre_mix,
            tap_rows,
            taps,
        })
    }

    /// The decoder: the bank entries stored, the chain by the row rule, the head and
    /// the sampler. Returns each sampling row's token and, when asked, its logits.
    /// The decoder half up to the head: the bank entries stored, the chain by the row
    /// rule, and the logits of the plan's head rows left in the half's scratch. Returns
    /// the head rows. `tap` runs after each layer.
    pub fn decode_half(
        n: &mut HalfNet,
        c: &DsConfig,
        shape: &DsShape,
        plan: &StepPlan,
        crossing: &ToDecoder,
        tap: &mut dyn FnMut(usize, &HalfNet) -> Result<()>,
    ) -> Result<u32> {
        let d = n.d.clone();
        let boundary = plan.boundary_rows();

        // The bank entries of every position, section by section.
        let mut written = Vec::with_capacity(crossing.positions.len());
        {
            let dev = n.dev.as_ref();
            let gs = &n.kv.groups[BANK];
            let (entries, block_bytes) = (gs.geom.entries(), gs.geom.block_bytes());
            for (row, (slot, pos)) in crossing.positions.iter().enumerate() {
                let target = n.kv.target(BANK, *slot, *pos);
                written.push((target, *pos));
                let block = (target / entries) as usize * block_bytes;
                let cell = &crossing.entries[row * crossing.entry_bytes..][..crossing.entry_bytes];
                let mut from = 0;
                for (at, bytes) in gs.geom.runs(0, target % entries) {
                    gs.pool.write(dev, block + at, &cell[from..from + bytes])?;
                    from += bytes;
                }
            }
        }
        for (target, pos) in written {
            n.kv.tag(BANK, 0, target, pos);
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
            // The encoder's taps: the first columns of each row's taps.
            if crossing.tap_rows > 0 {
                let (ds, draft) = n.drafting()?;
                let (from, to) = (
                    (draft.encoder_targets * d.dim) as usize,
                    (draft.targets * d.dim) as usize,
                );
                let mut hidden = vec![0u16; crossing.tap_rows as usize * to];
                for (row, src) in hidden
                    .chunks_exact_mut(to)
                    .zip(crossing.taps.chunks_exact(from))
                {
                    row[..from].copy_from_slice(src);
                }
                ds.hidden.write(dev, 0, &hidden)?;
            }
        }
        for j in 0..shape.depth() {
            let l = shape.split + j;
            let run = LayerRun {
                active: plan.chain_rows[j],
                emit: plan.emit_rows(j),
                window_layer: narrow(j, "chain layer")?,
                bank_layer: c.layers[l].kv_source.map(|_| 0),
                build: None,
                tap: match shape.decoder_targets.iter().position(|t| *t == l) {
                    Some(at) => Some(narrow(shape.encoder_targets.len() + at, "target layer")?),
                    None => None,
                },
            };
            n.layer(l, &run)?;
            tap(l, n)?;
        }
        // The drafter's context KV, wherever the top layer computed.
        if shape.drafting() {
            let rows = plan.tap_rows();
            let group = n.kv.groups.len() - 1;
            let targets: Vec<u32> = plan.items[..rows as usize]
                .iter()
                .map(|it| n.kv.target(group, it.slot, it.pos))
                .collect();
            let (ds, _) = n.drafting()?;
            ds.targets.write(n.dev.as_ref(), 0, &targets)?;
            n.drafter_context(rows)?;
            for b in 0..shape.drafter_blocks {
                n.tag_rows(group, narrow(b, "draft block")?, rows);
            }
        }

        let top = plan.top_rows;
        if top == 0 {
            check_status(n.status()?)?;
            return Ok(0);
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
        // SAFETY: `top` normed rows, the head's rows and `top` rows of logits.
        unsafe {
            dev.kernels().head(&LinearParams {
                x: n.s.u.addr,
                w: head,
                out: n.s.logits.addr,
                rows: top,
                n: d.vocab,
                k: d.dim,
                x_dtype: DT_BF16,
                w_dtype: DT_BF16,
                out_dtype: DT_F32,
                ..LinearParams::default()
            })?;
        }
        Ok(top)
    }

    /// The draft pass of the step's drafting rows, on the decoder's device before the
    /// forward, from context KV earlier steps wrote: the pass, then the slots sampled
    /// left to right on the draft stream, each slot's logits conditioned on the token
    /// before it through the Markov head (the previous slot's tokens are read on the
    /// device). Returns each drafting row's drafts, read back for the encoder's device.
    fn draft(&mut self, plan: &DraftPlan, step: &StepInput, last: &[u32]) -> Result<Vec<Vec<u32>>> {
        let count = plan.count();
        if count == 0 {
            return Ok(Vec::new());
        }
        let n = &mut self.decoder;
        let d = n.d.clone();
        let (_, draft) = n.drafting()?;
        let pass = count * plan.slots;
        let group = n.kv.groups.len() - 1;
        {
            let (ds, _) = n.drafting()?;
            let dev = n.dev.as_ref();
            // Slot 0 takes the row's last host token, the others the noise token.
            let mut tokens = last.to_vec();
            tokens.resize(pass as usize, draft.noise_token);
            let slots = plan.slot_lists(d.window, |slot, pos| n.kv.entry(group, slot, pos));
            if n.kv.tagged() {
                for row in &plan.rows {
                    for b in 0..draft.blocks {
                        n.check_window(group, b, row.1, row.2 - 1)?;
                    }
                }
            }
            let mut pre = vec![0.0f32; pass as usize * d.mix_words as usize];
            for row in pre.chunks_exact_mut(d.mix_words as usize) {
                row[0] = 1.0;
            }
            let positions = plan.positions();
            if positions.iter().any(|p| *p >= d.positions) {
                return Err(CudaError::new(
                    "a draft pass stands past the positions the RoPE tables hold",
                ));
            }
            n.s.tokens.write(dev, 0, &tokens)?;
            n.s.positions.write(dev, 0, &positions)?;
            n.s.mix_b.write(dev, 0, &pre)?;
            n.s.status.write(dev, 0, &[0u32])?;
            ds.slots.write(dev, 0, &slots)?;
            ds.previous.write(dev, 0, last)?;
        }
        n.draft_pass(pass)?;
        let sampleable = self.spec.sampleable_vocab_size;
        let lane = |i: u32| (i * count) as usize;
        let mut recorded: Vec<Vec<Vec<f32>>> = vec![Vec::new(); count as usize];
        for i in 0..plan.width() {
            let rows = plan.drafting(i);
            let (ds, _) = n.drafting()?;
            let previous = match i {
                0 => ds.previous.addr,
                _ => ds.tokens.at(lane(i - 1) * 4, 0),
            };
            n.draft_logits(rows, plan.row(i, 0), previous)?;
            let (ds, _) = n.drafting()?;
            let dev = n.dev.as_ref();
            if self.record {
                let all: Vec<f32> = ds.logits.read(dev, 0, (rows * d.vocab) as usize)?;
                for (r, l) in all.chunks_exact(d.vocab as usize).enumerate() {
                    recorded[r].push(l.to_vec());
                }
            }
            let samples: Vec<SampleRow> = plan.rows[..rows as usize]
                .iter()
                .zip(0u32..)
                .map(|(r, at)| SampleRow::new(&step.seqs[r.0 as usize].sampling, r.2 + 1 + i, at))
                .collect();
            // SAFETY: `rows` rows of one slot's logits; the slot's lane of the
            // distributions and of the tokens.
            unsafe {
                dev.sample(&SampleArgs {
                    logits: ds.logits.addr,
                    logits_stride: d.vocab,
                    logit_rows: rows,
                    n: sampleable,
                    rows: &samples,
                    rows_dev: ds.rows.addr,
                    probs: ds.probs.at(lane(i) * sampleable as usize * 8, 0),
                    tokens: ds.tokens.at(lane(i) * 4, 0),
                    status: n.s.status.addr,
                    draw: Some(Stream::Draft),
                })?;
            }
        }
        let (ds, _) = n.drafting()?;
        let tokens: Vec<u32> = ds.tokens.read(n.dev.as_ref(), 0, lane(plan.width()))?;
        check_status(n.status()?)?;
        if self.record {
            let mut order: Vec<usize> = (0..plan.rows.len()).collect();
            order.sort_by_key(|r| plan.rows[*r].0);
            for r in order {
                self.draft_logits
                    .push((plan.rows[r].1, std::mem::take(&mut recorded[r])));
            }
        }
        Ok(plan
            .rows
            .iter()
            .enumerate()
            .map(|(r, row)| (0..row.3).map(|i| tokens[lane(i) + r]).collect())
            .collect())
    }

    /// The decoder: [`DsCudaExecutor::decode_half`], then the sampler and, for the
    /// drafting rows, chain acceptance over the target's distributions at the drafted
    /// positions and the conditionals the drafts were drawn from. Returns each row's
    /// produced tokens and, when asked, the logits each was drawn from.
    #[allow(clippy::type_complexity)]
    fn decode(
        &mut self,
        plan: &StepPlan,
        dplan: &DraftPlan,
        rows: &[PlanRow],
        step: &StepInput,
        crossing: &ToDecoder,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<Vec<f32>>>)> {
        let top = Self::decode_half(
            &mut self.decoder,
            &self.c,
            &self.shape,
            plan,
            crossing,
            &mut Self::no_tap,
        )?;
        let mut produced = vec![Vec::new(); rows.len()];
        let mut logits = vec![Vec::new(); rows.len()];
        if top == 0 {
            return Ok((produced, logits));
        }
        let n = &self.decoder;
        let (d, dev) = (&n.d, n.dev.as_ref());
        let sampleable = self.spec.sampleable_vocab_size;
        let logit_row = |row: usize, pos: u32| {
            plan.row_of(u32::try_from(row).expect("rows of a step"), pos)
                .expect("the head read every sampling position")
        };
        // Rows without drafts sample plainly.
        let plain: Vec<usize> = (0..rows.len())
            .filter(|i| rows[*i].sample && rows[*i].drafts == 0)
            .collect();
        let samples: Vec<SampleRow> = plain
            .iter()
            .map(|i| {
                SampleRow::new(
                    &step.seqs[*i].sampling,
                    rows[*i].p + 1,
                    logit_row(*i, rows[*i].p),
                )
            })
            .collect();
        // SAFETY: the logits of `top` rows; the sampler's buffers hold a row per head
        // row of the largest step.
        unsafe {
            dev.sample(&SampleArgs {
                logits: n.s.logits.addr,
                logits_stride: d.vocab,
                logit_rows: top,
                n: sampleable,
                rows: &samples,
                rows_dev: n.s.srows.addr,
                probs: n.s.probs.addr,
                tokens: n.s.stokens.addr,
                status: n.s.status.addr,
                draw: Some(Stream::Sample),
            })?;
        }
        // Drafting rows: the target's distributions at `p ..= p + k`, then acceptance.
        let count = dplan.count();
        let stride = self.spec.max_draft_tokens + 1;
        if count > 0 {
            let (ds, _) = n.drafting()?;
            let first = plain.len();
            let (mut targets, mut target_row, mut accept) = (Vec::new(), Vec::new(), Vec::new());
            for r in &dplan.rows {
                let params = &step.seqs[r.0 as usize].sampling;
                target_row.push(narrow::<u32, usize>(first + targets.len(), "target rows")?);
                for pos in r.2..=r.2 + r.3 {
                    targets.push(SampleRow::new(
                        params,
                        pos + 1,
                        logit_row(r.0 as usize, pos),
                    ));
                }
                accept.push(SampleRow::new(params, r.2 + 1, 0));
            }
            let draft_row: Vec<u32> = (0..count).collect();
            let num_drafts: Vec<u32> = dplan.rows.iter().map(|r| r.3).collect();
            ds.accept_rows.write(dev, 0, &accept)?;
            ds.target_row.write(dev, 0, &target_row)?;
            ds.draft_row.write(dev, 0, &draft_row)?;
            ds.num_drafts.write(dev, 0, &num_drafts)?;
            // SAFETY: as above, after the plain rows' distributions; the acceptance
            // reads the rows just uploaded, the drafts where the draft pass left them.
            unsafe {
                dev.sample(&SampleArgs {
                    logits: n.s.logits.addr,
                    logits_stride: d.vocab,
                    logit_rows: top,
                    n: sampleable,
                    rows: &targets,
                    rows_dev: n.s.srows.at(first * 32, 0),
                    probs: n.s.probs.at(first * sampleable as usize * 8, 0),
                    tokens: n.s.stokens.at(first * 4, 0),
                    status: n.s.status.addr,
                    draw: None,
                })?;
                dev.accept(&AcceptArgs {
                    target: n.s.probs.addr,
                    draft: ds.probs.addr,
                    n: sampleable,
                    rows: ds.accept_rows.addr,
                    target_row: ds.target_row.addr,
                    draft_row: ds.draft_row.addr,
                    draft_step: count,
                    num_drafts: ds.num_drafts.addr,
                    drafts: ds.tokens.addr,
                    stride,
                    scratch: ds.residual.addr,
                    out: ds.out.addr,
                    counts: ds.counts.addr,
                    status: n.s.status.addr,
                    num_rows: count,
                })?;
            }
            let out: Vec<u32> = ds.out.read(dev, 0, (count * stride) as usize)?;
            let counts: Vec<u32> = ds.counts.read(dev, 0, count as usize)?;
            for (r, row) in dplan.rows.iter().enumerate() {
                let got = counts[r] as usize;
                if got == 0 || got > row.3 as usize + 1 {
                    return Err(CudaError::new("chain acceptance produced no tokens"));
                }
                produced[row.0 as usize] = out[r * stride as usize..][..got].to_vec();
            }
        }
        let tokens: Vec<u32> = n.s.stokens.read(dev, 0, plain.len())?;
        check_status(n.status()?)?;
        for (i, token) in plain.iter().zip(tokens) {
            produced[*i] = vec![token];
        }
        if step.return_logits {
            let all: Vec<f32> = n.s.logits.read(dev, 0, top as usize * d.vocab as usize)?;
            for (i, tokens) in produced.iter().enumerate() {
                for j in 0..tokens.len() {
                    let at = logit_row(i, rows[i].p + u32::try_from(j).expect("a draft")) as usize
                        * d.vocab as usize;
                    logits[i].push(all[at..at + d.vocab as usize].to_vec());
                }
            }
        }
        Ok((produced, logits))
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

        // Drafting, on the decoder's device, before the forward.
        let at = Instant::now();
        let dplan = DraftPlan::new(&rows, narrow(self.shape.draft_slots, "draft slots")?);
        let last: Vec<u32> = dplan
            .rows
            .iter()
            .map(|r| {
                *known[r.0 as usize]
                    .1
                    .last()
                    .expect("a row has a host token")
            })
            .collect();
        self.check_tokens(&last)?;
        let drafts = self.draft(&dplan, step, &last)?;
        let mut known = known;
        for (r, tokens) in dplan.rows.iter().zip(&drafts) {
            known[r.0 as usize].1.extend(tokens);
        }
        stats.draft_ms = ms(at);

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
        self.check_tokens(&tokens)?;
        let ids = self.engram_ids(&plan, &rows, &known)?;

        let at = Instant::now();
        let to_encoder = self.gather(&ids, plan.rows())?;
        check_status(self.decoder.status()?)?;
        stats.gather_ms = ms(at);

        let at = Instant::now();
        let crossing = Self::encode_half(
            &mut self.encoder,
            &self.c,
            &self.shape,
            &plan,
            &rows,
            &tokens,
            &ids,
            &to_encoder,
            &mut Self::no_tap,
        )?;
        stats.encoder_ms = ms(at);

        let at = Instant::now();
        let d = &self.encoder.d;
        crossing.check(
            &plan,
            (d.copies * d.dim) as usize,
            d.copies as usize,
            self.shape.encoder_targets.len() * d.dim as usize,
        )?;
        stats.crossing_ms = ms(at);

        let at = Instant::now();
        let (produced, logits) = self.decode(&plan, &dplan, &rows, step, &crossing)?;
        stats.decoder_ms = ms(at);

        let mut sampled = 0;
        for (i, (tokens, logits)) in produced.into_iter().zip(logits).enumerate() {
            if let Some(&bad) = tokens
                .iter()
                .find(|t| **t >= self.spec.sampleable_vocab_size)
            {
                return Err(CudaError::new(format!(
                    "sampled id {bad} is not sampleable"
                )));
            }
            sampled += tokens.len();
            out.tokens[i * stride as usize..][..tokens.len()].copy_from_slice(&tokens);
            out.num_tokens[i] = narrow(tokens.len(), "produced tokens")?;
            if let Some(l) = out.logits.as_mut() {
                l.push(logits);
            }
        }
        self.chain_work += plan.chain_work();
        if self.record {
            let mut transfer = StepTransfer::of(&plan, &to_encoder, &crossing);
            // In step order, as the reference executor lists them.
            let mut order: Vec<usize> = (0..dplan.rows.len()).collect();
            order.sort_by_key(|r| dplan.rows[*r].0);
            transfer.drafts = order
                .into_iter()
                .map(|r| (dplan.rows[r].1, drafts[r].clone()))
                .collect();
            self.transfers.push(transfer);
            stats.rows = plan.rows();
            stats.chain_work = plan.chain_work();
            stats.sampled = narrow(sampled, "sampled tokens")?;
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
    /// Zeroes every pool, table, state and scratch buffer and the host tier before the
    /// memory is freed:
    /// best effort, since on a failed device the writes fail too.
    fn drop(&mut self) {
        for n in [&mut self.encoder, &mut self.decoder] {
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
            window_blocks: 12,
            num_state_slots: 4,
            max_model_len: 64,
            buckets: vec![Bucket {
                max_seqs: 2,
                max_tokens: 16,
            }],
            sampleable_vocab_size: 95,
            devices: [0, 0],
            draft_width: 0,
            host_blocks: 0,
            position_tags: true,
            record: true,
            gemm: GemmForm::Exact,
        }
    }

    fn executor() -> DsCudaExecutor {
        executor_with(|_| ())
    }

    fn executor_with(change: impl FnOnce(&mut DsCudaExecutorConfig)) -> DsCudaExecutor {
        let mut cfg = config();
        change(&mut cfg);
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
            &cfg,
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
    fn host_blocks_outside_the_tier_are_refused_before_the_step_takes_effect() {
        for host_blocks in [0, 4] {
            let mut x = executor_with(|c| c.host_blocks = host_blocks);
            let mut s = first_row(&x, 1, 3, &[5, 6, 7]);
            for op in [
                Maintenance::Demote {
                    group: 0,
                    block: 2,
                    host: host_blocks,
                },
                Maintenance::Promote {
                    group: 1,
                    host: 0,
                    block: 2,
                },
                Maintenance::ZeroHost {
                    group: 1,
                    host: host_blocks + 1,
                },
            ] {
                s.maintenance = vec![Maintenance::Zero { group: 0, block: 2 }, op];
                let e = x.execute(&s).unwrap_err();
                assert!(e.0.contains("outside a host tier"), "{e}");
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
    }

    /// A block goes to the host tier as the ciphertext it is, byte for byte, and comes
    /// back into another block as the same plaintext; the host block's zero, listed in
    /// the step that promotes it, lands after the copy.
    #[test]
    fn the_host_tier_round_trips_ciphertext() {
        let mut x = executor_with(|c| c.host_blocks = 4);
        x.execute(&first_row(&x, 1, 3, &[5, 6, 7])).unwrap();
        let key = eidola_engine::secret::KvKey::from_bytes([7; 32]);
        let nonce = eidola_engine::kv_cipher::KvNonce::from_words([1, 2, 3]);
        for group in 0..4u32 {
            let plain = x.block_bytes(group, 3).unwrap();
            assert!(plain.iter().any(|b| *b != 0));
            let mut down = step(&x);
            down.kv_keys = vec![key.clone()];
            down.maintenance = vec![
                Maintenance::Encrypt {
                    group,
                    block: 3,
                    key: 0,
                    nonce,
                },
                Maintenance::Demote {
                    group,
                    block: 3,
                    host: 2,
                },
                Maintenance::Zero { group, block: 3 },
            ];
            x.execute(&down).unwrap();
            let held = x.host_block_bytes(group, 2).unwrap();
            assert!(x.host_block_is_encrypted(group, 2));
            assert!(x.block_is_zero(group, 3).unwrap());
            let same = held.iter().zip(&plain).filter(|(a, b)| a == b).count();
            assert!(
                same * 10 < plain.len(),
                "group {group}: the host holds plaintext"
            );
            assert!(
                x.host_block_bytes(group, 1)
                    .unwrap()
                    .iter()
                    .all(|b| *b == 0)
            );

            let mut up = step(&x);
            up.kv_keys = vec![key.clone()];
            up.maintenance = vec![
                Maintenance::Promote {
                    group,
                    host: 2,
                    block: 5,
                },
                Maintenance::ZeroHost { group, host: 2 },
                Maintenance::Decrypt {
                    group,
                    block: 5,
                    key: 0,
                    nonce,
                },
            ];
            x.execute(&up).unwrap();
            assert_eq!(x.block_bytes(group, 5).unwrap(), plain, "group {group}");
            assert!(!x.host_block_is_encrypted(group, 2));
            assert!(
                x.host_block_bytes(group, 2)
                    .unwrap()
                    .iter()
                    .all(|b| *b == 0)
            );
        }
    }

    #[test]
    #[should_panic(expected = "demote of a plaintext block")]
    fn plaintext_never_reaches_the_host_tier() {
        let mut x = executor_with(|c| c.host_blocks = 4);
        x.execute(&first_row(&x, 1, 3, &[5, 6, 7])).unwrap();
        let mut s = step(&x);
        s.maintenance = vec![Maintenance::Demote {
            group: 1,
            block: 3,
            host: 2,
        }];
        let _ = x.execute(&s);
    }

    /// The rows of `tokens` at `context ..` of slot 0, every group's first two blocks
    /// mapped to blocks 1 and 2.
    fn later_row(x: &DsCudaExecutor, context: u32, tokens: &[u32], chain_start: u32) -> StepInput {
        let mut s = first_row(x, 0, 1, tokens);
        s.table_updates.extend((0..4).map(|group| TableUpdate {
            slot: 0,
            group,
            index: 1,
            block: 2,
        }));
        s.seqs[0].context_len = context;
        let n = u32::try_from(tokens.len()).unwrap();
        s.positions = (context..context + n).collect();
        s.aux.chain_start = vec![chain_start];
        s
    }

    /// With position tags a step that reads KV no step computed is refused: a chain
    /// layer's window before the layer's start, and a context whose blocks are mapped
    /// and empty. Without them the same steps read whatever the blocks hold.
    #[test]
    fn a_read_of_kv_no_step_computed_is_refused() {
        let tokens: Vec<u32> = (0..10).collect();
        for tags in [true, false] {
            // The top layer starts at 6 and the row samples at 9: its top-layer window
            // reaches back to 0, which that layer does not compute.
            let mut x = executor_with(|c| c.position_tags = tags);
            let got = x.execute(&later_row(&x, 0, &tokens, 6));
            match tags {
                true => {
                    let e = got.unwrap_err();
                    assert!(e.0.contains("group 3 layer 3"), "{e}");
                    assert!(e.0.contains("no step computed"), "{e}");
                }
                false => assert!(got.is_ok()),
            }
            // The whole chain from 0 is a context every layer computed.
            let mut x = executor_with(|c| c.position_tags = tags);
            x.execute(&later_row(&x, 0, &tokens, 0)).unwrap();
            x.execute(&later_row(&x, 10, &[3], 10)).unwrap();

            // A context nothing computed.
            let mut x = executor_with(|c| c.position_tags = tags);
            let got = x.execute(&later_row(&x, 10, &[3], 10));
            assert_eq!(got.is_err(), tags);
        }
    }

    /// A zeroed block's tags go with its bytes, and a copy carries them.
    #[test]
    fn tags_follow_zero_and_copy() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut x = executor();
        x.execute(&later_row(&x, 0, &tokens, 0)).unwrap();
        // Every group's second block moves to block 7.
        let mut s = later_row(&x, 10, &[3], 10);
        s.maintenance = (0..4)
            .map(|group| Maintenance::Copy {
                group,
                src: 2,
                dst: 7,
            })
            .collect();
        for u in s.table_updates.iter_mut().filter(|u| u.index == 1) {
            u.block = 7;
        }
        x.execute(&s).unwrap();
        // Zeroed, the first block no longer holds the context.
        let mut s = later_row(&x, 11, &[4], 11);
        for u in s.table_updates.iter_mut().filter(|u| u.index == 1) {
            u.block = 7;
        }
        s.maintenance = vec![Maintenance::Zero { group: 3, block: 1 }];
        let e = x.execute(&s).unwrap_err();
        assert!(e.0.contains("no step computed"), "{e}");
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
    /// rowless step after rows scrubs every scratch buffer, the drafter's included, once.
    #[test]
    fn resets_and_idle_steps_scrub() {
        let mut x = executor_with(|c| c.draft_width = 4);
        let mut first = first_row(&x, 1, 3, &[5, 6, 7]);
        let drafter = |s: &mut StepInput, block| {
            let slot = s.seqs[0].slot;
            s.table_updates.push(TableUpdate {
                slot,
                group: 4,
                index: 0,
                block,
            });
        };
        drafter(&mut first, 3);
        x.execute(&first).unwrap();
        // A drafting step, so the drafter's scratch holds a pass.
        let mut next = first_row(&x, 1, 3, &[8]);
        next.table_updates.clear();
        next.seqs[0].context_len = 3;
        next.seqs[0].num_drafts = 2;
        next.positions = vec![3];
        next.aux.chain_start = vec![3];
        x.execute(&next).unwrap();
        let mut other = first_row(&x, 2, 4, &[9]);
        drafter(&mut other, 4);
        x.execute(&other).unwrap();
        assert!(!x.slot_state_is_zero(1).unwrap());
        assert!(!x.slot_state_is_zero(2).unwrap());
        assert!(x.slot_state_is_zero(0).unwrap());
        assert!(!x.scratch_is_zero().unwrap());
        let (ds, _) = x.half(Half::Decoder).drafting().unwrap();
        let pass: Vec<u8> = ds
            .base
            .read(x.half(Half::Decoder).dev.as_ref(), 0, ds.base.bytes)
            .unwrap();
        assert!(
            pass.iter().any(|b| *b != 0),
            "the draft pass left its logits"
        );

        let mut idle = step(&x);
        idle.maintenance = vec![Maintenance::ResetSlot { slot: 1 }];
        x.execute(&idle).unwrap();
        assert!(x.slot_state_is_zero(1).unwrap());
        assert!(!x.slot_state_is_zero(2).unwrap(), "another slot's state");
        assert!(x.scratch_is_zero().unwrap());
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

    /// The released configuration with five drafts, laid out with no checkpoint on two
    /// devices of one B300's memory that only check their launches: both halves fit
    /// (the decoder's with the drafter and its Engram table), and a prefill step and a
    /// drafting step make only launches their wrappers accept. The dry device reads
    /// back zeros, so the drafting step ends where it needs the acceptance's counts.
    #[test]
    fn the_released_model_drafts_within_one_device_a_half() {
        use super::super::dry::DryDevice;
        let c = super::super::support::tests::released_config();
        let e = c.engram.clone().unwrap();
        let map: Vec<u32> = (0..c.vocab_size)
            .map(|i| u32::try_from(i % e.compressed_vocab).unwrap())
            .collect();
        let cfg = DsCudaExecutorConfig {
            block_size: 32,
            num_blocks: 4096,
            window_blocks: 4096,
            num_state_slots: 16,
            max_model_len: 4096,
            buckets: vec![Bucket {
                max_seqs: 16,
                max_tokens: 256,
            }],
            sampleable_vocab_size: u32::try_from(c.vocab_size).unwrap(),
            devices: [0, 1],
            draft_width: 5,
            host_blocks: 8192,
            position_tags: true,
            record: false,
            gemm: GemmForm::Exact,
        };
        let capacity = 275_040usize << 20;
        let device = || -> Box<dyn Device> { Box::new(DryDevice::new(Some(capacity))) };
        let mut x = DsCudaExecutor::layout(device(), device(), &c, Some(map), &cfg).unwrap();
        assert_eq!(x.spec.kv_groups.len(), 5);
        assert_eq!(x.spec.max_draft_tokens, 5);
        let [enc, dec] = x.memory();
        eprintln!("{enc:?}\n{dec:?}");
        assert!(enc.total() < capacity as u64 && dec.total() < capacity as u64);
        // The drafter's weights are on the decoder's device beside its checkpoint side.
        assert!(dec.weights > 250_527_512_880 + 7_000_000_000);
        assert!(x.decoder.kv.host_bytes() > 0);

        let groups = 0..5u32;
        let map = |slot: Slot, index: u32, block: u32| {
            groups.clone().map(move |group| TableUpdate {
                slot,
                group,
                index,
                block,
            })
        };
        let row = |slot, context: u32, n: u32, drafts| SeqEntry {
            slot,
            token_start: context,
            num_tokens: n,
            context_len: context,
            num_drafts: drafts,
            sample: true,
            sampling: SamplingParams::greedy(),
        };
        // Two rows of 100 and 60 tokens.
        let mut s = step(&x);
        s.table_updates = (0..4)
            .flat_map(|i| map(0, i, 1 + i))
            .chain((0..2).flat_map(|i| map(1, i, 5 + i)))
            .collect();
        s.seqs = vec![row(0, 0, 100, 0), row(1, 0, 60, 0)];
        s.seqs[1].token_start = 100;
        s.token_ids = (0..160).collect();
        s.positions = (0..100).chain(0..60).collect();
        s.aux = StepAux {
            chain_start: vec![0, 0],
            history_ids: vec![0; 2 * x.spec.token_history as usize],
        };
        let out = x.execute(&s).unwrap();
        assert_eq!(out.num_tokens, vec![1, 1]);

        // Both draft, five and three tokens.
        let mut s = step(&x);
        s.seqs = vec![row(0, 100, 1, 5), row(1, 60, 1, 3)];
        s.seqs[0].token_start = 0;
        s.seqs[1].token_start = 1;
        s.token_ids = vec![7, 8];
        s.positions = vec![100, 60];
        s.aux = StepAux {
            chain_start: vec![100, 60],
            history_ids: vec![0; 2 * x.spec.token_history as usize],
        };
        let e = x.execute(&s).unwrap_err();
        assert!(e.0.contains("chain acceptance produced no tokens"), "{e}");
    }

    /// What dropping the executor runs leaves nothing but weights: every pool, table,
    /// slot state, scratch buffer and host block is zero.
    #[test]
    fn the_drop_scrub_covers_every_allocation() {
        let mut x = executor_with(|c| c.host_blocks = 4);
        x.execute(&first_row(&x, 1, 3, &[5, 6, 7])).unwrap();
        let mut down = step(&x);
        down.kv_keys = vec![eidola_engine::secret::KvKey::from_bytes([7; 32])];
        let nonce = eidola_engine::kv_cipher::KvNonce::from_words([1, 2, 3]);
        down.maintenance = vec![
            Maintenance::Encrypt {
                group: 1,
                block: 3,
                key: 0,
                nonce,
            },
            Maintenance::Demote {
                group: 1,
                block: 3,
                host: 2,
            },
        ];
        // Rows again, so the scratch holds a step when the scrub runs.
        let mut s = first_row(&x, 2, 4, &[9]);
        s.kv_keys = std::mem::take(&mut down.kv_keys);
        s.maintenance = std::mem::take(&mut down.maintenance);
        x.execute(&s).unwrap();
        assert!(x.host_block_bytes(1, 2).unwrap().iter().any(|b| *b != 0));
        for n in [&mut x.encoder, &mut x.decoder] {
            n.scrub_all().unwrap();
        }
        assert!(x.scratch_is_zero().unwrap());
        for group in 0..4 {
            for block in 0..12 {
                assert!(x.block_is_zero(group, block).unwrap());
            }
            for host in 1..4 {
                assert!(
                    x.host_block_bytes(group, host)
                        .unwrap()
                        .iter()
                        .all(|b| *b == 0)
                );
            }
        }
        for n in [&x.encoder, &x.decoder] {
            for b in n.buffers() {
                let bytes: Vec<u8> = b.read(n.dev.as_ref(), 0, b.bytes).unwrap();
                assert!(bytes.iter().all(|b| *b == 0), "{b:?}");
            }
            for g in 0..2 {
                assert!(
                    n.kv.read_table(n.dev.as_ref(), g, 1)
                        .unwrap()
                        .iter()
                        .all(|w| *w == 0)
                );
            }
        }
    }
}
