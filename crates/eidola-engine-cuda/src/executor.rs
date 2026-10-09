//! [`CudaExecutor`]: the serving core's [`Executor`] on one GPU.
//!
//! A step runs the seam's sequence exactly (`eidola-engine/src/executor.rs`):
//! maintenance in order, table updates in order, the target forward over every
//! host token (KV written through the tables), then sampling on the device with
//! the core's sampling semantics over the sampleable vocabulary. With a draft
//! width ([`CudaExecutorConfig::draft_tokens`]) every step is a drafted step
//! ([`crate::draft`]): MTP drafter rows for every row, and for the rows the
//! host gave drafts, drafting, verification and chain acceptance inside the
//! step. Drafts at position 0, more drafts than depths, and drafts on a row
//! that is not one sampled host token are host bugs.
//!
//! Contract checks mirror the CPU reference executor's: a KV write into a block
//! mapped by more than one table entry, or a read through an unmapped entry,
//! panics before anything is launched.
//!
//! With [`CudaGraphs::On`], a pure-decode step (or, drafting, a uniform drafted
//! decode step) that fits a rung of its ladder replays that rung's graph
//! instead of launching eagerly (see [`crate::graph`] and [`crate::draft`]),
//! and without drafting so does a mixed step whose work lists tile at 128 and
//! whose tokens fit a rung of the mixed ladder ([`crate::mixed`]); every
//! check above runs first, the same either way.

use std::cell::Cell;
use std::sync::Arc;

use crate::module::KernelDir;
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::sampling::Stream;
use eidola_engine::spec::{AttentionKind, Bucket, KvGroupSpec, KvRole, ModelSpec};
use eidola_engine_model::config::AttentionKind as ModelAttention;
use eidola_engine_model::config::AttentionSpec;
use eidola_engine_model::safetensors::WeightSet;

use crate::attention::{AttnPlan, AttnRequest, HostPlan, PlanShape};
use crate::device::ImageArch;
use crate::draft::MtpHidden;
use crate::graph::{
    CudaGraphs, DecodeGraphs, DecodePath, DecodeRows, GroupSlots, ProgramArgs, decode_ladder,
    decode_rows, max_decode_pages, pack, rung_for, split_readback,
};
use crate::kv::{GroupGeometry, KvLayout, KvStore};
use crate::launch::dptr;
use crate::mixed::{MixedGraphs, MixedRows, mixed_ladder, mixed_rung};
use crate::model::{
    ForwardInput, GpuModel, InputParts, Kernels, group_geometry, group_layers, read_config,
};
use crate::sampler::{STATUS_NON_FINITE, SampleRow};
use crate::support::{
    Unsupported, check_context, check_device, check_drafting, check_sampleable, check_supported,
};
use crate::weights::ModelWeights;
use crate::{CudaError, Gpu, Result, narrow};

/// Executor configuration: memory geometry and step limits.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaExecutorConfig {
    /// Positions per KV block (every group).
    pub block_size: u32,
    /// Physical blocks per KV group, by kind.
    pub num_blocks: KvBlocks,
    pub num_state_slots: u32,
    pub max_model_len: u32,
    /// Captured step shapes; the last bounds every step.
    pub buckets: Vec<Bucket>,
    /// The tokenizer's vocabulary size (never taken from the weights).
    pub sampleable_vocab_size: u32,
    /// Kernel image to run (the device's own when `None`; `Sm100f` runs the
    /// family image on any CC 10.x part).
    pub image: Option<ImageArch>,
    /// Whether steps replay captured graphs (captured here, at construction:
    /// one per rung of [`decode_ladder`] and of [`mixed_ladder`], or of
    /// [`crate::draft::draft_ladder`] when drafting).
    pub graphs: CudaGraphs,
    /// Draft width `k`: MTP depths drafted per decode step, depth `d` served
    /// by the checkpoint's MTP layer `d` (0 disables drafting; at most
    /// [`crate::support::MTP_LAYERS`]).
    pub draft_tokens: u32,
    /// Which of the target's hidden states every MTP depth conditions on.
    pub mtp_hidden: MtpHidden,
}

/// Physical blocks per KV group, keyed by the group's kind, each including
/// the null block 0. A group the retained layers use needs at least two; one
/// they do not use must have none. The drafter group (the MTP depths' KV and
/// boundary taps) exists exactly when drafting does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvBlocks {
    pub global: u32,
    pub sliding: u32,
    pub drafter: u32,
}

impl KvBlocks {
    /// The block count of each group `keys` names, in order; refuses a count
    /// for a group no retained layer uses.
    pub fn resolve(
        &self,
        keys: &[AttentionSpec],
        drafting: bool,
    ) -> std::result::Result<Vec<u32>, Unsupported> {
        if !drafting && self.drafter != 0 {
            return Err(Unsupported {
                field: "num_blocks.drafter".into(),
                found: self.drafter.to_string(),
                required: "0 (nothing is drafted)".into(),
            });
        }
        let used = |global: bool| {
            keys.iter()
                .any(|k| (k.kind == ModelAttention::Global) == global)
        };
        for (name, global, n) in [
            ("global", true, self.global),
            ("sliding", false, self.sliding),
        ] {
            if n != 0 && !used(global) {
                return Err(Unsupported {
                    field: format!("num_blocks.{name}"),
                    found: n.to_string(),
                    required: "0 (no retained layer uses this group)".into(),
                });
            }
        }
        Ok(keys
            .iter()
            .map(|k| match k.kind {
                ModelAttention::Global => self.global,
                ModelAttention::Sliding { .. } => self.sliding,
            })
            .collect())
    }
}

pub struct CudaExecutor {
    gpu: Gpu,
    spec: ModelSpec,
    cfg: CudaExecutorConfig,
    kv: KvStore,
    model: GpuModel,
    /// Drafting: the planner's constants, the device buffers, and per slot
    /// which positions its state holds.
    draft: Option<Drafting>,
    // Sampling scratch.
    sample_rows: cudarc::driver::CudaSlice<SampleRow>,
    probs: cudarc::driver::CudaSlice<f64>,
    tokens: cudarc::driver::CudaSlice<u32>,
    status: cudarc::driver::CudaSlice<u32>,
    steps: u64,
    /// Testing aid: compute logits at every host token of every step and keep
    /// the last step's (`[tokens, vocab]`, see [`CudaExecutor::take_all_logits`]).
    /// Steps run eagerly while it is set (not with drafting).
    pub record_all_logits: bool,
    /// Testing aid: keep every drafted step's drafts per row (see
    /// [`CudaExecutor::take_drafts`]).
    pub record_drafts: bool,
    drafts: std::cell::RefCell<Vec<DraftRecord>>,
    all_logits: Option<Vec<f32>>,
    /// The captured decode graphs (graphs on).
    decode: Option<DecodeGraphs>,
    /// The captured mixed-step graphs (graphs on, not drafting).
    mixed: Option<MixedGraphs>,
    /// How a step that fits a rung (decode or mixed) runs. A cell so a test
    /// or a measurement can switch it on an executor an engine owns.
    decode_path: Cell<DecodePath>,
    stats: DecodeStats,
}

/// Steps by how they ran (see [`CudaExecutor::decode_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeStats {
    /// Steps launched eagerly (any step no rung holds, or any step with
    /// graphs off).
    pub eager: u64,
    /// Decode steps run as the padded launch sequence without its graph.
    pub direct: u64,
    /// Decode steps that replayed a captured graph.
    pub replayed: u64,
    /// Mixed steps run as the padded launch sequence without its graph.
    pub mixed_direct: u64,
    /// Mixed steps that replayed a captured graph.
    pub mixed_replayed: u64,
}

impl std::fmt::Debug for CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaExecutor")
            .field("spec", &self.spec)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

/// A drafting executor's own state.
struct Drafting {
    ctx: crate::draft::PlanCtx,
    bufs: crate::draft::DraftBuffers,
    /// Per slot: `(base, at)` when its state holds the levels at `base ..`
    /// (entry `j` at `base + j`) and the sequence continues from `at`.
    slots: Vec<Option<(u32, u32)>>,
    graphs: Option<crate::draft::DraftGraphs>,
}

/// One row's drafts in a drafted step (see [`CudaExecutor::record_drafts`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftRecord {
    pub slot: u32,
    /// The row's first host position.
    pub context_len: u32,
    /// The drafts for positions `p + 1 ..= p + k`.
    pub drafts: Vec<u32>,
}

/// What [`CudaExecutor::new`] builds from, once every host-side check passed.
struct Plan {
    config: eidola_engine_model::ModelConfig,
    layer_kv: Vec<crate::model::LayerKv>,
    geometry: Vec<crate::kv::GroupGeometry>,
    spec: ModelSpec,
}

impl CudaExecutor {
    /// Every check [`CudaExecutor::new`] makes that needs no device, in the
    /// same order: the configuration (optionally only `keep_layers` of the
    /// checkpoint), the sampleable vocabulary, the context, the KV geometry
    /// and the resulting [`ModelSpec`], which it returns. Reads only the
    /// checkpoint's `config.json`; nothing is loaded or allocated, so a host
    /// can refuse a configuration on a machine without a device.
    pub fn preflight(
        store: &WeightSet,
        keep_layers: Option<&[usize]>,
        cfg: &CudaExecutorConfig,
    ) -> Result<ModelSpec> {
        Ok(Self::plan(store, keep_layers, cfg)?.spec)
    }

    fn plan(
        store: &WeightSet,
        keep_layers: Option<&[usize]>,
        cfg: &CudaExecutorConfig,
    ) -> Result<Plan> {
        let config = read_config(store, keep_layers)?;
        check_supported(&config)?;
        let depths = cfg.draft_tokens as usize;
        check_drafting(&config, depths)?;
        check_sampleable(cfg.sampleable_vocab_size as usize, config.vocab_size)?;
        check_context(cfg.max_model_len, &config)?;
        let (keys, layer_kv) = group_layers(&config);
        let num_blocks = cfg.num_blocks.resolve(&keys, depths > 0)?;
        let mut geometry = group_geometry(&keys, &layer_kv, cfg.block_size, &num_blocks)?;
        if depths > 0 {
            // The drafter group: one plane pair per depth, and a tap of
            // every depth's level per block.
            let a = &config.mtp.attention;
            geometry.push(GroupGeometry {
                num_layers: cfg.draft_tokens,
                num_kv_heads: narrow(a.num_kv_heads, "KV heads")?,
                head_dim_qk: narrow(a.head_dim_qk, "QK head dim")?,
                head_dim_v: narrow(a.head_dim_v, "V head dim")?,
                block_size: cfg.block_size,
                num_blocks: cfg.num_blocks.drafter,
                layout: KvLayout::Planar {
                    taps: cfg.draft_tokens,
                    hidden: narrow(config.hidden_size, "hidden size")?,
                },
            });
        }
        for g in &geometry {
            g.validate()?;
        }
        let kv_groups = keys
            .iter()
            .enumerate()
            .map(|(g, a)| -> Result<KvGroupSpec> {
                let attention = match a.kind {
                    ModelAttention::Global => AttentionKind::Full,
                    ModelAttention::Sliding { window } => AttentionKind::Sliding {
                        window: narrow(window, "window")?,
                    },
                };
                Ok(KvGroupSpec {
                    name: match attention {
                        AttentionKind::Full => "global".into(),
                        AttentionKind::Sliding { window } => format!("sliding-{window}"),
                    },
                    role: KvRole::Target,
                    attention,
                    num_layers: geometry[g].num_layers,
                    num_kv_heads: geometry[g].num_kv_heads,
                    head_dim_qk: geometry[g].head_dim_qk,
                    head_dim_v: geometry[g].head_dim_v,
                    num_blocks: geometry[g].num_blocks,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut kv_groups = kv_groups;
        if let Some(g) = geometry.get(keys.len()) {
            let window = match config.mtp.attention.kind {
                ModelAttention::Sliding { window } => narrow(window, "window")?,
                ModelAttention::Global => unreachable!("check_drafting: a sliding MTP layer"),
            };
            kv_groups.push(KvGroupSpec {
                name: "mtp".into(),
                role: KvRole::Drafter,
                attention: AttentionKind::Sliding { window },
                num_layers: g.num_layers,
                num_kv_heads: g.num_kv_heads,
                head_dim_qk: g.head_dim_qk,
                head_dim_v: g.head_dim_v,
                num_blocks: g.num_blocks,
            });
        }
        let spec = ModelSpec {
            vocab_size: narrow(config.vocab_size, "vocabulary")?,
            sampleable_vocab_size: cfg.sampleable_vocab_size,
            block_size: cfg.block_size,
            max_model_len: cfg.max_model_len,
            kv_groups,
            max_draft_tokens: cfg.draft_tokens,
            // A target pass of at most this many tokens takes the masked
            // expert layout; past it the contiguous layout costs several
            // times as much (`AGENTS.md` → Drafting).
            draft_step_tokens: crate::model::MASKED_TOKENS,
            num_state_slots: cfg.num_state_slots,
            buckets: cfg.buckets.clone(),
        };
        spec.validate().map_err(CudaError::new)?;
        Ok(Plan {
            config,
            layer_kv,
            geometry,
            spec,
        })
    }

    /// Load the model (optionally only `keep_layers` of the checkpoint) and
    /// allocate every pool and buffer.
    pub fn new(
        gpu: Gpu,
        kernels_dir: &KernelDir,
        store: Arc<WeightSet>,
        keep_layers: Option<&[usize]>,
        cfg: CudaExecutorConfig,
    ) -> Result<CudaExecutor> {
        // Everything that can be refused is checked before any device memory
        // is touched: the configuration, the sampleable vocabulary, the
        // context and the KV geometry (`preflight`), then the device; then
        // the kernels are loaded and verified; only then do the weights move.
        let Plan {
            config,
            layer_kv,
            geometry,
            spec,
        } = Self::plan(&store, keep_layers, &cfg)?;
        let image = check_device(gpu.info(), &config, cfg.image)?;
        let kernels = Kernels::load(&gpu, kernels_dir, Some(image))?;
        let depths = cfg.draft_tokens;
        let hidden = config.hidden_size;
        let weights = ModelWeights::load(&gpu, &kernels, store, config, depths as usize)?;
        let last = *spec.buckets.last().expect("validated: a bucket");
        let max_tokens = last.max_tokens as usize;
        let max_rows = last.max_seqs as usize;
        // Per slot, with drafting: every depth's level at each of the
        // `depths + 1` positions a step can leave it at.
        let state_width = (depths as usize + 1)
            .checked_mul(depths as usize)
            .and_then(|x| x.checked_mul(hidden))
            .ok_or_else(|| CudaError::new("drafter state overflows"))?;
        let kv = KvStore::new(
            &gpu,
            geometry,
            cfg.num_state_slots,
            spec.max_blocks_per_seq(),
            state_width,
        )?;
        let model = GpuModel::new(
            &gpu,
            weights,
            kernels,
            layer_kv,
            max_tokens,
            max_tokens,
            cfg.max_model_len as usize,
        )?;
        let s = gpu.stream();
        let v = cfg.sampleable_vocab_size as usize;
        let decode_path = match cfg.graphs {
            CudaGraphs::On => DecodePath::Replay,
            CudaGraphs::Off => DecodePath::Eager,
        };
        let draft = if depths > 0 {
            let ctx = crate::draft::PlanCtx {
                depths,
                hidden: cfg.mtp_hidden,
                block_size: cfg.block_size,
                targets: Vec::new(),
                drafter_window: 0,
                drafter_group_size: 0,
                cap_rows: last.max_seqs,
                max_blocks: spec.max_blocks_per_seq(),
                graph: false,
            };
            Some(Drafting {
                bufs: crate::draft::DraftBuffers::new(
                    &gpu,
                    &ctx,
                    hidden,
                    max_tokens,
                    cfg.sampleable_vocab_size,
                )?,
                ctx,
                slots: vec![None; cfg.num_state_slots as usize],
                graphs: None,
            })
        } else {
            None
        };
        // The plain sampler's scratch (drafting steps use their own).
        let sample_rows = if depths > 0 { 1 } else { max_rows.max(1) };
        let mut ex = CudaExecutor {
            sample_rows: s.alloc_zeros(sample_rows)?,
            probs: s.alloc_zeros(
                sample_rows
                    .checked_mul(v)
                    .ok_or_else(|| CudaError::new("sampler scratch overflows"))?,
            )?,
            tokens: s.alloc_zeros(sample_rows)?,
            status: s.alloc_zeros(1)?,
            gpu,
            spec,
            cfg,
            kv,
            model,
            draft,
            steps: 0,
            record_all_logits: false,
            record_drafts: false,
            drafts: std::cell::RefCell::new(Vec::new()),
            all_logits: None,
            decode: None,
            mixed: None,
            decode_path: Cell::new(decode_path),
            stats: DecodeStats::default(),
        };
        if let Some(d) = ex.draft.as_mut() {
            d.ctx = draft_ctx(&ex.spec, &ex.model, &ex.cfg)?;
        }
        if ex.cfg.graphs == CudaGraphs::On {
            if ex.draft.is_some() {
                ex.capture_draft_graphs()?;
            } else {
                ex.capture_decode_graphs()?;
                ex.capture_mixed_graphs()?;
            }
        }
        Ok(ex)
    }

    /// Query heads per KV head, per target group.
    fn group_sizes(&self) -> Result<Vec<u32>> {
        self.spec
            .kv_groups
            .iter()
            .take(self.model.num_groups())
            .enumerate()
            .map(|(g, group)| {
                let nq = self
                    .model
                    .config()
                    .layers
                    .iter()
                    .zip(&self.model.layer_kv)
                    .find(|(_, l)| l.group == g)
                    .map(|(l, _)| l.attention.num_q_heads)
                    .expect("a layer per group");
                let nq: u32 = narrow(nq, "query heads")?;
                Ok(nq / group.num_kv_heads)
            })
            .collect()
    }

    /// Every target group as the graphs see it.
    fn graph_slots(&self) -> Result<Vec<GroupSlots>> {
        let bs = self.spec.block_size;
        let max_blocks = self.spec.max_blocks_per_seq();
        Ok(self
            .group_sizes()?
            .into_iter()
            .zip(&self.spec.kv_groups)
            .zip(self.kv.geometry())
            .map(|((group_size, group), geom)| GroupSlots {
                group_size,
                page_size: bs,
                pad_block: geom.pad_block(),
                max_pages: max_decode_pages(group.attention, bs, max_blocks),
            })
            .collect())
    }

    /// Allocate the decode graphs' buffers and capture one graph per rung.
    fn capture_decode_graphs(&mut self) -> Result<()> {
        let slots = self.graph_slots()?;
        let mut graphs = DecodeGraphs::new(&self.gpu, decode_ladder(&self.spec.buckets), slots)?;
        let probs = dptr(&self.probs, self.gpu.stream());
        graphs.capture(
            &self.gpu,
            &mut ProgramArgs {
                model: &mut self.model,
                kv: &self.kv,
                probs,
                vocab: self.spec.vocab_size,
                sampleable: self.spec.sampleable_vocab_size,
            },
        )?;
        self.decode = Some(graphs);
        Ok(())
    }

    /// Allocate the mixed-step graphs' buffers and capture one graph per
    /// rung of the mixed ladder.
    fn capture_mixed_graphs(&mut self) -> Result<()> {
        let slots = self.graph_slots()?;
        let seats = self.spec.buckets.last().map_or(0, |b| b.max_seqs);
        let mut graphs =
            MixedGraphs::new(&self.gpu, mixed_ladder(&self.spec.buckets), seats, slots)?;
        let probs = dptr(&self.probs, self.gpu.stream());
        graphs.capture(
            &self.gpu,
            &mut ProgramArgs {
                model: &mut self.model,
                kv: &self.kv,
                probs,
                vocab: self.spec.vocab_size,
                sampleable: self.spec.sampleable_vocab_size,
            },
        )?;
        self.mixed = Some(graphs);
        Ok(())
    }

    /// The captured decode graphs, when graphs are on.
    pub fn decode_graphs(&self) -> Option<&DecodeGraphs> {
        self.decode.as_ref()
    }

    /// The captured mixed-step graphs, when graphs are on without drafting.
    pub fn mixed_graphs(&self) -> Option<&MixedGraphs> {
        self.mixed.as_ref()
    }

    /// How steps that fit a rung (decode or mixed) run from now on. With
    /// graphs off only [`DecodePath::Eager`] is available; a step asking for
    /// another is refused.
    pub fn set_decode_path(&self, path: DecodePath) {
        self.decode_path.set(path);
    }

    pub fn decode_path(&self) -> DecodePath {
        self.decode_path.get()
    }

    /// Steps so far, by how they ran.
    pub fn decode_stats(&self) -> DecodeStats {
        self.stats
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn kv(&self) -> &KvStore {
        &self.kv
    }

    pub fn model(&self) -> &GpuModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut GpuModel {
        &mut self.model
    }

    pub fn config(&self) -> &CudaExecutorConfig {
        &self.cfg
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The logits of every host token of the last step, when
    /// [`CudaExecutor::record_all_logits`] is set.
    pub fn take_all_logits(&mut self) -> Option<Vec<f32>> {
        self.all_logits.take()
    }

    /// Whether every byte of a block is zero (copied back from the device).
    pub fn block_is_zero(&self, group: u32, block: u32) -> Result<bool> {
        Ok(self
            .kv
            .read_block(&self.gpu, group as usize, block)?
            .iter()
            .all(|&x| x == 0))
    }

    fn step(&mut self, step: &StepInput) -> Result<StepOutput> {
        // Scratch is sized for the configured ladder: a step names one of its
        // buckets, exactly, or is refused before anything is staged.
        if !self.spec.buckets.contains(&step.bucket) {
            return Err(CudaError::new(format!(
                "bucket {:?} is not one of the configured {:?}",
                step.bucket, self.spec.buckets
            )));
        }
        if step.seqs.len() > step.bucket.max_seqs as usize
            || step.query_tokens() > step.bucket.max_tokens
        {
            return Err(CudaError::new("batch exceeds its bucket"));
        }
        if self.draft.is_some() {
            return self.draft_step(step);
        }
        // The whole step is validated before any of it takes effect: the
        // maintenance and table updates are staged (checked, and the tables
        // they lead to computed, with nothing applied), the rows are checked
        // against those staged tables, and the forward's input against the
        // model; only then is the staged work committed.
        let staged = self.kv.stage(&step.maintenance, &step.table_updates);
        let tables = staged.mirror();

        // Validate rows; gather token-level inputs, KV targets and plans.
        let groups = self.spec.kv_groups.clone();
        let bs = self.spec.block_size;
        let mut kv_targets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); groups.len()];
        let mut requests: Vec<Vec<AttnRequest>> = vec![Vec::new(); groups.len()];
        let mut logit_rows = Vec::new();
        let mut q_start = 0u32;
        for e in &step.seqs {
            assert!(e.num_tokens >= 1, "a row needs at least one host token");
            let start = e.token_start as usize;
            let end = start + e.num_tokens as usize;
            assert_eq!(
                start, q_start as usize,
                "rows' host tokens must be contiguous in row order"
            );
            for (i, &pos) in step.positions[start..end].iter().enumerate() {
                assert_eq!(
                    pos,
                    e.context_len + u32::try_from(i).expect("i < num_tokens, a u32"),
                    "positions disagree with context_len"
                );
            }
            let p = e.context_len + e.num_tokens - 1;
            let k = if e.sample { e.num_drafts } else { 0 };
            // No drafter row exists for position 0, and the seam forbids asking.
            assert!(
                p > 0 || k == 0,
                "slot {}: {k} drafts requested at position 0",
                e.slot
            );
            assert!(k == 0, "{k} drafts requested; this executor drafts none");
            assert!(p < self.spec.max_model_len, "row past the model length");
            for (g, group) in groups.iter().enumerate() {
                for pos in e.context_len..=p {
                    let block = tables.writable_block(e.slot, g, pos);
                    kv_targets[g].push((block, pos % bs));
                }
                // Visible KV of the row: from the first position its first
                // query sees, whole pages, through p.
                let first = group.attention.first_visible(e.context_len);
                let first_page = first / bs;
                let pages: Vec<u32> = (first_page..=p / bs)
                    .map(|i| tables.block_of(e.slot, g, i * bs))
                    .collect();
                requests[g].push(AttnRequest {
                    q_start,
                    qo_len: e.num_tokens,
                    pages,
                    kv_len: p + 1 - first_page * bs,
                });
            }
            if e.sample && !self.record_all_logits {
                logit_rows.push(q_start + e.num_tokens - 1);
            }
            q_start += e.num_tokens;
        }
        let total = q_start as usize;
        if self.record_all_logits {
            logit_rows = (0..q_start).collect();
        }
        assert_eq!(
            total,
            step.token_ids.len(),
            "token_ids holds every row's host tokens"
        );
        let vocab = self.spec.vocab_size;
        for &t in &step.token_ids {
            if t >= vocab {
                return Err(CudaError::new(format!("token {t} outside vocab {vocab}")));
            }
        }
        let group_sizes = self.group_sizes()?;
        let hosts = group_sizes
            .iter()
            .zip(&requests)
            .map(|(&group_size, r)| HostPlan::new(r, group_size, bs))
            .collect::<Result<Vec<HostPlan>>>()?;
        let path = self.decode_path.get();
        if path != DecodePath::Eager && !self.record_all_logits && !self.model.capture_layers {
            let decode = self
                .decode
                .as_ref()
                .ok_or_else(|| CudaError::new(format!("{path:?} decode needs graphs on")))?;
            if let Some(rung) = decode_rows(step).and_then(|n| rung_for(decode.ladder(), n)) {
                return self.decode_step(step, staged, &kv_targets, &requests, &group_sizes, rung);
            }
            let tiles: Vec<u32> = hosts.iter().map(HostPlan::tile).collect();
            if let Some(rung) = self
                .mixed
                .as_ref()
                .and_then(|m| mixed_rung(m.ladder(), total, &tiles))
            {
                let shapes: Vec<PlanShape> = hosts.iter().map(HostPlan::shape).collect();
                return self.mixed_step(
                    step,
                    staged,
                    &kv_targets,
                    &requests,
                    &shapes,
                    &logit_rows,
                    rung,
                );
            }
        }
        let mut plans: Vec<AttnPlan> = Vec::with_capacity(groups.len());
        for host in hosts {
            // Fresh buffers for this step only: no executor state changes.
            plans.push(self.model.kernels.attention.upload(&self.gpu, host)?);
        }
        self.stats.eager += 1;
        let input = ForwardInput {
            tokens: &step.token_ids,
            positions: &step.positions,
            kv_targets: &kv_targets,
            plans: &plans,
            logit_rows: &logit_rows,
        };
        self.model.check_input(&self.kv, &input)?;

        // 1-2. Maintenance, then table updates.
        self.kv.commit(&self.gpu, staged)?;

        // 3. The forward.
        self.model.forward(&self.gpu, &self.kv, &input)?;

        // 4. Sampling over the sampleable vocabulary.
        let s = self.gpu.stream().clone();
        let mut sample_rows = Vec::new();
        let mut which = Vec::new();
        for (i, e) in step.seqs.iter().enumerate() {
            if e.sample {
                let p = e.context_len + e.num_tokens - 1;
                let logit_row = if self.record_all_logits {
                    e.token_start + e.num_tokens - 1
                } else {
                    narrow(sample_rows.len(), "sample row")?
                };
                sample_rows.push(SampleRow::new(&e.sampling, p + 1, logit_row));
                which.push(i);
            }
        }
        if self.record_all_logits {
            let n = total * vocab as usize;
            self.all_logits = Some(s.clone_dtoh(&self.model.logits().slice(..n))?);
        }
        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; step.seqs.len() * stride as usize],
            stride,
            num_tokens: vec![0; step.seqs.len()],
            logits: step.return_logits.then(Vec::new),
        };
        if !sample_rows.is_empty() {
            let n = sample_rows.len();
            s.memset_zeros(&mut self.status)?;
            self.model.kernels.sampler.sample(
                &self.gpu,
                self.model.logits(),
                vocab as usize,
                self.spec.sampleable_vocab_size,
                &sample_rows,
                &mut self.sample_rows,
                Some(Stream::Sample),
                &mut self.probs,
                &mut self.tokens,
                &mut self.status,
            )?;
            let tokens = s.clone_dtoh(&self.tokens.slice(..n))?;
            let status = s.clone_dtoh(&self.status)?[0];
            if status & STATUS_NON_FINITE != 0 {
                return Err(CudaError::new("non-finite logits"));
            }
            let logits = if step.return_logits {
                let rows: Vec<usize> = sample_rows.iter().map(|r| r.logit_row as usize).collect();
                let all = s.clone_dtoh(
                    &self
                        .model
                        .logits()
                        .slice(..(rows.iter().max().unwrap() + 1) * vocab as usize),
                )?;
                let v = vocab as usize;
                Some(
                    rows.iter()
                        .flat_map(|&r| all[r * v..(r + 1) * v].to_vec())
                        .collect::<Vec<f32>>(),
                )
            } else {
                None
            };
            for (j, &i) in which.iter().enumerate() {
                out.tokens[i * stride as usize] = tokens[j];
                out.num_tokens[i] = 1;
            }
            if let (Some(all), Some(rows)) = (logits, out.logits.as_mut()) {
                let mut j = 0;
                for e in &step.seqs {
                    if e.sample {
                        rows.push(vec![
                            all[j * vocab as usize..(j + 1) * vocab as usize].to_vec(),
                        ]);
                        j += 1;
                    } else {
                        rows.push(Vec::new());
                    }
                }
            }
        } else if let Some(rows) = out.logits.as_mut() {
            rows.resize(step.seqs.len(), Vec::new());
        }
        Ok(out)
    }
}

impl CudaExecutor {
    /// A pure-decode step on rung `rung`: the eager path's checks over the
    /// real rows, then (nothing having changed yet) the padded step table
    /// packed; then maintenance and table updates, one upload, the graph (or
    /// its launches directly), and one read-back.
    fn decode_step(
        &mut self,
        step: &StepInput,
        staged: crate::kv::Staged,
        kv_targets: &[Vec<(u32, u32)>],
        requests: &[Vec<AttnRequest>],
        group_sizes: &[u32],
        rung: usize,
    ) -> Result<StepOutput> {
        let n = step.seqs.len();
        let bs = self.spec.block_size;
        let shapes = requests
            .iter()
            .zip(group_sizes)
            .map(|(r, &gs)| Ok(HostPlan::new(r, gs, bs)?.shape()))
            .collect::<Result<Vec<PlanShape>>>()?;
        let logit_rows: Vec<u32> = (0..narrow(n, "decode rows")?).collect();
        self.model.check_parts(
            &self.kv,
            &InputParts {
                tokens: &step.token_ids,
                positions: &step.positions,
                kv_targets,
                plans: &shapes,
                logit_rows: &logit_rows,
            },
        )?;
        let samples: Vec<SampleRow> = step
            .seqs
            .iter()
            .zip(&logit_rows)
            .map(|(e, &row)| SampleRow::new(&e.sampling, e.context_len + 1, row))
            .collect();
        let decode = self.decode.as_mut().expect("checked by the caller");
        let packed = pack(
            decode.layout(rung),
            decode.slots(),
            &DecodeRows {
                tokens: &step.token_ids,
                positions: &step.positions,
                kv_targets,
                requests,
                samples: &samples,
            },
        )?;

        // 1-2. Maintenance, then table updates.
        self.kv.commit(&self.gpu, staged)?;
        // 3-4. The forward and sampling, over the padded step.
        decode.upload(&self.gpu, &packed)?;
        let direct = self.decode_path.get() == DecodePath::Direct;
        let probs = dptr(&self.probs, self.gpu.stream());
        decode.run(
            &self.gpu,
            rung,
            direct,
            &mut ProgramArgs {
                model: &mut self.model,
                kv: &self.kv,
                probs,
                vocab: self.spec.vocab_size,
                sampleable: self.spec.sampleable_vocab_size,
            },
        )?;
        if direct {
            self.stats.direct += 1;
        } else {
            self.stats.replayed += 1;
        }
        let readback = decode.readback(&self.gpu, rung)?;
        let rows = decode.layout(rung).rows as usize;
        let (tokens, status) = split_readback(&readback, n, rows);
        if status & STATUS_NON_FINITE != 0 {
            return Err(CudaError::new("non-finite logits"));
        }
        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; n * stride as usize],
            stride,
            num_tokens: vec![1; n],
            logits: None,
        };
        for (i, &t) in tokens.iter().enumerate() {
            out.tokens[i * stride as usize] = t;
        }
        if step.return_logits {
            let v = self.spec.vocab_size as usize;
            let all = self
                .gpu
                .stream()
                .clone_dtoh(&self.model.logits().slice(..n * v))?;
            out.logits = Some(all.chunks(v).map(|row| vec![row.to_vec()]).collect());
        }
        Ok(out)
    }
}

impl CudaExecutor {
    /// A mixed step on mixed rung `rung`: the eager path's checks over the
    /// real tokens, then (nothing having changed yet) the padded step table
    /// packed; then maintenance and table updates, one upload, the graph (or
    /// its launches directly), and one read-back. Its output is the eager
    /// path's: one token per sampled row, logits per sampled row when asked.
    #[allow(clippy::too_many_arguments)]
    fn mixed_step(
        &mut self,
        step: &StepInput,
        staged: crate::kv::Staged,
        kv_targets: &[Vec<(u32, u32)>],
        requests: &[Vec<AttnRequest>],
        shapes: &[PlanShape],
        logit_rows: &[u32],
        rung: usize,
    ) -> Result<StepOutput> {
        self.model.check_parts(
            &self.kv,
            &InputParts {
                tokens: &step.token_ids,
                positions: &step.positions,
                kv_targets,
                plans: shapes,
                logit_rows,
            },
        )?;
        let mut samples = Vec::with_capacity(logit_rows.len());
        let mut which = Vec::with_capacity(logit_rows.len());
        for (i, e) in step.seqs.iter().enumerate() {
            if e.sample {
                let row = narrow(samples.len(), "sample row")?;
                samples.push(SampleRow::new(
                    &e.sampling,
                    e.context_len + e.num_tokens,
                    row,
                ));
                which.push(i);
            }
        }
        let mixed = self.mixed.as_mut().expect("checked by the caller");
        let packed = crate::mixed::pack(
            mixed.layout(rung),
            mixed.slots(),
            &MixedRows {
                tokens: &step.token_ids,
                positions: &step.positions,
                kv_targets,
                requests,
                logit_rows,
                samples: &samples,
            },
        )?;

        // 1-2. Maintenance, then table updates.
        self.kv.commit(&self.gpu, staged)?;
        // 3-4. The forward and sampling, over the padded step.
        mixed.upload(&self.gpu, &packed)?;
        let direct = self.decode_path.get() == DecodePath::Direct;
        let probs = dptr(&self.probs, self.gpu.stream());
        mixed.run(
            &self.gpu,
            rung,
            direct,
            &mut ProgramArgs {
                model: &mut self.model,
                kv: &self.kv,
                probs,
                vocab: self.spec.vocab_size,
                sampleable: self.spec.sampleable_vocab_size,
            },
        )?;
        if direct {
            self.stats.mixed_direct += 1;
        } else {
            self.stats.mixed_replayed += 1;
        }
        let readback = mixed.readback(&self.gpu, rung)?;
        let rows = mixed.layout(rung).logits as usize;
        let (tokens, status) = split_readback(&readback, samples.len(), rows);
        if status & STATUS_NON_FINITE != 0 {
            return Err(CudaError::new("non-finite logits"));
        }
        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; step.seqs.len() * stride as usize],
            stride,
            num_tokens: vec![0; step.seqs.len()],
            logits: step.return_logits.then(Vec::new),
        };
        for (&i, &t) in which.iter().zip(tokens) {
            out.tokens[i * stride as usize] = t;
            out.num_tokens[i] = 1;
        }
        if let Some(rows) = out.logits.as_mut() {
            let v = self.spec.vocab_size as usize;
            let all = if which.is_empty() {
                Vec::new()
            } else {
                self.gpu
                    .stream()
                    .clone_dtoh(&self.model.logits().slice(..which.len() * v))?
            };
            let mut j = 0;
            for e in &step.seqs {
                if e.sample {
                    rows.push(vec![all[j * v..(j + 1) * v].to_vec()]);
                    j += 1;
                } else {
                    rows.push(Vec::new());
                }
            }
        }
        Ok(out)
    }
}

/// The planner's constants for this executor.
fn draft_ctx(
    spec: &ModelSpec,
    model: &GpuModel,
    cfg: &CudaExecutorConfig,
) -> Result<crate::draft::PlanCtx> {
    let c = model.config();
    let targets = model.num_groups();
    let group_size = |kv_heads: u32| -> Result<u32> {
        let nq: u32 = narrow(c.layers[0].attention.num_q_heads, "query heads")?;
        Ok(nq / kv_heads)
    };
    let drafter = &spec.kv_groups[targets];
    let AttentionKind::Sliding { window } = drafter.attention else {
        return Err(CudaError::new("the drafter group is a sliding window"));
    };
    let last = *spec.buckets.last().expect("validated: a bucket");
    Ok(crate::draft::PlanCtx {
        depths: cfg.draft_tokens,
        hidden: cfg.mtp_hidden,
        block_size: spec.block_size,
        targets: spec.kv_groups[..targets]
            .iter()
            .map(|g| {
                Ok(crate::draft::TargetGroup {
                    attention: g.attention,
                    group_size: group_size(g.num_kv_heads)?,
                })
            })
            .collect::<Result<_>>()?,
        drafter_window: window,
        drafter_group_size: narrow(c.mtp.attention.num_q_heads, "query heads")
            .map(|nq: u32| nq / drafter.num_kv_heads)?,
        cap_rows: last.max_seqs,
        max_blocks: spec.max_blocks_per_seq(),
        graph: false,
    })
}

impl CudaExecutor {
    /// The drafts of every drafted step since the last call, row by row in
    /// step order, while [`CudaExecutor::record_drafts`] is set (rows without
    /// drafts included, with none). Takes `&self` so a caller that only
    /// borrows the executor (through the engine) can drain them.
    pub fn take_drafts(&self) -> Vec<DraftRecord> {
        std::mem::take(&mut *self.drafts.borrow_mut())
    }

    /// The captured drafted-step graphs, when drafting with graphs on.
    pub fn draft_graphs(&self) -> Option<&crate::draft::DraftGraphs> {
        self.draft.as_ref().and_then(|d| d.graphs.as_ref())
    }

    /// Capture one graph per rung of the drafted ladder.
    fn capture_draft_graphs(&mut self) -> Result<()> {
        let d = self.draft.as_mut().expect("drafting");
        let drafter = self.model.num_groups();
        let graphs = crate::draft::DraftGraphs::capture(
            &self.gpu,
            crate::draft::draft_ladders(&self.spec.buckets, d.ctx.depths),
            &d.ctx,
            &mut crate::draft::RunTarget {
                model: &mut self.model,
                kv: &self.kv,
                bufs: &mut d.bufs,
                vocab: self.spec.vocab_size,
                sampleable: self.spec.sampleable_vocab_size,
                drafter,
            },
        )?;
        d.graphs = Some(graphs);
        Ok(())
    }

    /// A step with drafting configured: every row's drafter rows, and
    /// drafting, verification and acceptance for the rows the host gave
    /// drafts (`crate::draft`). Checked whole (rows, tables, the plan and its
    /// work lists) before anything takes effect.
    fn draft_step(&mut self, step: &StepInput) -> Result<StepOutput> {
        use crate::draft::{Load, Row, RowOut};
        let staged = self.kv.stage(&step.maintenance, &step.table_updates);
        let d = self.draft.as_ref().expect("drafting");
        let depths = d.ctx.depths;
        let bs = self.spec.block_size;
        let drafter = self.model.num_groups();
        // The slots' state as it will be after this step's resets.
        let mut slots = d.slots.clone();
        for m in &step.maintenance {
            if let eidola_engine::executor::Maintenance::ResetSlot { slot } = *m {
                slots[slot as usize] = None;
            }
        }
        let mirror = staged.mirror();
        let mut rows = Vec::with_capacity(step.seqs.len());
        let mut q_start = 0u32;
        for e in &step.seqs {
            assert!(e.num_tokens >= 1, "a row needs at least one host token");
            let start = e.token_start as usize;
            let end = start + e.num_tokens as usize;
            assert_eq!(
                start, q_start as usize,
                "rows' host tokens must be contiguous in row order"
            );
            for (i, &pos) in step.positions[start..end].iter().enumerate() {
                assert_eq!(
                    pos,
                    e.context_len + u32::try_from(i).expect("i < num_tokens, a u32"),
                    "positions disagree with context_len"
                );
            }
            let p = e.context_len + e.num_tokens - 1;
            let k = if e.sample { e.num_drafts } else { 0 };
            assert!(
                p > 0 || k == 0,
                "slot {}: {k} drafts requested at position 0",
                e.slot
            );
            assert!(
                k <= depths,
                "{k} drafts requested; the drafter has {depths} depths"
            );
            assert!(p + k < self.spec.max_model_len, "row past the model length");
            let c = e.context_len;
            let load = if c == 0 {
                Load::None
            } else {
                match slots.get(e.slot as usize).copied().flatten() {
                    Some((base, at)) if at == c - 1 => Load::State { entry: at - base },
                    _ => {
                        assert!(
                            c.is_multiple_of(bs),
                            "slot {}: no drafter state at position {} (not a block boundary \
                             of a cached prefix, and not this slot's last position)",
                            e.slot,
                            c - 1
                        );
                        Load::Tap {
                            block: mirror.block_of(e.slot, drafter, c - 1),
                        }
                    }
                }
            };
            rows.push(Row {
                slot: Some(e.slot),
                c,
                tokens: step.token_ids[start..end].to_vec(),
                k,
                sample: e.sample,
                sampling: e.sampling,
                load,
            });
            q_start += e.num_tokens;
        }
        assert_eq!(
            q_start as usize,
            step.token_ids.len(),
            "token_ids holds every row's host tokens"
        );
        let vocab = self.spec.vocab_size;
        if let Some(&t) = step.token_ids.iter().find(|&&t| t >= vocab) {
            return Err(CudaError::new(format!("token {t} outside vocab {vocab}")));
        }
        if self.record_all_logits || self.model.capture_layers {
            return Err(CudaError::new(
                "per-token logits and per-layer copies are not recorded while drafting",
            ));
        }
        // A uniform drafted decode step that fits a rung replays (or runs
        // directly) padded to it.
        let path = self.decode_path.get();
        // Every row one host token, sampled, at one draft width (any of `0
        // ..= depths`: the serving core narrows it), past the first `depths`
        // positions so every depth has its rows.
        let width = rows.first().map_or(0, |r| r.k);
        let uniform = !rows.is_empty()
            && rows
                .iter()
                .all(|r| r.tokens.len() == 1 && r.sample && r.k == width && r.c >= depths);
        let rung = match (path, d.graphs.as_ref()) {
            (DecodePath::Eager, _) => None,
            (_, None) => {
                return Err(CudaError::new(format!("{path:?} decode needs graphs on")));
            }
            (_, Some(g)) if uniform => g.rung_for(width, rows.len()),
            _ => None,
        };
        let real = rows.len();
        let mut ctx = d.ctx.clone();
        if let Some(r) = rung {
            let rung_rows = d.graphs.as_ref().expect("a rung").rows(r);
            rows.extend((real..rung_rows).map(|_| crate::draft::pad_row(depths, width)));
            ctx.graph = true;
        }
        let kvmap = crate::draft::StepKv {
            mirror,
            geometry: self.kv.geometry(),
            drafter,
        };
        let plan = crate::draft::plan(&rows, &ctx, &kvmap);
        let built = crate::draft::build(&plan, &ctx)?;
        // Every forward input the plan holds is checked as the eager path
        // checks its own (tokens, positions, KV targets and pages inside
        // their pools, logit rows inside each forward).
        crate::draft::check_plan(&plan, &self.model, &self.kv, drafter)?;
        let layout;
        let launches;
        let mut fresh_table = None;
        match rung {
            Some(r) => {
                let g = self
                    .draft
                    .as_ref()
                    .expect("drafting")
                    .graphs
                    .as_ref()
                    .expect("a rung");
                layout = g.layout(r).clone();
                launches = crate::draft::launches(&built, &layout);
                if launches != g.launches(r) {
                    return Err(CudaError::new(format!(
                        "a uniform drafted step of {real} rows does not match rung {}'s capture",
                        g.rows(r)
                    )));
                }
            }
            None => {
                layout = crate::draft::Layout::new(&built, &ctx, false)?;
                launches = crate::draft::launches(&built, &layout);
            }
        }
        let words = crate::draft::pack(&built, &layout, &ctx)?;

        // 1-2. Maintenance, then table updates.
        self.kv.commit(&self.gpu, staged)?;
        // 3-4. The step's launches, then its one read-back.
        let s = self.gpu.stream().clone();
        let d = self.draft.as_mut().expect("drafting");
        let table = match rung {
            Some(_) => {
                let g = d.graphs.as_mut().expect("a rung");
                g.upload(&self.gpu, &words)?;
                g.table_ptr(&s)
            }
            None => {
                let t = s.clone_htod(&words)?;
                let ptr = crate::launch::dptr(&t, &s);
                fresh_table = Some(t);
                ptr
            }
        };
        let table_words = match (&fresh_table, d.graphs.as_ref(), rung) {
            (Some(t), _, _) => t.len(),
            (None, Some(g), Some(_)) => g.table_words(),
            _ => unreachable!("a table was chosen above"),
        };
        let mut target = crate::draft::RunTarget {
            model: &mut self.model,
            kv: &self.kv,
            bufs: &mut d.bufs,
            vocab,
            sampleable: self.spec.sampleable_vocab_size,
            drafter,
        };
        match (rung, path) {
            (Some(r), DecodePath::Replay) => {
                d.graphs.as_ref().expect("a rung").replay(r)?;
                self.stats.replayed += 1;
            }
            _ => {
                // SAFETY: the table holds the step packed for `layout`, from
                // a plan over these buffers, pools and model.
                unsafe {
                    crate::draft::run(
                        &self.gpu,
                        &launches,
                        &layout,
                        &mut target.args(&ctx, table, table_words),
                    )?;
                }
                if rung.is_some() {
                    self.stats.direct += 1;
                } else {
                    self.stats.eager += 1;
                }
            }
        }
        let rb_words = s.clone_dtoh(&d.bufs.readback)?;
        let rb = crate::draft::split(&rb_words, ctx.cap_rows, depths);
        if rb.status & STATUS_NON_FINITE != 0 {
            return Err(CudaError::new("non-finite logits"));
        }
        if rb.status & crate::sampler::STATUS_BAD_TOKEN != 0 {
            return Err(CudaError::new("a draft outside the sampleable vocabulary"));
        }
        if rb.status & crate::engine_ops::STATUS_BAD_INDEX != 0 {
            return Err(CudaError::new("a drafted step's copy index out of range"));
        }
        if self.record_drafts {
            let lanes = match &fresh_table {
                Some(t) => s.clone_dtoh(&t.slice(layout.lanes..layout.kv_chunk))?,
                None => d
                    .graphs
                    .as_ref()
                    .expect("a rung")
                    .read_words(&self.gpu, layout.lanes..layout.kv_chunk)?,
            };
            self.drafts
                .borrow_mut()
                .extend(plan.out[..real].iter().zip(&rows).map(|(o, row)| {
                    DraftRecord {
                        slot: row.slot.expect("a real row"),
                        context_len: row.c,
                        drafts: match *o {
                            RowOut::Drafted(rd) => (0..row.k)
                                .map(|i| lanes[layout.lane(&ctx, i, rd) - layout.lanes])
                                .collect(),
                            _ => Vec::new(),
                        },
                    }
                }));
        }
        let stride = depths + 1;
        let mut out = StepOutput {
            tokens: vec![0; real * stride as usize],
            stride,
            num_tokens: vec![0; real],
            logits: step.return_logits.then(Vec::new),
        };
        let mut logit_rows: Vec<Vec<usize>> = Vec::with_capacity(real);
        for (i, row) in rows[..real].iter().enumerate() {
            let produced: Vec<u32> = match plan.out[i] {
                RowOut::Drafted(rd) => {
                    let n = rb.counts[rd as usize];
                    if n == 0 || n > row.k + 1 {
                        return Err(CudaError::new(format!(
                            "row {i}: acceptance produced {n} tokens for {} drafts",
                            row.k
                        )));
                    }
                    let at = rd as usize * stride as usize;
                    rb.out[at..at + n as usize].to_vec()
                }
                RowOut::Plain(j) => vec![rb.plain[j as usize]],
                RowOut::None => Vec::new(),
            };
            let first = plan.logits[i];
            logit_rows.push(
                (0..produced.len())
                    .map(|j| first.expect("a sampled row has logits") as usize + j)
                    .collect(),
            );
            let n = u32::try_from(produced.len()).expect("at most k + 1 tokens");
            out.tokens[i * stride as usize..][..produced.len()].copy_from_slice(&produced);
            out.num_tokens[i] = n;
            // The slot continues from its last valid position; its state
            // holds the levels at p ..= p + k.
            let p = row.p();
            d.slots[row.slot.expect("a real row") as usize] = Some((p, p + n.max(1) - 1));
        }
        for m in &step.maintenance {
            if let eidola_engine::executor::Maintenance::ResetSlot { slot } = *m
                && !step.seqs.iter().any(|e| e.slot == slot)
            {
                d.slots[slot as usize] = None;
            }
        }
        if let Some(l) = out.logits.as_mut() {
            let v = vocab as usize;
            let max = logit_rows.iter().flatten().max().map_or(0, |&m| m + 1);
            let all = if max > 0 {
                s.clone_dtoh(&self.model.logits().slice(..max * v))?
            } else {
                Vec::new()
            };
            for rows in &logit_rows {
                l.push(
                    rows.iter()
                        .map(|&r| all[r * v..(r + 1) * v].to_vec())
                        .collect(),
                );
            }
        }
        drop(fresh_table);
        Ok(out)
    }
}

impl Executor for CudaExecutor {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> std::result::Result<StepOutput, ExecutorError> {
        self.steps += 1;
        self.step(step).map_err(|e| ExecutorError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(keep: &[usize]) -> Vec<AttentionSpec> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eidola-engine-model/tests/data/flash-mopd.config.json"
        );
        let c = eidola_engine_model::ModelConfig::from_file(std::path::Path::new(path))
            .unwrap()
            .truncated(keep)
            .unwrap();
        group_layers(&c).0
    }

    /// Capacities follow their group's kind, not the layer selection's order,
    /// and a capacity for a group no retained layer uses is refused.
    #[test]
    fn block_counts_are_keyed_by_kind() {
        let b = KvBlocks {
            global: 10,
            sliding: 20,
            drafter: 0,
        };
        assert_eq!(b.resolve(&keys(&[0, 1]), false).unwrap(), vec![10, 20]);
        assert_eq!(b.resolve(&keys(&[1, 5]), false).unwrap(), vec![10, 20]);
        let e = b.resolve(&keys(&[1, 2]), false).unwrap_err();
        assert_eq!(e.field, "num_blocks.global");
        let e = b.resolve(&keys(&[0]), false).unwrap_err();
        assert_eq!(e.field, "num_blocks.sliding");
        let sliding = KvBlocks { global: 0, ..b };
        assert_eq!(sliding.resolve(&keys(&[1, 2]), false).unwrap(), vec![20]);
        // The drafter group has blocks exactly when drafting.
        let drafter = KvBlocks { drafter: 30, ..b };
        assert_eq!(drafter.resolve(&keys(&[0, 1]), true).unwrap(), vec![10, 20]);
        let e = drafter.resolve(&keys(&[0, 1]), false).unwrap_err();
        assert_eq!(e.field, "num_blocks.drafter");
    }
}
