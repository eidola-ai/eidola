//! The model architectures the engine knows, and the numbers each one fixes.
//!
//! A model in the catalog (`models/` at the workspace root) names one
//! [`Architecture`], and an engine deployment's `deployment.json` names the
//! architecture of the weights it serves. Everything about a deployment that
//! depends on the model's shape (KV bytes per position, the sliding window,
//! the drafter's layers, the executor's per-token scratch) is read through
//! [`Architecture::devices`], never from a free constant, so adding a model
//! family is adding a variant and the [`Shape`] of each device it runs on.
//!
//! An architecture runs on one GPU or is split across several, each device
//! holding its own layers, KV pools and weights: [`Architecture::devices`]
//! lists one [`Shape`] per device, in device order.
//!
//! An architecture is **implemented** when the engine can serve it: its
//! shapes are known here and the engine crates load and run it. A variant
//! that is not implemented is a deliberate stub: [`Architecture::shape`]
//! panics for it, and the gateway's build refuses any catalog row or engine
//! deployment that names it, so a stub can be written down before its engine
//! work lands without anything being able to serve it.
//!
//! Adding a model family:
//!
//! 1. add the variant, with its [`name`](Architecture::name) (the spelling
//!    the catalog and `deployment.json` use) and a stub `shape` arm;
//! 2. implement it in the engine crates (`eidola-engine-model`'s config and
//!    loader, the CUDA executor's support check and kernels);
//! 3. replace the stub with its devices' [`Shape`]s, derived from the
//!    engine's own layouts, and return `true` from [`is_implemented`]
//!    (Architecture::is_implemented).

/// A model architecture, as the catalog and deployments name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Architecture {
    /// MiMo-V2.6-Flash: 9 global and 39 sliding attention layers, three MTP
    /// layers, the architecture the engine serves.
    MimoV26Flash,
    /// MiMo-V2.6-Pro. **Not implemented**: a stub, so the pattern for adding
    /// a model family is in the tree. Its shapes, loader and kernels do not
    /// exist yet, and nothing may name it in a catalog row or a deployment.
    MimoV26Pro,
    /// DeepSeek-V4.1-Flash, text only: 40 layers split across two GPUs, the
    /// first 20 with the embedding on device 0 and the rest with the head and
    /// the drafter on device 1.
    DeepSeekV41Flash,
    /// Qwen3.8-27B, text only: a dense model of 48 Gated DeltaNet and 16
    /// gated full-attention layers, with one MTP layer, on one GPU.
    Qwen38Dense27B,
}

impl Architecture {
    /// Every architecture, implemented or not.
    pub const ALL: [Architecture; 4] = [
        Architecture::MimoV26Flash,
        Architecture::MimoV26Pro,
        Architecture::DeepSeekV41Flash,
        Architecture::Qwen38Dense27B,
    ];

    /// The spelling the catalog and `deployment.json` use.
    pub const fn name(self) -> &'static str {
        match self {
            Architecture::MimoV26Flash => "MimoV26Flash",
            Architecture::MimoV26Pro => "MimoV26Pro",
            Architecture::DeepSeekV41Flash => "DeepSeekV41Flash",
            Architecture::Qwen38Dense27B => "Qwen38Dense27B",
        }
    }

    /// The architecture spelled `name`, if there is one.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }

    /// Whether the engine can serve this architecture: its [`Shape`] is
    /// known and the engine crates run it.
    pub const fn is_implemented(self) -> bool {
        match self {
            Architecture::MimoV26Flash => true,
            Architecture::MimoV26Pro => false,
            Architecture::DeepSeekV41Flash => true,
            Architecture::Qwen38Dense27B => true,
        }
    }

    /// The numbers this architecture fixes on each device it runs on, in
    /// device order: one [`Shape`] for a model on one GPU, one per GPU for a
    /// model split across several.
    ///
    /// # Panics
    ///
    /// For an architecture that is not [implemented](Self::is_implemented).
    /// Every reader of a configuration checks that first and refuses, so
    /// reaching the panic is a missing check.
    pub fn devices(self) -> &'static [Shape] {
        match self {
            Architecture::MimoV26Flash => core::slice::from_ref(&MIMO_V26_FLASH),
            Architecture::MimoV26Pro => unimplemented!(
                "MiMo-V2.6-Pro: its KV layout, drafter, executor scratch and kernels are not \
                 implemented; nothing may name this architecture yet"
            ),
            Architecture::DeepSeekV41Flash => &DEEPSEEK_V41_FLASH,
            Architecture::Qwen38Dense27B => core::slice::from_ref(&QWEN38_DENSE_27B),
        }
    }

    /// The numbers a one-device architecture fixes: its only device's
    /// [`Shape`].
    ///
    /// # Panics
    ///
    /// For an architecture that is not [implemented](Self::is_implemented), or
    /// one split across several devices (read [`Self::devices`] for those).
    pub fn shape(self) -> &'static Shape {
        match self.devices() {
            [shape] => shape,
            _ => panic!("{self} runs on several devices; read `Architecture::devices`"),
        }
    }

    /// The widest draft the CUDA executor can run for this architecture: the
    /// most [`Shape::mtp_layers`] any of its devices holds.
    pub fn mtp_layers(self) -> u32 {
        self.devices()
            .iter()
            .map(|d| d.mtp_layers)
            .max()
            .unwrap_or(0)
    }
}

impl core::fmt::Display for Architecture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// The numbers an architecture fixes for what a deployment allocates on one
/// device (`engine_deployment::check_resources`,
/// `engine_deployment::check_device_resources` and the bounds they are built
/// from).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// KV-cache bytes per position of the global attention layers.
    pub global_kv_bytes_per_position: u64,
    /// KV-cache bytes per position of the sliding attention layers.
    pub sliding_kv_bytes_per_position: u64,
    /// Positions a sliding layer sees.
    pub sliding_window: u64,
    /// The widest draft the CUDA executor can run: the MTP layers the
    /// checkpoint ships where each serves one draft depth (MiMo-V2.6's), or
    /// the grammar's cuda cap for a drafter that runs any depth
    /// (DeepSeek-V4.1's slots, Qwen3.8's one layer run on its own output).
    pub mtp_layers: u32,
    /// Drafter KV bytes per position for each draft depth (one MTP layer).
    pub drafter_kv_bytes_per_position_per_depth: u64,
    /// Drafter boundary-tap bytes per block for each draft depth: one chain
    /// level of the hidden size, in f32.
    pub drafter_tap_bytes_per_depth: u64,
    /// Drafter KV bytes per position whatever the draft width, for a drafter
    /// whose every block runs at any width (0 where the drafter's KV grows
    /// with the depth instead).
    pub drafter_kv_bytes_per_position: u64,
    /// Bytes of one row of the hidden size in f32 (the drafter's per-seat
    /// state rows).
    pub hidden_row_bytes: u64,
    /// Device scratch per token a step may hold (`MAX_BATCHED_TOKENS`).
    pub step_token_device_bytes: u64,
    /// The expert scratch held whatever the step.
    pub expert_floor_device_bytes: u64,
    /// Device memory per sampler row.
    pub sampler_row_device_bytes: u64,
    /// Device bytes of one split-attention partial row.
    pub split_partial_row_bytes: u64,
    /// RoPE table bytes per position (`MAX_MODEL_LEN`).
    pub rope_device_bytes_per_position: u64,
    /// The weights this device holds, bounded from above, for an
    /// architecture split across devices; `None` for one on a single device,
    /// whose weights are bounded by the whole pack
    /// (`engine_deployment::artifact_data_bytes`).
    pub weights_bytes: Option<u64>,
    /// For an executor that gives every bank group of every device one block
    /// count and every window group another (DeepSeek-V4.1's), this device's
    /// groups, in the executor's block layout; empty for one that sizes each
    /// group's pool by its own rule (MiMo-V2.6's), whose bounds read the
    /// per-position fields above.
    pub uniform_kv_groups: &'static [UniformKvGroup],
    /// Of [`Shape::weights_bytes`], the drafter's, which an executor that does
    /// not draft never loads (bounded from below, so the weights without it
    /// stay bounded from above); 0 where the drafter is elsewhere or none.
    pub drafter_weights_bytes: u64,
    /// For an executor whose step scratch this crate counts allocation by
    /// allocation (DeepSeek-V4.1's), that count; `None` for one whose reserve
    /// reads the per-token fields above (MiMo-V2.6's).
    pub step_scratch: Option<StepScratch>,
    /// For a hybrid model whose linear-attention layers keep a recurrent state
    /// (Qwen3.8's), its dimensions, from which this crate counts the state
    /// pool, each seat's state and the executor's step scratch
    /// ([`HybridDims`]); `None` for a model with no recurrent state.
    pub recurrent: Option<HybridDims>,
}

/// f32 values a row of a recurrent model's split FP8 GEMMs' partial sums takes at
/// most (`eidola-engine-cuda`'s `qwen::kernels::split_width`): a GEMM splits K into
/// `s` ranges only while its `s` ranges' 128-output tiles fit 148 CTAs, so `s · n`
/// is at most 148 · 128.
pub const SPLIT_PARTIAL_FLOATS: u64 = 148 * 128;

/// The dimensions of a dense hybrid of Gated DeltaNet and gated
/// full-attention layers with one MTP layer (Qwen3.8's), as the CUDA
/// executor (`eidola-engine-cuda`'s `QwenCudaExecutor`) lays its memory out
/// from them. Every byte count below is that executor's own, which
/// `eidola-server-engine`'s `cuda::tests` hold to its allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridDims {
    pub hidden: u64,
    pub intermediate: u64,
    /// Rows of the embedding and the head.
    pub vocab: u64,
    pub attention_layers: u64,
    pub deltanet_layers: u64,
    pub q_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    /// Dimensions of each head that carry the rotary position.
    pub rotary: u64,
    pub k_heads: u64,
    pub v_heads: u64,
    pub k_dim: u64,
    pub v_dim: u64,
}

impl HybridDims {
    const fn key_dim(&self) -> u64 {
        self.k_heads * self.k_dim
    }

    const fn value_dim(&self) -> u64 {
        self.v_heads * self.v_dim
    }

    /// The DeltaNet's convolved width: queries, keys and values.
    const fn conv_dim(&self) -> u64 {
        2 * self.key_dim() + self.value_dim()
    }

    const fn q_width(&self) -> u64 {
        self.q_heads * self.head_dim
    }

    const fn kv_width(&self) -> u64 {
        self.kv_heads * self.head_dim
    }

    /// Target KV bytes per position: BF16 K and V of every attention layer.
    pub const fn kv_bytes_per_position(&self) -> u64 {
        self.attention_layers * 2 * self.kv_width() * 2
    }

    /// The drafter's KV bytes per position: its one attention layer.
    pub const fn drafter_kv_bytes_per_position(&self) -> u64 {
        2 * self.kv_width() * 2
    }

    /// A drafter block's boundary tap: one BF16 row of the hidden size.
    pub const fn tap_bytes(&self) -> u64 {
        self.hidden * 2
    }

    /// Device bytes of one state buffer: every DeltaNet layer's convolution
    /// history (three inputs of the convolved width) and state matrix, f32,
    /// then a 4-byte tag, padded to a whole number of convolution rows
    /// (`4 × conv_dim` bytes) at a multiple of 256, as the executor strides
    /// its images. A seat's live image is one more of the same.
    pub const fn state_image_bytes(&self) -> u64 {
        let data =
            self.deltanet_layers * (3 * self.conv_dim() + self.v_heads * self.k_dim * self.v_dim);
        let row = self.conv_dim() * 4;
        let stride = lcm(256, row);
        (data * 4 + 4).div_ceil(stride) * stride
    }

    /// Device bytes one position of a seat's tail takes: every DeltaNet
    /// layer's convolved inputs, decay and write strength, f32.
    pub const fn tail_bytes_per_position(&self) -> u64 {
        self.deltanet_layers * (self.conv_dim() + 2 * self.v_heads) * 4
    }

    /// One kept normed state of a seat: a BF16 row of the hidden size.
    pub const fn kept_row_bytes(&self) -> u64 {
        self.hidden * 2
    }

    /// RoPE table bytes per position: f32 cosines and sines of the rotary
    /// dimensions.
    pub const fn rope_bytes_per_position(&self) -> u64 {
        2 * self.rotary * 4
    }

    /// The widest input of an FP8 linear: the quantiser's scratch width.
    pub const fn widest_input(&self) -> u64 {
        let mut k = self.hidden;
        if self.intermediate > k {
            k = self.intermediate;
        }
        if self.q_width() > k {
            k = self.q_width();
        }
        if self.value_dim() > k {
            k = self.value_dim();
        }
        k
    }

    /// The step program's scratch per row of its widest forward (the
    /// executor's `program::alloc_scratch`): the residual stream and its
    /// three working rows; the quantised input and its scales; the split FP8
    /// GEMMs' partial sums ([`SPLIT_PARTIAL_FLOATS`], a bound); the MLP's
    /// gate, up and gated rows; the DeltaNet's projections, gated norm,
    /// widened inputs, decay and write rows and output rows; the attention's
    /// gated query, K, V, normed query and key, output and gated output; and
    /// the drafter's embedding, hidden, concatenation, residual and normed
    /// rows.
    pub const fn program_row_bytes(&self) -> u64 {
        let (h, cd, vd, vh) = (self.hidden, self.conv_dim(), self.value_dim(), self.v_heads);
        let (qw, kvw) = (self.q_width(), self.kv_width());
        let k = self.widest_input();
        4 * h * 2
            + k
            + k / 128 * 4
            + SPLIT_PARTIAL_FLOATS * 4
            + 3 * self.intermediate * 2
            + cd * 2
            + vd * 2
            + 2 * vh * 2
            + vd * 2
            + cd * 4
            + 2 * vh * 4
            + vd * 4
            + 2 * qw * 2
            + 3 * kvw * 2
            + 3 * qw * 2
            + 4 * h * 2
            + 2 * h * 2
    }

    /// The DeltaNet's staging per staged position (a row's new positions and
    /// the kept ones it folds first): decay and write inputs, queries, keys,
    /// values, both gates and the output, f32.
    pub const fn staged_position_bytes(&self) -> u64 {
        let v = self.value_dim();
        2 * self.v_heads * 4 + 2 * self.key_dim() * 4 + v * 4 + 2 * self.v_heads * 4 + v * 4
    }

    /// One convolution input row of the staging, f32.
    pub const fn conv_row_bytes(&self) -> u64 {
        self.conv_dim() * 4
    }

    /// One FA2 partial row: every query head's BF16 output and f32
    /// log-sum-exp.
    pub const fn partial_row_bytes(&self) -> u64 {
        self.q_heads * (self.head_dim * 2 + 4)
    }

    /// The rows of the stacked DeltaNet `a` and `b` weight: `2 × v_heads`,
    /// to a multiple of 8.
    pub const fn ab_rows(&self) -> u64 {
        (2 * self.v_heads).next_multiple_of(8)
    }

    /// The decay and write rows' bytes, f32.
    pub const fn ab_row_bytes(&self) -> u64 {
        2 * self.v_heads * 4
    }

    pub const fn value_row_bytes(&self) -> u64 {
        self.value_dim() * 4
    }
}

/// The least common multiple of two positive numbers.
const fn lcm(a: u64, b: u64) -> u64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    a / x * b
}

/// A device's step scratch, as an executor sizes it beside its weights and KV
/// (`eidola-engine-cuda`'s `deepseek::model::Scratch` and `DraftScratch`, the
/// pooled sources' open entries): `engine_deployment::cuda_device_reserve_bytes`
/// reads it for a shape that has it. The byte counts are the executor's own,
/// held to its allocations by `eidola-server-engine`'s `cuda::tests`.
///
/// A step's **rows** are `MAX_BATCHED_TOKENS`, or with drafting
/// `MAX_SEQS × draft_slots` when that is more (a draft pass computes every
/// slot of every drafting row). The scratch is then:
/// - per row, `row_bytes`, with drafting `draft_row_bytes` more, and the
///   indexer's scores and candidate mask over every visible entry of a
///   `MAX_MODEL_LEN` sequence: `2 × (MAX_MODEL_LEN / index_ratio + 1)` bytes
///   of scores and a mask word per 32 candidate blocks of `candidate_entries`
///   entries (`engine_deployment::index_scratch_row_bytes`), which is what
///   grows with step tokens × context;
/// - per seat, `seat_bytes`, `per_draft_seat_bytes` per drafted token and,
///   with drafting, `drafting_seat_bytes`;
/// - `fixed_bytes`, which bounds the few allocations of no row or seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepScratch {
    pub row_bytes: u64,
    pub draft_row_bytes: u64,
    /// Positions per index-score entry: the pooling ratio of the bank this
    /// device's indexer scores.
    pub index_ratio: u64,
    /// Entries per candidate block the mask has a bit for.
    pub candidate_entries: u64,
    pub seat_bytes: u64,
    pub per_draft_seat_bytes: u64,
    pub drafting_seat_bytes: u64,
    /// The drafter's slots, every one computed for every drafting row.
    pub draft_slots: u64,
    pub fixed_bytes: u64,
}

/// One KV group of an executor whose bank groups all hold one number of
/// blocks and whose window groups all hold another
/// (`Shape::uniform_kv_groups`). A block of it is `layers` layers of
/// `KV_BLOCK_SIZE / ratio` entries of `entry_bytes` each, padded to 16 bytes
/// (`eidola-engine-cuda`'s `deepseek::kv::GroupGeometry::block_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformKvGroup {
    pub layers: u64,
    /// Positions per entry (the pooling ratio; 1 for a group of one entry
    /// per position).
    pub ratio: u64,
    /// Bytes of one entry of one layer.
    pub entry_bytes: u64,
    /// A sliding-window group, whose pool holds running sequences' live
    /// windows and the windows retained at hit points
    /// (`engine_deployment::uniform_kv_window_blocks`); otherwise a bank
    /// (full attention), whose pool holds whole sequences.
    pub window: bool,
    /// The drafter's group: present only when drafting (`DRAFT_TOKENS` > 0),
    /// and the same size at every draft width.
    pub drafter: bool,
}

impl Shape {
    /// KV-cache bytes per position, every layer.
    pub const fn kv_bytes_per_position(&self) -> u64 {
        self.global_kv_bytes_per_position + self.sliding_kv_bytes_per_position
    }
}

/// MiMo-V2.6-Flash's numbers (`eidola-engine-cuda` `support`, `kv`,
/// `model::ScratchSizes`, `attention`, the sampler).
pub const MIMO_V26_FLASH: Shape = Shape {
    // 9 global layers of 4 KV heads, each head 192 K + 128 V dims, bf16.
    global_kv_bytes_per_position: 9 * 4 * (192 + 128) * 2,
    // 39 sliding layers of 8 KV heads, each head 192 K + 128 V dims, bf16.
    sliding_kv_bytes_per_position: 39 * 8 * (192 + 128) * 2,
    // `support::WINDOW`.
    sliding_window: 128,
    // `num_nextn_predict_layers`, `support::MTP_LAYERS`.
    mtp_layers: 3,
    // One MTP layer: 8 KV heads of 192 K + 128 V dims, bf16
    // (`kv::GroupGeometry`, planar layout).
    drafter_kv_bytes_per_position_per_depth: 8 * (192 + 128) * 2,
    // One chain level of the hidden size (4,096) in f32.
    drafter_tap_bytes_per_depth: 4096 * 4,
    drafter_kv_bytes_per_position: 0,
    hidden_row_bytes: 4096 * 4,
    // The f32 logits row (152,576 × 4 bytes), the residual stream and its
    // projections, the quantized activations, Q, K, V and attention output,
    // the dense FFN, and eight expert rows of 22,580 bytes each, about
    // 1.03 MB in all, rounded up to 1.25 MiB.
    step_token_device_bytes: 5 << 18,
    // The expert layout's padding (up to 127 rows for each of the 256
    // experts, and a block's rounding), bounded by 65,536 rows of at most
    // 24 KiB.
    expert_floor_device_bytes: 65_536 * (24 << 10),
    // The f64 distribution over at most the head's 152,576 ids (1,220,608
    // bytes) and the row's buffers, rounded up to 1.25 MiB.
    sampler_row_device_bytes: 5 << 18,
    // 64 query heads, each a BF16 output of 128 and an f32 log-sum-exp
    // (`attention::PARTIAL_BYTES_PER_HEAD`).
    split_partial_row_bytes: 64 * (128 * 2 + 4),
    // 64 f32 for each distinct θ, at most three.
    rope_device_bytes_per_position: 3 * 64 * 4,
    weights_bytes: None,
    uniform_kv_groups: &[],
    drafter_weights_bytes: 0,
    step_scratch: None,
    recurrent: None,
};

/// Qwen3.8-27B's dimensions (the released `config.json`'s `text_config`):
/// 64 layers, every fourth gated full attention.
pub const QWEN38_DIMS: HybridDims = HybridDims {
    hidden: 5120,
    intermediate: 17_408,
    vocab: 248_320,
    attention_layers: 16,
    deltanet_layers: 48,
    q_heads: 24,
    kv_heads: 4,
    head_dim: 256,
    // `partial_rotary_factor` 0.25 of the 256.
    rotary: 64,
    k_heads: 16,
    v_heads: 48,
    k_dim: 128,
    v_dim: 128,
};

/// Qwen3.8-27B's numbers on its one GPU. Its KV is full attention only: the
/// target's 16 layers (64 KiB a position) and, drafting, the drafter's one
/// layer and a boundary tap per block. Its pools, seats and step scratch are
/// counted from [`QWEN38_DIMS`] (`engine_deployment`'s recurrent rules), not
/// from the per-token fields, which are 0.
pub const QWEN38_DENSE_27B: Shape = Shape {
    global_kv_bytes_per_position: QWEN38_DIMS.kv_bytes_per_position(),
    sliding_kv_bytes_per_position: 0,
    sliding_window: 0,
    // One MTP layer, run again on its own output for each further depth: the
    // executor drafts at any width, and the grammar holds a CUDA draft to
    // `caps::CUDA_DRAFT_TOKENS`.
    mtp_layers: 3,
    drafter_kv_bytes_per_position_per_depth: 0,
    drafter_tap_bytes_per_depth: 0,
    drafter_kv_bytes_per_position: QWEN38_DIMS.drafter_kv_bytes_per_position(),
    hidden_row_bytes: 0,
    step_token_device_bytes: 0,
    expert_floor_device_bytes: 0,
    sampler_row_device_bytes: 0,
    split_partial_row_bytes: 0,
    rope_device_bytes_per_position: QWEN38_DIMS.rope_bytes_per_position(),
    weights_bytes: None,
    uniform_kv_groups: &[],
    drafter_weights_bytes: 0,
    step_scratch: None,
    recurrent: Some(QWEN38_DIMS),
};

/// One window-KV cell of DeepSeek-V4.1: 512 FP8 codes and an E8M0 scale per
/// 32 (`eidola-engine-cpu` → DeepSeek-V4.1 → Memory images).
const DS_WINDOW_CELL: u64 = 512 + 512 / 32;
/// One global-KV entry of DeepSeek-V4.1: the FP4 KV with an E4M3 scale per
/// 16, then the FP4 index key with an E8M0 scale per 32.
const DS_GLOBAL_CELL: u64 = (256 + 512 / 16) + (64 + 128 / 32);
/// Layers per device: the model splits before layer 20, its first
/// per-position KV source.
const DS_LAYERS_PER_DEVICE: u64 = 20;

/// DeepSeek-V4.1-Flash's numbers, device 0 then device 1 (the model crate's
/// `DsConfig` of the released `config.json`; the KV groups and cells of
/// `eidola-engine-cpu`'s DeepSeek executor, whose packed forms are the bytes
/// a device executor holds).
///
/// The executor's step scratch is its own count ([`StepScratch`]): the
/// device reserve reads that, not the per-token, expert-floor and sampler
/// fields, and `eidola-server-engine`'s `cuda::tests` hold it to the
/// executor's allocations (the index scratch exactly, the rest within its
/// `fixed_bytes`) and lay the released model out wherever the shared check
/// admits it. The KV groups are that executor's (its block layouts, which
/// the same tests hold these equal to).
pub const DEEPSEEK_V41_FLASH: [Shape; 2] = [
    // Device 0: the embedding, layers 0-19 with the first Engram table, and
    // layer 20's bank stub.
    Shape {
        // Three pooled KV sources (layers 2, 8, 14) at ratio 2: one entry
        // per two positions each.
        global_kv_bytes_per_position: 3 * DS_GLOBAL_CELL / 2,
        // Layers 0-19's window KV, one cell per layer.
        sliding_kv_bytes_per_position: DS_LAYERS_PER_DEVICE * DS_WINDOW_CELL,
        // `sliding_window`.
        sliding_window: 128,
        // The drafter is on device 1.
        mtp_layers: 0,
        drafter_kv_bytes_per_position_per_depth: 0,
        drafter_tap_bytes_per_depth: 0,
        drafter_kv_bytes_per_position: 0,
        // The drafter keeps no per-seat state.
        hidden_row_bytes: 0,
        step_token_device_bytes: DS_STEP_TOKEN_DEVICE_BYTES,
        expert_floor_device_bytes: DS_EXPERT_FLOOR_DEVICE_BYTES,
        sampler_row_device_bytes: DS_SAMPLER_ROW_DEVICE_BYTES,
        // Attention reads at most the window and 512 selected entries per
        // query, never a whole context: no split partial rows.
        split_partial_row_bytes: 0,
        rope_device_bytes_per_position: DS_ROPE_DEVICE_BYTES_PER_POSITION,
        // 250.86 GB in the released checkpoint's headers, rounded up.
        weights_bytes: Some(250_870_000_000),
        drafter_weights_bytes: 0,
        step_scratch: Some(StepScratch {
            row_bytes: 602_248,
            draft_row_bytes: 0,
            // The encoder's indexers score the pooled bank.
            index_ratio: 2,
            candidate_entries: 8,
            // The pooled sources' open entries: 2 × 3 sources × 512 f32 for
            // each of `ratio + 1` kept positions, one more per draft.
            seat_bytes: 36_864,
            per_draft_seat_bytes: 12_288,
            drafting_seat_bytes: 0,
            draft_slots: 5,
            fixed_bytes: DS_SCRATCH_FIXED_BYTES,
        }),
        recurrent: None,
        // The executor's encoder groups: the three pooled sources' bank, and
        // layers 0-19's window.
        uniform_kv_groups: &[
            UniformKvGroup {
                layers: 3,
                ratio: 2,
                entry_bytes: DS_GLOBAL_CELL,
                window: false,
                drafter: false,
            },
            UniformKvGroup {
                layers: DS_LAYERS_PER_DEVICE,
                ratio: 1,
                entry_bytes: DS_WINDOW_CELL,
                window: true,
                drafter: false,
            },
        ],
    },
    // Device 1: layers 20-39, the norm and head, the second Engram table
    // (layer 14's), and the drafter with its own embedding copy.
    Shape {
        // Layer 20's per-position KV source.
        global_kv_bytes_per_position: DS_GLOBAL_CELL,
        // Layers 20-39's window KV.
        sliding_kv_bytes_per_position: DS_LAYERS_PER_DEVICE * DS_WINDOW_CELL,
        sliding_window: 128,
        // `num_nextn_predict_layers`; the drafter has five slots, and the
        // grammar holds a CUDA draft to `caps::CUDA_DRAFT_TOKENS`.
        mtp_layers: 3,
        drafter_kv_bytes_per_position_per_depth: 0,
        drafter_tap_bytes_per_depth: 0,
        // Every one of the drafter's three blocks keeps window KV at any
        // draft width.
        drafter_kv_bytes_per_position: 3 * DS_WINDOW_CELL,
        hidden_row_bytes: 0,
        step_token_device_bytes: DS_STEP_TOKEN_DEVICE_BYTES,
        expert_floor_device_bytes: DS_EXPERT_FLOOR_DEVICE_BYTES,
        sampler_row_device_bytes: DS_SAMPLER_ROW_DEVICE_BYTES,
        split_partial_row_bytes: 0,
        rope_device_bytes_per_position: DS_ROPE_DEVICE_BYTES_PER_POSITION,
        // 259.78 GB in the released checkpoint's headers, rounded up.
        weights_bytes: Some(259_790_000_000),
        // The drafter with its own embedding copy: 9,257,832,840 bytes on
        // the device, rounded down.
        drafter_weights_bytes: 9_257_000_000,
        step_scratch: Some(StepScratch {
            row_bytes: 538_964,
            // The draft pass's taps and the main model's rows.
            draft_row_bytes: 56_804,
            // The split layer's per-position source.
            index_ratio: 1,
            candidate_entries: 8,
            // A sampling row: the f32 logits over the head's 129,280 rows,
            // the f64 distribution over the 129,264 sampleable ids, and 36
            // bytes of row state; one per drafted token more.
            seat_bytes: 1_551_268,
            per_draft_seat_bytes: 1_551_268,
            // The draft pass of every slot (its KV, lists, base logits and
            // distributions) and the drafter's own logits row.
            drafting_seat_bytes: 9_312_270,
            draft_slots: 5,
            fixed_bytes: DS_SCRATCH_FIXED_BYTES,
        }),
        recurrent: None,
        // The executor's decoder groups: layer 20's bank, layers 20-39's
        // window, and when drafting the drafter's three blocks' window.
        uniform_kv_groups: &[
            UniformKvGroup {
                layers: 1,
                ratio: 1,
                entry_bytes: DS_GLOBAL_CELL,
                window: false,
                drafter: false,
            },
            UniformKvGroup {
                layers: DS_LAYERS_PER_DEVICE,
                ratio: 1,
                entry_bytes: DS_WINDOW_CELL,
                window: true,
                drafter: false,
            },
            UniformKvGroup {
                layers: 3,
                ratio: 1,
                entry_bytes: DS_WINDOW_CELL,
                window: true,
                drafter: true,
            },
        ],
    },
];

/// Bounds a device's scratch allocations of no row or seat (the status
/// words, the expert groups' ends, the drafter's per-step words) and what the
/// expert layout's rounding adds: at most 37,768 bytes over the released
/// configuration's sizings, rounded up to 40 KiB.
const DS_SCRATCH_FIXED_BYTES: u64 = 40 << 10;

/// The f32 logits row (129,280 × 4 bytes), the four residual copies of the
/// 5,120-wide stream, the 64 query heads of 512 and the six experts' rows:
/// under 1 MB, rounded up to 1.25 MiB.
const DS_STEP_TOKEN_DEVICE_BYTES: u64 = 5 << 18;
/// The expert layout's padding, bounded as MiMo's is: 65,536 rows of at most
/// 24 KiB.
const DS_EXPERT_FLOOR_DEVICE_BYTES: u64 = 65_536 * (24 << 10);
/// The f64 distribution over the head's 129,280 ids (1,034,240 bytes) and
/// the row's buffers, rounded up to 1.25 MiB.
const DS_SAMPLER_ROW_DEVICE_BYTES: u64 = 5 << 18;
/// 64 f32 for each of the two θ (the plain table and the compressed one).
const DS_ROPE_DEVICE_BYTES_PER_POSITION: u64 = 2 * 64 * 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_architecture_round_trips_its_name() {
        for arch in Architecture::ALL {
            assert_eq!(Architecture::from_name(arch.name()), Some(arch));
        }
        assert_eq!(Architecture::from_name("mimo-v2.6-flash"), None);
        assert_eq!(Architecture::from_name(""), None);
    }

    #[test]
    fn flash_is_the_implemented_architecture() {
        assert!(Architecture::MimoV26Flash.is_implemented());
        let flash = Architecture::MimoV26Flash.shape();
        assert_eq!(
            flash.kv_bytes_per_position(),
            (9 * 4 + 39 * 8) * (192 + 128) * 2
        );
        assert_eq!(flash.mtp_layers, 3);
    }

    #[test]
    fn every_implemented_architecture_drafts_within_the_nodes_cap() {
        for arch in Architecture::ALL.into_iter().filter(|a| a.is_implemented()) {
            assert!(arch.mtp_layers() <= crate::engine_deployment::caps::CUDA_DRAFT_TOKENS);
        }
    }

    /// DeepSeek-V4.1-Flash on two devices, at the released checkpoint's
    /// cell sizes: 534 bytes of global KV per position on device 0 (three
    /// sources at ratio 2), 356 on device 1, 528 bytes of window KV per layer
    /// and position, the drafter's three blocks on device 1.
    #[test]
    fn deepseek_is_split_across_two_devices() {
        let ds = Architecture::DeepSeekV41Flash;
        assert!(ds.is_implemented());
        let [d0, d1] = ds.devices() else {
            panic!("two devices")
        };
        assert_eq!((DS_WINDOW_CELL, DS_GLOBAL_CELL), (528, 356));
        assert_eq!(d0.global_kv_bytes_per_position, 534);
        assert_eq!(d1.global_kv_bytes_per_position, 356);
        assert_eq!(d0.sliding_kv_bytes_per_position, 20 * 528);
        assert_eq!(d1.sliding_kv_bytes_per_position, 20 * 528);
        assert_eq!((d0.mtp_layers, d1.mtp_layers, ds.mtp_layers()), (0, 3, 3));
        assert_eq!(d1.drafter_kv_bytes_per_position, 3 * 528);
        assert_eq!(d0.weights_bytes, Some(250_870_000_000));
        assert_eq!(d1.weights_bytes, Some(259_790_000_000));
        // Every split device states its weights; a one-device architecture
        // leaves them to its pack.
        for arch in Architecture::ALL.into_iter().filter(|a| a.is_implemented()) {
            let devices = arch.devices();
            assert!(
                devices
                    .iter()
                    .all(|d| d.weights_bytes.is_some() == (devices.len() > 1)),
                "{arch}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "runs on several devices")]
    fn a_split_architecture_has_no_one_device_shape() {
        let _ = Architecture::DeepSeekV41Flash.shape();
    }

    /// The stub stays a stub until its engine work lands: asking for its
    /// shape is a missing `is_implemented` check, and panics.
    #[test]
    #[should_panic(expected = "MiMo-V2.6-Pro")]
    fn the_pro_stub_has_no_shape() {
        assert!(!Architecture::MimoV26Pro.is_implemented());
        let _ = Architecture::MimoV26Pro.shape();
    }

    /// Qwen3.8-27B on one device, at the released configuration's sizes: 64
    /// KiB of target KV a position over its 16 attention layers, 4 KiB of
    /// the drafter's, a 10 KiB boundary tap, a state image of the 48 DeltaNet
    /// layers (about 157 MB, the executor's padded stride) and about 2 MB of
    /// tail a position.
    #[test]
    fn qwen_is_one_device_with_a_recurrent_state() {
        let qwen = Architecture::Qwen38Dense27B;
        assert!(qwen.is_implemented());
        let shape = qwen.shape();
        assert_eq!(qwen.devices().len(), 1);
        let dims = shape.recurrent.expect("a recurrent state");
        assert_eq!(shape.kv_bytes_per_position(), 65_536);
        assert_eq!(shape.drafter_kv_bytes_per_position, 4096);
        assert_eq!(dims.tap_bytes(), 10_240);
        // 48 × (3 × 10,240 + 48 × 128 × 128) f32 and a tag, to whole conv rows of
        // 40,960 bytes.
        assert_eq!(dims.state_image_bytes(), 156_917_760);
        assert_eq!(dims.tail_bytes_per_position(), 48 * 10_336 * 4);
        assert_eq!(dims.rope_bytes_per_position(), 512);
        assert_eq!(shape.weights_bytes, None);
        assert_eq!(qwen.mtp_layers(), 3);
    }
}
