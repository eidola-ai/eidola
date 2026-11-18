//! The Qwen3.8 kernels' parameter structs: `repr(C)` mirrors of
//! `eidola-engine-kernels`' `csrc/qwen_abi.cuh`, which is normative and states the
//! conventions (rows, BF16 arrays, the FP8 activation and weight layouts, the paged KV
//! block image). A device address is a `u64`. Names are the header's without their
//! `Qw` / `QW_` prefix.
//!
//! Every struct's size, alignment and field offsets are asserted at compile time against
//! the numbers the header asserts.

macro_rules! layout {
    ($t:ty, $size:expr, { $($field:ident: $offset:expr),* $(,)? }) => {
        const _: () = {
            assert!(std::mem::size_of::<$t>() == $size);
            assert!(std::mem::align_of::<$t>() == 8);
            $(assert!(std::mem::offset_of!($t, $field) == $offset);)*
        };
    };
}

/// Status bit: a token id at or past the vocabulary.
pub const STATUS_BAD_TOKEN: u32 = 0x400;
/// Columns of one FP8 quantisation block, and rows/columns of a weight scale block.
pub const BLOCK: u32 = 128;
/// [`KvWriteParams::targets`]: writes nothing.
pub const NO_TARGET: u32 = 0xffff_ffff;
/// [`GateParams::kind`]: `bf16(bf16(silu(g)) · a)`.
pub const GATE_SWIGLU: u32 = 0;
/// [`GateParams::kind`]: `bf16(a · bf16(σ(g)))`.
pub const GATE_SIGMOID: u32 = 1;
/// The DeltaNet key dimension the recurrence is compiled for.
pub const KEY_DIM: u32 = 128;
/// Widest unit a rotated norm stages.
pub const ROPE_MAX: u32 = 512;
/// Widest unit a norm stages (`kNormMax`).
pub const NORM_MAX: u32 = 8192;
/// State writes one [`DnSeq`] carries.
pub const SEQ_WRITES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmbedParams {
    pub tokens: u64,
    pub table: u64,
    pub out: u64,
    pub status: u64,
    pub rows: u32,
    pub width: u32,
    pub vocab: u32,
    pub out_stride: u32,
}
layout!(EmbedParams, 48, { status: 24, rows: 32, out_stride: 44 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormParams {
    pub x: u64,
    pub add: u64,
    pub weight: u64,
    pub out: u64,
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub positions: u64,
    pub rows: u32,
    pub units: u32,
    pub n: u32,
    pub x_stride: u32,
    pub x_unit_stride: u32,
    pub add_stride: u32,
    pub out_stride: u32,
    pub out_unit_stride: u32,
    pub rotary: u32,
    pub eps: f32,
}
layout!(NormParams, 96, { positions: 48, rows: 56, rotary: 88, eps: 92 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantParams {
    pub x: u64,
    pub codes: u64,
    pub scales: u64,
    pub rows: u32,
    pub k: u32,
    pub x_stride: u32,
    pub m_pad: u32,
}
layout!(QuantParams, 40, { rows: 24, m_pad: 36 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lin8Params {
    pub a: u64,
    pub a_scales: u64,
    pub w: u64,
    pub w_scales: u64,
    pub out: u64,
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub m_pad: u32,
    pub out_stride: u32,
    pub reserved: u32,
}
layout!(Lin8Params, 64, { rows: 40, out_stride: 56 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearParams {
    pub x: u64,
    pub w: u64,
    pub out: u64,
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub x_stride: u32,
    pub out_stride: u32,
    pub out_f32: u32,
}
layout!(LinearParams, 48, { rows: 24, out_f32: 44 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GateParams {
    pub a: u64,
    pub g: u64,
    pub out: u64,
    pub rows: u32,
    pub width: u32,
    pub a_stride: u32,
    pub g_stride: u32,
    pub out_stride: u32,
    pub unit: u32,
    pub g_unit_stride: u32,
    pub kind: u32,
}
layout!(GateParams, 56, { rows: 24, kind: 52 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvWriteParams {
    pub k: u64,
    pub v: u64,
    pub pool: u64,
    pub targets: u64,
    pub rows: u32,
    pub width: u32,
    pub k_stride: u32,
    pub v_stride: u32,
    pub layer: u32,
    pub layers: u32,
    pub block_size: u32,
    pub reserved: u32,
}
layout!(KvWriteParams, 64, { rows: 32, reserved: 60 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AttnParams {
    pub q: u64,
    pub pool: u64,
    pub tables: u64,
    pub positions: u64,
    pub out: u64,
    pub scores: u64,
    pub rows: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_stride: u32,
    pub q_head_stride: u32,
    pub out_stride: u32,
    pub table_stride: u32,
    pub layer: u32,
    pub layers: u32,
    pub block_size: u32,
    pub score_stride: u32,
    pub key_first: u32,
    pub scale: f32,
}
layout!(AttnParams, 104, { rows: 48, scale: 100 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DnPrepParams {
    pub conv_in: u64,
    pub window: u64,
    pub ab: u64,
    pub conv_w: u64,
    pub exp_a_log: u64,
    pub dt_bias: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub beta: u64,
    pub alpha: u64,
    pub positions: u32,
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    /// 1: `alpha` gets the log decay `g` (the fast profile's blocked DeltaNet).
    pub log_decay: u32,
}
layout!(DnPrepParams, 112, { positions: 88, log_decay: 108 });

/// One sequence's run through a DeltaNet layer's recurrence (`QwDnSeq`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DnSeq {
    pub src: u64,
    pub live: u64,
    pub write_dst: [u64; SEQ_WRITES],
    pub first: u32,
    pub count: u32,
    pub out_from: u32,
    pub live_after: u32,
    pub write_after: [u32; SEQ_WRITES],
    pub writes: u32,
    /// The run's first block among the launch's, for the blocked form
    /// ([`DnChunkParams`]); the token recurrence ignores it.
    pub tile: u32,
}
layout!(DnSeq, 88, { first: 48, writes: 80, tile: 84 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DnRecParams {
    pub seqs: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub beta: u64,
    pub alpha: u64,
    pub out: u64,
    pub state_offset: u64,
    pub seq_count: u32,
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub reserved: u32,
}
layout!(DnRecParams, 88, { state_offset: 56, seq_count: 64 });

/// The fast profile's DeltaNet in blocks (`QwDnChunkParams`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DnChunkParams {
    pub seqs: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub beta: u64,
    /// The log decay per position and value head.
    pub g: u64,
    pub out: u64,
    pub w: u64,
    pub u: u64,
    pub pm: u64,
    pub fac: u64,
    pub state_offset: u64,
    pub seq_count: u32,
    /// Blocks the scratch holds: the intra kernel's grid.
    pub tiles: u32,
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    /// Positions a block, 1 to [`CHUNK_TILE`].
    pub chunk: u32,
    pub reserved: u32,
}
layout!(DnChunkParams, 128, { state_offset: 88, seq_count: 96, chunk: 120 });

/// The rows of a block's tile: the longest block of the blocked DeltaNet.
pub const CHUNK_TILE: u32 = 16;
/// Floats of the blocked DeltaNet's scratch per (block, value head): `W` (16 × 128), `U`
/// (16 × `v_dim`), `P` (16 × 16) and the factors (48).
pub fn chunk_scratch_floats(v_dim: u32) -> (usize, usize, usize, usize) {
    let t = CHUNK_TILE as usize;
    (t * KEY_DIM as usize, t * v_dim as usize, t * t, 48)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DnNormParams {
    pub o: u64,
    pub z: u64,
    pub weight: u64,
    pub out: u64,
    pub rows: u32,
    pub v_heads: u32,
    pub v_dim: u32,
    pub z_stride: u32,
    pub eps: f32,
    pub reserved: u32,
}
layout!(DnNormParams, 56, { rows: 32, eps: 48 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WidenParams {
    pub x: u64,
    pub out: u64,
    pub rows: u32,
    pub width: u32,
    pub x_stride: u32,
    pub out_stride: u32,
}
layout!(WidenParams, 32, { out_stride: 28 });

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoundParams {
    pub x: u64,
    pub out: u64,
    pub rows: u32,
    pub width: u32,
    pub x_stride: u32,
    pub out_stride: u32,
    pub out_f32: u32,
    pub reserved: u32,
}
layout!(RoundParams, 40, { out_f32: 32 });
