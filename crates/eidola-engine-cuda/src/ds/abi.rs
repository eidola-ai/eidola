//! The DeepSeek-V4.1 kernels' parameter structs: `repr(C)` mirrors of
//! `eidola-engine-kernels`' `csrc/ds_abi.cuh`, which is normative and states
//! the conventions (rows, BF16 and packed arrays, the KV block images, entry
//! indices, the attention slot list). A device address is a `u64`. Names are
//! the header's without their `Ds` / `DS_` prefix.
//!
//! Every struct's size, alignment and field offsets are asserted at compile
//! time against the numbers the header asserts.

/// Asserts a mirror's layout: its size, 8-byte alignment and every field's
/// offset.
macro_rules! layout {
    ($t:ty, $size:expr, { $($field:ident: $offset:expr),* $(,)? }) => {
        const _: () = {
            assert!(std::mem::size_of::<$t>() == $size);
            assert!(std::mem::align_of::<$t>() == 8);
            $(assert!(std::mem::offset_of!($t, $field) == $offset);)*
        };
    };
}

/// An empty attention slot, and a KV target that writes nothing.
pub const SLOT_EMPTY: u32 = 0xffff_ffff;

/// Status bit: a token id outside the vocabulary.
pub const STATUS_BAD_TOKEN: u32 = 0x100;
/// Status bit: a table row, block-table word or entry out of range.
pub const STATUS_BAD_INDEX: u32 = 0x200;

/// E4M3 codes, a power-of-two scale byte per 32: window KV, attention outputs.
pub const KV_FP8: u32 = 0;
/// E2M1 nibbles, an E4M3 scale byte per 16: the global KV.
pub const KV_FP4_E4M3: u32 = 1;
/// E2M1 nibbles, a power-of-two scale byte per 32: index keys and queries.
pub const KV_FP4_E8M0: u32 = 2;

/// Bytes of one entry's codes.
pub const fn kv_code_bytes(format: u32, width: u32) -> u32 {
    if format == KV_FP8 { width } else { width / 2 }
}

/// Bytes of one entry's scales.
pub const fn kv_scale_bytes(format: u32, width: u32) -> u32 {
    if format == KV_FP4_E4M3 {
        width / 16
    } else {
        width / 32
    }
}

/// [`QuantFp8Params::x_format`]: BF16 values.
pub const X_BF16: u32 = 0;
/// [`QuantFp8Params::x_format`]: stored FP8 rows.
pub const X_FP8: u32 = 1;

/// [`RotQuantParams::rotation`].
pub const ROT_NONE: u32 = 0;
pub const ROT_ROTATE: u32 = 1;
pub const ROT_UNROTATE: u32 = 2;

/// [`LinearParams`] element types.
pub const DT_BF16: u32 = 0;
pub const DT_F32: u32 = 1;

/// [`WindowSlotsParams::order`].
pub const ORDER_POSITION: u32 = 0;
pub const ORDER_RING: u32 = 1;

/// f32 words of one mix row ([`MhcMixParams::out`]).
pub const fn mix_words(copies: u32) -> u32 {
    (2 + copies) * copies
}

/// `DsKvPlane`. One plane of packed KV entries inside a pool: where entry `e`'s codes and scales
/// are. With `block = e / entries_per_block` and `o = e % entries_per_block`, the codes are the
/// `ds_kv_code_bytes(format, width)` bytes at `base + block · block_bytes + codes_offset + o · code
/// bytes`, and the scales the `ds_kv_scale_bytes(format, width)` bytes at `base + block ·
/// block_bytes + scales_offset + o · scale bytes`. A scratch buffer is a plane too (one block, or
/// `block_bytes` its row pitch).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KvPlane {
    /// The pool's (or buffer's) first byte.
    pub base: u64,
    /// Bytes from one block's start to the next's.
    pub block_bytes: u64,
    /// Byte offset of entry 0's codes in a block.
    pub codes_offset: u64,
    /// Byte offset of entry 0's scales in a block.
    pub scales_offset: u64,
    /// Entries a block holds: its positions divided by the group's ratio. Not 0.
    pub entries_per_block: u32,
    /// `DS_KV_*`.
    pub format: u32,
    /// Elements per entry: a multiple of 32.
    pub width: u32,
    /// 0.
    pub reserved: u32,
}
layout!(KvPlane, 48, {
    base: 0, block_bytes: 8, codes_offset: 16, scales_offset: 24, entries_per_block: 32, format: 36, width: 40, reserved: 44,
});

/// `DsEmbedParams`. Embedding: row `r`'s token row, copied into each of the `copies` residual
/// copies. A token at or past `vocab` writes zeros and raises `DS_STATUS_BAD_TOKEN`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmbedParams {
    /// BF16 `[vocab, dim]`.
    pub table: u64,
    /// `[rows]` token ids.
    pub tokens: u64,
    /// BF16 `[rows, copies, dim]`.
    pub out: u64,
    /// One status word, OR-ed into.
    pub status: u64,
    pub rows: u32,
    pub dim: u32,
    /// Residual copies (`hc_mult`).
    pub copies: u32,
    pub vocab: u32,
}
layout!(EmbedParams, 48, {
    table: 0, tokens: 8, out: 16, status: 24, rows: 32, dim: 36, copies: 40, vocab: 44,
});

/// `DsQuantFp8Params`. The FP8 activation quantiser (`quant::act_quant`): per row, per 32-wide
/// block, E4M3 codes of `x / scale` and the block's scale byte (`ds::act_scale_byte`). The input is
/// BF16 values, or stored FP8 rows that are first dequantised as the reference reads them
/// (`bf16(code · 2^(scale - 127))`, one E8M0 scale per 32: Engram rows).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuantFp8Params {
    /// `[rows, width]`: BF16 values, or E4M3 codes when `x_format` is `DS_X_FP8`.
    pub x: u64,
    /// `[rows, width / 32]` E8M0, for `DS_X_FP8`; else unused.
    pub x_scales: u64,
    /// Out: `[rows, width]` E4M3.
    pub codes: u64,
    /// Out: `[rows, width / 32]` activation scale bytes.
    pub scales: u64,
    pub rows: u32,
    /// A multiple of 32.
    pub width: u32,
    /// `DS_X_BF16` or `DS_X_FP8`.
    pub x_format: u32,
    /// 0.
    pub reserved: u32,
}
layout!(QuantFp8Params, 48, {
    x: 0, x_scales: 8, codes: 16, scales: 24, rows: 32, width: 36, x_format: 40, reserved: 44,
});

/// `DsRotQuantParams`. Rotary rotation and a fake quantisation in one pass, into packed codes and
/// scales: the window KV (`rotate`, FP8, into the pool), an attention output (`unrotate` per head,
/// FP8, dense), the global KV (`rotate`, FP4 with E4M3 scales, into the pool), index keys
/// (`rotate`, FP4 with E8M0 scales, into the pool) and index queries (`rotate` per head, FP4 with
/// E8M0 scales, dense). Row `r` is `heads` vectors of `head_dim`; the last `rope_dim` elements of
/// each are rotated by the pairs of row `positions[r]` of `cis` (`rope::rotate` / `unrotate`:
/// nothing fused, each element rounded to BF16), then the whole row is quantised in the format's
/// blocks (`quant::fake_quant_*`). With `targets` the row is written as entry `targets[r]` of `dst`
/// (`heads` 1, `head_dim` the plane's width; an entry of `DS_SLOT_EMPTY` writes nothing); without,
/// to row `r` of `dst_codes` and `dst_scales`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RotQuantParams {
    /// BF16 `[rows, heads · head_dim]`.
    pub x: u64,
    /// `[cis_positions, rope_dim / 2, 2]`: `(cos, sin)` per pair, `RopeTable::cis`. Unused with
    /// `DS_ROT_NONE`.
    pub cis: u64,
    /// `[rows]`: each row's table position (for a pooled entry, its group's first position).
    pub positions: u64,
    /// `[rows]` entries of `dst`, or null for dense output.
    pub targets: u64,
    /// Dense out: `[rows, code bytes of heads · head_dim]`.
    pub dst_codes: u64,
    /// Dense out: `[rows, scale bytes of heads · head_dim]`.
    pub dst_scales: u64,
    /// The pool plane, with `targets`; its `format` is `format`.
    pub dst: KvPlane,
    pub rows: u32,
    pub heads: u32,
    /// A multiple of 32.
    pub head_dim: u32,
    /// Even, at most `head_dim`.
    pub rope_dim: u32,
    /// `DS_ROT_NONE`, `DS_ROT_ROTATE` or `DS_ROT_UNROTATE`.
    pub rotation: u32,
    /// `DS_KV_*`.
    pub format: u32,
    /// Rows of `cis`; every position is below it (the host checks).
    pub cis_positions: u32,
    /// 0.
    pub reserved: u32,
}
layout!(RotQuantParams, 128, {
    x: 0, cis: 8, positions: 16, targets: 24, dst_codes: 32, dst_scales: 40, dst: 48, rows: 96, heads: 100, head_dim: 104, rope_dim: 108, rotation: 112, format: 116, cis_positions: 120, reserved: 124,
});

/// `DsRmsNormParams`. `weights::rms`: `bf16(weight · (x · rsqrt(dot(x, x) / width + eps)))`, the
/// dot in the pinned order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RmsNormParams {
    /// BF16 `[rows, width]`.
    pub x: u64,
    /// `[width]`.
    pub weight: u64,
    /// BF16 `[rows, width]`; may be `x`.
    pub out: u64,
    pub rows: u32,
    pub width: u32,
    pub eps: f32,
    /// 0.
    pub reserved: u32,
}
layout!(RmsNormParams, 40, {
    x: 0, weight: 8, out: 16, rows: 24, width: 28, eps: 32, reserved: 36,
});

/// `DsMhcMixParams`. `mhc::mixes`: a sub-layer's coefficients from the stream entering it. Row `r`
/// of `out` is `pre[copies]`, `post[copies]`, then `comb[copies · copies]` row-major (`comb[i ·
/// copies + j]` mixes old copy `i` into new copy `j`): `DS_MIX_WORDS(copies)` f32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MhcMixParams {
    /// BF16 `[rows, copies · dim]`, copy-major.
    pub x: u64,
    /// `[(2 + copies) · copies, copies · dim]`.
    pub projection: u64,
    /// `[(2 + copies) · copies]`.
    pub base: u64,
    /// `[rows, (2 + copies) · copies]`.
    pub out: u64,
    pub rows: u32,
    pub dim: u32,
    pub copies: u32,
    /// Sinkhorn iterations: one softmax-and-column step, then `iters - 1` row-and-column steps.
    pub iters: u32,
    /// The pre, post and comb scales.
    pub scale: [f32; 3],
    pub norm_eps: f32,
    /// `hc_eps`.
    pub eps: f32,
    /// 0.
    pub reserved: u32,
}
layout!(MhcMixParams, 72, {
    x: 0, projection: 8, base: 16, out: 24, rows: 32, dim: 36, copies: 40, iters: 44, scale: 48, norm_eps: 60, eps: 64, reserved: 68,
});

/// `DsCollapseParams`. `mhc::collapse`: `bf16(Σ_j pre[j] · x[j])`, copies left to right.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CollapseParams {
    /// BF16 `[rows, copies, dim]`.
    pub x: u64,
    /// Row `r`'s `copies` weights at `pre + r · pre_stride`: the previous sub-layer's mix row, or
    /// one-hot rows for a stack's first sub-layer.
    pub pre: u64,
    /// BF16 `[rows, dim]`.
    pub out: u64,
    pub rows: u32,
    pub dim: u32,
    pub copies: u32,
    /// f32 elements between rows of `pre`.
    pub pre_stride: u32,
}
layout!(CollapseParams, 40, {
    x: 0, pre: 8, out: 16, rows: 24, dim: 28, copies: 32, pre_stride: 36,
});

/// `DsExpandParams`. `mhc::expand`: `out[j] = bf16(post[j] · y + Σ_i comb[i][j] · x[i])`, `i`
/// ascending.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpandParams {
    /// BF16 `[rows, copies, dim]`.
    pub x: u64,
    /// BF16 `[rows, dim]`: the sub-layer's output.
    pub y: u64,
    /// Row `r`'s mix row (`DsMhcMixParams::out`'s layout) at `mixes + r · mix_stride`.
    pub mixes: u64,
    /// BF16 `[rows, copies, dim]`; may be `x`.
    pub out: u64,
    pub rows: u32,
    pub dim: u32,
    pub copies: u32,
    /// f32 elements between mix rows.
    pub mix_stride: u32,
}
layout!(ExpandParams, 48, {
    x: 0, y: 8, mixes: 16, out: 24, rows: 32, dim: 36, copies: 40, mix_stride: 44,
});

/// `DsLin8Params`. `linear::lin8` after its quantiser: per row and output, the exact 32-wide FP8 ×
/// FP8 block dots, accumulated in ascending block order as `(dot · activation scale) · weight
/// scale`, then BF16. No weight code is NaN (the loader refuses one).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lin8Params {
    /// `[rows, k]` E4M3 (`DsQuantFp8Params::codes`).
    pub codes: u64,
    /// `[rows, k / 32]` activation scale bytes.
    pub act_scales: u64,
    /// `[n, k]` E4M3, as stored.
    pub w_codes: u64,
    /// `[ceil(n / 32), k / 32]` E8M0, as stored: one per 32 × 32 tile.
    pub w_scales: u64,
    /// BF16 `[rows, n]`.
    pub out: u64,
    pub rows: u32,
    /// Outputs.
    pub n: u32,
    /// Inputs: a multiple of 32.
    pub k: u32,
    /// 0.
    pub reserved: u32,
}
layout!(Lin8Params, 56, {
    codes: 0, act_scales: 8, w_codes: 16, w_scales: 24, out: 32, rows: 40, n: 44, k: 48, reserved: 52,
});

/// `DsLin4Params`. `linear::lin4` after its quantiser, over an expert-major layout: group `g` owns
/// the layout rows from `align_up(group_end[g - 1], row_align)` (0 for the first) to `group_end[g]`
/// and multiplies them by its weight. Rows between runs are padding: never read, their outputs
/// undefined.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lin4Params {
    /// `[layout_rows, k]` E4M3.
    pub codes: u64,
    /// `[layout_rows, k / 32]` activation scale bytes.
    pub act_scales: u64,
    /// `[groups, n, k / 2]` E2M1, element `2i` in the low nibble of byte `i`.
    pub w_packed: u64,
    /// `[groups, n, k / 32]` E8M0.
    pub w_scales: u64,
    /// `[groups]`: each group's end row (`DsPlaceParams::group_end`).
    pub group_end: u64,
    /// BF16 `[layout_rows, n]`.
    pub out: u64,
    pub layout_rows: u32,
    pub n: u32,
    /// A multiple of 32.
    pub k: u32,
    pub groups: u32,
    /// Rows a run starts on a multiple of; not 0.
    pub row_align: u32,
    /// 0.
    pub reserved: u32,
}
layout!(Lin4Params, 72, {
    codes: 0, act_scales: 8, w_packed: 16, w_scales: 24, group_end: 32, out: 40, layout_rows: 48, n: 52, k: 56, groups: 60, row_align: 64, reserved: 68,
});

/// `DsGroupedDotParams`. The attention output's grouped projection (`wo_a`): output `g · rank + i`
/// of a row is `bf16(dot(o_g, w_row))` in the pinned order, where `o_g` is the row's group `g`
/// slice of `k` values read from codes and scales (`ds::dequant_e4m3`) and the weight row is
/// dequantised likewise from its stored FP8 (`Fp8Linear::dequantised_bf16`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroupedDotParams {
    /// `[rows, groups · k]` E4M3: the fake-FP8 attention output.
    pub codes: u64,
    /// `[rows, groups · k / 32]` activation scale bytes.
    pub act_scales: u64,
    /// `[groups · rank, k]` E4M3, as stored.
    pub w_codes: u64,
    /// `[ceil(groups · rank / 32), k / 32]` E8M0.
    pub w_scales: u64,
    /// BF16 `[rows, groups · rank]`.
    pub out: u64,
    pub rows: u32,
    pub groups: u32,
    /// Outputs per group.
    pub rank: u32,
    /// Inputs per group: a multiple of 32.
    pub k: u32,
}
layout!(GroupedDotParams, 56, {
    codes: 0, act_scales: 8, w_codes: 16, w_scales: 24, out: 32, rows: 40, groups: 44, rank: 48, k: 52,
});

/// `DsAttnParams`. `attention::sparse_attend` over each row's slot list. Row `r` lists `a_slots`
/// entries of plane `a` then `b_slots` entries of plane `b`, at `slots + r · slot_stride`; a slot's
/// index in the list fixes its 64-slot block, and `DS_SLOT_EMPTY` is an empty slot. An entry's
/// value is read from its codes and scales (`ds::dequant_*`). For a layer of the backbone `a` is
/// its window plane (`a_slots` the window, position order, right-aligned) and `b` the global KV
/// plane of the bank it reads (ascending entries); for a drafter block `a` is its window plane in
/// ring order and `b` the draft slots' KV in scratch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AttnParams {
    /// BF16 `[rows, heads · head_dim]`: rotated queries.
    pub q: u64,
    /// `[rows, slot_stride]` entry indices.
    pub slots: u64,
    /// `[heads]`.
    pub sink: u64,
    /// BF16 `[rows, heads · head_dim]`.
    pub out: u64,
    /// Entries of the first `a_slots` slots; `width` is `head_dim`.
    pub a: KvPlane,
    /// Entries of the next `b_slots` slots.
    pub b: KvPlane,
    pub rows: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub a_slots: u32,
    pub b_slots: u32,
    /// At least `a_slots + b_slots`.
    pub slot_stride: u32,
    /// `softmax_scale`.
    pub scale: f32,
    /// 0.
    pub reserved: u32,
}
layout!(AttnParams, 160, {
    q: 0, slots: 8, sink: 16, out: 24, a: 32, b: 80, rows: 128, heads: 132, head_dim: 136, a_slots: 140, b_slots: 144, slot_stride: 148, scale: 152, reserved: 156,
});

/// `DsLinearParams`. A plain product in the pinned order (`linear::linear_f32` / `linear_bf16`):
/// `out[r][n] = dot(x_row, w[n])`, plus `addend[r][n]` when given, rounded to BF16 when `out_dtype`
/// says so. The compressor's projections, the index-key projection, the indexer's head weights, the
/// router's logits, the head and the Markov head.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinearParams {
    /// `[*, k]` of `x_dtype`.
    pub x: u64,
    /// `[rows]`: the row of `x` each output row reads, or null for row `r`.
    pub x_rows: u64,
    /// `[n, k]` of `w_dtype`.
    pub w: u64,
    /// `[rows, n]` f32 added to the dots (`addend + dot`), or null.
    pub addend: u64,
    /// `[rows, n]` of `out_dtype`.
    pub out: u64,
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    /// `DS_DT_*`.
    pub x_dtype: u32,
    /// `DS_DT_*`.
    pub w_dtype: u32,
    /// `DS_DT_*`.
    pub out_dtype: u32,
}
layout!(LinearParams, 64, {
    x: 0, x_rows: 8, w: 16, addend: 24, out: 32, rows: 40, n: 44, k: 48, x_dtype: 52, w_dtype: 56, out_dtype: 60,
});

/// `DsPoolParams`. A ratio-2 compressor's pooling: entry `e` pools rows `first[e]` (the earlier
/// position) and `second[e]` of the f32 projections, per channel a softmax over the two scores
/// (`exp32(score - max)`, summed first then second) weighting the two values, rounded to BF16.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PoolParams {
    /// `[*, width]`: `wkv` projections.
    pub kv: u64,
    /// `[*, width]`: `wgate` projections.
    pub score: u64,
    /// `[entries]` rows.
    pub first: u64,
    /// `[entries]` rows.
    pub second: u64,
    /// BF16 `[entries, width]`.
    pub out: u64,
    pub entries: u32,
    pub width: u32,
}
layout!(PoolParams, 48, {
    kv: 0, score: 8, first: 16, second: 24, out: 32, entries: 40, width: 44,
});

/// `DsIndexScoreParams`. `attention::index_score` of every visible entry of each row's bank. Entry
/// `t` of row `r` is entry `tables[table_row[r] · table_stride + t / keys.entries_per_block] ·
/// keys.entries_per_block + t % keys.entries_per_block` of `keys`; a negative table word raises
/// `DS_STATUS_BAD_INDEX`. The head weights are `bf16(head_dots · weight_scale)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IndexScoreParams {
    /// `[rows, heads · dim / 2]` E2M1: the index queries (`DsRotQuantParams`, dense).
    pub q_codes: u64,
    /// `[rows, heads · dim / 32]` activation scale bytes.
    pub q_scales: u64,
    /// BF16 `[rows, heads]`: `bf16(dot(u, weights_proj[h]))`.
    pub head_dots: u64,
    /// The group's block tables, `[*, table_stride]`.
    pub tables: u64,
    /// `[rows]`: each row's table.
    pub table_row: u64,
    /// `[rows]`: entries the row can see, `(position + 1) / ratio`.
    pub visible: u64,
    /// Out: BF16 `[rows, score_stride]`; elements from `visible[r]` on are not written.
    pub scores: u64,
    pub status: u64,
    /// The index-key plane (`DS_KV_FP4_E8M0`, `width` = `dim`).
    pub keys: KvPlane,
    pub rows: u32,
    /// Index heads.
    pub heads: u32,
    /// Index head width.
    pub dim: u32,
    /// Words per table.
    pub table_stride: u32,
    pub score_stride: u32,
    pub weight_scale: f32,
}
layout!(IndexScoreParams, 136, {
    q_codes: 0, q_scales: 8, head_dots: 16, tables: 24, table_row: 32, visible: 40, scores: 48, status: 56, keys: 64, rows: 112, heads: 116, dim: 120, table_stride: 124, score_stride: 128, weight_scale: 132,
});

/// `DsSelectParams`. The total-order top-k over index scores (`attention::top_k`: higher score
/// first, then lower index; `-0` and `+0` one score), the candidate blocks
/// (`attention::candidate_mask`), and the global part of the slot lists. Per row: with `mask_out`,
/// the kept candidate blocks of the row's `visible` entries are written (bit `b % 32` of word `b /
/// 32` for block `b` of `block_entries` entries); the best `top_k` of the visible entries (those in
/// a block of `mask_in`, when given) are written in ascending entry order, as entries of the bank's
/// planes, to slots `slot_first .. slot_first + top_k` of the row's list, the rest `DS_SLOT_EMPTY`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelectParams {
    /// BF16 `[rows, score_stride]`.
    pub scores: u64,
    /// `[rows]`.
    pub visible: u64,
    /// `[rows, mask_words]`, or null to select over everything visible.
    pub mask_in: u64,
    /// `[rows, mask_words]`, or null.
    pub mask_out: u64,
    /// The bank's block tables, `[*, table_stride]`.
    pub tables: u64,
    /// `[rows]`.
    pub table_row: u64,
    /// `[rows, slot_stride]`.
    pub slots: u64,
    pub status: u64,
    pub rows: u32,
    pub score_stride: u32,
    pub mask_words: u32,
    pub table_stride: u32,
    pub slot_stride: u32,
    /// The first global slot: the list's `a_slots`.
    pub slot_first: u32,
    pub top_k: u32,
    /// Candidate blocks kept (the newest always is).
    pub top_blocks: u32,
    /// Entries per candidate block.
    pub block_entries: u32,
    /// Entries per pool block of the bank.
    pub entries_per_block: u32,
}
layout!(SelectParams, 104, {
    scores: 0, visible: 8, mask_in: 16, mask_out: 24, tables: 32, table_row: 40, slots: 48, status: 56, rows: 64, score_stride: 68, mask_words: 72, table_stride: 76, slot_stride: 80, slot_first: 84, top_k: 88, top_blocks: 92, block_entries: 96, entries_per_block: 100,
});

/// `DsWindowSlotsParams`. The window part of the slot lists: slots `0 .. window` of each row. In
/// `DS_ORDER_POSITION` slot `i` is position `p + i + 1 - window` (`p` the row's position), empty
/// when that is negative or below `first[r]`; in `DS_ORDER_RING` (`drafter::ring_order`) slot `i`
/// is `p - (p - i) mod window` for `i < min(window, p + 1)` and empty after. Position `x` is entry
/// `tables[table_row[r] · table_stride + x / block_positions] · block_positions + x %
/// block_positions`; a negative table word raises `DS_STATUS_BAD_INDEX`. The words are the same for
/// every layer of a window group.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindowSlotsParams {
    /// `[rows]`.
    pub positions: u64,
    /// `[rows]`: the lowest position a row lists.
    pub first: u64,
    /// The window group's block tables.
    pub tables: u64,
    /// `[rows]`.
    pub table_row: u64,
    /// `[rows, slot_stride]`.
    pub slots: u64,
    pub status: u64,
    pub rows: u32,
    pub window: u32,
    /// Positions per pool block.
    pub block_positions: u32,
    pub table_stride: u32,
    pub slot_stride: u32,
    /// `DS_ORDER_*`.
    pub order: u32,
}
layout!(WindowSlotsParams, 72, {
    positions: 0, first: 8, tables: 16, table_row: 24, slots: 32, status: 40, rows: 48, window: 52, block_positions: 56, table_stride: 60, slot_stride: 64, order: 68,
});

/// `DsRouteParams`. `moe::route` after the logits: scores `sqrt(softplus(logit))`, the top `active`
/// of `score + bias` under the total order, weights the unbiased scores divided by their sum (added
/// in selection order, best first) plus 1e-20 when `normalize`, times `route_scale`. A row's
/// experts are written in ascending expert id with their weights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RouteParams {
    /// `[rows, experts]`.
    pub logits: u64,
    /// `[experts]`.
    pub bias: u64,
    /// Out: `[rows, active]`, ascending.
    pub ids: u64,
    /// Out: `[rows, active]`.
    pub weights: u64,
    pub rows: u32,
    pub experts: u32,
    pub active: u32,
    /// `norm_topk_prob` and `active > 1`.
    pub normalize: u32,
    pub route_scale: f32,
    /// 0.
    pub reserved: u32,
}
layout!(RouteParams, 56, {
    logits: 0, bias: 8, ids: 16, weights: 24, rows: 32, experts: 36, active: 40, normalize: 44, route_scale: 48, reserved: 52,
});

/// `DsPlaceParams`. The expert-major layout of a step's routed pairs: pairs of one expert keep
/// their (row, slot) order, expert `g`'s run starts at the previous run's start plus its length
/// rounded up to `row_align`, and `group_end[g]` is the end of its pairs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaceParams {
    /// `[rows, active]` (`DsRouteParams::ids`).
    pub ids: u64,
    /// `[rows, active]`.
    pub weights: u64,
    /// Out: `[rows, active]`, each pair's layout row.
    pub row_of: u64,
    /// Out: `[layout_rows]`, each placed row's routing weight.
    pub row_weight: u64,
    /// Out: `[groups]`.
    pub group_end: u64,
    /// Per-block counts for a multi-block launch, or null.
    pub scratch: u64,
    pub rows: u32,
    pub active: u32,
    pub groups: u32,
    /// Not 0.
    pub row_align: u32,
    /// Rows the layout buffers hold.
    pub layout_rows: u32,
    /// 0.
    pub reserved: u32,
}
layout!(PlaceParams, 72, {
    ids: 0, weights: 8, row_of: 16, row_weight: 24, group_end: 32, scratch: 40, rows: 48, active: 52, groups: 56, row_align: 60, layout_rows: 64, reserved: 68,
});

/// `DsSwigluParams`. `moe::swiglu_hidden` and the quantiser of the down-projection: per element
/// `bf16(weight · (silu(min(gate, limit)) · clamp(up, ±limit)))` (no weight for the shared expert,
/// no clamp when `limit <= 0`), then `quant::act_quant` of the row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwigluParams {
    /// BF16, row `r` at `gate + r · in_stride`.
    pub gate: u64,
    /// BF16, row `r` at `up + r · in_stride`.
    pub up: u64,
    /// `[rows]` routing weights, or null.
    pub weight: u64,
    /// Out: `[rows, width]` E4M3.
    pub codes: u64,
    /// Out: `[rows, width / 32]` activation scale bytes.
    pub scales: u64,
    pub rows: u32,
    /// A multiple of 32.
    pub width: u32,
    /// BF16 elements between input rows.
    pub in_stride: u32,
    pub limit: f32,
}
layout!(SwigluParams, 56, {
    gate: 0, up: 8, weight: 16, codes: 24, scales: 32, rows: 40, width: 44, in_stride: 48, limit: 52,
});

/// `DsCombineParams`. The feed-forward sum (`forward::moe_forward`): from +0, a row's routed
/// outputs added in slot order (ascending expert id), then `bf16(sum + shared)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CombineParams {
    /// BF16 `[layout_rows, dim]`.
    pub routed: u64,
    /// BF16 `[rows, dim]`.
    pub shared: u64,
    /// `[rows, active]`.
    pub row_of: u64,
    /// BF16 `[rows, dim]`.
    pub out: u64,
    pub rows: u32,
    pub active: u32,
    pub dim: u32,
    /// 0.
    pub reserved: u32,
}
layout!(CombineParams, 48, {
    routed: 0, shared: 8, row_of: 16, out: 24, rows: 32, active: 36, dim: 40, reserved: 44,
});

/// `DsEngramGatherParams`. Engram table rows, copied as stored: row `r`'s `ids_per_row` table rows
/// back to back. An id at or past `table_rows` copies zeros and raises `DS_STATUS_BAD_INDEX`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngramGatherParams {
    /// `[table_rows, width]` E4M3.
    pub codes: u64,
    /// `[table_rows, width / 32]` E8M0.
    pub scales: u64,
    /// `[rows, ids_per_row]` (`EngramHasher::rows`).
    pub ids: u64,
    /// `[rows, ids_per_row · width]`.
    pub out_codes: u64,
    /// `[rows, ids_per_row · width / 32]`.
    pub out_scales: u64,
    pub status: u64,
    pub table_rows: u64,
    pub rows: u32,
    pub ids_per_row: u32,
    /// A multiple of 32.
    pub width: u32,
    /// 0.
    pub reserved: u32,
}
layout!(EngramGatherParams, 72, {
    codes: 0, scales: 8, ids: 16, out_codes: 24, out_scales: 32, status: 40, table_rows: 48, rows: 56, ids_per_row: 60, width: 64, reserved: 68,
});

/// `DsEngramInjectParams`. `engram::inject` after its projection: per copy the gate from the signed
/// square root of the normalised three-way product (`pinned_sum`), and `bf16(x + gate · value)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngramInjectParams {
    /// BF16 `[rows, copies, dim]`.
    pub x: u64,
    /// BF16 `[rows, (copies + 1) · dim]`: a key per copy, then the value.
    pub kv: u64,
    /// `[copies, dim]`.
    pub gate_weight: u64,
    /// BF16 `[rows, copies, dim]`; may be `x`.
    pub out: u64,
    pub rows: u32,
    pub dim: u32,
    pub copies: u32,
    pub norm_eps: f32,
}
layout!(EngramInjectParams, 48, {
    x: 0, kv: 8, gate_weight: 16, out: 24, rows: 32, dim: 36, copies: 40, norm_eps: 44,
});

/// `DsStreamMeanParams`. `forward::stream_mean`: `bf16((Σ_j x[j]) / copies)`, copies left to right.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamMeanParams {
    /// BF16 `[rows, copies, dim]`.
    pub x: u64,
    /// BF16, row `r` at `out + r · out_stride`.
    pub out: u64,
    pub rows: u32,
    pub dim: u32,
    pub copies: u32,
    /// BF16 elements between output rows.
    pub out_stride: u32,
}
layout!(StreamMeanParams, 32, {
    x: 0, out: 8, rows: 16, dim: 20, copies: 24, out_stride: 28,
});

/// `DsProbeParams`: the numerics probe (`ds_probe.cu`). Item `i` reads
/// `in_words` 64-bit words at `input + i · in_words` and writes `out_words` at
/// `out + i · out_words`, through one function of the numerics header; an f32
/// travels as its 32 bits in the low half of a word, an f64 as its 64, a code
/// or a count as its value. That is the record layout of the host vectors,
/// whose cases a device test replays through this kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProbeParams {
    pub input: u64,
    pub out: u64,
    pub items: u32,
    /// `PROBE_*`.
    pub op: u32,
    /// Words per item; the op fixes them except where its comment gives a rule.
    pub in_words: u32,
    pub out_words: u32,
}
layout!(ProbeParams, 32, {
    input: 0, out: 8, items: 16, op: 20, in_words: 24, out_words: 28,
});

/// Probe ops, named after the vector file's functions. 1 → 1 unless noted.
pub const PROBE_BF16_ROUND: u32 = 0;
pub const PROBE_ROUND_E4M3: u32 = 1;
pub const PROBE_ROUND_E2M1: u32 = 2;
pub const PROBE_EXP32: u32 = 3;
pub const PROBE_SIGMOID32: u32 = 4;
pub const PROBE_SILU32: u32 = 5;
/// A NaN stays a NaN.
pub const PROBE_SOFTPLUS32: u32 = 6;
pub const PROBE_RSQRT32: u32 = 7;
/// amax, inv → scale.
pub const PROBE_ROUND_SCALE: u32 = 8;
pub const PROBE_DET_EXP: u32 = 9;
pub const PROBE_DET_LN: u32 = 10;
/// x → sin, cos.
pub const PROBE_DET_SIN_COS: u32 = 11;
/// base (f64), exponent → power.
pub const PROBE_POW32: u32 = 12;
/// 32 values → 32 code values, scale.
pub const PROBE_ACT_QUANT: u32 = 13;
/// 32 → 32.
pub const PROBE_FQ8: u32 = 14;
/// 32 → 32.
pub const PROBE_FQ4M: u32 = 15;
/// 16 → 16.
pub const PROBE_FQ4N: u32 = 16;
/// 32 codes, 32 codes → dot.
pub const PROBE_FP8_DOT: u32 = 17;
/// 32 codes, 16 packed bytes → dot.
pub const PROBE_FP4_DOT: u32 = 18;
/// a\[n\], b\[n\] → dot; n = `in_words / 2`, at most 128.
pub const PROBE_DOT: u32 = 19;
/// x\[n\] → sum; n = `in_words`, at most 128.
pub const PROBE_PINNED_SUM: u32 = 20;
/// a, b, cos, sin → a', b'.
pub const PROBE_ROTATE: u32 = 21;
pub const PROBE_UNROTATE: u32 = 22;
/// position, frequency → cos, sin.
pub const PROBE_CIS: u32 = 23;
/// gate, up, limit, weighted, weight → hidden.
pub const PROBE_SWIGLU: u32 = 24;
/// score → rank key.
pub const PROBE_TOPK_KEY: u32 = 25;
