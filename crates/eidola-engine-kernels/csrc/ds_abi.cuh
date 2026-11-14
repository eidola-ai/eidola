// Parameter structs of the DeepSeek-V4.1 kernels: the by-value argument of each
// kernel family, mirrored field for field by `eidola-engine-cuda`'s
// `src/ds/abi.rs`. Sizes, alignments and offsets are asserted here and there.
// A kernel's launch-contract record carries `sizeof` its struct.
//
// Conventions, for every struct:
//
// - Pointers are device addresses; arrays are dense and row-major. BF16 is the
//   16 stored bits (`ds::bf16_bits`); every value the reference rounds to BF16
//   is kept so between kernels. f32 arrays are the values the reference keeps
//   in f32 (coefficients, router and compressor projections, logits).
// - Activation codes are E4M3 bytes with one scale byte per 32
//   (`ds::act_scale_byte`); FP4 rows are packed two elements a byte, element
//   `2i` in the low nibble of byte `i`. Fake-quantised values are never stored
//   widened: a consumer reads codes and scales (`ds::dequant_*`), which gives
//   the reference's value exactly.
// - A row is one position of the step on this device. A kernel computes rows
//   `0 .. rows` of its arrays, each from that row alone, with a launch geometry
//   that is a function of `rows` only. The executor orders a step's positions
//   so that the positions a layer computes are a prefix of every array:
//   deepest-computed first. A layer of the window chain then runs its KV write
//   over the rows it computes and everything after its queries over the
//   (shorter) prefix the next layer computes.
// - Indices that are device data (tokens, table rows, slot entries, block
//   tables) are bounded by the kernel where a struct says it raises a status
//   bit; every other count, stride and position is the host's to check.
//
// KV entries. A pool holds blocks; a block holds, for each layer (or bank
// source) of its group, planes of packed entries, `DsKvPlane` naming one plane.
// The block images, in the order a block is zeroed, copied and enciphered:
//
// - window groups (the encoder's, the decoder's): per layer, codes
//   `[B, width]` then scales `[B, width / 32]` (`DS_KV_FP8`), `B` the block's
//   positions;
// - bank groups (ratio-2 sources, the ratio-1 bank): per source, with
//   `E = B / ratio` entries: KV codes `[E, width / 2]`, KV scales
//   `[E, width / 16]` (`DS_KV_FP4_E4M3`), index-key codes `[E, index / 2]`,
//   index-key scales `[E, index / 32]` (`DS_KV_FP4_E8M0`);
// - the drafter's windows: per draft block, codes then scales as a window
//   group's, then the block's boundary tap (BF16 stream means of the target
//   layers at the block's last position).
//
// An entry index is `block id · entries per block + offset`. The canonical
// attention slot list of a query is `DsAttnParams`'s.
#pragma once

#include <cstddef>
#include <cstdint>

#define DS_ABI_STRUCT(T, size) \
  static_assert(sizeof(T) == (size) && alignof(T) == 8, #T " layout is mirrored on the host")
#define DS_ABI_FIELD(T, field, offset) \
  static_assert(offsetof(T, field) == (offset), #T "::" #field " offset is mirrored on the host")

// An empty attention slot, and a KV target that writes nothing.
constexpr uint32_t DS_SLOT_EMPTY = 0xffffffffu;

// Status bits, apart from the sampler's and the row copy's.
constexpr uint32_t DS_STATUS_BAD_TOKEN = 0x100u;
constexpr uint32_t DS_STATUS_BAD_INDEX = 0x200u;

// Packed entry formats.
// E4M3 codes, a power-of-two scale byte per 32: window KV, attention outputs.
constexpr uint32_t DS_KV_FP8 = 0;
// E2M1 nibbles, an E4M3 scale byte per 16: the global KV.
constexpr uint32_t DS_KV_FP4_E4M3 = 1;
// E2M1 nibbles, a power-of-two scale byte per 32: index keys and queries.
constexpr uint32_t DS_KV_FP4_E8M0 = 2;

constexpr uint32_t ds_kv_code_bytes(uint32_t format, uint32_t width) {
  return format == DS_KV_FP8 ? width : width / 2;
}
constexpr uint32_t ds_kv_scale_bytes(uint32_t format, uint32_t width) {
  return format == DS_KV_FP4_E4M3 ? width / 16 : width / 32;
}

// `DsQuantFp8Params::x_format`.
constexpr uint32_t DS_X_BF16 = 0;
constexpr uint32_t DS_X_FP8 = 1;

// `DsRotQuantParams::rotation`.
constexpr uint32_t DS_ROT_NONE = 0;
constexpr uint32_t DS_ROT_ROTATE = 1;
constexpr uint32_t DS_ROT_UNROTATE = 2;

// `DsLinearParams` element types.
constexpr uint32_t DS_DT_BF16 = 0;
constexpr uint32_t DS_DT_F32 = 1;

// `DsWindowSlotsParams::order`.
constexpr uint32_t DS_ORDER_POSITION = 0;
constexpr uint32_t DS_ORDER_RING = 1;

// f32 words of one mix row (`DsMhcMixParams::out`).
constexpr uint32_t ds_mix_words(uint32_t copies) { return (2 + copies) * copies; }

// One plane of packed KV entries inside a pool: where entry `e`'s codes and scales are. With `block
// = e / entries_per_block` and `o = e % entries_per_block`, the codes are the
// `ds_kv_code_bytes(format, width)` bytes at `base + block · block_bytes + codes_offset + o · code
// bytes`, and the scales the `ds_kv_scale_bytes(format, width)` bytes at `base + block ·
// block_bytes + scales_offset + o · scale bytes`. A scratch buffer is a plane too (one block, or
// `block_bytes` its row pitch).
struct DsKvPlane {
  // The pool's (or buffer's) first byte.
  uint8_t* base;
  // Bytes from one block's start to the next's.
  uint64_t block_bytes;
  // Byte offset of entry 0's codes in a block.
  uint64_t codes_offset;
  // Byte offset of entry 0's scales in a block.
  uint64_t scales_offset;
  // Entries a block holds: its positions divided by the group's ratio. Not 0.
  uint32_t entries_per_block;
  // `DS_KV_*`.
  uint32_t format;
  // Elements per entry: a multiple of 32.
  uint32_t width;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsKvPlane, 48);
DS_ABI_FIELD(DsKvPlane, base, 0);
DS_ABI_FIELD(DsKvPlane, block_bytes, 8);
DS_ABI_FIELD(DsKvPlane, codes_offset, 16);
DS_ABI_FIELD(DsKvPlane, scales_offset, 24);
DS_ABI_FIELD(DsKvPlane, entries_per_block, 32);
DS_ABI_FIELD(DsKvPlane, format, 36);
DS_ABI_FIELD(DsKvPlane, width, 40);
DS_ABI_FIELD(DsKvPlane, reserved, 44);

// Embedding: row `r`'s token row, copied into each of the `copies` residual copies. A token at or
// past `vocab` writes zeros and raises `DS_STATUS_BAD_TOKEN`.
struct DsEmbedParams {
  // BF16 `[vocab, dim]`.
  const uint16_t* table;
  // `[rows]` token ids.
  const uint32_t* tokens;
  // BF16 `[rows, copies, dim]`.
  uint16_t* out;
  // One status word, OR-ed into.
  uint32_t* status;
  uint32_t rows;
  uint32_t dim;
  // Residual copies (`hc_mult`).
  uint32_t copies;
  uint32_t vocab;
};
DS_ABI_STRUCT(DsEmbedParams, 48);
DS_ABI_FIELD(DsEmbedParams, table, 0);
DS_ABI_FIELD(DsEmbedParams, tokens, 8);
DS_ABI_FIELD(DsEmbedParams, out, 16);
DS_ABI_FIELD(DsEmbedParams, status, 24);
DS_ABI_FIELD(DsEmbedParams, rows, 32);
DS_ABI_FIELD(DsEmbedParams, dim, 36);
DS_ABI_FIELD(DsEmbedParams, copies, 40);
DS_ABI_FIELD(DsEmbedParams, vocab, 44);

// The FP8 activation quantiser (`quant::act_quant`): per row, per 32-wide block, E4M3 codes of `x /
// scale` and the block's scale byte (`ds::act_scale_byte`). The input is BF16 values, or stored FP8
// rows that are first dequantised as the reference reads them (`bf16(code · 2^(scale - 127))`, one
// E8M0 scale per 32: Engram rows).
struct DsQuantFp8Params {
  // `[rows, width]`: BF16 values, or E4M3 codes when `x_format` is `DS_X_FP8`.
  const void* x;
  // `[rows, width / 32]` E8M0, for `DS_X_FP8`; else unused.
  const uint8_t* x_scales;
  // Out: `[rows, width]` E4M3.
  uint8_t* codes;
  // Out: `[rows, width / 32]` activation scale bytes.
  uint8_t* scales;
  uint32_t rows;
  // A multiple of 32.
  uint32_t width;
  // `DS_X_BF16` or `DS_X_FP8`.
  uint32_t x_format;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsQuantFp8Params, 48);
DS_ABI_FIELD(DsQuantFp8Params, x, 0);
DS_ABI_FIELD(DsQuantFp8Params, x_scales, 8);
DS_ABI_FIELD(DsQuantFp8Params, codes, 16);
DS_ABI_FIELD(DsQuantFp8Params, scales, 24);
DS_ABI_FIELD(DsQuantFp8Params, rows, 32);
DS_ABI_FIELD(DsQuantFp8Params, width, 36);
DS_ABI_FIELD(DsQuantFp8Params, x_format, 40);
DS_ABI_FIELD(DsQuantFp8Params, reserved, 44);

// Rotary rotation and a fake quantisation in one pass, into packed codes and scales: the window KV
// (`rotate`, FP8, into the pool), an attention output (`unrotate` per head, FP8, dense), the global
// KV (`rotate`, FP4 with E4M3 scales, into the pool), index keys (`rotate`, FP4 with E8M0 scales,
// into the pool) and index queries (`rotate` per head, FP4 with E8M0 scales, dense). Row `r` is
// `heads` vectors of `head_dim`; the last `rope_dim` elements of each are rotated by the pairs of
// row `positions[r]` of `cis` (`rope::rotate` / `unrotate`: nothing fused, each element rounded to
// BF16), then the whole row is quantised in the format's blocks (`quant::fake_quant_*`). With
// `targets` the row is written as entry `targets[r]` of `dst` (`heads` 1, `head_dim` the plane's
// width; an entry of `DS_SLOT_EMPTY` writes nothing); without, to row `r` of `dst_codes` and
// `dst_scales`.
struct DsRotQuantParams {
  // BF16 `[rows, heads · head_dim]`.
  const uint16_t* x;
  // `[cis_positions, rope_dim / 2, 2]`: `(cos, sin)` per pair, `RopeTable::cis`. Unused with
  // `DS_ROT_NONE`.
  const float* cis;
  // `[rows]`: each row's table position (for a pooled entry, its group's first position).
  const uint32_t* positions;
  // `[rows]` entries of `dst`, or null for dense output.
  const uint32_t* targets;
  // Dense out: `[rows, code bytes of heads · head_dim]`.
  uint8_t* dst_codes;
  // Dense out: `[rows, scale bytes of heads · head_dim]`.
  uint8_t* dst_scales;
  // The pool plane, with `targets`; its `format` is `format`.
  DsKvPlane dst;
  uint32_t rows;
  uint32_t heads;
  // A multiple of 32.
  uint32_t head_dim;
  // Even, at most `head_dim`.
  uint32_t rope_dim;
  // `DS_ROT_NONE`, `DS_ROT_ROTATE` or `DS_ROT_UNROTATE`.
  uint32_t rotation;
  // `DS_KV_*`.
  uint32_t format;
  // Rows of `cis`; every position is below it (the host checks).
  uint32_t cis_positions;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsRotQuantParams, 128);
DS_ABI_FIELD(DsRotQuantParams, x, 0);
DS_ABI_FIELD(DsRotQuantParams, cis, 8);
DS_ABI_FIELD(DsRotQuantParams, positions, 16);
DS_ABI_FIELD(DsRotQuantParams, targets, 24);
DS_ABI_FIELD(DsRotQuantParams, dst_codes, 32);
DS_ABI_FIELD(DsRotQuantParams, dst_scales, 40);
DS_ABI_FIELD(DsRotQuantParams, dst, 48);
DS_ABI_FIELD(DsRotQuantParams, rows, 96);
DS_ABI_FIELD(DsRotQuantParams, heads, 100);
DS_ABI_FIELD(DsRotQuantParams, head_dim, 104);
DS_ABI_FIELD(DsRotQuantParams, rope_dim, 108);
DS_ABI_FIELD(DsRotQuantParams, rotation, 112);
DS_ABI_FIELD(DsRotQuantParams, format, 116);
DS_ABI_FIELD(DsRotQuantParams, cis_positions, 120);
DS_ABI_FIELD(DsRotQuantParams, reserved, 124);

// `weights::rms`: `bf16(weight · (x · rsqrt(dot(x, x) / width + eps)))`, the dot in the pinned
// order.
struct DsRmsNormParams {
  // BF16 `[rows, width]`.
  const uint16_t* x;
  // `[width]`.
  const float* weight;
  // BF16 `[rows, width]`; may be `x`.
  uint16_t* out;
  uint32_t rows;
  uint32_t width;
  float eps;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsRmsNormParams, 40);
DS_ABI_FIELD(DsRmsNormParams, x, 0);
DS_ABI_FIELD(DsRmsNormParams, weight, 8);
DS_ABI_FIELD(DsRmsNormParams, out, 16);
DS_ABI_FIELD(DsRmsNormParams, rows, 24);
DS_ABI_FIELD(DsRmsNormParams, width, 28);
DS_ABI_FIELD(DsRmsNormParams, eps, 32);
DS_ABI_FIELD(DsRmsNormParams, reserved, 36);

// `mhc::mixes`: a sub-layer's coefficients from the stream entering it. Row `r` of `out` is
// `pre[copies]`, `post[copies]`, then `comb[copies · copies]` row-major (`comb[i · copies + j]`
// mixes old copy `i` into new copy `j`): `DS_MIX_WORDS(copies)` f32.
struct DsMhcMixParams {
  // BF16 `[rows, copies · dim]`, copy-major.
  const uint16_t* x;
  // `[(2 + copies) · copies, copies · dim]`.
  const float* projection;
  // `[(2 + copies) · copies]`.
  float* base;
  // `[rows, (2 + copies) · copies]`.
  float* out;
  uint32_t rows;
  uint32_t dim;
  uint32_t copies;
  // Sinkhorn iterations: one softmax-and-column step, then `iters - 1` row-and-column steps.
  uint32_t iters;
  // The pre, post and comb scales.
  float scale[3];
  float norm_eps;
  // `hc_eps`.
  float eps;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsMhcMixParams, 72);
DS_ABI_FIELD(DsMhcMixParams, x, 0);
DS_ABI_FIELD(DsMhcMixParams, projection, 8);
DS_ABI_FIELD(DsMhcMixParams, base, 16);
DS_ABI_FIELD(DsMhcMixParams, out, 24);
DS_ABI_FIELD(DsMhcMixParams, rows, 32);
DS_ABI_FIELD(DsMhcMixParams, dim, 36);
DS_ABI_FIELD(DsMhcMixParams, copies, 40);
DS_ABI_FIELD(DsMhcMixParams, iters, 44);
DS_ABI_FIELD(DsMhcMixParams, scale, 48);
DS_ABI_FIELD(DsMhcMixParams, norm_eps, 60);
DS_ABI_FIELD(DsMhcMixParams, eps, 64);
DS_ABI_FIELD(DsMhcMixParams, reserved, 68);

// `mhc::collapse`: `bf16(Σ_j pre[j] · x[j])`, copies left to right.
struct DsCollapseParams {
  // BF16 `[rows, copies, dim]`.
  const uint16_t* x;
  // Row `r`'s `copies` weights at `pre + r · pre_stride`: the previous sub-layer's mix row, or one-
  // hot rows for a stack's first sub-layer.
  const float* pre;
  // BF16 `[rows, dim]`.
  uint16_t* out;
  uint32_t rows;
  uint32_t dim;
  uint32_t copies;
  // f32 elements between rows of `pre`.
  uint32_t pre_stride;
};
DS_ABI_STRUCT(DsCollapseParams, 40);
DS_ABI_FIELD(DsCollapseParams, x, 0);
DS_ABI_FIELD(DsCollapseParams, pre, 8);
DS_ABI_FIELD(DsCollapseParams, out, 16);
DS_ABI_FIELD(DsCollapseParams, rows, 24);
DS_ABI_FIELD(DsCollapseParams, dim, 28);
DS_ABI_FIELD(DsCollapseParams, copies, 32);
DS_ABI_FIELD(DsCollapseParams, pre_stride, 36);

// `mhc::expand`: `out[j] = bf16(post[j] · y + Σ_i comb[i][j] · x[i])`, `i` ascending.
struct DsExpandParams {
  // BF16 `[rows, copies, dim]`.
  const uint16_t* x;
  // BF16 `[rows, dim]`: the sub-layer's output.
  const uint16_t* y;
  // Row `r`'s mix row (`DsMhcMixParams::out`'s layout) at `mixes + r · mix_stride`.
  const float* mixes;
  // BF16 `[rows, copies, dim]`; may be `x`.
  uint16_t* out;
  uint32_t rows;
  uint32_t dim;
  uint32_t copies;
  // f32 elements between mix rows.
  uint32_t mix_stride;
};
DS_ABI_STRUCT(DsExpandParams, 48);
DS_ABI_FIELD(DsExpandParams, x, 0);
DS_ABI_FIELD(DsExpandParams, y, 8);
DS_ABI_FIELD(DsExpandParams, mixes, 16);
DS_ABI_FIELD(DsExpandParams, out, 24);
DS_ABI_FIELD(DsExpandParams, rows, 32);
DS_ABI_FIELD(DsExpandParams, dim, 36);
DS_ABI_FIELD(DsExpandParams, copies, 40);
DS_ABI_FIELD(DsExpandParams, mix_stride, 44);

// `linear::lin8` after its quantiser: per row and output, the exact 32-wide FP8 × FP8 block dots,
// accumulated in ascending block order as `(dot · activation scale) · weight scale`, then BF16. No
// weight code is NaN (the loader refuses one).
struct DsLin8Params {
  // `[rows, k]` E4M3 (`DsQuantFp8Params::codes`).
  uint8_t* codes;
  // `[rows, k / 32]` activation scale bytes.
  const uint8_t* act_scales;
  // `[n, k]` E4M3, as stored.
  const uint8_t* w_codes;
  // `[ceil(n / 32), k / 32]` E8M0, as stored: one per 32 × 32 tile.
  const uint8_t* w_scales;
  // BF16 `[rows, n]`.
  uint16_t* out;
  uint32_t rows;
  // Outputs.
  uint32_t n;
  // Inputs: a multiple of 32.
  uint32_t k;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsLin8Params, 56);
DS_ABI_FIELD(DsLin8Params, codes, 0);
DS_ABI_FIELD(DsLin8Params, act_scales, 8);
DS_ABI_FIELD(DsLin8Params, w_codes, 16);
DS_ABI_FIELD(DsLin8Params, w_scales, 24);
DS_ABI_FIELD(DsLin8Params, out, 32);
DS_ABI_FIELD(DsLin8Params, rows, 40);
DS_ABI_FIELD(DsLin8Params, n, 44);
DS_ABI_FIELD(DsLin8Params, k, 48);
DS_ABI_FIELD(DsLin8Params, reserved, 52);

// `linear::lin4` after its quantiser, over an expert-major layout: group `g` owns the layout rows
// from `align_up(group_end[g - 1], row_align)` (0 for the first) to `group_end[g]` and multiplies
// them by its weight. Rows between runs are padding: never read, their outputs undefined.
struct DsLin4Params {
  // `[layout_rows, k]` E4M3.
  uint8_t* codes;
  // `[layout_rows, k / 32]` activation scale bytes.
  const uint8_t* act_scales;
  // `[groups, n, k / 2]` E2M1, element `2i` in the low nibble of byte `i`.
  const uint8_t* w_packed;
  // `[groups, n, k / 32]` E8M0.
  const uint8_t* w_scales;
  // `[groups]`: each group's end row (`DsPlaceParams::group_end`).
  uint32_t* group_end;
  // BF16 `[layout_rows, n]`.
  uint16_t* out;
  uint32_t layout_rows;
  uint32_t n;
  // A multiple of 32.
  uint32_t k;
  uint32_t groups;
  // Rows a run starts on a multiple of; not 0.
  uint32_t row_align;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsLin4Params, 72);
DS_ABI_FIELD(DsLin4Params, codes, 0);
DS_ABI_FIELD(DsLin4Params, act_scales, 8);
DS_ABI_FIELD(DsLin4Params, w_packed, 16);
DS_ABI_FIELD(DsLin4Params, w_scales, 24);
DS_ABI_FIELD(DsLin4Params, group_end, 32);
DS_ABI_FIELD(DsLin4Params, out, 40);
DS_ABI_FIELD(DsLin4Params, layout_rows, 48);
DS_ABI_FIELD(DsLin4Params, n, 52);
DS_ABI_FIELD(DsLin4Params, k, 56);
DS_ABI_FIELD(DsLin4Params, groups, 60);
DS_ABI_FIELD(DsLin4Params, row_align, 64);
DS_ABI_FIELD(DsLin4Params, reserved, 68);

// The attention output's grouped projection (`wo_a`): output `g · rank + i` of a row is
// `bf16(dot(o_g, w_row))` in the pinned order, where `o_g` is the row's group `g` slice of `k`
// values read from codes and scales (`ds::dequant_e4m3`) and the weight row is dequantised likewise
// from its stored FP8 (`Fp8Linear::dequantised_bf16`).
struct DsGroupedDotParams {
  // `[rows, groups · k]` E4M3: the fake-FP8 attention output.
  uint8_t* codes;
  // `[rows, groups · k / 32]` activation scale bytes.
  const uint8_t* act_scales;
  // `[groups · rank, k]` E4M3, as stored.
  const uint8_t* w_codes;
  // `[ceil(groups · rank / 32), k / 32]` E8M0.
  const uint8_t* w_scales;
  // BF16 `[rows, groups · rank]`.
  uint16_t* out;
  uint32_t rows;
  uint32_t groups;
  // Outputs per group.
  uint32_t rank;
  // Inputs per group: a multiple of 32.
  uint32_t k;
};
DS_ABI_STRUCT(DsGroupedDotParams, 56);
DS_ABI_FIELD(DsGroupedDotParams, codes, 0);
DS_ABI_FIELD(DsGroupedDotParams, act_scales, 8);
DS_ABI_FIELD(DsGroupedDotParams, w_codes, 16);
DS_ABI_FIELD(DsGroupedDotParams, w_scales, 24);
DS_ABI_FIELD(DsGroupedDotParams, out, 32);
DS_ABI_FIELD(DsGroupedDotParams, rows, 40);
DS_ABI_FIELD(DsGroupedDotParams, groups, 44);
DS_ABI_FIELD(DsGroupedDotParams, rank, 48);
DS_ABI_FIELD(DsGroupedDotParams, k, 52);

// `attention::sparse_attend` over each row's slot list. Row `r` lists `a_slots` entries of plane
// `a` then `b_slots` entries of plane `b`, at `slots + r · slot_stride`; a slot's index in the list
// fixes its 64-slot block, and `DS_SLOT_EMPTY` is an empty slot. An entry's value is read from its
// codes and scales (`ds::dequant_*`). For a layer of the backbone `a` is its window plane
// (`a_slots` the window, position order, right-aligned) and `b` the global KV plane of the bank it
// reads (ascending entries); for a drafter block `a` is its window plane in ring order and `b` the
// draft slots' KV in scratch.
struct DsAttnParams {
  // BF16 `[rows, heads · head_dim]`: rotated queries.
  const uint16_t* q;
  // `[rows, slot_stride]` entry indices.
  uint32_t* slots;
  // `[heads]`.
  const float* sink;
  // BF16 `[rows, heads · head_dim]`.
  uint16_t* out;
  // Entries of the first `a_slots` slots; `width` is `head_dim`.
  DsKvPlane a;
  // Entries of the next `b_slots` slots.
  DsKvPlane b;
  uint32_t rows;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t a_slots;
  uint32_t b_slots;
  // At least `a_slots + b_slots`.
  uint32_t slot_stride;
  // `softmax_scale`.
  float scale;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsAttnParams, 160);
DS_ABI_FIELD(DsAttnParams, q, 0);
DS_ABI_FIELD(DsAttnParams, slots, 8);
DS_ABI_FIELD(DsAttnParams, sink, 16);
DS_ABI_FIELD(DsAttnParams, out, 24);
DS_ABI_FIELD(DsAttnParams, a, 32);
DS_ABI_FIELD(DsAttnParams, b, 80);
DS_ABI_FIELD(DsAttnParams, rows, 128);
DS_ABI_FIELD(DsAttnParams, heads, 132);
DS_ABI_FIELD(DsAttnParams, head_dim, 136);
DS_ABI_FIELD(DsAttnParams, a_slots, 140);
DS_ABI_FIELD(DsAttnParams, b_slots, 144);
DS_ABI_FIELD(DsAttnParams, slot_stride, 148);
DS_ABI_FIELD(DsAttnParams, scale, 152);
DS_ABI_FIELD(DsAttnParams, reserved, 156);

// A plain product in the pinned order (`linear::linear_f32` / `linear_bf16`): `out[r][n] =
// dot(x_row, w[n])`, plus `addend[r][n]` when given, rounded to BF16 when `out_dtype` says so. The
// compressor's projections, the index-key projection, the indexer's head weights, the router's
// logits, the head and the Markov head.
struct DsLinearParams {
  // `[*, k]` of `x_dtype`.
  const void* x;
  // `[rows]`: the row of `x` each output row reads, or null for row `r`.
  const uint32_t* x_rows;
  // `[n, k]` of `w_dtype`.
  const void* w;
  // `[rows, n]` f32 added to the dots (`addend + dot`), or null.
  const float* addend;
  // `[rows, n]` of `out_dtype`.
  void* out;
  uint32_t rows;
  uint32_t n;
  uint32_t k;
  // `DS_DT_*`.
  uint32_t x_dtype;
  // `DS_DT_*`.
  uint32_t w_dtype;
  // `DS_DT_*`.
  uint32_t out_dtype;
};
DS_ABI_STRUCT(DsLinearParams, 64);
DS_ABI_FIELD(DsLinearParams, x, 0);
DS_ABI_FIELD(DsLinearParams, x_rows, 8);
DS_ABI_FIELD(DsLinearParams, w, 16);
DS_ABI_FIELD(DsLinearParams, addend, 24);
DS_ABI_FIELD(DsLinearParams, out, 32);
DS_ABI_FIELD(DsLinearParams, rows, 40);
DS_ABI_FIELD(DsLinearParams, n, 44);
DS_ABI_FIELD(DsLinearParams, k, 48);
DS_ABI_FIELD(DsLinearParams, x_dtype, 52);
DS_ABI_FIELD(DsLinearParams, w_dtype, 56);
DS_ABI_FIELD(DsLinearParams, out_dtype, 60);

// A ratio-2 compressor's pooling: entry `e` pools rows `first[e]` (the earlier position) and
// `second[e]` of the f32 projections, per channel a softmax over the two scores (`exp32(score -
// max)`, summed first then second) weighting the two values, rounded to BF16.
struct DsPoolParams {
  // `[*, width]`: `wkv` projections.
  const float* kv;
  // `[*, width]`: `wgate` projections.
  const float* score;
  // `[entries]` rows.
  const uint32_t* first;
  // `[entries]` rows.
  const uint32_t* second;
  // BF16 `[entries, width]`.
  uint16_t* out;
  uint32_t entries;
  uint32_t width;
};
DS_ABI_STRUCT(DsPoolParams, 48);
DS_ABI_FIELD(DsPoolParams, kv, 0);
DS_ABI_FIELD(DsPoolParams, score, 8);
DS_ABI_FIELD(DsPoolParams, first, 16);
DS_ABI_FIELD(DsPoolParams, second, 24);
DS_ABI_FIELD(DsPoolParams, out, 32);
DS_ABI_FIELD(DsPoolParams, entries, 40);
DS_ABI_FIELD(DsPoolParams, width, 44);

// `attention::index_score` of every visible entry of each row's bank. Entry `t` of row `r` is entry
// `tables[table_row[r] · table_stride + t / keys.entries_per_block] · keys.entries_per_block + t %
// keys.entries_per_block` of `keys`; a negative table word raises `DS_STATUS_BAD_INDEX`. The head
// weights are `bf16(head_dots · weight_scale)`.
struct DsIndexScoreParams {
  // `[rows, heads · dim / 2]` E2M1: the index queries (`DsRotQuantParams`, dense).
  const uint8_t* q_codes;
  // `[rows, heads · dim / 32]` activation scale bytes.
  const uint8_t* q_scales;
  // BF16 `[rows, heads]`: `bf16(dot(u, weights_proj[h]))`.
  const uint16_t* head_dots;
  // The group's block tables, `[*, table_stride]`.
  const int32_t* tables;
  // `[rows]`: each row's table.
  const uint32_t* table_row;
  // `[rows]`: entries the row can see, `(position + 1) / ratio`.
  const uint32_t* visible;
  // Out: BF16 `[rows, score_stride]`; elements from `visible[r]` on are not written.
  uint16_t* scores;
  uint32_t* status;
  // The index-key plane (`DS_KV_FP4_E8M0`, `width` = `dim`).
  DsKvPlane keys;
  uint32_t rows;
  // Index heads.
  uint32_t heads;
  // Index head width.
  uint32_t dim;
  // Words per table.
  uint32_t table_stride;
  uint32_t score_stride;
  float weight_scale;
};
DS_ABI_STRUCT(DsIndexScoreParams, 136);
DS_ABI_FIELD(DsIndexScoreParams, q_codes, 0);
DS_ABI_FIELD(DsIndexScoreParams, q_scales, 8);
DS_ABI_FIELD(DsIndexScoreParams, head_dots, 16);
DS_ABI_FIELD(DsIndexScoreParams, tables, 24);
DS_ABI_FIELD(DsIndexScoreParams, table_row, 32);
DS_ABI_FIELD(DsIndexScoreParams, visible, 40);
DS_ABI_FIELD(DsIndexScoreParams, scores, 48);
DS_ABI_FIELD(DsIndexScoreParams, status, 56);
DS_ABI_FIELD(DsIndexScoreParams, keys, 64);
DS_ABI_FIELD(DsIndexScoreParams, rows, 112);
DS_ABI_FIELD(DsIndexScoreParams, heads, 116);
DS_ABI_FIELD(DsIndexScoreParams, dim, 120);
DS_ABI_FIELD(DsIndexScoreParams, table_stride, 124);
DS_ABI_FIELD(DsIndexScoreParams, score_stride, 128);
DS_ABI_FIELD(DsIndexScoreParams, weight_scale, 132);

// The total-order top-k over index scores (`attention::top_k`: higher score first, then lower
// index; `-0` and `+0` one score), the candidate blocks (`attention::candidate_mask`), and the
// global part of the slot lists. Per row: with `mask_out`, the kept candidate blocks of the row's
// `visible` entries are written (bit `b % 32` of word `b / 32` for block `b` of `block_entries`
// entries); the best `top_k` of the visible entries (those in a block of `mask_in`, when given) are
// written in ascending entry order, as entries of the bank's planes, to slots `slot_first ..
// slot_first + top_k` of the row's list, the rest `DS_SLOT_EMPTY`.
struct DsSelectParams {
  // BF16 `[rows, score_stride]`.
  uint16_t* scores;
  // `[rows]`.
  const uint32_t* visible;
  // `[rows, mask_words]`, or null to select over everything visible.
  const uint32_t* mask_in;
  // `[rows, mask_words]`, or null.
  uint32_t* mask_out;
  // The bank's block tables, `[*, table_stride]`.
  const int32_t* tables;
  // `[rows]`.
  const uint32_t* table_row;
  // `[rows, slot_stride]`.
  uint32_t* slots;
  uint32_t* status;
  uint32_t rows;
  uint32_t score_stride;
  uint32_t mask_words;
  uint32_t table_stride;
  uint32_t slot_stride;
  // The first global slot: the list's `a_slots`.
  uint32_t slot_first;
  uint32_t top_k;
  // Candidate blocks kept (the newest always is).
  uint32_t top_blocks;
  // Entries per candidate block.
  uint32_t block_entries;
  // Entries per pool block of the bank.
  uint32_t entries_per_block;
};
DS_ABI_STRUCT(DsSelectParams, 104);
DS_ABI_FIELD(DsSelectParams, scores, 0);
DS_ABI_FIELD(DsSelectParams, visible, 8);
DS_ABI_FIELD(DsSelectParams, mask_in, 16);
DS_ABI_FIELD(DsSelectParams, mask_out, 24);
DS_ABI_FIELD(DsSelectParams, tables, 32);
DS_ABI_FIELD(DsSelectParams, table_row, 40);
DS_ABI_FIELD(DsSelectParams, slots, 48);
DS_ABI_FIELD(DsSelectParams, status, 56);
DS_ABI_FIELD(DsSelectParams, rows, 64);
DS_ABI_FIELD(DsSelectParams, score_stride, 68);
DS_ABI_FIELD(DsSelectParams, mask_words, 72);
DS_ABI_FIELD(DsSelectParams, table_stride, 76);
DS_ABI_FIELD(DsSelectParams, slot_stride, 80);
DS_ABI_FIELD(DsSelectParams, slot_first, 84);
DS_ABI_FIELD(DsSelectParams, top_k, 88);
DS_ABI_FIELD(DsSelectParams, top_blocks, 92);
DS_ABI_FIELD(DsSelectParams, block_entries, 96);
DS_ABI_FIELD(DsSelectParams, entries_per_block, 100);

// The window part of the slot lists: slots `0 .. window` of each row. In `DS_ORDER_POSITION` slot
// `i` is position `p + i + 1 - window` (`p` the row's position), empty when that is negative or
// below `first[r]`; in `DS_ORDER_RING` (`drafter::ring_order`) slot `i` is `p - (p - i) mod window`
// for `i < min(window, p + 1)` and empty after. Position `x` is entry `tables[table_row[r] ·
// table_stride + x / block_positions] · block_positions + x % block_positions`; a negative table
// word raises `DS_STATUS_BAD_INDEX`. The words are the same for every layer of a window group.
struct DsWindowSlotsParams {
  // `[rows]`.
  const uint32_t* positions;
  // `[rows]`: the lowest position a row lists.
  const uint32_t* first;
  // The window group's block tables.
  const int32_t* tables;
  // `[rows]`.
  const uint32_t* table_row;
  // `[rows, slot_stride]`.
  uint32_t* slots;
  uint32_t* status;
  uint32_t rows;
  uint32_t window;
  // Positions per pool block.
  uint32_t block_positions;
  uint32_t table_stride;
  uint32_t slot_stride;
  // `DS_ORDER_*`.
  uint32_t order;
};
DS_ABI_STRUCT(DsWindowSlotsParams, 72);
DS_ABI_FIELD(DsWindowSlotsParams, positions, 0);
DS_ABI_FIELD(DsWindowSlotsParams, first, 8);
DS_ABI_FIELD(DsWindowSlotsParams, tables, 16);
DS_ABI_FIELD(DsWindowSlotsParams, table_row, 24);
DS_ABI_FIELD(DsWindowSlotsParams, slots, 32);
DS_ABI_FIELD(DsWindowSlotsParams, status, 40);
DS_ABI_FIELD(DsWindowSlotsParams, rows, 48);
DS_ABI_FIELD(DsWindowSlotsParams, window, 52);
DS_ABI_FIELD(DsWindowSlotsParams, block_positions, 56);
DS_ABI_FIELD(DsWindowSlotsParams, table_stride, 60);
DS_ABI_FIELD(DsWindowSlotsParams, slot_stride, 64);
DS_ABI_FIELD(DsWindowSlotsParams, order, 68);

// `moe::route` after the logits: scores `sqrt(softplus(logit))`, the top `active` of `score + bias`
// under the total order, weights the unbiased scores divided by their sum (added in selection
// order, best first) plus 1e-20 when `normalize`, times `route_scale`. A row's experts are written
// in ascending expert id with their weights.
struct DsRouteParams {
  // `[rows, experts]`.
  const float* logits;
  // `[experts]`.
  const float* bias;
  // Out: `[rows, active]`, ascending.
  uint32_t* ids;
  // Out: `[rows, active]`.
  float* weights;
  uint32_t rows;
  uint32_t experts;
  uint32_t active;
  // `norm_topk_prob` and `active > 1`.
  uint32_t normalize;
  float route_scale;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsRouteParams, 56);
DS_ABI_FIELD(DsRouteParams, logits, 0);
DS_ABI_FIELD(DsRouteParams, bias, 8);
DS_ABI_FIELD(DsRouteParams, ids, 16);
DS_ABI_FIELD(DsRouteParams, weights, 24);
DS_ABI_FIELD(DsRouteParams, rows, 32);
DS_ABI_FIELD(DsRouteParams, experts, 36);
DS_ABI_FIELD(DsRouteParams, active, 40);
DS_ABI_FIELD(DsRouteParams, normalize, 44);
DS_ABI_FIELD(DsRouteParams, route_scale, 48);
DS_ABI_FIELD(DsRouteParams, reserved, 52);

// The expert-major layout of a step's routed pairs: pairs of one expert keep their (row, slot)
// order, expert `g`'s run starts at the previous run's start plus its length rounded up to
// `row_align`, and `group_end[g]` is the end of its pairs.
struct DsPlaceParams {
  // `[rows, active]` (`DsRouteParams::ids`).
  uint32_t* ids;
  // `[rows, active]`.
  float* weights;
  // Out: `[rows, active]`, each pair's layout row.
  uint32_t* row_of;
  // Out: `[layout_rows]`, each placed row's routing weight.
  float* row_weight;
  // Out: `[groups]`.
  uint32_t* group_end;
  // Per-block counts for a multi-block launch, or null.
  uint32_t* scratch;
  uint32_t rows;
  uint32_t active;
  uint32_t groups;
  // Not 0.
  uint32_t row_align;
  // Rows the layout buffers hold.
  uint32_t layout_rows;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsPlaceParams, 72);
DS_ABI_FIELD(DsPlaceParams, ids, 0);
DS_ABI_FIELD(DsPlaceParams, weights, 8);
DS_ABI_FIELD(DsPlaceParams, row_of, 16);
DS_ABI_FIELD(DsPlaceParams, row_weight, 24);
DS_ABI_FIELD(DsPlaceParams, group_end, 32);
DS_ABI_FIELD(DsPlaceParams, scratch, 40);
DS_ABI_FIELD(DsPlaceParams, rows, 48);
DS_ABI_FIELD(DsPlaceParams, active, 52);
DS_ABI_FIELD(DsPlaceParams, groups, 56);
DS_ABI_FIELD(DsPlaceParams, row_align, 60);
DS_ABI_FIELD(DsPlaceParams, layout_rows, 64);
DS_ABI_FIELD(DsPlaceParams, reserved, 68);

// `moe::swiglu_hidden` and the quantiser of the down-projection: per element `bf16(weight ·
// (silu(min(gate, limit)) · clamp(up, ±limit)))` (no weight for the shared expert, no clamp when
// `limit <= 0`), then `quant::act_quant` of the row.
struct DsSwigluParams {
  // BF16, row `r` at `gate + r · in_stride`.
  const uint16_t* gate;
  // BF16, row `r` at `up + r · in_stride`.
  const uint16_t* up;
  // `[rows]` routing weights, or null.
  const float* weight;
  // Out: `[rows, width]` E4M3.
  uint8_t* codes;
  // Out: `[rows, width / 32]` activation scale bytes.
  uint8_t* scales;
  uint32_t rows;
  // A multiple of 32.
  uint32_t width;
  // BF16 elements between input rows.
  uint32_t in_stride;
  float limit;
};
DS_ABI_STRUCT(DsSwigluParams, 56);
DS_ABI_FIELD(DsSwigluParams, gate, 0);
DS_ABI_FIELD(DsSwigluParams, up, 8);
DS_ABI_FIELD(DsSwigluParams, weight, 16);
DS_ABI_FIELD(DsSwigluParams, codes, 24);
DS_ABI_FIELD(DsSwigluParams, scales, 32);
DS_ABI_FIELD(DsSwigluParams, rows, 40);
DS_ABI_FIELD(DsSwigluParams, width, 44);
DS_ABI_FIELD(DsSwigluParams, in_stride, 48);
DS_ABI_FIELD(DsSwigluParams, limit, 52);

// The feed-forward sum (`forward::moe_forward`): from +0, a row's routed outputs added in slot
// order (ascending expert id), then `bf16(sum + shared)`.
struct DsCombineParams {
  // BF16 `[layout_rows, dim]`.
  const uint16_t* routed;
  // BF16 `[rows, dim]`.
  const uint16_t* shared;
  // `[rows, active]`.
  uint32_t* row_of;
  // BF16 `[rows, dim]`.
  uint16_t* out;
  uint32_t rows;
  uint32_t active;
  uint32_t dim;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsCombineParams, 48);
DS_ABI_FIELD(DsCombineParams, routed, 0);
DS_ABI_FIELD(DsCombineParams, shared, 8);
DS_ABI_FIELD(DsCombineParams, row_of, 16);
DS_ABI_FIELD(DsCombineParams, out, 24);
DS_ABI_FIELD(DsCombineParams, rows, 32);
DS_ABI_FIELD(DsCombineParams, active, 36);
DS_ABI_FIELD(DsCombineParams, dim, 40);
DS_ABI_FIELD(DsCombineParams, reserved, 44);

// Engram table rows, copied as stored: row `r`'s `ids_per_row` table rows back to back. An id at or
// past `table_rows` copies zeros and raises `DS_STATUS_BAD_INDEX`.
struct DsEngramGatherParams {
  // `[table_rows, width]` E4M3.
  uint8_t* codes;
  // `[table_rows, width / 32]` E8M0.
  uint8_t* scales;
  // `[rows, ids_per_row]` (`EngramHasher::rows`).
  uint32_t* ids;
  // `[rows, ids_per_row · width]`.
  uint8_t* out_codes;
  // `[rows, ids_per_row · width / 32]`.
  uint8_t* out_scales;
  uint32_t* status;
  uint64_t table_rows;
  uint32_t rows;
  uint32_t ids_per_row;
  // A multiple of 32.
  uint32_t width;
  // 0.
  uint32_t reserved;
};
DS_ABI_STRUCT(DsEngramGatherParams, 72);
DS_ABI_FIELD(DsEngramGatherParams, codes, 0);
DS_ABI_FIELD(DsEngramGatherParams, scales, 8);
DS_ABI_FIELD(DsEngramGatherParams, ids, 16);
DS_ABI_FIELD(DsEngramGatherParams, out_codes, 24);
DS_ABI_FIELD(DsEngramGatherParams, out_scales, 32);
DS_ABI_FIELD(DsEngramGatherParams, status, 40);
DS_ABI_FIELD(DsEngramGatherParams, table_rows, 48);
DS_ABI_FIELD(DsEngramGatherParams, rows, 56);
DS_ABI_FIELD(DsEngramGatherParams, ids_per_row, 60);
DS_ABI_FIELD(DsEngramGatherParams, width, 64);
DS_ABI_FIELD(DsEngramGatherParams, reserved, 68);

// `engram::inject` after its projection: per copy the gate from the signed square root of the
// normalised three-way product (`pinned_sum`), and `bf16(x + gate · value)`.
struct DsEngramInjectParams {
  // BF16 `[rows, copies, dim]`.
  const uint16_t* x;
  // BF16 `[rows, (copies + 1) · dim]`: a key per copy, then the value.
  const uint16_t* kv;
  // `[copies, dim]`.
  const float* gate_weight;
  // BF16 `[rows, copies, dim]`; may be `x`.
  uint16_t* out;
  uint32_t rows;
  uint32_t dim;
  uint32_t copies;
  float norm_eps;
};
DS_ABI_STRUCT(DsEngramInjectParams, 48);
DS_ABI_FIELD(DsEngramInjectParams, x, 0);
DS_ABI_FIELD(DsEngramInjectParams, kv, 8);
DS_ABI_FIELD(DsEngramInjectParams, gate_weight, 16);
DS_ABI_FIELD(DsEngramInjectParams, out, 24);
DS_ABI_FIELD(DsEngramInjectParams, rows, 32);
DS_ABI_FIELD(DsEngramInjectParams, dim, 36);
DS_ABI_FIELD(DsEngramInjectParams, copies, 40);
DS_ABI_FIELD(DsEngramInjectParams, norm_eps, 44);

// `forward::stream_mean`: `bf16((Σ_j x[j]) / copies)`, copies left to right.
struct DsStreamMeanParams {
  // BF16 `[rows, copies, dim]`.
  const uint16_t* x;
  // BF16, row `r` at `out + r · out_stride`.
  uint16_t* out;
  uint32_t rows;
  uint32_t dim;
  uint32_t copies;
  // BF16 elements between output rows.
  uint32_t out_stride;
};
DS_ABI_STRUCT(DsStreamMeanParams, 32);
DS_ABI_FIELD(DsStreamMeanParams, x, 0);
DS_ABI_FIELD(DsStreamMeanParams, out, 8);
DS_ABI_FIELD(DsStreamMeanParams, rows, 16);
DS_ABI_FIELD(DsStreamMeanParams, dim, 20);
DS_ABI_FIELD(DsStreamMeanParams, copies, 24);
DS_ABI_FIELD(DsStreamMeanParams, out_stride, 28);

// The numerics probe (`ds_probe.cu`): item `i` reads `in_words` 64-bit words at
// `in + i · in_words` and writes `out_words` at `out + i · out_words`, through
// one function of `ds_numeric.cuh`. An f32 travels as its 32 bits in the low
// half of a word, an f64 as its 64, a code or a count as its value: the
// record layout of the host vectors (`tests/data/ds_numeric.vectors`), whose
// `v` and `s` cases a device test replays through this kernel.
struct DsProbeParams {
  const uint64_t* in;
  uint64_t* out;
  uint32_t items;
  // `DS_PROBE_*`.
  uint32_t op;
  // Words per item; the op fixes them except where its comment gives a rule.
  uint32_t in_words;
  uint32_t out_words;
};
DS_ABI_STRUCT(DsProbeParams, 32);
DS_ABI_FIELD(DsProbeParams, in, 0);
DS_ABI_FIELD(DsProbeParams, out, 8);
DS_ABI_FIELD(DsProbeParams, items, 16);
DS_ABI_FIELD(DsProbeParams, op, 20);
DS_ABI_FIELD(DsProbeParams, in_words, 24);
DS_ABI_FIELD(DsProbeParams, out_words, 28);

// Probe ops, named after the vector file's functions. 1 → 1 unless noted.
constexpr uint32_t DS_PROBE_BF16_ROUND = 0;
constexpr uint32_t DS_PROBE_ROUND_E4M3 = 1;
constexpr uint32_t DS_PROBE_ROUND_E2M1 = 2;
constexpr uint32_t DS_PROBE_EXP32 = 3;
constexpr uint32_t DS_PROBE_SIGMOID32 = 4;
constexpr uint32_t DS_PROBE_SILU32 = 5;
constexpr uint32_t DS_PROBE_SOFTPLUS32 = 6;  // a NaN stays a NaN
constexpr uint32_t DS_PROBE_RSQRT32 = 7;
constexpr uint32_t DS_PROBE_ROUND_SCALE = 8;   // amax, inv → scale
constexpr uint32_t DS_PROBE_DET_EXP = 9;
constexpr uint32_t DS_PROBE_DET_LN = 10;
constexpr uint32_t DS_PROBE_DET_SIN_COS = 11;  // x → sin, cos
constexpr uint32_t DS_PROBE_POW32 = 12;        // base (f64), exponent → power
constexpr uint32_t DS_PROBE_ACT_QUANT = 13;    // 32 values → 32 code values, scale
constexpr uint32_t DS_PROBE_FQ8 = 14;          // 32 → 32
constexpr uint32_t DS_PROBE_FQ4M = 15;         // 32 → 32
constexpr uint32_t DS_PROBE_FQ4N = 16;         // 16 → 16
constexpr uint32_t DS_PROBE_FP8_DOT = 17;      // 32 codes, 32 codes → dot
constexpr uint32_t DS_PROBE_FP4_DOT = 18;      // 32 codes, 16 packed bytes → dot
constexpr uint32_t DS_PROBE_DOT = 19;          // a[n], b[n] → dot; n = in_words / 2, at most 128
constexpr uint32_t DS_PROBE_PINNED_SUM = 20;   // x[n] → sum; n = in_words, at most 128
constexpr uint32_t DS_PROBE_ROTATE = 21;       // a, b, cos, sin → a', b'
constexpr uint32_t DS_PROBE_UNROTATE = 22;
constexpr uint32_t DS_PROBE_CIS = 23;          // position, frequency → cos, sin
constexpr uint32_t DS_PROBE_SWIGLU = 24;       // gate, up, limit, weighted, weight → hidden
constexpr uint32_t DS_PROBE_TOPK_KEY = 25;     // score → rank key
