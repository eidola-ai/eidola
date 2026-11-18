// Parameter structs of the Qwen3.8 kernels: the by-value argument of each
// kernel family, mirrored field for field by `eidola-engine-cuda`'s
// `src/qwen/abi.rs`. Sizes, alignments and offsets are asserted here and there.
// A kernel's launch-contract record carries `sizeof` its struct.
//
// Conventions, for every struct:
//
// - Pointers are device addresses; arrays are dense and row-major, a row's
//   stride given in elements. A BF16 array holds the 16 stored bits
//   (`ds::bf16_store`); every value the reference rounds to BF16 is kept so
//   between kernels. f32 arrays hold what the reference keeps in f32.
// - A row is one position of a launch. A kernel computes rows `0 .. rows`,
//   each from that row's inputs alone, on a grid that is a function of the
//   row count and the widths only.
// - FP8 activations are E4M3 codes `[rows, k]` with one f32 scale per row and
//   128 columns, stored `[k / 128][m_pad]` (the CUTLASS blockwise kernel's
//   SFA layout: rows contiguous). FP8 weights are codes `[n, k]` with one f32
//   scale per 128 × 128 block, stored `[k / 128][n / 128]` (SFB: output
//   blocks contiguous).
// - Indices that are device data (tokens) are bounded by the kernel where a
//   struct says it raises a status bit; every other count, stride and
//   position is the host's to check.
#pragma once

#include <cstddef>
#include <cstdint>

#define QW_ABI_STRUCT(T, size) \
  static_assert(sizeof(T) == (size) && alignof(T) == 8, #T " layout is mirrored on the host")
#define QW_ABI_FIELD(T, field, offset) \
  static_assert(offsetof(T, field) == (offset), #T "::" #field " offset is mirrored on the host")

// A token id at or past the vocabulary (the row reads as zeros).
constexpr uint32_t QW_STATUS_BAD_TOKEN = 0x400u;

// The quantisation block of the FP8 linears.
constexpr uint32_t QW_BLOCK = 128;

// Embedding: `out[r] = table[tokens[r]]`.
struct QwEmbedParams {
  const uint32_t* tokens;  // [rows]
  const uint16_t* table;   // [vocab, width] BF16
  uint16_t* out;           // [rows, out_stride] BF16
  uint32_t* status;        // QW_STATUS_BAD_TOKEN
  uint32_t rows;
  uint32_t width;
  uint32_t vocab;
  uint32_t out_stride;
};
QW_ABI_STRUCT(QwEmbedParams, 48);
QW_ABI_FIELD(QwEmbedParams, status, 24);
QW_ABI_FIELD(QwEmbedParams, rows, 32);
QW_ABI_FIELD(QwEmbedParams, out_stride, 44);

// The zero-centred RMS norm (`forward::rms_norm_zc`) over `units` units of
// `n` elements per row: `out = bf16((x · r) · w)`, `r = rsqrt(dot(x, x) / n +
// eps)` with the dot in the pinned order. With `add`, first `x ← bf16(x +
// add)` in place (`forward::residual_add`; one unit per row). With `rotary`,
// the first `rotary` elements of every unit are then rotated at the row's
// position (`attention::apply_rope`: `bf16(bf16(x·cos) + bf16(rot(x)·sin))`,
// pairs `(j, j + rotary / 2)`) from the host's BF16-valued tables
// `[positions][rotary]`.
struct QwNormParams {
  uint16_t* x;                 // [rows, x_stride]; unit u at u · x_unit_stride
  const uint16_t* add;         // [rows, add_stride] or null
  const float* weight;         // [n]: `1 + w`
  uint16_t* out;               // [rows, out_stride]; unit u at u · out_unit_stride
  const float* rope_cos;       // [positions, rotary] or null
  const float* rope_sin;       // [positions, rotary] or null
  const uint32_t* positions;   // [rows] when rotary > 0
  uint32_t rows;
  uint32_t units;
  uint32_t n;
  uint32_t x_stride;
  uint32_t x_unit_stride;
  uint32_t add_stride;
  uint32_t out_stride;
  uint32_t out_unit_stride;
  uint32_t rotary;
  float eps;
};
QW_ABI_STRUCT(QwNormParams, 96);
QW_ABI_FIELD(QwNormParams, positions, 48);
QW_ABI_FIELD(QwNormParams, rows, 56);
QW_ABI_FIELD(QwNormParams, rotary, 88);
QW_ABI_FIELD(QwNormParams, eps, 92);

// The FP8 activation quantiser (`linear::act_quant`): per row and 128
// columns, `s = amax / 448` and `code = e4m3(x / s)`.
struct QwQuantParams {
  const uint16_t* x;  // [rows, x_stride] BF16
  uint8_t* codes;     // [rows, k]
  float* scales;      // [k / 128, m_pad]
  uint32_t rows;
  uint32_t k;
  uint32_t x_stride;
  uint32_t m_pad;
};
QW_ABI_STRUCT(QwQuantParams, 40);
QW_ABI_FIELD(QwQuantParams, rows, 24);
QW_ABI_FIELD(QwQuantParams, m_pad, 36);

// The FP8 linear (`linear::lin8_rows` after its quantiser): per output, the
// 128-blocks ascending, `acc + (dot · s_a) · s_w` with `dot` the block's exact
// dot product rounded once to f32, then BF16.
struct QwLin8Params {
  const uint8_t* a;        // [rows, k]
  const float* a_scales;   // [k / 128, m_pad]
  const uint8_t* w;        // [n, k]
  const float* w_scales;   // [k / 128, n / 128]
  uint16_t* out;           // [rows, out_stride] BF16
  uint32_t rows;
  uint32_t n;
  uint32_t k;
  uint32_t m_pad;
  uint32_t out_stride;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwLin8Params, 64);
QW_ABI_FIELD(QwLin8Params, rows, 40);
QW_ABI_FIELD(QwLin8Params, out_stride, 56);

// A BF16 linear (`linear::linear_bf16_rows`): per output, the f32 dot of the
// row with the weight row in the pinned order (`tensor::dot`), rounded to
// BF16; stored as BF16 bits, or with `out_f32` as the f32 of that BF16 value
// (logits).
struct QwLinearParams {
  const uint16_t* x;  // [rows, x_stride] BF16
  const uint16_t* w;  // [n, k] BF16
  void* out;          // [rows, out_stride]
  uint32_t rows;
  uint32_t n;
  uint32_t k;
  uint32_t x_stride;
  uint32_t out_stride;
  uint32_t out_f32;
};
QW_ABI_STRUCT(QwLinearParams, 48);
QW_ABI_FIELD(QwLinearParams, rows, 24);
QW_ABI_FIELD(QwLinearParams, out_f32, 44);

// Element-wise: the MLP's `bf16(bf16(silu(g)) · u)` (`kind` 0), or the
// attention's output gate `bf16(a · bf16(σ(g)))` (`kind` 1). `g` is read with
// its own row stride and, per unit of `unit` elements, `g_unit_stride` (the
// gate halves of the interleaved query projection).
struct QwGateParams {
  const uint16_t* a;  // [rows, a_stride]: `u` or the attention output
  const uint16_t* g;  // [rows, g_stride]
  uint16_t* out;      // [rows, out_stride]
  uint32_t rows;
  uint32_t width;
  uint32_t a_stride;
  uint32_t g_stride;
  uint32_t out_stride;
  uint32_t unit;
  uint32_t g_unit_stride;
  uint32_t kind;
};
QW_ABI_STRUCT(QwGateParams, 56);
QW_ABI_FIELD(QwGateParams, rows, 24);
QW_ABI_FIELD(QwGateParams, kind, 52);

constexpr uint32_t QW_GATE_SWIGLU = 0;
constexpr uint32_t QW_GATE_SIGMOID = 1;

// Paged KV write: row `r`'s keys and values into entry `targets[r]` (block ·
// block_size + offset; `QW_NO_TARGET` writes nothing). A block holds, per
// layer, keys `[block_size, width]` then values `[block_size, width]`, BF16.
struct QwKvWriteParams {
  const uint16_t* k;         // [rows, k_stride]
  const uint16_t* v;         // [rows, v_stride]
  uint16_t* pool;
  const uint32_t* targets;   // [rows]
  uint32_t rows;
  uint32_t width;            // kv heads · head dim
  uint32_t k_stride;
  uint32_t v_stride;
  uint32_t layer;
  uint32_t layers;
  uint32_t block_size;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwKvWriteParams, 64);
QW_ABI_FIELD(QwKvWriteParams, rows, 32);
QW_ABI_FIELD(QwKvWriteParams, reserved, 60);

constexpr uint32_t QW_NO_TARGET = 0xffffffffu;

// The exact attention core (`attention::attend`) of every (row, query head)
// over keys `0 ..= positions[r] - key_first` of the row's sequence, read
// through its block table: scores are pinned-order dots times `scale`; minus
// their maximum, `exp`; the normaliser summed left to right; output
// `Σ (w / Σw) · v` in ascending position, rounded once to BF16. Query head `h`
// reads KV head `h / group`. `scores` is scratch of `rows · q_heads ·
// score_stride` f32.
struct QwAttnParams {
  const uint16_t* q;         // [rows, q_stride], head h at h · q_head_stride
  const uint16_t* pool;      // as QwKvWriteParams
  const int32_t* tables;     // [rows, table_stride]: block of each logical block
  const uint32_t* positions; // [rows]: the query's position
  uint16_t* out;             // [rows, out_stride], head h at h · head_dim
  float* scores;
  uint32_t rows;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t q_stride;
  uint32_t q_head_stride;
  uint32_t out_stride;
  uint32_t table_stride;
  uint32_t layer;
  uint32_t layers;
  uint32_t block_size;
  uint32_t score_stride;
  uint32_t key_first;        // the first position with KV (the drafter's is 1)
  float scale;
};
QW_ABI_STRUCT(QwAttnParams, 104);
QW_ABI_FIELD(QwAttnParams, rows, 48);
QW_ABI_FIELD(QwAttnParams, scale, 100);

// DeltaNet, per position (`deltanet::conv_step`, `gates`, `l2_normalise`,
// `scale_query`): the depthwise convolution of the position's window of four
// raw rows `conv_in[window[p] .. window[p] + 4]` (oldest first), SiLU, then
// the split into queries, keys and values, the L2 norms of queries and keys,
// the query scale, and each value head's `β` and `α` from `ab[p]` (`a` then
// `b`, value heads each).
struct QwDnPrepParams {
  const float* conv_in;     // [*, conv_dim]: raw `mixed` rows (BF16 values)
  const uint32_t* window;   // [positions]
  const float* ab;          // [positions, 2 · v_heads]
  const float* conv_w;      // [conv_dim, 4]
  const float* exp_a_log;   // [v_heads]: exp32(A_log)
  const float* dt_bias;     // [v_heads]
  float* q;                 // [positions, k_heads · k_dim]
  float* k;                 // [positions, k_heads · k_dim]
  float* v;                 // [positions, v_heads · v_dim]
  float* beta;              // [positions, v_heads]
  float* alpha;             // [positions, v_heads]
  uint32_t positions;
  uint32_t k_heads;
  uint32_t v_heads;
  uint32_t k_dim;
  uint32_t v_dim;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwDnPrepParams, 112);
QW_ABI_FIELD(QwDnPrepParams, positions, 88);
QW_ABI_FIELD(QwDnPrepParams, reserved, 108);

// One sequence's run through a DeltaNet layer's recurrence: positions
// `first .. first + count` of the prep arrays, in order, from the state at
// `src` (null: the zero state). Positions before `out_from` (relative) are
// folded with no output. After `live_after` positions (relative; `count + 1`
// for never) the state is stored to `live`; after `write_after[i]` positions,
// to `write_dst[i]`. The state of a layer is `[v_heads][k_dim][v_dim]` f32,
// key dimension major (`DeltaNetState::recurrent`), at the address given plus
// the layer's offset, which the launch's `state_offset` adds to every
// address.
struct QwDnSeq {
  uint64_t src;
  uint64_t live;
  uint64_t write_dst[4];
  uint32_t first;
  uint32_t count;
  uint32_t out_from;
  uint32_t live_after;
  uint32_t write_after[4];
  uint32_t writes;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwDnSeq, 88);
QW_ABI_FIELD(QwDnSeq, first, 48);
QW_ABI_FIELD(QwDnSeq, writes, 80);

// `deltanet::recurrent_step` per value head and token, every sequence of
// `seqs`; the output `o` (rounded to BF16, as f32) of every position at or
// past its sequence's `out_from`.
struct QwDnRecParams {
  const QwDnSeq* seqs;
  const float* q;           // as QwDnPrepParams
  const float* k;
  const float* v;
  const float* beta;
  const float* alpha;
  float* out;               // [positions, v_heads · v_dim]
  uint64_t state_offset;    // bytes added to every state address
  uint32_t seq_count;
  uint32_t k_heads;
  uint32_t v_heads;
  uint32_t k_dim;
  uint32_t v_dim;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwDnRecParams, 88);
QW_ABI_FIELD(QwDnRecParams, state_offset, 56);
QW_ABI_FIELD(QwDnRecParams, seq_count, 64);

// The DeltaNet's gated RMS norm per value head (`deltanet::gated_norm`):
// `bf16(bf16(w · bf16(o · r)) · silu(z))`, `r = rsqrt(Σo² / v_dim + eps)`.
struct QwDnNormParams {
  const float* o;      // [rows, v_heads · v_dim], BF16 values
  const uint16_t* z;   // [rows, z_stride]
  const float* weight; // [v_dim], plain
  uint16_t* out;       // [rows, v_heads · v_dim]
  uint32_t rows;
  uint32_t v_heads;
  uint32_t v_dim;
  uint32_t z_stride;
  float eps;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwDnNormParams, 56);
QW_ABI_FIELD(QwDnNormParams, rows, 32);
QW_ABI_FIELD(QwDnNormParams, eps, 48);

// BF16 → f32, element-wise (BF16 values into the DeltaNet's f32 staging).
struct QwWidenParams {
  const uint16_t* x;  // [rows, x_stride]
  float* out;         // [rows, out_stride]
  uint32_t rows;
  uint32_t width;
  uint32_t x_stride;
  uint32_t out_stride;
};
QW_ABI_STRUCT(QwWidenParams, 32);
QW_ABI_FIELD(QwWidenParams, out_stride, 28);

// f32 → BF16, element-wise: the fast profile's f32 GEMM outputs rounded to the
// reference's BF16 (`bf16_round`), as BF16 bits, or with `out_f32` as the f32
// of that BF16 value (logits).
struct QwRoundParams {
  const float* x;  // [rows, x_stride]
  void* out;       // [rows, out_stride]
  uint32_t rows;
  uint32_t width;
  uint32_t x_stride;
  uint32_t out_stride;
  uint32_t out_f32;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwRoundParams, 40);
QW_ABI_FIELD(QwRoundParams, out_f32, 32);

// The fast profile's split FP8 GEMM, its partial sums added
// (`eidola_qwen_fast_split_sum`): `out[r][n] = bf16(((p[0][r][n] + p[1][r][n])
// + …) + p[splits - 1][r][n])`, the splits in order from the first, each
// split's array `part_rows` rows of `n` f32 values, rounded once to the
// nearest BF16 (ties to even; a NaN as the one quiet NaN).
struct QwSplitSumParams {
  const float* parts;  // [splits][part_rows][n]
  uint16_t* out;       // [rows, out_stride] BF16
  uint32_t rows;
  uint32_t n;
  uint32_t splits;
  uint32_t part_rows;
  uint32_t out_stride;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwSplitSumParams, 40);
QW_ABI_FIELD(QwSplitSumParams, splits, 24);
QW_ABI_FIELD(QwSplitSumParams, out_stride, 32);

// Row copies of 32-bit words (`eidola_qwen_copy_rows`): up to
// `QW_COPY_GROUPS` groups in one launch, each a table of up to
// `QW_COPY_BUFFERS` buffers of `width`-word rows and `items` copies (source
// buffer and row, destination buffer and row: device data). An item whose
// buffer or row is out of range copies nothing and raises
// `QW_STATUS_BAD_INDEX`; an item copying a row onto itself (the padding of a
// rung's lists) copies nothing.
constexpr uint32_t QW_COPY_GROUPS = 3;
constexpr uint32_t QW_COPY_BUFFERS = 8;
constexpr uint32_t QW_STATUS_BAD_INDEX = 4u;

struct QwCopyGroup {
  uint32_t* buf[QW_COPY_BUFFERS];
  uint64_t rows[QW_COPY_BUFFERS];  // rows each buffer holds
  const uint32_t* src_buf;         // [items]
  const uint32_t* src_row;
  const uint32_t* dst_buf;
  const uint32_t* dst_row;
  uint32_t width;  // words a row
  uint32_t items;
};
QW_ABI_STRUCT(QwCopyGroup, 168);
QW_ABI_FIELD(QwCopyGroup, src_buf, 128);
QW_ABI_FIELD(QwCopyGroup, width, 160);

struct QwCopyParams {
  QwCopyGroup group[QW_COPY_GROUPS];
  uint32_t* status;
  uint32_t groups;
  uint32_t reserved;
};
QW_ABI_STRUCT(QwCopyParams, 520);
QW_ABI_FIELD(QwCopyParams, status, 504);
QW_ABI_FIELD(QwCopyParams, groups, 512);
