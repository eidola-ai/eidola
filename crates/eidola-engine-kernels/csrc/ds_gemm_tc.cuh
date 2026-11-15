// The DeepSeek-V4.1 quantised GEMMs on tensor cores (`ds_gemm_tc.cu`): the
// pieces of a thread's work that are plain functions of its place in the
// block, so the host check (`oracle/ds_gemm_tc_host.cc`) runs the same index
// arithmetic, band splits and block arithmetic the device runs, around an
// emulated MMA.
//
// The products are `lin8` and `lin4` of `DsLin8Params` / `DsLin4Params`
// (`ds_abi.cuh`): per row and output, for 32-wide blocks ascending,
// `acc = acc + (dot · activation scale) · weight scale` in f32, `dot` the
// block's 32-term dot product rounded once, then BF16. A block's dot product
// runs on the tensor cores as a warp-level MMA with an f32 result from a zero
// accumulator. The operands are swapped, as the A operand has 16 rows and the
// B operand 8: A is 16 weight rows (outputs), B is 8 activation rows
// (tokens), both K-contiguous as stored, so a lane reads whole words of one
// row's codes.
//
// Two forms of the block product:
//
// - **Banded (exact).** The operands' codes are split by exponent field into
//   bands (a code outside a band is replaced by +0 there), and each pair of
//   bands is one block product on the tensor cores. Within a pair every product is an integer multiple of
//   one power of two and the 32 products' magnitudes sum below 2^24 of it, so
//   every partial sum of any order of accumulation is an f32 value: an
//   adder that is exact on exactly representable sums returns the pair's
//   exact sum, whatever its order or alignment. The pairs' sums are
//   converted to integers in the reference's unit (2^-18 for FP8 × FP8,
//   2^-10 for FP8 × FP4), added exactly, and rounded once to f32: the
//   reference's `fp8_fp8_block_dot` / `fp8_fp4_block_dot`, bit for bit. The
//   one hardware fact this rests on is that the tensor cores return an
//   exactly representable sum exactly; the probe (`eidola_ds_gemm_probe`) measures
//   it.
// - **Tolerance.** One block product of the whole codes: the hardware's
//   32-term dot, rounded however the hardware rounds it. Equal to the
//   reference when that is the correctly rounded sum; within the measured
//   tolerance otherwise.
//
// Everything after the block product is the reference's: the accumulation
// in ascending block order with `ds::gemm_accumulate`, BF16 by `bf16_round`
// and `bf16_store`. A row's arithmetic is a fixed sequence of instructions
// over its own codes, its scales and the weights, whichever tile, column or
// launch shape carries it (the probe also checks that an MMA's result for
// one element does not depend on its place in the fragment).
#pragma once

#include "ds_abi.cuh"
#include "ds_numeric.cuh"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define DSTC_FN __host__ __device__ __forceinline__
#else
#define DSTC_FN inline
#endif

namespace dstc {

// ---- launch geometry (mirrored by `eidola-engine-cuda`'s `ds::gemm`) ----

constexpr uint32_t kThreads = 128;
constexpr uint32_t kWarps = kThreads / 32;
// Padding after each staged row: a fragment load of 8 rows × 4 lanes then
// touches 32 distinct banks.
constexpr uint32_t kPad = 16;

// A block's shape: each warp computes `kMTiles` 16-output MMA tiles by
// `kNTiles` 8-row tiles; `kFeatureWarps` warps lie side by side over the
// outputs, and the `kWarps / kFeatureWarps` warps over the same outputs share
// out each step's blocks. A step stages `kStepBlocks` 32-wide blocks, in
// `kStages` stages.
template <bool kFp4_, uint32_t kMTiles_, uint32_t kNTiles_, uint32_t kFeatureWarps_,
          uint32_t kStepBlocks_, uint32_t kStages_>
struct Cfg {
  static constexpr bool kFp4 = kFp4_;
  static constexpr uint32_t kMTiles = kMTiles_;
  static constexpr uint32_t kNTiles = kNTiles_;
  static constexpr uint32_t kFeatureWarps = kFeatureWarps_;
  static constexpr uint32_t kStepBlocks = kStepBlocks_;
  static constexpr uint32_t kStages = kStages_;
  static constexpr uint32_t kKWarps = kWarps / kFeatureWarps;
  static constexpr uint32_t kTokens = 8 * kNTiles;
  static constexpr uint32_t kWarpFeatures = 16 * kMTiles;
  static constexpr uint32_t kFeatures = kFeatureWarps * kWarpFeatures;
  // Bytes of one row in a step: activation codes, and weight codes (FP8) or
  // packed FP4.
  static constexpr uint32_t kXBytes = 32 * kStepBlocks;
  static constexpr uint32_t kWBytes = kFp4 ? kXBytes / 2 : kXBytes;
  static constexpr uint32_t kWRow = kWBytes + kPad;
  static constexpr uint32_t kXRow = kXBytes + kPad;
  static_assert(kWarps % kFeatureWarps == 0 && kStepBlocks % (kWarps / kFeatureWarps) == 0);
  static_assert(kStepBlocks % 4 == 0, "scales are staged in 4-block words");
};

// The tiles: four warps side by side over the outputs, each computing every
// block of a step; steps of 4 blocks, double-buffered. 8 rows by 64 outputs
// (`_t8`) or 32 rows by 128 (`_t32`).
template <bool kFp4, uint32_t kTokens>
using TileCfg = Cfg<kFp4, kTokens == 8 ? 1 : 2, kTokens / 8, kWarps, 4, 2>;
// The split block (`_s8`): 8 rows by 16 outputs. Each warp computes a
// quarter of each step's 16 blocks, and the blocks' contributions are then
// added in ascending block order, one thread per output and row. Many small
// blocks with deep staging, for row counts whose GEMM is a stream of the
// weights.
template <bool kFp4>
using SplitCfg = Cfg<kFp4, 1, 1, 1, 16, kFp4 ? 8 : 6>;

// One staged step: weight rows, activation rows, and the scale bytes of the
// step's blocks (a weight scale per output row, expanded from the 32 × 32
// tile for FP8 weights).
template <class C>
struct alignas(16) Stage {
  uint8_t w[C::kFeatures][C::kWRow];
  uint8_t x[C::kTokens][C::kXRow];
  uint8_t ws[C::kFeatures][C::kStepBlocks];
  uint8_t xs[C::kTokens][C::kStepBlocks];
};

// The split block's contributions of one step, `(dot · activation scale) ·
// weight scale` per block, output and row, before they are added in order.
template <class C>
struct Contributions {
  float c[C::kStepBlocks][C::kFeatures][C::kTokens];
};

// Dynamic shared memory of the split block: its stages, then its contributions.
template <class C>
constexpr uint32_t split_smem_bytes() {
  return static_cast<uint32_t>(C::kStages * sizeof(Stage<C>) + sizeof(Contributions<C>));
}

// What one pass over K computes: the rows `[lo, hi)` of a tile (relative to
// its first row) against one weight. `lin8` is one pass a tile; `lin4` one
// pass per group whose run meets the tile.
struct Pass {
  const uint8_t* x;   // the tile's first row of activation codes
  const uint8_t* xs;  // its first row of scale bytes
  const uint8_t* w;   // the weight's first row (codes, or packed FP4)
  const uint8_t* ws;  // its scales: `[ceil(n / 32), k / 32]` (FP8) or `[n, k / 32]` (FP4)
  uint16_t* out;      // the tile's first output row
  uint32_t lo, hi;
  uint32_t n, k;
  uint32_t f0;  // the block's first output
};

template <class C>
DSTC_FN uint32_t steps(const Pass& p) {
  return (p.k / 32 + C::kStepBlocks - 1) / C::kStepBlocks;
}
template <class C>
DSTC_FN uint32_t blocks_in_step(const Pass& p, uint32_t step) {
  const uint32_t kb = p.k / 32, first = step * C::kStepBlocks;
  return kb - first < C::kStepBlocks ? kb - first : C::kStepBlocks;
}

// ---- staging: the copies of one step, enumerated per thread ----

// Copy `i` of the step's 16-byte weight copies (`i < kFeatures · kWBytes /
// 16`): its staged row and offset, and its source, or `nullptr` for a
// zero-filled piece (an output past `n`, a block past `k`).
template <class C>
DSTC_FN const uint8_t* w_copy(const Pass& p, uint32_t step, uint32_t i, uint32_t* row,
                              uint32_t* off) {
  constexpr uint32_t segs = C::kWBytes / 16;
  *row = i / segs;
  *off = (i % segs) * 16;
  const uint32_t f = p.f0 + *row;
  const uint32_t valid = blocks_in_step<C>(p, step) * (C::kFp4 ? 16u : 32u);
  if (f >= p.n || *off >= valid) return nullptr;
  const uint64_t row_bytes = C::kFp4 ? p.k / 2 : p.k;
  return p.w + f * row_bytes + static_cast<uint64_t>(step) * C::kWBytes + *off;
}

// Copy `i` of the step's 16-byte activation copies (`i < kTokens · kXBytes / 16`).
template <class C>
DSTC_FN const uint8_t* x_copy(const Pass& p, uint32_t step, uint32_t i, uint32_t* row,
                              uint32_t* off) {
  constexpr uint32_t segs = C::kXBytes / 16;
  *row = i / segs;
  *off = (i % segs) * 16;
  const uint32_t valid = blocks_in_step<C>(p, step) * 32u;
  if (*row < p.lo || *row >= p.hi || *off >= valid) return nullptr;
  return p.x + static_cast<uint64_t>(*row) * p.k + static_cast<uint64_t>(step) * C::kXBytes + *off;
}

// Whether a pass's scale bytes can be staged by 4-byte asynchronous copies:
// every row's scale bytes start on a multiple of 4 (`k / 32` a multiple of 4
// and both arrays 4-byte aligned), so a step's 4-block groups are whole
// words, each wholly inside `k` or wholly past it. Otherwise they are staged
// byte by byte (`stage_scale`), which waits for each load.
DSTC_FN bool scales_by_words(const Pass& p) {
  return (p.k / 32) % 4 == 0 && reinterpret_cast<uintptr_t>(p.ws) % 4 == 0 &&
         reinterpret_cast<uintptr_t>(p.xs) % 4 == 0;
}

// Copy `i` of the step's 4-byte scale copies (`i < (kFeatures + kTokens) ·
// kStepBlocks / 4`, weight scales first) when `scales_by_words`: where it
// lands, and its source, or `nullptr` for a zero-filled word (an output or a
// row outside the pass, blocks past `k`: nothing reads those scales).
template <class C>
DSTC_FN const uint8_t* scale_copy(const Pass& p, uint32_t step, uint32_t i, Stage<C>& s,
                                  uint8_t** dst) {
  constexpr uint32_t groups = C::kStepBlocks / 4;
  const uint32_t kb = p.k / 32, first = step * C::kStepBlocks + (i % groups) * 4;
  if (i < C::kFeatures * groups) {
    const uint32_t r = i / groups, f = p.f0 + r;
    *dst = &s.ws[r][(i % groups) * 4];
    if (f >= p.n || first >= kb) return nullptr;
    return p.ws + (C::kFp4 ? static_cast<uint64_t>(f) * kb : static_cast<uint64_t>(f / 32) * kb) + first;
  }
  const uint32_t r = (i - C::kFeatures * groups) / groups;
  *dst = &s.xs[r][(i % groups) * 4];
  if (r < p.lo || r >= p.hi || first >= kb) return nullptr;
  return p.xs + static_cast<uint64_t>(r) * kb + first;
}

// The scale bytes of the step, item `i < (kFeatures + kTokens) · kStepBlocks`:
// weight scales first. A scale nothing reads (an output or a row outside the
// pass, a block past `k`) is staged as 127.
template <class C>
DSTC_FN void stage_scale(const Pass& p, uint32_t step, uint32_t i, Stage<C>& s) {
  const uint32_t kb = p.k / 32, vb = blocks_in_step<C>(p, step);
  const uint32_t b = i % C::kStepBlocks;
  const uint32_t blk = step * C::kStepBlocks + b;
  if (i < C::kFeatures * C::kStepBlocks) {
    const uint32_t r = i / C::kStepBlocks, f = p.f0 + r;
    uint8_t v = 127;
    if (f < p.n && b < vb) {
      const uint64_t at = C::kFp4 ? static_cast<uint64_t>(f) * kb + blk
                                  : static_cast<uint64_t>(f / 32) * kb + blk;
      v = p.ws[at];
    }
    s.ws[r][b] = v;
  } else {
    const uint32_t r = (i - C::kFeatures * C::kStepBlocks) / C::kStepBlocks;
    s.xs[r][b] = (r >= p.lo && r < p.hi && b < vb) ? p.xs[static_cast<uint64_t>(r) * kb + blk] : 127;
  }
}

// ---- operands in half precision ----
//
// `sm_100a` and `sm_103a` have no warp-level FP8 MMA: an E4M3 `mma.sync`
// compiles to conversions to F16 and two `HMMA.16816.F32`. The kernels do
// that themselves, so a code is converted once and the bands are cut on the
// F16 values: a block's product is two chained `mma.m16n8k16` with F16
// operands and an f32 accumulator, the second taking the first's result.
// Every E4M3 and every E2M1 value is an F16 value, so the conversions are
// exact.

// Two E4M3 codes (bits 0-7, 8-15) as two F16 values (bits 0-15, 16-31).
DSTC_FN uint32_t e4m3x2_to_f16x2(uint32_t v) {
#if defined(__CUDA_ARCH__)
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(static_cast<uint16_t>(v)));
  return r;
#else
  uint32_t r = 0;
  for (uint32_t j = 0; j < 2; ++j) {
    const uint32_t c = (v >> (8 * j)) & 0xffu;
    const uint32_t sign = (c & 0x80u) << 8, e = (c >> 3) & 0x0fu, m = c & 7u;
    uint32_t h;
    if (e != 0) {
      h = (e + 8u) << 10 | m << 7;
    } else if (m == 0) {
      h = 0;
    } else {
      // m · 2^-9 with m in 1..7: normalise the leading bit.
      uint32_t lead = 0;
      while ((m >> (lead + 1)) != 0) ++lead;
      h = (lead + 6u) << 10 | ((m << (10 - lead)) & 0x3ffu);
    }
    r |= (sign | h) << (16 * j);
  }
  return r;
#endif
}

DSTC_FN uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
#if defined(__CUDA_ARCH__)
  return __byte_perm(a, b, sel);
#else
  const uint64_t pool = static_cast<uint64_t>(b) << 32 | a;
  uint32_t r = 0;
  for (uint32_t j = 0; j < 4; ++j) {
    r |= static_cast<uint32_t>((pool >> (8 * ((sel >> (4 * j)) & 7u))) & 0xffu) << (8 * j);
  }
  return r;
#endif
}

// Four E2M1 codes (bits 0-3, 4-7, 8-11, 12-15, in element order: the
// little-endian 16 bits of two packed bytes) as F16: elements 0, 1 in `lo`,
// 2, 3 in `hi`. The magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 are the F16 values
// whose low byte is zero and whose high byte is 0x00, 0x38, 0x3c, 0x3e,
// 0x40, 0x42, 0x44, 0x46; `prmt` picks each high byte by the code (selector
// nibbles with bit 3 clear copy bytes, replicating no sign) and a zero byte
// (pool byte 0, the code 0's) for each low byte; the sign moves to bit 15.
DSTC_FN void e2m1x4_to_f16(uint32_t v, uint32_t* lo, uint32_t* hi) {
  constexpr uint32_t kPoolLo = 0x3e3c3800u, kPoolHi = 0x46444240u;
  *lo = byte_perm(kPoolLo, kPoolHi, (v << 4 & 0x0070u) | (v << 8 & 0x7000u)) |
        (v & 0x0008u) << 12 | (v & 0x0080u) << 24;
  *hi = byte_perm(kPoolLo, kPoolHi, (v >> 4 & 0x0070u) | (v & 0x7000u)) |
        (v & 0x0800u) << 4 | (v & 0x8000u) << 16;
}

// ---- bands ----

// A mask of the halves of `x` (two F16 values) whose exponent field is at
// least `field`. Lane-wise without carries: a magnitude is at most 0x7fff and
// `0x8000 - (field << 10)` at most 0x8000, so their sum stays within a 16-bit
// lane and its bit 15 is the comparison.
DSTC_FN uint32_t at_least(uint32_t x, uint32_t field) {
  const uint32_t s = ((x & 0x7fff7fffu) + (0x8000u - (field << 10)) * 0x00010001u) & 0x80008000u;
  return (s >> 15) * 0xffffu;
}

// Bands by E4M3 exponent field. Within a band `[lo, hi]` an E4M3 value is
// `m · 2^(max(lo, 1) - 1)` in units of 2^-9 with `m` at most
// `15 · 2^(hi - max(lo, 1))`; the bands below bound each pair's sum of 32
// magnitudes:
//
//   FP8 × FP8: weights {[0, 8], [9, 15]} (m at most 1,920 and 960),
//   activations {[0, 5], [6, 10], [11, 15]} (m at most 240 each):
//   32 · 1,920 · 240 < 2^24.
//   FP8 × FP4: weights unbanded (twice an E2M1 value is at most 12),
//   activations {[0, 8], [9, 15]}: 32 · 1,920 · 12 < 2^20.
//
// They are cut on the F16 values: an E4M3 field `E` of 1 or more is F16 field
// `E + 8`, and the E4M3 subnormals (field 0) are F16 fields 6 to 8 (zero is
// 0), so a band starting at E4M3 field `lo` of 1 or more starts at F16 field
// `lo + 8`. A code outside a band is +0 there.
constexpr uint32_t kW8Bands = 2;
constexpr uint32_t kX8Bands = 3;
constexpr uint32_t kX4Bands = 2;
constexpr uint32_t kW8Cut = 9 + 8;
constexpr uint32_t kX8Cut1 = 6 + 8, kX8Cut2 = 11 + 8;
constexpr uint32_t kX4Cut = 9 + 8;

// `x` cut at F16 field `t`: below it, and from it up.
DSTC_FN void split2(uint32_t x, uint32_t t, uint32_t* lo, uint32_t* hi) {
  const uint32_t m = at_least(x, t);
  *lo = x & ~m;
  *hi = x & m;
}
// `x` cut at F16 fields `t1 < t2`.
DSTC_FN void split3(uint32_t x, uint32_t t1, uint32_t t2, uint32_t* b0, uint32_t* b1, uint32_t* b2) {
  const uint32_t m1 = at_least(x, t1), m2 = at_least(x, t2);
  *b0 = x & ~m1;
  *b1 = x & m1 & ~m2;
  *b2 = x & m2;
}

// A band pair's exact sum (an f32 holding it exactly) as an integer in the
// block dot's unit: 2^-18 for FP8 × FP8 (below 2^41 in all), 2^-10 for
// FP8 × FP4 (below 2^27). Every product is a multiple of the unit, so the
// scaled sum is an integer and the conversion exact.
DSTC_FN int64_t band_units8(float sum) { return static_cast<int64_t>(sum * 262144.0f); }
DSTC_FN int32_t band_units4(float sum) { return static_cast<int32_t>(sum * 1024.0f); }
// The block dot from the pairs' integer sum, rounded once: `ds::fp8_fp8_block_dot`
// rounds `sum · 2^-18` from f64; rounding `sum` and then scaling by a power of
// two is the same value, since the result is at least 2^-18 when not zero.
DSTC_FN float block_dot8(int64_t units) { return static_cast<float>(units) * 3.814697265625e-06f; }
// `ds::fp8_fp4_block_dot`'s own expression.
DSTC_FN float block_dot4(int32_t units) { return static_cast<float>(units) * 0.0009765625f; }

// ---- fragments ----
//
// `mma.m16n8k16` with F16 operands, lane `l`, `g = l / 4`, `q = l % 4`
// (CUTLASS's `SM80_16x8x16_F32F16F16F32_TN` traits), slot `s` of K:
//   A (16 × 16): a0 = row g, slots 2q, 2q + 1; a1 = row g + 8, the same;
//     a2 = row g, slots 8 + 2q, 9 + 2q; a3 = row g + 8, the same.
//   B (16 × 8): b0 = column g, slots 2q, 2q + 1; b1 = column g, slots 8 + 2q, 9 + 2q.
//   D (16 × 8, f32): d0, d1 = row g, columns 2q, 2q + 1; d2, d3 = row g + 8.
// A dot product does not depend on which slot holds which `k`, as long as A
// and B agree, so lane `q` takes four consecutive codes of its rows:
// half `h` of a block puts `k = 16h + 4q + j` in slot `2q + j` and
// `k = 16h + 4q + 2 + j` in slot `8 + 2q + j` (`j` 0 or 1). A lane then reads
// one 32-bit word (four FP8 codes, or 16 bits of packed FP4) per row and half.

DSTC_FN uint32_t load32(const uint8_t* p) {
#if defined(__CUDA_ARCH__)
  return *reinterpret_cast<const uint32_t*>(p);
#else
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
         static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
#endif
}
DSTC_FN uint32_t load16(const uint8_t* p) {
#if defined(__CUDA_ARCH__)
  return *reinterpret_cast<const uint16_t*>(p);
#else
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8;
#endif
}

// Four consecutive codes as the two F16 words of slots `2q..` and `8 + 2q..`.
DSTC_FN void fp8_word(uint32_t w, uint32_t* lo, uint32_t* hi) {
  *lo = e4m3x2_to_f16x2(w & 0xffffu);
  *hi = e4m3x2_to_f16x2(w >> 16);
}

// The A fragments of the 16-row tiles from staged weight row `frow` and
// every B fragment (the staged rows) of block `blk` of the step:
// `a[mt][h]`, `b[nt][h]` for the two halves `h` of the block.
template <class C>
DSTC_FN void load_frags(const Stage<C>& s, uint32_t frow, uint32_t lane, uint32_t blk,
                        uint32_t (&a)[C::kMTiles][2][4], uint32_t (&b)[C::kNTiles][2][2]) {
  const uint32_t g = lane >> 2, q = lane & 3u;
  for (uint32_t mt = 0; mt < C::kMTiles; ++mt) {
    const uint32_t r = frow + mt * 16 + g;
    for (uint32_t h = 0; h < 2; ++h) {
      if (C::kFp4) {
        const uint32_t at = blk * 16 + h * 8 + q * 2;
        e2m1x4_to_f16(load16(&s.w[r][at]), &a[mt][h][0], &a[mt][h][2]);
        e2m1x4_to_f16(load16(&s.w[r + 8][at]), &a[mt][h][1], &a[mt][h][3]);
      } else {
        const uint32_t at = blk * 32 + h * 16 + q * 4;
        fp8_word(load32(&s.w[r][at]), &a[mt][h][0], &a[mt][h][2]);
        fp8_word(load32(&s.w[r + 8][at]), &a[mt][h][1], &a[mt][h][3]);
      }
    }
  }
  for (uint32_t nt = 0; nt < C::kNTiles; ++nt) {
    const uint32_t j = nt * 8 + g;
    for (uint32_t h = 0; h < 2; ++h) {
      fp8_word(load32(&s.x[j][blk * 32 + h * 16 + q * 4]), &b[nt][h][0], &b[nt][h][1]);
    }
  }
}

// The staged output row and tile row of element `i` of D for tiles `mt`, `nt`.
DSTC_FN uint32_t d_feature(uint32_t frow, uint32_t lane, uint32_t mt, uint32_t i) {
  return frow + mt * 16 + (lane >> 2) + (i >= 2 ? 8u : 0u);
}
DSTC_FN uint32_t d_token(uint32_t lane, uint32_t nt, uint32_t i) {
  return nt * 8 + (lane & 3u) * 2 + (i & 1u);
}

// The n-tiles of the tile that hold a row of the pass: `[first, last)`.
DSTC_FN uint32_t ntile_first(const Pass& p) { return p.lo / 8; }
DSTC_FN uint32_t ntile_last(const Pass& p) { return (p.hi + 7) / 8; }

// Block `blk`'s contribution to output `f` of row `r`: `(dot · activation
// scale) · weight scale`, the product `ds::gemm_accumulate` adds.
template <class C>
DSTC_FN float contribution(const Stage<C>& s, uint32_t f, uint32_t r, uint32_t blk, float dot) {
  return dot * ds::act_scale_decode(s.xs[r][blk]) * ds::e8m0_decode(s.ws[f][blk]);
}

// The accumulation step of staged output `f`, row `r`, block `blk`.
template <class C>
DSTC_FN float accumulate(const Stage<C>& s, uint32_t f, uint32_t r, uint32_t blk, float acc,
                         float dot) {
  return ds::gemm_accumulate(acc, dot, ds::act_scale_decode(s.xs[r][blk]),
                             ds::e8m0_decode(s.ws[f][blk]));
}

// Store staged output `f` of tile row `r` if it is an output of the pass.
DSTC_FN void store(const Pass& p, uint32_t f, uint32_t r, float acc) {
  const uint32_t n = p.f0 + f;
  if (n < p.n && r >= p.lo && r < p.hi) {
    p.out[static_cast<uint64_t>(r) * p.n + n] = ds::bf16_store(ds::bf16_round(acc));
  }
}

// ---- lin4's groups ----

DSTC_FN uint32_t align_up(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

// The first group whose run ends after `row` (`groups` if none).
DSTC_FN uint32_t first_group(const DsLin4Params& p, uint32_t row) {
  uint32_t lo = 0, hi = p.groups;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (p.group_end[mid] > row) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}
DSTC_FN uint32_t group_start(const DsLin4Params& p, uint32_t g) {
  return g == 0 ? 0u : align_up(p.group_end[g - 1], p.row_align);
}

// The pass of group `g` over the tile of rows `[r0, r1)` and outputs from
// `f0`; its rows are empty when the group's run misses the tile.
DSTC_FN Pass lin4_pass(const DsLin4Params& p, uint32_t g, uint32_t r0, uint32_t r1, uint32_t f0) {
  const uint32_t start = group_start(p, g), end = p.group_end[g];
  const uint32_t lo = start > r0 ? start : r0, hi = end < r1 ? end : r1;
  const uint32_t kb = p.k / 32;
  Pass s;
  s.x = p.codes + static_cast<uint64_t>(r0) * p.k;
  s.xs = p.act_scales + static_cast<uint64_t>(r0) * kb;
  s.w = p.w_packed + static_cast<uint64_t>(g) * p.n * (p.k / 2);
  s.ws = p.w_scales + static_cast<uint64_t>(g) * p.n * kb;
  s.out = p.out + static_cast<uint64_t>(r0) * p.n;
  s.lo = lo < hi ? lo - r0 : 0;
  s.hi = lo < hi ? hi - r0 : 0;
  s.n = p.n;
  s.k = p.k;
  s.f0 = f0;
  return s;
}

DSTC_FN Pass lin8_pass(const DsLin8Params& p, uint32_t r0, uint32_t r1, uint32_t f0) {
  Pass s;
  s.x = p.codes + static_cast<uint64_t>(r0) * p.k;
  s.xs = p.act_scales + static_cast<uint64_t>(r0) * (p.k / 32);
  s.w = p.w_codes;
  s.ws = p.w_scales;
  s.out = p.out + static_cast<uint64_t>(r0) * p.n;
  s.lo = 0;
  s.hi = r1 - r0;
  s.n = p.n;
  s.k = p.k;
  s.f0 = f0;
  return s;
}

}  // namespace dstc

// ---- the probe ----

// `eidola_ds_gemm_probe`: what a block's product (`dstc` fragments, two
// chained F16 `mma.m16n8k16` from a zero accumulator, as both forms run it)
// returns, against the exact sum. Each warp runs `trials` trials; a trial draws a 16 × 32 A and a
// 32 × 8 B from a counter-based generator of `seed`, the warp and the trial,
// runs the MMA, and compares every element with the exact dot product
// rounded once to f32 (an integer sum, as `ds::fp8_fp8_block_dot`). It also
// runs the MMA again with A's rows and B's columns rotated, and requires
// every element to be the same bits as the element it moved from.
struct DsGemmProbeParams {
  // [5] u64: dot products checked; those not equal to the exact sum
  // rounded once; elements whose bits moved with their place; the largest
  // distance in f32 units in the last place seen; records claimed.
  unsigned long long* counts;
  // `[max_records][8]`: the first differing dot products (global warp,
  // trial, row, column, the bits returned, the bits of the rounded exact
  // sum, the exact sum's low and high words in units of 2^-18 or 2^-10).
  uint32_t* records;
  uint64_t seed;
  uint32_t trials;
  uint32_t max_records;
  // Exponent-field bands the codes are drawn from: `lo | hi << 8`. For
  // `a_format` 1 the A band is not used.
  uint32_t a_band;
  uint32_t b_band;
  // 0: A is E4M3 codes. 1: A is E2M1 codes, converted as the FP4 weights'
  // load converts them (`dstc::e2m1x4_to_f16`); the exact sum is then in
  // units of 2^-10.
  uint32_t a_format;
  // How codes are drawn (`DS_PROBE_*`).
  uint32_t family;
};
DS_ABI_STRUCT(DsGemmProbeParams, 48);
DS_ABI_FIELD(DsGemmProbeParams, counts, 0);
DS_ABI_FIELD(DsGemmProbeParams, records, 8);
DS_ABI_FIELD(DsGemmProbeParams, seed, 16);
DS_ABI_FIELD(DsGemmProbeParams, trials, 24);
DS_ABI_FIELD(DsGemmProbeParams, max_records, 28);
DS_ABI_FIELD(DsGemmProbeParams, a_band, 32);
DS_ABI_FIELD(DsGemmProbeParams, b_band, 36);
DS_ABI_FIELD(DsGemmProbeParams, a_format, 40);
DS_ABI_FIELD(DsGemmProbeParams, family, 44);

// Codes uniform over the band: sign, exponent field and mantissa; a sixteenth
// of them zero.
#define DS_PROBE_UNIFORM 0u
// Each code the band's largest or smallest magnitude, either sign: the
// widest alignment within a sum.
#define DS_PROBE_EXTREMES 1u
// Every code the band's largest positive magnitude: the largest sum, every
// carry.
#define DS_PROBE_CARRY 2u
// Elements 16..31 repeat elements 0..15 of A and negate those of B, but for
// B's element 31: the large products cancel and leave a residue.
#define DS_PROBE_CANCEL 3u
