// Qwen3.8's fast-profile kernels of our own: bodies for the device and for
// the host check (`eidola-engine-cuda`'s `oracle/qwen_launch_host.cc`), like
// `qwen_ops.cuh`'s, but not equal to the model crate: they belong to the fast
// profile, whose contract is that a row's result is a function of that row
// and the weights alone (`eidola-engine-cuda/AGENTS.md` → Qwen3.8). Their
// arithmetic is written out (`fma32`, every sum's operands in a fixed order),
// so the host build, which contracts nothing, computes what the device does.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "qwen_abi.cuh"

#if defined(__CUDACC__)
#define QF_FN __host__ __device__ __forceinline__
#else
#define QF_FN inline
#endif

namespace qwf {

// A BF16 value's f32 (its 16 stored bits widened).
QF_FN float bf16_value(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// `fma(a, b, c)`, rounded once, on both sides.
QF_FN float fma32(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

// ---- the small-N BF16 GEMM (`eidola_qwen_fast_gemm_bf16`) ----
//
// `out[r][n] (f32) = Σ_k x[r][k] · w[n][k]` for a weight of few rows (the
// DeltaNet's stacked `a` and `b`: 96 × 5,120), where a tensor-core tile of 128
// outputs left one CTA streaming the whole weight. A warp computes one output
// for `kSmallRows` rows: lane `l` takes the 8-element chunks `c = l, l + 32,
// …` of K (a 16-byte load of the weight row and of each row), each element of
// a chunk in turn, `acc = fma(x, w, acc)` from +0; the lanes' sums are then
// combined by an xor butterfly (16, 8, 4, 2, 1), whose additions are
// commutative, so every lane holds the same sum. (A product of two BF16
// values is exact in f32, so the fused and the unfused step give the same
// bits; the order of the sum is what fixes them.) A row's value depends on its
// own elements and the weight row only: never on the rows beside it, their
// count, or where the grid put it.

constexpr uint32_t kWarp = 32;
constexpr uint32_t kSmallWarps = 4;   // outputs per CTA, a warp each
constexpr uint32_t kSmallRows = 4;    // rows per CTA
constexpr uint32_t kSmallChunk = 8;   // a lane's elements per load (16 bytes of BF16)

// Lane `l`'s partial sums of output `n` for rows `r0 ..` (`rows` of them, at
// most `kSmallRows`), into `acc`.
QF_FN void small_gemm_lane(const QwLinearParams& p, uint32_t n, uint32_t r0, uint32_t rows,
                           uint32_t l, float* acc) {
  for (uint32_t r = 0; r < kSmallRows; ++r) acc[r] = 0.0f;
  const uint32_t chunks = p.k / kSmallChunk;
  const uint16_t* w = p.w + static_cast<uint64_t>(n) * p.k;
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#endif
  for (uint32_t c = l; c < chunks; c += kWarp) {
    uint16_t wv[kSmallChunk];
    uint16_t xv[kSmallRows][kSmallChunk];
#if defined(__CUDA_ARCH__)
    {
      const uint4 u = *reinterpret_cast<const uint4*>(w + c * kSmallChunk);
      const uint32_t words[4] = {u.x, u.y, u.z, u.w};
      for (uint32_t e = 0; e < 4; ++e) {
        wv[2 * e] = static_cast<uint16_t>(words[e] & 0xffffu);
        wv[2 * e + 1] = static_cast<uint16_t>(words[e] >> 16);
      }
    }
    for (uint32_t r = 0; r < kSmallRows; ++r) {
      if (r < rows) {
        const uint4 u = *reinterpret_cast<const uint4*>(
            p.x + static_cast<uint64_t>(r0 + r) * p.x_stride + c * kSmallChunk);
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
        for (uint32_t e = 0; e < 4; ++e) {
          xv[r][2 * e] = static_cast<uint16_t>(words[e] & 0xffffu);
          xv[r][2 * e + 1] = static_cast<uint16_t>(words[e] >> 16);
        }
      }
    }
#else
    for (uint32_t e = 0; e < kSmallChunk; ++e) wv[e] = w[c * kSmallChunk + e];
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t e = 0; e < kSmallChunk; ++e) {
        xv[r][e] = p.x[static_cast<uint64_t>(r0 + r) * p.x_stride + c * kSmallChunk + e];
      }
    }
#endif
    for (uint32_t r = 0; r < kSmallRows; ++r) {
      if (r < rows) {
        for (uint32_t e = 0; e < kSmallChunk; ++e) {
          acc[r] = fma32(bf16_value(xv[r][e]), bf16_value(wv[e]), acc[r]);
        }
      }
    }
  }
}

// The CTA (`bx`, `by`)'s output and first row for warp `w`, and its row count;
// false past the outputs or rows.
QF_FN bool small_gemm_unit(const QwLinearParams& p, uint32_t bx, uint32_t by, uint32_t w,
                           uint32_t* n, uint32_t* r0, uint32_t* rows) {
  *n = bx * kSmallWarps + w;
  *r0 = by * kSmallRows;
  if (*n >= p.n || *r0 >= p.rows) return false;
  *rows = p.rows - *r0 < kSmallRows ? p.rows - *r0 : kSmallRows;
  return true;
}

// Lane 0's stores of the combined sums.
QF_FN void small_gemm_store(const QwLinearParams& p, uint32_t n, uint32_t r0, uint32_t rows,
                            const float* sum) {
  for (uint32_t r = 0; r < rows; ++r) {
    static_cast<float*>(p.out)[static_cast<uint64_t>(r0 + r) * p.out_stride + n] = sum[r];
  }
}

// ---- the tiled form (`eidola_qwen_fast_gemm_bf16_tiled`) ----
//
// At prefill sizes the form above reads a weight row once per four rows and
// each row once per output. The tiled form gives a warp `kTileRows` rows by
// `kTileOuts` outputs (a CTA's four warps share its rows, so they read them
// from L1): per chunk a lane loads each of its rows and each of its weight rows
// once and makes every one of the tile's products from them. Every (row,
// output) is summed exactly as above: lane `l` over its chunks `l, l + 32, …`,
// each chunk's elements in turn, by fused multiply-adds from +0, then the same
// butterfly. So a row's bits do not depend on which form ran, which is what
// lets the executor choose by row count. On the device two rows' sums advance
// in one paired fused multiply-add (`fma.rn.f32x2`: each half is rounded
// once, as `fma32`).

constexpr uint32_t kTileRows = 8;   // rows a warp of the tiled form takes
constexpr uint32_t kTileOuts = 4;   // outputs a warp of the tiled form takes
constexpr uint32_t kTileWarps = 4;  // warps a CTA: its outputs side by side, the rows shared
static_assert(kTileRows % 2 == 0, "rows go in pairs");

// The CTA (`bx`, `by`)'s outputs and rows for warp `w`, and their counts;
// false past the outputs or rows.
QF_FN bool tile_gemm_unit(const QwLinearParams& p, uint32_t bx, uint32_t by, uint32_t w,
                          uint32_t* n0, uint32_t* outs, uint32_t* r0, uint32_t* rows) {
  *n0 = (bx * kTileWarps + w) * kTileOuts;
  *r0 = by * kTileRows;
  if (*n0 >= p.n || *r0 >= p.rows) return false;
  *outs = p.n - *n0 < kTileOuts ? p.n - *n0 : kTileOuts;
  *rows = p.rows - *r0 < kTileRows ? p.rows - *r0 : kTileRows;
  return true;
}

#if defined(__CUDA_ARCH__)
// Two f32 values in one 64-bit register pair, and `fma.rn.f32x2`.
__device__ __forceinline__ uint64_t pair32(float lo, float hi) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ void unpair32(uint64_t v, float* lo, float* hi) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(*lo), "=f"(*hi) : "l"(v));
}
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}
#endif

// Lane `l`'s partial sums of the tile: `acc[r · kTileOuts + o]` for row
// `r0 + r` and output `n0 + o`. A row or output past the tile's counts reads
// the first one instead and is never stored.
QF_FN void tile_gemm_lane(const QwLinearParams& p, uint32_t n0, uint32_t outs, uint32_t r0,
                          uint32_t rows, uint32_t l, float* acc) {
  const uint32_t chunks = p.k / kSmallChunk;
  // The tile's first row and weight row, and each one's offset from it.
  const uint16_t* x0 = p.x + static_cast<uint64_t>(r0) * p.x_stride;
  const uint16_t* w0 = p.w + static_cast<uint64_t>(n0) * p.k;
  uint32_t xo[kTileRows];
  uint32_t wo[kTileOuts];
  for (uint32_t r = 0; r < kTileRows; ++r) xo[r] = r < rows ? r * p.x_stride : 0;
  for (uint32_t o = 0; o < kTileOuts; ++o) wo[o] = o < outs ? o * p.k : 0;
#if defined(__CUDA_ARCH__)
  uint64_t a2[kTileRows / 2][kTileOuts];
#pragma unroll
  for (uint32_t r = 0; r < kTileRows / 2; ++r)
#pragma unroll
    for (uint32_t o = 0; o < kTileOuts; ++o) a2[r][o] = 0;
  for (uint32_t c = l; c < chunks; c += kWarp) {
    uint4 xv[kTileRows];
    uint4 wv[kTileOuts];
#pragma unroll
    for (uint32_t r = 0; r < kTileRows; ++r) {
      xv[r] = __ldg(reinterpret_cast<const uint4*>(x0 + xo[r] + c * kSmallChunk));
    }
#pragma unroll
    for (uint32_t o = 0; o < kTileOuts; ++o) {
      wv[o] = __ldg(reinterpret_cast<const uint4*>(w0 + wo[o] + c * kSmallChunk));
    }
#pragma unroll
    for (uint32_t e = 0; e < kSmallChunk; ++e) {
      // Element `e` of a 16-byte chunk: the low or high half of word `e / 2`.
      auto half = [e](const uint4& u) {
        const uint32_t word = e / 2 == 0 ? u.x : e / 2 == 1 ? u.y : e / 2 == 2 ? u.z : u.w;
        return __uint_as_float(e % 2 == 0 ? word << 16 : word & 0xffff0000u);
      };
      float wf[kTileOuts];
#pragma unroll
      for (uint32_t o = 0; o < kTileOuts; ++o) wf[o] = half(wv[o]);
#pragma unroll
      for (uint32_t r = 0; r < kTileRows / 2; ++r) {
        const uint64_t xp = pair32(half(xv[2 * r]), half(xv[2 * r + 1]));
#pragma unroll
        for (uint32_t o = 0; o < kTileOuts; ++o) {
          a2[r][o] = fma2(xp, pair32(wf[o], wf[o]), a2[r][o]);
        }
      }
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kTileRows / 2; ++r)
#pragma unroll
    for (uint32_t o = 0; o < kTileOuts; ++o) {
      unpair32(a2[r][o], &acc[(2 * r) * kTileOuts + o], &acc[(2 * r + 1) * kTileOuts + o]);
    }
#else
  for (uint32_t i = 0; i < kTileRows * kTileOuts; ++i) acc[i] = 0.0f;
  for (uint32_t c = l; c < chunks; c += kWarp) {
    for (uint32_t e = 0; e < kSmallChunk; ++e) {
      for (uint32_t r = 0; r < kTileRows; ++r) {
        const float xv = bf16_value(x0[xo[r] + c * kSmallChunk + e]);
        for (uint32_t o = 0; o < kTileOuts; ++o) {
          float& a = acc[r * kTileOuts + o];
          a = fma32(xv, bf16_value(w0[wo[o] + c * kSmallChunk + e]), a);
        }
      }
    }
  }
#endif
}

// Lane 0's stores of the tile's combined sums.
QF_FN void tile_gemm_store(const QwLinearParams& p, uint32_t n0, uint32_t outs, uint32_t r0,
                           uint32_t rows, const float* sum) {
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t o = 0; o < outs; ++o) {
      static_cast<float*>(p.out)[static_cast<uint64_t>(r0 + r) * p.out_stride + n0 + o] =
          sum[r * kTileOuts + o];
    }
  }
}

}  // namespace qwf
