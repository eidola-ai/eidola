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

}  // namespace qwf
