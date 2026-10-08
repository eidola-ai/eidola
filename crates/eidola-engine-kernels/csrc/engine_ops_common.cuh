// Helpers shared by the executor's own kernels (engine_ops.cu) and the
// reference kernels the GPU tests compare them against
// (engine_ops_reference.cu): conversions, warp reductions, the UE8M0 recipe
// and the packed scale layout. Internal linkage, as in a single translation
// unit.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreads = 256;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__device__ __forceinline__ uint16_t f2bf16(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over a kThreads block.
__device__ float block_sum(float v) {
  __shared__ float part[kThreads / 32];
  __shared__ float total;
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < kThreads / 32 ? part[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) total = t;
  }
  __syncthreads();
  const float out = total;
  __syncthreads();
  return out;
}

// UE8M0 exponent byte for a group's amax: 2^ceil(log2(amax / 448)).
__device__ __forceinline__ uint8_t ue8m0_for(float amax) {
  const float r = fmaxf(amax, 1e-10f) / kFp8Max;
  int e = static_cast<int>(ceilf(log2f(r)));
  e = max(-127, min(127, e));
  return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float ue8m0_value(uint8_t b) { return exp2f(static_cast<float>(b) - 127.f); }

// Index of SFA word `w` (of `words` per row) for row `r`: [words][rows4]
// when cap == 0, else [r / cap][words][cap].
__device__ __forceinline__ size_t sfa_index(uint32_t r, uint32_t w, uint32_t words, uint32_t rows4,
                                            uint32_t cap) {
  if (cap == 0) return static_cast<size_t>(w) * rows4 + r;
  return (static_cast<size_t>(r / cap) * words + w) * cap + r % cap;
}

}  // namespace
