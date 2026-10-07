// RMSNorm over BF16 rows: out = w * x / sqrt(mean(x^2) + eps), accumulated
// in f32. One thread block per row; the host launches `rows` blocks.
//
// This is the smallest kernel the pipeline builds and exists to prove the path
// for our own CUDA: no vendored headers beyond the toolkit's.

#include <cuda_bf16.h>

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_rmsnorm_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ x,
                        const __nv_bfloat16* __restrict__ weight, uint32_t hidden, float eps) {
  __shared__ float partial[kWarps];
  __shared__ float inv_rms;

  const __nv_bfloat16* row = x + static_cast<size_t>(blockIdx.x) * hidden;
  __nv_bfloat16* out_row = out + static_cast<size_t>(blockIdx.x) * hidden;

  float sum_sq = 0.f;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = __bfloat162float(row[i]);
    sum_sq += v * v;
  }
  sum_sq = warp_sum(sum_sq);
  if (threadIdx.x % 32 == 0) {
    partial[threadIdx.x / 32] = sum_sq;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float total = threadIdx.x < kWarps ? partial[threadIdx.x] : 0.f;
    total = warp_sum(total);
    if (threadIdx.x == 0) {
      inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
  }
  __syncthreads();

  const float scale = inv_rms;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = __bfloat162float(row[i]) * scale;
    out_row[i] = __float2bfloat16(v * __bfloat162float(weight[i]));
  }
}

EIDOLA_KERNEL_META(eidola_rmsnorm_bf16, kThreads, 1, 1, 0, 1, 1, 1, 0);
