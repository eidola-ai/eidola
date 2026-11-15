// The DeepSeek-V4.1 executor's scalar kernels that divide or take a square
// root: the global KV's fake quantisation, RMS norm, the hyper-connection
// coefficients, the stream mean, the ratio-2 pooling, the router's selection,
// SwiGLU and the Engram injection. ptxas expands a division into fused
// multiply-adds of its own, so this image's SASS cannot be required to hold
// none; its PTX is held like every exact image's.

#include "ds_ops.cuh"
#include "ds_shuffle.cuh"
#include "eidola_kernel.cuh"

extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_rot_quant_e4m3(const __grid_constant__ DsRotQuantParams p) {
  dsk::rot_quant_thread(p, blockIdx.x, blockIdx.y, threadIdx.x, true);
}
EIDOLA_KERNEL_META(eidola_ds_rot_quant_e4m3, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsRotQuantParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_rms_norm(const __grid_constant__ DsRmsNormParams p) {
  __shared__ float factor;
  const uint32_t t = threadIdx.x;
  if (t < 32) {
    float lanes[16];
    dsk::gather16(t < 16 ? dsk::rms_lane(p, blockIdx.x, t) : 0.0f, lanes);
    if (t == 0) factor = dsk::rms_finish(p, blockIdx.x, lanes);
  }
  // The row may be normed in place: no element is written before the sum is
  // complete.
  __syncthreads();
  dsk::rms_store(p, blockIdx.x, t, factor);
}
EIDOLA_KERNEL_META(eidola_ds_rms_norm, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsRmsNormParams));

extern "C" __global__ void __launch_bounds__(dsk::kMixThreads)
    eidola_ds_mhc_mix(const __grid_constant__ DsMhcMixParams p) {
  __shared__ float dots[dsk::kMixThreads / 16];
  const uint32_t g = threadIdx.x >> 4, lane = threadIdx.x & 15u;
  const bool live = g < dsk::mix_groups(p.copies);
  float lanes[16];
  dsk::gather16(live ? dsk::mhc_lane(p, blockIdx.x, g, lane) : 0.0f, lanes);
  if (live && lane == 0) dots[g] = dsk::mhc_group(p, blockIdx.x, g, lanes);
  __syncthreads();
  if (threadIdx.x == 0) dsk::mhc_finish(p, blockIdx.x, dots);
}
EIDOLA_KERNEL_META(eidola_ds_mhc_mix, dsk::kMixThreads, 1, 1, 0, 1, 1, 1, sizeof(DsMhcMixParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_stream_mean(const __grid_constant__ DsStreamMeanParams p) {
  dsk::stream_mean_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_stream_mean, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsStreamMeanParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_pool(const __grid_constant__ DsPoolParams p) {
  dsk::pool_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_pool, dsk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(DsPoolParams));

extern "C" __global__ void __launch_bounds__(dsk::kRouteThreads)
    eidola_ds_route(const __grid_constant__ DsRouteParams p) {
  __shared__ float score[dsk::kMaxExperts];
  __shared__ float biased[dsk::kMaxExperts];
  for (uint32_t e = threadIdx.x; e < p.experts; e += dsk::kRouteThreads) {
    dsk::route_score(p, blockIdx.x, e, score, biased);
  }
  __syncthreads();
  if (threadIdx.x == 0) dsk::route_select(p, blockIdx.x, score, biased);
}
EIDOLA_KERNEL_META(eidola_ds_route, dsk::kRouteThreads, 1, 1, 0, 1, 1, 1, sizeof(DsRouteParams));

extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_swiglu(const __grid_constant__ DsSwigluParams p) {
  dsk::swiglu_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_swiglu, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1, sizeof(DsSwigluParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_engram_inject(const __grid_constant__ DsEngramInjectParams p) {
  __shared__ float sums[3];
  __shared__ float gate;
  const uint32_t t = threadIdx.x, g = t >> 4, lane = t & 15u;
  const uint32_t copy = blockIdx.x, row = blockIdx.y;
  if (t < 64) {
    float lanes[16];
    dsk::gather16(g < 3 ? dsk::inject_lane(p, row, copy, g, lane) : 0.0f, lanes);
    if (g < 3 && lane == 0) sums[g] = dsk::inject_group(p, row, copy, g, lanes);
  }
  __syncthreads();
  if (t == 0) gate = dsk::inject_gate(p, sums);
  // The copy may be updated in place: its sums are complete.
  __syncthreads();
  dsk::inject_store(p, row, copy, t, gate);
}
EIDOLA_KERNEL_META(eidola_ds_engram_inject, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsEngramInjectParams));
