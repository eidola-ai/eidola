// The DeepSeek-V4.1 executor's scalar kernels that neither divide nor take a
// square root: embedding, the FP8 activation quantiser, RoPE with the
// power-of-two fake quantisations, the hyper-connection collapse and expand,
// plain pinned-order products, the expert layout, the feed-forward sum and
// the Engram gather. Their image's SASS holds no fused multiply-add
// (`tests/manifest.rs`); the kernels that divide are `ds_ops_div.cu`.
//
// Each entry is its body in `ds_ops.cuh` placed on a grid; the bodies say
// which reference function they are and why a row's result is its own.

#include "ds_ops.cuh"
#include "ds_shuffle.cuh"
#include "eidola_kernel.cuh"

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_embed(const __grid_constant__ DsEmbedParams p) {
  dsk::embed_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_embed, dsk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(DsEmbedParams));

extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_quant_fp8(const __grid_constant__ DsQuantFp8Params p) {
  dsk::quant_fp8_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_quant_fp8, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsQuantFp8Params));

// `DS_KV_FP8` and `DS_KV_FP4_E8M0`; `DS_KV_FP4_E4M3` is
// `eidola_ds_rot_quant_e4m3`.
extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_rot_quant(const __grid_constant__ DsRotQuantParams p) {
  dsk::rot_quant_thread(p, blockIdx.x, blockIdx.y, threadIdx.x, false);
}
EIDOLA_KERNEL_META(eidola_ds_rot_quant, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsRotQuantParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_collapse(const __grid_constant__ DsCollapseParams p) {
  dsk::collapse_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_collapse, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsCollapseParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_expand(const __grid_constant__ DsExpandParams p) {
  dsk::expand_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_expand, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsExpandParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_linear(const __grid_constant__ DsLinearParams p) {
  const uint32_t lane = threadIdx.x & 15u;
  const uint32_t n0 = (blockIdx.x * 16 + (threadIdx.x >> 4)) * dsk::kLinearChains;
  float acc[dsk::kLinearChains];
  dsk::linear_lane(p, blockIdx.y, n0, lane, acc);
  for (uint32_t u = 0; u < dsk::kLinearChains; ++u) {
    float lanes[16];
    dsk::gather16(acc[u], lanes);
    if (lane == 0 && n0 + u < p.n) dsk::linear_finish(p, blockIdx.y, n0 + u, lanes);
  }
}
EIDOLA_KERNEL_META(eidola_ds_linear, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsLinearParams));

extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_place_count(const __grid_constant__ DsPlaceParams p) {
  dsk::place_count_thread(p, blockIdx.x, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_place_count, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsPlaceParams));

extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_place(const __grid_constant__ DsPlaceParams p) {
  dsk::place_thread(p, blockIdx.x, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_place, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1, sizeof(DsPlaceParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_place_rows(const __grid_constant__ DsPlaceRowsParams p) {
  dsk::place_rows_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_place_rows, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsPlaceRowsParams));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_combine(const __grid_constant__ DsCombineParams p) {
  dsk::combine_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_combine, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsCombineParams));

extern "C" __global__ void __launch_bounds__(dsk::kBlockThreads)
    eidola_ds_engram_gather(const __grid_constant__ DsEngramGatherParams p) {
  dsk::engram_gather_thread(p, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_engram_gather, dsk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsEngramGatherParams));
