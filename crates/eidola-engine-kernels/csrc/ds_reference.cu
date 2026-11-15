// Reference forms of the DeepSeek-V4.1 products: the quantised GEMMs as
// scalar integer block dots (`lin8`, `lin4`: what a tensor-core form must
// equal, and what the tests and a bring-up run), and the attention output's
// grouped projection under its current definition, a pinned dot of the two
// dequantised operands.

#include "ds_ops.cuh"
#include "ds_shuffle.cuh"
#include "eidola_kernel.cuh"

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_reference_lin8(const __grid_constant__ DsLin8Params p) {
  dsk::lin8_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_reference_lin8, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsLin8Params));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_reference_lin4(const __grid_constant__ DsLin4Params p) {
  dsk::lin4_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_reference_lin4, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsLin4Params));

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_grouped_dot(const __grid_constant__ DsGroupedDotParams p) {
  const uint32_t lane = threadIdx.x & 15u;
  const uint32_t i0 = (blockIdx.x * 16 + (threadIdx.x >> 4)) * dsk::kLinearChains;
  const uint32_t group = blockIdx.y, row = blockIdx.z;
  float acc[dsk::kLinearChains];
  dsk::grouped_lane(p, row, group, i0, lane, acc);
  for (uint32_t u = 0; u < dsk::kLinearChains; ++u) {
    float lanes[16];
    dsk::gather16(acc[u], lanes);
    if (lane == 0 && i0 + u < p.rank) dsk::grouped_finish(p, row, group, i0 + u, lanes);
  }
}
EIDOLA_KERNEL_META(eidola_ds_grouped_dot, dsk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsGroupedDotParams));
