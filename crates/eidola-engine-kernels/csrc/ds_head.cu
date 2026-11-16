// The output head and the drafter's Markov head: `linear::linear_f32` over a
// weight with one row per vocabulary entry, laid out weight row by weight
// row so that one launch reads the weight once for every 8 rows of logits.
// The Markov head adds its products to the base logits (`addend`) and reads
// each row's input through `x_rows` (the Markov embedding of the token
// before).

#include "ds_ops.cuh"
#include "ds_shuffle.cuh"
#include "eidola_kernel.cuh"

extern "C" __global__ void __launch_bounds__(dsk::kElementThreads)
    eidola_ds_head(const __grid_constant__ DsLinearParams p) {
  const uint32_t lane = threadIdx.x & 15u;
  const uint32_t n = blockIdx.x * 16 + (threadIdx.x >> 4);
  const uint32_t row0 = blockIdx.y * dsk::kHeadChains;
  float acc[dsk::kHeadChains];
  dsk::head_lane(p, row0, n, lane, acc);
  for (uint32_t u = 0; u < dsk::kHeadChains; ++u) {
    float lanes[16];
    dsk::gather16(acc[u], lanes);
    if (lane == 0 && n < p.n && row0 + u < p.rows) dsk::linear_finish(p, row0 + u, n, lanes);
  }
}
EIDOLA_KERNEL_META(eidola_ds_head, dsk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(DsLinearParams));
