// Sparse attention of the DeepSeek-V4.1 forward (`attention::sparse_attend`)
// and the window part of its slot lists. The bodies are `ds_attention.cuh`'s.
//
// `eidola_ds_attn`: grid `(x, rows)`, 512 threads. Block `(bx, row)` computes
// heads `bx · g .. (bx + 1) · g` of the row, `g = ceil(heads / x)` at most 8,
// one after another over the row's slot blocks: the block's entries are
// dequantised into shared memory once and serve every head of the block, so a
// wider grid trades that sharing for more blocks in flight. A row's bits do not
// depend on `x`.
//
// `eidola_ds_window_slots`: grid `(ceil(window / 128), rows)`, a thread per
// slot.

#include "ds_attention.cuh"
#include "eidola_kernel.cuh"

extern "C" __global__ void __launch_bounds__(ds_attn::kThreads)
    eidola_ds_attn(const __grid_constant__ DsAttnParams p) {
  extern __shared__ __align__(8) unsigned char smem[];
  ds_attn::attend(p, *reinterpret_cast<ds_attn::Shared*>(smem), blockIdx.x, gridDim.x, blockIdx.y,
                  ds::Par{});
}
EIDOLA_KERNEL_META(eidola_ds_attn, ds_attn::kThreads, 1, 1, sizeof(ds_attn::Shared), 1, 1, 1,
                   sizeof(DsAttnParams));

extern "C" __global__ void __launch_bounds__(ds_attn::kSlotThreads)
    eidola_ds_window_slots(const __grid_constant__ DsWindowSlotsParams p) {
  ds_attn::window_slot(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_ds_window_slots, ds_attn::kSlotThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(DsWindowSlotsParams));
