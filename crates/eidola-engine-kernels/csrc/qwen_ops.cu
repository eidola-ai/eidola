// The Qwen3.8 executor's kernels that neither divide nor take a square root:
// embedding, the exact FP8 and BF16 linears, the paged KV write, the BF16
// widening and the f32 rounding. Their image's SASS holds no fused multiply-add
// (`tests/manifest.rs`); the kernels that divide are `qwen_ops_div.cu`, the
// DeltaNet recurrence is `qwen_deltanet.cu`.
//
// Each entry is its body in `qwen_ops.cuh` placed on a grid.

#include "eidola_kernel.cuh"
#include "qwen_ops.cuh"

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_embed(const __grid_constant__ QwEmbedParams p) {
  qwk::embed_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_embed, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwEmbedParams));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_lin8(const __grid_constant__ QwLin8Params p) {
  qwk::lin8_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_lin8, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1, sizeof(QwLin8Params));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_linear(const __grid_constant__ QwLinearParams p) {
  qwk::linear_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_linear, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwLinearParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_kv_write(const __grid_constant__ QwKvWriteParams p) {
  qwk::kv_write_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_kv_write, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwKvWriteParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_widen(const __grid_constant__ QwWidenParams p) {
  qwk::widen_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_widen, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwWidenParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_round(const __grid_constant__ QwRoundParams p) {
  qwk::round_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_round, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwRoundParams));
