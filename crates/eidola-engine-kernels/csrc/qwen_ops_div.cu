// The Qwen3.8 executor's kernels that divide or take a square root: the
// zero-centred RMS norm (with the residual add and RoPE), the FP8 activation
// quantiser, the element-wise gates (their sigmoid and SiLU divide in f64),
// the exact attention core, and the DeltaNet's per-position preparation and
// gated norm. The fused multiply-adds in this image's SASS are ptxas's
// expansions of those divisions and roots; its PTX holds none
// (`tests/manifest.rs`).

#include "eidola_kernel.cuh"
#include "qwen_ops.cuh"

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_norm(const __grid_constant__ QwNormParams p) {
  __shared__ qwk::NormShared sh;
  qwk::norm_block(p, blockIdx.y, blockIdx.x, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_norm, qwk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(QwNormParams));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_quant(const __grid_constant__ QwQuantParams p) {
  qwk::quant_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_quant, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1, sizeof(QwQuantParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_gate(const __grid_constant__ QwGateParams p) {
  qwk::gate_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_gate, qwk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(QwGateParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_attn(const __grid_constant__ QwAttnParams p) {
  __shared__ qwk::AttnShared sh;
  qwk::attn_block(p, blockIdx.y, blockIdx.x, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_attn, qwk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(QwAttnParams));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_dn_prep(const __grid_constant__ QwDnPrepParams p) {
  __shared__ qwk::DnPrepShared sh;
  qwk::dn_prep_block(p, blockIdx.y, blockIdx.x, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_dn_prep, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnPrepParams));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_dn_norm(const __grid_constant__ QwDnNormParams p) {
  __shared__ qwk::DnNormShared sh;
  qwk::dn_norm_block(p, blockIdx.y, blockIdx.x, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_dn_norm, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnNormParams));
