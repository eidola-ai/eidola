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

// The warp form (`qwen_ops.cuh`): a warp per block and `kQuantRows` rows, its
// lanes' amaxes combined by a butterfly of shuffles.
extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_quant(const __grid_constant__ QwQuantParams p) {
  const uint32_t l = threadIdx.x % qwk::kWarp;
  uint32_t b;
  // Uniform across the warp: the block is the warp's.
  if (!qwk::quant_block_of(p, blockIdx.x, threadIdx.x / qwk::kWarp, &b)) return;
  const uint32_t r0 = blockIdx.y * qwk::kQuantRows;
  float v[qwk::kQuantRows][qwk::kQuantLane];
  float m[qwk::kQuantRows];
#pragma unroll
  for (uint32_t r = 0; r < qwk::kQuantRows; ++r) {
    m[r] = r0 + r < p.rows ? qwk::quant_lane_load(p, r0 + r, b, l, v[r]) : 0.0f;
  }
#pragma unroll
  for (uint32_t r = 0; r < qwk::kQuantRows; ++r) {
    for (uint32_t off = qwk::kWarp / 2; off > 0; off >>= 1) {
      m[r] = ds::max32(m[r], __shfl_xor_sync(0xffffffffu, m[r], off));
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < qwk::kQuantRows; ++r) {
    if (r0 + r < p.rows) qwk::quant_lane_store(p, r0 + r, b, l, v[r], m[r]);
  }
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
