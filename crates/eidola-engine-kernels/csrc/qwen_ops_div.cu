// The Qwen3.8 executor's kernels that divide or take a square root: the
// zero-centred RMS norm (with the residual add and RoPE, and its fused forms),
// the FP8 activation
// quantiser, the element-wise gates (their sigmoid and SiLU divide in f64),
// the exact attention core, and the DeltaNet's per-position preparation and
// gated norm. The fused multiply-adds in this image's SASS are ptxas's
// expansions of those divisions and roots; its PTX holds none
// (`tests/manifest.rs`).

#include "eidola_kernel.cuh"
#include "qwen_ops.cuh"

// A unit wider than `kNormWarpMax` a CTA (grid: units by rows); narrower
// units a warp each, eight a CTA, numbered row by row (grid: their count
// over eight). The choice is the launch's (its width), uniform over the CTA.
template <bool Fused>
__device__ __forceinline__ void norm_entry(const QwNormParams& p) {
  __shared__ qwk::NormShared sh;
  if (p.n > qwk::kNormWarpMax) {
    qwk::norm_item<qwk::kElementThreads, Fused>(p, blockIdx.y, blockIdx.x, ds::Par{}, sh.ln[0],
                                                sh.row, sh.gm);
    return;
  }
  const uint32_t w = threadIdx.x / qwk::kWarp;
  const uint64_t item = static_cast<uint64_t>(blockIdx.x) * qwk::kNormItems + w;
  // Uniform across the warp: the unit is the warp's.
  if (item >= static_cast<uint64_t>(p.rows) * p.units) return;
  qwk::norm_item<qwk::kWarp, Fused>(p, static_cast<uint32_t>(item / p.units),
                                    static_cast<uint32_t>(item % p.units), qwk::WarpPar{},
                                    sh.ln[w], sh.row + w * qwk::kNormWarpMax,
                                    sh.gm + w * (qwk::kNormWarpMax / 4));
}

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_norm(const __grid_constant__ QwNormParams p) {
  norm_entry<false>(p);
}
EIDOLA_KERNEL_META(eidola_qwen_norm, qwk::kElementThreads, 1, 1, 0, 1, 1, 1, sizeof(QwNormParams));

// The norm's fused forms (`qwen_ops.cuh`): the residual add from a split
// GEMM's partials, the normed rows quantised. The same grid.
extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_norm_fused(const __grid_constant__ QwNormParams p) {
  norm_entry<true>(p);
}
EIDOLA_KERNEL_META(eidola_qwen_norm_fused, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwNormParams));

// The warp form (`qwen_ops.cuh`): a warp per block and up to `kQuantRows` rows
// (the rows over the grid's y: one a CTA at decode sizes, four at prefill
// sizes), every row's load issued first, its lanes' amaxes combined by a
// butterfly of shuffles.
extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_quant(const __grid_constant__ QwQuantParams p) {
  const uint32_t l = threadIdx.x % qwk::kWarp;
  uint32_t b;
  // Uniform across the warp: the block is the warp's.
  if (!qwk::quant_block_of(p, blockIdx.x, threadIdx.x / qwk::kWarp, &b)) return;
  const uint32_t per = (p.rows + gridDim.y - 1) / gridDim.y;
  const uint32_t r0 = blockIdx.y * per;
  if (r0 >= p.rows) return;
  qwk::QuantLane v[qwk::kQuantRows];
  uint32_t m[qwk::kQuantRows];
#pragma unroll
  for (uint32_t r = 0; r < qwk::kQuantRows; ++r) {
    const bool own = r < per && r0 + r < p.rows;
    v[r] = qwk::quant_lane_load(p, own ? r0 + r : r0, b, l);
  }
#pragma unroll
  for (uint32_t r = 0; r < qwk::kQuantRows; ++r) {
    m[r] = qwk::quant_lane_amax(v[r]);
    for (uint32_t off = qwk::kWarp / 2; off > 0; off >>= 1) {
      m[r] = max(m[r], __shfl_xor_sync(0xffffffffu, m[r], off));
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < qwk::kQuantRows; ++r) {
    if (r < per && r0 + r < p.rows) qwk::quant_lane_store(p, r0 + r, b, l, v[r], m[r]);
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

// A warp a (row, head), four a CTA, numbered row by row.
extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_dn_norm(const __grid_constant__ QwDnNormParams p) {
  __shared__ qwk::DnNormShared sh[qwk::kDnNormItems];
  const uint32_t w = threadIdx.x / qwk::kWarp;
  const uint64_t item = static_cast<uint64_t>(blockIdx.x) * qwk::kDnNormItems + w;
  // Uniform across the warp: the head is the warp's.
  if (item >= static_cast<uint64_t>(p.rows) * p.v_heads) return;
  qwk::dn_norm_item(p, static_cast<uint32_t>(item / p.v_heads),
                    static_cast<uint32_t>(item % p.v_heads), qwk::WarpPar{}, sh[w]);
}
EIDOLA_KERNEL_META(eidola_qwen_dn_norm, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnNormParams));
