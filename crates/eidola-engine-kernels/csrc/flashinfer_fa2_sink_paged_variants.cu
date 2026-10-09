// Experimental FA2 sink-attention instances for the attention-invariance
// bench only; the executor never loads this image. Same FlashInfer paged
// prefill template, BF16 Q/KV/O, head dims 192/128, as
// flashinfer_fa2_sink_paged.cu, at explicit warp and KV-tile shapes instead
// of the ones FlashInfer's dispatcher derives from the query tile.
//
// A row's softmax over its keys is reduced in an order fixed by three things:
// how many warps split each KV tile (NUM_WARPS_KV, merged once at the end),
// how wide a tile is (CTA_TILE_KV = NUM_MMA_KV x 16 x NUM_WARPS_KV), and where
// the tiles start. The executor's three instances differ in the first two:
//
//   q16   1 Q warp x 4 KV warps, NUM_MMA_KV 2: 128-key tiles in 32-key stripes
//   q64   4 Q warps x 1 KV warp, NUM_MMA_KV 8: 128-key tiles
//   q128  4 Q warps x 1 KV warp, NUM_MMA_KV 4: 64-key tiles
//
// and the kernel starts a sliding layer's tiles at the first key the query
// tile's first row can see, so a row's tile boundaries move with the rows it
// shares a tile with. Instances here name their shape: q<CTA_TILE_Q>_w<NUM_WARPS_Q>x<NUM_WARPS_KV>_k<NUM_MMA_KV>.
//
// Two families:
//
//   stock     FlashInfer's AttentionSink variant, as the executor's. Tiles
//             start where the kernel puts them (key 0 for global layers).
//   anchored  AnchoredSink below: the tiles of every query tile start at a
//             multiple of CTA_TILE_KV counted from the first listed page, so
//             a host that lists pages from a multiple of 128 positions puts
//             every tile boundary at an absolute position. The window is
//             enforced on every key by LogitsTransform, since the kernel only
//             calls the mask on the iterations it predicts need it.
//
// Within one (NUM_WARPS_KV, NUM_MMA_KV) and one anchoring, a row's arithmetic
// is expected to be independent of CTA_TILE_Q and NUM_MMA_Q (each MMA
// fragment runs the same code); the bench measures whether it is.

#include "batch_prefill_config.inc"

#include <flashinfer/attention/prefill.cuh>

#include "eidola_kernel.cuh"

namespace eidola_fa2_variants {

using namespace flashinfer;

constexpr uint32_t kSmemPerBlockOptin = 227 * 1024;

// AttentionSink with every query tile's KV tiles anchored at a multiple of
// CTA_TILE_KV. The kernel starts a query tile's keys at
// sub_if_greater_or_zero(kv_len + q0, qo_len + window_left) for its first row
// q0; this variant reports a window widened just enough that the start lands
// on the multiple of CTA_TILE_KV at or below that row's first visible key.
// Keys before a row's window, in the widened leading tile or in any tile the
// kernel does not mask, are set to -inf in LogitsTransform: a fully masked
// tile leaves a row's (m, d, o) exactly as they were.
template <uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV>
struct AnchoredSink : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  // What the kernel reads: the request's true lengths and the widened window.
  uint32_t window_left, qo_len, kv_len;
  // The window the mask and transform enforce.
  uint32_t true_window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ AnchoredSink(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    true_window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = true_window_left;
#ifdef __CUDA_ARCH__
    // Anchoring moves where a query tile's keys start, which a KV split's
    // chunk boundaries would then no longer follow.
    if (params.partition_kv) __trap();
    const uint32_t tile = static_cast<uint32_t>(params.qo_tile_indices[blockIdx.x]);
    const uint32_t q0 = (tile * CTA_TILE_Q) / params.group_size;
    // The first key row q0 sees, as the kernel computes its start.
    const uint32_t first = sub_if_greater_or_zero(kv_len + q0, qo_len + true_window_left);
    // A tile whose first row sees key 0 starts there already. Otherwise
    // kv_len + q0 > qo_len + true_window_left and anchored <= first, so the
    // window below is wider than the true one.
    if (first > 0) {
      const uint32_t anchored = (first / CTA_TILE_KV) * CTA_TILE_KV;
      window_left = kv_len + q0 - qo_len - anchored;
    }
#endif
  }

  __device__ __forceinline__ bool visible(uint32_t qo_idx, uint32_t kv_idx) const {
    return kv_idx + qo_len + true_window_left >= kv_len + qo_idx;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return visible(qo_idx, kv_idx) ? logits : T(-math::inf);
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                       { return visible(qo_idx, kv_idx); })

  // AttentionSink's sink and output, unchanged. The kernel passes
  // update_m_d its work item's KV chunk (kv_tile_indices[blockIdx.x], 0 for
  // every work item without a split), not the tile the traversal starts at,
  // so the sink enters each row exactly once wherever an anchored start
  // begins. Anchored instances never run split (the constructor traps on
  // partition_kv); split stock instances add it in chunk 0 only, and the
  // merge carries it into the row once.
  REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, {
    float log_sink = (kv_tile_idx == 0 && qo_head_idx < params.num_qo_heads)
                         ? params.sink[qo_head_idx] * math::log2e
                         : -math::inf;
    float m_new = (log_sink > m) ? log_sink : m;
    scale = math::ptx_exp2(max(m - m_new, -math::inf));
    float d_new = math::ptx_exp2(max(log_sink - m_new, -math::inf)) + d * scale;
    m = m_new;
    d = d_new;
  })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    return output * scale * d_rcp;
  });
};

template <uint32_t CTA_TILE_Q, uint32_t WARPS_Q, uint32_t WARPS_KV, uint32_t MMA_KV, bool ANCHORED>
struct Shape {
  static constexpr uint32_t MMA_Q = CTA_TILE_Q / (16 * WARPS_Q);
  static_assert(MMA_Q * 16 * WARPS_Q == CTA_TILE_Q, "query tile is whole fragments per warp");
  static constexpr uint32_t CTA_TILE_KV = MMA_KV * 16 * WARPS_KV;
  using Variant = std::conditional_t<ANCHORED, AnchoredSink<CTA_TILE_Q, CTA_TILE_KV>, AttentionSink>;
  using KTraits =
      KernelTraits<MaskMode::kCausal, CTA_TILE_Q, MMA_Q, MMA_KV, HEAD_DIM_QK / 16,
                   HEAD_DIM_VO / 16, WARPS_Q, WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV,
                   DTypeO, float, IdType, Variant, /*ENABLE_FP4_REPACK=*/true>;
  static_assert(!KTraits::IsInvalid(), "FlashInfer rejects this shape");
  static_assert(KTraits::CTA_TILE_KV == CTA_TILE_KV);
  static constexpr uint32_t kSmemBytes = sizeof(typename KTraits::SharedStoragePaged);
  static_assert(kSmemBytes <= kSmemPerBlockOptin, "shape exceeds the per-block opt-in limit");
};

// The executor's instances' shapes (flashinfer_fa2_sink_paged.cu derives
// them from shared-memory and register limits); the "today" shapes below
// must equal them.
constexpr uint32_t executor_mma_kv(uint32_t cta_tile_q) {
  const uint32_t warps_q = get_num_warps_q(cta_tile_q);
  const uint32_t warps_kv = get_num_warps_kv(cta_tile_q);
  const uint32_t mma_q = get_num_mma_q(cta_tile_q);
  const uint32_t kv_per_mma = (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * warps_kv * sizeof(DTypeKV);
  const uint32_t fixed = cta_tile_q * HEAD_DIM_QK * sizeof(DTypeQ);
  const uint32_t per_sm = 228 * 1024;
  const uint32_t ctas = per_sm >= 2 * (fixed + kv_per_mma) ? 2 : 1;
  const uint32_t per_cta = per_sm / ctas < kSmemPerBlockOptin ? per_sm / ctas : kSmemPerBlockOptin;
  const uint32_t by_reg = 8 / mma_q;
  const uint32_t by_smem = (per_cta - fixed) / kv_per_mma;
  const uint32_t most = by_smem < by_reg ? by_smem : by_reg;
  (void)warps_q;
  return most >= 8 ? 8 : (most >= 4 ? 4 : (most >= 2 ? 2 : 1));
}
static_assert(get_num_warps_kv(16) == 4 && executor_mma_kv(16) == 2, "executor q16: w1x4_k2");
static_assert(get_num_warps_kv(64) == 1 && executor_mma_kv(64) == 8, "executor q64: w4x1_k8");
static_assert(get_num_warps_kv(128) == 1 && get_num_mma_q(128) == 2 && executor_mma_kv(128) == 4,
              "executor q128: w4x1_k4");

}  // namespace eidola_fa2_variants

#define EIDOLA_FA2_VARIANT(NAME, TILE, WQ, WKV, MKV, ANCHORED)                                 \
  using EidolaFa2Variant_##NAME = eidola_fa2_variants::Shape<TILE, WQ, WKV, MKV, ANCHORED>;    \
  extern "C" __global__ __launch_bounds__(EidolaFa2Variant_##NAME::KTraits::NUM_THREADS) void \
      eidola_fa2v_##NAME(const __grid_constant__ PagedParams params) {                          \
    extern __shared__ uint8_t smem[];                                                          \
    auto& smem_storage =                                                                       \
        reinterpret_cast<typename EidolaFa2Variant_##NAME::KTraits::SharedStoragePaged&>(smem); \
    flashinfer::BatchPrefillWithPagedKVCacheDevice<EidolaFa2Variant_##NAME::KTraits,           \
                                                   /*SAME_KV_STRIDES=*/false>(params,          \
                                                                              smem_storage);   \
  }                                                                                            \
  EIDOLA_KERNEL_META(eidola_fa2v_##NAME, 32, WQ, WKV, EidolaFa2Variant_##NAME::kSmemBytes, 1,  \
                     1, 1, sizeof(PagedParams))

// One-warp KV at 64-key tiles (the executor's q128 shape) for every query
// tile, and the four-warp shape (the executor's q16) at a 64-row query tile.
EIDOLA_FA2_VARIANT(stock_q16_w1x1_k4, 16, 1, 1, 4, false);
EIDOLA_FA2_VARIANT(stock_q64_w4x1_k4, 64, 4, 1, 4, false);
EIDOLA_FA2_VARIANT(stock_q64_w4x4_k2, 64, 4, 4, 2, false);

// Anchored: the executor's three shapes, and the three above.
EIDOLA_FA2_VARIANT(anch_q16_w1x4_k2, 16, 1, 4, 2, true);
EIDOLA_FA2_VARIANT(anch_q64_w4x1_k8, 64, 4, 1, 8, true);
EIDOLA_FA2_VARIANT(anch_q128_w4x1_k4, 128, 4, 1, 4, true);
EIDOLA_FA2_VARIANT(anch_q16_w1x1_k4, 16, 1, 1, 4, true);
EIDOLA_FA2_VARIANT(anch_q64_w4x1_k4, 64, 4, 1, 4, true);
EIDOLA_FA2_VARIANT(anch_q64_w4x4_k2, 64, 4, 4, 2, true);
