// Paged attention with learned sinks and a sliding window: FlashInfer's FA2
// prefill template (mma.sync tensor cores, not tcgen05) with its AttentionSink
// variant, BF16 Q/KV/O, head_dim_qk 192 / head_dim_vo 128, causal mask.
//
// Batch invariance. A query row's output must not depend on the rows it is
// launched with, so every row's softmax over its keys is reduced in one
// order, fixed by the row's own position:
//
//   - One KV warp and 64-key tiles (NUM_WARPS_KV 1, NUM_MMA_KV 4) at every
//     query tile. FlashInfer's dispatcher would pick the KV shape from the
//     query tile (four KV warps over 128-key tiles at 16 rows, one warp over
//     128-key tiles at 64), which reduces a row in another order whenever a
//     step's longest query run moves it to another tile. Within one KV shape
//     each query fragment runs the same code, so the query tile (16, 64 or
//     128 rows, picked by the host for speed) does not change a row's
//     arithmetic.
//   - Sliding layers' tiles at absolute positions (AnchoredSink below). The
//     kernel starts a query tile's keys at its first row's first visible key,
//     so a row's tile boundaries would move with the rows it shares a tile
//     with (a verify run, a chunk's start). The host lists a sliding request's
//     pages from an anchor (a fixed origin plus a multiple of 64 positions),
//     and the variant moves every query tile's start down to a multiple of 64
//     keys from it, masking the keys before each row's true window itself.
//   - Global layers split their keys at absolute multiples of the chunk size
//     the host passes (kv_chunk_size_ptr, 1,024 keys), each chunk a work item
//     of its own, merged by PersistentVariableLengthMergeStatesKernel below in
//     chunk order. AttentionSink adds the sink to chunk 0's state only
//     (kv_tile_idx == 0), so the merge carries it into each row exactly once.
//     The host splits a request at multiples of the chunk size of its query
//     positions, so every row of a work item sees at least one key of every
//     chunk the item covers, and a row's merge reads exactly its own chunks.
//
// Work items. Each CTA walks the work list from its blockIdx.x in steps of
// gridDim.x up to a count read from device memory, so a captured launch
// with a fixed grid serves a step whose work list (global layers' chunk
// count grows with the context) is device data. A work item is computed by
// the same code whichever CTA runs it.
//
// The configuration prologue (`PagedParams`, the `AttentionSink` variant) is
// rendered at build time from the pinned FlashInfer tree; see
// nix/render-flashinfer-sink.sh. Every instantiation is checked against the
// Blackwell datacenter parts' per-block shared-memory limit (227 KiB).

#include "batch_prefill_config.inc"

#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/prefill.cuh>

#include "eidola_kernel.cuh"

// What a launch takes by value: FlashInfer's parameters and the device word
// holding the number of work items.
struct EidolaAttnParams {
  PagedParams paged;
  const uint32_t* work_items;
};

namespace eidola_fa2_sink {

using namespace flashinfer;

constexpr uint32_t kSmemPerBlockOptin = 227 * 1024;
// Keys per KV tile: one KV warp of four 16-key MMA fragments.
constexpr uint32_t kKvTile = 64;
constexpr uint32_t kNumMmaKv = kKvTile / 16;

// The parameters one work item is computed with: the launch's, and the item.
struct ItemParams : PagedParams {
  uint32_t item;
};

// AttentionSink with every query tile's KV tiles anchored at a multiple of
// CTA_TILE_KV from the first listed key. The kernel starts a query tile's
// keys at sub_if_greater_or_zero(kv_len + q0, qo_len + window_left) for the
// tile's first row q0; this variant reports a window widened just enough
// that the start lands on the multiple of CTA_TILE_KV at or below that row's
// first visible key. Keys before a row's window, in the widened leading tile
// or in any tile the kernel does not mask, are set to -inf in
// LogitsTransform (the kernel calls the mask only on the iterations it
// predicts need it): a fully masked tile leaves a row's (m, d, o) exactly as
// they were.
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
    const uint32_t tile = static_cast<uint32_t>(params.qo_tile_indices[params.item]);
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

  // AttentionSink's sink and output, unchanged. The kernel passes update_m_d
  // its work item's KV chunk (0 for every work item without a split), so the
  // sink enters each row exactly once wherever an anchored start begins.
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

// One query tile: WARPS_Q query warps of MMA_Q 16-row fragments each, over
// the one KV shape.
template <uint32_t CTA_TILE_Q, uint32_t WARPS_Q, bool ANCHORED>
struct Shape {
  static constexpr uint32_t MMA_Q = CTA_TILE_Q / (16 * WARPS_Q);
  static_assert(MMA_Q * 16 * WARPS_Q == CTA_TILE_Q, "query tile is whole fragments per warp");
  using Variant =
      std::conditional_t<ANCHORED, AnchoredSink<CTA_TILE_Q, kKvTile>, AttentionSink>;
  using KTraits =
      KernelTraits<MaskMode::kCausal, CTA_TILE_Q, MMA_Q, kNumMmaKv, HEAD_DIM_QK / 16,
                   HEAD_DIM_VO / 16, WARPS_Q, /*NUM_WARPS_KV=*/1, POS_ENCODING_MODE, DTypeQ,
                   DTypeKV, DTypeO, float, IdType, Variant, /*ENABLE_FP4_REPACK=*/true>;
  static_assert(!KTraits::IsInvalid(), "FlashInfer rejects this shape");
  static_assert(KTraits::CTA_TILE_KV == kKvTile, "one KV warp of 64-key tiles");
  static constexpr uint32_t kSmemBytes = sizeof(typename KTraits::SharedStoragePaged);
  static_assert(kSmemBytes <= kSmemPerBlockOptin, "shape exceeds the per-block opt-in limit");
};

// Every work item from blockIdx.x up to the count, gridDim.x apart.
template <typename KTraits>
__device__ __forceinline__ void run_work_items(const EidolaAttnParams& args, uint8_t* smem) {
  auto& storage = *reinterpret_cast<typename KTraits::SharedStoragePaged*>(smem);
  const uint32_t n = *args.work_items;
  for (uint32_t w = blockIdx.x; w < n; w += gridDim.x) {
    const ItemParams params{args.paged, w};
    BatchPrefillWithPagedKVCacheDevice<KTraits, /*SAME_KV_STRIDES=*/false>(
        params, storage, threadIdx, w, blockIdx.z, gridDim.z);
    // The next item reuses the shared memory this one read.
    __syncthreads();
  }
}

// head_dim 128, BF16 partials: FlashInfer's VariableLengthMergeStates choice.
constexpr uint32_t kMergeVec = 16 / sizeof(DTypeO) > HEAD_DIM_VO / 32 ? 16 / sizeof(DTypeO)
                                                                       : HEAD_DIM_VO / 32;
constexpr uint32_t kMergeBdx = HEAD_DIM_VO / kMergeVec;
constexpr uint32_t kMergeBdy = 128 / kMergeBdx;
constexpr uint32_t kMergeStages = 4;
constexpr uint32_t kMergeSmem =
    kMergeStages * kMergeBdy * HEAD_DIM_VO * sizeof(DTypeO) + 128 * sizeof(float);

}  // namespace eidola_fa2_sink

#define EIDOLA_FA2_SINK_ENTRY(NAME, TILE, WARPS_Q, ANCHORED)                                   \
  using EidolaFa2Shape_##NAME = eidola_fa2_sink::Shape<TILE, WARPS_Q, ANCHORED>;               \
  extern "C" __global__ __launch_bounds__(EidolaFa2Shape_##NAME::KTraits::NUM_THREADS) void   \
      NAME(const __grid_constant__ EidolaAttnParams args) {                                    \
    extern __shared__ uint8_t smem[];                                                          \
    eidola_fa2_sink::run_work_items<EidolaFa2Shape_##NAME::KTraits>(args, smem);               \
  }                                                                                            \
  EIDOLA_KERNEL_META(NAME, 32, WARPS_Q, 1, EidolaFa2Shape_##NAME::kSmemBytes, 1, 1, 1,         \
                     sizeof(EidolaAttnParams))

// Global layers (split KV, sink in chunk 0) and sliding layers (anchored), at
// query tiles of 16, 64 and 128 rows.
EIDOLA_FA2_SINK_ENTRY(eidola_fa2_sink_paged_bf16_q16, 16, 1, false);
EIDOLA_FA2_SINK_ENTRY(eidola_fa2_sink_paged_bf16_q64, 64, 4, false);
EIDOLA_FA2_SINK_ENTRY(eidola_fa2_sink_paged_bf16_q128, 128, 4, false);
EIDOLA_FA2_SINK_ENTRY(eidola_fa2_sink_paged_anchored_bf16_q16, 16, 1, true);
EIDOLA_FA2_SINK_ENTRY(eidola_fa2_sink_paged_anchored_bf16_q64, 64, 4, true);
EIDOLA_FA2_SINK_ENTRY(eidola_fa2_sink_paged_anchored_bf16_q128, 128, 4, true);

template __global__ void flashinfer::PersistentVariableLengthMergeStatesKernel<
    eidola_fa2_sink::kMergeVec, eidola_fa2_sink::kMergeBdx, eidola_fa2_sink::kMergeBdy,
    eidola_fa2_sink::kMergeStages, DTypeO, DTypeO, IdType>(DTypeO*, float*, IdType*, DTypeO*,
                                                           float*, uint32_t, uint32_t*, uint32_t);

// The merge kernel's name is mangled; `meta_aliases` in kernels.json binds
// this record to it.
EIDOLA_KERNEL_META(eidola_fa2_merge_states_bf16_d128, eidola_fa2_sink::kMergeBdx,
                   eidola_fa2_sink::kMergeBdy, 1, eidola_fa2_sink::kMergeSmem, 1, 1, 1, 0);
