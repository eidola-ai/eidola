// Paged attention with learned sinks and a sliding window: FlashInfer's FA2
// prefill template (mma.sync tensor cores, not tcgen05) with its AttentionSink
// variant, BF16 Q/KV/O, head_dim_qk 192 / head_dim_vo 128, causal mask.
//
// FlashInfer serves both prefill and decode for this variant from the paged
// prefill kernel — its decode template requires equal QK and VO head dims —
// and picks the query tile from the batch's average query length. Three tiles
// are built, matching the set FlashInfer instantiates for these head dims:
//
//   q16   decode and speculative verify (one to a few query rows per request)
//   q64   short prefill chunks
//   q128  long prefill chunks
//
// The configuration prologue (`PagedParams`, the `AttentionSink` variant) is
// rendered at build time from the pinned FlashInfer tree; see
// nix/render-flashinfer-sink.sh. FlashInfer's host dispatcher chooses the KV
// tile (NUM_MMA_KV) at launch from the device's shared-memory limits; the
// same arithmetic is evaluated here at compile time for the Blackwell
// datacenter parts (228 KiB per SM, 227 KiB per block), and every resulting
// instantiation is checked against those limits.
//
// When the host splits long KV across CTAs (partition_kv), the partial
// outputs are merged by FlashInfer's PersistentVariableLengthMergeStatesKernel,
// which is instantiated here for head_dim 128 as well.

#include "batch_prefill_config.inc"

#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/prefill.cuh>

#include "eidola_kernel.cuh"

namespace eidola_fa2_sink {

using namespace flashinfer;

constexpr uint32_t kSmemPerSm = 228 * 1024;
constexpr uint32_t kSmemPerBlockOptin = 227 * 1024;

// BatchPrefillWithPagedKVCacheDispatchedImpl's tile arithmetic, specialised to
// 16-bit KV (no repack staging, no FP4 scale buffers) with HEAD_DIM_QK !=
// HEAD_DIM_VO (no shared K/V buffer) and HEAD_DIM_VO <= 256 (no VO split).
template <uint32_t CTA_TILE_Q>
struct Tile {
  static_assert(sizeof(DTypeKV) == 2 && HEAD_DIM_QK != HEAD_DIM_VO && HEAD_DIM_VO <= 256,
                "the arithmetic below covers exactly this configuration");
  static constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  static constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  static constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
  static constexpr uint32_t kKVSmemPerMmaKV =
      (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV);
  static constexpr uint32_t kFixedSmem = CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
  static constexpr uint32_t kCtasPerSm =
      kSmemPerSm >= 2 * (kFixedSmem + kKVSmemPerMmaKV) ? 2 : 1;
  static constexpr uint32_t kSmemPerCta = kSmemPerSm / kCtasPerSm < kSmemPerBlockOptin
                                              ? kSmemPerSm / kCtasPerSm
                                              : kSmemPerBlockOptin;
  static constexpr uint32_t kMaxMmaKvReg = 8 / NUM_MMA_Q;
  static constexpr uint32_t kMaxMmaKvSmem = (kSmemPerCta - kFixedSmem) / kKVSmemPerMmaKV;
  static constexpr uint32_t kMaxMmaKv = kMaxMmaKvSmem < kMaxMmaKvReg ? kMaxMmaKvSmem : kMaxMmaKvReg;
  static constexpr uint32_t NUM_MMA_KV =
      kMaxMmaKv >= 8 ? 8 : (kMaxMmaKv >= 4 ? 4 : (kMaxMmaKv >= 2 ? 2 : 1));
  static_assert(kMaxMmaKvSmem >= 1, "smallest KV tile does not fit");

  using KTraits =
      KernelTraits<MaskMode::kCausal, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, HEAD_DIM_QK / 16,
                   HEAD_DIM_VO / 16, NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV,
                   DTypeO, float, IdType, AttentionSink, /*ENABLE_FP4_REPACK=*/true>;
  static_assert(!KTraits::IsInvalid(), "FlashInfer rejects this tile");
  static constexpr uint32_t kSmemBytes = sizeof(typename KTraits::SharedStoragePaged);
  static_assert(kSmemBytes <= kSmemPerBlockOptin, "tile exceeds the per-block opt-in limit");
};

// head_dim 128, BF16 partials: FlashInfer's VariableLengthMergeStates choice.
constexpr uint32_t kMergeVec = 16 / sizeof(DTypeO) > HEAD_DIM_VO / 32 ? 16 / sizeof(DTypeO)
                                                                       : HEAD_DIM_VO / 32;
constexpr uint32_t kMergeBdx = HEAD_DIM_VO / kMergeVec;
constexpr uint32_t kMergeBdy = 128 / kMergeBdx;
constexpr uint32_t kMergeStages = 4;
constexpr uint32_t kMergeSmem =
    kMergeStages * kMergeBdy * HEAD_DIM_VO * sizeof(DTypeO) + 128 * sizeof(float);

}  // namespace eidola_fa2_sink

#define EIDOLA_FA2_SINK_ENTRY(TILE)                                                              \
  using EidolaFa2SinkQ##TILE = eidola_fa2_sink::Tile<TILE>;                                      \
  extern "C" __global__ __launch_bounds__(EidolaFa2SinkQ##TILE::KTraits::NUM_THREADS) void        \
      eidola_fa2_sink_paged_bf16_q##TILE(const __grid_constant__ PagedParams params) {           \
    extern __shared__ uint8_t smem[];                                                            \
    auto& smem_storage =                                                                         \
        reinterpret_cast<typename EidolaFa2SinkQ##TILE::KTraits::SharedStoragePaged&>(smem);     \
    flashinfer::BatchPrefillWithPagedKVCacheDevice<EidolaFa2SinkQ##TILE::KTraits,                \
                                                   /*SAME_KV_STRIDES=*/false>(params,            \
                                                                              smem_storage);     \
  }                                                                                              \
  EIDOLA_KERNEL_META(eidola_fa2_sink_paged_bf16_q##TILE, 32, EidolaFa2SinkQ##TILE::NUM_WARPS_Q,  \
                     EidolaFa2SinkQ##TILE::NUM_WARPS_KV, EidolaFa2SinkQ##TILE::kSmemBytes, 1, 1, \
                     1, sizeof(PagedParams))

EIDOLA_FA2_SINK_ENTRY(16);
EIDOLA_FA2_SINK_ENTRY(64);
EIDOLA_FA2_SINK_ENTRY(128);

template __global__ void flashinfer::PersistentVariableLengthMergeStatesKernel<
    eidola_fa2_sink::kMergeVec, eidola_fa2_sink::kMergeBdx, eidola_fa2_sink::kMergeBdy,
    eidola_fa2_sink::kMergeStages, DTypeO, DTypeO, IdType>(DTypeO*, float*, IdType*, DTypeO*,
                                                           float*, uint32_t, uint32_t*, uint32_t);

EIDOLA_KERNEL_META(eidola_fa2_merge_states_bf16_d128, eidola_fa2_sink::kMergeBdx,
                   eidola_fa2_sink::kMergeBdy, 1, eidola_fa2_sink::kMergeSmem, 1, 1, 1, 0);
