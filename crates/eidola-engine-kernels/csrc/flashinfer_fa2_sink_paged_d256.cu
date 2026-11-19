// Paged attention at head dimension 256 (Qwen3.8's full-attention layers, the
// fast profile): FlashInfer's FA2 prefill template (mma.sync tensor cores, not
// tcgen05) with its AttentionSink variant, BF16 Q/KV/O, head_dim_qk =
// head_dim_vo = 256, causal mask. The model has no sinks: the host passes a
// sink of -inf for every head, which leaves every row's running maximum and
// normaliser as they were (`exp2(-inf) = 0`, a scale of `exp2(0) = 1`).
//
// Batch invariance is the structure of `flashinfer_fa2_sink_paged.cu`'s
// global-layer instances, unchanged: one KV warp over 64-key tiles at every
// query tile, keys split at absolute multiples of the chunk size the host
// passes (1,024 keys), each chunk a work item, the partials merged in chunk
// order by the d256 merge instance below, and every entry walking its work list
// from device memory with a fixed grid.
//
// The configuration prologue is the one `nix/render-flashinfer-sink.sh`
// renders for the sink variant: its `PagedParams` and `AttentionSink` do not
// depend on the head dimension, which is a template argument of the kernel
// traits here (its `HEAD_DIM_QK` / `HEAD_DIM_VO` constants are not read).

#include "batch_prefill_config.inc"

#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/prefill.cuh>

#include "eidola_kernel.cuh"

// What a launch takes by value: FlashInfer's parameters and the device word
// holding the number of work items (`EidolaAttnParams`, as the d192/d128
// instances take it).
struct EidolaAttnParamsD256 {
  PagedParams paged;
  const uint32_t* work_items;
};

namespace eidola_fa2_d256 {

using namespace flashinfer;

constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kSmemPerBlockOptin = 227 * 1024;
constexpr uint32_t kKvTile = 64;
constexpr uint32_t kNumMmaKv = kKvTile / 16;

template <uint32_t CTA_TILE_Q, uint32_t WARPS_Q>
struct Shape {
  static constexpr uint32_t MMA_Q = CTA_TILE_Q / (16 * WARPS_Q);
  static_assert(MMA_Q * 16 * WARPS_Q == CTA_TILE_Q, "query tile is whole fragments per warp");
  using KTraits = KernelTraits<MaskMode::kCausal, CTA_TILE_Q, MMA_Q, kNumMmaKv, kHeadDim / 16,
                               kHeadDim / 16, WARPS_Q, /*NUM_WARPS_KV=*/1, POS_ENCODING_MODE,
                               DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionSink,
                               /*ENABLE_FP4_REPACK=*/true>;
  static_assert(!KTraits::IsInvalid(), "FlashInfer rejects this shape");
  static_assert(KTraits::CTA_TILE_KV == kKvTile, "one KV warp of 64-key tiles");
  static constexpr uint32_t kSmemBytes = sizeof(typename KTraits::SharedStoragePaged);
  static_assert(kSmemBytes <= kSmemPerBlockOptin, "shape exceeds the per-block opt-in limit");
};

template <typename KTraits>
__device__ __forceinline__ void run_work_items(const EidolaAttnParamsD256& args, uint8_t* smem) {
  auto& storage = *reinterpret_cast<typename KTraits::SharedStoragePaged*>(smem);
  const uint32_t n = *args.work_items;
  for (uint32_t w = blockIdx.x; w < n; w += gridDim.x) {
    BatchPrefillWithPagedKVCacheDevice<KTraits, /*SAME_KV_STRIDES=*/false>(
        args.paged, storage, threadIdx, w, blockIdx.z, gridDim.z);
    __syncthreads();
  }
}

// head_dim 256, BF16 partials: FlashInfer's VariableLengthMergeStates choice.
constexpr uint32_t kMergeVec = 16 / sizeof(DTypeO) > kHeadDim / 32 ? 16 / sizeof(DTypeO)
                                                                    : kHeadDim / 32;
constexpr uint32_t kMergeBdx = kHeadDim / kMergeVec;
constexpr uint32_t kMergeBdy = 128 / kMergeBdx;
constexpr uint32_t kMergeStages = 4;
constexpr uint32_t kMergeSmem =
    kMergeStages * kMergeBdy * kHeadDim * sizeof(DTypeO) + 128 * sizeof(float);

}  // namespace eidola_fa2_d256

#define EIDOLA_FA2_D256_ENTRY(NAME, TILE, WARPS_Q)                                           \
  using EidolaFa2D256Shape_##NAME = eidola_fa2_d256::Shape<TILE, WARPS_Q>;                   \
  extern "C" __global__ __launch_bounds__(EidolaFa2D256Shape_##NAME::KTraits::NUM_THREADS)   \
      void NAME(const __grid_constant__ EidolaAttnParamsD256 args) {                         \
    extern __shared__ uint8_t smem[];                                                        \
    eidola_fa2_d256::run_work_items<EidolaFa2D256Shape_##NAME::KTraits>(args, smem);         \
  }                                                                                          \
  EIDOLA_KERNEL_META(NAME, 32, WARPS_Q, 1, EidolaFa2D256Shape_##NAME::kSmemBytes, 1, 1, 1,   \
                     sizeof(EidolaAttnParamsD256))

EIDOLA_FA2_D256_ENTRY(eidola_fa2_sink_paged_d256_bf16_q16, 16, 1);
EIDOLA_FA2_D256_ENTRY(eidola_fa2_sink_paged_d256_bf16_q64, 64, 4);

template __global__ void flashinfer::PersistentVariableLengthMergeStatesKernel<
    eidola_fa2_d256::kMergeVec, eidola_fa2_d256::kMergeBdx, eidola_fa2_d256::kMergeBdy,
    eidola_fa2_d256::kMergeStages, DTypeO, DTypeO, IdType>(DTypeO*, float*, IdType*, DTypeO*,
                                                           float*, uint32_t, uint32_t*, uint32_t);

EIDOLA_KERNEL_META(eidola_fa2_merge_states_bf16_d256, eidola_fa2_d256::kMergeBdx,
                   eidola_fa2_d256::kMergeBdy, 1, eidola_fa2_d256::kMergeSmem, 1, 1, 1, 0);
