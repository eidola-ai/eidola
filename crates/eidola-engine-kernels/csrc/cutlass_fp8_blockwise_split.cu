// The blockwise FP8 GEMMs of `cutlass_fp8_blockwise_gemm.cu` and
// `cutlass_fp8_blockwise_swap.cu` with an f32 output and no rounding, for
// Qwen3.8's fast profile to run a GEMM split over K:
//
//   P[s][m, n] (f32) = sum_{k in split s} (A[m, k] * sfa[m, k / 128]) * (W[n, k] * sfw[n / 128, k / 128])
//
// The host runs each split as one batch of CUTLASS's problem (its L mode):
// split `s` is K columns `s · K/S .. (s + 1) · K/S` of every row of A and of
// W (an L stride of K/S elements inside the rows, a row stride of K), the
// scale blocks `s · K/(128 S) ..` (the scale arrays are stored with their K
// block outermost, so a split's scales are one contiguous run), and its
// output is the f32 array `P[s]`. One launch covers every split, so a GEMM
// of few output tiles (N = 5,120: 40 tiles of 128) puts S times as many CTAs
// on the device as the unsplit kernel does. Qwen's `eidola_qwen_fast_split_sum`
// then adds the S partial rows in split order and rounds once to BF16.
//
// Per output element and split the arithmetic is the BF16-output kernels':
// per 128-wide K block the MMA's dot from zero, then `acc += (s_a · s_w) ·
// dot` in K block order from zero, then the identity epilogue (`alpha` = 1),
// here without the BF16 rounding. The activation-tile instances are the
// swapped kernels' (weight rows on the MMA's 128-row side, activation rows
// 16, 32 or 64 on its narrow side); the wide instance is the unswapped
// kernel's (128 × 128 tiles). Every instance computes each split's sums of
// a row in the same order, so a row's partials, and therefore its sum, are
// the same whichever instance its row count routes it to and whatever rows
// share its batch: `eidola-engine-cuda`'s `tests/qwen_ops.rs`
// (`fast_fp8_split_forms`) asserts it bit for bit.
//
// Four instances, each an `extern "C"` entry equivalent to
// `cutlass::device_kernel` as in the BF16-output sources.

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "eidola_kernel.cuh"

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "the blockwise FP8 GEMM needs a tcgen05 target (sm_100a, sm_103a, or sm_100f)"
#endif

namespace eidola_fp8_blockwise_split {

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = void;
using ElementD = float;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ElementCompute = float;

using ClusterShape_MNK = Shape<_1, _1, _1>;

// The swapped instances: CUTLASS's A the weight (scales per 128 × 128, its
// M × K), its B the activation rows (scales per row and 128 of K), D
// column-major `[N, M]`, which is `[M, N]` row-major.
using SwapScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128>;
using SwapLayoutSFA = decltype(SwapScaleConfig::deduce_layoutSFA());
using SwapLayoutSFB = decltype(SwapScaleConfig::deduce_layoutSFB());

template <int TileRows>
struct Swap {
  using MmaTileShape_MNK = Shape<_128, Int<TileRows>, _128>;
  using LayoutD = cutlass::layout::ColumnMajor;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutD, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, SwapLayoutSFA>, AlignmentA, ElementB,
      cute::tuple<LayoutB, SwapLayoutSFB>, AlignmentB, ElementAccumulator, MmaTileShape_MNK,
      ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
};

// The wide instance: the unswapped kernel's operands (A the activation rows,
// B the weight), 128 × 128 tiles, D row-major `[M, N]`.
using WideScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128>;
using WideLayoutSFA = decltype(WideScaleConfig::deduce_layoutSFA());
using WideLayoutSFB = decltype(WideScaleConfig::deduce_layoutSFB());

struct Wide {
  using MmaTileShape_MNK = Shape<_128, _128, _128>;
  using LayoutD = cutlass::layout::RowMajor;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutD, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, WideLayoutSFA>, AlignmentA, ElementB,
      cute::tuple<LayoutB, WideLayoutSFB>, AlignmentB, ElementAccumulator, MmaTileShape_MNK,
      ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
};

}  // namespace eidola_fp8_blockwise_split

using EidolaFp8Split16 = eidola_fp8_blockwise_split::Swap<16>::GemmKernel;
using EidolaFp8Split32 = eidola_fp8_blockwise_split::Swap<32>::GemmKernel;
using EidolaFp8Split64 = eidola_fp8_blockwise_split::Swap<64>::GemmKernel;
using EidolaFp8SplitWide = eidola_fp8_blockwise_split::Wide::GemmKernel;

#define EIDOLA_FP8_SPLIT_ENTRY(entry, Kernel)                                                 \
  extern "C" __global__ void __launch_bounds__(Kernel::MaxThreadsPerBlock,                    \
                                               Kernel::MinBlocksPerMultiprocessor)            \
      entry(CUTLASS_GRID_CONSTANT Kernel::Params const params) {                              \
    extern __shared__ char smem[];                                                            \
    Kernel op;                                                                                \
    op(params, smem);                                                                         \
  }                                                                                           \
  EIDOLA_KERNEL_META(entry, Kernel::MaxThreadsPerBlock, 1, 1, Kernel::SharedStorageSize, 1, 1, \
                     1, sizeof(Kernel::Params))

EIDOLA_FP8_SPLIT_ENTRY(eidola_cutlass_fp8_blockwise_split16_f32, EidolaFp8Split16);
EIDOLA_FP8_SPLIT_ENTRY(eidola_cutlass_fp8_blockwise_split32_f32, EidolaFp8Split32);
EIDOLA_FP8_SPLIT_ENTRY(eidola_cutlass_fp8_blockwise_split64_f32, EidolaFp8Split64);
EIDOLA_FP8_SPLIT_ENTRY(eidola_cutlass_fp8_blockwise_split128_f32, EidolaFp8SplitWide);
