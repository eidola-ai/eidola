// Dense FP8 GEMM with F32 block scales on Blackwell tcgen05 (CUTLASS's
// KernelScheduleSm100Blockwise), the shape of every FP8-quantised dense
// projection in the model family:
//
//   D[m, n] (bf16) = sum_k (A[m, k] * sfa[m, k / 128]) * (B[n, k] * sfb[n / 128, k / 128])
//
//   A    fp8 e4m3, row-major [M, K]     activations, dynamically quantised
//   sfa  f32, one scale per (row, 128-wide K block)          (1 x 128)
//   B    fp8 e4m3, column-major [K, N]  weights, i.e. K-major rows of W[n, :]
//   sfb  f32, one scale per (128 rows of N, 128-wide K block) (128 x 128)
//
// The scales are arbitrary f32, not powers of two, which is why this kernel and
// not a UE8M0 block-scaled one carries the checkpoint's dense FP8 weights.
//
// The kernel is CUTLASS's own GemmUniversal; the only thing written here is an
// `extern "C"` entry point equivalent to `cutlass::device_kernel`, so the
// symbol a host loads by name does not depend on C++ mangling. The by-value
// `Params` is what `GemmUniversalAdapter::initialize` builds on the host
// (including the TMA descriptors); its size is published in the meta record.

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

namespace eidola_fp8_blockwise {

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// No C operand: D = A·B with block scales, no residual.
using ElementC = void;
using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ElementCompute = float;

// One SM per 128x128x128 MMA tile, no cluster.
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
    LayoutD, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>,
    AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue,
                                                        void>;

}  // namespace eidola_fp8_blockwise

using EidolaFp8BlockwiseGemm = eidola_fp8_blockwise::GemmKernel;

extern "C" __global__ void __launch_bounds__(EidolaFp8BlockwiseGemm::MaxThreadsPerBlock,
                                             EidolaFp8BlockwiseGemm::MinBlocksPerMultiprocessor)
    eidola_cutlass_fp8_blockwise_gemm_bf16(
        CUTLASS_GRID_CONSTANT EidolaFp8BlockwiseGemm::Params const params) {
  extern __shared__ char smem[];
  EidolaFp8BlockwiseGemm op;
  op(params, smem);
}

EIDOLA_KERNEL_META(eidola_cutlass_fp8_blockwise_gemm_bf16, EidolaFp8BlockwiseGemm::MaxThreadsPerBlock,
                   1, 1, EidolaFp8BlockwiseGemm::SharedStorageSize, 1, 1, 1,
                   sizeof(EidolaFp8BlockwiseGemm::Params));
