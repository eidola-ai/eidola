// Dense BF16 GEMM with an f32 result on Blackwell tcgen05: CUTLASS's
// GemmUniversal with the collective builders' default (warp-specialized TMA)
// schedule. The model's BF16 projections run on it: attention `o_proj`, the
// MTP `eh_proj`, and the `lm_head` (whose logits stay f32 for sampling).
//
//   D[m, n] (f32) = alpha * sum_k A[m, k] * B[n, k]
//
//   A    bf16, row-major [M, K]      activations
//   B    bf16, column-major [K, N]   weights, i.e. K-major rows of W[n, :]
//   D    f32, row-major [M, N]
//
// `alpha` is a runtime scalar: attention's value scale is applied here, in
// f32, rather than folded into the BF16 `o_proj` weights (0.707 * W is not
// representable in BF16). There is no C operand; residual adds are separate
// f32 kernels.
//
// As for the FP8 GEMM, the entry point is an `extern "C"` equivalent of
// `cutlass::device_kernel`, and its by-value `Params` (TMA descriptors
// included) is built by the host to the size the meta record publishes.

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "eidola_kernel.cuh"

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "the BF16 GEMM needs a tcgen05 target (sm_100a, sm_103a, or sm_100f)"
#endif

namespace eidola_bf16_gemm {

using namespace cute;

using ElementA = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = void;
using ElementD = float;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ElementCompute = float;

// One SM per 128x128x64 MMA tile, no cluster: the same launch shape as the
// FP8 GEMM, so one host path serves both.
using MmaTileShape_MNK = Shape<_128, _128, _64>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
    LayoutD, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB,
    LayoutB, AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue,
                                                        void>;

}  // namespace eidola_bf16_gemm

using EidolaBf16Gemm = eidola_bf16_gemm::GemmKernel;

extern "C" __global__ void __launch_bounds__(EidolaBf16Gemm::MaxThreadsPerBlock,
                                             EidolaBf16Gemm::MinBlocksPerMultiprocessor)
    eidola_cutlass_bf16_gemm_f32(CUTLASS_GRID_CONSTANT EidolaBf16Gemm::Params const params) {
  extern __shared__ char smem[];
  EidolaBf16Gemm op;
  op(params, smem);
}

EIDOLA_KERNEL_META(eidola_cutlass_bf16_gemm_f32, EidolaBf16Gemm::MaxThreadsPerBlock, 1, 1,
                   EidolaBf16Gemm::SharedStorageSize, 1, 1, 1, sizeof(EidolaBf16Gemm::Params));
