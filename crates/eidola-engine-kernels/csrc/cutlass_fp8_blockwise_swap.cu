// The blockwise FP8 GEMM of `cutlass_fp8_blockwise_gemm.cu` with its operands
// swapped, for few activation rows (Qwen3.8's fast profile at decode sizes):
//
//   D[m, n] (bf16) = sum_k (A[m, k] * sfa[m, k / 128]) * (W[n, k] * sfw[n / 128, k / 128])
//
// computed as the transposed product Dᵀ = W · Aᵀ, so the weight's rows fill
// the 128-row side of the tcgen05 MMA and the activation rows its narrow
// side (16, 32 or 64). At a few rows the unswapped kernel spends each
// 128-wide K block promoting a 128 × 128 accumulator of which one row is
// real, and holds half its shared memory for activation rows that are not
// there; here a K block promotes 128 × 16 (or 32, 64), and a pipeline stage
// is the weight block and a sliver of activations, so more stages stay in
// flight. This is the shape upstream serving stacks dispatch at small M.
//
//   CUTLASS A = W     fp8 e4m3, row-major [N, K]            (the weight as stored)
//   CUTLASS B = A     fp8 e4m3, column-major [K, M]         (activation rows, K-major)
//   CUTLASS D         bf16, column-major [N, M]             (= row-major [M, N])
//   CUTLASS SFA       f32 per 128 × 128 weight block, stored [K / 128][N / 128]
//   CUTLASS SFB       f32 per row and 128-wide K block, stored [K / 128][M]
//
// The scale arrays are the unswapped kernel's SFB and SFA, byte for byte, so
// the executor's buffers serve either kernel.
//
// Per output element the arithmetic is the unswapped kernel's: per 128-wide
// K block the MMA's dot of one weight row and one activation row from zero,
// then `acc += (s_w · s_a) · dot` in K block order (the mainloop's scale
// product commutes exactly), then the identity epilogue and the BF16
// rounding. Whether the tensor core's sum of a row pair is the same whichever
// operand is which, and whatever the MMA's N, is a fact of the hardware, not
// of this source: `eidola-engine-cuda`'s `tests/qwen_ops.rs`
// (`fast_fp8_configs_agree`) asserts every output's bits equal across every
// instance a row can be routed to before the executor mixes them.
//
// Three instances, one per activation tile (16, 32, 64 rows), each an
// `extern "C"` entry equivalent to `cutlass::device_kernel` as in the
// unswapped source.

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

namespace eidola_fp8_blockwise_swap {

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = void;
using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::ColumnMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ElementCompute = float;

using ClusterShape_MNK = Shape<_1, _1, _1>;

// Weight scales per 128 × 128 (CUTLASS's M × K), activation scales per row
// and 128 of K (N × K), both stored with their M or N index contiguous.
using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

template <int TileRows>
struct Instance {
  using MmaTileShape_MNK = Shape<_128, Int<TileRows>, _128>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutD, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
};

}  // namespace eidola_fp8_blockwise_swap

using EidolaFp8Swap16 = eidola_fp8_blockwise_swap::Instance<16>::GemmKernel;
using EidolaFp8Swap32 = eidola_fp8_blockwise_swap::Instance<32>::GemmKernel;
using EidolaFp8Swap64 = eidola_fp8_blockwise_swap::Instance<64>::GemmKernel;

#define EIDOLA_FP8_SWAP_ENTRY(entry, Kernel)                                                  \
  extern "C" __global__ void __launch_bounds__(Kernel::MaxThreadsPerBlock,                    \
                                               Kernel::MinBlocksPerMultiprocessor)            \
      entry(CUTLASS_GRID_CONSTANT Kernel::Params const params) {                              \
    extern __shared__ char smem[];                                                            \
    Kernel op;                                                                                \
    op(params, smem);                                                                         \
  }                                                                                           \
  EIDOLA_KERNEL_META(entry, Kernel::MaxThreadsPerBlock, 1, 1, Kernel::SharedStorageSize, 1, 1, \
                     1, sizeof(Kernel::Params))

EIDOLA_FP8_SWAP_ENTRY(eidola_cutlass_fp8_blockwise_swap16_bf16, EidolaFp8Swap16);
EIDOLA_FP8_SWAP_ENTRY(eidola_cutlass_fp8_blockwise_swap32_bf16, EidolaFp8Swap32);
EIDOLA_FP8_SWAP_ENTRY(eidola_cutlass_fp8_blockwise_swap64_bf16, EidolaFp8Swap64);
