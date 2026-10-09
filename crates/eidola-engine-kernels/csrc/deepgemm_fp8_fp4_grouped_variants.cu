// Experimental tilings of the routed-expert grouped GEMM, for the kernel
// bench only: the executor loads `deepgemm_fp8_fp4_grouped` and never this
// image. Each variant is the executor's instance (same upstream template, the
// psum layout, FP8 A x MXFP4 B, BF16 out, swap-AB, 128-byte swizzles, 148
// SMs, 2 TMA store stages, 128 + 128 threads) with a few template arguments
// changed, so that the bench can time and compare them on the same data:
//
//   name      BLOCK_M  cluster  stages  smem (DeepGEMM's accounting)
//   m64s8          64        2       8  9020 + 8 * 21504  = 181052
//   m64s10         64        2      10  9020 + 10 * 21504 = 224060
//   m32s11         32        2      11  9020 + 11 * 19456 = 223036
//   m64s8c1        64        1       8  9020 + 8 * 25600  = 213820
//
// Per stage a CTA holds LOAD_BLOCK_M = BLOCK_M / cluster rows of A (FP8,
// 128 of K), 128 rows of B (FP4 unpacked to one byte by the TMA: 16 KiB),
// and 512 bytes each of SFA and SFB; the fixed part is the swap-AB C/D
// staging (16 * 128 * 2 B * 2 stores = 8192), the barriers DeepGEMM reserves
// for 32 stages (824) and the TMEM pointer (4). The stage counts are the most
// that fit the 232,448 bytes an SM100 block can opt into (DeepGEMM's
// `get_pipeline_config`), except m64s8, which holds the executor's count so
// the bench can separate a smaller BLOCK_M from more stages, and m64s8c1,
// which is the most a 1-CTA instance with BLOCK_M 64 fits.
//
// A smaller BLOCK_M halves (or quarters) the A tile each stage carries and
// leaves room for more stages, so more weight bytes are in flight per SM;
// it costs a second block per expert for every expert with more than BLOCK_M
// rows (a second pass over its weights, from L2 if it is still there). With a
// cluster of 1 each CTA runs its own UMMA (M = 128) and loads its own A, and
// no CTA waits on a peer's weight tile.
//
// The psum layout's runs start on multiples of BLOCK_M (the scheduler aligns
// each group's start to it), so a variant with BLOCK_M 64 or 32 needs a
// layout placed with that alignment; the executor's placement (128) is not
// one. The bench places its own.

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kNumSms = 148;
constexpr uint32_t kGroups = 256;
// Hidden 4096; expert intermediate 2048, gate and up fused along N.
constexpr uint32_t kGateUpN = 4096, kGateUpK = 4096;
constexpr uint32_t kDownN = 4096, kDownK = 2048;
constexpr uint32_t kThreads = 256;

#define EIDOLA_DEEPGEMM_VARIANT(SHAPE_N, SHAPE_K, BLOCK_M, STAGES, CLUSTER)                          \
  template __global__ void sm100_fp8_fp4_gemm_1d1d_impl<                                         \
      cute::UMMA::Major::K, cute::UMMA::Major::K, /*gran_k A, B, k_alignment=*/128, 32, 128,     \
      /*SHAPE_M, N, K=*/0, SHAPE_N, SHAPE_K, /*BLOCK_M, N, K=*/BLOCK_M, 128, 128, kGroups,       \
      /*swizzle A, B, CD=*/128, 128, 128, /*stages, TMA store stages=*/STAGES, 2,                \
      /*non-epilogue, epilogue threads=*/128, 128, /*multicast, on A=*/CLUSTER, true, kNumSms,   \
      /*swap_ab, ensure_zero_padding=*/true, false,                                              \
      GemmType::MGroupedContiguousWithPsumLayout, /*with_accumulation=*/false,                   \
      cutlass::float_e4m3_t, cutlass::detail::float_e2m1_unpacksmem_t, cutlass::bfloat16_t,      \
      epilogue::operators::Identity>(                                                            \
      int*, uint32_t, uint32_t, uint32_t,                                                        \
      const __grid_constant__ epilogue::operators::Identity,                                     \
      const __grid_constant__ cute::TmaDescriptor, const __grid_constant__ cute::TmaDescriptor,  \
      const __grid_constant__ cute::TmaDescriptor, const __grid_constant__ cute::TmaDescriptor,  \
      const __grid_constant__ cute::TmaDescriptor)

#define EIDOLA_DEEPGEMM_VARIANT_PAIR(BLOCK_M, STAGES, CLUSTER)       \
  EIDOLA_DEEPGEMM_VARIANT(kGateUpN, kGateUpK, BLOCK_M, STAGES, CLUSTER); \
  EIDOLA_DEEPGEMM_VARIANT(kDownN, kDownK, BLOCK_M, STAGES, CLUSTER)

}  // namespace

// Explicit instantiations with the full parameter list, as in
// deepgemm_fp8_fp4_grouped.cu; the entries keep internal linkage.
namespace deep_gemm {
EIDOLA_DEEPGEMM_VARIANT_PAIR(64, 8, 2);
EIDOLA_DEEPGEMM_VARIANT_PAIR(64, 10, 2);
EIDOLA_DEEPGEMM_VARIANT_PAIR(32, 11, 2);
EIDOLA_DEEPGEMM_VARIANT_PAIR(64, 8, 1);
}  // namespace deep_gemm

// Launch contracts, bound to the mangled entries by `meta_aliases` in
// kernels.json.
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m64s8_gate_up, kThreads, 1, 1, 181052, 2, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m64s8_down, kThreads, 1, 1, 181052, 2, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m64s10_gate_up, kThreads, 1, 1, 224060, 2, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m64s10_down, kThreads, 1, 1, 224060, 2, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m32s11_gate_up, kThreads, 1, 1, 223036, 2, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m32s11_down, kThreads, 1, 1, 223036, 2, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m64s8c1_gate_up, kThreads, 1, 1, 213820, 1, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_variant_m64s8c1_down, kThreads, 1, 1, 213820, 1, 1, 1, 0);
