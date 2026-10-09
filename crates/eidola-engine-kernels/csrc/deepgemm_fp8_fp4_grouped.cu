// Routed-expert grouped GEMM: FP8 activations x MXFP4 weights on Blackwell
// tcgen05, instantiated ahead of time from DeepGEMM's device headers. DeepGEMM
// normally generates exactly this translation unit at runtime and compiles it
// with its JIT; here the instantiation is fixed at build time and nothing on
// the host ever invokes a compiler.
//
//   A    fp8 e4m3, K-major, per-(row, 128-wide K block) UE8M0 scales
//   B    MXFP4 (e2m1, two per byte), K-major, per-(row, 32-wide K block)
//        UE8M0 scales — the checkpoint's expert layout
//   D    bf16
//
// Template arguments mirror what DeepGEMM's own host wrapper
// (`sm100_m_grouped_fp8_fp4_gemm_{masked,contiguous}_1d1d`) and its SM100
// heuristics select for these shapes, with `compiled_dims = "nk"` (M stays a
// runtime argument):
//
//   swap_ab, BLOCK_M/N/K = 128/128/128, cluster 2 multicast on A (N/128 is
//   even), LOAD_BLOCK_M 64, STORE_BLOCK_M 16, 128-byte swizzles everywhere,
//   8 pipeline stages, 2 TMA store stages, 128 + 128 threads.
//
// Shared memory, per DeepGEMM's pipeline accounting:
//   C/D 16*128*2B*2 = 8192; barriers 32*8*3 + 2*8*3 + 8 = 824; TMEM ptr 4;
//   per stage A 64*128 + B 128*128 + SFA 512 + SFB 512 = 25600; 8 stages
//   -> 9020 + 204800 = 213820 bytes.
//
// The two layouts:
//
//   masked   decode-sized batches: A is [256][M][K], `grouped_layout[g]` the
//            number of valid rows of group g.
//   psum     prefill-sized batches (DeepGEMM's `use_psum_layout`): every
//            group's rows back to back in group order, group g's run starting
//            at the previous run's end rounded up to 128 rows;
//            `grouped_layout[g]` is the end of group g's rows (a prefix sum).
//            The scheduler visits only each group's own 128-row blocks, and
//            with `ensure_zero_padding` false it narrows a group's last block
//            to its rows rounded up to 16 (the swap-AB UMMA N), so empty
//            experts cost nothing and a sparse expert one narrow block. The
//            plain contiguous layout (`grouped_layout[row]` per row, -1 for
//            padding) is not used: on SM100 its scheduler runs every block of
//            M through the tensor cores, padding blocks included (against
//            group 0's weights), so its cost follows the layout's worst-case
//            bound rather than the routed rows.
//
// The persistent scheduler bakes the SM count into the instantiation, so a
// kernel built for 148 SMs must be launched with a 148-block grid on a part
// with at least that many SMs.
//
// Entry points are DeepGEMM's own templated kernels, so their symbols are
// C++-mangled; the manifest records each mangled name with its demangled form.

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kNumSms = 148;
constexpr uint32_t kGroups = 256;
// Hidden 4096; expert intermediate 2048, gate and up fused along N.
constexpr uint32_t kGateUpN = 4096, kGateUpK = 4096;
constexpr uint32_t kDownN = 4096, kDownK = 2048;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kSmemBytes = 213820;
constexpr uint32_t kCluster = 2;

#define EIDOLA_DEEPGEMM_INSTANCE(SHAPE_N, SHAPE_K, TYPE)                                                   \
  template __global__ void sm100_fp8_fp4_gemm_1d1d_impl<                                         \
      cute::UMMA::Major::K, cute::UMMA::Major::K, /*gran_k A, B, k_alignment=*/128, 32, 128,     \
      /*SHAPE_M, N, K=*/0, SHAPE_N, SHAPE_K, /*BLOCK_M, N, K=*/128, 128, 128, kGroups,           \
      /*swizzle A, B, CD=*/128, 128, 128, /*stages, TMA store stages=*/8, 2,                     \
      /*non-epilogue, epilogue threads=*/128, 128, /*multicast, on A=*/kCluster, true, kNumSms,  \
      /*swap_ab, ensure_zero_padding=*/true, false, TYPE, /*with_accumulation=*/false,           \
      cutlass::float_e4m3_t, cutlass::detail::float_e2m1_unpacksmem_t, cutlass::bfloat16_t,      \
      epilogue::operators::Identity>(                                                         \
      int*, uint32_t, uint32_t, uint32_t,                                                        \
      const __grid_constant__ epilogue::operators::Identity,                                     \
      const __grid_constant__ cute::TmaDescriptor, const __grid_constant__ cute::TmaDescriptor,  \
      const __grid_constant__ cute::TmaDescriptor, const __grid_constant__ cute::TmaDescriptor,  \
      const __grid_constant__ cute::TmaDescriptor)

}  // namespace

// Explicit instantiations, spelled out with the full parameter list so a
// signature change upstream fails here rather than at launch. DeepGEMM
// declares its kernels `__global__ static` (CUTLASS_GLOBAL), so the entries
// keep internal linkage (STB_LOCAL in the cubin's symbol table).
namespace deep_gemm {
EIDOLA_DEEPGEMM_INSTANCE(kGateUpN, kGateUpK, GemmType::MGroupedMasked);
EIDOLA_DEEPGEMM_INSTANCE(kDownN, kDownK, GemmType::MGroupedMasked);
EIDOLA_DEEPGEMM_INSTANCE(kGateUpN, kGateUpK, GemmType::MGroupedContiguousWithPsumLayout);
EIDOLA_DEEPGEMM_INSTANCE(kDownN, kDownK, GemmType::MGroupedContiguousWithPsumLayout);
}  // namespace deep_gemm

// The entries' names are mangled, so each launch-contract record is bound to
// its entry explicitly: `meta_aliases` in kernels.json maps every record below
// to the mangled symbol it describes, and the build refuses an image whose
// entries and records do not correspond one to one.
EIDOLA_KERNEL_META(eidola_deepgemm_fp8_fp4_masked_gate_up, kThreads, 1, 1, kSmemBytes, kCluster, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_fp8_fp4_masked_down, kThreads, 1, 1, kSmemBytes, kCluster, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_fp8_fp4_psum_gate_up, kThreads, 1, 1, kSmemBytes, kCluster, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_deepgemm_fp8_fp4_psum_down, kThreads, 1, 1, kSmemBytes, kCluster, 1, 1, 0);
