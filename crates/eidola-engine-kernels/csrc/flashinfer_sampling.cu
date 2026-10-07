// Token sampling from FlashInfer's sampling.cuh: top-k, top-p, joint top-k +
// top-p, and chain speculative sampling (verify k draft tokens against the
// target distribution and emit the accepted prefix plus one corrected or bonus
// token). Rejection-based, sorting-free, with Philox randomness whose seed and
// offset come from the caller, so a step is reproducible from (seed, offset).
//
// Instantiation mirrors FlashInfer's host dispatch on compute capability >= 8:
// 1024 threads, cub warp-scan / warp-reduction block algorithms, f32
// probabilities, int32 token ids. VEC_SIZE is FlashInfer's
// gcd(16 / sizeof(float), vocab): 4, which needs a vocabulary that is a
// multiple of 4 (the padded vocabulary is). DETERMINISTIC is on: the
// deterministic block scan gives the same result regardless of warp timing.
//
// Entries are FlashInfer's templated kernels (mangled symbols); each takes a
// plain argument list and dynamic shared memory of
// sizeof(SamplingTempStorage<...>), published in the meta records.

#include <flashinfer/sampling.cuh>

#include "eidola_kernel.cuh"

namespace eidola_sampling {

using namespace flashinfer::sampling;

constexpr uint32_t kThreads = 1024;
constexpr uint32_t kVec = 4;
constexpr bool kDeterministic = true;
using DType = float;
using IdType = int32_t;

constexpr uint32_t kSmemBytes = sizeof(SamplingTempStorage<kThreads, SCAN_ALGO, REDUCE_ALGO>);

}  // namespace eidola_sampling

namespace flashinfer::sampling {

using eidola_sampling::DType;
using eidola_sampling::IdType;

template __global__ void TopKSamplingFromProbKernel<eidola_sampling::kThreads, SCAN_ALGO,
                                                    REDUCE_ALGO, eidola_sampling::kVec,
                                                    eidola_sampling::kDeterministic, DType, IdType>(
    DType*, IdType*, bool*, IdType*, IdType*, uint32_t, uint32_t, uint64_t*, uint64_t, uint64_t*,
    uint64_t);

template __global__ void TopPSamplingFromProbKernel<eidola_sampling::kThreads, SCAN_ALGO,
                                                    REDUCE_ALGO, eidola_sampling::kVec,
                                                    eidola_sampling::kDeterministic, DType, IdType>(
    DType*, IdType*, bool*, IdType*, float*, float, uint32_t, uint64_t*, uint64_t, uint64_t*,
    uint64_t);

template __global__ void TopKTopPSamplingFromProbKernel<
    eidola_sampling::kThreads, SCAN_ALGO, REDUCE_ALGO, eidola_sampling::kVec,
    eidola_sampling::kDeterministic, DType, IdType>(DType*, IdType*, float*, IdType*, bool*,
                                                     IdType*, IdType, float, uint32_t, uint64_t*,
                                                     uint64_t, uint64_t*, uint64_t);

template __global__ void ChainSpeculativeSampling<eidola_sampling::kThreads, SCAN_ALGO,
                                                  REDUCE_ALGO, eidola_sampling::kVec,
                                                  eidola_sampling::kDeterministic, DType, IdType>(
    DType*, IdType*, DType*, IdType*, IdType*, IdType*, uint32_t, uint32_t, uint64_t*, uint64_t,
    uint64_t*, uint64_t);

}  // namespace flashinfer::sampling

EIDOLA_KERNEL_META(eidola_sampling_top_k_f32, eidola_sampling::kThreads, 1, 1,
                   eidola_sampling::kSmemBytes, 1, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_sampling_top_p_f32, eidola_sampling::kThreads, 1, 1,
                   eidola_sampling::kSmemBytes, 1, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_sampling_top_k_top_p_f32, eidola_sampling::kThreads, 1, 1,
                   eidola_sampling::kSmemBytes, 1, 1, 1, 0);
EIDOLA_KERNEL_META(eidola_sampling_chain_speculative_f32, eidola_sampling::kThreads, 1, 1,
                   eidola_sampling::kSmemBytes, 1, 1, 1, 0);
