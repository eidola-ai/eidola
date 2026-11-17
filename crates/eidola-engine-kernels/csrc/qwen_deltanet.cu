// The Qwen3.8 DeltaNet recurrence (`deltanet::recurrent_step`): the token
// recurrence, left to right, per sequence and value head. A block owns
// (sequence, value head) and a thread one value column `j`, holding the
// column `S[·][j]` (`kKeyDim` f32) in its own array across tokens: columns
// share nothing but the position's keys and queries, which the block stages.
// One kernel serves a prefill, a decode, a draft check and a fold: the
// positions it runs, which of them produce outputs, and where the state is
// stored are the launch's per-sequence data (`QwDnSeq`).
//
// No division and no square root: this image's SASS holds no fused
// multiply-add (`tests/manifest.rs`).

#include "eidola_kernel.cuh"
#include "qwen_ops.cuh"

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_dn_rec(const __grid_constant__ QwDnRecParams p) {
  __shared__ float k[qwk::kKeyDim];
  __shared__ float q[qwk::kKeyDim];
  const uint32_t h = blockIdx.y;
  const QwDnSeq s = p.seqs[blockIdx.x];
  const uint32_t j = threadIdx.x;
  const bool owns = j < p.v_dim;
  float st[qwk::kKeyDim];
  if (owns) {
    qwk::dn_rec_load(p, s, h, j, st);
    qwk::dn_rec_stores(p, s, 0, h, j, st);
  }
  for (uint32_t i = 0; i < s.count; ++i) {
    __syncthreads();
    qwk::dn_rec_stage(p, s.first + i, h, threadIdx.x, qwk::kBlockThreads, k, q);
    __syncthreads();
    if (owns) qwk::dn_rec_token(p, s, h, j, i, k, q, st);
  }
}
EIDOLA_KERNEL_META(eidola_qwen_dn_rec, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnRecParams));
