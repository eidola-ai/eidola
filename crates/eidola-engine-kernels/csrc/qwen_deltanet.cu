// The Qwen3.8 DeltaNet recurrence (`deltanet::recurrent_step`): the token
// recurrence, left to right, per sequence and value head. Block `x` owns
// sequence `x / slices` and value columns `32 · (x % slices) ..` of value
// head `y`; four threads share a column, each holding 32 of its state values
// across tokens (`qwen_ops.cuh`). One kernel serves a prefill, a decode, a
// draft check and a fold: the positions it runs, which of them produce
// outputs, and where the state is stored are the launch's per-sequence data
// (`QwDnSeq`).
//
// No division and no square root: this image's SASS holds no fused
// multiply-add (`tests/manifest.rs`).

#include "eidola_kernel.cuh"
#include "qwen_ops.cuh"

namespace {

// Steps 2 and 1 of a column's tree across its four threads, then the +0 tail.
__device__ __forceinline__ float group_sum(float part, uint32_t l) {
  part = qwk::dn_tree_step(part, __shfl_xor_sync(0xffffffffu, part, 2), l, 2);
  part = qwk::dn_tree_step(part, __shfl_xor_sync(0xffffffffu, part, 1), l, 1);
  return part + 0.0f;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(qwk::kDnThreads)
    eidola_qwen_dn_rec(const __grid_constant__ QwDnRecParams p) {
  __shared__ qwk::DnRecShared sh;
  const uint32_t t = threadIdx.x;
  const uint32_t slices = qwk::dn_rec_slices(p.v_dim);
  const uint32_t slice = blockIdx.x % slices;
  const uint32_t h = blockIdx.y;
  const QwDnSeq s = p.seqs[blockIdx.x / slices];
  const uint32_t l = t % qwk::kDnGroup;
  const uint32_t c = t / qwk::kDnGroup;
  const uint32_t j = slice * qwk::kDnCols + c;
  qwk::DnFetch f;
  qwk::dn_rec_fetch(p, s, h, slice, 0, t, f);
  float st[qwk::kDnPer];
  qwk::dn_rec_load(p, s, h, j, l, st);
  qwk::dn_rec_stores(p, s, 0, h, j, l, st);
  qwk::dn_rec_put(f, t, sh, 0);
  __syncthreads();
  uint32_t b = 0;
  for (uint32_t base = 0; base < s.count; base += qwk::kDnChunk, b ^= 1) {
    const bool more = base + qwk::kDnChunk < s.count;
    if (more) qwk::dn_rec_fetch(p, s, h, slice, base + qwk::kDnChunk, t, f);
    const uint32_t n = s.count - base < qwk::kDnChunk ? s.count - base : qwk::kDnChunk;
    for (uint32_t u = 0; u < n; ++u) {
      const uint32_t i = base + u;
      float x[qwk::kDnPer];
      qwk::dn_row(sh.k[b][u][l], x);
      const float r = group_sum(qwk::dn_rec_decay(st, x, sh.alpha[b][u]), l);
      qwk::dn_rec_update(st, x, sh.v[b][u][c], sh.beta[b][u], r);
      if (i >= s.out_from) {
        qwk::dn_row(sh.q[b][u][l], x);
        const float o = group_sum(qwk::dn_partial(st, x), l);
        if (l == 0 && j < p.v_dim) qwk::dn_rec_out(p, s.first + i, h, j, o);
      }
      qwk::dn_rec_stores(p, s, i + 1, h, j, l, st);
    }
    if (more) qwk::dn_rec_put(f, t, sh, b ^ 1);
    __syncthreads();
  }
}
EIDOLA_KERNEL_META(eidola_qwen_dn_rec, qwk::kDnThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnRecParams));
