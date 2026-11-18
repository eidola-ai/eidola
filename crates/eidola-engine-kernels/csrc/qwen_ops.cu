// The Qwen3.8 executor's kernels that neither divide nor take a square root:
// embedding, the exact FP8 and BF16 linears, the paged KV write, the BF16
// widening, the f32 rounding and the grouped row copy. Their image's SASS holds
// no fused multiply-add (`tests/manifest.rs`); the kernels that divide are
// `qwen_ops_div.cu`, the DeltaNet recurrence is `qwen_deltanet.cu`.
//
// Each entry but the row copy is its body in `qwen_ops.cuh` placed on a grid;
// the row copy has no host body (the host device reads its items as it does).

#include "eidola_kernel.cuh"
#include "qwen_ops.cuh"

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_embed(const __grid_constant__ QwEmbedParams p) {
  qwk::embed_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_embed, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwEmbedParams));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_lin8(const __grid_constant__ QwLin8Params p) {
  qwk::lin8_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_lin8, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1, sizeof(QwLin8Params));

extern "C" __global__ void __launch_bounds__(qwk::kBlockThreads)
    eidola_qwen_linear(const __grid_constant__ QwLinearParams p) {
  qwk::linear_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_linear, qwk::kBlockThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwLinearParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_kv_write(const __grid_constant__ QwKvWriteParams p) {
  qwk::kv_write_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_kv_write, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwKvWriteParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_widen(const __grid_constant__ QwWidenParams p) {
  qwk::widen_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_widen, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwWidenParams));

extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_round(const __grid_constant__ QwRoundParams p) {
  qwk::round_thread(p, blockIdx.x, blockIdx.y, threadIdx.x);
}
EIDOLA_KERNEL_META(eidola_qwen_round, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwRoundParams));

// Item `blockIdx.x` of the groups taken in order: its row copied by the
// block, sixteen bytes a thread at a time when both rows are aligned to them.
extern "C" __global__ void __launch_bounds__(qwk::kElementThreads)
    eidola_qwen_copy_rows(const __grid_constant__ QwCopyParams p) {
  uint32_t i = blockIdx.x, g = 0;
  while (g < p.groups && g < QW_COPY_GROUPS && i >= p.group[g].items) i -= p.group[g++].items;
  if (g >= p.groups || g >= QW_COPY_GROUPS) return;
  const QwCopyGroup& c = p.group[g];
  const uint32_t sb = c.src_buf[i], db = c.dst_buf[i];
  const uint32_t sr = c.src_row[i], dr = c.dst_row[i];
  if (sb >= QW_COPY_BUFFERS || db >= QW_COPY_BUFFERS || sr >= c.rows[sb] || dr >= c.rows[db]) {
    if (threadIdx.x == 0) atomicOr(p.status, QW_STATUS_BAD_INDEX);
    return;
  }
  if (sb == db && sr == dr) return;
  const uint32_t* src = c.buf[sb] + static_cast<uint64_t>(sr) * c.width;
  uint32_t* dst = c.buf[db] + static_cast<uint64_t>(dr) * c.width;
  if (c.width % 4 == 0 &&
      ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (uint32_t w = threadIdx.x; w < c.width / 4; w += qwk::kElementThreads) d4[w] = s4[w];
  } else {
    for (uint32_t w = threadIdx.x; w < c.width; w += qwk::kElementThreads) dst[w] = src[w];
  }
}
EIDOLA_KERNEL_META(eidola_qwen_copy_rows, qwk::kElementThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwCopyParams));
