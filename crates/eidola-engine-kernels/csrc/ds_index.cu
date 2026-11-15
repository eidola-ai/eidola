// The indexer's scores (`attention::index_score`) and the selections over them
// (`attention::top_k`, `attention::candidate_mask`, the global part of the
// attention slot lists). The bodies are `ds_index.cuh`'s. Nothing here divides,
// so the image holds no fused multiply-add at all (`tests/ds_attention_manifest.rs`).
//
// `eidola_ds_index_score`: grid `(x, rows)`, 256 threads. A block scores 8
// entries a pass, a warp per entry and a lane per index head, and walks the
// passes `bx, bx + x, …` of its row up to the row's visible entries, which are
// device data: the grid is the host's choice and no function of them.
//
// `eidola_ds_select`: grid `rows`, 512 threads, one block per row: two
// 256-bucket counting passes over the 16-bit rank keys find the k-th best key,
// a counting pass ranks every thread's run of candidates, and a last pass
// writes the selection in index order.

#include "ds_index.cuh"
#include "eidola_kernel.cuh"

extern "C" __global__ void __launch_bounds__(ds_index::kScoreThreads)
    eidola_ds_index_score(const __grid_constant__ DsIndexScoreParams p) {
  extern __shared__ __align__(8) unsigned char smem[];
  ds_index::index_score(p, *reinterpret_cast<ds_index::ScoreShared*>(smem), blockIdx.x, gridDim.x,
                        blockIdx.y, ds::Par{});
}
EIDOLA_KERNEL_META(eidola_ds_index_score, ds_index::kScoreThreads, 1, 1,
                   sizeof(ds_index::ScoreShared), 1, 1, 1, sizeof(DsIndexScoreParams));

extern "C" __global__ void __launch_bounds__(ds_index::kSelectThreads)
    eidola_ds_select(const __grid_constant__ DsSelectParams p) {
  extern __shared__ __align__(8) unsigned char smem[];
  ds_index::select(p, *reinterpret_cast<ds_index::SelectShared*>(smem), blockIdx.x, ds::Par{});
}
EIDOLA_KERNEL_META(eidola_ds_select, ds_index::kSelectThreads, 1, 1, sizeof(ds_index::SelectShared),
                   1, 1, 1, sizeof(DsSelectParams));
