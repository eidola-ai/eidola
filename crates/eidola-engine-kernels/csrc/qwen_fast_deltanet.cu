// The fast profile's Gated DeltaNet in blocks (`qwen_fast_deltanet.cuh` holds
// the definition and why it is reproducible): `eidola_qwen_fast_dn_intra`, a
// block's state-free parts (the body the host check runs too), and
// `eidola_qwen_fast_dn_state`, each run's blocks in order from its source, its
// products with the state on tensor cores (`mma.sync` TF32 through WMMA, f32
// accumulators). Built in the `exact` profile: no contraction, so every scalar
// step rounds as written, here and on the host.

#include <mma.h>

#include "eidola_kernel.cuh"
#include "qwen_fast_deltanet.cuh"

using namespace nvcuda;

extern "C" __global__ void __launch_bounds__(qwd::kThreads)
    eidola_qwen_fast_dn_intra(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::IntraShared sh;
  qwd::dn_intra_block(p, blockIdx.x, blockIdx.y, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_intra, qwd::kThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));

namespace {

static_assert(qwd::kThreads == qwd::kKey, "a thread per key column");
static_assert(qwd::kTile * qwd::kCols % qwd::kThreads == 0, "whole value tiles per thread");

constexpr uint32_t kUPer = qwd::kTile * qwd::kCols / qwd::kThreads;
constexpr uint32_t kPPer = qwd::kTile * qwd::kTile / qwd::kThreads;

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragAc = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;

template <class F>
__device__ __forceinline__ void to_tf32(F& f) {
#pragma unroll
  for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
}

// A thread's share of one block's operands, loaded while the block before it
// runs: column `t` of `W`, of the queries and of the keys, its elements of `U`'s
// slice and of `P`, and (threads 0 to 32) one factor.
struct Fetch {
  float w[qwd::kTile];
  float q[qwd::kTile];
  float k[qwd::kTile];
  float u[kUPer];
  float pm[kPPer];
  float fac;
};

__device__ __forceinline__ void fetch(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t b,
                                      uint32_t h, uint32_t kh, uint32_t slice, uint32_t t,
                                      Fetch& f) {
  const uint32_t from = b * p.chunk;
  const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
  const uint64_t th = qwd::tile_head(p, s.tile + b, h);
  const float* w = p.w + th * qwd::kTile * qwd::kKey;
  const float* u = p.u + th * qwd::kTile * p.v_dim;
  const float* pm = p.pm + th * qwd::kTile * qwd::kTile;
  const float* fac = p.fac + th * qwd::kFac;
  const uint32_t key_dim = p.k_heads * qwd::kKey;
#pragma unroll
  for (uint32_t r = 0; r < qwd::kTile; ++r) {
    const uint64_t at =
        static_cast<uint64_t>(s.first + from + r) * key_dim + kh * qwd::kKey + t;
    f.w[r] = w[r * qwd::kKey + t];
    f.q[r] = r < n ? p.q[at] : 0.0f;
    f.k[r] = r < n ? p.k[at] : 0.0f;
  }
#pragma unroll
  for (uint32_t m = 0; m < kUPer; ++m) {
    const uint32_t e = t + m * qwd::kThreads;
    f.u[m] = u[(e / qwd::kCols) * p.v_dim + slice * qwd::kCols + e % qwd::kCols];
  }
#pragma unroll
  for (uint32_t m = 0; m < kPPer; ++m) f.pm[m] = pm[t + m * qwd::kThreads];
  f.fac = t <= qwd::kFacGamma ? fac[t] : 0.0f;
}

// The block's `W`, queries, `U`, `P` and factors into shared memory.
__device__ __forceinline__ void put(const Fetch& f, uint32_t t, qwd::StateShared& sh) {
#pragma unroll
  for (uint32_t r = 0; r < qwd::kTile; ++r) {
    sh.w[r][t] = f.w[r];
    sh.q[r][t] = f.q[r];
  }
#pragma unroll
  for (uint32_t m = 0; m < kUPer; ++m) {
    const uint32_t e = t + m * qwd::kThreads;
    sh.u[e / qwd::kCols][e % qwd::kCols] = f.u[m];
  }
#pragma unroll
  for (uint32_t m = 0; m < kPPer; ++m) {
    const uint32_t e = t + m * qwd::kThreads;
    sh.pm[e / qwd::kTile][e % qwd::kTile] = f.pm[m];
  }
  static_assert(qwd::kFacG == 0 && qwd::kFacGL == qwd::kTile, "the factors' layout");
  if (t < qwd::kTile) {
    sh.eg[t] = f.fac;
  } else if (t < 2 * qwd::kTile) {
    sh.egl[t - qwd::kTile] = f.fac;
  } else if (t == qwd::kFacGamma) {
    sh.gamma = f.fac;
  }
}

// `a · S[:, 16·nt ..]` over the key dimension: the 16 K-steps alternating
// between two accumulators, then their sum.
__device__ __forceinline__ void times_state(const float (*a)[qwd::kAP], const qwd::StateShared& sh,
                                            uint32_t nt, FragC& out) {
  FragC acc[2];
  wmma::fill_fragment(acc[0], 0.0f);
  wmma::fill_fragment(acc[1], 0.0f);
#pragma unroll
  for (uint32_t ks = 0; ks < qwd::kKey / 8; ++ks) {
    FragA fa;
    FragB fb;
    wmma::load_matrix_sync(fa, &a[0][ks * 8], qwd::kAP);
    wmma::load_matrix_sync(fb, &sh.s[ks * 8][nt * 16], qwd::kSP);
    to_tf32(fa);
    to_tf32(fb);
    wmma::mma_sync(acc[ks % 2], fa, fb, acc[ks % 2]);
  }
#pragma unroll
  for (int i = 0; i < out.num_elements; ++i) out.x[i] = acc[0].x[i] + acc[1].x[i];
}

}  // namespace

// Three CTAs an SM (at most 168 registers a thread): a decode step's many short runs
// overlap their state loads.
extern "C" __global__ void __launch_bounds__(qwd::kThreads, 3)
    eidola_qwen_fast_dn_state(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::StateShared sh;
  const uint32_t t = threadIdx.x;
  const uint32_t warp = t / 32;
  const uint32_t slices = p.v_dim / qwd::kCols;
  const uint32_t seq = blockIdx.x / slices;
  const uint32_t slice = blockIdx.x % slices;
  const uint32_t h = blockIdx.y;
  if (seq >= p.seq_count) return;
  const QwDnSeq s = p.seqs[seq];
  const uint32_t nb = qwd::run_tiles(s, p.chunk);
  if (s.tile + nb > p.tiles) return;
  const uint32_t kh = h / (p.v_heads / p.k_heads);
  qwd::state_load(p, s, h, slice, t, sh.s);
  qwd::state_stores(p, s, 0, h, slice, t, sh.s);
  Fetch f;
  if (nb > 0) fetch(p, s, 0, h, kh, slice, t, f);
  for (uint32_t b = 0; b < nb; ++b) {
    const uint32_t from = b * p.chunk;
    const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
    const bool whole = n == p.chunk;
    put(f, t, sh);
    __syncthreads();
    // The next block's operands are in flight while this one runs.
    Fetch next;
    if (b + 1 < nb) fetch(p, s, b + 1, h, kh, slice, t, next);
    // Warps 0 and 1: Δ = U − W S, a column tile each. Warps 2 and 3: q S.
    const uint32_t nt = warp % 2;
    FragC x;
    times_state(warp < 2 ? sh.w : sh.q, sh, nt, x);
    if (warp < 2) {
      FragC uu;
      wmma::load_matrix_sync(uu, &sh.u[0][nt * 16], qwd::kSP, wmma::mem_row_major);
#pragma unroll
      for (int i = 0; i < uu.num_elements; ++i) uu.x[i] = uu.x[i] - x.x[i];
      wmma::store_matrix_sync(&sh.d[0][nt * 16], uu, qwd::kSP, wmma::mem_row_major);
    } else {
      wmma::store_matrix_sync(&sh.y[0][nt * 16], x, qwd::kSP, wmma::mem_row_major);
    }
    __syncthreads();
    qwd::block_rows(n, t, sh);
    if (whole) {
#pragma unroll
      for (uint32_t r = 0; r < qwd::kTile; ++r) sh.w[r][t] = f.k[r];
    }
    __syncthreads();
    if (whole) {
      // S · γ + Kᵀ Δ′: warp w's rows 32w .. 32w + 32, both column tiles.
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t mt = 2 * warp + i / 2;
        const uint32_t ct = i % 2;
        FragC dd, ss;
        wmma::fill_fragment(dd, 0.0f);
#pragma unroll
        for (uint32_t ks = 0; ks < qwd::kTile / 8; ++ks) {
          FragAc fa;
          FragB fb;
          wmma::load_matrix_sync(fa, &sh.w[ks * 8][mt * 16], qwd::kAP);
          wmma::load_matrix_sync(fb, &sh.u[ks * 8][ct * 16], qwd::kSP);
          to_tf32(fa);
          to_tf32(fb);
          wmma::mma_sync(dd, fa, fb, dd);
        }
        wmma::load_matrix_sync(ss, &sh.s[mt * 16][ct * 16], qwd::kSP, wmma::mem_row_major);
#pragma unroll
        for (int e = 0; e < ss.num_elements; ++e) ss.x[e] = ss.x[e] * sh.gamma + dd.x[e];
        wmma::store_matrix_sync(&sh.s[mt * 16][ct * 16], ss, qwd::kSP, wmma::mem_row_major);
      }
    }
    qwd::block_out(p, s, from, n, h, slice, t, sh.z);
    __syncthreads();
    if (whole) qwd::state_stores(p, s, from + n, h, slice, t, sh.s);
    if (b + 1 < nb) f = next;
  }
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_state, qwd::kThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));
