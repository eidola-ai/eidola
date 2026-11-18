// The fast profile's Gated DeltaNet in blocks (`qwen_fast_deltanet.cuh` holds
// the definition and why it is reproducible): `eidola_qwen_fast_dn_intra`, a
// block's state-free parts (the body the host check runs too), and
// `eidola_qwen_fast_dn_state`, each run's blocks in order from its source, its
// products with the state on tensor cores (`mma.sync` m16n8k8 TF32, f32
// accumulators). Built in the `exact` profile: no contraction, so every scalar
// step rounds as written, here and on the host.

#include "eidola_kernel.cuh"
#include "qwen_fast_deltanet.cuh"

// Four CTAs an SM (at most 128 registers a thread): a decode step's blocks are
// a position or two each, and many.
extern "C" __global__ void __launch_bounds__(qwd::kThreads, 4)
    eidola_qwen_fast_dn_intra(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::IntraShared sh;
  qwd::dn_intra_block(p, blockIdx.x, blockIdx.y, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_intra, qwd::kThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));

// After the state kernel: the kept new positions into the tail cache.
extern "C" __global__ void __launch_bounds__(qwd::kThreads, 8)
    eidola_qwen_fast_dn_keep(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::KeepShared sh;
  qwd::dn_keep_block(p, blockIdx.x, blockIdx.y, ds::Par{}, sh);
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_keep, qwd::kThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));

namespace {

constexpr uint32_t kWarps = qwd::kThreads / 32;
static_assert(kWarps * 16 == qwd::kSlice, "a warp per 16 value columns of the slice");

__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

// `d += a · b` on one m16n8k8 tile: `a` 16 × 8 (rows `g`, `g + 8`; columns
// `t`, `t + 4`), `b` 8 × 8 (rows `t`, `t + 4`; column `g`), `d` 16 × 8 (rows
// `g`, `g + 8`; columns `2t`, `2t + 1`), for lane `4g + t`.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16 bytes from global to shared memory, asynchronously; zeros, reading nothing,
// when not `valid`.
__device__ __forceinline__ void cp16(float* smem, const float* gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem),
               "r"(valid ? 16u : 0u)
               : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// Block `b`'s rows: its positions `n`, its cached rows `lo`.
struct Rows {
  uint32_t from, n, lo;
};
__device__ __forceinline__ Rows rows_of(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t b) {
  const uint32_t from = b * p.chunk;
  const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
  return {from, n, qwd::cached_rows(s, from, n)};
}

// Block `b`'s new rows of `W` and the rounded queries into shared memory, the
// others zeros.
__device__ __forceinline__ void issue_wq(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t b,
                                         uint32_t h, uint32_t kh, uint32_t t,
                                         qwd::StateShared& sh) {
  const Rows rw = rows_of(p, s, b);
  const float* w = p.w + qwd::tile_head(p, s.tile + b, h) * qwd::kTile * qwd::kKey;
  const float* q = p.qr + qwd::tile_key(p, s.tile + b, kh) * qwd::kTile * qwd::kKey;
#pragma unroll
  for (uint32_t m = 0; m < qwd::kTile * qwd::kKey / 4 / qwd::kThreads; ++m) {
    const uint32_t e = t + m * qwd::kThreads;
    const uint32_t r = e / (qwd::kKey / 4);
    const uint32_t c = e % (qwd::kKey / 4) * 4;
    const bool fresh = r >= rw.lo && r < rw.n;
    cp16(&sh.w[r][c], fresh ? w + r * qwd::kKey + c : w, fresh);
    cp16(&sh.q[r][c], fresh ? q + r * qwd::kKey + c : q, fresh);
  }
}

// Block `b`'s rounded keys (a whole block's only), `U`'s new rows and the cached
// rows' `Δ` of the slice, `P` and factors into shared memory.
__device__ __forceinline__ void issue_rest(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t b,
                                           uint32_t h, uint32_t kh, uint32_t vbase, uint32_t t,
                                           qwd::StateShared& sh) {
  const Rows rw = rows_of(p, s, b);
  const uint64_t th = qwd::tile_head(p, s.tile + b, h);
  const float* k = p.kr + qwd::tile_key(p, s.tile + b, kh) * qwd::kTile * qwd::kKey;
  const float* u = p.u + th * qwd::kTile * p.v_dim;
  const float* pm = p.pm + th * qwd::kTile * qwd::kTile;
  const float* fac = p.fac + th * qwd::kFac;
  if (rw.n == p.chunk) {
#pragma unroll
    for (uint32_t m = 0; m < qwd::kTile * qwd::kKey / 4 / qwd::kThreads; ++m) {
      const uint32_t e = t + m * qwd::kThreads;
      const uint32_t r = e / (qwd::kKey / 4);
      const uint32_t c = e % (qwd::kKey / 4) * 4;
      cp16(&sh.k[r][c], k + r * qwd::kKey + c, true);
    }
  }
#pragma unroll
  for (uint32_t m = 0; m < qwd::kTile * qwd::kSlice / 4 / qwd::kThreads; ++m) {
    const uint32_t e = t + m * qwd::kThreads;
    const uint32_t r = e / (qwd::kSlice / 4);
    const uint32_t c = e % (qwd::kSlice / 4) * 4;
    const bool in = vbase + c < p.v_dim;
    const bool fresh = in && r >= rw.lo && r < rw.n;
    cp16(&sh.u[r][c], fresh ? u + r * p.v_dim + vbase + c : u, fresh);
    if (r < rw.lo && in) {
      const float* d = qwd::rec_at(p, qwd::entry_at(p, s, rw.from + r), h) + p.v_dim;
      cp16(&sh.cd[r][c], d + vbase + c, true);
    }
  }
  if (t < qwd::kTile * qwd::kTile / 4) {
    cp16(&sh.pm[t / 4][t % 4 * 4], pm + t * 4, true);
  } else if (t < qwd::kTile * qwd::kTile / 4 + qwd::kFac / 4) {
    const uint32_t f = t - qwd::kTile * qwd::kTile / 4;
    cp16(&sh.fac[f * 4], fac + f * 4, true);
  }
}

// Xᵀ = Sᵀ Wᵀ and Yᵀ = Sᵀ qᵀ for a warp's 16 columns by the block's 16 rows: `wq(r, c,
// w0, w1, q0, q1)` gives the right operands' TF32 bits at row `r`, columns `c, c + 1`.
template <class WQ>
__device__ __forceinline__ void xy_products(const float (&st)[16][4], uint32_t g, uint32_t q4,
                                            float (&x)[2][4], float (&y)[2][4], const WQ& wq) {
#pragma unroll
  for (uint32_t ks = 0; ks < 16; ++ks) {
    const uint32_t a[4] = {tf32(st[ks][0]), tf32(st[ks][2]), tf32(st[ks][1]), tf32(st[ks][3])};
#pragma unroll
    for (uint32_t nt = 0; nt < 2; ++nt) {
      uint32_t w0, w1, q0, q1;
      wq(8 * nt + g, 8 * ks + 2 * q4, w0, w1, q0, q1);
      mma(x[nt], a, w0, w1);
      mma(y[nt], a, q0, q1);
    }
  }
}

// The rest of a block for a warp's 16 columns: Δ = U − X (a cached row's from `cd`, a
// kept new row's into the tail cache) into the warp's rows of `d`, the outputs, and for a
// whole block the update; `kb(j, c)` gives the rounded keys' TF32 bits. `sh` holds `u`,
// `cd`, `pm`, `fac` and `d`.
template <class Sh, class KB>
__device__ __forceinline__ void block_rest(const QwDnChunkParams& p, const QwDnSeq& s,
                                           const Rows& rw, uint32_t h, uint32_t wp, uint32_t v0,
                                           uint32_t g, uint32_t q4, float (&st)[16][4],
                                           float (&x)[2][4], const float (&y)[2][4], Sh& sh,
                                           const KB& kb) {
  const uint32_t from = rw.from;
  const uint32_t n = rw.n;
  float(*dw)[qwd::kDP] = sh.d[wp];
#pragma unroll
  for (uint32_t nt = 0; nt < 2; ++nt) {
    const uint32_t r = 8 * nt + 2 * q4;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t rr = r + (e & 1);
      const uint32_t vl = g + 8 * (e >> 1);
      float d = sh.u[rr][16 * wp + vl] - x[nt][e];
      if (rr < rw.lo) {
        d = sh.cd[rr][16 * wp + vl];
      } else if (rr < n && from + rr >= s.keep_from) {
        qwd::rec_at(p, qwd::entry_at(p, s, from + rr), h)[p.v_dim + v0 + vl] = d;
      }
      x[nt][e] = d;
    }
    *reinterpret_cast<float2*>(&dw[g][r]) = make_float2(x[nt][0], x[nt][1]);
    *reinterpret_cast<float2*>(&dw[g + 8][r]) = make_float2(x[nt][2], x[nt][3]);
  }
  __syncwarp();
  // o = exp(G) · Y + Z for the rows that produce outputs.
#pragma unroll
  for (uint32_t nt = 0; nt < 2; ++nt)
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t r = 8 * nt + 2 * q4 + (e & 1);
      const uint32_t vl = g + 8 * (e >> 1);
      if (r < n && from + r >= s.out_from) {
        const float o = sh.fac[qwd::kFacG + r] * y[nt][e] + qwd::ztree(sh.pm[r], dw[vl]);
        const uint64_t pos = qwd::staged_at(s, from + r);
        p.out[(pos * p.v_heads + h) * p.v_dim + v0 + vl] = ds::bf16_round(o);
      }
    }
  if (n == p.chunk) {
    // Sᵀ ← Sᵀ · γ + Δ′ᵀ K, Δ′ = exp(G_{L−1} − G) · Δ on the block's rows.
    uint32_t a[2][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 2; ++nt) {
      float dl[4];
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t r = 8 * nt + 2 * q4 + (e & 1);
        dl[e] = r < n ? sh.fac[qwd::kFacGL + r] * x[nt][e] : 0.0f;
      }
      a[nt][0] = tf32(dl[0]);
      a[nt][1] = tf32(dl[2]);
      a[nt][2] = tf32(dl[1]);
      a[nt][3] = tf32(dl[3]);
    }
    const float gamma = sh.fac[qwd::kFacGamma];
#pragma unroll
    for (uint32_t nt = 0; nt < 16; ++nt) {
      float dd[4] = {};
#pragma unroll
      for (uint32_t ks = 0; ks < 2; ++ks) {
        const uint32_t j = 8 * ks + 2 * q4;
        mma(dd, a[ks], kb(j, 8 * nt + g), kb(j + 1, 8 * nt + g));
      }
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) st[nt][e] = st[nt][e] * gamma + dd[e];
    }
  }
}

// A slice of the state into registers (`st[nt][e]`: key `8 nt + 2t + (e & 1)`, column
// `g + 8 (e >> 1)` of the warp's 16), zeros without a source.
__device__ __forceinline__ void load_state(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t h,
                                           uint32_t v0, bool active, uint32_t g, uint32_t q4,
                                           float (&st)[16][4]) {
  const float* src = s.src != 0 ? qwd::state_at(s.src, p.state_offset) : nullptr;
#pragma unroll
  for (uint32_t nt = 0; nt < 16; ++nt)
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t k = 8 * nt + 2 * q4 + (e & 1);
      const uint32_t v = v0 + g + 8 * (e >> 1);
      st[nt][e] = active && src != nullptr ? src[qwd::state_index(h, k, v, p.v_dim)] : 0.0f;
    }
}

// The stores due once `done` positions have run, of the warp's slice of the state. A
// store's address and stride pass through an empty `asm volatile`: stores are rare, and
// their 64 offsets hoisted out of the block loop would hold registers across it.
__device__ __forceinline__ void state_stores(const QwDnChunkParams& p, const QwDnSeq& s,
                                             uint32_t done, uint32_t h, uint32_t v0, bool active,
                                             uint32_t g, uint32_t q4, const float (&st)[16][4]) {
  auto store = [&](uint64_t addr) {
    if (!active) return;
    uint64_t at = addr + p.state_offset + qwd::state_index(h, 2 * q4, v0 + g, p.v_dim) * 4;
    uint64_t row = static_cast<uint64_t>(p.v_dim) * 4;
    asm volatile("" : "+l"(at), "+l"(row));
#pragma unroll
    for (uint32_t nt = 0; nt < 16; ++nt) {
      float* r0 = reinterpret_cast<float*>(at);
      float* r1 = reinterpret_cast<float*>(at + row);
      r0[0] = st[nt][0];
      r1[0] = st[nt][1];
      r0[8] = st[nt][2];
      r1[8] = st[nt][3];
      at += 8 * row;
    }
  };
  if (s.live != 0 && s.live_after == done) store(s.live);
  for (uint32_t w = 0; w < s.writes && w < 4; ++w)
    if (s.write_after[w] == done) store(s.write_dst[w]);
}

}  // namespace

// A CTA per (run, 64 value columns, value head): four warps, each 16 columns of
// the state slice in registers, three barriers a block, the next block's operands
// copied in while this one finishes; at most 128 registers a thread, four CTAs an SM.
extern "C" __global__ void __launch_bounds__(qwd::kThreads, 4)
    eidola_qwen_fast_dn_state(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::StateShared sh;
  const uint32_t t = threadIdx.x;
  const uint32_t lane = t % 32;
  const uint32_t wp = t / 32;
  const uint32_t g = lane / 4;
  const uint32_t q4 = lane % 4;
  const uint32_t slices = (p.v_dim + qwd::kSlice - 1) / qwd::kSlice;
  const uint32_t seq = blockIdx.x / slices;
  const uint32_t vbase = blockIdx.x % slices * qwd::kSlice;
  const uint32_t h = blockIdx.y;
  if (seq >= p.seq_count) return;
  const QwDnSeq s = p.seqs[seq];
  const uint32_t nb = qwd::run_tiles(s, p.chunk);
  if (s.tile + nb > p.tiles) return;
  const uint32_t kh = h / (p.v_heads / p.k_heads);
  const uint32_t v0 = vbase + 16 * wp;
  const bool active = v0 < p.v_dim;
  if (nb > 0) {
    issue_wq(p, s, 0, h, kh, t, sh);
    issue_rest(p, s, 0, h, kh, vbase, t, sh);
    cp_commit();
  }
  float st[16][4];
  load_state(p, s, h, v0, active, g, q4, st);
  state_stores(p, s, 0, h, v0, active, g, q4, st);
  for (uint32_t b = 0; b < nb; ++b) {
    const Rows rw = rows_of(p, s, b);
    cp_wait();
    __syncthreads();
    float x[2][4] = {};
    float y[2][4] = {};
    if (active) {
      xy_products(st, g, q4, x, y,
                  [&](uint32_t r, uint32_t c, uint32_t& w0, uint32_t& w1, uint32_t& q0,
                      uint32_t& q1) {
                    const float2 wv = *reinterpret_cast<const float2*>(&sh.w[r][c]);
                    const float2 qv = *reinterpret_cast<const float2*>(&sh.q[r][c]);
                    w0 = __float_as_uint(wv.x);
                    w1 = __float_as_uint(wv.y);
                    q0 = __float_as_uint(qv.x);
                    q1 = __float_as_uint(qv.y);
                  });
    }
    __syncthreads();
    if (b + 1 < nb) {
      issue_wq(p, s, b + 1, h, kh, t, sh);
      cp_commit();
    }
    if (active) {
      block_rest(p, s, rw, h, wp, v0, g, q4, st, x, y, sh,
                 [&](uint32_t j, uint32_t c) { return __float_as_uint(sh.k[j][c]); });
    }
    __syncthreads();
    if (rw.n == p.chunk) state_stores(p, s, rw.from + rw.n, h, v0, active, g, q4, st);
    if (b + 1 < nb) {
      issue_rest(p, s, b + 1, h, kh, vbase, t, sh);
      cp_commit();
    }
  }
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_state, qwd::kThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));

// The fused form, for a launch whose every run is at most the ring's positions: the
// state kernel's grid, each block's intra body for the CTA's value head and columns in
// shared memory (`qwd::fused_intra`), the state kernel's block on it, then the block's
// kept positions into the tail cache (`qwd::fused_keep`). The queries and keys are read
// as the intra body loaded them and rounded as the unfused scratch rounds them.
extern "C" __global__ void __launch_bounds__(qwd::kThreads, 4)
    eidola_qwen_fast_dn_fused(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::FusedShared sh;
  const uint32_t t = threadIdx.x;
  const uint32_t lane = t % 32;
  const uint32_t wp = t / 32;
  const uint32_t g = lane / 4;
  const uint32_t q4 = lane % 4;
  const uint32_t slices = (p.v_dim + qwd::kSlice - 1) / qwd::kSlice;
  const uint32_t seq = blockIdx.x / slices;
  const uint32_t vbase = blockIdx.x % slices * qwd::kSlice;
  const uint32_t h = blockIdx.y;
  if (seq >= p.seq_count) return;
  const QwDnSeq s = p.seqs[seq];
  const uint32_t nb = qwd::run_tiles(s, p.chunk);
  if (s.tile + nb > p.tiles) return;
  const uint32_t v0 = vbase + 16 * wp;
  const bool active = v0 < p.v_dim;
  float st[16][4];
  load_state(p, s, h, v0, active, g, q4, st);
  state_stores(p, s, 0, h, v0, active, g, q4, st);
  for (uint32_t b = 0; b < nb; ++b) {
    const Rows rw = rows_of(p, s, b);
    qwd::fused_intra(p, s, b, h, vbase, ds::Par{}, sh);
    float x[2][4] = {};
    float y[2][4] = {};
    if (active) {
      xy_products(st, g, q4, x, y,
                  [&](uint32_t r, uint32_t c, uint32_t& w0, uint32_t& w1, uint32_t& q0,
                      uint32_t& q1) {
                    const float2 wv = *reinterpret_cast<const float2*>(&sh.w[r][c]);
                    w0 = __float_as_uint(wv.x);
                    w1 = __float_as_uint(wv.y);
                    q0 = __float_as_uint(qwd::tf32_rna(sh.in.q[r][c]));
                    q1 = __float_as_uint(qwd::tf32_rna(sh.in.q[r][c + 1]));
                  });
      block_rest(p, s, rw, h, wp, v0, g, q4, st, x, y, sh, [&](uint32_t j, uint32_t c) {
        return __float_as_uint(qwd::tf32_rna(sh.in.k[j][c]));
      });
    }
    __syncthreads();
    qwd::fused_keep(p, s, b, h, vbase, ds::Par{}, sh);
    if (rw.n == p.chunk) state_stores(p, s, rw.from + rw.n, h, v0, active, g, q4, st);
  }
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_fused, qwd::kThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));
