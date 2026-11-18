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

namespace {

constexpr uint32_t kST = qwd::kStateThreads;
static_assert(kST / 32 == 2 * (qwd::kSlice / 16), "two warps per 16 value columns of the slice");

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

// 16 bytes from global to shared memory, asynchronously; zeros when not `valid`.
__device__ __forceinline__ void cp16(float* smem, const float* gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem),
               "r"(valid ? 16u : 0u)
               : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// Block `b`'s `W` and rounded queries into shared memory.
__device__ __forceinline__ void issue_wq(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t b,
                                         uint32_t h, uint32_t kh, uint32_t t,
                                         qwd::StateShared& sh) {
  const float* w = p.w + qwd::tile_head(p, s.tile + b, h) * qwd::kTile * qwd::kKey;
  const float* q = p.qr + qwd::tile_key(p, s.tile + b, kh) * qwd::kTile * qwd::kKey;
#pragma unroll
  for (uint32_t m = 0; m < qwd::kTile * qwd::kKey / 4 / kST; ++m) {
    const uint32_t e = t + m * kST;
    const uint32_t r = e / (qwd::kKey / 4);
    const uint32_t c = e % (qwd::kKey / 4) * 4;
    cp16(&sh.w[r][c], w + r * qwd::kKey + c, true);
    cp16(&sh.q[r][c], q + r * qwd::kKey + c, true);
  }
}

// Block `b`'s rounded keys, `U`'s slice, `P` and factors into shared memory.
__device__ __forceinline__ void issue_rest(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t b,
                                           uint32_t h, uint32_t kh, uint32_t vbase, uint32_t t,
                                           qwd::StateShared& sh) {
  const uint64_t th = qwd::tile_head(p, s.tile + b, h);
  const float* k = p.kr + qwd::tile_key(p, s.tile + b, kh) * qwd::kTile * qwd::kKey;
  const float* u = p.u + th * qwd::kTile * p.v_dim;
  const float* pm = p.pm + th * qwd::kTile * qwd::kTile;
  const float* fac = p.fac + th * qwd::kFac;
#pragma unroll
  for (uint32_t m = 0; m < qwd::kTile * qwd::kKey / 4 / kST; ++m) {
    const uint32_t e = t + m * kST;
    const uint32_t r = e / (qwd::kKey / 4);
    const uint32_t c = e % (qwd::kKey / 4) * 4;
    cp16(&sh.k[r][c], k + r * qwd::kKey + c, true);
  }
#pragma unroll
  for (uint32_t m = 0; m < qwd::kTile * qwd::kSlice / 4 / kST; ++m) {
    const uint32_t e = t + m * kST;
    const uint32_t r = e / (qwd::kSlice / 4);
    const uint32_t c = e % (qwd::kSlice / 4) * 4;
    const bool in = vbase + c < p.v_dim;
    cp16(&sh.u[r][c], in ? u + r * p.v_dim + vbase + c : u, in);
  }
  if (t < qwd::kTile * qwd::kTile / 4) {
    cp16(&sh.pm[t / 4][t % 4 * 4], pm + t * 4, true);
  } else if (t < qwd::kTile * qwd::kTile / 4 + qwd::kFac / 4) {
    const uint32_t f = t - qwd::kTile * qwd::kTile / 4;
    cp16(&sh.fac[f * 4], fac + f * 4, true);
  }
}

}  // namespace

// A CTA per (run, 64 value columns, value head): eight warps, warp `w` 16 columns
// (`w % 4`) over half the key dimension (`w / 4`), its share of the state slice
// in registers (`st[nt][e]`: key `64 half + 8 nt + 2t + (e & 1)`, column `g + 8 (e
// >> 1)` of the 16), four barriers a block, the next block's operands copied in
// while this one finishes; at most 128 registers a thread, two CTAs an SM.
extern "C" __global__ void __launch_bounds__(qwd::kStateThreads, 2)
    eidola_qwen_fast_dn_state(const __grid_constant__ QwDnChunkParams p) {
  __shared__ qwd::StateShared sh;
  const uint32_t t = threadIdx.x;
  const uint32_t lane = t % 32;
  const uint32_t wp = t / 32;
  const uint32_t cg = wp % 4;
  const uint32_t half = wp / 4;
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
  const uint32_t v0 = vbase + 16 * cg;
  const bool active = v0 < p.v_dim;
  const uint32_t k0 = 64 * half;
  float st[8][4];
  {
    const float* src = s.src != 0 ? qwd::state_at(s.src, p.state_offset) : nullptr;
#pragma unroll
    for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t k = k0 + 8 * nt + 2 * q4 + (e & 1);
        const uint32_t v = v0 + g + 8 * (e >> 1);
        st[nt][e] = active && src != nullptr ? src[qwd::state_index(h, k, v, p.v_dim)] : 0.0f;
      }
  }
  // A store's address and stride pass through an empty `asm volatile`: stores are rare,
  // and their offsets hoisted out of the block loop would hold registers across it.
  auto store = [&](uint64_t addr) {
    if (!active) return;
    uint64_t at =
        addr + p.state_offset + qwd::state_index(h, k0 + 2 * q4, v0 + g, p.v_dim) * 4;
    uint64_t row = static_cast<uint64_t>(p.v_dim) * 4;
    asm volatile("" : "+l"(at), "+l"(row));
#pragma unroll
    for (uint32_t nt = 0; nt < 8; ++nt) {
      float* r0 = reinterpret_cast<float*>(at);
      float* r1 = reinterpret_cast<float*>(at + row);
      r0[0] = st[nt][0];
      r1[0] = st[nt][1];
      r0[8] = st[nt][2];
      r1[8] = st[nt][3];
      at += 8 * row;
    }
  };
  auto stores = [&](uint32_t done) {
    if (s.live != 0 && s.live_after == done) store(s.live);
    for (uint32_t w = 0; w < s.writes && w < 4; ++w)
      if (s.write_after[w] == done) store(s.write_dst[w]);
  };
  stores(0);
  if (nb > 0) {
    issue_wq(p, s, 0, h, kh, t, sh);
    issue_rest(p, s, 0, h, kh, vbase, t, sh);
    cp_commit();
  }
  float(*dx)[qwd::kDP] = sh.d[cg];
  float(*dy)[qwd::kDP] = sh.y[cg];
  for (uint32_t b = 0; b < nb; ++b) {
    const uint32_t from = b * p.chunk;
    const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
    const bool whole = n == p.chunk;
    const uint32_t lo = qwd::cached_rows(s, from, n);
    cp_wait();
    __syncthreads();
    // This half's sums of Xᵀ = Sᵀ Wᵀ and Yᵀ = Sᵀ qᵀ: 16 columns by the block's 16 rows.
    float x[2][4] = {};
    float y[2][4] = {};
    if (active) {
#pragma unroll
      for (uint32_t ks = 0; ks < 8; ++ks) {
        const uint32_t a[4] = {tf32(st[ks][0]), tf32(st[ks][2]), tf32(st[ks][1]), tf32(st[ks][3])};
        const uint32_t kc = k0 + 8 * ks + 2 * q4;
#pragma unroll
        for (uint32_t nt = 0; nt < 2; ++nt) {
          const float2 wv = *reinterpret_cast<const float2*>(&sh.w[8 * nt + g][kc]);
          const float2 qv = *reinterpret_cast<const float2*>(&sh.q[8 * nt + g][kc]);
          mma(x[nt], a, __float_as_uint(wv.x), __float_as_uint(wv.y));
          mma(y[nt], a, __float_as_uint(qv.x), __float_as_uint(qv.y));
        }
      }
      if (half == 1) {
#pragma unroll
        for (uint32_t nt = 0; nt < 2; ++nt) {
          const uint32_t r = 8 * nt + 2 * q4;
          *reinterpret_cast<float2*>(&dx[g][r]) = make_float2(x[nt][0], x[nt][1]);
          *reinterpret_cast<float2*>(&dx[g + 8][r]) = make_float2(x[nt][2], x[nt][3]);
          *reinterpret_cast<float2*>(&dy[g][r]) = make_float2(y[nt][0], y[nt][1]);
          *reinterpret_cast<float2*>(&dy[g + 8][r]) = make_float2(y[nt][2], y[nt][3]);
        }
      }
    }
    __syncthreads();
    if (b + 1 < nb) {
      issue_wq(p, s, b + 1, h, kh, t, sh);
      cp_commit();
    }
    // The low half adds the high half's sums: Δ = U − X (a cached row's as it was
    // computed, a kept new row's into the tail cache) and Y, into the column group's
    // rows of `d` and `y` (`[v][r]`).
    if (active && half == 0) {
#pragma unroll
      for (uint32_t nt = 0; nt < 2; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          const uint32_t rr = 8 * nt + 2 * q4 + (e & 1);
          const uint32_t vl = g + 8 * (e >> 1);
          float d = sh.u[rr][16 * cg + vl] - (x[nt][e] + dx[vl][rr]);
          if (rr < lo) {
            d = qwd::rec_at(p, qwd::entry_at(p, s, from + rr), h)[p.v_dim + v0 + vl];
          } else if (rr < n && from + rr >= s.keep_from) {
            qwd::rec_at(p, qwd::entry_at(p, s, from + rr), h)[p.v_dim + v0 + vl] = d;
          }
          dx[vl][rr] = d;
          dy[vl][rr] = y[nt][e] + dy[vl][rr];
        }
    }
    __syncthreads();
    if (active) {
      // o = exp(G) · Y + Z for the rows that produce outputs: this warp's row tile.
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t r = 8 * half + 2 * q4 + (e & 1);
        const uint32_t vl = g + 8 * (e >> 1);
        if (r < n && from + r >= s.out_from) {
          const float o = sh.fac[qwd::kFacG + r] * dy[vl][r] + qwd::ztree(sh.pm[r], dx[vl]);
          const uint64_t pos = qwd::staged_at(s, from + r);
          p.out[(pos * p.v_heads + h) * p.v_dim + v0 + vl] = ds::bf16_round(o);
        }
      }
      if (whole) {
        // Sᵀ ← Sᵀ · γ + Δ′ᵀ K over this half's keys, Δ′ = exp(G_{L−1} − G) · Δ on the
        // block's rows.
        uint32_t a[2][4];
#pragma unroll
        for (uint32_t ks = 0; ks < 2; ++ks) {
          float dl[4];
#pragma unroll
          for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t r = 8 * ks + 2 * q4 + (e & 1);
            const uint32_t vl = g + 8 * (e >> 1);
            dl[e] = r < n ? sh.fac[qwd::kFacGL + r] * dx[vl][r] : 0.0f;
          }
          a[ks][0] = tf32(dl[0]);
          a[ks][1] = tf32(dl[2]);
          a[ks][2] = tf32(dl[1]);
          a[ks][3] = tf32(dl[3]);
        }
        const float gamma = sh.fac[qwd::kFacGamma];
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt) {
          float dd[4] = {};
#pragma unroll
          for (uint32_t ks = 0; ks < 2; ++ks) {
            const uint32_t j = 8 * ks + 2 * q4;
            mma(dd, a[ks], __float_as_uint(sh.k[j][k0 + 8 * nt + g]),
                __float_as_uint(sh.k[j + 1][k0 + 8 * nt + g]));
          }
#pragma unroll
          for (uint32_t e = 0; e < 4; ++e) st[nt][e] = st[nt][e] * gamma + dd[e];
        }
      }
    }
    __syncthreads();
    {
      const uint32_t cols = p.v_dim - vbase < qwd::kSlice ? p.v_dim - vbase : qwd::kSlice;
      for (uint32_t r = lo; r < n; ++r)
        if (from + r >= s.keep_from)
          qwd::keep_row(p, s, s.tile + b, from, r, h, vbase, cols, t, kST);
    }
    if (whole) stores(from + n);
    if (b + 1 < nb) {
      issue_rest(p, s, b + 1, h, kh, vbase, t, sh);
      cp_commit();
    }
  }
}
EIDOLA_KERNEL_META(eidola_qwen_fast_dn_state, qwd::kStateThreads, 1, 1, 0, 1, 1, 1,
                   sizeof(QwDnChunkParams));
