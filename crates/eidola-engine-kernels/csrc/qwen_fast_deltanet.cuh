// The fast profile's Gated DeltaNet in blocks (`QwDnChunkParams`): the bodies
// a GPU and the host check share (`eidola-engine-cuda`'s
// `oracle/qwen_launch_host.cc`), and the host form of the state kernel.
//
// The definition. A run starts at a block boundary, from the state `S` there,
// and its positions fall in blocks of `L = chunk` positions (at most 16). For a
// block of `n ≤ L` positions (relative `r = 0 .. n`) of one value head, with
// queries `q_r`, keys `k_r` (its key head's), values `v_r`, write strengths
// `β_r` and log decays `g_r`:
//
//   G_r    = g_0 + … + g_r                           (in order)
//   E_rj   = exp(G_r − G_j)                          (j ≤ r)
//   A_rj   = (β_r · (k_r·k_j)) · E_rj                (j < r)
//   T      = (I + A)⁻¹, unit lower triangular: column j by forward
//            substitution, T_rj = −(A_rj + Σ_{m=j+1}^{r−1} A_rm · T_mj)
//   W_r    = Σ_{j≤r} (T_rj · c_j) · k_j,  c_j = β_j · exp(G_j)
//   U_r    = Σ_{j≤r} (T_rj · β_j) · v_j
//   P_rj   = (q_r·k_j) · E_rj                        (j ≤ r)
//   Δ_r    = U_r − (W_r S)                            (W S on tensor cores)
//   o_r    = exp(G_r) · (q_r S) + Σ_j P_rj · Δ_j       (q S on tensor cores, the
//            sum `ztree`'s), rounded to BF16
//
// and when the block is whole (`n = L`) the state after it is
// `S' = S · exp(G_{L−1}) + Kᵀ Δ′` with `Δ′_j = exp(G_{L−1} − G_j) · Δ_j` (Kᵀ Δ′ on
// tensor cores), which is the token recurrence's algebra
// (`deltanet::recurrent_step`) regrouped: the same output and state in exact
// arithmetic, different roundings. The dot products `k_r·k_j` and `q_r·k_j`
// are `pinned_dot`s, every other sum runs in the order written, from +0, IEEE
// `+ - *` (the image is built without contraction), `exp` is `exp32`. Every
// tensor-core operand is rounded to TF32 (nearest, ties away: `W`, the queries
// and the keys once, into the scratch; `S` and `Δ′` as they are read) and every
// product of one accumulates in f32; `S` stays f32 between blocks.
//
// Why a position's output and every stored state are functions of the token
// prefix alone. A block is always a block of the same absolute positions (runs
// start at boundaries), its tile always 16 rows with positions past `n` zero
// in every operand (`q = k = v = β = g = 0`, and `W`, `U`, `P` zero there). Row
// `r` of every quantity reads rows `≤ r` only: each loop over `j` or `m` stops
// at `r` or reads a `P_rj` that is +0 past `r` (a +0 product adds nothing to a
// sum that starts at +0), and a tensor-core product gives position `r`'s
// result from position `r`'s row of `W` or `q` and the whole state, which
// never depends on the rows after it. So row `r`'s bits are the same whether
// the block holds `r + 1` positions or `L`, whether it is a prefill's, a decode
// step's recomputation from the boundary or a fold's, and whatever else the
// launch holds; the state is updated only from whole blocks, whose tiles are
// the same in every path. Nothing reduces over positions with a tensor core but
// the update, so no result rests on a tensor core adding an exact zero.

#pragma once

#include "ds_body.cuh"
#include "qwen_abi.cuh"

#if defined(__CUDACC__)
#define QD_ALIGN32 __align__(32)
#else
#define QD_ALIGN32 alignas(32)
#endif

namespace qwd {

constexpr uint32_t kTile = 16;     // rows of a block's tile: the longest block
constexpr uint32_t kKey = 128;     // the key dimension
constexpr uint32_t kThreads = 128; // threads of either kernel
constexpr uint32_t kFac = 48;      // floats of a block's factors

// A block's factors: `exp(G_r)` at `kFacG`, `exp(G_{L−1} − G_r)` at
// `kFacGL`, `exp(G_{L−1})` at `kFacGamma`.
constexpr uint32_t kFacG = 0;
constexpr uint32_t kFacGL = kTile;
constexpr uint32_t kFacGamma = 2 * kTile;

DS_FN uint32_t run_tiles(const QwDnSeq& s, uint32_t chunk) {
  return (s.count + chunk - 1) / chunk;
}

// The scratch's index of (tile, value head), and of (tile, key head).
DS_FN uint64_t tile_head(const QwDnChunkParams& p, uint32_t tile, uint32_t h) {
  return static_cast<uint64_t>(tile) * p.v_heads + h;
}
DS_FN uint64_t tile_key(const QwDnChunkParams& p, uint32_t tile, uint32_t kh) {
  return static_cast<uint64_t>(tile) * p.k_heads + kh;
}

// State value `i` of column `j` of value head `h` in a layer's state.
DS_FN uint64_t state_index(uint32_t h, uint32_t i, uint32_t j, uint32_t vd) {
  return (static_cast<uint64_t>(h) * kKey + i) * vd + j;
}

DS_FN float* state_at(uint64_t addr, uint64_t offset) {
  return reinterpret_cast<float*>(addr + offset);
}

// TF32, nearest with ties away from zero (`cvt.rna.tf32.f32`), as f32 bits
// with the low 13 bits clear; infinities and NaNs as they are.
DS_FN float tf32_rna(float x) {
  const uint32_t u = ds::f32_bits(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  return ds::f32_from_bits((u + 0x1000u) & 0xffffe000u);
}

// `pinned_dot` of two 128-wide rows (16 lanes from +0 over the chunks of 16,
// then the halving tree).
DS_FN float dot128(const float* a, const float* b) {
  float acc[16];
  for (uint32_t j = 0; j < 16; ++j) acc[j] = 0.0f;
  for (uint32_t i = 0; i < kKey; i += 16) {
    for (uint32_t j = 0; j < 16; ++j) acc[j] = acc[j] + a[i + j] * b[i + j];
  }
  for (uint32_t w = 8; w > 0; w >>= 1) {
    for (uint32_t j = 0; j < w; ++j) acc[j] = acc[j] + acc[j + w];
  }
  return acc[0] + 0.0f;
}

// `Z = Σ_j P_rj · Δ_j` over all 16 rows of a tile (`P_rj` is +0 past `r`, so
// those terms add exact zeros to sums that start at +0): four partial sums,
// thread `t`'s rows `2t, 2t + 1, 2t + 8, 2t + 9` in that order from +0, then
// `(p0 + p1) + (p2 + p3)`.
DS_FN float ztree(const float* pr, const float* dj) {
  float p[4];
  for (uint32_t t = 0; t < 4; ++t) {
    float a = 0.0f;
    a = a + pr[2 * t] * dj[2 * t];
    a = a + pr[2 * t + 1] * dj[2 * t + 1];
    a = a + pr[2 * t + 8] * dj[2 * t + 8];
    a = a + pr[2 * t + 9] * dj[2 * t + 9];
    p[t] = a;
  }
  return (p[0] + p[1]) + (p[2] + p[3]);
}

// ---- the block's state-free parts (`eidola_qwen_fast_dn_intra`) ----
//
// Every row past the block's `n` positions is zero in what it writes (`W`, `U`,
// `P`, the rounded queries and keys), and every loop over rows stops at `n`,
// so a decode step's block of one or two positions costs that much.

struct QD_ALIGN32 IntraShared {
  float q[kTile][kKey];
  float k[kTile][kKey];
  float kk[kTile][kTile];
  float qk[kTile][kTile];
  float a[kTile][kTile];
  float t[kTile][kTile];
  float tc[kTile][kTile];
  float tb[kTile][kTile];
  float beta[kTile];
  float g[kTile];
  float c[kTile];
  uint32_t seq;
  uint32_t found;
};

// CTA (`x`, key head `kh`): block `x` of the launch, its key head's queries and
// keys (and their TF32 roundings into the scratch), then each of the key
// head's value heads: `W` (rounded to TF32), `U`, `P` and the factors.
template <class P>
DS_FN void dn_intra_block(const QwDnChunkParams& p, uint32_t x, uint32_t kh, const P& par,
                          IntraShared& sh) {
  par.run([&](uint32_t t) {
    if (t == 0) sh.found = 0;
  });
  // The run holding block `x`: the one whose blocks `tile ..` reach past it.
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < p.seq_count; i += kThreads) {
      const QwDnSeq& s = p.seqs[i];
      if (x >= s.tile && x - s.tile < run_tiles(s, p.chunk)) {
        sh.seq = i;
        sh.found = 1;
      }
    }
  });
  if (sh.found == 0 || x >= p.tiles) return;
  const QwDnSeq& s = p.seqs[sh.seq];
  const uint32_t from = (x - s.tile) * p.chunk;
  const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
  const uint32_t pos0 = s.first + from;
  const uint32_t key_dim = p.k_heads * kKey;
  const uint32_t group = p.v_heads / p.k_heads;
  par.run([&](uint32_t t) {
    const uint64_t tk = tile_key(p, x, kh) * kTile * kKey;
    for (uint32_t e = t; e < kTile * kKey; e += kThreads) {
      const uint32_t r = e / kKey;
      const uint32_t i = e % kKey;
      const uint64_t at = static_cast<uint64_t>(pos0 + r) * key_dim + kh * kKey + i;
      const float q = r < n ? p.q[at] : 0.0f;
      const float k = r < n ? p.k[at] : 0.0f;
      sh.q[r][i] = q;
      sh.k[r][i] = k;
      p.qr[tk + e] = tf32_rna(q);
      p.kr[tk + e] = tf32_rna(k);
    }
  });
  // The lower triangle's dots over the present rows, pair `e` = (r, j ≤ r) in
  // row order.
  par.run([&](uint32_t t) {
    for (uint32_t e = t; e < n * (n + 1) / 2; e += kThreads) {
      uint32_t r = 0;
      while ((r + 1) * (r + 2) / 2 <= e) ++r;
      const uint32_t j = e - r * (r + 1) / 2;
      sh.kk[r][j] = dot128(sh.k[r], sh.k[j]);
      sh.qk[r][j] = dot128(sh.q[r], sh.k[j]);
    }
  });
  for (uint32_t gi = 0; gi < group; ++gi) {
    const uint32_t h = kh * group + gi;
    const uint64_t th = tile_head(p, x, h);
    float* pm = p.pm + th * kTile * kTile;
    float* fac = p.fac + th * kFac;
    float* w = p.w + th * kTile * kKey;
    float* u = p.u + th * kTile * p.v_dim;
    par.run([&](uint32_t t) {
      if (t < kTile) {
        const uint64_t at = static_cast<uint64_t>(pos0 + t) * p.v_heads + h;
        sh.beta[t] = t < n ? p.beta[at] : 0.0f;
        sh.g[t] = t < n ? p.g[at] : 0.0f;
      }
    });
    // The cumulative log decay, every thread its own copy (the same sums).
    float gc[kTile];
    {
      float acc = sh.g[0];
      gc[0] = acc;
      for (uint32_t r = 1; r < kTile; ++r) {
        acc = acc + sh.g[r];
        gc[r] = acc;
      }
    }
    par.run([&](uint32_t t) {
      for (uint32_t e = t; e < kTile * kTile; e += kThreads) {
        const uint32_t r = e / kTile;
        const uint32_t j = e % kTile;
        float a = 0.0f;
        float pv = 0.0f;
        if (j <= r && r < n) {
          const float d = ds::exp32(gc[r] - gc[j]);
          pv = sh.qk[r][j] * d;
          if (j < r) a = (sh.beta[r] * sh.kk[r][j]) * d;
        }
        sh.a[r][j] = a;
        pm[e] = pv;
      }
      if (t < kTile) {
        const float eg = ds::exp32(gc[t]);
        sh.c[t] = sh.beta[t] * eg;
        fac[kFacG + t] = eg;
        fac[kFacGL + t] = ds::exp32(gc[p.chunk - 1] - gc[t]);
        if (t == 0) fac[kFacGamma] = ds::exp32(gc[p.chunk - 1]);
      }
    });
    // T, a column a thread: column j reads only its own entries above.
    par.run([&](uint32_t t) {
      if (t < kTile) {
        const uint32_t j = t;
        for (uint32_t r = 0; r < kTile; ++r) {
          float v = 0.0f;
          if (r == j) {
            v = 1.0f;
          } else if (r > j && r < n) {
            float acc = sh.a[r][j];
            for (uint32_t m = j + 1; m < r; ++m) acc = acc + sh.a[r][m] * sh.t[m][j];
            v = -acc;
          }
          sh.t[r][j] = v;
        }
      }
    });
    par.run([&](uint32_t t) {
      for (uint32_t e = t; e < kTile * kTile; e += kThreads) {
        const uint32_t r = e / kTile;
        const uint32_t j = e % kTile;
        const bool in = j <= r && r < n;
        sh.tc[r][j] = in ? sh.t[r][j] * sh.c[j] : 0.0f;
        sh.tb[r][j] = in ? sh.t[r][j] * sh.beta[j] : 0.0f;
      }
    });
    // W and U a column a thread, the column's present rows in registers.
    par.run([&](uint32_t t) {
      for (uint32_t i = t; i < kKey; i += kThreads) {
        float col[kTile];
        for (uint32_t j = 0; j < kTile; ++j) col[j] = sh.k[j][i];
        for (uint32_t r = 0; r < kTile; ++r) {
          float acc = 0.0f;
          if (r < n) {
            for (uint32_t j = 0; j <= r; ++j) acc = acc + sh.tc[r][j] * col[j];
          }
          w[r * kKey + i] = tf32_rna(acc);
        }
      }
      for (uint32_t c = t; c < p.v_dim; c += kThreads) {
        float col[kTile];
        for (uint32_t j = 0; j < kTile; ++j) {
          col[j] =
              j < n ? p.v[(static_cast<uint64_t>(pos0 + j) * p.v_heads + h) * p.v_dim + c] : 0.0f;
        }
        for (uint32_t r = 0; r < kTile; ++r) {
          float acc = 0.0f;
          if (r < n) {
            for (uint32_t j = 0; j <= r; ++j) acc = acc + sh.tb[r][j] * col[j];
          }
          u[r * p.v_dim + c] = acc;
        }
      }
    });
  }
}

// ---- the state kernel (`eidola_qwen_fast_dn_state`) ----
//
// A CTA owns one run, one value head and `kSlice` of its value columns (a warp
// 16 of them; columns are independent in the recurrence), and keeps that slice
// of the state in registers across the run's blocks, transposed (`Sᵀ`, value
// rows by key columns), in the tensor core's accumulator layout. Per block,
// on tensor cores (`mma.sync` m16n8k8, TF32): `Xᵀ = Sᵀ Wᵀ` and `Yᵀ = Sᵀ qᵀ`
// (16 K-steps over the key dimension, `Sᵀ` read as the left operand straight
// from its accumulator registers), then element by element `Δ = U − X`, `o =
// exp(G) · Y + Z` (`ztree`, over `Δ` staged in shared memory), and for a whole
// block `Sᵀ ← Sᵀ · γ + Δ′ᵀ K` (each warp its 16 rows by 16 eight-column tiles,
// two K-steps, from zero accumulators). A K-step's eight products pair `Sᵀ`'s
// (or `Δ′ᵀ`'s) columns `2t, 2t + 1` with the right operand's rows `2t, 2t + 1`,
// which is how the accumulator layout hands them over; a product's column is
// its right operand's column alone, so a position's results read only its own
// rows of `W`, `q` and `U`.

constexpr uint32_t kSlice = 64;     // value columns of a state CTA
constexpr uint32_t kWP = kKey + 8;  // W's and the queries' row stride
constexpr uint32_t kKP = kKey + 4;  // the keys' row stride
constexpr uint32_t kUP = kSlice + 4;
constexpr uint32_t kDP = kTile + 4; // a warp's Δ rows

struct QD_ALIGN32 StateShared {
  float w[kTile][kWP];
  float q[kTile][kWP];
  float k[kTile][kKP];
  float u[kTile][kUP];
  float pm[kTile][kTile];
  float fac[kFac];
  float d[kThreads / 32][16][kDP];
};

// The stores due once `done` positions have run (`done` a block boundary: the
// state exists nowhere else), from a host array `st[key][column]` of the
// slice's `cols` columns.
DS_FN void slice_store(const QwDnChunkParams& p, uint64_t addr, uint32_t h, uint32_t vbase,
                       uint32_t cols, const float* st) {
  float* dst = state_at(addr, p.state_offset);
  for (uint32_t i = 0; i < kKey; ++i)
    for (uint32_t c = 0; c < cols; ++c) dst[state_index(h, i, vbase + c, p.v_dim)] = st[i * kSlice + c];
}

#if !defined(__CUDA_ARCH__)

// ---- the state kernel on the host ----
//
// The device's steps with each tensor-core K-step read as: the 8 products of
// TF32 operands (exact in f32) summed from +0 in K order, then added to the
// accumulator. A tensor core sums a K-step its own way, so the host's bits are a
// reading of the form, not the device's: the tests hold the device to them
// within a bound and the host form to the token recurrence.

inline float host_kstep(const float* a, uint32_t as, const float* b, uint32_t bs) {
  float sum = 0.0f;
  for (uint32_t k = 0; k < 8; ++k) sum = sum + tf32_rna(a[k * as]) * tf32_rna(b[k * bs]);
  return sum;
}

// CTA (`x`, value head `h`) of the state kernel.
inline void dn_state_host(const QwDnChunkParams& p, uint32_t x, uint32_t h) {
  const uint32_t slices = (p.v_dim + kSlice - 1) / kSlice;
  const uint32_t seq = x / slices;
  const uint32_t vbase = (x % slices) * kSlice;
  if (seq >= p.seq_count) return;
  const QwDnSeq s = p.seqs[seq];
  const uint32_t nb = run_tiles(s, p.chunk);
  if (s.tile + nb > p.tiles) return;
  const uint32_t cols = p.v_dim - vbase < kSlice ? p.v_dim - vbase : kSlice;
  const uint32_t kh = h / (p.v_heads / p.k_heads);
  static thread_local float st[kKey * kSlice];
  for (uint32_t i = 0; i < kKey; ++i)
    for (uint32_t c = 0; c < kSlice; ++c)
      st[i * kSlice + c] = s.src == 0 || c >= cols
                               ? 0.0f
                               : state_at(s.src, p.state_offset)[state_index(h, i, vbase + c,
                                                                             p.v_dim)];
  auto stores = [&](uint32_t done) {
    if (s.live != 0 && s.live_after == done) slice_store(p, s.live, h, vbase, cols, st);
    for (uint32_t w = 0; w < s.writes && w < 4; ++w)
      if (s.write_after[w] == done) slice_store(p, s.write_dst[w], h, vbase, cols, st);
  };
  stores(0);
  for (uint32_t b = 0; b < nb; ++b) {
    const uint32_t from = b * p.chunk;
    const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
    const uint64_t th = tile_head(p, s.tile + b, h);
    const uint64_t tk = tile_key(p, s.tile + b, kh);
    const float* w = p.w + th * kTile * kKey;
    const float* u = p.u + th * kTile * p.v_dim;
    const float* pm = p.pm + th * kTile * kTile;
    const float* fac = p.fac + th * kFac;
    const float* qr = p.qr + tk * kTile * kKey;
    const float* kr = p.kr + tk * kTile * kKey;
    float d[kTile][kSlice];
    float y[kTile][kSlice];
    for (uint32_t r = 0; r < kTile; ++r)
      for (uint32_t c = 0; c < cols; ++c) {
        float xa = 0.0f;
        float ya = 0.0f;
        for (uint32_t ks = 0; ks < kKey / 8; ++ks) {
          xa = xa + host_kstep(w + r * kKey + 8 * ks, 1, st + 8 * ks * kSlice + c, kSlice);
          ya = ya + host_kstep(qr + r * kKey + 8 * ks, 1, st + 8 * ks * kSlice + c, kSlice);
        }
        d[r][c] = u[r * p.v_dim + vbase + c] - xa;
        y[r][c] = ya;
      }
    for (uint32_t r = 0; r < n; ++r) {
      if (from + r < s.out_from) continue;
      for (uint32_t c = 0; c < cols; ++c) {
        float dj[kTile];
        for (uint32_t j = 0; j < kTile; ++j) dj[j] = d[j][c];
        const float o = fac[kFacG + r] * y[r][c] + ztree(pm + r * kTile, dj);
        const uint64_t pos = s.first + from + r;
        p.out[(pos * p.v_heads + h) * p.v_dim + vbase + c] = ds::bf16_round(o);
      }
    }
    if (n == p.chunk) {
      float dl[kTile][kSlice];
      for (uint32_t j = 0; j < kTile; ++j)
        for (uint32_t c = 0; c < cols; ++c) dl[j][c] = j < n ? fac[kFacGL + j] * d[j][c] : 0.0f;
      const float gamma = fac[kFacGamma];
      for (uint32_t i = 0; i < kKey; ++i)
        for (uint32_t c = 0; c < cols; ++c) {
          float dd = 0.0f;
          for (uint32_t ks = 0; ks < kTile / 8; ++ks)
            dd = dd + host_kstep(kr + 8 * ks * kKey + i, kKey, &dl[8 * ks][c], kSlice);
          float& sv = st[i * kSlice + c];
          sv = sv * gamma + dd;
        }
      stores(from + n);
    }
  }
}

#endif

}  // namespace qwd
