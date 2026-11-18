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
//   o_r    = exp(G_r) · (q_r S) + Σ_{j≤r} P_rj · Δ_j   (q S on tensor cores, the
//            sum in order), rounded to BF16
//
// and when the block is whole (`n = L`) the state after it is
// `S' = S · exp(G_{L−1}) + Kᵀ Δ′` with `Δ′_j = exp(G_{L−1} − G_j) · Δ_j` (Kᵀ Δ′ on
// tensor cores), which is the token recurrence's algebra
// (`deltanet::recurrent_step`) regrouped: the same output and state in exact
// arithmetic, different roundings. The dot products `k_r·k_j` and `q_r·k_j` are `pinned_dot`s, every
// other sum runs in the order written, from +0, IEEE `+ - *` (the image is
// built without contraction), `exp` is `exp32`. Every tensor-core operand is
// rounded to TF32 (nearest, ties away) and every product of one accumulates in
// f32; `S` stays f32 between blocks.
//
// Why a position's output and every stored state are functions of the token
// prefix alone. A block is always a block of the same absolute positions (runs
// start at boundaries), its tile always 16 rows with positions past `n` zero
// in every operand (`q = k = v = β = g = 0`). Row `r` of every quantity above
// reads rows `≤ r` only: each loop over `j` or `m` stops at `r`, and the
// tensor-core products give row `r` from row `r` of their left operand and the
// whole state, which never depends on the rows after it. So row `r`'s bits are
// the same whether the block holds `r + 1` positions or `L`, whether it is a
// prefill's, a decode step's recomputation from the boundary or a fold's, and
// whatever else the launch holds; the state is updated only from whole blocks,
// whose tiles are the same in every path. Nothing reduces over positions with
// a tensor core but the update, so no result rests on a tensor core adding an
// exact zero.

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
constexpr uint32_t kCols = 32;     // value columns of a state CTA
constexpr uint32_t kFac = 48;      // floats of a block's factors
// Padded row strides of the state CTA's staged tiles (multiples of 4 floats;
// every tile's first row 32-byte aligned).
constexpr uint32_t kSP = kCols + 4;
constexpr uint32_t kAP = kKey + 4;

// A block's factors: `exp(G_r)` at `kFacG`, `exp(G_{L−1} − G_r)` at
// `kFacGL`, `exp(G_{L−1})` at `kFacGamma`.
constexpr uint32_t kFacG = 0;
constexpr uint32_t kFacGL = kTile;
constexpr uint32_t kFacGamma = 2 * kTile;

DS_FN uint32_t run_tiles(const QwDnSeq& s, uint32_t chunk) {
  return (s.count + chunk - 1) / chunk;
}

// The scratch's index of (tile, value head).
DS_FN uint64_t tile_head(const QwDnChunkParams& p, uint32_t tile, uint32_t h) {
  return static_cast<uint64_t>(tile) * p.v_heads + h;
}

// State value `i` of column `j` of value head `h` in a layer's state.
DS_FN uint64_t state_index(uint32_t h, uint32_t i, uint32_t j, uint32_t vd) {
  return (static_cast<uint64_t>(h) * kKey + i) * vd + j;
}

DS_FN float* state_at(uint64_t addr, uint64_t offset) {
  return reinterpret_cast<float*>(addr + offset);
}

// ---- the block's state-free parts (`eidola_qwen_fast_dn_intra`) ----

struct IntraShared {
  float q[kTile][kKey];
  float k[kTile][kKey];
  float v[kTile][kKey];
  float kk[kTile][kTile];
  float qk[kTile][kTile];
  float a[kTile][kTile];
  float t[kTile][kTile];
  float beta[kTile];
  float g[kTile];
  float gc[kTile];
  float c[kTile];
  uint32_t seq;
  uint32_t found;
};

// CTA (`x`, key head `kh`): block `x` of the launch, its key head's queries and
// keys, then each of the key head's value heads: `W`, `U`, `P` and the
// factors into the scratch.
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
    for (uint32_t e = t; e < kTile * kKey; e += kThreads) {
      const uint32_t r = e / kKey;
      const uint32_t i = e % kKey;
      const uint64_t at = static_cast<uint64_t>(pos0 + r) * key_dim + kh * kKey + i;
      sh.q[r][i] = r < n ? p.q[at] : 0.0f;
      sh.k[r][i] = r < n ? p.k[at] : 0.0f;
    }
  });
  // The lower triangle's dots, pair `e` = (r, j ≤ r) in row order.
  par.run([&](uint32_t t) {
    for (uint32_t e = t; e < kTile * (kTile + 1) / 2; e += kThreads) {
      uint32_t r = 0;
      while ((r + 1) * (r + 2) / 2 <= e) ++r;
      const uint32_t j = e - r * (r + 1) / 2;
      sh.kk[r][j] = ds::pinned_dot(sh.k[r], sh.k[j], kKey);
      sh.qk[r][j] = ds::pinned_dot(sh.q[r], sh.k[j], kKey);
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
      for (uint32_t e = t; e < kTile * p.v_dim; e += kThreads) {
        const uint32_t r = e / p.v_dim;
        const uint32_t c = e % p.v_dim;
        sh.v[r][c] =
            r < n ? p.v[(static_cast<uint64_t>(pos0 + r) * p.v_heads + h) * p.v_dim + c] : 0.0f;
      }
      if (t < kTile) {
        const uint64_t at = static_cast<uint64_t>(pos0 + t) * p.v_heads + h;
        sh.beta[t] = t < n ? p.beta[at] : 0.0f;
        sh.g[t] = t < n ? p.g[at] : 0.0f;
      }
    });
    par.run([&](uint32_t t) {
      if (t == 0) {
        float acc = sh.g[0];
        sh.gc[0] = acc;
        for (uint32_t r = 1; r < kTile; ++r) {
          acc = acc + sh.g[r];
          sh.gc[r] = acc;
        }
      }
    });
    par.run([&](uint32_t t) {
      for (uint32_t e = t; e < kTile * kTile; e += kThreads) {
        const uint32_t r = e / kTile;
        const uint32_t j = e % kTile;
        float a = 0.0f;
        float pv = 0.0f;
        if (j <= r) {
          const float d = ds::exp32(sh.gc[r] - sh.gc[j]);
          pv = sh.qk[r][j] * d;
          if (j < r) a = (sh.beta[r] * sh.kk[r][j]) * d;
        }
        sh.a[r][j] = a;
        pm[e] = pv;
      }
      if (t < kTile) {
        const float eg = ds::exp32(sh.gc[t]);
        sh.c[t] = sh.beta[t] * eg;
        fac[kFacG + t] = eg;
        fac[kFacGL + t] = ds::exp32(sh.gc[p.chunk - 1] - sh.gc[t]);
        if (t == 0) fac[kFacGamma] = ds::exp32(sh.gc[p.chunk - 1]);
      }
    });
    // T, a column a thread: column j reads only its own entries above.
    par.run([&](uint32_t t) {
      if (t < kTile) {
        const uint32_t j = t;
        for (uint32_t r = 0; r < kTile; ++r) {
          float v;
          if (r < j) {
            v = 0.0f;
          } else if (r == j) {
            v = 1.0f;
          } else {
            float acc = sh.a[r][j];
            for (uint32_t m = j + 1; m < r; ++m) acc = acc + sh.a[r][m] * sh.t[m][j];
            v = -acc;
          }
          sh.t[r][j] = v;
        }
      }
    });
    par.run([&](uint32_t t) {
      for (uint32_t i = t; i < kKey; i += kThreads) {
        for (uint32_t r = 0; r < kTile; ++r) {
          float acc = 0.0f;
          for (uint32_t j = 0; j <= r; ++j) acc = acc + (sh.t[r][j] * sh.c[j]) * sh.k[j][i];
          w[r * kKey + i] = acc;
        }
      }
      for (uint32_t c = t; c < p.v_dim; c += kThreads) {
        for (uint32_t r = 0; r < kTile; ++r) {
          float acc = 0.0f;
          for (uint32_t j = 0; j <= r; ++j) acc = acc + (sh.t[r][j] * sh.beta[j]) * sh.v[j][c];
          u[r * p.v_dim + c] = acc;
        }
      }
    });
  }
}

// ---- the state kernel's scalar steps (`eidola_qwen_fast_dn_state`) ----
//
// A CTA owns one run, one value head and `kCols` of its value columns (columns
// are independent in the recurrence) and keeps that slice of `S` in shared
// memory across the run's blocks. Per block: `W S` and `q S` on tensor cores
// (four warps, a 16 × 16 output tile each: two of `W S`, then `Δ = U − W S`,
// and two of `q S`; the 16 K-steps alternating between two accumulators that
// are added at the end); then element by element `o` and `Δ′`; and for a whole
// block the update, `Kᵀ Δ′` on tensor cores (each warp four tiles of `S`'s 128
// rows by 32 columns, two K-steps) and `S · γ + D` element by element.

struct QD_ALIGN32 StateShared {
  float s[kKey][kSP];    // the state slice
  float w[kTile][kAP];   // W, then the keys
  float q[kTile][kAP];   // the queries
  float u[kTile][kSP];   // U's slice, then Δ′
  float d[kTile][kSP];   // Δ
  float y[kTile][kSP];   // q S
  float z[kTile][kSP];   // o
  float pm[kTile][kTile];
  float eg[kTile];
  float egl[kTile];
  float gamma;
};

// The run's state slice into `s` (zeros from the null source), threads in
// turn: row `i = e / kCols`, column `e % kCols` of element `e = t + kThreads·n`.
DS_FN void state_load(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t h, uint32_t slice,
                      uint32_t t, float (*st)[kSP]) {
  for (uint32_t e = t; e < kKey * kCols; e += kThreads) {
    const uint32_t i = e / kCols;
    const uint32_t c = e % kCols;
    st[i][c] = s.src == 0 ? 0.0f
                          : state_at(s.src, p.state_offset)[state_index(
                                h, i, slice * kCols + c, p.v_dim)];
  }
}

DS_FN void state_store(const QwDnChunkParams& p, uint64_t addr, uint32_t h, uint32_t slice,
                       uint32_t t, const float (*st)[kSP]) {
  float* dst = state_at(addr, p.state_offset);
  for (uint32_t e = t; e < kKey * kCols; e += kThreads) {
    const uint32_t i = e / kCols;
    const uint32_t c = e % kCols;
    dst[state_index(h, i, slice * kCols + c, p.v_dim)] = st[i][c];
  }
}

// The stores due once `done` positions have run (`done` a block boundary: the
// state exists nowhere else).
DS_FN void state_stores(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t done, uint32_t h,
                        uint32_t slice, uint32_t t, const float (*st)[kSP]) {
  if (s.live != 0 && s.live_after == done) state_store(p, s.live, h, slice, t, st);
  for (uint32_t w = 0; w < s.writes && w < 4; ++w) {
    if (s.write_after[w] == done) state_store(p, s.write_dst[w], h, slice, t, st);
  }
}

// Element `e = t + kThreads·m` of the 16 × kCols tile, for the block's `n`
// rows (zero past them): `o_r = exp(G_r) · (q S)_r + Σ_{j≤r} P_rj · Δ_j` into
// `sh.z`, and `Δ′_r = exp(G_{L−1} − G_r) · Δ_r` into `sh.u` (its `U` read).
DS_FN void block_rows(uint32_t n, uint32_t t, StateShared& sh) {
  for (uint32_t e = t; e < kTile * kCols; e += kThreads) {
    const uint32_t r = e / kCols;
    const uint32_t c = e % kCols;
    float o = 0.0f;
    float dl = 0.0f;
    if (r < n) {
      float acc = 0.0f;
      for (uint32_t j = 0; j <= r; ++j) acc = acc + sh.pm[r][j] * sh.d[j][c];
      o = sh.eg[r] * sh.y[r][c] + acc;
      dl = sh.egl[r] * sh.d[r][c];
    }
    sh.z[r][c] = o;
    sh.u[r][c] = dl;
  }
}

// The block's outputs: row `r < n` at relative position `from + r ≥ out_from`,
// `o` rounded to BF16.
DS_FN void block_out(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t from, uint32_t n,
                     uint32_t h, uint32_t slice, uint32_t t, const float (*o)[kSP]) {
  for (uint32_t e = t; e < kTile * kCols; e += kThreads) {
    const uint32_t r = e / kCols;
    const uint32_t c = e % kCols;
    if (r < n && from + r >= s.out_from) {
      const uint64_t pos = s.first + from + r;
      p.out[(pos * p.v_heads + h) * p.v_dim + slice * kCols + c] = ds::bf16_round(o[r][c]);
    }
  }
}

// TF32, nearest with ties away from zero (`cvt.rna.tf32.f32`), as f32 bits
// with the low 13 bits clear; infinities and NaNs as they are.
DS_FN float tf32_rna(float x) {
  const uint32_t u = ds::f32_bits(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  return ds::f32_from_bits((u + 0x1000u) & 0xffffe000u);
}

#if !defined(__CUDA_ARCH__)

// ---- the state kernel on the host ----
//
// The device's steps in the device's order, with each tensor-core K-step of 8
// read as: the 8 products of TF32-rounded operands (exact in f32) summed from
// +0 in K order, then added to the accumulator. A tensor core sums a K-step its
// own way, so the host's bits are a reading of the form, not the device's:
// the tests hold the device to them within a bound and the host form to the
// token recurrence.

// `acc[m][c] += Σ_k a(m, k) · b(k, c)` over K-steps `k0 .. k0 + 8` of an
// `rows × cols` tile.
template <class A, class B>
inline void host_kstep(float (*acc)[kSP], uint32_t rows, uint32_t cols, uint32_t k0, A a, B b) {
  for (uint32_t m = 0; m < rows; ++m)
    for (uint32_t c = 0; c < cols; ++c) {
      float sum = 0.0f;
      for (uint32_t k = k0; k < k0 + 8; ++k) sum = sum + tf32_rna(a(m, k)) * tf32_rna(b(k, c));
      acc[m][c] = acc[m][c] + sum;
    }
}

// CTA (`x`, value head `h`) of the state kernel.
inline void dn_state_host(const QwDnChunkParams& p, uint32_t x, uint32_t h, StateShared& sh) {
  const uint32_t slices = p.v_dim / kCols;
  const uint32_t seq = x / slices;
  const uint32_t slice = x % slices;
  if (seq >= p.seq_count) return;
  const QwDnSeq s = p.seqs[seq];
  const uint32_t nb = run_tiles(s, p.chunk);
  if (s.tile + nb > p.tiles) return;
  const uint32_t key_dim = p.k_heads * kKey;
  const uint32_t kh = h / (p.v_heads / p.k_heads);
  for (uint32_t t = kThreads; t-- > 0;) state_load(p, s, h, slice, t, sh.s);
  for (uint32_t t = kThreads; t-- > 0;) state_stores(p, s, 0, h, slice, t, sh.s);
  for (uint32_t b = 0; b < nb; ++b) {
    const uint32_t from = b * p.chunk;
    const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
    const uint64_t th = tile_head(p, s.tile + b, h);
    const float* w = p.w + th * kTile * kKey;
    const float* u = p.u + th * kTile * p.v_dim;
    const float* pm = p.pm + th * kTile * kTile;
    const float* fac = p.fac + th * kFac;
    float kr[kTile][kKey];
    for (uint32_t r = 0; r < kTile; ++r)
      for (uint32_t i = 0; i < kKey; ++i) {
        const uint64_t at = static_cast<uint64_t>(s.first + from + r) * key_dim + kh * kKey + i;
        sh.w[r][i] = w[r * kKey + i];
        sh.q[r][i] = r < n ? p.q[at] : 0.0f;
        kr[r][i] = r < n ? p.k[at] : 0.0f;
      }
    for (uint32_t r = 0; r < kTile; ++r) {
      for (uint32_t c = 0; c < kCols; ++c) sh.u[r][c] = u[r * p.v_dim + slice * kCols + c];
      for (uint32_t j = 0; j < kTile; ++j) sh.pm[r][j] = pm[r * kTile + j];
      sh.eg[r] = fac[kFacG + r];
      sh.egl[r] = fac[kFacGL + r];
    }
    sh.gamma = fac[kFacGamma];
    // W S and q S, the K-steps alternating between two accumulators.
    float x0[kTile][kSP] = {}, x1[kTile][kSP] = {}, y0[kTile][kSP] = {}, y1[kTile][kSP] = {};
    for (uint32_t ks = 0; ks < kKey / 8; ++ks) {
      auto bs = [&](uint32_t k, uint32_t c) { return sh.s[k][c]; };
      host_kstep(ks % 2 == 0 ? x0 : x1, kTile, kCols, 8 * ks,
                 [&](uint32_t m, uint32_t k) { return sh.w[m][k]; }, bs);
      host_kstep(ks % 2 == 0 ? y0 : y1, kTile, kCols, 8 * ks,
                 [&](uint32_t m, uint32_t k) { return sh.q[m][k]; }, bs);
    }
    for (uint32_t r = 0; r < kTile; ++r)
      for (uint32_t c = 0; c < kCols; ++c) {
        sh.d[r][c] = sh.u[r][c] - (x0[r][c] + x1[r][c]);
        sh.y[r][c] = y0[r][c] + y1[r][c];
      }
    for (uint32_t t = kThreads; t-- > 0;) block_rows(n, t, sh);
    if (n == p.chunk) {
      for (uint32_t r = 0; r < kTile; ++r)
        for (uint32_t i = 0; i < kKey; ++i) sh.w[r][i] = kr[r][i];
      for (uint32_t mt = 0; mt < kKey / kTile; ++mt) {
        float dd[kTile][kSP] = {};
        for (uint32_t ks = 0; ks < kTile / 8; ++ks)
          host_kstep(
              dd, kTile, kCols, 8 * ks,
              [&](uint32_t m, uint32_t k) { return sh.w[k][mt * kTile + m]; },
              [&](uint32_t k, uint32_t c) { return sh.u[k][c]; });
        for (uint32_t m = 0; m < kTile; ++m)
          for (uint32_t c = 0; c < kCols; ++c) {
            float& sv = sh.s[mt * kTile + m][c];
            sv = sv * sh.gamma + dd[m][c];
          }
      }
    }
    for (uint32_t t = kThreads; t-- > 0;) block_out(p, s, from, n, h, slice, t, sh.z);
    if (n == p.chunk)
      for (uint32_t t = kThreads; t-- > 0;) state_stores(p, s, from + n, h, slice, t, sh.s);
  }
}

#endif

}  // namespace qwd
