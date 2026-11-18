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

// ---- the tail cache ----
//
// A position's intermediates, kept from the step that first computes it until
// its block's state is stored: they are what a later step needs of it, so a
// decode step computes only its new rows. Per layer and tail row, an entry of
// `entry_floats` floats: per key head the position's key (`k_dim`), then per
// value head a record of `rec_floats`: its value and `Δ` (`v_dim` each), its
// row of `T` (16), its write strength, its cumulative decay `G` and two pad
// floats. Every value is the bits the position's own computation produced,
// and a row of each quantity reads only rows at or before it, so a row
// computed beside cached rows is the row computed with the whole block.

DS_FN uint32_t rec_floats(const QwDnChunkParams& p) { return 2 * p.v_dim + kTile + 4; }
// Floats of a row of the `rec` scratch: its row of `T`, `β`, `G` and two pad floats.
constexpr uint32_t kRec = kTile + 4;
DS_FN uint32_t entry_floats(const QwDnChunkParams& p) {
  return p.k_heads * kKey + p.v_heads * rec_floats(p);
}
// The entry of the run's relative position `rel`.
DS_FN float* entry_at(const QwDnChunkParams& p, const QwDnSeq& s, uint32_t rel) {
  const uint64_t row = s.tail_row + (static_cast<uint64_t>(s.pos0) + rel) % p.ring;
  return p.tail + row * entry_floats(p);
}
// Value head `h`'s record in an entry.
DS_FN float* rec_at(const QwDnChunkParams& p, float* entry, uint32_t h) {
  return entry + p.k_heads * kKey + h * rec_floats(p);
}
// The staged index of relative position `rel` (at or past `out_from`).
DS_FN uint64_t staged_at(const QwDnSeq& s, uint32_t rel) {
  return static_cast<uint64_t>(s.first) + (rel - s.out_from);
}
// The block's cached rows: those before the run's `out_from`.
DS_FN uint32_t cached_rows(const QwDnSeq& s, uint32_t from, uint32_t n) {
  if (s.out_from <= from) return 0;
  return s.out_from - from < n ? s.out_from - from : n;
}

// The run holding block `x` of the launch (the one whose blocks `tile ..` reach past
// it) into `seq`, `found` 1; `found` 0 when no run does.
template <class P>
DS_FN void find_run(const QwDnChunkParams& p, uint32_t x, const P& par, uint32_t& seq,
                    uint32_t& found) {
  par.run([&](uint32_t t) {
    if (t == 0) found = 0;
  });
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < p.seq_count; i += kThreads) {
      const QwDnSeq& s = p.seqs[i];
      if (x >= s.tile && x - s.tile < run_tiles(s, p.chunk)) {
        seq = i;
        found = 1;
      }
    }
  });
}

// ---- keeping the new positions (`eidola_qwen_fast_dn_keep`) ----
//
// CTA (`x`, key head `kh`), after the state kernel: block `x`'s kept new positions'
// key, and each of the key head's value heads' value, row of `T`, `β` and `G`, from
// what the intra kernel left, into the tail cache (the state kernel writes `Δ`). It
// runs after both kernels have read the cache: a long run's cached and kept
// positions can share a ring row.

struct QD_ALIGN32 KeepShared {
  uint32_t seq;
  uint32_t found;
};

template <class P>
DS_FN void dn_keep_block(const QwDnChunkParams& p, uint32_t x, uint32_t kh, const P& par,
                         KeepShared& sh) {
  find_run(p, x, par, sh.seq, sh.found);
  if (sh.found == 0 || x >= p.tiles) return;
  const QwDnSeq& s = p.seqs[sh.seq];
  const uint32_t from = (x - s.tile) * p.chunk;
  const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
  const uint32_t lo = cached_rows(s, from, n);
  const uint32_t group = p.v_heads / p.k_heads;
  par.run([&](uint32_t t) {
    for (uint32_t r = lo; r < n; ++r) {
      if (from + r < s.keep_from) continue;
      float* entry = entry_at(p, s, from + r);
      const float* k = p.kraw + (tile_key(p, x, kh) * kTile + r) * kKey;
      for (uint32_t i = t; i < kKey; i += kThreads) entry[kh * kKey + i] = k[i];
      const uint64_t at = staged_at(s, from + r);
      for (uint32_t e = t; e < group * p.v_dim; e += kThreads) {
        const uint32_t h = kh * group + e / p.v_dim;
        const uint32_t c = e % p.v_dim;
        rec_at(p, entry, h)[c] = p.v[(at * p.v_heads + h) * p.v_dim + c];
      }
      for (uint32_t e = t; e < group * (kTile + 2); e += kThreads) {
        const uint32_t h = kh * group + e / (kTile + 2);
        const uint32_t j = e % (kTile + 2);
        rec_at(p, entry, h)[2 * p.v_dim + j] = p.rec[(tile_head(p, x, h) * kTile + r) * kRec + j];
      }
    }
  });
}

// ---- the block's state-free parts (`eidola_qwen_fast_dn_intra`) ----
//
// Every loop over rows runs over the block's new ones, and it writes the rows the
// state kernel reads: the new rows of `W`, `U` and the rounded queries (the state
// kernel reads the others as zeros), and the rounded keys of every row only for a
// whole block, the only one whose update reads them (zero past its `n` positions).
// So a decode step's block costs its new position.

// The key and query rows are 129 floats apart, so the threads of a pair's dot,
// each reading another row at one column, meet in distinct banks.
constexpr uint32_t kQK = kKey + 1;
// The most value heads a key head groups.
constexpr uint32_t kMaxGroup = 4;

struct QD_ALIGN32 IntraShared {
  float q[kTile][kQK];
  float k[kTile][kQK];
  float kk[kTile][kTile];
  float qk[kTile][kTile];
  float a[kMaxGroup][kTile][kTile];
  float t[kMaxGroup][kTile][kTile];
  float tc[kMaxGroup][kTile][kTile];
  float tb[kMaxGroup][kTile][kTile];
  float beta[kMaxGroup][kTile];
  float g[kMaxGroup][kTile];
  float gc[kMaxGroup][kTile];
  float c[kMaxGroup][kTile];
  uint32_t seq;
  uint32_t found;
};

// CTA (`x`, key head `kh`): block `x` of the launch, its key head's queries and
// keys (and their TF32 roundings, and the keys, into the scratch), then all of
// the key head's value heads at once: `W` (rounded to TF32), `U`, `P`, the
// factors, and the rows of `T`, write strengths and cumulative decays the state
// kernel keeps in the tail cache.
template <class P>
DS_FN void dn_intra_block(const QwDnChunkParams& p, uint32_t x, uint32_t kh, const P& par,
                          IntraShared& sh) {
  find_run(p, x, par, sh.seq, sh.found);
  if (sh.found == 0 || x >= p.tiles) return;
  const QwDnSeq& s = p.seqs[sh.seq];
  const uint32_t from = (x - s.tile) * p.chunk;
  const uint32_t n = s.count - from < p.chunk ? s.count - from : p.chunk;
  const uint32_t lo = cached_rows(s, from, n);
  const bool whole = n == p.chunk;
  const uint32_t key_dim = p.k_heads * kKey;
  const uint32_t group = p.v_heads / p.k_heads;
  const uint32_t tr = 2 * p.v_dim;  // a record's row of `T`
  // The key head's queries and keys (a cached row's key from the cache), and each
  // value head's write strengths and log decays (a cached row's cumulative decay and
  // row of `T` from the cache).
  par.run([&](uint32_t t) {
    const uint64_t tk = tile_key(p, x, kh) * kTile * kKey;
    for (uint32_t e = t; e < kTile * kKey; e += kThreads) {
      const uint32_t r = e / kKey;
      const uint32_t i = e % kKey;
      float q = 0.0f;
      float k = 0.0f;
      if (r < lo) {
        k = entry_at(p, s, from + r)[kh * kKey + i];
      } else if (r < n) {
        const uint64_t at = staged_at(s, from + r) * key_dim + kh * kKey + i;
        q = p.q[at];
        k = p.k[at];
      }
      sh.q[r][i] = q;
      sh.k[r][i] = k;
      if (r >= lo && r < n) {
        p.qr[tk + e] = tf32_rna(q);
        p.kraw[tk + e] = k;
      }
      if (whole) p.kr[tk + e] = tf32_rna(k);
    }
    for (uint32_t e = t; e < group * kTile; e += kThreads) {
      const uint32_t gi = e / kTile;
      const uint32_t r = e % kTile;
      const uint32_t h = kh * group + gi;
      float beta = 0.0f;
      float g = 0.0f;
      if (r < lo) {
        const float* rec = rec_at(p, entry_at(p, s, from + r), h);
        beta = rec[tr + kTile];
        g = rec[tr + kTile + 1];  // the cached row's cumulative decay
      } else if (r < n) {
        const uint64_t at = staged_at(s, from + r) * p.v_heads + h;
        beta = p.beta[at];
        g = p.g[at];
      }
      sh.beta[gi][r] = beta;
      sh.g[gi][r] = g;
    }
    for (uint32_t e = t; e < group * lo * kTile; e += kThreads) {
      const uint32_t gi = e / (lo * kTile);
      const uint32_t r = e / kTile % lo;
      const uint32_t j = e % kTile;
      sh.t[gi][r][j] = rec_at(p, entry_at(p, s, from + r), kh * group + gi)[tr + j];
    }
  });
  // The new rows' dots, pair `e` = (r, j ≤ r) in row order; and each value head's
  // cumulative log decay: the cached rows' as they were computed, then on from them.
  par.run([&](uint32_t t) {
    for (uint32_t e = lo * (lo + 1) / 2 + t; e < n * (n + 1) / 2; e += kThreads) {
      uint32_t r = 0;
      while ((r + 1) * (r + 2) / 2 <= e) ++r;
      const uint32_t j = e - r * (r + 1) / 2;
      sh.kk[r][j] = dot128(sh.k[r], sh.k[j]);
      sh.qk[r][j] = dot128(sh.q[r], sh.k[j]);
    }
    if (t < group) {
      float acc = 0.0f;
      for (uint32_t r = 0; r < kTile; ++r) {
        if (r < lo) {
          acc = sh.g[t][r];
        } else if (r == 0) {
          acc = sh.g[t][0];
        } else {
          acc = acc + sh.g[t][r];
        }
        sh.gc[t][r] = acc;
      }
    }
  });
  par.run([&](uint32_t t) {
    for (uint32_t e = t; e < group * kTile * kTile; e += kThreads) {
      const uint32_t gi = e / (kTile * kTile);
      const uint32_t r = e / kTile % kTile;
      const uint32_t j = e % kTile;
      const float* gc = sh.gc[gi];
      float a = 0.0f;
      float pv = 0.0f;
      if (j <= r && r >= lo && r < n) {
        const float d = ds::exp32(gc[r] - gc[j]);
        pv = sh.qk[r][j] * d;
        if (j < r) a = (sh.beta[gi][r] * sh.kk[r][j]) * d;
      }
      sh.a[gi][r][j] = a;
      p.pm[(tile_head(p, x, kh * group + gi) * kTile + r) * kTile + j] = pv;
    }
    for (uint32_t e = t; e < group * kTile; e += kThreads) {
      const uint32_t gi = e / kTile;
      const uint32_t r = e % kTile;
      const float* gc = sh.gc[gi];
      float* fac = p.fac + tile_head(p, x, kh * group + gi) * kFac;
      const float eg = ds::exp32(gc[r]);
      sh.c[gi][r] = sh.beta[gi][r] * eg;
      fac[kFacG + r] = eg;
      fac[kFacGL + r] = ds::exp32(gc[p.chunk - 1] - gc[r]);
      if (r == 0) fac[kFacGamma] = ds::exp32(gc[p.chunk - 1]);
    }
  });
  // T, a column a thread for every value head: column j reads only its own entries
  // above (the cached rows' as they were computed).
  par.run([&](uint32_t t) {
    if (t < group * kTile) {
      const uint32_t gi = t / kTile;
      const uint32_t j = t % kTile;
      for (uint32_t r = lo; r < kTile; ++r) {
        float v = 0.0f;
        if (r == j) {
          v = 1.0f;
        } else if (r > j && r < n) {
          float acc = sh.a[gi][r][j];
          for (uint32_t m = j + 1; m < r; ++m) acc = acc + sh.a[gi][r][m] * sh.t[gi][m][j];
          v = -acc;
        }
        sh.t[gi][r][j] = v;
      }
    }
  });
  par.run([&](uint32_t t) {
    for (uint32_t e = t; e < group * kTile * kTile; e += kThreads) {
      const uint32_t gi = e / (kTile * kTile);
      const uint32_t r = e / kTile % kTile;
      const uint32_t j = e % kTile;
      const bool in = j <= r && r >= lo && r < n;
      sh.tc[gi][r][j] = in ? sh.t[gi][r][j] * sh.c[gi][j] : 0.0f;
      sh.tb[gi][r][j] = in ? sh.t[gi][r][j] * sh.beta[gi][j] : 0.0f;
      p.rec[(tile_head(p, x, kh * group + gi) * kTile + r) * kRec + j] = sh.t[gi][r][j];
    }
    for (uint32_t e = t; e < group * kTile; e += kThreads) {
      const uint32_t gi = e / kTile;
      const uint32_t r = e % kTile;
      float* rs = p.rec + (tile_head(p, x, kh * group + gi) * kTile + r) * kRec;
      rs[kTile] = sh.beta[gi][r];
      rs[kTile + 1] = sh.gc[gi][r];
    }
  });
  // W and U a column a thread, the column's rows in registers (the key column once for
  // every value head).
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < kKey; i += kThreads) {
      float col[kTile];
      for (uint32_t j = 0; j < kTile; ++j) col[j] = sh.k[j][i];
      for (uint32_t gi = 0; gi < group; ++gi) {
        float* w = p.w + tile_head(p, x, kh * group + gi) * kTile * kKey;
        for (uint32_t r = 0; r < kTile; ++r) {
          if (r < lo || r >= n) continue;
          float acc = 0.0f;
          for (uint32_t j = 0; j <= r; ++j) acc = acc + sh.tc[gi][r][j] * col[j];
          w[r * kKey + i] = tf32_rna(acc);
        }
      }
    }
    for (uint32_t e = t; e < group * p.v_dim; e += kThreads) {
      const uint32_t gi = e / p.v_dim;
      const uint32_t c = e % p.v_dim;
      const uint32_t h = kh * group + gi;
      float* u = p.u + tile_head(p, x, h) * kTile * p.v_dim;
      float col[kTile];
      for (uint32_t j = 0; j < kTile; ++j) {
        float v = 0.0f;
        if (j < lo) {
          v = rec_at(p, entry_at(p, s, from + j), h)[c];
        } else if (j < n) {
          v = p.v[(staged_at(s, from + j) * p.v_heads + h) * p.v_dim + c];
        }
        col[j] = v;
      }
      for (uint32_t r = 0; r < kTile; ++r) {
        if (r < lo || r >= n) continue;
        float acc = 0.0f;
        for (uint32_t j = 0; j <= r; ++j) acc = acc + sh.tb[gi][r][j] * col[j];
        u[r * p.v_dim + c] = acc;
      }
    }
  });
}

// ---- the state kernel (`eidola_qwen_fast_dn_state`) ----
//
// A CTA owns one run, one value head and `kSlice` of its value columns (a warp 16
// of them; columns are independent in the recurrence), and keeps that slice of
// the state in registers across the run's blocks, transposed (`Sᵀ`, value rows by
// key columns), in the tensor core's accumulator layout. Per block, on tensor
// cores (`mma.sync` m16n8k8, TF32): `Xᵀ = Sᵀ Wᵀ` and `Yᵀ = Sᵀ qᵀ` (16 K-steps
// over the key dimension, `Sᵀ` read as the left operand straight from its
// accumulator registers), then element by element `Δ = U − X` (a cached row's
// from the tail cache, a kept new row's into it), `o = exp(G) · Y + Z` (`ztree`,
// over `Δ` staged in shared memory), and for a whole block `Sᵀ ← Sᵀ · γ + Δ′ᵀ K`
// (each warp its 16 rows by 16 eight-column tiles, two K-steps, from zero
// accumulators). The rows of `W`, `q` and `U` outside the block's new ones are
// read as zeros, and the keys only for a whole block. A K-step's eight products pair `Sᵀ`'s
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
  float cd[kTile][kUP];  // the cached rows' `Δ`, the slice's columns
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
    const uint32_t lo = cached_rows(s, from, n);
    // The rows of `W`, `q` and `U` outside the new ones read as zeros.
    static const float zero[kKey] = {};
    for (uint32_t r = 0; r < kTile; ++r) {
      const bool fresh = r >= lo && r < n;
      const float* wr = fresh ? w + r * kKey : zero;
      const float* qrr = fresh ? qr + r * kKey : zero;
      for (uint32_t c = 0; c < cols; ++c) {
        float xa = 0.0f;
        float ya = 0.0f;
        for (uint32_t ks = 0; ks < kKey / 8; ++ks) {
          xa = xa + host_kstep(wr + 8 * ks, 1, st + 8 * ks * kSlice + c, kSlice);
          ya = ya + host_kstep(qrr + 8 * ks, 1, st + 8 * ks * kSlice + c, kSlice);
        }
        d[r][c] = (fresh ? u[r * p.v_dim + vbase + c] : 0.0f) - xa;
        y[r][c] = ya;
      }
    }
    // A cached row's `Δ` as it was computed; a kept new row's into the tail cache.
    for (uint32_t r = 0; r < n; ++r)
      for (uint32_t c = 0; c < cols; ++c) {
        float* rec = rec_at(p, entry_at(p, s, from + r), h);
        if (r < lo) {
          d[r][c] = rec[p.v_dim + vbase + c];
        } else if (from + r >= s.keep_from) {
          rec[p.v_dim + vbase + c] = d[r][c];
        }
      }
    for (uint32_t r = 0; r < n; ++r) {
      if (from + r < s.out_from) continue;
      for (uint32_t c = 0; c < cols; ++c) {
        float dj[kTile];
        for (uint32_t j = 0; j < kTile; ++j) dj[j] = d[j][c];
        const float o = fac[kFacG + r] * y[r][c] + ztree(pm + r * kTile, dj);
        const uint64_t pos = staged_at(s, from + r);
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
