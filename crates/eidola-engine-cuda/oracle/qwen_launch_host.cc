// The Qwen3.8 kernels' bodies (`eidola-engine-kernels`' `csrc/qwen_ops.cuh`
// and the fast profile's `csrc/qwen_fast.cuh` and `csrc/qwen_fast_deltanet.cuh`,
// the C++ a GPU compiles, and the blocked DeltaNet's state kernel's host form) as a
// host library: one function per launch, each repeating its kernel's entry
// over the grid the Rust wrapper computed, and the reference forms of the
// bodies arranged for speed over their own grids (`*_reference`). Blocks
// and threads run in descending order and shared memory starts as garbage, so
// a body that reads what another thread writes inside a phase, or reads
// shared memory it never wrote, shows up. Dev-only: built by the crate's tests
// with the system C++ compiler at `-ffp-contract=off`; never part of the
// engine.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "qwen_fast.cuh"
#include "qwen_fast_deltanet.cuh"
#include "qwen_ops.cuh"

namespace {

constexpr uint8_t kGarbage = 0xa5;

template <class S>
void garbage(S* s) {
  std::memset(static_cast<void*>(s), kGarbage, sizeof(S));
}

const ds::Par kPar{qwk::kElementThreads, true};
const ds::Par kParBlock{qwk::kBlockThreads, true};

}  // namespace

extern "C" {

void qwen_host_embed(const QwEmbedParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kElementThreads; t-- > 0;) qwk::embed_thread(*p, x, y, t);
}

// `eidola_qwen_norm`'s entry: a unit wider than `kNormWarpMax` a CTA over the
// grid's (units, rows); narrower ones a warp each, eight a CTA, numbered row
// by row over the grid's x.
// `eidola_qwen_norm_fused` (`fused`) or `eidola_qwen_norm`'s, over its grid.
static void host_norm(const QwNormParams* p, uint32_t gx, uint32_t gy, bool fused) {
  qwk::NormShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      if (p->n > qwk::kNormWarpMax) {
        if (fused) {
          qwk::norm_item<qwk::kElementThreads, true>(*p, y, x, kPar, sh.ln[0], sh.row, sh.gm,
                                                     sh.sc);
        } else {
          qwk::norm_item<qwk::kElementThreads, false>(*p, y, x, kPar, sh.ln[0], sh.row, sh.gm,
                                                      sh.sc);
        }
        continue;
      }
      for (uint32_t w = qwk::kNormItems; w-- > 0;) {
        const uint64_t item = static_cast<uint64_t>(x) * qwk::kNormItems + w;
        if (item >= static_cast<uint64_t>(p->rows) * p->units) continue;
        const uint32_t r = static_cast<uint32_t>(item / p->units);
        const uint32_t u = static_cast<uint32_t>(item % p->units);
        uint16_t* b = sh.row + w * qwk::kNormWarpMax;
        uint32_t* g = sh.gm + w * qwk::kNormWarpQuantWords;
        qwk::QuantScale* c = sh.sc + w * (qwk::kNormWarpMax / QW_BLOCK);
        if (fused) {
          qwk::norm_item<qwk::kWarp, true>(*p, r, u, qwk::WarpPar{}, sh.ln[w], b, g, c);
        } else {
          qwk::norm_item<qwk::kWarp, false>(*p, r, u, qwk::WarpPar{}, sh.ln[w], b, g, c);
        }
      }
    }
}

void qwen_host_norm(const QwNormParams* p, uint32_t gx, uint32_t gy) {
  host_norm(p, gx, gy, false);
}

void qwen_host_norm_fused(const QwNormParams* p, uint32_t gx, uint32_t gy) {
  host_norm(p, gx, gy, true);
}

// The norm's reference form over its own grid: a CTA a (unit, row).
void qwen_host_norm_reference(const QwNormParams* p, uint32_t gx, uint32_t gy) {
  qwk::NormRefShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwk::norm_reference_block(*p, y, x, kPar, sh);
    }
}

// `eidola_qwen_quant`'s entry: a warp per block and row (however the grid
// groups rows into CTAs), its lanes' amaxes combined by the butterfly the
// device's shuffles run.
void qwen_host_quant(const QwQuantParams* p, uint32_t gx, uint32_t gy) {
  constexpr uint32_t kW = qwk::kWarp;
  qwk::QuantLane v[kW];
  uint32_t m[kW];
  for (uint32_t y = gy * qwk::kQuantRows; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t w = qwk::kQuantWarps; w-- > 0;) {
        uint32_t b;
        if (y >= p->rows || !qwk::quant_block_of(*p, x, w, &b)) continue;
        for (uint32_t l = kW; l-- > 0;) {
          v[l] = qwk::quant_lane_load(*p, y, b, l);
          m[l] = qwk::quant_lane_amax(v[l]);
        }
        for (uint32_t off = kW / 2; off > 0; off >>= 1) {
          uint32_t next[kW];
          for (uint32_t l = kW; l-- > 0;) next[l] = m[l] > m[l ^ off] ? m[l] : m[l ^ off];
          std::memcpy(m, next, sizeof m);
        }
        for (uint32_t l = kW; l-- > 0;) qwk::quant_lane_store(*p, y, b, l, v[l], m[l]);
      }
}

// `qwk::e4m3_code` against `ds::e4m3_encode` over every f32 bit pattern from
// `first` to `last` inclusive: the number that differ, and the first of them
// in `*at`.
uint64_t qwen_host_e4m3_check(uint32_t first, uint32_t last, uint32_t* at) {
  uint64_t bad = 0;
  for (uint64_t b = first; b <= last; ++b) {
    float x;
    const uint32_t bits = static_cast<uint32_t>(b);
    std::memcpy(&x, &bits, sizeof x);
    if (qwk::e4m3_code(x) != ds::e4m3_encode(x)) {
      if (bad++ == 0) *at = bits;
    }
  }
  return bad;
}

// `eidola_qwen_fast_gemm_bf16`'s entry (the fast profile's small-N GEMM): a
// warp per output, its lanes' sums combined by the butterfly the device's
// shuffles run.
void qwen_host_fast_gemm(const QwLinearParams* p, uint32_t gx, uint32_t gy) {
  constexpr uint32_t kW = qwf::kWarp;
  float acc[kW][qwf::kSmallRows];
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t w = qwf::kSmallWarps; w-- > 0;) {
        uint32_t n, r0, rows;
        if (!qwf::small_gemm_unit(*p, x, y, w, &n, &r0, &rows)) continue;
        for (uint32_t l = kW; l-- > 0;) qwf::small_gemm_lane(*p, n, r0, rows, l, acc[l]);
        for (uint32_t off = kW / 2; off > 0; off >>= 1) {
          float next[kW][qwf::kSmallRows];
          for (uint32_t l = kW; l-- > 0;)
            for (uint32_t r = 0; r < qwf::kSmallRows; ++r) next[l][r] = acc[l][r] + acc[l ^ off][r];
          std::memcpy(acc, next, sizeof acc);
        }
        qwf::small_gemm_store(*p, n, r0, rows, acc[0]);
      }
}

// `eidola_qwen_fast_gemm_bf16_tiled`'s entry (the small-N GEMM's tiled form):
// a warp per tile, each of its sums combined by the same butterfly.
void qwen_host_fast_gemm_tiled(const QwLinearParams* p, uint32_t gx, uint32_t gy) {
  constexpr uint32_t kW = qwf::kWarp;
  constexpr uint32_t kT = qwf::kTileRows * qwf::kTileOuts;
  std::vector<float> acc(kW * kT);
  std::vector<float> next(kW * kT);
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t w = qwf::kTileWarps; w-- > 0;) {
        uint32_t n0, outs, r0, rows;
        if (!qwf::tile_gemm_unit(*p, x, y, w, &n0, &outs, &r0, &rows)) continue;
        for (uint32_t l = kW; l-- > 0;)
          qwf::tile_gemm_lane(*p, n0, outs, r0, rows, l, &acc[l * kT]);
        for (uint32_t off = kW / 2; off > 0; off >>= 1) {
          for (uint32_t l = kW; l-- > 0;)
            for (uint32_t i = 0; i < kT; ++i)
              next[l * kT + i] = acc[l * kT + i] + acc[(l ^ off) * kT + i];
          acc.swap(next);
        }
        qwf::tile_gemm_store(*p, n0, outs, r0, rows, &acc[0]);
      }
}

// The reference form (`quant_thread`) over its own grid: a thread per block,
// `kBlockThreads` blocks of a row per CTA. The warp form is held to it.
void qwen_host_quant_thread(const QwQuantParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kBlockThreads; t-- > 0;) qwk::quant_thread(*p, x, y, t);
}

void qwen_host_lin8(const QwLin8Params* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kBlockThreads; t-- > 0;) qwk::lin8_thread(*p, x, y, t);
}

void qwen_host_linear(const QwLinearParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kBlockThreads; t-- > 0;) qwk::linear_thread(*p, x, y, t);
}

void qwen_host_gate(const QwGateParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kElementThreads; t-- > 0;) qwk::gate_thread(*p, x, y, t);
}

void qwen_host_kv_write(const QwKvWriteParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kElementThreads; t-- > 0;) qwk::kv_write_thread(*p, x, y, t);
}

void qwen_host_attn(const QwAttnParams* p, uint32_t gx, uint32_t gy) {
  qwk::AttnShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwk::attn_block(*p, y, x, kPar, sh);
    }
}

void qwen_host_dn_prep(const QwDnPrepParams* p, uint32_t gx, uint32_t gy) {
  qwk::DnPrepShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwk::dn_prep_block(*p, y, x, kParBlock, sh);
    }
}

// `eidola_qwen_dn_rec`'s entry: every thread's share of its column kept across
// the tokens, the block's staged chunks between barriers, and a column's four
// threads exchanging their partial sums as the device's shuffles do.
void qwen_host_dn_rec(const QwDnRecParams* p, uint32_t gx, uint32_t gy) {
  constexpr uint32_t kT = qwk::kDnThreads;
  constexpr uint32_t kG = qwk::kDnGroup;
  qwk::DnRecShared sh;
  std::vector<float> st(static_cast<size_t>(kT) * qwk::kDnPer);
  std::vector<qwk::DnFetch> f(kT);
  float part[kT];
  float x[qwk::kDnPer];
  // Steps 2 and 1 of every column's tree, then the +0 tail.
  auto group_sum = [&]() {
    for (uint32_t w = 2; w > 0; w >>= 1) {
      float next[kT];
      for (uint32_t t = kT; t-- > 0;)
        next[t] = qwk::dn_tree_step(part[t], part[t ^ w], t % kG, w);
      std::memcpy(part, next, sizeof part);
    }
    for (uint32_t t = kT; t-- > 0;) part[t] = part[t] + 0.0f;
  };
  const uint32_t slices = qwk::dn_rec_slices(p->v_dim);
  for (uint32_t h = gy; h-- > 0;)
    for (uint32_t bx = gx; bx-- > 0;) {
      garbage(&sh);
      const uint32_t slice = bx % slices;
      const QwDnSeq s = p->seqs[bx / slices];
      auto col = [&](uint32_t t) { return slice * qwk::kDnCols + t / kG; };
      for (uint32_t t = kT; t-- > 0;) {
        qwk::dn_rec_fetch(*p, s, h, slice, 0, t, f[t]);
        qwk::dn_rec_load(*p, s, h, col(t), t % kG, &st[t * qwk::kDnPer]);
        qwk::dn_rec_stores(*p, s, 0, h, col(t), t % kG, &st[t * qwk::kDnPer]);
      }
      for (uint32_t t = kT; t-- > 0;) qwk::dn_rec_put(f[t], t, sh, 0);
      uint32_t b = 0;
      for (uint32_t base = 0; base < s.count; base += qwk::kDnChunk, b ^= 1) {
        const bool more = base + qwk::kDnChunk < s.count;
        if (more)
          for (uint32_t t = kT; t-- > 0;)
            qwk::dn_rec_fetch(*p, s, h, slice, base + qwk::kDnChunk, t, f[t]);
        const uint32_t n = s.count - base < qwk::kDnChunk ? s.count - base : qwk::kDnChunk;
        for (uint32_t u = 0; u < n; ++u) {
          const uint32_t i = base + u;
          for (uint32_t t = kT; t-- > 0;) {
            qwk::dn_row(sh.k[b][u][t % kG], x);
            part[t] = qwk::dn_rec_decay(&st[t * qwk::kDnPer], x, sh.alpha[b][u]);
          }
          group_sum();
          for (uint32_t t = kT; t-- > 0;) {
            qwk::dn_row(sh.k[b][u][t % kG], x);
            qwk::dn_rec_update(&st[t * qwk::kDnPer], x, sh.v[b][u][t / kG], sh.beta[b][u],
                               part[t]);
          }
          if (i >= s.out_from) {
            for (uint32_t t = kT; t-- > 0;) {
              qwk::dn_row(sh.q[b][u][t % kG], x);
              part[t] = qwk::dn_partial(&st[t * qwk::kDnPer], x);
            }
            group_sum();
            for (uint32_t t = kT; t-- > 0;)
              if (t % kG == 0 && col(t) < p->v_dim)
                qwk::dn_rec_out(*p, s.first + i, h, col(t), part[t]);
          }
          for (uint32_t t = kT; t-- > 0;)
            qwk::dn_rec_stores(*p, s, i + 1, h, col(t), t % kG, &st[t * qwk::kDnPer]);
        }
        if (more) {
          garbage(&sh.k[b ^ 1]);
          garbage(&sh.q[b ^ 1]);
          garbage(&sh.v[b ^ 1]);
          garbage(&sh.alpha[b ^ 1]);
          garbage(&sh.beta[b ^ 1]);
          for (uint32_t t = kT; t-- > 0;) qwk::dn_rec_put(f[t], t, sh, b ^ 1);
        }
      }
    }
}

// The fast profile's blocked DeltaNet: the intra and keep kernels' bodies, and the
// state kernel's host form (`qwd::dn_state_host`).
void qwen_host_dn_intra(const QwDnChunkParams* p, uint32_t gx, uint32_t gy) {
  qwd::IntraShared sh;
  const ds::Par par{qwd::kThreads, true};
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwd::dn_intra_block(*p, x, y, par, sh);
    }
}

void qwen_host_dn_state(const QwDnChunkParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) qwd::dn_state_host(*p, x, y);
}

// The fused form's intra body and keeping, CTA by CTA of its grid (`gx` runs by
// slices, `gy` value heads), after the intra kernel's body filled the scratch: each
// block's shared memory held to the scratch bit for bit where the state step reads it
// (a mismatch aborts, naming it), then its kept positions into the tail cache. The
// state kernel's host form runs after, as for the unfused form.
void qwen_host_dn_fused(const QwDnChunkParams* p, uint32_t gx, uint32_t gy) {
  static thread_local qwd::FusedShared sh;
  const ds::Par par{qwd::kThreads, true};
  const uint32_t slices = (p->v_dim + qwd::kSlice - 1) / qwd::kSlice;
  auto bits = [](float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
  };
  auto check = [&](bool ok, const char* what, uint32_t seq, uint32_t b, uint32_t h, uint32_t r,
                   uint32_t c) {
    if (ok) return;
    std::fprintf(stderr, "fused intra differs: %s, run %u block %u head %u row %u column %u\n",
                 what, seq, b, h, r, c);
    std::abort();
  };
  for (uint32_t h = gy; h-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      const uint32_t seq = x / slices;
      const uint32_t vbase = x % slices * qwd::kSlice;
      if (seq >= p->seq_count) continue;
      const QwDnSeq& s = p->seqs[seq];
      const uint32_t nb = qwd::run_tiles(s, p->chunk);
      if (s.tile + nb > p->tiles) continue;
      const uint32_t cols = p->v_dim - vbase < qwd::kSlice ? p->v_dim - vbase : qwd::kSlice;
      const uint32_t kh = h / (p->v_heads / p->k_heads);
      for (uint32_t b = 0; b < nb; ++b) {
        garbage(&sh);
        qwd::fused_intra(*p, s, b, h, vbase, par, sh);
        const uint32_t tile = s.tile + b;
        const uint32_t from = b * p->chunk;
        const uint32_t n = s.count - from < p->chunk ? s.count - from : p->chunk;
        const uint32_t lo = qwd::cached_rows(s, from, n);
        const uint64_t th = qwd::tile_head(*p, tile, h);
        const uint64_t tk = qwd::tile_key(*p, tile, kh);
        for (uint32_t r = 0; r < qwd::kTile; ++r) {
          const bool fresh = r >= lo && r < n;
          for (uint32_t i = 0; i < qwd::kKey; ++i) {
            const uint64_t at = (tk * qwd::kTile + r) * qwd::kKey + i;
            if (fresh) {
              check(bits(sh.w[r][i]) == bits(p->w[(th * qwd::kTile + r) * qwd::kKey + i]), "W",
                    seq, b, h, r, i);
              check(bits(qwd::tf32_rna(sh.in.q[r][i])) == bits(p->qr[at]), "queries", seq, b, h,
                    r, i);
            } else {
              check(bits(sh.w[r][i]) == 0 && bits(qwd::tf32_rna(sh.in.q[r][i])) == 0,
                    "a row read as zeros", seq, b, h, r, i);
            }
            if (n == p->chunk)
              check(bits(qwd::tf32_rna(sh.in.k[r][i])) == bits(p->kr[at]), "keys", seq, b, h, r, i);
          }
          for (uint32_t c = 0; c < cols; ++c) {
            const float want = fresh ? p->u[(th * qwd::kTile + r) * p->v_dim + vbase + c] : 0.0f;
            check(bits(sh.u[r][c]) == bits(want), "U", seq, b, h, r, c);
            if (r < lo)
              check(bits(sh.cd[r][c]) ==
                        bits(qwd::rec_at(*p, qwd::entry_at(*p, s, from + r), h)[p->v_dim + vbase + c]),
                    "cached delta", seq, b, h, r, c);
          }
          for (uint32_t j = 0; j < qwd::kTile; ++j)
            check(bits(sh.pm[r][j]) == bits(p->pm[(th * qwd::kTile + r) * qwd::kTile + j]), "P", seq,
                  b, h, r, j);
          const float* rec = p->rec + (th * qwd::kTile + r) * qwd::kRec;
          for (uint32_t j = 0; j < qwd::kTile; ++j)
            check(bits(sh.in.t[0][r][j]) == bits(rec[j]), "T", seq, b, h, r, j);
          check(bits(sh.in.beta[0][r]) == bits(rec[qwd::kTile]), "write strength", seq, b, h, r, 0);
          check(bits(sh.in.gc[0][r]) == bits(rec[qwd::kTile + 1]), "decay", seq, b, h, r, 0);
        }
        for (uint32_t f = 0; f <= qwd::kFacGamma; ++f)
          check(bits(sh.fac[f]) == bits(p->fac[th * qwd::kFac + f]), "factors", seq, b, h, f, 0);
        qwd::fused_keep(*p, s, b, h, vbase, par, sh);
      }
    }
}

void qwen_host_dn_keep(const QwDnChunkParams* p, uint32_t gx, uint32_t gy) {
  qwd::KeepShared sh;
  const ds::Par par{qwd::kThreads, true};
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwd::dn_keep_block(*p, x, y, par, sh);
    }
}

// `eidola_qwen_dn_norm`'s entry: a warp a (row, head), four a CTA, numbered
// row by row over the grid's x.
void qwen_host_dn_norm(const QwDnNormParams* p, uint32_t gx, uint32_t) {
  qwk::DnNormShared sh[qwk::kDnNormItems];
  for (uint32_t x = gx; x-- > 0;) {
    garbage(&sh);
    for (uint32_t w = qwk::kDnNormItems; w-- > 0;) {
      const uint64_t item = static_cast<uint64_t>(x) * qwk::kDnNormItems + w;
      if (item >= static_cast<uint64_t>(p->rows) * p->v_heads) continue;
      qwk::dn_norm_item(*p, static_cast<uint32_t>(item / p->v_heads),
                        static_cast<uint32_t>(item % p->v_heads), qwk::WarpPar{}, sh[w]);
    }
  }
}

// The gated norm's reference form over its own grid: a CTA a (head, row).
void qwen_host_dn_norm_reference(const QwDnNormParams* p, uint32_t gx, uint32_t gy) {
  qwk::DnNormRefShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwk::dn_norm_reference_block(*p, y, x, kParBlock, sh);
    }
}

void qwen_host_widen(const QwWidenParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kElementThreads; t-- > 0;) qwk::widen8_thread(*p, x, y, t);
}

// The widening's reference form over its own grid: an element a thread.
void qwen_host_widen_reference(const QwWidenParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kElementThreads; t-- > 0;) qwk::widen_thread(*p, x, y, t);
}

}  // extern "C"

extern "C" void qwen_host_round(const QwRoundParams* p, uint32_t gx, uint32_t gy) {
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t t = qwk::kElementThreads; t-- > 0;) qwk::round_thread(*p, x, y, t);
}
