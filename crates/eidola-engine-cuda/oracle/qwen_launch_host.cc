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
void qwen_host_norm(const QwNormParams* p, uint32_t gx, uint32_t gy) {
  qwk::NormShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      if (p->n > qwk::kNormWarpMax) {
        qwk::norm_item<qwk::kElementThreads>(*p, y, x, kPar, sh.ln[0], sh.row);
        continue;
      }
      for (uint32_t w = qwk::kNormItems; w-- > 0;) {
        const uint64_t item = static_cast<uint64_t>(x) * qwk::kNormItems + w;
        if (item >= static_cast<uint64_t>(p->rows) * p->units) continue;
        qwk::norm_item<qwk::kWarp>(*p, static_cast<uint32_t>(item / p->units),
                                   static_cast<uint32_t>(item % p->units), qwk::WarpPar{},
                                   sh.ln[w], sh.row + w * qwk::kNormWarpMax);
      }
    }
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

// The fast profile's blocked DeltaNet: the intra kernel's body, and the state
// kernel's host form (`qwd::dn_state_host`).
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
