// The Qwen3.8 kernels' bodies (`eidola-engine-kernels`' `csrc/qwen_ops.cuh`,
// the C++ a GPU compiles) as a host library: one function per launch, each
// repeating its kernel's entry over the grid the Rust wrapper computed. Blocks
// and threads run in descending order and shared memory starts as garbage, so
// a body that reads what another thread writes inside a phase, or reads
// shared memory it never wrote, shows up. Dev-only: built by the crate's tests
// with the system C++ compiler at `-ffp-contract=off`; never part of the
// engine.

#include <cstdint>
#include <cstring>
#include <vector>

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

void qwen_host_norm(const QwNormParams* p, uint32_t gx, uint32_t gy) {
  qwk::NormShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwk::norm_block(*p, y, x, kPar, sh);
    }
}

// `eidola_qwen_quant`'s entry: a warp per block, its lanes' amaxes combined by
// the butterfly the device's shuffles run, in the same operand order.
void qwen_host_quant(const QwQuantParams* p, uint32_t gx, uint32_t gy) {
  constexpr uint32_t kW = qwk::kWarp;
  float v[kW][qwk::kQuantLane];
  float m[kW];
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;)
      for (uint32_t w = qwk::kQuantWarps; w-- > 0;) {
        uint32_t b;
        if (!qwk::quant_block_of(*p, x, y, w, &b)) continue;
        for (uint32_t l = kW; l-- > 0;) m[l] = qwk::quant_lane_load(*p, y, b, l, v[l]);
        for (uint32_t off = kW / 2; off > 0; off >>= 1) {
          float next[kW];
          for (uint32_t l = kW; l-- > 0;) next[l] = ds::max32(m[l], m[l ^ off]);
          std::memcpy(m, next, sizeof m);
        }
        for (uint32_t l = kW; l-- > 0;) qwk::quant_lane_store(*p, y, b, l, v[l], m[l]);
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

void qwen_host_dn_norm(const QwDnNormParams* p, uint32_t gx, uint32_t gy) {
  qwk::DnNormShared sh;
  for (uint32_t y = gy; y-- > 0;)
    for (uint32_t x = gx; x-- > 0;) {
      garbage(&sh);
      qwk::dn_norm_block(*p, y, x, kParBlock, sh);
    }
}

void qwen_host_widen(const QwWidenParams* p, uint32_t gx, uint32_t gy) {
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
