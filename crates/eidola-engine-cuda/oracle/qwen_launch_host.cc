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

void qwen_host_quant(const QwQuantParams* p, uint32_t gx, uint32_t gy) {
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

// `eidola_qwen_dn_rec`'s entry: every thread's column array kept across the
// tokens, the block's staged keys and queries between barriers.
void qwen_host_dn_rec(const QwDnRecParams* p, uint32_t gx, uint32_t gy) {
  std::vector<float> st(static_cast<size_t>(qwk::kBlockThreads) * qwk::kKeyDim);
  float k[qwk::kKeyDim];
  float q[qwk::kKeyDim];
  for (uint32_t h = gy; h-- > 0;)
    for (uint32_t b = gx; b-- > 0;) {
      const QwDnSeq s = p->seqs[b];
      for (uint32_t j = qwk::kBlockThreads; j-- > 0;) {
        if (j >= p->v_dim) continue;
        qwk::dn_rec_load(*p, s, h, j, &st[j * qwk::kKeyDim]);
        qwk::dn_rec_stores(*p, s, 0, h, j, &st[j * qwk::kKeyDim]);
      }
      for (uint32_t i = 0; i < s.count; ++i) {
        std::memset(k, kGarbage, sizeof k);
        std::memset(q, kGarbage, sizeof q);
        for (uint32_t t = qwk::kBlockThreads; t-- > 0;)
          qwk::dn_rec_stage(*p, s.first + i, h, t, qwk::kBlockThreads, k, q);
        for (uint32_t j = qwk::kBlockThreads; j-- > 0;) {
          if (j >= p->v_dim) continue;
          qwk::dn_rec_token(*p, s, h, j, i, k, q, &st[j * qwk::kKeyDim]);
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
