// The Qwen3.8 kernels' bodies: one source for the device and for the host
// check (`eidola-engine-cuda`'s `oracle/qwen_launch_host.cc`). Every one is
// bit-identical to a function of `eidola-engine-model`'s `src/qwen/`, named
// beside it, written from `ds_numeric.cuh` (whose functions are the Qwen
// numeric module's: that module is a subset of DeepSeek's, function for
// function) and IEEE `+ - * /` in the reference's order. The `exact` profile
// fuses and reorders nothing.
//
// Two shapes of body:
//
// - A thread computes a whole unit (an element, a 128-wide quantisation block,
//   an output for up to 8 rows): `<kernel>_thread`, the entire kernel.
// - A block cooperates through phases (`ds::Par`, `ds_body.cuh`): norms,
//   attention, the DeltaNet's per-position preparation and gated norm. State
//   that crosses a phase lives in the block's shared struct.
//
// The DeltaNet recurrence keeps each thread's share of a state column in its
// own array across tokens; its device and host drivers call the same staging
// and per-token steps and differ only in how a column's four threads exchange
// partial sums (shuffles on the device).
//
// Why a row's result is its own: a thread or block reads its own row's
// arrays (and the weights) and writes its own row's outputs; no value of the
// launch's shape enters any arithmetic.
#pragma once

#include "ds_body.cuh"
#include "qwen_abi.cuh"

namespace qwk {

constexpr uint32_t kElementThreads = 256;
constexpr uint32_t kBlockThreads = 128;
// Rows an output thread of the linears computes, reading each weight block once.
constexpr uint32_t kLinRows = 8;
// The DeltaNet's key dimension, a template constant so a thread's state
// column lives in registers.
constexpr uint32_t kKeyDim = 128;
// The L2 norms' epsilon, inside the root (`deltanet::L2_EPS`).
constexpr float kL2Eps = 1e-6f;

DS_FN float bf16_in(const uint16_t* p, uint64_t i) { return ds::bf16_value(p[i]); }

// `tensor::dot` of two BF16 vectors: 16 interleaved partial sums from +0 over
// the full 16-wide chunks, the halving tree, then the tail added last.
DS_FN float pinned_dot_bf16(const uint16_t* a, const uint16_t* b, uint32_t n) {
  float acc[16];
  for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
  const uint32_t full = n & ~15u;
  for (uint32_t i = 0; i < full; i += 16) {
    for (uint32_t j = 0; j < 16; ++j) acc[j] = acc[j] + bf16_in(a, i + j) * bf16_in(b, i + j);
  }
  float tail = 0.0f;
  for (uint32_t i = full; i < n; ++i) tail = tail + bf16_in(a, i) * bf16_in(b, i);
  return ds::pinned_reduce(acc, tail);
}

// `pinned_sum` of the squares of `n` f32 values (each square rounded, as the
// reference's `x * x` or `v * v` before the sum).
DS_FN float pinned_sum_sq(const float* x, uint32_t n) {
  float acc[16];
  for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
  const uint32_t full = n & ~15u;
  for (uint32_t i = 0; i < full; i += 16) {
    for (uint32_t j = 0; j < 16; ++j) acc[j] = acc[j] + x[i + j] * x[i + j];
  }
  float tail = 0.0f;
  for (uint32_t i = full; i < n; ++i) tail = tail + x[i] * x[i];
  return ds::pinned_reduce(acc, tail);
}

// ---- embedding (`QwenModel::embed`) ----

DS_FN void embed_thread(const QwEmbedParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i = bx * kElementThreads + t;
  if (row >= p.rows || i >= p.width) return;
  const uint32_t token = p.tokens[row];
  uint16_t* dst = p.out + static_cast<uint64_t>(row) * p.out_stride;
  if (token >= p.vocab) {
    if (i == 0) ds::status_or(p.status, QW_STATUS_BAD_TOKEN);
    dst[i] = 0;
    return;
  }
  dst[i] = p.table[static_cast<uint64_t>(token) * p.width + i];
}

// ---- the zero-centred RMS norm (`forward::rms_norm_zc`, `residual_add`,
// `attention::apply_rope`) ----

constexpr uint32_t kRopeMax = 512;  // widest unit a rotated norm stages

struct NormShared {
  float lanes[16];
  float tail;
  float r;
  float staged[kRopeMax];
};

template <class P>
DS_FN void norm_block(const QwNormParams& p, uint32_t row, uint32_t unit, const P& par,
                      NormShared& sh) {
  uint16_t* x = p.x + static_cast<uint64_t>(row) * p.x_stride +
                static_cast<uint64_t>(unit) * p.x_unit_stride;
  uint16_t* out = p.out + static_cast<uint64_t>(row) * p.out_stride +
                  static_cast<uint64_t>(unit) * p.out_unit_stride;
  const uint32_t threads = kElementThreads;
  if (p.add != nullptr) {
    const uint16_t* add = p.add + static_cast<uint64_t>(row) * p.add_stride;
    par.run([&](uint32_t t) {
      for (uint32_t i = t; i < p.n; i += threads) {
        x[i] = ds::bf16_store(ds::bf16_round(ds::bf16_value(x[i]) + ds::bf16_value(add[i])));
      }
    });
  }
  const uint32_t full = p.n & ~15u;
  par.run([&](uint32_t t) {
    if (t < 16) {
      float acc = 0.0f;
      for (uint32_t i = t; i < full; i += 16) {
        const float v = ds::bf16_value(x[i]);
        acc = acc + v * v;
      }
      sh.lanes[t] = acc;
    } else if (t == 16) {
      float tail = 0.0f;
      for (uint32_t i = full; i < p.n; ++i) {
        const float v = ds::bf16_value(x[i]);
        tail = tail + v * v;
      }
      sh.tail = tail;
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = sh.lanes[j];
      sh.r = ds::rms_factor(ds::pinned_reduce(acc, sh.tail), p.n, p.eps);
    }
  });
  if (p.rotary == 0) {
    par.run([&](uint32_t t) {
      for (uint32_t i = t; i < p.n; i += threads) {
        out[i] = ds::bf16_store(ds::rms_element(ds::bf16_value(x[i]), sh.r, p.weight[i]));
      }
    });
    return;
  }
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < p.n; i += threads) {
      sh.staged[i] = ds::rms_element(ds::bf16_value(x[i]), sh.r, p.weight[i]);
    }
  });
  const uint64_t at = static_cast<uint64_t>(p.positions[row]) * p.rotary;
  const uint32_t half = p.rotary / 2;
  par.run([&](uint32_t t) {
    for (uint32_t j = t; j < p.n; j += threads) {
      float v = sh.staged[j];
      if (j < p.rotary) {
        const float rot = j < half ? -sh.staged[j + half] : sh.staged[j - half];
        v = ds::bf16_round(ds::bf16_round(v * p.rope_cos[at + j]) +
                           ds::bf16_round(rot * p.rope_sin[at + j]));
      }
      out[j] = ds::bf16_store(v);
    }
  });
}

// ---- the FP8 activation quantiser (`linear::act_quant`) ----

// One 128-wide block of one row.
DS_FN void quant_thread(const QwQuantParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t b = bx * kBlockThreads + t;
  if (row >= p.rows || b >= p.k / QW_BLOCK) return;
  const uint16_t* x = p.x + static_cast<uint64_t>(row) * p.x_stride + b * QW_BLOCK;
  float amax = 0.0f;
  for (uint32_t j = 0; j < QW_BLOCK; ++j) amax = ds::amax_step(amax, ds::bf16_value(x[j]));
  const float s = amax / 448.0f;
  uint8_t* codes = p.codes + static_cast<uint64_t>(row) * p.k + b * QW_BLOCK;
  for (uint32_t j = 0; j < QW_BLOCK; ++j) codes[j] = ds::e4m3_encode(ds::bf16_value(x[j]) / s);
  p.scales[static_cast<uint64_t>(b) * p.m_pad + row] = s;
}

// ---- the FP8 linear (`linear::fp8_block_dot`, `fp8_row_dot`) ----

// The exact dot product of two 128-wide blocks of E4M3 codes, rounded once to
// f32: in units of 2^-18 every partial sum is an integer below 2^43.
DS_FN float block_dot128(const uint8_t* a, const uint8_t* w) {
  int64_t sum = 0;
  for (uint32_t j = 0; j < QW_BLOCK; ++j) {
    sum += static_cast<int64_t>(ds::e4m3_times_512(a[j])) *
           static_cast<int64_t>(ds::e4m3_times_512(w[j]));
  }
  return static_cast<float>(static_cast<double>(sum) * 3.814697265625e-06 /* 2^-18 */);
}

// Output `bx · 128 + t` for rows `by · 8 ..`.
DS_FN void lin8_thread(const QwLin8Params& p, uint32_t bx, uint32_t by, uint32_t t) {
  const uint32_t n = bx * kBlockThreads + t;
  if (n >= p.n) return;
  const uint32_t r0 = by * kLinRows;
  const uint32_t rows = ds::min_u32(kLinRows, p.rows > r0 ? p.rows - r0 : 0);
  const uint32_t kb = p.k / QW_BLOCK;
  const uint32_t nb = p.n / QW_BLOCK;
  float acc[kLinRows];
  for (uint32_t r = 0; r < kLinRows; ++r) acc[r] = 0.0f;
  const uint8_t* w = p.w + static_cast<uint64_t>(n) * p.k;
  for (uint32_t b = 0; b < kb; ++b) {
    const float sw = p.w_scales[static_cast<uint64_t>(b) * nb + n / QW_BLOCK];
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t row = r0 + r;
      const float dot = block_dot128(p.a + row * p.k + b * QW_BLOCK, w + b * QW_BLOCK);
      acc[r] = ds::gemm_accumulate(acc[r], dot, p.a_scales[static_cast<uint64_t>(b) * p.m_pad + row],
                                   sw);
    }
  }
  for (uint32_t r = 0; r < rows; ++r) {
    p.out[static_cast<uint64_t>(r0 + r) * p.out_stride + n] = ds::bf16_store(ds::bf16_round(acc[r]));
  }
}

// ---- the BF16 linear (`linear::linear_bf16_rows`, `QwenModel::head`) ----

DS_FN void linear_thread(const QwLinearParams& p, uint32_t bx, uint32_t by, uint32_t t) {
  const uint32_t n = bx * kBlockThreads + t;
  if (n >= p.n) return;
  const uint32_t r0 = by * kLinRows;
  const uint16_t* w = p.w + static_cast<uint64_t>(n) * p.k;
  for (uint32_t r = 0; r < kLinRows && r0 + r < p.rows; ++r) {
    const uint64_t row = r0 + r;
    const float v = ds::bf16_round(pinned_dot_bf16(p.x + row * p.x_stride, w, p.k));
    if (p.out_f32 != 0) {
      static_cast<float*>(p.out)[row * p.out_stride + n] = v;
    } else {
      static_cast<uint16_t*>(p.out)[row * p.out_stride + n] = ds::bf16_store(v);
    }
  }
}

// ---- element-wise gates (`QwenModel::mlp`'s product, `attention_mix`'s gate) ----

DS_FN void gate_thread(const QwGateParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i = bx * kElementThreads + t;
  if (row >= p.rows || i >= p.width) return;
  const float a = ds::bf16_value(p.a[static_cast<uint64_t>(row) * p.a_stride + i]);
  const uint64_t gi = static_cast<uint64_t>(row) * p.g_stride +
                      static_cast<uint64_t>(i / p.unit) * p.g_unit_stride + i % p.unit;
  const float g = ds::bf16_value(p.g[gi]);
  float y;
  if (p.kind == QW_GATE_SWIGLU) {
    y = ds::bf16_round(ds::bf16_round(ds::silu32(g)) * a);
  } else {
    y = ds::bf16_round(a * ds::bf16_round(ds::sigmoid32(g)));
  }
  p.out[static_cast<uint64_t>(row) * p.out_stride + i] = ds::bf16_store(y);
}

// ---- paged KV (`KvCache`, as the executor pages it) ----

DS_FN uint64_t kv_block_base(uint32_t block, uint32_t layer, uint32_t layers, uint32_t bs,
                             uint32_t width) {
  return (static_cast<uint64_t>(block) * layers + layer) * 2u * bs * width;
}

DS_FN void kv_write_thread(const QwKvWriteParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i = bx * kElementThreads + t;
  if (row >= p.rows || i >= p.width) return;
  const uint32_t e = p.targets[row];
  if (e == QW_NO_TARGET) return;
  const uint64_t base =
      kv_block_base(e / p.block_size, p.layer, p.layers, p.block_size, p.width) +
      static_cast<uint64_t>(e % p.block_size) * p.width;
  p.pool[base + i] = p.k[static_cast<uint64_t>(row) * p.k_stride + i];
  p.pool[base + static_cast<uint64_t>(p.block_size) * p.width + i] =
      p.v[static_cast<uint64_t>(row) * p.v_stride + i];
}

// ---- the exact attention core (`attention::attend`) ----

struct AttnShared {
  float m;
  float denom;
};

// Key `x` (a position) of KV head `g` of row `row`: keys, or values with `values`.
DS_FN const uint16_t* attn_kv(const QwAttnParams& p, uint32_t row, uint32_t x, uint32_t g,
                              bool values) {
  const int32_t block = p.tables[static_cast<uint64_t>(row) * p.table_stride + x / p.block_size];
  const uint32_t width = p.kv_heads * p.head_dim;
  const uint64_t base =
      kv_block_base(static_cast<uint32_t>(block), p.layer, p.layers, p.block_size, width) +
      (values ? static_cast<uint64_t>(p.block_size) * width : 0) +
      static_cast<uint64_t>(x % p.block_size) * width + static_cast<uint64_t>(g) * p.head_dim;
  return p.pool + base;
}

template <class P>
DS_FN void attn_block(const QwAttnParams& p, uint32_t row, uint32_t head, const P& par,
                      AttnShared& sh) {
  const uint32_t g = head / (p.q_heads / p.kv_heads);
  const uint32_t visible = p.positions[row] + 1 - p.key_first;
  const uint16_t* q = p.q + static_cast<uint64_t>(row) * p.q_stride +
                      static_cast<uint64_t>(head) * p.q_head_stride;
  float* s = p.scores + (static_cast<uint64_t>(row) * p.q_heads + head) * p.score_stride;
  const uint32_t threads = kElementThreads;
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < visible; i += threads) {
      s[i] = pinned_dot_bf16(q, attn_kv(p, row, p.key_first + i, g, false), p.head_dim) * p.scale;
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float m = -ds::f32_from_bits(0x7f800000u);
      for (uint32_t i = 0; i < visible; ++i) m = ds::max32(m, s[i]);
      sh.m = m;
    }
  });
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < visible; i += threads) s[i] = ds::exp32(s[i] - sh.m);
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float d = 0.0f;
      for (uint32_t i = 0; i < visible; ++i) d = d + s[i];
      sh.denom = d;
    }
  });
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < visible; i += threads) s[i] = s[i] / sh.denom;
  });
  par.run([&](uint32_t t) {
    if (t >= p.head_dim) return;
    float o = 0.0f;
    for (uint32_t i = 0; i < visible; ++i) {
      o = o + s[i] * ds::bf16_value(attn_kv(p, row, p.key_first + i, g, true)[t]);
    }
    p.out[static_cast<uint64_t>(row) * p.out_stride + static_cast<uint64_t>(head) * p.head_dim +
          t] = ds::bf16_store(ds::bf16_round(o));
  });
}

// ---- DeltaNet: per position (`deltanet::conv_step`, `gates`, `l2_normalise`,
// `scale_query`) ----

struct DnPrepShared {
  float q[kKeyDim];
  float k[kKeyDim];
  float rq;
  float rk;
};

// The convolution of channel `ch` over a position's window, then SiLU:
// `bf16(silu(bf16(Σ w_j · x_j)))`, taps oldest first, the first product
// starting the sum.
DS_FN float dn_conv(const QwDnPrepParams& p, uint32_t pos, uint32_t ch) {
  const uint32_t cd = 2 * p.k_heads * p.k_dim + p.v_heads * p.v_dim;
  const float* rows = p.conv_in + static_cast<uint64_t>(p.window[pos]) * cd + ch;
  const float* w = p.conv_w + static_cast<uint64_t>(ch) * 4;
  float acc = w[0] * rows[0];
  for (uint32_t j = 1; j < 4; ++j) acc = acc + w[j] * rows[static_cast<uint64_t>(j) * cd];
  return ds::bf16_round(ds::silu32(ds::bf16_round(acc)));
}

// Block (position, key head): the key head's queries and keys, its value
// heads' values, `β` and `α`.
template <class P>
DS_FN void dn_prep_block(const QwDnPrepParams& p, uint32_t pos, uint32_t kh, const P& par,
                         DnPrepShared& sh) {
  const uint32_t kd = p.k_dim;
  const uint32_t key_dim = p.k_heads * kd;
  const uint32_t group = p.v_heads / p.k_heads;
  const uint32_t threads = kBlockThreads;
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < kd; i += threads) {
      sh.q[i] = dn_conv(p, pos, kh * kd + i);
      sh.k[i] = dn_conv(p, pos, key_dim + kh * kd + i);
    }
    for (uint32_t i = t; i < group * p.v_dim; i += threads) {
      const uint32_t c = kh * group * p.v_dim + i;
      p.v[static_cast<uint64_t>(pos) * p.v_heads * p.v_dim + c] = dn_conv(p, pos, 2 * key_dim + c);
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) sh.rq = ds::rsqrt32(pinned_sum_sq(sh.q, kd) + kL2Eps);
    if (t == 1) sh.rk = ds::rsqrt32(pinned_sum_sq(sh.k, kd) + kL2Eps);
  });
  par.run([&](uint32_t t) {
    const float d = static_cast<float>(sqrt(static_cast<double>(kd)));
    for (uint32_t i = t; i < kd; i += threads) {
      const uint64_t at = static_cast<uint64_t>(pos) * key_dim + kh * kd + i;
      p.q[at] = (sh.q[i] * sh.rq) / d;
      p.k[at] = sh.k[i] * sh.rk;
    }
    if (t < group) {
      const uint32_t h = kh * group + t;
      const float* ab = p.ab + static_cast<uint64_t>(pos) * 2 * p.v_heads;
      const float a = ab[h];
      const float b = ab[p.v_heads + h];
      const float g = -p.exp_a_log[h] * ds::softplus32(a + p.dt_bias[h]);
      p.beta[static_cast<uint64_t>(pos) * p.v_heads + h] = ds::bf16_round(ds::sigmoid32(b));
      p.alpha[static_cast<uint64_t>(pos) * p.v_heads + h] = ds::exp32(g);
    }
  });
}

// ---- DeltaNet: the recurrence (`deltanet::recurrent_step`) ----
//
// A block computes `kDnCols` value columns of one (sequence, value head), a
// group of `kDnGroup` adjacent threads per column. Thread `l` of a group holds
// the column's state values `i = l + 4n` (`n < kDnPer`): the reference's
// accumulator lanes `l`, `l + 4`, `l + 8` and `l + 12`, each a run of 8 terms.
// It adds its lanes from +0 in ascending `i` and takes them through the
// halving tree's steps 8 and 4 (`dn_partial`); steps 2 and 1 combine the
// group's threads (`dn_tree_step`, the operands in the reference's order on
// every thread, so the whole group holds the same sum), and the +0 tail is
// added last. Columns share nothing but the position's keys and queries, so a
// head's columns split across blocks freely.
//
// Keys, queries, values and gates are staged `kDnChunk` positions at a time,
// keys and queries rearranged so that thread `l`'s values are contiguous
// (`DnRecShared`); a thread fetches its share of the next chunk while the
// block runs this one (`DnFetch`).

constexpr uint32_t kDnThreads = 128;
constexpr uint32_t kDnGroup = 4;
constexpr uint32_t kDnCols = kDnThreads / kDnGroup;
constexpr uint32_t kDnPer = kKeyDim / kDnGroup;
constexpr uint32_t kDnChunk = 8;
// A staged key or query row of one group thread, padded so that the four
// rows a warp reads at once fall in distinct banks.
constexpr uint32_t kDnRow = kDnPer + 4;
// Four-value units of keys (or queries) per position, and per thread a chunk.
constexpr uint32_t kDnQuads = kKeyDim / 4;
constexpr uint32_t kDnQuadLoads = kDnChunk * kDnQuads / kDnThreads;
constexpr uint32_t kDnValueLoads = kDnChunk * kDnCols / kDnThreads;
static_assert(kDnChunk * kDnQuads % kDnThreads == 0, "whole quads per thread");
static_assert(kDnChunk * kDnCols % kDnThreads == 0, "whole values per thread");
static_assert(kDnChunk <= kDnThreads, "a gate per thread");

#if defined(__CUDACC__)
#define QW_ALIGN16 __align__(16)
#else
#define QW_ALIGN16 alignas(16)
#endif
// A loop over a thread's state values unrolled, so they stay in registers.
#if defined(__CUDA_ARCH__)
#define QW_UNROLL _Pragma("unroll")
#else
#define QW_UNROLL
#endif

struct QW_ALIGN16 DnRecShared {
  float k[2][kDnChunk][kDnGroup][kDnRow];
  float q[2][kDnChunk][kDnGroup][kDnRow];
  float v[2][kDnChunk][kDnCols];
  float alpha[2][kDnChunk];
  float beta[2][kDnChunk];
};

// One thread's share of a chunk, between its loads and its stores to shared
// memory.
struct DnFetch {
  float k[kDnQuadLoads][4];
  float q[kDnQuadLoads][4];
  float v[kDnValueLoads];
  float alpha;
  float beta;
};

// The blocks a (sequence, value head) takes.
DS_FN uint32_t dn_rec_slices(uint32_t v_dim) { return (v_dim + kDnCols - 1) / kDnCols; }

// Four consecutive f32 values, 16-byte aligned.
DS_FN void dn_load4(const float* src, float* dst) {
#if defined(__CUDACC__)
  const float4 x = *reinterpret_cast<const float4*>(src);
  dst[0] = x.x;
  dst[1] = x.y;
  dst[2] = x.z;
  dst[3] = x.w;
#else
  for (uint32_t e = 0; e < 4; ++e) dst[e] = src[e];
#endif
}

// A group thread's staged row (`kDnPer` values, 16-byte aligned).
DS_FN void dn_row(const float* src, float* dst) {
  QW_UNROLL
  for (uint32_t n = 0; n < kDnPer; n += 4) dn_load4(src + n, dst + n);
}

// State value `i` of column `j` of value head `h` in a layer's state.
DS_FN uint64_t dn_state_index(uint32_t h, uint32_t i, uint32_t j, uint32_t vd) {
  return (static_cast<uint64_t>(h) * kKeyDim + i) * vd + j;
}

DS_FN float* dn_state_at(uint64_t addr, uint64_t offset) {
  return reinterpret_cast<float*>(addr + offset);
}

// Thread `l`'s lanes of the dot of `st` and the row `x` (both its values
// `i = l + 4n`), each from +0 in ascending `i`, through the tree's steps 8 and
// 4: the tree's value at index `l`.
DS_FN float dn_partial(const float* st, const float* x) {
  float acc[4];
  QW_UNROLL
  for (uint32_t a = 0; a < 4; ++a) acc[a] = 0.0f;
  QW_UNROLL
  for (uint32_t m = 0; m < kDnPer / 4; ++m) {
    QW_UNROLL
    for (uint32_t a = 0; a < 4; ++a) acc[a] = acc[a] + st[4 * m + a] * x[4 * m + a];
  }
  return (acc[0] + acc[2]) + (acc[1] + acc[3]);
}

// Step `w` (2 or 1) of the tree on group thread `l`, holding index `l`'s value
// `own`, given index `l ^ w`'s `other`: index `min` plus index `max`, the
// reference's operands in its order on both threads.
DS_FN float dn_tree_step(float own, float other, uint32_t l, uint32_t w) {
  const bool upper = (l & w) != 0;
  return (upper ? other : own) + (upper ? own : other);
}

// Decay the state by `alpha`, then the partial of `r = Σ_i S_ij k_i`.
DS_FN float dn_rec_decay(float* st, const float* k, float alpha) {
  QW_UNROLL
  for (uint32_t n = 0; n < kDnPer; ++n) st[n] = st[n] * alpha;
  return dn_partial(st, k);
}

// `δ = (v_j − r)·β` and `S_ij ← S_ij + k_i δ`.
DS_FN void dn_rec_update(float* st, const float* k, float vj, float beta, float r) {
  const float delta = (vj - r) * beta;
  QW_UNROLL
  for (uint32_t n = 0; n < kDnPer; ++n) st[n] = st[n] + k[n] * delta;
}

// Group thread `l`'s values of column `j` from the sequence's source state
// (zeros for none, or a column past `v_dim`).
DS_FN void dn_rec_load(const QwDnRecParams& p, const QwDnSeq& s, uint32_t h, uint32_t j,
                       uint32_t l, float* st) {
  QW_UNROLL
  for (uint32_t n = 0; n < kDnPer; ++n) {
    st[n] = s.src == 0 || j >= p.v_dim
                ? 0.0f
                : dn_state_at(s.src, p.state_offset)[dn_state_index(h, l + 4 * n, j, p.v_dim)];
  }
}

// A store's address and stride, opaque to the compiler: stores are rare, and
// their 32 addresses hoisted out of the token loop would take 64 registers.
DS_FN uint64_t dn_opaque(uint64_t x) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+l"(x));
#endif
  return x;
}

DS_FN void dn_rec_store(const QwDnRecParams& p, uint64_t dst, uint32_t h, uint32_t j, uint32_t l,
                        const float* st) {
  float* at = reinterpret_cast<float*>(
      dn_opaque(dst + p.state_offset + dn_state_index(h, l, j, p.v_dim) * sizeof(float)));
  const uint64_t step = dn_opaque(static_cast<uint64_t>(kDnGroup) * p.v_dim);
  QW_UNROLL
  for (uint32_t n = 0; n < kDnPer; ++n) {
    *at = st[n];
    at += step;
  }
}

// After the step that leaves `done` positions run: the stores due then.
DS_FN void dn_rec_stores(const QwDnRecParams& p, const QwDnSeq& s, uint32_t done, uint32_t h,
                         uint32_t j, uint32_t l, const float* st) {
  if (j >= p.v_dim) return;
  if (s.live != 0 && s.live_after == done) dn_rec_store(p, s.live, h, j, l, st);
  for (uint32_t w = 0; w < s.writes && w < 4; ++w) {
    if (s.write_after[w] == done) dn_rec_store(p, s.write_dst[w], h, j, l, st);
  }
}

// Thread `t`'s share of the chunk of positions `base ..` (relative) of
// column block `slice`; zeros past the sequence's positions or `v_dim`.
DS_FN void dn_rec_fetch(const QwDnRecParams& p, const QwDnSeq& s, uint32_t h, uint32_t slice,
                        uint32_t base, uint32_t t, DnFetch& f) {
  const uint32_t kh = h / (p.v_heads / p.k_heads);
  for (uint32_t r = 0; r < kDnQuadLoads; ++r) {
    const uint32_t x = t + r * kDnThreads;
    const uint32_t u = x / kDnQuads;
    if (base + u < s.count) {
      const uint64_t at = (static_cast<uint64_t>(s.first + base + u) * p.k_heads + kh) * kKeyDim +
                          (x % kDnQuads) * 4;
      dn_load4(p.k + at, f.k[r]);
      dn_load4(p.q + at, f.q[r]);
    } else {
      for (uint32_t e = 0; e < 4; ++e) f.k[r][e] = f.q[r][e] = 0.0f;
    }
  }
  for (uint32_t r = 0; r < kDnValueLoads; ++r) {
    const uint32_t x = t + r * kDnThreads;
    const uint32_t u = x / kDnCols;
    const uint32_t j = slice * kDnCols + x % kDnCols;
    f.v[r] = base + u < s.count && j < p.v_dim
                 ? p.v[(static_cast<uint64_t>(s.first + base + u) * p.v_heads + h) * p.v_dim + j]
                 : 0.0f;
  }
  const bool gate = t < kDnChunk && base + t < s.count;
  const uint64_t vh = static_cast<uint64_t>(s.first + base + t) * p.v_heads + h;
  f.alpha = gate ? p.alpha[vh] : 0.0f;
  f.beta = gate ? p.beta[vh] : 0.0f;
}

// Thread `t`'s share into buffer `b`: key and query element `4c + e` of a
// position to group thread `e`'s row at `c`.
DS_FN void dn_rec_put(const DnFetch& f, uint32_t t, DnRecShared& sh, uint32_t b) {
  for (uint32_t r = 0; r < kDnQuadLoads; ++r) {
    const uint32_t x = t + r * kDnThreads;
    const uint32_t u = x / kDnQuads;
    const uint32_t c = x % kDnQuads;
    for (uint32_t e = 0; e < 4; ++e) {
      sh.k[b][u][e][c] = f.k[r][e];
      sh.q[b][u][e][c] = f.q[r][e];
    }
  }
  for (uint32_t r = 0; r < kDnValueLoads; ++r) {
    const uint32_t x = t + r * kDnThreads;
    sh.v[b][x / kDnCols][x % kDnCols] = f.v[r];
  }
  if (t < kDnChunk) {
    sh.alpha[b][t] = f.alpha;
    sh.beta[b][t] = f.beta;
  }
}

// The output of position `pos` for column `j`: `o` rounded to BF16.
DS_FN void dn_rec_out(const QwDnRecParams& p, uint32_t pos, uint32_t h, uint32_t j, float o) {
  p.out[(static_cast<uint64_t>(pos) * p.v_heads + h) * p.v_dim + j] = ds::bf16_round(o);
}

// ---- DeltaNet: the gated norm (`deltanet::gated_norm`) ----

struct DnNormShared {
  float lanes[16];
  float r;
};

template <class P>
DS_FN void dn_norm_block(const QwDnNormParams& p, uint32_t row, uint32_t h, const P& par,
                         DnNormShared& sh) {
  const uint32_t vd = p.v_dim;
  const float* o = p.o + (static_cast<uint64_t>(row) * p.v_heads + h) * vd;
  const uint32_t full = vd & ~15u;
  par.run([&](uint32_t t) {
    if (t < 16) {
      float acc = 0.0f;
      for (uint32_t i = t; i < full; i += 16) acc = acc + o[i] * o[i];
      sh.lanes[t] = acc;
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float tail = 0.0f;
      for (uint32_t i = full; i < vd; ++i) tail = tail + o[i] * o[i];
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = sh.lanes[j];
      sh.r = ds::rms_factor(ds::pinned_reduce(acc, tail), vd, p.eps);
    }
  });
  par.run([&](uint32_t t) {
    for (uint32_t j = t; j < vd; j += kBlockThreads) {
      const float n = ds::bf16_round(p.weight[j] * ds::bf16_round(o[j] * sh.r));
      const float z = ds::bf16_value(p.z[static_cast<uint64_t>(row) * p.z_stride + h * vd + j]);
      p.out[(static_cast<uint64_t>(row) * p.v_heads + h) * vd + j] =
          ds::bf16_store(ds::bf16_round(n * ds::silu32(z)));
    }
  });
}

// ---- BF16 → f32 ----

DS_FN void widen_thread(const QwWidenParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i = bx * kElementThreads + t;
  if (row >= p.rows || i >= p.width) return;
  p.out[static_cast<uint64_t>(row) * p.out_stride + i] =
      ds::bf16_value(p.x[static_cast<uint64_t>(row) * p.x_stride + i]);
}

// ---- f32 → BF16 ----

DS_FN void round_thread(const QwRoundParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i = bx * kElementThreads + t;
  if (row >= p.rows || i >= p.width) return;
  const float v = ds::bf16_round(p.x[static_cast<uint64_t>(row) * p.x_stride + i]);
  const uint64_t o = static_cast<uint64_t>(row) * p.out_stride + i;
  if (p.out_f32 != 0) {
    static_cast<float*>(p.out)[o] = v;
  } else {
    static_cast<uint16_t*>(p.out)[o] = ds::bf16_store(v);
  }
}

}  // namespace qwk
