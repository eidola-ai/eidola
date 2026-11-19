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
// - A thread computes a whole unit (an element, eight elements, a 128-wide
//   quantisation block, an output for up to 8 rows): `<kernel>_thread`, the
//   entire kernel.
// - A block or a warp cooperates through phases (`ds::Par`, `ds_body.cuh`, and
//   `WarpPar` below): norms, attention, the DeltaNet's per-position
//   preparation and gated norm. State that crosses a phase lives in the
//   block's (or the warp's) shared struct.
//
// A kernel whose body is arranged for speed has a plain form beside it, its
// reference form (`*_reference_block`, `widen_thread`, `quant_thread`: a
// thread or a block a unit, in the reference's order of operations), which the
// host check and the device tests hold the kernel to bit for bit.
//
// The DeltaNet recurrence keeps each thread's share of a state column in its
// own array across tokens; its device and host drivers call the same staging
// and per-token steps and differ only in how a column's four threads exchange
// partial sums (shuffles on the device). The quantiser is a warp per block the
// same way: per-lane load and store steps, the lanes' amaxes combined by
// shuffles on the device and by the same butterfly on the host.
//
// Why a row's result is its own: a thread or block reads its own row's
// arrays (and the weights) and writes its own row's outputs; no value of the
// launch's shape enters any arithmetic.
#pragma once

#include "ds_body.cuh"
#include "qwen_abi.cuh"

// A loop over a thread's own values unrolled, so they stay in registers.
#if defined(__CUDA_ARCH__)
#define QW_UNROLL _Pragma("unroll")
#else
#define QW_UNROLL
#endif

#if defined(__CUDACC__)
#define QW_ALIGN16 __align__(16)
#else
#define QW_ALIGN16 alignas(16)
#endif

namespace qwk {

constexpr uint32_t kElementThreads = 256;
constexpr uint32_t kBlockThreads = 128;
constexpr uint32_t kWarp = 32;

// A body's phases over one warp: on the device each lane runs its share and
// meets a warp barrier; on a host the lanes run in descending order. A warp
// that runs such a body owns its item alone (`ds::Par`'s rules, `ds_body.cuh`,
// at the scale of a warp).
#if defined(__CUDACC__)
struct WarpPar {
  template <class F>
  DS_DEV void run(F phase) const {
    phase(threadIdx.x % kWarp);
    __syncwarp();
  }
};
#else
struct WarpPar {
  template <class F>
  void run(F phase) const {
    for (uint32_t l = kWarp; l-- > 0;) phase(l);
  }
};
#endif

// Eight BF16 values a 16-byte access moves: element `e` in bits `16 (e % 2)`
// of word `e / 2`.
struct Bf16x8 {
  uint32_t w[4];
};
// Eight f32 values, two 16-byte accesses.
struct F32x8 {
  float f[8];
};

DS_FN Bf16x8 load8(const uint16_t* p) {
#if defined(__CUDACC__)
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  return Bf16x8{{u.x, u.y, u.z, u.w}};
#else
  Bf16x8 v;
  std::memcpy(v.w, p, sizeof v.w);
  return v;
#endif
}
DS_FN void store8(uint16_t* p, const Bf16x8& v) {
#if defined(__CUDACC__)
  *reinterpret_cast<uint4*>(p) = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]);
#else
  std::memcpy(p, v.w, sizeof v.w);
#endif
}
DS_FN uint16_t elem8(const Bf16x8& v, uint32_t e) {
  return static_cast<uint16_t>(v.w[e / 2] >> (16 * (e % 2)));
}
DS_FN void set8(Bf16x8& v, uint32_t e, uint16_t h) {
  const uint32_t s = 16 * (e % 2);
  v.w[e / 2] = (v.w[e / 2] & ~(0xffffu << s)) | (static_cast<uint32_t>(h) << s);
}
DS_FN F32x8 load8f(const float* p) {
  F32x8 v;
#if defined(__CUDACC__)
  const float4 a = reinterpret_cast<const float4*>(p)[0];
  const float4 b = reinterpret_cast<const float4*>(p)[1];
  v.f[0] = a.x, v.f[1] = a.y, v.f[2] = a.z, v.f[3] = a.w;
  v.f[4] = b.x, v.f[5] = b.y, v.f[6] = b.z, v.f[7] = b.w;
#else
  std::memcpy(v.f, p, sizeof v.f);
#endif
  return v;
}
DS_FN void store8f(float* p, const F32x8& v) {
#if defined(__CUDACC__)
  reinterpret_cast<float4*>(p)[0] = make_float4(v.f[0], v.f[1], v.f[2], v.f[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v.f[4], v.f[5], v.f[6], v.f[7]);
#else
  std::memcpy(p, v.f, sizeof v.f);
#endif
}
// Whether every address is a multiple of 16 bytes.
DS_FN bool aligned16(const void* a, const void* b, const void* c, const void* d) {
  return ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
           reinterpret_cast<uintptr_t>(c) | reinterpret_cast<uintptr_t>(d)) &
          15u) == 0;
}
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

constexpr uint32_t kRopeMax = 512;   // widest unit a rotated norm stages
constexpr uint32_t kNormMax = 8192;  // widest unit a norm stages

// The kernel's form (`norm_item`): a unit's BF16 inputs (after the residual
// add) are staged in shared memory, so the sum's 16 serial lanes, in the
// reference's order, read them at shared-memory latency. A unit wider than
// `kNormWarpMax` takes a CTA of 256 threads; narrower units (the query and key
// heads) take a warp each, eight to a CTA, so a launch of many small units
// keeps eight in flight a CTA and meets no block barrier. Every phase that
// touches global memory issues a batch of a thread's loads before any of its
// stores: three of its 16-byte vectors `t`, `t + T`, … (at most four; a
// 5,120-wide unit's whole share, so a decode-size launch makes one round trip a
// phase) when the unit's rows, strides and arrays allow them, else eight of its
// elements `t`, `t + T`, … (48 registers a thread, five CTAs an SM). The
// rotated units stage their normed values in the same buffer,
// as BF16 (a normed value is BF16-valued, and a NaN stays a NaN), before the
// rotation reads each one's partner. Only where the loads sit and who computes
// which element differ from the reference form (`norm_reference_block`, the
// host check's and the device test's reference): every value is computed by
// the same expressions, and the sum in the same order.
constexpr uint32_t kNormWarpMax = 1024;                          // widest unit a warp takes
constexpr uint32_t kNormItems = kElementThreads / kWarp;          // a warp's units a CTA
constexpr uint32_t kNormVecs = kNormMax / 8 / kElementThreads;    // a thread's vectors
constexpr uint32_t kNormScalar = kNormMax / kElementThreads;      // a thread's elements
constexpr uint32_t kNormBatch = 8;     // elements a thread loads before it stores any
constexpr uint32_t kNormVecBatch = 3;  // vectors a thread loads before it stores any
static_assert(kNormWarpMax / 8 / kWarp == kNormVecs, "a warp's thread holds as many vectors");
static_assert(kNormWarpMax / kWarp == kNormScalar, "and as many elements");
static_assert(kNormWarpMax * kNormItems == kNormMax, "eight warp units fill the buffer");
static_assert(kRopeMax <= kNormWarpMax, "a rotated unit is a warp's");

// The 16 lane sums, the tail and the factor of one unit.
struct NormLanes {
  float lanes[16];
  float tail;
  float r;
};
// An FP8 quantisation block's scale, its reciprocal rounded once, and whether
// every lane divides (a scale that is not a normal number): `quant_block_scale`.
struct QuantScale {
  float s;
  float r;
  bool divide;
};

// A fused quantisation's group maxima (a word of padding a block of 32) and
// block scales, for a unit of `n` elements.
constexpr uint32_t kNormQuantWords = kNormMax / 4 + kNormMax / QW_BLOCK;
constexpr uint32_t kNormWarpQuantWords = kNormWarpMax / 4 + kNormWarpMax / QW_BLOCK;

struct QW_ALIGN16 NormShared {
  uint16_t row[kNormMax];  // one CTA unit, or eight warp units of kNormWarpMax
  NormLanes ln[kNormItems];
  // A fused quantisation's: one CTA unit's, or eight warp units' in turn.
  uint32_t gm[kNormQuantWords];
  QuantScale sc[kNormMax / QW_BLOCK];
};

// The fused quantisation of a unit's normed values staged in `buf`
// (`norm_item` with `codes`; defined with the quantiser below).
template <uint32_t T, class P>
DS_FN void norm_quant(const QwNormParams& p, uint32_t row, const P& par, const uint16_t* buf,
                      uint32_t* gm, QuantScale* sc);

// Whether every unit of the launch moves in 16-byte vectors: whole vectors
// per unit, every row and unit offset and every array aligned to them.
DS_FN bool norm_vectors(const QwNormParams& p) {
  const uint32_t strides = p.n | p.x_stride | p.x_unit_stride | p.out_stride |
                           p.out_unit_stride | (p.add != nullptr ? p.add_stride : 0u);
  return strides % 8 == 0 && aligned16(p.x, p.out, p.add, p.weight) &&
         aligned16(p.add_parts, p.add_parts, p.add_parts, p.add_parts);
}

// Element `i` of the residual add's input for `row`: the BF16 `add`, or (the
// fused form) the split GEMM's partials added in split order and rounded once
// (`add_parts`).
template <bool Fused>
DS_FN uint16_t norm_add_element(const QwNormParams& p, const uint16_t* add, uint32_t row,
                                uint32_t i) {
  if (!Fused || add != nullptr) return add[i];
  const uint64_t at = static_cast<uint64_t>(row) * p.n + i;
  const uint64_t step = static_cast<uint64_t>(p.add_part_rows) * p.n;
  float acc = p.add_parts[at];
  for (uint32_t s = 1; s < p.add_splits; ++s) acc = acc + p.add_parts[at + s * step];
  return ds::bf16_store(ds::bf16_round(acc));
}

// Elements `8q .. 8q + 8` of the residual add's input, as `norm_add_element`.
template <bool Fused>
DS_FN Bf16x8 norm_add_vector(const QwNormParams& p, const uint16_t* add, uint32_t row,
                             uint32_t q) {
  if (!Fused || add != nullptr) return load8(add + 8 * q);
  const uint64_t at = static_cast<uint64_t>(row) * p.n + 8 * q;
  const uint64_t step = static_cast<uint64_t>(p.add_part_rows) * p.n;
  F32x8 acc = load8f(p.add_parts + at);
  for (uint32_t s = 1; s < p.add_splits; ++s) {
    const F32x8 v = load8f(p.add_parts + at + s * step);
    QW_UNROLL
    for (uint32_t e = 0; e < 8; ++e) acc.f[e] = acc.f[e] + v.f[e];
  }
  Bf16x8 a;
  QW_UNROLL
  for (uint32_t e = 0; e < 8; ++e) set8(a, e, ds::bf16_store(ds::bf16_round(acc.f[e])));
  return a;
}

// Unit `unit` of row `row` over `T` threads, staged in `buf` (at least `n`
// values, 16-byte aligned).
// `Fused`: the kernel of the fused forms (`eidola_qwen_norm_fused`), which reads
// `add_parts` and `codes`; the plain kernel is compiled without them.
template <uint32_t T, bool Fused, class P>
DS_FN void norm_item(const QwNormParams& p, uint32_t row, uint32_t unit, const P& par,
                     NormLanes& ln, uint16_t* buf, uint32_t* gm, QuantScale* sc) {
  uint16_t* x = p.x + static_cast<uint64_t>(row) * p.x_stride +
                static_cast<uint64_t>(unit) * p.x_unit_stride;
  uint16_t* out = p.out + static_cast<uint64_t>(row) * p.out_stride +
                  static_cast<uint64_t>(unit) * p.out_unit_stride;
  const uint16_t* add =
      p.add != nullptr ? p.add + static_cast<uint64_t>(row) * p.add_stride : nullptr;
  const bool adding = add != nullptr || (Fused && p.add_parts != nullptr);
  const bool quantise = Fused && p.codes != nullptr;
  const uint32_t n = p.n;
  const uint32_t nv = n / 8;
  const bool vec = norm_vectors(p);
  // The residual add and the staging.
  par.run([&](uint32_t t) {
    if (vec) {
      for (uint32_t c0 = 0; c0 < kNormVecs && t + c0 * T < nv; c0 += kNormVecBatch) {
        Bf16x8 v[kNormVecBatch];
        Bf16x8 a[kNormVecBatch];
        QW_UNROLL
        for (uint32_t c = 0; c < kNormVecBatch; ++c) {
          if (t + (c0 + c) * T < nv) v[c] = load8(x + 8 * (t + (c0 + c) * T));
        }
        if (adding) {
          QW_UNROLL
          for (uint32_t c = 0; c < kNormVecBatch; ++c) {
            if (t + (c0 + c) * T < nv) a[c] = norm_add_vector<Fused>(p, add, row, t + (c0 + c) * T);
          }
        }
        QW_UNROLL
        for (uint32_t c = 0; c < kNormVecBatch; ++c) {
          const uint32_t q = t + (c0 + c) * T;
          if (q >= nv) continue;
          if (adding) {
            QW_UNROLL
            for (uint32_t e = 0; e < 8; ++e) {
              set8(v[c], e,
                   ds::bf16_store(ds::bf16_round(ds::bf16_value(elem8(v[c], e)) +
                                                 ds::bf16_value(elem8(a[c], e)))));
            }
            store8(x + 8 * q, v[c]);
          }
          store8(buf + 8 * q, v[c]);
        }
      }
      return;
    }
    for (uint32_t c0 = 0; c0 < kNormScalar && t + c0 * T < n; c0 += kNormBatch) {
      uint16_t v[kNormBatch];
      uint16_t a[kNormBatch];
      QW_UNROLL
      for (uint32_t c = 0; c < kNormBatch; ++c) {
        const uint32_t i = t + (c0 + c) * T;
        v[c] = i < n ? x[i] : uint16_t{0};
      }
      if (adding) {
        QW_UNROLL
        for (uint32_t c = 0; c < kNormBatch; ++c) {
          const uint32_t i = t + (c0 + c) * T;
          a[c] = i < n ? norm_add_element<Fused>(p, add, row, i) : uint16_t{0};
        }
      }
      QW_UNROLL
      for (uint32_t c = 0; c < kNormBatch; ++c) {
        const uint32_t i = t + (c0 + c) * T;
        if (i >= n) continue;
        if (adding) {
          v[c] = ds::bf16_store(ds::bf16_round(ds::bf16_value(v[c]) + ds::bf16_value(a[c])));
          x[i] = v[c];
        }
        buf[i] = v[c];
      }
    }
  });
  const uint32_t full = n & ~15u;
  par.run([&](uint32_t t) {
    if (t < 16) {
      float acc = 0.0f;
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
      for (uint32_t i = t; i < full; i += 16) {
        const float v = ds::bf16_value(buf[i]);
        acc = acc + v * v;
      }
      ln.lanes[t] = acc;
    } else if (t == 16) {
      float tail = 0.0f;
      for (uint32_t i = full; i < n; ++i) {
        const float v = ds::bf16_value(buf[i]);
        tail = tail + v * v;
      }
      ln.tail = tail;
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = ln.lanes[j];
      ln.r = ds::rms_factor(ds::pinned_reduce(acc, ln.tail), n, p.eps);
    }
  });
  // The normed values: into `out`, or (rotated units) back into `buf`.
  uint16_t* const normed = p.rotary == 0 ? out : buf;
  par.run([&](uint32_t t) {
    const float r = ln.r;
    if (vec) {
      for (uint32_t c0 = 0; c0 < kNormVecs && t + c0 * T < nv; c0 += kNormVecBatch) {
        F32x8 w[kNormVecBatch];
        QW_UNROLL
        for (uint32_t c = 0; c < kNormVecBatch; ++c) {
          if (t + (c0 + c) * T < nv) w[c] = load8f(p.weight + 8 * (t + (c0 + c) * T));
        }
        QW_UNROLL
        for (uint32_t c = 0; c < kNormVecBatch; ++c) {
          const uint32_t q = t + (c0 + c) * T;
          if (q >= nv) continue;
          Bf16x8 h = load8(buf + 8 * q);
          QW_UNROLL
          for (uint32_t e = 0; e < 8; ++e) {
            set8(h, e,
                 ds::bf16_store(ds::rms_element(ds::bf16_value(elem8(h, e)), r, w[c].f[e])));
          }
          store8(normed + 8 * q, h);
          // A fused quantisation reads the normed values back from the staging.
          if (quantise) store8(buf + 8 * q, h);
        }
      }
      return;
    }
    for (uint32_t c0 = 0; c0 < kNormScalar && t + c0 * T < n; c0 += kNormBatch) {
      float w[kNormBatch];
      QW_UNROLL
      for (uint32_t c = 0; c < kNormBatch; ++c) {
        const uint32_t i = t + (c0 + c) * T;
        w[c] = i < n ? p.weight[i] : 0.0f;
      }
      QW_UNROLL
      for (uint32_t c = 0; c < kNormBatch; ++c) {
        const uint32_t i = t + (c0 + c) * T;
        if (i >= n) continue;
        const uint16_t h = ds::bf16_store(ds::rms_element(ds::bf16_value(buf[i]), r, w[c]));
        normed[i] = h;
        if (quantise) buf[i] = h;
      }
    }
  });
  if (quantise) {
    norm_quant<T>(p, row, par, buf, gm, sc);
    return;
  }
  if (p.rotary == 0) return;
  const uint64_t at = static_cast<uint64_t>(p.positions[row]) * p.rotary;
  const uint32_t half = p.rotary / 2;
  // Element `j`'s rotated value from the staged normed values.
  auto rotated = [&](uint32_t j) {
    float v = ds::bf16_value(buf[j]);
    if (j < p.rotary) {
      const float rot = j < half ? -ds::bf16_value(buf[j + half]) : ds::bf16_value(buf[j - half]);
      v = ds::bf16_round(ds::bf16_round(v * p.rope_cos[at + j]) +
                         ds::bf16_round(rot * p.rope_sin[at + j]));
    }
    return ds::bf16_store(v);
  };
  par.run([&](uint32_t t) {
    if (vec) {
      for (uint32_t q = t; q < nv; q += T) {
        Bf16x8 o;
        QW_UNROLL
        for (uint32_t e = 0; e < 8; ++e) set8(o, e, rotated(8 * q + e));
        store8(out + 8 * q, o);
      }
      return;
    }
    for (uint32_t j = t; j < n; j += T) out[j] = rotated(j);
  });
}

// The reference form: a CTA a unit, the unit's elements `t`, `t + 256`, …
// a thread, the normed values of a rotated unit staged as f32. The host check
// and the device test hold `norm_item` to it bit for bit.
struct NormRefShared {
  float lanes[16];
  float tail;
  float r;
  uint16_t row[kNormMax];
  float staged[kRopeMax];
};

// The elements one thread of a norm holds.
constexpr uint32_t kNormPer = kNormMax / kElementThreads;

template <class P>
DS_FN void norm_reference_block(const QwNormParams& p, uint32_t row, uint32_t unit, const P& par,
                                NormRefShared& sh) {
  uint16_t* x = p.x + static_cast<uint64_t>(row) * p.x_stride +
                static_cast<uint64_t>(unit) * p.x_unit_stride;
  uint16_t* out = p.out + static_cast<uint64_t>(row) * p.out_stride +
                  static_cast<uint64_t>(unit) * p.out_unit_stride;
  const uint32_t threads = kElementThreads;
  const uint32_t n = p.n;
  const uint16_t* add =
      p.add != nullptr ? p.add + static_cast<uint64_t>(row) * p.add_stride : nullptr;
  par.run([&](uint32_t t) {
    uint16_t v[kNormPer];
    uint16_t a[kNormPer];
    QW_UNROLL
    for (uint32_t c = 0; c < kNormPer; ++c) {
      const uint32_t i = t + c * threads;
      v[c] = i < n ? x[i] : uint16_t{0};
    }
    if (add != nullptr) {
      QW_UNROLL
      for (uint32_t c = 0; c < kNormPer; ++c) {
        const uint32_t i = t + c * threads;
        a[c] = i < n ? add[i] : uint16_t{0};
      }
      QW_UNROLL
      for (uint32_t c = 0; c < kNormPer; ++c) {
        const uint32_t i = t + c * threads;
        if (i < n) {
          v[c] = ds::bf16_store(ds::bf16_round(ds::bf16_value(v[c]) + ds::bf16_value(a[c])));
          x[i] = v[c];
        }
      }
    }
    QW_UNROLL
    for (uint32_t c = 0; c < kNormPer; ++c) {
      const uint32_t i = t + c * threads;
      if (i < n) sh.row[i] = v[c];
    }
  });
  const uint32_t full = n & ~15u;
  par.run([&](uint32_t t) {
    if (t < 16) {
      float acc = 0.0f;
      const uint16_t* r = sh.row;
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
      for (uint32_t i = t; i < full; i += 16) {
        const float v = ds::bf16_value(r[i]);
        acc = acc + v * v;
      }
      sh.lanes[t] = acc;
    } else if (t == 16) {
      float tail = 0.0f;
      for (uint32_t i = full; i < n; ++i) {
        const float v = ds::bf16_value(sh.row[i]);
        tail = tail + v * v;
      }
      sh.tail = tail;
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = sh.lanes[j];
      sh.r = ds::rms_factor(ds::pinned_reduce(acc, sh.tail), n, p.eps);
    }
  });
  if (p.rotary == 0) {
    par.run([&](uint32_t t) {
      float w[kNormPer];
      QW_UNROLL
      for (uint32_t c = 0; c < kNormPer; ++c) {
        const uint32_t i = t + c * threads;
        w[c] = i < n ? p.weight[i] : 0.0f;
      }
      const float r = sh.r;
      QW_UNROLL
      for (uint32_t c = 0; c < kNormPer; ++c) {
        const uint32_t i = t + c * threads;
        if (i < n) out[i] = ds::bf16_store(ds::rms_element(ds::bf16_value(sh.row[i]), r, w[c]));
      }
    });
    return;
  }
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < n; i += threads) {
      sh.staged[i] = ds::rms_element(ds::bf16_value(sh.row[i]), sh.r, p.weight[i]);
    }
  });
  const uint64_t at = static_cast<uint64_t>(p.positions[row]) * p.rotary;
  const uint32_t half = p.rotary / 2;
  par.run([&](uint32_t t) {
    for (uint32_t j = t; j < n; j += threads) {
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

// The reference form: one thread quantises one 128-wide block of one row, in
// the model crate's order. The kernel runs the warp form below; the host check
// holds that form to this one bit for bit (`qwen_host_quant_thread`).
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

// The warp form (`eidola_qwen_quant`): a warp per 128-wide block, lane `l`
// holding the block's elements `4l .. 4l + 4`, `kQuantWarps` blocks per CTA,
// each for one row at decode sizes, or for up to `kQuantRows` consecutive rows
// at prefill sizes (every row's load issued before any row's arithmetic: a
// row past the CTA's reads its first row and is dropped, so the loads carry no
// branch between them); the grid's y says which. Bit for bit `quant_thread`:
//
// - The amax is taken on the BF16 magnitudes' bits as integers, a NaN's as 0:
//   for non-negative floats, +0 to infinity, the order of the bits is the
//   order of the values, so this is the reference's `max32` fold from +0 (which
//   drops NaNs) in any order, and the lanes combine theirs by a butterfly.
// - Each code is `e4m3_code` (`ds::e4m3_encode` without branches) of the
//   element divided by the scale, as there, computed as a product with the
//   scale's reciprocal wherever that gives the same code
//   (`quant_lane_store`).
constexpr uint32_t kQuantLane = QW_BLOCK / kWarp;
constexpr uint32_t kQuantWarps = kBlockThreads / kWarp;
constexpr uint32_t kQuantRows = 4;
static_assert(kQuantLane == 4, "four elements a lane: one 8-byte load, one 4-byte store");

// `ds::e4m3_encode(x)` for every f32 `x` (the host check runs both over all
// 2^32: `qwen_host_e4m3_check`), without a branch. The codes step by 2^-9 from
// +0 to 2^-5, so below 2^-6 a code is `x`'s magnitude rounded to a multiple of
// 2^-9, which adding 2^14 (whose unit in the last place is 2^-9) rounds to
// nearest, ties to even, as the encoder's integer rounding does. From 2^-6 the
// code is the magnitude's bits rebiased by 120 exponents and rounded to their
// top 8 bits, ties to even, the carry running into the exponent, saturating at
// 0x7e (448, which the infinities take too). A NaN gives 0; the sign is the
// top bit.
DS_FN uint32_t e4m3_code(float x) {
  const uint32_t b = ds::f32_bits(x);
  const uint32_t a = b & 0x7fffffffu;
  const uint32_t sub = ds::f32_bits(ds::abs32(x) + 16384.0f) - 0x46800000u;
  const uint32_t nor =
      ds::min_u32((a - 0x3c000000u + 0x7ffffu + ((a >> 20) & 1u)) >> 20, 0x7eu);
  const uint32_t c = (a < 0x3c800000u ? sub : nor) | ((b >> 24) & 0x80u);
  return a > 0x7f800000u ? 0u : c;
}

// The block warp `w` of CTA column `bx` quantises (in each of the CTA's rows),
// or false past the row's blocks.
DS_FN bool quant_block_of(const QwQuantParams& p, uint32_t bx, uint32_t w, uint32_t* b) {
  *b = bx * kQuantWarps + w;
  return *b < p.k / QW_BLOCK;
}

// Lane `l`'s BF16 elements of block `b` of `row`: element `j` in bits
// `16 (j % 2)` of word `j / 2`.
struct QuantLane {
  uint32_t w[kQuantLane / 2];
};

DS_FN QuantLane quant_lane_load(const QwQuantParams& p, uint32_t row, uint32_t b, uint32_t l) {
  const uint16_t* x =
      p.x + static_cast<uint64_t>(row) * p.x_stride + b * QW_BLOCK + l * kQuantLane;
  QuantLane v;
#if defined(__CUDACC__)
  const uint2 u = *reinterpret_cast<const uint2*>(x);
  v.w[0] = u.x;
  v.w[1] = u.y;
#else
  std::memcpy(v.w, x, sizeof v.w);
#endif
  return v;
}

// The bits of the largest non-NaN magnitude of the lane's elements, or 0.
DS_FN uint32_t quant_lane_amax(const QuantLane& v) {
  uint32_t m = 0;
  for (uint32_t j = 0; j < kQuantLane; ++j) {
    const uint32_t a = (v.w[j / 2] >> (16 * (j % 2))) & 0x7fffu;
    m = a > 0x7f80u || a < m ? m : a;
  }
  return m;
}

// Lane `l`'s codes of block `b` given the block's amax bits; lane 0 stores the
// scale.
//
// A code is `E(x / s)`: `E` (`e4m3_code`) of the correctly rounded quotient.
// The lane takes `E(q)` of `q = x · r` instead, `r = 1 / s` correctly rounded
// once, unless `q` falls where that could differ, and then divides as the
// reference does. Why `E(q)` is `E(x / s)` elsewhere, for a normal `s` (and `r`
// is then normal too, since `s` is at most 2^128 / 448): `q` is within a
// relative 2u + u² of `x / s` (u = 2^-24), and the rounded quotient within u,
// so `q` is within 3.0001 units in the last place of the rounded quotient. `E`
// is constant between consecutive midpoints of E4M3 codes and changes only at
// them. From 2^-6 up a midpoint's low 20 bits are 0x80000 and it sits in the
// middle of its binade, so a `q` whose low 20 bits are more than 8 from 0x80000
// has no midpoint within 8 units of it, and the rounded quotient lies in the
// same open interval between midpoints. Below 2^-11 both are under 2^-10, so
// both encode to a signed zero, with `x`'s sign. Between 2^-11 and 2^-6 (eight
// units past it, to be safe) the midpoints are E4M3's subnormal ones, and the
// lane divides. A NaN `q` encodes to 0 either way. A block whose scale is 0,
// infinite or subnormal divides throughout.
//
// Split in two so a caller can compute a block's scale once
// (`quant_block_scale`) and each lane's codes from it (`quant_lane_word`).

DS_FN QuantScale quant_block_scale(uint32_t amax) {
  const float s = ds::bf16_value(static_cast<uint16_t>(amax)) / 448.0f;
  return QuantScale{s, 1.0f / s, ds::f32_bits(s) - 0x00800000u >= 0x7f000000u};
}

DS_FN uint32_t quant_lane_word(const QuantLane& v, const QuantScale& sc) {
  const float s = sc.s;
  const float r = sc.r;
  bool divide = sc.divide;
  uint32_t word = 0;
  for (uint32_t j = 0; j < kQuantLane; ++j) {
    const float q = ds::bf16_value(static_cast<uint16_t>(v.w[j / 2] >> (16 * (j % 2)))) * r;
    const uint32_t a = ds::f32_bits(q) & 0x7fffffffu;
    divide = divide || a - 0x3a000000u < 0x02800008u || (a & 0xfffffu) - 0x7fff8u < 17u;
    word |= e4m3_code(q) << (8 * j);
  }
  if (divide) {
    word = 0;
    for (uint32_t j = 0; j < kQuantLane; ++j) {
      const float x = ds::bf16_value(static_cast<uint16_t>(v.w[j / 2] >> (16 * (j % 2))));
      word |= e4m3_code(x / s) << (8 * j);
    }
  }
  return word;
}

DS_FN void quant_lane_store(const QwQuantParams& p, uint32_t row, uint32_t b, uint32_t l,
                            const QuantLane& v, uint32_t amax) {
  const QuantScale sc = quant_block_scale(amax);
  const float s = sc.s;
  const uint32_t word = quant_lane_word(v, sc);
  uint8_t* codes = p.codes + static_cast<uint64_t>(row) * p.k + b * QW_BLOCK + l * kQuantLane;
#if defined(__CUDACC__)
  *reinterpret_cast<uint32_t*>(codes) = word;
#else
  std::memcpy(codes, &word, sizeof word);
#endif
  if (l == 0) p.scales[static_cast<uint64_t>(b) * p.m_pad + row] = s;
}

// The fused quantisation (`norm_item` with `codes`): the unit's normed values,
// staged in `buf`, quantised exactly as the warp form above quantises them
// from memory. Group `g` of four elements is lane `g % 32` of block `g / 32`:
// its amax bits into `gm`; then each block's amax, the integer maximum of its
// 32 lanes' (the butterfly's maximum, in any order), and from it the block's
// scale once (`quant_block_scale`), stored with the scales; then each group's
// codes from that scale (`quant_lane_word`). A group's amax sits at `g + g /
// 32` (a word of padding a block), so a block's 32 are read by its thread
// without two blocks' threads meeting in one bank.
static_assert(kNormWarpMax % QW_BLOCK == 0, "a warp unit's blocks are whole");
static_assert(kQuantLane == 4, "the group maxima count groups of four");

template <uint32_t T, class P>
DS_FN void norm_quant(const QwNormParams& p, uint32_t row, const P& par, const uint16_t* buf,
                      uint32_t* gm, QuantScale* sc) {
  const uint32_t groups = p.n / kQuantLane;
  const uint32_t blocks = p.n / QW_BLOCK;
  auto lane = [&](uint32_t g) {
    QuantLane v;
    for (uint32_t j = 0; j < kQuantLane / 2; ++j) {
      v.w[j] = static_cast<uint32_t>(buf[kQuantLane * g + 2 * j]) |
               (static_cast<uint32_t>(buf[kQuantLane * g + 2 * j + 1]) << 16);
    }
    return v;
  };
  par.run([&](uint32_t t) {
    for (uint32_t g = t; g < groups; g += T) gm[g + g / kWarp] = quant_lane_amax(lane(g));
  });
  par.run([&](uint32_t t) {
    for (uint32_t b = t; b < blocks; b += T) {
      const uint32_t* m = gm + b * (kWarp + 1);
      uint32_t a = m[0];
      for (uint32_t l = 1; l < kWarp; ++l) a = m[l] > a ? m[l] : a;
      sc[b] = quant_block_scale(a);
      p.scales[static_cast<uint64_t>(b) * p.m_pad + row] = sc[b].s;
    }
  });
  par.run([&](uint32_t t) {
    for (uint32_t g = t; g < groups; g += T) {
      const uint32_t word = quant_lane_word(lane(g), sc[g / kWarp]);
      uint8_t* codes = p.codes + static_cast<uint64_t>(row) * p.n + kQuantLane * g;
#if defined(__CUDACC__)
      *reinterpret_cast<uint32_t*>(codes) = word;
#else
      std::memcpy(codes, &word, sizeof word);
#endif
    }
  });
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

// Value channels a dn_prep thread convolves before storing any.
constexpr uint32_t kDnPrepBatch = 4;

// Block (position, key head): the key head's queries and keys, its value
// heads' values, `β` and `α`.
template <class P>
DS_FN void dn_prep_block(const QwDnPrepParams& p, uint32_t pos, uint32_t kh, const P& par,
                         DnPrepShared& sh) {
  const uint32_t kd = p.k_dim;
  const uint32_t key_dim = p.k_heads * kd;
  const uint32_t group = p.v_heads / p.k_heads;
  const uint32_t threads = kBlockThreads;
  // A thread's convolutions are computed a batch at a time before their
  // stores, so their loads overlap (a store between them would order every
  // later load behind it).
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < kd; i += threads) {
      const float q = dn_conv(p, pos, kh * kd + i);
      const float k = dn_conv(p, pos, key_dim + kh * kd + i);
      sh.q[i] = q;
      sh.k[i] = k;
    }
    const uint32_t values = group * p.v_dim;
    for (uint32_t base = 0; base < values; base += kDnPrepBatch * threads) {
      float y[kDnPrepBatch];
      QW_UNROLL
      for (uint32_t b = 0; b < kDnPrepBatch; ++b) {
        const uint32_t i = base + b * threads + t;
        y[b] = i < values ? dn_conv(p, pos, 2 * key_dim + kh * values + i) : 0.0f;
      }
      QW_UNROLL
      for (uint32_t b = 0; b < kDnPrepBatch; ++b) {
        const uint32_t i = base + b * threads + t;
        if (i < values) p.v[static_cast<uint64_t>(pos) * p.v_heads * p.v_dim + kh * values + i] = y[b];
      }
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

// The kernel's form (`dn_norm_item`): a warp a (row, head), four to a CTA, the
// head's outputs staged in shared memory (a lane's four consecutive values in
// one 16-byte load when the head's arrays allow it, else values `l`,
// `l + 32`, …), so the 16 serial lanes read them at shared-memory latency;
// the last phase issues a lane's loads before its stores. The reference form
// (`dn_norm_reference_block`: a CTA a head, a thread a value) is what the host
// check and the device test hold it to: the same expressions, the sum in the
// same order.
constexpr uint32_t kDnNormMax = kBlockThreads;  // widest head the gated norm stages
constexpr uint32_t kDnNormItems = kBlockThreads / kWarp;  // heads a CTA

struct QW_ALIGN16 DnNormShared {
  float o[kDnNormMax];
  float lanes[16];
  float r;
};

// Whether a lane moves four consecutive values at once: heads of whole
// fours, and every array and row aligned to them.
DS_FN bool dn_norm_vectors(const QwDnNormParams& p) {
  return (p.v_dim | p.z_stride) % 4 == 0 &&
         ((reinterpret_cast<uintptr_t>(p.o) | reinterpret_cast<uintptr_t>(p.weight)) & 15u) == 0 &&
         ((reinterpret_cast<uintptr_t>(p.z) | reinterpret_cast<uintptr_t>(p.out)) & 7u) == 0;
}

// Value `i`'s output given the head's factor.
DS_FN uint16_t dn_norm_value(float o, float r, float w, uint16_t zb) {
  const float n = ds::bf16_round(w * ds::bf16_round(o * r));
  return ds::bf16_store(ds::bf16_round(n * ds::silu32(ds::bf16_value(zb))));
}

template <class P>
DS_FN void dn_norm_item(const QwDnNormParams& p, uint32_t row, uint32_t h, const P& par,
                        DnNormShared& sh) {
  const uint32_t vd = p.v_dim;
  const uint64_t head = (static_cast<uint64_t>(row) * p.v_heads + h) * vd;
  const float* o = p.o + head;
  const uint16_t* z = p.z + static_cast<uint64_t>(row) * p.z_stride + h * vd;
  uint16_t* out = p.out + head;
  const uint32_t full = vd & ~15u;
  const bool vec = dn_norm_vectors(p);
  par.run([&](uint32_t l) {
    if (vec) {
      if (4 * l < vd) {
#if defined(__CUDACC__)
        *reinterpret_cast<float4*>(sh.o + 4 * l) = *reinterpret_cast<const float4*>(o + 4 * l);
#else
        std::memcpy(sh.o + 4 * l, o + 4 * l, 4 * sizeof(float));
#endif
      }
      return;
    }
    float v[kDnNormMax / kWarp];
    QW_UNROLL
    for (uint32_t c = 0; c < kDnNormMax / kWarp; ++c) {
      v[c] = l + c * kWarp < vd ? o[l + c * kWarp] : 0.0f;
    }
    QW_UNROLL
    for (uint32_t c = 0; c < kDnNormMax / kWarp; ++c) {
      if (l + c * kWarp < vd) sh.o[l + c * kWarp] = v[c];
    }
  });
  par.run([&](uint32_t l) {
    if (l < 16) {
      float acc = 0.0f;
      for (uint32_t i = l; i < full; i += 16) acc = acc + sh.o[i] * sh.o[i];
      sh.lanes[l] = acc;
    }
  });
  par.run([&](uint32_t l) {
    if (l == 0) {
      float tail = 0.0f;
      for (uint32_t i = full; i < vd; ++i) tail = tail + sh.o[i] * sh.o[i];
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = sh.lanes[j];
      sh.r = ds::rms_factor(ds::pinned_reduce(acc, tail), vd, p.eps);
    }
  });
  par.run([&](uint32_t l) {
    const float r = sh.r;
    if (vec) {
      if (4 * l >= vd) return;
      float w[4];
      uint16_t zb[4];
#if defined(__CUDACC__)
      const float4 wv = *reinterpret_cast<const float4*>(p.weight + 4 * l);
      const uint2 zv = *reinterpret_cast<const uint2*>(z + 4 * l);
      w[0] = wv.x, w[1] = wv.y, w[2] = wv.z, w[3] = wv.w;
      zb[0] = static_cast<uint16_t>(zv.x), zb[1] = static_cast<uint16_t>(zv.x >> 16);
      zb[2] = static_cast<uint16_t>(zv.y), zb[3] = static_cast<uint16_t>(zv.y >> 16);
#else
      std::memcpy(w, p.weight + 4 * l, sizeof w);
      std::memcpy(zb, z + 4 * l, sizeof zb);
#endif
      uint16_t y[4];
      QW_UNROLL
      for (uint32_t e = 0; e < 4; ++e) y[e] = dn_norm_value(sh.o[4 * l + e], r, w[e], zb[e]);
#if defined(__CUDACC__)
      *reinterpret_cast<uint2*>(out + 4 * l) =
          make_uint2(static_cast<uint32_t>(y[0]) | (static_cast<uint32_t>(y[1]) << 16),
                     static_cast<uint32_t>(y[2]) | (static_cast<uint32_t>(y[3]) << 16));
#else
      std::memcpy(out + 4 * l, y, sizeof y);
#endif
      return;
    }
    float w[kDnNormMax / kWarp];
    uint16_t zb[kDnNormMax / kWarp];
    QW_UNROLL
    for (uint32_t c = 0; c < kDnNormMax / kWarp; ++c) {
      const uint32_t i = l + c * kWarp;
      w[c] = i < vd ? p.weight[i] : 0.0f;
      zb[c] = i < vd ? z[i] : uint16_t{0};
    }
    QW_UNROLL
    for (uint32_t c = 0; c < kDnNormMax / kWarp; ++c) {
      const uint32_t i = l + c * kWarp;
      if (i < vd) out[i] = dn_norm_value(sh.o[i], r, w[c], zb[c]);
    }
  });
}

// The reference form: a CTA of 128 threads a (row, head), a thread a value.
struct DnNormRefShared {
  float lanes[16];
  float r;
  float o[kDnNormMax];
};

template <class P>
DS_FN void dn_norm_reference_block(const QwDnNormParams& p, uint32_t row, uint32_t h,
                                   const P& par, DnNormRefShared& sh) {
  const uint32_t vd = p.v_dim;
  const float* o = p.o + (static_cast<uint64_t>(row) * p.v_heads + h) * vd;
  const uint32_t full = vd & ~15u;
  par.run([&](uint32_t t) {
    if (t < vd) sh.o[t] = o[t];
  });
  par.run([&](uint32_t t) {
    if (t < 16) {
      float acc = 0.0f;
      for (uint32_t i = t; i < full; i += 16) acc = acc + sh.o[i] * sh.o[i];
      sh.lanes[t] = acc;
    }
  });
  par.run([&](uint32_t t) {
    if (t == 0) {
      float tail = 0.0f;
      for (uint32_t i = full; i < vd; ++i) tail = tail + sh.o[i] * sh.o[i];
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = sh.lanes[j];
      sh.r = ds::rms_factor(ds::pinned_reduce(acc, tail), vd, p.eps);
    }
  });
  par.run([&](uint32_t t) {
    if (t >= vd) return;
    const float w = p.weight[t];
    const uint16_t zb = p.z[static_cast<uint64_t>(row) * p.z_stride + h * vd + t];
    const float n = ds::bf16_round(w * ds::bf16_round(sh.o[t] * sh.r));
    const float z = ds::bf16_value(zb);
    p.out[(static_cast<uint64_t>(row) * p.v_heads + h) * vd + t] =
        ds::bf16_store(ds::bf16_round(n * ds::silu32(z)));
  });
}

// ---- BF16 → f32 ----

// The kernel's form: a thread widens 8 consecutive elements of a row, one
// 16-byte load and two 16-byte stores when the rows and arrays allow, else
// element by element (every load before any store). The reference form
// (`widen_thread`, an element a thread) is what the host check and the device
// test hold it to.
constexpr uint32_t kWidenPer = 8;

DS_FN void widen8_thread(const QwWidenParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i0 = (bx * kElementThreads + t) * kWidenPer;
  if (row >= p.rows || i0 >= p.width) return;
  const uint16_t* x = p.x + static_cast<uint64_t>(row) * p.x_stride;
  float* out = p.out + static_cast<uint64_t>(row) * p.out_stride;
  if ((p.width | p.x_stride) % 8 == 0 && p.out_stride % 4 == 0 &&
      aligned16(p.x, p.out, p.x, p.out)) {
    const Bf16x8 v = load8(x + i0);
    F32x8 f;
    QW_UNROLL
    for (uint32_t e = 0; e < kWidenPer; ++e) f.f[e] = ds::bf16_value(elem8(v, e));
    store8f(out + i0, f);
    return;
  }
  uint16_t v[kWidenPer];
  QW_UNROLL
  for (uint32_t e = 0; e < kWidenPer; ++e) v[e] = i0 + e < p.width ? x[i0 + e] : uint16_t{0};
  QW_UNROLL
  for (uint32_t e = 0; e < kWidenPer; ++e) {
    if (i0 + e < p.width) out[i0 + e] = ds::bf16_value(v[e]);
  }
}

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
