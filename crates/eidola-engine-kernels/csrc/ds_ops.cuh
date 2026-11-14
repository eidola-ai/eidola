// The bodies of the DeepSeek-V4.1 scalar kernels (`ds_ops.cu`,
// `ds_ops_div.cu`, `ds_reference.cu`, `ds_head.cu`): what one thread computes,
// as plain functions of the kernel's parameter struct and the thread's place
// in the launch, so the same source runs on a device and in the host check
// (`oracle/ds_kernels_host.cc`). Every floating-point operation is
// `ds_numeric.cuh`'s or an IEEE `+ - * /` written out here in the reference's
// order.
//
// Two shapes of kernel:
//
// - A thread computes a whole unit by itself (an element, a 32-wide block, an
//   output of a quantised GEMM): `<kernel>_thread(p, block x, block y[, z],
//   thread)` is the entire kernel.
// - A pinned-order dot or sum (`tensor::dot`) runs on 16 threads, thread `j`
//   of a group holding lane `j`: the terms `i ≡ j (mod 16)` of the full
//   16-wide chunks, added in ascending `i` from +0 (`<kernel>_lane`). The
//   group's first thread collects the 16 lane sums and finishes with
//   `ds::pinned_reduce` and the tail (`<kernel>_finish`). Collecting is the
//   only step that differs between the device (warp shuffles) and the host.
//
// A thread reads its own row's inputs only and no launch-geometry value
// enters any arithmetic, so a row's result does not depend on the other rows
// or on how the launch is shaped.
#pragma once

#include "ds_abi.cuh"
#include "ds_numeric.cuh"

#if defined(__CUDACC__)
#define DSK_FN __device__ __forceinline__
#define DSK_STATUS_OR(ptr, bits) atomicOr((ptr), (bits))
#else
#define DSK_FN inline
#define DSK_STATUS_OR(ptr, bits) (*(ptr) |= (bits))
#endif

namespace dsk {

// ---- launch geometry (mirrored by `eidola-engine-cuda`'s `ds::ops`) ----

// Threads of a block, per kernel family.
constexpr uint32_t kElementThreads = 256;  // one element (or output) a thread
constexpr uint32_t kBlockThreads = 128;    // one 32-wide block a thread
constexpr uint32_t kMixThreads = 1024;     // 64 lane groups
constexpr uint32_t kRouteThreads = 128;
// Outputs one lane thread carries at once (independent chains).
constexpr uint32_t kLinearChains = 4;
constexpr uint32_t kHeadChains = 8;
// Limits the wrappers refuse beyond.
constexpr uint32_t kMaxCopies = 8;      // collapse, expand, stream mean, embed
constexpr uint32_t kMaxMixCopies = 6;   // (2 + copies) · copies + 1 lane groups in 64
constexpr uint32_t kMaxExperts = 1024;  // router scores in shared memory
constexpr uint32_t kMaxActive = 16;

constexpr uint32_t kRouteNormFloorBits = 0x1e3ce508u;  // 1e-20 as f32
constexpr uint32_t kGateFloorBits = 0x358637bdu;       // 1e-6 as f32

DSK_FN float load_bf16(const uint16_t* p, uint64_t i) { return ds::bf16_value(p[i]); }
DSK_FN void store_bf16(uint16_t* p, uint64_t i, float v) {
  p[i] = ds::bf16_bits(ds::bf16_round(v));
}
DSK_FN float load_dt(const void* p, uint32_t dtype, uint64_t i) {
  return dtype == DS_DT_F32 ? static_cast<const float*>(p)[i]
                            : ds::bf16_value(static_cast<const uint16_t*>(p)[i]);
}

// `ds_kv_code_bytes` and `ds_kv_scale_bytes`, callable on a device.
DSK_FN uint32_t code_bytes(uint32_t format, uint32_t width) {
  return format == DS_KV_FP8 ? width : width / 2;
}
DSK_FN uint32_t scale_bytes(uint32_t format, uint32_t width) {
  return format == DS_KV_FP4_E4M3 ? width / 16 : width / 32;
}

// Where entry `e` of a plane keeps its codes and its scales.
DSK_FN void plane_entry(const DsKvPlane& plane, uint32_t e, uint8_t** codes, uint8_t** scales) {
  const uint64_t block = e / plane.entries_per_block;
  const uint64_t o = e % plane.entries_per_block;
  uint8_t* at = plane.base + block * plane.block_bytes;
  *codes = at + plane.codes_offset + o * code_bytes(plane.format, plane.width);
  *scales = at + plane.scales_offset + o * scale_bytes(plane.format, plane.width);
}

// ---- embedding ----

// Grid (ceil(dim / 256), rows). Thread: element `d` of the row, into every copy.
DSK_FN void embed_thread(const DsEmbedParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t d = bx * kElementThreads + t;
  if (d >= p.dim) return;
  const uint32_t token = p.tokens[row];
  uint16_t v = 0;
  if (token < p.vocab) {
    v = p.table[static_cast<uint64_t>(token) * p.dim + d];
  } else if (d == 0) {
    DSK_STATUS_OR(p.status, DS_STATUS_BAD_TOKEN);
  }
  for (uint32_t c = 0; c < p.copies; ++c) {
    p.out[(static_cast<uint64_t>(row) * p.copies + c) * p.dim + d] = v;
  }
}

// ---- quantisers ----

// `quant::act_quant`. Grid (ceil(width / 32 / 128), rows). Thread: one
// 32-wide block, its codes and its scale byte.
DSK_FN void quant_fp8_thread(const DsQuantFp8Params& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t u = bx * kBlockThreads + t;
  const uint32_t blocks = p.width / 32;
  if (u >= blocks) return;
  const uint64_t at = static_cast<uint64_t>(row) * p.width + u * 32;
  float x[32];
  if (p.x_format == DS_X_FP8) {
    // `engram::decode_row`: `bf16(e4m3(code) · 2^(scale - 127))`.
    const uint8_t* codes = static_cast<const uint8_t*>(p.x) + at;
    const float s = ds::e8m0_decode(p.x_scales[static_cast<uint64_t>(row) * blocks + u]);
    for (int j = 0; j < 32; ++j) x[j] = ds::bf16_round(ds::e4m3_decode(codes[j]) * s);
  } else {
    const uint16_t* v = static_cast<const uint16_t*>(p.x) + at;
    for (int j = 0; j < 32; ++j) x[j] = ds::bf16_value(v[j]);
  }
  uint8_t codes[32];
  const uint8_t scale = ds::act_quant_block(x, codes);
  for (int j = 0; j < 32; ++j) p.codes[at + j] = codes[j];
  p.scales[static_cast<uint64_t>(row) * blocks + u] = scale;
}

// `rope::rotate` / `unrotate` then `quant::fake_quant_*`, as codes and
// scales. Grid (ceil(heads · head_dim / 32 / 128), rows). Thread: 32
// consecutive elements. A rotated pair is two adjacent elements at an even
// offset, so it never straddles two threads. `e4m3` selects the global KV's
// format (the only one that divides), which has its own entry.
DSK_FN void rot_quant_thread(const DsRotQuantParams& p, uint32_t bx, uint32_t row, uint32_t t,
                             bool e4m3) {
  const uint32_t u = bx * kBlockThreads + t;
  const uint32_t width = p.heads * p.head_dim;
  if (u >= width / 32) return;
  uint32_t target = 0;
  if (p.targets != nullptr) {
    target = p.targets[row];
    if (target == DS_SLOT_EMPTY) return;
  }
  const uint16_t* v = p.x + static_cast<uint64_t>(row) * width + u * 32;
  float x[32];
  for (int j = 0; j < 32; ++j) x[j] = ds::bf16_value(v[j]);
  if (p.rotation != DS_ROT_NONE) {
    const uint32_t in_head = (u * 32) % p.head_dim;
    const uint32_t start = p.head_dim - p.rope_dim;
    const float* cis = p.cis + static_cast<uint64_t>(p.positions[row]) * p.rope_dim;
    for (uint32_t j = 0; j < 32; j += 2) {
      if (in_head + j < start) continue;
      const uint32_t pair = (in_head + j - start) / 2;
      const float c = cis[2 * pair], s = cis[2 * pair + 1];
      if (p.rotation == DS_ROT_ROTATE) {
        ds::rope_rotate(x[j], x[j + 1], c, s, &x[j], &x[j + 1]);
      } else {
        ds::rope_unrotate(x[j], x[j + 1], c, s, &x[j], &x[j + 1]);
      }
    }
  }
  uint8_t* codes;
  uint8_t* scales;
  if (p.targets != nullptr) {
    plane_entry(p.dst, target, &codes, &scales);
  } else {
    codes = p.dst_codes + static_cast<uint64_t>(row) * code_bytes(p.format, width);
    scales = p.dst_scales + static_cast<uint64_t>(row) * scale_bytes(p.format, width);
  }
  uint8_t c[32];
  if (e4m3) {
    scales[2 * u] = ds::fp4_e4m3_quant_block(x, c);
    scales[2 * u + 1] = ds::fp4_e4m3_quant_block(x + 16, c + 16);
  } else if (p.format == DS_KV_FP8) {
    scales[u] = ds::act_quant_block(x, c);
    for (int j = 0; j < 32; ++j) codes[u * 32 + j] = c[j];
    return;
  } else {
    scales[u] = ds::fp4_pow2_quant_block(x, c);
  }
  for (int j = 0; j < 16; ++j) {
    codes[u * 16 + j] = static_cast<uint8_t>(c[2 * j] | (c[2 * j + 1] << 4));
  }
}

// ---- RMS norm ----

// Grid (rows), 256 threads. Threads 0..15 hold the lanes of `dot(x, x)`;
// after the factor is known every thread writes elements `t, t + 256, …`.
DSK_FN float rms_lane(const DsRmsNormParams& p, uint32_t row, uint32_t lane) {
  const uint16_t* x = p.x + static_cast<uint64_t>(row) * p.width;
  float acc = 0.0f;
  for (uint32_t i = lane; i < (p.width & ~15u); i += 16) {
    const float v = ds::bf16_value(x[i]);
    acc = acc + v * v;
  }
  return acc;
}
DSK_FN float rms_finish(const DsRmsNormParams& p, uint32_t row, float* lanes) {
  const uint16_t* x = p.x + static_cast<uint64_t>(row) * p.width;
  float tail = 0.0f;
  for (uint32_t i = p.width & ~15u; i < p.width; ++i) {
    const float v = ds::bf16_value(x[i]);
    tail = tail + v * v;
  }
  return ds::rms_factor(ds::pinned_reduce(lanes, tail), p.width, p.eps);
}
DSK_FN void rms_store(const DsRmsNormParams& p, uint32_t row, uint32_t t, float factor) {
  const uint64_t at = static_cast<uint64_t>(row) * p.width;
  for (uint32_t e = t; e < p.width; e += kElementThreads) {
    p.out[at + e] = ds::bf16_bits(ds::rms_element(ds::bf16_value(p.x[at + e]), factor, p.weight[e]));
  }
}

// ---- hyper-connections ----

// `mhc::mixes`. Grid (rows), 1024 threads: lane group `g` (threads
// `16g .. 16g + 16`) is `dot(x, projection[g])` for `g < (2 + copies) ·
// copies` and `dot(x, x)` for the group after; thread 0 then runs the scalar
// part over the group results.
DSK_FN uint32_t mix_groups(uint32_t copies) { return (2 + copies) * copies + 1; }

DSK_FN float mhc_lane(const DsMhcMixParams& p, uint32_t row, uint32_t g, uint32_t lane) {
  const uint32_t len = p.copies * p.dim;
  const uint16_t* x = p.x + static_cast<uint64_t>(row) * len;
  const bool square = g + 1 == mix_groups(p.copies);
  const float* w = p.projection + static_cast<uint64_t>(square ? 0 : g) * len;
  float acc = 0.0f;
  for (uint32_t i = lane; i < (len & ~15u); i += 16) {
    const float v = ds::bf16_value(x[i]);
    acc = acc + v * (square ? v : w[i]);
  }
  return acc;
}
DSK_FN float mhc_group(const DsMhcMixParams& p, uint32_t row, uint32_t g, float* lanes) {
  const uint32_t len = p.copies * p.dim;
  const uint16_t* x = p.x + static_cast<uint64_t>(row) * len;
  const bool square = g + 1 == mix_groups(p.copies);
  const float* w = p.projection + static_cast<uint64_t>(square ? 0 : g) * len;
  float tail = 0.0f;
  for (uint32_t i = len & ~15u; i < len; ++i) {
    const float v = ds::bf16_value(x[i]);
    tail = tail + v * (square ? v : w[i]);
  }
  return ds::pinned_reduce(lanes, tail);
}
// `dots` holds the `mix_groups` results in group order.
DSK_FN void mhc_finish(const DsMhcMixParams& p, uint32_t row, const float* dots) {
  const uint32_t hc = p.copies, n = (2 + hc) * hc;
  const float rs = ds::rsqrt32(dots[n] / static_cast<float>(hc * p.dim) + p.norm_eps);
  float* out = p.out + static_cast<uint64_t>(row) * n;
  for (uint32_t j = 0; j < hc; ++j) {
    out[j] = ds::sigmoid32(dots[j] * rs * p.scale[0] + p.base[j]) + p.eps;
    out[hc + j] = 2.0f * ds::sigmoid32(dots[hc + j] * rs * p.scale[1] + p.base[hc + j]);
  }
  float comb[kMaxMixCopies * kMaxMixCopies];
  for (uint32_t j = 0; j < hc * hc; ++j) {
    comb[j] = dots[2 * hc + j] * rs * p.scale[2] + p.base[2 * hc + j];
  }
  // Row softmax, plus `eps`.
  for (uint32_t i = 0; i < hc; ++i) {
    float* r = comb + i * hc;
    float max = ds::f32_from_bits(0xff800000u);
    for (uint32_t j = 0; j < hc; ++j) max = ds::max32(max, r[j]);
    for (uint32_t j = 0; j < hc; ++j) r[j] = ds::exp32(r[j] - max);
    float sum = 0.0f;
    for (uint32_t j = 0; j < hc; ++j) sum = sum + r[j];
    for (uint32_t j = 0; j < hc; ++j) r[j] = r[j] / sum + p.eps;
  }
  for (uint32_t step = 0;; ++step) {
    // Columns.
    for (uint32_t k = 0; k < hc; ++k) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < hc; ++j) sum = sum + comb[j * hc + k];
      for (uint32_t j = 0; j < hc; ++j) comb[j * hc + k] = comb[j * hc + k] / (sum + p.eps);
    }
    if (step + 1 >= p.iters) break;
    // Rows.
    for (uint32_t i = 0; i < hc; ++i) {
      float* r = comb + i * hc;
      float sum = 0.0f;
      for (uint32_t j = 0; j < hc; ++j) sum = sum + r[j];
      for (uint32_t j = 0; j < hc; ++j) r[j] = r[j] / (sum + p.eps);
    }
  }
  for (uint32_t j = 0; j < hc * hc; ++j) out[2 * hc + j] = comb[j];
}

// `mhc::collapse`. Grid (ceil(dim / 256), rows). Thread: element `d`.
DSK_FN void collapse_thread(const DsCollapseParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t d = bx * kElementThreads + t;
  if (d >= p.dim) return;
  const float* pre = p.pre + static_cast<uint64_t>(row) * p.pre_stride;
  const uint64_t at = static_cast<uint64_t>(row) * p.copies * p.dim;
  float s = 0.0f;
  for (uint32_t j = 0; j < p.copies; ++j) {
    s = s + pre[j] * load_bf16(p.x, at + static_cast<uint64_t>(j) * p.dim + d);
  }
  store_bf16(p.out, static_cast<uint64_t>(row) * p.dim + d, s);
}

// `mhc::expand`. Grid (ceil(dim / 256), rows). Thread: element `d` of every
// copy, the old copies read before any is written (so `out` may be `x`).
DSK_FN void expand_thread(const DsExpandParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t d = bx * kElementThreads + t;
  if (d >= p.dim) return;
  const uint32_t hc = p.copies;
  const float* mix = p.mixes + static_cast<uint64_t>(row) * p.mix_stride;
  const uint64_t at = static_cast<uint64_t>(row) * hc * p.dim;
  float old[kMaxCopies];
  for (uint32_t i = 0; i < hc; ++i) old[i] = load_bf16(p.x, at + static_cast<uint64_t>(i) * p.dim + d);
  const float y = load_bf16(p.y, static_cast<uint64_t>(row) * p.dim + d);
  for (uint32_t j = 0; j < hc; ++j) {
    float s = 0.0f;
    for (uint32_t i = 0; i < hc; ++i) s = s + mix[2 * hc + i * hc + j] * old[i];
    store_bf16(p.out, at + static_cast<uint64_t>(j) * p.dim + d, mix[hc + j] * y + s);
  }
}

// `forward::stream_mean`. Grid (ceil(dim / 256), rows). Thread: element `d`.
DSK_FN void stream_mean_thread(const DsStreamMeanParams& p, uint32_t bx, uint32_t row,
                               uint32_t t) {
  const uint32_t d = bx * kElementThreads + t;
  if (d >= p.dim) return;
  const uint64_t at = static_cast<uint64_t>(row) * p.copies * p.dim;
  float s = 0.0f;
  for (uint32_t j = 0; j < p.copies; ++j) s = s + load_bf16(p.x, at + static_cast<uint64_t>(j) * p.dim + d);
  store_bf16(p.out, static_cast<uint64_t>(row) * p.out_stride + d, s / static_cast<float>(p.copies));
}

// ---- plain products in the pinned order ----

// `linear::linear_f32` / `linear_bf16`, a row at a time. Grid
// (ceil(n / 64), rows), 256 threads: lane group `g` of block `bx` carries
// outputs `(16 bx + g) · 4 ..+ 4`, each lane thread one chain per output.
DSK_FN uint64_t linear_x_row(const DsLinearParams& p, uint32_t row) {
  return static_cast<uint64_t>(p.x_rows != nullptr ? p.x_rows[row] : row) * p.k;
}
DSK_FN void linear_lane(const DsLinearParams& p, uint32_t row, uint32_t n0, uint32_t lane,
                        float* acc) {
  const uint64_t xa = linear_x_row(p, row);
  for (uint32_t u = 0; u < kLinearChains; ++u) acc[u] = 0.0f;
  if (n0 >= p.n) return;
  const uint32_t live = p.n - n0 < kLinearChains ? p.n - n0 : kLinearChains;
  for (uint32_t i = lane; i < (p.k & ~15u); i += 16) {
    const float x = load_dt(p.x, p.x_dtype, xa + i);
    for (uint32_t u = 0; u < live; ++u) {
      acc[u] = acc[u] + x * load_dt(p.w, p.w_dtype, static_cast<uint64_t>(n0 + u) * p.k + i);
    }
  }
}
// The tail of one output, then the reduction, the addend and the store.
DSK_FN void linear_finish(const DsLinearParams& p, uint32_t row, uint32_t n, float* lanes) {
  const uint64_t xa = linear_x_row(p, row);
  float tail = 0.0f;
  for (uint32_t i = p.k & ~15u; i < p.k; ++i) {
    tail = tail + load_dt(p.x, p.x_dtype, xa + i) *
                      load_dt(p.w, p.w_dtype, static_cast<uint64_t>(n) * p.k + i);
  }
  float v = ds::pinned_reduce(lanes, tail);
  const uint64_t at = static_cast<uint64_t>(row) * p.n + n;
  if (p.addend != nullptr) v = p.addend[at] + v;
  if (p.out_dtype == DS_DT_F32) {
    static_cast<float*>(p.out)[at] = v;
  } else {
    store_bf16(static_cast<uint16_t*>(p.out), at, v);
  }
}

// The same products weight row by weight row (`ds_head.cu`): grid
// (ceil(n / 16), ceil(rows / 8)), 256 threads; lane group `g` of block `bx`
// is weight row `16 bx + g`, each lane thread one chain per row of the
// block's 8 rows, so a weight row is read once for all of them.
DSK_FN void head_lane(const DsLinearParams& p, uint32_t row0, uint32_t n, uint32_t lane,
                      float* acc) {
  for (uint32_t u = 0; u < kHeadChains; ++u) acc[u] = 0.0f;
  if (n >= p.n) return;
  const uint32_t live = p.rows - row0 < kHeadChains ? p.rows - row0 : kHeadChains;
  uint64_t xa[kHeadChains];
  for (uint32_t u = 0; u < live; ++u) xa[u] = linear_x_row(p, row0 + u);
  for (uint32_t i = lane; i < (p.k & ~15u); i += 16) {
    const float w = load_dt(p.w, p.w_dtype, static_cast<uint64_t>(n) * p.k + i);
    for (uint32_t u = 0; u < live; ++u) acc[u] = acc[u] + load_dt(p.x, p.x_dtype, xa[u] + i) * w;
  }
}

// ---- the ratio-2 compressor's pooling ----

// Grid (ceil(width / 256), entries). Thread: one channel of one entry.
DSK_FN void pool_thread(const DsPoolParams& p, uint32_t bx, uint32_t entry, uint32_t t) {
  const uint32_t ch = bx * kElementThreads + t;
  if (ch >= p.width) return;
  const uint64_t a = static_cast<uint64_t>(p.first[entry]) * p.width + ch;
  const uint64_t b = static_cast<uint64_t>(p.second[entry]) * p.width + ch;
  const float max = ds::max32(ds::max32(ds::f32_from_bits(0xff800000u), p.score[a]), p.score[b]);
  const float ea = ds::exp32(p.score[a] - max), eb = ds::exp32(p.score[b] - max);
  const float sum = (0.0f + ea) + eb;
  float pooled = 0.0f;
  pooled = pooled + p.kv[a] * (ea / sum);
  pooled = pooled + p.kv[b] * (eb / sum);
  store_bf16(p.out, static_cast<uint64_t>(entry) * p.width + ch, pooled);
}

// ---- router ----

// `moe::route` after the logits. Grid (rows), 128 threads: thread `t`
// scores experts `t, t + 128, …` into shared memory; thread 0 selects.
DSK_FN void route_score(const DsRouteParams& p, uint32_t row, uint32_t e, float* score,
                        float* biased) {
  const float s = sqrtf(ds::softplus32(p.logits[static_cast<uint64_t>(row) * p.experts + e]));
  score[e] = s;
  biased[e] = s + p.bias[e];
}
DSK_FN void route_select(const DsRouteParams& p, uint32_t row, const float* score,
                         const float* biased) {
  uint32_t ids[kMaxActive];
  float w[kMaxActive];
  // Round `r` takes the best expert ranked after round `r - 1`'s: the order
  // is total and strict, so that is the best not yet taken.
  for (uint32_t r = 0; r < p.active; ++r) {
    uint32_t best = p.experts;
    for (uint32_t e = 0; e < p.experts; ++e) {
      if (r > 0 && !ds::topk_before(biased[ids[r - 1]], ids[r - 1], biased[e], e)) continue;
      if (best == p.experts || ds::topk_before(biased[e], e, biased[best], best)) best = e;
    }
    ids[r] = best;
    w[r] = score[best];
  }
  if (p.normalize != 0) {
    float sum = 0.0f;
    for (uint32_t r = 0; r < p.active; ++r) sum = sum + w[r];
    const float den = sum + ds::f32_from_bits(kRouteNormFloorBits);
    for (uint32_t r = 0; r < p.active; ++r) w[r] = w[r] / den;
  }
  for (uint32_t r = 0; r < p.active; ++r) w[r] = w[r] * p.route_scale;
  // Ascending expert id.
  for (uint32_t r = 1; r < p.active; ++r) {
    const uint32_t id = ids[r];
    const float weight = w[r];
    uint32_t at = r;
    for (; at > 0 && ids[at - 1] > id; --at) {
      ids[at] = ids[at - 1];
      w[at] = w[at - 1];
    }
    ids[at] = id;
    w[at] = weight;
  }
  for (uint32_t r = 0; r < p.active; ++r) {
    p.ids[static_cast<uint64_t>(row) * p.active + r] = ids[r];
    p.weights[static_cast<uint64_t>(row) * p.active + r] = w[r];
  }
}

// ---- expert layout ----

// Two launches over grid (ceil(groups / 128)), a thread per group. The first
// counts each group's pairs into `scratch[groups]`; the second derives the
// group's first row from the counts before it and places its pairs in (row,
// slot) order. A pair whose id is not below `groups`, or whose row is not
// below `layout_rows`, is not placed.
DSK_FN void place_count_thread(const DsPlaceParams& p, uint32_t bx, uint32_t t) {
  const uint32_t g = bx * kBlockThreads + t;
  if (g >= p.groups) return;
  const uint64_t pairs = static_cast<uint64_t>(p.rows) * p.active;
  uint32_t count = 0;
  for (uint64_t i = 0; i < pairs; ++i) count += p.ids[i] == g ? 1u : 0u;
  p.scratch[g] = count;
}
DSK_FN uint32_t align_up(uint32_t x, uint32_t a) {
  const uint32_t over = x % a;
  return over == 0 ? x : x + (a - over);
}
DSK_FN void place_thread(const DsPlaceParams& p, uint32_t bx, uint32_t t) {
  const uint32_t g = bx * kBlockThreads + t;
  if (g >= p.groups) return;
  uint32_t end = 0;
  for (uint32_t h = 0; h < g; ++h) end = align_up(end, p.row_align) + p.scratch[h];
  uint32_t at = align_up(end, p.row_align);
  const uint64_t pairs = static_cast<uint64_t>(p.rows) * p.active;
  for (uint64_t i = 0; i < pairs; ++i) {
    if (p.ids[i] != g) continue;
    if (at < p.layout_rows) {
      p.row_of[i] = at;
      p.row_weight[at] = p.weights[i];
    }
    ++at;
  }
  p.group_end[g] = at;
}

// ---- feed-forward ----

// `moe::swiglu_hidden` then `quant::act_quant`. Grid (ceil(width / 32 /
// 128), rows). Thread: one 32-wide block.
DSK_FN void swiglu_thread(const DsSwigluParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t u = bx * kBlockThreads + t;
  const uint32_t blocks = p.width / 32;
  if (u >= blocks) return;
  const uint64_t in = static_cast<uint64_t>(row) * p.in_stride + u * 32;
  const bool weighted = p.weight != nullptr;
  const float weight = weighted ? p.weight[row] : 0.0f;
  float x[32];
  for (int j = 0; j < 32; ++j) {
    x[j] = ds::swiglu_hidden(ds::bf16_value(p.gate[in + j]), ds::bf16_value(p.up[in + j]), p.limit,
                             weighted, weight);
  }
  uint8_t codes[32];
  const uint8_t scale = ds::act_quant_block(x, codes);
  const uint64_t out = static_cast<uint64_t>(row) * p.width + u * 32;
  for (int j = 0; j < 32; ++j) p.codes[out + j] = codes[j];
  p.scales[static_cast<uint64_t>(row) * blocks + u] = scale;
}

// `forward::moe_forward`'s sum. Grid (ceil(dim / 256), rows). Thread: element `d`.
DSK_FN void combine_thread(const DsCombineParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t d = bx * kElementThreads + t;
  if (d >= p.dim) return;
  const uint32_t* row_of = p.row_of + static_cast<uint64_t>(row) * p.active;
  float s = 0.0f;
  for (uint32_t j = 0; j < p.active; ++j) {
    s = s + load_bf16(p.routed, static_cast<uint64_t>(row_of[j]) * p.dim + d);
  }
  const uint64_t at = static_cast<uint64_t>(row) * p.dim + d;
  store_bf16(p.out, at, s + load_bf16(p.shared, at));
}

// ---- Engram ----

// Grid (ceil(width / 32 / 128), ids_per_row, rows). Thread: 32 codes and
// their scale byte of one table row.
DSK_FN void engram_gather_thread(const DsEngramGatherParams& p, uint32_t bx, uint32_t slot,
                                 uint32_t row, uint32_t t) {
  const uint32_t u = bx * kBlockThreads + t;
  const uint32_t blocks = p.width / 32;
  if (u >= blocks) return;
  const uint64_t pair = static_cast<uint64_t>(row) * p.ids_per_row + slot;
  const uint64_t id = p.ids[pair];
  const bool ok = id < p.table_rows;
  if (!ok && u == 0) DSK_STATUS_OR(p.status, DS_STATUS_BAD_INDEX);
  for (int j = 0; j < 32; ++j) {
    p.out_codes[pair * p.width + u * 32 + j] = ok ? p.codes[id * p.width + u * 32 + j] : 0;
  }
  p.out_scales[pair * blocks + u] = ok ? p.scales[id * blocks + u] : 0;
}

// `engram::inject` after its projection. Grid (copies, rows), 256 threads.
// Lane groups 0, 1 and 2 (threads 0..47) are `dot(x, x)`, `dot(key, key)`
// and the pinned sum of `(x · w) · key`; then every thread writes elements
// `t, t + 256, …` of its copy.
DSK_FN void inject_inputs(const DsEngramInjectParams& p, uint32_t row, uint32_t copy,
                          const uint16_t** x, const uint16_t** key, const float** w) {
  *x = p.x + (static_cast<uint64_t>(row) * p.copies + copy) * p.dim;
  *key = p.kv + (static_cast<uint64_t>(row) * (p.copies + 1) + copy) * p.dim;
  *w = p.gate_weight + static_cast<uint64_t>(copy) * p.dim;
}
DSK_FN float inject_term(const uint16_t* x, const uint16_t* key, const float* w, uint32_t g,
                         uint32_t i) {
  const float xv = ds::bf16_value(x[i]), kv = ds::bf16_value(key[i]);
  return g == 0 ? xv * xv : g == 1 ? kv * kv : xv * w[i] * kv;
}
DSK_FN float inject_lane(const DsEngramInjectParams& p, uint32_t row, uint32_t copy, uint32_t g,
                         uint32_t lane) {
  const uint16_t *x, *key;
  const float* w;
  inject_inputs(p, row, copy, &x, &key, &w);
  float acc = 0.0f;
  for (uint32_t i = lane; i < (p.dim & ~15u); i += 16) acc = acc + inject_term(x, key, w, g, i);
  return acc;
}
DSK_FN float inject_group(const DsEngramInjectParams& p, uint32_t row, uint32_t copy, uint32_t g,
                          float* lanes) {
  const uint16_t *x, *key;
  const float* w;
  inject_inputs(p, row, copy, &x, &key, &w);
  float tail = 0.0f;
  for (uint32_t i = p.dim & ~15u; i < p.dim; ++i) tail = tail + inject_term(x, key, w, g, i);
  return ds::pinned_reduce(lanes, tail);
}
// The gate from the three group results.
DSK_FN float inject_gate(const DsEngramInjectParams& p, const float* sums) {
  const float dim = static_cast<float>(p.dim);
  const float rstd =
      ds::rsqrt32(sums[0] / dim + p.norm_eps) * ds::rsqrt32(sums[1] / dim + p.norm_eps);
  const float scale = static_cast<float>(1.0 / sqrt(static_cast<double>(p.dim)));
  const float dot = sums[2] * rstd * scale;
  const float root = sqrtf(ds::max32(ds::abs32(dot), ds::f32_from_bits(kGateFloorBits)));
  return ds::sigmoid32(ds::copysign32(root, dot));
}
DSK_FN void inject_store(const DsEngramInjectParams& p, uint32_t row, uint32_t copy, uint32_t t,
                         float gate) {
  const uint64_t at = (static_cast<uint64_t>(row) * p.copies + copy) * p.dim;
  const uint64_t value = (static_cast<uint64_t>(row) * (p.copies + 1) + p.copies) * p.dim;
  for (uint32_t d = t; d < p.dim; d += kElementThreads) {
    store_bf16(p.out, at + d, load_bf16(p.x, at + d) + gate * load_bf16(p.kv, value + d));
  }
}

// ---- quantised GEMMs, scalar integer form ----

// `linear::lin8` after its quantiser. Grid (ceil(n / 256), rows). Thread:
// one output of one row, blocks ascending.
DSK_FN void lin8_thread(const DsLin8Params& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t n = bx * kElementThreads + t;
  if (n >= p.n) return;
  const uint32_t kb = p.k / 32;
  const uint8_t* a = p.codes + static_cast<uint64_t>(row) * p.k;
  const uint8_t* as = p.act_scales + static_cast<uint64_t>(row) * kb;
  const uint8_t* w = p.w_codes + static_cast<uint64_t>(n) * p.k;
  const uint8_t* ws = p.w_scales + static_cast<uint64_t>(n / 32) * kb;
  float acc = 0.0f;
  for (uint32_t b = 0; b < kb; ++b) {
    acc = ds::gemm_accumulate(acc, ds::fp8_fp8_block_dot(a + b * 32, w + b * 32),
                              ds::act_scale_decode(as[b]), ds::e8m0_decode(ws[b]));
  }
  store_bf16(p.out, static_cast<uint64_t>(row) * p.n + n, acc);
}

// `linear::lin4` after its quantiser, over the expert-major layout. Grid
// (ceil(n / 256), layout_rows). Thread: one output of one layout row, with
// the weight of the group whose run holds the row; a padding row writes
// nothing.
DSK_FN void lin4_thread(const DsLin4Params& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t n = bx * kElementThreads + t;
  if (n >= p.n) return;
  // The first group whose run ends after the row.
  uint32_t lo = 0, hi = p.groups;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (p.group_end[mid] > row) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  if (lo == p.groups) return;
  if (lo > 0 && row < align_up(p.group_end[lo - 1], p.row_align)) return;
  const uint32_t kb = p.k / 32;
  const uint8_t* a = p.codes + static_cast<uint64_t>(row) * p.k;
  const uint8_t* as = p.act_scales + static_cast<uint64_t>(row) * kb;
  const uint64_t wrow = static_cast<uint64_t>(lo) * p.n + n;
  const uint8_t* w = p.w_packed + wrow * (p.k / 2);
  const uint8_t* ws = p.w_scales + wrow * kb;
  float acc = 0.0f;
  for (uint32_t b = 0; b < kb; ++b) {
    acc = ds::gemm_accumulate(acc, ds::fp8_fp4_block_dot(a + b * 32, w, b * 32),
                              ds::act_scale_decode(as[b]), ds::e8m0_decode(ws[b]));
  }
  store_bf16(p.out, static_cast<uint64_t>(row) * p.n + n, acc);
}

// The grouped projection (`wo_a`): grid (ceil(rank / 64), groups, rows), 256
// threads; lane group `g` of block `bx` carries outputs `(16 bx + g) · 4 ..+
// 4` of the block's group. Both operands are dequantised on load.
DSK_FN float grouped_x(const DsGroupedDotParams& p, uint32_t row, uint32_t group, uint32_t i) {
  const uint64_t at = static_cast<uint64_t>(row) * p.groups * p.k + static_cast<uint64_t>(group) * p.k + i;
  return ds::dequant_e4m3(p.codes[at], ds::act_scale_decode(p.act_scales[at / 32]));
}
DSK_FN float grouped_w(const DsGroupedDotParams& p, uint32_t wrow, uint32_t i) {
  const float s = ds::e8m0_decode(p.w_scales[static_cast<uint64_t>(wrow / 32) * (p.k / 32) + i / 32]);
  return ds::bf16_round(ds::e4m3_decode(p.w_codes[static_cast<uint64_t>(wrow) * p.k + i]) * s);
}
DSK_FN void grouped_lane(const DsGroupedDotParams& p, uint32_t row, uint32_t group, uint32_t i0,
                         uint32_t lane, float* acc) {
  for (uint32_t u = 0; u < kLinearChains; ++u) acc[u] = 0.0f;
  if (i0 >= p.rank) return;
  const uint32_t live = p.rank - i0 < kLinearChains ? p.rank - i0 : kLinearChains;
  for (uint32_t i = lane; i < (p.k & ~15u); i += 16) {
    const float x = grouped_x(p, row, group, i);
    for (uint32_t u = 0; u < live; ++u) {
      acc[u] = acc[u] + x * grouped_w(p, group * p.rank + i0 + u, i);
    }
  }
}
// `k` is a multiple of 32, so there is no tail.
DSK_FN void grouped_finish(const DsGroupedDotParams& p, uint32_t row, uint32_t group, uint32_t i,
                           float* lanes) {
  store_bf16(p.out, static_cast<uint64_t>(row) * p.groups * p.rank + group * p.rank + i,
             ds::pinned_reduce(lanes, 0.0f));
}

}  // namespace dsk
