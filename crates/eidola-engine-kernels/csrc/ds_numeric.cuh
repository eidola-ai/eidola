// The numerics of the DeepSeek-V4.1 forward: every rounding point, scale
// rule, exact block product, pinned summation order and transcendental, each
// bit-identical to its definition in `eidola-engine-model`'s
// `src/deepseek/{numeric,quant,linear,rope,attention}.rs` and `src/tensor.rs`
// (named beside each function).
//
// The arithmetic is plain C++: IEEE `+ - * /`, `sqrt`, `floor`, integer
// conversions and bit casts, so one source serves a device kernel and the
// host check (`oracle/ds_numeric_host.cc`). It is correct only when the
// compiler neither fuses a multiply-add nor reassociates: device translation
// units that include it build under the `exact` profile of `kernels.json`,
// and a host build passes `-ffp-contract=off` and no fast-math option.
//
// Values outside what the forward produces keep the reference's behaviour
// where it is defined (NaN and infinite inputs of the casts and of
// `bf16_round`, signed zeros); `det_ln` and `det_sin_cos` have the
// reference's preconditions.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define DS_FN __host__ __device__ __forceinline__
#else
#define DS_FN inline
#endif

namespace ds {

// ---- bit casts ----

DS_FN uint32_t f32_bits(float x) {
#if defined(__CUDACC__)
  return __float_as_uint(x);
#else
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return b;
#endif
}
DS_FN float f32_from_bits(uint32_t b) {
#if defined(__CUDACC__)
  return __uint_as_float(b);
#else
  float x;
  std::memcpy(&x, &b, 4);
  return x;
#endif
}
DS_FN uint64_t f64_bits(double x) {
#if defined(__CUDACC__)
  return static_cast<uint64_t>(__double_as_longlong(x));
#else
  uint64_t b;
  std::memcpy(&b, &x, 8);
  return b;
#endif
}
DS_FN double f64_from_bits(uint64_t b) {
#if defined(__CUDACC__)
  return __longlong_as_double(static_cast<long long>(b));
#else
  double x;
  std::memcpy(&x, &b, 8);
  return x;
#endif
}

constexpr uint32_t kF32Nan = 0x7fc00000u;  // Rust's `f32::NAN`
constexpr uint32_t kSign32 = 0x80000000u;
constexpr uint64_t kSign64 = 0x8000000000000000ull;

DS_FN bool is_nan(float x) { return (f32_bits(x) & ~kSign32) > 0x7f800000u; }
DS_FN bool is_nan(double x) { return (f64_bits(x) & ~kSign64) > 0x7ff0000000000000ull; }
DS_FN float abs32(float x) { return f32_from_bits(f32_bits(x) & ~kSign32); }
DS_FN double abs64(double x) { return f64_from_bits(f64_bits(x) & ~kSign64); }
// `mag.copysign(sign)`.
DS_FN float copysign32(float mag, float sign) {
  return f32_from_bits((f32_bits(mag) & ~kSign32) | (f32_bits(sign) & kSign32));
}
DS_FN double copysign64(double mag, double sign) {
  return f64_from_bits((f64_bits(mag) & ~kSign64) | (f64_bits(sign) & kSign64));
}
// Rust's `f32::max`: a NaN operand loses.
DS_FN float max32(float a, float b) {
  if (is_nan(a)) return b;
  if (is_nan(b)) return a;
  return a > b ? a : b;
}
// Rust's `f32::min`.
DS_FN float min32(float a, float b) {
  if (is_nan(a)) return b;
  if (is_nan(b)) return a;
  return a < b ? a : b;
}
// Rust's `f32::clamp(lo, hi)`: NaN stays NaN.
DS_FN float clamp32(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// ---- BF16 ----

// `numeric::bf16_round`: nearest BF16, ties to even, widened back. NaN is
// returned unchanged; a value above the largest BF16 becomes infinite.
DS_FN float bf16_round(float x) {
  if (is_nan(x)) return x;
  const uint32_t b = f32_bits(x);
  return f32_from_bits((b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u);
}
// The 16 stored bits of a BF16-valued f32 (after `bf16_round`), and back.
DS_FN uint16_t bf16_bits(float x) { return static_cast<uint16_t>(f32_bits(x) >> 16); }
DS_FN float bf16_value(uint16_t b) { return f32_from_bits(static_cast<uint32_t>(b) << 16); }

// ---- storage formats: decodes (`crate::numeric`) ----

// OCP FP8 E4M3 (`fp8_e4m3_to_f32`): bias 7, no infinities, `S.1111.111` NaN.
DS_FN float e4m3_decode(uint8_t code) {
  const uint32_t exp = (code >> 3) & 0x0fu;
  const uint32_t man = code & 0x07u;
  if (exp == 0x0fu && man == 0x07u) return f32_from_bits(kF32Nan);
  const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
  // Subnormal: man · 2^-9. Normal: (8 + man) · 2^(exp - 10).
  const float mag = exp == 0 ? static_cast<float>(man) * f32_from_bits((127u - 9u) << 23)
                             : static_cast<float>(8u + man) * f32_from_bits((exp + 117u) << 23);
  return f32_from_bits(f32_bits(mag) | sign);
}
DS_FN bool e4m3_is_nan(uint8_t code) { return (code & 0x7fu) == 0x7fu; }
// 512 times an E4M3 value: an integer of magnitude at most 229,376. The code
// must not be NaN.
DS_FN int32_t e4m3_times_512(uint8_t code) {
  const int32_t exp = (code >> 3) & 0x0f;
  const int32_t man = code & 0x07;
  const int32_t mag = exp == 0 ? man : (8 + man) << (exp - 1);
  return (code & 0x80u) ? -mag : mag;
}

// OCP FP4 E2M1 (`E2M1_VALUES`), bit 3 the sign.
DS_FN float e2m1_decode(uint8_t code) {
  const uint32_t c = code & 0x07u;
  // 0, 0.5, then (2 + m) · 2^(e - 1) for exponent field e = 1..3.
  const float mag =
      c < 2 ? static_cast<float>(c) * 0.5f
            : static_cast<float>(2u + (c & 1u)) * f32_from_bits(((c >> 1) + 125u) << 23);
  return f32_from_bits(f32_bits(mag) | (static_cast<uint32_t>(code & 0x08u) << 28));
}
// Twice an E2M1 value: an integer in -12..=12.
DS_FN int32_t e2m1_times_2(uint8_t code) {
  const int32_t c = code & 0x07;
  const int32_t mag = c < 2 ? c : (2 + (c & 1)) << ((c >> 1) - 1);
  return (code & 0x08u) ? -mag : mag;
}
// Element `i` of a packed FP4 row: element `2j` is the low nibble of byte `j`.
DS_FN uint8_t fp4_nibble(const uint8_t* packed, uint32_t i) {
  return static_cast<uint8_t>((packed[i >> 1] >> ((i & 1u) * 4u)) & 0x0fu);
}

// OCP E8M0 (`e8m0_to_f32`): 2^(code - 127); 255 is NaN.
DS_FN float e8m0_decode(uint8_t code) {
  if (code == 0xffu) return f32_from_bits(kF32Nan);
  // 2^-127 is an f32 subnormal.
  return f32_from_bits(code == 0 ? (1u << 22) : (static_cast<uint32_t>(code) << 23));
}

// ---- casts to FP8 and FP4 ----

// Nearest code of a small float format to the magnitude `abs_bits` (an f32's
// bits without its sign, not NaN), ties to the even code, saturating at
// `max_code` from `sat_bits` up. The format has `mbits` mantissa bits and its
// smallest normal is 2^emin; below it the codes step by 2^(emin - mbits).
DS_FN uint32_t rne_code(uint32_t abs_bits, uint32_t mbits, int32_t emin, uint32_t max_code,
                        uint32_t sat_bits) {
  if (abs_bits >= sat_bits) return max_code;
  const uint32_t field = abs_bits >> 23;
  if (field == 0) return 0;  // an f32 subnormal is far below half a step
  const int32_t e = static_cast<int32_t>(field) - 127;
  const uint32_t s = (abs_bits & 0x7fffffu) | 0x800000u;
  uint32_t shift = 23u - mbits;
  uint32_t code;
  if (e >= emin) {
    code = (s >> shift) + (static_cast<uint32_t>(e - emin) << mbits);
  } else {
    shift += static_cast<uint32_t>(emin - e);
    if (shift >= 25u) return 0;  // under half a step
    code = s >> shift;
  }
  const uint32_t rem = s & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1u);
  return code + ((rem > half || (rem == half && (code & 1u))) ? 1u : 0u);
}

// f32 → E4M3 code: `numeric::round_e4m3` as the stored byte (saturating at
// ±448; a NaN becomes a zero of its sign, as the reference's table search
// leaves it).
DS_FN uint8_t e4m3_encode(float v) {
  const uint32_t b = f32_bits(v);
  const uint32_t sign = (b >> 24) & 0x80u;
  if (is_nan(v)) return static_cast<uint8_t>(sign);
  return static_cast<uint8_t>(sign | rne_code(b & ~kSign32, 3, -6, 0x7eu, 0x43e00000u));
}
// f32 → E2M1 code: `numeric::round_e2m1` as the stored nibble (saturating at
// ±6).
DS_FN uint8_t e2m1_encode(float v) {
  const uint32_t b = f32_bits(v);
  const uint32_t sign = (b >> 28) & 0x08u;
  if (is_nan(v)) return static_cast<uint8_t>(sign);
  return static_cast<uint8_t>(sign | rne_code(b & ~kSign32, 1, 0, 0x07u, 0x40c00000u));
}
DS_FN float round_e4m3(float v) { return e4m3_decode(e4m3_encode(v)); }
DS_FN float round_e2m1(float v) { return e2m1_decode(e2m1_encode(v)); }

// ---- activation scales (`numeric::round_scale`, `quant`) ----

// `2^ceil(log2(amax · inv))` from the f32 product's bits: its exponent, plus
// one when its mantissa is non-zero.
DS_FN float round_scale(float amax, float inv) {
  const uint32_t bits = f32_bits(amax * inv);
  const uint32_t e = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0 ? 1u : 0u);
  return f32_from_bits(e << 23);
}
// The stored byte of a power-of-two activation scale `round_scale` returned
// (its exponent field), and the scale of a stored byte. Finite activations
// give 1..=254, which is E8M0; 255 is the infinite scale of a block holding
// an infinite activation, where a weight's E8M0 (`e8m0_decode`) has NaN.
DS_FN uint8_t act_scale_byte(float scale) { return static_cast<uint8_t>(f32_bits(scale) >> 23); }
DS_FN float act_scale_decode(uint8_t byte) {
  return f32_from_bits(static_cast<uint32_t>(byte) << 23);
}
// The reciprocal of a stored power-of-two activation scale (byte 1..=255; 0
// for the infinite scale). Multiplying by it is dividing by the scale: both
// round the same real number once, 2^-127 being an exact f32.
DS_FN float act_scale_recip(uint8_t byte) {
  if (byte == 0xffu) return 0.0f;
  return f32_from_bits(byte == 0xfeu ? (1u << 22) : ((254u - byte) << 23));
}

constexpr uint32_t kInv448Bits = 0x3b124925u;   // (1.0 / 448.0) as f32
constexpr uint32_t kInv6Bits = 0x3e2aaaabu;     // (1.0 / 6.0) as f32
constexpr uint32_t kFp8FloorBits = 0x38d1b717u; // 1e-4 as f32
constexpr uint32_t kFp4Pow2FloorBits = 0x01c00000u;  // 6 · 2^-126
constexpr uint32_t kFp4E4m3FloorBits = 0x3c400000u;  // 6 · 2^-9

// A block's largest magnitude: `fold(0, |m, v| m.max(v.abs()))`. A NaN
// element is skipped.
DS_FN float amax_step(float m, float v) { return max32(m, abs32(v)); }

// The FP8 block scale (32 elements): `2^ceil(log2(max(amax, 1e-4) / 448))`.
DS_FN float fp8_scale(float amax) {
  return round_scale(max32(amax, f32_from_bits(kFp8FloorBits)), f32_from_bits(kInv448Bits));
}
// The power-of-two FP4 block scale (32 elements; index keys and queries):
// `2^ceil(log2(max(amax, 6·2^-126) / 6))`.
DS_FN float fp4_pow2_scale(float amax) {
  return round_scale(max32(amax, f32_from_bits(kFp4Pow2FloorBits)), f32_from_bits(kInv6Bits));
}
// The E4M3 FP4 block scale (16 elements; the global KV):
// `e4m3(max(amax, 6·2^-9) / 6)`, as its code. Never zero and never negative.
DS_FN uint8_t fp4_e4m3_scale_code(float amax) {
  return e4m3_encode(max32(amax, f32_from_bits(kFp4E4m3FloorBits)) / 6.0f);
}

// One element of a quantiser: the stored code of `x / scale`, for a
// power-of-two scale given as its reciprocal (`act_scale_recip`), and for an
// E4M3 scale by an IEEE division.
DS_FN uint8_t quant_e4m3_pow2(float x, float recip) { return e4m3_encode(x * recip); }
DS_FN uint8_t quant_e2m1_pow2(float x, float recip) { return e2m1_encode(x * recip); }
DS_FN uint8_t quant_e2m1(float x, float scale) { return e2m1_encode(x / scale); }
// What a fake quantisation leaves in place of the element:
// `bf16(code · scale)`. The product of a code and its scale is a BF16 value
// already whenever it is a normal f32, so a consumer of stored codes and
// scales reads the same value with this.
DS_FN float dequant_e4m3(uint8_t code, float scale) {
  return bf16_round(e4m3_decode(code) * scale);
}
DS_FN float dequant_e2m1(uint8_t code, float scale) {
  return bf16_round(e2m1_decode(code) * scale);
}

// `quant::act_quant` over one 32-wide block: codes and the scale byte.
DS_FN uint8_t act_quant_block(const float* x, uint8_t* codes) {
  float amax = 0.0f;
  for (int j = 0; j < 32; ++j) amax = amax_step(amax, x[j]);
  const uint8_t byte = act_scale_byte(fp8_scale(amax));
  const float r = act_scale_recip(byte);
  for (int j = 0; j < 32; ++j) codes[j] = quant_e4m3_pow2(x[j], r);
  return byte;
}
// `quant::fake_quant_fp4_pow2` over one 32-wide block: unpacked E2M1 codes
// (one per byte) and the scale byte.
DS_FN uint8_t fp4_pow2_quant_block(const float* x, uint8_t* codes) {
  float amax = 0.0f;
  for (int j = 0; j < 32; ++j) amax = amax_step(amax, x[j]);
  const uint8_t byte = act_scale_byte(fp4_pow2_scale(amax));
  const float r = act_scale_recip(byte);
  for (int j = 0; j < 32; ++j) codes[j] = quant_e2m1_pow2(x[j], r);
  return byte;
}
// `quant::fake_quant_fp4_e4m3` over one 16-wide block: unpacked E2M1 codes
// and the E4M3 scale code.
DS_FN uint8_t fp4_e4m3_quant_block(const float* x, uint8_t* codes) {
  float amax = 0.0f;
  for (int j = 0; j < 16; ++j) amax = amax_step(amax, x[j]);
  const uint8_t sc = fp4_e4m3_scale_code(amax);
  const float s = e4m3_decode(sc);
  for (int j = 0; j < 16; ++j) codes[j] = quant_e2m1(x[j], s);
  return sc;
}

// ---- exact block products (`linear`) ----

// The dot product of two 32-wide blocks of E4M3 codes, exact, rounded once to
// f32 (`fp8_block_dot`). In units of 2^-18 every term is an integer below
// 2^36 and the sum below 2^41. No code may be NaN. A zero sum is +0.
DS_FN float fp8_fp8_block_dot(const uint8_t* a, const uint8_t* w) {
  int64_t sum = 0;
  for (int j = 0; j < 32; ++j) {
    sum += static_cast<int64_t>(e4m3_times_512(a[j])) * static_cast<int64_t>(e4m3_times_512(w[j]));
  }
  return static_cast<float>(static_cast<double>(sum) * 3.814697265625e-06 /* 2^-18 */);
}
// The dot product of a 32-wide block of E4M3 codes with 32 E2M1 elements
// (packed, element 0 at nibble `first`), exact, rounded once to f32 (`lin4`).
// In units of 2^-10 the sum is an integer below 2^27.
DS_FN float fp8_fp4_block_dot(const uint8_t* a, const uint8_t* w_packed, uint32_t first) {
  int32_t sum = 0;
  for (uint32_t j = 0; j < 32; ++j) {
    sum += e4m3_times_512(a[j]) * e2m1_times_2(fp4_nibble(w_packed, first + j));
  }
  return static_cast<float>(sum) * 0.0009765625f /* 2^-10 */;
}
// One block's step of a quantised GEMM's accumulation, blocks ascending:
// `acc + (dot · activation scale) · weight scale`.
DS_FN float gemm_accumulate(float acc, float dot, float act_scale, float w_scale) {
  return acc + dot * act_scale * w_scale;
}

// ---- pinned orders (`tensor::dot`, `numeric::pinned_sum`) ----

// 16 interleaved partial sums over the full 16-wide chunks, a halving tree
// over them, then the tail left to right, added last.
DS_FN float pinned_dot(const float* a, const float* b, uint32_t n) {
  float acc[16];
  for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
  const uint32_t full = n & ~15u;
  for (uint32_t i = 0; i < full; i += 16) {
    for (uint32_t j = 0; j < 16; ++j) acc[j] = acc[j] + a[i + j] * b[i + j];
  }
  for (uint32_t w = 8; w > 0; w >>= 1) {
    for (uint32_t j = 0; j < w; ++j) acc[j] = acc[j] + acc[j + w];
  }
  float tail = 0.0f;
  for (uint32_t i = full; i < n; ++i) tail = tail + a[i] * b[i];
  return acc[0] + tail;
}
DS_FN float pinned_sum(const float* x, uint32_t n) {
  float acc[16];
  for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
  const uint32_t full = n & ~15u;
  for (uint32_t i = 0; i < full; i += 16) {
    for (uint32_t j = 0; j < 16; ++j) acc[j] = acc[j] + x[i + j];
  }
  for (uint32_t w = 8; w > 0; w >>= 1) {
    for (uint32_t j = 0; j < w; ++j) acc[j] = acc[j] + acc[j + w];
  }
  float tail = 0.0f;
  for (uint32_t i = full; i < n; ++i) tail = tail + x[i];
  return acc[0] + tail;
}
// The tree over 16 lane sums, for a kernel that keeps the lanes itself: lane
// `j` holds the terms `i ≡ j (mod 16)` of the full chunks, added in ascending
// `i` from +0.
DS_FN float pinned_reduce(float* acc, float tail) {
  for (uint32_t w = 8; w > 0; w >>= 1) {
    for (uint32_t j = 0; j < w; ++j) acc[j] = acc[j] + acc[j + w];
  }
  return acc[0] + tail;
}

// ---- transcendentals (`numeric`) ----

DS_FN double pow2_f64(int64_t e) { return f64_from_bits(static_cast<uint64_t>(e + 1023) << 52); }

// `numeric::det_exp`: `n = round(x / ln 2)` (half away from zero),
// `r = (x - n·ln2_hi) - n·ln2_lo`, a degree-13 Horner polynomial with the
// correctly rounded `1 / k!`, then a power-of-two scale. Below -708 it is 0.
DS_FN double det_exp(double x) {
  if (is_nan(x)) return x;
  if (x > 709.782712893384) return f64_from_bits(0x7ff0000000000000ull);
  if (x < -708.0) return 0.0;
  const uint64_t c[14] = {0x3ff0000000000000ull, 0x3ff0000000000000ull, 0x3fe0000000000000ull,
                          0x3fc5555555555555ull, 0x3fa5555555555555ull, 0x3f81111111111111ull,
                          0x3f56c16c16c16c17ull, 0x3f2a01a01a01a01aull, 0x3efa01a01a01a01aull,
                          0x3ec71de3a556c734ull, 0x3e927e4fb7789f5cull, 0x3e5ae64567f544e4ull,
                          0x3e21eed8eff8d898ull, 0x3de6124613a86d09ull};
  const double t = x * f64_from_bits(0x3ff71547652b82feull);
  const double n = copysign64(floor(abs64(t) + 0.5), t);
  const double r = (x - n * f64_from_bits(0x3fe62e42fee00000ull)) -
                   n * f64_from_bits(0x3dea39ef35793c76ull);
  double p = f64_from_bits(c[13]);
  for (int k = 12; k >= 0; --k) p = p * r + f64_from_bits(c[k]);
  const int64_t ni = static_cast<int64_t>(n);
  if (ni > 1023) return p * pow2_f64(1023) * pow2_f64(ni - 1023);
  return p * pow2_f64(ni);
}

// `numeric::det_ln` of a positive normal f64: `x = m·2^e` with `m` in
// `(sqrt ½, sqrt 2]`, `s = (m - 1) / (m + 1)`,
// `ln m = 2s·(1 + s²/3 + … + s²²/23)`, `ln x = e·ln2_hi + (ln m + e·ln2_lo)`.
DS_FN double det_ln(double x) {
  const uint64_t bits = f64_bits(x);
  int64_t ei = static_cast<int64_t>((bits >> 52) & 0x7ffu) - 1023;
  double m = f64_from_bits((bits & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
  if (m > f64_from_bits(0x3ff6a09e667f3bcdull) /* sqrt 2 */) {
    m = m * 0.5;
    ei += 1;
  }
  const double e = static_cast<double>(ei);
  const double s = (m - 1.0) / (m + 1.0);
  const double z = s * s;
  double p = 1.0 / 23.0;
  for (int k = 10; k >= 0; --k) p = p * z + 1.0 / static_cast<double>(2 * k + 1);
  return e * f64_from_bits(0x3fe62e42fee00000ull) +
         (2.0 * s * p + e * f64_from_bits(0x3dea39ef35793c76ull));
}

// `numeric::det_sin_cos` for `|x|` up to about 1.6e6: `n = round(x·2/π)`
// (half away from zero), `r = ((x - n·p1) - n·p2) - n·p3` with π/2 in three
// 33-bit pieces, polynomials in `r` on `[-π/4, π/4]`, then the quadrant.
DS_FN void det_sin_cos(double x, double* sin_out, double* cos_out) {
  const uint64_t sc[6] = {0xbfc5555555555549ull, 0x3f8111111110f8a6ull, 0xbf2a01a019c161d5ull,
                          0x3ec71de357b1fe7dull, 0xbe5ae5e68a2b9cebull, 0x3de5d93a5acfd57cull};
  const uint64_t cc[6] = {0x3fa555555555554cull, 0xbf56c16c16c15177ull, 0x3efa01a019cb1590ull,
                          0xbe927e4f809c52adull, 0x3e21ee9ebdb4b1c4ull, 0xbda8fae9be8838d4ull};
  const double t = x * f64_from_bits(0x3fe45f306dc9c883ull);
  const double n = copysign64(floor(abs64(t) + 0.5), t);
  const double r = ((x - n * f64_from_bits(0x3ff921fb54400000ull)) -
                    n * f64_from_bits(0x3dd0b4611a600000ull)) -
                   n * f64_from_bits(0x3ba3198a2e000000ull);
  const double z = r * r;
  double ps = f64_from_bits(sc[5]);
  double pc = f64_from_bits(cc[5]);
  for (int k = 4; k >= 0; --k) {
    ps = ps * z + f64_from_bits(sc[k]);
    pc = pc * z + f64_from_bits(cc[k]);
  }
  const double sin = r + (r * z) * ps;
  const double cos = (1.0 - 0.5 * z) + (z * z) * pc;
  switch (static_cast<int64_t>(n) & 3) {  // `rem_euclid(4)`
    case 0:
      *sin_out = sin;
      *cos_out = cos;
      break;
    case 1:
      *sin_out = cos;
      *cos_out = -sin;
      break;
    case 2:
      *sin_out = -sin;
      *cos_out = -cos;
      break;
    default:
      *sin_out = -cos;
      *cos_out = sin;
      break;
  }
}

// `numeric::exp32`.
DS_FN float exp32(float x) { return static_cast<float>(det_exp(static_cast<double>(x))); }
// `numeric::sigmoid32`: `1 / (1 + e^-x)` in f64, rounded to f32.
DS_FN float sigmoid32(float x) {
  return static_cast<float>(1.0 / (1.0 + det_exp(-static_cast<double>(x))));
}
// `numeric::silu32`: `x / (1 + e^-x)` in f64, rounded to f32.
DS_FN float silu32(float x) {
  const double xd = static_cast<double>(x);
  return static_cast<float>(xd / (1.0 + det_exp(-xd)));
}
// `numeric::softplus32`: `ln(1 + e^x)`, `x` itself above 20, and
// `e - e²/2` while `e^x < 2^-20`.
DS_FN float softplus32(float x) {
  if (x > 20.0f) return x;
  const double e = det_exp(static_cast<double>(x));
  const double v = e < 9.5367431640625e-7 ? e - 0.5 * e * e : det_ln(1.0 + e);
  return static_cast<float>(v);
}
// `numeric::pow32`: `det_exp(e · det_ln(base))` rounded to f32.
DS_FN float pow32(double base, float e) {
  return static_cast<float>(det_exp(static_cast<double>(e) * det_ln(base)));
}
// `numeric::rsqrt32`: two correctly rounded f32 operations.
DS_FN float rsqrt32(float x) { return 1.0f / sqrtf(x); }

// ---- composites the kernels share ----

// `tensor::rms_norm`'s factor: `1 / sqrt(dot(x, x) / n + eps)`; an element is
// then `bf16(weight · (x · factor))`.
DS_FN float rms_factor(float sum_sq, uint32_t n, float eps) {
  return 1.0f / sqrtf(sum_sq / static_cast<float>(n) + eps);
}
DS_FN float rms_element(float x, float factor, float weight) {
  return bf16_round(weight * (x * factor));
}

// `rope::rotate` of one pair by `(c, s)`: `(a·c − b·s, a·s + b·c)`, each
// rounded to BF16; nothing fused.
DS_FN void rope_rotate(float a, float b, float c, float s, float* out_a, float* out_b) {
  *out_a = bf16_round(a * c - b * s);
  *out_b = bf16_round(a * s + b * c);
}
// `rope::unrotate`: `(a·c + b·s, b·c − a·s)`.
DS_FN void rope_unrotate(float a, float b, float c, float s, float* out_a, float* out_b) {
  *out_a = bf16_round(a * c + b * s);
  *out_b = bf16_round(b * c - a * s);
}
// `RopeTable::cis` of one pair: the angle is the f32 product
// `position · freq`; `(cos, sin)` rounded to f32.
DS_FN void rope_cis(uint32_t position, float freq, float* cos_out, float* sin_out) {
  double s, c;
  det_sin_cos(static_cast<double>(static_cast<float>(position) * freq), &s, &c);
  *cos_out = static_cast<float>(c);
  *sin_out = static_cast<float>(s);
}

// `moe::swiglu_hidden` of one element: `silu(min(gate, limit)) ·
// clamp(up, ±limit)` (no clamp when `limit <= 0`), times the routing weight
// when `weighted`, rounded to BF16.
DS_FN float swiglu_hidden(float gate, float up, float limit, bool weighted, float weight) {
  if (limit > 0.0f) {
    gate = min32(gate, limit);
    up = clamp32(up, -limit, limit);
  }
  const float h = silu32(gate) * up;
  return bf16_round(weighted ? weight * h : h);
}

// ---- the total order of every top-k (`attention::top_k`) ----

// A score's rank key: unsigned, larger is better, `-0` and `+0` one key, NaN
// ordered as `f32::total_cmp` orders it.
DS_FN uint32_t topk_key(float score) {
  uint32_t b = score == 0.0f ? 0u : f32_bits(score);
  // `total_cmp`'s map to a signed integer, then to unsigned order.
  b ^= static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) >> 1;
  return b ^ kSign32;
}
// Whether candidate `a` ranks before candidate `b`: higher score first, then
// lower index.
DS_FN bool topk_before(float score_a, uint32_t index_a, float score_b, uint32_t index_b) {
  const uint32_t ka = topk_key(score_a), kb = topk_key(score_b);
  return ka > kb || (ka == kb && index_a < index_b);
}

}  // namespace ds
