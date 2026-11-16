// The numerics probe: every function of `ds_numeric.cuh` behind one entry, an
// item per thread, so a device test replays the host vectors through the code
// nvcc emits for them. Only tests load it.
//
// `ds_probe_nodiv.cu` compiles this file again without the ops that divide or
// take a square root, as `eidola_ds_probe_nodiv`: a division's SASS expansion
// holds fused multiply-adds of its own, so only an image without one can be
// required to hold none (`tests/manifest.rs`).

#include "ds_abi.cuh"
#include "ds_numeric.cuh"
#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxTerms = 128;

__device__ __forceinline__ float f(uint64_t w) {
  return ds::f32_from_bits(static_cast<uint32_t>(w));
}
__device__ __forceinline__ uint64_t w32(float x) { return ds::f32_bits(x); }

}  // namespace

#if defined(DS_PROBE_NO_DIVISION)
#define DS_PROBE_ENTRY eidola_ds_probe_nodiv
#else
#define DS_PROBE_ENTRY eidola_ds_probe
#endif

extern "C" __global__ void __launch_bounds__(kThreads)
    DS_PROBE_ENTRY(const __grid_constant__ DsProbeParams p) {
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= p.items) return;
  const uint64_t* in = p.in + i * p.in_words;
  uint64_t* out = p.out + i * p.out_words;
  switch (p.op) {
    case DS_PROBE_BF16_ROUND:
      out[0] = w32(ds::bf16_round(f(in[0])));
      break;
    case DS_PROBE_ROUND_E4M3:
      out[0] = w32(ds::round_e4m3(f(in[0])));
      break;
    case DS_PROBE_ROUND_E2M1:
      out[0] = w32(ds::round_e2m1(f(in[0])));
      break;
    case DS_PROBE_EXP32:
      out[0] = w32(ds::exp32(f(in[0])));
      break;
#if !defined(DS_PROBE_NO_DIVISION)
    case DS_PROBE_SIGMOID32:
      out[0] = w32(ds::sigmoid32(f(in[0])));
      break;
    case DS_PROBE_SILU32:
      out[0] = w32(ds::silu32(f(in[0])));
      break;
    case DS_PROBE_SOFTPLUS32: {
      const float x = f(in[0]);
      out[0] = w32(ds::is_nan(x) ? x : ds::softplus32(x));
      break;
    }
    case DS_PROBE_RSQRT32:
      out[0] = w32(ds::rsqrt32(f(in[0])));
      break;
#endif
    case DS_PROBE_ROUND_SCALE:
      out[0] = w32(ds::round_scale(f(in[0]), f(in[1])));
      break;
    case DS_PROBE_DET_EXP:
      out[0] = ds::f64_bits(ds::det_exp(ds::f64_from_bits(in[0])));
      break;
#if !defined(DS_PROBE_NO_DIVISION)
    case DS_PROBE_DET_LN:
      out[0] = ds::f64_bits(ds::det_ln(ds::f64_from_bits(in[0])));
      break;
#endif
    case DS_PROBE_DET_SIN_COS: {
      double s, c;
      ds::det_sin_cos(ds::f64_from_bits(in[0]), &s, &c);
      out[0] = ds::f64_bits(s);
      out[1] = ds::f64_bits(c);
      break;
    }
#if !defined(DS_PROBE_NO_DIVISION)
    case DS_PROBE_POW32:
      out[0] = w32(ds::pow32(ds::f64_from_bits(in[0]), f(in[1])));
      break;
#endif
    case DS_PROBE_ACT_QUANT:
    case DS_PROBE_FQ8:
    case DS_PROBE_FQ4M: {
      float x[32];
      uint8_t codes[32];
      for (int j = 0; j < 32; ++j) x[j] = f(in[j]);
      if (p.op == DS_PROBE_FQ4M) {
        const float s = ds::act_scale_decode(ds::fp4_pow2_quant_block(x, codes));
        for (int j = 0; j < 32; ++j) out[j] = w32(ds::dequant_e2m1(codes[j], s));
      } else {
        const float s = ds::act_scale_decode(ds::act_quant_block(x, codes));
        if (p.op == DS_PROBE_FQ8) {
          for (int j = 0; j < 32; ++j) out[j] = w32(ds::dequant_e4m3(codes[j], s));
        } else {
          for (int j = 0; j < 32; ++j) out[j] = w32(ds::e4m3_decode(codes[j]));
          out[32] = w32(s);
        }
      }
      break;
    }
#if !defined(DS_PROBE_NO_DIVISION)
    case DS_PROBE_FQ4N: {
      float x[16];
      uint8_t codes[16];
      for (int j = 0; j < 16; ++j) x[j] = f(in[j]);
      const float s = ds::e4m3_decode(ds::fp4_e4m3_quant_block(x, codes));
      for (int j = 0; j < 16; ++j) out[j] = w32(ds::dequant_e2m1(codes[j], s));
      break;
    }
#endif
    case DS_PROBE_FP8_DOT:
    case DS_PROBE_FP4_DOT: {
      uint8_t a[32], w[32];
      const int w_bytes = p.op == DS_PROBE_FP8_DOT ? 32 : 16;
      for (int j = 0; j < 32; ++j) a[j] = static_cast<uint8_t>(in[j]);
      for (int j = 0; j < w_bytes; ++j) w[j] = static_cast<uint8_t>(in[32 + j]);
      out[0] = w32(p.op == DS_PROBE_FP8_DOT ? ds::fp8_fp8_block_dot(a, w)
                                             : ds::fp8_fp4_block_dot(a, w, 0));
      break;
    }
    case DS_PROBE_DOT:
    case DS_PROBE_PINNED_SUM: {
      float a[kMaxTerms], b[kMaxTerms];
      const bool dot = p.op == DS_PROBE_DOT;
      const uint32_t n = dot ? p.in_words / 2 : p.in_words;
      if (n > kMaxTerms) return;
      for (uint32_t j = 0; j < n; ++j) {
        a[j] = f(in[j]);
        b[j] = dot ? f(in[n + j]) : 0.0f;
      }
      out[0] = w32(dot ? ds::pinned_dot(a, b, n) : ds::pinned_sum(a, n));
      break;
    }
    case DS_PROBE_ROTATE:
    case DS_PROBE_UNROTATE: {
      float a, b;
      if (p.op == DS_PROBE_ROTATE) {
        ds::rope_rotate(f(in[0]), f(in[1]), f(in[2]), f(in[3]), &a, &b);
      } else {
        ds::rope_unrotate(f(in[0]), f(in[1]), f(in[2]), f(in[3]), &a, &b);
      }
      out[0] = w32(a);
      out[1] = w32(b);
      break;
    }
    case DS_PROBE_CIS: {
      float c, s;
      ds::rope_cis(static_cast<uint32_t>(in[0]), f(in[1]), &c, &s);
      out[0] = w32(c);
      out[1] = w32(s);
      break;
    }
#if !defined(DS_PROBE_NO_DIVISION)
    case DS_PROBE_SWIGLU:
      out[0] = w32(ds::swiglu_hidden(f(in[0]), f(in[1]), f(in[2]), in[3] != 0, f(in[4])));
      break;
#endif
    case DS_PROBE_TOPK_KEY:
      out[0] = ds::topk_key(f(in[0]));
      break;
    default:
      break;
  }
}
#if defined(DS_PROBE_NO_DIVISION)
EIDOLA_KERNEL_META(eidola_ds_probe_nodiv, kThreads, 1, 1, 0, 1, 1, 1, sizeof(DsProbeParams));
#else
EIDOLA_KERNEL_META(eidola_ds_probe, kThreads, 1, 1, 0, 1, 1, 1, sizeof(DsProbeParams));
#endif
