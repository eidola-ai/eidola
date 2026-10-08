// The executor's own small kernels: everything between the GEMMs and the
// attention kernel. Activations between layers are f32 (the residual stream
// and norm outputs); GEMM inputs are quantized here, GEMM outputs are read
// back in BF16 or f32. Each kernel names its layout and quantization recipe.
//
// Quantization recipes:
//   - FP8 for the CUTLASS blockwise GEMM: per (row, 128 of K), an f32 scale
//     amax / 448 (1 for an all-zero group), q = sat_e4m3(x / scale).
//   - FP8 for DeepGEMM: per (row, 128 of K), a power-of-two scale
//     2^ceil(log2(amax / 448)) stored as UE8M0 (exponent + 127), four K blocks
//     packed little-endian per i32, words laid out [K/512][rows'] with rows'
//     the row count rounded up to 4 (contiguous layout), or per expert
//     [E][K/512][cap] for the masked layout's `cap` rows per expert.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kThreads = 256;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__device__ __forceinline__ uint16_t f2bf16(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over a kThreads block.
__device__ float block_sum(float v) {
  __shared__ float part[kThreads / 32];
  __shared__ float total;
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < kThreads / 32 ? part[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) total = t;
  }
  __syncthreads();
  const float out = total;
  __syncthreads();
  return out;
}

// UE8M0 exponent byte for a group's amax: 2^ceil(log2(amax / 448)).
__device__ __forceinline__ uint8_t ue8m0_for(float amax) {
  const float r = fmaxf(amax, 1e-10f) / kFp8Max;
  int e = static_cast<int>(ceilf(log2f(r)));
  e = max(-127, min(127, e));
  return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float ue8m0_value(uint8_t b) { return exp2f(static_cast<float>(b) - 127.f); }

// Index of SFA word `w` (of `words` per row) for row `r`: [words][rows4]
// when cap == 0, else [r / cap][words][cap].
__device__ __forceinline__ size_t sfa_index(uint32_t r, uint32_t w, uint32_t words, uint32_t rows4,
                                            uint32_t cap) {
  if (cap == 0) return static_cast<size_t>(w) * rows4 + r;
  return (static_cast<size_t>(r / cap) * words + w) * cap + r % cap;
}

}  // namespace

// out[r] = table[tokens[r]] (BF16 -> f32). One block per row.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_embed(float* __restrict__ out, const uint16_t* __restrict__ table,
                 const uint32_t* __restrict__ tokens, uint32_t hidden) {
  const uint16_t* src = table + static_cast<size_t>(tokens[blockIdx.x]) * hidden;
  float* dst = out + static_cast<size_t>(blockIdx.x) * hidden;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) dst[i] = bf16f(src[i]);
}
EIDOLA_KERNEL_META(eidola_embed, kThreads, 1, 1, 0, 1, 1, 1, 0);

// RMSNorm f32 -> f32 (weight f32), and optionally a BF16 copy:
//   out[r] = w * (x[r] * rsqrt(mean(x[r]^2) + eps)). One block per row.
// `x` may live in a wider buffer: rows are `x_stride` apart.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_rmsnorm_f32(float* __restrict__ out, uint16_t* __restrict__ out_bf16,
                       const float* __restrict__ x, uint32_t x_stride,
                       const float* __restrict__ weight, uint32_t hidden, float eps) {
  const float* row = x + static_cast<size_t>(blockIdx.x) * x_stride;
  float ss = 0.f;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) ss += row[i] * row[i];
  const float inv = rsqrtf(block_sum(ss) / static_cast<float>(hidden) + eps);
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = weight[i] * (row[i] * inv);
    if (out) out[static_cast<size_t>(blockIdx.x) * hidden + i] = v;
    if (out_bf16) out_bf16[static_cast<size_t>(blockIdx.x) * hidden + i] = f2bf16(v);
  }
}
EIDOLA_KERNEL_META(eidola_rmsnorm_f32, kThreads, 1, 1, 0, 1, 1, 1, 0);

// FP8 with f32 scales (the CUTLASS blockwise recipe) of `rows` f32 rows of K:
// q [rows][K] e4m3, sf [K/128][m_pad] f32. Grid (rows, K/128), one warp per
// group of 128.
extern "C" __global__ void __launch_bounds__(32)
    eidola_quant_fp8_f32scale(uint8_t* __restrict__ q, float* __restrict__ sf,
                              const float* __restrict__ x, uint32_t k, uint32_t m_pad) {
  const uint32_t r = blockIdx.x, g = blockIdx.y;
  const float* src = x + static_cast<size_t>(r) * k + g * 128;
  float v[4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = src[threadIdx.x + 32 * j];
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = warp_max(amax);
  const float scale = amax > 0.f ? amax / kFp8Max : 1.f;
  uint8_t* dst = q + static_cast<size_t>(r) * k + g * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[threadIdx.x + 32 * j] = f2e4m3(v[j] / scale);
  if (threadIdx.x == 0) sf[static_cast<size_t>(g) * m_pad + r] = scale;
}
EIDOLA_KERNEL_META(eidola_quant_fp8_f32scale, 32, 1, 1, 0, 1, 1, 1, 0);

// h[i] += d[i] (f32 or BF16 delta), n elements.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_add_f32(float* __restrict__ h, const float* __restrict__ d, uint64_t n) {
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(kThreads) + threadIdx.x; i < n;
       i += static_cast<uint64_t>(gridDim.x) * kThreads)
    h[i] += d[i];
}
EIDOLA_KERNEL_META(eidola_add_f32, kThreads, 1, 1, 0, 1, 1, 1, 0);

// h[r][i] += d[r][i] for `rows` rows of `hidden`, `d` BF16 rows `d_stride` apart.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_add_bf16(float* __restrict__ h, const uint16_t* __restrict__ d, uint32_t hidden,
                    uint32_t d_stride) {
  const uint32_t r = blockIdx.x;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads)
    h[static_cast<size_t>(r) * hidden + i] += bf16f(d[static_cast<size_t>(r) * d_stride + i]);
}
EIDOLA_KERNEL_META(eidola_add_bf16, kThreads, 1, 1, 0, 1, 1, 1, 0);

// Fused QKV (BF16, the checkpoint's rank-major layout padded per chunk) ->
// rotated Q for attention, rotated K and V written into the paged pool.
//
// qkv row t holds `chunks` chunks of `chunk_stride` elements, chunk c =
// [Q heads c*qh .. (c+1)*qh) x 192 | K heads c*kh .. x 192 | V heads c*kh .. x 128].
// Partial NeoX RoPE on dims [0, 64) of every Q and K head with the position's
// cos/sin from `rope` ([max_pos][32 cos | 32 sin], f32, computed on the host
// exactly as the reference does).
//
// KV: K of token t goes to pool + kv_block[t] * block_elems + k_off +
// kv_slot[t] * (num_kv_heads * 192), V likewise with v_off and 128.
// One block per token.
struct EidolaQkvArgs {
  const uint16_t* qkv;
  uint16_t* q_out;  // [T][num_q_heads][192]
  uint16_t* pool;
  const uint32_t* positions;  // RoPE position per token
  const uint32_t* kv_block;
  const uint32_t* kv_slot;  // position within the block
  const float* rope;
  uint64_t block_elems;
  uint64_t k_off;
  uint64_t v_off;
  uint32_t chunk_stride;
  uint32_t chunks;
  uint32_t q_heads_per_chunk;
  uint32_t kv_heads_per_chunk;
};

extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_qkv_rope_kv(const __grid_constant__ EidolaQkvArgs a) {
  constexpr uint32_t D = 192, DV = 128, R = 64, H = R / 2;
  const uint32_t t = blockIdx.x;
  const uint16_t* row = a.qkv + static_cast<size_t>(t) * a.chunk_stride * a.chunks;
  const float* cs = a.rope + static_cast<size_t>(a.positions[t]) * R;
  const uint32_t nq = a.q_heads_per_chunk * a.chunks, nkv = a.kv_heads_per_chunk * a.chunks;
  const size_t kv_base = static_cast<size_t>(a.kv_block[t]) * a.block_elems;
  uint16_t* kdst = a.pool + kv_base + a.k_off + static_cast<size_t>(a.kv_slot[t]) * nkv * D;
  uint16_t* vdst = a.pool + kv_base + a.v_off + static_cast<size_t>(a.kv_slot[t]) * nkv * DV;
  // Q and K: one (head, dim) per thread step.
  const uint32_t per_chunk_qk = (a.q_heads_per_chunk + a.kv_heads_per_chunk) * D;
  for (uint32_t i = threadIdx.x; i < a.chunks * per_chunk_qk; i += kThreads) {
    const uint32_t c = i / per_chunk_qk, within = i % per_chunk_qk;
    const uint32_t head = within / D, d = within % D;
    const uint16_t* src = row + static_cast<size_t>(c) * a.chunk_stride + head * D;
    float v = bf16f(src[d]);
    if (d < R) {
      const uint32_t j = d % H;
      const float cosv = cs[j], sinv = cs[H + j];
      const float x1 = bf16f(src[j]), x2 = bf16f(src[j + H]);
      v = d < H ? x1 * cosv + (-x2) * sinv : x2 * cosv + x1 * sinv;
    }
    if (head < a.q_heads_per_chunk) {
      const uint32_t qh = c * a.q_heads_per_chunk + head;
      a.q_out[(static_cast<size_t>(t) * nq + qh) * D + d] = f2bf16(v);
    } else {
      const uint32_t kh = c * a.kv_heads_per_chunk + (head - a.q_heads_per_chunk);
      kdst[kh * D + d] = f2bf16(v);
    }
  }
  // V: copied as is.
  const uint32_t per_chunk_v = a.kv_heads_per_chunk * DV;
  for (uint32_t i = threadIdx.x; i < a.chunks * per_chunk_v; i += kThreads) {
    const uint32_t c = i / per_chunk_v, within = i % per_chunk_v;
    const uint16_t* src = row + static_cast<size_t>(c) * a.chunk_stride + per_chunk_qk + within;
    vdst[c * per_chunk_v + within] = *src;
  }
}
EIDOLA_KERNEL_META(eidola_qkv_rope_kv, kThreads, 1, 1, 0, 1, 1, 1, sizeof(EidolaQkvArgs));

// SwiGLU over BF16 [rows][2I] (gate | up), silu(g) * u in f32, quantized for
// the CUTLASS blockwise GEMM: q [rows][I], sf [I/128][m_pad]. Grid (rows,
// I/128), one warp per group.
extern "C" __global__ void __launch_bounds__(32)
    eidola_swiglu_quant_fp8_f32scale(uint8_t* __restrict__ q, float* __restrict__ sf,
                                     const uint16_t* __restrict__ gu, uint32_t inter,
                                     uint32_t m_pad) {
  const uint32_t r = blockIdx.x, g = blockIdx.y;
  const uint16_t* gate = gu + static_cast<size_t>(r) * 2 * inter + g * 128;
  const uint16_t* up = gate + inter;
  float v[4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t i = threadIdx.x + 32 * j;
    const float x = bf16f(gate[i]);
    v[j] = x / (1.f + expf(-x)) * bf16f(up[i]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = warp_max(amax);
  const float scale = amax > 0.f ? amax / kFp8Max : 1.f;
  uint8_t* dst = q + static_cast<size_t>(r) * inter + g * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[threadIdx.x + 32 * j] = f2e4m3(v[j] / scale);
  if (threadIdx.x == 0) sf[static_cast<size_t>(g) * m_pad + r] = scale;
}
EIDOLA_KERNEL_META(eidola_swiglu_quant_fp8_f32scale, 32, 1, 1, 0, 1, 1, 1, 0);

// SwiGLU over BF16 [rows][2I] quantized for DeepGEMM: q [rows][I], packed
// UE8M0 sf (see sfa_index). Grid (rows, I/512), 4 warps (one per 128 group).
extern "C" __global__ void __launch_bounds__(128)
    eidola_swiglu_quant_fp8_ue8m0(uint8_t* __restrict__ q, int32_t* __restrict__ sf,
                                  const uint16_t* __restrict__ gu, uint32_t inter,
                                  uint32_t rows4, uint32_t cap) {
  __shared__ uint8_t exps[4];
  const uint32_t r = blockIdx.x, w = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t g = w * 4 + warp;
  const uint16_t* gate = gu + static_cast<size_t>(r) * 2 * inter + g * 128;
  const uint16_t* up = gate + inter;
  float v[4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t i = lane + 32 * j;
    const float x = bf16f(gate[i]);
    v[j] = x / (1.f + expf(-x)) * bf16f(up[i]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = warp_max(amax);
  const uint8_t e = ue8m0_for(amax);
  const float inv = 1.f / ue8m0_value(e);
  uint8_t* dst = q + static_cast<size_t>(r) * inter + g * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = f2e4m3(v[j] * inv);
  if (lane == 0) exps[warp] = e;
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t word = exps[0] | (exps[1] << 8) | (exps[2] << 16) | (static_cast<uint32_t>(exps[3]) << 24);
    sf[sfa_index(r, w, inter / 512, rows4, cap)] = static_cast<int32_t>(word);
  }
}
EIDOLA_KERNEL_META(eidola_swiglu_quant_fp8_ue8m0, 128, 1, 1, 0, 1, 1, 1, 0);

// Router: logits = x · Wᵀ (x f32 [T][H], W BF16 [E][H], f32 accumulation),
// scores = sigmoid(logits), top-k of scores + bias (ties to the lower expert),
// experts sorted ascending, weights = scores / (sum + 1e-20) * scaling.
// One block per token; E <= 256, k <= 8.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_router_topk(int32_t* __restrict__ topk_ids, float* __restrict__ topk_w,
                       const float* __restrict__ x, const uint16_t* __restrict__ w,
                       const float* __restrict__ bias, uint32_t hidden, uint32_t experts,
                       uint32_t top_k, float scaling) {
  __shared__ float score[256];
  __shared__ float choice[256];
  __shared__ int32_t picked[8];
  const uint32_t t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const float* xr = x + static_cast<size_t>(t) * hidden;
  for (uint32_t e = warp; e < experts; e += kThreads / 32) {
    const uint16_t* wr = w + static_cast<size_t>(e) * hidden;
    float acc = 0.f;
    for (uint32_t i = lane; i < hidden; i += 32) acc += xr[i] * bf16f(wr[i]);
    acc = warp_sum(acc);
    if (lane == 0) {
      const float s = 1.f / (1.f + expf(-acc));
      score[e] = s;
      choice[e] = s + bias[e];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // k rounds of argmax over the remaining experts (ties to the lower id).
    for (uint32_t j = 0; j < top_k; ++j) {
      int32_t best = -1;
      for (uint32_t e = 0; e < experts; ++e) {
        bool taken = false;
        for (uint32_t p = 0; p < j; ++p) taken |= picked[p] == static_cast<int32_t>(e);
        if (!taken && (best < 0 || choice[e] > choice[best])) best = static_cast<int32_t>(e);
      }
      picked[j] = best;
    }
    // Ascending expert order.
    for (uint32_t a = 1; a < top_k; ++a)
      for (uint32_t b = a; b > 0 && picked[b - 1] > picked[b]; --b) {
        const int32_t tmp = picked[b];
        picked[b] = picked[b - 1];
        picked[b - 1] = tmp;
      }
    float sum = 0.f;
    for (uint32_t j = 0; j < top_k; ++j) sum += score[picked[j]];
    const float denom = sum + 1e-20f;
    for (uint32_t j = 0; j < top_k; ++j) {
      topk_ids[t * top_k + j] = picked[j];
      topk_w[t * top_k + j] = score[picked[j]] / denom * scaling;
    }
  }
}
EIDOLA_KERNEL_META(eidola_router_topk, kThreads, 1, 1, 0, 1, 1, 1, 0);

// Expert-major placement of the T*k routed (token, slot) pairs, deterministic
// (tokens ascending within an expert). One block of 256 threads, thread e
// owning expert e.
//
//   contiguous (cap == 0): expert e's rows start at a multiple of 128 in
//     expert order; grouped_layout[row] = e for its rows, -1 for padding;
//     the layout covers `rows_bound` rows.
//   masked (cap > 0): expert e's rows are e*cap + i; grouped_layout[e] =
//     its row count (at most cap).
//
// row_of[t*k + j] = the row (token t, slot j) landed in.
// row_src[row] = the token feeding it (-1 for padding), for the gather.
extern "C" __global__ void __launch_bounds__(256)
    eidola_moe_permute(int32_t* __restrict__ grouped_layout, int32_t* __restrict__ row_of,
                       int32_t* __restrict__ row_src, const int32_t* __restrict__ topk_ids,
                       uint32_t tokens, uint32_t top_k, uint32_t cap, uint32_t rows_bound) {
  __shared__ uint32_t counts[256];
  __shared__ uint32_t starts[256];
  const uint32_t e = threadIdx.x;
  const uint32_t n = tokens * top_k;
  uint32_t c = 0;
  for (uint32_t i = 0; i < n; ++i) c += topk_ids[i] == static_cast<int32_t>(e);
  counts[e] = c;
  __syncthreads();
  if (e == 0) {
    uint32_t s = 0;
    for (uint32_t x = 0; x < 256; ++x) {
      starts[x] = cap ? x * cap : s;
      s += cap ? 0 : (counts[x] + 127) / 128 * 128;
    }
  }
  __syncthreads();
  // Clear the layout (contiguous: every row -1; masked: counts below).
  if (cap == 0) {
    for (uint32_t r = e; r < rows_bound; r += 256) {
      grouped_layout[r] = -1;
      row_src[r] = -1;
    }
  } else {
    for (uint32_t r = e; r < 256 * cap; r += 256) row_src[r] = -1;
  }
  __syncthreads();
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (topk_ids[i] == static_cast<int32_t>(e)) {
      const uint32_t row = starts[e] + k++;
      row_of[i] = static_cast<int32_t>(row);
      row_src[row] = static_cast<int32_t>(i / top_k);
      if (cap == 0) grouped_layout[row] = static_cast<int32_t>(e);
    }
  }
  if (cap == 0) {
    // Padding rows of this expert's run belong to it too (blocks are
    // assigned by their first row).
    for (uint32_t row = starts[e] + counts[e]; row < starts[e] + (counts[e] + 127) / 128 * 128; ++row)
      grouped_layout[row] = static_cast<int32_t>(e);
  } else {
    grouped_layout[e] = static_cast<int32_t>(counts[e]);
  }
}
EIDOLA_KERNEL_META(eidola_moe_permute, 256, 1, 1, 0, 1, 1, 1, 0);

// Gather f32 rows by row_src into DeepGEMM's FP8 A with packed UE8M0 scales:
// a [rows][K], sf (see sfa_index); rows with row_src -1 become zeros (scale
// byte 127). Grid (rows, K/512), 4 warps.
extern "C" __global__ void __launch_bounds__(128)
    eidola_gather_quant_ue8m0(uint8_t* __restrict__ a, int32_t* __restrict__ sf,
                              const float* __restrict__ x, const int32_t* __restrict__ row_src,
                              uint32_t k, uint32_t rows4, uint32_t cap) {
  __shared__ uint8_t exps[4];
  const uint32_t r = blockIdx.x, w = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t g = w * 4 + warp;
  const int32_t src = row_src[r];
  float v[4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = src >= 0 ? x[static_cast<size_t>(src) * k + g * 128 + lane + 32 * j] : 0.f;
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = warp_max(amax);
  const uint8_t e = src >= 0 ? ue8m0_for(amax) : 127;
  const float inv = 1.f / ue8m0_value(e);
  uint8_t* dst = a + static_cast<size_t>(r) * k + g * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = f2e4m3(v[j] * inv);
  if (lane == 0) exps[warp] = e;
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t word = exps[0] | (exps[1] << 8) | (exps[2] << 16) | (static_cast<uint32_t>(exps[3]) << 24);
    sf[sfa_index(r, w, k / 512, rows4, cap)] = static_cast<int32_t>(word);
  }
}
EIDOLA_KERNEL_META(eidola_gather_quant_ue8m0, 128, 1, 1, 0, 1, 1, 1, 0);

// out[t] = sum over slots j (experts ascending) of w[t][j] * d[row_of[t][j]]
// (d BF16 [rows][H]), accumulated in f32. One block per token.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_moe_combine(float* __restrict__ out, const uint16_t* __restrict__ d,
                       const int32_t* __restrict__ row_of, const float* __restrict__ topk_w,
                       uint32_t hidden, uint32_t top_k) {
  const uint32_t t = blockIdx.x;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) {
    float acc = 0.f;
    for (uint32_t j = 0; j < top_k; ++j) {
      const int32_t row = row_of[t * top_k + j];
      acc += topk_w[t * top_k + j] * bf16f(d[static_cast<size_t>(row) * hidden + i]);
    }
    out[static_cast<size_t>(t) * hidden + i] = acc;
  }
}
EIDOLA_KERNEL_META(eidola_moe_combine, kThreads, 1, 1, 0, 1, 1, 1, 0);

// out[i] = BF16(x[rows[i]]) for selected f32 rows of `hidden`. One block per
// output row.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_gather_rows_bf16(uint16_t* __restrict__ out, const float* __restrict__ x,
                            const uint32_t* __restrict__ rows, uint32_t hidden) {
  const float* src = x + static_cast<size_t>(rows[blockIdx.x]) * hidden;
  uint16_t* dst = out + static_cast<size_t>(blockIdx.x) * hidden;
  for (uint32_t i = threadIdx.x; i < hidden; i += kThreads) dst[i] = f2bf16(src[i]);
}
EIDOLA_KERNEL_META(eidola_gather_rows_bf16, kThreads, 1, 1, 0, 1, 1, 1, 0);
