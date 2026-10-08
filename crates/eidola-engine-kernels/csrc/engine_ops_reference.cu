// Reference kernels for the GPU tests and the kernel bench: the router, the
// UE8M0 SwiGLU, the UE8M0 gather, the fused-QKV RoPE + KV write and the expert
// combine in their single-block, row-per-block forms, which define the
// numerics the executor's kernels (engine_ops.cu) reproduce bit for bit. Not
// loaded by the executor.
//
// The reference gather walks every row of the expert layout and reads the
// token feeding it from `row_src` (-1 for padding); the reference SwiGLU
// walks every row. Rows named by a routed (token, slot) pair are the ones the
// two forms must agree on.

#include <cstdint>

#include "eidola_kernel.cuh"
#include "engine_ops_common.cuh"

// Router: logits = x · Wᵀ (x f32 [T][H], W BF16 [E][H], f32 accumulation),
// scores = sigmoid(logits), top-k of scores + bias (ties to the lower expert),
// experts sorted ascending, weights = scores / (sum + 1e-20) * scaling.
// One block per token; E <= 256, k <= 8.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_reference_router_topk(int32_t* __restrict__ topk_ids, float* __restrict__ topk_w,
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
EIDOLA_KERNEL_META(eidola_reference_router_topk, kThreads, 1, 1, 0, 1, 1, 1, 0);

// SwiGLU over BF16 [rows][2I] quantized for DeepGEMM: q [rows][I], packed
// UE8M0 sf (see sfa_index). Grid (rows, I/512), 4 warps (one per 128 group).
extern "C" __global__ void __launch_bounds__(128)
    eidola_reference_swiglu_quant_fp8_ue8m0(uint8_t* __restrict__ q, int32_t* __restrict__ sf,
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
EIDOLA_KERNEL_META(eidola_reference_swiglu_quant_fp8_ue8m0, 128, 1, 1, 0, 1, 1, 1, 0);

// Gather f32 rows by row_src into DeepGEMM's FP8 A with packed UE8M0 scales:
// a [rows][K], sf (see sfa_index); rows with row_src -1 become zeros (scale
// byte 127). Grid (rows, K/512), 4 warps.
extern "C" __global__ void __launch_bounds__(128)
    eidola_reference_gather_quant_ue8m0(uint8_t* __restrict__ a, int32_t* __restrict__ sf,
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
EIDOLA_KERNEL_META(eidola_reference_gather_quant_ue8m0, 128, 1, 1, 0, 1, 1, 1, 0);

// Fused QKV RoPE + paged KV write (EidolaQkvArgs; the layout is described at
// eidola_qkv_rope_kv in engine_ops.cu). One block per token.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_reference_qkv_rope_kv(const __grid_constant__ EidolaQkvArgs a) {
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
EIDOLA_KERNEL_META(eidola_reference_qkv_rope_kv, kThreads, 1, 1, 0, 1, 1, 1, sizeof(EidolaQkvArgs));

// out[t] = sum over slots j (experts ascending) of w[t][j] * d[row_of[t][j]]
// (d BF16 [rows][H]), accumulated in f32. One block per token.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_reference_moe_combine(float* __restrict__ out, const uint16_t* __restrict__ d,
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
EIDOLA_KERNEL_META(eidola_reference_moe_combine, kThreads, 1, 1, 0, 1, 1, 1, 0);
