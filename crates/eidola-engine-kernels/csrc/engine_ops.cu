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
//     the expert layout's row count rounded up to 4.

#include <cooperative_groups.h>

#include <cstdint>

#include "eidola_kernel.cuh"
#include "engine_ops_common.cuh"

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
// rotated Q for attention, rotated K and V written into the paged pool
// (EidolaQkvArgs in engine_ops_common.cuh).
//
// qkv row t holds `chunks` chunks of `chunk_stride` elements, chunk c =
// [Q heads c*qh .. (c+1)*qh) x 192 | K heads c*kh .. x 192 | V heads c*kh .. x 128].
// Partial NeoX RoPE on dims [0, 64) of every Q and K head with the position's
// cos/sin from `rope` ([max_pos][32 cos | 32 sin], f32, computed on the host
// exactly as the reference does).
//
// KV: K of token t goes to pool + kv_block[t] * block_elems + k_off +
// kv_slot[t] * (num_kv_heads * 192), V likewise with v_off and 128.
//
// One thread per kQkvVec consecutive output elements: grid (tokens,
// ceil(per-token elements / (kQkvVec * kThreads))), where a token's elements
// are its chunks' Q, K and V elements in the row's own order (per chunk
// (qh + kh) x 192 then kh x 128). Heads of 192 and 128 are whole vectors, so
// a vector never straddles a head, and the 64 rotated dims are whole vectors:
// a rotated vector also loads its 8 partners 32 dims away and its 8 cos/sin
// pairs. Every load and store is 16 bytes (the host checks the alignment of
// every base, stride and offset). Each element is computed independently,
// with the arithmetic of the per-token form.
constexpr uint32_t kQkvVec = 8;

__device__ __forceinline__ void unpack_bf16x8(const uint4 w, float (&f)[kQkvVec]) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (uint32_t e = 0; e < kQkvVec; ++e) f[e] = bf16f(static_cast<uint16_t>(words[e / 2] >> (16 * (e % 2))));
}

__device__ __forceinline__ uint4 pack_bf16x8(const float (&f)[kQkvVec]) {
  uint32_t words[4];
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
    words[i] = static_cast<uint32_t>(f2bf16(f[2 * i])) | (static_cast<uint32_t>(f2bf16(f[2 * i + 1])) << 16);
  return make_uint4(words[0], words[1], words[2], words[3]);
}

extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_qkv_rope_kv(const __grid_constant__ EidolaQkvArgs a) {
  constexpr uint32_t D = 192, DV = 128, R = 64, H = R / 2;
  const uint32_t t = blockIdx.x;
  const uint32_t per_chunk_qk = (a.q_heads_per_chunk + a.kv_heads_per_chunk) * D;
  const uint32_t per_chunk_v = a.kv_heads_per_chunk * DV;
  const uint32_t per_chunk = per_chunk_qk + per_chunk_v;
  const uint32_t el = (blockIdx.y * kThreads + threadIdx.x) * kQkvVec;
  if (el >= a.chunks * per_chunk) return;
  const uint32_t c = el / per_chunk, within = el % per_chunk;
  const uint16_t* row = a.qkv + static_cast<size_t>(t) * a.chunk_stride * a.chunks;
  const uint32_t nq = a.q_heads_per_chunk * a.chunks, nkv = a.kv_heads_per_chunk * a.chunks;
  const size_t kv_base = static_cast<size_t>(a.kv_block[t]) * a.block_elems;
  if (within < per_chunk_qk) {
    // Q and K: one head's dims d0 .. d0 + 8.
    const uint32_t head = within / D, d0 = within % D;
    const uint16_t* src = row + static_cast<size_t>(c) * a.chunk_stride + head * D;
    float v[kQkvVec];
    unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(src + d0)), v);
    if (d0 < R) {
      // The pair of dims d and d +- 32: x1 = src[j], x2 = src[j + H] with
      // j = d % H; this vector holds x1 below H and x2 above it.
      const uint32_t j0 = d0 % H;
      const float* cs = a.rope + static_cast<size_t>(a.positions[t]) * R;
      float other[kQkvVec], cosv[kQkvVec], sinv[kQkvVec];
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(src + (d0 < H ? d0 + H : d0 - H))), other);
#pragma unroll
      for (uint32_t q4 = 0; q4 < kQkvVec / 4; ++q4) {
        const float4 cv = __ldg(reinterpret_cast<const float4*>(cs + j0 + 4 * q4));
        const float4 sv = __ldg(reinterpret_cast<const float4*>(cs + H + j0 + 4 * q4));
        cosv[4 * q4] = cv.x, cosv[4 * q4 + 1] = cv.y, cosv[4 * q4 + 2] = cv.z, cosv[4 * q4 + 3] = cv.w;
        sinv[4 * q4] = sv.x, sinv[4 * q4 + 1] = sv.y, sinv[4 * q4 + 2] = sv.z, sinv[4 * q4 + 3] = sv.w;
      }
#pragma unroll
      for (uint32_t e = 0; e < kQkvVec; ++e) {
        // x1 * cos - x2 * sin fuses the first product onto the rounded
        // second; x2 * cos + x1 * sin fuses the second product onto the
        // rounded first. Both are the contractions the reference form
        // compiles to, so every rotated element is bit-identical to it.
        const float x1 = d0 < H ? v[e] : other[e], x2 = d0 < H ? other[e] : v[e];
        v[e] = d0 < H ? __fmaf_rn(x1, cosv[e], -__fmul_rn(x2, sinv[e]))
                      : __fmaf_rn(x1, sinv[e], __fmul_rn(x2, cosv[e]));
      }
    }
    uint16_t* dst;
    if (head < a.q_heads_per_chunk) {
      const uint32_t qh = c * a.q_heads_per_chunk + head;
      dst = a.q_out + (static_cast<size_t>(t) * nq + qh) * D;
    } else {
      const uint32_t kh = c * a.kv_heads_per_chunk + (head - a.q_heads_per_chunk);
      dst = a.pool + kv_base + a.k_off + static_cast<size_t>(a.kv_slot[t]) * nkv * D + kh * D;
    }
    *reinterpret_cast<uint4*>(dst + d0) = pack_bf16x8(v);
  } else {
    // V: copied as is.
    uint16_t* vdst = a.pool + kv_base + a.v_off + static_cast<size_t>(a.kv_slot[t]) * nkv * DV;
    const uint32_t vi = within - per_chunk_qk;
    *reinterpret_cast<uint4*>(vdst + c * per_chunk_v + vi) =
        __ldg(reinterpret_cast<const uint4*>(row + static_cast<size_t>(c) * a.chunk_stride + per_chunk_qk + vi));
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
// UE8M0 sf (see sfa_index), for exactly the rows a routed (token, slot) pair
// landed in: block x is pair i, its row row_of[i]. Rows no pair names (the
// expert layout's padding) are not touched; the grouped GEMMs compute every
// row independently, so what they hold reaches only padding outputs, which
// the combine never reads.
//
// Grid (pairs, I/1024), 4 warps: block y covers two 512-wide scale words
// (eight 128-wide groups), each half-warp one group, each lane 8 consecutive
// elements (16-byte gate and up loads, an 8-byte store). gu and q must be
// 16- and 8-byte aligned (the host checks both).
constexpr uint32_t kSwigluVec = 8;
constexpr uint32_t kSwigluWidth = 128 * kSwigluVec;

__device__ __forceinline__ float half_warp_max(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

extern "C" __global__ void __launch_bounds__(128)
    eidola_swiglu_quant_fp8_ue8m0(uint8_t* __restrict__ q, int32_t* __restrict__ sf,
                                  const uint16_t* __restrict__ gu,
                                  const int32_t* __restrict__ row_of, uint32_t inter,
                                  uint32_t rows4) {
  __shared__ uint8_t exps[8];
  const uint32_t r = static_cast<uint32_t>(row_of[blockIdx.x]);
  const uint32_t gb = threadIdx.x / 16, i0 = (threadIdx.x % 16) * kSwigluVec;
  const uint32_t g = blockIdx.y * 8 + gb;
  const uint16_t* gate = gu + static_cast<size_t>(r) * 2 * inter + g * 128 + i0;
  const uint4 gw = *reinterpret_cast<const uint4*>(gate);
  const uint4 uw = *reinterpret_cast<const uint4*>(gate + inter);
  const uint32_t gws[4] = {gw.x, gw.y, gw.z, gw.w}, uws[4] = {uw.x, uw.y, uw.z, uw.w};
  float v[kSwigluVec];
  float amax = 0.f;
#pragma unroll
  for (uint32_t e = 0; e < kSwigluVec; ++e) {
    const float x = bf16f(static_cast<uint16_t>(gws[e / 2] >> (16 * (e % 2))));
    v[e] = x / (1.f + expf(-x)) * bf16f(static_cast<uint16_t>(uws[e / 2] >> (16 * (e % 2))));
    amax = fmaxf(amax, fabsf(v[e]));
  }
  amax = half_warp_max(amax);
  const uint8_t e8 = ue8m0_for(amax);
  const float inv = 1.f / ue8m0_value(e8);
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (uint32_t e = 0; e < 4; ++e) {
    lo |= static_cast<uint32_t>(f2e4m3(v[e] * inv)) << (8 * e);
    hi |= static_cast<uint32_t>(f2e4m3(v[e + 4] * inv)) << (8 * e);
  }
  *reinterpret_cast<uint2*>(q + static_cast<size_t>(r) * inter + g * 128 + i0) = make_uint2(lo, hi);
  if (threadIdx.x % 16 == 0) exps[gb] = e8;
  __syncthreads();
  if (threadIdx.x < 2) {
    const uint8_t* e4 = exps + 4 * threadIdx.x;
    const uint32_t word = e4[0] | (e4[1] << 8) | (e4[2] << 16) | (static_cast<uint32_t>(e4[3]) << 24);
    sf[sfa_index(r, blockIdx.y * 2 + threadIdx.x, rows4)] = static_cast<int32_t>(word);
  }
}
EIDOLA_KERNEL_META(eidola_swiglu_quant_fp8_ue8m0, 128, 1, 1, 0, 1, 1, 1, 0);

// Router: logits = x · Wᵀ (x f32 [T][H], W BF16 [E][H], f32 accumulation),
// scores = sigmoid(logits), top-k of scores + bias (ties to the lower expert),
// experts sorted ascending, weights = scores / (sum + 1e-20) * scaling.
// E <= 256, k <= 8, H a multiple of kRouterChunk and at most kRouterMaxHidden.
//
// The numerics, which every form below shares: a logit is lane l's sequential
// sum of x[i] * w[i] over i = l, l + 32, ... (fused multiply-add, ascending
// i), then the xor-butterfly over the 32 lanes, then 1 / (1 + expf(-x)).
// The selection takes, per round, the best untaken expert under (choice
// greater, or equal and lower id), unless the lowest untaken expert's choice
// is NaN, in which case that expert. This is exactly what a scan in expert
// order keeping the first strictly greater choice selects.
//
// The executor runs the split form, two launches with the scores between them
// in global memory:
//
//   eidola_router_scores: kRouterScoresTile tokens x kRouterScoresExperts
//     experts per block, no cluster, so the grid grows with both the token
//     count and the expert count (16 blocks per 8 tokens at 256 experts). Each
//     lane runs the chains of 4 experts x 8 tokens for its i; the rows stream
//     through shared memory a chunk at a time, kRouterScoresStages chunks in
//     flight. Writes every (token, expert)'s choice and score.
//   eidola_router_select: one warp per token, reading the token's 256 choices
//     from global memory and selecting as below.
//
// The two cluster forms below compute the same ids and weights in one launch
// and are kept for the kernel bench and the tests. Both run one cluster of
// kRouterCluster blocks over a group of tokens, block rank b scoring experts
// 32b .. 32b + 31; rank 0 then reads every block's scores through distributed
// shared memory and selects, one warp per token. They differ in how many
// tokens a cluster takes:
//
//   eidola_router_topk: one token per cluster, 32 warps per block, one expert
//     per warp; the token's row staged in shared memory, each lane keeping
//     kRouterBatch weight loads in flight ahead of its multiply-adds.
//   eidola_router_topk_tiled: kRouterTile tokens per cluster, 4 warps per
//     block, each lane running the chains of 8 experts x kRouterTile tokens
//     for its i, so each weight is read once per tile instead of once per
//     token. The tile's rows and the block's weight rows stream through
//     shared memory kRouterChunk elements of i at a time (cp.async,
//     kRouterStages chunks in flight).
//
// engine_ops.rs (RouterForm) records why the executor runs the split form.
constexpr uint32_t kRouterCluster = 8;
constexpr uint32_t kRouterWarps = 32;
constexpr uint32_t kRouterThreads = kRouterWarps * 32;
constexpr uint32_t kRouterMaxHidden = 4096;
// W loads in flight per lane before their multiply-adds.
constexpr uint32_t kRouterBatch = 32;

constexpr uint32_t kRouterTile = 8;
constexpr uint32_t kRouterTiledWarps = 4;
constexpr uint32_t kRouterTiledThreads = kRouterTiledWarps * 32;
constexpr uint32_t kRouterWarpExperts = kRouterWarps / kRouterTiledWarps;
constexpr uint32_t kRouterChunk = 128;
constexpr uint32_t kRouterStages = 3;

namespace {

constexpr uint32_t kNone = 0xffffffffu;

// a before b in the selection order: a valid, and b invalid or a's choice
// greater, or equal with a lower id. A strict total order over valid
// (non-NaN) candidates with distinct ids, so any reduction tree finds its
// maximum.
__device__ __forceinline__ bool router_before(float av, uint32_t ai, float bv, uint32_t bi) {
  if (ai == kNone) return false;
  if (bi == kNone) return true;
  return av > bv || (av == bv && ai < bi);
}

// The score of a logit, as the one-token form computes it inline.
__device__ __forceinline__ float router_sigmoid(float logit) { return 1.f / (1.f + expf(-logit)); }

// One token's selection for the tiled form, by a whole warp of rank 0 (the
// one-token form runs the same code inline): `choice` and `score` are this
// block's rows of the token's 32 choices and scores, which every rank holds at
// the same offset. Writes the token's k ids (ascending) and weights.
__device__ __forceinline__ void router_select(const cooperative_groups::cluster_group& cluster,
                                              float* choice, float* score, uint32_t lane,
                                              uint32_t experts, uint32_t top_k, float scaling,
                                              int32_t* __restrict__ ids_out,
                                              float* __restrict__ w_out) {
  // Lane l holds experts 32q + l for q < kRouterCluster (block q's slot l).
  float c[kRouterCluster];
  uint32_t valid = 0;
#pragma unroll
  for (uint32_t q = 0; q < kRouterCluster; ++q) {
    c[q] = 0.f;
    if (q * kRouterWarps + lane < experts) {
      c[q] = cluster.map_shared_rank(choice, q)[lane];
      valid |= 1u << q;
    }
  }
  int32_t picked[8];
  for (uint32_t j = 0; j < top_k; ++j) {
    // The lowest untaken expert, and its choice.
    // `valid` loses an expert's bit when it is taken.
    uint32_t low = kNone;
#pragma unroll
    for (uint32_t q = 0; q < kRouterCluster; ++q)
      if ((valid >> q & 1u) && low == kNone) low = q * kRouterWarps + lane;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) low = min(low, __shfl_xor_sync(0xffffffffu, low, o));
    float low_c = 0.f;
#pragma unroll
    for (uint32_t q = 0; q < kRouterCluster; ++q)
      if (q == low / kRouterWarps) low_c = c[q];
    low_c = __shfl_sync(0xffffffffu, low_c, low % kRouterWarps);
    uint32_t best = low;
    if (!isnan(low_c)) {
      float bv = 0.f;
      uint32_t bi = kNone;
#pragma unroll
      for (uint32_t q = 0; q < kRouterCluster; ++q) {
        const uint32_t ei = q * kRouterWarps + lane;
        if ((valid >> q & 1u) && !isnan(c[q]) && router_before(c[q], ei, bv, bi)) {
          bv = c[q];
          bi = ei;
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (router_before(ov, oi, bv, bi)) {
          bv = ov;
          bi = oi;
        }
      }
      best = bi;
    }
    picked[j] = static_cast<int32_t>(best);
    if (lane == best % kRouterWarps) valid &= ~(1u << (best / kRouterWarps));
  }
  if (lane == 0) {
    // Ascending expert order.
    for (uint32_t a = 1; a < top_k; ++a)
      for (uint32_t b = a; b > 0 && picked[b - 1] > picked[b]; --b) {
        const int32_t tmp = picked[b];
        picked[b] = picked[b - 1];
        picked[b - 1] = tmp;
      }
    float sel[8];
    for (uint32_t j = 0; j < top_k; ++j) {
      const uint32_t p = static_cast<uint32_t>(picked[j]);
      sel[j] = cluster.map_shared_rank(score, p / kRouterWarps)[p % kRouterWarps];
    }
    float sum = 0.f;
    for (uint32_t j = 0; j < top_k; ++j) sum += sel[j];
    const float denom = sum + 1e-20f;
    for (uint32_t j = 0; j < top_k; ++j) {
      ids_out[j] = picked[j];
      w_out[j] = sel[j] / denom * scaling;
    }
  }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kRouterThreads)
    eidola_router_topk(int32_t* __restrict__ topk_ids, float* __restrict__ topk_w,
                       const float* __restrict__ x, const uint16_t* __restrict__ w,
                       const float* __restrict__ bias, uint32_t hidden, uint32_t experts,
                       uint32_t top_k, float scaling) {
  namespace cg = cooperative_groups;
  __shared__ float xs[kRouterMaxHidden];
  __shared__ float score[kRouterWarps];
  __shared__ float choice[kRouterWarps];
  const cg::cluster_group cluster = cg::this_cluster();
  // A launch without the cluster attribute would score a fraction of the
  // experts and select among unwritten memory.
  if (cluster.num_blocks() != kRouterCluster) __trap();
  const uint32_t t = blockIdx.x / kRouterCluster, rank = cluster.block_rank();
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const float* xr = x + static_cast<size_t>(t) * hidden;
  for (uint32_t i = threadIdx.x; i < hidden; i += kRouterThreads) xs[i] = xr[i];
  __syncthreads();
  const uint32_t e = rank * kRouterWarps + warp;
  if (e < experts) {
    const uint16_t* wr = w + static_cast<size_t>(e) * hidden;
    float acc = 0.f;
    uint32_t i = lane;
    for (; i + 32 * (kRouterBatch - 1) < hidden; i += 32 * kRouterBatch) {
      uint16_t wv[kRouterBatch];
#pragma unroll
      for (uint32_t b = 0; b < kRouterBatch; ++b) wv[b] = __ldg(wr + i + 32 * b);
#pragma unroll
      for (uint32_t b = 0; b < kRouterBatch; ++b) acc = __fmaf_rn(xs[i + 32 * b], bf16f(wv[b]), acc);
    }
    for (; i < hidden; i += 32) acc = __fmaf_rn(xs[i], bf16f(__ldg(wr + i)), acc);
    acc = warp_sum(acc);
    if (lane == 0) {
      const float s = 1.f / (1.f + expf(-acc));
      score[warp] = s;
      choice[warp] = s + bias[e];
    }
  }
  cluster.sync();
  if (rank == 0 && warp == 0) {
    // Lane l holds experts 32q + l for q < kRouterCluster (block q's slot l).
    float c[kRouterCluster];
    uint32_t valid = 0;
#pragma unroll
    for (uint32_t q = 0; q < kRouterCluster; ++q) {
      c[q] = 0.f;
      if (q * kRouterWarps + lane < experts) {
        c[q] = cluster.map_shared_rank(choice, q)[lane];
        valid |= 1u << q;
      }
    }
    int32_t picked[8];
    for (uint32_t j = 0; j < top_k; ++j) {
      // The lowest untaken expert, and its choice.
      // `valid` loses an expert's bit when it is taken.
      uint32_t low = kNone;
#pragma unroll
      for (uint32_t q = 0; q < kRouterCluster; ++q)
        if ((valid >> q & 1u) && low == kNone) low = q * kRouterWarps + lane;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) low = min(low, __shfl_xor_sync(0xffffffffu, low, o));
      float low_c = 0.f;
#pragma unroll
      for (uint32_t q = 0; q < kRouterCluster; ++q)
        if (q == low / kRouterWarps) low_c = c[q];
      low_c = __shfl_sync(0xffffffffu, low_c, low % kRouterWarps);
      uint32_t best = low;
      if (!isnan(low_c)) {
        float bv = 0.f;
        uint32_t bi = kNone;
#pragma unroll
        for (uint32_t q = 0; q < kRouterCluster; ++q) {
          const uint32_t ei = q * kRouterWarps + lane;
          if ((valid >> q & 1u) && !isnan(c[q]) && router_before(c[q], ei, bv, bi)) {
            bv = c[q];
            bi = ei;
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
          const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (router_before(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
          }
        }
        best = bi;
      }
      picked[j] = static_cast<int32_t>(best);
      if (lane == best % kRouterWarps) valid &= ~(1u << (best / kRouterWarps));
    }
    if (lane == 0) {
      // Ascending expert order.
      for (uint32_t a = 1; a < top_k; ++a)
        for (uint32_t b = a; b > 0 && picked[b - 1] > picked[b]; --b) {
          const int32_t tmp = picked[b];
          picked[b] = picked[b - 1];
          picked[b - 1] = tmp;
        }
      float sel[8];
      for (uint32_t j = 0; j < top_k; ++j) {
        const uint32_t p = static_cast<uint32_t>(picked[j]);
        sel[j] = cluster.map_shared_rank(score, p / kRouterWarps)[p % kRouterWarps];
      }
      float sum = 0.f;
      for (uint32_t j = 0; j < top_k; ++j) sum += sel[j];
      const float denom = sum + 1e-20f;
      for (uint32_t j = 0; j < top_k; ++j) {
        topk_ids[t * top_k + j] = picked[j];
        topk_w[t * top_k + j] = sel[j] / denom * scaling;
      }
    }
  }
  // Every block's scores stay readable until rank 0 has read them.
  cluster.sync();
}
EIDOLA_KERNEL_META(eidola_router_topk, kRouterThreads, 1, 1, 0, kRouterCluster, 1, 1, 0);

// The tiled form: grid kRouterCluster * ceil(tokens / kRouterTile); x and w
// 16-byte aligned (the host checks). Warp v of rank b owns experts
// 32b + 8v .. 32b + 8v + 7; its lane l runs, for every token of the tile and
// each of those experts, the chain over i = l, l + 32, ..., in ascending i
// across the chunks.
extern "C" __global__ void __launch_bounds__(kRouterTiledThreads)
    eidola_router_topk_tiled(int32_t* __restrict__ topk_ids, float* __restrict__ topk_w,
                             const float* __restrict__ x, const uint16_t* __restrict__ w,
                             const float* __restrict__ bias, uint32_t tokens, uint32_t hidden,
                             uint32_t experts, uint32_t top_k, float scaling) {
  namespace cg = cooperative_groups;
  __shared__ __align__(16) float xs_stage[kRouterStages][kRouterTile][kRouterChunk];
  __shared__ __align__(16) uint16_t ws_stage[kRouterStages][kRouterWarps][kRouterChunk];
  __shared__ float score[kRouterTile][kRouterWarps];
  __shared__ float choice[kRouterTile][kRouterWarps];
  const cg::cluster_group cluster = cg::this_cluster();
  if (cluster.num_blocks() != kRouterCluster) __trap();
  const uint32_t t0 = blockIdx.x / kRouterCluster * kRouterTile, rank = cluster.block_rank();
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t live = min(kRouterTile, tokens - t0);
  const uint32_t e0 = rank * kRouterWarps;
  // Chunk c of the tile's rows and the block's weight rows into stage s. Rows
  // past the tokens or the experts are not loaded, and nothing reads them.
  auto load = [&](uint32_t c, uint32_t s) {
    float* xs = &xs_stage[s][0][0];
    uint16_t* ws = &ws_stage[s][0][0];
    const uint32_t i0 = c * kRouterChunk;
    for (uint32_t v = threadIdx.x; v < kRouterTile * kRouterChunk / 4; v += kRouterTiledThreads) {
      const uint32_t r = v / (kRouterChunk / 4), col = v % (kRouterChunk / 4) * 4;
      if (r < live) cp_async16(xs + r * kRouterChunk + col, x + static_cast<size_t>(t0 + r) * hidden + i0 + col);
    }
    for (uint32_t v = threadIdx.x; v < kRouterWarps * kRouterChunk / 8; v += kRouterTiledThreads) {
      const uint32_t r = v / (kRouterChunk / 8), col = v % (kRouterChunk / 8) * 8;
      if (e0 + r < experts)
        cp_async16(ws + r * kRouterChunk + col, w + static_cast<size_t>(e0 + r) * hidden + i0 + col);
    }
  };
  float acc[kRouterTile][kRouterWarpExperts];
#pragma unroll
  for (uint32_t r = 0; r < kRouterTile; ++r)
#pragma unroll
    for (uint32_t q = 0; q < kRouterWarpExperts; ++q) acc[r][q] = 0.f;
  const uint32_t chunks = hidden / kRouterChunk;
#pragma unroll
  for (uint32_t s = 0; s + 1 < kRouterStages; ++s) {
    if (s < chunks) load(s, s);
    cp_async_commit();
  }
  for (uint32_t c = 0; c < chunks; ++c) {
    // Chunk c's group is complete (one group per chunk, committed in order),
    // and every thread is done with chunk c - 1, whose stage is refilled next.
    cp_async_wait<kRouterStages - 2>();
    __syncthreads();
    if (c + kRouterStages - 1 < chunks) load(c + kRouterStages - 1, (c + kRouterStages - 1) % kRouterStages);
    cp_async_commit();
    const float* xs = &xs_stage[c % kRouterStages][0][0];
    const uint16_t* ws = &ws_stage[c % kRouterStages][warp * kRouterWarpExperts][0];
#pragma unroll
    for (uint32_t jj = 0; jj < kRouterChunk / 32; ++jj) {
      const uint32_t i = lane + 32 * jj;
      float xv[kRouterTile], wv[kRouterWarpExperts];
#pragma unroll
      for (uint32_t r = 0; r < kRouterTile; ++r) xv[r] = xs[r * kRouterChunk + i];
#pragma unroll
      for (uint32_t q = 0; q < kRouterWarpExperts; ++q) wv[q] = bf16f(ws[q * kRouterChunk + i]);
#pragma unroll
      for (uint32_t r = 0; r < kRouterTile; ++r)
#pragma unroll
        for (uint32_t q = 0; q < kRouterWarpExperts; ++q) acc[r][q] = __fmaf_rn(xv[r], wv[q], acc[r][q]);
    }
  }
  // Each chain's butterfly; lane 0's sum, as the one-token form takes it.
  // The sigmoids are spread over the lanes.
#pragma unroll
  for (uint32_t r = 0; r < kRouterTile; ++r)
#pragma unroll
    for (uint32_t q = 0; q < kRouterWarpExperts; ++q) {
      const float logit = __shfl_sync(0xffffffffu, warp_sum(acc[r][q]), 0);
      const uint32_t el = warp * kRouterWarpExperts + q;
      if (lane == (r * kRouterWarpExperts + q) % 32 && e0 + el < experts) {
        const float s = router_sigmoid(logit);
        score[r][el] = s;
        choice[r][el] = s + bias[e0 + el];
      }
    }
  cluster.sync();
  if (rank == 0)
    for (uint32_t r = warp; r < live; r += kRouterTiledWarps)
      router_select(cluster, choice[r], score[r], lane, experts, top_k, scaling,
                    topk_ids + static_cast<size_t>(t0 + r) * top_k,
                    topk_w + static_cast<size_t>(t0 + r) * top_k);
  cluster.sync();
}
EIDOLA_KERNEL_META(eidola_router_topk_tiled, kRouterTiledThreads, 1, 1, 0, kRouterCluster, 1, 1, 0);

// The split form's geometry. A block scores kRouterScoresTile tokens x
// kRouterScoresExperts experts with 4 warps; warp v owns the block's experts
// 4v .. 4v + 3 for every token of the tile, so each lane runs 32 chains, one
// per lane again after the butterflies. A stage holds one chunk of the tile's
// rows (f32) and the block's weight rows (BF16): 8 KiB.
constexpr uint32_t kRouterScoresTile = 8;
constexpr uint32_t kRouterScoresWarps = 4;
constexpr uint32_t kRouterScoresThreads = kRouterScoresWarps * 32;
constexpr uint32_t kRouterScoresWarpExperts = 4;
constexpr uint32_t kRouterScoresExperts = kRouterScoresWarps * kRouterScoresWarpExperts;
constexpr uint32_t kRouterScoresStages = 8;
constexpr uint32_t kRouterScoresStageXBytes = kRouterScoresTile * kRouterChunk * 4;
constexpr uint32_t kRouterScoresStageBytes = kRouterScoresStageXBytes + kRouterScoresExperts * kRouterChunk * 2;
constexpr uint32_t kRouterScoresSmem = kRouterScoresStages * kRouterScoresStageBytes;
static_assert(kRouterScoresTile * kRouterScoresWarpExperts == 32, "one chain per lane after the butterflies");
// Experts a select lane holds: lane l holds experts 32q + l.
constexpr uint32_t kRouterLaneExperts = 256 / 32;
constexpr uint32_t kRouterSelectWarps = 4;

// Scores of every (token, expert): `out` is [2][tokens][experts] f32, the
// choices (score + bias) then the scores. Grid (ceil(experts / 16),
// ceil(tokens / 8)); x and w 16-byte aligned (the host checks). Lane l of
// warp v in block (b, y) runs, for token 8y + r and expert 16b + 4v + q, the
// chain over i = l, l + 32, ..., in ascending i across the chunks: the same
// chains, butterfly, lane-0 sum and sigmoid as the cluster forms.
extern "C" __global__ void __launch_bounds__(kRouterScoresThreads)
    eidola_router_scores(float* __restrict__ out, const float* __restrict__ x,
                         const uint16_t* __restrict__ w, const float* __restrict__ bias,
                         uint32_t tokens, uint32_t hidden, uint32_t experts) {
  extern __shared__ __align__(16) uint8_t router_smem[];
  const uint32_t e0 = blockIdx.x * kRouterScoresExperts, t0 = blockIdx.y * kRouterScoresTile;
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t live = min(kRouterScoresTile, tokens - t0);
  auto stage_x = [&](uint32_t s) {
    return reinterpret_cast<float*>(router_smem + s * kRouterScoresStageBytes);
  };
  auto stage_w = [&](uint32_t s) {
    return reinterpret_cast<uint16_t*>(router_smem + s * kRouterScoresStageBytes + kRouterScoresStageXBytes);
  };
  // Chunk c of the tile's rows and the block's weight rows into stage s. Rows
  // past the tokens or the experts are not loaded, and no output reads them.
  auto load = [&](uint32_t c, uint32_t s) {
    float* xs = stage_x(s);
    uint16_t* ws = stage_w(s);
    const uint32_t i0 = c * kRouterChunk;
    for (uint32_t v = threadIdx.x; v < kRouterScoresTile * kRouterChunk / 4; v += kRouterScoresThreads) {
      const uint32_t r = v / (kRouterChunk / 4), col = v % (kRouterChunk / 4) * 4;
      if (r < live) cp_async16(xs + r * kRouterChunk + col, x + static_cast<size_t>(t0 + r) * hidden + i0 + col);
    }
    for (uint32_t v = threadIdx.x; v < kRouterScoresExperts * kRouterChunk / 8; v += kRouterScoresThreads) {
      const uint32_t r = v / (kRouterChunk / 8), col = v % (kRouterChunk / 8) * 8;
      if (e0 + r < experts)
        cp_async16(ws + r * kRouterChunk + col, w + static_cast<size_t>(e0 + r) * hidden + i0 + col);
    }
  };
  float acc[kRouterScoresTile][kRouterScoresWarpExperts];
#pragma unroll
  for (uint32_t r = 0; r < kRouterScoresTile; ++r)
#pragma unroll
    for (uint32_t q = 0; q < kRouterScoresWarpExperts; ++q) acc[r][q] = 0.f;
  const uint32_t chunks = hidden / kRouterChunk;
#pragma unroll
  for (uint32_t s = 0; s + 1 < kRouterScoresStages; ++s) {
    if (s < chunks) load(s, s);
    cp_async_commit();
  }
  for (uint32_t c = 0; c < chunks; ++c) {
    // Chunk c's group is complete (one group per chunk, committed in order),
    // and every thread is done with chunk c - 1, whose stage is refilled next.
    cp_async_wait<kRouterScoresStages - 2>();
    __syncthreads();
    if (c + kRouterScoresStages - 1 < chunks)
      load(c + kRouterScoresStages - 1, (c + kRouterScoresStages - 1) % kRouterScoresStages);
    cp_async_commit();
    const float* xs = stage_x(c % kRouterScoresStages);
    const uint16_t* ws = stage_w(c % kRouterScoresStages) + warp * kRouterScoresWarpExperts * kRouterChunk;
#pragma unroll
    for (uint32_t jj = 0; jj < kRouterChunk / 32; ++jj) {
      const uint32_t i = lane + 32 * jj;
      float xv[kRouterScoresTile], wv[kRouterScoresWarpExperts];
#pragma unroll
      for (uint32_t r = 0; r < kRouterScoresTile; ++r) xv[r] = xs[r * kRouterChunk + i];
#pragma unroll
      for (uint32_t q = 0; q < kRouterScoresWarpExperts; ++q) wv[q] = bf16f(ws[q * kRouterChunk + i]);
#pragma unroll
      for (uint32_t r = 0; r < kRouterScoresTile; ++r)
#pragma unroll
        for (uint32_t q = 0; q < kRouterScoresWarpExperts; ++q) acc[r][q] = __fmaf_rn(xv[r], wv[q], acc[r][q]);
    }
  }
  // Each chain's butterfly; lane 0's sum, as the cluster forms take it. Chain
  // (r, q) goes to lane 4r + q, so every lane computes one sigmoid.
  float logit = 0.f;
#pragma unroll
  for (uint32_t r = 0; r < kRouterScoresTile; ++r)
#pragma unroll
    for (uint32_t q = 0; q < kRouterScoresWarpExperts; ++q) {
      const float v = __shfl_sync(0xffffffffu, warp_sum(acc[r][q]), 0);
      if (lane == r * kRouterScoresWarpExperts + q) logit = v;
    }
  const uint32_t r = lane / kRouterScoresWarpExperts;
  const uint32_t e = e0 + warp * kRouterScoresWarpExperts + lane % kRouterScoresWarpExperts;
  if (r < live && e < experts) {
    const float s = router_sigmoid(logit);
    const size_t o = static_cast<size_t>(t0 + r) * experts + e;
    out[o] = s + bias[e];
    out[static_cast<size_t>(tokens) * experts + o] = s;
  }
}
EIDOLA_KERNEL_META(eidola_router_scores, kRouterScoresThreads, 1, 1, kRouterScoresSmem, 1, 1, 1, 0);

// The selection of eidola_router_scores' output: one warp per token, grid
// ceil(tokens / kRouterSelectWarps). Each round takes the untaken expert the
// cluster forms' rank-0 warp takes (the same code over the same choices),
// the k picks are put in ascending order by a sorting network (they are
// distinct), and the weights are the same expressions over the scores in that
// order.
extern "C" __global__ void __launch_bounds__(kRouterSelectWarps * 32)
    eidola_router_select(int32_t* __restrict__ topk_ids, float* __restrict__ topk_w,
                         const float* __restrict__ scores, uint32_t tokens, uint32_t experts,
                         uint32_t top_k, float scaling) {
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t t = blockIdx.x * kRouterSelectWarps + warp;
  // Uniform over the warp.
  if (t >= tokens) return;
  const float* choice = scores + static_cast<size_t>(t) * experts;
  const float* score = choice + static_cast<size_t>(tokens) * experts;
  float c[kRouterLaneExperts];
  uint32_t valid = 0;
#pragma unroll
  for (uint32_t q = 0; q < kRouterLaneExperts; ++q) {
    c[q] = 0.f;
    if (q * 32 + lane < experts) {
      c[q] = choice[q * 32 + lane];
      valid |= 1u << q;
    }
  }
  // Every lane ends each round with the same pick; unused slots sort last.
  uint32_t picked[8];
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    picked[j] = kNone;
    if (j >= top_k) continue;
    // The lowest untaken expert, and its choice.
    uint32_t low = kNone;
#pragma unroll
    for (uint32_t q = 0; q < kRouterLaneExperts; ++q)
      if ((valid >> q & 1u) && low == kNone) low = q * 32 + lane;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) low = min(low, __shfl_xor_sync(0xffffffffu, low, o));
    float low_c = 0.f;
#pragma unroll
    for (uint32_t q = 0; q < kRouterLaneExperts; ++q)
      if (q == low / 32) low_c = c[q];
    low_c = __shfl_sync(0xffffffffu, low_c, low % 32);
    uint32_t best = low;
    if (!isnan(low_c)) {
      float bv = 0.f;
      uint32_t bi = kNone;
#pragma unroll
      for (uint32_t q = 0; q < kRouterLaneExperts; ++q) {
        const uint32_t ei = q * 32 + lane;
        if ((valid >> q & 1u) && !isnan(c[q]) && router_before(c[q], ei, bv, bi)) {
          bv = c[q];
          bi = ei;
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (router_before(ov, oi, bv, bi)) {
          bv = ov;
          bi = oi;
        }
      }
      best = bi;
    }
    picked[j] = best;
    if (lane == best % 32) valid &= ~(1u << (best / 32));
  }
  // Ascending expert order: Batcher's odd-even merge sort of 8 (19
  // compare-exchanges), in registers.
  auto cx = [&](uint32_t a, uint32_t b) {
    const uint32_t lo = min(picked[a], picked[b]), hi = max(picked[a], picked[b]);
    picked[a] = lo;
    picked[b] = hi;
  };
  cx(0, 1); cx(2, 3); cx(4, 5); cx(6, 7);
  cx(0, 2); cx(1, 3); cx(4, 6); cx(5, 7);
  cx(1, 2); cx(5, 6);
  cx(0, 4); cx(1, 5); cx(2, 6); cx(3, 7);
  cx(2, 4); cx(3, 5);
  cx(1, 2); cx(3, 4); cx(5, 6);
  if (lane == 0) {
    float sel[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) sel[j] = j < top_k ? score[picked[j]] : 0.f;
    float sum = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j)
      if (j < top_k) sum += sel[j];
    const float denom = sum + 1e-20f;
    int32_t* ids_out = topk_ids + static_cast<size_t>(t) * top_k;
    float* w_out = topk_w + static_cast<size_t>(t) * top_k;
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j)
      if (j < top_k) {
        ids_out[j] = static_cast<int32_t>(picked[j]);
        w_out[j] = sel[j] / denom * scaling;
      }
  }
}
EIDOLA_KERNEL_META(eidola_router_select, kRouterSelectWarps * 32, 1, 1, 0, 1, 1, 1, 0);

// Expert-major placement of the T*k routed (token, slot) pairs, deterministic
// (pairs ascending within an expert, i.e. tokens ascending). One block of
// 256 threads: thread e owns expert e's count and start, and warp w places
// the w-th eighth of the pairs.
//
// Expert e's rows start at the previous expert's end rounded up to 128 rows,
// in expert order; grouped_layout[e] = the end of expert e's rows
// (DeepGEMM's psum layout). An expert's rows past its pairs, to the next
// multiple of 128, are padding.
//
// row_of[t*k + j] = the row (token t, slot j) landed in; the gather, the
// SwiGLU and the combine address rows through it. An id outside 0..256 is
// placed nowhere (its row_of entry is left as it is); the router writes none.
//
// Each warp counts its eighth into its own histogram (shared atomics: the
// counts do not depend on their order), thread e turns the histograms into
// each warp's first row for expert e, and each warp then walks its eighth in
// order 32 pairs at a time: a pair's row is its warp's next row for its
// expert plus the lanes before it holding the same expert.
constexpr uint32_t kPermuteExperts = 256;
constexpr uint32_t kPermuteWarps = kPermuteExperts / 32;

extern "C" __global__ void __launch_bounds__(kPermuteExperts)
    eidola_moe_permute(int32_t* __restrict__ grouped_layout, int32_t* __restrict__ row_of,
                       const int32_t* __restrict__ topk_ids, uint32_t tokens, uint32_t top_k) {
  __shared__ uint32_t next[kPermuteWarps][kPermuteExperts];
  __shared__ uint32_t scan[kPermuteExperts];
  const uint32_t e = threadIdx.x, warp = e / 32, lane = e % 32;
  const uint32_t n = tokens * top_k;
  const uint32_t seg = (n + kPermuteWarps - 1) / kPermuteWarps;
  const uint32_t lo = min(warp * seg, n), hi = min(lo + seg, n);
#pragma unroll
  for (uint32_t w = 0; w < kPermuteWarps; ++w) next[w][e] = 0;
  __syncthreads();
  for (uint32_t i = lo + lane; i < hi; i += 32) {
    const uint32_t id = static_cast<uint32_t>(topk_ids[i]);
    if (id < kPermuteExperts) atomicAdd(&next[warp][id], 1u);
  }
  __syncthreads();
  // Expert e's count, and each warp's offset inside the expert's rows.
  uint32_t count = 0;
#pragma unroll
  for (uint32_t w = 0; w < kPermuteWarps; ++w) {
    const uint32_t h = next[w][e];
    next[w][e] = count;
    count += h;
  }
  // Starts: an exclusive scan of the runs.
  scan[e] = (count + 127) / 128 * 128;
  __syncthreads();
  for (uint32_t d = 1; d < kPermuteExperts; d *= 2) {
    const uint32_t add = e >= d ? scan[e - d] : 0;
    __syncthreads();
    scan[e] += add;
    __syncthreads();
  }
  const uint32_t run = (count + 127) / 128 * 128;
  const uint32_t start = scan[e] - run;
  grouped_layout[e] = static_cast<int32_t>(start + count);
#pragma unroll
  for (uint32_t w = 0; w < kPermuteWarps; ++w) next[w][e] += start;
  __syncthreads();
  for (uint32_t base = lo; base < hi; base += 32) {
    const uint32_t i = base + lane;
    const uint32_t id = i < hi ? static_cast<uint32_t>(topk_ids[i]) : kPermuteExperts;
    const bool placed = id < kPermuteExperts;
    const uint32_t same = __match_any_sync(0xffffffffu, placed ? id : kPermuteExperts);
    const uint32_t before = __popc(same & ((1u << lane) - 1u));
    if (placed) row_of[i] = static_cast<int32_t>(next[warp][id] + before);
    __syncwarp();
    if (placed && before == 0) next[warp][id] += __popc(same);
    __syncwarp();
  }
}
EIDOLA_KERNEL_META(eidola_moe_permute, 256, 1, 1, 0, 1, 1, 1, 0);

// Gather f32 token rows into DeepGEMM's FP8 A with packed UE8M0 scales:
// a [rows][K], sf (see sfa_index). Every routed (token, slot) pair i gets
// token i / top_k's row in row row_of[i]. Rows no pair names (the expert
// layout's padding) are not touched; see eidola_swiglu_quant_fp8_ue8m0.
//
// Grid (tokens, K/512), 4 warps: block (t, w) quantizes token t's 512-wide
// word w once (each warp one 128-wide group, each lane 4 consecutive
// elements: a 16-byte load, a 4-byte store) and stores it into the row of
// every one of the token's pairs. A token's pairs carry the same row, so the
// codes and scale word are the ones a block per pair would compute. x must be
// 16-byte aligned and a 4-byte aligned (the host checks both).
extern "C" __global__ void __launch_bounds__(128)
    eidola_gather_quant_ue8m0(uint8_t* __restrict__ a, int32_t* __restrict__ sf,
                              const float* __restrict__ x, const int32_t* __restrict__ row_of,
                              uint32_t top_k, uint32_t k, uint32_t rows4) {
  __shared__ uint8_t exps[4];
  const uint32_t t = blockIdx.x, w = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t g = w * 4 + warp;
  const float4 x4 = *reinterpret_cast<const float4*>(x + static_cast<size_t>(t) * k + g * 128 + lane * 4);
  const float v[4] = {x4.x, x4.y, x4.z, x4.w};
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(v[j]));
  amax = warp_max(amax);
  const uint8_t e = ue8m0_for(amax);
  const float inv = 1.f / ue8m0_value(e);
  uint32_t codes = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) codes |= static_cast<uint32_t>(f2e4m3(v[j] * inv)) << (8 * j);
  if (lane == 0) exps[warp] = e;
  __syncthreads();
  const uint32_t word = exps[0] | (exps[1] << 8) | (exps[2] << 16) | (static_cast<uint32_t>(exps[3]) << 24);
  const int32_t* rows = row_of + static_cast<size_t>(t) * top_k;
  for (uint32_t j = 0; j < top_k; ++j) {
    const uint32_t r = static_cast<uint32_t>(rows[j]);
    *reinterpret_cast<uint32_t*>(a + static_cast<size_t>(r) * k + g * 128 + lane * 4) = codes;
    if (threadIdx.x == 0) sf[sfa_index(r, w, rows4)] = static_cast<int32_t>(word);
  }
}
EIDOLA_KERNEL_META(eidola_gather_quant_ue8m0, 128, 1, 1, 0, 1, 1, 1, 0);

// out[t] = sum over slots j (experts ascending) of w[t][j] * d[row_of[t][j]]
// (d BF16 [rows][H]), accumulated in f32: per element, acc = 0 then
// acc = fma(w[t][j], d, acc) for j ascending.
//
// Grid (tokens, H / kCombineWidth), kCombineThreads threads, each owning
// kCombineVec consecutive elements of one token: it loads the token's k rows'
// 16-byte pieces at once, then runs each element's chain. H is a multiple of
// kCombineWidth and d, out are 16-byte aligned (the host checks both).
constexpr uint32_t kCombineThreads = 128;
constexpr uint32_t kCombineVec = 8;
constexpr uint32_t kCombineWidth = kCombineThreads * kCombineVec;
constexpr uint32_t kCombineMaxK = 8;

extern "C" __global__ void __launch_bounds__(kCombineThreads)
    eidola_moe_combine(float* __restrict__ out, const uint16_t* __restrict__ d,
                       const int32_t* __restrict__ row_of, const float* __restrict__ topk_w,
                       uint32_t hidden, uint32_t top_k) {
  const uint32_t t = blockIdx.x;
  const uint32_t i0 = blockIdx.y * kCombineWidth + threadIdx.x * kCombineVec;
  uint4 dv[kCombineMaxK];
  float wv[kCombineMaxK];
#pragma unroll
  for (uint32_t j = 0; j < kCombineMaxK; ++j) {
    if (j < top_k) {
      const int32_t row = row_of[t * top_k + j];
      wv[j] = topk_w[t * top_k + j];
      dv[j] = *reinterpret_cast<const uint4*>(d + static_cast<size_t>(row) * hidden + i0);
    }
  }
  float acc[kCombineVec];
#pragma unroll
  for (uint32_t e = 0; e < kCombineVec; ++e) acc[e] = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kCombineMaxK; ++j) {
    if (j < top_k) {
      const uint32_t words[4] = {dv[j].x, dv[j].y, dv[j].z, dv[j].w};
#pragma unroll
      for (uint32_t e = 0; e < kCombineVec; ++e) {
        const uint16_t b = static_cast<uint16_t>(words[e / 2] >> (16 * (e % 2)));
        acc[e] = __fmaf_rn(wv[j], bf16f(b), acc[e]);
      }
    }
  }
  float4* dst = reinterpret_cast<float4*>(out + static_cast<size_t>(t) * hidden + i0);
  dst[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  dst[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}
EIDOLA_KERNEL_META(eidola_moe_combine, kCombineThreads, 1, 1, 0, 1, 1, 1, 0);

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

// Row copies between buffers (the speculative step's gathers and scatters
// of hidden states, drafter state, boundary taps and token ids). Buffer `b`
// holds `rows[b]` rows of `width` 32-bit words; item i copies row
// `src_row[i]` of buffer `src_buf[i]` to row `dst_row[i]` of buffer
// `dst_buf[i]`, word for word. Every index is bounded here: an item naming a
// buffer past kCopyBuffers or a row past its buffer copies nothing and sets
// kStatusBadIndex in `status`. Items must not overlap (no item's destination
// row is another item's source or destination row): the host plans them so.
//
// Grid (items, ceil(width / kThreads)): one thread per word.
namespace {
constexpr uint32_t kCopyBuffers = 8;
constexpr uint32_t kStatusBadIndex = 4;
}  // namespace

struct EidolaCopyArgs {
  uint32_t* buf[kCopyBuffers];
  uint64_t rows[kCopyBuffers];
  const uint32_t* src_buf;
  const uint32_t* src_row;
  const uint32_t* dst_buf;
  const uint32_t* dst_row;
  uint32_t* status;
  uint32_t width;
  uint32_t items;
};
static_assert(sizeof(EidolaCopyArgs) == 176, "host mirror layout");

extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_copy_rows(const __grid_constant__ EidolaCopyArgs a) {
  const uint32_t i = blockIdx.x;
  if (i >= a.items) return;
  const uint32_t sb = a.src_buf[i], db = a.dst_buf[i];
  const uint32_t sr = a.src_row[i], dr = a.dst_row[i];
  if (sb >= kCopyBuffers || db >= kCopyBuffers || sr >= a.rows[sb] || dr >= a.rows[db]) {
    if (blockIdx.y == 0 && threadIdx.x == 0) atomicOr(a.status, kStatusBadIndex);
    return;
  }
  const uint32_t w = blockIdx.y * kThreads + threadIdx.x;
  if (w >= a.width) return;
  a.buf[db][static_cast<uint64_t>(dr) * a.width + w] =
      a.buf[sb][static_cast<uint64_t>(sr) * a.width + w];
}
EIDOLA_KERNEL_META(eidola_copy_rows, kThreads, 1, 1, 0, 1, 1, 1, sizeof(EidolaCopyArgs));
