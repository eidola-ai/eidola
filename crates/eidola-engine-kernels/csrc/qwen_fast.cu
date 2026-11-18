// Qwen3.8's fast-profile kernels of our own: the small-N BF16 GEMM
// (`qwen_fast.cuh`, whose body the host check runs too: it spells out every
// fused multiply-add it means, `qwf::fma32`) and the element-wise gates in
// f32. Built in the `own` profile (contraction allowed).

#include "eidola_kernel.cuh"
#include "qwen_fast.cuh"

extern "C" __global__ void __launch_bounds__(qwf::kSmallWarps * qwf::kWarp)
    eidola_qwen_fast_gemm_bf16(const __grid_constant__ QwLinearParams p) {
  const uint32_t l = threadIdx.x % qwf::kWarp;
  uint32_t n, r0, rows;
  // Uniform across the warp: the output is the warp's.
  if (!qwf::small_gemm_unit(p, blockIdx.x, blockIdx.y, threadIdx.x / qwf::kWarp, &n, &r0, &rows))
    return;
  float acc[qwf::kSmallRows];
  qwf::small_gemm_lane(p, n, r0, rows, l, acc);
#pragma unroll
  for (uint32_t r = 0; r < qwf::kSmallRows; ++r) {
    for (uint32_t off = qwf::kWarp / 2; off > 0; off >>= 1) {
      acc[r] = acc[r] + __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
  }
  if (l == 0) qwf::small_gemm_store(p, n, r0, rows, acc);
}
EIDOLA_KERNEL_META(eidola_qwen_fast_gemm_bf16, qwf::kSmallWarps * qwf::kWarp, 1, 1, 0, 1, 1, 1,
                   sizeof(QwLinearParams));

// The tiled form (`qwen_fast.cuh`): a warp per `kTileRows` rows by `kTileOuts`
// outputs, the same sums in the same order, the same butterfly.
extern "C" __global__ void __launch_bounds__(qwf::kTileWarps * qwf::kWarp)
    eidola_qwen_fast_gemm_bf16_tiled(const __grid_constant__ QwLinearParams p) {
  const uint32_t l = threadIdx.x % qwf::kWarp;
  uint32_t n0, outs, r0, rows;
  // Uniform across the warp: the tile is the warp's.
  if (!qwf::tile_gemm_unit(p, blockIdx.x, blockIdx.y, threadIdx.x / qwf::kWarp, &n0, &outs, &r0,
                           &rows))
    return;
  float acc[qwf::kTileRows * qwf::kTileOuts];
  qwf::tile_gemm_lane(p, n0, outs, r0, rows, l, acc);
#pragma unroll
  for (uint32_t i = 0; i < qwf::kTileRows * qwf::kTileOuts; ++i) {
    for (uint32_t off = qwf::kWarp / 2; off > 0; off >>= 1) {
      acc[i] = acc[i] + __shfl_xor_sync(0xffffffffu, acc[i], off);
    }
  }
  if (l == 0) qwf::tile_gemm_store(p, n0, outs, r0, rows, acc);
}
EIDOLA_KERNEL_META(eidola_qwen_fast_gemm_bf16_tiled, qwf::kTileWarps * qwf::kWarp, 1, 1, 0, 1, 1,
                   1, sizeof(QwLinearParams));

// ---- the gates in f32 (`eidola_qwen_fast_gate`) ----
//
// The exact gates (`eidola_qwen_gate`) take the reference's sigmoid and SiLU
// in f64 (`det_exp` and an f64 division): at a prefill's 2,048 × 17,408
// elements that is about 100 µs of f64 arithmetic, six times the memory
// time. The fast profile computes them in f32 with the hardware exponential,
// `s = g / (1 + exp(-g))` (SiLU) or `1 / (1 + exp(-g))` (sigmoid), then rounds
// exactly as the exact gates do: `bf16(bf16(s) · a)` and `bf16(a · bf16(s))`.
// Each element is a function of its own `a` and `g`; a thread takes 8
// consecutive elements of a row (16-byte loads and store when every stride,
// the unit and the width are multiples of 8 and the arrays 16-byte aligned,
// one element at a time otherwise).

namespace {

constexpr uint32_t kGateThreads = 256;
constexpr uint32_t kGateChunk = 8;

__device__ __forceinline__ float gate_value(uint16_t b) {
  return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

// Nearest BF16, ties to even, a NaN stored as the one quiet NaN: the exact
// kernels' `bf16_round` and `bf16_store`.
__device__ __forceinline__ float gate_round(float x) {
  if (x != x) return x;
  const uint32_t b = __float_as_uint(x);
  return __uint_as_float((b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u);
}
__device__ __forceinline__ uint16_t gate_store(float x) {
  return x != x ? uint16_t{0x7fc0} : static_cast<uint16_t>(__float_as_uint(x) >> 16);
}

__device__ __forceinline__ uint16_t gate_one(uint32_t kind, uint16_t ab, uint16_t gb) {
  const float a = gate_value(ab);
  const float g = gate_value(gb);
  const float e = 1.0f + __expf(-g);
  if (kind == QW_GATE_SWIGLU) return gate_store(gate_round(gate_round(__fdiv_rn(g, e)) * a));
  return gate_store(gate_round(a * gate_round(__fdiv_rn(1.0f, e))));
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kGateThreads)
    eidola_qwen_fast_gate(const __grid_constant__ QwGateParams p) {
  const uint32_t row = blockIdx.y;
  const uint32_t i0 = (blockIdx.x * kGateThreads + threadIdx.x) * kGateChunk;
  if (row >= p.rows || i0 >= p.width) return;
  const uint16_t* a = p.a + static_cast<uint64_t>(row) * p.a_stride;
  const uint16_t* g = p.g + static_cast<uint64_t>(row) * p.g_stride;
  uint16_t* out = p.out + static_cast<uint64_t>(row) * p.out_stride;
  const bool vec = ((p.width | p.unit | p.a_stride | p.g_stride | p.g_unit_stride | p.out_stride) %
                        kGateChunk ==
                    0) &&
                   ((reinterpret_cast<uint64_t>(p.a) | reinterpret_cast<uint64_t>(p.g) |
                     reinterpret_cast<uint64_t>(p.out)) %
                        16 ==
                    0);
  if (vec) {
    // A chunk lies in one unit: the unit is a multiple of 8.
    const uint64_t gi = static_cast<uint64_t>(i0 / p.unit) * p.g_unit_stride + i0 % p.unit;
    const uint4 av = *reinterpret_cast<const uint4*>(a + i0);
    const uint4 gv = *reinterpret_cast<const uint4*>(g + gi);
    const uint32_t aw[4] = {av.x, av.y, av.z, av.w};
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    uint32_t ow[4];
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint16_t lo = gate_one(p.kind, static_cast<uint16_t>(aw[e] & 0xffffu),
                                   static_cast<uint16_t>(gw[e] & 0xffffu));
      const uint16_t hi = gate_one(p.kind, static_cast<uint16_t>(aw[e] >> 16),
                                   static_cast<uint16_t>(gw[e] >> 16));
      ow[e] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    *reinterpret_cast<uint4*>(out + i0) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    return;
  }
  uint16_t av[kGateChunk], gv[kGateChunk];
#pragma unroll
  for (uint32_t e = 0; e < kGateChunk; ++e) {
    const uint32_t i = i0 + e;
    if (i < p.width) {
      av[e] = a[i];
      gv[e] = g[static_cast<uint64_t>(i / p.unit) * p.g_unit_stride + i % p.unit];
    }
  }
#pragma unroll
  for (uint32_t e = 0; e < kGateChunk; ++e) {
    if (i0 + e < p.width) out[i0 + e] = gate_one(p.kind, av[e], gv[e]);
  }
}
EIDOLA_KERNEL_META(eidola_qwen_fast_gate, kGateThreads, 1, 1, 0, 1, 1, 1, sizeof(QwGateParams));
