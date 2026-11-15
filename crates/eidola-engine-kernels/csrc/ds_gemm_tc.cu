// DeepSeek-V4.1's quantised GEMMs on tensor cores, and the probe of the MMA
// they rest on. `ds_gemm_tc.cuh` states the forms and holds every piece a
// thread computes; this file places them on the grid: staging through
// shared memory with `cp.async`, the MMAs, the ordered sums of the split
// block, and the passes of `lin4`'s groups.
//
// Entries, each taking the scalar reference's parameter struct unchanged
// (`DsLin8Params`, `DsLin4Params`; `ds_reference.cu` is the definition):
//
//   eidola_ds_{lin8,lin4}_banded_{s8,t8,t32}     exact: banded block products
//   eidola_ds_{lin8,lin4}_tolerance_{s8,t8,t32}  one block product per block
//   eidola_ds_gemm_probe                         the MMA against the exact sum
//
// Grid `(ceil(n / features), ceil(rows / tokens))`, 128 threads; `lin4` takes
// `layout_rows` for `rows`:
//
//   _t8, _t32  8 rows by 64 outputs, 32 rows by 128: each warp owns 16 or 32
//              outputs and computes every block of every step, accumulating
//              in registers; steps of 4 blocks, double-buffered.
//   _s8        8 rows by 16 outputs: each warp computes a quarter of each
//              step's 16 blocks and writes each block's contribution
//              `(dot · activation scale) · weight scale` to shared memory;
//              then each thread adds its output's contributions in ascending
//              block order (the reference's accumulation, the same operations
//              in the same order); 6 stages (FP8) or 8 (FP4) in flight.
//
// Every output of a row is computed from that row's codes and scales and the
// weight, by the same operations whichever block shape holds it; the shape is
// chosen by the host from the row count.
//
// `lin4`'s tile may meet several groups' runs (any `row_align`): it makes one
// pass over K per group, each pass computing that group's rows of the tile
// against that group's weight. A `row_align` that is a multiple of 8 gives
// every 8-row tile at most one group. Padding rows are not written.

#include "ds_gemm_tc.cuh"
#include "eidola_kernel.cuh"

namespace {

using dstc::Cfg;
using dstc::kThreads;
using dstc::Pass;
using dstc::Stage;

// D = A · B over one half of a block's K (`mma.m16n8k16`, F16 operands, f32
// accumulator C).
__device__ __forceinline__ void mma16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                      const float (&c)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]));
}
// A block's product: both halves, chained from a zero accumulator.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[2][4],
                                    const uint32_t (&b)[2][2]) {
  const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float half[4];
  mma16(half, a[0], b[0], zero);
  mma16(d, a[1], b[1], half);
}

// 16 bytes from global to shared memory, or 16 zero bytes when `src` is
// null (`any` is then a valid global address that is not read).
__device__ __forceinline__ void copy16(uint8_t* dst, const uint8_t* src, const uint8_t* any) {
  const uint32_t to = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(to),
               "l"(src != nullptr ? src : any), "r"(src != nullptr ? 16u : 0u)
               : "memory");
}
// 4 bytes, or 4 zero bytes when `src` is null.
__device__ __forceinline__ void copy4(uint8_t* dst, const uint8_t* src, const uint8_t* any) {
  const uint32_t to = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(to),
               "l"(src != nullptr ? src : any), "r"(src != nullptr ? 4u : 0u)
               : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int kPending>
__device__ __forceinline__ void wait_copies() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending) : "memory");
}

template <class C>
__device__ __forceinline__ void fill(Stage<C>& s, const Pass& p, uint32_t step) {
  const uint32_t t = threadIdx.x;
  for (uint32_t i = t; i < C::kFeatures * (C::kWBytes / 16); i += kThreads) {
    uint32_t row, off;
    const uint8_t* src = dstc::w_copy<C>(p, step, i, &row, &off);
    copy16(&s.w[row][off], src, p.w);
  }
  for (uint32_t i = t; i < C::kTokens * (C::kXBytes / 16); i += kThreads) {
    uint32_t row, off;
    const uint8_t* src = dstc::x_copy<C>(p, step, i, &row, &off);
    copy16(&s.x[row][off], src, p.x);
  }
  if (dstc::scales_by_words(p)) {
    for (uint32_t i = t; i < (C::kFeatures + C::kTokens) * (C::kStepBlocks / 4); i += kThreads) {
      uint8_t* dst;
      const uint8_t* src = dstc::scale_copy<C>(p, step, i, s, &dst);
      copy4(dst, src, p.ws);
    }
  } else {
    for (uint32_t i = t; i < (C::kFeatures + C::kTokens) * C::kStepBlocks; i += kThreads) {
      dstc::stage_scale<C>(p, step, i, s);
    }
  }
}

// Any of the warp's operand words non-zero (a band that holds a code).
template <uint32_t kN, uint32_t kW>
__device__ __forceinline__ bool any_code(const uint32_t (&v)[kN][2][kW]) {
  uint32_t o = 0;
#pragma unroll
  for (uint32_t i = 0; i < kN; ++i) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
#pragma unroll
      for (uint32_t j = 0; j < kW; ++j) o |= v[i][h][j];
    }
  }
  return __any_sync(0xffffffffu, o != 0);
}

// Every word cut in two at F16 field `t`.
template <uint32_t kN, uint32_t kW>
__device__ __forceinline__ void cut2(const uint32_t (&v)[kN][2][kW], uint32_t t,
                                     uint32_t (&lo)[kN][2][kW], uint32_t (&hi)[kN][2][kW]) {
#pragma unroll
  for (uint32_t i = 0; i < kN; ++i) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
#pragma unroll
      for (uint32_t j = 0; j < kW; ++j) dstc::split2(v[i][h][j], t, &lo[i][h][j], &hi[i][h][j]);
    }
  }
}

// Every word cut in three at F16 fields `t1 < t2`.
template <uint32_t kN, uint32_t kW>
__device__ __forceinline__ void cut3(const uint32_t (&v)[kN][2][kW], uint32_t t1, uint32_t t2,
                                     uint32_t (&out)[3][kN][2][kW]) {
#pragma unroll
  for (uint32_t i = 0; i < kN; ++i) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
#pragma unroll
      for (uint32_t j = 0; j < kW; ++j) {
        dstc::split3(v[i][h][j], t1, t2, &out[0][i][h][j], &out[1][i][h][j], &out[2][i][h][j]);
      }
    }
  }
}

// The block dot products of block `blk` of the staged step, for the warp's
// tiles from staged weight row `frow` and the n-tiles `[nt0, nt1)`. A band
// empty across the warp is skipped: its products would add exact zeros.
template <class C, bool kBanded>
__device__ __forceinline__ void block_dots(const Stage<C>& s, uint32_t frow, uint32_t lane,
                                           uint32_t blk, uint32_t nt0, uint32_t nt1,
                                           float (&dot)[C::kMTiles][C::kNTiles][4]) {
  constexpr uint32_t kMT = C::kMTiles, kNT = C::kNTiles;
  uint32_t a[kMT][2][4], b[kNT][2][2];
  dstc::load_frags<C>(s, frow, lane, blk, a, b);
  if (!kBanded) {
#pragma unroll
    for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
      for (uint32_t nt = 0; nt < kNT; ++nt) {
        if (nt >= nt0 && nt < nt1) mma(dot[mt][nt], a[mt], b[nt]);
      }
    }
  } else if (C::kFp4) {
    int32_t u[kMT][kNT][4] = {};
    uint32_t bm[2][kNT][2][2];
    cut2(b, dstc::kX4Cut, bm[0], bm[1]);
#pragma unroll
    for (uint32_t xb = 0; xb < dstc::kX4Bands; ++xb) {
      if (!any_code(bm[xb])) continue;
#pragma unroll
      for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
        for (uint32_t nt = 0; nt < kNT; ++nt) {
          if (nt < nt0 || nt >= nt1) continue;
          float d[4];
          mma(d, a[mt], bm[xb][nt]);
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) u[mt][nt][i] += dstc::band_units4(d[i]);
        }
      }
    }
#pragma unroll
    for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
      for (uint32_t nt = 0; nt < kNT; ++nt) {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) dot[mt][nt][i] = dstc::block_dot4(u[mt][nt][i]);
      }
    }
  } else {
    int64_t u[kMT][kNT][4] = {};
    uint32_t bm[3][kNT][2][2], am[2][kMT][2][4];
    cut3(b, dstc::kX8Cut1, dstc::kX8Cut2, bm);
    cut2(a, dstc::kW8Cut, am[0], am[1]);
    bool on[3];
#pragma unroll
    for (uint32_t xb = 0; xb < dstc::kX8Bands; ++xb) on[xb] = any_code(bm[xb]);
#pragma unroll
    for (uint32_t wb = 0; wb < dstc::kW8Bands; ++wb) {
      if (!any_code(am[wb])) continue;
#pragma unroll
      for (uint32_t xb = 0; xb < dstc::kX8Bands; ++xb) {
        if (!on[xb]) continue;
#pragma unroll
        for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
          for (uint32_t nt = 0; nt < kNT; ++nt) {
            if (nt < nt0 || nt >= nt1) continue;
            float d[4];
            mma(d, am[wb][mt], bm[xb][nt]);
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) u[mt][nt][i] += dstc::band_units8(d[i]);
          }
        }
      }
    }
#pragma unroll
    for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
      for (uint32_t nt = 0; nt < kNT; ++nt) {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) dot[mt][nt][i] = dstc::block_dot8(u[mt][nt][i]);
      }
    }
  }
}

// One pass over K in a tile: stage `step % 2` holds the step while the next
// is copied in; each warp accumulates its outputs in registers.
template <class C, bool kBanded>
__device__ __forceinline__ void tile_pass(Stage<C> (&s)[2], const Pass& p) {
  constexpr uint32_t kMT = C::kMTiles, kNT = C::kNTiles;
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const uint32_t frow = warp * C::kWarpFeatures;
  const uint32_t steps = dstc::steps<C>(p);
  const bool warp_on = p.f0 + frow < p.n;
  const uint32_t nt0 = dstc::ntile_first(p), nt1 = dstc::ntile_last(p);
  float acc[kMT][kNT][4];
#pragma unroll
  for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (uint32_t nt = 0; nt < kNT; ++nt) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;
    }
  }
  fill<C>(s[0], p, 0);
  commit();
  for (uint32_t step = 0; step < steps; ++step) {
    if (step + 1 < steps) {
      fill<C>(s[(step + 1) & 1u], p, step + 1);
      commit();
      wait_copies<1>();
    } else {
      wait_copies<0>();
    }
    __syncthreads();
    if (warp_on) {
      const Stage<C>& cur = s[step & 1u];
      const uint32_t vb = dstc::blocks_in_step<C>(p, step);
#pragma unroll
      for (uint32_t blk = 0; blk < C::kStepBlocks; ++blk) {
        if (blk >= vb) continue;
        float dot[kMT][kNT][4];
        block_dots<C, kBanded>(cur, frow, lane, blk, nt0, nt1, dot);
#pragma unroll
        for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
          for (uint32_t nt = 0; nt < kNT; ++nt) {
            if (nt < nt0 || nt >= nt1) continue;
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              acc[mt][nt][i] = dstc::accumulate<C>(cur, dstc::d_feature(frow, lane, mt, i),
                                                   dstc::d_token(lane, nt, i), blk, acc[mt][nt][i],
                                                   dot[mt][nt][i]);
            }
          }
        }
      }
    }
    __syncthreads();
  }
  if (!warp_on) return;
#pragma unroll
  for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (uint32_t nt = 0; nt < kNT; ++nt) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        dstc::store(p, dstc::d_feature(frow, lane, mt, i), dstc::d_token(lane, nt, i), acc[mt][nt][i]);
      }
    }
  }
}

// One pass over K in the split block: `kStages` stages in flight; each warp
// computes the contributions of its quarter of the step's blocks, then
// thread `t` adds output `t / 8`'s contributions to row `t % 8` in ascending
// block order.
template <class C, bool kBanded>
__device__ __forceinline__ void split_pass(Stage<C>* s, dstc::Contributions<C>& c, const Pass& p) {
  static_assert(C::kFeatures * C::kTokens == kThreads, "a thread per output and row");
  constexpr uint32_t kNS = C::kStages, kPerWarp = C::kStepBlocks / C::kKWarps;
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const uint32_t steps = dstc::steps<C>(p);
  const uint32_t nt0 = dstc::ntile_first(p), nt1 = dstc::ntile_last(p);
  const uint32_t mine_f = threadIdx.x / C::kTokens, mine_r = threadIdx.x % C::kTokens;
  float acc = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j + 1 < kNS; ++j) {
    if (j < steps) fill<C>(s[j], p, j);
    commit();
  }
  for (uint32_t step = 0; step < steps; ++step) {
    if (step + kNS - 1 < steps) fill<C>(s[(step + kNS - 1) % kNS], p, step + kNS - 1);
    commit();
    wait_copies<kNS - 1>();
    __syncthreads();
    const Stage<C>& cur = s[step % kNS];
    const uint32_t vb = dstc::blocks_in_step<C>(p, step);
#pragma unroll
    for (uint32_t j = 0; j < kPerWarp; ++j) {
      const uint32_t blk = warp * kPerWarp + j;
      if (blk >= vb) continue;
      float dot[1][1][4];
      block_dots<C, kBanded>(cur, 0, lane, blk, nt0, nt1, dot);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t f = dstc::d_feature(0, lane, 0, i), r = dstc::d_token(lane, 0, i);
        c.c[blk][f][r] = dstc::contribution<C>(cur, f, r, blk, dot[0][0][i]);
      }
    }
    __syncthreads();
    for (uint32_t blk = 0; blk < vb; ++blk) acc = acc + c.c[blk][mine_f][mine_r];
  }
  dstc::store(p, mine_f, mine_r, acc);
  // The next pass's first contributions are written only after its first
  // barrier, which every thread reaches after this pass's last sums.
}

template <uint32_t kTokens, bool kBanded>
__device__ __forceinline__ void lin8_tile(const DsLin8Params& p) {
  using C = dstc::TileCfg<false, kTokens>;
  __shared__ Stage<C> s[2];
  const uint32_t r0 = blockIdx.y * kTokens;
  const uint32_t r1 = p.rows - r0 < kTokens ? p.rows : r0 + kTokens;
  tile_pass<C, kBanded>(s, dstc::lin8_pass(p, r0, r1, blockIdx.x * C::kFeatures));
}

template <uint32_t kTokens, bool kBanded>
__device__ __forceinline__ void lin4_tile(const DsLin4Params& p) {
  using C = dstc::TileCfg<true, kTokens>;
  __shared__ Stage<C> s[2];
  const uint32_t r0 = blockIdx.y * kTokens;
  const uint32_t r1 = p.layout_rows - r0 < kTokens ? p.layout_rows : r0 + kTokens;
  const uint32_t f0 = blockIdx.x * C::kFeatures;
  for (uint32_t g = dstc::first_group(p, r0); g < p.groups; ++g) {
    if (dstc::group_start(p, g) >= r1) break;
    const Pass pass = dstc::lin4_pass(p, g, r0, r1, f0);
    if (pass.lo < pass.hi) tile_pass<C, kBanded>(s, pass);
  }
}

extern __shared__ __align__(16) uint8_t split_smem[];

template <class C>
__device__ __forceinline__ Stage<C>* split_stages() {
  return reinterpret_cast<Stage<C>*>(split_smem);
}
template <class C>
__device__ __forceinline__ dstc::Contributions<C>& split_contributions() {
  return *reinterpret_cast<dstc::Contributions<C>*>(split_smem + C::kStages * sizeof(Stage<C>));
}

template <bool kBanded>
__device__ __forceinline__ void lin8_split(const DsLin8Params& p) {
  using C = dstc::SplitCfg<false>;
  const uint32_t r0 = blockIdx.y * C::kTokens;
  const uint32_t r1 = p.rows - r0 < C::kTokens ? p.rows : r0 + C::kTokens;
  split_pass<C, kBanded>(split_stages<C>(), split_contributions<C>(),
                         dstc::lin8_pass(p, r0, r1, blockIdx.x * C::kFeatures));
}

template <bool kBanded>
__device__ __forceinline__ void lin4_split(const DsLin4Params& p) {
  using C = dstc::SplitCfg<true>;
  const uint32_t r0 = blockIdx.y * C::kTokens;
  const uint32_t r1 = p.layout_rows - r0 < C::kTokens ? p.layout_rows : r0 + C::kTokens;
  const uint32_t f0 = blockIdx.x * C::kFeatures;
  for (uint32_t g = dstc::first_group(p, r0); g < p.groups; ++g) {
    if (dstc::group_start(p, g) >= r1) break;
    const Pass pass = dstc::lin4_pass(p, g, r0, r1, f0);
    if (pass.lo < pass.hi) {
      split_pass<C, kBanded>(split_stages<C>(), split_contributions<C>(), pass);
      // Every thread has read its sums before another pass refills the stages.
      __syncthreads();
    }
  }
}

constexpr uint32_t kSplit8Smem = dstc::split_smem_bytes<dstc::SplitCfg<false>>();
constexpr uint32_t kSplit4Smem = dstc::split_smem_bytes<dstc::SplitCfg<true>>();

}  // namespace

#define DS_GEMM_TC_ENTRY(NAME, BODY, PARAMS, SMEM)                                    \
  extern "C" __global__ void __launch_bounds__(kThreads)                              \
      NAME(const __grid_constant__ PARAMS p) {                                        \
    BODY(p);                                                                          \
  }                                                                                   \
  EIDOLA_KERNEL_META(NAME, kThreads, 1, 1, SMEM, 1, 1, 1, sizeof(PARAMS))

DS_GEMM_TC_ENTRY(eidola_ds_lin8_banded_s8, (lin8_split<true>), DsLin8Params, kSplit8Smem);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_banded_t8, (lin8_tile<8, true>), DsLin8Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_banded_t32, (lin8_tile<32, true>), DsLin8Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_tolerance_s8, (lin8_split<false>), DsLin8Params, kSplit8Smem);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_tolerance_t8, (lin8_tile<8, false>), DsLin8Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_tolerance_t32, (lin8_tile<32, false>), DsLin8Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_banded_s8, (lin4_split<true>), DsLin4Params, kSplit4Smem);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_banded_t8, (lin4_tile<8, true>), DsLin4Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_banded_t32, (lin4_tile<32, true>), DsLin4Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_tolerance_s8, (lin4_split<false>), DsLin4Params, kSplit4Smem);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_tolerance_t8, (lin4_tile<8, false>), DsLin4Params, 0);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_tolerance_t32, (lin4_tile<32, false>), DsLin4Params, 0);

// ---- the probe ----

namespace {

__device__ __forceinline__ uint64_t mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// An E4M3 code of the band `lo | hi << 8` drawn by `family` from `h`.
__device__ __forceinline__ uint8_t draw_e4m3(uint64_t h, uint32_t band, uint32_t family) {
  const uint32_t lo = band & 0xffu, hi = (band >> 8) & 0xffu;
  uint32_t sign = static_cast<uint32_t>(h >> 63) << 7, e, m;
  if (family == DS_PROBE_UNIFORM) {
    if ((h & 15u) == 0) return static_cast<uint8_t>(sign);
    e = lo + static_cast<uint32_t>((h >> 8) % (hi - lo + 1));
    m = static_cast<uint32_t>(h >> 16) & 7u;
  } else if (family == DS_PROBE_CARRY) {
    sign = 0;
    e = hi;
    m = 7;
  } else if ((h >> 4) & 1u) {
    e = hi;
    m = 7;
  } else {
    e = lo;
    m = lo == 0 ? 1u : 0u;
  }
  if (e == 15 && m == 7) m = 6;
  return static_cast<uint8_t>(sign | e << 3 | m);
}

// An E2M1 code drawn by `family` from `h`.
__device__ __forceinline__ uint8_t draw_e2m1(uint64_t h, uint32_t family) {
  if (family == DS_PROBE_UNIFORM) return static_cast<uint8_t>(h & 15u);
  if (family == DS_PROBE_CARRY) return 7;
  const uint32_t sign = static_cast<uint32_t>(h >> 63) << 3;
  return static_cast<uint8_t>(sign | (((h >> 4) & 1u) ? 7u : 1u));
}

// Element `k` of row `idx` of operand `op` (0: A, 1: B) of the trial `key`:
// an E4M3 code, or for an E2M1 A the E2M1 code.
__device__ uint8_t element(const DsGemmProbeParams& p, uint64_t key, uint32_t op, uint32_t idx,
                           uint32_t k) {
  uint32_t family = p.family, kk = k;
  bool negate = false;
  if (family == DS_PROBE_CANCEL) {
    const bool residue = op == 1 && k == 31;
    if (k >= 16 && !residue) {
      kk = k - 16;
      negate = op == 1;
    }
    family = residue ? DS_PROBE_UNIFORM : DS_PROBE_EXTREMES;
  }
  const uint64_t h = mix(key ^ (static_cast<uint64_t>(op) << 40 | static_cast<uint64_t>(idx) << 8 | kk));
  if (op == 0 && p.a_format == 1) return draw_e2m1(h, family);
  const uint8_t c = draw_e4m3(h, op == 0 ? p.a_band : p.b_band, family);
  return negate ? static_cast<uint8_t>(c ^ 0x80u) : c;
}

// Elements `k0 .. k0 + 4` of row `idx` as the two F16 words the kernels'
// loads make of them (`lo`: elements 0, 1; `hi`: 2, 3).
__device__ void words(const DsGemmProbeParams& p, uint64_t key, uint32_t op, uint32_t idx,
                      uint32_t k0, uint32_t* lo, uint32_t* hi) {
  uint32_t v = 0;
  if (op == 0 && p.a_format == 1) {
    for (uint32_t j = 0; j < 4; ++j) v |= static_cast<uint32_t>(element(p, key, op, idx, k0 + j)) << (4 * j);
    dstc::e2m1x4_to_f16(v, lo, hi);
  } else {
    for (uint32_t j = 0; j < 4; ++j) v |= static_cast<uint32_t>(element(p, key, op, idx, k0 + j)) << (8 * j);
    dstc::fp8_word(v, lo, hi);
  }
}

// A trial's fragments, with A's rows taken from `(m + ra) % 16` and B's
// columns from `(n + rb) % 8`.
__device__ void fragments(const DsGemmProbeParams& p, uint64_t key, uint32_t lane, uint32_t ra,
                          uint32_t rb, uint32_t (&a)[2][4], uint32_t (&b)[2][2]) {
  const uint32_t g = lane >> 2, q = lane & 3u;
  for (uint32_t h = 0; h < 2; ++h) {
    const uint32_t k0 = h * 16 + q * 4;
    words(p, key, 0, (g + ra) % 16, k0, &a[h][0], &a[h][2]);
    words(p, key, 0, (g + 8 + ra) % 16, k0, &a[h][1], &a[h][3]);
    words(p, key, 1, (g + rb) % 8, k0, &b[h][0], &b[h][1]);
  }
}

__device__ __forceinline__ int64_t ordered(uint32_t b) {
  return (b & 0x80000000u) ? -static_cast<int64_t>(b & 0x7fffffffu) : static_cast<int64_t>(b);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_ds_gemm_probe(const __grid_constant__ DsGemmProbeParams p) {
  __shared__ float stash[dstc::kWarps][16][8];
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const uint32_t g = lane >> 2, q = lane & 3u;
  const uint64_t gw = static_cast<uint64_t>(blockIdx.x) * dstc::kWarps + warp;
  unsigned long long checked = 0, wrong = 0, moved = 0, worst = 0;
  bool recorded = false;
  for (uint32_t trial = 0; trial < p.trials; ++trial) {
    const uint64_t key = mix(p.seed ^ mix(gw << 32 | trial));
    const uint32_t ra = 1 + static_cast<uint32_t>(key % 15), rb = 1 + static_cast<uint32_t>((key >> 8) % 7);
    uint32_t a[2][4], b[2][2], a2[2][4], b2[2][2];
    fragments(p, key, lane, 0, 0, a, b);
    fragments(p, key, lane, ra, rb, a2, b2);
    float d[4], d2[4];
    mma(d, a, b);
    mma(d2, a2, b2);
    for (uint32_t i = 0; i < 4; ++i) stash[warp][g + (i >= 2 ? 8 : 0)][2 * q + (i & 1u)] = d[i];
    __syncwarp();
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t m = g + (i >= 2 ? 8 : 0), n = 2 * q + (i & 1u);
      // The exact sum, in units of 2^-18 (2^-10 for an E2M1 A).
      int64_t sum = 0;
      for (uint32_t k = 0; k < 32; ++k) {
        const uint8_t ac = element(p, key, 0, m, k);
        const int64_t av = p.a_format == 1 ? ds::e2m1_times_2(ac) : ds::e4m3_times_512(ac);
        sum += av * ds::e4m3_times_512(element(p, key, 1, n, k));
      }
      const double unit = p.a_format == 1 ? 0.0009765625 : 3.814697265625e-06;
      const float want = static_cast<float>(static_cast<double>(sum) * unit);
      ++checked;
      if (!(d[i] == want)) {
        ++wrong;
        const int64_t dist = ordered(ds::f32_bits(d[i])) - ordered(ds::f32_bits(want));
        const unsigned long long ad = static_cast<unsigned long long>(dist < 0 ? -dist : dist);
        if (ad > worst) worst = ad;
        if (!recorded) {
          recorded = true;
          const unsigned long long slot = atomicAdd(&p.counts[4], 1ull);
          if (slot < p.max_records) {
            uint32_t* rec = p.records + slot * 8;
            rec[0] = static_cast<uint32_t>(gw);
            rec[1] = trial;
            rec[2] = m;
            rec[3] = n;
            rec[4] = ds::f32_bits(d[i]);
            rec[5] = ds::f32_bits(want);
            rec[6] = static_cast<uint32_t>(static_cast<uint64_t>(sum));
            rec[7] = static_cast<uint32_t>(static_cast<uint64_t>(sum) >> 32);
          }
        }
      }
      // The rotated product's element (m, n) is the first's ((m + ra) % 16, (n + rb) % 8).
      if (ds::f32_bits(d2[i]) != ds::f32_bits(stash[warp][(m + ra) % 16][(n + rb) % 8])) ++moved;
    }
    __syncwarp();
  }
  atomicAdd(&p.counts[0], checked);
  atomicAdd(&p.counts[1], wrong);
  atomicAdd(&p.counts[2], moved);
  atomicMax(&p.counts[3], worst);
}
EIDOLA_KERNEL_META(eidola_ds_gemm_probe, kThreads, 1, 1, 0, 1, 1, 1, sizeof(DsGemmProbeParams));
