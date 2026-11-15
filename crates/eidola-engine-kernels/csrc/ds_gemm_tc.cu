// DeepSeek-V4.1's quantised GEMMs on tensor cores, and the probe of the MMA
// they rest on. `ds_gemm_tc.cuh` states the forms and holds every piece a
// thread computes; this file places them on the grid: staging through
// shared memory with `cp.async`, the MMAs, and the passes of `lin4`'s groups.
//
// Entries, each taking the scalar reference's parameter struct unchanged
// (`DsLin8Params`, `DsLin4Params`; `ds_reference.cu` is the definition):
//
//   eidola_ds_{lin8,lin4}_banded_t{8,32}     exact: banded block products
//   eidola_ds_{lin8,lin4}_tolerance_t{8,32}  one MMA per block
//   eidola_ds_gemm_probe                     the MMA against the exact sum
//
// Grid `(ceil(n / features), ceil(rows / tokens))`, 128 threads: a block
// computes `tokens` rows (8 or 32) by `features` outputs (64 or 128); `lin4`
// takes `layout_rows` for `rows`. Every output of a row is computed from that
// row's codes and scales and the weight, by the same instructions whichever
// tile holds it; the tile size is chosen by the host from the row count.
//
// `lin4`'s tile may meet several groups' runs (any `row_align`): it makes one
// pass over K per group, each pass computing that group's rows of the tile
// against that group's weight. A `row_align` that is a multiple of the token
// tile gives every tile at most one group. Padding rows are not written.

#include "ds_gemm_tc.cuh"
#include "eidola_kernel.cuh"

namespace {

using dstc::Cfg;
using dstc::kChunkBlocks;
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
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int kPending>
__device__ __forceinline__ void wait_copies() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending) : "memory");
}

template <bool kFp4, uint32_t kTokens>
__device__ __forceinline__ void fill(Stage<kFp4, kTokens>& s, const Pass& p, uint32_t step) {
  using C = Cfg<kFp4, kTokens>;
  const uint32_t t = threadIdx.x;
  for (uint32_t i = t; i < C::kFeatures * (C::kWBytes / 16); i += kThreads) {
    uint32_t row, off;
    const uint8_t* src = dstc::w_copy<kFp4, kTokens>(p, step, i, &row, &off);
    copy16(&s.w[row][off], src, p.w);
  }
  for (uint32_t i = t; i < kTokens * (dstc::kChunkBytes / 16); i += kThreads) {
    uint32_t row, off;
    const uint8_t* src = dstc::x_copy<kTokens>(p, step, i, &row, &off);
    copy16(&s.x[row][off], src, p.x);
  }
  for (uint32_t i = t; i < (C::kFeatures + kTokens) * kChunkBlocks; i += kThreads) {
    dstc::stage_scale(p, step, i, s);
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

// The band `[lo, hi]` of every word.
template <uint32_t kN, uint32_t kW>
__device__ __forceinline__ void mask(const uint32_t (&v)[kN][2][kW], uint32_t lo, uint32_t hi,
                                     uint32_t (&out)[kN][2][kW]) {
#pragma unroll
  for (uint32_t i = 0; i < kN; ++i) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
#pragma unroll
      for (uint32_t j = 0; j < kW; ++j) out[i][h][j] = dstc::band_mask(v[i][h][j], lo, hi);
    }
  }
}

// One block of the staged step into the accumulators. A band empty across
// the warp is skipped: its MMAs would add exact zeros.
template <bool kFp4, uint32_t kTokens, bool kBanded>
__device__ __forceinline__ void block(const Stage<kFp4, kTokens>& s, uint32_t warp, uint32_t lane,
                                      uint32_t blk, uint32_t nt0, uint32_t nt1,
                                      float (&acc)[Cfg<kFp4, kTokens>::kMTiles]
                                                  [Cfg<kFp4, kTokens>::kNTiles][4]) {
  using C = Cfg<kFp4, kTokens>;
  constexpr uint32_t kMT = C::kMTiles, kNT = C::kNTiles;
  uint32_t a[kMT][2][4], b[kNT][2][2];
  dstc::load_frags(s, warp, lane, blk, a, b);
  float dot[kMT][kNT][4];
  if (!kBanded) {
#pragma unroll
    for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
      for (uint32_t nt = 0; nt < kNT; ++nt) {
        if (nt >= nt0 && nt < nt1) mma(dot[mt][nt], a[mt], b[nt]);
      }
    }
  } else if (kFp4) {
    int32_t u[kMT][kNT][4] = {};
#pragma unroll
    for (uint32_t xb = 0; xb < dstc::kX4Bands; ++xb) {
      uint32_t bm[kNT][2][2];
      mask(b, dstc::x4_band_lo(xb), dstc::x4_band_hi(xb), bm);
      if (!any_code(bm)) continue;
#pragma unroll
      for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
        for (uint32_t nt = 0; nt < kNT; ++nt) {
          if (nt < nt0 || nt >= nt1) continue;
          float d[4];
          mma(d, a[mt], bm[nt]);
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
    uint32_t bm[dstc::kX8Bands][kNT][2][2];
    bool on[dstc::kX8Bands];
#pragma unroll
    for (uint32_t xb = 0; xb < dstc::kX8Bands; ++xb) {
      mask(b, dstc::x8_band_lo(xb), dstc::x8_band_hi(xb), bm[xb]);
      on[xb] = any_code(bm[xb]);
    }
#pragma unroll
    for (uint32_t wb = 0; wb < dstc::kW8Bands; ++wb) {
      uint32_t am[kMT][2][4];
      mask(a, dstc::w8_band_lo(wb), dstc::w8_band_hi(wb), am);
      if (!any_code(am)) continue;
#pragma unroll
      for (uint32_t xb = 0; xb < dstc::kX8Bands; ++xb) {
        if (!on[xb]) continue;
#pragma unroll
        for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
          for (uint32_t nt = 0; nt < kNT; ++nt) {
            if (nt < nt0 || nt >= nt1) continue;
            float d[4];
            mma(d, am[mt], bm[xb][nt]);
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
#pragma unroll
  for (uint32_t mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (uint32_t nt = 0; nt < kNT; ++nt) {
      if (nt < nt0 || nt >= nt1) continue;
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        acc[mt][nt][i] = dstc::accumulate(s, warp, lane, mt, nt, i, blk, acc[mt][nt][i],
                                          dot[mt][nt][i]);
      }
    }
  }
}

// One pass over K: stage `s` holds step `j` while step `j + 1` is copied in.
template <bool kFp4, uint32_t kTokens, bool kBanded>
__device__ __forceinline__ void run_pass(Stage<kFp4, kTokens> (&s)[2], const Pass& p) {
  using C = Cfg<kFp4, kTokens>;
  constexpr uint32_t kMT = C::kMTiles, kNT = C::kNTiles;
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const uint32_t steps = (p.k / 32 + kChunkBlocks - 1) / kChunkBlocks;
  const bool warp_on = p.f0 + warp * C::kWarpFeatures < p.n;
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
  fill(s[0], p, 0);
  commit();
  for (uint32_t step = 0; step < steps; ++step) {
    if (step + 1 < steps) {
      fill(s[(step + 1) & 1u], p, step + 1);
      commit();
      wait_copies<1>();
    } else {
      wait_copies<0>();
    }
    __syncthreads();
    if (warp_on) {
      const Stage<kFp4, kTokens>& cur = s[step & 1u];
      const uint32_t vb = dstc::blocks_in_step(p, step);
#pragma unroll
      for (uint32_t blk = 0; blk < kChunkBlocks; ++blk) {
        if (blk < vb) block<kFp4, kTokens, kBanded>(cur, warp, lane, blk, nt0, nt1, acc);
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
        dstc::store<kFp4, kTokens>(p, warp, lane, mt, nt, i, acc[mt][nt][i]);
      }
    }
  }
}

template <uint32_t kTokens, bool kBanded>
__device__ __forceinline__ void lin8(const DsLin8Params& p) {
  __shared__ Stage<false, kTokens> s[2];
  const uint32_t r0 = blockIdx.y * kTokens;
  const uint32_t r1 = p.rows - r0 < kTokens ? p.rows : r0 + kTokens;
  run_pass<false, kTokens, kBanded>(s, dstc::lin8_pass(p, r0, r1, blockIdx.x * Cfg<false, kTokens>::kFeatures));
}

template <uint32_t kTokens, bool kBanded>
__device__ __forceinline__ void lin4(const DsLin4Params& p) {
  __shared__ Stage<true, kTokens> s[2];
  const uint32_t r0 = blockIdx.y * kTokens;
  const uint32_t r1 = p.layout_rows - r0 < kTokens ? p.layout_rows : r0 + kTokens;
  const uint32_t f0 = blockIdx.x * Cfg<true, kTokens>::kFeatures;
  for (uint32_t g = dstc::first_group(p, r0); g < p.groups; ++g) {
    if (dstc::group_start(p, g) >= r1) break;
    const Pass pass = dstc::lin4_pass(p, g, r0, r1, f0);
    if (pass.lo < pass.hi) run_pass<true, kTokens, kBanded>(s, pass);
  }
}

}  // namespace

#define DS_GEMM_TC_ENTRY(NAME, BODY, PARAMS)                                          \
  extern "C" __global__ void __launch_bounds__(kThreads)                              \
      NAME(const __grid_constant__ PARAMS p) {                                        \
    BODY(p);                                                                          \
  }                                                                                   \
  EIDOLA_KERNEL_META(NAME, kThreads, 1, 1, 0, 1, 1, 1, sizeof(PARAMS))

DS_GEMM_TC_ENTRY(eidola_ds_lin8_banded_t8, (lin8<8, true>), DsLin8Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_banded_t32, (lin8<32, true>), DsLin8Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_tolerance_t8, (lin8<8, false>), DsLin8Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin8_tolerance_t32, (lin8<32, false>), DsLin8Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_banded_t8, (lin4<8, true>), DsLin4Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_banded_t32, (lin4<32, true>), DsLin4Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_tolerance_t8, (lin4<8, false>), DsLin4Params);
DS_GEMM_TC_ENTRY(eidola_ds_lin4_tolerance_t32, (lin4<32, false>), DsLin4Params);

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
