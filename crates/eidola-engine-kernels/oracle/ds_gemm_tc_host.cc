// Host check of the DeepSeek tensor-core GEMMs (`csrc/ds_gemm_tc.cuh`): the
// device kernels' staging, fragment, band and block arithmetic run on the
// host around an emulated `mma.m16n8k16`, against the scalar reference
// bodies (`dsk::lin8_thread`, `dsk::lin4_thread` of `csrc/ds_ops.cuh`).
//
// Dev-only: never a build input. It shows the index arithmetic (copies,
// fragments, the slots of K, the D layout, the passes over `lin4`'s groups),
// the conversions and the bands are right, and that the banded form is exact
// for any adder that is exact on exactly representable sums: the emulated
// MMA adds the 16 products in a random order with f32 rounding after every
// addition, or rounds the exact sum once. What only the device shows is
// what its tensor cores do (`eidola_ds_gemm_probe`) and what nvcc emits.
//
//   c++ -std=c++17 -O2 -ffp-contract=off -I csrc oracle/ds_gemm_tc_host.cc -o ds_gemm_tc_host
//   ./ds_gemm_tc_host
//
// `tests/deepseek_gemm_tc_host.rs` runs exactly that. It prints one line per
// check and exits 0 when every banded output equals the reference.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "ds_gemm_tc.cuh"
#include "ds_ops.cuh"

namespace {

uint64_t fin(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

struct Rng {
  uint64_t state;
  uint64_t next() {
    state += 0x9e3779b97f4a7c15ull;
    return fin(state);
  }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
};

int failures = 0;
// Steps staged with each path of the scale bytes.
uint64_t word_staged = 0, byte_staged = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    ++failures;
    std::printf("FAILED %s\n", what);
  }
}

float f16_value(uint32_t h) {
  const uint32_t sign = h >> 15 & 1u, e = h >> 10 & 0x1fu, m = h & 0x3ffu;
  float v;
  if (e == 0) {
    v = static_cast<float>(m) * ds::f32_from_bits((127u - 24u) << 23);
  } else {
    v = static_cast<float>(1024u + m) * ds::f32_from_bits((e + 127u - 25u) << 23);
  }
  return sign ? -v : v;
}

// ---- conversions and bands ----

void conversions() {
  bool ok = true;
  for (uint32_t c = 0; c < 256; ++c) {
    if (ds::e4m3_is_nan(static_cast<uint8_t>(c))) continue;
    const uint32_t h = dstc::e4m3x2_to_f16x2(c | (c ^ 0x80u) << 8);
    const float want = ds::e4m3_decode(static_cast<uint8_t>(c));
    ok &= ds::f32_bits(f16_value(h & 0xffffu)) == ds::f32_bits(want);
    ok &= ds::f32_bits(f16_value(h >> 16)) == ds::f32_bits(-want);
  }
  check(ok, "E4M3 to F16");
  ok = true;
  for (uint32_t v = 0; v < 65536; ++v) {
    uint32_t lo, hi;
    dstc::e2m1x4_to_f16(v, &lo, &hi);
    const uint32_t got[4] = {lo & 0xffffu, lo >> 16, hi & 0xffffu, hi >> 16};
    for (uint32_t j = 0; j < 4; ++j) {
      const float want = ds::e2m1_decode(static_cast<uint8_t>(v >> (4 * j) & 15u));
      ok &= ds::f32_bits(f16_value(got[j])) == ds::f32_bits(want);
    }
  }
  check(ok, "E2M1 to F16");
  std::printf("conversions: every E4M3 and E2M1 code\n");
}

// Each set of bands, as the kernels cut a word: the parts of `x`.
uint32_t cut(uint32_t set, uint32_t x, uint32_t (&parts)[3]) {
  parts[2] = 0;
  if (set == 0) {
    dstc::split2(x, dstc::kW8Cut, &parts[0], &parts[1]);
    return dstc::kW8Bands;
  }
  if (set == 1) {
    dstc::split3(x, dstc::kX8Cut1, dstc::kX8Cut2, &parts[0], &parts[1], &parts[2]);
    return dstc::kX8Bands;
  }
  dstc::split2(x, dstc::kX4Cut, &parts[0], &parts[1]);
  return dstc::kX4Bands;
}

// The parts of every word partition it, a code's band does not depend on the
// other half of its word, and each band pair's sum of 32 magnitudes, in units
// of its smallest product, is below 2^24.
void bands() {
  bool ok = true;
  // Per set and band: the largest magnitude and the smallest unit, times 512.
  int64_t big[3][3] = {}, unit[3][3];
  for (auto& row : unit) {
    for (auto& u : row) u = 1ll << 40;
  }
  for (uint32_t set = 0; set < 3; ++set) {
    for (uint32_t c = 0; c < 256; ++c) {
      if (ds::e4m3_is_nan(static_cast<uint8_t>(c))) continue;
      for (uint32_t other : {0x00u, 0x7eu, 0x80u, 0x01u, 0x38u}) {
        const uint32_t x = dstc::e4m3x2_to_f16x2(c | other << 8);
        uint32_t parts[3];
        const uint32_t n = cut(set, x, parts);
        uint32_t sum = 0, hits = 0, band = 0;
        for (uint32_t j = 0; j < n; ++j) {
          ok &= (parts[j] & ~x) == 0 && (sum & parts[j]) == 0;
          sum |= parts[j];
          if ((parts[j] & 0xffffu) == (x & 0xffffu) && (x & 0x7fffu) != 0) {
            ++hits;
            band = j;
          }
        }
        ok &= sum == x;
        ok &= (x & 0x7fffu) == 0 || hits == 1;
        if ((c & 0x7fu) == 0) continue;
        const int64_t v = std::llabs(ds::e4m3_times_512(static_cast<uint8_t>(c)));
        big[set][band] = std::max(big[set][band], v);
        unit[set][band] = std::min(unit[set][band], v & -v);  // the lowest set bit
      }
    }
  }
  for (uint32_t wb = 0; wb < dstc::kW8Bands; ++wb) {
    for (uint32_t xb = 0; xb < dstc::kX8Bands; ++xb) {
      ok &= 32 * (big[0][wb] / unit[0][wb]) * (big[1][xb] / unit[1][xb]) < (1ll << 24);
    }
  }
  for (uint32_t xb = 0; xb < dstc::kX4Bands; ++xb) {
    ok &= 32 * 12 * (big[2][xb] / unit[2][xb]) < (1ll << 24);
  }
  check(ok, "bands partition the codes and bound every pair's sum below 2^24");
  std::printf("bands: %u x %u FP8 pairs, %u FP4 bands\n", dstc::kW8Bands, dstc::kX8Bands,
              dstc::kX4Bands);
}

// ---- the emulated MMA ----

enum class Adder { Exact, Shuffled };

// One `mma.m16n8k16` (F16 operands, f32 accumulator) over a warp's lanes.
void mma16(Adder adder, Rng& rng, const uint32_t (&a)[32][4], const uint32_t (&b)[32][2],
           const float (&c)[32][4], float (&d)[32][4]) {
  float A[16][16], B[16][8], C[16][8];
  for (uint32_t l = 0; l < 32; ++l) {
    const uint32_t g = l >> 2, q = l & 3u;
    for (uint32_t r = 0; r < 4; ++r) {
      const uint32_t row = g + (r & 1u) * 8, slot = (r >> 1) * 8 + 2 * q;
      A[row][slot] = f16_value(a[l][r] & 0xffffu);
      A[row][slot + 1] = f16_value(a[l][r] >> 16);
    }
    for (uint32_t r = 0; r < 2; ++r) {
      B[r * 8 + 2 * q][g] = f16_value(b[l][r] & 0xffffu);
      B[r * 8 + 2 * q + 1][g] = f16_value(b[l][r] >> 16);
    }
    for (uint32_t i = 0; i < 4; ++i) C[g + (i >= 2 ? 8 : 0)][2 * q + (i & 1u)] = c[l][i];
  }
  for (uint32_t l = 0; l < 32; ++l) {
    const uint32_t g = l >> 2, q = l & 3u;
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t m = g + (i >= 2 ? 8 : 0), n = 2 * q + (i & 1u);
      if (adder == Adder::Exact) {
        // Products of F16 values are exact in f64, and so is their sum here.
        double s = C[m][n];
        for (uint32_t k = 0; k < 16; ++k) s += static_cast<double>(A[m][k]) * B[k][n];
        d[l][i] = static_cast<float>(s);
      } else {
        float terms[17];
        for (uint32_t k = 0; k < 16; ++k) terms[k] = A[m][k] * B[k][n];
        terms[16] = C[m][n];
        for (uint32_t k = 16; k > 0; --k) std::swap(terms[k], terms[rng.below(k + 1)]);
        float s = terms[0];
        for (uint32_t k = 1; k < 17; ++k) s = s + terms[k];
        d[l][i] = s;
      }
    }
  }
}

// A block's product over a warp: both halves, chained from zero.
void mma(Adder adder, Rng& rng, const uint32_t (&a)[32][2][4], const uint32_t (&b)[32][2][2],
         float (&d)[32][4]) {
  float zero[32][4] = {}, half[32][4];
  uint32_t ah[32][4], bh[32][2];
  for (uint32_t h = 0; h < 2; ++h) {
    for (uint32_t l = 0; l < 32; ++l) {
      std::memcpy(ah[l], a[l][h], sizeof ah[l]);
      std::memcpy(bh[l], b[l][h], sizeof bh[l]);
    }
    if (h == 0) {
      mma16(adder, rng, ah, bh, zero, half);
    } else {
      mma16(adder, rng, ah, bh, half, d);
    }
  }
}

// ---- the blocks of the grid, as `ds_gemm_tc.cu` runs them ----

// `frag_dots` of `ds_gemm_tc.cu` over a warp's 32 lanes' fragments.
template <bool kFp4, bool kBanded, uint32_t kMT, uint32_t kNT>
void frag_dots(const uint32_t (&a)[32][kMT][2][4], const uint32_t (&b)[32][kNT][2][2], uint32_t nt0,
               uint32_t nt1, Adder adder, Rng& rng, float (&dot)[32][kMT][kNT][4]) {
  for (uint32_t mt = 0; mt < kMT; ++mt) {
    for (uint32_t nt = nt0; nt < nt1; ++nt) {
      static uint32_t am[32][2][4], bm[32][2][2];
      float d[32][4];
      if (!kBanded) {
        for (uint32_t l = 0; l < 32; ++l) {
          std::memcpy(am[l], a[l][mt], sizeof am[l]);
          std::memcpy(bm[l], b[l][nt], sizeof bm[l]);
        }
        mma(adder, rng, am, bm, d);
        for (uint32_t l = 0; l < 32; ++l) {
          for (uint32_t i = 0; i < 4; ++i) dot[l][mt][nt][i] = d[l][i];
        }
        continue;
      }
      float hi[32][4] = {}, lo[32][4] = {};
      const uint32_t wbands = kFp4 ? 1 : dstc::kW8Bands;
      const uint32_t xbands = kFp4 ? dstc::kX4Bands : dstc::kX8Bands;
      for (uint32_t wb = 0; wb < wbands; ++wb) {
        for (uint32_t xb = 0; xb < xbands; ++xb) {
          for (uint32_t l = 0; l < 32; ++l) {
            for (uint32_t h = 0; h < 2; ++h) {
              uint32_t parts[3];
              for (uint32_t j = 0; j < 4; ++j) {
                if (kFp4) {
                  am[l][h][j] = a[l][mt][h][j];
                } else {
                  cut(0, a[l][mt][h][j], parts);
                  am[l][h][j] = parts[wb];
                }
              }
              for (uint32_t j = 0; j < 2; ++j) {
                cut(kFp4 ? 2 : 1, b[l][nt][h][j], parts);
                bm[l][h][j] = parts[xb];
              }
            }
          }
          mma(adder, rng, am, bm, d);
          for (uint32_t l = 0; l < 32; ++l) {
            for (uint32_t i = 0; i < 4; ++i) {
              if (kFp4) {
                dstc::pair4(d[l][i], &hi[l][i], &lo[l][i]);
              } else {
                dstc::pair8(d[l][i], &hi[l][i], &lo[l][i]);
              }
            }
          }
        }
      }
      for (uint32_t l = 0; l < 32; ++l) {
        for (uint32_t i = 0; i < 4; ++i) {
          dot[l][mt][nt][i] =
              kFp4 ? dstc::block_dot4(hi[l][i], lo[l][i]) : dstc::block_dot8(hi[l][i], lo[l][i]);
        }
      }
    }
  }
}

// `block_dots`: a staged block's fragments, then their products.
template <class C, bool kBanded>
void block_dots(const dstc::Stage<C>& s, uint32_t frow, uint32_t blk, uint32_t nt0, uint32_t nt1,
                Adder adder, Rng& rng, float (&dot)[32][C::kMTiles][C::kNTiles][4]) {
  static uint32_t a[32][C::kMTiles][2][4], b[32][C::kNTiles][2][2];
  for (uint32_t l = 0; l < 32; ++l) dstc::load_frags<C>(s, frow, l, blk, a[l], b[l]);
  frag_dots<C::kFp4, kBanded>(a, b, nt0, nt1, adder, rng, dot);
}

// One step's stage, copied or zeroed as `fill` does; staged bytes start as
// garbage, so every byte a block reads must have been copied or zeroed.
template <class C>
void fill(dstc::Stage<C>& s, const dstc::Pass& p, uint32_t step) {
  std::memset(&s, 0xa5, sizeof s);
  for (uint32_t i = 0; i < C::kFeatures * (C::kWBytes / 16); ++i) {
    uint32_t row, off;
    const uint8_t* src = dstc::w_copy<C>(p, step, i, &row, &off);
    if (src) {
      std::memcpy(&s.w[row][off], src, 16);
    } else {
      std::memset(&s.w[row][off], 0, 16);
    }
  }
  for (uint32_t i = 0; i < C::kTokens * (C::kXBytes / 16); ++i) {
    uint32_t row, off;
    const uint8_t* src = dstc::x_copy<C>(p, step, i, &row, &off);
    if (src) {
      std::memcpy(&s.x[row][off], src, 16);
    } else {
      std::memset(&s.x[row][off], 0, 16);
    }
  }
  if (dstc::scales_by_words(p)) {
    ++word_staged;
    for (uint32_t i = 0; i < (C::kFeatures + C::kTokens) * (C::kStepBlocks / 4); ++i) {
      uint8_t* dst;
      const uint8_t* src = dstc::scale_copy<C>(p, step, i, s, &dst);
      if (src) {
        std::memcpy(dst, src, 4);
      } else {
        std::memset(dst, 0, 4);
      }
    }
  } else {
    ++byte_staged;
    for (uint32_t i = 0; i < (C::kFeatures + C::kTokens) * C::kStepBlocks; ++i) {
      dstc::stage_scale<C>(p, step, i, s);
    }
  }
}

// `tile_pass`: warps side by side, accumulating in registers.
template <class C, bool kBanded>
void tile_pass(const dstc::Pass& p, Adder adder, Rng& rng) {
  constexpr uint32_t kMT = C::kMTiles, kNT = C::kNTiles;
  auto s = std::make_unique<dstc::Stage<C>>();
  const uint32_t nt0 = dstc::ntile_first(p), nt1 = dstc::ntile_last(p);
  static float acc[dstc::kWarps][32][kMT][kNT][4];
  std::memset(acc, 0, sizeof acc);
  for (uint32_t step = 0; step < dstc::steps<C>(p); ++step) {
    fill<C>(*s, p, step);
    const uint32_t vb = dstc::blocks_in_step<C>(p, step);
    for (uint32_t warp = 0; warp < dstc::kWarps; ++warp) {
      const uint32_t frow = warp * C::kWarpFeatures;
      if (p.f0 + frow >= p.n) continue;
      for (uint32_t blk = 0; blk < vb; ++blk) {
        static float dot[32][kMT][kNT][4];
        block_dots<C, kBanded>(*s, frow, blk, nt0, nt1, adder, rng, dot);
        for (uint32_t l = 0; l < 32; ++l) {
          for (uint32_t mt = 0; mt < kMT; ++mt) {
            for (uint32_t nt = nt0; nt < nt1; ++nt) {
              for (uint32_t i = 0; i < 4; ++i) {
                acc[warp][l][mt][nt][i] =
                    dstc::accumulate<C>(*s, dstc::d_feature(frow, l, mt, i), dstc::d_token(l, nt, i),
                                        blk, acc[warp][l][mt][nt][i], dot[l][mt][nt][i]);
              }
            }
          }
        }
      }
    }
  }
  for (uint32_t warp = 0; warp < dstc::kWarps; ++warp) {
    const uint32_t frow = warp * C::kWarpFeatures;
    if (p.f0 + frow >= p.n) continue;
    for (uint32_t l = 0; l < 32; ++l) {
      for (uint32_t mt = 0; mt < kMT; ++mt) {
        for (uint32_t nt = 0; nt < kNT; ++nt) {
          for (uint32_t i = 0; i < 4; ++i) {
            dstc::store(p, dstc::d_feature(frow, l, mt, i), dstc::d_token(l, nt, i),
                        acc[warp][l][mt][nt][i]);
          }
        }
      }
    }
  }
}

// `split_pass`: per chunk, each warp's blocks (warps in descending order)
// from operands loaded per lane into contributions that start as garbage,
// then each thread's ordered sum.
template <bool kFp4, bool kBanded>
void split_pass(const dstc::Pass& p, Adder adder, Rng& rng) {
  constexpr uint32_t kPer = dstc::split_per_warp(kBanded), kChunk = dstc::kSplitWarps * kPer;
  const uint32_t kb = p.k / 32;
  const uint32_t nt0 = dstc::ntile_first(p), nt1 = dstc::ntile_last(p);
  auto c = std::make_unique<dstc::SplitContributions>();
  constexpr uint32_t kSums = dstc::kSplitFeatures * dstc::kSplitTokens;
  float acc[kSums] = {};
  for (uint32_t c0 = 0; c0 < kb; c0 += kChunk) {
    std::memset(c.get(), 0x5a, sizeof *c);
    for (uint32_t warp = dstc::kSplitWarps; warp-- > 0;) {
      for (uint32_t j = 0; j < kPer; ++j) {
        const uint32_t blk = c0 + warp * kPer + j;
        if (blk >= kb) continue;
        static dstc::SplitOperands o[32];
        static uint32_t a[32][1][2][4], b[32][1][2][2];
        static float dot[32][1][1][4];
        for (uint32_t l = 0; l < 32; ++l) {
          dstc::split_operands<kFp4>(dstc::split_rows<kFp4>(p, l), blk, &o[l]);
          dstc::split_frags<kFp4>(o[l], a[l], b[l]);
        }
        frag_dots<kFp4, kBanded>(a, b, nt0, nt1, adder, rng, dot);
        for (uint32_t l = 0; l < 32; ++l) {
          for (uint32_t i = 0; i < 4; ++i) {
            c->c[warp * kPer + j][dstc::d_feature(0, l, 0, i)][dstc::d_token(l, 0, i)] =
                dstc::split_contribution(o[l], i, dot[l][0][0][i]);
          }
        }
      }
    }
    const uint32_t n = std::min(kChunk, kb - c0);
    for (uint32_t t = 0; t < kSums; ++t) {
      for (uint32_t blk = 0; blk < n; ++blk) {
        acc[t] = acc[t] + c->c[blk][t / dstc::kSplitTokens][t % dstc::kSplitTokens];
      }
    }
  }
  for (uint32_t t = 0; t < kSums; ++t) {
    dstc::store(p, t / dstc::kSplitTokens, t % dstc::kSplitTokens, acc[t]);
  }
}

// The split block's shape, for the grids below.
template <bool kFp4_>
struct SplitShape {
  static constexpr bool kFp4 = kFp4_;
  static constexpr uint32_t kFeatures = dstc::kSplitFeatures;
  static constexpr uint32_t kTokens = dstc::kSplitTokens;
};
using SplitShape8 = SplitShape<false>;
using SplitShape4 = SplitShape<true>;

template <class C, bool kSplit, bool kBanded>
void run_pass(const dstc::Pass& p, Adder adder, Rng& rng) {
  if constexpr (kSplit) {
    split_pass<C::kFp4, kBanded>(p, adder, rng);
  } else {
    tile_pass<C, kBanded>(p, adder, rng);
  }
}

// The grid of `ds_gemm_tc.cu`'s entries, blocks in descending order.
template <class C, bool kSplit, bool kBanded>
void lin8(const DsLin8Params& p, Adder adder, Rng& rng) {
  const uint32_t gx = (p.n + C::kFeatures - 1) / C::kFeatures;
  const uint32_t gy = (p.rows + C::kTokens - 1) / C::kTokens;
  for (uint32_t y = gy; y-- > 0;) {
    for (uint32_t x = gx; x-- > 0;) {
      const uint32_t r0 = y * C::kTokens, r1 = std::min(p.rows, r0 + C::kTokens);
      run_pass<C, kSplit, kBanded>(dstc::lin8_pass(p, r0, r1, x * C::kFeatures), adder, rng);
    }
  }
}

template <class C, bool kSplit, bool kBanded>
void lin4(const DsLin4Params& p, Adder adder, Rng& rng) {
  const uint32_t gx = (p.n + C::kFeatures - 1) / C::kFeatures;
  const uint32_t gy = (p.layout_rows + C::kTokens - 1) / C::kTokens;
  for (uint32_t y = gy; y-- > 0;) {
    for (uint32_t x = gx; x-- > 0;) {
      const uint32_t r0 = y * C::kTokens, r1 = std::min(p.layout_rows, r0 + C::kTokens);
      for (uint32_t g = dstc::first_group(p, r0); g < p.groups; ++g) {
        if (dstc::group_start(p, g) >= r1) break;
        const dstc::Pass pass = dstc::lin4_pass(p, g, r0, r1, x * C::kFeatures);
        if (pass.lo < pass.hi) run_pass<C, kSplit, kBanded>(pass, adder, rng);
      }
    }
  }
}

// ---- cases ----

// A code: any non-NaN E4M3 code (every exponent, both signs), or one of a
// quantised block (exponents near the top, as a scaled activation's are).
uint8_t code(Rng& g, bool wide) {
  for (;;) {
    uint8_t c = static_cast<uint8_t>(g.next());
    if (!wide) c = static_cast<uint8_t>((c & 0x87u) | (10u + g.below(6)) << 3);
    if (g.below(13) == 0) c &= 0x80u;
    if (!ds::e4m3_is_nan(c)) return c;
  }
}

std::vector<uint8_t> codes(Rng& g, size_t n, bool wide) {
  std::vector<uint8_t> v(n);
  for (auto& c : v) c = code(g, wide);
  return v;
}

std::vector<uint8_t> scales(Rng& g, size_t n, uint32_t base, uint32_t spread) {
  std::vector<uint8_t> v(n);
  for (auto& c : v) c = static_cast<uint8_t>(base + g.below(spread));
  return v;
}

struct Tally {
  uint64_t outputs = 0, banded_diff = 0, tolerance_diff = 0;
};

void compare(const std::vector<uint16_t>& want, const std::vector<uint16_t>& got, bool banded,
             Tally& t, const char* what) {
  uint64_t diff = 0;
  for (size_t i = 0; i < want.size(); ++i) diff += want[i] != got[i];
  t.outputs += want.size();
  if (banded) {
    t.banded_diff += diff;
    if (diff) std::printf("FAILED %s: %llu of %zu outputs differ\n", what,
                          static_cast<unsigned long long>(diff), want.size());
    if (diff) ++failures;
  } else {
    t.tolerance_diff += diff;
  }
}

void lin8_cases(Rng& g, Tally& t) {
  for (uint32_t c = 0; c < 48; ++c) {
    const uint32_t rows = 1 + g.below(c < 40 ? 40 : 70), n = 1 + g.below(c < 24 ? 300 : 700);
    const uint32_t k = 32 * (1 + g.below(c < 24 ? 9 : 20));
    const bool wide = g.below(2) == 0;
    auto x = codes(g, size_t(rows) * k, wide);
    // One case in four over every finite scale: products that underflow,
    // overflow and turn infinite, where the order of the two scalings shows.
    const bool extreme = c % 4 == 3;
    auto xs = extreme ? scales(g, size_t(rows) * k / 32, 1, 254) : scales(g, size_t(rows) * k / 32, 100, 50);
    auto w = codes(g, size_t(n) * k, true);
    auto ws = extreme ? scales(g, size_t((n + 31) / 32) * (k / 32), 0, 255)
                      : scales(g, size_t((n + 31) / 32) * (k / 32), 110, 30);
    if (c == 5) xs[0] = 255;  // an infinite activation scale
    if (c % 8 == 6) {
      // The largest block dot: every code 448 (or -448 in the activations'
      // first row), where the exact sum is near its bound.
      for (auto& v : x) v = 0x7e;
      for (uint32_t j = 0; j < k; ++j) x[j] = 0xfe;
      for (auto& v : w) v = 0x7e;
    }
    std::vector<uint16_t> want(size_t(rows) * n, 0xdead);
    DsLin8Params p{x.data(), xs.data(), w.data(), ws.data(), want.data(), rows, n, k, 0};
    for (uint32_t row = rows; row-- > 0;) {
      for (uint32_t bx = (n + 255) / 256; bx-- > 0;) {
        for (uint32_t th = dsk::kElementThreads; th-- > 0;) dsk::lin8_thread(p, bx, row, th);
      }
    }
    for (int form = 0; form < 6; ++form) {
      std::vector<uint16_t> got(want.size(), 0xdead);
      p.out = got.data();
      const Adder adder = c % 2 ? Adder::Shuffled : Adder::Exact;
      using S = SplitShape8;
      using T8 = dstc::TileCfg<false, 8>;
      using T32 = dstc::TileCfg<false, 32>;
      switch (form) {
        case 0: lin8<S, true, true>(p, adder, g); break;
        case 1: lin8<T8, false, true>(p, adder, g); break;
        case 2: lin8<T32, false, true>(p, adder, g); break;
        case 3: lin8<S, true, false>(p, adder, g); break;
        case 4: lin8<T8, false, false>(p, adder, g); break;
        default: lin8<T32, false, false>(p, adder, g); break;
      }
      compare(want, got, form < 3, t, "lin8");
    }
  }
}

void lin4_cases(Rng& g, Tally& t) {
  for (uint32_t c = 0; c < 48; ++c) {
    const uint32_t groups = 1 + g.below(c < 24 ? 6 : 24), n = 1 + g.below(c < 24 ? 260 : 600);
    const uint32_t k = 32 * (1 + g.below(c < 24 ? 9 : 16));
    const uint32_t aligns[] = {1, 3, 8, 16, 32};
    const uint32_t row_align = aligns[g.below(5)];
    std::vector<uint32_t> end(groups);
    uint32_t at = 0;
    for (uint32_t i = 0; i < groups; ++i) {
      at = dstc::align_up(at, row_align);
      // Some groups empty, some spanning tiles.
      const uint32_t count = g.below(4) == 0 ? 0 : g.below(c < 24 ? 12 : 70);
      at += count;
      end[i] = at;
    }
    const uint32_t layout_rows = at + g.below(9);
    if (layout_rows == 0) continue;
    const bool wide = g.below(2) == 0;
    auto x = codes(g, size_t(layout_rows) * k, wide);
    const bool extreme = c % 4 == 3;
    auto xs = extreme ? scales(g, size_t(layout_rows) * k / 32, 1, 254)
                      : scales(g, size_t(layout_rows) * k / 32, 100, 50);
    std::vector<uint8_t> w(size_t(groups) * n * k / 2);
    for (auto& b : w) b = static_cast<uint8_t>(g.next());
    auto ws = extreme ? scales(g, size_t(groups) * n * (k / 32), 0, 255)
                      : scales(g, size_t(groups) * n * (k / 32), 110, 30);
    std::vector<uint16_t> want(size_t(layout_rows) * n, 0xdead);
    DsLin4Params p{x.data(), xs.data(), w.data(), ws.data(), end.data(), want.data(),
                   layout_rows, n, k, groups, row_align, 0};
    for (uint32_t row = layout_rows; row-- > 0;) {
      for (uint32_t bx = (n + 255) / 256; bx-- > 0;) {
        for (uint32_t th = dsk::kElementThreads; th-- > 0;) dsk::lin4_thread(p, bx, row, th);
      }
    }
    for (int form = 0; form < 6; ++form) {
      std::vector<uint16_t> got(want.size(), 0xdead);
      p.out = got.data();
      const Adder adder = c % 2 ? Adder::Shuffled : Adder::Exact;
      using S = SplitShape4;
      using T8 = dstc::TileCfg<true, 8>;
      using T32 = dstc::TileCfg<true, 32>;
      switch (form) {
        case 0: lin4<S, true, true>(p, adder, g); break;
        case 1: lin4<T8, false, true>(p, adder, g); break;
        case 2: lin4<T32, false, true>(p, adder, g); break;
        case 3: lin4<S, true, false>(p, adder, g); break;
        case 4: lin4<T8, false, false>(p, adder, g); break;
        default: lin4<T32, false, false>(p, adder, g); break;
      }
      compare(want, got, form < 3, t, "lin4");
    }
  }
}

}  // namespace

int main() {
  conversions();
  bands();
  Rng g{0x7c9e};
  Tally t8, t4;
  lin8_cases(g, t8);
  lin4_cases(g, t4);
  std::printf("lin8: %llu outputs, banded differing %llu, tolerance differing %llu\n",
              static_cast<unsigned long long>(t8.outputs), static_cast<unsigned long long>(t8.banded_diff),
              static_cast<unsigned long long>(t8.tolerance_diff));
  std::printf("lin4: %llu outputs, banded differing %llu, tolerance differing %llu\n",
              static_cast<unsigned long long>(t4.outputs), static_cast<unsigned long long>(t4.banded_diff),
              static_cast<unsigned long long>(t4.tolerance_diff));
  std::printf("scale staging: %llu steps by words, %llu by bytes\n",
              static_cast<unsigned long long>(word_staged), static_cast<unsigned long long>(byte_staged));
  check(word_staged > 1000 && byte_staged > 1000, "both scale staging paths ran");
  std::printf("failures: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
