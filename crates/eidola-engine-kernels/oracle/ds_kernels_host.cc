// Host check of the DeepSeek scalar kernels' bodies: `csrc/ds_ops.cuh`,
// compiled for the host and run thread by thread over each kernel's launch
// grid, against the digests the Rust reference emits
// (`eidola-engine-cuda`'s `tests/ds_cases`, which documents the file format
// and draws every case's inputs; `make` here repeats its draws).
//
// Dev-only: never a build input of the kernels or of anything shipped. It
// shows the bodies compute the reference's outputs and that the grids reach
// every unit; what nvcc emits for the device is held by the `exact` profile,
// the SASS assertions and the GPU tests. Blocks and threads run in descending
// order here, so a body that leaned on the order of a launch would differ.
//
//   c++ -std=c++17 -O2 -ffp-contract=off -I csrc oracle/ds_kernels_host.cc -o ds_kernels_host
//   ./ds_kernels_host tests/data/ds_ops.vectors tests/data/ds_gemm_ref.vectors
//
// `tests/ds_kernels_host.rs` runs exactly that. Exit status 0 means every
// record matched.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ds_ops.cuh"

namespace {

constexpr uint64_t kGolden = 0x9e3779b97f4a7c15ull;

uint64_t fin(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

using U8 = std::vector<uint8_t>;
using U16 = std::vector<uint16_t>;
using U32 = std::vector<uint32_t>;
using F32 = std::vector<float>;

struct Rng {
  uint64_t state;
  uint64_t next() {
    state += kGolden;
    return fin(state);
  }
  size_t below(size_t n) { return static_cast<size_t>(next() % n); }
  float f32_in(uint32_t base, uint32_t spread) {
    const uint64_t r = next();
    const uint32_t exp = base + static_cast<uint32_t>((r >> 32) & 0xff) % spread;
    return ds::f32_from_bits(static_cast<uint32_t>(r >> 63) << 31 | exp << 23 |
                             static_cast<uint32_t>(r & 0x7fffff));
  }
  F32 f32s(size_t n, uint32_t base, uint32_t spread) {
    F32 v(n);
    for (auto& x : v) x = f32_in(base, spread);
    return v;
  }
  void bf16_range(U16& out, size_t n, uint32_t base, uint32_t spread) {
    for (size_t i = 0; i < n; ++i) {
      const uint64_t r = next();
      const uint16_t sign = static_cast<uint16_t>((r >> 63) << 15);
      if ((r >> 40) % 61 == 0) {
        out.push_back(sign);
      } else {
        const uint32_t exp = base + static_cast<uint32_t>((r >> 32) & 0xff) % spread;
        out.push_back(static_cast<uint16_t>(sign | exp << 7 | (r & 0x7f)));
      }
    }
  }
  void bf16s(U16& out, size_t n) {
    const uint32_t base = 100 + static_cast<uint32_t>(below(40));
    bf16_range(out, n, base, 8);
  }
  U16 bf16_rows(size_t rows, size_t n) {
    U16 out;
    for (size_t r = 0; r < rows; ++r) bf16s(out, n);
    return out;
  }
  void bf16_wide(U16& out, size_t n) {
    const uint32_t base = 90 + static_cast<uint32_t>(below(20));
    bf16_range(out, n, base, 40);
  }
  U16 bf16_blocks(size_t n) {
    U16 out;
    for (size_t b = 0; b < n / 32; ++b) {
      const uint32_t base = 1 + static_cast<uint32_t>(below(200));
      bf16_range(out, 32, base, 12);
    }
    return out;
  }
  U8 codes(size_t n) {
    U8 v(n);
    for (auto& c : v) {
      const uint8_t b = static_cast<uint8_t>(next() & 0xff);
      c = (b & 0x7f) == 0x7f ? b - 1 : b;
    }
    return v;
  }
  U8 bytes(size_t n) {
    U8 v(n);
    for (auto& c : v) c = static_cast<uint8_t>(next() & 0xff);
    return v;
  }
  U8 scales(size_t n) {
    U8 v(n);
    for (auto& c : v) c = static_cast<uint8_t>(100 + below(56));
    return v;
  }
  void ascending_ids(U32& out, size_t active, size_t groups) {
    size_t at = 0;
    for (size_t j = 0; j < active; ++j) {
      at += below(groups - at - (active - j) + 1);
      out.push_back(static_cast<uint32_t>(at++));
    }
  }
};

struct Digest {
  uint64_t h = 0;
  void word(uint64_t w) { h = fin(h + kGolden + w); }
  void add(const U8& v) {
    for (auto x : v) word(x);
  }
  void add(const U16& v) {
    for (auto x : v) word(x);
  }
  void add(const U32& v) {
    for (auto x : v) word(x);
  }
  void add(const F32& v) {
    for (auto x : v) word(ds::f32_bits(x));
  }
};

// What an output buffer holds before a launch.
constexpr uint8_t kS8 = 0xa5;
constexpr uint16_t kS16 = 0xa5a5;
constexpr uint32_t kS32 = 0xa5a5a5a5u;
constexpr float kSF = -7777.0f;
constexpr float kEps = 1e-6f;

uint32_t u32(size_t x) { return static_cast<uint32_t>(x); }
uint32_t ceil_div(size_t a, size_t b) { return u32((a + b - 1) / b); }

// A launch of independent threads over grid (gx, gy, gz), last block and
// thread first.
template <class F>
void launch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t threads, F body) {
  for (uint32_t z = gz; z-- > 0;)
    for (uint32_t y = gy; y-- > 0;)
      for (uint32_t x = gx; x-- > 0;)
        for (uint32_t t = threads; t-- > 0;) body(x, y, z, t);
}

// One block of a pinned-order kernel: `lane(group, lane, acc)` on every
// thread, then `finish(group, chain, lanes)` on each group's first thread
// with the group's 16 lane sums of that chain (what the device collects by
// warp shuffles).
template <class L, class F>
void lane_block(uint32_t groups, uint32_t chains, L lane, F finish) {
  std::vector<float> acc(static_cast<size_t>(groups) * 16 * chains);
  for (uint32_t t = groups * 16; t-- > 0;) {
    lane(t / 16, t % 16, &acc[static_cast<size_t>(t) * chains]);
  }
  for (uint32_t g = groups; g-- > 0;) {
    for (uint32_t u = 0; u < chains; ++u) {
      float lanes[16];
      for (uint32_t j = 0; j < 16; ++j) {
        lanes[j] = acc[(static_cast<size_t>(g) * 16 + j) * chains + u];
      }
      finish(g, u, lanes);
    }
  }
}

size_t sz(Rng& g, bool wide, size_t wide_value, size_t bound) {
  return wide ? wide_value : 1 + g.below(bound);
}

// `quant_fp8` over BF16 rows.
void quant_rows(const U16& x, size_t rows, size_t width, U8& codes, U8& scales) {
  codes.assign(rows * width, kS8);
  scales.assign(rows * width / 32, kS8);
  DsQuantFp8Params p{};
  p.x = x.data();
  p.codes = codes.data();
  p.scales = scales.data();
  p.rows = u32(rows);
  p.width = u32(width);
  p.x_format = DS_X_BF16;
  launch(ceil_div(width / 32, dsk::kBlockThreads), p.rows, 1, dsk::kBlockThreads,
         [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { dsk::quant_fp8_thread(p, bx, y, t); });
}

// Both placement launches.
void place(DsPlaceParams p) {
  U32 scratch(p.groups, kS32);
  p.scratch = scratch.data();
  const uint32_t gx = ceil_div(p.groups, dsk::kBlockThreads);
  launch(gx, 1, 1, dsk::kBlockThreads,
         [&](uint32_t x, uint32_t, uint32_t, uint32_t t) { dsk::place_count_thread(p, x, t); });
  launch(gx, 1, 1, dsk::kBlockThreads,
         [&](uint32_t x, uint32_t, uint32_t, uint32_t t) { dsk::place_thread(p, x, t); });
}

// Rows an expert-major layout uses (`ds::ops::expert_layout`).
size_t layout_rows(const U32& ids, size_t groups, size_t row_align) {
  std::vector<size_t> count(groups, 0);
  for (auto id : ids) ++count[id];
  size_t end = 0;
  for (auto c : count) end = (end + row_align - 1) / row_align * row_align + c;
  return end;
}

void linear_case(bool head, bool wide, Rng& g, Digest& d) {
  const size_t rows = sz(g, wide, head ? 9 : 2, head ? 20 : 3);
  const size_t n = sz(g, wide, head ? 100 : 300, head ? 40 : 70);
  const size_t k = sz(g, wide, 5120, 100);
  const bool x_f32 = g.below(2) == 1;
  const bool w_f32 = g.below(2) == 1;
  const bool out_f32 = g.below(2) == 1;
  const bool indexed = g.below(2) == 1;
  const bool added = g.below(2) == 1;
  const size_t x_rows = indexed ? rows + g.below(3) : rows;
  F32 xf, wf;
  U16 xb, wb;
  if (x_f32) {
    xf = g.f32s(x_rows * k, 115, 10);
  } else {
    g.bf16s(xb, x_rows * k);
  }
  if (w_f32) {
    wf = g.f32s(n * k, 115, 10);
  } else {
    g.bf16s(wb, n * k);
  }
  U32 index;
  if (indexed) {
    for (size_t r = 0; r < rows; ++r) index.push_back(u32(g.below(x_rows)));
  }
  F32 addend;
  if (added) addend = g.f32s(rows * n, 115, 14);
  F32 out_f(out_f32 ? rows * n : 0, kSF);
  U16 out_b(out_f32 ? 0 : rows * n, kS16);
  DsLinearParams p{};
  p.x = x_f32 ? static_cast<const void*>(xf.data()) : xb.data();
  p.w = w_f32 ? static_cast<const void*>(wf.data()) : wb.data();
  p.x_rows = indexed ? index.data() : nullptr;
  p.addend = added ? addend.data() : nullptr;
  p.out = out_f32 ? static_cast<void*>(out_f.data()) : out_b.data();
  p.rows = u32(rows);
  p.n = u32(n);
  p.k = u32(k);
  p.x_dtype = x_f32 ? DS_DT_F32 : DS_DT_BF16;
  p.w_dtype = w_f32 ? DS_DT_F32 : DS_DT_BF16;
  p.out_dtype = out_f32 ? DS_DT_F32 : DS_DT_BF16;
  const uint32_t groups = dsk::kElementThreads / 16;
  if (head) {
    for (uint32_t by = ceil_div(rows, dsk::kHeadChains); by-- > 0;) {
      for (uint32_t bx = ceil_div(n, 16); bx-- > 0;) {
        const uint32_t row0 = by * dsk::kHeadChains;
        lane_block(
            groups, dsk::kHeadChains,
            [&](uint32_t grp, uint32_t lane, float* acc) {
              dsk::head_lane(p, row0, bx * 16 + grp, lane, acc);
            },
            [&](uint32_t grp, uint32_t u, float* lanes) {
              const uint32_t nn = bx * 16 + grp;
              if (nn < p.n && row0 + u < p.rows) dsk::linear_finish(p, row0 + u, nn, lanes);
            });
      }
    }
  } else {
    for (uint32_t row = p.rows; row-- > 0;) {
      for (uint32_t bx = ceil_div(n, 16 * dsk::kLinearChains); bx-- > 0;) {
        lane_block(
            groups, dsk::kLinearChains,
            [&](uint32_t grp, uint32_t lane, float* acc) {
              dsk::linear_lane(p, row, (bx * 16 + grp) * dsk::kLinearChains, lane, acc);
            },
            [&](uint32_t grp, uint32_t u, float* lanes) {
              const uint32_t nn = (bx * 16 + grp) * dsk::kLinearChains + u;
              if (nn < p.n) dsk::linear_finish(p, row, nn, lanes);
            });
      }
    }
  }
  if (out_f32) {
    d.add(out_f);
  } else {
    d.add(out_b);
  }
}

// One case of `kernel`: its inputs drawn as `ds_cases::build` draws them, its
// launch run, its outputs folded into `d`.
void make(const std::string& kernel, bool wide, Rng& g, Digest& d) {
  using namespace dsk;
  if (kernel == "embed") {
    const size_t vocab = sz(g, wide, 300, 40);
    const size_t dim = sz(g, wide, 5120, 600);
    const size_t copies = sz(g, wide, 4, 4);
    const size_t rows = sz(g, wide, 3, 5);
    U16 table;
    g.bf16s(table, vocab * dim);
    U32 tokens;
    for (size_t r = 0; r < rows; ++r) {
      size_t t = g.below(vocab);
      if (g.below(8) == 0) t = vocab + g.below(3);
      tokens.push_back(u32(t));
    }
    U16 out(rows * copies * dim, kS16);
    U32 status(1, 0);
    DsEmbedParams p{table.data(), tokens.data(), out.data(), status.data(),
                    u32(rows),    u32(dim),      u32(copies), u32(vocab)};
    launch(ceil_div(dim, kElementThreads), p.rows, 1, kElementThreads,
           [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { embed_thread(p, x, y, t); });
    d.add(out);
    d.add(status);
  } else if (kernel == "quant_fp8") {
    const size_t rows = sz(g, wide, 2, 4);
    const size_t width = 32 * sz(g, wide, 160, 8);
    const bool fp8 = g.below(2) == 1;
    U8 xc, xs;
    U16 xb;
    if (fp8) {
      xc = g.codes(rows * width);
      xs = g.scales(rows * width / 32);
    } else {
      xb = g.bf16_blocks(rows * width);
    }
    U8 codes(rows * width, kS8), scales(rows * width / 32, kS8);
    DsQuantFp8Params p{};
    p.x = fp8 ? static_cast<const void*>(xc.data()) : xb.data();
    p.x_scales = fp8 ? xs.data() : nullptr;
    p.codes = codes.data();
    p.scales = scales.data();
    p.rows = u32(rows);
    p.width = u32(width);
    p.x_format = fp8 ? DS_X_FP8 : DS_X_BF16;
    launch(ceil_div(width / 32, kBlockThreads), p.rows, 1, kBlockThreads,
           [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { quant_fp8_thread(p, x, y, t); });
    d.add(codes);
    d.add(scales);
  } else if (kernel == "rot_quant" || kernel == "rot_quant_e4m3") {
    const bool e4m3 = kernel == "rot_quant_e4m3";
    const size_t rows = sz(g, wide, 2, 3);
    const bool pooled = g.below(2) == 1;
    const size_t heads = pooled ? 1 : sz(g, wide, 4, 3);
    const size_t head_dim = 32 * sz(g, wide, 16, 4);
    const size_t rope_dim = wide ? 64 : 2 * (1 + g.below(head_dim / 2));
    const size_t rotation = g.below(3);
    const uint32_t format = e4m3 ? DS_KV_FP4_E4M3 : g.below(2) == 0 ? DS_KV_FP8 : DS_KV_FP4_E8M0;
    const size_t cis_positions = 4 + g.below(8);
    const F32 cis = g.f32s(cis_positions * rope_dim, 117, 10);
    U32 positions;
    for (size_t r = 0; r < rows; ++r) positions.push_back(u32(g.below(cis_positions)));
    const size_t width = heads * head_dim;
    const U16 x = g.bf16_blocks(rows * width);
    const size_t cb = ds_kv_code_bytes(format, u32(width));
    const size_t sb = ds_kv_scale_bytes(format, u32(width));
    DsRotQuantParams p{};
    p.x = x.data();
    p.cis = cis.data();
    p.positions = positions.data();
    p.rows = u32(rows);
    p.heads = u32(heads);
    p.head_dim = u32(head_dim);
    p.rope_dim = u32(rope_dim);
    p.rotation = u32(rotation);
    p.format = format;
    p.cis_positions = u32(cis_positions);
    U8 pool, codes, scales;
    U32 targets;
    if (pooled) {
      const size_t epb = 1 + g.below(5);
      const size_t blocks = 3 + g.below(3);
      const size_t codes_offset = g.below(4);
      const size_t scales_offset = codes_offset + epb * cb;
      const size_t block_bytes = scales_offset + epb * sb + g.below(4);
      const size_t total = epb * blocks;
      const size_t first = g.below(total);
      for (size_t r = 0; r < rows; ++r) {
        targets.push_back(g.below(4) == 0 ? DS_SLOT_EMPTY : u32((first + r) % total));
      }
      pool.assign(blocks * block_bytes, kS8);
      p.targets = targets.data();
      p.dst = DsKvPlane{pool.data(), block_bytes, codes_offset, scales_offset,
                        u32(epb),    format,      u32(width),   0};
    } else {
      codes.assign(rows * cb, kS8);
      scales.assign(rows * sb, kS8);
      p.dst_codes = codes.data();
      p.dst_scales = scales.data();
    }
    launch(ceil_div(width / 32, kBlockThreads), p.rows, 1, kBlockThreads,
           [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) {
             rot_quant_thread(p, bx, y, t, e4m3);
           });
    if (pooled) {
      d.add(pool);
    } else {
      d.add(codes);
      d.add(scales);
    }
  } else if (kernel == "rms_norm") {
    const size_t rows = sz(g, wide, 2, 3);
    const size_t width = sz(g, wide, 5120, 100);
    const U16 x = g.bf16_rows(rows, width);
    const F32 weight = g.f32s(width, 118, 8);
    U16 out(rows * width, kS16);
    DsRmsNormParams p{x.data(), weight.data(), out.data(), u32(rows), u32(width), kEps, 0};
    for (uint32_t row = p.rows; row-- > 0;) {
      float lanes[16];
      for (uint32_t j = 16; j-- > 0;) lanes[j] = rms_lane(p, row, j);
      const float factor = rms_finish(p, row, lanes);
      for (uint32_t t = kElementThreads; t-- > 0;) rms_store(p, row, t, factor);
    }
    d.add(out);
  } else if (kernel == "mhc_mix") {
    const size_t rows = sz(g, wide, 2, 2);
    const size_t copies = sz(g, wide, 4, 4);
    const size_t dim = sz(g, wide, 5120, 40);
    const size_t iters = sz(g, wide, 20, 20);
    const size_t n = (2 + copies) * copies;
    const U16 x = g.bf16_rows(rows, copies * dim);
    const F32 projection = g.f32s(n * copies * dim, 120, 4);
    F32 base = g.f32s(n, 122, 4);
    F32 out(rows * n, kSF);
    DsMhcMixParams p{};
    p.x = x.data();
    p.projection = projection.data();
    p.base = base.data();
    p.out = out.data();
    p.rows = u32(rows);
    p.dim = u32(dim);
    p.copies = u32(copies);
    p.iters = u32(iters);
    for (int j = 0; j < 3; ++j) p.scale[j] = g.f32_in(124, 3);
    p.norm_eps = kEps;
    p.eps = kEps;
    for (uint32_t row = p.rows; row-- > 0;) {
      float dots[kMixThreads / 16];
      lane_block(
          mix_groups(p.copies), 1,
          [&](uint32_t grp, uint32_t lane, float* acc) { *acc = mhc_lane(p, row, grp, lane); },
          [&](uint32_t grp, uint32_t, float* lanes) { dots[grp] = mhc_group(p, row, grp, lanes); });
      mhc_finish(p, row, dots);
    }
    d.add(out);
  } else if (kernel == "collapse" || kernel == "expand" || kernel == "stream_mean") {
    const size_t rows = sz(g, wide, 2, 3);
    const size_t copies = sz(g, wide, 4, 4);
    const size_t dim = sz(g, wide, 5120, 300);
    U16 x;
    for (size_t r = 0; r < rows; ++r) g.bf16_wide(x, copies * dim);
    const uint32_t gx = ceil_div(dim, kElementThreads);
    if (kernel == "collapse") {
      const size_t stride = copies + g.below(3);
      const F32 pre = g.f32s(rows * stride, 120, 6);
      U16 out(rows * dim, kS16);
      DsCollapseParams p{x.data(), pre.data(), out.data(), u32(rows),
                         u32(dim), u32(copies), u32(stride)};
      launch(gx, p.rows, 1, kElementThreads,
             [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { collapse_thread(p, bx, y, t); });
      d.add(out);
    } else if (kernel == "expand") {
      const U16 y = g.bf16_rows(rows, dim);
      const size_t stride = (2 + copies) * copies + g.below(3);
      const F32 mixes = g.f32s(rows * stride, 120, 6);
      U16 out(rows * copies * dim, kS16);
      DsExpandParams p{x.data(), y.data(), mixes.data(), out.data(),
                       u32(rows), u32(dim), u32(copies), u32(stride)};
      launch(gx, p.rows, 1, kElementThreads,
             [&](uint32_t bx, uint32_t by, uint32_t, uint32_t t) { expand_thread(p, bx, by, t); });
      d.add(out);
    } else {
      const size_t stride = dim + g.below(3);
      U16 out(rows * stride, kS16);
      DsStreamMeanParams p{x.data(), out.data(), u32(rows), u32(dim), u32(copies), u32(stride)};
      launch(gx, p.rows, 1, kElementThreads, [&](uint32_t bx, uint32_t by, uint32_t, uint32_t t) {
        stream_mean_thread(p, bx, by, t);
      });
      d.add(out);
    }
  } else if (kernel == "linear" || kernel == "head") {
    linear_case(kernel == "head", wide, g, d);
  } else if (kernel == "pool") {
    const size_t entries = sz(g, wide, 3, 4);
    const size_t width = sz(g, wide, 512, 300);
    const size_t src = 2 * entries + g.below(3);
    const F32 kv = g.f32s(src * width, 115, 10);
    const F32 score = g.f32s(src * width, 122, 8);
    U32 first, second;
    for (size_t e = 0; e < entries; ++e) first.push_back(u32(g.below(src)));
    for (size_t e = 0; e < entries; ++e) second.push_back(u32(g.below(src)));
    U16 out(entries * width, kS16);
    DsPoolParams p{kv.data(), score.data(), first.data(), second.data(),
                   out.data(), u32(entries), u32(width)};
    launch(ceil_div(width, kElementThreads), p.entries, 1, kElementThreads,
           [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { pool_thread(p, bx, y, t); });
    d.add(out);
  } else if (kernel == "route") {
    const size_t rows = sz(g, wide, 3, 3);
    const size_t experts = sz(g, wide, 384, 40);
    const size_t active = wide ? 6 : 1 + g.below(experts < 8 ? experts : 8);
    const bool drawn = g.below(2) == 1;
    const bool normalize = drawn && active > 1;
    const float route_scale = wide ? 1.5f : ds::abs32(g.f32_in(125, 3));
    const size_t mode = g.below(4);
    const bool ties = mode == 0, low = mode == 1;
    F32 logits(rows * experts), bias(experts);
    for (auto& v : logits) {
      v = ties  ? (static_cast<float>(g.below(7)) - 3.0f) * 0.5f
          : low ? -ds::abs32(g.f32_in(133, 1))
                : g.f32_in(120, 10);
    }
    for (auto& v : bias) v = ties ? static_cast<float>(g.below(3)) * 0.25f : g.f32_in(118, 6);
    U32 ids(rows * active, kS32);
    F32 weights(rows * active, kSF);
    DsRouteParams p{logits.data(), bias.data(),  ids.data(),         weights.data(), u32(rows),
                    u32(experts),  u32(active), normalize ? 1u : 0u, route_scale,    0};
    for (uint32_t row = p.rows; row-- > 0;) {
      float score[kMaxExperts], biased[kMaxExperts];
      for (uint32_t t = kRouteThreads; t-- > 0;) {
        for (uint32_t e = t; e < p.experts; e += kRouteThreads) {
          route_score(p, row, e, score, biased);
        }
      }
      route_select(p, row, score, biased);
    }
    d.add(ids);
    d.add(weights);
  } else if (kernel == "place") {
    const size_t rows = sz(g, wide, 64, 6);
    const size_t active = sz(g, wide, 6, 4);
    const size_t groups = wide ? 384 : active + g.below(12);
    const size_t row_align = sz(g, wide, 16, 4);
    U32 ids;
    for (size_t r = 0; r < rows; ++r) g.ascending_ids(ids, active, groups);
    F32 weights = g.f32s(rows * active, 120, 6);
    const size_t lrows = layout_rows(ids, groups, row_align) + g.below(3);
    U32 row_of(rows * active, kS32), group_end(groups, kS32);
    F32 row_weight(lrows, kSF);
    place(DsPlaceParams{ids.data(), weights.data(), row_of.data(), row_weight.data(),
                        group_end.data(), nullptr, u32(rows), u32(active), u32(groups),
                        u32(row_align), u32(lrows), 0});
    d.add(row_of);
    d.add(row_weight);
    d.add(group_end);
  } else if (kernel == "swiglu") {
    const size_t rows = sz(g, wide, 2, 3);
    const size_t width = 32 * sz(g, wide, 72, 4);
    const size_t in_stride = width + 32 * g.below(2);
    const uint32_t base = 120 + u32(g.below(12));
    U16 gate, up;
    g.bf16_range(gate, rows * in_stride, base, 8);
    g.bf16_range(up, rows * in_stride, base, 8);
    const float limit = g.below(2) == 0 ? 0.0f : 10.0f;
    const bool weighted = g.below(2) == 1;
    const F32 weight = g.f32s(rows, 120, 8);
    U8 codes(rows * width, kS8), scales(rows * width / 32, kS8);
    DsSwigluParams p{gate.data(),  up.data(),  weighted ? weight.data() : nullptr,
                     codes.data(), scales.data(), u32(rows), u32(width), u32(in_stride), limit};
    launch(ceil_div(width / 32, kBlockThreads), p.rows, 1, kBlockThreads,
           [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { swiglu_thread(p, bx, y, t); });
    d.add(codes);
    d.add(scales);
  } else if (kernel == "combine") {
    const size_t rows = sz(g, wide, 3, 4);
    const size_t active = sz(g, wide, 6, 6);
    const size_t dim = sz(g, wide, 5120, 300);
    const size_t lrows = rows * active + g.below(4);
    U32 row_of;
    for (size_t i = 0; i < rows * active; ++i) row_of.push_back(u32(g.below(lrows)));
    U16 routed, shared;
    g.bf16_wide(routed, lrows * dim);
    g.bf16_wide(shared, rows * dim);
    U16 out(rows * dim, kS16);
    DsCombineParams p{routed.data(), shared.data(), row_of.data(), out.data(),
                      u32(rows),     u32(active),   u32(dim),      0};
    launch(ceil_div(dim, kElementThreads), p.rows, 1, kElementThreads,
           [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { combine_thread(p, bx, y, t); });
    d.add(out);
  } else if (kernel == "engram_gather") {
    const size_t table_rows = 2 + g.below(20);
    const size_t width = 32 * sz(g, wide, 8, 3);
    const size_t rows = sz(g, wide, 2, 3);
    const size_t per = sz(g, wide, 24, 4);
    U8 codes = g.bytes(table_rows * width);
    U8 scales = g.bytes(table_rows * width / 32);
    U32 ids;
    for (size_t i = 0; i < rows * per; ++i) {
      size_t id = g.below(table_rows);
      if (g.below(8) == 0) id = table_rows + g.below(3);
      ids.push_back(u32(id));
    }
    U8 out_codes(rows * per * width, kS8), out_scales(rows * per * width / 32, kS8);
    U32 status(1, 0);
    DsEngramGatherParams p{codes.data(),      scales.data(), ids.data(), out_codes.data(),
                           out_scales.data(), status.data(), table_rows, u32(rows),
                           u32(per),          u32(width),    0};
    launch(ceil_div(width / 32, kBlockThreads), p.ids_per_row, p.rows, kBlockThreads,
           [&](uint32_t bx, uint32_t y, uint32_t z, uint32_t t) {
             engram_gather_thread(p, bx, y, z, t);
           });
    d.add(out_codes);
    d.add(out_scales);
    d.add(status);
  } else if (kernel == "engram_inject") {
    const size_t rows = sz(g, wide, 2, 2);
    const size_t copies = sz(g, wide, 4, 4);
    const size_t dim = sz(g, wide, 5120, 120);
    const U16 x = g.bf16_rows(rows, copies * dim);
    const U16 kv = g.bf16_rows(rows, (copies + 1) * dim);
    const F32 gate = g.f32s(copies * dim, 118, 8);
    U16 out(rows * copies * dim, kS16);
    DsEngramInjectParams p{x.data(), kv.data(), gate.data(), out.data(),
                           u32(rows), u32(dim), u32(copies), kEps};
    for (uint32_t row = p.rows; row-- > 0;) {
      for (uint32_t copy = p.copies; copy-- > 0;) {
        float sums[3];
        lane_block(
            3, 1,
            [&](uint32_t grp, uint32_t lane, float* acc) {
              *acc = inject_lane(p, row, copy, grp, lane);
            },
            [&](uint32_t grp, uint32_t, float* lanes) {
              sums[grp] = inject_group(p, row, copy, grp, lanes);
            });
        const float gate_value = inject_gate(p, sums);
        for (uint32_t t = kElementThreads; t-- > 0;) inject_store(p, row, copy, t, gate_value);
      }
    }
    d.add(out);
  } else if (kernel == "lin8") {
    const size_t rows = sz(g, wide, 2, 3);
    const size_t n = sz(g, wide, 300, 70);
    const size_t k = 32 * sz(g, wide, 160, 4);
    const U16 x = g.bf16_blocks(rows * k);
    const U8 w_codes = g.codes(n * k);
    const U8 w_scales = g.scales((n + 31) / 32 * (k / 32));
    U8 codes, scales;
    quant_rows(x, rows, k, codes, scales);
    U16 out(rows * n, kS16);
    DsLin8Params p{codes.data(), scales.data(), w_codes.data(), w_scales.data(), out.data(),
                   u32(rows),    u32(n),        u32(k),         0};
    launch(ceil_div(n, kElementThreads), p.rows, 1, kElementThreads,
           [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { lin8_thread(p, bx, y, t); });
    d.add(codes);
    d.add(scales);
    d.add(out);
  } else if (kernel == "lin4") {
    const size_t tokens = sz(g, wide, 4, 4);
    const size_t active = sz(g, wide, 3, 3);
    const size_t groups = wide ? 8 : active + g.below(5);
    const size_t row_align = 1 + g.below(4);
    const size_t n = sz(g, wide, 300, 40);
    const size_t k = 32 * sz(g, wide, 72, 3);
    const size_t kb = k / 32;
    U32 ids;
    for (size_t r = 0; r < tokens; ++r) g.ascending_ids(ids, active, groups);
    const U16 x = g.bf16_blocks(tokens * k);
    const U8 w_packed = g.bytes(groups * n * k / 2);
    const U8 w_scales = g.scales(groups * n * kb);
    const size_t lrows = layout_rows(ids, groups, row_align) + g.below(3);
    U8 codes, scales;
    quant_rows(x, tokens, k, codes, scales);
    // The layout from the placement kernels, the activations copied to it.
    F32 weights(tokens * active, 1.0f), row_weight(lrows, kSF);
    U32 row_of(tokens * active, kS32), group_end(groups, kS32);
    place(DsPlaceParams{ids.data(), weights.data(), row_of.data(), row_weight.data(),
                        group_end.data(), nullptr, u32(tokens), u32(active), u32(groups),
                        u32(row_align), u32(lrows), 0});
    U8 lcodes(lrows * k, 0), lscales(lrows * kb, 0);
    for (size_t pair = 0; pair < row_of.size(); ++pair) {
      const size_t token = pair / active, row = row_of[pair];
      std::copy_n(&codes[token * k], k, &lcodes[row * k]);
      std::copy_n(&scales[token * kb], kb, &lscales[row * kb]);
    }
    U16 out(lrows * n, kS16);
    DsLin4Params p{lcodes.data(), lscales.data(), w_packed.data(), w_scales.data(),
                   group_end.data(), out.data(), u32(lrows), u32(n), u32(k), u32(groups),
                   u32(row_align), 0};
    launch(ceil_div(n, kElementThreads), p.layout_rows, 1, kElementThreads,
           [&](uint32_t bx, uint32_t y, uint32_t, uint32_t t) { lin4_thread(p, bx, y, t); });
    d.add(out);
  } else if (kernel == "grouped_dot") {
    const size_t rows = sz(g, wide, 2, 2);
    const size_t groups = sz(g, wide, 2, 3);
    const size_t rank = sz(g, wide, 70, 70);
    const size_t k = 32 * sz(g, wide, 128, 4);
    const U16 o = g.bf16_blocks(rows * groups * k);
    const U8 w_codes = g.codes(groups * rank * k);
    const U8 w_scales = g.scales((groups * rank + 31) / 32 * (k / 32));
    U8 codes, scales;
    quant_rows(o, rows, groups * k, codes, scales);
    U16 out(rows * groups * rank, kS16);
    DsGroupedDotParams p{codes.data(), scales.data(), w_codes.data(), w_scales.data(), out.data(),
                         u32(rows),    u32(groups),   u32(rank),      u32(k)};
    for (uint32_t row = p.rows; row-- > 0;) {
      for (uint32_t grp = p.groups; grp-- > 0;) {
        for (uint32_t bx = ceil_div(rank, 16 * kLinearChains); bx-- > 0;) {
          lane_block(
              kElementThreads / 16, kLinearChains,
              [&](uint32_t lg, uint32_t lane, float* acc) {
                grouped_lane(p, row, grp, (bx * 16 + lg) * kLinearChains, lane, acc);
              },
              [&](uint32_t lg, uint32_t u, float* lanes) {
                const uint32_t i = (bx * 16 + lg) * kLinearChains + u;
                if (i < p.rank) grouped_finish(p, row, grp, i, lanes);
              });
        }
      }
    }
    d.add(out);
  } else {
    std::fprintf(stderr, "no kernel %s\n", kernel.c_str());
    std::exit(2);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectors>...\n", argv[0]);
    return 2;
  }
  size_t sweeps = 0, cases = 0;
  int failures = 0;
  for (int a = 1; a < argc; ++a) {
    std::ifstream in(argv[a]);
    if (!in) {
      std::fprintf(stderr, "cannot read %s\n", argv[a]);
      return 1;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream tokens(line);
      std::string kind, kernel, regime, seed, log2, equals, want;
      tokens >> kind >> kernel >> regime >> seed >> log2 >> equals >> want;
      if (kind != "s" || equals != "=" || (regime != "s" && regime != "w")) {
        std::fprintf(stderr, "unknown record: %s\n", line.c_str());
        return 1;
      }
      Rng g{std::stoull(seed, nullptr, 16)};
      Digest d;
      const uint64_t items = 1ull << std::stoull(log2, nullptr, 16);
      for (uint64_t i = 0; i < items; ++i) make(kernel, regime == "w", g, d);
      ++sweeps;
      cases += items;
      if (d.h != std::stoull(want, nullptr, 16)) {
        ++failures;
        std::fprintf(stderr, "MISMATCH %s %s: want %s, got %llx\n", kernel.c_str(), regime.c_str(),
                     want.c_str(), static_cast<unsigned long long>(d.h));
      }
    }
  }
  std::printf("ds_kernels_host: %zu sweeps, %zu cases; %d mismatches\n", sweeps, cases, failures);
  return failures == 0 ? 0 : 1;
}
