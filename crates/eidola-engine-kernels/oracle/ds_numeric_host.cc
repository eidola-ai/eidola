// Host check of the DeepSeek numerics header: every function of
// `csrc/ds_numeric.cuh`, compiled for the host, against the vectors the Rust
// reference emits (`eidola-engine-model`'s `tests/deepseek_numeric_vectors.rs`,
// which documents the file format and the sweep generators mirrored here).
//
// Dev-only: never a build input of the kernels or of anything shipped. It
// shows the header's arithmetic is the reference's; what nvcc emits for the
// device is held by the `exact` profile and the SASS assertions.
//
//   c++ -std=c++17 -O2 -ffp-contract=off -I csrc oracle/ds_numeric_host.cc -o ds_numeric_host
//   ./ds_numeric_host tests/data/ds_numeric.vectors [tests/data/ds_numeric.exhaustive]
//
// `tests/ds_numeric_host.rs` runs exactly that. Exit status 0 means every
// record matched.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ds_abi.cuh"
#include "ds_numeric.cuh"

namespace {

using Words = std::vector<uint64_t>;

constexpr uint64_t kGolden = 0x9e3779b97f4a7c15ull;

uint64_t fin(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

struct Rng {
  uint64_t state;
  uint64_t next() {
    state += kGolden;
    return fin(state);
  }
  uint64_t below(uint64_t n) { return next() % n; }
  uint32_t f32_in(uint32_t base, uint32_t spread) {
    const uint64_t r = next();
    const uint32_t exp = base + static_cast<uint32_t>((r >> 32) & 0xff) % spread;
    return static_cast<uint32_t>(r >> 63) << 31 | exp << 23 | static_cast<uint32_t>(r & 0x7fffff);
  }
  uint32_t moderate() { return f32_in(103, 48); }
  uint64_t f64_in(uint64_t base, uint64_t spread) {
    const uint64_t r = next();
    const uint64_t exp = base + below(spread);
    return (r >> 63) << 63 | exp << 52 | (r & 0x000fffffffffffffull);
  }
  void block(Words& out, size_t n) {
    const uint32_t base = 1 + static_cast<uint32_t>(below(230));
    for (size_t i = 0; i < n; ++i) out.push_back(f32_in(base, 12));
  }
  uint64_t code() {
    const uint64_t b = next() & 0xff;
    return (b & 0x7f) == 0x7f ? b - 1 : b;
  }
};

float f(uint64_t bits) { return ds::f32_from_bits(static_cast<uint32_t>(bits)); }
uint64_t w32(float x) { return ds::is_nan(x) ? 0x7fc00000ull : ds::f32_bits(x); }
uint64_t w64(double x) { return ds::is_nan(x) ? 0x7ff8000000000000ull : ds::f64_bits(x); }

const std::map<std::string, float (*)(float)> kUnary = {
    {"bf16_round", [](float x) { return ds::bf16_round(x); }},
    {"round_e4m3", [](float x) { return ds::round_e4m3(x); }},
    {"round_e2m1", [](float x) { return ds::round_e2m1(x); }},
    {"exp32", [](float x) { return ds::exp32(x); }},
    {"sigmoid32", [](float x) { return ds::sigmoid32(x); }},
    {"silu32", [](float x) { return ds::silu32(x); }},
    // The reference refuses a NaN here; the vectors record a NaN.
    {"softplus32", [](float x) { return ds::is_nan(x) ? x : ds::softplus32(x); }},
    {"rsqrt32", [](float x) { return ds::rsqrt32(x); }},
};

std::vector<float> floats(const Words& i, size_t from, size_t n) {
  std::vector<float> v(n);
  for (size_t j = 0; j < n; ++j) v[j] = f(i[from + j]);
  return v;
}
std::vector<uint8_t> bytes(const Words& i, size_t from, size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t j = 0; j < n; ++j) v[j] = static_cast<uint8_t>(i[from + j]);
  return v;
}

// A quantised GEMM row by row: quantise `x` in 32-wide blocks, then per
// output accumulate the exact block dots in ascending block order.
Words lin(const Words& i, bool fp4) {
  const size_t rows = i[0], kb = i[1], k = kb * 32;
  const std::vector<float> x = floats(i, 2, k);
  std::vector<uint8_t> codes(k), scales(kb);
  for (size_t b = 0; b < kb; ++b) scales[b] = ds::act_quant_block(&x[b * 32], &codes[b * 32]);
  const size_t wbytes = fp4 ? rows * k / 2 : rows * k;
  const std::vector<uint8_t> w = bytes(i, 2 + k, wbytes);
  const std::vector<uint8_t> ws = bytes(i, 2 + k + wbytes, i.size() - 2 - k - wbytes);
  Words out;
  for (size_t n = 0; n < rows; ++n) {
    float acc = 0.0f;
    for (size_t b = 0; b < kb; ++b) {
      const float d = fp4 ? ds::fp8_fp4_block_dot(&codes[b * 32], &w[n * k / 2],
                                                  static_cast<uint32_t>(b * 32))
                          : ds::fp8_fp8_block_dot(&codes[b * 32], &w[n * k + b * 32]);
      const uint8_t wscale = fp4 ? ws[n * kb + b] : ws[(n / 32) * kb + b];
      acc = ds::gemm_accumulate(acc, d, ds::act_scale_decode(scales[b]), ds::e8m0_decode(wscale));
    }
    out.push_back(w32(ds::bf16_round(acc)));
  }
  return out;
}

Words eval(const std::string& name, const Words& i) {
  const auto unary = kUnary.find(name);
  if (unary != kUnary.end()) return {w32(unary->second(f(i[0])))};
  if (name == "e4m3_decode") return {w32(ds::e4m3_decode(static_cast<uint8_t>(i[0])))};
  if (name == "e2m1_decode") return {w32(ds::e2m1_decode(static_cast<uint8_t>(i[0])))};
  if (name == "e8m0_decode") return {w32(ds::e8m0_decode(static_cast<uint8_t>(i[0])))};
  if (name == "round_scale") return {w32(ds::round_scale(f(i[0]), f(i[1])))};
  if (name == "det_exp") return {w64(ds::det_exp(ds::f64_from_bits(i[0])))};
  if (name == "det_ln") return {w64(ds::det_ln(ds::f64_from_bits(i[0])))};
  if (name == "det_sin_cos") {
    double s, c;
    ds::det_sin_cos(ds::f64_from_bits(i[0]), &s, &c);
    return {w64(s), w64(c)};
  }
  if (name == "pow32") return {w32(ds::pow32(ds::f64_from_bits(i[0]), f(i[1])))};
  if (name == "act_quant") {
    const std::vector<float> x = floats(i, 0, 32);
    uint8_t codes[32];
    const uint8_t scale = ds::act_quant_block(x.data(), codes);
    Words out;
    for (uint8_t c : codes) out.push_back(w32(ds::e4m3_decode(c)));
    out.push_back(w32(ds::act_scale_decode(scale)));
    return out;
  }
  if (name == "fq8" || name == "fq4m") {
    const std::vector<float> x = floats(i, 0, 32);
    uint8_t codes[32];
    Words out;
    if (name == "fq8") {
      const float s = ds::act_scale_decode(ds::act_quant_block(x.data(), codes));
      for (uint8_t c : codes) out.push_back(w32(ds::dequant_e4m3(c, s)));
    } else {
      const float s = ds::act_scale_decode(ds::fp4_pow2_quant_block(x.data(), codes));
      for (uint8_t c : codes) out.push_back(w32(ds::dequant_e2m1(c, s)));
    }
    return out;
  }
  if (name == "fq4n") {
    const std::vector<float> x = floats(i, 0, 16);
    uint8_t codes[16];
    const float s = ds::e4m3_decode(ds::fp4_e4m3_quant_block(x.data(), codes));
    Words out;
    for (uint8_t c : codes) out.push_back(w32(ds::dequant_e2m1(c, s)));
    return out;
  }
  if (name == "fp8_dot") {
    const std::vector<uint8_t> b = bytes(i, 0, 64);
    return {w32(ds::fp8_fp8_block_dot(&b[0], &b[32]))};
  }
  if (name == "fp4_dot") {
    const std::vector<uint8_t> b = bytes(i, 0, 48);
    return {w32(ds::fp8_fp4_block_dot(&b[0], &b[32], 0))};
  }
  if (name == "lin8") return lin(i, false);
  if (name == "lin4") return lin(i, true);
  if (name == "dot") {
    const size_t n = i[0];
    const std::vector<float> a = floats(i, 1, n), b = floats(i, 1 + n, n);
    return {w32(ds::pinned_dot(a.data(), b.data(), static_cast<uint32_t>(n)))};
  }
  if (name == "pinned_sum") {
    const std::vector<float> x = floats(i, 1, i[0]);
    return {w32(ds::pinned_sum(x.data(), static_cast<uint32_t>(i[0])))};
  }
  if (name == "rotate" || name == "unrotate") {
    float a, b;
    if (name == "rotate") {
      ds::rope_rotate(f(i[0]), f(i[1]), f(i[2]), f(i[3]), &a, &b);
    } else {
      ds::rope_unrotate(f(i[0]), f(i[1]), f(i[2]), f(i[3]), &a, &b);
    }
    return {w32(a), w32(b)};
  }
  if (name == "cis") {
    float c, s;
    ds::rope_cis(static_cast<uint32_t>(i[0]), f(i[1]), &c, &s);
    return {w32(c), w32(s)};
  }
  if (name == "rms") {
    const size_t n = i[0];
    const std::vector<float> x = floats(i, 2, n), w = floats(i, 2 + n, n);
    const float factor =
        ds::rms_factor(ds::pinned_dot(x.data(), x.data(), static_cast<uint32_t>(n)),
                       static_cast<uint32_t>(n), f(i[1]));
    Words out;
    for (size_t j = 0; j < n; ++j) out.push_back(w32(ds::rms_element(x[j], factor, w[j])));
    return out;
  }
  if (name == "swiglu") {
    return {w32(ds::swiglu_hidden(f(i[0]), f(i[1]), f(i[2]), i[3] != 0, f(i[4])))};
  }
  if (name == "top_k") {
    const size_t n = i[0], k = i[1] < n ? i[1] : n;
    const std::vector<float> scores = floats(i, 2, n);
    std::vector<bool> taken(n, false);
    Words out;
    for (size_t round = 0; round < k; ++round) {
      size_t best = n;
      for (size_t j = 0; j < n; ++j) {
        if (taken[j]) continue;
        if (best == n || ds::topk_before(scores[j], static_cast<uint32_t>(j), scores[best],
                                         static_cast<uint32_t>(best))) {
          best = j;
        }
      }
      taken[best] = true;
      out.push_back(best);
    }
    return out;
  }
  std::fprintf(stderr, "no function %s\n", name.c_str());
  std::exit(2);
}

Words sweep_item(const std::string& name, Rng& g) {
  Words v;
  if (kUnary.count(name)) {
    const uint64_t raw = g.next();
    const uint32_t moderate = g.moderate();
    v.push_back((raw >> 63) == 0 ? (raw & 0xffffffffull) : moderate);
  } else if (name == "round_scale") {
    v.push_back(g.f32_in(1, 254) & 0x7fffffffu);
    v.push_back((g.next() & 1) == 0 ? ds::kInv448Bits : ds::kInv6Bits);
  } else if (name == "det_exp") {
    v.push_back(g.f64_in(993, 42));
  } else if (name == "det_ln") {
    v.push_back(g.f64_in(1, 2046) & ~ds::kSign64);
  } else if (name == "det_sin_cos") {
    v.push_back(g.f64_in(983, 60));
  } else if (name == "pow32") {
    v.push_back(g.f64_in(1013, 30) & ~ds::kSign64);
    v.push_back(g.f32_in(110, 20));
  } else if (name == "act_quant" || name == "fq8" || name == "fq4m") {
    g.block(v, 32);
  } else if (name == "fq4n") {
    g.block(v, 16);
  } else if (name == "fp8_dot" || name == "fp4_dot") {
    v.push_back(0x7e | (g.next() & 0x80));
    for (int j = 1; j < 32; ++j) v.push_back(g.code());
    if (name == "fp8_dot") {
      for (int j = 0; j < 32; ++j) v.push_back(g.code());
    } else {
      for (int j = 0; j < 16; ++j) v.push_back(g.next() & 0xff);
    }
  } else if (name == "lin8" || name == "lin4") {
    const uint64_t rows = 1 + g.below(40);
    const uint64_t blocks = 1 + g.below(4);
    v.push_back(rows);
    v.push_back(blocks);
    for (uint64_t b = 0; b < blocks; ++b) g.block(v, 32);
    const uint64_t k = blocks * 32;
    if (name == "lin8") {
      for (uint64_t j = 0; j < rows * k; ++j) v.push_back(g.code());
      for (uint64_t j = 0; j < ((rows + 31) / 32) * blocks; ++j) v.push_back(100 + g.below(56));
    } else {
      for (uint64_t j = 0; j < rows * k / 2; ++j) v.push_back(g.next() & 0xff);
      for (uint64_t j = 0; j < rows * blocks; ++j) v.push_back(100 + g.below(56));
    }
  } else if (name == "dot" || name == "pinned_sum") {
    const uint64_t n = g.below(97);
    v.push_back(n);
    for (uint64_t j = 0; j < (name == "dot" ? 2 * n : n); ++j) v.push_back(g.moderate());
  } else if (name == "rotate" || name == "unrotate") {
    for (int j = 0; j < 4; ++j) v.push_back(g.moderate());
  } else if (name == "cis") {
    v.push_back(g.below(1 << 20));
    v.push_back(g.f32_in(103, 24) & 0x7fffffffu);
  } else if (name == "rms") {
    const uint64_t n = 1 + g.below(48);
    v.push_back(n);
    v.push_back(0x358637bdu);  // 1e-6
    g.block(v, n);
    for (uint64_t j = 0; j < n; ++j) v.push_back(g.moderate());
  } else if (name == "swiglu") {
    v.push_back(g.f32_in(117, 16));
    v.push_back(g.f32_in(117, 16));
    v.push_back((g.next() & 1) == 0 ? 0u : 0x41200000u);  // no limit, or 10
    v.push_back(g.next() & 1);
    v.push_back(g.f32_in(117, 12));
  } else if (name == "top_k") {
    const uint64_t n = 1 + g.below(64);
    const uint64_t k = 1 + g.below(n);
    v.push_back(n);
    v.push_back(k);
    for (uint64_t j = 0; j < n; ++j) {
      const uint64_t r = g.below(12);
      v.push_back(r == 0   ? 0x80000000u
                  : r == 1 ? 0x7fc00000u
                  : r == 2 ? 0xffc00000u
                           : ds::f32_bits(static_cast<float>(r) - 6.0f));
    }
  } else {
    std::fprintf(stderr, "no sweep for %s\n", name.c_str());
    std::exit(2);
  }
  return v;
}

uint64_t sweep(const std::string& name, uint64_t seed, uint64_t log2) {
  Rng g{seed};
  uint64_t h = 0;
  for (uint64_t item = 0; item < (1ull << log2); ++item) {
    for (uint64_t w : eval(name, sweep_item(name, g))) h = fin(h + kGolden + w);
  }
  return h;
}

uint64_t exhaustive(float (*func)(float)) {
  const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<uint64_t> sums(threads, 0);
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      uint64_t sum = 0;
      for (uint64_t hi = t; hi < 65536; hi += threads) {
        for (uint64_t lo = 0; lo < 65536; ++lo) {
          const uint64_t bits = hi << 16 | lo;
          sum += fin(bits << 32 | w32(func(f(bits))));
        }
      }
      sums[t] = sum;
    });
  }
  uint64_t total = 0;
  for (unsigned t = 0; t < threads; ++t) {
    pool[t].join();
    total += sums[t];
  }
  return total;
}

std::string hex(const Words& w) {
  std::ostringstream s;
  for (uint64_t v : w) s << ' ' << std::hex << v;
  return s.str();
}

int check_file(const char* path, std::map<std::string, size_t>& counts) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", path);
    return 1;
  }
  const std::map<std::string, uint32_t> constants = {
      {"inv448", ds::kInv448Bits},
      {"inv6", ds::kInv6Bits},
      {"fp8_floor", ds::kFp8FloorBits},
      {"fp4_pow2_floor", ds::kFp4Pow2FloorBits},
      {"fp4_e4m3_floor", ds::kFp4E4m3FloorBits},
  };
  int failures = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream tokens(line);
    std::string kind, name, token;
    tokens >> kind >> name;
    Words before, after;
    bool seen_equals = false;
    while (tokens >> token) {
      if (token == "=") {
        seen_equals = true;
      } else {
        (seen_equals ? after : before).push_back(std::stoull(token, nullptr, 16));
      }
    }
    Words got;
    if (kind == "c") {
      const auto c = constants.find(name);
      after = before;
      got = {c == constants.end() ? ~0ull : c->second};
    } else if (kind == "v") {
      got = eval(name, before);
    } else if (kind == "s") {
      got = {sweep(name, before.at(0), before.at(1))};
    } else if (kind == "x") {
      got = {exhaustive(kUnary.at(name))};
    } else {
      std::fprintf(stderr, "unknown record: %s\n", line.c_str());
      return 1;
    }
    ++counts[kind];
    if (got != after) {
      if (++failures <= 20) {
        std::fprintf(stderr, "MISMATCH %s %s%s\n  want%s\n  got %s\n", kind.c_str(), name.c_str(),
                     hex(before).c_str(), hex(after).c_str(), hex(got).c_str());
      }
    }
  }
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectors> [<exhaustive>]\n", argv[0]);
    return 2;
  }
  std::map<std::string, size_t> counts;
  int failures = 0;
  for (int a = 1; a < argc; ++a) failures += check_file(argv[a], counts);
  std::printf("ds_numeric_host: %zu constants, %zu cases, %zu sweeps, %zu exhaustive; %d mismatches\n",
              counts["c"], counts["v"], counts["s"], counts["x"], failures);
  return failures == 0 ? 0 : 1;
}
