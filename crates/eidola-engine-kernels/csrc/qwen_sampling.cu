// The Qwen3.8 executor's sampler and chain acceptance: the serving core's
// sampler (eidola-engine/src/sampling.rs) on the device, bit for bit given
// equal logits, as `sampling.cu` is, but spread over the GPU and selecting
// top-k and top-p from a short list of candidates instead of searching the
// whole vocabulary bit by bit.
//
// The arithmetic is the reference's, operation for operation: f64 throughout,
// every add, subtract, multiply and divide an explicit round-to-nearest
// intrinsic (never contracted), exponentials from det_exp, and every sum over
// a set of probabilities its pinned grouping (chunks of 1024 consecutive ids
// summed left to right from +0, then the chunk sums left to right from +0).
// A chunk's sum is always one thread's left-to-right loop; what is spread is
// which thread sums which chunk. Max, counts, comparisons and f32 -> f64
// conversion are exact, so their parallel reductions are order-free.
//
// A sample launch is four kernels over a work area of one `QwSampleWork` a
// row:
//
// 1. `eidola_qwen_sample_top` (a block a row): the lowest-id argmax of the
//    f32 logits, the non-finite check, a greedy row's token; clears the row's
//    histogram.
// 2. `eidola_qwen_sample_exp` (a block a chunk of a row): e_i = det_exp(l_i/t
//    - l_top/t) into the row's distribution, and the chunk's pinned sum.
// 3. `eidola_qwen_sample_norm` (a block a chunk of a row): the chunk sums
//    added in order into the softmax's denominator, p_i = e_i / sum in place
//    (a greedy row's one-hot instead), the chunk's pinned sum of p, and a
//    filtering row's histogram of p by its top twelve bits below the sign.
// 4. `eidola_qwen_sample_pick` (a block a row): an unfiltered row draws by
//    the chunk sums of p (one chunk scanned); a filtering row selects.
//
// Selection. Top-k, top-p and min-p keep a prefix of the order (p descending,
// id ascending). Non-negative doubles order like their bit patterns, so a
// prefix is every key (p's bits, then the id) at or above a threshold. The
// threshold is found from histograms of p's bits, twelve at a time, until the
// keys at or above it are at most `kCap` (or, at a tie on one value of p, by
// the lowest ids of that value): for top-k, the fewest bins holding k keys;
// for top-p alone, the fewest bins whose complement's mass is bounded (bin
// count times the bin's upper edge) below 1 - top_p of the whole; for min-p
// alone, every bin at or above the floor's. The candidates are gathered,
// ordered by key, and then the reference's own steps run over them: the
// masses of rank prefixes are the pinned sums of those ranks, computed over
// the candidates in id order (a chunk's kept ids in order from +0, chunks in
// order: exactly the reference's sum, whose other terms are zeros), top-p's
// shortest prefix by the reference's binary search, min-p's floor, the
// renormalisation, and the draw scanning the kept ids in vocabulary order.
// A top-p prefix whose candidates do not reach its mass, a top-k past
// `kCap`, a row filtering nothing but asking for top-k (a renormalisation of
// the whole vocabulary), and a min-p floor past `kCap` take the bitwise
// search of `sampling.cu` instead (`slow_filter`), which is exact and slow.
//
// Chain acceptance is three kernels: the accept tests (`decide`), the pinned
// chunk sums of the one distribution each sequence then draws from
// (`sums`, a block a chunk: the residual at the first rejection, or the
// bonus row), and the draw (`finish`), which scans one chunk.
//
// The sampleable vocabulary is at most `kMaxChunks` chunks (262,144 ids).

#include <cstdint>

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kThreads = 1024;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kChunk = 1024;  // SUM_CHUNK in sampling.rs
constexpr uint32_t kMaxChunks = 256;
// Threads of a chunk block: four ids each.
constexpr uint32_t kChunkThreads = 256;
constexpr uint32_t kPerThread = kChunk / kChunkThreads;
// Histogram bins: twelve bits of p.
constexpr uint32_t kBins = 4096;
// The most candidates a selection gathers.
constexpr uint32_t kCap = 2048;
// The digits of p's bits a selection resolves in turn (bit 63 is the sign,
// bit 62 is zero for p <= 1): bits 61..50, 49..38, 37..26, 25..14, 13..2, 1..0.
constexpr uint32_t kLevels = 6;
constexpr uint32_t kTopShift = 50;
__device__ __forceinline__ uint32_t level_shift(uint32_t level) {
  return level + 1 < kLevels ? kTopShift - 12 * level : 0;
}
__device__ __forceinline__ uint32_t level_width(uint32_t level) {
  return level + 1 < kLevels ? 12 : 2;
}

constexpr uint64_t kStreamAccept = 1;
constexpr uint64_t kStreamResidual = 2;
constexpr uint64_t kStreamSample = 0;
constexpr uint32_t kNoStream = 0xffffffffu;

constexpr uint32_t kStatusNonFinite = 1;
constexpr uint32_t kStatusBadToken = 2;

constexpr uint32_t kNone = 0xffffffffu;

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ double uniform(uint64_t seed, uint64_t position, uint64_t stream) {
  const uint64_t w = mix64(mix64(seed) ^ mix64(position * 0x9e3779b97f4a7c15ull + stream));
  return __dmul_rn(static_cast<double>(w >> 40), 1.0 / 16777216.0);
}

__device__ __forceinline__ double from_bits(uint64_t b) {
  return __longlong_as_double(static_cast<long long>(b));
}
__device__ __forceinline__ uint64_t to_bits(double d) {
  return static_cast<uint64_t>(__double_as_longlong(d));
}

__device__ __forceinline__ double pow2(int64_t e) {
  return from_bits(static_cast<uint64_t>(e + 1023) << 52);
}

// det_exp in sampling.rs, operation for operation.
__device__ double det_exp(double x) {
  if (isnan(x)) return x;
  if (x > 709.782712893384) return __longlong_as_double(0x7ff0000000000000ll);
  if (x < -745.2) return 0.0;
  const uint64_t c[14] = {0x3ff0000000000000ull, 0x3ff0000000000000ull, 0x3fe0000000000000ull,
                          0x3fc5555555555555ull, 0x3fa5555555555555ull, 0x3f81111111111111ull,
                          0x3f56c16c16c16c17ull, 0x3f2a01a01a01a01aull, 0x3efa01a01a01a01aull,
                          0x3ec71de3a556c734ull, 0x3e927e4fb7789f5cull, 0x3e5ae64567f544e4ull,
                          0x3e21eed8eff8d898ull, 0x3de6124613a86d09ull};
  const double n = round(__dmul_rn(x, from_bits(0x3ff71547652b82feull)));
  const double r = __dsub_rn(__dsub_rn(x, __dmul_rn(n, from_bits(0x3fe62e42fee00000ull))),
                             __dmul_rn(n, from_bits(0x3dea39ef35793c76ull)));
  double p = from_bits(c[13]);
#pragma unroll
  for (int k = 12; k >= 0; --k) p = __dadd_rn(__dmul_rn(p, r), from_bits(c[k]));
  const int64_t ni = static_cast<int64_t>(n);
  if (ni > 1023) return __dmul_rn(__dmul_rn(p, pow2(1023)), pow2(ni - 1023));
  if (ni < -1022) return __dmul_rn(__dmul_rn(p, pow2(-600)), pow2(ni + 600));
  return __dmul_rn(p, pow2(ni));
}

__device__ __forceinline__ uint32_t chunks_of(uint32_t n) { return (n + kChunk - 1) / kChunk; }

// The digit of p's bits a histogram bins by at `shift` (12 bits, or the
// level's width).
__device__ __forceinline__ uint32_t digit(uint64_t key, uint32_t shift, uint32_t width) {
  return static_cast<uint32_t>(key >> shift) & ((1u << width) - 1u);
}

// ---- block reductions and scans (blockDim.x == kThreads) ----

struct Red {
  double d[kWarps];
  uint64_t u[kWarps];
  double result;
  uint64_t word;
  uint32_t index;
  uint32_t count;
  uint32_t sel[4];
  double dsel[2];
};

template <typename T, typename Op>
__device__ T block_reduce(T v, Op op, T* scratch) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = scratch[threadIdx.x];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) scratch[0] = v;
  }
  __syncthreads();
  const T out = scratch[0];
  __syncthreads();
  return out;
}

__device__ uint64_t block_sum_u64(uint64_t v, Red& sh) {
  return block_reduce<uint64_t>(v, [](uint64_t a, uint64_t b) { return a + b; }, sh.u);
}
__device__ uint64_t block_max_u64(uint64_t v, Red& sh) {
  return block_reduce<uint64_t>(v, [](uint64_t a, uint64_t b) { return a > b ? a : b; }, sh.u);
}
__device__ uint64_t block_min_u64(uint64_t v, Red& sh) {
  return block_reduce<uint64_t>(v, [](uint64_t a, uint64_t b) { return a < b ? a : b; }, sh.u);
}

// Inclusive scan of `v` over threads in ascending order, and the total. The
// order of its additions is fixed (it serves heuristics and counts only).
template <typename T>
__device__ T block_scan(T v, T* total, T* scratch) {
  const uint32_t lane = threadIdx.x % 32, w = threadIdx.x / 32;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const T x = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= static_cast<uint32_t>(o)) v = v + x;
  }
  if (lane == 31) scratch[w] = v;
  __syncthreads();
  if (w == 0) {
    T s = scratch[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const T x = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= static_cast<uint32_t>(o)) s = s + x;
    }
    scratch[lane] = s;
  }
  __syncthreads();
  const T before = w > 0 ? scratch[w - 1] : T(0);
  *total = scratch[kWarps - 1];
  __syncthreads();
  return before + v;
}

// Lowest-id argmax of f32 values (sampling::argmax), with a non-finite check.
__device__ uint32_t argmax_f32(const float* l, uint32_t n, bool* non_finite, Red& sh) {
  float best = -INFINITY;
  uint32_t idx = kNone;
  bool bad = false;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    const float v = l[i];
    if (isnan(v) || v == INFINITY) bad = true;
    if (idx == kNone || v > best) {
      best = v;
      idx = i;
    }
  }
  // Larger value wins, ties to the lower id: (orderable value bits, inverted
  // id) as one u64. -0 and +0 are equal (the reference's `>` sees a tie).
  uint32_t vb = __float_as_uint(best == 0.0f ? 0.0f : best);
  vb = (vb & 0x80000000u) ? ~vb : (vb | 0x80000000u);
  const uint64_t key = idx == kNone ? 0 : (static_cast<uint64_t>(vb) << 32) | (kNone - idx);
  const uint64_t top = block_max_u64(key, sh);
  const uint64_t any_bad = block_max_u64(bad ? 1 : 0, sh);
  *non_finite = any_bad != 0;
  return kNone - static_cast<uint32_t>(top & 0xffffffffu);
}

// Lowest-id argmax of non-negative f64 values (chain_accept's argmax_f64).
__device__ uint32_t argmax_f64(const double* p, uint32_t n, Red& sh) {
  uint64_t best = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (to_bits(p[i]) > best) best = to_bits(p[i]);
  }
  const uint64_t top = block_max_u64(best, sh);
  uint64_t lo = ~0ull;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (to_bits(p[i]) == top && i < lo) lo = i;
  }
  return static_cast<uint32_t>(block_min_u64(lo, sh));
}

__device__ __forceinline__ uint32_t bounded(uint32_t token, uint32_t n, uint32_t* status) {
  if (token < n) return token;
  atomicOr(status, kStatusBadToken);
  return 0;
}

// The largest id with a positive value (sample_from's fallback), or 0.
template <typename Val>
__device__ uint32_t last_positive(Val val, uint32_t n, Red& sh) {
  uint64_t last = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (val(i) > 0.0) last = i;
  }
  return static_cast<uint32_t>(block_max_u64(last, sh));
}

// sample_from(values, u) given the pinned chunk sums of `values`: the chunks
// added in order find the first whose end passes the target, and that one
// chunk, staged, is scanned left to right by one thread. If (by rounding) no
// id of it passes, the scan goes on to the next chunk whose end passes, as
// the reference's loop does; failing everything, the last positive id.
template <typename Val>
__device__ uint32_t draw_by_sums(Val val, const double* sums, uint32_t n, double u, Red& sh,
                                 double* stage) {
  const uint32_t chunks = chunks_of(n);
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) total = __dadd_rn(total, sums[c]);
    sh.dsel[0] = __dmul_rn(u, total);  // target
    sh.dsel[1] = 0.0;                   // base
    sh.sel[0] = 0;                      // next chunk
    sh.sel[1] = kNone;                  // token
  }
  __syncthreads();
  while (true) {
    if (threadIdx.x == 0) {
      const double target = sh.dsel[0];
      double base = sh.dsel[1];
      uint32_t c = sh.sel[0];
      while (c < chunks && !(__dadd_rn(base, sums[c]) > target)) {
        base = __dadd_rn(base, sums[c]);
        ++c;
      }
      sh.dsel[1] = base;
      sh.sel[0] = c;
    }
    __syncthreads();
    const uint32_t c = sh.sel[0];
    if (c >= chunks) break;
    const uint32_t lo = c * kChunk;
    const uint32_t len = min(kChunk, n - lo);
    for (uint32_t j = threadIdx.x; j < len; j += kThreads) stage[j] = val(lo + j);
    __syncthreads();
    if (threadIdx.x == 0) {
      const double target = sh.dsel[0], base = sh.dsel[1];
      double cum = 0.0;
      for (uint32_t j = 0; j < len; ++j) {
        const double v = stage[j];
        cum = __dadd_rn(cum, v);
        if (v > 0.0 && __dadd_rn(base, cum) > target) {
          sh.sel[1] = lo + j;
          break;
        }
      }
      sh.dsel[1] = __dadd_rn(base, sums[c]);
      sh.sel[0] = c + 1;
    }
    __syncthreads();
    if (sh.sel[1] != kNone) break;
  }
  const uint32_t token = sh.sel[1];
  __syncthreads();
  return token != kNone ? token : last_positive(val, n, sh);
}

// The pinned chunk sums of `val` over [0, n) into `sums`: thread c sums chunk
// c left to right (the slow form, for rare paths).
template <typename Val>
__device__ void chunk_sums_serial(Val val, uint32_t n, double* sums) {
  const uint32_t chunks = chunks_of(n);
  if (threadIdx.x < chunks) {
    const uint32_t lo = threadIdx.x * kChunk;
    const uint32_t hi = min(lo + kChunk, n);
    double s = 0.0;
    for (uint32_t i = lo; i < hi; ++i) s = __dadd_rn(s, val(i));
    sums[threadIdx.x] = s;
  }
  __syncthreads();
}

// ---- the bitwise search of sampling.cu, for the rows selection hands it ----

struct RankSet {
  uint64_t pt;
  uint32_t it;
  double floor;
  __device__ bool contains(double p, uint32_t i) const {
    const uint64_t pb = to_bits(p);
    return (pb > pt || (pb == pt && i <= it)) && p >= floor;
  }
};

template <typename Keep>
__device__ double pinned_sum(const double* x, uint32_t n, Keep keep, Red& sh, double* chunk) {
  const uint32_t chunks = chunks_of(n);
  if (threadIdx.x < chunks) {
    const uint32_t lo = threadIdx.x * kChunk;
    const uint32_t hi = min(lo + kChunk, n);
    double s = 0.0;
    for (uint32_t i = lo; i < hi; ++i) {
      if (keep(i)) s = __dadd_rn(s, x[i]);
    }
    chunk[threadIdx.x] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) total = __dadd_rn(total, chunk[c]);
    sh.result = total;
  }
  __syncthreads();
  const double out = sh.result;
  __syncthreads();
  return out;
}

__device__ uint64_t count_where_bits_ge(const double* p, uint32_t n, const RankSet& set,
                                        uint64_t v, Red& sh) {
  uint64_t c = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (set.contains(p[i], i) && to_bits(p[i]) >= v) ++c;
  }
  return block_sum_u64(c, sh);
}

template <typename Pred>
__device__ uint64_t search_bits(Pred pred) {
  uint64_t v = 0;
  for (int b = 63; b >= 0; --b) {
    const uint64_t t = v | (1ull << b);
    if (pred(t)) v = t;
  }
  return v;
}

template <typename Pred>
__device__ uint32_t search_id(uint32_t n, Pred pred) {
  uint32_t lo = 0, hi = n - 1;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (pred(mid)) hi = mid; else lo = mid + 1;
  }
  return lo;
}

__device__ uint64_t min_member_bits_ge(const double* p, uint32_t n, const RankSet& set, uint64_t v,
                                       Red& sh) {
  uint64_t m = ~0ull;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    const uint64_t b = to_bits(p[i]);
    if (set.contains(p[i], i) && b >= v && b < m) m = b;
  }
  return block_min_u64(m, sh);
}

// Steps 3-6 of the definition in place over `p` (the softmax), by the
// bitwise search: sampling.cu's filtering, unchanged.
__device__ void slow_filter(double* p, uint32_t n, uint32_t top, uint32_t top_k, float top_p,
                            float min_p, Red& sh, double* chunk) {
  RankSet set{0, n - 1, 0.0};
  if (top_k > 0 && top_k < n) {
    const uint64_t keep = top_k;
    const uint64_t pt =
        search_bits([&](uint64_t v) { return count_where_bits_ge(p, n, set, v, sh) >= keep; });
    const uint64_t above = count_where_bits_ge(p, n, set, pt + 1, sh);
    const uint64_t need = keep - above;
    const uint32_t it = search_id(n, [&](uint32_t id) {
      uint64_t c = 0;
      for (uint32_t i = threadIdx.x; i <= id && i < n; i += kThreads) {
        if (to_bits(p[i]) == pt) ++c;
      }
      return block_sum_u64(c, sh) >= need;
    });
    set = RankSet{pt, it, 0.0};
  }
  if (top_p < 1.0f) {
    const RankSet kept = set;
    const double mass =
        pinned_sum(p, n, [&](uint32_t i) { return kept.contains(p[i], i); }, sh, chunk);
    const double target = __dmul_rn(static_cast<double>(top_p), mass);
    const auto mass_ge = [&](uint64_t v) {
      return pinned_sum(
          p, n, [&](uint32_t i) { return kept.contains(p[i], i) && to_bits(p[i]) >= v; }, sh,
          chunk);
    };
    const uint64_t v = search_bits([&](uint64_t b) { return mass_ge(b) >= target; });
    const uint64_t pt = min_member_bits_ge(p, n, kept, v, sh);
    const uint32_t it = search_id(n, [&](uint32_t id) {
      return pinned_sum(p, n,
                        [&](uint32_t i) {
                          const uint64_t b = to_bits(p[i]);
                          return kept.contains(p[i], i) && (b > pt || (b == pt && i <= id));
                        },
                        sh, chunk) >= target;
    });
    set = RankSet{pt, it, 0.0};
  }
  if (min_p > 0.0f) set.floor = __dmul_rn(static_cast<double>(min_p), p[top]);
  const RankSet kept = set;
  const double mass =
      pinned_sum(p, n, [&](uint32_t i) { return kept.contains(p[i], i); }, sh, chunk);
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    p[i] = kept.contains(p[i], i) ? __ddiv_rn(p[i], mass) : 0.0;
  }
  __syncthreads();
}

// ---- selection from candidates ----

struct Picked {
  // The level-0 histogram, then (once the candidates are gathered) the
  // candidates in id order: their ids and ranks.
  union {
    uint32_t hist[kBins];
    struct {
      uint32_t id[kCap];
      uint32_t rank[kCap];
    } by_id;
  } u;
  // The candidates: p's bits and ids, gathered, then sorted by rank.
  uint64_t key[kCap];
  uint32_t id[kCap];
  // Per chunk: a mass's chunk sums, and the index (in id order) of its first
  // candidate.
  double run_sum[kMaxChunks];
  uint32_t run_first[kMaxChunks];
};

union PickShared {
  Picked f;
  double chunk[kChunk];
};

// In `h[0 .. bins)` (bins a power of two, at most kBins), the largest bin s
// with sum(h[s ..]) >= need: sel[0] = s, sel[1] = sum(h[s ..]), sel[2] =
// sum(h[s + 1 ..]). Requires sum(h) >= need >= 1.
__device__ void select_bin(const uint32_t* h, uint32_t bins, uint32_t need, Red& sh) {
  constexpr uint32_t kPer = kBins / kThreads;
  const uint32_t b0 = threadIdx.x * kPer;
  uint32_t own = 0;
  for (uint32_t j = 0; j < kPer; ++j) own += b0 + j < bins ? h[b0 + j] : 0;
  uint32_t total;
  const uint32_t incl = block_scan<uint32_t>(own, &total, reinterpret_cast<uint32_t*>(sh.u));
  // Keys in bins after this thread's.
  uint32_t run = total - incl;
  for (uint32_t j = kPer; j-- > 0;) {
    const uint32_t b = b0 + j;
    if (b >= bins) continue;
    const uint32_t ge = run + h[b];
    if (ge >= need && run < need) {
      sh.sel[0] = b;
      sh.sel[1] = ge;
      sh.sel[2] = run;
    }
    run = ge;
  }
  __syncthreads();
}

// Bitonic sort of `count` entries (padded to a power of two with entries
// that sort last), ascending by `before`.
template <typename Before, typename Swap>
__device__ void bitonic(uint32_t count, Before before, Swap swap) {
  uint32_t size = 2;
  while (size < count) size <<= 1;
  for (uint32_t k = 2; k <= size; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      for (uint32_t i = threadIdx.x; i < size; i += kThreads) {
        const uint32_t o = i ^ j;
        if (o > i) {
          const bool up = (i & k) == 0;
          if (up ? before(o, i) : before(i, o)) swap(i, o);
        }
      }
      __syncthreads();
    }
  }
}

// The pinned sum of `val(rank)` over the candidate ranks below `m`, in id
// order: each chunk's kept ids left to right from +0, the chunk sums in
// order (a chunk with none adds +0, which changes nothing). Leaves the chunk
// sums in `run_sum`.
template <typename Val>
__device__ double ranks_mass(Picked& f, uint32_t count, uint32_t chunks, uint32_t m, Val val,
                             Red& sh) {
  for (uint32_t c = threadIdx.x; c < chunks; c += kThreads) f.run_sum[c] = 0.0;
  __syncthreads();
  for (uint32_t t = threadIdx.x; t < count; t += kThreads) {
    const uint32_t c = f.u.by_id.id[t] / kChunk;
    if (t > 0 && f.u.by_id.id[t - 1] / kChunk == c) continue;
    double s = 0.0;
    for (uint32_t j = t; j < count && f.u.by_id.id[j] / kChunk == c; ++j) {
      const uint32_t r = f.u.by_id.rank[j];
      if (r < m) s = __dadd_rn(s, val(r));
    }
    f.run_sum[c] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) total = __dadd_rn(total, f.run_sum[c]);
    sh.result = total;
  }
  __syncthreads();
  const double out = sh.result;
  __syncthreads();
  return out;
}

}  // namespace

// Per-row sampling parameters, the device copy of SamplingParams (as
// sampling.cu's).
struct EidolaSampleRow {
  float temperature;  // 0: greedy
  uint32_t top_k;     // 0: off
  float top_p;        // 1: off
  float min_p;        // 0: off
  uint64_t seed;
  uint32_t position;  // the position of the token being chosen
  uint32_t logit_row; // row of `logits` this row samples from
};
static_assert(sizeof(EidolaSampleRow) == 32, "host mirror layout");

// A sample row's work area.
struct QwSampleWork {
  double esum[kMaxChunks];  // chunk sums of the exponentials
  double psum[kMaxChunks];  // chunk sums of p
  uint32_t hist[kBins];     // p's level-0 histogram (filtering rows)
  uint32_t top;             // the logits' argmax
  uint32_t reserved[31];
};
static_assert(sizeof(QwSampleWork) == 20608, "host mirror layout");

__device__ __forceinline__ bool filtering(const EidolaSampleRow& row) {
  return row.top_k > 0 || row.top_p < 1.0f || row.min_p > 0.0f;
}

// 1. The argmax, the non-finite check, a greedy row's token; the histogram
// cleared.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_qwen_sample_top(const float* __restrict__ logits, uint64_t logits_stride, uint32_t n,
                           const EidolaSampleRow* __restrict__ rows, uint32_t draw_stream,
                           uint32_t* __restrict__ tokens, QwSampleWork* __restrict__ work,
                           uint32_t* __restrict__ status) {
  __shared__ Red sh;
  const EidolaSampleRow row = rows[blockIdx.x];
  const float* l = logits + static_cast<uint64_t>(row.logit_row) * logits_stride;
  QwSampleWork& w = work[blockIdx.x];
  for (uint32_t b = threadIdx.x; b < kBins; b += kThreads) w.hist[b] = 0;
  bool non_finite = false;
  const uint32_t top = argmax_f32(l, n, &non_finite, sh);
  if (threadIdx.x != 0) return;
  w.top = top;
  if (row.temperature <= 0.0f) {
    if (draw_stream != kNoStream) tokens[blockIdx.x] = bounded(top, n, status);
    if (non_finite) atomicOr(status, kStatusNonFinite);
    return;
  }
  const double zmax = __ddiv_rn(static_cast<double>(l[top]), static_cast<double>(row.temperature));
  if (non_finite || isinf(zmax)) atomicOr(status, kStatusNonFinite);
}
EIDOLA_KERNEL_META(eidola_qwen_sample_top, kThreads, 1, 1, 0, 1, 1, 1, 0);

// 2. Chunk `blockIdx.x` of row `blockIdx.y`: the exponentials (processed_probs
// step 2, before the division) into `probs`, and their pinned chunk sum.
extern "C" __global__ void __launch_bounds__(kChunkThreads)
    eidola_qwen_sample_exp(const float* __restrict__ logits, uint64_t logits_stride, uint32_t n,
                           const EidolaSampleRow* __restrict__ rows, double* __restrict__ probs,
                           QwSampleWork* __restrict__ work) {
  __shared__ double e[kChunk];
  const uint32_t r = blockIdx.y, c = blockIdx.x;
  const EidolaSampleRow row = rows[r];
  if (row.temperature <= 0.0f) return;
  const float* l = logits + static_cast<uint64_t>(row.logit_row) * logits_stride;
  double* p = probs + static_cast<uint64_t>(r) * n;
  const uint32_t lo = c * kChunk;
  const uint32_t len = min(kChunk, n - lo);
  const double t = static_cast<double>(row.temperature);
  const double zmax = __ddiv_rn(static_cast<double>(l[work[r].top]), t);
#pragma unroll
  for (uint32_t k = 0; k < kPerThread; ++k) {
    const uint32_t j = threadIdx.x + k * kChunkThreads;
    if (j < len) {
      const double v = det_exp(__dsub_rn(__ddiv_rn(static_cast<double>(l[lo + j]), t), zmax));
      e[j] = v;
      p[lo + j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (uint32_t j = 0; j < len; ++j) s = __dadd_rn(s, e[j]);
    work[r].esum[c] = s;
  }
}
EIDOLA_KERNEL_META(eidola_qwen_sample_exp, kChunkThreads, 1, 1, 0, 1, 1, 1, 0);

// 3. Chunk `blockIdx.x` of row `blockIdx.y`: p = e / sum (the chunk sums added
// in order), a greedy row's one-hot instead; p's pinned chunk sum; and a
// filtering row's histogram.
extern "C" __global__ void __launch_bounds__(kChunkThreads)
    eidola_qwen_sample_norm(uint32_t n, const EidolaSampleRow* __restrict__ rows,
                            double* __restrict__ probs, QwSampleWork* __restrict__ work) {
  __shared__ double pv[kChunk];
  __shared__ uint32_t hist[kBins];
  __shared__ double total_sh;
  const uint32_t r = blockIdx.y, c = blockIdx.x;
  const EidolaSampleRow row = rows[r];
  double* p = probs + static_cast<uint64_t>(r) * n;
  QwSampleWork& w = work[r];
  const uint32_t lo = c * kChunk;
  const uint32_t len = min(kChunk, n - lo);
  if (row.temperature <= 0.0f) {
    const uint32_t top = w.top;
    for (uint32_t j = threadIdx.x; j < len; j += kChunkThreads) p[lo + j] = lo + j == top ? 1.0 : 0.0;
    return;
  }
  const bool filter = filtering(row);
  if (threadIdx.x == 0) {
    const uint32_t chunks = chunks_of(n);
    double total = 0.0;
    for (uint32_t k = 0; k < chunks; ++k) total = __dadd_rn(total, w.esum[k]);
    total_sh = total;
  }
  if (filter) {
    for (uint32_t b = threadIdx.x; b < kBins; b += kChunkThreads) hist[b] = 0;
  }
  __syncthreads();
  const double total = total_sh;
#pragma unroll
  for (uint32_t k = 0; k < kPerThread; ++k) {
    const uint32_t j = threadIdx.x + k * kChunkThreads;
    if (j < len) {
      const double v = __ddiv_rn(p[lo + j], total);
      pv[j] = v;
      p[lo + j] = v;
      if (filter) atomicAdd(&hist[digit(to_bits(v), kTopShift, 12)], 1u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (uint32_t j = 0; j < len; ++j) s = __dadd_rn(s, pv[j]);
    w.psum[c] = s;
  }
  if (filter) {
    for (uint32_t b = threadIdx.x; b < kBins; b += kChunkThreads) {
      if (hist[b] != 0) atomicAdd(&w.hist[b], hist[b]);
    }
  }
}
EIDOLA_KERNEL_META(eidola_qwen_sample_norm, kChunkThreads, 1, 1, 0, 1, 1, 1, 0);

// 4. Row `blockIdx.x`: an unfiltered row's draw; a filtering row's selection,
// distribution and draw.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_qwen_sample_pick(uint32_t n, const EidolaSampleRow* __restrict__ rows,
                            uint32_t draw_stream, double* __restrict__ probs,
                            uint32_t* __restrict__ tokens, QwSampleWork* __restrict__ work,
                            uint32_t* __restrict__ status) {
  __shared__ Red sh;
  __shared__ PickShared sm;
  const uint32_t r = blockIdx.x;
  const EidolaSampleRow row = rows[r];
  if (row.temperature <= 0.0f) return;
  double* p = probs + static_cast<uint64_t>(r) * n;
  const QwSampleWork& w = work[r];
  const uint32_t chunks = chunks_of(n);
  const bool draw = draw_stream != kNoStream;
  const double u = uniform(row.seed, row.position, draw_stream);
  const auto pval = [&](uint32_t i) { return p[i]; };

  if (!filtering(row)) {
    if (draw) {
      const uint32_t token = draw_by_sums(pval, w.psum, n, u, sh, sm.chunk);
      if (threadIdx.x == 0) tokens[r] = bounded(token, n, status);
    }
    return;
  }

  Picked& f = sm.f;
  const bool has_k = row.top_k > 0 && row.top_k < n;
  const uint32_t keep_k = has_k ? row.top_k : n;
  // The pinned sum of every p: the chunk sums in order.
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) total = __dadd_rn(total, w.psum[c]);
    sh.result = total;
  }
  for (uint32_t b = threadIdx.x; b < kBins; b += kThreads) f.u.hist[b] = w.hist[b];
  __syncthreads();
  const double mass_all = sh.result;
  __syncthreads();

  // The threshold: candidates are keys >= `floor_key`, plus, at a tie, the
  // `tie_take` lowest ids whose key is `tie_key`.
  bool slow = false;
  uint64_t floor_key = 0;
  bool tie = false;
  uint64_t tie_key = 0;
  uint32_t tie_take = 0;
  if (has_k) {
    if (keep_k > kCap) {
      slow = true;
    } else {
      uint64_t prefix = 0;
      uint32_t above = 0;
      for (uint32_t level = 0; level < kLevels; ++level) {
        const uint32_t shift = level_shift(level), width = level_width(level), bins = 1u << width;
        if (level > 0) {
          const uint32_t prev = level_shift(level - 1);
          for (uint32_t b = threadIdx.x; b < bins; b += kThreads) f.u.hist[b] = 0;
          __syncthreads();
          for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
            const uint64_t key = to_bits(p[i]);
            if ((key >> prev) == prefix) atomicAdd(&f.u.hist[digit(key, shift, width)], 1u);
          }
          __syncthreads();
        }
        select_bin(f.u.hist, bins, keep_k - above, sh);
        const uint32_t s = sh.sel[0], ge = sh.sel[1], gt = sh.sel[2];
        __syncthreads();
        prefix = (prefix << width) | s;
        if (above + ge <= kCap) {
          floor_key = prefix << shift;
          break;
        }
        above += gt;
        if (level + 1 == kLevels) {
          // Every bit fixed: the boundary is one value of p, cut by id.
          tie = true;
          tie_key = prefix;
          tie_take = keep_k - above;
          floor_key = prefix + 1;
        }
      }
    }
  } else if (row.top_p < 1.0f) {
    // The fewest level-0 bins whose complement's mass is bounded below
    // (1 - top_p) of the whole, with a margin for the sums' rounding.
    const double limit = __dmul_rn(
        mass_all, __dsub_rn(__dsub_rn(1.0, static_cast<double>(row.top_p)), 9.313225746154785e-10));
    if (!(limit > 0.0)) {
      slow = true;
    } else {
      constexpr uint32_t kPer = kBins / kThreads;
      const uint32_t b0 = threadIdx.x * kPer;
      double own[kPer];
      double sum = 0.0;
      for (uint32_t j = 0; j < kPer; ++j) {
        // An upper bound of every p in bin b: the next bin's lower edge.
        own[j] = static_cast<double>(f.u.hist[b0 + j]) *
                 from_bits(static_cast<uint64_t>(b0 + j + 1) << kTopShift);
        sum += own[j];
      }
      double total;
      const double incl = block_scan<double>(sum, &total, sh.d);
      // The bound of the mass below bin b is the sum over bins below b.
      double below = incl - sum;
      uint64_t best = 0;
      for (uint32_t j = 0; j < kPer; ++j) {
        if (below * 1.0000001 <= limit) best = b0 + j;
        below += own[j];
      }
      const uint64_t b = block_max_u64(best, sh);
      floor_key = b << kTopShift;
    }
  } else if (row.min_p > 0.0f) {
    const double floor = __dmul_rn(static_cast<double>(row.min_p), p[w.top]);
    floor_key = (to_bits(floor) >> kTopShift) << kTopShift;
  } else {
    // top_k at or past the vocabulary alone: the whole vocabulary renormalised.
    slow = true;
  }
  if (!slow && !has_k) {
    // The candidates' count from the histogram.
    uint32_t c = 0;
    const uint32_t b0 = static_cast<uint32_t>(floor_key >> kTopShift);
    for (uint32_t b = threadIdx.x; b < kBins; b += kThreads) c += b >= b0 ? f.u.hist[b] : 0;
    slow = block_sum_u64(c, sh) > kCap;
  }

  uint32_t count = 0;
  if (!slow) {
    // Gather.
    if (threadIdx.x == 0) sh.count = 0;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
      const uint64_t key = to_bits(p[i]);
      if (key >= floor_key) {
        const uint32_t at = atomicAdd(&sh.count, 1u);
        // Never past the capacity (only a non-finite row's keys could be).
        if (at < kCap) {
          f.key[at] = key;
          f.id[at] = i;
        }
      }
    }
    __syncthreads();
    if (tie) {
      // The lowest `tie_take` ids whose key is `tie_key`, in id order.
      uint32_t taken = 0;
      for (uint32_t base = 0; base < n && taken < tie_take; base += kThreads) {
        const uint32_t i = base + threadIdx.x;
        const bool hit = i < n && to_bits(p[i]) == tie_key;
        const uint32_t ballot = __ballot_sync(0xffffffffu, hit);
        const uint32_t lane = threadIdx.x % 32;
        uint32_t warp_total;
        const uint32_t warp_incl = block_scan<uint32_t>(
            threadIdx.x % 32 == 0 ? static_cast<uint32_t>(__popc(ballot)) : 0u, &warp_total,
            reinterpret_cast<uint32_t*>(sh.u));
        // Hits before this warp: the inclusive scan at its lane 0, less its own.
        const uint32_t warp_before = __shfl_sync(0xffffffffu, warp_incl, 0) -
                                     static_cast<uint32_t>(__popc(ballot));
        const uint32_t at = taken + warp_before + __popc(ballot & ((1u << lane) - 1u));
        if (hit && at < tie_take && sh.count + at < kCap) {
          f.key[sh.count + at] = tie_key;
          f.id[sh.count + at] = i;
        }
        taken += warp_total;
      }
      __syncthreads();
      if (threadIdx.x == 0) sh.count += tie_take;
      __syncthreads();
    }
    count = min(sh.count, kCap);
    __syncthreads();

    // Rank order: key descending, id ascending.
    const uint32_t size_hint = count;
    for (uint32_t i = count + threadIdx.x; i < kCap; i += kThreads) {
      f.key[i] = 0;
      f.id[i] = kNone;
    }
    __syncthreads();
    bitonic(
        size_hint,
        [&](uint32_t a, uint32_t b) {
          return f.key[a] > f.key[b] || (f.key[a] == f.key[b] && f.id[a] < f.id[b]);
        },
        [&](uint32_t a, uint32_t b) {
          const uint64_t k = f.key[a];
          f.key[a] = f.key[b];
          f.key[b] = k;
          const uint32_t i = f.id[a];
          f.id[a] = f.id[b];
          f.id[b] = i;
        });
    // Id order, with ranks.
    for (uint32_t i = threadIdx.x; i < kCap; i += kThreads) {
      f.u.by_id.id[i] = i < count ? f.id[i] : kNone;
      f.u.by_id.rank[i] = i;
    }
    __syncthreads();
    bitonic(
        size_hint, [&](uint32_t a, uint32_t b) { return f.u.by_id.id[a] < f.u.by_id.id[b]; },
        [&](uint32_t a, uint32_t b) {
          const uint32_t i = f.u.by_id.id[a];
          f.u.by_id.id[a] = f.u.by_id.id[b];
          f.u.by_id.id[b] = i;
          const uint32_t k = f.u.by_id.rank[a];
          f.u.by_id.rank[a] = f.u.by_id.rank[b];
          f.u.by_id.rank[b] = k;
        });
    for (uint32_t c = threadIdx.x; c < kMaxChunks; c += kThreads) f.run_first[c] = kNone;
    __syncthreads();
    for (uint32_t t = threadIdx.x; t < count; t += kThreads) {
      const uint32_t c = f.u.by_id.id[t] / kChunk;
      if (t == 0 || f.u.by_id.id[t - 1] / kChunk != c) f.run_first[c] = t;
    }
    __syncthreads();
  }

  const auto pr = [&](uint32_t rank) { return from_bits(f.key[rank]); };
  uint32_t keep = 0;
  if (!slow) {
    keep = has_k ? keep_k : count;
    if (row.top_p < 1.0f) {
      const double mass = has_k ? ranks_mass(f, count, chunks, keep, pr, sh) : mass_all;
      const double target = __dmul_rn(static_cast<double>(row.top_p), mass);
      // The candidates must reach the target: top-p's prefix ends inside them.
      if (!has_k && !(ranks_mass(f, count, chunks, count, pr, sh) >= target)) {
        slow = true;
      } else {
        uint32_t lo = 1, hi = keep;
        while (lo < hi) {
          const uint32_t mid = (lo + hi) / 2;
          if (ranks_mass(f, count, chunks, mid, pr, sh) >= target) hi = mid; else lo = mid + 1;
        }
        keep = lo > 1 ? lo : 1;
      }
    }
  }
  if (!slow && row.min_p > 0.0f) {
    const double floor = __dmul_rn(static_cast<double>(row.min_p), pr(0));
    // The ranks are in descending p: the prefix at or above the floor.
    uint64_t cut = 0;
    for (uint32_t t = threadIdx.x; t < keep; t += kThreads) cut += pr(t) >= floor ? 1 : 0;
    const uint32_t c = static_cast<uint32_t>(block_sum_u64(cut, sh));
    keep = c > 1 ? c : 1;
  }

  if (slow) {
    slow_filter(p, n, w.top, row.top_k, row.top_p, row.min_p, sh, sm.chunk);
    if (draw) {
      // The sums lie past the staging area draw_by_sums uses.
      chunk_sums_serial(pval, n, f.run_sum);
      const uint32_t token = draw_by_sums(pval, f.run_sum, n, u, sh, sm.chunk);
      if (threadIdx.x == 0) tokens[r] = bounded(token, n, status);
    }
    return;
  }

  // The kept ranks' renormalisation, and the distribution written.
  const double mass = ranks_mass(f, count, chunks, keep, pr, sh);
  const auto q = [&](uint32_t rank) { return __ddiv_rn(pr(rank), mass); };
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) p[i] = 0.0;
  __syncthreads();
  for (uint32_t t = threadIdx.x; t < keep; t += kThreads) p[f.id[t]] = q(t);
  if (!draw) return;

  // The draw (sample_from over the distribution): its total is the pinned sum
  // of the kept q, and its scan visits the kept ids in vocabulary order (the
  // other terms are zeros, which change no partial sum and pass no test).
  const double total = ranks_mass(f, count, chunks, keep, q, sh);
  if (threadIdx.x == 0) {
    const double target = __dmul_rn(u, total);
    double base = 0.0;
    uint32_t token = kNone;
    for (uint32_t c = 0; c < chunks && token == kNone; ++c) {
      const double cs = f.run_sum[c];
      if (__dadd_rn(base, cs) > target && f.run_first[c] != kNone) {
        double cum = 0.0;
        for (uint32_t j = f.run_first[c]; j < count && f.u.by_id.id[j] / kChunk == c; ++j) {
          const uint32_t rk = f.u.by_id.rank[j];
          if (rk >= keep) continue;
          const double v = q(rk);
          cum = __dadd_rn(cum, v);
          if (v > 0.0 && __dadd_rn(base, cum) > target) {
            token = f.u.by_id.id[j];
            break;
          }
        }
      }
      base = __dadd_rn(base, cs);
    }
    if (token == kNone) {
      // The largest kept id with a positive probability, or 0.
      token = 0;
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t rk = f.u.by_id.rank[j];
        if (rk < keep && q(rk) > 0.0) token = f.u.by_id.id[j];
      }
    }
    tokens[r] = bounded(token, n, status);
  }
}
EIDOLA_KERNEL_META(eidola_qwen_sample_pick, kThreads, 1, 1, 0, 1, 1, 1, 0);

// ---- chain acceptance ----

// A sequence's acceptance work: the pinned chunk sums of the distribution it
// draws from, and the accept tests' outcome.
struct QwAcceptWork {
  double sums[kMaxChunks];
  uint32_t accepted;  // drafts accepted
  uint32_t kind;      // kAcceptResidual, kAcceptBonus, or kAcceptDone
  uint32_t reserved[2];
};
static_assert(sizeof(QwAcceptWork) == 2064, "host mirror layout");

namespace {
constexpr uint32_t kAcceptResidual = 0;  // rejected at `accepted`: draw from the residual
constexpr uint32_t kAcceptBonus = 1;     // every draft accepted: the bonus from p_k
constexpr uint32_t kAcceptDone = 2;      // a bad draft id: nothing more
}  // namespace

// Chain speculative acceptance (sampling::chain_accept), one sequence a
// thread: the accept tests, each accepted draft into `out`, and the outcome
// into the sequence's work. The arguments are eidola_chain_accept's, its
// scratch the work array (`QwAcceptWork` a sequence). A draft id at or past
// `n` raises kStatusBadToken and ends the row with no tokens, before
// anything is read through it.
extern "C" __global__ void __launch_bounds__(128)
    eidola_qwen_accept_decide(const double* __restrict__ target, const double* __restrict__ draft,
                              uint32_t n, const EidolaSampleRow* __restrict__ rows,
                              const uint32_t* __restrict__ target_row,
                              const uint32_t* __restrict__ draft_row, uint32_t draft_step,
                              const uint32_t* __restrict__ num_drafts,
                              const uint32_t* __restrict__ drafts, uint32_t stride,
                              QwAcceptWork* __restrict__ work, uint32_t* __restrict__ out,
                              uint32_t* __restrict__ counts, uint32_t* __restrict__ status,
                              uint32_t num_rows) {
  const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_rows) return;
  const EidolaSampleRow row = rows[r];
  const bool greedy = row.temperature <= 0.0f;
  const uint32_t k = num_drafts[r];
  const uint64_t first = row.position;
  QwAcceptWork& w = work[r];
  for (uint32_t i = 0; i < k; ++i) {
    const uint64_t dr = static_cast<uint64_t>(draft_row[r]) + static_cast<uint64_t>(i) * draft_step;
    const uint32_t d = drafts[dr];
    if (d >= n) {
      atomicOr(status, kStatusBadToken);
      counts[r] = 0;
      w.kind = kAcceptDone;
      return;
    }
    const double* pi = target + static_cast<uint64_t>(target_row[r] + i) * n;
    const double* qi = draft + dr * n;
    const bool accepted =
        greedy ? pi[d] > 0.0 : __dmul_rn(uniform(row.seed, first + i, kStreamAccept), qi[d]) < pi[d];
    if (!accepted) {
      w.accepted = i;
      w.kind = kAcceptResidual;
      return;
    }
    out[r * stride + i] = d;
  }
  w.accepted = k;
  w.kind = kAcceptBonus;
}
EIDOLA_KERNEL_META(eidola_qwen_accept_decide, 128, 1, 1, 0, 1, 1, 1, 0);

// Chunk `blockIdx.x` of sequence `blockIdx.y` of a sampled row: the pinned
// chunk sum of the distribution it draws from, the residual max(0, p_i - q_i)
// at its rejection or the bonus row p_k.
extern "C" __global__ void __launch_bounds__(kChunkThreads)
    eidola_qwen_accept_sums(const double* __restrict__ target, const double* __restrict__ draft,
                            uint32_t n, const EidolaSampleRow* __restrict__ rows,
                            const uint32_t* __restrict__ target_row,
                            const uint32_t* __restrict__ draft_row, uint32_t draft_step,
                            QwAcceptWork* __restrict__ work) {
  __shared__ double x[kChunk];
  const uint32_t r = blockIdx.y, c = blockIdx.x;
  const EidolaSampleRow row = rows[r];
  QwAcceptWork& w = work[r];
  if (row.temperature <= 0.0f || w.kind == kAcceptDone) return;
  const uint32_t i = w.accepted;
  const bool residual = w.kind == kAcceptResidual;
  const double* pi = target + static_cast<uint64_t>(target_row[r] + i) * n;
  const double* qi =
      draft + (static_cast<uint64_t>(draft_row[r]) + static_cast<uint64_t>(i) * draft_step) * n;
  const uint32_t lo = c * kChunk;
  const uint32_t len = min(kChunk, n - lo);
#pragma unroll
  for (uint32_t k = 0; k < kPerThread; ++k) {
    const uint32_t j = threadIdx.x + k * kChunkThreads;
    if (j < len) {
      double v = pi[lo + j];
      if (residual) {
        const double d = __dsub_rn(v, qi[lo + j]);
        v = d > 0.0 ? d : 0.0;
      }
      x[j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (uint32_t j = 0; j < len; ++j) s = __dadd_rn(s, x[j]);
    w.sums[c] = s;
  }
}
EIDOLA_KERNEL_META(eidola_qwen_accept_sums, kChunkThreads, 1, 1, 0, 1, 1, 1, 0);

// Sequence `blockIdx.x`: the replacement or bonus token after its accepted
// drafts, and the count.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_qwen_accept_finish(const double* __restrict__ target, const double* __restrict__ draft,
                              uint32_t n, const EidolaSampleRow* __restrict__ rows,
                              const uint32_t* __restrict__ target_row,
                              const uint32_t* __restrict__ draft_row, uint32_t draft_step,
                              uint32_t stride, QwAcceptWork* __restrict__ work,
                              uint32_t* __restrict__ out, uint32_t* __restrict__ counts) {
  __shared__ Red sh;
  __shared__ double stage[kChunk];
  __shared__ double psums[kMaxChunks];
  const uint32_t r = blockIdx.x;
  const EidolaSampleRow row = rows[r];
  const QwAcceptWork& w = work[r];
  if (w.kind == kAcceptDone) return;
  const bool greedy = row.temperature <= 0.0f;
  const uint32_t i = w.accepted;
  const uint64_t pos = static_cast<uint64_t>(row.position) + i;
  const double* pi = target + static_cast<uint64_t>(target_row[r] + i) * n;
  const double* qi =
      draft + (static_cast<uint64_t>(draft_row[r]) + static_cast<uint64_t>(i) * draft_step) * n;
  const auto pval = [&](uint32_t j) { return pi[j]; };
  uint32_t token;
  if (greedy) {
    token = argmax_f64(pi, n, sh);
  } else if (w.kind == kAcceptBonus) {
    token = draw_by_sums(pval, w.sums, n, uniform(row.seed, pos, kStreamSample), sh, stage);
  } else {
    const auto res = [&](uint32_t j) {
      const double d = __dsub_rn(pi[j], qi[j]);
      return d > 0.0 ? d : 0.0;
    };
    if (threadIdx.x == 0) {
      double rs = 0.0;
      for (uint32_t c = 0; c < chunks_of(n); ++c) rs = __dadd_rn(rs, w.sums[c]);
      sh.result = rs;
    }
    __syncthreads();
    const double rs = sh.result;
    __syncthreads();
    const double ur = uniform(row.seed, pos, kStreamResidual);
    if (rs > 0.0) {
      token = draw_by_sums(res, w.sums, n, ur, sh, stage);
    } else {
      chunk_sums_serial(pval, n, psums);
      token = draw_by_sums(pval, psums, n, ur, sh, stage);
    }
  }
  if (threadIdx.x == 0) {
    out[r * stride + i] = token;
    counts[r] = i + 1;
  }
}
EIDOLA_KERNEL_META(eidola_qwen_accept_finish, kThreads, 1, 1, 0, 1, 1, 1, 0);
