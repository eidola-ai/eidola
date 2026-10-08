// Sampling and chain speculative acceptance: the serving core's sampler
// (eidola-engine/src/sampling.rs) on the device, reproducing it bit for bit
// given equal logits.
//
// The core pins its arithmetic for exactly this: f64 throughout, exponentials
// from det_exp (correctly rounded + - * / only), and every sum over the
// vocabulary in one fixed grouping (pinned_sum: chunks of 1024 consecutive ids
// summed left to right, then the chunk sums left to right). This file performs
// the same operations in the same grouping:
//
// - every f64 add, subtract, multiply and divide is an explicit
//   round-to-nearest intrinsic (__dadd_rn, ...), which nvcc never contracts
//   into an FMA (the DFMAs in the SASS belong to the correctly rounded
//   division routine);
// - a chunk sum is one thread's left-to-right loop over its chunk, and the
//   chunk sums are added left to right by one thread;
// - max, counts, comparisons and f32 -> f64 conversion are exact, so their
//   parallel reductions are order-free.
//
// Top-k, top-p and min-p select sets by rank in (p descending, id ascending)
// order. Nothing is sorted: a set of top ranks is the set of keys at or above
// a threshold (p_T, id_T) — "p > p_T, or p == p_T and id <= id_T" — found by
// bitwise searches over the bits of p (non-negative doubles order like their
// bit patterns) and then over ids, each step one block-wide count or pinned
// sum. Masses grow with the set, so top-p's "shortest prefix whose mass
// reaches the target" is the largest threshold whose set still reaches it.
//
// One block of 1024 threads per row. The sampleable vocabulary may be at most
// 1024 chunks (1,048,576 ids).

#include <cstdint>

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kThreads = 1024;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kChunk = 1024;  // SUM_CHUNK in sampling.rs

// Stream ids (sampling::Stream).
constexpr uint64_t kStreamAccept = 1;
constexpr uint64_t kStreamResidual = 2;
constexpr uint64_t kStreamSample = 0;

// Status bits.
constexpr uint32_t kStatusNonFinite = 1;

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ double uniform(uint64_t seed, uint64_t position, uint64_t stream) {
  const uint64_t w = mix64(mix64(seed) ^ mix64(position * 0x9e3779b97f4a7c15ull + stream));
  return __dmul_rn(static_cast<double>(w >> 40), 1.0 / 16777216.0);
}

__device__ __forceinline__ double from_bits(uint64_t b) { return __longlong_as_double(static_cast<long long>(b)); }
__device__ __forceinline__ uint64_t to_bits(double d) { return static_cast<uint64_t>(__double_as_longlong(d)); }

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

// ---- block reductions (blockDim.x == kThreads) ----

template <typename T, typename Op>
__device__ T block_reduce(T v, Op op, T* scratch) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = scratch[threadIdx.x];  // kWarps == 32
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) scratch[0] = v;
  }
  __syncthreads();
  const T out = scratch[0];
  __syncthreads();
  return out;
}

struct Shared {
  double d[kWarps];
  uint64_t u[kWarps];
  double chunk[kChunk];
  double result;
  uint32_t index;
};

__device__ uint64_t block_sum_u64(uint64_t v, Shared& sh) {
  return block_reduce<uint64_t>(v, [](uint64_t a, uint64_t b) { return a + b; }, sh.u);
}

__device__ uint64_t block_max_u64(uint64_t v, Shared& sh) {
  return block_reduce<uint64_t>(v, [](uint64_t a, uint64_t b) { return a > b ? a : b; }, sh.u);
}

__device__ uint64_t block_min_u64(uint64_t v, Shared& sh) {
  return block_reduce<uint64_t>(v, [](uint64_t a, uint64_t b) { return a < b ? a : b; }, sh.u);
}

// Membership of id i (probability p) in a rank set: key >= (pt, it) in
// (p descending, id ascending) order, and p >= floor.
struct RankSet {
  uint64_t pt;   // bits of p_T
  uint32_t it;   // id_T
  double floor;  // min-p floor (0 when unused)
  __device__ bool contains(double p, uint32_t i) const {
    const uint64_t pb = to_bits(p);
    return (pb > pt || (pb == pt && i <= it)) && p >= floor;
  }
};

// pinned_sum over the ids `keep` selects: thread c sums chunk c left to right,
// thread 0 adds the chunk sums left to right.
template <typename Keep>
__device__ double pinned_sum(const double* x, uint32_t n, Keep keep, Shared& sh) {
  const uint32_t chunks = (n + kChunk - 1) / kChunk;
  if (threadIdx.x < chunks) {
    const uint32_t lo = threadIdx.x * kChunk;
    const uint32_t hi = min(lo + kChunk, n);
    double s = 0.0;
    for (uint32_t i = lo; i < hi; ++i) {
      if (keep(i)) s = __dadd_rn(s, x[i]);
    }
    sh.chunk[threadIdx.x] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) total = __dadd_rn(total, sh.chunk[c]);
    sh.result = total;
  }
  __syncthreads();
  const double out = sh.result;
  __syncthreads();
  return out;
}

__device__ uint64_t count_where_bits_ge(const double* p, uint32_t n, const RankSet& set,
                                        uint64_t v, Shared& sh) {
  uint64_t c = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (set.contains(p[i], i) && to_bits(p[i]) >= v) ++c;
  }
  return block_sum_u64(c, sh);
}

// The largest p-bits value v with pred(v) true, where pred is monotone
// (true up to some value, false above) and pred(0) holds.
template <typename Pred>
__device__ uint64_t search_bits(Pred pred) {
  uint64_t v = 0;
  for (int b = 63; b >= 0; --b) {
    const uint64_t t = v | (1ull << b);
    if (pred(t)) v = t;
  }
  return v;
}

// The smallest id I in [0, n) with pred(I) true, pred monotone (false then
// true) and pred(n - 1) true.
template <typename Pred>
__device__ uint32_t search_id(uint32_t n, Pred pred) {
  uint32_t lo = 0, hi = n - 1;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (pred(mid)) hi = mid; else lo = mid + 1;
  }
  return lo;
}

// The smallest key value (as RankSet threshold) among members whose p bits are
// >= v: the value p_T itself, for a following id search.
__device__ uint64_t min_member_bits_ge(const double* p, uint32_t n, const RankSet& set, uint64_t v,
                                       Shared& sh) {
  uint64_t m = ~0ull;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    const uint64_t b = to_bits(p[i]);
    if (set.contains(p[i], i) && b >= v && b < m) m = b;
  }
  return block_min_u64(m, sh);
}

// Inverse-CDF draw with the pinned chunked prefix sums (sample_from).
__device__ uint32_t draw(const double* p, uint32_t n, double u, Shared& sh) {
  const uint32_t chunks = (n + kChunk - 1) / kChunk;
  // Last nonzero id (the fallback).
  uint64_t last = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (p[i] > 0.0) last = max(last, static_cast<uint64_t>(i));
  }
  last = block_max_u64(last, sh);
  if (threadIdx.x < chunks) {
    const uint32_t lo = threadIdx.x * kChunk;
    const uint32_t hi = min(lo + kChunk, n);
    double s = 0.0;
    for (uint32_t i = lo; i < hi; ++i) s = __dadd_rn(s, p[i]);
    sh.chunk[threadIdx.x] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) total = __dadd_rn(total, sh.chunk[c]);
    const double target = __dmul_rn(u, total);
    uint32_t token = static_cast<uint32_t>(last);
    double base = 0.0;
    for (uint32_t c = 0; c < chunks; ++c) {
      const double cs = sh.chunk[c];
      if (__dadd_rn(base, cs) > target) {
        const uint32_t lo = c * kChunk;
        const uint32_t hi = min(lo + kChunk, n);
        double cum = 0.0;
        for (uint32_t i = lo; i < hi; ++i) {
          cum = __dadd_rn(cum, p[i]);
          if (p[i] > 0.0 && __dadd_rn(base, cum) > target) {
            token = i;
            break;
          }
        }
        break;
      }
      base = __dadd_rn(base, cs);
    }
    sh.index = token;
  }
  __syncthreads();
  const uint32_t out = sh.index;
  __syncthreads();
  return out;
}

// Lowest-id argmax of f32 values (sampling::argmax), with a non-finite check.
__device__ uint32_t argmax_f32(const float* l, uint32_t n, bool* non_finite, Shared& sh) {
  float best = -INFINITY;
  uint32_t idx = 0xffffffffu;
  bool bad = false;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    const float v = l[i];
    if (isnan(v) || v == INFINITY) bad = true;
    if (idx == 0xffffffffu || v > best) {
      best = v;
      idx = i;
    }
  }
  // Order-free reduction: larger value wins, ties to the lower id. Pack as
  // (orderable value bits, inverted id) into one u64 and take the max. -0.0
  // and +0.0 are equal (the reference's `>` sees a tie), so both pack as +0.0
  // and the lower id wins.
  uint32_t vb = __float_as_uint(best == 0.0f ? 0.0f : best);
  vb = (vb & 0x80000000u) ? ~vb : (vb | 0x80000000u);
  const uint64_t key = idx == 0xffffffffu ? 0 : (static_cast<uint64_t>(vb) << 32) | (0xffffffffu - idx);
  const uint64_t top = block_max_u64(key, sh);
  const uint64_t any_bad = block_max_u64(bad ? 1 : 0, sh);
  *non_finite = any_bad != 0;
  return 0xffffffffu - static_cast<uint32_t>(top & 0xffffffffu);
}

__device__ uint32_t argmax_f64(const double* p, uint32_t n, Shared& sh) {
  // p >= 0: bit patterns order like values.
  uint64_t best = 0;
  uint32_t idx = 0xffffffffu;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (idx == 0xffffffffu || to_bits(p[i]) > best) {
      best = to_bits(p[i]);
      idx = i;
    }
  }
  // Ties go low: reduce on (bits, inverted id). Bits fit 63 bits for p >= 0,
  // ids below 2^20, so split as max over bits then min id at that value.
  const uint64_t top = block_max_u64(idx == 0xffffffffu ? 0 : best, sh);
  uint64_t lo = ~0ull;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    if (to_bits(p[i]) == top && i < lo) lo = i;
  }
  return static_cast<uint32_t>(block_min_u64(lo, sh));
}

}  // namespace

// Per-row sampling parameters (32 bytes), the device copy of SamplingParams
// plus where the row's logits are and which position it draws for.
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

// For each row: the processed distribution (probs, [rows, n]) and, when
// `draw_stream` is not ~0, a token drawn from it with that stream (greedy rows
// take the argmax and draw nothing). `status` gets kStatusNonFinite when a
// row's logits hold NaN or +inf, or are all -inf on a non-greedy row.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_sample(const float* __restrict__ logits, uint64_t logits_stride, uint32_t n,
                  const EidolaSampleRow* __restrict__ rows, uint32_t draw_stream,
                  double* __restrict__ probs, uint32_t* __restrict__ tokens,
                  uint32_t* __restrict__ status) {
  __shared__ Shared sh;
  const EidolaSampleRow row = rows[blockIdx.x];
  const float* l = logits + static_cast<uint64_t>(row.logit_row) * logits_stride;
  double* p = probs + static_cast<uint64_t>(blockIdx.x) * n;

  bool non_finite = false;
  const uint32_t top = argmax_f32(l, n, &non_finite, sh);
  if (row.temperature <= 0.0f) {
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) p[i] = i == top ? 1.0 : 0.0;
    if (threadIdx.x == 0) {
      if (draw_stream != 0xffffffffu) tokens[blockIdx.x] = top;
      if (non_finite) atomicOr(status, kStatusNonFinite);
    }
    return;
  }

  // Softmax at temperature t (processed_probs steps 2).
  const double t = static_cast<double>(row.temperature);
  // z = l / t; its max is at the f32 argmax (division by t > 0 is monotone).
  const double zmax = __ddiv_rn(static_cast<double>(l[top]), t);
  if (threadIdx.x == 0 && (non_finite || isinf(zmax))) atomicOr(status, kStatusNonFinite);
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    p[i] = det_exp(__dsub_rn(__ddiv_rn(static_cast<double>(l[i]), t), zmax));
  }
  __syncthreads();
  const double sum = pinned_sum(p, n, [](uint32_t) { return true; }, sh);
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) p[i] = __ddiv_rn(p[i], sum);
  __syncthreads();

  const bool filtering = row.top_k > 0 || row.top_p < 1.0f || row.min_p > 0.0f;
  if (filtering) {
    RankSet set{0, n - 1, 0.0};  // everything
    // Top-k: the threshold at rank keep - 1.
    if (row.top_k > 0 && row.top_k < n) {
      const uint64_t keep = row.top_k;
      const uint64_t pt =
          search_bits([&](uint64_t v) { return count_where_bits_ge(p, n, set, v, sh) >= keep; });
      const uint64_t above = count_where_bits_ge(p, n, set, pt + 1, sh);
      const uint64_t need = keep - above;  // >= 1 ids at p == p_T, lowest first
      const uint32_t it = search_id(n, [&](uint32_t id) {
        uint64_t c = 0;
        for (uint32_t i = threadIdx.x; i <= id && i < n; i += kThreads) {
          if (to_bits(p[i]) == pt) ++c;
        }
        return block_sum_u64(c, sh) >= need;
      });
      set = RankSet{pt, it, 0.0};
    }
    // Top-p: the shortest prefix of the kept set reaching top_p of its mass.
    if (row.top_p < 1.0f) {
      const RankSet kept = set;
      const double mass = pinned_sum(p, n, [&](uint32_t i) { return kept.contains(p[i], i); }, sh);
      const double target = __dmul_rn(static_cast<double>(row.top_p), mass);
      const auto mass_ge = [&](uint64_t v) {
        return pinned_sum(p, n,
                          [&](uint32_t i) { return kept.contains(p[i], i) && to_bits(p[i]) >= v; },
                          sh);
      };
      const uint64_t v = search_bits([&](uint64_t b) { return mass_ge(b) >= target; });
      const uint64_t pt = min_member_bits_ge(p, n, kept, v, sh);
      const uint32_t it = search_id(n, [&](uint32_t id) {
        return pinned_sum(p, n,
                          [&](uint32_t i) {
                            const uint64_t b = to_bits(p[i]);
                            return kept.contains(p[i], i) && (b > pt || (b == pt && i <= id));
                          },
                          sh) >= target;
      });
      set = RankSet{pt, it, 0.0};
    }
    // Min-p: tokens of the kept set with p >= min_p * max(p).
    if (row.min_p > 0.0f) {
      set.floor = __dmul_rn(static_cast<double>(row.min_p), p[top]);
    }
    const RankSet kept = set;
    const double mass = pinned_sum(p, n, [&](uint32_t i) { return kept.contains(p[i], i); }, sh);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
      p[i] = kept.contains(p[i], i) ? __ddiv_rn(p[i], mass) : 0.0;
    }
    __syncthreads();
  }

  if (draw_stream != 0xffffffffu) {
    const double u = uniform(row.seed, row.position, draw_stream);
    const uint32_t token = draw(p, n, u, sh);
    if (threadIdx.x == 0) tokens[blockIdx.x] = token;
  }
}

EIDOLA_KERNEL_META(eidola_sample, kThreads, 1, 1, 0, 1, 1, 1, 0);

// Chain speculative acceptance (sampling::chain_accept) for one sequence per
// block. Row r has `num_drafts[r] = k` drafts; its target distributions are
// `target[target_row[r] + 0 ..= k]` and its draft distributions
// `draft[draft_row[r] + 0 .. k]` (each a [n] f64 row from eidola_sample).
// Writes 1..=k+1 tokens to out[r * stride ..] and their count to counts[r].
// `scratch` holds one [n] row per block for the residual.
extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_chain_accept(const double* __restrict__ target, const double* __restrict__ draft,
                        uint32_t n, const EidolaSampleRow* __restrict__ rows,
                        const uint32_t* __restrict__ target_row,
                        const uint32_t* __restrict__ draft_row,
                        const uint32_t* __restrict__ num_drafts, const uint32_t* __restrict__ drafts,
                        uint32_t stride, double* __restrict__ scratch, uint32_t* __restrict__ out,
                        uint32_t* __restrict__ counts) {
  __shared__ Shared sh;
  const uint32_t r = blockIdx.x;
  const EidolaSampleRow row = rows[r];
  const bool greedy = row.temperature <= 0.0f;
  const uint32_t k = num_drafts[r];
  // `position` is the first drafted position (p + 1).
  const uint64_t first = row.position;
  double* res = scratch + static_cast<uint64_t>(r) * n;
  uint32_t produced = 0;
  for (uint32_t i = 0; i < k; ++i) {
    const double* p = target + static_cast<uint64_t>(target_row[r] + i) * n;
    const double* q = draft + static_cast<uint64_t>(draft_row[r] + i) * n;
    const uint32_t d = drafts[r * stride + i];
    const uint64_t pos = first + i;
    const bool accepted =
        greedy ? p[d] > 0.0
               : __dmul_rn(uniform(row.seed, pos, kStreamAccept), q[d]) < p[d];
    if (accepted) {
      if (threadIdx.x == 0) out[r * stride + produced] = d;
      ++produced;
      continue;
    }
    uint32_t token;
    if (greedy) {
      token = argmax_f64(p, n, sh);
    } else {
      for (uint32_t j = threadIdx.x; j < n; j += kThreads) {
        const double v = __dsub_rn(p[j], q[j]);
        res[j] = v > 0.0 ? v : 0.0;
      }
      __syncthreads();
      const double rs = pinned_sum(res, n, [](uint32_t) { return true; }, sh);
      token = draw(rs > 0.0 ? res : p, n, uniform(row.seed, pos, kStreamResidual), sh);
    }
    if (threadIdx.x == 0) {
      out[r * stride + produced] = token;
      counts[r] = produced + 1;
    }
    return;
  }
  const double* pk = target + static_cast<uint64_t>(target_row[r] + k) * n;
  const uint32_t bonus =
      greedy ? argmax_f64(pk, n, sh) : draw(pk, n, uniform(row.seed, first + k, kStreamSample), sh);
  if (threadIdx.x == 0) {
    out[r * stride + produced] = bonus;
    counts[r] = produced + 1;
  }
}

EIDOLA_KERNEL_META(eidola_chain_accept, kThreads, 1, 1, 0, 1, 1, 1, 0);
