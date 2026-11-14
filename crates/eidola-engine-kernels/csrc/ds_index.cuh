// The indexer's scores and the total-order selections over them: the bodies of
// `ds_index.cu`'s kernels, compiled unchanged by the host check.
#pragma once

#include "ds_body.cuh"

namespace ds_index {

// ---- index scores (`attention::index_score`) ----

constexpr uint32_t kScoreThreads = 256;
constexpr uint32_t kWarp = 32;
// Entries a block scores per pass: one per warp, a lane per index head.
constexpr uint32_t kPass = kScoreThreads / kWarp;
constexpr uint32_t kMaxHeads = kWarp;
constexpr uint32_t kMaxDim = 128;

struct ScoreShared {
  const uint8_t* codes[kPass];
  const uint8_t* scales[kPass];
  // The row's index queries and head weights, dequantised once.
  float q[kMaxHeads][kMaxDim];
  float key[kPass][kMaxDim];
  float term[kPass][kMaxHeads];
  float weight[kMaxHeads];
  uint32_t ok[kPass];
};

// Block `bx` of `gx`, row `row`: the passes `bx, bx + gx, …` of the row's visible entries, pass
// `g` being entries `g · kPass ..`.
template <class Par>
DS_DEV void index_score(const DsIndexScoreParams& p, ScoreShared& sh, uint32_t bx, uint32_t gx,
                        uint32_t row, const Par& par) {
  const uint32_t heads = p.heads, dim = p.dim;
  if (row >= p.rows || heads > kMaxHeads || dim > kMaxDim) return;
  const uint32_t seen = p.visible[row];
  const uint32_t visible = ds::min_u32(seen, p.score_stride);
  if (static_cast<uint64_t>(bx) * kPass >= visible) return;
  const uint32_t width = heads * dim;
  const uint8_t* q_codes = p.q_codes + static_cast<uint64_t>(row) * (width / 2);
  const uint8_t* q_scales = p.q_scales + static_cast<uint64_t>(row) * (width / 32);
  const uint16_t* head_dots = p.head_dots + static_cast<uint64_t>(row) * heads;
  const int32_t* table = p.tables + static_cast<uint64_t>(p.table_row[row]) * p.table_stride;
  uint16_t* scores = p.scores + static_cast<uint64_t>(row) * p.score_stride;
  const uint32_t per_block = p.keys.entries_per_block;

  par.run([&](uint32_t t) {
    if (t == 0 && bx == 0 && seen > p.score_stride) ds::status_or(p.status, DS_STATUS_BAD_INDEX);
    for (uint32_t e = t; e < width; e += kScoreThreads) {
      sh.q[e / dim][e % dim] = ds::kv_element(DS_KV_FP4_E8M0, q_codes, q_scales, e);
    }
    // `bf16(bf16(dot(u, weights_proj[h])) · weight_scale)`.
    if (t < heads) sh.weight[t] = ds::bf16_round(ds::bf16_value(head_dots[t]) * p.weight_scale);
  });

  for (uint64_t g = bx; g * kPass < visible; g += gx) {
    // The entry's key.
    par.run([&](uint32_t t) {
      const uint32_t w = t / kWarp, lane = t % kWarp;
      const uint64_t entry = g * kPass + w;
      if (entry >= visible) {
        if (lane == 0) sh.ok[w] = 0;
        return;
      }
      const uint32_t e = static_cast<uint32_t>(entry);
      const int32_t block = table[e / per_block];
      if (lane == 0) {
        sh.ok[w] = block >= 0;
        if (block < 0) ds::status_or(p.status, DS_STATUS_BAD_INDEX);
      }
      if (block < 0) return;
      const uint8_t *codes, *scales;
      ds::kv_entry(p.keys, static_cast<uint32_t>(block) * per_block + e % per_block, &codes,
                   &scales);
      for (uint32_t i = lane; i < dim; i += kWarp) {
        sh.key[w][i] = ds::kv_element(p.keys.format, codes, scales, i);
      }
    });
    // Per head: `bf16(dot(q_h, key))`, rectified, times the head's weight, rounded.
    par.run([&](uint32_t t) {
      const uint32_t w = t / kWarp, h = t % kWarp;
      if (h >= heads || !sh.ok[w]) return;
      const float s = ds::bf16_round(ds::pinned_dot(sh.q[h], sh.key[w], dim));
      sh.term[w][h] = ds::bf16_round((s > 0.0f ? s : 0.0f) * sh.weight[h]);
    });
    // The head terms in the pinned order, rounded. An entry behind an unmapped table word
    // scores +0.
    par.run([&](uint32_t t) {
      const uint32_t w = t / kWarp;
      const uint64_t entry = g * kPass + w;
      if (t % kWarp != 0 || entry >= visible) return;
      const float score = sh.ok[w] ? ds::bf16_round(ds::pinned_sum(sh.term[w], heads)) : 0.0f;
      scores[entry] = ds::bf16_store(score);
    });
  }
}

// ---- selection (`attention::top_k`, `attention::candidate_mask`) ----

constexpr uint32_t kSelectThreads = 512;

struct SelectShared {
  uint32_t hist[256];
  // Per thread, over its run of candidates: how many rank above the threshold and how many tie
  // with it; then the same counts over the runs before it.
  uint32_t above[kSelectThreads];
  uint32_t ties[kSelectThreads];
  uint32_t high;
  uint32_t rest;
  uint32_t threshold;
  uint32_t take_ties;
  // The candidates are fewer than `k`: every one is selected.
  uint32_t all;
  uint32_t total;
};

// A BF16 score's rank key in 16 bits: `ds::topk_key` of a BF16 value has no information below
// them.
DS_FN uint32_t key16(float score) { return ds::topk_key(score) >> 16; }

// Leaves in `sh` the selection of the best `k` of `n` candidates under the total order (key
// descending, index ascending): `threshold` and `take_ties` (every candidate above the
// threshold, and the first `take_ties` in index order that equal it), or `all`. `key(i)` is
// candidate `i`'s 16-bit key, or negative when `i` is not a candidate.
template <class Par, class Key>
DS_DEV void threshold(SelectShared& sh, const Key& key, uint32_t n, uint32_t k, const Par& par) {
  par.run([&](uint32_t t) {
    if (t < 256) sh.hist[t] = 0;
  });
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < n; i += kSelectThreads) {
      const int32_t v = key(i);
      if (v >= 0) ds::count_add(&sh.hist[v >> 8], 1);
    }
  });
  // The high byte whose candidates hold the k-th best.
  par.run([&](uint32_t t) {
    if (t != 0) return;
    uint32_t above = 0, found = 0;
    for (uint32_t b = 256; b-- > 0;) {
      if (!found && above + sh.hist[b] >= k) {
        sh.high = b;
        sh.rest = k - above;
        found = 1;
      }
      if (!found) above += sh.hist[b];
      sh.hist[b] = 0;
    }
    sh.all = !found;
    sh.threshold = 0;
    sh.take_ties = 0;
  });
  if (sh.all) return;
  const uint32_t high = sh.high;
  par.run([&](uint32_t t) {
    for (uint32_t i = t; i < n; i += kSelectThreads) {
      const int32_t v = key(i);
      if (v >= 0 && static_cast<uint32_t>(v) >> 8 == high) ds::count_add(&sh.hist[v & 255], 1);
    }
  });
  par.run([&](uint32_t t) {
    if (t != 0) return;
    uint32_t above = 0;
    for (uint32_t b = 256; b-- > 0;) {
      if (above + sh.hist[b] >= sh.rest) {
        sh.threshold = high << 8 | b;
        sh.take_ties = sh.rest - above;
        break;
      }
      above += sh.hist[b];
    }
  });
}

// Whether a candidate of key `v`, with `ties_before` equal candidates before it in index order,
// is selected.
DS_FN bool selected(const SelectShared& sh, uint32_t v, uint32_t ties_before) {
  return sh.all || v > sh.threshold || (v == sh.threshold && ties_before < sh.take_ties);
}

// The rank of every thread's run: thread `t` owns candidates `t · run .. (t + 1) · run`. Leaves
// in `above[t]` and `ties[t]` the counts over the runs before `t`, and `total` selected.
template <class Par, class Key>
DS_DEV void rank(SelectShared& sh, const Key& key, uint32_t n, uint32_t run, const Par& par) {
  par.run([&](uint32_t t) {
    const uint64_t lo = static_cast<uint64_t>(t) * run;
    uint32_t above = 0, ties = 0;
    for (uint64_t i = lo; i < lo + run && i < n; ++i) {
      const int32_t v = key(static_cast<uint32_t>(i));
      if (v < 0) continue;
      const uint32_t u = static_cast<uint32_t>(v);
      if (sh.all || u > sh.threshold) {
        ++above;
      } else if (u == sh.threshold) {
        ++ties;
      }
    }
    sh.above[t] = above;
    sh.ties[t] = ties;
  });
  par.run([&](uint32_t t) {
    if (t != 0) return;
    uint32_t above = 0, ties = 0;
    for (uint32_t u = 0; u < kSelectThreads; ++u) {
      const uint32_t a = sh.above[u], e = sh.ties[u];
      sh.above[u] = above;
      sh.ties[u] = ties;
      above += a;
      ties += e;
    }
    sh.total = above + ds::min_u32(ties, sh.take_ties);
  });
}

// A candidate block's score: its best entry (a NaN never wins), +inf for the block holding the
// newest entry.
struct BlockKey {
  const uint16_t* scores;
  uint32_t visible;
  uint32_t block_entries;
  DS_FN float score(uint32_t b) const {
    if (b == (visible - 1) / block_entries) return ds::f32_from_bits(0x7f800000u);
    float best = ds::f32_from_bits(0xff800000u);
    // `b` is a block of the visible entries, so its first entry is below `visible`.
    const uint32_t lo = b * block_entries;
    const uint32_t n = ds::min_u32(block_entries, visible - lo);
    for (uint32_t j = 0; j < n; ++j) best = ds::max32(best, ds::bf16_value(scores[lo + j]));
    return best;
  }
  DS_FN int32_t operator()(uint32_t b) const { return static_cast<int32_t>(key16(score(b))); }
};

// An entry's key when the row may select it: every visible entry, or those inside a block of
// `mask`.
struct EntryKey {
  const uint16_t* scores;
  const uint32_t* mask;
  uint32_t mask_bits;
  uint32_t block_entries;
  DS_FN int32_t operator()(uint32_t i) const {
    if (mask != nullptr) {
      const uint32_t b = i / block_entries;
      if (b >= mask_bits || !((mask[b >> 5] >> (b & 31u)) & 1u)) return -1;
    }
    return static_cast<int32_t>(key16(ds::bf16_value(scores[i])));
  }
};

// Row `row` of `DsSelectParams`, one block.
template <class Par>
DS_DEV void select(const DsSelectParams& p, SelectShared& sh, uint32_t row, const Par& par) {
  if (row >= p.rows) return;
  const uint32_t seen = p.visible[row];
  const uint32_t visible = ds::min_u32(seen, p.score_stride);
  const uint16_t* scores = p.scores + static_cast<uint64_t>(row) * p.score_stride;
  const uint32_t mask_bits = p.mask_words * 32u;

  if (p.mask_out != nullptr) {
    uint32_t* out = p.mask_out + static_cast<uint64_t>(row) * p.mask_words;
    const uint32_t blocks =
        ds::min_u32(visible / p.block_entries + (visible % p.block_entries != 0), mask_bits);
    const BlockKey key{scores, visible, p.block_entries};
    threshold(sh, key, blocks, p.top_blocks, par);
    // A thread owns whole mask words.
    const uint32_t words = (p.mask_words + kSelectThreads - 1) / kSelectThreads;
    rank(sh, key, blocks, words * 32u, par);
    par.run([&](uint32_t t) {
      uint32_t ties = sh.ties[t];
      for (uint64_t w = static_cast<uint64_t>(t) * words; w < (t + 1ull) * words && w < p.mask_words;
           ++w) {
        uint32_t bits = 0;
        for (uint32_t j = 0; j < 32; ++j) {
          const uint64_t b = w * 32 + j;
          if (b >= blocks) break;
          const float s = key.score(static_cast<uint32_t>(b));
          const uint32_t v = key16(s);
          const bool take = selected(sh, v, ties);
          if (!sh.all && v == sh.threshold) ++ties;
          // A kept block holds a score above -inf.
          if (take && s > ds::f32_from_bits(0xff800000u)) bits |= 1u << j;
        }
        out[w] = bits;
      }
    });
  }

  const uint32_t* mask_in =
      p.mask_in == nullptr ? nullptr : p.mask_in + static_cast<uint64_t>(row) * p.mask_words;
  const EntryKey key{scores, mask_in, mask_bits, p.block_entries};
  const int32_t* table = p.tables + static_cast<uint64_t>(p.table_row[row]) * p.table_stride;
  uint32_t* slots = p.slots + static_cast<uint64_t>(row) * p.slot_stride + p.slot_first;
  threshold(sh, key, visible, p.top_k, par);
  const uint32_t run = visible / kSelectThreads + (visible % kSelectThreads != 0);
  rank(sh, key, visible, run, par);
  // The selected entries in ascending order, as entries of the bank's planes.
  par.run([&](uint32_t t) {
    if (t == 0 && seen > p.score_stride) ds::status_or(p.status, DS_STATUS_BAD_INDEX);
    uint32_t ties = sh.ties[t];
    uint32_t at = sh.above[t] + ds::min_u32(ties, sh.take_ties);
    const uint64_t lo = static_cast<uint64_t>(t) * run;
    for (uint64_t i = lo; i < lo + run && i < visible; ++i) {
      const int32_t v = key(static_cast<uint32_t>(i));
      if (v < 0) continue;
      const uint32_t u = static_cast<uint32_t>(v);
      const bool take = selected(sh, u, ties);
      if (!sh.all && u == sh.threshold) ++ties;
      if (take) {
        slots[at++] =
            ds::table_entry(table, static_cast<uint32_t>(i), p.entries_per_block, p.status);
      }
    }
  });
  const uint32_t total = sh.total;
  par.run([&](uint32_t t) {
    for (uint64_t i = static_cast<uint64_t>(total) + t; i < p.top_k; i += kSelectThreads) {
      slots[i] = DS_SLOT_EMPTY;
    }
  });
}

}  // namespace ds_index
