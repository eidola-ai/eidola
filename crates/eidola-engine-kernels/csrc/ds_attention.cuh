// Sparse attention and the window part of its slot lists: the bodies of
// `ds_attention.cu`'s kernels, compiled unchanged by the host check.
//
// `attend` is `attention::sparse_attend` for one block's heads of one row,
// operation for operation; the comments name the reference's steps.
#pragma once

#include "ds_body.cuh"

namespace ds_attn {

constexpr uint32_t kThreads = 512;
// Slots per block of the online softmax (`attention::ATTENTION_BLOCK`).
constexpr uint32_t kBlock = 64;
constexpr uint32_t kLanes = 16;
// The widest head, and the most heads one block computes.
constexpr uint32_t kMaxDim = 512;
constexpr uint32_t kMaxGroup = 8;

// One block's shared memory. `kv` and `q` are BF16 bits: every value held is a
// BF16 value, so 16 bits keep it exactly.
struct Shared {
  const uint8_t* codes[kBlock];
  const uint8_t* scales[kBlock];
  // Slots of the current 64 that hold an entry.
  uint64_t mask;
  float acc[kMaxGroup][kMaxDim];
  float lanes[kBlock][kLanes];
  float score[kBlock];
  float p[kBlock];
  float weight[kBlock];
  uint32_t format[kBlock];
  // Per head: the running maximum and the normaliser.
  float max[kMaxGroup];
  float den[kMaxGroup];
  float next_max;
  float rescale;
  uint32_t skip;
  uint32_t a_len;
  uint16_t kv[kBlock][kMaxDim];
  uint16_t q[kMaxGroup][kMaxDim];
  uint8_t present[kBlock];
};

// The heads one block computes when the grid is `gx` blocks wide.
DS_FN uint32_t heads_per_block(uint32_t heads, uint32_t gx) { return (heads + gx - 1) / gx; }

// Block `bx` of `gx`, row `row`: heads `bx · group ..` of the row.
template <class Par>
DS_DEV void attend(const DsAttnParams& p, Shared& sh, uint32_t bx, uint32_t gx, uint32_t row,
                   const Par& par) {
  static_assert(kThreads == (kBlock / 2) * kLanes, "a thread scores one lane of two slots");
  const uint32_t group = heads_per_block(p.heads, gx);
  const uint32_t h0 = bx * group;
  const uint32_t dim = p.head_dim;
  if (row >= p.rows || h0 >= p.heads || group > kMaxGroup || dim > kMaxDim) return;
  const uint32_t heads = ds::min_u32(group, p.heads - h0);
  const uint32_t* slots = p.slots + static_cast<uint64_t>(row) * p.slot_stride;
  const uint64_t at = (static_cast<uint64_t>(row) * p.heads + h0) * dim;
  const uint16_t* q = p.q + at;
  uint16_t* out = p.out + at;

  par.run([&](uint32_t t) {
    if (t == 0) {
      uint32_t a_len = p.a_slots;
      if (p.layout == DS_ATTN_PACKED) {
        while (a_len > 0 && slots[a_len - 1] == DS_SLOT_EMPTY) --a_len;
      }
      sh.a_len = a_len;
    }
    if (t < heads) {
      // `(max, den) = (-1e30, 0)`.
      sh.max[t] = -1e30f;
      sh.den[t] = 0.0f;
    }
    if (t < dim) {
      for (uint32_t g = 0; g < heads; ++g) {
        sh.q[g][t] = q[g * dim + t];
        sh.acc[g][t] = 0.0f;
      }
    }
  });
  const uint32_t a_len = sh.a_len;
  const uint32_t len = a_len + p.b_slots;

  for (uint32_t first = 0; first < len; first += kBlock) {
    // Where each of the block's slots keeps its entry.
    par.run([&](uint32_t t) {
      if (t >= kBlock) return;
      const uint32_t s = first + t;
      const bool in_a = s < a_len;
      const uint32_t word =
          s >= len ? DS_SLOT_EMPTY : in_a ? slots[s] : slots[p.a_slots + (s - a_len)];
      sh.present[t] = word != DS_SLOT_EMPTY;
      if (word == DS_SLOT_EMPTY) return;
      const DsKvPlane& plane = in_a ? p.a : p.b;
      sh.format[t] = plane.format;
      ds::kv_entry(plane, word, &sh.codes[t], &sh.scales[t]);
    });
    // The entries' values, dequantised once for every head of the block.
    par.run([&](uint32_t t) {
      if (t == 0) {
        uint64_t mask = 0;
        for (uint32_t j = 0; j < kBlock; ++j) mask |= static_cast<uint64_t>(sh.present[j]) << j;
        sh.mask = mask;
      }
      if (t >= dim) return;
      for (uint32_t j = 0; j < kBlock; ++j) {
        if (!sh.present[j]) continue;
        sh.kv[j][t] = ds::bf16_store(ds::kv_element(sh.format[j], sh.codes[j], sh.scales[j], t));
      }
    });
    const uint64_t mask = sh.mask;
    // A block without an entry changes nothing.
    if (mask == 0) continue;

    for (uint32_t g = 0; g < heads; ++g) {
      // `dot(qh, kv)`: lane `l` of a slot adds its products `i ≡ l (mod 16)` in ascending `i`
      // from +0.
      par.run([&](uint32_t t) {
        const uint32_t lane = t % kLanes, j0 = t / kLanes, j1 = j0 + kBlock / 2;
        const bool on0 = (mask >> j0) & 1, on1 = (mask >> j1) & 1;
        float s0 = 0.0f, s1 = 0.0f;
        for (uint32_t i = lane; i < dim; i += kLanes) {
          const float qi = ds::bf16_value(sh.q[g][i]);
          if (on0) s0 = s0 + qi * ds::bf16_value(sh.kv[j0][i]);
          if (on1) s1 = s1 + qi * ds::bf16_value(sh.kv[j1][i]);
        }
        sh.lanes[j0][lane] = s0;
        sh.lanes[j1][lane] = s1;
      });
      // The halving tree, the (empty) tail, and `· scale`.
      par.run([&](uint32_t t) {
        if (t >= kBlock || !((mask >> t) & 1)) return;
        float lanes[kLanes];
        for (uint32_t l = 0; l < kLanes; ++l) lanes[l] = sh.lanes[t][l];
        sh.score[t] = ds::pinned_reduce(lanes, 0.0f) * p.scale;
      });
      // The block's maximum (a NaN score never wins), then per entry `p = exp(s - max)` and
      // its BF16 weight. A block whose maximum is -inf is skipped whole.
      par.run([&](uint32_t t) {
        if (t >= kBlock) return;
        float block_max = ds::f32_from_bits(0xff800000u);
        for (uint32_t j = 0; j < kBlock; ++j) {
          if ((mask >> j) & 1) block_max = ds::max32(block_max, sh.score[j]);
        }
        const bool skip = block_max == ds::f32_from_bits(0xff800000u);
        const float prev = sh.max[g];
        const float max = ds::max32(prev, block_max);
        if (t == 0) {
          sh.skip = skip;
          sh.next_max = max;
          sh.rescale = skip ? 1.0f : ds::exp32(prev - max);
        }
        if (skip || !((mask >> t) & 1)) return;
        const float pj = ds::exp32(sh.score[t] - max);
        sh.p[t] = pj;
        sh.weight[t] = ds::bf16_round(pj);
      });
      if (sh.skip) continue;
      // `acc *= r`, then each entry in slot order; the normaliser gains the block's `p` left
      // to right.
      par.run([&](uint32_t t) {
        const float r = sh.rescale;
        if (t < dim) {
          float a = sh.acc[g][t] * r;
          for (uint32_t j = 0; j < kBlock; ++j) {
            if ((mask >> j) & 1) a = a + sh.weight[j] * ds::bf16_value(sh.kv[j][t]);
          }
          sh.acc[g][t] = a;
        }
        if (t == 0) {
          float block_sum = 0.0f;
          for (uint32_t j = 0; j < kBlock; ++j) {
            if ((mask >> j) & 1) block_sum = block_sum + sh.p[j];
          }
          sh.den[g] = sh.den[g] * r + block_sum;
          sh.max[g] = sh.next_max;
        }
      });
    }
  }

  // The sink joins the normaliser and adds no value.
  par.run([&](uint32_t t) {
    if (t < heads) sh.den[t] = sh.den[t] + ds::exp32(p.sink[h0 + t] - sh.max[t]);
  });
  par.run([&](uint32_t t) {
    if (t >= dim) return;
    for (uint32_t g = 0; g < heads; ++g) {
      out[g * dim + t] = ds::bf16_store(ds::bf16_round(sh.acc[g][t] / sh.den[g]));
    }
  });
}

constexpr uint32_t kSlotThreads = 128;

// Slot `bx · kSlotThreads + t` of row `row`'s window (`DsWindowSlotsParams`).
DS_DEV void window_slot(const DsWindowSlotsParams& p, uint32_t bx, uint32_t row, uint32_t t) {
  const uint32_t i = bx * kSlotThreads + t;
  if (row >= p.rows || i >= p.window) return;
  const uint32_t pos = p.positions[row];
  bool listed;
  uint32_t x;
  if (p.order == DS_ORDER_RING) {
    // `drafter::ring_order`: slot `i` holds the newest position congruent to `i`.
    listed = i <= pos;
    x = listed ? pos - (pos - i) % p.window : 0;
  } else {
    // Right-aligned: the row's own position is the last slot.
    const uint64_t shifted = static_cast<uint64_t>(pos) + i + 1;
    listed = shifted >= p.window;
    x = listed ? static_cast<uint32_t>(shifted - p.window) : 0;
    listed = listed && x >= p.first[row];
  }
  uint32_t word = DS_SLOT_EMPTY;
  if (listed) {
    const int32_t* table = p.tables + static_cast<uint64_t>(p.table_row[row]) * p.table_stride;
    word = ds::table_entry(table, x, p.block_positions, p.status);
  }
  p.slots[static_cast<uint64_t>(row) * p.slot_stride + i] = word;
}

}  // namespace ds_attn
