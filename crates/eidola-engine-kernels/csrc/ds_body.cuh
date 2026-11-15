// How a DeepSeek-V4.1 kernel with cooperating threads is written once, for the
// device and for the host check (`oracle/ds_attention_host.cc`).
//
// A kernel body is a function template over a phase runner. It is a sequence
// of phases, `par.run([&](uint32_t t) { … })`: thread `t`'s share of one
// phase. On the device every thread runs the body, executes each phase for its
// own index and meets a block barrier after it; on a host the runner calls the
// phase for every thread index in turn. The two agree when
//
// - no thread reads, inside a phase, memory another thread writes in that
//   phase (the host check runs every body in ascending and in descending
//   thread order, which exposes most such reads), and
// - everything outside a phase is a function of the parameters and of shared
//   memory read after a barrier, so every thread takes the same path.
//
// State that crosses a phase lives in the block's shared struct, never in a
// thread's locals.
#pragma once

#include "ds_abi.cuh"
#include "ds_numeric.cuh"

#if defined(__CUDACC__)
#define DS_DEV __device__ __forceinline__
#else
#define DS_DEV inline
#endif

namespace ds {

#if defined(__CUDACC__)
struct Par {
  template <class F>
  DS_DEV void run(F phase) const {
    phase(threadIdx.x);
    __syncthreads();
  }
};
DS_DEV void status_or(uint32_t* status, uint32_t bits) { atomicOr(status, bits); }
DS_DEV void count_add(uint32_t* counter, uint32_t n) { atomicAdd(counter, n); }
#else
struct Par {
  uint32_t threads;
  bool descending;
  template <class F>
  void run(F phase) const {
    for (uint32_t i = 0; i < threads; ++i) phase(descending ? threads - 1 - i : i);
  }
};
inline void status_or(uint32_t* status, uint32_t bits) { *status |= bits; }
inline void count_add(uint32_t* counter, uint32_t n) { *counter += n; }
#endif

// `ds_kv_code_bytes` and `ds_kv_scale_bytes` for device code: the header's are host `constexpr`
// functions, and a device call of one compiles, without a diagnostic, to a trap that takes the
// code after it with it.
DS_FN uint32_t kv_code_bytes(uint32_t format, uint32_t width) {
  return format == DS_KV_FP8 ? width : width / 2;
}
DS_FN uint32_t kv_scale_bytes(uint32_t format, uint32_t width) {
  return format == DS_KV_FP4_E4M3 ? width / 16 : width / 32;
}

// Where entry `e` of a plane keeps its codes and its scales.
DS_FN void kv_entry(const DsKvPlane& plane, uint32_t e, const uint8_t** codes,
                    const uint8_t** scales) {
  const uint32_t block = e / plane.entries_per_block;
  const uint64_t o = e % plane.entries_per_block;
  const uint8_t* at = plane.base + static_cast<uint64_t>(block) * plane.block_bytes;
  *codes = at + plane.codes_offset + o * kv_code_bytes(plane.format, plane.width);
  *scales = at + plane.scales_offset + o * kv_scale_bytes(plane.format, plane.width);
}

// Element `d` of a packed entry, as the value the reference keeps
// (`quant::fake_quant_*`'s result).
DS_FN float kv_element(uint32_t format, const uint8_t* codes, const uint8_t* scales, uint32_t d) {
  if (format == DS_KV_FP8) return dequant_e4m3(codes[d], act_scale_decode(scales[d >> 5]));
  const uint8_t nibble = fp4_nibble(codes, d);
  if (format == DS_KV_FP4_E4M3) return dequant_e2m1(nibble, e4m3_decode(scales[d >> 4]));
  return dequant_e2m1(nibble, act_scale_decode(scales[d >> 5]));
}

// The pool entry of logical index `x` (a position, or a bank entry) through
// one block table: `table[x / per_block] · per_block + x % per_block`. A
// negative table word raises `DS_STATUS_BAD_INDEX` and gives an empty slot.
DS_DEV uint32_t table_entry(const int32_t* table, uint32_t x, uint32_t per_block,
                            uint32_t* status) {
  const int32_t block = table[x / per_block];
  if (block < 0) {
    status_or(status, DS_STATUS_BAD_INDEX);
    return DS_SLOT_EMPTY;
  }
  return static_cast<uint32_t>(block) * per_block + x % per_block;
}

DS_FN uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace ds
