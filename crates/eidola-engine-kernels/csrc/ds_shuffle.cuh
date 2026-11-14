// The one device-only step of a pinned-order kernel (`ds_ops.cuh`): a lane
// group's 16 partial sums, collected by each of its threads.
#pragma once

#include <cstdint>

namespace dsk {

// Lane groups are threads `16g .. 16g + 16` of a block whose warps are full,
// so a group is half a warp. Every thread of the warp must call this
// together; `lanes[j]` is lane `j`'s `v`.
__device__ __forceinline__ void gather16(float v, float* lanes) {
  const uint32_t base = threadIdx.x & 16u;
  for (uint32_t j = 0; j < 16; ++j) lanes[j] = __shfl_sync(0xffffffffu, v, base + j);
}

}  // namespace dsk
