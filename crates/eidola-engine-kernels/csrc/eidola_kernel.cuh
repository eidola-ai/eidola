// Shared definitions for every translation unit this crate compiles to a cubin.
//
// Each kernel entry point publishes its launch contract as a device global
// named `<entry>_meta`, so the host reads the contract out of the very bytes
// the manifest hashes (cuModuleGetGlobal + a 32-byte copy) instead of
// re-deriving tile sizes, thread counts, or shared-memory budgets that only
// the C++ template instantiation knows.
#pragma once

#include <cstdint>

struct EidolaKernelMeta {
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
  // Dynamic shared memory the launch must request (and opt into with
  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES when above 48 KiB).
  uint32_t dynamic_smem_bytes;
  // Thread-block cluster shape; (1, 1, 1) means no cluster launch attribute.
  uint32_t cluster_x;
  uint32_t cluster_y;
  uint32_t cluster_z;
  // sizeof the single by-value parameter struct, or 0 for kernels taking a
  // plain argument list. A host-side mirror asserts against this.
  uint32_t params_bytes;
};
static_assert(sizeof(EidolaKernelMeta) == 32, "launch contract layout is fixed");

#define EIDOLA_KERNEL_META(entry, bx, by, bz, smem, cx, cy, cz, params) \
  extern "C" __device__ const EidolaKernelMeta entry##_meta = {        \
      (bx), (by), (bz), (smem), (cx), (cy), (cz), (params)}
