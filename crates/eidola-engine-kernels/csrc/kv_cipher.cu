// ChaCha20 (RFC 8439 block function, 20 rounds, 32-bit counter, 96-bit nonce)
// as a bare stream cipher over idle prefix-cache KV: the device form of
// `eidola-engine`'s `kv_cipher`, which defines it and which the GPU tests
// compare this kernel with byte for byte.
//
// Additions, rotations and XORs on 32-bit words only: no table and no
// secret-dependent address or branch (the one branch below depends on a
// range's length and alignment, which are public geometry).
//
// One launch applies a list of items. Item i XORs `words` little-endian
// 32-bit words at `base` with the keystream of (`key`, `nonce`) from block
// `counter` on: word j takes keystream word j % 16 of block
// `counter + j / 16`. Encrypting and decrypting are the same operation. The
// host builds one item per contiguous range of a block (its image order is
// the executor's), and no two items of a launch overlap.
//
// The item list holds the keys; the host zeroes it on the stream right after
// the launch. No key or keystream word is stored outside registers here: the
// keystream block is 16 scalars, the tail loop is unrolled with constant
// indices, and the build's resource report must show no stack frame
// (`inspect/kv_cipher.<arch>.res-usage`).
//
// Geometry: grid (X, Y), 256 threads. Block row y walks items y, y + Y, ...;
// within an item, thread t of block x takes 64-byte keystream blocks
// x * 256 + t, then strides by X * 256.

#include "eidola_kernel.cuh"

namespace {

constexpr uint32_t kThreads = 256;

struct EidolaKvCipherItem {
  uint64_t base;      // device address of the range (4-byte aligned)
  uint32_t words;     // 32-bit words in the range
  uint32_t counter;   // keystream block of the range's first word
  uint32_t key[8];    // little-endian key words
  uint32_t nonce[3];  // little-endian nonce words
  uint32_t reserved;  // zero
};
static_assert(sizeof(EidolaKvCipherItem) == 64, "item layout is fixed");

__device__ __forceinline__ uint32_t rotl(uint32_t x, uint32_t n) {
  return __funnelshift_l(x, x, n);
}

#define EIDOLA_QR(a, b, c, d) \
  a += b;                     \
  d = rotl(d ^ a, 16);        \
  c += d;                     \
  b = rotl(b ^ c, 12);        \
  a += b;                     \
  d = rotl(d ^ a, 8);         \
  c += d;                     \
  b = rotl(b ^ c, 7)

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads)
    eidola_kv_chacha20(const EidolaKvCipherItem* __restrict__ items, uint32_t n_items) {
  for (uint32_t it = blockIdx.y; it < n_items; it += gridDim.y) {
    const EidolaKvCipherItem* item = items + it;
    const uint32_t words = item->words;
    const uint32_t chunks = (words + 15u) / 16u;
    uint32_t* data = reinterpret_cast<uint32_t*>(item->base);
    const bool vec = (item->base & 15u) == 0;
    const uint32_t k0 = item->key[0], k1 = item->key[1], k2 = item->key[2], k3 = item->key[3];
    const uint32_t k4 = item->key[4], k5 = item->key[5], k6 = item->key[6], k7 = item->key[7];
    const uint32_t n0 = item->nonce[0], n1 = item->nonce[1], n2 = item->nonce[2];
    const uint32_t counter = item->counter;
    for (uint32_t c = blockIdx.x * kThreads + threadIdx.x; c < chunks; c += gridDim.x * kThreads) {
      uint32_t x0 = 0x61707865u, x1 = 0x3320646eu, x2 = 0x79622d32u, x3 = 0x6b206574u;
      uint32_t x4 = k0, x5 = k1, x6 = k2, x7 = k3, x8 = k4, x9 = k5, x10 = k6, x11 = k7;
      uint32_t x12 = counter + c, x13 = n0, x14 = n1, x15 = n2;
#pragma unroll
      for (int round = 0; round < 10; ++round) {
        EIDOLA_QR(x0, x4, x8, x12);
        EIDOLA_QR(x1, x5, x9, x13);
        EIDOLA_QR(x2, x6, x10, x14);
        EIDOLA_QR(x3, x7, x11, x15);
        EIDOLA_QR(x0, x5, x10, x15);
        EIDOLA_QR(x1, x6, x11, x12);
        EIDOLA_QR(x2, x7, x8, x13);
        EIDOLA_QR(x3, x4, x9, x14);
      }
      const uint32_t ks[16] = {
          x0 + 0x61707865u, x1 + 0x3320646eu, x2 + 0x79622d32u, x3 + 0x6b206574u,
          x4 + k0,          x5 + k1,          x6 + k2,          x7 + k3,
          x8 + k4,          x9 + k5,          x10 + k6,         x11 + k7,
          x12 + counter + c, x13 + n0,        x14 + n1,         x15 + n2};
      const uint32_t first = c * 16u;
      const uint32_t n = min(16u, words - first);
      if (vec && n == 16u) {
        uint4* v = reinterpret_cast<uint4*>(data + first);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint4 w = v[q];
          w.x ^= ks[4 * q];
          w.y ^= ks[4 * q + 1];
          w.z ^= ks[4 * q + 2];
          w.w ^= ks[4 * q + 3];
          v[q] = w;
        }
      } else {
#pragma unroll
        for (uint32_t j = 0; j < 16u; ++j) {
          if (j < n) data[first + j] ^= ks[j];
        }
      }
    }
  }
}
EIDOLA_KERNEL_META(eidola_kv_chacha20, kThreads, 1, 1, 0, 1, 1, 1, 0);
