// Dev-only oracle for the host-side CUTLASS launch parameters. Never part of
// the engine: it is compiled on a GPU development host with any CUDA 13
// toolkit and the pinned CUTLASS tree, and it includes the kernels crate's own
// translation units so its types are exactly the AOT kernels' types.
//
// For a list of problem shapes and fixed fake device addresses it runs
// CUTLASS's own host path (`GemmKernel::to_underlying_arguments`,
// `get_grid_shape`, `can_implement`, `get_workspace_size`) and prints, one
// JSON object per line: the `Params` bytes, the grid, the workspace size, and
// every `cuTensorMapEncodeTiled` call it made. The Rust mirror in
// `src/gemm.rs` must reproduce those bytes exactly (`tests/cutlass_params.rs`
// compares against the committed output in `tests/data/`).
//
//   nvcc -std=c++17 -O1 --expt-relaxed-constexpr -arch=sm_100a -DNDEBUG \
//     -I<cutlass>/include -I<cutlass>/tools/util/include -I<repo>/crates/eidola-engine-kernels/csrc \
//     crates/eidola-engine-cuda/oracle/cutlass_params.cu -lcuda -ldl -o cutlass-params
//   ./cutlass-params > crates/eidola-engine-cuda/tests/data/cutlass_params.jsonl

#include <cuda.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// Route CUTLASS's descriptor encoding through a logger that forwards to the
// driver.
static std::vector<std::string> g_encodes;
CUresult oracle_encode(CUtensorMap* map, CUtensorMapDataType dtype, cuuint32_t rank, void* addr,
                       const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                       const cuuint32_t* elem_strides, CUtensorMapInterleave interleave,
                       CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2,
                       CUtensorMapFloatOOBfill oob);
#define CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 1
#define cuTensorMapEncodeTiled oracle_encode

#include "cutlass_bf16_gemm.cu"
#include "cutlass_fp8_blockwise_gemm.cu"

#undef cuTensorMapEncodeTiled

#include <cutlass/util/packed_stride.hpp>

CUresult oracle_encode(CUtensorMap* map, CUtensorMapDataType dtype, cuuint32_t rank, void* addr,
                       const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                       const cuuint32_t* elem_strides, CUtensorMapInterleave interleave,
                       CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2,
                       CUtensorMapFloatOOBfill oob) {
  static char buf[1024];
  int n = snprintf(buf, sizeof buf, "{\"dtype\":%d,\"rank\":%u,\"addr\":%llu,\"dims\":[", (int)dtype,
                   rank, (unsigned long long)(uintptr_t)addr);
  for (cuuint32_t i = 0; i < rank; ++i)
    n += snprintf(buf + n, sizeof buf - n, "%s%llu", i ? "," : "", (unsigned long long)dims[i]);
  n += snprintf(buf + n, sizeof buf - n, "],\"strides\":[");
  for (cuuint32_t i = 0; i + 1 < rank; ++i)
    n += snprintf(buf + n, sizeof buf - n, "%s%llu", i ? "," : "", (unsigned long long)strides[i]);
  n += snprintf(buf + n, sizeof buf - n, "],\"box\":[");
  for (cuuint32_t i = 0; i < rank; ++i)
    n += snprintf(buf + n, sizeof buf - n, "%s%u", i ? "," : "", box[i]);
  n += snprintf(buf + n, sizeof buf - n, "],\"elem_strides\":[");
  for (cuuint32_t i = 0; i < rank; ++i)
    n += snprintf(buf + n, sizeof buf - n, "%s%u", i ? "," : "", elem_strides[i]);
  snprintf(buf + n, sizeof buf - n, "],\"interleave\":%d,\"swizzle\":%d,\"l2\":%d,\"oob\":%d}",
           (int)interleave, (int)swizzle, (int)l2, (int)oob);
  g_encodes.push_back(buf);
  return cuTensorMapEncodeTiled(map, dtype, rank, addr, dims, strides, box, elem_strides, interleave,
                                swizzle, l2, oob);
}

template <class T>
static std::string hex(const T& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
  for (size_t i = 0; i < sizeof(T); ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

static std::string encodes() {
  std::string s = "[";
  for (size_t i = 0; i < g_encodes.size(); ++i) s += (i ? "," : "") + g_encodes[i];
  g_encodes.clear();
  return s + "]";
}

constexpr uint64_t kA = 0x7f0000000000ull, kB = 0x7f1000000000ull, kD = 0x7f2000000000ull,
                   kSFA = 0x7f3000000000ull, kSFB = 0x7f4000000000ull;

// Padding inside Params is never written by CUTLASS, so its bytes are whatever
// the stack held. Params is built three times under different stack contents
// (plain, after scribbling a pattern, and deeper in the stack after scribbling
// another); bytes that agree in all three are the defined ones, and only
// those are compared.
__attribute__((noinline)) static void scribble(unsigned char v) {
  volatile unsigned char junk[1 << 16];
  for (size_t i = 0; i < sizeof junk; ++i) junk[i] = v;
}

template <class Kernel>
__attribute__((noinline)) static typename Kernel::Params build(typename Kernel::Arguments const& args,
                                                               int depth) {
  volatile unsigned char pad[4096];
  pad[0] = static_cast<unsigned char>(depth);
  if (depth > 0) return build<Kernel>(args, depth - 1);
  (void)pad[0];
  return Kernel::to_underlying_arguments(args, nullptr);
}

// Build Params into storage pre-filled with `fill` (guaranteed copy elision
// constructs the result in place, so bytes CUTLASS never writes keep the
// fill), after scribbling `scratch` over the stack below.
template <class Kernel>
static void build_into(unsigned char* storage, unsigned char fill, unsigned char scratch, int depth,
                       typename Kernel::Arguments const& args) {
  memset(storage, fill, sizeof(typename Kernel::Params));
  scribble(scratch);
  new (storage) typename Kernel::Params(build<Kernel>(args, depth));
}

template <class Kernel>
static void emit(const char* name, int m, int n, int k, double alpha,
                 typename Kernel::Arguments const& args) {
  using Params = typename Kernel::Params;
  const size_t ws = Kernel::get_workspace_size(args);
  const bool ok = Kernel::can_implement(args);
  alignas(Params) static unsigned char s1[sizeof(Params)], s2[sizeof(Params)], s3[sizeof(Params)];
  build_into<Kernel>(s1, 0x00, 0x11, 0, args);
  const std::string enc = encodes();
  build_into<Kernel>(s2, 0xff, 0x5a, 0, args);
  build_into<Kernel>(s3, 0x77, 0xa5, 7, args);
  g_encodes.clear();
  Params const& params = *reinterpret_cast<Params const*>(s1);
  std::string mask;
  for (size_t i = 0; i < sizeof(Params); ++i) mask += (s1[i] == s2[i] && s1[i] == s3[i]) ? '1' : '0';
  const dim3 grid = Kernel::get_grid_shape(params);
  const char* base = reinterpret_cast<const char*>(&params);
  printf("{\"kernel\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"alpha\":%.17g,\"can_implement\":%s,"
         "\"workspace\":%zu,\"grid\":[%u,%u,%u],\"params_bytes\":%zu,"
         "\"offsets\":{\"problem_shape\":%td,\"mainloop\":%td,\"epilogue\":%td,\"scheduler\":%td,"
         "\"hw_info\":%td,\"sizeof_mainloop\":%zu,\"sizeof_epilogue\":%zu,\"sizeof_scheduler\":%zu},"
         "\"params\":\"%s\",\"defined\":\"%s\",\"encodes\":%s}\n",
         name, m, n, k, alpha, ok ? "true" : "false", ws, grid.x, grid.y, grid.z, sizeof(params),
         reinterpret_cast<const char*>(&params.problem_shape) - base,
         reinterpret_cast<const char*>(&params.mainloop) - base,
         reinterpret_cast<const char*>(&params.epilogue) - base,
         reinterpret_cast<const char*>(&params.scheduler) - base,
         reinterpret_cast<const char*>(&params.hw_info) - base, sizeof(params.mainloop),
         sizeof(params.epilogue), sizeof(params.scheduler), hex(params).c_str(), mask.c_str(),
         enc.c_str());
}

static cutlass::KernelHardwareInfo hw() {
  cutlass::KernelHardwareInfo h;
  h.device_id = 0;
  h.sm_count = 148;
  return h;
}

static void fp8(int m, int n, int k) {
  using K = EidolaFp8BlockwiseGemm;
  using namespace eidola_fp8_blockwise;
  using StrideA = typename K::StrideA;
  using StrideB = typename K::StrideB;
  using StrideD = typename K::StrideD;
  using StrideC = typename K::StrideC;
  auto sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  auto sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  auto lsfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto lsfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename K::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<ElementA*>(kA), sa, reinterpret_cast<ElementB*>(kB), sb,
       reinterpret_cast<float*>(kSFA), lsfa, reinterpret_cast<float*>(kSFB), lsfb},
      {{}, nullptr, sc, reinterpret_cast<ElementD*>(kD), sd},
      hw()};
  emit<K>("fp8_blockwise", m, n, k, 1.0, args);
}

static void bf16(int m, int n, int k, float alpha) {
  using K = EidolaBf16Gemm;
  using namespace eidola_bf16_gemm;
  using StrideA = typename K::StrideA;
  using StrideB = typename K::StrideB;
  using StrideD = typename K::StrideD;
  using StrideC = typename K::StrideC;
  auto sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  auto sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  typename K::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<ElementA*>(kA), sa, reinterpret_cast<ElementB*>(kB), sb},
      {{alpha, 0.0f}, nullptr, sc, reinterpret_cast<ElementD*>(kD), sd},
      hw()};
  emit<K>("bf16", m, n, k, alpha, args);
}

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    fprintf(stderr, "cuInit failed\n");
    return 1;
  }
  // Descriptor encoding needs a current context.
  CUdevice dev;
  CUcontext ctx;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    fprintf(stderr, "no device context\n");
    return 1;
  }
  for (auto [m, n, k] : std::vector<std::tuple<int, int, int>>{
           {1, 13824, 4096}, {37, 14848, 4096}, {300, 32768, 4096}, {5, 4096, 16384},
           {128, 128, 128}, {1000, 4096, 2048}, {4, 256, 512}, {2048, 13824, 4096}})
    fp8(m, n, k);
  for (auto [m, n, k, a] : std::vector<std::tuple<int, int, int, float>>{
           {1, 4096, 8192, 0.707f}, {37, 4096, 8192, 1.0f}, {300, 152576, 4096, 1.0f},
           {4, 4096, 8192, 0.5f}, {1, 256, 4096, 1.0f}, {2048, 4096, 8192, 0.707f},
           {64, 128, 64, 1.0f}})
    bf16(m, n, k, a);
  return 0;
}
