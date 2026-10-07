# eidola-engine-cuda

The CUDA executor for the MiMo-V2.6 engine. It runs the kernels `eidola-engine-kernels` builds ahead of time, through the CUDA driver API, and is held to the CPU reference executor (`eidola-engine-cpu/AGENTS.md`) and the seam contract (`eidola-engine/src/executor.rs`). AGPL-3.0-only. Workspace conventions live in the root `AGENTS.md`.

| Module | Contents |
|---|---|
| `device.rs` | `Gpu` (primary context, one stream, `DeviceInfo`), `ImageArch` (which cubin a compute capability runs). |
| `module.rs` | `KernelModule` (loads manifest-verified images), `Kernel` (entry + launch contract, `cuLaunchKernelEx` with the contract's block, shared memory and cluster). |
| `ops.rs` | Typed launch wrappers for our own kernels. |
| `bf16.rs` | Host BF16 bit conversions (device BF16 tensors are `u16`). |

## Host boundary

- **Driver API only.** `cudarc` with `driver` + `dynamic-loading`, bindings pinned to CUDA 13.0 (`cuda-13000`, the R580 driver's API). No CUDA runtime, no NVRTC, nothing linked at build time: the crate builds and its tests run on a machine without CUDA. Do not enable `cudarc`'s `nvrtc` feature (its `load_module` path needs it; `module.rs` calls `cuModuleLoadData` directly instead).
- **Images only through `ArtifactDir`.** Bytes reach `cuModuleLoadData` only after their size and SHA-256 match the compiled-in kernel manifest.
- **Image choice.** CC 10.0 runs `sm_100a`, CC 10.3 runs `sm_103a`, any other 10.x runs `sm_100f`; nothing else runs. The images carry no PTX, so a foreign architecture's cubin is refused (`CUDA_ERROR_NO_BINARY_FOR_GPU`), never JIT-compiled; `tests/smoke.rs` asserts that.

## Measured on the B300 (driver 580.173.02, CUDA 13.0 driver API)

- `NVIDIA B300 SXM6 AC`, CC 10.3, **148 SMs**, 232,448 bytes opt-in shared memory per block, 287.4 GB device memory.
- The CUDA 13.2-built cubins load on the 13.0 driver: they are SASS-only, and the driver accepts them under minor-version compatibility.
- `sm_103a`, `sm_100f` and the fatbin all load and resolve every entry and launch-contract record, including DeepGEMM's mangled `__global__ static` instances and FlashInfer's merge kernel (`STB_LOCAL` symbols resolve through `cuModuleGetFunction` / `cuModuleGetGlobal`). `sm_100a` is refused.
- RMSNorm (BF16 in/out, f32 accumulation, `rsqrtf`) against the f32 reference rounded to BF16: bit-equal on 1×4096, 5×1000 and 3×256; 5 of 151,552 outputs one BF16 ulp off on 37×4096. Tolerance: one BF16 ulp.

## DeepGEMM SM count

DeepGEMM's persistent scheduler takes the SM count as a template argument and uses it as the grid stride, so an instance is correct only when launched with exactly that many blocks on a part with at least that many SMs. The built instances use 148, which equals the B300's count. `ops::deepgemm_grid` launches with the instance's count (`DEEPGEMM_INSTANCE_SMS`, checked against the template argument in every manifest symbol by a unit test) and refuses a device with fewer SMs; a part with more runs correctly on 148 of them. A part with a different count that needs every SM gets its own instance in the kernel build.

## Tests

`cargo test -p eidola-engine-cuda` runs everywhere. GPU tests need a CC 10.x device and `EIDOLA_ENGINE_KERNELS_DIR` pointing at a kernel build output (`target/engine-kernels/<arch>/` from `scripts/build-in-docker.sh`); without either they print why and pass.

`cargo run --release -p eidola-engine-cuda --example smoke -- <kernel build output>` prints the device report above.
