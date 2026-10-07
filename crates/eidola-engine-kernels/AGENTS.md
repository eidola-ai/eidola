# eidola-engine-kernels — Agent Guide

The engine's GPU kernels, compiled ahead of time from open source on a GPU-less Linux builder, and the manifest that pins their bytes. Targets are NVIDIA Blackwell datacenter parts: `sm_100a` (B200), `sm_103a` (B300), and the `sm_100f` family target (runs on any CC 10.x part). Workspace context: root `AGENTS.md`.

## Layout

| Path | What |
|---|---|
| `csrc/*.cu`, `csrc/eidola_kernel.cuh` | Our translation units: one per kernel family, each instantiating upstream templates (or our own code) and exposing entry points |
| `csrc/kernels.json` | The build matrix: kernels, target archs, flag profiles. The single place a kernel is added |
| `nix/default.nix` | The derivation: pinned nixpkgs, CUDA 13.2, unfree allow-list, inputs → `$out/{cubin,fatbin,inspect,manifest.json}` |
| `nix/sources.nix` | Upstream pins (commit + NAR hash) |
| `nix/build-kernels.sh` | Compiles, bundles, inspects, writes `manifest.json`; also runnable by hand inside `nix-shell nix` |
| `nix/render-flashinfer-sink.sh` | Renders FlashInfer's FA2 config template + attention-sink variant without Python |
| `scripts/build-in-docker.sh` | The builder: Nix in a pinned `nixos/nix` container; `--check`, `--verify`, `--update` |
| `kernels.manifest.json` | The committed build manifest, compiled into the crate |
| `src/lib.rs` | `Manifest` (parse/validate), `ArtifactDir` (verified loading), `KernelMeta` (launch contract decode) |

## The closed-tool boundary

Every kernel comes from source we fetch by hash and compile ourselves: **CUTLASS** (BSD-3-Clause), **DeepGEMM** device headers (MIT), **FlashInfer** headers and templates (Apache-2.0), and our own CUDA. Nothing upstream is committed here; the FlashInfer prologue is rendered from the pinned tree at build time, so no third-party text enters this directory.

The only closed software in the build is NVIDIA's CUDA 13.2 toolkit, at build time only: `nvcc`, `cudafe++`, `cicc` (libnvvm), `ptxas`, `fatbinary`, plus `cuobjdump`/`nvdisasm`/`cu++filt` for inspection. `nix/default.nix` admits exactly those components (and the cudart/crt/curand/cccl headers) through `allowUnfreePredicate`; adding any other NVIDIA package must be an explicit edit to that list.

Never, in this crate or anything it feeds:

- **trtllm-gen or any other prebuilt cubin.** We cannot rebuild them, so we cannot pin them to source.
- **Runtime compilation.** No nvrtc, no driver PTX JIT, no DeepGEMM/FlashInfer JIT. Images are SASS for the exact target; there is deliberately no PTX in them, so the driver has nothing to JIT.
- **The CuTe Python DSL** (closed compiler) or Python anywhere in the build.

## Toolchain pins

- **nixpkgs** `78e9c786…` — the root flake's revision, so one nixpkgs serves the repository.
- **CUDA 13.2.51** (`cudaPackages_13_2`). `sm_100a`/`sm_103a`/`sm_100f` need ≥ 12.9; CUTLASS 4.8 and current DeepGEMM/FlashInfer are developed on 13.x; 13.3 is the newest in this nixpkgs but its CCCL is marked unsupported there. Moving the toolkit moves every byte: rebuild and `--update` in the same change.
- **Host compiler** gcc 15.3.0 (`cudaPackages.backendStdenv`). nvcc uses it to preprocess device code, and `cicc` is told its version; swapping in gcc 14 produced byte-identical cubins, but the version stays pinned and recorded.

## Kernel set

| Kernel | Source | Entries | Tensor-core path (SASS) |
|---|---|---|---|
| Dense FP8 GEMM, f32 scales 1×128 (activations) / 128×128 (weights), BF16 out | CUTLASS 4.8 `KernelScheduleSm100Blockwise` | `eidola_cutlass_fp8_blockwise_gemm_bf16` | tcgen05 (`UTCQMMA`) |
| Routed-expert grouped GEMM, FP8 × MXFP4, masked (decode) and contiguous (prefill), gate/up and down | DeepGEMM `sm100_fp8_fp4_gemm_1d1d` | 4 mangled `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<…>` | tcgen05 (`UTCQMMA`, 2-CTA) |
| Paged attention, learned sinks, sliding window, head dims 192/128, BF16; query tiles 16/64/128 + split-KV merge | FlashInfer FA2 prefill template + `AttentionSink` | `eidola_fa2_sink_paged_bf16_q{16,64,128}`, mangled `PersistentVariableLengthMergeStatesKernel<…>` | `mma.sync` (`HMMA`), no tcgen05 |
| Sampling: top-k, top-p, top-k+top-p, chain speculative | FlashInfer `sampling.cuh` | 4 mangled `flashinfer::sampling::…` | none |
| RMSNorm, BF16 | ours | `eidola_rmsnorm_bf16` | none |

Every kernel is built for all three targets; each fatbin bundles the `sm_100a` and `sm_103a` cubins so the driver picks the exact match. Instantiation choices (tile shapes, stages, thread counts) mirror what the upstream host dispatchers pick for these shapes; each source file explains its own.

`tests/manifest.rs` holds the SASS facts above as assertions on the manifest, so a source or toolkit change that silently drops a GEMM off tcgen05 fails `cargo test`.

## How a host uses the images

Cubins and fatbins are loaded through the CUDA driver API (`cuModuleLoadData`, e.g. via `cudarc`); there is no host-side object code and no CUDA runtime library involved.

- **Load through `ArtifactDir`.** It reads an image from a build output and returns the bytes only if size and SHA-256 match the compiled-in manifest.
- **Entries by symbol.** Kernels we wrap are `extern "C"`. Upstream template kernels keep their mangled names, which the manifest lists with their demangled forms. Template instances, DeepGEMM's `__global__ static` kernels and every `_meta` global have local binding (`STB_LOCAL`) in a whole-program cubin. The CUDA runtime resolves exactly such symbols by name for ordinary programs, so `cuModuleGetFunction` / `cuModuleGetGlobal` should accept them. That is inferred, not yet observed on a GPU; the fallback is `cuModuleEnumerateFunctions`.
- **Launch contract in the image.** Each entry has a `<entry>_meta` device global (`EidolaKernelMeta`, 32 bytes) holding:
  - block shape,
  - dynamic shared memory,
  - cluster shape,
  - the size of its by-value parameter struct.

  Read it with `cuModuleGetGlobal` plus a 32-byte copy, and decode it with `KernelMeta::from_bytes`. Launches above 48 KiB of dynamic shared memory must first set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. Cluster launches need `cuLaunchKernelEx` with the cluster attribute.
- **Parameters.** These are what the host must build to call each kernel:
  - **FlashInfer kernels:** a plain `PagedParams` struct (296 bytes) or plain argument lists.
  - **DeepGEMM:** five `CUtensorMap`s, which the host encodes with `cuTensorMapEncodeTiled`.
  - **CUTLASS GEMM:** its 2048-byte `Params`, which includes TMA descriptors that CUTLASS builds in host C++. Providing that, either as a host-only C++ shim compiled without device code or as a Rust mirror checked against `params_bytes`, is the first job when the launch path is written.
- **DeepGEMM SM count.** The persistent scheduler bakes in the SM count (148). The grid must be exactly that, and the part must have at least that many SMs; confirm the B300 count before relying on the `sm_103a` instance.

## Determinism doctrine

The kernel bytes are a function of the inputs listed in the manifest: the sources, the flags, and the toolchain. They do not depend on the build directory, parallelism, clock, `SOURCE_DATE_EPOCH`, or the builder's CPU architecture. Two runs of an aarch64 builder produce identical manifests, and so do an aarch64 and an x86_64 builder (emulated on Apple Silicon). The rules that make this hold, each learned by breaking it:

- **Pin the source path nvcc sees.** `cicc` names internal-linkage device symbols (`_INTERNAL_<hash>_…`) after a module ID derived from the absolute source path.
  - The symptom: the same source in two directories gives different cubins for every kernel that has such symbols. Only RMSNorm was unaffected.
  - Why it matters here: Nix 2.32 builds in a randomized `/nix/var/nix/builds/nix-<pid>-<rand>` directory, so without the pin the build is not reproducible even run to run.
  - The fix: every compile passes `-Xcicc --orig_src_path_name -Xcicc crates/eidola-engine-kernels/csrc/<file>`. This is an undocumented `cicc` option; nvcc's own `-frandom-seed` does not affect it.
- **No `-lineinfo`, no `-G`.** Both embed source paths, and builds from two directories then differ.
- **No `-split-compile`.** With it, three compiles of the same file in the same directory produced two different `.text` sizes. This is code-level nondeterminism, not metadata.
- **One arch per compile.** CUB's ABI namespace embeds the list of target archs in the translation unit (`cub::_V_300200_SM_1000…`). The same source compiled for `sm_100a` alone and for `sm_100a`+`sm_103a` together therefore yields different `sm_100a` cubins. Per-arch compiles keep each cubin independent of the target set.
- **Explicit fatbin settings.** The nixpkgs nvcc setup hook would prepend `-Xfatbin=-compress-all`. The derivation disables the hook (`dontSetupCUDAToolkitCompilers`), and the build script runs `fatbinary --compress=false` directly. The compressor itself was observed to be deterministic, but uncompressed images let a reader check that the fatbin holds the cubins byte for byte.
- **No ambient flags.** The build script unsets the following before calling nvcc, so nothing reaches nvcc or its preprocessor that the manifest does not list:
  - `NVCC_PREPEND_FLAGS`
  - `NVCC_APPEND_FLAGS`
  - `NIX_CFLAGS_COMPILE`
  - the hardening variables
- **`-DNDEBUG` everywhere.** Device `assert` would embed `__FILE__` strings.

Each rule has a control experiment: build with the rule broken, from two directories, and compare. Keep that discipline when changing flags.

## Building and verifying

```sh
crates/eidola-engine-kernels/scripts/build-in-docker.sh --verify          # build, compare to kernels.manifest.json
crates/eidola-engine-kernels/scripts/build-in-docker.sh --check --verify  # also rebuild and require identical bytes
crates/eidola-engine-kernels/scripts/build-in-docker.sh --platform linux/amd64 --verify  # the other builder arch
crates/eidola-engine-kernels/scripts/build-in-docker.sh --update          # after an intended change
```

The first run downloads the toolkit into a named Docker volume (`eidola-kernels-nix-<arch>`). The x86_64 toolkit is also unpacked and patched locally, because unfree packages are not in the public cache. After that, a build takes under a minute natively. Output lands in `target/engine-kernels/<arch>/`:

- the cubins and fatbins,
- full SASS and register/shared-memory usage under `inspect/`,
- the rendered FlashInfer config,
- `manifest.json`.

Inside the container Nix runs with `sandbox = false` and `filter-syscalls = false`. The container is the isolation boundary, Nix's sandbox needs privileges it lacks, and the seccomp filter cannot load under amd64 emulation. Purity still holds: every input is fixed by hash, and the build script removes the build directory from the output.

`cargo test -p eidola-engine-kernels` needs no toolkit. It checks that the committed manifest still describes the committed sources:

- every `csrc/` file and build script hashes as recorded, and no new file has appeared;
- the pins agree with `nix/sources.nix`;
- the kernel × arch matrix and flags agree with `kernels.json`;
- the SASS facts above hold.

Editing a kernel without rebuilding fails here.

## Adding a kernel

1. Write `csrc/<name>.cu`. Give each entry point an `extern "C"` wrapper where the upstream code has a `__device__` body to call, or explicitly instantiate the upstream `__global__` template with its full signature. Add an `EIDOLA_KERNEL_META` record for each entry. Do not use `assert`-dependent or path-dependent constructs.
2. Add it to `csrc/kernels.json` with a flag profile. Add a new profile only when the upstream project's own build flags differ.
3. Run `scripts/build-in-docker.sh --check --update`, inspect `inspect/<name>.<arch>.sass` / `.res-usage` (register spills show as `STACK`), and commit the source and manifest together.
4. If the kernel exists for a particular tensor-core path, assert it in `tests/manifest.rs`.

## Repository integration (not yet wired)

The derivation is self-contained so the root flake can adopt it unchanged:

```nix
engineKernels = import ./crates/eidola-engine-kernels/nix {
  nixpkgsSrc = nixpkgs;
  nixpkgsRev = nixpkgs.rev;
  system = "x86_64-linux";
};
```

Any import must pass the root's own nixpkgs revision, or the manifest's toolchain record changes. A CI job would run the build on a native x86_64 Linux runner and `cmp` its `manifest.json` against `kernels.manifest.json`. That run also closes the gap left by building x86_64 only under emulation so far.
