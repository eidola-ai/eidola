# eidola-engine-kernels — Agent Guide

The engine's GPU kernels, compiled ahead of time from open source on a GPU-less Linux builder, and the manifest that pins their bytes. Targets are NVIDIA Blackwell datacenter parts: `sm_100a` (B200), `sm_103a` (B300), and the `sm_100f` family target (runs on any CC 10.x part). Workspace context: root `AGENTS.md`.

## Layout

| Path | What |
|---|---|
| `csrc/*.cu`, `csrc/eidola_kernel.cuh` | Our translation units: one per kernel family, each instantiating upstream templates (or our own code) and exposing entry points |
| `csrc/engine_ops_common.cuh` | Helpers `engine_ops.cu` and `engine_ops_reference.cu` share (conversions, warp reductions, the UE8M0 recipe, the packed scale layout, the fused-QKV parameter struct) |
| `csrc/kernels.json` | The build matrix: kernels, target archs, flag profiles, and the `meta_aliases` binding mangled entries to their launch-contract records. The single place a kernel is added |
| `nix/default.nix` | The derivation: pinned nixpkgs, CUDA 13.2, unfree allow-list, inputs → `$out/{cubin,fatbin,inspect,manifest.json}` |
| `nix/sources.nix` | Upstream pins (commit + NAR hash) |
| `nix/build-kernels.sh` | Compiles, bundles, inspects, writes `manifest.json`; also runnable by hand inside `nix-shell nix` |
| `nix/render-flashinfer-sink.sh` | Renders FlashInfer's FA2 config template + attention-sink variant without Python |
| `scripts/build-in-docker.sh` | The builder: Nix in a pinned `nixos/nix` container; `--check`, `--verify`, `--update` |
| `kernels.manifest.json` | The committed build manifest, compiled into the crate |
| `src/lib.rs` | `Manifest` (parse/validate), `Cubin::entry` / `Cubin::entry_for_meta` (entry ↔ record lookups), `ArtifactDir` (verified loading), `KernelMeta` (launch contract decode) |

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
| Dense BF16 GEMM, f32 out, runtime `alpha` (`o_proj`, `eh_proj`, `lm_head`) | CUTLASS 4.8 collective-builder default schedule | `eidola_cutlass_bf16_gemm_f32` | tcgen05 (`UTCHMMA`) |
| Routed-expert grouped GEMM, FP8 × MXFP4, in DeepGEMM's psum layout (`MGroupedContiguousWithPsumLayout`) at every token count, gate/up and down | DeepGEMM `sm100_fp8_fp4_gemm_1d1d` | 2 mangled `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<…>` | tcgen05 (`UTCQMMA`, 2-CTA) |
| Paged attention, learned sinks, sliding window, head dims 192/128, BF16; query tiles 16/64/128 + split-KV merge | FlashInfer FA2 prefill template + `AttentionSink` | `eidola_fa2_sink_paged_bf16_q{16,64,128}`, mangled `PersistentVariableLengthMergeStatesKernel<…>` | `mma.sync` (`HMMA`), no tcgen05 |
| Sampling (filters, inverse-CDF draw) and chain speculative acceptance | ours (`sampling.cu`) | `eidola_sample`, `eidola_chain_accept` | none |
| The executor's glue: embedding, RMSNorm, FP8 activation quantization (f32-scale and UE8M0 recipes), fused-QKV RoPE + paged KV write, SwiGLU + quantization, router top-k (a one-cluster form and a two-launch form, the executor picking by token count), expert placement, gather, combine, bounded row copies | ours (`engine_ops.cu`) | 16 `eidola_*` entries | none |
| Reference forms of the router, the f32-scale activation quantization, both SwiGLUs (f32 and UE8M0 scales), the UE8M0 gather, the fused-QKV RoPE + KV write and the combine, for the GPU tests and the kernel bench only (the executor never loads them) | ours (`engine_ops_reference.cu`) | `eidola_reference_{router_topk,quant_fp8_f32scale,swiglu_quant_fp8_f32scale,swiglu_quant_fp8_ue8m0,gather_quant_ue8m0,qkv_rope_kv,moe_combine}` | none |
| RMSNorm, BF16 | ours | `eidola_rmsnorm_bf16` | none |

**Sampling never comes from upstream.** The serving core (`eidola-engine/src/sampling.rs`) defines what a sample is: draws from a SplitMix-based counter RNG keyed by `(seed, position, stream)`, with separate streams for target samples, drafts, speculative acceptance and the residual, its own filtering, CDF walk and chain acceptance, and pinned `f64` arithmetic (its own `exp`, sums in a fixed chunked order). Recompute preemption and the CPU oracle depend on those exact draws. FlashInfer's sampling kernels draw from Philox with a caller-supplied seed and offset and cannot reproduce them. `sampling.cu` performs the core's operations in the core's grouping with explicit round-to-nearest intrinsics (nvcc never contracts those into FMAs; the `DFMA`s in its SASS belong to the correctly rounded division routine), so it returns the same probabilities and tokens bit for bit given equal logits; `eidola-engine-cuda`'s `tests/sampler.rs` checks every probability and token against the core on both device images.

**Draft ids are bounded on the device.** A drafted step's drafts are drawn by `eidola_sample` and read by the next forward's embedding and by `eidola_chain_accept` without passing through the host, so the two kernels bound them: `eidola_sample` writes no token id at or past the sampleable vocabulary `n` (none can arise: every argmax and draw it makes is below `n`; it would write 0 and raise `kStatusBadToken`, status bit 2), and `eidola_chain_accept` checks every draft id against `n` before reading a distribution through it (raising the bit and ending the row with no tokens). Acceptance reads draft `i` of row `r` at row `draft_row[r] + i × draft_step` of the draft distributions and of the ids: contiguous per row with a step of 1 (a host plan), or one depth's drafts of every row together with a step of the row count (the drafter's lanes). The arithmetic is unchanged.

Every kernel is built for all three targets; each fatbin bundles the `sm_100a` and `sm_103a` cubins so the driver picks the exact match. Instantiation choices (tile shapes, stages, thread counts) mirror what the upstream host dispatchers pick for these shapes; each source file explains its own.

### The decode path's small kernels

At one decode row these launch per layer, so their geometry is chosen for small token counts first; every launch's shape is a function of the token count alone (a decode graph's rung). The ones whose work grows with a prefill's tokens (the fused QKV, the f32-scale quantization, the gather, both SwiGLUs, the combine) move 16-byte vectors per thread, so a mixed or prefill step's hundreds to thousands of tokens stream at a large share of memory bandwidth rather than being bound by one scalar load chain per element. Flash: hidden 4096, 256 experts, top 8, expert intermediate 2048, dense intermediate 16,384; per attention layer 4 rank chunks of 16 query heads and 1 (global) or 2 (sliding) KV heads.

| Entry | What | Geometry |
|---|---|---|
| `eidola_qkv_rope_kv` | Partial NeoX RoPE of every Q and K head of the fused QKV output; Q out for attention, K and V into the paged pool | Grid (T, per-token elements / (8 × 256)), 256 threads: one thread per 8 consecutive output elements (a 16-byte BF16 vector), in the row's own order (per chunk its Q and K heads of 192, then its V heads of 128, all whole vectors); a rotated vector also loads its 8 partners 32 dims away and its cos/sin as 16-byte vectors |
| `eidola_router_scores` | Logits `x · Wᵀ` (BF16 weights, f32) and sigmoid scores of every (token, expert), with their choices (score + bias), into a `[2][T][E]` f32 scratch: the first of the executor's two router launches | Grid (⌈E/16⌉, ⌈T/8⌉), 128 threads, no cluster: a block takes 8 tokens × 16 experts, each warp 4 of the experts, each lane the 32 (token, expert) chains of its residue class. The tile's rows and the block's weight rows stream through dynamic shared memory 128 elements of the row at a time (`cp.async`, 8 stages of 8 KiB). After the butterflies each lane holds one chain's logit and computes one sigmoid |
| `eidola_router_select` | Top-k of each token's choices with ties to the lower id, ids ascending, renormalized weights: the second launch | Grid ⌈T/4⌉, 128 threads: one warp per token reads its 256 choices (8 a lane), selects with warp shuffles, orders the k picks with a sorting network in registers; lane 0 reads the k scores and writes the ids and weights |
| `eidola_router_topk` | The whole router (logits, scores, selection) in one cluster launch: the one-token form, which the executor runs at up to 15 tokens | One cluster of 8 blocks per token (grid `8T`, cluster attribute from the launch contract), 1024 threads: one warp per expert, the token's row staged in shared memory (hidden ≤ 4096). Rank 0's first warp reads the 256 choices through distributed shared memory and selects with warp shuffles |
| `eidola_quant_fp8_f32scale` | The dense path's activations (the QKV and dense gate/up GEMMs' input), FP8 with f32 scales | Grid (T, ⌈K/512⌉), 4 warps: a warp per 128-wide group, each lane 4 consecutive elements (a 16-byte load, a 4-byte store) |
| `eidola_swiglu_quant_fp8_f32scale` | The dense layers' (and MTP layers') SwiGLU, FP8 with f32 scales | As the expert SwiGLU below, over every row: items of 256 elements, `rows × I/256` of them |
| `eidola_moe_permute` | Expert-major placement: `row_of` per (token, slot) pair and the grouped layout (each expert's end row) | ⌈pairs/512⌉ blocks of 256 threads (at least one, at most 128), each over a contiguous run: per-warp shared-memory histograms; above one block the counts meet in a scratch buffer behind a grid barrier, each block reading every block's counts of each expert; a scan of the runs, then each warp places its eighth of the run in order, 32 at a time (`__match_any_sync` ranks) |
| `eidola_gather_quant_ue8m0` | Each pair's token row, FP8 with UE8M0 scales, into the row `row_of` names | Grid (T, K/512), 4 warps: one block per token and 512-wide word, each lane 4 consecutive elements (a 16-byte load); the word is quantized once and stored into the rows of all the token's `top_k` pairs |
| `eidola_swiglu_quant_fp8_ue8m0` | SwiGLU of each routed row of the gate/up GEMM output, FP8 with UE8M0 scales | Items of 256 elements of a routed pair's row (`pairs × I/256`), a warp each at a time: a half-warp per 128-wide group, each lane 8 consecutive elements (16-byte gate and up loads, an 8-byte store), each group's exponent stored as its byte of the scale word. At most 4,736 warps (blocks of 4, 8 blocks an SM on 148 SMs, all resident at once): each warp walks its share, loading the next item (and the row of the one after) before computing the current |
| `eidola_moe_combine` | Each token's routed down-projection rows, weighted, summed in ascending slot order | Grid (T, H/1024), 128 threads of 8 consecutive elements: each thread loads its piece of the token's 8 rows at once (16 bytes each), then runs each element's chain |

The executor picks the form by the step's token count (`eidola-engine-cuda`'s `engine_ops.rs`, `executor_router`; `ROUTER_PER_TOKEN_MAX`'s comment records the measurements behind the bound): the one-token form at up to 15 tokens, the two-launch form above. The one-token form puts a token's whole expert dimension on one 8-block cluster and selects on its rank 0, so each cluster reads the whole 2 MiB router weight for its token; on a B300 it costs 20.5–22.4 µs a layer at 1 to 15 tokens, 4–6 µs under the two-launch form, and then grows in steps once its clusters no longer all run at once (37.7–38.9 µs at 16, 93.3 at 64). The two-launch form spreads the expert dimension over 16 blocks per 8 tokens, without a cluster barrier, and selects with one warp per token; it costs 25.6–27.6 µs from 1 to 64 tokens.

**Why the SwiGLUs walk items.** The SwiGLU is the one glue kernel with real arithmetic per element: the reference's exact `expf` (a `MUFU.EX2` and seven FP32 instructions) and an IEEE division (`MUFU.RCP`, five `FFMA`s and an `FCHK` guarding a slow-path call) per element, the dense form a second division by the scale. A warp issues about 360 instructions per 256 elements; at 2,048 tokens the expert SwiGLU's 131,072 warp-items are about 47 M warp instructions, roughly 40 µs of issue on 148 SMs, more than the roughly 24 µs its 168 MB take at memory bandwidth. As one block per pair and 1024 elements, each block loaded its row index, then its gate and up, then computed, with nothing in flight while it did (and each element's division is its own call region, so the compiler could not overlap one element's exponential with the next's): issue and memory time added rather than overlapped, which is why moving it to 16-byte vectors gained only 1.2–1.5× where the memory-bound gather and fused QKV gained 2.6–3.7×. Now each warp issues the next item's loads, and the row index of the one after, before computing the current item, the whole grid is resident at once, and a lane computes its eight exponentials before its eight divisions. The arithmetic cannot shrink without changing bits, so issue time, not bandwidth, is the floor.

**The numerics are the reference forms'**, bit for bit; the geometry only moves where the arithmetic runs:

- **Router.** A logit is lane `l`'s sequential fused multiply-add over `i = l, l + 32, …` in ascending order, then the xor butterfly over the warp, then `1 / (1 + expf(-x))`; the selection takes, per round, the untaken expert a scan in expert order keeping the first strictly greater choice would take. Under the order (choice greater, or equal and lower id) that is the maximum over non-NaN choices, unless the lowest untaken expert's choice is NaN, in which case that expert; the warp computes exactly that, and the order is total, so the shuffle tree cannot change the result. The weight sum runs over the selection in ascending id order, as before. The two-launch form keeps every chain as it is: their lane `l` runs the same residue class in the same ascending order for each (token, expert) it owns (the chunks are consumed in order), the butterfly is the same `warp_sum`, lane 0's sum is the one used, and the sigmoid's SASS matches the one-token form's (the same `FFMA` contraction of `1 + expf(-x)`, then the reciprocal). Only which thread runs a chain and how its operands reach shared memory change. The two-launch form's selection runs the one-token form's rounds over the same choices, read from global memory instead of distributed shared memory; the picks are distinct, so the sorting network orders them as the insertion sort did, and the weight sum runs over them in that ascending order. The reference form ran 32 experts per warp on one block per token, its dot products bound by one load round trip per eight elements, and the selection on one thread.
- **Quantization, gather and SwiGLU** compute each row (each routed row, for the gather and the expert SwiGLU) exactly as the per-row reference does: each element with the same expressions, each group's amax a maximum (which no reduction order changes; NaN never wins `fmaxf`). The f32-scale forms keep the reference's IEEE division by the scale (`x / scale`, not a reciprocal), the SwiGLUs its `x / (1 + expf(-x)) * u` (computing a lane's eight denominators first only reorders independent work; the SASS keeps the same `+ 1` contraction into the exponential's last `FFMA`). The reference forms of the f32-scale quantization and dense SwiGLU are the kernels they replace, and compile to the same SASS. The gather quantizes a token's word once for all its pairs, which carry the same row. The gather and the expert SwiGLU do not touch the layout's padding rows: the reference forms launch over every row of the layout (every pair plus up to 127 padding rows per expert reached), zero-filling them in the gather. The grouped GEMMs compute each output row from its own A row and scales only, and the combine reads only the rows `row_of` names, so a padding row's contents reach nothing the executor uses.
- **Placement.** `eidola_moe_permute` has no arithmetic; its host form is `eidola-engine-cuda`'s `engine_ops::expert_placement` (pairs in order within an expert), which `tests/moe_ops.rs` compares with the kernel word for word. Blocks, warps and lanes cover the pairs in order, and each block's first row in an expert's run is that expert's count in the blocks before it, so the multi-block form places every pair where the one-block form did. Its grid barrier needs every block resident at once: at most 128 blocks of 256 threads and 10 KiB of shared memory, a small fraction of any part the executor accepts.
- **Combine.** Each element is `acc = 0`, then `acc = fma(w_j, d_j, acc)` for slots `j` ascending: the contraction the reference's `acc += w * d` compiles to (its SASS is that `FFMA` chain from `RZ`), written out. The reference ran one 256-thread block per token, each thread walking 16 elements with the slots' loads serialized behind each other.
- **Fused QKV.** Each output element is computed by one thread with the reference's expressions; the rotation is written as the contraction the reference compiles to (`fma(x1, cos, -(x2 · sin))` for the first half, `fma(x2, cos, x1 · sin)` for the second: the rounded product is the second one in both). The reference ran one 256-thread block per token, about 58 serial steps a thread, each step's loads ordered behind the previous step's stores.

**Row copies** (`eidola_copy_rows`): the drafted step's gathers and scatters of hidden states, drafter state, boundary taps and token ids. One launch copies items of `width` 32-bit words between up to eight buffers (a by-value `EidolaCopyArgs` of 176 bytes: the buffers' addresses and row counts, the four item arrays, the status word), item `i` row `src_row[i]` of buffer `src_buf[i]` to row `dst_row[i]` of buffer `dst_buf[i]`; every buffer and row index is bounded by the kernel against the buffer's rows, and an item out of range copies nothing and raises `kStatusBadIndex` (bit 4). Grid (items, ⌈width / 256⌉), a thread per word. Items must not overlap (the host plans them so). A copy has no arithmetic, so its reference form is the host gather itself: `eidola-engine-cuda`'s `tests/copy_rows.rs` compares word for word.

`eidola-engine-cuda`'s `tests/moe_ops.rs` checks the router (both forms), gather, SwiGLU and combine against the reference image bit for bit from 1 to 8,192 tokens (ties, NaN and infinite choices and rows, scale-range extremes), `tests/quant_ops.rs` the f32-scale quantization and dense SwiGLU, `tests/qkv_rope_kv.rs` checks the fused-QKV kernel's Q and every pool byte for both KV head counts, and the `moe_kernel_bench` example times each against its reference form, and the router in every form, per token count.

`tests/manifest.rs` holds the SASS facts above as assertions on the manifest, so a source or toolkit change that silently drops a GEMM off tcgen05 fails `cargo test`.

## How a host uses the images

Cubins and fatbins are loaded through the CUDA driver API (`cuModuleLoadData`, e.g. via `cudarc`); there is no host-side object code and no CUDA runtime library involved.

- **Load through `ArtifactDir`.** It reads an image from a build output and returns the bytes only if size and SHA-256 match the compiled-in manifest.
- **Entries by symbol.** Kernels we wrap are `extern "C"`. Upstream template kernels keep their mangled names, which the manifest lists with their demangled forms. Template instances, DeepGEMM's `__global__ static` kernels and every `_meta` global have local binding (`STB_LOCAL`) in a whole-program cubin. `cuModuleGetFunction` / `cuModuleGetGlobal` resolve them by name (observed on a B300, R580 driver; `eidola-engine-cuda`'s smoke tests cover every entry and record).
- **Launch contract in the image.** Each entry has a launch-contract device global (`EidolaKernelMeta`, 32 bytes), named by the entry's `meta` field in the manifest. For an entry we name ourselves it is `<entry>_meta`. A mangled template instance cannot carry a matching name, so its record has a readable alias (`eidola_deepgemm_fp8_fp4_psum_gate_up_meta`, …) and `meta_aliases` in `csrc/kernels.json` binds the alias to the mangled symbol. The build fails unless every entry resolves to exactly one record and every record to exactly one entry, `Manifest::parse` re-checks that correspondence, and `Cubin::entry` / `Cubin::entry_for_meta` look it up in either direction. The record holds:
  - block shape,
  - dynamic shared memory,
  - cluster shape,
  - the size of its by-value parameter struct.

  Read it with `cuModuleGetGlobal` plus a 32-byte copy, and decode it with `KernelMeta::from_bytes`. Launches above 48 KiB of dynamic shared memory must first set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. Cluster launches need `cuLaunchKernelEx` with the cluster attribute.
- **Parameters.** These are what the host must build to call each kernel:
  - **FlashInfer attention:** a plain `PagedParams` struct (296 bytes); the split-KV merge takes a plain argument list.
  - **DeepGEMM:** five `CUtensorMap`s, which the host encodes with `cuTensorMapEncodeTiled`.
  - **CUTLASS GEMMs:** their 2048-byte `GemmUniversal::Params`, including TMA descriptors that CUTLASS builds in host C++. `eidola-engine-cuda` builds the same bytes in Rust (`src/gemm.rs`), pinned against CUTLASS's own host path by a dev-only oracle.
- **DeepGEMM SM count.** The persistent scheduler bakes in the SM count (148). The grid must be exactly that, and the part must have at least that many SMs. The B300 has 148 (measured), the same as the B200.

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

- every file under `csrc/` and `nix/` (the sources and the whole build recipe: `default.nix`, `sources.nix` and both scripts, all in the derivation's source set) hashes as recorded, and no new file has appeared in either;
- the pins agree with `nix/sources.nix`;
- the kernel × arch matrix and flags agree with `kernels.json`;
- the SASS facts above hold;
- every entry is bound to exactly one launch-contract record, and the aliases are exactly the bindings that differ from `<entry>_meta`.

Editing a kernel or the build recipe without rebuilding fails here.

## Adding a kernel

1. Write `csrc/<name>.cu`. Give each entry point an `extern "C"` wrapper where the upstream code has a `__device__` body to call, or explicitly instantiate the upstream `__global__` template with its full signature. Add an `EIDOLA_KERNEL_META` record for each entry: `<entry>` for an `extern "C"` entry, a readable alias for a mangled one. Do not use `assert`-dependent or path-dependent constructs.
2. Add it to `csrc/kernels.json` with a flag profile, and a `meta_aliases` entry (`"<alias>_meta": "<mangled symbol>"`) for every aliased record; the first build prints the mangled symbols if you do not have them yet. Add a new profile only when the upstream project's own build flags differ.
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
