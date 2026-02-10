# On-Device Inference with Burn

## Context

The macOS app runs AI inference locally on the user's device. This requires
choosing an ML framework, deciding how to structure model implementations, and
solving the practical problem of GPU acceleration across platforms.

### Alternatives considered

- **llama.cpp (via bindings):** The dominant C++ inference engine. Excellent
  performance, huge model ecosystem (GGUF format), active development. However,
  it is a C++ library — linking it violates the pure-Rust constraint
  (see [Pure Rust, Zero C Dependencies](pure-rust-zero-c-dependencies.md)) and complicates
  cross-compilation. It also brings its own runtime, threading model, and
  memory management that would need to interoperate with the Rust application.

- **Candle (Hugging Face):** Pure Rust ML framework from Hugging Face.
  Mature, well-documented, with many pre-built model implementations. However,
  its Metal backend requires linking against Apple's Metal framework (C/ObjC
  dependency), and its model implementations are tightly coupled to the candle
  crate — harder to extract as standalone libraries.

- **ONNX Runtime (via bindings):** Cross-platform, well-optimized, but a C++
  library with complex build requirements. Same cross-compilation concerns as
  llama.cpp.

- **Burn:** Pure Rust ML framework with a pluggable backend system. WGPU
  backend provides GPU acceleration via Metal (macOS), Vulkan (Linux/Windows),
  and WebGPU (browser) — all through a single Rust crate. NdArray backend
  provides a CPU fallback. The entire stack compiles with `rustc` alone.

## Decision

Use **Burn** as the ML framework for on-device inference.

### Why Burn

1. **Pure Rust.** No C dependencies. Compiles to all targets in
   `rust-toolchain.toml` without a cross-compiler.
2. **Backend abstraction.** The same model code runs on WGPU (GPU) or
   NdArray (CPU) with no code changes. Adding new backends (e.g., WebGPU for
   WASM) requires no model changes.
3. **WGPU for GPU.** A single backend covers Metal (macOS/iOS), Vulkan
   (Linux/Android), and WebGPU (browser). No platform-specific GPU code.
4. **Autodiff support.** While not needed for inference today, Burn's autodiff
   backend enables future on-device fine-tuning and adaptation without
   switching frameworks.

### Model implementations as standalone crates

Each model architecture is implemented as its own crate (e.g., `qwen3-burn`),
separate from the application integration layer (`eidolons-perception`). This
separation provides:

- **Isolation:** Each model crate has its own tests and benchmarks, runnable
  without the full application stack.
- **Reusability:** Model crates depend only on Burn and standard libraries.
  They can be published independently or used in other projects.
- **Clear boundaries:** The model crate handles architecture, weight loading,
  and generation. The perception crate handles model downloading, backend
  selection, and application integration.

The perception crate (`eidolons-perception`) is the integration layer:
- Downloads models from HuggingFace Hub via `hf-hub`
- Detects model architecture from `config.json`
- Selects the appropriate backend (WGPU with NdArray fallback)
- Wraps model crates with the `CausalLM` trait for uniform generation

### GPU as a compile-time feature flag

GPU support (WGPU) is gated behind a `gpu` Cargo feature, enabled by default.
This exists because WGPU types are not `Send+Sync`, which creates a
fundamental incompatibility with UniFFI's requirements for FFI-exported types.

For FFI consumers (the `eidolons-shared` crate), GPU support is still used
but the model is owned by a dedicated inference thread that communicates via
channels. The perception crate is included with default features (GPU enabled),
and the thread boundary isolates the non-Send types.

The backend selection at runtime uses `catch_unwind` around WGPU initialization
to handle environments where GPU access fails (sandboxed apps, headless
servers, CI). Failure falls back silently to the NdArray CPU backend.

### SafeTensors as the only weight format

Only the SafeTensors format is supported for loading model weights. PyTorch
`.bin` files (which use Python's pickle and can execute arbitrary code during
deserialization) are explicitly rejected with a helpful error message.

SafeTensors was security-audited by Trail of Bits with no critical findings.
Combined with hash verification at load time
(see [Model Weight Management](model-weight-management.md)), this provides both
integrity and safety at the weight-loading boundary.

## Consequences

**Benefits:**

- The entire ML stack — framework, backends, model implementations — is pure
  Rust, maintaining the zero-C-dependency guarantee.
- GPU acceleration works across macOS (Metal), Linux (Vulkan), and eventually
  browsers (WebGPU) from the same code.
- Model crates are testable in isolation without GPU hardware (NdArray backend
  for unit tests).
- SafeTensors eliminates code execution risks during weight loading.
- The backend abstraction means performance improvements in Burn (new backends,
  kernel optimizations) benefit all models without code changes.

**Trade-offs we accept:**

- Burn is younger and less optimized than llama.cpp or ONNX Runtime.
  Inference performance will be lower, especially for quantized models where
  llama.cpp has heavily optimized kernels.
- No quantization support yet. Burn operates on full-precision (f32) or
  half-precision (f16/bf16) tensors. GGUF-style 4-bit quantization, which
  is critical for running large models on consumer hardware, is not available
  in Burn.
- The WGPU backend's performance depends on the graphics driver and may vary
  across hardware. Metal on Apple Silicon is well-supported; other platforms
  are less tested.
- Implementing model architectures from scratch in Burn is more work than
  using pre-built implementations from llama.cpp or candle. Each new
  architecture (Llama, Qwen3, etc.) requires a dedicated implementation
  effort.
- The `catch_unwind` fallback for GPU initialization is a blunt instrument.
  It catches panics, not errors, which may mask real bugs in GPU setup.

## Future Considerations

- **Quantization.** When Burn gains quantization support (or via a custom
  quantization layer), enabling 4-bit inference would dramatically expand the
  set of models that can run on consumer hardware.

- **On-device adaptation.** Burn's autodiff backend enables gradient
  computation, which opens the door to on-device LoRA fine-tuning or
  preference learning. The adaptation layers discussed in
  [Model Weight Management](model-weight-management.md) would be trained using this
  capability.

- **Additional model architectures.** New architectures follow the same
  pattern: standalone crate implementing the model on Burn, integrated via
  `eidolons-perception`. Candidates include Mistral, Phi, Gemma.

- **WebAssembly inference.** The pure-Rust stack and WGPU's WebGPU support
  make browser-based inference theoretically possible. This would require
  testing Burn's WGPU backend in a browser environment.
