# Self-Hosted Inference

Eidola's long-term intent — named in [gaps.md § Inference upstream](gaps.md#inference-upstream) and [upstream.md](upstream.md) — is to end the inference-layer trust chain where the rest of Eidola's does: reproducible builds, measurements Eidola owns and statically pins, human-attested releases. This document describes the first concrete step: a **self-hosted inference container, co-located with the eidola-server inside the same measured enclave**, serving a modern Gemma model through the same OpenAI-compatible surface as the Tinfoil upstream.

## What runs where

```text
┌─ Tinfoil CVM (one SEV-SNP measurement covers everything below) ──────┐
│                                                                      │
│  shim (TLS, attestation)                                             │
│    └── eidola-server ──────────────┐                                 │
│          │ per-handshake attested  │ plain HTTP, CVM-internal        │
│          │ HTTPS (tinfoil-verifier)│ loopback (127.0.0.1:8081)       │
│          ▼                         ▼                                 │
│   inference.tinfoil.sh      eidola-inference                         │
│   (Tinfoil-hosted models)     ├── eidola-inference (boot supervisor) │
│                               │     fetch → verify SHA-256 → exec    │
│                               └── llama-server (llama.cpp, static)   │
└──────────────────────────────────────────────────────────────────────┘
```

The eidola-server routes each chat-completion request by model id: catalog models go to Tinfoil through the attesting client exactly as before; the self-hosted model goes to the co-located `eidola-inference` container over the CVM-internal loopback. There is no attestation between the server and the inference container **because there is no trust boundary between them** — both containers' image digests are declared in `tinfoil-config.yml`, whose SHA-256 is bound into the kernel command line and therefore into the single enclave measurement the client already pins and verifies on every handshake.

This matches the deployment scoping deliberately: one inference instance on the same machine. Scaling to load-balanced inference machines is a later iteration and brings real requirements this design intentionally excludes (server→inference attestation with a statically pinned measurement — the documented replacement for `upstream_trust` — plus server-to-server authentication).

## The trust chain extension

Nothing about the chain's *shape* changes; the measurement's *value* now covers more:

1. **Inference engine**: `oci/eidola-inference/Containerfile` builds llama.cpp from a checksum-pinned source tarball with the StageX toolchain into a static musl `llama-server`, plus the Rust boot supervisor — reproducible, `FROM scratch`, non-root, 25 MB. Its digest is recorded in `artifact-manifest.json` and stamped into `tinfoil-config.yml` by the same `just update-manifest` flow that stamps the server digest.
2. **Model weights**: `MODEL_URL` and `MODEL_SHA256` are environment variables of the inference container *declared in `tinfoil-config.yml`* — measured, not operator-supplied. At boot, the supervisor fetches the weights, hashes them **as they exist on disk** (the same bytes the engine will mmap), refuses to start the engine on mismatch, and only then execs `llama-server`. A compromised or swapped weight file cannot serve a single token.
3. **Model identity**: the model id/name/description/context the eidola-server advertises in `/v1/models` come from `EIDOLA_INFERENCE_*` env vars in the same measured config, so what clients see is bound to the same measurement as the weight hash it refers to.
4. **Client**: unchanged. The client's pinned `server-enclave.json` measurement now transitively covers the inference engine and weights. `verify-full` cross-checks all of it in CI.

The weights themselves are pinned to the **official Google QAT release** on Hugging Face (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf` in production; the E2B variant in dev), by immutable revision URL plus the Git-LFS SHA-256 content hash. The fetch transport doesn't need to be trusted — only the hash does.

## Why llama.cpp

Considered: llama.cpp, vLLM, SGLang, TensorRT-LLM, TGI, Ollama, and the Rust-native engines (mistral.rs, candle, burn). Decision criteria, in order: verifiability/reproducibility (StageX), maintenance burden, OpenAI-surface parity, Gemma 4 support, and a credible path to NVIDIA-CC GPU serving.

- **llama.cpp** is the only engine that clears the StageX bar today: CMake + a C/C++ toolchain and *zero* external dependencies (HTTP is vendored; curl and OpenSSL are disabled), building to a single static musl binary from a hash-pinned tarball. It has day-0 Gemma 4 support (Google publishes first-party QAT GGUFs), a mature OpenAI-compatible server (`/v1/chat/completions` with SSE + `stream_options.include_usage`, `/v1/models`), and per-commit release tags that pin one tag to one commit.
- **vLLM / SGLang** are the production standard for GPU serving — vLLM in particular is what Tinfoil runs for its per-model enclaves, including Gemma 4 31B — but they are ~170-package Python trees that no one (including Tinfoil) builds reproducibly; the honest integrity story there is digest-pinning an upstream image, which is a *lower* bar than the rest of Eidola's chain.
- **TensorRT-LLM** ships closed components and telemetry; fails the source-audit bar outright. **TGI** is archived. **Ollama** is a runtime-model-pull wrapper around llama.cpp's engine — the wrong shape for measured deployments.
- **mistral.rs** is the interesting Rust convergence candidate (Cargo.lock pins the whole tree, real OpenAI server, real Gemma 4 support, CUDA + Metal) but has no confidential-compute production record yet. Worth benchmarking under CC later; not the launch engine.

The performance trade is real and accepted for this iteration: llama.cpp is competitive at low concurrency and clearly behind vLLM at high concurrency (no PagedAttention-class batching). This instance is not expected to carry high-concurrency production load; it is the trust-chain proof.

## CPU now, GPU next

This iteration serves the model on CPU. That is not the end state, and the GPU path has hard constraints worth recording:

- **NVIDIA CC mode is transparent to the engine** — no inference-code changes; the taxes are at the encrypted CPU↔GPU boundary (bounce buffers) and in kernel-launch latency. Tinfoil's [confidential-computing-overhead post](https://tinfoil.sh/blog/2026-06-23-confidential-computing-overhead) measures ~10–25% end-to-end overhead for a single-GPU Gemma-4-31B-class enclave at moderate batch sizes, with CUDA graphs mandatory and speculative decoding needing re-profiling under CC.
- **A CUDA build can never be a StageX artifact.** `libcuda.so` is injected by the host driver stack at container start, is glibc-linked, and is proprietary — a static musl CUDA binary is structurally impossible, and the driver injection point sits outside any reproducible-image boundary. This is the same class of exception already documented for the Linux GUI (glibc + GPU stack, built via Nix instead of StageX — see [gaps.md § Build chain opacity](gaps.md#build-chain-opacity)). The GPU inference image will be a Nix-built glibc image (llama.cpp CUDA) or a digest-pinned upstream engine image (vLLM, Tinfoil-style), with the *weights and config* still measured exactly as here.
- **GPU attestation is additive, not architectural.** Tinfoil CVMs collect GPU evidence in measured boot code (NVIDIA attestation-sdk local verifier, SHA-pinned) and fold its hash into the same `REPORT_DATA` binding (`SHA-256(tls_key_fp ‖ hpke_key ‖ nonce ‖ gpu_hash ‖ nvswitch_hash)`) that `tinfoil-verifier` already recomputes — today those fields are empty for CPU-only enclaves; on GPU hardware they stop being empty. The client-side verification machinery needs no redesign.

## The boot supervisor (`crates/eidola-inference/`)

A ~300-line Rust binary, PID 1 of the inference container:

1. Reads `MODEL_URL` + `MODEL_SHA256` (+ optional `MODEL_PATH`, `INFERENCE_EXTRA_ARGS`) from the measured environment.
2. Reuses an existing on-disk file only if its hash matches (a warm dev volume; enclave boots are cold).
3. Otherwise streams the fetch to `MODEL_PATH.partial` (rustls + webpki roots; the image has no system trust store), then **re-reads the file from disk** and compares its SHA-256 — verifying the bytes the engine will actually mmap, not the bytes that happened to cross the network — and atomically renames it into place.
4. `exec`s the engine command given after `--` in its argv (fixed in the Containerfile `ENTRYPOINT`, hence measured via the image digest), substituting `{model}` with the verified path and appending `INFERENCE_EXTRA_ARGS` (measured via the config).

Any failure exits without exec'ing the engine: **no verified weights, no inference.** In production the weights land in RAM-backed container storage (the CVM has no persistent disk), so the enclave's memory budget must cover weights + KV cache + processes; `tinfoil-config.yml` sizes this.

## Server integration (`crates/eidola-server/src/backend.rs`)

`TinfoilBackend` became `InferenceBackend`: same `ChatBackend` trait, same catalog, plus an optional self-hosted upstream gated on `EIDOLA_INFERENCE_URL`. Routing is by model id; the self-hosted entry joins `/v1/models` with pricing from the same override mechanism (`TINFOIL_PRICING_OVERRIDES`, keyed by the self-hosted model id) over conservative defaults. Backend metadata reports `provider: "eidola"` and `TeeType::EidolaEnclave`, so the client-visible privacy metadata distinguishes the two upstreams honestly.

The server takes no fail-fast readiness dependency on the inference container: it boots concurrently and spends its first minutes fetching and verifying weights, so "not ready yet" is its normal boot state; requests routed to it fail with an ordinary upstream error until it comes up. (The Tinfoil upstream keeps its existing fail-fast smoke test.)

`src/upstream_trust` is untouched: Tinfoil-hosted models still resolve the router measurement at runtime. When inference is *fully* self-hosted (or split onto Eidola-controlled inference machines), that subsystem is deleted per its own module docs.

## Development workflow

```sh
just dev --inference        # full container stack + inference
just services --inference   # host-mode server + inference container
```

The dev model is the official Google QAT q4_0 GGUF of **Gemma 4 E2B** (~3.3 GiB), pinned by revision URL + SHA-256 in `compose.yaml` and cached in the `models` volume across restarts (the supervisor re-verifies the hash every boot). The server routes model id `gemma4-e2b` to it; everything else still goes to Tinfoil. Expect modest token rates locally: the container is linux/amd64 (x86-64-v3), so on Apple Silicon it runs CPU inference under Rosetta emulation — fine for exercising the full path, not for throughput.

To point dev at different weights, set `EIDOLA_INFERENCE_MODEL_URL` / `EIDOLA_INFERENCE_MODEL_SHA256` / `EIDOLA_INFERENCE_EXTRA_ARGS` in `.env` (the `--alias` must match the server's `EIDOLA_INFERENCE_MODEL`).

## Known limitations (this iteration)

- **CPU-only.** The production-shaped GPU/NVIDIA-CC deployment is the named next step (see above); this iteration proves the trust chain, not the throughput.
- **Boot-time fetch.** Weights are fetched from Hugging Face on every enclave boot (no persistent disks in Tinfoil CVMs yet): boot depends on HF availability, and the weights live in RAM. The stronger future shape is Tinfoil's `modelwrap` pattern — weights baked into a dm-verity read-only image referenced from the measured config — which makes verification continuous and free instead of a boot-time pass, and removes the boot-time network dependency. Eidola-controlled mirrors are the cheap intermediate step.
- **Single-file GGUF only.** Multi-part weight files (needed for much larger models) are a supervisor extension, not a design change.
- **Placeholder pricing.** Self-hosted cost is fixed infrastructure, not a per-token upstream invoice; defaults are set conservatively and can be overridden until real utilization data exists.
- **Model id ↔ alias coupling is manual.** `EIDOLA_INFERENCE_MODEL` (server) and `--alias` (engine) must agree; both live side-by-side in `tinfoil-config.yml`, but nothing machine-checks them yet.
- **Same-CVM loopback is assumed.** Containers in one Tinfoil CVM are expected to share a network namespace (`127.0.0.1`); if a future cvmimage isolates them, the URL in the measured config changes accordingly.
