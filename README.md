# Eidolons

We are building an app that functions as an AI extension of the user.

By solving privacy concerns at a foundational level (using embedded and self-hosted inference, zero knowledge services, risk-based data retention, etc) we can expand the data that the tool collects to pretty much everything, including continuous screen recording, audio recording, location data, streams of emails, photos, etc, as well as direct browser plugins.

Initially, we will use a familiar chat type user interface with a "spotlight"-like quick-access interface. However, the magic will come from predicted interactions.

### Initial Technologies
Due to the need for broad system access and heavy requirements for on-device processing, this will start as a native app. The core will be written in rust (following the crux architecture), with the initial interface in SwiftUI, targeting macOS.

Inference and on-device training will leverage the burn framework, and structured storage will initially use Turso.

Additional targets are expected, including a CLI, Linux, iOS, Android, and web, although these are not immediate goals.

## Security, Privacy, and Confidentiality

The very first iteration will exclusively use embedded models. It will be limited to inbound event streams, and models will have no ability to change data outside the app's own data store, nor the ability to call external tools. No cloud services will be integrated by default. An "off the record" mode will completely prevent persistence while active.

In subsequent near-term iterations, local models will continue to coordinate all requests and event processing. We may introduce realtime cloud services that the local agent can use to search the web or delegate complex tasks to hosted models. In such cases, local models will be responsible for "sandboxing" such external requests, making them generic and unlinking personal or sensitive data, and users will approve each call by default. Privacy Pass tokens or similar anonymizing authorization technologies will ensure user-verifiable unlinkability of requests to their payment account, and between requests. Privacy-preserving networking technologies like Tor hidden services or oHTTP will be used to provide user controllable freedom from linking requests by IP address. Optional zero-knowledge cloud storage and syncing may be made available, built on the same privacy preserving technology.

In a more distant iteration, we intend to explore per-user evolution of actual models, providing some form of "memory" or "tuning" that remains inherently confidential over the specific interactions.

With our first public launch, we will ensure our entire source code is publicly available (we may either use a "source available" license or full "open source" license, TBD) and that builds are fully hermetic and reproducible. This will allow an end-user to verify that their binary corresponds exactly to a particular git commit. We will additionally utilize some sort of public append-only log (potentially written to a public blockchain, and potentially with external notarization/attestation) to identify official release hashes (both source and binary) which will be required for automatic updates.

## Architecture

We are building a **Local-First Data Lakehouse** condensed into a single application. The system operates on a biological "Senses to Synapses" loop, prioritizing privacy and user ownership by design.

**Core Philosophy:**

* **Local-First:** All data collection, inference, and storage happen on-device.
* **Event Sourcing:** The system state is derived from an immutable stream of atomic events.
* **Bicameral Processing:** We separate the "Conscious" (UI/Application Logic) from the "Subconscious" (Perception/Inference) to ensure a responsive, non-blocking user experience.

### 1. The Ingestion Layer ("Senses")

The system captures raw input streams (screen, audio, location) into transient, encrypted buffers.

* **Time-Chunking:** Streams are sliced into manageable chunks (e.g., 30s) stored on the local filesystem.
* **Pre-Filtering:** Lightweight gating mechanisms (e.g., pixel-diffing, VAD) discard static or silent chunks to conserve battery and storage before deep inference occurs.

### 2. The Perception Layer ("Synapses")

A background **Perception Engine** (built with **Rust** and **Burn**) processes raw chunks to extract structured meaning. This engine runs in a separate thread/process from the main application logic.

* **Inference:** Using embedded ONNX models (e.g., CLIP for vision, Whisper for audio), the engine converts raw media into **Atomic Events**.
* **Vectorization:** Semantic embeddings are generated for searchable content, enabling "vague" queries (e.g., "that email about the budget").
* **Privacy:** All inference is local. No raw data leaves the device.

### 3. The Storage Layer ("Memory")

We utilize a hybrid storage approach for performance and query capability:

* **Turso (SQLite):** Stores structured metadata, atomic events, and full-text search indices. This is the "Long-term Memory."
* **Filesystem:** Stores the raw, encrypted blob data (referenced by SQLite).
* **Retention:** A risk-based retention policy allows the system to "forget" (delete) raw blobs while preserving the distilled metadata and insights.

### 4. The Prediction Loop ("Dreaming")

To power the "Spotlight" interface, the system continuously learns from user behavior:

* **Real-time:** A lightweight MLP model predicts the next likely action based on the current screen context and recent event history.
* **Async Optimization:** When the device is idle and charging, the system performs maintenance (compaction, index optimization) and fine-tunes local models on the day's interactions to improve future predictions.

### 5. The Event Bus

Communication between the application core (Crux) and the Perception Engine is handled via a strict Command-Event protocol:

* **Downstream (Crux → Perception):** `RequestPrediction`, `SetPrivacyMode`, `UpdateConfig`.
* **Upstream (Perception → Crux):** `PredictionReady`, `InsightDerived`, `StatusUpdate`.

This architecture ensures that the "Conscious" mind (the UI) remains snappy and responsive, while the "Subconscious" mind (the AI) creates deep understanding in the background.

## Developing

Eidolons consists of these components:
- **Crates** (`crates/`) - Rust crates including:
  - `eidolons-server` - OpenAI-compatible proxy that routes requests to AI providers (currently Anthropic Claude)
  - `eidolons-hello` - Example capability implementation
  - `eidolons-shared` - Crux-based cross-platform app core managing state and effects, exposing capabilities via FFI
- **APP: macOS** (`apps/macos/`) - SwiftUI shell that renders the shared core's view model

All builds are deterministic and reproducible via Nix.

Enter a development shell with Rust toolchain and tools:
```bash
nix develop  # Provides Rust toolchain, cargo-watch, rust-analyzer
```

Or use your own Rust installation with the toolchain specified in `rust-toolchain.toml`.

```bash
# Lint and format
cargo fmt
cargo clippy

# Run tests
cargo test

# Run the server
ANTHROPIC_API_KEY="<sk-ant-YOUR_API_KEY>" cargo run -p eidolons-server

```

**Updating generated files:**
If you change Rust APIs or types, you must update the committed Swift bindings or OpenAPI spec. The preferred way is using Nix, which ensures the same environment as CI:
```bash
nix run '.#update-eidolons-shared-swift-bindings'     # Update bindings and types
nix run '.#update-eidolons-shared-swift-xcframework'  # Update the static XCFramework
nix run '.#update-server-openapi'                     # Update OpenAPI spec
```

These can also be updated outside of Nix using your local Rust toolchain (auto-generates via `cargo run` and `cargo build`):
```bash
scripts/update-shared-bindings.sh
scripts/update-server-openapi.sh
scripts/update-shared-xcframework-dev.sh  # Fast: native architecture only (preferred for development)
scripts/update-shared-xcframework.sh      # Full: all architectures (used in CI)
```
*Note: XCFramework scripts require a macOS host.*

Generated artifacts are committed and verified by CI:
- `crates/eidolons-shared/swift/` - Shared core bindings (UniFFI + Crux types)
- `crates/eidolons-shared/target/apple/` - Compiled XCFramework
- `crates/eidolons-server/openapi.json` - Server API specification

## Building for release

This project uses Nix for reproducible builds. [Install Nix](https://nixos.org/download.html) with flakes enabled.

```bash
# Build targets
nix build '.#server'                            # Server binary (native)
nix build '.#server-oci'                        # Server OCI image (native)
nix build '.#eidolons-shared-swift-xcframework' # Shared core XCFramework

# Cross-compile Linux binaries
nix build '.#server--aarch64-unknown-linux-musl' # Linux ARM64 binary
nix build '.#server--x86_64-unknown-linux-musl'  # Linux x86_64 binary

# Build the OCI (docker) image
nix build '.#server-oci--aarch64-unknown-linux-musl' # ARM64 OCI image
nix build '.#server-oci--x86_64-unknown-linux-musl'  # x86_64 OCI image

# Run checks (tests, linting, formatting)
nix flake check
```

## Server

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint that proxies to Anthropic's Claude API, handling format translation and streaming.

### Running with Docker

```bash
# Build the Linux container image
nix build '.#server-oci--aarch64-unknown-linux-musl'  # ARM64
# OR
nix build '.#server-oci--x86_64-unknown-linux-musl'   # x86_64

# Load and run
docker load < result
docker run --rm -d -p 8080:8080 -e ANTHROPIC_API_KEY="<sk-ant-YOUR_API_KEY>" eidolons-server:latest

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}]}'
```
