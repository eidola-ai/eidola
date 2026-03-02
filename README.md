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
  - `eidolons-server` - Privacy-transparent OpenAI-compatible proxy that routes requests through RedPill.ai with inline attestation metadata
  - `eidolons-hello` - Example capability implementation
  - `eidolons-shared` - Crux-based cross-platform app core managing state and effects, exposing capabilities via FFI
- **APP: macOS** (`apps/macos/`) - SwiftUI shell that renders the shared core's view model

**Prerequisites:** `rustup`, `just`, `docker`

The Rust toolchain version is pinned in `rust-toolchain.toml` and installed automatically by rustup. Run `just` to see all available recipes.

```bash
# Start postgres for local development
just db

# Run the server on the host (fast iteration with incremental compilation)
REDPILL_API_KEY="<YOUR_KEY>" cargo run -p eidolons-server

# Lint and format
cargo fmt
cargo clippy

# Run tests
cargo test

# Full stack in containers (postgres + server)
just dev
```

**Updating generated files:**
If you change Rust APIs or types, update the committed Swift bindings or OpenAPI spec:
```bash
just update-bindings      # UniFFI Swift bindings + Crux types
just update-openapi       # OpenAPI spec
just update-xcframework   # XCFramework (dev, native arch only)
```

For CI parity, Nix-based equivalents are available:
```bash
nix run '.#update-eidolons-shared-swift-bindings'
nix run '.#update-eidolons-shared-swift-xcframework'
nix run '.#update-server-openapi'
```

Generated artifacts are committed and verified by CI:
- `crates/eidolons-shared/swift/` - Shared core bindings (UniFFI + Crux types)
- `crates/eidolons-shared/target/apple/` - Compiled XCFramework
- `crates/eidolons-server/openapi.json` - Server API specification

## Building for release

### Server OCI image

The server image is built with a [StageX](https://stagex.tools/)-based Containerfile for reproducible, fully-bootstrapped builds:

```bash
# Build the server OCI image
just oci-build

# Or directly:
docker build -f crates/eidolons-server/Containerfile -t eidolons-server:dev .
```

The image is `FROM scratch` with a statically-linked musl binary (~9MB, runs as non-root UID 65534).

### Swift / macOS

Nix handles XCFramework and macOS app builds:

```bash
nix build '.#eidolons-shared-swift-xcframework'   # Shared core XCFramework

# macOS app
( cd apps/macos && Support/package-app.sh )
```

### CI checks

```bash
nix flake check   # Runs: cargo fmt, clippy, tests, openapi/binding freshness, Swift formatting
```

## Server

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint that proxies requests through RedPill.ai, enriching responses with privacy-transparent attestation metadata.

### Running with Docker

```bash
# Build the image
just oci-build

# Run
docker run --rm -d -p 8080:8080 -e REDPILL_API_KEY="<YOUR_KEY>" eidolons-server:dev

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}]}'
```

### Running with Docker Compose

```bash
# Copy and fill in your API key
cp .env.example .env

# Start postgres + server
just dev

# Or just postgres (run server on the host for fast iteration)
just db
cargo run -p eidolons-server
```

### Production deployment (dstack)

The server is deployed to Phala dstack TEEs. All services run inside a single Confidential VM with encrypted storage.
