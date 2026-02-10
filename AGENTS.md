# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Structure

```
eidolons/
├── crates/           # Rust crates
│   ├── eidolons-server/  # OpenAI-compatible AI proxy server
│   │   └── src/
│   │       ├── main.rs       # HTTP server (hyper + tokio)
│   │       ├── openai.rs     # OpenAI API types
│   │       ├── anthropic.rs  # Anthropic API types
│   │       ├── transform.rs  # Format conversion
│   │       └── proxy.rs      # Upstream HTTP client
│   └── eidolons-hello/   # Hello capability (example)
│       └── src/lib.rs
├── apps/
│   ├── eidolons-shared/  # Crux-based shared core (exclusive FFI generator)
│   │   ├── src/
│   │   │   ├── lib.rs        # FFI bridge (processEvent, handleResponse, view, capabilities)
│   │   │   ├── app.rs        # Crux App impl (Event, Model, ViewModel, Effect)
│   │   │   └── capabilities/ # Crux capabilities (e.g., eidolons)
│   │   ├── swift/            # Generated bindings (UniFFI + Crux types)
│   │   └── Package.swift     # Swift Package exposing EidolonsShared + SharedTypes
│   └── macos/            # macOS app (SwiftPM + Xcode wrapper)
│       ├── Sources/
│       │   ├── Eidolons/         # SwiftUI shell (Core.swift, ContentView.swift)
│       │   └── EidolonsEntrypoint/  # App entrypoint
│       ├── Xcode/            # Xcode project wrapper
│       ├── Support/          # Shared build files (Info.plist, scripts)
│       └── Package.swift     # Swift Package Manager config
├── docs/
│   └── design/           # Architecture Decision Records (ADRs)
├── tools/            # Build tooling
│   ├── uniffi-bindgen-swift/  # UniFFI binding generator
│   └── shared-typegen/        # Crux type generator for Swift
└── flake.nix         # Nix build definitions
```

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers.

**Current upstream:** Anthropic Claude (Messages API)

**Key design decisions:**
- Pure Rust TLS via `rustls-rustcrypto` (enables cross-compilation, no C dependencies)
- Statically linked musl binaries for Linux deployment
- Distroless OCI images (~9MB, runs as non-root)
- Request-based (no sessions/caching)

**API endpoints:**
- `GET /health` - Health check
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (proxied to Anthropic)

**Environment variables:**
- `ANTHROPIC_API_KEY` (required) - Anthropic API key
- `BIND_ADDR` (default: `127.0.0.1:8080`) - Address to bind

## Crux Architecture

The macOS app uses [Crux](https://redbadger.github.io/crux/) for cross-platform state management. The architecture separates the core (Rust) from the shell (Swift):

```
┌─────────────────────────────────────────────────────────┐
│  Swift Shell (apps/macos)                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Core.swift - handles event/effect loop           │  │
│  │  - Sends Events to core via processEvent()        │  │
│  │  - Handles Effects (Render, Eidolons capability)  │  │
│  │  - Updates ViewModel for SwiftUI                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │ FFI (UniFFI + bincode)
┌─────────────────────▼───────────────────────────────────┐
│  Crux Core (apps/eidolons-shared)                       │
│  - Event: user actions (e.g., Greet)                    │
│  - Model: private app state                             │
│  - ViewModel: public view state                         │
│  - Effect: side-effects for shell to handle             │
│  - Capabilities: Render, Eidolons (calls capability impls) │
└─────────────────────────────────────────────────────────┘
```

**Key pattern:** The core never performs side-effects directly. It emits Effects that the shell handles, then the shell sends responses back via `handleResponse()`.

**Capability implementations:** Pure Rust crates in `crates/` (e.g., `eidolons-hello`) implement capability logic. These are compiled into `eidolons-shared` and exposed via UniFFI, so the Swift shell can call them directly.

**Two codegen pipelines:**
- `uniffi-bindgen-swift` → FFI bridge (`processEvent`, `handleResponse`, `view`)
- `crux_core::typegen` → Domain types (`Event`, `Effect`, `ViewModel`) with bincode serialization

## Build Commands

For rapid iteration, **prefer the local Rust toolchain** inside the Nix development shell. Nix is the final source of authority for CI and releases, but standard `cargo` commands allow for incremental compilation.

```bash
# 1. Enter the development environment
nix develop

# 2. Development (Inner Loop - PREFERRED)
cargo build -p eidolons-server    # Quick local build
cargo test                        # Run tests
cargo clippy                      # Lint
cargo fmt                         # Format

# 3. Updating generated files (after changing Rust APIs/types)
# These scripts auto-generate artifacts using your local toolchain.
scripts/update-shared-bindings.sh
scripts/update-server-openapi.sh
scripts/update-shared-xcframework-dev.sh  # Fast dev build (native arch only)
scripts/update-shared-xcframework.sh      # Full build (all architectures, CI)

# 4. Verification (Simulates CI)
# Optimized for speed; focuses on correctness, not heavy artifact building.
nix flake check

# 5. Production builds (Nix - Final Authority)
nix build '.#server'                              # Native binary
nix build '.#server-oci'                          # OCI image
nix build '.#eidolons-shared-swift-xcframework'   # Shared core XCFramework

# 6. Nix-based updates (for perfect CI parity)
nix run '.#update-eidolons-shared-swift-bindings'
nix run '.#update-server-openapi'
```

## Cross-Compilation

Targets defined in `rust-toolchain.toml`:
- macOS: `aarch64-apple-darwin`, `x86_64-apple-darwin`
- Linux: `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`

**OCI images:** Use `server-oci--<linux-target>` for Docker. The native `server-oci` builds for macOS and won't run in Docker.

## Key Files

| File | Purpose |
|------|---------|
| `flake.nix` | Nix build definitions, cross-compile targets, CI checks |
| `rust-toolchain.toml` | Pinned Rust version (1.92.0) and targets |
| `Cargo.toml` | Workspace config, release profile (LTO, single codegen unit) |
| `crates/eidolons-server/Cargo.toml` | Server dependencies |
| `crates/eidolons-hello/src/lib.rs` | Hello capability implementation (pure Rust) |
| `apps/eidolons-shared/Package.swift` | Shared core Swift Package (EidolonsShared + SharedTypes) |
| `apps/eidolons-shared/src/lib.rs` | FFI bridge + capability re-exports |
| `apps/eidolons-shared/src/app.rs` | Crux App implementation (Event, Model, ViewModel, Effect) |
| `apps/macos/Package.swift` | macOS app Swift Package config |
| `apps/macos/Sources/Eidolons/Core.swift` | Swift shell bridge (handles Crux event/effect loop) |
| `apps/macos/Support/Info.plist` | Shared app Info.plist |
| `apps/macos/Support/package-app.sh` | CLI build script for .app bundle |

## Design Documents

Architecture decisions are recorded in [`docs/design/`](docs/design/). See the
[index](docs/design/README.md) for a full list. Key decisions:

- [Model Weight Management](docs/design/model-weight-management.md) — weights as pinned dependencies, hash-verified at every boundary
- [Pure Rust, Zero C Dependencies](docs/design/pure-rust-zero-c-dependencies.md) — rustls-rustcrypto, webpki-roots, no C cross-compiler needed
- [Reproducible Builds](docs/design/reproducible-builds.md) — Nix, Crane, deterministic settings, CI-verified generated artifacts
- [Crux Cross-Platform Architecture](docs/design/crux-cross-platform-architecture.md) — Elm-like core/shell split, UniFFI, bincode FFI bridge
- [OpenAI-Compatible Proxy Server](docs/design/openai-compatible-proxy-server.md) — canonical API format, stateless proxy, distroless OCI
- [On-Device Inference with Burn](docs/design/on-device-inference-with-burn.md) — pure Rust ML, WGPU GPU backend, model-per-crate

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- No caching/state in server (request-based)
- OpenAI API format as the canonical interface
- Deterministic builds (no timestamps, fixed codegen)
