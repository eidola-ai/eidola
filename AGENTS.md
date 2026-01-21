# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Structure

```
eidolons/
├── eidolons-server/  # OpenAI-compatible AI proxy server
│   └── src/
│       ├── main.rs       # HTTP server (hyper + tokio)
│       ├── openai.rs     # OpenAI API types
│       ├── anthropic.rs  # Anthropic API types
│       ├── transform.rs  # Format conversion
│       └── proxy.rs      # Upstream HTTP client
├── eidolons/         # Rust library with Swift bindings
│   ├── src/lib.rs        # Exports via #[uniffi::export]
│   ├── swift/            # Generated Swift bindings (committed)
│   └── Package.swift     # Swift Package Manager config
├── apps/
│   ├── eidolons-shared/  # Crux-based shared core
│   │   ├── src/
│   │   │   ├── lib.rs        # FFI bridge (processEvent, handleResponse, view)
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
│  - Capabilities: Render, Eidolons (calls eidolons lib)  │
└─────────────────────────────────────────────────────────┘
```

**Key pattern:** The core never performs side-effects directly. It emits Effects that the shell handles, then the shell sends responses back via `handleResponse()`.

**Two codegen pipelines:**
- `uniffi-bindgen-swift` → FFI bridge (`processEvent`, `handleResponse`, `view`)
- `crux_core::typegen` → Domain types (`Event`, `Effect`, `ViewModel`) with bincode serialization

## Build Commands

All builds use Nix for reproducibility. Run `nix develop` to enter dev shell.

```bash
# Development
cargo build -p eidolons-server    # Quick local build
cargo clippy -p eidolons-server   # Lint
cargo fmt                         # Format

# Production builds (Nix)
nix build '.#server'                              # Native binary
nix build '.#server--aarch64-unknown-linux-musl'  # Linux ARM64
nix build '.#server-oci--aarch64-unknown-linux-musl'  # Linux ARM64 container

# Checks
nix flake check   # All checks (fmt, clippy, tests, binding sync)

# Swift bindings (after changing Rust APIs)
nix run '.#update-core-swift-bindings'            # eidolons/ bindings
nix run '.#update-eidolons-shared-swift-bindings' # eidolons-shared/ bindings
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
| `eidolons-server/Cargo.toml` | Server dependencies |
| `eidolons/Package.swift` | Core library Swift Package config |
| `apps/eidolons-shared/Package.swift` | Shared core Swift Package (EidolonsShared + SharedTypes) |
| `apps/eidolons-shared/src/app.rs` | Crux App implementation (Event, Model, ViewModel, Effect) |
| `apps/macos/Package.swift` | macOS app Swift Package config |
| `apps/macos/Sources/Eidolons/Core.swift` | Swift shell bridge (handles Crux event/effect loop) |
| `apps/macos/Support/Info.plist` | Shared app Info.plist |
| `apps/macos/Support/package-app.sh` | CLI build script for .app bundle |

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- No caching/state in server (request-based)
- OpenAI API format as the canonical interface
- Deterministic builds (no timestamps, fixed codegen)
