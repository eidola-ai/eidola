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
├── apps/macos/       # macOS app (SwiftPM + Xcode wrapper)
│   ├── Sources/
│   │   ├── Eidolons/         # EidolonsApp library (SwiftUI views)
│   │   └── EidolonsEntrypoint/  # App entrypoint
│   ├── Xcode/            # Xcode project wrapper
│   ├── Support/          # Shared build files (Info.plist, scripts)
│   └── Package.swift     # Swift Package Manager config
├── tools/            # Build tooling
│   └── uniffi-bindgen-swift/  # Custom binding generator
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

# Swift bindings (after changing eidolons/ API)
nix run '.#update-core-swift-bindings'
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
| `apps/macos/Package.swift` | macOS app Swift Package config |
| `apps/macos/Support/Info.plist` | Shared app Info.plist |
| `apps/macos/Support/package-app.sh` | CLI build script for .app bundle |

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- No caching/state in server (request-based)
- OpenAI API format as the canonical interface
- Deterministic builds (no timestamps, fixed codegen)
