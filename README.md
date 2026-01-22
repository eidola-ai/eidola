# Eidolons

Eidolons consists of four components:
- **Server** (`eidolons-server/`) - An OpenAI-compatible proxy that routes requests to AI providers (currently Anthropic Claude)
- **Core library** (`eidolons/`) - A Rust library with Swift bindings for business logic
- **Shared core** (`apps/eidolons-shared/`) - Crux-based cross-platform app core managing state and effects
- **macOS app** (`apps/macos/`) - SwiftUI shell that renders the shared core's view model

All builds are deterministic and reproducible via Nix.

## Developing

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
```bash
nix run '.#update-core-swift-bindings'           # Update eidolons/ Swift bindings
nix run '.#update-eidolons-shared-swift-bindings' # Update shared core Swift bindings
nix run '.#update-server-openapi'                # Update OpenAPI spec
```

Generated Swift bindings are committed and verified by CI:
- `eidolons/swift/` - Core library bindings
- `apps/eidolons-shared/swift/` - Shared core bindings (UniFFI + Crux types)

## Building for release

This project uses Nix for reproducible builds. [Install Nix](https://nixos.org/download.html) with flakes enabled.

```bash
# Build targets
nix build '.#server'                          # Server binary (native)
nix build '.#server-oci'                      # Server OCI image (native, for macOS won't run in Docker)
nix build '.#eidolons-swift-xcframework'      # Core library XCFramework
nix build '.#eidolons-shared-swift-xcframework' # Shared core XCFramework

# Cross-compile Linux binaries
nix build '.#server--aarch64-unknown-linux-musl' # Linux ARM64 binary
nix build '.#server--x86_64-unknown-linux-musl'  # Linux x86_64 binary

# Build the OCI (docker) image
nix build '.#server-oci--aarch64-unknown-linux-musl' # ARM64 OCI image
nix build '.#server-oci--x86_64-unknown-linux-musl'  # x86_64 OCI image

# Run all checks
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
