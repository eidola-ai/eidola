# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eidolons is a Rust library with Swift bindings for iOS/macOS apps, plus a server component. The primary focus is demonstrating deterministic, reproducible builds via Nix.

**Structure:**
- `core/` - Rust library with uniffi-generated Swift bindings
- `server/` - Rust server binary
- `apps/apple/Eidolons/` - Xcode project for macOS/iOS app

## Build Commands

This project uses Nix for reproducible builds. All commands require Nix with flakes enabled.

```bash
# Enter development shell (provides Rust toolchain, cargo-watch, rust-analyzer)
nix develop

# Run all checks (formatting, clippy, tests, binding sync verification)
nix flake check

# Build targets (native)
nix build '.#server'                 # Server binary (native, e.g. macOS on macOS)
nix build '.#server-oci'             # Server OCI/Docker image (native target)
nix build '.#core'                   # Core static library (native)
nix build '.#core-swift-xcframework' # XCFramework for Apple platforms

# Cross-compile targets (append --<target> to package name)
nix build '.#server--x86_64-unknown-linux-musl'   # Linux x86_64
nix build '.#server--aarch64-unknown-linux-musl'  # Linux ARM64

# Update Swift bindings after changing core Rust API
nix run '.#update-core-swift-bindings'
nix run '.#update-core-swift-xcframework'

# Swift tests
cd core && swift test
```

## Architecture

### Rust-Swift FFI Bridge

The core library uses [uniffi](https://github.com/mozilla/uniffi-rs) v0.30.0 for FFI generation:

1. **Rust code** in `core/src/lib.rs` exports functions via `#[uniffi::export]`
2. **Custom binding generator** in `core/uniffi-bindgen-swift/` produces Swift code
3. **Generated artifacts** (committed):
   - `core/swift/Sources/EidolonsCore/` - Swift bindings
   - `core/swift/Sources/EidolonsCoreFFI/` - C FFI header
4. **XCFramework** built for: macOS (arm64+x86_64), iOS device, iOS simulator

CI verifies committed bindings match generated output via `nix flake check`.

### Cross-Compilation Targets

Defined in `rust-toolchain.toml` (Rust 1.92.0):
- Apple: `aarch64-apple-darwin`, `x86_64-apple-darwin`, `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios`
- Linux: `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`

### Reproducibility

All Rust/Nix builds are fully reproducible. See `REPRODUCIBILITY.md` for details on:
- Deterministic build settings used in Nix
- XCFramework and artifact verification
- CI outputs (OCI images pushed to GHCR)

## Key Configuration Files

- `flake.nix` - Nix build definitions, cross-compile targets, CI checks
- `rust-toolchain.toml` - Pinned Rust version and targets
- `Cargo.toml` (workspace root) - Workspace members, optimized release profile
- `core/Package.swift` - Swift Package Manager config with binary target support
