# Eidolons

This is a nearly-empty rust project with two packages: an "eidolons" library (in the "core" directory) that will be used by client applications and a "eidolons-server" executable (in the "server" directory).

The immediate purpose of this repository is to create a working proof of concept, demonstrating the ability to create deterministic builds.

## Building

This project uses Nix for reproducible builds. [Install Nix](https://nixos.org/download.html) with flakes enabled.

**Build commands:**
```bash
nix build '.#server'                  # Server binary (musl, static)
nix build '.#core'                    # Core library (glibc, dynamic)
nix build '.#server-oci'              # Server OCI/Docker image

# Cross-compile to Linux (from macOS)
nix build '.#server-x86_64-linux'
nix build '.#server-aarch64-linux'

# Run all checks (builds, formatting, clippy, tests)
nix flake check
```

**Development shell:**
```bash
nix develop  # Provides Rust toolchain, cargo-watch, rust-analyzer
```
