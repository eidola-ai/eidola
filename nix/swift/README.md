# Hermetic Swift 6.2

This Nix flake provides the latest stable Swift (6.2) targeting macOS ARM64 (`aarch64-darwin`). 

## The Pragmatic Compromise

This repository takes a pragmatic approach to providing a reproducible Swift compiler: rather than painstakingly building the massive Swift and LLVM toolchains from source within Nix (which is notoriously fragile, lengthy, and difficult to maintain across platforms), this flake **downloads and wraps Apple's official pre-built macOS binaries**. 

The wrapper scripts dynamicially intercept compilation commands and strictly limit the compiler to Nix's sandbox environment. By automatically injecting `apple-sdk` paths and transparently stripping unsupported framework flags injected by conventional Nix cross-compilers (such as `-Werror=unguarded-availability`), we obtain a robust and extremely fast installation.

This significantly lowers the maintenance burden compared to the enormous effort required to build Swift strictly from source via CMake. For more context, see the broader upstream discussions:

- [Nixpkgs Issue #343210: Update request: swift 5.8 → 6](https://github.com/NixOS/nixpkgs/issues/343210)
- [Matrix Channel: #nixpkgs-swift:matrix.org](https://matrix.to/#/#nixpkgs-swift:matrix.org)