# Design Documents

Architectural decisions for the eidolons project.

Each document captures a significant design decision with its context, rationale,
and consequences. These are **living documents** — when a decision evolves, update
the document directly. Git history tracks what changed and why.

## Format

```
# Title

## Context
## Decision
## Consequences
## Future Considerations (optional)
```

## Index

| Document | Summary |
|----------|---------|
| [Model Weight Management](model-weight-management.md) | Weights as pinned dependencies, hash-verified at every boundary |
| [Pure Rust, Zero C Dependencies](pure-rust-zero-c-dependencies.md) | rustls-rustcrypto, webpki-roots, no C cross-compiler needed |
| [Reproducible Builds](reproducible-builds.md) | Nix, Crane, deterministic settings, CI-verified generated artifacts |
| [Crux Cross-Platform Architecture](crux-cross-platform-architecture.md) | Elm-like core/shell split, UniFFI, bincode FFI bridge |
| [OpenAI-Compatible Proxy Server](openai-compatible-proxy-server.md) | Canonical API format, stateless proxy, distroless OCI |
| [On-Device Inference with Burn](on-device-inference-with-burn.md) | Pure Rust ML, WGPU GPU backend, model-per-crate |
