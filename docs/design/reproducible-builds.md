# Reproducible Builds via Nix

## Context

Reproducible builds are a core project value, not a nice-to-have. If a user
cannot independently verify that a binary was built from a specific source
commit, they must trust the builder. For a project that runs AI models with
direct influence on user behavior, that trust requirement is unacceptable.

The build pipeline must produce bit-identical outputs given the same inputs
(source commit + lock files), across different machines and at different times.

### The Apple problem

macOS and iOS apps cannot be fully reproducible end-to-end because Apple's
code signing, provisioning profiles, and notarization inject non-deterministic
artifacts. The build must be reproducible *up to the signing boundary*, so
that the unsigned app can be verified independently.

### Why not just `cargo build`?

Cargo alone does not guarantee reproducibility:
- Incremental compilation introduces non-determinism
- Build parallelism can affect output ordering
- System-installed tools (linker version, SDK version) vary across machines
- Network access during builds is uncontrolled
- Timestamps leak into archives and debug info

## Decision

Use **Nix** as the authoritative build system for all artifacts that leave the
developer's machine (CI, releases, distribution). Standard `cargo` commands
remain available for inner-loop development where reproducibility is not
required.

### Build system layering

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Fully Reproducible (Nix)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Rust crates (server, perception, shared core)                      │
│  Swift bindings generation (UniFFI + Crux typegen)                  │
│  XCFramework (static libraries for macOS arm64 + x86_64)           │
│  Server binaries (Linux musl, macOS)                                │
│  OCI container images                                               │
│  OpenAPI specification                                              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│            Deterministic (Same Xcode + macOS version)               │
├─────────────────────────────────────────────────────────────────────┤
│  Unsigned macOS app (.app bundle) — planned                         │
│  Uses ZERO_AR_DATE, SOURCE_DATE_EPOCH, path remapping               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Non-Reproducible (Signing)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Code signature (developer identity, timestamps)                    │
│  Provisioning profile                                               │
│  Notarization ticket                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Nix tooling choices

- **Crane** (ipetkov/crane) for Rust builds. Crane separates dependency
  compilation from crate compilation, enabling better caching. Preferred over
  `naersk` and nixpkgs' `buildRustPackage` for its workspace-aware caching.

- **Fenix** (nix-community/fenix) for Rust toolchain management. Reads
  `rust-toolchain.toml` directly, so one file is the source of truth for both
  Nix and local development. The toolchain derivation is additionally pinned
  by SHA256 in `flake.nix`.

- **Workspace source filtering.** A custom `mkFilteredSrc` function in
  `flake.nix` parses the Cargo dependency graph and creates minimal source
  trees for each package build. Changing one crate's source does not invalidate
  the Nix cache for unrelated crates.

### Deterministic build settings

Applied to all Nix-built Rust artifacts:

| Setting | Value | Purpose |
|---------|-------|---------|
| `CARGO_INCREMENTAL` | `false` | Disable non-deterministic incremental compilation |
| `SOURCE_DATE_EPOCH` | `0` | Fixed timestamp for all date-dependent operations |
| `ZERO_AR_DATE` | `1` | Reproducible ar/ranlib archives |
| `CARGO_NET_OFFLINE` | `true` | Network isolation during build |
| `RUSTFLAGS` | `-C debuginfo=0 -C target-cpu=generic` | No debug info (path-dependent), generic codegen |

Cargo release profile (`Cargo.toml`):

| Setting | Value | Purpose |
|---------|-------|---------|
| `codegen-units` | `1` | Single codegen unit eliminates parallelism-dependent output |
| `incremental` | `false` | Matches Nix setting |
| `lto` | `true` | Whole-program optimization for deterministic dead code elimination |
| `strip` | `true` | Remove symbols (which contain build paths) |

### Generated artifacts: committed and CI-verified

Two categories of generated code are committed to the repository:

1. **Swift bindings** (UniFFI + Crux typegen) — committed in
   `apps/eidolons-shared/swift/`. Swift developers can build without running
   the Rust codegen toolchain.
2. **OpenAPI specification** — generated from Rust type annotations via utoipa.

CI verifies freshness: `nix flake check` regenerates both artifacts and diffs
them against the committed versions. A mismatch fails the build. This is a
"golden file" pattern — always generated from code, never hand-edited, but
committed for convenience.

### OCI images

Server containers are built with `pkgs.dockerTools.buildLayeredImage` (not
Docker). Images are:
- **Distroless**: no shell, package manager, or libc. Only the static binary.
- **Non-root**: runs as UID 65534 (nobody).
- **Timestamped to epoch**: `created = "1970-01-01T00:00:00Z"` for
  reproducibility.
- **~9MB**: the static musl binary is the only content.

OCI operations use the `crane` CLI (google/go-containerregistry), not the
Docker daemon. No Docker installation is required anywhere in the pipeline.

### CI environment

- **Self-hosted runners** for Apple Silicon support, Nix binary cache
  integration, and long build timeouts (30 minutes for cross-compilation).
- **GitHub Actions pinned by SHA** (not tag) to prevent supply-chain attacks
  through compromised action tags. A `pinact` workflow automates SHA pinning.
- A `nix-hash-suggester` bot proposes hash updates when Dependabot PRs cause
  Nix fixed-output hash mismatches.

### Verification

Given the same git commit and `flake.lock`, anyone can reproduce and verify
any Nix-built artifact:

```bash
# Core library
nix build '.#core'
sha256sum result/lib/libeidolons.a

# XCFramework
nix build '.#eidolons-shared-swift-xcframework'
sha256sum result/libeidolons-rs.xcframework/*/libeidolons.a

# Server binary
nix build '.#server--x86_64-unknown-linux-musl'
sha256sum result/bin/eidolons-server

# OCI image
nix build '.#server-oci--x86_64-unknown-linux-musl'
sha256sum result
```

## Consequences

**Benefits:**

- Bit-identical outputs from any machine with Nix installed.
- The Nix binary cache eliminates redundant builds across the team and CI.
- Generated artifacts are always verifiably fresh.
- OCI images are minimal, signed by content hash, and reproducible.
- `flake.lock` pins the entire dependency tree (Nix packages, Rust toolchain,
  Crane, Fenix) to exact revisions.

**Trade-offs we accept:**

- Nix has a steep learning curve. Contributors must understand Nix to modify
  CI or release infrastructure.
- Inner-loop development uses `cargo` directly, which may diverge from the
  Nix build. `nix flake check` catches this, but only when run explicitly.
- `CARGO_BUILD_JOBS=1` (single-threaded compilation) would guarantee
  determinism from proc macros but is prohibitively slow. It is currently
  disabled — if a proc macro introduces non-determinism, this setting can be
  re-enabled.
- Self-hosted CI runners require maintenance and are not ephemeral like
  GitHub-hosted runners.

## Future Considerations

- **Tart for fully reproducible macOS builds.** Use Tart (cirruslabs/tart) to
  provide pinned macOS VM images for CI, achieving bit-identical unsigned .app
  bundles by fixing the macOS version, Xcode version, and build environment.

- **Unsigned app verification.** When implemented, unsigned macOS app builds
  will use `ZERO_AR_DATE`, `SOURCE_DATE_EPOCH`, debug prefix remapping, and
  fixed version numbers to maximize reproducibility. Known sources of
  variation: SDK version in `LC_BUILD_VERSION`, linker version metadata
  (normalizable with `vtool`), module cache paths.

- **Nix binary cache for model weights.** Model weights pinned by hash
  (see [Model Weight Management](model-weight-management.md)) can be distributed
  through the same Nix binary cache infrastructure used for build artifacts.
