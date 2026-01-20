# Reproducible Builds

This document describes Eidolons' approach to reproducible and verifiable builds.

## Overview

Eidolons uses a hybrid build strategy that maximizes reproducibility while acknowledging
the constraints of Apple's toolchain:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Fully Reproducible (Nix)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Rust core library (eidolons)                                       │
│  Swift bindings generation (uniffi-bindgen-swift)                   │
│  XCFramework creation (static libraries for macOS)                  │
│  Server binaries (Linux musl, macOS)                                │
│  OCI container images                                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Deterministic (Same Xcode + macOS version)             │
├─────────────────────────────────────────────────────────────────────┤
│  Unsigned macOS app (.app bundle)                                   │
│  Build uses ZERO_AR_DATE, SOURCE_DATE_EPOCH, path remapping         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Non-reproducible (Signing)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Code signature (developer identity, timestamps)                    │
│  Provisioning profile                                               │
│  Notarization ticket                                                │
└─────────────────────────────────────────────────────────────────────┘
```

## What You Can Verify

### 1. Rust Core Library (Fully Reproducible)

The core library is built entirely within Nix's hermetic sandbox. Given the same:
- Git commit
- Nix flake lock

You will get bit-identical outputs:

```bash
# Build and check hash
nix build '.#core'
sha256sum result/lib/libeidolons.a
```

### 2. Swift Bindings (Fully Reproducible)

Generated Swift code and FFI headers are deterministic:

```bash
nix build '.#core-swift-bindings'
diff -r result/Sources/EidolonsCore core/swift/Sources/EidolonsCore
```

CI verifies committed bindings match generated ones via `nix flake check`.

### 3. XCFramework (Fully Reproducible)

The universal XCFramework containing the static library for macOS:

```bash
nix build '.#eidolons-swift-xcframework'

# Check hash of the library
sha256sum result/libeidolons-rs.xcframework/macos-arm64_x86_64/libeidolons.a

# Check hash of the framework metadata
sha256sum result/libeidolons-rs.xcframework/Info.plist
```

The XCFramework contains:
- `macos-arm64_x86_64/libeidolons.a` - Universal macOS (arm64 + x86_64)
- `Info.plist` - Framework metadata

All libraries are built hermetically within Nix.

### 4. Unsigned macOS App (Planned)

Unsigned app builds with deterministic settings are planned but not yet implemented.
The build would use settings like `ZERO_AR_DATE`, `SOURCE_DATE_EPOCH`, and path remapping
to maximize reproducibility while depending on system Xcode.

## Verifying Builds

### Rust/Nix Artifacts

All Rust artifacts built through Nix are fully reproducible. Given the same git commit
and `flake.lock`, you will get bit-identical outputs:

```bash
# Clone and build
git clone https://github.com/eidolons-ai/eidolons.git
cd eidolons

# Verify core library
nix build '.#core'
sha256sum result/lib/libeidolons.a

# Verify XCFramework
nix build '.#core-swift-xcframework'
sha256sum result/libeidolons-rs.xcframework/*/libeidolons.a

# Verify server binaries
nix build '.#server--x86_64-unknown-linux-musl'
sha256sum result/bin/eidolons-server
```

## Build Settings for Determinism

The Nix builds use these settings to maximize reproducibility (from `flake.nix`):

| Setting | Value | Purpose |
|---------|-------|---------|
| `CARGO_BUILD_JOBS` | `1` | Single-threaded for reproducibility |
| `CARGO_INCREMENTAL` | `false` | Disable incremental compilation |
| `SOURCE_DATE_EPOCH` | `0` | Fixed timestamp |
| `ZERO_AR_DATE` | `1` | Reproducible ar/ranlib archives |
| `CARGO_NET_OFFLINE` | `true` | Network isolation during build |
| `RUSTFLAGS` | `-C debuginfo=0 -C target-cpu=generic` | Deterministic codegen |

For future unsigned Xcode app builds, these additional settings would be used:

| Setting | Value | Purpose |
|---------|-------|---------|
| `ZERO_AR_DATE` | `1` | Zero timestamps in archives |
| `SOURCE_DATE_EPOCH` | `0` | Fixed `__DATE__`/`__TIME__` macros |
| `CODE_SIGN_IDENTITY` | `""` | Disable signing |
| `CODE_SIGNING_REQUIRED` | `NO` | Allow unsigned build |
| `-Wl,-oso_prefix` | `$SRCROOT/` | Relativize debug paths |
| `-fdebug-prefix-map` | `$SRCROOT=.` | Relativize source paths |
| `-debug-prefix-map` | `$SRCROOT=.` | Swift debug path remapping |
| `CURRENT_PROJECT_VERSION` | `1` | Fixed version (not auto-increment) |

## Known Limitations

### What's Fully Reproducible Now

- Rust core library and server binaries (via Nix)
- Swift bindings generation
- XCFramework creation
- OCI container images

### Future: Unsigned App Builds

When unsigned Xcode app builds are implemented, they will depend on:
- **macOS version** - Different SDK versions embed different metadata
- **Xcode version** - Compiler/linker versions affect output
- **Xcode Command Line Tools** - Must match Xcode version

Even with deterministic settings, these may vary across environments:
- `LC_BUILD_VERSION` SDK version in Mach-O headers
- Linker version metadata (can be normalized with `vtool`)
- Module cache paths (shouldn't affect final binary)

### Not Currently Implemented

- Unsigned macOS app builds
- iOS/visionOS builds
- App Store builds (require signing)

## Future: Bit-Identical Builds with Tart

We plan to use [Tart](https://github.com/cirruslabs/tart) to provide fully reproducible
macOS VM images for CI builds. This will enable:

1. **Pinned macOS version** - Exact OS version in VM image
2. **Pinned Xcode version** - Installed in VM image
3. **Hermetic builds** - No host system dependencies
4. **Verifier reproduction** - Anyone can run the same VM

The workflow will be:

```
Tart VM Image (versioned)
  └── macOS 26.x
      └── Xcode 26.x
          └── Nix
              └── xcodebuild (unsigned)
                  └── Bit-identical .app
```

## CI Outputs

Every CI run on `main` produces OCI images pushed to GitHub Container Registry:

| Image | Description |
|-------|-------------|
| `ghcr.io/eidolons-ai/eidolons-server:latest` | Multi-arch manifest (amd64 + arm64) |
| `ghcr.io/eidolons-ai/eidolons-server:sha-<commit>` | Commit-specific multi-arch manifest |
| `ghcr.io/eidolons-ai/eidolons-server:latest-amd64` | Linux x86_64 (musl, static) |
| `ghcr.io/eidolons-ai/eidolons-server:latest-arm64` | Linux ARM64 (musl, static) |

Pull requests produce images tagged with `pr-<number>` instead of `latest`/`sha-*`.

CI also runs:
- `nix flake check` (formatting, clippy, tests, binding sync verification)
- Swift tests via `swift test`

## References

- [Apple's Linker & Deterministic Builds](https://milen.me/writings/apple-linker-ld64-deterministic-builds-oso-prefix/)
- [Reproducible Builds for macOS](https://gist.github.com/pudquick/89c90421a9582f88741b21d10c6a155e)
- [SOURCE_DATE_EPOCH Specification](https://reproducible-builds.org/specs/source-date-epoch/)
- [Telegram Reproducible Builds](https://core.telegram.org/reproducible-builds)
