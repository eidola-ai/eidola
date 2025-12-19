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
│  XCFramework creation (static libraries for all Apple platforms)    │
│  Server binaries (Linux musl, macOS)                                │
│  OCI container images                                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Deterministic (Same Xcode + macOS version)             │
├─────────────────────────────────────────────────────────────────────┤
│  Unsigned macOS/iOS app (.app bundle)                               │
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
nix build '.#swift-bindings'
diff -r result/Sources/EidolonsCore core/swift/Sources/EidolonsCore
```

CI verifies committed bindings match generated ones via `nix flake check`.

### 3. XCFramework (Fully Reproducible)

The universal XCFramework containing static libraries for all Apple platforms:

```bash
nix build '.#xcframework'

# Check hashes of all platform libraries
sha256sum result/macos-arm64_x86_64/libeidolons.a
sha256sum result/ios-arm64/libeidolons.a
sha256sum result/ios-arm64_x86_64-simulator/libeidolons.a

# Check hash of the framework metadata
sha256sum result/Info.plist
```

The XCFramework contains:
- `macos-arm64_x86_64/libeidolons.a` - Universal macOS (arm64 + x86_64)
- `ios-arm64/libeidolons.a` - iOS device (arm64)
- `ios-arm64_x86_64-simulator/libeidolons.a` - iOS simulator (arm64 + x86_64)
- `Info.plist` - Framework metadata

All libraries are built hermetically within Nix - no iOS SDK required for static library compilation.

### 4. Unsigned macOS App (Deterministic with Caveats)

The unsigned app is built with deterministic settings but depends on system Xcode:

```bash
# Via Nix (requires system Xcode)
nix run '.#build-app-unsigned'

# Or directly
build-app-unsigned  # if in nix develop shell
```

**Output:** `build/unsigned/Eidolons.app` with `BUILD-INFO.txt` and `checksums.sha256`

## Verifying a Release

### Prerequisites

To reproduce our CI builds, you need:
- macOS (version specified in release's `BUILD-INFO.txt`)
- Xcode (version specified in release's `BUILD-INFO.txt`)
- Nix with flakes enabled

### Step-by-Step Verification

1. **Clone at the release tag:**
   ```bash
   git clone https://github.com/anthropic/eidolons.git
   cd eidolons
   git checkout v1.0.0  # or specific release tag
   ```

2. **Build the unsigned app:**
   ```bash
   nix run .#build-app-unsigned
   ```

3. **Compare checksums:**
   ```bash
   # Compare against published checksums
   diff build/unsigned/checksums.sha256 <(curl -sL https://github.com/.../checksums.sha256)
   ```

4. **Verify signed release matches unsigned build:**
   ```bash
   # Download signed release
   curl -LO https://github.com/.../Eidolons-signed.app.zip
   unzip Eidolons-signed.app.zip

   # Strip signature for comparison
   codesign --remove-signature Eidolons-signed.app/Contents/MacOS/Eidolons

   # Compare binaries (excluding _CodeSignature and provisioning)
   diff -r build/unsigned/Eidolons.app Eidolons-signed.app \
     --exclude="_CodeSignature" \
     --exclude="embedded.provisionprofile"
   ```

## Build Settings for Determinism

The unsigned app build uses these settings to maximize reproducibility:

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

### Environment Dependencies

The unsigned app build depends on:
- **macOS version** - Different SDK versions embed different metadata
- **Xcode version** - Compiler/linker versions affect output
- **Xcode Command Line Tools** - Must match Xcode version

### What May Differ

Even with deterministic settings, these may vary across environments:
- `LC_BUILD_VERSION` SDK version in Mach-O headers
- Linker version metadata (can be normalized with `vtool`)
- Module cache paths (shouldn't affect final binary)

### Not Currently Verified

- iOS app builds (requires additional iOS SDK setup)
- visionOS builds
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
  └── macOS 15.1
      └── Xcode 16.1
          └── Nix
              └── build-app-unsigned
                  └── Bit-identical .app
```

Until then, we document exact versions in `BUILD-INFO.txt` for each release.

## CI Artifacts

Every CI run produces:

| Artifact | Description |
|----------|-------------|
| `eidolons-app-unsigned-macos` | Unsigned .app bundle with checksums |
| `eidolons-server-macos-aarch64` | Native macOS server binary |
| `eidolons-server-linux-x86_64` | Linux x86_64 server (musl, static) |
| `eidolons-server-linux-aarch64` | Linux ARM64 server (musl, static) |

The unsigned app artifact includes:
- `Eidolons.app/` - The unsigned application bundle
- `BUILD-INFO.txt` - Build environment details
- `checksums.sha256` - SHA256 of binary and bundle

## References

- [Apple's Linker & Deterministic Builds](https://milen.me/writings/apple-linker-ld64-deterministic-builds-oso-prefix/)
- [Reproducible Builds for macOS](https://gist.github.com/pudquick/89c90421a9582f88741b21d10c6a155e)
- [SOURCE_DATE_EPOCH Specification](https://reproducible-builds.org/specs/source-date-epoch/)
- [Telegram Reproducible Builds](https://core.telegram.org/reproducible-builds)
