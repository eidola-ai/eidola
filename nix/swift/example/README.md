# SwiftUI Minimal

A minimal proof-of-concept for truly reproducible macOS apps using SwiftUI and direct `swiftc` compilation.

See also [`../swiftui-package`](../swiftui-package) for an alternative approach using Swift Package Manager that supports both Xcode development and Nix builds.

## Approach

This variant uses `swiftc` directly rather than Swift Package Manager. This provides:

- Maximum control over compiler flags
- No SPM resolution step or `.build` artifacts
- Simpler Nix integration
- No Xcode/SPM compatibility (Nix-only)

## Nix

We're using Nix to provide a hermetic toolchain. For the initial iteration, we are using `apple-sdk_14` rather than `apple-sdk_26`, as the latter requires Swift 6 which is not yet available via nix (see [this issue](https://github.com/NixOS/nixpkgs/issues/343210)).

## Signing and Verification

The build produces an unsigned `.app` bundle with bit-for-bit reproducibility. For distribution, signing and notarization are handled separately:

1. **Build** with Nix (reproducible, unsigned)
2. **Sign** with the Xcode toolchain (`codesign`)
3. **Notarize** via Apple (required for distribution since macOS Catalina)

### Verifying a signed distribution

Code signing modifies the Mach-O binary structure (adds load commands, padding, signature data). Running `codesign --remove-signature` removes the signature but doesn't perfectly restore the original structure. To verify a distributed binary:

```bash
# Build from source
nix build .#default
cp -r result/Applications/MinimalSwiftUI.app ./local-build.app

# Strip signatures from both (normalizes structure)
codesign --remove-signature ./local-build.app
codesign --remove-signature ./distributed.app

# Remove notarization artifacts
xattr -cr ./distributed.app

# Compare
diff -r ./local-build.app ./distributed.app
```

Stripping both binaries—even the nominally unsigned local build—ensures structural normalization for comparison.

### Notes

- **Entitlements** and **Hardened Runtime** are applied at signing time, not build time, so they don't affect reproducibility
- The `Info.plist` contains only static values; adding timestamps or build numbers would break reproducibility
