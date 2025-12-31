{
  description = "Eidolons";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    # Utilities for multi-system support
    flake-utils.url = "github:numtide/flake-utils";

    # Rust toolchain manager that respects rust-toolchain.toml
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Efficient Rust builds with incremental caching
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
      crane,
    }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # SHA256 for rust-toolchain.toml (single source of truth)
        rustToolchainSha256 = "sha256-sqSWJDUxc+zaz1nBWMAJKTAGBuGWP25GCftIOlCEAtA=";

        # Get the exact Rust toolchain specified in rust-toolchain.toml
        rustToolchain = fenix.packages.${system}.fromToolchainFile {
          file = ./rust-toolchain.toml;
          sha256 = rustToolchainSha256;
        };

        # Create crane library with our Rust toolchain (function form for consistency)
        craneLib = (crane.mkLib pkgs).overrideToolchain (_: rustToolchain);

        # Map Nix system to Rust target triple for native builds
        nativeRustTarget =
          {
            "aarch64-darwin" = "aarch64-apple-darwin";
            "x86_64-darwin" = "x86_64-apple-darwin";
            "aarch64-linux" = "aarch64-unknown-linux-musl";
            "x86_64-linux" = "x86_64-unknown-linux-musl";
          }
          .${system};

        # Full repo source for checks that compare committed vs generated files
        repoSrc = craneLib.path ./.;

        # Source filtering for Rust builds - excludes generated outputs
        rustSrc = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = pkgs.lib.fileset.unions [
            (craneLib.fileset.commonCargoSources ./.)
            ./rust-toolchain.toml
          ];
        };

        # Minimal source for dependency builds - only Cargo files
        # This ensures .rs file changes don't invalidate the deps cache
        rustDepsSrc = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = pkgs.lib.fileset.unions [
            (craneLib.fileset.cargoTomlAndLock ./.)
            ./rust-toolchain.toml
          ];
        };

        # Common arguments for all Rust builds - ensures determinism
        # Note: src is NOT included here; add it per-derivation
        commonArgs = {
          strictDeps = true;

          # Deterministic build settings
          CARGO_INCREMENTAL = "false"; # Disable incremental compilation
          SOURCE_DATE_EPOCH = "0"; # Fixed timestamp
          ZERO_AR_DATE = "1"; # Reproducible ar/ranlib archives

          # Single-threaded for reproducibility.
          # Note: Setting this to 1 causes a major hit to compilation. It
          # should have no impact on reproducibility *unless* a proc macro
          # is not designed to function deterministically. If such a case
          # emerges, this can be uncommented as a temporary workaround.
          # CARGO_BUILD_JOBS = "1";

          # Network isolation during build
          CARGO_NET_OFFLINE = "true";

          # Rust flags for deterministic builds
          # Note: Nix automatically handles path remapping via build sandbox
          RUSTFLAGS = "-C debuginfo=0 -C target-cpu=generic";
        };

        # Target configuration helper
        # Takes explicit Rust target and optional Nix cross-system (pkgsCross attr name).
        # - rustTarget: Rust target triple (e.g., "aarch64-apple-darwin")
        # - nixCrossSystem: pkgsCross attr name (e.g., "aarch64-multiplatform-musl"), or null for native pkgs
        mkTargetConfig =
          rustTarget: nixCrossSystem:
          let
            isNative = rustTarget == nativeRustTarget;
            isLinuxMusl = builtins.match ".*-linux-musl" rustTarget != null;

            # Use pkgsCross if specified, otherwise native pkgs
            targetPkgs = if nixCrossSystem == null then pkgs else pkgs.pkgsCross.${nixCrossSystem};

            # Crane uses target pkgs (for linker/libc) but host toolchain (for cargo)
            craneLibTarget = (crane.mkLib targetPkgs).overrideToolchain (_: rustToolchain);

            # Cross-compilation needs CARGO_BUILD_TARGET set.
            # For Linux musl targets without pkgsCross, use rust-lld (bundled with Rust)
            # instead of system cc. If nixCrossSystem is set, pkgsCross provides a cross-linker.
            targetArgs =
              if isNative then
                { }
              else if isLinuxMusl && nixCrossSystem == null then
                {
                  CARGO_BUILD_TARGET = rustTarget;

                  # The linker env var is dynamically generated from the target triple:
                  # "aarch64-unknown-linux-musl" -> "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER"
                  "CARGO_TARGET_${pkgs.lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] rustTarget)}_LINKER" =
                    "rust-lld";
                }
              else
                { CARGO_BUILD_TARGET = rustTarget; };
          in
          {
            inherit
              isNative
              targetPkgs
              rustTarget
              craneLibTarget
              targetArgs
              ;
          };

        # Build workspace dependencies for a target (shared across all packages)
        # Uses cargoSrc (Cargo files only) so .rs changes don't invalidate cache
        mkWorkspaceDeps =
          rustTarget: nixCrossSystem:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            # Host-only tools that should never be cross-compiled
          in
          cfg.craneLibTarget.buildDepsOnly (
            commonArgs
            // cfg.targetArgs
            // {
              src = rustDepsSrc;
              pname = "eidolons-workspace-deps--${rustTarget}";
            }
          );

        # Pre-built workspace deps for native target (shared by all native builds)
        nativeWorkspaceDeps = mkWorkspaceDeps nativeRustTarget null;

        # Build the uniffi-bindgen-swift tool (native only)
        uniffiBindgenSwift = craneLib.buildPackage (
          commonArgs
          // {
            src = rustSrc;
            cargoArtifacts = nativeWorkspaceDeps;
            pname = "uniffi-bindgen-swift";
            cargoExtraArgs = "--bin uniffi-bindgen-swift";
          }
        );

        # Generate Swift bindings from the core library
        coreSwiftBindings = pkgs.stdenv.mkDerivation {
          name = "eidolons-swift-bindings";

          nativeBuildInputs = [
            uniffiBindgenSwift
            rustToolchain # Provides cargo for metadata extraction
          ];

          # Use same deterministic settings
          SOURCE_DATE_EPOCH = "0";

          dontUnpack = true;

          buildPhase = ''
                        # Create output directories
                        mkdir -p $out/Sources/EidolonsCore
                        mkdir -p $out/Sources/EidolonsCoreFFI

                        # uniffi-bindgen-swift needs access to Cargo.toml for metadata
                        cp -r ${rustSrc}/* .
                        chmod -R +w .

                        # Find the dylib (native build, cdylib for uniffi-bindgen-swift)
                        DYLIB="${mkCore nativeRustTarget null "cdylib"}/lib/libeidolons.dylib"

                        # Generate Swift bindings to a temp directory
                        TEMP_OUT=$(mktemp -d)
                        uniffi-bindgen-swift \
                            --swift-sources --headers --modulemap \
                            --metadata-no-deps \
                            "$DYLIB" \
                            "$TEMP_OUT" \
                            --module-name eidolonsFFI \
                            --modulemap-filename module.modulemap

                        # Move files to their proper locations:
                        # - Swift source goes to EidolonsCore
                        # - Header and modulemap go to EidolonsCoreFFI
                        mv "$TEMP_OUT"/*.swift $out/Sources/EidolonsCore/
                        mv "$TEMP_OUT"/*.h $out/Sources/EidolonsCoreFFI/
                        mv "$TEMP_OUT"/module.modulemap $out/Sources/EidolonsCoreFFI/

                        # Create stub C file for SPM (requires at least one compilable source per C target)
                        cat > $out/Sources/EidolonsCoreFFI/eidolonsFFI.c << 'STUB'
            // This file exists so Swift Package Manager has something to compile for the eidolonsFFI module.
            // The actual implementation is in the XCFramework (libeidolons.a).
            // This module just exposes the C header interface to Swift.
            #include "eidolonsFFI.h"
            STUB
          '';

          installPhase = ''
            echo "Generated Swift bindings:"
            echo "EidolonsCore (Swift):"
            ls -la $out/Sources/EidolonsCore/
            echo "EidolonsCoreFFI (C headers):"
            ls -la $out/Sources/EidolonsCoreFFI/
          '';
        };

        # Build XCFramework containing static libraries for all Apple platforms
        coreSwiftXCFramework = pkgs.stdenv.mkDerivation {
          name = "eidolons-xcframework";

          nativeBuildInputs = [ pkgs.darwin.cctools ]; # Provides lipo

          # Use same deterministic settings
          SOURCE_DATE_EPOCH = "0";
          ZERO_AR_DATE = "1";

          dontUnpack = true;

          # Reference all the Apple target builds (all use native pkgs, Rust handles cross-compilation)
          macosArm64 = mkCore "aarch64-apple-darwin" null "staticlib";
          macosX86_64 = mkCore "x86_64-apple-darwin" null "staticlib";
          iosArm64 = mkCore "aarch64-apple-ios" null "staticlib";
          iosSimArm64 = mkCore "aarch64-apple-ios-sim" null "staticlib";
          iosSimX86_64 = mkCore "x86_64-apple-ios" null "staticlib";

          buildPhase = ''
            mkdir -p "$out/libeidolons-rs.xcframework/macos-arm64_x86_64"
            mkdir -p "$out/libeidolons-rs.xcframework/ios-arm64"
            mkdir -p "$out/libeidolons-rs.xcframework/ios-arm64_x86_64-simulator"

            # macOS: combine arm64 + x86_64 into universal binary
            lipo -create \
              "$macosArm64/lib/libeidolons.a" \
              "$macosX86_64/lib/libeidolons.a" \
              -output "$out/libeidolons-rs.xcframework/macos-arm64_x86_64/libeidolons.a"

            # iOS device: arm64 only
            cp "$iosArm64/lib/libeidolons.a" \
              "$out/libeidolons-rs.xcframework/ios-arm64/libeidolons.a"

            # iOS simulator: combine arm64 + x86_64 into universal binary
            lipo -create \
              "$iosSimArm64/lib/libeidolons.a" \
              "$iosSimX86_64/lib/libeidolons.a" \
              -output "$out/libeidolons-rs.xcframework/ios-arm64_x86_64-simulator/libeidolons.a"

            # Create Info.plist for XCFramework
            cat > "$out/libeidolons-rs.xcframework/Info.plist" << 'EOF'
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
              <key>AvailableLibraries</key>
              <array>
                <dict>
                  <key>LibraryIdentifier</key>
                  <string>macos-arm64_x86_64</string>
                  <key>LibraryPath</key>
                  <string>libeidolons.a</string>
                  <key>SupportedArchitectures</key>
                  <array>
                    <string>arm64</string>
                    <string>x86_64</string>
                  </array>
                  <key>SupportedPlatform</key>
                  <string>macos</string>
                </dict>
                <dict>
                  <key>LibraryIdentifier</key>
                  <string>ios-arm64</string>
                  <key>LibraryPath</key>
                  <string>libeidolons.a</string>
                  <key>SupportedArchitectures</key>
                  <array><string>arm64</string></array>
                  <key>SupportedPlatform</key>
                  <string>ios</string>
                </dict>
                <dict>
                  <key>LibraryIdentifier</key>
                  <string>ios-arm64_x86_64-simulator</string>
                  <key>LibraryPath</key>
                  <string>libeidolons.a</string>
                  <key>SupportedArchitectures</key>
                  <array>
                    <string>arm64</string>
                    <string>x86_64</string>
                  </array>
                  <key>SupportedPlatform</key>
                  <string>ios</string>
                  <key>SupportedPlatformVariant</key>
                  <string>simulator</string>
                </dict>
              </array>
              <key>CFBundlePackageType</key>
              <string>XFWK</string>
              <key>XCFrameworkFormatVersion</key>
              <string>1.0</string>
            </dict>
            </plist>
            EOF
          '';

          installPhase = ''
            echo "XCFramework contents:"
            find "$out" -name "*.a" -exec ls -lh {} \;
            echo ""
            echo "Architecture info:"
            for lib in "$out"/libeidolons-rs.xcframework/*/libeidolons.a; do
              echo "$lib:"
              lipo -info "$lib"
            done
          '';
        };

        # Build the core library
        # - rustTarget: Rust target triple
        # - nixCrossSystem: pkgsCross attr name, or null for native pkgs
        # - crateType: "staticlib" (default) or "cdylib
        mkCore =
          rustTarget: nixCrossSystem: crateType:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            effectiveCrateType = if crateType == null then "staticlib" else crateType;

            # Override crate-type to build only what's requested (only needed for package, not deps)
            preBuildHook = ''
              sed -i 's/crate-type = .*/crate-type = ["${effectiveCrateType}"]/' core/Cargo.toml
            '';

          in
          cfg.craneLibTarget.buildPackage (
            commonArgs
            // cfg.targetArgs
            // {
              src = rustSrc;
              cargoArtifacts = mkWorkspaceDeps rustTarget nixCrossSystem;
              pname = "eidolons-core--${rustTarget}";
              cargoExtraArgs = "--lib --package eidolons";
              preBuild = preBuildHook;

              # Skip tests when cross-compiling (can't run foreign binaries)
              doCheck = cfg.isNative;
            }
          );

        # Cross-compilation targets: rustTarget -> nixCrossSystem (null = use native pkgs)
        # All targets use null (Rust handles cross-compilation) because the server
        # has pure Rust dependencies with no C library requirements.
        #
        # If this changes, we can map rust targets to complete pkgsCross targets, like:
        #   "aarch64-unknown-linux-musl" = "aarch64-multiplatform-musl";
        #   "x86_64-unknown-linux-musl" = "musl64";
        crossTargets = {
          "aarch64-unknown-linux-musl" = "aarch64-multiplatform-musl";
          "x86_64-unknown-linux-musl" = "musl64";
          "aarch64-apple-darwin" = null;
          "x86_64-apple-darwin" = null;
        };

        # Flatten cross-compiled packages: { "server--aarch64-unknown-linux-musl" = ...; ... }
        crossPackages = builtins.listToAttrs (
          builtins.concatMap (
            rustTarget:
            let
              nixCrossSystem = crossTargets.${rustTarget};
            in
            [
              {
                name = "core--${rustTarget}";
                value = mkCore rustTarget nixCrossSystem null; # staticlib
              }
            ]
          ) (builtins.attrNames crossTargets)
        );
        packages = {
          core = mkCore nativeRustTarget null null; # staticlib

          # Swift binding generation (native only)
          core-swift-bindings = coreSwiftBindings;
          core-swift-xcframework = coreSwiftXCFramework;
        }
        // crossPackages;

      in
      {
        inherit packages;

        # Development shell with Rust toolchain and tools
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Rust toolchain
            rustToolchain

            # Additional rust tools
            pkgs.cargo-watch
            pkgs.rust-analyzer
          ];
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify code formatting
          formatting = craneLib.cargoFmt {
            src = rustSrc;
            pname = "eidolons-fmt";
          };

          # Verify no Clippy warnings
          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              src = rustSrc;
              cargoArtifacts = nativeWorkspaceDeps;
              pname = "eidolons-clippy";
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          # Run unit tests
          tests = craneLib.cargoTest (
            commonArgs
            // {
              src = rustSrc;
              cargoArtifacts = nativeWorkspaceDeps;
              pname = "eidolons-tests";
            }
          );

          # Checks that committed bindings are up to date with the generated ones
          bindings-current =
            pkgs.runCommand "check-swift-bindings"
              {
                buildInputs = [ pkgs.diffutils ];
              }
              ''
                echo "Checking if committed Swift bindings match generated ones..."

                # Check Swift sources
                GENERATED_SWIFT="${packages.core-swift-bindings}/Sources/EidolonsCore"
                COMMITTED_SWIFT="${repoSrc}/core/swift/Sources/EidolonsCore"

                if [ ! -d "$COMMITTED_SWIFT" ] || [ -z "$(ls -A "$COMMITTED_SWIFT" 2>/dev/null)" ]; then
                  echo "ERROR: No committed Swift bindings found at core/swift/Sources/EidolonsCore/"
                  echo "Run: nix run .#update-core-swift-bindings"
                  echo "Then commit the generated files."
                  exit 1
                fi

                # Check FFI headers
                GENERATED_FFI="${packages.core-swift-bindings}/Sources/EidolonsCoreFFI"
                COMMITTED_FFI="${repoSrc}/core/swift/Sources/EidolonsCoreFFI"

                if [ ! -d "$COMMITTED_FFI" ] || [ -z "$(ls -A "$COMMITTED_FFI" 2>/dev/null)" ]; then
                  echo "ERROR: No committed FFI headers found at core/swift/Sources/EidolonsCoreFFI/"
                  echo "Run: nix run '.#update-core-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                # Compare generated vs committed (Swift)
                if ! diff -r "$GENERATED_SWIFT" "$COMMITTED_SWIFT"; then
                  echo ""
                  echo "ERROR: Committed Swift bindings don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-core-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated bindings"
                  echo ""
                  exit 1
                fi

                # Compare generated vs committed (FFI headers)
                if ! diff -r "$GENERATED_FFI" "$COMMITTED_FFI"; then
                  echo ""
                  echo "ERROR: Committed FFI headers don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-core-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated bindings"
                  echo ""
                  exit 1
                fi

                echo "✓ Swift bindings are up to date"
                touch $out
              '';

        };

        apps = {
          update-core-swift-bindings = {
            type = "app";
            meta.description = "Update committed Swift bindings from generated sources";
            program = "${
              pkgs.writeShellApplication {
                name = "update-core-swift-bindings";
                runtimeInputs = [
                  pkgs.coreutils
                  pkgs.git
                ];

                text = ''
                  set -euo pipefail

                  # Sanity check: must run from repo root (or adjust logic)
                  if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
                    echo "error: not in a git repository" >&2
                    exit 1
                  fi

                  repo_root="$(git rev-parse --show-toplevel)"
                  dest="$repo_root/core/swift/Sources"

                  echo "Syncing Swift bindings from Nix store:"
                  echo "  source: ${packages.core-swift-bindings}"
                  echo "  dest:   $dest"

                  mkdir -p "$dest"
                  rm -rf "$dest"
                  cp -R "${packages.core-swift-bindings}/Sources" "$dest"
                  chmod -R +w "$dest"

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/update-core-swift-bindings";
          };
          update-core-swift-xcframework = {
            type = "app";
            meta.description = "Update XCFramework with compiled static libraries";
            program = "${
              pkgs.writeShellApplication {
                name = "update-core-swift-xcframework";
                runtimeInputs = [
                  pkgs.coreutils
                  pkgs.git
                ];

                text = ''
                  set -euo pipefail

                  # Sanity check: must run from repo root (or adjust logic)
                  if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
                    echo "error: not in a git repository" >&2
                    exit 1
                  fi

                  repo_root="$(git rev-parse --show-toplevel)"
                  dest="$repo_root/core/target/apple/libeidolons-rs.xcframework"

                  echo "Copying core Swift XCframework from Nix store:"
                  echo "  source: ${packages.core-swift-xcframework}"
                  echo "  dest:   $dest"

                  mkdir -p "$dest"
                  rm -rf "$dest"
                  cp -R "${packages.core-swift-xcframework}/libeidolons-rs.xcframework" "$dest"
                  chmod -R +w "$dest"

                  echo "Done."
                '';
              }
            }/bin/update-core-swift-xcframework";
          };
        };
      }
    );
}
