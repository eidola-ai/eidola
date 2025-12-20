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

      # By default fenix uses rust-analyzer nightly. Uncommenting the bellow
      # will use the latest stable version instead.
      # inputs.rust-analyzer-src = {
      #   url = "github:rust-lang/rust-analyzer/release";
      #   flake = false;
      # };
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

        # Source filtering - only include files needed for Rust builds
        src = pkgs.lib.cleanSourceWith {
          src = craneLib.path ./.;
          filter =
            path: type:
            let
              baseName = builtins.baseNameOf path;
              pathStr = toString path;
            in
            # Include all Rust source files and Cargo files
            (craneLib.filterCargoSources path type)
            # Also include rust-toolchain.toml for fenix
            || (baseName == "rust-toolchain.toml")
            # Include Swift bindings and tests for checks
            || (pkgs.lib.hasInfix "/core/swift" pathStr)
            # Include Package.swift for Swift builds
            || (baseName == "Package.swift")
            # Include OpenAPI spec for checks
            || (baseName == "openapi.json");
        };

        # Common arguments for all Rust builds - ensures determinism
        commonArgs = {
          inherit src;
          strictDeps = true;

          # Deterministic build settings
          CARGO_BUILD_JOBS = "1"; # Single-threaded for reproducibility
          CARGO_INCREMENTAL = "false"; # Disable incremental compilation
          SOURCE_DATE_EPOCH = "0"; # Fixed timestamp
          ZERO_AR_DATE = "1"; # Reproducible ar/ranlib archives

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

            # Use pkgsCross if specified, otherwise native pkgs
            targetPkgs = if nixCrossSystem == null then pkgs else pkgs.pkgsCross.${nixCrossSystem};

            # Crane uses target pkgs (for linker/libc) but host toolchain (for cargo)
            craneLibTarget = (crane.mkLib targetPkgs).overrideToolchain (_: rustToolchain);

            # Cross-compilation needs CARGO_BUILD_TARGET set
            targetArgs = if isNative then { } else { CARGO_BUILD_TARGET = rustTarget; };
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

        # Build the generate-openapi binary (native only, used for spec generation)
        generateOpenapiBin = craneLib.buildPackage (
          commonArgs
          // {
            cargoArtifacts = craneLib.buildDepsOnly (
              commonArgs
              // {
                pname = "generate-openapi-deps";
                cargoExtraArgs = "--package eidolons-server";
              }
            );
            pname = "generate-openapi";
            cargoExtraArgs = "--bin generate-openapi";
          }
        );

        # Generate OpenAPI specification from the server code
        serverOpenApiSpec = pkgs.runCommand "eidolons-openapi-spec"
          {
            nativeBuildInputs = [ generateOpenapiBin ];
            SOURCE_DATE_EPOCH = "0";
          }
          ''
            mkdir -p $out
            generate-openapi > $out/openapi.json
          '';

        # Build the server binary
        # - rustTarget: Rust target triple
        # - nixCrossSystem: pkgsCross attr name, or null for native pkgs
        mkServer =
          rustTarget: nixCrossSystem:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;

            serverArtifacts = cfg.craneLibTarget.buildDepsOnly (
              commonArgs
              // cfg.targetArgs
              // {
                pname = "eidolons-server-deps--${rustTarget}";
                cargoExtraArgs = "--package eidolons-server";
              }
            );
          in
          cfg.craneLibTarget.buildPackage (
            commonArgs
            // cfg.targetArgs
            // {
              cargoArtifacts = serverArtifacts;
              pname = "eidolons-server--${rustTarget}";
              cargoExtraArgs = "--bin eidolons-server";
            }
          );

        # Build the core library dependencies and package
        # - rustTarget: Rust target triple
        # - nixCrossSystem: pkgsCross attr name, or null for native pkgs
        # - crateType: "staticlib" (default) or "cdylib"
        # - doCheck: whether to run tests (default true)
        mkCoreInternals =
          rustTarget: nixCrossSystem: crateType: doCheck:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            effectiveCrateType = if crateType == null then "staticlib" else crateType;
            effectiveDoCheck = if doCheck == null then true else doCheck;

            # Override crate-type to build only what's requested
            preBuildHook = ''
              sed -i 's/crate-type = .*/crate-type = ["${effectiveCrateType}"]/' core/Cargo.toml
            '';

            coreArtifacts = cfg.craneLibTarget.buildDepsOnly (
              commonArgs
              // cfg.targetArgs
              // {
                pname = "eidolons-core-deps--${rustTarget}";
                cargoExtraArgs = "--package eidolons";
                preBuild = preBuildHook;
                doCheck = effectiveDoCheck;
              }
            );
            corePackage = cfg.craneLibTarget.buildPackage (
              commonArgs
              // cfg.targetArgs
              // {
                cargoArtifacts = coreArtifacts;
                pname = "eidolons-core--${rustTarget}";
                cargoExtraArgs = "--lib -p eidolons";
                preBuild = preBuildHook;
                doCheck = effectiveDoCheck;
              }
            );
          in
          {
            coreArtifacts = coreArtifacts;
            corePackage = corePackage;
          };

        # Build the core library
        # - rustTarget: Rust target triple
        # - nixCrossSystem: pkgsCross attr name, or null for native pkgs
        # - crateType: "staticlib" (default) or "cdylib"
        # - doCheck: whether to run tests (default true)
        mkCore =
          rustTarget: nixCrossSystem: crateType: doCheck:
          let
            mkCoreHelperResults = mkCoreInternals rustTarget nixCrossSystem crateType doCheck;
          in
          mkCoreHelperResults.corePackage;

        # Build OCI (Docker) image containing the server
        # - rustTarget: Rust target triple
        # - nixCrossSystem: pkgsCross attr name, or null for native pkgs
        #
        # Design decisions:
        # - Distroless: No shell or package manager (minimal attack surface)
        # - Static binary: musl-linked, self-contained (no libc dependency)
        # - CA certs: Embedded via webpki-roots crate (no system certs needed)
        # - Non-root: Runs as unprivileged user (UID 65534 = nobody)
        # - Entrypoint: Server is the only thing this container does
        mkServerOCI =
          rustTarget: nixCrossSystem:
          let
            server = mkServer rustTarget nixCrossSystem;
          in
          pkgs.dockerTools.buildLayeredImage {
            name = "eidolons-server";
            tag = "latest";

            contents = [ server ];

            config = {
              Entrypoint = [ "${server}/bin/eidolons-server" ];

              # Bind to all interfaces (required for container networking)
              # ANTHROPIC_API_KEY must be provided at runtime
              Env = [
                "BIND_ADDR=0.0.0.0:8080"
              ];

              # Run as unprivileged user (nobody)
              User = "65534:65534";

              # Document the exposed port
              ExposedPorts = {
                "8080/tcp" = { };
              };
            };

            # Reproducible timestamp for deterministic builds
            created = "1970-01-01T00:00:00Z";
          };

        # Build all packages for a target
        # - rustTarget: Rust target triple
        # - nixCrossSystem: pkgsCross attr name, or null for native pkgs
        mkSystemPackages = rustTarget: nixCrossSystem: {
          server = mkServer rustTarget nixCrossSystem;
          server-oci = mkServerOCI rustTarget nixCrossSystem;
          core = mkCore rustTarget nixCrossSystem null null; # staticlib, run tests
        };

        # Build the uniffi-bindgen-swift tool (native only)
        uniffiBindgenSwift = craneLib.buildPackage (
          commonArgs
          // {
            cargoArtifacts = craneLib.buildDepsOnly (
              commonArgs
              // {
                pname = "uniffi-bindgen-swift-deps";
                cargoExtraArgs = "--package uniffi-bindgen-swift";
              }
            );
            pname = "uniffi-bindgen-swift";
            cargoExtraArgs = "--bin uniffi-bindgen-swift";
          }
        );

        # Generate Swift bindings from the core library
        mkCoreSwiftBindings = pkgs.stdenv.mkDerivation {
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
            cp -r ${src}/* .
            chmod -R +w .

            # Find the dylib (native build, cdylib for uniffi-bindgen-swift)
            DYLIB="${mkCore nativeRustTarget null "cdylib" null}/lib/libeidolons.dylib"

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
        mkCoreSwiftXCFramework = pkgs.stdenv.mkDerivation {
          name = "eidolons-xcframework";

          nativeBuildInputs = [ pkgs.darwin.cctools ]; # Provides lipo

          # Use same deterministic settings
          SOURCE_DATE_EPOCH = "0";
          ZERO_AR_DATE = "1";

          dontUnpack = true;

          # Reference all the Apple target builds (all use native pkgs, Rust handles cross-compilation)
          macosArm64 = mkCore "aarch64-apple-darwin" null "staticlib" null;
          macosX86_64 = mkCore "x86_64-apple-darwin" null "staticlib" null;
          iosArm64 = mkCore "aarch64-apple-ios" null "staticlib" false;
          iosSimArm64 = mkCore "aarch64-apple-ios-sim" null "staticlib" false;
          iosSimX86_64 = mkCore "x86_64-apple-ios" null "staticlib" false;

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

        # Cross-compilation targets: rustTarget -> nixCrossSystem (null = use native pkgs)
        crossTargets = {
          "aarch64-unknown-linux-musl" = "aarch64-multiplatform-musl";
          "x86_64-unknown-linux-musl" = "musl64";
          "aarch64-apple-darwin" = null; # Rust handles cross-compilation
          "x86_64-apple-darwin" = null; # Rust handles cross-compilation
        };

        # Flatten cross-compiled packages: { "server--aarch64-unknown-linux-musl" = ...; ... }
        crossPackages = builtins.listToAttrs (
          builtins.concatMap (
            rustTarget:
            let
              nixCrossSystem = crossTargets.${rustTarget};
              targetPkgs = mkSystemPackages rustTarget nixCrossSystem;
            in
            [
              {
                name = "server--${rustTarget}";
                value = targetPkgs.server;
              }
              {
                name = "server-oci--${rustTarget}";
                value = targetPkgs.server-oci;
              }
              {
                name = "core--${rustTarget}";
                value = targetPkgs.core;
              }
            ]
          ) (builtins.attrNames crossTargets)
        );
        packages =
          # Default packages for current system
          mkSystemPackages nativeRustTarget null
          // crossPackages
          // {
            # Swift binding generation (native only)
            core-swift-bindings = mkCoreSwiftBindings;
            core-swift-xcframework = mkCoreSwiftXCFramework;

            # OpenAPI spec generation
            server-openapi-spec = serverOpenApiSpec;
          };

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

            # Pin GitHub Actions to SHAs
            pkgs.pinact

            # Interact with OCI images
            pkgs.crane
          ];
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify code formatting
          formatting = craneLib.cargoFmt {
            inherit (commonArgs) src;
            pname = "eidolons-fmt";
          };

          # Verify no Clippy warnings
          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              cargoArtifacts = (mkCoreInternals nativeRustTarget null null null).coreArtifacts;
              pname = "eidolons-clippy";
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          # Run unit tests
          tests = craneLib.cargoTest (
            commonArgs
            // {
              cargoArtifacts = (mkCoreInternals nativeRustTarget null null null).coreArtifacts;
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
                COMMITTED_SWIFT="${src}/core/swift/Sources/EidolonsCore"

                if [ ! -d "$COMMITTED_SWIFT" ] || [ -z "$(ls -A "$COMMITTED_SWIFT" 2>/dev/null)" ]; then
                  echo "ERROR: No committed Swift bindings found at core/swift/Sources/EidolonsCore/"
                  echo "Run: nix run .#update-core-swift-bindings"
                  echo "Then commit the generated files."
                  exit 1
                fi

                # Check FFI headers
                GENERATED_FFI="${packages.core-swift-bindings}/Sources/EidolonsCoreFFI"
                COMMITTED_FFI="${src}/core/swift/Sources/EidolonsCoreFFI"

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

          # Checks that committed OpenAPI spec is up to date with the generated one
          openapi-current =
            pkgs.runCommand "check-openapi-spec"
              {
                buildInputs = [ pkgs.diffutils ];
              }
              ''
                echo "Checking if committed OpenAPI spec matches generated one..."

                GENERATED="${packages.server-openapi-spec}/openapi.json"
                COMMITTED="${src}/server/openapi.json"

                if [ ! -f "$COMMITTED" ]; then
                  echo "ERROR: No committed OpenAPI spec found at server/openapi.json"
                  echo "Run: nix run '.#update-server-openapi'"
                  echo "Then commit the generated file."
                  exit 1
                fi

                if ! diff "$GENERATED" "$COMMITTED"; then
                  echo ""
                  echo "ERROR: Committed OpenAPI spec doesn't match generated one!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-server-openapi'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated spec"
                  echo ""
                  exit 1
                fi

                echo "OpenAPI spec is up to date"
                touch $out
              '';

          # Ensure the primary artifacts are built
          build-server-oci = packages.server-oci;

        };

        apps = {
          update-core-swift-bindings = {
            type = "app";
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
          update-server-openapi = {
            type = "app";
            program = "${
              pkgs.writeShellApplication {
                name = "update-server-openapi";
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
                  dest="$repo_root/server/openapi.json"

                  echo "Copying OpenAPI spec from Nix store:"
                  echo "  source: ${packages.server-openapi-spec}/openapi.json"
                  echo "  dest:   $dest"

                  cp "${packages.server-openapi-spec}/openapi.json" "$dest"
                  chmod +w "$dest"

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/update-server-openapi";
          };
        };
      }
    );
}
