{
  description = "Eidolons";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    # Rust toolchain manager that respects rust-toolchain.toml
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src = {
        url = "github:rust-lang/rust-analyzer/release";
        flake = false;
      };
    };

    # Efficient Rust builds with incremental caching
    crane.url = "github:ipetkov/crane";

    # Utilities for multi-system support
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, fenix, crane, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
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

        # Source filtering - only include files needed for Rust builds
        src = pkgs.lib.cleanSourceWith {
          src = craneLib.path ./.;
          filter = path: type:
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
            || (baseName == "Package.swift");
        };

        # Common arguments for all Rust builds - ensures determinism
        commonArgs = {
          inherit src;
          strictDeps = true;

          # Deterministic build settings
          CARGO_BUILD_JOBS = "1";  # Single-threaded for reproducibility
          SOURCE_DATE_EPOCH = "0"; # Fixed timestamp

          # Network isolation during build
          CARGO_NET_OFFLINE = "true";

          # Rust flags for deterministic builds
          # Note: Nix automatically handles path remapping via build sandbox
          RUSTFLAGS = "-C debuginfo=0 -C target-cpu=generic";
        };

        # Helper to create cross-compilation pkgs with specific libc
        mkCrossSystem = targetSystem: libc:
          if system == targetSystem then pkgs
          else import nixpkgs {
            inherit system;
            crossSystem = {
              config = if targetSystem == "x86_64-linux" then "x86_64-unknown-linux-${libc}"
                       else if targetSystem == "aarch64-linux" then "aarch64-unknown-linux-${libc}"
                       else throw "Unsupported target: ${targetSystem}";
            };
          };

        # Build packages for a specific target
        mkPackagesForTarget = targetSystem: let
          # Server uses musl for static linking (containers)
          pkgsServerTarget = mkCrossSystem targetSystem "musl";
          craneLibServer = (crane.mkLib pkgsServerTarget).overrideToolchain (_: fenix.packages.${system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = rustToolchainSha256;
          });

          # Core library uses glibc for Linux (ignored on macOS where native toolchain is used)
          pkgsCoreTarget = mkCrossSystem targetSystem "gnu";
          craneLibCore = (crane.mkLib pkgsCoreTarget).overrideToolchain (_: fenix.packages.${system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = rustToolchainSha256;
          });

          # Build server dependencies (musl)
          serverArtifacts = craneLibServer.buildDepsOnly (commonArgs // {
            pname = "eidolons-server-deps";
            cargoExtraArgs = "--package eidolons-server";
          });

          # Build core dependencies (gnu)
          coreArtifacts = craneLibCore.buildDepsOnly (commonArgs // {
            pname = "eidolons-core-deps";
            cargoExtraArgs = "--package eidolons";
          });

          # Build the server binary with musl (static)
          server = craneLibServer.buildPackage (commonArgs // {
            cargoArtifacts = serverArtifacts;
            pname = "eidolons-server";
            cargoExtraArgs = "--bin eidolons-server";
          });

          # Build the core library with gnu (dynamic, app-compatible)
          core = craneLibCore.buildPackage (commonArgs // {
            cargoArtifacts = coreArtifacts;
            pname = "eidolons-core";
            cargoExtraArgs = "--lib -p eidolons";
          });

          # Build OCI (Docker) image containing the musl server
          server-oci = pkgsServerTarget.dockerTools.buildLayeredImage {
            name = "eidolons-server";
            tag = "latest";

            contents = [ server ];

            config = {
              Cmd = [ "${server}/bin/eidolons-server" ];
              Env = [
                "SOURCE_DATE_EPOCH=0"
              ];
            };

            # Reproducible timestamps
            created = "1970-01-01T00:00:00Z";
          };
        in {
          inherit server core server-oci;
        };

        # Build cargo dependencies once for checks (reused across fmt/clippy/test)
        cargoArtifacts = craneLib.buildDepsOnly (commonArgs // {
          pname = "eidolons-deps";
        });

        # Build for the current system
        nativePackages = mkPackagesForTarget system;

        # For macOS hosts, also provide cross-compiled Linux packages
        linuxPackages = if pkgs.stdenv.isDarwin then {
          x86_64-linux = mkPackagesForTarget "x86_64-linux";
          aarch64-linux = mkPackagesForTarget "aarch64-linux";
        } else {};

        # Swift/Apple-specific packages (only on macOS)
        applePackages = if pkgs.stdenv.isDarwin then {
          # Build uniffi-bindgen-swift tool
          uniffi-bindgen-swift = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            pname = "uniffi-bindgen-swift";
            cargoExtraArgs = "--bin uniffi-bindgen-swift";
          });

          # Build core library for Swift bindings generation (native macOS)
          # Note: This is separate from nativePackages.core because it shares cargoArtifacts
          # with the checks (clippy, tests, fmt), making CI builds more efficient.
          core-swift = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            pname = "eidolons-swift";
            cargoExtraArgs = "--lib -p eidolons";
          });

          # Generate Swift bindings from the core library
          swift-bindings = pkgs.stdenv.mkDerivation {
            name = "eidolons-swift-bindings";

            # Need the built library, bindgen tool, and cargo for metadata extraction
            nativeBuildInputs = [
              applePackages.uniffi-bindgen-swift
              rustToolchain  # Provides cargo for metadata extraction
            ];

            # Use same deterministic settings
            SOURCE_DATE_EPOCH = "0";

            # The generation happens in buildPhase
            dontUnpack = true;

            buildPhase = ''
              # Create output directories
              mkdir -p $out/Sources/EidolonsCore
              mkdir -p $out/Sources/EidolonsCoreFFI

              # uniffi-bindgen-swift needs access to the Cargo.toml to extract metadata
              # Copy the source tree to allow cargo metadata to work
              cp -r ${src}/* .
              chmod -R +w .

              # Find the dylib - it will be in the lib output
              DYLIB="${applePackages.core-swift}/lib/libeidolons.dylib"

              # Generate Swift bindings to a temp directory first
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
            '';

            installPhase = ''
              # Output is already in $out from buildPhase
              echo "Generated Swift bindings:"
              echo "EidolonsCore (Swift):"
              ls -la $out/Sources/EidolonsCore/
              echo "EidolonsCoreFFI (C headers):"
              ls -la $out/Sources/EidolonsCoreFFI/
            '';
          };

          # Script to build full XCFramework with all iOS targets
          # This uses system Xcode and is not fully deterministic
          build-xcframework = pkgs.writeShellScriptBin "build-xcframework" ''
            set -euo pipefail

            echo "Building XCFramework for all Apple platforms..."
            echo "Note: This requires Xcode and uses system SDKs"

            # Define architectures and their Rust targets
            declare -A TARGETS=(
              ["aarch64-apple-darwin"]="macOS (Apple Silicon)"
              ["x86_64-apple-darwin"]="macOS (Intel)"
              ["aarch64-apple-ios"]="iOS (Device)"
              ["aarch64-apple-ios-sim"]="iOS Simulator (Apple Silicon)"
              ["x86_64-apple-ios"]="iOS Simulator (Intel)"
            )

            # Build directory
            BUILD_DIR="target/apple-libs"
            XCFRAMEWORK_DIR="target/apple"
            mkdir -p "$BUILD_DIR" "$XCFRAMEWORK_DIR"

            # Build for each target
            for target in "''${!TARGETS[@]}"; do
              echo "Building for $target (''${TARGETS[$target]})..."
              cargo build --release --lib -p eidolons --target "$target"
            done

            # Create XCFramework
            echo "Creating XCFramework..."
            rm -rf "$XCFRAMEWORK_DIR/libeidolons-rs.xcframework"

            xcodebuild -create-xcframework \
              -library "target/aarch64-apple-darwin/release/libeidolons.dylib" \
              -library "target/x86_64-apple-darwin/release/libeidolons.dylib" \
              -library "target/aarch64-apple-ios/release/libeidolons.a" \
              -library "target/aarch64-apple-ios-sim/release/libeidolons.a" \
              -library "target/x86_64-apple-ios/release/libeidolons.a" \
              -output "$XCFRAMEWORK_DIR/libeidolons-rs.xcframework"

            echo "XCFramework created at: $XCFRAMEWORK_DIR/libeidolons-rs.xcframework"
            echo ""
            echo "Next steps:"
            echo "1. Swift bindings are at: core/swift/Sources/EidolonsCore/"
            echo "2. Run Swift Package Manager: cd core && swift build"
          '';

          # Script to run Swift tests (builds XCFramework first)
          swift-test = pkgs.writeShellScriptBin "swift-test" ''
            set -euo pipefail

            echo "Building macOS library..."
            cargo build --release --lib -p eidolons

            echo "Creating XCFramework for testing..."
            XCFRAMEWORK_DIR="target/apple"
            mkdir -p "$XCFRAMEWORK_DIR"
            rm -rf "$XCFRAMEWORK_DIR/libeidolons-rs.xcframework"

            xcodebuild -create-xcframework \
              -library "target/release/libeidolons.dylib" \
              -output "$XCFRAMEWORK_DIR/libeidolons-rs.xcframework"

            echo "Running Swift tests..."
            cd core
            swift test
          '';

          # Script to update Swift bindings in the source tree
          update-swift-bindings = pkgs.writeShellScriptBin "update-swift-bindings" ''
            set -euo pipefail

            echo "Generating Swift bindings..."

            # Build the core library and bindgen tool
            echo "Building core library..."
            cargo build --release --lib -p eidolons

            echo "Building uniffi-bindgen-swift..."
            cargo build --release -p uniffi-bindgen-swift

            # Find the built dylib (will be in target/release)
            DYLIB="target/release/libeidolons.dylib"

            if [ ! -f "$DYLIB" ]; then
              echo "Error: Could not find $DYLIB"
              exit 1
            fi

            # Clean output directories
            SWIFT_DIR="core/swift/Sources/EidolonsCore"
            FFI_DIR="core/swift/Sources/EidolonsCoreFFI"
            rm -rf "$SWIFT_DIR" "$FFI_DIR"
            mkdir -p "$SWIFT_DIR" "$FFI_DIR"

            # Generate bindings to temp directory
            TEMP_OUT=$(mktemp -d)
            echo "Generating Swift bindings..."
            target/release/uniffi-bindgen-swift \
              --swift-sources --headers --modulemap \
              --metadata-no-deps \
              "$DYLIB" \
              "$TEMP_OUT" \
              --module-name eidolonsFFI \
              --modulemap-filename module.modulemap

            # Move files to their proper locations
            mv "$TEMP_OUT"/*.swift "$SWIFT_DIR/"
            mv "$TEMP_OUT"/*.h "$FFI_DIR/"
            mv "$TEMP_OUT"/module.modulemap "$FFI_DIR/"
            rm -rf "$TEMP_OUT"

            echo "Swift bindings updated successfully!"
            echo ""
            echo "Generated Swift files:"
            ls -lh "$SWIFT_DIR"
            echo ""
            echo "Generated FFI headers:"
            ls -lh "$FFI_DIR"
            echo ""
            echo "Don't forget to commit these changes!"
          '';
        } else {};

      in {
        # Default packages for current system
        packages = {
          default = nativePackages.server;
          server = nativePackages.server;
          core = nativePackages.core;
          server-oci = nativePackages.server-oci;
          crane-cli = pkgs.crane;
        } // (if pkgs.stdenv.isDarwin then {
          # On macOS, provide Linux cross-compiled packages
          server-x86_64-linux = linuxPackages.x86_64-linux.server;
          server-aarch64-linux = linuxPackages.aarch64-linux.server;
          server-oci-x86_64-linux = linuxPackages.x86_64-linux.server-oci;
          server-oci-aarch64-linux = linuxPackages.aarch64-linux.server-oci;

          # Swift/Apple packages
          inherit (applePackages)
            uniffi-bindgen-swift
            core-swift
            swift-bindings
            build-xcframework
            update-swift-bindings
            swift-test;
        } else {});

        # Development shell with Rust toolchain and tools
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            cargo-watch
            rust-analyzer
            pinact  # Pin GitHub Actions to SHAs
          ] ++ (if pkgs.stdenv.isDarwin then [
            # Swift/Apple development tools
            applePackages.update-swift-bindings
            applePackages.build-xcframework
            applePackages.swift-test
          ] else []);

          # Same environment variables as builds for consistency
          SOURCE_DATE_EPOCH = "0";
          RUSTFLAGS = "-C debuginfo=0 -C target-cpu=generic";

          shellHook = if pkgs.stdenv.isDarwin then ''
            echo "Swift commands available:"
            echo "  update-swift-bindings  - Generate Swift bindings"
            echo "  build-xcframework      - Build XCFramework for all Apple platforms"
            echo "  swift-test             - Run Swift tests"
          '' else "";
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify builds work
          server-builds = nativePackages.server;
          core-builds = nativePackages.core;

          # Verify code formatting
          formatting = craneLib.cargoFmt {
            inherit (commonArgs) src;
            pname = "eidolons-fmt";
          };

          # Verify no Clippy warnings
          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            pname = "eidolons-clippy";
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

          # Run unit tests
          tests = craneLib.cargoTest (commonArgs // {
            inherit cargoArtifacts;
            pname = "eidolons-tests";
          });
        } // (if pkgs.stdenv.isDarwin then {
          # Verify committed Swift bindings match generated ones
          swift-bindings-current = pkgs.runCommand "check-swift-bindings" {
            buildInputs = [ pkgs.diffutils ];
          } ''
            echo "Checking if committed Swift bindings match generated ones..."

            # Check Swift sources
            GENERATED_SWIFT="${applePackages.swift-bindings}/Sources/EidolonsCore"
            COMMITTED_SWIFT="${src}/core/swift/Sources/EidolonsCore"

            if [ ! -d "$COMMITTED_SWIFT" ] || [ -z "$(ls -A "$COMMITTED_SWIFT" 2>/dev/null)" ]; then
              echo "ERROR: No committed Swift bindings found at core/swift/Sources/EidolonsCore/"
              echo "Run: nix run .#update-swift-bindings"
              echo "Then commit the generated files."
              exit 1
            fi

            # Check FFI headers
            GENERATED_FFI="${applePackages.swift-bindings}/Sources/EidolonsCoreFFI"
            COMMITTED_FFI="${src}/core/swift/Sources/EidolonsCoreFFI"

            if [ ! -d "$COMMITTED_FFI" ] || [ -z "$(ls -A "$COMMITTED_FFI" 2>/dev/null)" ]; then
              echo "ERROR: No committed FFI headers found at core/swift/Sources/EidolonsCoreFFI/"
              echo "Run: nix run .#update-swift-bindings"
              echo "Then commit the generated files."
              exit 1
            fi

            # Compare generated vs committed (Swift)
            if ! diff -r "$GENERATED_SWIFT" "$COMMITTED_SWIFT"; then
              echo ""
              echo "ERROR: Committed Swift bindings don't match generated ones!"
              echo ""
              echo "To fix this:"
              echo "  1. Run: nix run .#update-swift-bindings"
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
              echo "  1. Run: nix run .#update-swift-bindings"
              echo "  2. Review the changes"
              echo "  3. Commit the updated bindings"
              echo ""
              exit 1
            fi

            echo "✓ Swift bindings are up to date"
            touch $out
          '';
        } else {});
      }
    );
}
