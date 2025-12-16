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

        # Get the exact Rust toolchain specified in rust-toolchain.toml
        rustToolchain = fenix.packages.${system}.fromToolchainFile {
          file = ./rust-toolchain.toml;
          sha256 = "sha256-sqSWJDUxc+zaz1nBWMAJKTAGBuGWP25GCftIOlCEAtA=";
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
              rootPath = toString ./.;
            in
            # Include all Rust source files and Cargo files
            (craneLib.filterCargoSources path type)
            # Also include rust-toolchain.toml for fenix
            || (baseName == "rust-toolchain.toml")
            # Include Swift bindings directory for checks
            || (pkgs.lib.hasInfix "/core/swift" pathStr);
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
            sha256 = "sha256-sqSWJDUxc+zaz1nBWMAJKTAGBuGWP25GCftIOlCEAtA=";
          });

          # Core library uses gnu for compatibility with apps
          pkgsCoreTarget = mkCrossSystem targetSystem "gnu";
          craneLibCore = (crane.mkLib pkgsCoreTarget).overrideToolchain (_: fenix.packages.${system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = "sha256-sqSWJDUxc+zaz1nBWMAJKTAGBuGWP25GCftIOlCEAtA=";
          });

          # Build server dependencies (musl)
          serverArtifacts = craneLibServer.buildDepsOnly (commonArgs // {
            pname = "eidolons-server-deps";
          });

          # Build core dependencies (gnu)
          coreArtifacts = craneLibCore.buildDepsOnly (commonArgs // {
            pname = "eidolons-core-deps";
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

          # Build core library as cdylib for Swift (native macOS architecture)
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
              mkdir -p $out/Sources/EidolonsCore

              # uniffi-bindgen-swift needs access to the Cargo.toml to extract metadata
              # Copy the source tree to allow cargo metadata to work
              cp -r ${src}/* .
              chmod -R +w .

              # Find the dylib - it will be in the lib output
              DYLIB="${applePackages.core-swift}/lib/libeidolons.dylib"

              # Generate Swift bindings
              # Note: Must specify what to generate (--swift-sources, --headers, --modulemap)
              # Then: library_path out_dir [options]
              # Use --metadata-no-deps to avoid network access during metadata extraction
              uniffi-bindgen-swift \
                --swift-sources --headers --modulemap \
                --metadata-no-deps \
                "$DYLIB" \
                $out/Sources/EidolonsCore \
                --module-name EidolonsCoreFFI \
                --modulemap-filename module.modulemap
            '';

            installPhase = ''
              # Output is already in $out from buildPhase
              echo "Generated Swift bindings:"
              ls -la $out/Sources/EidolonsCore/
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

          # Script to update Swift bindings in the source tree
          update-swift-bindings = pkgs.writeShellScriptBin "update-swift-bindings" ''
            set -euo pipefail

            echo "Generating Swift bindings..."

            # Build the core library and bindgen tool
            echo "Building core library..."
            cargo build --release --lib -p eidolons

            echo "Building uniffi-bindgen-swift..."
            cargo build --release --bin uniffi-bindgen-swift

            # Find the built dylib (will be in target/release)
            DYLIB="target/release/libeidolons.dylib"

            if [ ! -f "$DYLIB" ]; then
              echo "Error: Could not find $DYLIB"
              exit 1
            fi

            # Clean output directory
            OUTPUT_DIR="core/swift/Sources/EidolonsCore"
            rm -rf "$OUTPUT_DIR"
            mkdir -p "$OUTPUT_DIR"

            # Generate bindings
            echo "Generating Swift bindings to $OUTPUT_DIR..."
            target/release/uniffi-bindgen-swift \
              --swift-sources --headers --modulemap \
              "$DYLIB" \
              "$OUTPUT_DIR" \
              --module-name EidolonsCoreFFI \
              --modulemap-filename module.modulemap

            echo "Swift bindings updated successfully!"
            echo ""
            echo "Generated files:"
            ls -lh "$OUTPUT_DIR"
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
            update-swift-bindings;
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
          ] else []);

          # Same environment variables as builds for consistency
          SOURCE_DATE_EPOCH = "0";
          RUSTFLAGS = "-C debuginfo=0 -C target-cpu=generic";

          shellHook = if pkgs.stdenv.isDarwin then ''
            echo "Swift bindings commands available:"
            echo "  update-swift-bindings  - Generate Swift bindings"
            echo "  build-xcframework      - Build XCFramework for all Apple platforms"
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

            # Copy generated bindings to temp location
            GENERATED="${applePackages.swift-bindings}/Sources/EidolonsCore"
            COMMITTED="${src}/core/swift/Sources/EidolonsCore"

            # Check if committed bindings exist
            if [ ! -d "$COMMITTED" ] || [ -z "$(ls -A "$COMMITTED" 2>/dev/null)" ]; then
              echo "ERROR: No committed Swift bindings found at core/swift/Sources/EidolonsCore/"
              echo "Run: nix run .#update-swift-bindings"
              echo "Then commit the generated files."
              exit 1
            fi

            # Compare generated vs committed
            if ! diff -r "$GENERATED" "$COMMITTED"; then
              echo ""
              echo "ERROR: Committed Swift bindings don't match generated ones!"
              echo ""
              echo "The Swift bindings in core/swift/Sources/EidolonsCore/ are out of date."
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
