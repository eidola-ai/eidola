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
            || (baseName == "Package.swift");
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

        # Target system configuration helper
        # Provides pkgsCross mapping, Rust target triple, and crane setup for a target
        mkTargetConfig =
          targetSystem:
          let
            isNative = targetSystem == system;

            # Map system identifier to pkgsCross attribute name
            # Native builds use pkgs directly; cross builds need pkgsCross
            crossPkgsAttr =
              {
                "aarch64-darwin" = "aarch64-darwin";
                "x86_64-darwin" = "x86_64-darwin";
                "aarch64-linux" = "aarch64-multiplatform-musl"; # musl for static linking
                "x86_64-linux" = "musl64"; # musl for static linking
              }
              .${targetSystem} or (throw "Unknown target system: ${targetSystem}");

            targetPkgs = if isNative then pkgs else pkgs.pkgsCross.${crossPkgsAttr};

            # Map system identifier to Rust target triple
            rustTarget =
              {
                "aarch64-darwin" = "aarch64-apple-darwin";
                "x86_64-darwin" = "x86_64-apple-darwin";
                "aarch64-linux" = "aarch64-unknown-linux-musl";
                "x86_64-linux" = "x86_64-unknown-linux-musl";
              }
              .${targetSystem} or (throw "Unknown target system: ${targetSystem}");

            # Crane uses target pkgs (for linker/libc) but host toolchain (for cargo)
            craneLibTarget = (crane.mkLib targetPkgs).overrideToolchain (_: rustToolchain);

            # Cross-compilation needs CARGO_BUILD_TARGET set
            targetArgs = if isNative then { } else { CARGO_BUILD_TARGET = rustTarget; };
          in
          {
            inherit isNative targetPkgs rustTarget craneLibTarget targetArgs;
          };

        # Build the server binary for a target system
        mkServer =
          targetSystem:
          let
            cfg = mkTargetConfig targetSystem;

            serverArtifacts = cfg.craneLibTarget.buildDepsOnly (
              commonArgs
              // cfg.targetArgs
              // {
                pname = "eidolons-server-deps-${targetSystem}";
                cargoExtraArgs = "--package eidolons-server";
              }
            );
          in
          cfg.craneLibTarget.buildPackage (
            commonArgs
            // cfg.targetArgs
            // {
              cargoArtifacts = serverArtifacts;
              pname = "eidolons-server-${targetSystem}";
              cargoExtraArgs = "--bin eidolons-server";
            }
          );

        # Build the core library as a static lib for a target system
        mkCore =
          targetSystem:
          let
            cfg = mkTargetConfig targetSystem;

            coreArtifacts = cfg.craneLibTarget.buildDepsOnly (
              commonArgs
              // cfg.targetArgs
              // {
                pname = "eidolons-core-deps-${targetSystem}";
                cargoExtraArgs = "--package eidolons";
              }
            );
          in
          cfg.craneLibTarget.buildPackage (
            commonArgs
            // cfg.targetArgs
            // {
              cargoArtifacts = coreArtifacts;
              pname = "eidolons-core-${targetSystem}";
              cargoExtraArgs = "--lib -p eidolons";
            }
          );

        # Build OCI (Docker) image containing the server
        mkServerOCI =
          targetSystem:
          let
            server = mkServer targetSystem;
          in
          pkgs.dockerTools.buildLayeredImage {
            name = "eidolons-server";
            tag = "latest";

            contents = [ server ];

            config = {
              Cmd = [ "${server}/bin/eidolons-server" ];
              Env = [ "SOURCE_DATE_EPOCH=0" ];
            };

            # Reproducible timestamp for deterministic builds
            created = "1970-01-01T00:00:00Z";
          };

        mkSystemPackages = targetSystem: {
          server = mkServer targetSystem;
          server-oci = mkServerOCI targetSystem;
          core = mkCore targetSystem;
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

            # Find the dylib
            DYLIB="${mkCore system}/lib/libeidolons.dylib"

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
          '';

          installPhase = ''
            echo "Generated Swift bindings:"
            echo "EidolonsCore (Swift):"
            ls -la $out/Sources/EidolonsCore/
            echo "EidolonsCoreFFI (C headers):"
            ls -la $out/Sources/EidolonsCoreFFI/
          '';
        };

      in
      {
        # Default packages for current system
        packages = mkSystemPackages system // {
          # Swift binding generation (native only)
          core-swift-bindings = mkCoreSwiftBindings;
          # TODO: xcframework depends on cross-compiled apple targets

          # Cross-compiled packages for other target systems
          # Access via: nix build '.#cross.<target>.server'
          cross = builtins.listToAttrs (
            map
              (targetSystem: {
                name = targetSystem;
                value = mkSystemPackages targetSystem;
              })
              [
                # Linux targets (for containers/servers)
                "aarch64-linux"
                "x86_64-linux"

                # Other macOS arch
                "x86_64-darwin"
              ]
          );
        };

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
          ];
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify code formatting
          formatting = craneLib.cargoFmt {
            inherit (commonArgs) src;
            pname = "eidolons-fmt";
          };
        };
      }
    );
}
