{
  description = "Eidolons";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    # Rust toolchain manager that respects rust-toolchain.toml
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
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
          sha256 = "sha256-SDu4snEWjuZU475PERvu+iO50Mi39KVjqCeJeNvpguU=";
        };

        # Create crane library with our Rust toolchain (function form for consistency)
        craneLib = (crane.mkLib pkgs).overrideToolchain (_: rustToolchain);

        # Source filtering - only include files needed for Rust builds
        src = pkgs.lib.cleanSourceWith {
          src = craneLib.path ./.;
          filter = path: type:
            # Include all Rust source files and Cargo files
            (craneLib.filterCargoSources path type)
            # Also include rust-toolchain.toml for fenix
            || (builtins.baseNameOf path == "rust-toolchain.toml");
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
            sha256 = "sha256-SDu4snEWjuZU475PERvu+iO50Mi39KVjqCeJeNvpguU=";
          });

          # Core library uses gnu for compatibility with apps
          pkgsCoreTarget = mkCrossSystem targetSystem "gnu";
          craneLibCore = (crane.mkLib pkgsCoreTarget).overrideToolchain (_: fenix.packages.${system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = "sha256-SDu4snEWjuZU475PERvu+iO50Mi39KVjqCeJeNvpguU=";
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

        # Build for the current system
        nativePackages = mkPackagesForTarget system;

        # For macOS hosts, also provide cross-compiled Linux packages
        linuxPackages = if pkgs.stdenv.isDarwin then {
          x86_64-linux = mkPackagesForTarget "x86_64-linux";
          aarch64-linux = mkPackagesForTarget "aarch64-linux";
        } else {};

      in {
        # Default packages for current system
        packages = {
          default = nativePackages.server;
          server = nativePackages.server;
          core = nativePackages.core;
          server-oci = nativePackages.server-oci;
        } // (if pkgs.stdenv.isDarwin then {
          # On macOS, provide Linux cross-compiled packages
          server-x86_64-linux = linuxPackages.x86_64-linux.server;
          server-aarch64-linux = linuxPackages.aarch64-linux.server;
          server-oci-x86_64-linux = linuxPackages.x86_64-linux.server-oci;
          server-oci-aarch64-linux = linuxPackages.aarch64-linux.server-oci;
        } else {});

        # Development shell with Rust toolchain and tools
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            cargo-watch
            rust-analyzer
          ];

          # Same environment variables as builds for consistency
          SOURCE_DATE_EPOCH = "0";
          RUSTFLAGS = "-C debuginfo=0 -C target-cpu=generic";
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify builds work
          server-builds = nativePackages.server;
          core-builds = nativePackages.core;
        };
      }
    );
}
