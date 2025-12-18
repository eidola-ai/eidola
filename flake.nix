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

        mkSystemPackages = system: {
          server = mkServer system;
          server-oci = mkServerOCI system;
          core = mkCore system;
        };

      in
      {
        # Default packages for current system
        packages = {
          # Packages built using the native system.
          inherit mkSystemPackages system;
          cors-swift-bindings = mkCoresSwiftBindings;
          cors-swift-xcframework = mkCoresSwiftXCFramework system; # depends on all cross-compiled apple targets

          cross = builtins.listToAttrs (
            map
              (t: {
                name = t;
                value = mkSystemPackages pkgs.pkgsCross.${t};
              })
              [
                # Linux targets
                "aarch64-linux"
                "x86_64-linux"

                # Apple targets
                "aarch64-apple-darwin"
                "x86_64-apple-darwin"
                "aarch64-apple-ios"
                "aarch64-apple-ios-sim"
                "x86_64-apple-ios"
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
