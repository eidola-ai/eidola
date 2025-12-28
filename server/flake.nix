{
  description = "Eidolons Server";

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
    flake-utils.lib.eachSystem [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ] (
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

        # Build the generate-openapi binary (native only, used for spec generation)
        generateOpenapiBin = craneLib.buildPackage (
          commonArgs
          // {
            src = rustSrc;
            cargoArtifacts = nativeWorkspaceDeps;
            pname = "generate-openapi";
            cargoExtraArgs = "--bin generate-openapi";
          }
        );

        # Generate OpenAPI specification from the server code
        serverOpenApiSpec =
          pkgs.runCommand "eidolons-openapi-spec"
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
          in
          cfg.craneLibTarget.buildPackage (
            commonArgs
            // cfg.targetArgs
            // {
              cargoArtifacts = mkWorkspaceDeps rustTarget nixCrossSystem;
              src = rustSrc;
              pname = "eidolons-server--${rustTarget}";
              cargoExtraArgs = "--bin eidolons-server";

              # Skip tests when cross-compiling (can't run foreign binaries)
              doCheck = cfg.isNative;
            }
          );

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

        # Cross-compilation targets: rustTarget -> nixCrossSystem (null = use native pkgs)
        crossTargets = {
          "aarch64-unknown-linux-musl" = "aarch64-multiplatform-musl";
          "x86_64-unknown-linux-musl" = "musl64";
          "aarch64-apple-darwin" = "aarch64-darwin";
          "x86_64-apple-darwin" = "x86_64-darwin";
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
                name = "server--${rustTarget}";
                value = mkServer rustTarget nixCrossSystem;
              }
              {
                name = "server-oci--${rustTarget}";
                value = mkServerOCI rustTarget nixCrossSystem;
              }
            ]
          ) (builtins.attrNames crossTargets)
        );
        packages = {

          server = mkServer nativeRustTarget null;
          server-oci = mkServerOCI nativeRustTarget null;

          # OpenAPI spec generation
          server-openapi-spec = serverOpenApiSpec;
        }
        # Add cross-compiled targets
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

            # Interact with OCI images
            pkgs.crane
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

          # Checks that committed OpenAPI spec is up to date with the generated one
          openapi-current =
            pkgs.runCommand "check-openapi-spec"
              {
                buildInputs = [ pkgs.diffutils ];
              }
              ''
                echo "Checking if committed OpenAPI spec matches generated one..."

                GENERATED="${packages.server-openapi-spec}/openapi.json"
                COMMITTED="${repoSrc}/server/openapi.json"

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
          update-server-openapi = {
            type = "app";
            meta.description = "Update committed OpenAPI spec from generated sources";
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
