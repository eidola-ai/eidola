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
      ...
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

        # Parse workspace Cargo.toml to get member patterns
        workspaceCargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        memberPatterns = workspaceCargoToml.workspace.members or [ ];

        # Expand member patterns (handles both explicit paths and globs like "crates/*")
        expandMemberPattern =
          pattern:
          let
            # Check if pattern ends with /*
            globMatch = builtins.match "(.+)/\\*" pattern;
          in
          if globMatch != null then
            # It's a glob pattern like "crates/*" - list the directory
            let
              baseDir = builtins.head globMatch;
            in
            map (name: "${baseDir}/${name}") (
              builtins.attrNames (
                pkgs.lib.filterAttrs (_: type: type == "directory") (builtins.readDir ./${baseDir})
              )
            )
          else
            # Explicit path
            [ pattern ];

        # Get all workspace member paths (e.g., ["crates/foo", "crates/bar"])
        workspaceMemberPaths = builtins.concatMap expandMemberPattern memberPatterns;

        # Map from crate name to its path
        cratePaths = builtins.listToAttrs (
          map (path: {
            name = builtins.baseNameOf path;
            value = path;
          }) workspaceMemberPaths
        );

        # List of crate names
        workspaceCrates = builtins.attrNames cratePaths;

        # Parse a crate's Cargo.toml and extract workspace dependencies
        # (dependencies with path = "../<crate>" pointing to sibling crates)
        getWorkspaceDeps =
          pname:
          let
            cratePath = cratePaths.${pname};
            cargoToml = builtins.fromTOML (builtins.readFile ./${cratePath}/Cargo.toml);
            deps = cargoToml.dependencies or { };
            # Filter to only path dependencies that point to workspace crates
            workspaceDeps = pkgs.lib.filterAttrs (
              name: spec: builtins.isAttrs spec && spec ? path && builtins.elem name workspaceCrates
            ) deps;
          in
          builtins.attrNames workspaceDeps;

        # Recursively resolve all transitive workspace dependencies
        getAllDeps =
          pname:
          let
            directDeps = getWorkspaceDeps pname;
            transitiveDeps = builtins.concatMap getAllDeps directDeps;
          in
          pkgs.lib.unique (directDeps ++ transitiveDeps);

        # Build the full dependency graph (auto-discovered from Cargo.toml files)
        packageDeps = builtins.listToAttrs (
          map (pname: {
            name = pname;
            value = getAllDeps pname;
          }) workspaceCrates
        );

        # Create filtered source that only includes specific crates
        mkFilteredSrc =
          crates:
          let
            crateSet = builtins.listToAttrs (
              map (c: {
                name = c;
                value = true;
              }) crates
            );
            # Find which crate (if any) a path belongs to
            getCrateForPath =
              relPath:
              pkgs.lib.findFirst (
                name:
                let
                  p = cratePaths.${name};
                in
                relPath == "/${p}" || pkgs.lib.hasPrefix "/${p}/" relPath
              ) null workspaceCrates;
            # Check if path is a parent directory of any workspace crate
            isParentOfCrate =
              relPath: pkgs.lib.any (path: pkgs.lib.hasPrefix "${relPath}/" "/${path}") workspaceMemberPaths;

            # Filter source files
            filteredSrc = pkgs.lib.cleanSourceWith {
              src = ./.;
              filter =
                path: type:
                let
                  relPath = pkgs.lib.removePrefix (toString ./.) (toString path);
                  # Which crate does this path belong to?
                  matchingCrate = getCrateForPath relPath;
                  # Is this an irrelevant crate? (in a crate dir but not in our set)
                  isIrrelevantCrate = matchingCrate != null && !(crateSet ? ${matchingCrate});
                in
                # Exclude irrelevant crate directories entirely
                if isIrrelevantCrate then
                  false
                # Keep root-level files (except Cargo.toml which we'll replace)
                else if type == "regular" && builtins.match "/[^/]+" relPath != null then
                  true
                # Keep directories that are parents of crate paths
                else if type == "directory" && isParentOfCrate relPath then
                  true
                # For everything else, use crane's filter (which handles .rs, Cargo.toml, etc.)
                else
                  craneLib.filterCargoSources path type;
            };

            # Generate a Cargo.toml with only the relevant members listed
            filteredCargoTomlContent = (pkgs.formats.toml { }).generate "Cargo.toml" (
              workspaceCargoToml
              // {
                workspace = workspaceCargoToml.workspace // {
                  members = map (c: cratePaths.${c}) crates;
                };
              }
            );
          in
          # Combine filtered source with the modified Cargo.toml
          pkgs.runCommand "filtered-workspace-${builtins.concatStringsSep "-" crates}" { } ''
            cp -r ${filteredSrc} $out
            chmod -R u+w $out
            cp ${filteredCargoTomlContent} $out/Cargo.toml
          '';

        # Full repo source for checks that compare committed vs generated files
        repoSrc = craneLib.path ./.;

        # Full source for workspace-wide operations
        fullSrc = craneLib.cleanCargoSource ./.;

        # Common arguments for all Rust builds - ensures determinism
        # Note: src is NOT included here; add it per-derivation
        commonArgs = {
          strictDeps = true;

          # GNU tar needed for crane's reproducible archive creation (BSD tar lacks --sort)
          nativeBuildInputs = [ pkgs.gnutar ];

          # On Darwin, use Nix's hermetic Apple SDK instead of relying on host Xcode.
          # This ensures builds work regardless of host's xcode-select configuration.
          buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.apple-sdk ];

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
            isIOS = builtins.match ".*-apple-ios.*" rustTarget != null;

            # Use pkgsCross if specified, otherwise native pkgs
            targetPkgs = if nixCrossSystem == null then pkgs else pkgs.pkgsCross.${nixCrossSystem};

            # Crane uses target pkgs (for linker/libc) but host toolchain (for cargo)
            craneLibTarget = (crane.mkLib targetPkgs).overrideToolchain (_: rustToolchain);

            # Map iOS targets to their SDK names
            iosSdkName =
              if rustTarget == "aarch64-apple-ios" then
                "iphoneos"
              else if rustTarget == "aarch64-apple-ios-sim" || rustTarget == "x86_64-apple-ios" then
                "iphonesimulator"
              else
                null;

            # Linker env var name for this target (dynamically generated from target triple)
            linkerEnvVar = "CARGO_TARGET_${
              pkgs.lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] rustTarget)
            }_LINKER";

            # Cross-compilation needs CARGO_BUILD_TARGET set.
            # For Linux musl targets without pkgsCross, use rust-lld (bundled with Rust).
            # For iOS targets, use system clang and Xcode SDK (requires sandbox=false).
            targetArgs =
              if isNative then
                { }
              else if isLinuxMusl && nixCrossSystem == null then
                {
                  CARGO_BUILD_TARGET = rustTarget;
                  ${linkerEnvVar} = "rust-lld";
                }
              else if isIOS then
                {
                  CARGO_BUILD_TARGET = rustTarget;
                  ${linkerEnvVar} = "/usr/bin/clang";
                  preBuild = ''
                    export PATH="${pkgs.gnutar}/bin:$PATH:/usr/bin"
                    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
                    export SDKROOT="$(/usr/bin/xcrun --sdk ${iosSdkName} --show-sdk-path)"
                    echo "Using SDK: $SDKROOT"
                  '';
                  doCheck = false; # iOS binaries can't run on macOS
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

        # Build dependencies separately for caching (uses full workspace)
        # This is used for workspace-wide operations like clippy and tests
        cargoArtifacts = craneLib.buildDepsOnly (
          commonArgs
          // {
            src = fullSrc;
            pname = "workspace";
          }
        );

        # Build per-package dependencies (only compiles deps that package needs)
        mkPackageDeps =
          pname: rustTarget: nixCrossSystem:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            deps = packageDeps.${pname} or [ ];
            relevantCrates = [ pname ] ++ deps;
            filteredSrc = mkFilteredSrc relevantCrates;
          in
          cfg.craneLibTarget.buildDepsOnly (
            commonArgs
            // cfg.targetArgs
            // {
              src = filteredSrc;
              inherit pname;
              # Only build deps for this specific package
              cargoExtraArgs = "-p ${pname}";
            }
          );

        # Build individual packages with their own filtered deps
        mkPackage =
          {
            pname,
            rustTarget,
            nixCrossSystem,
            crateType ? null,
          }:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            deps = packageDeps.${pname} or [ ];
            relevantCrates = [ pname ] ++ deps;
            filteredSrc = mkFilteredSrc relevantCrates;
            packageCargoArtifacts = mkPackageDeps pname rustTarget nixCrossSystem;
            crateTypeSetup =
              if crateType == null then
                { }
              else
                {
                  preBuildHook = ''
                    sed -i 's/crate-type = .*/crate-type = ["${crateType}"]/' eidolons/Cargo.toml
                  '';
                };
          in
          cfg.craneLibTarget.buildPackage (
            commonArgs
            // cfg.targetArgs
            // crateTypeSetup
            // {
              src = filteredSrc;
              cargoArtifacts = packageCargoArtifacts;
              inherit pname;
              cargoExtraArgs = "-p ${pname}";
            }
          );

        # Build the generate-openapi binary (native only, used for spec generation)
        generateOpenapiBin = mkPackage {
          pname = "generate-openapi";
          rustTarget = nativeRustTarget;
          nixCrossSystem = null;
        };

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
            server = mkPackage {
              pname = "eidolons-server";
              inherit rustTarget nixCrossSystem;
            };
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

        # Build the uniffi-bindgen-swift tool (native only)
        uniffiBindgenSwift = mkPackage {
          pname = "uniffi-bindgen-swift";
          rustTarget = nativeRustTarget;
          nixCrossSystem = null;
        };

        # Generate Swift bindings from the core library
        eidolonsSwiftBindings = pkgs.stdenv.mkDerivation {
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
                        cp -r ${mkFilteredSrc ([ "eidolons" ] ++ packageDeps.eidolons or [ ])}/* .
                        chmod -R +w .

                        # Find the dylib (native build, cdylib for uniffi-bindgen-swift)
                        DYLIB="${
                          mkPackage {
                            pname = "eidolons";
                            rustTarget = nativeRustTarget;
                            nixCrossSystem = null;
                            crateType = "cdylib";
                          }
                        }/lib/libeidolons.dylib"

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
        # Note: iOS targets require system Xcode SDK (sandbox=false in nix.conf).
        eidolonsSwiftXCFramework = pkgs.stdenv.mkDerivation {
          name = "eidolons-xcframework";

          nativeBuildInputs = [ pkgs.darwin.cctools ]; # Provides lipo

          # Use same deterministic settings
          SOURCE_DATE_EPOCH = "0";
          ZERO_AR_DATE = "1";

          dontUnpack = true;

          # Reference all the Apple target builds
          macosArm64 = mkPackage {
            pname = "eidolons";
            rustTarget = "aarch64-apple-darwin";
            nixCrossSystem = null;
            crateType = "staticlib";
          };
          macosX86_64 = mkPackage {
            pname = "eidolons";
            rustTarget = "x86_64-apple-darwin";
            nixCrossSystem = null;
            crateType = "staticlib";
          };
          iosArm64 = mkPackage {
            pname = "eidolons";
            rustTarget = "aarch64-apple-ios";
            nixCrossSystem = null;
            crateType = "staticlib";
          };
          iosSimArm64 = mkPackage {
            pname = "eidolons";
            rustTarget = "aarch64-apple-ios-sim";
            nixCrossSystem = null;
            crateType = "staticlib";
          };
          iosSimX86_64 = mkPackage {
            pname = "eidolons";
            rustTarget = "x86_64-apple-ios";
            nixCrossSystem = null;
            crateType = "staticlib";
          };

          buildPhase = ''
            export PATH="$PATH:/usr/bin"

            # Create universal binaries for multi-arch slices
            WORKDIR=$(mktemp -d)

            # macOS: combine arm64 + x86_64
            lipo -create \
              "$macosArm64/lib/libeidolons.a" \
              "$macosX86_64/lib/libeidolons.a" \
              -output "$WORKDIR/libeidolons-macos.a"

            # iOS simulator: combine arm64 + x86_64
            lipo -create \
              "$iosSimArm64/lib/libeidolons.a" \
              "$iosSimX86_64/lib/libeidolons.a" \
              -output "$WORKDIR/libeidolons-ios-sim.a"

            # Use xcodebuild to create the XCFramework (handles Info.plist automatically)
            xcodebuild -create-xcframework \
              -library "$WORKDIR/libeidolons-macos.a" \
              -library "$iosArm64/lib/libeidolons.a" \
              -library "$WORKDIR/libeidolons-ios-sim.a" \
              -output "$out/libeidolons-rs.xcframework"
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

      in
      {
        packages = {
          server = mkPackage {
            pname = "eidolons-server";
            rustTarget = nativeRustTarget;
            nixCrossSystem = null;
          };
          server-oci = mkServerOCI nativeRustTarget null;
          server-openapi-spec = serverOpenApiSpec;

          # Server binaries cross-compiled for specific targets
          server--aarch64-unknown-linux-musl = mkPackage {
            pname = "eidolons-server";
            rustTarget = "aarch64-unknown-linux-musl";
            nixCrossSystem = "aarch64-multiplatform-musl";
          };
          server--x86_64-unknown-linux-musl = mkPackage {
            pname = "eidolons-server";
            rustTarget = "x86_64-unknown-linux-musl";
            nixCrossSystem = "musl64";
          };
          server--aarch64-apple-darwin = mkPackage {
            pname = "eidolons-server";
            rustTarget = "aarch64-apple-darwin";
            nixCrossSystem = "aarch64-darwin";
          };
          server--x86_64-apple-darwin = mkPackage {
            pname = "eidolons-server";
            rustTarget = "x86_64-apple-darwin";
            nixCrossSystem = "x86_64-darwin";
          };

          # Server OCI images cross-compiled for specific targets
          server-oci--aarch64-unknown-linux-musl = mkServerOCI "aarch64-unknown-linux-musl" "aarch64-multiplatform-musl";
          server-oci--x86_64-unknown-linux-musl = mkServerOCI "x86_64-unknown-linux-musl" "musl64";

          # Swift binding generation (native only)
          eidolons-swift-bindings = eidolonsSwiftBindings;
          eidolons-swift-xcframework = eidolonsSwiftXCFramework;
        };

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

            # Pin GitHub actions
            pkgs.pinact
          ];
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify code formatting
          formatting = craneLib.cargoFmt {
            src = fullSrc;
            pname = "fmt";
          };

          # Verify no Clippy warnings
          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              src = fullSrc;
              inherit cargoArtifacts;
              pname = "clippy";
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          # Run unit tests
          tests = craneLib.cargoTest (
            commonArgs
            // {
              src = fullSrc;
              inherit cargoArtifacts;
              pname = "tests";
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

                GENERATED="${self.packages.${system}.server-openapi-spec}/openapi.json"
                COMMITTED="${repoSrc}/eidolons-server/openapi.json"

                if [ ! -f "$COMMITTED" ]; then
                  echo "ERROR: No committed OpenAPI spec found at eidolons-server/openapi.json"
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

          # Checks that committed bindings are up to date with the generated ones
          bindings-current =
            pkgs.runCommand "check-swift-bindings"
              {
                buildInputs = [ pkgs.diffutils ];
              }
              ''
                echo "Checking if committed Swift bindings match generated ones..."

                # Check Swift sources
                GENERATED_SWIFT="${self.packages.${system}.eidolons-swift-bindings}/Sources/EidolonsCore"
                COMMITTED_SWIFT="${repoSrc}/eidolons/swift/Sources/EidolonsCore"

                if [ ! -d "$COMMITTED_SWIFT" ] || [ -z "$(ls -A "$COMMITTED_SWIFT" 2>/dev/null)" ]; then
                  echo "ERROR: No committed Swift bindings found at eidolons/swift/Sources/EidolonsCore/"
                  echo "Run: nix run .#update-eidolons-swift-bindings"
                  echo "Then commit the generated files."
                  exit 1
                fi

                # Check FFI headers
                GENERATED_FFI="${self.packages.${system}.eidolons-swift-bindings}/Sources/EidolonsCoreFFI"
                COMMITTED_FFI="${repoSrc}/eidolons/swift/Sources/EidolonsCoreFFI"

                if [ ! -d "$COMMITTED_FFI" ] || [ -z "$(ls -A "$COMMITTED_FFI" 2>/dev/null)" ]; then
                  echo "ERROR: No committed FFI headers found at eidolons/swift/Sources/EidolonsCoreFFI/"
                  echo "Run: nix run '.#update-eidolons-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                # Compare generated vs committed (Swift)
                if ! diff -r "$GENERATED_SWIFT" "$COMMITTED_SWIFT"; then
                  echo ""
                  echo "ERROR: Committed Swift bindings don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-eidolons-swift-bindings'"
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
                  echo "  1. Run: nix run '.#update-eidolons-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated bindings"
                  echo ""
                  exit 1
                fi

                echo "✓ Swift bindings are up to date"
                touch $out
              '';

          # Ensure the primary artifacts are built
          builds-server-oci = self.packages.${system}.server-oci;
          # Note: XCFramework requires sandbox=false for iOS builds (uses system Xcode SDK)
          builds-eidolons-swift-xcframework = self.packages.${system}.eidolons-swift-xcframework;

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
                  dest="$repo_root/eidolons-server/openapi.json"

                  echo "Copying OpenAPI spec from Nix store:"
                  echo "  source: ${self.packages.${system}.server-openapi-spec}/openapi.json"
                  echo "  dest:   $dest"

                  cp "${self.packages.${system}.server-openapi-spec}/openapi.json" "$dest"
                  chmod +w "$dest"

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/update-server-openapi";
          };

          update-eidolons-swift-bindings = {
            type = "app";
            meta.description = "Update committed Swift bindings from generated sources";
            program = "${
              pkgs.writeShellApplication {
                name = "update-eidolons-swift-bindings";
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
                  dest="$repo_root/eidolons/swift/Sources"

                  echo "Syncing Swift bindings from Nix store:"
                  echo "  source: ${self.packages.${system}.eidolons-swift-bindings}"
                  echo "  dest:   $dest"

                  mkdir -p "$dest"
                  rm -rf "$dest"
                  cp -R "${self.packages.${system}.eidolons-swift-bindings}/Sources" "$dest"
                  chmod -R +w "$dest"

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/update-eidolons-swift-bindings";
          };

          update-eidolons-swift-xcframework = {
            type = "app";
            meta.description = "Update XCFramework with compiled static libraries";
            program = "${
              pkgs.writeShellApplication {
                name = "update-eidolons-swift-xcframework";
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
                  dest="$repo_root/eidolons/target/apple/libeidolons-rs.xcframework"

                  echo "Copying core Swift XCframework from Nix store:"
                  echo "  source: ${self.packages.${system}.eidolons-swift-xcframework}"
                  echo "  dest:   $dest"

                  mkdir -p "$dest"
                  rm -rf "$dest"
                  cp -R "${self.packages.${system}.eidolons-swift-xcframework}/libeidolons-rs.xcframework" "$dest"
                  chmod -R +w "$dest"

                  echo "Done."
                '';
              }
            }/bin/update-eidolons-swift-xcframework";
          };
        };
      }
    );
}
