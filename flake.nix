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
        rustToolchainSha256 = "sha256-SBKjxhC6zHTu0SyJwxLlQHItzMzYZ71VCWQC2hOzpRY=";

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

          # Build tools required by native dependencies (e.g., mlx-sys-burn)
          nativeBuildInputs = [ pkgs.cmake ];

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

            # Linker env var name for this target (dynamically generated from target triple)
            linkerEnvVar = "CARGO_TARGET_${
              pkgs.lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] rustTarget)
            }_LINKER";

            # Cross-compilation needs CARGO_BUILD_TARGET set.
            # For Linux musl targets without pkgsCross, use rust-lld (bundled with Rust).
            targetArgs =
              if isNative then
                { }
              else if isLinuxMusl && nixCrossSystem == null then
                {
                  CARGO_BUILD_TARGET = rustTarget;
                  ${linkerEnvVar} = "rust-lld";
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
          {
            pname,
            rustTarget,
            nixCrossSystem,
            extraCargoArgs ? "",
          }:
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
              pname = "${pname}-deps";
              # Only build deps for this specific package
              cargoExtraArgs = "-p ${pname} ${extraCargoArgs}";
            }
          );

        # Build individual packages with their own filtered deps
        mkPackage =
          {
            pname,
            rustTarget,
            nixCrossSystem,
            crateType ? null,
            extraCargoArgs ? "",
          }:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            deps = packageDeps.${pname} or [ ];
            relevantCrates = [ pname ] ++ deps;
            filteredSrc = mkFilteredSrc relevantCrates;
            packageCargoArtifacts = mkPackageDeps {
              inherit pname rustTarget nixCrossSystem extraCargoArgs;
            };
            cratePath = cratePaths.${pname};
            crateTypeSetup =
              if crateType == null then
                { }
              else
                {
                  preBuildHook = ''
                    sed -i 's/crate-type = .*/crate-type = ["${crateType}"]/' ${cratePath}/Cargo.toml
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
              cargoExtraArgs = "-p ${pname} ${extraCargoArgs}";
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

        # Build the uniffi-bindgen-swift tool (native only)
        uniffiBindgenSwift = mkPackage {
          pname = "uniffi-bindgen-swift";
          rustTarget = nativeRustTarget;
          nixCrossSystem = null;
        };

        # Build the shared-typegen tool (native only)
        sharedTypegen = mkPackage {
          pname = "shared-typegen";
          rustTarget = nativeRustTarget;
          nixCrossSystem = null;
        };

        # Generate Swift types from the shared core using Crux typegen
        eidolonsSharedSwiftTypes = pkgs.stdenv.mkDerivation {
          name = "eidolons-shared-swift-types";

          nativeBuildInputs = [ sharedTypegen ];

          SOURCE_DATE_EPOCH = "0";

          dontUnpack = true;

          buildPhase = ''
            # Use NIX_BUILD_TOP as temp directory
            TEMP_OUT="$NIX_BUILD_TOP/typegen-output"
            mkdir -p "$TEMP_OUT"

            # Run typegen - ignore errors about Package.swift, we'll create our own
            export RUST_BACKTRACE=1
            shared-typegen "$TEMP_OUT" 2>&1 || true

            # Check if the Swift types were generated
            if [ ! -f "$TEMP_OUT/SharedTypes/Sources/SharedTypes/SharedTypes.swift" ]; then
              echo "Failed to generate Swift types"
              find "$TEMP_OUT" -type f 2>/dev/null || true
              exit 1
            fi

            # The typegen might fail trying to write Package.swift but we don't need it
            # since we embed the types directly in our package
            # Remove the generated Package.swift if it exists (we use our own)
            rm -f "$TEMP_OUT/SharedTypes/Package.swift"

            # Move to $out
            mkdir -p $out
            cp -r "$TEMP_OUT"/* $out/
          '';

          installPhase = ''
            echo "Generated Swift types:"
            find $out -type f -name "*.swift" | head -20
          '';
        };

        # Generate Swift bindings from the shared core library (UniFFI)
        eidolonsSharedSwiftBindings = pkgs.stdenv.mkDerivation {
          name = "eidolons-shared-swift-bindings";

          nativeBuildInputs = [
            uniffiBindgenSwift
            rustToolchain
          ];

          SOURCE_DATE_EPOCH = "0";

          dontUnpack = true;

          buildPhase = ''
            # Create output directories
            mkdir -p $out/Sources/EidolonsShared
            mkdir -p $out/Sources/EidolonsSharedFFI

            # uniffi-bindgen-swift needs access to Cargo.toml for metadata
            cp -r ${mkFilteredSrc ([ "eidolons-shared" ] ++ packageDeps.eidolons-shared or [ ])}/* .
            chmod -R +w .

            # Find the dylib (native build, cdylib for uniffi-bindgen-swift)
            # Built without default features (MLX) since bindings only need the FFI interface
            DYLIB="${
              mkPackage {
                pname = "eidolons-shared";
                rustTarget = nativeRustTarget;
                nixCrossSystem = null;
                crateType = "cdylib";
                extraCargoArgs = "--no-default-features";
              }
            }/lib/libeidolons_shared.dylib"

            # Generate Swift bindings to a temp directory
            TEMP_OUT=$(mktemp -d)
            uniffi-bindgen-swift \
                --swift-sources --headers --modulemap \
                --metadata-no-deps \
                "$DYLIB" \
                "$TEMP_OUT" \
                --module-name eidolons_sharedFFI \
                --modulemap-filename module.modulemap

            # Move files to their proper locations
            mv "$TEMP_OUT"/*.swift $out/Sources/EidolonsShared/
            mv "$TEMP_OUT"/*.h $out/Sources/EidolonsSharedFFI/
            mv "$TEMP_OUT"/module.modulemap $out/Sources/EidolonsSharedFFI/

            # Create stub C file for SPM
            cat > $out/Sources/EidolonsSharedFFI/eidolons_sharedFFI.c << 'STUB'
            // This file exists so Swift Package Manager has something to compile for the eidolons_sharedFFI module.
            // The actual implementation is in the XCFramework (libeidolons_shared.a).
            // This module just exposes the C header interface to Swift.
            #include "eidolons_sharedFFI.h"
            STUB
          '';

          installPhase = ''
            echo "Generated Swift bindings:"
            echo "EidolonsShared (Swift):"
            ls -la $out/Sources/EidolonsShared/
            echo "EidolonsSharedFFI (C headers):"
            ls -la $out/Sources/EidolonsSharedFFI/
          '';
        };

        # Build XCFramework for eidolons-shared
        eidolonsSharedSwiftXCFramework = pkgs.stdenv.mkDerivation {
          name = "eidolons-shared-xcframework";

          nativeBuildInputs = [ pkgs.darwin.cctools ];

          SOURCE_DATE_EPOCH = "0";
          ZERO_AR_DATE = "1";

          dontUnpack = true;

          macosArm64 = mkPackage {
            pname = "eidolons-shared";
            rustTarget = "aarch64-apple-darwin";
            nixCrossSystem = null;
            crateType = "staticlib";
          };
          macosX86_64 = mkPackage {
            pname = "eidolons-shared";
            rustTarget = "x86_64-apple-darwin";
            nixCrossSystem = null;
            crateType = "staticlib";
          };

          buildPhase = ''
            XCFW="$out/libeidolons_shared-rs.xcframework"
            MACOS_DIR="$XCFW/macos-arm64_x86_64"
            mkdir -p "$MACOS_DIR"

            lipo -create \
              "$macosArm64/lib/libeidolons_shared.a" \
              "$macosX86_64/lib/libeidolons_shared.a" \
              -output "$MACOS_DIR/libeidolons_shared.a"

            cat > "$XCFW/Info.plist" << 'EOF'
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
                  <string>libeidolons_shared.a</string>
                  <key>SupportedArchitectures</key>
                  <array>
                    <string>arm64</string>
                    <string>x86_64</string>
                  </array>
                  <key>SupportedPlatform</key>
                  <string>macos</string>
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
            find "$out" -type f -exec ls -lh {} \;
            echo ""
            echo "Architecture info:"
            lipo -info "$out/libeidolons_shared-rs.xcframework/macos-arm64_x86_64/libeidolons_shared.a"
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
          server-openapi-spec = serverOpenApiSpec;

          # Shared core Swift binding generation
          eidolons-shared-swift-types = eidolonsSharedSwiftTypes;
          eidolons-shared-swift-bindings = eidolonsSharedSwiftBindings;
          eidolons-shared-swift-xcframework = eidolonsSharedSwiftXCFramework;
        };

        # Development shell (lightweight — daily Rust dev uses rustup)
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Pin GitHub actions
            pkgs.pinact
          ];
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Verify Rust code formatting
          rust-formatting = craneLib.cargoFmt {
            src = fullSrc;
            pname = "rust-fmt";
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
                COMMITTED="${repoSrc}/crates/eidolons-server/openapi.json"

                if [ ! -f "$COMMITTED" ]; then
                  echo "ERROR: No committed OpenAPI spec found at crates/eidolons-server/openapi.json"
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

          # Checks that committed Swift bindings are up to date with generated ones
          swift-bindings-current =
            pkgs.runCommand "check-swift-bindings"
              {
                buildInputs = [ pkgs.diffutils ];
              }
              ''
                echo "Checking if committed Swift bindings match generated ones..."

                # Check EidolonsShared Swift files
                GENERATED_SWIFT="${self.packages.${system}.eidolons-shared-swift-bindings}/Sources/EidolonsShared"
                COMMITTED_SWIFT="${repoSrc}/crates/eidolons-shared/swift/Sources/EidolonsShared"

                if [ ! -d "$COMMITTED_SWIFT" ]; then
                  echo "ERROR: No committed Swift bindings found at crates/eidolons-shared/swift/Sources/EidolonsShared"
                  echo "Run: nix run '.#update-eidolons-shared-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                if ! diff -r "$GENERATED_SWIFT" "$COMMITTED_SWIFT"; then
                  echo ""
                  echo "ERROR: Committed EidolonsShared Swift bindings don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-eidolons-shared-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated bindings"
                  echo ""
                  exit 1
                fi

                # Check EidolonsSharedFFI C headers
                GENERATED_FFI="${self.packages.${system}.eidolons-shared-swift-bindings}/Sources/EidolonsSharedFFI"
                COMMITTED_FFI="${repoSrc}/crates/eidolons-shared/swift/Sources/EidolonsSharedFFI"

                if [ ! -d "$COMMITTED_FFI" ]; then
                  echo "ERROR: No committed FFI headers found at crates/eidolons-shared/swift/Sources/EidolonsSharedFFI"
                  echo "Run: nix run '.#update-eidolons-shared-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                if ! diff -r "$GENERATED_FFI" "$COMMITTED_FFI"; then
                  echo ""
                  echo "ERROR: Committed EidolonsSharedFFI headers don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-eidolons-shared-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated headers"
                  echo ""
                  exit 1
                fi

                # Check SharedTypes (Crux typegen)
                GENERATED_TYPES="${self.packages.${system}.eidolons-shared-swift-types}/SharedTypes"
                COMMITTED_TYPES="${repoSrc}/crates/eidolons-shared/swift/generated/SharedTypes"

                if [ ! -d "$COMMITTED_TYPES" ]; then
                  echo "ERROR: No committed SharedTypes found at crates/eidolons-shared/swift/generated/SharedTypes"
                  echo "Run: nix run '.#update-eidolons-shared-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                if ! diff -r "$GENERATED_TYPES" "$COMMITTED_TYPES"; then
                  echo ""
                  echo "ERROR: Committed SharedTypes don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-eidolons-shared-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated types"
                  echo ""
                  exit 1
                fi

                echo "Swift bindings are up to date"
                touch $out
              '';

          # Verify Swift formatting for all Swift files in the repo
          swift-formatting =
            pkgs.runCommand "check-swift-formatting"
              {
                nativeBuildInputs = [ pkgs.swift-format pkgs.findutils ];
              }
              ''
                echo "Checking Swift formatting..."

                # Find all Swift files, excluding:
                # - crates/eidolons-shared/swift/Sources/EidolonsShared (auto-generated bindings)
                # - crates/eidolons-shared/swift/generated (auto-generated Crux types)
                # - Any .build directories (SwiftPM build artifacts)
                # Note: .git is already excluded by crane's source filtering
                find ${repoSrc} \
                  -path '*/crates/eidolons-shared/swift/Sources/EidolonsShared' -prune -o \
                  -path '*/crates/eidolons-shared/swift/generated' -prune -o \
                  -path '*/.build' -prune -o \
                  -name '*.swift' -print0 \
                  | xargs -0 -r swift-format lint --strict

                echo "✓ Swift files are properly formatted"
                touch $out
              '';
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
                  ${./scripts/update-server-openapi.sh} "${self.packages.${system}.server-openapi-spec}/openapi.json"
                '';
              }
            }/bin/update-server-openapi";
          };

          update-eidolons-shared-swift-bindings = {
            type = "app";
            meta.description = "Update committed Swift bindings for shared core";
            program = "${
              pkgs.writeShellApplication {
                name = "update-eidolons-shared-swift-bindings";
                runtimeInputs = [
                  pkgs.coreutils
                  pkgs.git
                ];

                text = ''
                  ${./scripts/update-shared-bindings.sh} \
                    "${self.packages.${system}.eidolons-shared-swift-bindings}/Sources" \
                    "${self.packages.${system}.eidolons-shared-swift-types}"
                '';
              }
            }/bin/update-eidolons-shared-swift-bindings";
          };

          update-eidolons-shared-swift-xcframework = {
            type = "app";
            meta.description = "Update XCFramework for shared core";
            program = "${
              pkgs.writeShellApplication {
                name = "update-eidolons-shared-swift-xcframework";
                runtimeInputs = [
                  pkgs.coreutils
                  pkgs.git
                ];

                text = ''
                  ${./scripts/update-shared-xcframework.sh} "${self.packages.${system}.eidolons-shared-swift-xcframework}"
                '';
              }
            }/bin/update-eidolons-shared-swift-xcframework";
          };

          format-rust = {
            type = "app";
            meta.description = "Format all Rust files in the repo";
            program = "${
              pkgs.writeShellApplication {
                name = "format-rust";
                runtimeInputs = [
                  rustToolchain
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
                  cd "$repo_root"

                  echo "Formatting Rust files..."
                  cargo fmt

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/format-rust";
          };

          format-swift = {
            type = "app";
            meta.description = "Format all Swift files in the repo (excludes auto-generated bindings)";
            program = "${
              pkgs.writeShellApplication {
                name = "format-swift";
                runtimeInputs = [
                  pkgs.swift-format
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
                  cd "$repo_root"

                  echo "Formatting Swift files (excluding auto-generated bindings)..."

                  # Use git ls-files to respect .gitignore, exclude auto-generated bindings
                  git ls-files '*.swift' \
                    | grep -v '^crates/eidolons-shared/swift/Sources/EidolonsShared/' \
                    | grep -v '^crates/eidolons-shared/swift/generated/' \
                    | xargs -r swift-format format --in-place

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/format-swift";
          };
        };
      }
    );
}
