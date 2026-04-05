{
  description = "Eidola";

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

    # Hermetic Swift 6.2 toolchain (macOS ARM64)
    swift.url = "path:./nix/swift";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
      crane,
      swift,
      ...
    }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ] (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # SHA256 for rust-toolchain.toml (single source of truth)
        rustToolchainSha256 = "sha256-qqF33vNuAdU5vua96VKVIwuc43j4EFeEXbjQ6+l4mO4=";

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

        # Map from Cargo package name to its path
        cratePaths = builtins.listToAttrs (
          map (path: {
            name = (builtins.fromTOML (builtins.readFile ./${path}/Cargo.toml)).package.name;
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
            rootBuildFiles = {
              "Cargo.lock" = true;
              "rust-toolchain.toml" = true;
            };
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
                  baseName = builtins.baseNameOf path;
                  # Which crate does this path belong to?
                  matchingCrate = getCrateForPath relPath;
                  # Is this an irrelevant crate? (in a crate dir but not in our set)
                  isIrrelevantCrate = matchingCrate != null && !(crateSet ? ${matchingCrate});
                in
                # Exclude irrelevant crate directories entirely
                if isIrrelevantCrate then
                  false
                # Keep only root-level files that affect Cargo resolution/builds.
                # This avoids generated files like artifact-manifest.json from
                # perturbing package hashes for unrelated Nix builds.
                else if type == "regular" && builtins.match "/[^/]+" relPath != null then
                  builtins.hasAttr baseName rootBuildFiles
                # Keep directories that are parents of crate paths
                else if type == "directory" && isParentOfCrate relPath then
                  true
                # Include .sql files (used by include_str! in the CLI)
                else if type == "regular" && pkgs.lib.hasSuffix ".sql" path then
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

        # Swift toolchain (from the swift flake input) — used for formatting and compilation
        swiftPkg = swift.packages.${system}.swift or null;

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
              inherit
                pname
                rustTarget
                nixCrossSystem
                extraCargoArgs
                ;
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

        # Build the generate-openapi binary crate (native only, used for spec generation)
        generateOpenapiBin = mkPackage {
          pname = "generate-openapi";
          rustTarget = nativeRustTarget;
          nixCrossSystem = null;
        };

        # Generate OpenAPI specification from the server code
        serverOpenApiSpec =
          pkgs.runCommand "eidola-openapi-spec"
            {
              nativeBuildInputs = [ generateOpenapiBin ];
              SOURCE_DATE_EPOCH = "0";
            }
            ''
              mkdir -p $out
              generate-openapi > $out/openapi.json
            '';

        # Build the uniffi-bindgen-swift binary crate (native only)
        uniffiBindgenSwift = mkPackage {
          pname = "uniffi-bindgen-swift";
          rustTarget = nativeRustTarget;
          nixCrossSystem = null;
        };

        # Generate Swift bindings from the app core library (UniFFI)
        eidolaAppCoreSwiftBindings = pkgs.stdenv.mkDerivation {
          name = "eidola-app-core-swift-bindings";

          nativeBuildInputs = [
            uniffiBindgenSwift
            rustToolchain
          ];

          SOURCE_DATE_EPOCH = "0";

          dontUnpack = true;

          buildPhase = ''
            # Create output directories
            mkdir -p $out/Sources/EidolaAppCore
            mkdir -p $out/Sources/EidolaAppCoreFFI

            # uniffi-bindgen-swift needs access to Cargo.toml for metadata
            cp -r ${mkFilteredSrc ([ "eidola-app-core" ] ++ packageDeps.eidola-app-core or [ ])}/* .
            chmod -R +w .

            # Find the dylib (native build, cdylib for uniffi-bindgen-swift)
            DYLIB="${
              mkPackage {
                pname = "eidola-app-core";
                rustTarget = nativeRustTarget;
                nixCrossSystem = null;
                crateType = "cdylib";
                extraCargoArgs = "--no-default-features";
              }
            }/lib/libeidola_app_core.dylib"

            # Generate Swift bindings to a temp directory
            TEMP_OUT=$(mktemp -d)
            uniffi-bindgen-swift \
                --swift-sources --headers --modulemap \
                --metadata-no-deps \
                "$DYLIB" \
                "$TEMP_OUT" \
                --module-name eidola_app_coreFFI \
                --modulemap-filename module.modulemap

            # Move files to their proper locations
            mv "$TEMP_OUT"/*.swift $out/Sources/EidolaAppCore/
            mv "$TEMP_OUT"/*.h $out/Sources/EidolaAppCoreFFI/
            mv "$TEMP_OUT"/module.modulemap $out/Sources/EidolaAppCoreFFI/

            # Format generated Swift so it passes lint checks
            ${swiftPkg}/bin/swift format --in-place $out/Sources/EidolaAppCore/*.swift

            # Create stub C file for SPM
            cat > $out/Sources/EidolaAppCoreFFI/eidola_app_coreFFI.c << 'STUB'
            // This file exists so Swift Package Manager has something to compile for the eidola_app_coreFFI module.
            // The actual implementation is in the XCFramework (libeidola_app_core.a).
            // This module just exposes the C header interface to Swift.
            #include "eidola_app_coreFFI.h"
            STUB
          '';

          installPhase = ''
            echo "Generated Swift bindings:"
            echo "EidolaAppCore (Swift):"
            ls -la $out/Sources/EidolaAppCore/
            echo "EidolaAppCoreFFI (C headers):"
            ls -la $out/Sources/EidolaAppCoreFFI/
          '';
        };

        # Build XCFramework for eidola-app-core
        eidolaAppCoreSwiftXCFramework = pkgs.stdenv.mkDerivation {
          name = "eidola-app-core-xcframework";

          nativeBuildInputs = [ pkgs.darwin.cctools ];

          SOURCE_DATE_EPOCH = "0";
          ZERO_AR_DATE = "1";

          dontUnpack = true;

          macosArm64 = mkPackage {
            pname = "eidola-app-core";
            rustTarget = "aarch64-apple-darwin";
            nixCrossSystem = null;
            crateType = "staticlib";
          };
          macosX86_64 = mkPackage {
            pname = "eidola-app-core";
            rustTarget = "x86_64-apple-darwin";
            nixCrossSystem = null;
            crateType = "staticlib";
          };

          buildPhase = ''
            XCFW="$out/libeidola_app_core-rs.xcframework"
            MACOS_DIR="$XCFW/macos-arm64_x86_64"
            mkdir -p "$MACOS_DIR"

            lipo -create \
              "$macosArm64/lib/libeidola_app_core.a" \
              "$macosX86_64/lib/libeidola_app_core.a" \
              -output "$MACOS_DIR/libeidola_app_core.a"

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
                  <string>libeidola_app_core.a</string>
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
            lipo -info "$out/libeidola_app_core-rs.xcframework/macos-arm64_x86_64/libeidola_app_core.a"
          '';
        };

        # Build the CLI as a macOS universal binary (Darwin only)
        eidolaCliMacosUniversal =
          if !pkgs.stdenv.isDarwin then
            null
          else
            pkgs.stdenv.mkDerivation {
              pname = "eidola-cli-macos-universal";
              version = "1.0";

              nativeBuildInputs = [ pkgs.darwin.cctools ];

              SOURCE_DATE_EPOCH = "0";

              dontUnpack = true;

              arm64 = mkPackage {
                pname = "eidola-cli";
                rustTarget = "aarch64-apple-darwin";
                nixCrossSystem = null;
              };
              x86_64 = mkPackage {
                pname = "eidola-cli";
                rustTarget = "x86_64-apple-darwin";
                nixCrossSystem = null;
              };

              buildPhase = ''
                mkdir -p $out/bin
                lipo -create \
                  "$arm64/bin/eidola" \
                  "$x86_64/bin/eidola" \
                  -output "$out/bin/eidola"
              '';

              installPhase = ''
                echo "Universal binary:"
                lipo -info "$out/bin/eidola"
              '';

              meta = {
                description = "Eidola CLI (macOS universal binary)";
                platforms = [
                  "aarch64-darwin"
                  "x86_64-darwin"
                ];
              };
            };

        # Generate AppIcon.icns from the xcassets appiconset PNGs using libicns.
        # Returns null if no icon images are present (all slots empty).
        appIcon =
          let
            appiconset = ./apps/macos/Sources/EidolaEntrypoint/Assets.xcassets/AppIcon.appiconset;
            contentsJson = builtins.fromJSON (builtins.readFile (appiconset + "/Contents.json"));
            # An image entry has a "filename" key when a PNG is assigned
            hasImages = builtins.any (img: img ? filename) contentsJson.images;
          in
          if !hasImages then
            null
          else
            pkgs.runCommand "app-icon"
              {
                nativeBuildInputs = [ pkgs.libicns ];
                SOURCE_DATE_EPOCH = "0";
              }
              ''
                mkdir -p $out

                # Map xcassets size+scale to the pixel sizes png2icns expects.
                # png2icns auto-detects icon type from PNG dimensions.
                ${builtins.concatStringsSep "\n" (
                  builtins.filter (s: s != "") (
                    map (
                      img: if img ? filename then "cp ${appiconset}/${img.filename} $TMPDIR/${img.filename}" else ""
                    ) contentsJson.images
                  )
                )}

                png2icns $out/AppIcon.icns $TMPDIR/*.png
              '';

        # Build the macOS app with Swift 6.2 (Darwin only)
        # Uses swiftc directly (not SPM) to avoid xcrun/Xcode dependency.
        # Modules are compiled in dependency order, then linked into the final executable.
        eidolaMacosApp =
          let
            isDarwin = pkgs.stdenv.isDarwin;
          in
          if !isDarwin || swiftPkg == null then
            null
          else
            pkgs.stdenv.mkDerivation {
              pname = "eidola-macos";
              version = "1.0";

              src = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
                  ./apps/macos/Sources
                  ./apps/macos/Support/Info.plist
                  ./crates/eidola-app-core/swift
                ];
              };

              nativeBuildInputs = [ swiftPkg ];
              buildInputs = [ pkgs.apple-sdk_26 ];

              SOURCE_DATE_EPOCH = "0";

              # XCFramework static library (nix-built)
              xcframework = eidolaAppCoreSwiftXCFramework;

              buildPhase = ''
                export HOME=$TMPDIR
                export XDG_CACHE_HOME=$TMPDIR

                SHARED="crates/eidola-app-core/swift"
                XCFW_LIB="$xcframework/libeidola_app_core-rs.xcframework/macos-arm64_x86_64"
                FFI_HEADERS="$SHARED/Sources/EidolaAppCoreFFI"
                MODULEMAP="$FFI_HEADERS/module.modulemap"

                MODULES="$TMPDIR/modules"
                OBJS="$TMPDIR/objs"
                mkdir -p "$MODULES" "$OBJS"

                COMMON_FLAGS=(
                  -whole-module-optimization -parse-as-library
                  -enable-upcoming-feature MemberImportVisibility
                  -Xfrontend -no-serialize-debugging-options
                )

                echo "Building EidolaAppCore module..."
                swiftc -c "''${COMMON_FLAGS[@]}" \
                  -module-name EidolaAppCore \
                  -emit-module-path "$MODULES/EidolaAppCore.swiftmodule" \
                  -I "$MODULES" \
                  -I "$FFI_HEADERS" \
                  -Xcc -fmodule-map-file="$MODULEMAP" \
                  -o "$OBJS/EidolaAppCore.o" \
                  $(find "$SHARED/Sources/EidolaAppCore" -name '*.swift' | sort)

                echo "Building EidolaApp module..."
                swiftc -c "''${COMMON_FLAGS[@]}" \
                  -module-name EidolaApp \
                  -emit-module-path "$MODULES/EidolaApp.swiftmodule" \
                  -I "$MODULES" \
                  -I "$FFI_HEADERS" \
                  -Xcc -fmodule-map-file="$MODULEMAP" \
                  -o "$OBJS/EidolaApp.o" \
                  $(find "apps/macos/Sources/Eidola" -name '*.swift' | sort)

                echo "Linking Eidola..."
                swiftc \
                  -o Eidola \
                  -module-name EidolaEntrypoint \
                  -I "$MODULES" \
                  -I "$FFI_HEADERS" \
                  -Xcc -fmodule-map-file="$MODULEMAP" \
                  -L "$XCFW_LIB" -leidola_app_core \
                  -framework SwiftUI -framework AppKit -framework Foundation \
                  -framework SystemConfiguration \
                  -Xfrontend -no-serialize-debugging-options \
                  -Xlinker -reproducible \
                  -enable-upcoming-feature MemberImportVisibility \
                  "$OBJS/EidolaAppCore.o" "$OBJS/EidolaApp.o" \
                  apps/macos/Sources/EidolaEntrypoint/main.swift
              '';

              installPhase = ''
                APP="$out/Applications/Eidola.app"
                mkdir -p "$APP/Contents/MacOS"
                mkdir -p "$APP/Contents/Resources"

                cp Eidola "$APP/Contents/MacOS/Eidola"
                cp apps/macos/Support/Info.plist "$APP/Contents/"

                ${if appIcon != null then ''cp ${appIcon}/AppIcon.icns "$APP/Contents/Resources/"'' else ""}

                mkdir -p $out/bin
                ln -s "$APP/Contents/MacOS/Eidola" $out/bin/Eidola
              '';

              meta = {
                description = "Eidola macOS application";
                platforms = [
                  "aarch64-darwin"
                  "x86_64-darwin"
                ];
              };
            };

      in
      {
        packages = {
          server = mkPackage {
            pname = "eidola-server";
            rustTarget = nativeRustTarget;
            nixCrossSystem = null;
          };
          server-openapi-spec = serverOpenApiSpec;

          # App core Swift binding generation
          eidola-app-core-swift-bindings = eidolaAppCoreSwiftBindings;
          eidola-app-core-swift-xcframework = eidolaAppCoreSwiftXCFramework;
        }
        // pkgs.lib.optionalAttrs (eidolaCliMacosUniversal != null) {
          eidola-cli-macos-universal = eidolaCliMacosUniversal;
        }
        // pkgs.lib.optionalAttrs (eidolaMacosApp != null) {
          eidola-macos-app = eidolaMacosApp;
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
                COMMITTED="${repoSrc}/crates/eidola-server/openapi.json"

                if [ ! -f "$COMMITTED" ]; then
                  echo "ERROR: No committed OpenAPI spec found at crates/eidola-server/openapi.json"
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

                # Check EidolaAppCore Swift files
                GENERATED_SWIFT="${self.packages.${system}.eidola-app-core-swift-bindings}/Sources/EidolaAppCore"
                COMMITTED_SWIFT="${repoSrc}/crates/eidola-app-core/swift/Sources/EidolaAppCore"

                if [ ! -d "$COMMITTED_SWIFT" ]; then
                  echo "ERROR: No committed Swift bindings found at crates/eidola-app-core/swift/Sources/EidolaAppCore"
                  echo "Run: nix run '.#update-eidola-app-core-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                if ! diff -r "$GENERATED_SWIFT" "$COMMITTED_SWIFT"; then
                  echo ""
                  echo "ERROR: Committed EidolaAppCore Swift bindings don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-eidola-app-core-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated bindings"
                  echo ""
                  exit 1
                fi

                # Check EidolaAppCoreFFI C headers
                GENERATED_FFI="${self.packages.${system}.eidola-app-core-swift-bindings}/Sources/EidolaAppCoreFFI"
                COMMITTED_FFI="${repoSrc}/crates/eidola-app-core/swift/Sources/EidolaAppCoreFFI"

                if [ ! -d "$COMMITTED_FFI" ]; then
                  echo "ERROR: No committed FFI headers found at crates/eidola-app-core/swift/Sources/EidolaAppCoreFFI"
                  echo "Run: nix run '.#update-eidola-app-core-swift-bindings'"
                  echo "Then commit the generated files."
                  exit 1
                fi

                if ! diff -r "$GENERATED_FFI" "$COMMITTED_FFI"; then
                  echo ""
                  echo "ERROR: Committed EidolaAppCoreFFI headers don't match generated ones!"
                  echo ""
                  echo "To fix this:"
                  echo "  1. Run: nix run '.#update-eidola-app-core-swift-bindings'"
                  echo "  2. Review the changes"
                  echo "  3. Commit the updated headers"
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
                nativeBuildInputs = [ pkgs.findutils ];
              }
              ''
                echo "Checking Swift formatting..."

                # Find all Swift files, excluding .build directories (SwiftPM build artifacts)
                # Generated bindings are pre-formatted during generation, so no exclusions needed
                # Note: .git is already excluded by crane's source filtering
                find ${repoSrc} \
                  -path '*/.build' -prune -o \
                  -name '*.swift' -print0 \
                  | xargs -0 -r ${swiftPkg}/bin/swift format lint --strict

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

          update-eidola-app-core-swift-bindings = {
            type = "app";
            meta.description = "Update committed Swift bindings for app core";
            program = "${
              pkgs.writeShellApplication {
                name = "update-eidola-app-core-swift-bindings";
                runtimeInputs = [
                  pkgs.coreutils
                  pkgs.git
                ];

                text = ''
                  ${./scripts/update-shared-bindings.sh} \
                    "${self.packages.${system}.eidola-app-core-swift-bindings}/Sources"
                '';
              }
            }/bin/update-eidola-app-core-swift-bindings";
          };

          update-eidola-app-core-swift-xcframework = {
            type = "app";
            meta.description = "Update XCFramework for app core";
            program = "${
              pkgs.writeShellApplication {
                name = "update-eidola-app-core-swift-xcframework";
                runtimeInputs = [
                  pkgs.coreutils
                  pkgs.git
                ];

                text = ''
                  ${./scripts/update-xcframework.sh} "${self.packages.${system}.eidola-app-core-swift-xcframework}"
                '';
              }
            }/bin/update-eidola-app-core-swift-xcframework";
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

                  echo "Formatting Swift files..."

                  # Use git ls-files to respect .gitignore
                  # Generated bindings are pre-formatted during generation, so no exclusions needed
                  git ls-files '*.swift' \
                    | xargs -r ${swiftPkg}/bin/swift format --in-place

                  echo "Done. Review changes and commit:"
                  echo "  git status"
                '';
              }
            }/bin/format-swift";
          };
        }
        // pkgs.lib.optionalAttrs (eidolaMacosApp != null) {
          run-eidola = {
            type = "app";
            meta.description = "Build and launch the Eidola macOS app";
            program = "${pkgs.writeShellScript "run-eidola" ''
              open "${eidolaMacosApp}/Applications/Eidola.app"
            ''}";
          };
        };
      }
    );
}
