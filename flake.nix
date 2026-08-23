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

    # Apple code-signature tool (pure Python). Build-time and CI only, and
    # only as an *independent check*: its `apply` reattaches a detached
    # signature, which is what we differentially test our own apply against.
    # It is not the signer and not the detacher — it cannot detach a
    # signature it did not create, and its only detach path wants a PKCS#12
    # archive, which a non-exportable token key can never supply (measured:
    # scripts/fixtures/apple-roundtrip/round-trip.md §1.1). Detaching is
    # ours. It never enters Cargo.lock, never ships in an artifact, and never
    # runs on a user's machine — see .github/AGENTS.md ("Pinned build
    # tools"). flake.lock is the pin; if a patch is ever needed, fork per the
    # fork-branch practice in crates/eidola-gui/AGENTS.md.
    signapple = {
      url = "github:achow101/signapple/3fab3bb57f227f0dd31007b417683035f5204838";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
      crane,
      signapple,
      ...
    }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ] (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # SHA256 for rust-toolchain.toml (single source of truth)
        rustToolchainSha256 = "sha256-A1abGIbOtcBSdrUMhDGrER3pRM1hQP4fp9gh3Y4PKc8=";

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

        readCrateToml = pname: builtins.fromTOML (builtins.readFile ./${cratePaths.${pname}}/Cargo.toml);

        # Dependency-table kinds. Ordinary and build dependencies are actually
        # compiled; dev-dependencies are compiled only for the crate whose
        # tests are being run.
        buildDepKinds = [
          "dependencies"
          "build-dependencies"
        ];
        devDepKind = "dev-dependencies";

        # Every dependency table of the requested kinds in a manifest: the
        # top-level ones plus their `[target.'cfg(...)'.*]` counterparts.
        depTables =
          cargoToml: kinds:
          map (k: cargoToml.${k} or { }) kinds
          ++ builtins.concatMap (t: map (k: t.${k} or { }) kinds) (
            builtins.attrValues (cargoToml.target or { })
          );

        isWorkspacePathDep =
          name: spec: builtins.isAttrs spec && spec ? path && builtins.elem name workspaceCrates;

        # Sibling workspace crates a crate declares as path dependencies.
        # `kinds` selects which dependency tables count. Self-references (the
        # `{ path = "." }` idiom for enabling a crate's own test-only feature)
        # are dropped so the graph stays acyclic.
        getWorkspaceDeps =
          kinds: pname:
          let
            cargoToml = readCrateToml pname;
            names = builtins.concatMap (
              tbl: builtins.attrNames (pkgs.lib.filterAttrs isWorkspacePathDep tbl)
            ) (depTables cargoToml kinds);
          in
          pkgs.lib.unique (builtins.filter (n: n != pname) names);

        # Recursively resolve all transitive workspace dependencies that are
        # actually compiled.
        getAllDeps =
          pname:
          let
            directDeps = getWorkspaceDeps buildDepKinds pname;
            transitiveDeps = builtins.concatMap getAllDeps directDeps;
          in
          pkgs.lib.unique (directDeps ++ transitiveDeps);

        # Crates whose sources a package build needs: the package itself, the
        # workspace crates its own dev-dependencies name (a package build may
        # run `cargo test` for that one package), and the compiled closure of
        # all of those.
        packageCrates =
          pname:
          let
            devDeps = getWorkspaceDeps [ devDepKind ] pname;
          in
          pkgs.lib.unique (
            [ pname ] ++ getAllDeps pname ++ builtins.concatMap (d: [ d ] ++ getAllDeps d) devDeps
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
                  # Both `eidola-app-core/build.rs` and `eidola-server/build.rs`
                  # consume pinned trust data from `releases/`:
                  #   * eidola-app-core → all of `releases/trust/*.json` +
                  #     `releases/schema/*.json` (client trust root)
                  #   * eidola-server → just `releases/trust/sigstore-trusted-root.json`
                  #     (pinned Sigstore root; the runtime measurement resolver
                  #     verifies Tinfoil release attestations against it)
                  # Include `releases/` when either crate is in the set, but
                  # only for those crates — unrelated crates shouldn't
                  # cache-bust on every manifest regeneration.
                  # `artifact-manifest.json` is deliberately excluded — it
                  # records the eidola-cli narHash itself, so including it
                  # would create a self-reference that prevents the build
                  # from reaching a fixed point.
                  trustRootFiles =
                    crateSet ? "eidola-app-core"
                    || crateSet ? "eidola-server";
                  isTrustRootPath =
                    relPath == "/releases"
                    || pkgs.lib.hasPrefix "/releases/" relPath;
                in
                # Exclude irrelevant crate directories entirely
                if isIrrelevantCrate then
                  false
                # Trust-root build inputs for `eidola-app-core/build.rs`.
                else if trustRootFiles && isTrustRootPath then
                  true
                # Keep only root-level files that affect Cargo resolution/builds.
                # This avoids generated files like artifact-manifest.json from
                # perturbing package hashes for unrelated Nix builds.
                else if type == "regular" && builtins.match "/[^/]+" relPath != null then
                  builtins.hasAttr baseName rootBuildFiles
                # Keep directories that are parents of crate paths
                else if type == "directory" && isParentOfCrate relPath then
                  true
                # Include .sql files (used by include_str! in the CLI), .ttf
                # font files (used by include_bytes! in the GUI), and .ftl
                # localization files (read by the GUI's build.rs, which emits
                # them as string literals — every shipped string is a build
                # input inside the measured artifact, never loaded at runtime).
                # All three feed compile-time inputs; craneLib.filterCargoSources
                # discards them by default because they aren't Rust source.
                else if type == "regular" && (
                  pkgs.lib.hasSuffix ".sql" path
                  || pkgs.lib.hasSuffix ".ttf" path
                  || pkgs.lib.hasSuffix ".ftl" path
                ) then
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

            # Cargo insists every `path` dependency's manifest resolve, even
            # for a dependency it will never build — a dev-dependency on a
            # workspace crate is enough to fail manifest loading for the whole
            # workspace if that crate's directory isn't here. Only the package
            # being built ever has its tests compiled (`packageCrates` keeps
            # *its* dev-dependencies in the set), so instead of dragging the
            # other crates' test-only graphs into every desktop closure —
            # which would tie the desktop narHashes to unrelated crates'
            # churn forever — materialize those manifests with the out-of-set
            # dev-dependency edges dropped. Cargo then resolves and prunes
            # them from the in-sandbox lockfile exactly as it already does for
            # the workspace members this filter omits.
            pruneDevDeps = pkgs.lib.filterAttrs (
              name: spec: !(isWorkspacePathDep name spec) || crateSet ? ${name}
            );
            pruneDevDepTables =
              attrs:
              attrs
              // pkgs.lib.optionalAttrs (attrs ? ${devDepKind}) {
                ${devDepKind} = pruneDevDeps attrs.${devDepKind};
              };
            pruneManifest =
              cargoToml:
              pruneDevDepTables cargoToml
              // pkgs.lib.optionalAttrs (cargoToml ? target) {
                target = builtins.mapAttrs (_: pruneDevDepTables) cargoToml.target;
              };
            prunedManifests = builtins.filter (m: m != null) (
              map (
                c:
                let
                  cargoToml = readCrateToml c;
                  pruned = pruneManifest cargoToml;
                in
                if pruned == cargoToml then
                  null
                else
                  {
                    path = cratePaths.${c};
                    file = (pkgs.formats.toml { }).generate "Cargo.toml" pruned;
                  }
              ) crates
            );
          in
          # Combine filtered source with the modified Cargo.toml
          pkgs.runCommand "filtered-workspace-${builtins.concatStringsSep "-" crates}" { } ''
            cp -r ${filteredSrc} $out
            chmod -R u+w $out
            cp ${filteredCargoTomlContent} $out/Cargo.toml
            ${pkgs.lib.concatMapStrings (m: ''
              cp ${m.file} $out/${m.path}/Cargo.toml
            '') prunedManifests}
          '';

        # Full repo source for checks that compare committed vs generated files
        repoSrc = craneLib.path ./.;

        # Full source for workspace-wide operations. Carries the same `.ftl`
        # exception as `filteredSrc`: craneLib's filter keeps only Rust and
        # Cargo files, and the GUI's build script reads `locales/` to generate
        # its typed accessors, so the workspace-wide clippy/test derivations
        # cannot compile it without them.
        fullSrc = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter =
            path: type:
            craneLib.filterCargoSources path type
            || (type == "regular" && pkgs.lib.hasSuffix ".ftl" path)
            # .ttf and .sql feed include_bytes!/include_str! the same way .ftl
            # feeds the localization build script; without them the workspace
            # check derivations cannot compile eidola-gui or eidola-app-core.
            || (type == "regular" && pkgs.lib.hasSuffix ".ttf" path)
            || (type == "regular" && pkgs.lib.hasSuffix ".sql" path);
        };

        # Base RUSTFLAGS for deterministic builds (extended per-target in mkTargetConfig)
        baseRustFlags = "-C debuginfo=0 -C target-cpu=generic";

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

          RUSTFLAGS = baseRustFlags;

          # Remap $NIX_BUILD_TOP to a fixed `/build` prefix in compile-time
          # paths. Without this, rustc embeds absolute paths into the binary
          # via `file!()` (used implicitly by panic location info) for any
          # source file under the build sandbox — including `OUT_DIR` files
          # generated by dependency build scripts (e.g. html5ever's
          # `rules.rs`). Those paths contain the per-invocation build-instance
          # number (`nix-NNNN-NNNNNNNN`), which is unique to each build and
          # was empirically the last source of byte-level drift after thin→fat
          # LTO eliminated codegen-ordering nondeterminism (~45 bytes per
          # arch all clustered in one rules.rs path string + cascading
          # codesign hash). `$NIX_BUILD_TOP` is only knowable at sandbox
          # runtime so this must go through preBuild, not the RUSTFLAGS attr.
          # The llama.cpp sidecar has the C/ObjC equivalent (`-ffile-prefix-map`
          # in `llamaServer`); both are required for the macOS universal
          # narHash to be a function of source.
          preBuild = ''
            export RUSTFLAGS="$RUSTFLAGS --remap-path-prefix $NIX_BUILD_TOP=/build"
          '';
        };

        # Target configuration helper
        # Takes explicit Rust target and optional Nix cross-system (pkgsCross attr name).
        # - rustTarget: Rust target triple (e.g., "aarch64-apple-darwin")
        # - nixCrossSystem: pkgsCross attr name (e.g., "aarch64-multiplatform-musl"), or null for native pkgs
        mkTargetConfig =
          rustTarget: nixCrossSystem:
          let
            isNative = rustTarget == nativeRustTarget;
            isDarwin = builtins.match ".*-apple-darwin" rustTarget != null;
            isLinuxMusl = builtins.match ".*-linux-musl" rustTarget != null;

            # Use pkgsCross if specified, otherwise native pkgs
            targetPkgs = if nixCrossSystem == null then pkgs else pkgs.pkgsCross.${nixCrossSystem};

            # Crane uses target pkgs (for linker/libc) but host toolchain (for cargo)
            craneLibTarget = (crane.mkLib targetPkgs).overrideToolchain (_: rustToolchain);

            # Linker env var name for this target (dynamically generated from target triple)
            linkerEnvVar = "CARGO_TARGET_${
              pkgs.lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] rustTarget)
            }_LINKER";

            # Use rust-lld (bundled with the Rust toolchain) for Darwin targets.
            # The system ld64 (from cctools) produces nondeterministic Mach-O output
            # depending on which macOS version's framework stubs are visible in the
            # build sandbox. rust-lld resolves symbols solely from the Nix-provided
            # apple-sdk, making builds reproducible across macOS environments.
            rustLld = "${rustToolchain}/lib/rustlib/${nativeRustTarget}/bin/rust-lld";
            darwinRustFlags = builtins.concatStringsSep " " [
              baseRustFlags
              "-C linker-flavor=ld64.lld"
              "-C linker=${rustLld}"
              "-C link-arg=-L${pkgs.libiconv}/lib"
            ];

            # Cross-compilation needs CARGO_BUILD_TARGET set.
            # For Linux musl targets without pkgsCross, use rust-lld (bundled with Rust).
            # For Darwin targets, use rust-lld with ld64.lld flavor for reproducibility.
            targetArgs =
              if isDarwin && isNative then
                {
                  RUSTFLAGS = darwinRustFlags;
                }
              else if isDarwin then
                {
                  CARGO_BUILD_TARGET = rustTarget;
                  RUSTFLAGS = darwinRustFlags;
                }
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
            extraBuildArgs ? { },
          }:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            relevantCrates = packageCrates pname;
            filteredSrc = mkFilteredSrc relevantCrates;
          in
          cfg.craneLibTarget.buildDepsOnly (
            commonArgs
            // cfg.targetArgs
            // extraBuildArgs
            // {
              src = filteredSrc;
              pname = "${pname}-deps";
              # Only build deps for this specific package
              cargoExtraArgs = "-p ${pname} ${extraCargoArgs}";
            }
          );

        # Build individual packages with their own filtered deps. `doCheck`
        # toggles whether crane runs `cargo test` as part of the build
        # (default true). Pass false for release artifacts whose
        # integration tests can't run in the Nix sandbox — e.g. the GUI's
        # `tests/visual.rs`, which talks to live AppKit and is documented
        # in crates/eidola-gui/AGENTS.md as a local-only debug aid.
        # The workspace-wide `checks.tests` target still runs the full
        # test suite separately, so coverage isn't lost.
        mkPackage =
          {
            pname,
            rustTarget,
            nixCrossSystem,
            crateType ? null,
            extraCargoArgs ? "",
            doCheck ? true,
            # Extra derivation attrs merged into both the deps-only and the
            # package build (e.g. buildInputs for system libraries the crate
            # links via pkg-config — used by the Linux GUI build).
            extraBuildArgs ? { },
            # Extra derivation attrs merged into the package build ONLY —
            # for hooks like postFixup that reference $out/bin/<binary>,
            # which doesn't exist in the deps-only derivation.
            extraPackageArgs ? { },
          }:
          let
            cfg = mkTargetConfig rustTarget nixCrossSystem;
            relevantCrates = packageCrates pname;
            filteredSrc = mkFilteredSrc relevantCrates;
            packageCargoArtifacts = mkPackageDeps {
              inherit
                pname
                rustTarget
                nixCrossSystem
                extraCargoArgs
                extraBuildArgs
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
            // extraBuildArgs
            // extraPackageArgs
            // {
              src = filteredSrc;
              cargoArtifacts = packageCargoArtifacts;
              inherit pname doCheck;
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

        # ── ELF assertions for the shipped Linux executables ────────────────
        #
        # Both live *inside* the derivation that produces the binary, so a
        # nixpkgs bump that changes linkage or raises a symbol requirement
        # fails the build. Asserting either of these outside the build — in a
        # script or a workflow step — would let a raised requirement reach a
        # user, who would meet it as a loader error at launch.
        #
        # readelf is called by absolute path rather than added to
        # nativeBuildInputs: the sidecar's static build is a cross build
        # (build gnu → host musl), where an unprefixed binutils on PATH would
        # shadow the cross toolchain's own linker wrappers.
        readelfBin = "${pkgs.binutils-unwrapped}/bin/readelf";

        # No PT_INTERP (which would name a loader, and a store-built binary's
        # loader is a /nix/store path) and no DT_NEEDED (which would name
        # libraries the host must supply). This is the property that lets a
        # binary be copied out of the store into a host-distro package with
        # nothing following it.
        assertFullyStatic = binary: ''
          echo "checking static linkage: ${binary}"
          if ${readelfBin} -lW "${binary}" | grep -q INTERP; then
            echo "error: ${binary} has a PT_INTERP program header, so it is dynamically linked" >&2
            ${readelfBin} -lW "${binary}" | grep -A1 INTERP >&2
            exit 1
          fi
          if ${readelfBin} -dW "${binary}" 2>/dev/null | grep -q NEEDED; then
            echo "error: ${binary} declares DT_NEEDED entries, so it is dynamically linked" >&2
            ${readelfBin} -dW "${binary}" | grep NEEDED >&2
            exit 1
          fi
        '';

        # The oldest glibc a shipped Linux executable may demand, expressed as
        # the highest `GLIBC_` symbol version it is allowed to reference.
        # 2.39 is Ubuntu 24.04 LTS and Debian 13; a host older than that
        # cannot run the binary at all, so this number *is* the supported
        # floor and moving it is a product decision, not a build detail.
        glibcSymbolFloor = "2.39";

        # Versioned-symbol requirements are what actually bind a glibc binary
        # to a minimum host, and they move on their own: a nixpkgs bump can
        # raise one without a line of our code changing.
        #
        # `GLIBCXX_`/`CXXABI_` are asserted absent rather than floored. No
        # shipped executable links libstdc++ today (the sidecar is static,
        # the GUI is Rust), and the two families version independently of
        # glibc, so there is no honest single floor to compare them against.
        # If C++ ever enters the link, this fails and someone picks a floor
        # for it deliberately.
        assertGlibcFloor = binary: ''
          echo "checking symbol-version floor (GLIBC_${glibcSymbolFloor}): ${binary}"
          symbolVersions="$(
            ${readelfBin} --dyn-syms -W "${binary}" \
              | grep -oE '@@?(GLIBC|GLIBCXX|CXXABI)_[0-9][0-9.]*' \
              | tr -d '@' | sort -u
          )"
          maxGlibc="$(
            printf '%s\n' "$symbolVersions" | grep '^GLIBC_' \
              | cut -d_ -f2 | sort -V | tail -1
          )"
          if [ -n "$maxGlibc" ] \
            && [ "$(printf '%s\n%s\n' "$maxGlibc" "${glibcSymbolFloor}" | sort -V | tail -1)" != "${glibcSymbolFloor}" ]; then
            echo "error: ${binary} requires GLIBC_$maxGlibc, above the supported floor GLIBC_${glibcSymbolFloor}" >&2
            ${readelfBin} --dyn-syms -W "${binary}" | grep -E "@GLIBC_$maxGlibc\$" >&2
            exit 1
          fi
          cxxVersions="$(printf '%s\n' "$symbolVersions" | grep -E '^(GLIBCXX|CXXABI)_' || true)"
          if [ -n "$cxxVersions" ]; then
            echo "error: ${binary} requires libstdc++ symbol versions, which no shipped executable is expected to:" >&2
            printf '%s\n' "$cxxVersions" >&2
            echo "pick a libstdc++ floor deliberately before shipping this." >&2
            exit 1
          fi
          echo "symbol-version floor ok (max GLIBC_''${maxGlibc:-none})"
        '';

        # llama.cpp `llama-server`, shipped as the `local` backend's on-device
        # inference engine sidecar. Bundled inside the macOS .app
        # (Contents/MacOS), next to the CLI binary, and in the Linux GUI
        # closure (exposed via EIDOLA_LLAMA_SERVER by the wrapper). App-core
        # resolves the engine without ever scanning $PATH, so no system
        # llama.cpp install is required.
        #
        # Linkage differs per platform, and the difference is load-bearing:
        #
        #   * Darwin: llama.cpp's own libraries are linked in, leaving only
        #     the system frameworks (Metal/Accelerate/libc++/libSystem). That
        #     is as static as an Apple platform gets — libSystem is never
        #     statically linked — and it is what makes the binary relocatable
        #     out of the store into the .app.
        #   * Linux: fully static against musl (`pkgsStatic`), no PT_INTERP
        #     and no DT_NEEDED at all. `assertFullyStatic` checks that in
        #     the derivation, because the property is what lets the
        #     sidecar be copied into a host-distro package: a dynamic sidecar
        #     carries a /nix/store interpreter and a glibc/libstdc++ symbol
        #     floor that would follow the artifact onto every user's machine.
        #
        # The base package fn hardcodes LLAMA_CURL and BUILD_SHARED_LIBS to ON
        # and exposes no curlSupport arg; we flip both OFF via *appended*
        # cmakeFlags (later -D wins), which drops the libcurl runtime dep and
        # folds llama.cpp's own libs into the one binary we ship. On
        # aarch64-darwin metalSupport is ON by default and the Metal library
        # is *embedded* (LLAMA_METAL_EMBED_LIBRARY), compiled at runtime — no
        # Xcode, no external .metallib, works inside the sandbox.
        #
        # Linux is CPU-only for now, and with BLAS off: nixpkgs' default BLAS
        # backend is a shared `libblas.so.3`, which a static binary cannot
        # carry — and measured on a small model it slowed prompt processing
        # by an order of magnitude anyway. OpenMP stays on; the static
        # toolchain has libgomp.a. A Vulkan build (`vulkanSupport = true`,
        # shaderc compiles SPIR-V at build time) is coherent to *build*, but
        # usable GPU inference also needs Vulkan ICD driver files at runtime
        # and a loader to dlopen, which a static binary cannot do. Wiring
        # that is the follow-up; CPU-only ships a working engine on every
        # Linux host today.
        #
        # The pinned nixpkgs ships llama.cpp build 6981, which predates the
        # `gemma4` GGUF architecture the curated catalog uses ("unknown model
        # architecture: 'gemma4'"), so the source is bumped to release b9960
        # (verified against the catalog models) while keeping the nixpkgs
        # build recipe. The base recipe derives its src tag from `version` and
        # only uses `leaveDotGit` to extract the short commit into a COMMIT
        # file for `LLAMA_BUILD_COMMIT`; a plain pinned fetch plus injecting
        # the known commit directly is equivalent and more reproducible.
        llamaServerVersion = "9960";
        llamaServerCommit = "a935fbff"; # short rev of tag b9960
        llamaServerBase =
          if pkgs.stdenv.hostPlatform.isDarwin then
            pkgs.llama-cpp
          else
            pkgs.pkgsStatic.llama-cpp.override { blasSupport = false; };
        llamaServer = llamaServerBase.overrideAttrs (o: {
          version = llamaServerVersion;
          src = pkgs.fetchFromGitHub {
            owner = "ggml-org";
            repo = "llama.cpp";
            tag = "b${llamaServerVersion}";
            hash = "sha256-FheVvdqpF3pqxmovFXBh65iNAH+lSM+jqGrM8CpLHF8=";
          };
          # python3 is only for the Darwin LC_UUID zeroing in postInstall.
          nativeBuildInputs = (o.nativeBuildInputs or [ ]) ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.python3
          ];
          preConfigure = ''
            prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=${llamaServerCommit}"
            # $NIX_BUILD_TOP is only knowable at sandbox runtime
            # (`nix-NNNN-NNNNNNNN`). llama.cpp bakes those paths into the
            # binary via `__FILE__` (~209 assertion / log strings). The
            # decimal width of the build-instance id varies per invocation,
            # which shifts `__cstring` size, Mach-O section layout, and every
            # pointer into the string table — empirically the whole
            # macos-universal narHash drift vs CI (PR #317). Same class of
            # leak the Rust builds already close with `--remap-path-prefix`.
            # Darwin's cc-wrapper does not apply `-ffile-prefix-map` the way
            # Linux's does, so this has to be explicit. The cc-wrapper reads
            # NIX_CFLAGS_COMPILE at compile time, so CMake cannot bake it
            # away; ObjC `.m` files go through the same wrapper.
            export NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE -ffile-prefix-map=$NIX_BUILD_TOP=/build -fmacro-prefix-map=$NIX_BUILD_TOP=/build"
            export NIX_CXXFLAGS_COMPILE="$NIX_CXXFLAGS_COMPILE -ffile-prefix-map=$NIX_BUILD_TOP=/build -fmacro-prefix-map=$NIX_BUILD_TOP=/build"
          '';
          cmakeFlags = o.cmakeFlags ++ [
            "-DLLAMA_CURL=OFF"
            "-DBUILD_SHARED_LIBS=OFF"
            # The server auto-detects OpenSSL for httplib TLS; we speak plain
            # HTTP over loopback only, and a libssl dep would pin the binary
            # to nix-store paths (not relocatable on user machines). This is
            # the pre-b9960 spelling of the option, kept because changing it
            # would move the macOS sidecar's bytes; what makes the detection
            # miss on Darwin is that nothing in the sandbox provides OpenSSL
            # once curl is out of buildInputs.
            "-DLLAMA_SERVER_SSL=OFF"
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            # Left on deliberately: the static toolchain has libgomp.a, so
            # OpenMP costs nothing in linkage, and dropping it for ggml's own
            # threadpool measured ~35% slower token generation.
            "-DGGML_OPENMP=ON"
            # The vendored cpp-httplib auto-detects OpenSSL under this option
            # (`LLAMA_SERVER_SSL` above is the older spelling and is inert at
            # this llama.cpp revision). We speak plain HTTP over loopback
            # only, and a TLS stack would drag its CA-path assumptions into a
            # binary that must run on any host.
            "-DLLAMA_OPENSSL=OFF"
            # tools/ui builds an asset-embedding generator that has to run on
            # the *build* machine; a static build is a cross build, so CMake
            # cannot reuse the target compiler for it.
            "-DHOST_CXX_COMPILER=${pkgs.buildPackages.stdenv.cc}/bin/c++"
          ];
          # curl is unused with LLAMA_CURL=OFF, and its presence propagates
          # OpenSSL into the server's auto-detection — drop it entirely.
          # Matched by prefix because a static build's package names carry a
          # `-static-<triple>` suffix.
          buildInputs = pkgs.lib.filter (d: !(pkgs.lib.hasPrefix "curl" (d.pname or ""))) o.buildInputs;
          # Trim the closure to just the one tool we ship. The static build
          # embeds llama.cpp's libs into the binary, so the sibling llama-*
          # tools, the `llama` symlink, and the installed headers/archives are
          # all dead weight for a sidecar.
          postInstall =
            (o.postInstall or "")
            + ''
              find "$out/bin" -mindepth 1 ! -name llama-server -delete
              rm -rf "$out/include" "$out/lib"
            ''
            + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
              # ld64's LC_UUID includes nondeterministic linker state even
              # when the compiled code is identical (same reason the
              # universal wrapper zeros the Rust slices). Zero it here so
              # the sidecar NAR itself is a function of source; autoSign
              # re-signs in postFixup.
              chmod u+w "$out/bin/llama-server"
              python3 -c '
              import struct, sys
              def zero_uuid(data, offset):
                  magic = struct.unpack_from("<I", data, offset)[0]
                  assert magic == 0xFEEDFACF, f"bad Mach-O magic at {offset:#x}"
                  ncmds = struct.unpack_from("<I", data, offset + 16)[0]
                  pos = offset + 32
                  for _ in range(ncmds):
                      cmd, cmdsize = struct.unpack_from("<II", data, pos)
                      if cmd == 0x1B:
                          data[pos + 8 : pos + 24] = b"\x00" * 16
                          return
                      pos += cmdsize
                  sys.exit(f"LC_UUID not found at offset {offset:#x}")

              path = sys.argv[1]
              with open(path, "r+b") as f:
                  data = bytearray(f.read())
              magic = struct.unpack_from(">I", data, 0)[0]
              if magic == 0xCAFEBABE:
                  nfat = struct.unpack_from(">I", data, 4)[0]
                  for i in range(nfat):
                      offset = struct.unpack_from(">I", data, 8 + i * 20 + 8)[0]
                      zero_uuid(data, offset)
              else:
                  zero_uuid(data, 0)
              with open(path, "wb") as f:
                  f.write(data)
              ' "$out/bin/llama-server"
            '';
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # The build-platform compiler tools/ui's generator needs.
          depsBuildBuild = (o.depsBuildBuild or [ ]) ++ [ pkgs.buildPackages.stdenv.cc ];
          # postFixup, not postInstall: fixupPhase strips and rewrites RPATHs,
          # so this reads the bytes that actually ship.
          postFixup = assertFullyStatic "$out/bin/llama-server";
        });

        # signapple — the independent `apply` our own reattach is checked
        # against, not the signer and not the detacher. Build-time/CI only;
        # see the flake input comment above and .github/AGENTS.md.
        #
        # Two of its four Python dependencies are pinned to git revisions
        # upstream and are not usable from nixpkgs as-is, so they are built
        # here rather than taken from python3Packages:
        #   - certvalidator: signapple calls ValidationContext with
        #     `additional_critical_extensions=` (Apple's certs carry custom
        #     critical extensions). Only achow101's fork accepts that keyword;
        #     nixpkgs' vanilla 0.11.1 raises TypeError on every verify.
        #   - elfesteem: not in nixpkgs at all.
        # asn1crypto and oscrypto come from nixpkgs. nixpkgs' oscrypto is
        # 1.3.0 plus the two post-1.3.0 patches that matter (OpenSSL 3.0.10
        # version parsing, and the importlib-for-imp change), so it is
        # equivalent to signapple's pinned revision for our purposes.
        signappleElfesteem = pkgs.python3Packages.buildPythonPackage {
          pname = "elf-esteem";
          version = "0.1-unstable-2018-11-19";
          src = pkgs.fetchFromGitHub {
            owner = "LRGH";
            repo = "elfesteem";
            rev = "5800fcf150dec3ce524f14bc2f24dc037f4826e6";
            hash = "sha256-qy48vJMDQcoxLUraCPQizgt8hcrTrPdH3v3LpLqM0sw=";
          };
          # setup.py predates Python 3.12's removal of distutils.
          postPatch = ''
            substituteInPlace setup.py \
              --replace-fail "from distutils.core import setup" "from setuptools import setup"
          '';
          pyproject = true;
          build-system = [ pkgs.python3Packages.setuptools ];
          doCheck = false;
          pythonImportsCheck = [ "elfesteem.macho" ];
        };

        signappleCertvalidator = pkgs.python3Packages.buildPythonPackage {
          pname = "certvalidator";
          version = "0.12.0.dev1-unstable-2020-12-14";
          src = pkgs.fetchFromGitHub {
            owner = "achow101";
            repo = "certvalidator";
            rev = "e5bdb4bfcaa09fa0af355eb8867d00dfeecba08c";
            hash = "sha256-5TBCc94uz5FuZAM8fWHYzPV6i+kTbOrdfn+6effs+6I=";
          };
          pyproject = true;
          build-system = [ pkgs.python3Packages.setuptools ];
          dependencies = with pkgs.python3Packages; [
            asn1crypto
            oscrypto
          ];
          doCheck = false;
          pythonImportsCheck = [ "certvalidator" ];
        };

        signappleTool = pkgs.python3Packages.buildPythonApplication {
          pname = "signapple";
          version = "0.2.0-unstable-2026-05-26";
          src = signapple;
          # Poetry records the three git dependencies as direct URL
          # references, which no installed distribution can ever satisfy.
          # Relax them to plain constraints; the pins live in this file.
          postPatch = ''
            substituteInPlace pyproject.toml \
              --replace-fail 'oscrypto = { git = "https://github.com/wbond/oscrypto.git", rev = "1547f535001ba568b239b8797465536759c742a3" }' 'oscrypto = "*"' \
              --replace-fail 'certvalidator = { git = "https://github.com/achow101/certvalidator.git", rev = "e5bdb4bfcaa09fa0af355eb8867d00dfeecba08c" }' 'certvalidator = "*"' \
              --replace-fail 'elf-esteem = { git = "https://github.com/LRGH/elfesteem.git", rev = "5800fcf150dec3ce524f14bc2f24dc037f4826e6" }' 'elf-esteem = "*"'
          '';
          pyproject = true;
          build-system = [ pkgs.python3Packages.poetry-core ];
          dependencies = [
            pkgs.python3Packages.asn1crypto
            pkgs.python3Packages.oscrypto
            signappleCertvalidator
            signappleElfesteem
          ];
          doCheck = false;
          pythonImportsCheck = [ "signapple" ];
          meta = {
            description = "Signing and verification tool for macOS code signatures";
            license = pkgs.lib.licenses.mit;
            mainProgram = "signapple";
          };
        };

        # Build the CLI as a macOS universal binary (Darwin only)
        eidolaCliMacosUniversal =
          if !pkgs.stdenv.isDarwin then
            null
          else
            pkgs.stdenv.mkDerivation {
              pname = "eidola-cli-macos-universal";
              version = "1.0";

              nativeBuildInputs = [
                pkgs.darwin.cctools
                pkgs.darwin.autoSignDarwinBinariesHook
                pkgs.python3
              ];

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

                # Zero LC_UUID in each Mach-O slice for reproducibility.
                # lld computes the UUID from a hash that includes nondeterministic
                # linker state, producing different values across macOS environments
                # even when the compiled code is byte-for-byte identical.
                chmod +w "$out/bin/eidola"
                python3 -c '
import struct, sys
def zero_uuid(data, offset):
    magic = struct.unpack_from("<I", data, offset)[0]
    assert magic == 0xFEEDFACF, f"bad Mach-O magic at {offset:#x}"
    ncmds = struct.unpack_from("<I", data, offset + 16)[0]
    pos = offset + 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, pos)
        if cmd == 0x1B:
            data[pos + 8 : pos + 24] = b"\x00" * 16
            return
        pos += cmdsize
    sys.exit(f"LC_UUID not found at offset {offset:#x}")

path = sys.argv[1]
with open(path, "r+b") as f:
    data = bytearray(f.read())
magic = struct.unpack_from(">I", data, 0)[0]
if magic == 0xCAFEBABE:
    nfat = struct.unpack_from(">I", data, 4)[0]
    for i in range(nfat):
        offset = struct.unpack_from(">I", data, 8 + i * 20 + 8)[0]
        zero_uuid(data, offset)
else:
    zero_uuid(data, 0)
with open(path, "wb") as f:
    f.write(data)
' "$out/bin/eidola"

                # autoSignDarwinBinariesHook re-signs in postFixup
                chmod -w "$out/bin/eidola"

                # Ship the on-device inference engine sidecar next to the CLI
                # binary; app-core accepts a sibling `llama-server` next to the
                # exe. arm64-only (see the GUI bundle below for the rationale);
                # the autoSignDarwinBinariesHook postFixup pass walks every
                # Mach-O in $out and (re-)signs this one too. The sidecar
                # derivation is the reproducibility boundary (path remap +
                # Darwin LC_UUID zero); copying it is then deterministic.
                cp ${llamaServer}/bin/llama-server "$out/bin/llama-server"
                chmod u+w "$out/bin/llama-server"
              '';

              installPhase = ''
                echo "Universal binary:"
                lipo -info "$out/bin/eidola"
              '';

              meta = {
                description = "Eidola CLI (macOS universal binary)";
                platforms = [ "aarch64-darwin" ];
              };
            };

        # Build the GUI as a reproducible macOS .app bundle (Darwin only).
        # Layout: $out/Eidola.app/Contents/{MacOS/Eidola, Resources, Info.plist}.
        # The .app wrapper is *required*, not cosmetic — see
        # crates/eidola-gui/AGENTS.md (".app bundling") for why bare-binary
        # mode breaks menu key-equivalent dispatch under AppKit.
        eidolaGuiMacosUniversal =
          if !pkgs.stdenv.isDarwin then
            null
          else
            pkgs.stdenv.mkDerivation {
              pname = "eidola-gui-macos-universal";
              version = "1.0";

              nativeBuildInputs = [
                pkgs.darwin.cctools
                pkgs.darwin.autoSignDarwinBinariesHook
                pkgs.python3
              ];

              SOURCE_DATE_EPOCH = "0";

              dontUnpack = true;

              # tests/visual.rs talks to live AppKit (real Metal renderer +
              # offscreen window); the Nix sandbox can't host it. See the
              # `mkPackage` doc above and crates/eidola-gui/AGENTS.md's
              # Testing section for why this is local-only.
              arm64 = mkPackage {
                pname = "eidola-gui";
                rustTarget = "aarch64-apple-darwin";
                nixCrossSystem = null;
                doCheck = false;
              };
              x86_64 = mkPackage {
                pname = "eidola-gui";
                rustTarget = "x86_64-apple-darwin";
                nixCrossSystem = null;
                doCheck = false;
              };

              buildPhase = ''
                APP="$out/Eidola.app"
                mkdir -p "$APP/Contents/MacOS"
                mkdir -p "$APP/Contents/Resources"

                # Lipo into Contents/MacOS/Eidola. The binary is renamed to
                # match CFBundleExecutable in Info.plist; mismatch makes
                # AppKit fall back to tool-mode and breaks menu dispatch.
                lipo -create \
                  "$arm64/bin/eidola-gui" \
                  "$x86_64/bin/eidola-gui" \
                  -output "$APP/Contents/MacOS/Eidola"

                # Zero LC_UUID in each Mach-O slice. lld computes the UUID
                # from a hash that includes nondeterministic linker state,
                # producing different values across macOS environments even
                # when the compiled code is byte-for-byte identical. Same
                # trick as the cli universal-binary build above.
                chmod +w "$APP/Contents/MacOS/Eidola"
                python3 -c '
import struct, sys
def zero_uuid(data, offset):
    magic = struct.unpack_from("<I", data, offset)[0]
    assert magic == 0xFEEDFACF, f"bad Mach-O magic at {offset:#x}"
    ncmds = struct.unpack_from("<I", data, offset + 16)[0]
    pos = offset + 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, pos)
        if cmd == 0x1B:
            data[pos + 8 : pos + 24] = b"\x00" * 16
            return
        pos += cmdsize
    sys.exit(f"LC_UUID not found at offset {offset:#x}")

path = sys.argv[1]
with open(path, "r+b") as f:
    data = bytearray(f.read())
magic = struct.unpack_from(">I", data, 0)[0]
if magic == 0xCAFEBABE:
    nfat = struct.unpack_from(">I", data, 4)[0]
    for i in range(nfat):
        offset = struct.unpack_from(">I", data, 8 + i * 20 + 8)[0]
        zero_uuid(data, offset)
else:
    zero_uuid(data, 0)
with open(path, "wb") as f:
    f.write(data)
' "$APP/Contents/MacOS/Eidola"

                # autoSignDarwinBinariesHook re-signs the binary in
                # postFixup. The hook uses Nix's sigtool to produce a
                # deterministic ad-hoc signature, which is sufficient to
                # launch the binary on Apple Silicon (the bundle wraps it
                # for AppKit's benefit; signing happens at the Mach-O level).
                chmod -w "$APP/Contents/MacOS/Eidola"

                # Static bundle resources. Info.plist is referenced as a
                # direct path (outside the filtered cargo source) so its
                # content participates in the derivation hash — any edit
                # invalidates the cached build.
                cp ${./crates/eidola-gui/Support/Info.plist} "$APP/Contents/Info.plist"
                chmod -w "$APP/Contents/Info.plist"

                # The app icon. AppIcon.icns matches CFBundleIconFile
                # (pre-Tahoe); Assets.car matches CFBundleIconName (Tahoe
                # light / dark / tinted). Referenced as direct paths for the
                # same reason as Info.plist: the bytes then participate in
                # the derivation hash.
                cp ${./crates/eidola-gui/Support/AppIcon.icns} \
                  "$APP/Contents/Resources/AppIcon.icns"
                chmod -w "$APP/Contents/Resources/AppIcon.icns"
                cp ${./crates/eidola-gui/Support/Assets.car} \
                  "$APP/Contents/Resources/Assets.car"
                chmod -w "$APP/Contents/Resources/Assets.car"

                # Bundle the on-device inference engine sidecar next to the
                # main binary. app-core's rule is one line on every platform:
                # a `llama-server` sibling of the running executable — and
                # inside a bundle the executable's sibling is Contents/MacOS.
                # Contents/MacOS is also where Apple expects a bundled
                # executable to live, which is what keeps the signed bundle
                # and notarization straightforward.
                #
                # Deliberately arm64-only inside the universal app: the sidecar
                # comes from the aarch64-darwin llamaServer derivation and is
                # NOT lipo'd with an x86_64 slice. Intel Macs get no bundled
                # engine — Apple's Intel Metal stack is weak/EOL, so a CPU-only
                # x86_64 engine isn't worth its weight; the Local tab shows the
                # honest "engine not present" state there and `llama_server_path`
                # stays the escape hatch. Recorded as a product decision in
                # AGENTS.md.
                #
                # The sidecar derivation remaps `$NIX_BUILD_TOP` out of
                # `__FILE__` and zeros LC_UUID on Darwin, so its NAR is a
                # function of source; copying it is then deterministic.
                # autoSignDarwinBinariesHook's postFixup walks every Mach-O in
                # $out, so this sidecar is (re-)signed with the same deterministic
                # ad-hoc signature as the main binary; no explicit sign call.
                cp ${llamaServer}/bin/llama-server "$APP/Contents/MacOS/llama-server"
                chmod u+w "$APP/Contents/MacOS/llama-server"
              '';

              installPhase = ''
                echo "Eidola.app universal binary:"
                lipo -info "$out/Eidola.app/Contents/MacOS/Eidola"
              '';

              meta = {
                description = "Eidola GUI (macOS universal .app bundle)";
                platforms = [ "aarch64-darwin" ];
              };
            };

        # Build the GUI for Linux (Linux only). Unlike the server/cli — static
        # musl binaries — the GUI is a *glibc dynamic* binary by necessity:
        # a desktop app must interoperate with the host GPU stack (the Vulkan
        # loader dlopens Mesa ICDs, which every distro builds against glibc,
        # and a musl binary cannot dlopen glibc libraries). The Nix closure
        # supplies the full userland (wayland-client, libxkbcommon,
        # fontconfig, freetype, vulkan-loader), so the runtime host surface
        # is kernel + compositor socket + /dev/dri. Linux is Wayland-only by
        # decision — see crates/eidola-gui/AGENTS.md — so no X11 libraries
        # appear here. libxkbcommon *does* ship libxkbcommon-x11.so, which
        # upstream gpui unconditionally puts on the link line (its xkbcommon
        # crate features aren't gated by backend); --as-needed drops it from
        # DT_NEEDED, so the built binary stays X11-free at runtime.
        #
        # The artifact recorded in artifact-manifest.json is the *wrapped*
        # derivation below (eidolaGuiLinuxWrapped), which bundles Mesa; this
        # unwrapped build is the bare binary + library RUNPATH. End-user
        # packaging (AppImage/tarball/deb) is still a ship-time step on top.
        #
        # The library set below serves three roles: link-time inputs for the
        # release build, the binary's RUNPATH closure (rustc's link step does
        # not emit rpath entries for Nix store paths, and everything beyond
        # libxkbcommon is dlopened by bare soname — wayland-client, the
        # Vulkan loader, fontconfig, freetype — so without an explicit
        # RUNPATH the artifact can't resolve its libraries on any host), and
        # the `devShells.gui` link environment.
        guiLinuxLibs = [
          pkgs.wayland
          pkgs.libxkbcommon
          pkgs.fontconfig
          pkgs.freetype
          pkgs.vulkan-loader
        ];

        eidolaGuiLinux =
          if !pkgs.stdenv.isLinux then
            null
          else
            mkPackage {
              pname = "eidola-gui";
              rustTarget =
                {
                  "x86_64-linux" = "x86_64-unknown-linux-gnu";
                  "aarch64-linux" = "aarch64-unknown-linux-gnu";
                }
                .${system};
              nixCrossSystem = null;
              # Parity with the macOS GUI build: integration tests that need a
              # windowing/GPU environment can't run in the Nix sandbox; the
              # workspace-wide `checks.tests` target covers the test suite.
              doCheck = false;
              extraBuildArgs = {
                nativeBuildInputs = [
                  pkgs.cmake
                  pkgs.pkg-config
                ];
                buildInputs = guiLinuxLibs;
              };
              extraPackageArgs = {
                nativeBuildInputs = [
                  pkgs.cmake
                  pkgs.pkg-config
                  pkgs.patchelf
                ];
                # postFixup runs after fixupPhase's strip/shrink-rpath. That
                # pass keeps the RUNPATH entries for the binary's direct NEEDED
                # libs (glibc's libc/libm, libgcc_s, libxkbcommon) and drops the
                # dlopen-only ones (wayland, the Vulkan loader, fontconfig,
                # freetype), since nothing NEEDED resolves through them. *Append*
                # those back rather than `--set-rpath` overwriting: overwriting
                # would strip the compiler/runtime closure the linker recorded
                # (libc/libm/libgcc), so the artifact would hash fine but fail to
                # start once launched from the store. Deterministic: store paths
                # are pure functions of the flake inputs.
                postFixup = ''
                  patchelf --add-rpath "${pkgs.lib.makeLibraryPath guiLinuxLibs}" $out/bin/eidola-gui
                '';
              };
            };

        # Canonical byte-stream of a Nix payload for `archiveSha256`. GNU
        # tar POSIX/pax + gzip -n, timestamps/owners pinned, so the file
        # hash is a function of the payload tree rather than of the packer
        # host. `$out` is the `.tar.gz` itself (not a directory). See
        # docs/verification.md.
        #
        # pax, not ustar: ustar caps member names at 100 bytes (155-byte
        # prefix, splittable only at `/`) and — with no extension mechanism
        # — symlink targets at 100 bytes flat. A payload symlink pointing
        # into /nix/store blows that on its own. pax has no such limit.
        # The two pax options are what make it deterministic: the default
        # extended-header member name embeds tar's PID (`PaxHeaders.%p`),
        # and atime/ctime are packer-run state rather than payload content.
        # mtime is not deleted — --mtime=@0 already pins it, in both the
        # ustar-compatible header and the pax header.
        #
        # `--mode=u=rwX,go=rX` normalizes permissions to exactly the
        # information a NAR records (the executable bit and nothing else),
        # so the archive and narHash cannot disagree about modes.
        #
        # gzip's DEFLATE output is stable for a given implementation, so
        # the *gzip version* pinned by flake.lock is an input to
        # archiveSha256: a nixpkgs bump can move the archive hash with an
        # unchanged payload. That is exactly the divergence narHash exists
        # to isolate (see docs/verification.md).
        mkReproducibleArchive =
          {
            pname,
            payload,
          }:
          pkgs.runCommand "${pname}.tar.gz"
            {
              nativeBuildInputs = [
                pkgs.gnutar
                pkgs.gzip
              ];
            }
            ''
              export LC_ALL=C
              tar --format=posix \
                --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime \
                --sort=name \
                --mtime=@0 \
                --owner=0 --group=0 --numeric-owner \
                --mode=u=rwX,go=rX \
                -C ${payload} -cf - . \
              | gzip -n -9 > "$out"
            '';

        # The macOS shipping container: a deterministic zip of the payload,
        # in the shape a browser download and Gatekeeper expect. Distinct
        # from mkReproducibleArchive's `.tar.gz` and not a replacement for
        # it — `archiveSha256` stays the payload's canonical identity. What
        # this adds is a *published recipe* for the container itself, which
        # is what makes `apply(archive, envelope) = installable` checkable:
        # reconstructing the signed bundle only reproduces a tree, and the
        # shipped file is a zip, so re-zipping has to land on the same bytes
        # CI produced or the comparison has nothing to compare.
        #
        # The recipe itself lives in `scripts/pack-shipping-zip.sh`, which
        # this runs rather than restates. That is the whole point of the
        # split: a verifier has a tree this attribute cannot pack — some
        # other directory, usually on Linux, reconstructed by
        # `just verify-apple` — and two copies of a byte-exact recipe would
        # be two recipes. The script carries the reasoning for each flag,
        # including why Info-ZIP and never `ditto`.
        #
        # The copy is made here because the store path is read-only and the
        # script normalizes modes in place; `u+w` only makes the copy
        # writable, and the script then sets the exact mode set.
        mkShippingZip =
          {
            pname,
            payload,
          }:
          pkgs.runCommand "${pname}.zip"
            {
              nativeBuildInputs = [ pkgs.zip ];
            }
            ''
              cp -R ${payload} tree
              chmod -R u+w tree
              ${./scripts/pack-shipping-zip.sh} tree "$out"
            '';

        # The shippable Linux GUI: a *copied* GUI binary, a *copied*
        # llama-server sidecar (both sets of bytes in this NAR — same
        # measurement rule as macOS), and a *referenced* nixpkgs Mesa ICD
        # set.
        #
        # Both executables are copied rather than symlinked/wrapped-by-
        # reference on purpose. A makeWrapper over `${eidolaGuiLinux}/bin/
        # eidola-gui` would leave the application itself outside $out: the
        # payload would measure a wrapper script whose only claim on the
        # binary is a store *path*. Store paths here are input-addressed,
        # so that binds the build's inputs but not its output bytes — a
        # non-reproducible build or a compromised builder yields the same
        # path, therefore the same narHash/archiveSha256, while the user
        # resolves different bytes from their own substituter. Copying puts
        # the ELF in the measured tree, which is the rule docs/verification.md
        # states: functional bytes that ship inside the installable are in
        # the payload.
        #
        # The copy lands at `.eidola-gui-wrapped` (the nixpkgs wrapper
        # convention) so it stays a *sibling* of llama-server in $out/bin.
        # app-core resolves the engine from the running executable, which
        # after exec is the copy, so the sibling rule holds on its own;
        # EIDOLA_LLAMA_SERVER is set anyway as an explicit pin.
        #
        # Mesa stays a reference — the one deliberate exception, because
        # the device's GPU stack is pre-trusted. The host's own ICD
        # manifests are useless to a Nix binary — distros
        # reference drivers by bare soname (`libvulkan_radeon.so`), which
        # the loader resolves through the dynamic linker's search path,
        # i.e. our RUNPATH, never /usr/lib; and dlopening host Mesa into a
        # Nix glibc invites symbol-version mismatches. Bundling Mesa
        # (whose Nix ICD manifests carry absolute store paths) makes the
        # artifact's host surface kernel + compositor socket + /dev/dri,
        # at the cost of Mesa's closure (~1 GB, LLVM for the llvmpipe
        # fallback included). Those Mesa bytes are *not* copied into
        # $out: VK_ADD_DRIVER_FILES points at the store, so they sit
        # outside archiveSha256. A Flatpak against the Freedesktop GL
        # runtime is how an Ubuntu-shaped installable leaves Mesa out of
        # our payload entirely. VK_ADD_DRIVER_FILES is *additive* and set
        # only as a default, so a NixOS host's /run/opengl-driver drivers
        # coexist and a user can override outright.
        eidolaGuiLinuxWrapped =
          if eidolaGuiLinux == null then
            null
          else
            pkgs.runCommand "eidola-gui-linux"
              {
                nativeBuildInputs = [ pkgs.makeWrapper ];
              }
              ''
                mkdir -p $out/bin $out/share/applications
                cp ${eidolaGuiLinux}/bin/eidola-gui "$out/bin/.eidola-gui-wrapped"
                chmod u+w "$out/bin/.eidola-gui-wrapped"
                cp ${llamaServer}/bin/llama-server "$out/bin/llama-server"
                chmod u+w "$out/bin/llama-server"
                icds=$(ls ${pkgs.mesa}/share/vulkan/icd.d/*.json | tr '\n' ':')
                makeWrapper "$out/bin/.eidola-gui-wrapped" $out/bin/eidola-gui \
                  --set-default VK_ADD_DRIVER_FILES "''${icds%:}" \
                  --set-default EIDOLA_LLAMA_SERVER "$out/bin/llama-server"
                # Desktop entry — basename matches the Wayland app_id the
                # binary sets (lib.rs APP_ID), which is what lets the shell
                # resolve our windows to this entry. Strip the comment header
                # (leading lines starting with '#'): desktop-file spec allows
                # comments, but some validators are strict about the first
                # line being the group header.
                grep -v '^#' ${./releases/linux/ai.eidola.app.desktop} \
                  > $out/share/applications/ai.eidola.app.desktop
                # Themed icons the entry's Icon= key resolves to: the scalable
                # SVG every modern shell prefers (light/dark via
                # prefers-color-scheme), the symbolic mark, plus fixed-size
                # PNGs for shells that don't rasterize. Generated by
                # `just update-brand`.
                mkdir -p $out/share/icons
                cp -r --no-preserve=mode ${./releases/linux/icons}/. \
                  $out/share/icons/
              '';

        # ── The host-generic Linux payload and its .deb envelope ────────────
        #
        # The Nix installable above hands the GUI a complete userland out of
        # the store. This one does the opposite: it keeps only our own bytes
        # and resolves everything else from the host distro, which is what
        # lets the host's Vulkan loader and its Mesa or proprietary NVIDIA
        # drivers — all built against the host's glibc — load into our
        # process. The Nix-glibc/host-driver mismatch the wrapped build works
        # around by bundling Mesa does not arise here.
        #
        # Two pieces of post-link metadata are all that bind the release
        # binary to the store, and patchelf removes both:
        #
        #   * RUNPATH, which points every lookup at the store. Dropping it is
        #     safe because the only DT_NEEDED library outside glibc is
        #     libxkbcommon; the Wayland client, the Vulkan loader and EGL are
        #     dlopened by bare soname, and a bare soname is resolved through
        #     the loader's default search path.
        #   * PT_INTERP, the store's ld.so, rewritten to the path the
        #     filesystem hierarchy standard gives glibc's loader.
        #
        # Nothing else changes: same source, same release profile, same
        # linker output.
        fdoInterpreter =
          {
            "x86_64-linux" = "/lib64/ld-linux-x86-64.so.2";
            "aarch64-linux" = "/lib/ld-linux-aarch64.so.1";
          }
          .${system} or null;

        eidolaGuiLinuxFdo =
          if eidolaGuiLinux == null then
            null
          else
            pkgs.runCommand "eidola-gui-linux-fdo"
              {
                nativeBuildInputs = [ pkgs.patchelf ];
              }
              ''
                mkdir -p $out/bin
                cp ${eidolaGuiLinux}/bin/eidola-gui $out/bin/eidola-gui
                chmod u+w $out/bin/eidola-gui
                patchelf --remove-rpath $out/bin/eidola-gui
                patchelf --set-interpreter ${fdoInterpreter} $out/bin/eidola-gui
                ${assertGlibcFloor "$out/bin/eidola-gui"}
                chmod a-w $out/bin/eidola-gui
              '';

        # Debian dependencies, grouped by how each one is discovered, because
        # only the first group can be checked against the artifact.
        #
        # DT_NEEDED — resolved eagerly by the loader. `assertDebNeededCovered`
        # below fails the build if the shipped binary names a soname this
        # table does not, so the package can never declare less than it links.
        # The glibc entry carries the version floor, which turns an
        # unsupported host into an apt refusal instead of a loader error.
        debNeededPackages = {
          "libc.so.6" = "libc6 (>= ${glibcSymbolFloor})";
          "libm.so.6" = "libc6 (>= ${glibcSymbolFloor})";
          # aarch64 links the loader itself; x86-64 does not, but both are
          # libc6 and listing both keeps this table arch-independent.
          "ld-linux-aarch64.so.1" = "libc6 (>= ${glibcSymbolFloor})";
          "ld-linux-x86-64.so.2" = "libc6 (>= ${glibcSymbolFloor})";
          "libgcc_s.so.1" = "libgcc-s1";
          "libxkbcommon.so.0" = "libxkbcommon0";
        };

        # dlopened by bare soname, so absent from the ELF header and not
        # derivable from it. Read out of the binary's string table:
        # libwayland-client.so.0, libwayland-egl.so.1, libEGL.so.1,
        # libvulkan.so.1. Note what is *not* here — fontconfig and freetype
        # are Rust reimplementations in this build (`fontconfig_parser`), so
        # neither library is loaded.
        debDlopenPackages = [
          "libwayland-client0"
          "libwayland-egl1"
          "libegl1"
          "libvulkan1"
        ];

        # Not dependencies: a Vulkan ICD is a property of the machine's GPU
        # (a host with proprietary NVIDIA drivers must not be made to install
        # Mesa), and the font files the pure-Rust text stack reads off disk
        # are supplied by any desktop install. Both are needed for the app to
        # be useful, which is what Recommends means, and apt installs
        # Recommends by default.
        debRecommendsPackages = [
          "mesa-vulkan-drivers"
          "fonts-dejavu-core"
        ];

        assertDebNeededCovered = binary: ''
          echo "checking that every DT_NEEDED soname has a declared package: ${binary}"
          for soname in $(
            ${readelfBin} -dW "${binary}" \
              | grep NEEDED | tr -d '[]' | tr -s ' ' '\n' | grep '\.so'
          ); do
            case " ${pkgs.lib.concatStringsSep " " (builtins.attrNames debNeededPackages)} " in
              *" $soname "*) ;;
              *)
                echo "error: ${binary} needs $soname, which no Debian package is declared for" >&2
                echo "add it to debNeededPackages in flake.nix" >&2
                exit 1
                ;;
            esac
          done
        '';

        debArchitecture =
          {
            "x86_64-linux" = "amd64";
            "aarch64-linux" = "arm64";
          }
          .${system} or null;

        debVersion = workspaceCargoToml.workspace.package.version;

        # The installed tree, exactly as it lands on the host.
        #
        # Both executables sit in /usr/libexec/eidola and /usr/bin/eidola-gui
        # is a symlink to the GUI. That layout is not cosmetic: app-core
        # resolves the inference engine as a `llama-server` sibling of the
        # running executable, and /proc/self/exe reports the symlink's target,
        # so a launch through /usr/bin lands in /usr/libexec/eidola and finds
        # the sidecar there. The engine rule is unchanged on every platform.
        eidolaLinuxDebTree =
          if eidolaGuiLinuxFdo == null then
            null
          else
            pkgs.runCommand "eidola-deb-tree"
              {
                nativeBuildInputs = [
                  pkgs.appstream
                  pkgs.desktop-file-utils
                ];
              }
              ''
                install -Dm755 ${eidolaGuiLinuxFdo}/bin/eidola-gui \
                  $out/usr/libexec/eidola/eidola-gui
                install -Dm755 ${llamaServer}/bin/llama-server \
                  $out/usr/libexec/eidola/llama-server
                mkdir -p $out/usr/bin
                ln -s ../libexec/eidola/eidola-gui $out/usr/bin/eidola-gui

                # Desktop entry — see the Nix installable above for why the
                # comment header is stripped and why the basename is fixed.
                mkdir -p $out/usr/share/applications
                grep -v '^#' ${./releases/linux/ai.eidola.app.desktop} \
                  > $out/usr/share/applications/ai.eidola.app.desktop

                mkdir -p $out/usr/share/icons
                cp -r --no-preserve=mode ${./releases/linux/icons}/. \
                  $out/usr/share/icons/

                # App-centre metadata. GNOME Software and KDE Discover render
                # an installed app's page from this file.
                install -Dm644 ${./releases/linux/ai.eidola.app.metainfo.xml} \
                  $out/usr/share/metainfo/ai.eidola.app.metainfo.xml

                # Both files are read by software nobody here controls, on
                # machines we never see, so they are validated where a
                # mistake is still cheap.
                desktop-file-validate \
                  $out/usr/share/applications/ai.eidola.app.desktop
                appstreamcli validate --no-net --pedantic \
                  $out/usr/share/metainfo/ai.eidola.app.metainfo.xml
              '';

        # The .deb: an `ar` archive of three members in a fixed order, the
        # last two of which are tarballs of the control and data trees.
        #
        # Everything here is spelled out rather than delegated to dpkg-deb so
        # the byte stream is a function of the tree: `ar -D` zeroes member
        # timestamps, uids and modes, and the two tars carry the same
        # discipline as `mkReproducibleArchive` above — sorted members, pinned
        # mtime, no owner names, modes normalized. GNU tar format rather than
        # that function's pax, because dpkg's extractor reads ustar plus the
        # GNU long-name extensions and nothing else. dpkg-deb is still
        # present, as the check that what we assembled is a package dpkg
        # agrees with.
        eidolaLinuxDeb =
          if eidolaLinuxDebTree == null then
            null
          else
            pkgs.runCommand "eidola_${debVersion}_${debArchitecture}.deb"
              {
                nativeBuildInputs = [
                  pkgs.gnutar
                  pkgs.gzip
                  pkgs.binutils
                  pkgs.dpkg
                ];
              }
              ''
                export LC_ALL=C
                tree=${eidolaLinuxDebTree}

                ${assertDebNeededCovered "$tree/usr/libexec/eidola/eidola-gui"}
                ${assertGlibcFloor "$tree/usr/libexec/eidola/eidola-gui"}
                ${assertFullyStatic "$tree/usr/libexec/eidola/llama-server"}

                tarFlags="--format=gnu --sort=name --mtime=@0 \
                  --owner=0 --group=0 --numeric-owner --mode=u=rwX,go=rX"

                # data.tar.gz — the installed tree, rooted at `./`.
                # shellcheck disable=SC2086
                tar $tarFlags -C "$tree" -cf - . | gzip -n -9 > data.tar.gz

                mkdir control

                # md5sums, over every regular file in the payload. dpkg reads
                # it for `dpkg --verify` and for conffile-less integrity
                # checks; the symlink has no entry because it has no content.
                ( cd "$tree" && find . -type f | sort | \
                  while read -r f; do
                    md5sum "$f"
                  done ) | sed 's|\./||' > control/md5sums

                installedSize=$(du -s -k --apparent-size "$tree" | cut -f1)

                cat > control/control <<EOF
                Package: eidola
                Version: ${debVersion}
                Architecture: ${debArchitecture}
                Maintainer: Eidola, Inc. <hello@eidola.ai>
                Installed-Size: $installedSize
                Section: utils
                Priority: optional
                Homepage: https://www.eidola.ai
                Depends: ${
                  pkgs.lib.concatStringsSep ", " (
                    pkgs.lib.naturalSort (
                      pkgs.lib.unique (builtins.attrValues debNeededPackages ++ debDlopenPackages)
                    )
                  )
                }
                Recommends: ${pkgs.lib.concatStringsSep ", " debRecommendsPackages}
                Description: private AI chat client
                 A personal AI client whose privacy rests on verifiable architecture
                 and open code rather than on a policy taken on faith. Conversations
                 run against models in confidential-computing enclaves that the app
                 verifies on every connection, or entirely on this device.
                 .
                 The bundled inference engine and the application binary are the only
                 files this package installs beyond its desktop metadata; the Wayland,
                 Vulkan and GPU stacks come from the host.
                EOF

                # shellcheck disable=SC2086
                tar $tarFlags -C control -cf - . | gzip -n -9 > control.tar.gz

                echo "2.0" > debian-binary

                # `ar -D`: deterministic mode, which zeroes each member's
                # timestamp, uid, gid and mode. Member order is dpkg's
                # requirement, not ar's.
                ar rcD "$out" debian-binary control.tar.gz data.tar.gz

                echo "── dpkg-deb --info ──"
                dpkg-deb --info "$out"
                echo "── dpkg-deb --contents ──"
                dpkg-deb --contents "$out"
              '';

      in
      {
        packages = {
          server = mkPackage {
            pname = "eidola-server";
            rustTarget = nativeRustTarget;
            nixCrossSystem = null;
          };
          server-openapi-spec = serverOpenApiSpec;
          # Static llama.cpp `llama-server` — the bundled on-device inference
          # engine sidecar. Buildable on its own (`nix build .#llama-server`)
          # for the dev-path `just engine` recipe.
          llama-server = llamaServer;
          # Apple detached-signature tool. Build-time/CI only — never shipped.
          signapple = signappleTool;
        }
        // pkgs.lib.optionalAttrs (eidolaCliMacosUniversal != null) {
          eidola-cli-macos-universal = eidolaCliMacosUniversal;
          eidola-cli-macos-universal-archive = mkReproducibleArchive {
            pname = "eidola-cli-macos-universal";
            payload = eidolaCliMacosUniversal;
          };
        }
        // pkgs.lib.optionalAttrs (eidolaGuiMacosUniversal != null) {
          eidola-gui-macos-universal = eidolaGuiMacosUniversal;
          eidola-gui-macos-universal-archive = mkReproducibleArchive {
            pname = "eidola-gui-macos-universal";
            payload = eidolaGuiMacosUniversal;
          };
          # The unsigned shipping container. Signing happens outside Nix
          # (it needs a key), so this is the *unsigned* zip: the recipe a
          # signed release's zip is produced by, exercised on every build.
          eidola-gui-macos-universal-zip = mkShippingZip {
            pname = "eidola-gui-macos-universal";
            payload = eidolaGuiMacosUniversal;
          };
        }
        // pkgs.lib.optionalAttrs (eidolaGuiLinux != null) {
          eidola-gui-linux = eidolaGuiLinuxWrapped;
          eidola-gui-linux-unwrapped = eidolaGuiLinux;
          eidola-gui-linux-archive = mkReproducibleArchive {
            pname = "eidola-gui-linux";
            payload = eidolaGuiLinuxWrapped;
          };
          # The host-generic GUI binary, and the Debian package built over it.
          eidola-gui-linux-fdo = eidolaGuiLinuxFdo;
          eidola-linux-deb-tree = eidolaLinuxDebTree;
          eidola-linux-deb = eidolaLinuxDeb;
        };

        # Development shell (lightweight — daily Rust dev uses rustup)
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Pin GitHub actions
            pkgs.pinact
          ];
        };

        # GUI dev shell for Linux: provides the system libraries and
        # pkg-config that `cargo build -p eidola-gui` needs to link (daily
        # Rust dev still uses rustup — this shell only supplies the C-level
        # build environment). The built binary runs against the host's own
        # runtime libraries (wayland/vulkan/fontconfig are dlopened by
        # name), so this is link-time-only tooling. On macOS the system
        # frameworks cover everything and this shell is unnecessary.
        devShells.gui = pkgs.mkShell {
          buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux (
            [
              pkgs.pkg-config
              pkgs.cmake
            ]
            ++ guiLinuxLibs
          );
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

        };
      }
    );
}
