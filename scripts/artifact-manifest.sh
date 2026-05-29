#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/artifact-manifest.sh build [--push] [--metadata-file PATH] [--builder NAME] [--ensure-builder] [--targets GROUP] [--set PATTERN=VALUE ...]
  scripts/artifact-manifest.sh print [--push] [(--metadata-file PATH [--targets "NAME ..."])...]
  scripts/artifact-manifest.sh verify [--push] [(--metadata-file PATH [--targets "NAME ..."])...] [--manifest PATH]
  scripts/artifact-manifest.sh build-macos [--output PATH]
  scripts/artifact-manifest.sh measure [--config PATH] [--verify-attestations] [--server-enclave-output PATH]
  scripts/artifact-manifest.sh verify-full [--partial PATH ...] [--manifest PATH] [--config PATH] [--server-enclave PATH] [--output PATH] [--server-enclave-output PATH] [--verify-attestations]
  scripts/artifact-manifest.sh stamp-config [--metadata-file PATH] [--config PATH]
  scripts/artifact-manifest.sh update [--output PATH] [--metadata-file PATH] [--builder NAME] [--ensure-builder]

Options:
  --push                       Push images directly from BuildKit to the registry (uses ci
                               bake group with type=image,push=true). Requires REGISTRY and
                               TAGS env vars. Without this flag, images are built in BuildKit
                               without push for digest computation (type=image,push=false).
                               Requires a docker-container driver (--ensure-builder or
                               setup-buildx-action).
  --targets GROUP              For `build`: bake group to build (default: full manifest).
                               Recognized values: `all` (default), `server`, `cli`. Both push
                               and non-push modes accept the split selectors so CI's
                               two-phase build can push each phase independently.
                               For `print`/`verify`: space-separated list of target names
                               whose digests to read from the preceding --metadata-file
                               (e.g. "server postgres" or "cli"). Pairs are matched
                               positionally with --metadata-file occurrences.
  --metadata-file PATH         Path to a buildx bake metadata file. May be repeated for
                               `print`/`verify` to span multiple builds; each repetition
                               must be paired with a --targets value naming the targets to
                               read from that file.
  --server-enclave PATH        Path to server-enclave.json (default:
                               releases/trust/server-enclave.json). Valid for `verify-full`.
  --server-enclave-output PATH Write the computed enclave block (with `schema_version: 1`
                               envelope) to PATH. Valid for `measure` and `verify-full`.
  --verify-attestations        Verify CVM manifest provenance via Sigstore (requires gh CLI).
                               Fails the command if attestation verification fails.
EOF
}

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "error: not in a git repository" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILDKIT_IMAGE="moby/buildkit:v0.28.0@sha256:60bfb07e39a6e524e78e6c4723114902c6b61ee36714493e357e39861bea753b"

# CVM image artifacts for enclave measurement computation.
# The OVMF firmware version is pinned to match tinfoilsh/measure-image-action.
CVM_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/eidola/cvm"
OVMF_VERSION="v0.0.3"
OVMF_URL="https://github.com/tinfoilsh/edk2/releases/download/${OVMF_VERSION}/OVMF.fd"

COMMAND="${1:-}"
if [[ "$COMMAND" = "-h" || "$COMMAND" = "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$COMMAND" ]]; then
  usage >&2
  exit 1
fi
shift

METADATA_FILE="/tmp/bake-metadata.json"
OUTPUT_FILE=""
MANIFEST_FILE="$REPO_ROOT/artifact-manifest.json"
SERVER_ENCLAVE_FILE="$REPO_ROOT/releases/trust/server-enclave.json"
SERVER_ENCLAVE_OUTPUT=""
BUILDER_NAME="eidola"
ENSURE_BUILDER=0
CLI_PATH=""
GUI_PATH=""
CONFIG_FILE="$REPO_ROOT/tinfoil-config.yml"
VERIFY_ATTESTATIONS=0
PUSH_MODE=0
TARGETS="all"
PARTIAL_FILES=()
BUILDX_SET_ARGS=()
# Parallel arrays for `print`/`verify` multi-metadata mode. Each --metadata-file
# occurrence appends to PRINT_METADATA_FILES, each --targets to PRINT_TARGETS_LIST;
# the two are paired positionally. If TARGETS_LIST is shorter (i.e. fewer
# --targets than --metadata-files), the missing trailing entries default to
# "server cli postgres" for backward compatibility with single-file callers.
PRINT_METADATA_FILES=()
PRINT_TARGETS_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metadata-file)
      METADATA_FILE="$2"
      PRINT_METADATA_FILES+=("$2")
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_FILE="$2"
      shift 2
      ;;
    --builder)
      BUILDER_NAME="$2"
      shift 2
      ;;
    --ensure-builder)
      ENSURE_BUILDER=1
      shift
      ;;
    --partial)
      PARTIAL_FILES+=("$2")
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --push)
      PUSH_MODE=1
      shift
      ;;
    --set)
      BUILDX_SET_ARGS+=("$2")
      shift 2
      ;;
    --targets)
      TARGETS="$2"
      PRINT_TARGETS_LIST+=("$2")
      shift 2
      ;;
    --server-enclave)
      SERVER_ENCLAVE_FILE="$2"
      shift 2
      ;;
    --server-enclave-output)
      SERVER_ENCLAVE_OUTPUT="$2"
      shift 2
      ;;
    --verify-attestations)
      VERIFY_ATTESTATIONS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ── CVM artifact fetching ─────────────────────────────────────────────────────

# Read the cvm-version field from tinfoil-config.yml.
read_cvm_version() {
  grep -E '^cvm-version:' "$CONFIG_FILE" | sed 's/^cvm-version:[[:space:]]*//'
}

# Download a URL to a local cache path, skipping if already present.
cached_download() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "Downloading $(basename "$dest")..." >&2
  curl -fsSL --retry 3 -o "$dest" "$url"
}

# Fetch CVM image artifacts (OVMF, kernel, initrd, manifest) and verify
# kernel/initrd hashes against the CVM manifest.
fetch_cvm_artifacts() {
  local cvm_version cache_dir manifest_url manifest_file
  local kernel_url kernel_file kernel_hash
  local initrd_url initrd_file initrd_hash
  local ovmf_file

  cvm_version="$(read_cvm_version)"
  cache_dir="${CVM_CACHE_DIR}/${cvm_version}"

  # Fetch CVM manifest
  manifest_url="https://github.com/tinfoilsh/cvmimage/releases/download/v${cvm_version}/tinfoil-inference-v${cvm_version}-manifest.json"
  manifest_file="${cache_dir}/manifest.json"
  cached_download "$manifest_url" "$manifest_file"

  # Extract expected hashes
  kernel_hash="$(jq -er '.kernel' "$manifest_file")"
  initrd_hash="$(jq -er '.initrd' "$manifest_file")"

  # Fetch kernel
  kernel_url="https://images.tinfoil.sh/cvm/tinfoil-inference-v${cvm_version}.vmlinuz"
  kernel_file="${cache_dir}/vmlinuz"
  cached_download "$kernel_url" "$kernel_file"

  # Fetch initrd
  initrd_url="https://images.tinfoil.sh/cvm/tinfoil-inference-v${cvm_version}.initrd"
  initrd_file="${cache_dir}/initrd"
  cached_download "$initrd_url" "$initrd_file"

  # Fetch OVMF (version-independent, cached by OVMF version)
  ovmf_file="${CVM_CACHE_DIR}/OVMF-${OVMF_VERSION}.fd"
  cached_download "$OVMF_URL" "$ovmf_file"

  # Verify kernel and initrd against manifest hashes
  local actual_kernel_hash actual_initrd_hash

  compute_sha256() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$file" | cut -d' ' -f1
    else
      shasum -a 256 "$file" | cut -d' ' -f1
    fi
  }

  actual_kernel_hash="$(compute_sha256 "$kernel_file")"
  actual_initrd_hash="$(compute_sha256 "$initrd_file")"

  if [[ "$actual_kernel_hash" != "$kernel_hash" ]]; then
    echo "error: kernel hash mismatch" >&2
    echo "  expected: $kernel_hash" >&2
    echo "  actual:   $actual_kernel_hash" >&2
    rm -f "$kernel_file"
    exit 1
  fi

  if [[ "$actual_initrd_hash" != "$initrd_hash" ]]; then
    echo "error: initrd hash mismatch" >&2
    echo "  expected: $initrd_hash" >&2
    echo "  actual:   $actual_initrd_hash" >&2
    rm -f "$initrd_file"
    exit 1
  fi

  # When --verify-attestations is set, verify CVM manifest provenance via
  # Sigstore. This checks that the manifest was built on GitHub-hosted runners
  # in the tinfoilsh/cvmimage repo. Fails hard if verification fails.
  if [[ "$VERIFY_ATTESTATIONS" -eq 1 ]]; then
    echo "Verifying CVM manifest attestation..." >&2
    if ! gh attestation verify "$manifest_file" -R tinfoilsh/cvmimage --deny-self-hosted-runners; then
      echo "error: CVM manifest attestation verification failed" >&2
      echo "  The manifest hash checks passed, but Sigstore provenance could not be verified." >&2
      echo "  This may indicate the release was not built on GitHub-hosted runners." >&2
      exit 1
    fi
    echo "CVM manifest attestation verified." >&2
  fi

  # Export paths for use by compute_measurements
  CVM_OVMF="$ovmf_file"
  CVM_KERNEL="$kernel_file"
  CVM_INITRD="$initrd_file"
  CVM_ROOTHASH="$(jq -er '.root' "$manifest_file")"
}

# ── Enclave measurement ──────────────────────────────────────────────────────

# Update the eidola-server image digest in tinfoil-config.yml from build
# metadata. (Only the server runs inside the enclave; the database is
# hosted externally, so eidola-postgres's digest doesn't feed the
# measurement.)
stamp_config_digests() {
  local server_digest

  server_digest="$(metadata_digest "$METADATA_FILE" "$(target_key server)")"

  sed -i.bak \
    -e "s|ghcr.io/eidola-ai/eidola-server@sha256:[a-f0-9]*|ghcr.io/eidola-ai/eidola-server@${server_digest}|" \
    "$CONFIG_FILE"
  rm -f "${CONFIG_FILE}.bak"
}

# Compute enclave measurements using the measure-enclave binary.
# Requires CVM artifacts to be fetched first (sets CVM_* variables).
compute_measurements() {
  fetch_cvm_artifacts

  cargo run -q -p measure-enclave -- \
    --config "$CONFIG_FILE" \
    --ovmf "$CVM_OVMF" \
    --kernel "$CVM_KERNEL" \
    --initrd "$CVM_INITRD" \
    --roothash "$CVM_ROOTHASH"
}

# Wrap a bare enclave-measurement JSON (as emitted by measure-enclave) in the
# `{schema_version, snp_measurement, tdx_measurement, cmdline}` envelope used
# by `releases/trust/server-enclave.json`. Writes to PATH if given, else
# stdout. The schema_version is the same integer-versioned scheme used by
# the rest of the trust-root JSON files.
write_server_enclave_envelope() {
  local enclave="$1" out_path="$2"
  local enveloped

  enveloped="$(printf '%s\n' "$enclave" | jq -S '{schema_version: 1} + .')"

  if [[ -n "$out_path" ]]; then
    mkdir -p "$(dirname "$out_path")"
    printf '%s\n' "$enveloped" > "$out_path"
  else
    printf '%s\n' "$enveloped"
  fi
}

# ── Builder management ────────────────────────────────────────────────────────

ensure_builder() {
  local inspect_output needs_recreate

  needs_recreate=0
  if ! inspect_output="$(docker buildx inspect "$BUILDER_NAME" 2>/dev/null)"; then
    needs_recreate=1
  elif ! grep -Fq "image=\"${BUILDKIT_IMAGE}\"" <<<"$inspect_output"; then
    needs_recreate=1
  elif ! grep -Fq "linux/amd64*" <<<"$inspect_output"; then
    needs_recreate=1
  fi

  if [[ "$needs_recreate" -eq 1 ]]; then
    if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
      docker buildx rm "$BUILDER_NAME" >/dev/null
    fi

    echo "Creating docker-container builder '$BUILDER_NAME'..."
    docker buildx create \
      --name "$BUILDER_NAME" \
      --driver docker-container \
      --platform linux/amd64 \
      --driver-opt "image=${BUILDKIT_IMAGE}" \
      >/dev/null
  fi

  docker buildx inspect "$BUILDER_NAME" --bootstrap >/dev/null
}

# Return the bake metadata key for a target name.
# In push mode, the ci bake group prefixes targets with "ci-".
target_key() {
  local name="$1"
  if [[ "$PUSH_MODE" -eq 1 ]]; then
    echo "ci-${name}"
  else
    echo "$name"
  fi
}

build_metadata() {
  local -a builder_args buildx_set_args bake_targets
  builder_args=()
  buildx_set_args=()

  if [[ "$ENSURE_BUILDER" -eq 1 ]]; then
    ensure_builder
    builder_args=(--builder "$BUILDER_NAME")
  fi

  for buildx_set_arg in ${BUILDX_SET_ARGS[@]+"${BUILDX_SET_ARGS[@]}"}; do
    buildx_set_args+=(--set "$buildx_set_arg")
  done

  # Pick the exact bake targets for this phase. Push mode uses the
  # registry-push variants (`ci-*`) defined in docker-bake.hcl; non-push
  # mode uses the dev variants. Both modes accept the split `server` /
  # `cli` selectors so the two-phase build (server first, then cli after
  # the enclave is recomputed) can run in either mode — CI's `oci` job
  # uses this in push mode; the local `update_manifest` uses it without
  # --push.
  if [[ "$PUSH_MODE" -eq 1 ]]; then
    case "$TARGETS" in
      all)    bake_targets=(ci-server ci-cli ci-postgres) ;;
      server) bake_targets=(ci-server ci-postgres) ;;
      cli)    bake_targets=(ci-cli) ;;
      *)
        echo "error: unknown --targets value: $TARGETS (expected: all, server, cli)" >&2
        exit 1
        ;;
    esac
  else
    case "$TARGETS" in
      all)    bake_targets=(server cli postgres) ;;
      server) bake_targets=(server postgres) ;;
      cli)    bake_targets=(cli) ;;
      *)
        echo "error: unknown --targets value: $TARGETS (expected: all, server, cli)" >&2
        exit 1
        ;;
    esac
    # Build OCI images locally for digest computation. No push, no daemon load.
    # Requires a docker-container driver (--ensure-builder or setup-buildx-action).
    buildx_set_args+=(--set '*.output=type=image,push=false,rewrite-timestamp=true,force-compression=true,compression=gzip,oci-mediatypes=true')
  fi

  docker buildx bake "${bake_targets[@]}" \
    ${builder_args[@]+"${builder_args[@]}"} \
    ${buildx_set_args[@]+"${buildx_set_args[@]}"} \
    --metadata-file "$METADATA_FILE"
}

# Extract a single image digest from a bake metadata file by target name.
# Returns the bare `sha256:...` string. Caller supplies the (push-aware)
# target key.
metadata_digest() {
  local metadata_file="$1" tgt="$2"
  jq -er '."'"$tgt"'"."containerimage.digest" | select(type == "string" and startswith("sha256:"))' "$metadata_file"
}

# Build a partial artifact-manifest from a list of target names.
# Each target is read from $METADATA_FILE using its push-aware key.
print_oci_partial_for_targets() {
  local -a targets=("$@")
  local target digest jq_filter

  jq_filter='{ schema_version: 1, artifacts: {} }'
  for target in "${targets[@]}"; do
    digest="$(metadata_digest "$METADATA_FILE" "$(target_key "$target")")"
    jq_filter+=" | .artifacts[\"eidola-${target}\"] = { type: \"oci\", platform: \"linux/amd64\", digest: \"${digest}\" }"
  done

  jq -n "$jq_filter"
}

print_oci_manifest() {
  # Default behavior (no explicit pairs): emit server+cli+postgres digests
  # from the single legacy METADATA_FILE. Preserves backward compat with the
  # one-shot oci bake.
  if [[ "${#PRINT_METADATA_FILES[@]}" -eq 0 && "${#PRINT_TARGETS_LIST[@]}" -eq 0 ]]; then
    print_oci_partial_for_targets server cli postgres
    return
  fi

  # Single --metadata-file with no --targets: same legacy default but allow
  # the caller to point at a non-default metadata file.
  if [[ "${#PRINT_METADATA_FILES[@]}" -eq 1 && "${#PRINT_TARGETS_LIST[@]}" -eq 0 ]]; then
    print_oci_partial_for_targets server cli postgres
    return
  fi

  if [[ "${#PRINT_METADATA_FILES[@]}" != "${#PRINT_TARGETS_LIST[@]}" ]]; then
    echo "error: --metadata-file and --targets must be paired (got ${#PRINT_METADATA_FILES[@]} metadata files, ${#PRINT_TARGETS_LIST[@]} targets values)" >&2
    exit 1
  fi

  # Multi-mode: each (metadata-file, targets) pair produces a partial; merge
  # them all into a single artifact partial.
  local i partials_json saved_metadata_file partial
  saved_metadata_file="$METADATA_FILE"
  partials_json="[]"

  for ((i = 0; i < ${#PRINT_METADATA_FILES[@]}; i++)); do
    local -a tgts=()
    read -ra tgts <<<"${PRINT_TARGETS_LIST[i]}"
    METADATA_FILE="${PRINT_METADATA_FILES[i]}"
    partial="$(print_oci_partial_for_targets "${tgts[@]}")"
    partials_json="$(jq -n --argjson acc "$partials_json" --argjson p "$partial" '$acc + [$p]')"
  done

  METADATA_FILE="$saved_metadata_file"
  printf '%s\n' "$partials_json" | jq '{
    schema_version: 1,
    artifacts: (reduce .[] as $p ({}; . + ($p.artifacts // {})))
  }'
}

build_macos_artifacts() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: macOS artifact builds require a Darwin host" >&2
    exit 1
  fi

  CLI_PATH="$(
    nix build \
      '.#eidola-cli-macos-universal' \
      --no-link \
      --print-out-paths \
      --show-trace
  )"

  GUI_PATH="$(
    nix build \
      '.#eidola-gui-macos-universal' \
      --no-link \
      --print-out-paths \
      --show-trace
  )"
}

nix_nar_hash() {
  local store_path="$1"

  nix path-info --json "$store_path" \
    | jq -er --arg path "$store_path" '.[$path].narHash | select(type == "string" and startswith("sha256-"))'
}

print_macos_manifest() {
  local cli_hash gui_hash

  if [[ -z "${CLI_PATH:-}" || -z "${GUI_PATH:-}" ]]; then
    echo "error: print_macos_manifest called before build_macos_artifacts" >&2
    return 1
  fi

  cli_hash="$(nix_nar_hash "$CLI_PATH")"
  gui_hash="$(nix_nar_hash "$GUI_PATH")"

  jq -n \
    --arg cli_hash "$cli_hash" \
    --arg gui_hash "$gui_hash" \
    '{
      schema_version: 1,
      artifacts: {
        "eidola-cli-macos-universal": { type: "nix", platform: "darwin/universal", narHash: $cli_hash },
        "eidola-gui-macos-universal": { type: "nix", platform: "darwin/universal", narHash: $gui_hash }
      }
    }'
}

write_output() {
  local content="$1"

  if [[ -n "$OUTPUT_FILE" ]]; then
    printf '%s\n' "$content" > "$OUTPUT_FILE"
  else
    printf '%s\n' "$content"
  fi
}

write_temp_file() {
  local content="$1"
  local tmp_file

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/artifact-manifest.XXXXXX")"
  printf '%s\n' "$content" > "$tmp_file"
  printf '%s\n' "$tmp_file"
}

merge_partials() {
  if [[ "${#PARTIAL_FILES[@]}" -eq 0 ]]; then
    echo "error: provide at least one --partial file" >&2
    exit 1
  fi

  local merged
  merged="$(jq -s '
    {
      schema_version: 1,
      artifacts: (reduce .[] as $partial ({}; . + ($partial.artifacts // {})))
    }
  ' "${PARTIAL_FILES[@]}")"

  # If enclave measurements were computed, merge them in
  if [[ -n "${ENCLAVE_MEASUREMENTS:-}" ]]; then
    merged="$(printf '%s\n' "$merged" | jq \
      --argjson enclave "$ENCLAVE_MEASUREMENTS" \
      '. + {enclave: $enclave}')"
  fi

  # Sort keys for canonical output (matches the -cS normalization in verify)
  printf '%s\n' "$merged" | jq -S .
}

verify_full_manifest() {
  local actual_norm committed_norm actual_manifest
  local expected_envelope committed_envelope rc=0

  # Recompute enclave measurements from committed config if not already set
  if [[ -z "${ENCLAVE_MEASUREMENTS:-}" ]]; then
    ENCLAVE_MEASUREMENTS="$(compute_measurements)"
  fi

  expected_envelope="$(write_server_enclave_envelope "$ENCLAVE_MEASUREMENTS" "")"
  if [[ -n "$SERVER_ENCLAVE_OUTPUT" ]]; then
    mkdir -p "$(dirname "$SERVER_ENCLAVE_OUTPUT")"
    printf '%s\n' "$expected_envelope" > "$SERVER_ENCLAVE_OUTPUT"
  fi

  # Consistency check: the committed `releases/trust/server-enclave.json`
  # must match the enclave block we just recomputed from
  # `tinfoil-config.yml`. CI overwrites the committed file with the
  # recomputed value before the cli builds run (see the `enclave` job in
  # ci.yml), so the merged partials already contain reliable cli digests
  # even when the committed file is stale; this check is what surfaces
  # the drift to the developer so they know to update the committed file.
  if [[ -f "$SERVER_ENCLAVE_FILE" ]]; then
    committed_envelope="$(jq -cS . "$SERVER_ENCLAVE_FILE")"
    if [[ "$(printf '%s\n' "$expected_envelope" | jq -cS .)" != "$committed_envelope" ]]; then
      echo "::error::$SERVER_ENCLAVE_FILE does not match the enclave block recomputed from $CONFIG_FILE."
      echo "Committed:"
      echo "$committed_envelope" | jq .
      echo "Recomputed:"
      printf '%s\n' "$expected_envelope" | jq .
      rc=1
    fi
  else
    echo "::error::missing $SERVER_ENCLAVE_FILE — run \`just update-manifest\` to regenerate it"
    rc=1
  fi

  actual_manifest="$(merge_partials)"
  if [[ -n "$OUTPUT_FILE" ]]; then
    write_output "$actual_manifest"
  fi
  actual_norm="$(printf '%s\n' "$actual_manifest" | jq -cS .)"
  committed_norm="$(jq -cS . "$MANIFEST_FILE")"

  if [[ "$actual_norm" != "$committed_norm" ]]; then
    echo "::error::Artifact manifest does not match build output."
    echo "Committed:"
    echo "$committed_norm" | jq .
    echo "Actual:"
    echo "$actual_norm" | jq .
    rc=1
  fi

  if [[ "$rc" -eq 0 ]]; then
    echo "Artifact manifest and server-enclave.json match build output."
  fi
  return "$rc"
}

verify_oci_manifest() {
  local actual_norm committed_subset
  local tmp_partial

  tmp_partial="$(write_temp_file "$(print_oci_manifest)")"
  PARTIAL_FILES=("$tmp_partial")

  actual_norm="$(merge_partials | jq -cS .)"
  committed_subset="$(jq -cS '
    {
      schema_version: 1,
      artifacts: {
        "eidola-server": .artifacts["eidola-server"],
        "eidola-cli": .artifacts["eidola-cli"],
        "eidola-postgres": .artifacts["eidola-postgres"]
      }
    }
  ' "$MANIFEST_FILE")"

  rm -f "$tmp_partial"

  if [[ "$actual_norm" = "$committed_subset" ]]; then
    echo "Artifact manifest matches OCI build output."
    return 0
  fi
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::error::Artifact manifest does not match OCI build output."
  else
    echo "Artifact manifest does not match OCI build output."
  fi
  echo "Committed OCI subset:"
  echo "$committed_subset" | jq .
  echo "Actual OCI subset:"
  echo "$actual_norm" | jq .
  return 1
}

# Two-phase build:
#
#   Phase 1: build {server, postgres}. Neither image consumes the enclave
#            measurement, so they can be built first.
#   Phase 2: stamp the new server digest into `tinfoil-config.yml`, recompute
#            the enclave block, and write `releases/trust/server-enclave.json`.
#   Phase 3: build the cli OCI image and the macOS universal CLI. Both
#            consume the freshly-written `server-enclave.json` via
#            `eidola-app-core/build.rs`, so they have to come after phase 2.
#   Phase 4: compose `artifact-manifest.json` from all of the above.
#
# This breaks the previous self-reference (the cli build's OCI digest is
# recorded in the very file the cli build was COPYing into its build context),
# so `just update-manifest` converges in a single run.
update_manifest() {
  local server_oci_partial cli_oci_partial macos_partial actual_manifest
  local server_oci_partial_file cli_oci_partial_file macos_partial_file
  local server_metadata cli_metadata original_metadata original_targets

  if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="$REPO_ROOT/artifact-manifest.json"
  fi

  original_metadata="$METADATA_FILE"
  original_targets="$TARGETS"
  server_metadata="${TMPDIR:-/tmp}/bake-metadata-server.json"
  cli_metadata="${TMPDIR:-/tmp}/bake-metadata-cli.json"

  # ── Phase 1: build server + postgres ────────────────────────────────────
  METADATA_FILE="$server_metadata"
  TARGETS="server"
  build_metadata

  # ── Phase 2: stamp config, compute enclave, write server-enclave.json ───
  stamp_config_digests
  ENCLAVE_MEASUREMENTS="$(compute_measurements)"
  write_server_enclave_envelope "$ENCLAVE_MEASUREMENTS" "$SERVER_ENCLAVE_FILE"
  echo "Updated $SERVER_ENCLAVE_FILE"

  # Nix flakes only see git-tracked paths under dirty working trees, so a
  # brand-new `server-enclave.json` would be invisible to the macOS build
  # below. Mark it intent-to-add (no content staged) so flakes pick it up
  # via the working tree without staging anything for the developer.
  if [[ -d "$REPO_ROOT/.git" ]] && ! git -C "$REPO_ROOT" ls-files --error-unmatch releases/trust/server-enclave.json >/dev/null 2>&1; then
    git -C "$REPO_ROOT" add --intent-to-add releases/trust/server-enclave.json
  fi

  # ── Phase 3: build cli OCI + macOS universal ────────────────────────────
  METADATA_FILE="$cli_metadata"
  TARGETS="cli"
  build_metadata
  build_macos_artifacts

  # ── Phase 4: compose final artifact-manifest.json ───────────────────────
  METADATA_FILE="$server_metadata"
  server_oci_partial="$(print_oci_partial_for_targets server postgres)"
  METADATA_FILE="$cli_metadata"
  cli_oci_partial="$(print_oci_partial_for_targets cli)"
  macos_partial="$(print_macos_manifest)"

  server_oci_partial_file="$(write_temp_file "$server_oci_partial")"
  cli_oci_partial_file="$(write_temp_file "$cli_oci_partial")"
  macos_partial_file="$(write_temp_file "$macos_partial")"
  PARTIAL_FILES=("$server_oci_partial_file" "$cli_oci_partial_file" "$macos_partial_file")

  actual_manifest="$(merge_partials)"
  rm -f "$server_oci_partial_file" "$cli_oci_partial_file" "$macos_partial_file"

  write_output "$actual_manifest"
  if [[ -n "$OUTPUT_FILE" ]]; then
    echo "Updated $OUTPUT_FILE"
  fi

  # Restore globals so subsequent commands behave predictably.
  METADATA_FILE="$original_metadata"
  TARGETS="$original_targets"
}

case "$COMMAND" in
  build)
    build_metadata
    ;;
  print)
    print_oci_manifest
    ;;
  verify)
    verify_oci_manifest
    ;;
  build-macos)
    build_macos_artifacts
    write_output "$(print_macos_manifest)"
    ;;
  measure)
    enclave_json="$(compute_measurements)"
    if [[ -n "$SERVER_ENCLAVE_OUTPUT" ]]; then
      write_server_enclave_envelope "$enclave_json" "$SERVER_ENCLAVE_OUTPUT"
    else
      printf '%s\n' "$enclave_json"
    fi
    ;;
  stamp-config)
    # Stamp the freshly-built server digest from --metadata-file into
    # tinfoil-config.yml. Used by CI's two-phase oci job after the server
    # bake completes and before the enclave is recomputed.
    stamp_config_digests
    echo "Stamped $CONFIG_FILE with server digest from $METADATA_FILE"
    ;;
  verify-full)
    verify_full_manifest
    ;;
  update)
    update_manifest
    ;;
  *)
    echo "error: unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
