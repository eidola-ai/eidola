#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/artifact-manifest.sh build [--metadata-file PATH] [--builder NAME] [--ensure-builder] [--set PATTERN=VALUE ...]
  scripts/artifact-manifest.sh print [--metadata-file PATH]
  scripts/artifact-manifest.sh verify [--metadata-file PATH] [--manifest PATH]
  scripts/artifact-manifest.sh build-macos [--output PATH] [--macos-paths-file PATH]
  scripts/artifact-manifest.sh print-macos [--macos-paths-file PATH | --app-path PATH --cli-path PATH]
  scripts/artifact-manifest.sh measure [--config PATH] [--verify-attestations]
  scripts/artifact-manifest.sh merge [--partial PATH ...] [--output PATH]
  scripts/artifact-manifest.sh verify-full [--partial PATH ...] [--manifest PATH] [--output PATH] [--verify-attestations]
  scripts/artifact-manifest.sh update [--output PATH] [--metadata-file PATH] [--builder NAME] [--ensure-builder]

Options:
  --verify-attestations   Verify CVM manifest provenance via Sigstore (requires gh CLI).
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
BUILDER_NAME="eidola"
ENSURE_BUILDER=0
MACOS_PATHS_FILE=""
APP_PATH=""
CLI_PATH=""
CONFIG_FILE="$REPO_ROOT/tinfoil-config.yml"
VERIFY_ATTESTATIONS=0
PARTIAL_FILES=()
BUILDX_SET_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metadata-file)
      METADATA_FILE="$2"
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
    --macos-paths-file)
      MACOS_PATHS_FILE="$2"
      shift 2
      ;;
    --app-path)
      APP_PATH="$2"
      shift 2
      ;;
    --cli-path)
      CLI_PATH="$2"
      shift 2
      ;;
    --partial)
      PARTIAL_FILES+=("$2")
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --set)
      BUILDX_SET_ARGS+=("$2")
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
  if command -v sha256sum >/dev/null 2>&1; then
    actual_kernel_hash="$(sha256sum "$kernel_file" | cut -d' ' -f1)"
    actual_initrd_hash="$(sha256sum "$initrd_file" | cut -d' ' -f1)"
  else
    actual_kernel_hash="$(shasum -a 256 "$kernel_file" | cut -d' ' -f1)"
    actual_initrd_hash="$(shasum -a 256 "$initrd_file" | cut -d' ' -f1)"
  fi

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

# Update image digests in tinfoil-config.yml from build metadata.
stamp_config_digests() {
  local server_digest postgres_digest

  server_digest="$(jq -er '."server"."containerimage.digest" | select(type == "string" and startswith("sha256:"))' "$METADATA_FILE")"
  postgres_digest="$(jq -er '."postgres"."containerimage.digest" | select(type == "string" and startswith("sha256:"))' "$METADATA_FILE")"

  # Strip the sha256: prefix for the image reference
  sed -i.bak \
    -e "s|ghcr.io/eidola-ai/eidola-server@sha256:[a-f0-9]*|ghcr.io/eidola-ai/eidola-server@${server_digest}|" \
    -e "s|ghcr.io/eidola-ai/eidola-postgres@sha256:[a-f0-9]*|ghcr.io/eidola-ai/eidola-postgres@${postgres_digest}|" \
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

build_metadata() {
  local -a builder_args buildx_set_args
  builder_args=()
  buildx_set_args=()

  if [[ "$ENSURE_BUILDER" -eq 1 ]]; then
    ensure_builder
    builder_args=(--builder "$BUILDER_NAME")
  fi

  for buildx_set_arg in ${BUILDX_SET_ARGS[@]+"${BUILDX_SET_ARGS[@]}"}; do
    buildx_set_args+=(--set "$buildx_set_arg")
  done

  docker buildx bake manifest \
    ${builder_args[@]+"${builder_args[@]}"} \
    ${buildx_set_args[@]+"${buildx_set_args[@]}"} \
    --set '*.output=type=docker,rewrite-timestamp=true,force-compression=true,compression=gzip,oci-mediatypes=true' \
    --metadata-file "$METADATA_FILE"
}

print_oci_manifest() {
  local server cli postgres

  server="$(jq -er '."server"."containerimage.digest" | select(type == "string" and startswith("sha256:"))' "$METADATA_FILE")"
  cli="$(jq -er '."cli"."containerimage.digest" | select(type == "string" and startswith("sha256:"))' "$METADATA_FILE")"
  postgres="$(jq -er '."postgres"."containerimage.digest" | select(type == "string" and startswith("sha256:"))' "$METADATA_FILE")"

  jq -n \
    --arg server "$server" \
    --arg cli "$cli" \
    --arg postgres "$postgres" \
    '{
      version: 1,
      artifacts: {
        "eidola-server": { type: "oci", platform: "linux/amd64", digest: $server },
        "eidola-cli": { type: "oci", platform: "linux/amd64", digest: $cli },
        "eidola-postgres": { type: "oci", platform: "linux/amd64", digest: $postgres }
      }
    }'
}

build_macos_artifacts() {
  local -a out_paths
  local path

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: macOS artifact builds require a Darwin host" >&2
    exit 1
  fi

  out_paths=()
  while IFS= read -r path; do
    out_paths+=("$path")
  done < <(
    nix build \
      '.#eidola-macos-app' \
      '.#eidola-cli-macos-universal' \
      --no-link \
      --print-out-paths \
      --show-trace
  )

  if [[ "${#out_paths[@]}" -ne 2 ]]; then
    echo "error: expected 2 macOS output paths, got ${#out_paths[@]}" >&2
    exit 1
  fi

  APP_PATH="${out_paths[0]}"
  CLI_PATH="${out_paths[1]}"

  if [[ -n "$MACOS_PATHS_FILE" ]]; then
    jq -n \
      --arg app_path "$APP_PATH" \
      --arg cli_path "$CLI_PATH" \
      '{app_path: $app_path, cli_path: $cli_path}' \
      > "$MACOS_PATHS_FILE"
  fi
}

load_macos_paths() {
  if [[ -n "$MACOS_PATHS_FILE" ]]; then
    APP_PATH="$(jq -er '.app_path | select(type == "string" and length > 0)' "$MACOS_PATHS_FILE")"
    CLI_PATH="$(jq -er '.cli_path | select(type == "string" and length > 0)' "$MACOS_PATHS_FILE")"
  fi

  if [[ -z "$APP_PATH" || -z "$CLI_PATH" ]]; then
    echo "error: provide --macos-paths-file or both --app-path and --cli-path" >&2
    exit 1
  fi
}

nix_nar_hash() {
  local store_path="$1"

  nix path-info --json "$store_path" \
    | jq -er --arg path "$store_path" '.[$path].narHash | select(type == "string" and startswith("sha256-"))'
}

print_macos_manifest() {
  local app_hash cli_hash

  load_macos_paths

  app_hash="$(nix_nar_hash "$APP_PATH")"
  cli_hash="$(nix_nar_hash "$CLI_PATH")"

  jq -n \
    --arg app_hash "$app_hash" \
    --arg cli_hash "$cli_hash" \
    '{
      version: 1,
      artifacts: {
        "eidola-macos-app": { type: "nix", platform: "darwin/universal", narHash: $app_hash },
        "eidola-cli-macos-universal": { type: "nix", platform: "darwin/universal", narHash: $cli_hash }
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
      version: 1,
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

  # Recompute enclave measurements from committed config if not already set
  if [[ -z "${ENCLAVE_MEASUREMENTS:-}" ]]; then
    ENCLAVE_MEASUREMENTS="$(compute_measurements)"
  fi

  actual_manifest="$(merge_partials)"
  if [[ -n "$OUTPUT_FILE" ]]; then
    write_output "$actual_manifest"
  fi
  actual_norm="$(printf '%s\n' "$actual_manifest" | jq -cS .)"
  committed_norm="$(jq -cS . "$MANIFEST_FILE")"

  if [[ "$actual_norm" = "$committed_norm" ]]; then
    echo "Artifact manifest matches build output."
    return 0
  fi

  echo "::error::Artifact manifest does not match build output."
  echo "Committed:"
  echo "$committed_norm" | jq .
  echo "Actual:"
  echo "$actual_norm" | jq .
  return 1
}

verify_oci_manifest() {
  local actual_norm committed_subset
  local tmp_partial

  tmp_partial="$(write_temp_file "$(print_oci_manifest)")"
  PARTIAL_FILES=("$tmp_partial")

  actual_norm="$(merge_partials | jq -cS .)"
  committed_subset="$(jq -cS '
    {
      version: 1,
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
  echo "::error::Artifact manifest does not match OCI build output."
  echo "Committed OCI subset:"
  echo "$committed_subset" | jq .
  echo "Actual OCI subset:"
  echo "$actual_norm" | jq .
  return 1
}

update_manifest() {
  local oci_partial macos_partial actual_manifest
  local oci_partial_file macos_partial_file

  if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="$REPO_ROOT/artifact-manifest.json"
  fi

  build_metadata
  build_macos_artifacts

  # Stamp new OCI digests into tinfoil-config.yml before measuring
  stamp_config_digests

  # Compute enclave measurements from the updated config
  ENCLAVE_MEASUREMENTS="$(compute_measurements)"

  oci_partial="$(print_oci_manifest)"
  macos_partial="$(print_macos_manifest)"

  oci_partial_file="$(write_temp_file "$oci_partial")"
  macos_partial_file="$(write_temp_file "$macos_partial")"
  PARTIAL_FILES=("$oci_partial_file" "$macos_partial_file")

  actual_manifest="$(merge_partials)"
  rm -f "$oci_partial_file" "$macos_partial_file"

  write_output "$actual_manifest"
  if [[ -n "$OUTPUT_FILE" ]]; then
    echo "Updated $OUTPUT_FILE"
  fi
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
  print-macos)
    print_macos_manifest
    ;;
  measure)
    compute_measurements
    ;;
  merge)
    write_output "$(merge_partials)"
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
