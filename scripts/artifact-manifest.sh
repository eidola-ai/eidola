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
  scripts/artifact-manifest.sh merge [--partial PATH ...] [--output PATH]
  scripts/artifact-manifest.sh verify-full [--partial PATH ...] [--manifest PATH]
  scripts/artifact-manifest.sh update [--output PATH] [--metadata-file PATH] [--builder NAME] [--ensure-builder]
EOF
}

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "error: not in a git repository" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILDKIT_IMAGE="moby/buildkit:v0.28.0@sha256:60bfb07e39a6e524e78e6c4723114902c6b61ee36714493e357e39861bea753b"

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
BUILDER_NAME="eidolons"
ENSURE_BUILDER=0
MACOS_PATHS_FILE=""
APP_PATH=""
CLI_PATH=""
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
    --set)
      BUILDX_SET_ARGS+=("$2")
      shift 2
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

  for buildx_set in ${BUILDX_SET_ARGS[@]+"${BUILDX_SET_ARGS[@]}"}; do
    buildx_set_args+=(--set "$buildx_set")
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
        "eidolons-server": { type: "oci", platform: "linux/amd64", digest: $server },
        "eidolons-cli": { type: "oci", platform: "linux/amd64", digest: $cli },
        "eidolons-postgres": { type: "oci", platform: "linux/amd64", digest: $postgres }
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
      '.#eidolons-macos-app' \
      '.#eidolons-cli-macos-universal' \
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
        "eidolons-macos-app": { type: "nix", platform: "darwin/universal", narHash: $app_hash },
        "eidolons-cli-macos-universal": { type: "nix", platform: "darwin/universal", narHash: $cli_hash }
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

  jq -s '
    {
      version: 1,
      artifacts: (reduce .[] as $partial ({}; . + ($partial.artifacts // {})))
    }
  ' "${PARTIAL_FILES[@]}"
}

verify_full_manifest() {
  local actual_norm committed_norm actual_manifest

  actual_manifest="$(merge_partials)"
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
        "eidolons-server": .artifacts["eidolons-server"],
        "eidolons-cli": .artifacts["eidolons-cli"],
        "eidolons-postgres": .artifacts["eidolons-postgres"]
      }
    }
  ' "$MANIFEST_FILE")"

  rm -f "$tmp_partial"

  if [[ "$actual_norm" = "$committed_subset" ]]; then
    echo "Artifact manifest matches OCI build output."
    return 0
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
