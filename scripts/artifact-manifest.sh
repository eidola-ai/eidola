#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/artifact-manifest.sh build [--metadata-file PATH] [--builder NAME] [--ensure-builder]
  scripts/artifact-manifest.sh print [--metadata-file PATH]
  scripts/artifact-manifest.sh verify [--metadata-file PATH] [--manifest PATH]
  scripts/artifact-manifest.sh update [--metadata-file PATH] [--output PATH] [--builder NAME] [--ensure-builder]
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
OUTPUT_FILE="$REPO_ROOT/artifact-manifest.json"
MANIFEST_FILE="$REPO_ROOT/artifact-manifest.json"
BUILDER_NAME="eidolons"
ENSURE_BUILDER=0

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
  local -a builder_args
  builder_args=()

  if [[ "$ENSURE_BUILDER" -eq 1 ]]; then
    ensure_builder
    builder_args=(--builder "$BUILDER_NAME")
  fi

  docker buildx bake manifest \
    "${builder_args[@]}" \
    --set '*.output=type=docker,rewrite-timestamp=true,force-compression=true,compression=gzip,oci-mediatypes=true' \
    --metadata-file "$METADATA_FILE"
}

render_manifest() {
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

write_manifest() {
  local tmp_file

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/artifact-manifest.XXXXXX")"
  render_manifest > "$tmp_file"
  mv "$tmp_file" "$OUTPUT_FILE"
  echo "Updated $OUTPUT_FILE"
}

verify_manifest() {
  local actual_norm committed_norm

  actual_norm="$(render_manifest | jq -cS .)"
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

case "$COMMAND" in
  build)
    build_metadata
    ;;
  print)
    render_manifest
    ;;
  verify)
    verify_manifest
    ;;
  update)
    build_metadata
    write_manifest
    ;;
  *)
    echo "error: unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
