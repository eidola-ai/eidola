#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_SPEC="${1:-}"

if [[ -z "$SOURCE_SPEC" ]]; then
  echo "No input provided. Generating locally..."
  
  # Ensure target directory exists
  mkdir -p "$REPO_ROOT/target"
  
  TARGET_FILE="$REPO_ROOT/target/openapi_gen.json"
  
  # Run the generator from the repo root
  (cd "$REPO_ROOT" && cargo run -q -p generate-openapi > "$TARGET_FILE")
  
  SOURCE_SPEC="$TARGET_FILE"
fi

DEST="$REPO_ROOT/crates/eidolons-server/openapi.json"

echo "Copying OpenAPI spec..."
echo "  Source: $SOURCE_SPEC"
echo "  Dest:   $DEST"

cp "$SOURCE_SPEC" "$DEST"
chmod +w "$DEST"

echo "Done. Review changes and commit:"
echo "  git status"
