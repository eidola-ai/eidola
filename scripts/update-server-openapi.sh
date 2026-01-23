#!/usr/bin/env bash
set -euo pipefail

SOURCE_SPEC="${1:-}"

if [[ -z "$SOURCE_SPEC" ]]; then
  echo "Error: Source spec path not provided."
  echo "Usage: $0 <path-to-openapi.json>"
  exit 1
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
DEST="$REPO_ROOT/crates/eidolons-server/openapi.json"

echo "Copying OpenAPI spec..."
echo "  Source: $SOURCE_SPEC"
echo "  Dest:   $DEST"

cp "$SOURCE_SPEC" "$DEST"
chmod +w "$DEST"

echo "Done. Review changes and commit:"
echo "  git status"
