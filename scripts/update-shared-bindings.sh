#!/usr/bin/env bash
set -euo pipefail

SOURCE_BINDINGS="${1:-}"
SOURCE_TYPES="${2:-}"

if [[ -z "$SOURCE_BINDINGS" || -z "$SOURCE_TYPES" ]]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <path-to-bindings-sources> <path-to-types-root>"
  exit 1
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Update UniFFI bindings
DEST="$REPO_ROOT/apps/eidolons-shared/swift/Sources"
echo "Syncing Swift bindings..."
echo "  Source: $SOURCE_BINDINGS"
echo "  Dest:   $DEST"

mkdir -p "$DEST"
rm -rf "$DEST"
# Copy contents of Source/ to Source/
cp -R "$SOURCE_BINDINGS" "$DEST"
chmod -R +w "$DEST"

# Update Crux typegen types
TYPES_DEST="$REPO_ROOT/apps/eidolons-shared/swift/generated"
echo "Syncing Crux typegen Swift types..."
echo "  Source: $SOURCE_TYPES"
echo "  Dest:   $TYPES_DEST"

rm -rf "$TYPES_DEST"
mkdir -p "$TYPES_DEST"
# The source is expected to contain the SharedTypes folder
cp -R "$SOURCE_TYPES/SharedTypes" "$TYPES_DEST/SharedTypes"
chmod -R +w "$TYPES_DEST"

echo "Done. Review changes and commit:"
echo "  git status"
