#!/usr/bin/env bash
set -euo pipefail

SOURCE_XCFRAMEWORK="${1:-}"

if [[ -z "$SOURCE_XCFRAMEWORK" ]]; then
  echo "Error: Source XCFramework path not provided."
  echo "Usage: $0 <path-to-xcframework-root>"
  exit 1
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
DEST="$REPO_ROOT/apps/eidolons-shared/target/apple/libeidolons_shared-rs.xcframework"

echo "Copying shared core XCframework..."
echo "  Source: $SOURCE_XCFRAMEWORK"
echo "  Dest:   $DEST"

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
# Copy the xcframework folder itself
cp -R "$SOURCE_XCFRAMEWORK/libeidolons_shared-rs.xcframework" "$DEST"
chmod -R +w "$DEST"

echo "Done."
