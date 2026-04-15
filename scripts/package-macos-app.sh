#!/usr/bin/env bash
set -euo pipefail

# Assemble a .app bundle from swift build output.
# Usage: package-macos-app.sh [debug|release]

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: .app packaging is only supported on macOS" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG="${1:-debug}"

ARCH="$(uname -m)"
BIN_DIR="$REPO_ROOT/apps/macos/.build/${ARCH}-apple-macosx/${CONFIG}"
EXECUTABLE="$BIN_DIR/Eidola"

if [[ ! -f "$EXECUTABLE" ]]; then
  echo "error: executable not found at $EXECUTABLE" >&2
  echo "  Run 'swift build' in apps/macos first." >&2
  exit 1
fi

APP_DIR="$REPO_ROOT/apps/macos/build/Eidola.app"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

echo "Assembling Eidola.app..."

# Executable (renamed per CFBundleExecutable in Info.plist)
cp "$EXECUTABLE" "$APP_DIR/Contents/MacOS/Eidola"

# Info.plist
cp "$REPO_ROOT/apps/macos/Support/Info.plist" "$APP_DIR/Contents/"

# App icon (if present)
ICON="$REPO_ROOT/apps/macos/Support/AppIcon.icns"
if [[ -f "$ICON" ]]; then
  cp "$ICON" "$APP_DIR/Contents/Resources/"
fi

# Ad-hoc codesign for local dev
codesign --force --sign - "$APP_DIR"

echo "Done: $APP_DIR"
