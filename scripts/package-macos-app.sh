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

# Embed the Rust dylib framework in dev builds (release uses static linking)
FRAMEWORK_SRC="$BIN_DIR/EidolaAppCoreRS.framework"
if [[ -d "$FRAMEWORK_SRC" ]]; then
  mkdir -p "$APP_DIR/Contents/Frameworks"
  cp -R "$FRAMEWORK_SRC" "$APP_DIR/Contents/Frameworks/"
  install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP_DIR/Contents/MacOS/Eidola" 2>/dev/null || true
fi

# Info.plist
cp "$REPO_ROOT/apps/macos/Support/Info.plist" "$APP_DIR/Contents/"

# App icon (if present)
ICON="$REPO_ROOT/apps/macos/Support/AppIcon.icns"
if [[ -f "$ICON" ]]; then
  cp "$ICON" "$APP_DIR/Contents/Resources/"
fi

# Ad-hoc codesign for local dev (sign embedded frameworks first)
if [[ -d "$APP_DIR/Contents/Frameworks" ]]; then
  find "$APP_DIR/Contents/Frameworks" -name "*.framework" -exec codesign --force --sign - {} \;
fi
codesign --force --sign - "$APP_DIR"

echo "Done: $APP_DIR"
