#!/usr/bin/env bash
set -euo pipefail

# Assembles "Eidola Notepad.app" from the cargo-built `eidola-notepad` binary.
#
# Usage: package-notepad-app.sh [debug|release]
#
# The bundle goes to crates/eidola-notepad/build/Eidola Notepad.app. A bundle
# is required, not cosmetic: AppKit runs a bare binary as a command-line tool
# (unreliable menu key equivalents), and the Info.plist's document types are
# what let Finder hand the app a file dropped on its Dock icon.

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: .app packaging is only supported on macOS" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG="${1:-debug}"

case "$CONFIG" in
  debug | release) BIN_DIR="$REPO_ROOT/target/$CONFIG" ;;
  *)
    echo "error: unknown config '$CONFIG' (expected: debug, release)" >&2
    exit 1
    ;;
esac

EXECUTABLE="$BIN_DIR/eidola-notepad"
if [[ ! -f "$EXECUTABLE" ]]; then
  echo "error: executable not found at $EXECUTABLE" >&2
  echo "  Run 'cargo build -p eidola-notepad' (or with --release) first." >&2
  exit 1
fi

APP_DIR="$REPO_ROOT/crates/eidola-notepad/build/Eidola Notepad.app"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

echo "Assembling Eidola Notepad.app from $CONFIG build..."

# Renamed to match CFBundleExecutable; a mismatch falls back to tool mode.
cp "$EXECUTABLE" "$APP_DIR/Contents/MacOS/EidolaNotepad"
cp "$REPO_ROOT/crates/eidola-notepad/Support/Info.plist" "$APP_DIR/Contents/"

# The Eidola mark, copied from the generated app icon (see brand/AGENTS.md);
# the notepad has no identity asset of its own.
for asset in AppIcon.icns Assets.car; do
  SRC="$REPO_ROOT/crates/eidola-gui/Support/$asset"
  if [[ ! -f "$SRC" ]]; then
    echo "error: $asset not found at $SRC (run 'just update-brand')" >&2
    exit 1
  fi
  cp "$SRC" "$APP_DIR/Contents/Resources/"
done

# Ad-hoc codesign: required on Apple Silicon for the binary to launch at all.
codesign --force --sign - "$APP_DIR"

# Tell LaunchServices about the bundle now, so its document types (Open
# With, Dock-icon drops) are known before the first launch.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
if [[ -x "$LSREGISTER" ]]; then
  "$LSREGISTER" -f "$APP_DIR" || true
fi

echo "Done: $APP_DIR"
