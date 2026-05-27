#!/usr/bin/env bash
set -euo pipefail

# Assembles a .app bundle from the cargo-built `eidola-gui` binary.
#
# Usage: package-gui-app.sh [debug|release]
#
# The bundle goes to crates/eidola-gui/build/Eidola.app. Without this — i.e. running
# the bare `cargo run -p eidola-gui` binary — AppKit treats the process as a
# command-line tool, which breaks menu key-equivalent dispatch when no
# window has key focus (⌘N / ⌘Q etc. after ⌘Tab back, or after closing the
# last window). See crates/eidola-gui/Support/Info.plist for the full rationale.

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: .app packaging is only supported on macOS" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG="${1:-debug}"

case "$CONFIG" in
  debug)
    BIN_DIR="$REPO_ROOT/target/debug"
    ;;
  release)
    BIN_DIR="$REPO_ROOT/target/release"
    ;;
  *)
    echo "error: unknown config '$CONFIG' (expected: debug, release)" >&2
    exit 1
    ;;
esac

EXECUTABLE="$BIN_DIR/eidola-gui"
if [[ ! -f "$EXECUTABLE" ]]; then
  echo "error: executable not found at $EXECUTABLE" >&2
  echo "  Run 'cargo build -p eidola-gui' (or with --release) first." >&2
  exit 1
fi

APP_DIR="$REPO_ROOT/crates/eidola-gui/build/Eidola.app"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

echo "Assembling Eidola.app from $CONFIG build..."

# Rename to match CFBundleExecutable in Info.plist. macOS uses this to decide
# whether the binary "owns" the bundle; mismatch falls back to tool-mode.
cp "$EXECUTABLE" "$APP_DIR/Contents/MacOS/Eidola"

cp "$REPO_ROOT/crates/eidola-gui/Support/Info.plist" "$APP_DIR/Contents/"

# App icon (if present).
ICON="$REPO_ROOT/crates/eidola-gui/Support/AppIcon.icns"
if [[ -f "$ICON" ]]; then
  cp "$ICON" "$APP_DIR/Contents/Resources/"
fi

# Ad-hoc codesign for local dev. Required on Apple Silicon for the binary
# to launch at all; on Intel it's not strictly required but harmless.
codesign --force --sign - "$APP_DIR"

echo "Done: $APP_DIR"
