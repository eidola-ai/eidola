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

# App icon. Committed, generated from the vector master by `just update-brand`
# (see brand/AGENTS.md); the name matches CFBundleIconFile in Info.plist.
ICON="$REPO_ROOT/crates/eidola-gui/Support/AppIcon.icns"
if [[ ! -f "$ICON" ]]; then
  echo "error: app icon not found at $ICON" >&2
  echo "  Run 'just update-brand' to regenerate it." >&2
  exit 1
fi
cp "$ICON" "$APP_DIR/Contents/Resources/"

# Bundle the on-device inference engine sidecar, if one is available. This
# is best-effort for dev: the `local` backend needs a `llama-server` next to
# the main binary at Contents/MacOS/llama-server (app-core's one rule is a
# sibling of the running executable) — the same layout the Nix release
# bundle produces, so dev and release stay one shape.
# Look for an explicit override first, then the `just engine` nix result
# symlink. A dev without either just sees the honest missing-engine state in
# the Local tab (or points `llama_server_path` at their own build).
SIDECAR=""
if [[ -n "${EIDOLA_LLAMA_SERVER:-}" && -x "${EIDOLA_LLAMA_SERVER}" ]]; then
  SIDECAR="$EIDOLA_LLAMA_SERVER"
elif [[ -x "$REPO_ROOT/crates/eidola-gui/build/llama-server/bin/llama-server" ]]; then
  SIDECAR="$REPO_ROOT/crates/eidola-gui/build/llama-server/bin/llama-server"
fi

if [[ -n "$SIDECAR" ]]; then
  cp "$SIDECAR" "$APP_DIR/Contents/MacOS/llama-server"
  chmod u+w "$APP_DIR/Contents/MacOS/llama-server"
  echo "Bundled inference engine: $SIDECAR"
else
  echo "No inference engine bundled (run 'just engine' to add one)."
fi

# Ad-hoc codesign for local dev. Required on Apple Silicon for the binary
# to launch at all; on Intel it's not strictly required but harmless. Done
# after the sidecar copy so the bundle seal covers it; the nix-built sidecar
# already carries its own ad-hoc signature for subprocess exec regardless.
codesign --force --sign - "$APP_DIR"

echo "Done: $APP_DIR"
