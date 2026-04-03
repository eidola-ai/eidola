#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_BINDINGS="${1:-}"

if [[ -z "$SOURCE_BINDINGS" ]]; then
  echo "No inputs provided. Generating locally..."

  # Build the shared library first to get the dylib
  echo "Building eidola-app-core..."
  (cd "$REPO_ROOT" && cargo build -p eidola-app-core)

  # Locate the dylib (heuristic for macOS/Linux)
  DYLIB="$REPO_ROOT/target/debug/libeidola_app_core.dylib"
  if [[ ! -f "$DYLIB" ]]; then
    DYLIB="$REPO_ROOT/target/debug/libeidola_app_core.so"
  fi

  if [[ ! -f "$DYLIB" ]]; then
     echo "Error: Could not find built dylib at $REPO_ROOT/target/debug/libeidola_app_core.{dylib,so}"
     exit 1
  fi

  # Create temp output directory
  TEMP_OUT="$REPO_ROOT/target/generated-swift"
  mkdir -p "$TEMP_OUT"

  echo "Generating UniFFI bindings..."
  (cd "$REPO_ROOT" && cargo run -p uniffi-bindgen-swift -- \
    --swift-sources --headers --modulemap \
    --metadata-no-deps \
    "$DYLIB" \
    "$TEMP_OUT" \
    --module-name eidola_app_coreFFI \
    --modulemap-filename module.modulemap)

  # Create the stub C file that Flake creates
  cat > "$TEMP_OUT/eidola_app_coreFFI.c" << 'STUB'
// This file exists so Swift Package Manager has something to compile for the eidola_app_coreFFI module.
// The actual implementation is in the XCFramework (libeidola_app_core.a).
// This module just exposes the C header interface to Swift.
#include "eidola_app_coreFFI.h"
STUB

  SOURCE_BINDINGS="$TEMP_OUT"
fi

# Update UniFFI bindings
DEST="$REPO_ROOT/crates/eidola-app-core/swift/Sources"
echo "Syncing Swift bindings..."
echo "  Source: $SOURCE_BINDINGS"
echo "  Dest:   $DEST"

mkdir -p "$DEST/EidolaAppCore"
mkdir -p "$DEST/EidolaAppCoreFFI"

# Clean old files
rm -f "$DEST/EidolaAppCore/"*.swift
rm -f "$DEST/EidolaAppCoreFFI/"*.{h,c,modulemap}

# Copy new files
# Handle both flat outputs (local temp dir) and Nix outputs (Sources/EidolaAppCore, Sources/EidolaAppCoreFFI)
if [[ -d "$SOURCE_BINDINGS/EidolaAppCore" && -d "$SOURCE_BINDINGS/EidolaAppCoreFFI" ]]; then
  BINDINGS_SWIFT_DIR="$SOURCE_BINDINGS/EidolaAppCore"
  BINDINGS_FFI_DIR="$SOURCE_BINDINGS/EidolaAppCoreFFI"
elif [[ -d "$SOURCE_BINDINGS/Sources/EidolaAppCore" && -d "$SOURCE_BINDINGS/Sources/EidolaAppCoreFFI" ]]; then
  BINDINGS_SWIFT_DIR="$SOURCE_BINDINGS/Sources/EidolaAppCore"
  BINDINGS_FFI_DIR="$SOURCE_BINDINGS/Sources/EidolaAppCoreFFI"
else
  BINDINGS_SWIFT_DIR="$SOURCE_BINDINGS"
  BINDINGS_FFI_DIR="$SOURCE_BINDINGS"
fi

cp "$BINDINGS_SWIFT_DIR/"*.swift "$DEST/EidolaAppCore/"
cp "$BINDINGS_FFI_DIR/"*.h "$DEST/EidolaAppCoreFFI/"
cp "$BINDINGS_FFI_DIR/module.modulemap" "$DEST/EidolaAppCoreFFI/"
cp "$BINDINGS_FFI_DIR/eidola_app_coreFFI.c" "$DEST/EidolaAppCoreFFI/"

# Format generated Swift so it passes lint checks
echo "Formatting generated Swift..."
swift format --in-place "$DEST/EidolaAppCore/"*.swift

echo "Done. Review changes and commit:"
echo "  git status"
