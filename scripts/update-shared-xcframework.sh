#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_XCFRAMEWORK="${1:-}"

if [[ -z "$SOURCE_XCFRAMEWORK" ]]; then
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: Local XCFramework generation is only supported on macOS."
    exit 1
  fi

  echo "No input provided. Generating locally..."

  # Ensure target directory exists
  TARGET_DIR="$REPO_ROOT/target"
  mkdir -p "$TARGET_DIR"

  echo "Building for aarch64-apple-darwin..."
  (cd "$REPO_ROOT" && cargo build -p eidola-shared --release --target aarch64-apple-darwin)

  echo "Building for x86_64-apple-darwin..."
  (cd "$REPO_ROOT" && cargo build -p eidola-shared --release --target x86_64-apple-darwin)

  # Prepare temp XCFramework structure
  TEMP_ROOT="$TARGET_DIR/generated-xcframework"
  rm -rf "$TEMP_ROOT"
  mkdir -p "$TEMP_ROOT"
  
  XCFW_NAME="libeidola_shared-rs.xcframework"
  XCFW_PATH="$TEMP_ROOT/$XCFW_NAME"
  MACOS_DIR="$XCFW_PATH/macos-arm64_x86_64"
  
  mkdir -p "$MACOS_DIR"

  LIB_ARM64="$TARGET_DIR/aarch64-apple-darwin/release/libeidola_shared.a"
  LIB_X86_64="$TARGET_DIR/x86_64-apple-darwin/release/libeidola_shared.a"

  if [[ ! -f "$LIB_ARM64" || ! -f "$LIB_X86_64" ]]; then
    echo "Error: Static libraries not found after build."
    echo "Expected: $LIB_ARM64"
    echo "Expected: $LIB_X86_64"
    exit 1
  fi

  echo "Creating universal library with lipo..."
  lipo -create \
    "$LIB_ARM64" \
    "$LIB_X86_64" \
    -output "$MACOS_DIR/libeidola_shared.a"

  echo "Creating Info.plist..."
  cat > "$XCFW_PATH/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>AvailableLibraries</key>
  <array>
    <dict>
      <key>LibraryIdentifier</key>
      <string>macos-arm64_x86_64</string>
      <key>LibraryPath</key>
      <string>libeidola_shared.a</string>
      <key>SupportedArchitectures</key>
      <array>
        <string>arm64</string>
        <string>x86_64</string>
      </array>
      <key>SupportedPlatform</key>
      <string>macos</string>
    </dict>
  </array>
  <key>CFBundlePackageType</key>
  <string>XFWK</string>
  <key>XCFrameworkFormatVersion</key>
  <string>1.0</string>
</dict>
</plist>
EOF

  SOURCE_XCFRAMEWORK="$TEMP_ROOT"
fi

DEST="$REPO_ROOT/crates/eidola-shared/target/apple/libeidola_shared-rs.xcframework"

echo "Copying shared core XCframework..."
echo "  Source: $SOURCE_XCFRAMEWORK"
echo "  Dest:   $DEST"

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
# Copy the xcframework folder itself
cp -R "$SOURCE_XCFRAMEWORK/libeidola_shared-rs.xcframework" "$DEST"
chmod -R +w "$DEST"

echo "Done."
