#!/usr/bin/env bash
set -euo pipefail

# Development version of update-xcframework.sh
# - Only compiles for native architecture (no universal binary)
# - Uses debug build (no --release flag)

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: not in a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: XCFramework generation is only supported on macOS."
  exit 1
fi

# Detect native architecture
ARCH="$(uname -m)"
case "$ARCH" in
  arm64)
    TARGET="aarch64-apple-darwin"
    ;;
  x86_64)
    TARGET="x86_64-apple-darwin"
    ;;
  *)
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

echo "Building for $TARGET (debug)..."
(cd "$REPO_ROOT" && cargo build -p eidola-app-core --target "$TARGET")

# Prepare temp XCFramework structure
TARGET_DIR="$REPO_ROOT/target"
TEMP_ROOT="$TARGET_DIR/generated-xcframework"
rm -rf "$TEMP_ROOT"
mkdir -p "$TEMP_ROOT"

XCFW_NAME="libeidola_app_core-rs.xcframework"
XCFW_PATH="$TEMP_ROOT/$XCFW_NAME"
MACOS_DIR="$XCFW_PATH/macos-arm64_x86_64"

mkdir -p "$MACOS_DIR"

LIB_PATH="$TARGET_DIR/$TARGET/debug/libeidola_app_core.a"

if [[ ! -f "$LIB_PATH" ]]; then
  echo "Error: Static library not found after build."
  echo "Expected: $LIB_PATH"
  exit 1
fi

echo "Copying library..."
cp "$LIB_PATH" "$MACOS_DIR/libeidola_app_core.a"

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
      <string>libeidola_app_core.a</string>
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

DEST="$REPO_ROOT/crates/eidola-app-core/target/apple/libeidola_app_core-rs.xcframework"

echo "Copying shared core XCframework..."
echo "  Source: $TEMP_ROOT"
echo "  Dest:   $DEST"

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$TEMP_ROOT/$XCFW_NAME" "$DEST"
chmod -R +w "$DEST"

echo "Done (dev build for $ARCH only)."
