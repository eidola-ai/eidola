#!/usr/bin/env bash
set -euo pipefail

# Development version of update-xcframework.sh
# - Only compiles for native architecture (no universal binary)
# - Uses debug build (no --release flag)
# - Produces a dynamic framework (not static) so Xcode previews work
#   (the JIT linker can't process a 375MB+ staticlib in 30 seconds)

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

echo "Building for $TARGET (debug, dynamic)..."
(cd "$REPO_ROOT" && cargo build -p eidola-app-core --target "$TARGET")

TARGET_DIR="$REPO_ROOT/target"
DYLIB_PATH="$TARGET_DIR/$TARGET/debug/libeidola_app_core.dylib"

if [[ ! -f "$DYLIB_PATH" ]]; then
  echo "Error: Dynamic library not found after build."
  echo "Expected: $DYLIB_PATH"
  exit 1
fi

# Prepare temp XCFramework structure with a dynamic .framework
TEMP_ROOT="$TARGET_DIR/generated-xcframework"
rm -rf "$TEMP_ROOT"
mkdir -p "$TEMP_ROOT"

XCFW_NAME="libeidola_app_core-rs.xcframework"
XCFW_PATH="$TEMP_ROOT/$XCFW_NAME"
FW_DIR="$XCFW_PATH/macos-arm64_x86_64/EidolaAppCoreRS.framework"

mkdir -p "$FW_DIR"

echo "Creating dynamic framework (versioned bundle)..."

# macOS requires the versioned framework structure
VERSIONS_DIR="$FW_DIR/Versions/A"
mkdir -p "$VERSIONS_DIR/Resources"

cp "$DYLIB_PATH" "$VERSIONS_DIR/EidolaAppCoreRS"

# Fix the install_name so the linker embeds the correct rpath reference
install_name_tool -id "@rpath/EidolaAppCoreRS.framework/Versions/A/EidolaAppCoreRS" "$VERSIONS_DIR/EidolaAppCoreRS"

# Ad-hoc codesign (required for dylibs on Apple Silicon)
codesign --force --sign - "$VERSIONS_DIR/EidolaAppCoreRS"

# Symlinks for versioned bundle
ln -sfn A "$FW_DIR/Versions/Current"
ln -sfn Versions/Current/EidolaAppCoreRS "$FW_DIR/EidolaAppCoreRS"
ln -sfn Versions/Current/Resources "$FW_DIR/Resources"

# Framework Info.plist
cat > "$VERSIONS_DIR/Resources/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>EidolaAppCoreRS</string>
  <key>CFBundleIdentifier</key>
  <string>ai.eidola.EidolaAppCoreRS</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
</dict>
</plist>
EOF

# XCFramework Info.plist (declares a framework, not a static library)
cat > "$XCFW_PATH/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>AvailableLibraries</key>
  <array>
    <dict>
      <key>BinaryPath</key>
      <string>EidolaAppCoreRS</string>
      <key>LibraryIdentifier</key>
      <string>macos-arm64_x86_64</string>
      <key>LibraryPath</key>
      <string>EidolaAppCoreRS.framework</string>
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

echo "Installing XCFramework..."
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$TEMP_ROOT/$XCFW_NAME" "$DEST"
chmod -R +w "$DEST"

echo "Done (dev dynamic framework for $ARCH, $(du -sh "$FW_DIR/Versions/A/EidolaAppCoreRS" | cut -f1) dylib)."
