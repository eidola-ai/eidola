#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.build"
CONFIG="${1:-release}"

APP_NAME="Eidolons"
APP="${BUILD_DIR}/${APP_NAME}.app"

# Build if needed
if [[ ! -f "${BUILD_DIR}/${CONFIG}/EidolonsEntrypoint" ]]; then
  echo "Building (${CONFIG})..."
  swift build -c "${CONFIG}" --package-path "${ROOT_DIR}"
fi

# Clean previous bundle
rm -rf "${APP}"

# Create bundle structure
mkdir -p "${APP}/Contents/MacOS"
mkdir -p "${APP}/Contents/Resources"

# Copy executable
cp "${BUILD_DIR}/${CONFIG}/EidolonsEntrypoint" "${APP}/Contents/MacOS/${APP_NAME}"

# Copy Info.plist
cp "${SCRIPT_DIR}/Info.plist" "${APP}/Contents/"

# Copy SwiftPM resource bundle contents
RESOURCE_BUNDLE="${BUILD_DIR}/${CONFIG}/EidolonsApp_EidolonsEntrypoint.bundle"

if [[ -d "${RESOURCE_BUNDLE}" ]]; then
  echo "Copying resource bundle contents..."

  # Copy all non-xcassets files from the bundle (Info.plist, localized resources, etc.)
  find "${RESOURCE_BUNDLE}" -maxdepth 1 -type f -exec cp {} "${APP}/Contents/Resources/" \;

  # Handle assets - check for both compiled (Assets.car) and raw (Assets.xcassets)
  if [[ -f "${RESOURCE_BUNDLE}/Assets.car" ]]; then
    # SwiftPM compiled the assets - copy directly
    echo "Copying compiled assets (Assets.car)..."
    cp "${RESOURCE_BUNDLE}/Assets.car" "${APP}/Contents/Resources/"
    # Also copy AppIcon.icns if it exists
    [[ -f "${RESOURCE_BUNDLE}/AppIcon.icns" ]] && cp "${RESOURCE_BUNDLE}/AppIcon.icns" "${APP}/Contents/Resources/"
  elif [[ -d "${RESOURCE_BUNDLE}/Assets.xcassets" ]]; then
    # SwiftPM left raw assets - compile them with actool
    if command -v actool &>/dev/null || [[ -x /usr/bin/actool ]]; then
      echo "Compiling assets with actool..."
      actool "${RESOURCE_BUNDLE}/Assets.xcassets" \
        --compile "${APP}/Contents/Resources" \
        --platform macosx \
        --minimum-deployment-target 26.0 \
        --app-icon AppIcon \
        --accent-color AccentColor \
        --output-partial-info-plist /dev/null \
        >/dev/null 2>&1 || echo "Warning: actool failed, assets may be incomplete"
    else
      echo "Warning: actool not found and no compiled assets, app may lack icons"
    fi
  fi

  # Copy any other resource bundles or directories (excluding Assets.xcassets which we handled)
  find "${RESOURCE_BUNDLE}" -maxdepth 1 -type d ! -name "Assets.xcassets" ! -path "${RESOURCE_BUNDLE}" -exec cp -r {} "${APP}/Contents/Resources/" \;
else
  echo "Warning: Resource bundle not found at ${RESOURCE_BUNDLE}"
fi

echo "Created ${APP}"
