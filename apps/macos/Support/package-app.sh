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

# Compile assets with actool if available
RESOURCE_BUNDLE="${BUILD_DIR}/${CONFIG}/EidolonsApp_EidolonsEntrypoint.bundle"
XCASSETS="${RESOURCE_BUNDLE}/Assets.xcassets"

if [[ -d "${XCASSETS}" ]]; then
  if command -v actool &>/dev/null || [[ -x /usr/bin/actool ]]; then
    echo "Compiling assets..."
    actool "${XCASSETS}" \
      --compile "${APP}/Contents/Resources" \
      --platform macosx \
      --minimum-deployment-target 26.0 \
      --app-icon AppIcon \
      --accent-color AccentColor \
      --output-partial-info-plist /dev/null \
      >/dev/null 2>&1 || echo "Warning: actool failed, assets may be incomplete"
  else
    echo "Warning: actool not found, copying raw assets"
    cp -r "${XCASSETS}" "${APP}/Contents/Resources/"
  fi
fi

echo "Created ${APP}"
