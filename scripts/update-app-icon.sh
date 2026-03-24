#!/bin/bash
set -euo pipefail

# Converts AppIcon.appiconset (xcassets) to AppIcon.icns via iconutil.
# This avoids depending on actool/Xcode for the Nix hermetic build.
#
# Prerequisites: icon PNGs must be present in the appiconset directory.
# iconutil ships with macOS (no Xcode required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

APPICONSET="${ROOT_DIR}/apps/macos/Sources/EidolonsEntrypoint/Assets.xcassets/AppIcon.appiconset"
ICNS_OUTPUT="${ROOT_DIR}/apps/macos/Support/AppIcon.icns"

# Map xcassets size+scale to iconutil .iconset naming convention
# See: https://developer.apple.com/documentation/bundleresources/information-property-list/cfbundleiconfile
declare -A ICON_MAP=(
  ["16x16@1x"]="icon_16x16.png"
  ["16x16@2x"]="icon_16x16@2x.png"
  ["32x32@1x"]="icon_32x32.png"
  ["32x32@2x"]="icon_32x32@2x.png"
  ["128x128@1x"]="icon_128x128.png"
  ["128x128@2x"]="icon_128x128@2x.png"
  ["256x256@1x"]="icon_256x256.png"
  ["256x256@2x"]="icon_256x256@2x.png"
  ["512x512@1x"]="icon_512x512.png"
  ["512x512@2x"]="icon_512x512@2x.png"
)

# Parse Contents.json to find which images are present
CONTENTS="${APPICONSET}/Contents.json"
if [[ ! -f "$CONTENTS" ]]; then
  echo "Error: No Contents.json found at ${CONTENTS}"
  exit 1
fi

# Check if any images have filenames (i.e., actual PNGs are present)
IMAGE_COUNT=$(jq '[.images[] | select(.filename)] | length' "$CONTENTS")
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
  echo "No icon images found in AppIcon.appiconset (all slots empty)."
  echo "Add PNGs to the appiconset in Xcode, then re-run this script."
  # Remove stale .icns if it exists
  rm -f "$ICNS_OUTPUT"
  exit 0
fi

# Create temporary .iconset directory
ICONSET=$(mktemp -d)/AppIcon.iconset
mkdir -p "$ICONSET"

# Copy PNGs from appiconset to iconset with correct naming
jq -c '.images[] | select(.filename)' "$CONTENTS" | while read -r entry; do
  SIZE=$(echo "$entry" | jq -r '.size')
  SCALE=$(echo "$entry" | jq -r '.scale')
  FILENAME=$(echo "$entry" | jq -r '.filename')
  KEY="${SIZE}@${SCALE}"

  ICONSET_NAME="${ICON_MAP[$KEY]:-}"
  if [[ -z "$ICONSET_NAME" ]]; then
    echo "Warning: unknown size/scale ${KEY}, skipping"
    continue
  fi

  SRC="${APPICONSET}/${FILENAME}"
  if [[ ! -f "$SRC" ]]; then
    echo "Warning: ${FILENAME} referenced in Contents.json but not found, skipping"
    continue
  fi

  cp "$SRC" "${ICONSET}/${ICONSET_NAME}"
done

# Convert to .icns
iconutil --convert icns --output "$ICNS_OUTPUT" "$ICONSET"

# Clean up
rm -rf "$(dirname "$ICONSET")"

echo "Updated ${ICNS_OUTPUT}"
