#!/usr/bin/env bash
# Builds the iOS shell with the bundle it embeds.
#   build.sh [--bundle DIR] [--release] [--device]
# --bundle  the bundle to embed (default: the app, built by ../build-bundle.sh; the tests embed
#           ../contract-test)
# --release the Release configuration, as ai.eidola.mobile (default Debug, as ai.eidola.mobile.dev:
#           console in the system log, inspectable)
# --device  an iPhoneOS build signed by the team in the project (needs that Apple ID in Xcode);
#           default is an unsigned Simulator build.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUNDLE=
CONFIG=Debug
DEVICE=false
while (($#)); do
  case "$1" in
    --bundle) BUNDLE="$(cd "$2" && pwd)"; shift 2 ;;
    --release) CONFIG=Release; shift ;;
    --device) DEVICE=true; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$BUNDLE" ]]; then
  "$HERE/../build-bundle.sh" >&2
  BUNDLE="$HERE/../build/bundle"
fi
test -f "$BUNDLE/index.html" || { echo "$BUNDLE has no index.html" >&2; exit 1; }
# The digest tool refuses a file the shell would not serve; it prints the digest the shell will show.
DIGEST="$(python3 "$HERE/../tools/bundle_digest.py" "$BUNDLE")"
rm -rf "$HERE/build/bundle"
mkdir -p "$HERE/build"
cp -R "$BUNDLE" "$HERE/build/bundle"

if $DEVICE; then
  SDK=(-sdk iphoneos -destination generic/platform=iOS -allowProvisioningUpdates)
  PRODUCTS="$CONFIG-iphoneos"
else
  SDK=(-sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO)
  PRODUCTS="$CONFIG-iphonesimulator"
fi
xcodebuild -project "$HERE/EidolaShell.xcodeproj" -scheme EidolaShell -configuration "$CONFIG" \
  -derivedDataPath "$HERE/build/DerivedData" "${SDK[@]}" -quiet build
echo "app:    $HERE/build/DerivedData/Build/Products/$PRODUCTS/EidolaShell.app"
echo "bundle: $BUNDLE"
echo "digest: $DIGEST"
