#!/usr/bin/env bash
# Runs the contract test page in the iOS Simulator under XCUITest: the whole contract (seeing
# Safari open for shell.openExternal), the native bundle sheet (opened from the home-screen
# quick action) against the page's digest, and a cold relaunch. Screenshots go to SHOTS
# (default build/shots).
#   test-sim.sh [SHOTS]
# Uses (and creates once) its own simulator, "eidola-shell"; never another device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SHOTS="$(mkdir -p "${1:-$HERE/build/shots}" && cd "${1:-$HERE/build/shots}" && pwd)"
NAME=eidola-shell
DEVICE_TYPE="${DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro}"
RUNTIME="${RUNTIME:-com.apple.CoreSimulator.SimRuntime.iOS-27-0}"

UDID="$(xcrun simctl list devices -j | python3 -c '
import json, sys
for devices in json.load(sys.stdin)["devices"].values():
    for d in devices:
        if d["name"] == sys.argv[1] and d["isAvailable"]:
            print(d["udid"]); raise SystemExit' "$NAME")"
[[ -n "$UDID" ]] || UDID="$(xcrun simctl create "$NAME" "$DEVICE_TYPE" "$RUNTIME")"
echo "simulator $NAME: $UDID"
xcrun simctl bootstatus "$UDID" -b >/dev/null
# A fresh install, so no earlier run's files or bundles remain.
xcrun simctl uninstall "$UDID" ai.eidola.mobile.dev || true

python3 "$HERE/../tools/echo_server.py" 7777 &
ECHO=$!
trap 'kill $ECHO 2>/dev/null || true' EXIT

"$HERE/build.sh"
rm -rf "$HERE/build/test.xcresult"
TEST_RUNNER_SHOTS="$SHOTS" xcodebuild -project "$HERE/EidolaShell.xcodeproj" -scheme EidolaShell \
  -configuration Debug -derivedDataPath "$HERE/build/DerivedData" -destination "id=$UDID" \
  -resultBundlePath "$HERE/build/test.xcresult" CODE_SIGNING_ALLOWED=NO test 2>&1 |
  grep -E "Test Case|error:|\*\* TEST|contract:|restart:" 
echo "screenshots: $SHOTS"
