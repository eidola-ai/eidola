#!/usr/bin/env bash
# Runs the contract test page in the iOS Simulator under XCUITest: the whole contract (seeing
# Safari open for shell.openExternal), the native bundle sheet (opened from the home-screen
# quick action) against the page's digest, and a cold relaunch. Screenshots go to SHOTS
# (default build/shots).
#   test-sim.sh [SHOTS]
# Uses (and creates once per runtime) its own simulator, "eidola-shell"; never another device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SHOTS="$(mkdir -p "${1:-$HERE/build/shots}" && cd "${1:-$HERE/build/shots}" && pwd)"
NAME=eidola-shell
# DEVICE_TYPE, RUNTIME: see simulator.sh (default the newest installed iOS runtime).
source "$HERE/simulator.sh"

UDID="$(simulator_udid "$NAME")"
echo "simulator $NAME: $UDID ($RUNTIME)"
xcrun simctl bootstatus "$UDID" -b >/dev/null
# A fresh install, so no earlier run's files or bundles remain.
xcrun simctl uninstall "$UDID" ai.eidola.app.ios.dev || true

python3 "$HERE/../tools/echo_server.py" 7777 &
ECHO=$!
trap 'kill $ECHO 2>/dev/null || true' EXIT

"$HERE/build.sh" --bundle "$HERE/../contract-test"
rm -rf "$HERE/build/test.xcresult"
TEST_RUNNER_SHOTS="$SHOTS" xcodebuild -project "$HERE/EidolaShell.xcodeproj" -scheme EidolaShell \
  -configuration Debug -derivedDataPath "$HERE/build/DerivedData" -destination "id=$UDID" \
  -resultBundlePath "$HERE/build/test.xcresult" -only-testing:EidolaShellUITests/ContractUITests \
  CODE_SIGNING_ALLOWED=NO test 2>&1 |
  grep -E "Test Case|error:|\*\* TEST|contract:|restart:" 
echo "screenshots: $SHOTS"
