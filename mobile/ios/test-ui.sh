#!/usr/bin/env bash
# The phone interface's UI tests (UITests/PhoneUITests.swift) in the iOS Simulator: a development
# bundle whose default agent answers from tools/openai_stub.py on this machine, driven by real taps
# and typing through a whole conversation (a new conversation, the keyboard, a streamed answer with
# its reasoning, markdown, a reply branch, scrolling, the Library, an untouched conversation not
# kept), Chinese through the Pinyin IME in the app's zh-Hans locale, and a cold relaunch at night.
# Screenshots and a screen recording go to OUT (default build/ui).
#   test-ui.sh [OUT] [--bundle DIR]
# --bundle  a development bundle built for 127.0.0.1:$STUB_PORT (default: built here)
# STUB_PORT the model stub's port (default 8790), so two runs on one machine stay apart
# Uses (and creates once) its own simulators, "eidola-phone" and "eidola-phone-zh" (its language
# Chinese, so Pinyin is its first keyboard); never another device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=
BUNDLE=
while (($#)); do
  case "$1" in
    --bundle) BUNDLE="$(cd "$2" && pwd)"; shift 2 ;;
    *) OUT="$1"; shift ;;
  esac
done
OUT="$(mkdir -p "${OUT:-$HERE/build/ui}" && cd "${OUT:-$HERE/build/ui}" && pwd)"
DEVICE_TYPE="${DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro}"
RUNTIME="${RUNTIME:-com.apple.CoreSimulator.SimRuntime.iOS-27-0}"
PORT="${STUB_PORT:-8790}"

simulator() { # simulator NAME: its udid, created and booted
  local udid
  udid="$(xcrun simctl list devices -j | python3 -c '
import json, sys
for devices in json.load(sys.stdin)["devices"].values():
    for d in devices:
        if d["name"] == sys.argv[1] and d["isAvailable"]:
            print(d["udid"]); raise SystemExit' "$1")"
  if [[ -z "$udid" ]]; then
    udid="$(xcrun simctl create "$1" "$DEVICE_TYPE" "$RUNTIME")"
    if [[ "$1" == *-zh ]]; then
      xcrun simctl boot "$udid"
      xcrun simctl bootstatus "$udid" -b >/dev/null
      xcrun simctl spawn "$udid" defaults write -g AppleLanguages -array zh-Hans-CN en-US
      xcrun simctl spawn "$udid" defaults write -g AppleLocale -string zh_CN
      xcrun simctl spawn "$udid" defaults write -g AppleKeyboards -array \
        "zh_Hans-Pinyin@sw=Pinyin;hw=Automatic" "en_US@sw=QWERTY;hw=Automatic" "emoji@sw=Emoji"
      xcrun simctl shutdown "$udid"
    fi
  fi
  xcrun simctl boot "$udid" 2>/dev/null || true
  xcrun simctl bootstatus "$udid" -b >/dev/null
  echo "$udid"
}
UDID="$(simulator eidola-phone)"
ZH="$(simulator eidola-phone-zh)"
echo "simulators: eidola-phone $UDID, eidola-phone-zh $ZH"
xcrun simctl ui "$UDID" appearance light
xcrun simctl ui "$ZH" appearance light
# Fresh installs: no earlier run's conversations.
xcrun simctl uninstall "$UDID" ai.eidola.mobile.dev || true
xcrun simctl uninstall "$ZH" ai.eidola.mobile.dev || true

python3 "$HERE/../tools/openai_stub.py" --port "$PORT" --delay 0.08 >"$OUT/stub.log" 2>&1 &
STUB=$!
VIDEO=
cleanup() {
  [[ -n "$VIDEO" ]] && kill -INT "$VIDEO" 2>/dev/null && wait "$VIDEO" 2>/dev/null || true
  kill "$STUB" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -z "$BUNDLE" ]]; then
  "$HERE/../build-bundle.sh" --dev-backend "http://127.0.0.1:$PORT" >&2
  BUNDLE="$HERE/../build/bundle-dev"
fi
"$HERE/build.sh" --bundle "$BUNDLE"

run_test() { # run_test SIMULATOR METHOD
  rm -rf "$HERE/build/ui-$2.xcresult"
  TEST_RUNNER_SHOTS="$OUT" xcodebuild -project "$HERE/EidolaShell.xcodeproj" -scheme EidolaShell \
    -configuration Debug -derivedDataPath "$HERE/build/DerivedData" -destination "id=$1" \
    -resultBundlePath "$HERE/build/ui-$2.xcresult" -only-testing:"EidolaShellUITests/PhoneUITests/$2" \
    CODE_SIGNING_ALLOWED=NO test 2>&1 | grep -E "Test Case|error:|\*\* TEST|failed"
  return "${PIPESTATUS[0]}"
}
rm -f "$OUT/conversation.mp4"
xcrun simctl io "$UDID" recordVideo --codec h264 --force "$OUT/conversation.mp4" >/dev/null 2>&1 &
VIDEO=$!
set +e
run_test "$UDID" testConversation
STATUS=$?
# The same profile after a cold launch, with the device switched to dark.
xcrun simctl ui "$UDID" appearance dark
run_test "$UDID" testRelaunchAtNight
NIGHT=$?
xcrun simctl ui "$UDID" appearance light
kill -INT "$VIDEO" 2>/dev/null && wait "$VIDEO" 2>/dev/null
VIDEO=
xcrun simctl io "$ZH" recordVideo --codec h264 --force "$OUT/chinese-ime.mp4" >/dev/null 2>&1 &
VIDEO=$!
run_test "$ZH" testChineseInput
IME=$?
set -e
echo "screenshots and recordings: $OUT"
for s in $STATUS $NIGHT $IME; do [[ $s == 0 ]] || exit 1; done
