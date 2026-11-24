#!/usr/bin/env bash
# The phone interface's UI test on the Android emulator (ui_test.py): a development bundle whose
# default agent answers from tools/openai_stub.py on this machine (the emulator reaches it at
# 10.0.2.2), which trusts the local dev stack's gateway behind its attesting shim
# (tools/account_stack.sh) and takes its update from a fixture release (tools/update_fixtures.py),
# driven by real taps and typing (`adb shell input`) through the first run with a new account and
# a checkout in the browser, a whole conversation, settings applied live, the system back
# gesture, a cold relaunch and the update (approved, fallen back from, reverted); the elements
# found by probe name in the WebView's accessibility tree. Screenshots and a screen recording go
# to OUT (default build/ui). Needs docker (the database).
#   test-ui.sh [OUT]
# Uses (and creates once) the AVD "eidola-shell" (test-emulator.sh's), or the one UI_AVD names; never
# another AVD or a device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-}"
OUT="$(mkdir -p "${OUT:-$HERE/build/ui}" && cd "${OUT:-$HERE/build/ui}" && pwd)"
SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
# UI_AVD and UI_PORT name another AVD and console port, and STUB_PORT and UPDATE_PORT the model
# stub's and update fixtures' ports (default 8791, 8792), for a second checkout running at once.
AVD="${UI_AVD:-eidola-shell}"
PORT="${UI_PORT:-5590}"
PKG=ai.eidola.app.android.dev
ADB=("$SDK/platform-tools/adb" -s "emulator-$PORT")
IMAGE=system-images/android-36.1/google_apis_playstore/arm64-v8a/
STUB_PORT="${STUB_PORT:-8791}"
UPDATE_PORT="${UPDATE_PORT:-8792}"
STACK="$HERE/build/account-stack"
mkdir -p "$HERE/build"

if [[ ! -f "$HOME/.android/avd/$AVD.ini" ]]; then
  mkdir -p "$HOME/.android/avd/$AVD.avd"
  printf 'avd.ini.encoding=UTF-8\npath=%s\npath.rel=avd/%s.avd\ntarget=android-36.1\n' \
    "$HOME/.android/avd/$AVD.avd" "$AVD" > "$HOME/.android/avd/$AVD.ini"
  cat > "$HOME/.android/avd/$AVD.avd/config.ini" <<CONFIG
AvdId=$AVD
avd.ini.displayname=$AVD
abi.type=arm64-v8a
hw.cpu.arch=arm64
hw.cpu.ncore=4
hw.ramSize=4096
disk.dataPartition.size=6G
image.sysdir.1=$IMAGE
tag.id=google_apis_playstore
PlayStore.enabled=true
target=android-36.1
hw.device.name=medium_phone
hw.lcd.density=420
hw.lcd.width=1080
hw.lcd.height=2400
hw.gpu.enabled=yes
hw.gpu.mode=auto
hw.keyboard=yes
hw.mainKeys=no
showDeviceFrame=no
CONFIG
fi

if ! "${ADB[@]}" get-state >/dev/null 2>&1; then
  "$SDK/emulator/emulator" -avd "$AVD" -port "$PORT" -no-window -no-snapshot -no-audio -no-boot-anim \
    -gpu auto >"$HERE/build/emulator.log" 2>&1 &
fi
"${ADB[@]}" wait-for-device
until [[ "$("${ADB[@]}" shell getprop sys.boot_completed | tr -d '\r')" == 1 ]]; do sleep 2; done

mkdir -p "$HERE/.keys"
[[ -f "$HERE/.keys/debug.p12" ]] || keytool -genkeypair -keystore "$HERE/.keys/debug.p12" -storetype PKCS12 \
  -storepass android -alias shell -keyalg EC -groupname secp256r1 -validity 3650 -dname "CN=Eidola shell debug key"
FIXTURES=
STUB=
VIDEO=
cleanup() {
  [[ -n "$STUB" ]] && kill "$STUB" 2>/dev/null || true
  [[ -n "$VIDEO" ]] && kill -INT "$VIDEO" 2>/dev/null || true
  [[ -n "$FIXTURES" ]] && kill "$FIXTURES" 2>/dev/null || true
  "$HERE/../tools/account_stack.sh" down "$STACK" || true
}
trap cleanup EXIT
"$HERE/../tools/account_stack.sh" up "$STACK"
python3 - "$STACK/eidola.json" >"$STACK/eidola-android.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["base_url"] = "https://10.0.2.2:%d" % d.pop("port")
print(json.dumps(d))
PY
BUNDLE="$HERE/../build/bundle-dev-android"
"$HERE/../build-bundle.sh" --dev-backend "http://10.0.2.2:$STUB_PORT" \
  --dev-update "http://10.0.2.2:$UPDATE_PORT/update.json" --dev-eidola "$STACK/eidola-android.json" \
  --out "$BUNDLE" >&2
python3 "$HERE/../tools/update_fixtures.py" "$BUNDLE" "$HERE/build/update-fixtures" --port "$UPDATE_PORT" \
  --base "http://10.0.2.2:$UPDATE_PORT" >"$OUT/fixtures.log" 2>&1 &
FIXTURES=$!
KEYSTORE="$HERE/.keys/debug.p12" "$HERE/build.sh" --debug --bundle "$BUNDLE"
# A fresh install: no earlier run's conversations.
"${ADB[@]}" uninstall "$PKG" >/dev/null 2>&1 || true
"${ADB[@]}" install -r "$HERE/build/debug/eidola-shell-signed.apk" >/dev/null

python3 "$HERE/../tools/openai_stub.py" --port "$STUB_PORT" --delay 0.08 >"$OUT/stub.log" 2>&1 &
STUB=$!
"${ADB[@]}" shell rm -f /sdcard/conversation.mp4
# screenrecord stops at three minutes; the run is longer, so it records in segments.
( for n in 1 2 3 4 5 6 7 8 9; do "${ADB[@]}" shell screenrecord --bit-rate 6000000 "/sdcard/ui-$n.mp4" || break; done ) &
VIDEO=$!

"${ADB[@]}" logcat -c
set +e
python3 "$HERE/ui_test.py" "emulator-$PORT" "$PKG" "$OUT"
STATUS=$?
set -e
kill "$VIDEO" 2>/dev/null || true
VIDEO=
"${ADB[@]}" shell pkill -INT screenrecord || true
sleep 3
for n in 1 2 3 4 5 6 7 8 9; do "${ADB[@]}" pull "/sdcard/ui-$n.mp4" "$OUT/ui-$n.mp4" >/dev/null 2>&1 || break; done
"${ADB[@]}" logcat -d -s EidolaShell:V chromium:V >"$OUT/logcat.txt" || true
echo "screenshots and recording: $OUT"
exit "$STATUS"
