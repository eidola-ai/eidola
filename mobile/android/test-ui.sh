#!/usr/bin/env bash
# The phone interface's UI test on the Android emulator (ui_test.py): a development bundle whose
# default agent answers from tools/openai_stub.py on this machine (the emulator reaches it at
# 10.0.2.2), driven by real taps and typing (`adb shell input`) through a whole conversation, the
# elements found by probe name in the WebView's accessibility tree. Screenshots and a screen
# recording go to OUT (default build/ui).
#   test-ui.sh [OUT] [--bundle DIR]
# --bundle  a development bundle built for 10.0.2.2 (default: built here)
# Uses (and creates once) the AVD "eidola-shell" (test-emulator.sh's); never another AVD or a device.
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
SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
AVD=eidola-shell
PORT=5590
PKG=ai.eidola.mobile.dev
ADB=("$SDK/platform-tools/adb" -s "emulator-$PORT")
IMAGE=system-images/android-36.1/google_apis_playstore/arm64-v8a/
STUB_PORT="${STUB_PORT:-8791}"
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
if [[ -z "$BUNDLE" ]]; then
  "$HERE/../build-bundle.sh" --dev-backend "http://10.0.2.2:$STUB_PORT" --out "$HERE/../build/bundle-dev-android" >&2
  BUNDLE="$HERE/../build/bundle-dev-android"
fi
KEYSTORE="$HERE/.keys/debug.p12" "$HERE/build.sh" --debug --bundle "$BUNDLE"
# A fresh install: no earlier run's conversations.
"${ADB[@]}" uninstall "$PKG" >/dev/null 2>&1 || true
"${ADB[@]}" install -r "$HERE/build/debug/eidola-shell-signed.apk" >/dev/null

python3 "$HERE/../tools/openai_stub.py" --port "$STUB_PORT" --delay 0.08 >"$OUT/stub.log" 2>&1 &
STUB=$!
"${ADB[@]}" shell rm -f /sdcard/conversation.mp4
"${ADB[@]}" shell screenrecord --bit-rate 6000000 /sdcard/conversation.mp4 &
VIDEO=$!
cleanup() {
  kill "$STUB" 2>/dev/null || true
  kill -INT "$VIDEO" 2>/dev/null || true
}
trap cleanup EXIT

"${ADB[@]}" logcat -c
set +e
python3 "$HERE/ui_test.py" "emulator-$PORT" "$PKG" "$OUT"
STATUS=$?
set -e
"${ADB[@]}" shell pkill -INT screenrecord || true
sleep 3
"${ADB[@]}" pull /sdcard/conversation.mp4 "$OUT/conversation.mp4" >/dev/null || true
"${ADB[@]}" logcat -d -s EidolaShell:V chromium:V >"$OUT/logcat.txt" || true
echo "screenshots and recording: $OUT"
exit "$STATUS"
