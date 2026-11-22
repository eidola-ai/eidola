#!/usr/bin/env bash
# Runs the contract test page on an Android emulator: the whole contract, the native bundle
# sheet (the launcher shortcut's intent) against the page's digest, and a cold relaunch.
# Screenshots go to SHOTS (default build/shots).
#   test-emulator.sh [SHOTS]
# Uses (and creates once) its own AVD, "eidola-shell", on the android-36.1 Play image; never
# another AVD or a device. Needs the SDK's emulator, platform-tools and that system image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SHOTS="$(mkdir -p "${1:-$HERE/build/shots}" && cd "${1:-$HERE/build/shots}" && pwd)"
SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
AVD=eidola-shell
PORT=5590
PKG=ai.eidola.mobile.dev
ADB=("$SDK/platform-tools/adb" -s "emulator-$PORT")
IMAGE=system-images/android-36.1/google_apis_playstore/arm64-v8a/

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
KEYSTORE="$HERE/.keys/debug.p12" "$HERE/build.sh" --debug
# A fresh install, so no earlier run's files or bundles remain.
"${ADB[@]}" uninstall "$PKG" >/dev/null 2>&1 || true
"${ADB[@]}" install -r "$HERE/build/debug/eidola-shell-signed.apk" >/dev/null

python3 "$HERE/../tools/echo_server.py" 7777 &
ECHO=$!
trap 'kill $ECHO 2>/dev/null || true' EXIT

report() { "${ADB[@]}" shell run-as "$PKG" cat no_backup/files/contract-report.json 2>/dev/null || true; }
field() { python3 -c 'import json,sys; print(json.loads(sys.stdin.read() or "{}").get(sys.argv[1], ""))' "$1"; }
wait_for() { # wait_for FIELD SECONDS
  for _ in $(seq "$2"); do
    [[ -n "$(report | field "$1")" ]] && return 0
    sleep 1
  done
  echo "timed out waiting for $1" >&2
  return 1
}

"${ADB[@]}" logcat -c
"${ADB[@]}" shell am start -W -n "$PKG/ai.eidola.shell.MainActivity" >/dev/null
wait_for summary 180
sleep 1
"${ADB[@]}" exec-out screencap -p >"$SHOTS/1-contract.png"
SUMMARY="$(report | field summary)"
echo "$SUMMARY"

# The bundle sheet, through the intent the launcher shortcut sends.
"${ADB[@]}" shell am start -a ai.eidola.shell.SHOW_DIGEST -n "$PKG/ai.eidola.shell.MainActivity" >/dev/null
sleep 3
"${ADB[@]}" exec-out screencap -p >"$SHOTS/2-bundle-sheet.png"
"${ADB[@]}" shell uiautomator dump /sdcard/ui.xml >/dev/null
SHEET="$("${ADB[@]}" shell cat /sdcard/ui.xml)"
DIGEST="$(python3 "$HERE/../tools/bundle_digest.py" "$HERE/../contract-test")"
SHEET_OK=FAIL
[[ "$(tr -d '\n' <<<"$SHEET" | sed 's/&#10;//g')" == *"$DIGEST"* ]] && SHEET_OK=PASS
echo "sheet: $SHEET_OK (digest $DIGEST)"
"${ADB[@]}" shell input keyevent KEYCODE_BACK

# A cold relaunch: the synced file survived the process.
"${ADB[@]}" shell am force-stop "$PKG"
"${ADB[@]}" shell am start -W -n "$PKG/ai.eidola.shell.MainActivity" >/dev/null
wait_for restart 60
"${ADB[@]}" exec-out screencap -p >"$SHOTS/3-restart.png"
RESTART="$(report | field restart)"
echo "$RESTART"
"${ADB[@]}" logcat -d -s EidolaShell:V >"$SHOTS/logcat.txt"
echo "screenshots: $SHOTS"
[[ "$SUMMARY" == "contract: PASS"* && "$SHEET_OK" == PASS && "$RESTART" == "restart: PASS"* ]]
