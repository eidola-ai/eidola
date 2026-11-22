#!/usr/bin/env bash
# Runs the app's headless self-test (crates/eidola-mobile/src/selftest.rs) under a shell, in the iOS
# Simulator or the Android emulator, against attested shims run on this machine:
#   - a tinfoil-shim-mock attesting the measurement the app is told to trust, and one attesting
#     another, both in front of a static upstream that serves GET /v1/terms;
#   - tools/echo_server.py, for the pipe's throughput.
# The self-test bundle is built, embedded in a debug shell, launched (phase 1: boot, attestation,
# the barrier, the bundle store, throughput), killed and cold-launched (phase 2: the profile and
# its settings survived). Reports, logs and screenshots go to OUT (default build/self-test/PLATFORM).
#   self-test.sh ios|android [--out DIR]
# Uses its own simulator ("eidola-bridge") or AVD ("eidola-bridge"), created once; never a device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PLATFORM="${1:-}"
[[ "$PLATFORM" == ios || "$PLATFORM" == android ]] || { echo "usage: self-test.sh ios|android [--out DIR]" >&2; exit 2; }
shift
OUT="$HERE/build/self-test/$PLATFORM"
while (($#)); do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
rm -rf "$OUT" && mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
T="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$T"; }
trap cleanup EXIT

# The address the app reaches this machine at.
if [[ $PLATFORM == ios ]]; then HOST=127.0.0.1; else HOST=10.0.2.2; fi
GOOD="$(printf '6b%.0s' $(seq 48))"
BAD="$(printf '11%.0s' $(seq 48))"
ZERO="$(printf '0%.0s' $(seq 64))"

# 1. The peers.
(cd "$ROOT" && cargo build -q -p tinfoil-shim-mock)
SHIM="${CARGO_TARGET_DIR:-$ROOT/target}/debug/tinfoil-shim-mock"
mkdir -p "$T/www/v1" "$T/certs"
echo '{"documents":[]}' >"$T/www/v1/terms"
python3 -m http.server 18080 --bind 127.0.0.1 --directory "$T/www" >"$OUT/upstream.log" 2>&1 &
PIDS+=($!)
python3 "$HERE/tools/echo_server.py" 7777 >"$OUT/echo.log" 2>&1 &
PIDS+=($!)
shim() { # shim PORT MEASUREMENT NAME
  CERT_DIR="$T/certs" LISTEN_ADDR="127.0.0.1:$1" UPSTREAM_URL=http://127.0.0.1:18080 DEV_MEASUREMENT="$2" \
    EXTRA_TLS_NAMES=10.0.2.2 "$SHIM" >"$OUT/$3.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 50); do nc -z 127.0.0.1 "$1" 2>/dev/null && return 0; sleep 0.2; done
  echo "shim on $1 did not start" >&2; return 1
}
shim 18443 "$GOOD" shim-trusted
shim 18444 "$BAD" shim-untrusted

# 2. The self-test bundle, with this run's settings.
"$HERE/build-bundle.sh" --self-test --out "$T/bundle" | tee "$OUT/bundle.txt"
python3 - "$T" "$HOST" "$GOOD" "$ZERO" >"$T/bundle/selftest.json" <<'PY'
import json, sys
t, host, good, zero = sys.argv[1:]
pem = lambda name: open(f"{t}/certs/{name}").read()
print(json.dumps({
    "good_url": f"https://{host}:18443", "bad_url": f"https://{host}:18444",
    "measurement": good, "host_data": zero, "guest_policy": "0x30133",
    "ark_pem": pem("ark.pem"), "ask_pem": pem("ask.pem"), "tls_ca_pem": pem("tls-ca.pem"),
    "echo_host": host, "echo_port": 7777, "throughput_mib": 16,
}, indent=1))
PY

wait_for() { # wait_for SECONDS COMMAND...: until COMMAND succeeds
  local n="$1"; shift
  for _ in $(seq "$n"); do "$@" && return 0; sleep 1; done
  return 1
}

if [[ $PLATFORM == ios ]]; then
  UDID="$(xcrun simctl list devices -j | python3 -c '
import json, sys
for devices in json.load(sys.stdin)["devices"].values():
    for d in devices:
        if d["name"] == "eidola-bridge" and d["isAvailable"]:
            print(d["udid"]); raise SystemExit')"
  [[ -n "$UDID" ]] || UDID="$(xcrun simctl create eidola-bridge com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro \
    com.apple.CoreSimulator.SimRuntime.iOS-27-0)"
  xcrun simctl boot "$UDID" 2>/dev/null || true
  timeout 180 xcrun simctl bootstatus "$UDID" -b >/dev/null
  "$HERE/ios/build.sh" --bundle "$T/bundle" | tail -2
  APP="$HERE/ios/build/DerivedData/Build/Products/Debug-iphonesimulator/EidolaShell.app"
  PKG=ai.eidola.mobile.dev
  xcrun simctl uninstall "$UDID" "$PKG" 2>/dev/null || true
  xcrun simctl install "$UDID" "$APP"
  xcrun simctl spawn "$UDID" log stream --style compact --level debug \
    --predicate 'subsystem == "ai.eidola.shell"' >"$OUT/console.log" 2>&1 &
  PIDS+=($!)
  sleep 2
  FILES="$(xcrun simctl get_app_container "$UDID" "$PKG" data)/Library/Application Support/Eidola/files"
  report() { cat "$FILES/selftest-$1.json" 2>/dev/null; }
  launch() { xcrun simctl launch "$UDID" "$PKG" >/dev/null; }
  kill_app() { xcrun simctl terminate "$UDID" "$PKG"; }
  shot() { xcrun simctl io "$UDID" screenshot "$OUT/$1.png" >/dev/null 2>&1; }
  files() { ls -l "$FILES"; }
else
  SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
  AVD=eidola-bridge
  PORT=5592
  ADB=("$SDK/platform-tools/adb" -s "emulator-$PORT")
  if [[ ! -f "$HOME/.android/avd/$AVD.ini" ]]; then
    mkdir -p "$HOME/.android/avd/$AVD.avd"
    printf 'avd.ini.encoding=UTF-8\npath=%s\npath.rel=avd/%s.avd\ntarget=android-36.1\n' \
      "$HOME/.android/avd/$AVD.avd" "$AVD" >"$HOME/.android/avd/$AVD.ini"
    cat >"$HOME/.android/avd/$AVD.avd/config.ini" <<CONFIG
AvdId=$AVD
avd.ini.displayname=$AVD
abi.type=arm64-v8a
hw.cpu.arch=arm64
hw.cpu.ncore=4
hw.ramSize=4096
disk.dataPartition.size=6G
image.sysdir.1=system-images/android-36.1/google_apis_playstore/arm64-v8a/
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
      -gpu auto >"$OUT/emulator.log" 2>&1 &
  fi
  timeout 300 "${ADB[@]}" wait-for-device
  wait_for 300 bash -c "[[ \"\$('${ADB[0]}' -s emulator-$PORT shell getprop sys.boot_completed | tr -d '\r')\" == 1 ]]"
  mkdir -p "$HERE/android/.keys"
  [[ -f "$HERE/android/.keys/debug.p12" ]] || keytool -genkeypair -keystore "$HERE/android/.keys/debug.p12" \
    -storetype PKCS12 -storepass android -alias shell -keyalg EC -groupname secp256r1 -validity 3650 \
    -dname "CN=Eidola shell debug key"
  KEYSTORE="$HERE/android/.keys/debug.p12" "$HERE/android/build.sh" --debug --bundle "$T/bundle" --out "$T/apk" | tail -2
  PKG=ai.eidola.mobile.dev
  "${ADB[@]}" uninstall "$PKG" >/dev/null 2>&1 || true
  "${ADB[@]}" install -r "$T/apk/eidola-shell-signed.apk" >/dev/null
  "${ADB[@]}" logcat -c
  report() { "${ADB[@]}" shell run-as "$PKG" cat "no_backup/files/selftest-$1.json" 2>/dev/null | grep . ; }
  launch() { "${ADB[@]}" shell am start -W -n "$PKG/ai.eidola.shell.MainActivity" >/dev/null; }
  kill_app() { "${ADB[@]}" shell am force-stop "$PKG"; }
  shot() { "${ADB[@]}" exec-out screencap -p >"$OUT/$1.png"; }
  files() { "${ADB[@]}" shell run-as "$PKG" ls -l no_backup/files; }
fi

# 3. Phase 1, a kill, phase 2.
launch
wait_for 180 report 1 >/dev/null || { echo "phase 1 reported nothing" >&2; }
sleep 1
shot 1-first-launch
report 1 >"$OUT/selftest-1.json" || true
files >"$OUT/files-after-phase-1.txt" 2>&1 || true
kill_app
sleep 1
launch
wait_for 120 report 2 >/dev/null || { echo "phase 2 reported nothing" >&2; }
sleep 1
shot 2-relaunch
report 2 >"$OUT/selftest-2.json" || true
files >"$OUT/files-after-phase-2.txt" 2>&1 || true
[[ $PLATFORM == android ]] && "${ADB[@]}" logcat -d -s EidolaShell:V >"$OUT/console.log"
kill_app || true

python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
ok = True
for phase in (1, 2):
    try:
        report = json.load(open(f"{out}/selftest-{phase}.json"))
    except Exception as e:
        print(f"phase {phase}: no report ({e})"); ok = False; continue
    for check in report["checks"]:
        print(f"phase {phase}: {'PASS' if check['pass'] else 'FAIL'} {check['name']}")
    for m in report.get("measurements", []):
        print(f"phase {phase}: measured {json.dumps(m)}")
    ok = ok and report["pass"]
print("self-test:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
