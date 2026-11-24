#!/usr/bin/env bash
# Runs the app's headless self-test (crates/eidola-mobile/src/selftest.rs) under a shell, in the iOS
# Simulator or the Android emulator, against attested shims run on this machine:
#   - a tinfoil-shim-mock attesting the measurement the app is told to trust, and one attesting
#     another, both in front of a static upstream that serves GET /v1/terms;
#   - tools/echo_server.py, for the pipe's throughput.
# The self-test bundle is built, embedded in a debug shell, launched (phase 1: boot, attestation,
# the barrier, the bundle store, throughput; the self-update: a captured signed release verified
# in the app, and an update approved, checked, staged and activated), killed and cold-launched
# (phase 2: the profile and its settings survived; the update booted and confirmed with its
# release's digest, which the shell's native sheet shows on Android; a broken update approved),
# launched again (the broken bundle's trial, which never confirms) and once more (phase 3: the
# shell fell back to the confirmed update and recorded the failure; the user's revert).
# The releases are served over HTTPS by the trusted shim: the captured v0.0.8 release (genuinely
# signed; verified for real) and two bundle containers whose releases the app takes as verified,
# since no manifest recording a bundle can be signed outside a release.
# Reports, logs and screenshots go to OUT (default build/self-test/PLATFORM).
#   self-test.sh ios|android [--out DIR]
#   self-test.sh ios --device UDID [--host ADDRESS] [--out DIR]
# Uses its own simulator ("eidola-bridge") or AVD ("eidola-bridge"), created once; never another.
# --device runs it on that physical iOS device instead (mobile/device-checklist.md): a debug shell
# signed by the team (IOS_TEAM overrides the project's), reaching the shims and the echo server
# over the LAN at ADDRESS (default this Mac's en0 address), which their certificates name. It
# installs over the app already there, which keeps its local network permission, and starts the
# self-test afresh by blanking its state file and reports (the self-test removes the bundle
# versions an earlier run stored). On a device without the app, the first launch asks for local
# network access; the self-test waits up to three minutes for it to be allowed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PLATFORM="${1:-}"
[[ "$PLATFORM" == ios || "$PLATFORM" == android ]] || { echo "usage: self-test.sh ios|android [--out DIR]" >&2; exit 2; }
shift
OUT=
DEVICE=
HOST=
while (($#)); do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -z "$DEVICE" || $PLATFORM == ios ]] || { echo "--device is for ios" >&2; exit 2; }
OUT="${OUT:-$HERE/build/self-test/$PLATFORM${DEVICE:+-device}}"
rm -rf "$OUT" && mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
T="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$T"; }
trap cleanup EXIT

# The address the app reaches this machine at, and the one the peers listen on.
if [[ -n "$DEVICE" ]]; then
  HOST="${HOST:-$(ipconfig getifaddr en0)}"
  BIND=0.0.0.0
  [[ -n "$HOST" ]] || { echo "no LAN address (pass --host)" >&2; exit 2; }
elif [[ $PLATFORM == ios ]]; then HOST=127.0.0.1; BIND=127.0.0.1
else HOST=10.0.2.2; BIND=127.0.0.1; fi
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
python3 "$HERE/tools/echo_server.py" 7777 "$BIND" >"$OUT/echo.log" 2>&1 &
PIDS+=($!)
shim() { # shim PORT MEASUREMENT NAME
  CERT_DIR="$T/certs" LISTEN_ADDR="$BIND:$1" UPSTREAM_URL=http://127.0.0.1:18080 DEV_MEASUREMENT="$2" \
    EXTRA_TLS_NAMES="10.0.2.2${DEVICE:+,$HOST}" "$SHIM" >"$OUT/$3.log" 2>&1 &
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
    "update_url": f"https://{host}:18443/selftest-update.json",
}, indent=1))
PY

# The self-update's releases. The update is this bundle plus a marker (a digest of its own); the
# broken update is a page with no app, which never confirms.
cp -R "$T/bundle" "$T/update"
echo "the self-update's target bundle" >"$T/update/update-marker.txt"
mkdir -p "$T/broken"
echo '<!doctype html><title>a bundle that never confirms</title>' >"$T/broken/index.html"
FIXTURES="$ROOT/crates/eidola-app-core/tests/fixtures/v0.0.8"
# The broken release's version is this run's own: the shell remembers the last version that failed
# its trial, and the app refuses to install that one again, so a device that keeps its install
# from one run to the next must be offered a new one.
BROKEN="0.0.$(date +%s)"
mkdir -p "$T/www/latest/download" "$T/www/download/v0.0.8" "$T/www/download/v0.0.9" "$T/www/download/v$BROKEN"
cp "$FIXTURES"/artifact-manifest.json "$FIXTURES"/artifact-manifest.json.sigstore \
  "$FIXTURES"/attestation-throwaway.json "$FIXTURES"/attestation-throwaway.bundle.json "$T/www/download/v0.0.8/"
python3 "$HERE/tools/bundle_container.py" "$T/update" "$T/www/download/v0.0.9/eidola-mobile-bundle.tar" >"$T/update-row.json"
python3 "$HERE/tools/bundle_container.py" "$T/broken" "$T/www/download/v$BROKEN/eidola-mobile-bundle.tar" >"$T/broken-row.json"
python3 - "$T" "https://$HOST:18443" "$BROKEN" >"$T/www/selftest-update.json" <<'PY'
import base64, json, sys
t, base, broken = sys.argv[1:]
v8 = f"{base}/download/v0.0.8"
with open(f"{t}/www/latest/download/release.json", "w") as f:
    json.dump({
        "schema_version": 1, "version": "0.0.8", "git_commit": "bb7773fafbaa90d40e7c7634ff7f690b54dd9be9",
        "git_tag": "v0.0.8", "released_at": "2026-05-31T23:53:54Z",
        "previous_release": {"version": "0.0.6", "git_commit": "7ba323817820a04948ffc70e78955201c96c733f"},
        "artifact_manifest": {"url": f"{v8}/artifact-manifest.json",
                              "sigstore_bundle_url": f"{v8}/artifact-manifest.json.sigstore"},
        "human_attestations": [{"attestant_id": "throwaway", "url": f"{v8}/attestation-throwaway.json",
                                "bundle_url": f"{v8}/attestation-throwaway.bundle.json"}],
    }, f)
def staged(version, row):
    tag = f"v{version}"
    manifest = json.dumps({"schema_version": 6, "artifacts": {"eidola-mobile-bundle": row}}).encode()
    return {"index": {
        "schema_version": 2, "version": version, "git_commit": "9c3a" + "0" * 34 + version.replace(".", "")[-2:],
        "git_tag": tag, "released_at": "2026-11-23T00:00:00Z",
        "artifact_manifest": {"url": f"{base}/download/{tag}/artifact-manifest.json",
                              "sigstore_bundle_url": f"{base}/download/{tag}/artifact-manifest.json.sigstore"},
        "human_attestations": [],
        "artifacts": {"eidola-mobile-bundle": {"url": f"{base}/download/{tag}/eidola-mobile-bundle.tar"}},
    }, "manifest": base64.b64encode(manifest).decode()}
print(json.dumps({
    "releases_url": base, "installed_version": "0.0.6",
    "attestant_fingerprint": "9f2103792a8007816eb963d1b284d139ac3cf69f3d6ab7cb93878bb654d70915",
    "ci_identity_pattern": "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v*",
    "good": staged("0.0.9", json.load(open(f"{t}/update-row.json"))),
    "broken": staged(broken, json.load(open(f"{t}/broken-row.json"))),
}, indent=1))
PY
UPDATE_DIGEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["digest"][7:])' "$T/update-row.json")"
echo "update digest: $UPDATE_DIGEST" | tee "$OUT/update.txt"

wait_for() { # wait_for SECONDS COMMAND...: until COMMAND succeeds
  local n="$1"; shift
  for _ in $(seq "$n"); do "$@" && return 0; sleep 1; done
  return 1
}

if [[ -n "$DEVICE" ]]; then
  "$HERE/ios/build.sh" --device --bundle "$T/bundle" | tail -2
  APP="$HERE/ios/build/DerivedData/Build/Products/Debug-iphoneos/EidolaShell.app"
  PKG=ai.eidola.mobile.dev
  # devicectl takes its options before an app's bundle id or path.
  dev() { local verb=("$1" "$2" "$3"); shift 3; timeout 120 xcrun devicectl "${verb[@]}" --device "$DEVICE" -q "$@"; }
  dev device install app "$APP"
  # The page's console, which the debug shell writes to the system log.
  idevicesyslog -u "$DEVICE" -p EidolaShell >"$OUT/console.log" 2>&1 &
  PIDS+=($!)
  FILES="Library/Application Support/Eidola/files"
  report() { # an empty report is an earlier run's, blanked below
    rm -f "$T/report.json"
    dev device copy from --domain-type appDataContainer --domain-identifier "$PKG" \
      --source "$FILES/selftest-$1.json" --destination "$T/report.json" >/dev/null 2>&1 &&
      [[ -s "$T/report.json" ]] && cat "$T/report.json"
  }
  sheet() { :; } # opened from a home-screen quick action, which only XCUITest reaches
  launch() { dev device process launch --terminate-existing "$PKG" >/dev/null; }
  kill_app() { # the shell's process, found by its executable
    local pid
    pid="$(dev device info processes -j - |
      python3 -c 'import json,sys; print(" ".join(str(p["processIdentifier"]) for p in json.load(sys.stdin)["result"]["runningProcesses"] if p.get("executable","").endswith("/EidolaShell.app/EidolaShell")))')"
    for p in $pid; do dev device process terminate --kill --pid "$p" >/dev/null; done
  }
  shot() { dev device capture screenshot --destination "$OUT/$1.png" >/dev/null 2>&1 || true; }
  files() { dev device info files --domain-type appDataContainer --domain-identifier "$PKG" --subdirectory "$FILES"; }
  # A fresh start without reinstalling: no state (the next launch is a first one), no reports.
  : >"$T/empty"
  for f in selftest-state.json selftest-1.json selftest-2.json selftest-3.json; do
    dev device copy to --domain-type appDataContainer --domain-identifier "$PKG" \
      --source "$T/empty" --destination "$FILES/$f" >/dev/null 2>&1 || true # none yet on a new install
  done
elif [[ $PLATFORM == ios ]]; then
  # DEVICE_TYPE, RUNTIME: see ios/simulator.sh (default the newest installed iOS runtime).
  source "$HERE/ios/simulator.sh"
  UDID="$(simulator_udid eidola-bridge)"
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
  sheet() { :; } # opened from a home-screen quick action, which only XCUITest reaches
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
  sheet() { # the bundle sheet, through the intent the launcher shortcut sends; prints its text
    "${ADB[@]}" shell am start -a ai.eidola.shell.SHOW_DIGEST -n "$PKG/ai.eidola.shell.MainActivity" >/dev/null
    sleep 3
    shot "$1"
    "${ADB[@]}" shell uiautomator dump /sdcard/ui.xml >/dev/null
    "${ADB[@]}" shell cat /sdcard/ui.xml | tr -d '\n' | sed 's/&#10;//g'
    "${ADB[@]}" shell input keyevent KEYCODE_BACK
  }
  launch() { "${ADB[@]}" shell am start -W -n "$PKG/ai.eidola.shell.MainActivity" >/dev/null; }
  kill_app() { "${ADB[@]}" shell am force-stop "$PKG"; }
  shot() { "${ADB[@]}" exec-out screencap -p >"$OUT/$1.png"; }
  files() { "${ADB[@]}" shell run-as "$PKG" ls -l no_backup/files; }
fi

# 3. Phase 1, a kill, phase 2.
launch
wait_for 300 report 1 >/dev/null || { echo "phase 1 reported nothing" >&2; }
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
# The update is serving: the shell's own sheet shows its digest.
sheet 2-bundle-sheet >"$OUT/sheet.txt" || true
kill_app
sleep 1
# The broken update's trial: it never confirms.
launch
sleep 8
shot 3-broken-trial
kill_app
sleep 1
# The shell falls back to the confirmed update.
launch
wait_for 120 report 3 >/dev/null || { echo "phase 3 reported nothing" >&2; }
sleep 1
shot 4-after-fallback
report 3 >"$OUT/selftest-3.json" || true
[[ $PLATFORM == android ]] && "${ADB[@]}" logcat -d -s EidolaShell:V >"$OUT/console.log"
kill_app || true

python3 - "$OUT" "$PLATFORM" "$UPDATE_DIGEST" <<'PY'
import json, sys
out, platform, digest = sys.argv[1:]
ok = True
if platform == "android":
    shown = open(f"{out}/sheet.txt").read()
    sheet = digest in shown and "on trial" not in shown
    print(f"sheet: {'PASS' if sheet else 'FAIL'} the native bundle sheet shows the update's digest {digest}")
    ok = ok and sheet
else:
    print("sheet: not checked here (iOS opens it from a home-screen quick action; mobile/ios/test-sim.sh)")
for phase in (1, 2, 3):
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
