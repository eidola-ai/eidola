# The iOS Simulator devices the mobile scripts run on, sourced by ios/test-sim.sh, ios/test-ui.sh
# and self-test.sh. Each script owns its devices by name and never touches another.
#   DEVICE_TYPE  the device type a new simulator is created as (default an iPhone 17 Pro)
#   RUNTIME      the runtime it runs (default the newest installed iOS runtime that supports
#                DEVICE_TYPE), so the scripts follow whatever runtimes Xcode has installed
DEVICE_TYPE="${DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro}"

# The newest available iOS runtime that can run DEVICE_TYPE (by version, then the broadest one:
# Xcode also installs device-specific runtimes of the same version).
newest_ios_runtime() {
  xcrun simctl list runtimes -j | python3 -c '
import json, sys
kind = sys.argv[1]
runtimes = [
    r for r in json.load(sys.stdin)["runtimes"]
    if r.get("platform") == "iOS" and r["isAvailable"]
    and any(d["identifier"] == kind for d in r.get("supportedDeviceTypes", []))
]
if not runtimes:
    sys.exit(f"no installed iOS runtime runs {kind} (Xcode > Settings > Components)")
best = max(runtimes, key=lambda r: (tuple(int(p) for p in r["version"].split(".")), len(r["supportedDeviceTypes"])))
print(best["identifier"])' "$DEVICE_TYPE"
}
RUNTIME="${RUNTIME:-$(newest_ios_runtime)}"

# simulator_udid NAME: the simulator of that name on RUNTIME, created if there is none. A device
# of that name on another runtime (or one whose runtime was removed) is deleted first: the name
# is this script's, and its devices hold nothing but earlier runs' installs.
simulator_udid() {
  local found
  found="$(xcrun simctl list devices -j | python3 -c '
import json, sys
name, runtime = sys.argv[1:]
for rt, devices in json.load(sys.stdin)["devices"].items():
    for d in devices:
        if d["name"] == name:
            print(("keep" if rt == runtime and d["isAvailable"] else "stale"), d["udid"])' "$1" "$RUNTIME")"
  local verdict udid keep=
  while read -r verdict udid; do
    [[ -z "$udid" ]] && continue
    if [[ $verdict == keep && -z "$keep" ]]; then keep="$udid"; else xcrun simctl delete "$udid" >&2; fi
  done <<<"$found"
  if [[ -z "$keep" ]]; then
    keep="$(xcrun simctl create "$1" "$DEVICE_TYPE" "$RUNTIME")"
    echo "created simulator $1 ($keep) on $RUNTIME" >&2
  fi
  echo "$keep"
}
