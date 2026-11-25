#!/usr/bin/env bash
# The phone interface's UI tests (UITests/PhoneUITests.swift) in the iOS Simulator: a development
# bundle whose default agent answers from tools/openai_stub.py on this machine, which trusts the
# local dev stack's gateway behind its attesting shim (tools/account_stack.sh: the real gateway,
# a stand-in for the payment processor at its edge) and takes its update from a fixture release
# (tools/update_fixtures.py), driven by real taps and typing:
#   - the first run on a fresh install: an account created under the terms the consent page
#     rendered, its credentials, the plans, a checkout handed to the system browser, the balance
#     read again on coming back;
#   - a whole conversation (a new conversation, the keyboard, a streamed answer with its
#     reasoning, markdown, a reply branch, scrolling, the Library, an untouched one not kept);
#   - settings applied live (the night palette, the text size), the wallet, About, the licences;
#   - a cold relaunch at night;
#   - the update: approved, restarted into and confirmed; a broken one falling back; the way
#     back to the built-in bundle;
#   - Chinese through the Pinyin IME and the first run in the app's zh-Hans locale.
# Screenshots and screen recordings go to OUT (default build/ui). Needs docker (the database).
#   test-ui.sh [OUT]
#   test-ui.sh [OUT] --device UDID [--host ADDRESS]
# --device runs them on that physical iOS device instead (mobile/device-checklist.md), signed by the
# project's team (IOS_TEAM another) with the UI-test runner, everything it reaches served on the LAN
# at ADDRESS (default this Mac's en0 address). It keeps the app installed (and its local network
# grant; the tests allow the prompt when a new install raises it) and starts the profile afresh by
# blanking its files; the device's own language decides the keyboard, so the Pinyin test is
# Simulator-only.
# STUB_PORT, UPDATE_PORT  the model stub's and the update fixtures' ports on this machine
#                         (default 8790, 8791), so two runs on one machine stay apart
# Uses (and creates once per runtime) its own simulators, "eidola-phone" and "eidola-phone-zh" (its
# language Chinese, so Pinyin is its first keyboard), or the pair UI_SIMULATOR names; never another
# device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=
DEVICE=
HOST=
while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    -*) echo "unknown argument $1" >&2; exit 2 ;;
    *) OUT="$1"; shift ;;
  esac
done
OUT="$(mkdir -p "${OUT:-$HERE/build/ui}" && cd "${OUT:-$HERE/build/ui}" && pwd)"
PORT="${STUB_PORT:-8790}"
UPDATE_PORT="${UPDATE_PORT:-8791}"
PKG=ai.eidola.app.ios.dev

if [[ -n "$DEVICE" ]]; then
  HOST="${HOST:-$(ipconfig getifaddr en0)}"
  [[ -n "$HOST" ]] || { echo "no LAN address (pass --host)" >&2; exit 2; }
  BIND=0.0.0.0
  dev() { local verb=("$1" "$2" "$3"); shift 3; timeout 120 xcrun devicectl "${verb[@]}" --device "$DEVICE" -q "$@"; }
  appearance() { dev device settings appearance --mode "$2"; }
  pasteboard() { dev device pasteboard paste; }
  record() { # stopped by an interrupt, like simctl's
    xcrun devicectl device capture screen-record --device "$DEVICE" -q --destination "$2" >/dev/null 2>&1 &
    VIDEO=$!
  }
  # A fresh profile without reinstalling: the profile's files blanked (a new database, the default
  # configuration); none yet on a new install.
  : >"$OUT/.empty"
  for f in config.toml eidola.db eidola.db-wal; do
    dev device copy to --domain-type appDataContainer --domain-identifier "$PKG" --source "$OUT/.empty" \
      --destination "Library/Application Support/Eidola/files/$f" >/dev/null 2>&1 || true
  done
  rm -f "$OUT/.empty"
  UDID="$DEVICE"
  echo "device: $DEVICE, serving on $HOST"
else
  HOST=127.0.0.1
  BIND=127.0.0.1
  # DEVICE_TYPE, RUNTIME: see simulator.sh (default the newest installed iOS runtime).
  source "$HERE/simulator.sh"
  simulator() { # simulator NAME: its udid, created and booted
    local udid
    udid="$(simulator_udid "$1")"
    xcrun simctl boot "$udid" 2>/dev/null || true
    xcrun simctl bootstatus "$udid" -b >/dev/null
    # The -zh simulator's language is Chinese, so Pinyin is its first keyboard; set once, and it
    # takes effect from the next boot.
    local languages
    languages="$(xcrun simctl spawn "$udid" defaults read -g AppleLanguages 2>/dev/null || true)"
    if [[ "$1" == *-zh && "$languages" != *zh-Hans* ]]; then
      xcrun simctl spawn "$udid" defaults write -g AppleLanguages -array zh-Hans-CN en-US
      xcrun simctl spawn "$udid" defaults write -g AppleLocale -string zh_CN
      xcrun simctl spawn "$udid" defaults write -g AppleKeyboards -array \
        "zh_Hans-Pinyin@sw=Pinyin;hw=Automatic" "en_US@sw=QWERTY;hw=Automatic" "emoji@sw=Emoji"
      xcrun simctl shutdown "$udid" >&2
      xcrun simctl boot "$udid" >&2
      xcrun simctl bootstatus "$udid" -b >/dev/null
    fi
    echo "$udid"
  }
  # UI_SIMULATOR names another pair (NAME and NAME-zh), for a second checkout running at once.
  SIM="${UI_SIMULATOR:-eidola-phone}"
  UDID="$(simulator "$SIM")"
  ZH="$(simulator "$SIM-zh")"
  echo "simulators: $SIM $UDID, $SIM-zh $ZH"
  xcrun simctl ui "$UDID" appearance light
  xcrun simctl ui "$ZH" appearance light
  # Fresh installs: no earlier run's conversations.
  xcrun simctl uninstall "$UDID" "$PKG" || true
  xcrun simctl uninstall "$ZH" "$PKG" || true

  appearance() { xcrun simctl ui "$1" appearance "$2"; }
  pasteboard() { xcrun simctl pbpaste "$UDID"; }
  record() { xcrun simctl io "$1" recordVideo --codec h264 --force "$2" >/dev/null 2>&1 & VIDEO=$!; }
fi

python3 "$HERE/../tools/openai_stub.py" --port "$PORT" --delay 0.08 --bind "$BIND" >"$OUT/stub.log" 2>&1 &
STUB=$!
FIXTURES=
VIDEO=
STACK="$HERE/build/account-stack"
cleanup() {
  [[ -n "$VIDEO" ]] && kill -INT "$VIDEO" 2>/dev/null && wait "$VIDEO" 2>/dev/null || true
  kill "$STUB" 2>/dev/null || true
  [[ -n "$FIXTURES" ]] && kill "$FIXTURES" 2>/dev/null || true
  "$HERE/../tools/account_stack.sh" down "$STACK" || true
}
trap cleanup EXIT

ACCOUNT_STACK_HOST="${DEVICE:+$HOST}" "$HERE/../tools/account_stack.sh" up "$STACK"
python3 - "$STACK/eidola.json" "$HOST" >"$STACK/eidola-ios.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["base_url"] = "https://%s:%d" % (sys.argv[2], d.pop("port"))
print(json.dumps(d))
PY
BUNDLE="$HERE/../build/bundle-dev"
"$HERE/../build-bundle.sh" --dev-backend "http://$HOST:$PORT" \
  --dev-update "http://$HOST:$UPDATE_PORT/update.json" --dev-eidola "$STACK/eidola-ios.json" \
  --out "$BUNDLE" >&2
# The broken update's version is this run's own (tools/update_fixtures.py: a device keeps its install).
python3 "$HERE/../tools/update_fixtures.py" "$BUNDLE" "$HERE/build/update-fixtures" --port "$UPDATE_PORT" \
  --base "http://$HOST:$UPDATE_PORT" --bind "$BIND" --broken-version "0.0.$(date +%s)" >"$OUT/fixtures.log" 2>&1 &
FIXTURES=$!
if [[ -n "$DEVICE" ]]; then
  "$HERE/build.sh" --device --bundle "$BUNDLE"
  SIGNING=(-allowProvisioningUpdates ${IOS_TEAM:+"DEVELOPMENT_TEAM=$IOS_TEAM"})
else
  "$HERE/build.sh" --bundle "$BUNDLE"
  SIGNING=(CODE_SIGNING_ALLOWED=NO)
fi

run_test() { # run_test DESTINATION METHOD
  rm -rf "$HERE/build/ui-$2.xcresult"
  TEST_RUNNER_SHOTS="$OUT" xcodebuild -project "$HERE/EidolaShell.xcodeproj" -scheme EidolaShell \
    -configuration Debug -derivedDataPath "$HERE/build/DerivedData" -destination "id=$1" \
    -resultBundlePath "$HERE/build/ui-$2.xcresult" -only-testing:"EidolaShellUITests/PhoneUITests/$2" \
    "${SIGNING[@]}" test 2>&1 | grep -E "Test Case|error:|\*\* TEST|failed"
  return "${PIPESTATUS[0]}"
}
rm -f "$OUT/conversation.mp4"
record "$UDID" "$OUT/conversation.mp4"
set +e
run_test "$UDID" testOnboardingAccount
FIRST=$?
# The account secret, copied while masked on screen: the clipboard holds the secret itself.
if [[ $FIRST -eq 0 ]]; then
  WANT="$(curl -fsS http://127.0.0.1:18081/_stub/secret-sha256 | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha256"])')"
  GOT="$(pasteboard | python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
  if [[ -n "$WANT" && "$WANT" == "$GOT" ]]; then
    echo "clipboard: the masked secret's Copy copied the secret"
  else
    echo "clipboard: the masked secret's Copy did not copy the secret" >&2
    FIRST=1
  fi
fi
run_test "$UDID" testConversation
STATUS=$?
run_test "$UDID" testSettings
SETTINGS=$?
# The same profile after a cold launch, with the device switched to dark.
appearance "$UDID" dark
run_test "$UDID" testRelaunchAtNight
NIGHT=$?
appearance "$UDID" light
run_test "$UDID" testUpdate
UPDATE=$?
kill -INT "$VIDEO" 2>/dev/null && wait "$VIDEO" 2>/dev/null
VIDEO=
if [[ -n "$DEVICE" ]]; then
  IME=0
  echo "Chinese: not run on a device (the Simulator's own)"
else
  record "$ZH" "$OUT/chinese-ime.mp4"
  run_test "$ZH" testChineseInput
  IME=$?
fi
set -e
echo "screenshots and recordings: $OUT"
echo "results: first run $FIRST, conversation $STATUS, settings $SETTINGS, night $NIGHT, update $UPDATE, Chinese $IME"
for s in $FIRST $STATUS $SETTINGS $NIGHT $UPDATE $IME; do [[ $s == 0 ]] || exit 1; done
