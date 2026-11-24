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
# STUB_PORT, UPDATE_PORT  the model stub's and the update fixtures' ports on this machine
#                         (default 8790, 8791), so two runs on one machine stay apart
# Uses (and creates once per runtime) its own simulators, "eidola-phone" and "eidola-phone-zh" (its
# language Chinese, so Pinyin is its first keyboard), or the pair UI_SIMULATOR names; never another
# device.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-}"
OUT="$(mkdir -p "${OUT:-$HERE/build/ui}" && cd "${OUT:-$HERE/build/ui}" && pwd)"
# DEVICE_TYPE, RUNTIME: see simulator.sh (default the newest installed iOS runtime).
source "$HERE/simulator.sh"
PORT="${STUB_PORT:-8790}"
UPDATE_PORT="${UPDATE_PORT:-8791}"

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
xcrun simctl uninstall "$UDID" ai.eidola.app.ios.dev || true
xcrun simctl uninstall "$ZH" ai.eidola.app.ios.dev || true

python3 "$HERE/../tools/openai_stub.py" --port "$PORT" --delay 0.08 >"$OUT/stub.log" 2>&1 &
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

"$HERE/../tools/account_stack.sh" up "$STACK"
python3 - "$STACK/eidola.json" >"$STACK/eidola-ios.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["base_url"] = "https://127.0.0.1:%d" % d.pop("port")
print(json.dumps(d))
PY
BUNDLE="$HERE/../build/bundle-dev"
"$HERE/../build-bundle.sh" --dev-backend "http://127.0.0.1:$PORT" \
  --dev-update "http://127.0.0.1:$UPDATE_PORT/update.json" --dev-eidola "$STACK/eidola-ios.json" \
  --out "$BUNDLE" >&2
python3 "$HERE/../tools/update_fixtures.py" "$BUNDLE" "$HERE/build/update-fixtures" --port "$UPDATE_PORT" \
  --base "http://127.0.0.1:$UPDATE_PORT" >"$OUT/fixtures.log" 2>&1 &
FIXTURES=$!
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
run_test "$UDID" testOnboardingAccount
FIRST=$?
# The account secret, copied while masked on screen: the clipboard holds the secret itself.
if [[ $FIRST -eq 0 ]]; then
  WANT="$(curl -fsS http://127.0.0.1:18081/_stub/secret-sha256 | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha256"])')"
  GOT="$(xcrun simctl pbpaste "$UDID" | python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
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
xcrun simctl ui "$UDID" appearance dark
run_test "$UDID" testRelaunchAtNight
NIGHT=$?
xcrun simctl ui "$UDID" appearance light
run_test "$UDID" testUpdate
UPDATE=$?
kill -INT "$VIDEO" 2>/dev/null && wait "$VIDEO" 2>/dev/null
VIDEO=
xcrun simctl io "$ZH" recordVideo --codec h264 --force "$OUT/chinese-ime.mp4" >/dev/null 2>&1 &
VIDEO=$!
run_test "$ZH" testChineseInput
IME=$?
set -e
echo "screenshots and recordings: $OUT"
echo "results: first run $FIRST, conversation $STATUS, settings $SETTINGS, night $NIGHT, update $UPDATE, Chinese $IME"
for s in $FIRST $STATUS $SETTINGS $NIGHT $UPDATE $IME; do [[ $s == 0 ]] || exit 1; done
