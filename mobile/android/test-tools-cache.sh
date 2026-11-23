#!/usr/bin/env bash
# The pinned-tools cache cannot hand out a byte that was not checked in the current run
# (tools/fetch.sh). Drives fetch over a small fixture archive served from a file: URL, so it
# needs no network:
#   * a fresh fetch unpacks the archive that hashed to its pin;
#   * a later run re-unpacks: a tampered unpacked tree from an earlier run is replaced;
#   * a tampered cached archive is refused and fetched again from the source;
#   * a source that serves the tampered bytes too fails the fetch, and nothing is unpacked;
#   * within one run a tree is unpacked once.
#   test-tools-cache.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# A fixture archive with one "tool" in it, and its pin.
python3 -I - "$T/genuine.zip" "genuine tool" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr("bin/tool", sys.argv[2])
PY
python3 -I - "$T/tampered.zip" "tampered tool" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr("bin/tool", sys.argv[2])
PY
cp "$T/genuine.zip" "$T/served.zip"

run() {
  # One run of the build: a fresh process with its own run id.
  env -u EIDOLA_ANDROID_TOOLS_RUN ANDROID_TOOLS="$T/cache" SHA="$SHA" URL="file://$T/served.zip" \
    bash -c 'set -euo pipefail; source "$1"; d="$(fetch "$URL" "$SHA")"; cat "$d/bin/tool"' _ "$HERE/tools/fetch.sh"
}
source "$HERE/tools/fetch.sh"
SHA="$(sha256 "$T/genuine.zip" | cut -d' ' -f1)"
UNPACKED="$T/cache/unpacked/$SHA/bin/tool"
ARCHIVE="$T/cache/archives/$SHA/archive.zip"

[[ "$(run 2>/dev/null)" == "genuine tool" ]] || fail "a fresh fetch did not unpack the pinned archive"
echo "ok: a fresh fetch unpacks the pinned archive"

printf 'tampered tool' >"$UNPACKED"
[[ "$(run 2>/dev/null)" == "genuine tool" ]] || fail "an unpacked tree from an earlier run was used"
echo "ok: a tampered unpacked tree from an earlier run is replaced"

cp "$T/tampered.zip" "$ARCHIVE"
out="$(run 2>"$T/err")" || fail "a tampered cached archive was not fetched again"
[[ "$out" == "genuine tool" ]] || fail "a tampered cached archive was unpacked"
grep -q "does not hash to" "$T/err" || fail "a tampered cached archive was replaced without saying so"
echo "ok: a tampered cached archive is refused and fetched again"

cp "$T/tampered.zip" "$ARCHIVE"
cp "$T/tampered.zip" "$T/served.zip"
rm -rf "$T/cache/unpacked"
if run >/dev/null 2>"$T/err"; then fail "a source serving the wrong bytes was accepted"; fi
grep -q "expected $SHA" "$T/err" || fail "the fetch failed, but not on the hash"
[[ ! -e "$UNPACKED" ]] || fail "something was unpacked from bytes that failed their pin"
echo "ok: a source serving the wrong bytes fails the fetch, and nothing is unpacked"

cp "$T/genuine.zip" "$T/served.zip"
ANDROID_TOOLS="$T/cache" EIDOLA_ANDROID_TOOLS_RUN=one-run bash -c \
  'set -euo pipefail; source "$1"; fetch "file://$2" "$3" >/dev/null; touch "$4.marker"; fetch "file://$2" "$3" >/dev/null' \
  _ "$HERE/tools/fetch.sh" "$T/served.zip" "$SHA" "$UNPACKED" 2>/dev/null
[[ -e "$UNPACKED.marker" ]] || fail "one run unpacked the same tree twice"
echo "ok: within one run a tree is unpacked once"
echo "the pinned-tools cache holds"
