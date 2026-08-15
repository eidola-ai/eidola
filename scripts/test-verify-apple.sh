#!/bin/sh
# Exercise the public two-archive verifier with Nix-style read-only directories.
set -eu

command -v zip >/dev/null 2>&1 || {
  echo "test-verify-apple requires zip" >&2
  exit 2
}
command -v python3 >/dev/null 2>&1 || {
  echo "test-verify-apple requires python3" >&2
  exit 2
}

fixture=scripts/fixtures/apple-roundtrip/synthetic-universal
test_root=$(mktemp -d "${TMPDIR:-/tmp}/eidola-apple-archive-test.XXXXXX")
cleanup() {
  find "$test_root" -type d -exec chmod u+w {} + 2>/dev/null || true
  rm -rf "$test_root"
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$test_root/unsigned" "$test_root/detached"

run_verify_bounded() {
  python3 -c '
import os
import subprocess
import sys

environment = os.environ.copy()
environment["EIDOLA_VERIFY_TMPDIR"] = sys.argv[4]
with open(sys.argv[3], "wb") as output:
    try:
        result = subprocess.run(
            ["./scripts/verify-apple.sh", sys.argv[1], sys.argv[2]],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=10,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        sys.exit(124)
sys.exit(result.returncode)
' "$1" "$2" "$3" "$test_root/verifier-tmp"
}

assert_verify_refused() {
  unsigned_zip=$1
  detached_zip=$2
  log=$3
  expected=$4
  verify_status=0
  run_verify_bounded "$unsigned_zip" "$detached_zip" "$log" || verify_status=$?
  if [ "$verify_status" -eq 0 ]; then
    echo "verify-apple accepted an invalid archive root" >&2
    exit 1
  fi
  if [ "$verify_status" -eq 124 ]; then
    echo "verify-apple did not refuse an invalid archive within 10 seconds" >&2
    exit 1
  fi
  if grep -q '^replace .*?' "$log"; then
    echo "verify-apple reached an interactive archive overwrite prompt" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -q "$expected" "$log"; then
    echo "verify-apple failed for the wrong archive-root reason" >&2
    cat "$log" >&2
    exit 1
  fi
  if find "$test_root/verifier-tmp" -mindepth 1 -print -quit | grep -q .; then
    echo "failed verify-apple left its temporary extraction tree behind" >&2
    exit 1
  fi
}

cp -R "$fixture/settled/Fixture.app" "$test_root/unsigned/"
cp -R "$fixture/detached/Fixture.app" "$test_root/detached/"
cp "$fixture/detached/eidola-placement.json" "$test_root/detached/"
find "$test_root/unsigned/Fixture.app" -type d -exec chmod 0555 {} +
find "$test_root/detached/Fixture.app" -type d -exec chmod 0555 {} +

(cd "$test_root/unsigned" && zip -qry "$test_root/unsigned.zip" Fixture.app)
(cd "$test_root/detached" && zip -qry "$test_root/detached.zip" Fixture.app eidola-placement.json)
python3 -c '
import shutil
import sys
import warnings
import zipfile
import copy
import json

unsigned, detached = sys.argv[1:3]
cases = [
    (unsigned, sys.argv[3], "Fixture.app/Contents/Info.plist", None),
    (detached, sys.argv[4], "eidola-placement.json", None),
    (detached, sys.argv[5], "Fixture.app/Contents/MacOS", b"collision"),
    (detached, sys.argv[6], "./eidola-placement.json", None),
]
for source, destination, member, replacement in cases:
    shutil.copyfile(source, destination)
    with zipfile.ZipFile(destination, "a") as archive:
        payload = archive.read(member.removeprefix("./")) if replacement is None else replacement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr(member, payload)

shutil.copyfile(unsigned, sys.argv[7])
with zipfile.ZipFile(sys.argv[7], "a") as archive:
    archive.writestr(
        "fixture.app/Contents/Info.plist",
        archive.read("Fixture.app/Contents/Info.plist"),
    )

composed = "Fixtur\u00e9.app"
decomposed = "Fixture\u0301.app"
for source, destination, bundle in [
    (unsigned, sys.argv[8], composed),
    (detached, sys.argv[9], composed),
]:
    with zipfile.ZipFile(source) as input_archive, zipfile.ZipFile(destination, "w") as output_archive:
        for original in input_archive.infolist():
            member = copy.copy(original)
            member.filename = member.filename.replace("Fixture.app", bundle, 1)
            member.orig_filename = member.filename
            payload = input_archive.read(original)
            if member.filename == "eidola-placement.json":
                record = json.loads(payload)
                record["bundle"] = bundle
                payload = json.dumps(record, indent=2).encode() + b"\n"
            output_archive.writestr(member, payload)
with zipfile.ZipFile(sys.argv[8], "a") as archive:
    archive.writestr(
        f"{decomposed}/Contents/Info.plist",
        archive.read(f"{composed}/Contents/Info.plist"),
    )
' \
  "$test_root/unsigned.zip" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-duplicate.zip" \
  "$test_root/detached-duplicate.zip" \
  "$test_root/detached-directory-file.zip" \
  "$test_root/detached-dot-alias.zip" \
  "$test_root/unsigned-case-alias.zip" \
  "$test_root/unsigned-unicode-alias.zip" \
  "$test_root/detached-unicode.zip"
filesystem_equivalence=$(python3 -c '
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "filesystem-equivalence"
root.mkdir()

def aliases(directory, first, second):
    probe = root / directory
    probe.mkdir()
    (probe / first).mkdir()
    try:
        (probe / second).mkdir()
    except FileExistsError:
        return "1"
    return "0"

print(
    aliases("case", "CaseProbe", "caseprobe")
    + ":"
    + aliases("unicode", "\u00e9", "e\u0301")
)
' "$test_root")
case_equivalent=${filesystem_equivalence%%:*}
unicode_equivalent=${filesystem_equivalence##*:}
mkdir "$test_root/probe"
unzip -q "$test_root/unsigned.zip" -d "$test_root/probe"
if mkdir "$test_root/probe/Fixture.app/Contents/write-probe" 2>/dev/null; then
  echo "unsigned archive did not preserve the read-only Contents directory" >&2
  exit 1
fi
mkdir "$test_root/detached-probe"
unzip -q "$test_root/detached.zip" -d "$test_root/detached-probe"
if mkdir "$test_root/detached-probe/Fixture.app/Contents/write-probe" 2>/dev/null; then
  echo "detached archive did not preserve the read-only Contents directory" >&2
  exit 1
fi

mkdir "$test_root/verifier-tmp"
EIDOLA_VERIFY_TMPDIR="$test_root/verifier-tmp" \
  ./scripts/verify-apple.sh "$test_root/unsigned.zip" "$test_root/detached.zip"
if find "$test_root/verifier-tmp" -mindepth 1 -print -quit | grep -q .; then
  echo "verify-apple left its temporary extraction tree behind" >&2
  exit 1
fi

assert_verify_refused \
  "$test_root/unsigned-duplicate.zip" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-duplicate.log" \
  'duplicate or colliding archive member'
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-duplicate.zip" \
  "$test_root/detached-duplicate.log" \
  'duplicate or colliding archive member'
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-directory-file.zip" \
  "$test_root/detached-directory-file.log" \
  'duplicate or colliding archive member'
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-dot-alias.zip" \
  "$test_root/detached-dot-alias.log" \
  'duplicate or colliding archive member'
if [ "$case_equivalent" -eq 1 ]; then
  assert_verify_refused \
    "$test_root/unsigned-case-alias.zip" \
    "$test_root/detached.zip" \
    "$test_root/unsigned-case-alias.log" \
    'duplicate or colliding archive member'
else
  assert_verify_refused \
    "$test_root/unsigned-case-alias.zip" \
    "$test_root/detached.zip" \
    "$test_root/unsigned-case-alias.log" \
    'unsigned zip root'
fi
if [ "$unicode_equivalent" -eq 1 ]; then
  assert_verify_refused \
    "$test_root/unsigned-unicode-alias.zip" \
    "$test_root/detached-unicode.zip" \
    "$test_root/unsigned-unicode-alias.log" \
    'duplicate or colliding archive member'
else
  assert_verify_refused \
    "$test_root/unsigned-unicode-alias.zip" \
    "$test_root/detached-unicode.zip" \
    "$test_root/unsigned-unicode-alias.log" \
    'unsigned zip root'
fi

printf 'stale\n' >"$test_root/detached/stale.sign"
(cd "$test_root/detached" && zip -qry "$test_root/detached-extra.zip" Fixture.app eidola-placement.json stale.sign)
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-extra.zip" \
  "$test_root/unexpected-root.log" \
  'stale.sign'

mkdir -p "$test_root/detached-wrapped/wrapper"
cp -R "$fixture/detached/Fixture.app" "$test_root/detached-wrapped/wrapper/"
cp "$fixture/detached/eidola-placement.json" "$test_root/detached-wrapped/wrapper/"
printf 'ignored\n' >"$test_root/detached-wrapped/extra.txt"
(cd "$test_root/detached-wrapped" && zip -qry "$test_root/detached-wrapped-file.zip" wrapper extra.txt)
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-wrapped-file.zip" \
  "$test_root/detached-wrapped-file.log" \
  'signature-bundle zip root'

rm "$test_root/detached-wrapped/extra.txt"
ln -s wrapper/Fixture.app "$test_root/detached-wrapped/extra-link"
(cd "$test_root/detached-wrapped" && zip -qry -y "$test_root/detached-wrapped-link.zip" wrapper extra-link)
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-wrapped-link.zip" \
  "$test_root/detached-wrapped-link.log" \
  'signature-bundle zip root'

mkdir -p "$test_root/unsigned-wrapped/wrapper"
cp -R "$fixture/settled/Fixture.app" "$test_root/unsigned-wrapped/wrapper/"
printf 'ignored\n' >"$test_root/unsigned-wrapped/extra.txt"
(cd "$test_root/unsigned-wrapped" && zip -qry "$test_root/unsigned-wrapped.zip" wrapper extra.txt)
assert_verify_refused \
  "$test_root/unsigned-wrapped.zip" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-wrapped.log" \
  'unsigned zip root'
