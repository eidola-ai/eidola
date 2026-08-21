#!/bin/sh
# Exercise the public two-archive verifier over both containers it reads: the
# canonical gzip'd POSIX tar the flake publishes as the unsigned macOS
# archive, and zip. Includes Nix-style read-only directories.
set -eu

command -v zip >/dev/null 2>&1 || {
  echo "test-verify-apple requires zip" >&2
  exit 2
}
command -v python3 >/dev/null 2>&1 || {
  echo "test-verify-apple requires python3" >&2
  exit 2
}
command -v tar >/dev/null 2>&1 || {
  echo "test-verify-apple requires tar" >&2
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
    'unsigned archive root'
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
    'unsigned archive root'
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
  'signature-bundle archive root'

rm "$test_root/detached-wrapped/extra.txt"
ln -s wrapper/Fixture.app "$test_root/detached-wrapped/extra-link"
(cd "$test_root/detached-wrapped" && zip -qry -y "$test_root/detached-wrapped-link.zip" wrapper extra-link)
assert_verify_refused \
  "$test_root/unsigned.zip" \
  "$test_root/detached-wrapped-link.zip" \
  "$test_root/detached-wrapped-link.log" \
  'signature-bundle archive root'

mkdir -p "$test_root/unsigned-wrapped/wrapper"
cp -R "$fixture/settled/Fixture.app" "$test_root/unsigned-wrapped/wrapper/"
printf 'ignored\n' >"$test_root/unsigned-wrapped/extra.txt"
(cd "$test_root/unsigned-wrapped" && zip -qry "$test_root/unsigned-wrapped.zip" wrapper extra.txt)
assert_verify_refused \
  "$test_root/unsigned-wrapped.zip" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-wrapped.log" \
  'unsigned archive root'

# --- Canonical gzip'd POSIX tar ---------------------------------------------
# The unsigned macOS archive artifact-manifest.json binds through
# archiveSha256 is `nix build .#eidola-gui-macos-universal-archive` — a
# gzip'd POSIX tar of the payload directory packed as `.`, not a zip. These
# archives are built to that shape (a bare `./` root member, `./`-prefixed
# names, mode u=rwX,go=rX, mtime 0, numeric owner 0) rather than by shelling
# out to tar, because bsdtar has no --sort/--mtime/--mode. They are packed
# from the fixture roots rather than the staging copies above, which the zip
# cases mutate.
python3 -c '
import copy
import io
import os
import sys
import tarfile

test_root, unsigned_root, detached_root = sys.argv[1:4]


def members(root, readonly):
    """(TarInfo, payload) pairs in the flake archive'"'"'s shape."""
    items = []

    def info(name, mode, kind):
        entry = tarfile.TarInfo(name)
        entry.mode = mode
        entry.type = kind
        entry.mtime = 0
        entry.uid = entry.gid = 0
        entry.uname = entry.gname = ""
        return entry

    items.append((info("./", 0o555 if readonly else 0o755, tarfile.DIRTYPE), None))
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        files.sort()
        relative = os.path.relpath(directory, root)
        prefix = "./" if relative == "." else "./" + relative + "/"
        for name in subdirectories:
            items.append(
                (info(prefix + name + "/", 0o555 if readonly else 0o755, tarfile.DIRTYPE), None)
            )
        for name in files:
            with open(os.path.join(directory, name), "rb") as handle:
                payload = handle.read()
            entry = info(prefix + name, 0o444 if readonly else 0o644, tarfile.REGTYPE)
            entry.size = len(payload)
            items.append((entry, payload))
    return items


def regular(name, payload):
    entry = tarfile.TarInfo(name)
    entry.mode = 0o644
    entry.type = tarfile.REGTYPE
    entry.mtime = 0
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = ""
    entry.size = len(payload)
    return (entry, payload)


def write(path, items):
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for entry, payload in items:
            archive.addfile(entry, io.BytesIO(payload) if payload is not None else None)


unsigned = members(unsigned_root, False)
write(os.path.join(test_root, "unsigned.tar.gz"), unsigned)
write(os.path.join(test_root, "detached.tar.gz"), members(detached_root, True))
write(os.path.join(test_root, "unsigned-readonly.tar.gz"), members(unsigned_root, True))

plist = next(
    item for item in unsigned if item[0].name == "./Fixture.app/Contents/Info.plist"
)
write(os.path.join(test_root, "unsigned-tar-duplicate.tar.gz"), unsigned + [plist])
write(
    os.path.join(test_root, "unsigned-tar-traversal.tar.gz"),
    unsigned + [regular("./Fixture.app/../escape", b"escaped\n")],
)
write(
    os.path.join(test_root, "unsigned-tar-absolute.tar.gz"),
    unsigned + [regular("/escape", b"escaped\n")],
)

# A symlink member standing in for a directory the archive also writes
# through: named without the trailing slash a directory member carries, so
# the shared member validation types it as a file and its own children
# collide with it before anything is extracted.
link = tarfile.TarInfo("./Fixture.app/Contents")
link.mode = 0o777
link.type = tarfile.SYMTYPE
link.linkname = "/etc"
link.mtime = 0
link.uid = link.gid = 0
link.uname = link.gname = ""
write(
    os.path.join(test_root, "unsigned-tar-symlink-parent.tar.gz"),
    [item for item in unsigned if item[0].name != "./Fixture.app/Contents/"]
    + [(link, None)],
)

root = next(item for item in unsigned if item[0].name == "./")
wrapper = copy.copy(root[0])
wrapper.name = "./wrapper/"
wrapped = [root, (wrapper, None)]
for entry, payload in unsigned:
    if entry.name == "./":
        continue
    renamed = copy.copy(entry)
    renamed.name = "./wrapper/" + entry.name[2:]
    wrapped.append((renamed, payload))
write(
    os.path.join(test_root, "unsigned-tar-wrapped.tar.gz"),
    wrapped + [regular("./extra.txt", b"ignored\n")],
)
' "$test_root" "$fixture/settled" "$fixture/detached"

mkdir "$test_root/tar-probe"
tar -xzf "$test_root/unsigned-readonly.tar.gz" -C "$test_root/tar-probe"
if mkdir "$test_root/tar-probe/Fixture.app/Contents/write-probe" 2>/dev/null; then
  echo "unsigned tar archive did not preserve the read-only Contents directory" >&2
  exit 1
fi

# The canonical pairing: manifest-bound unsigned tar, zip signature bundle.
EIDOLA_VERIFY_TMPDIR="$test_root/verifier-tmp" \
  ./scripts/verify-apple.sh "$test_root/unsigned.tar.gz" "$test_root/detached.zip"
if find "$test_root/verifier-tmp" -mindepth 1 -print -quit | grep -q .; then
  echo "verify-apple left its temporary extraction tree behind" >&2
  exit 1
fi

# Both sides tar, with the read-only directory modes a Nix pack can carry.
EIDOLA_VERIFY_TMPDIR="$test_root/verifier-tmp" \
  ./scripts/verify-apple.sh "$test_root/unsigned-readonly.tar.gz" "$test_root/detached.tar.gz"
if find "$test_root/verifier-tmp" -mindepth 1 -print -quit | grep -q .; then
  echo "verify-apple left its temporary extraction tree behind" >&2
  exit 1
fi

assert_verify_refused \
  "$test_root/unsigned-tar-duplicate.tar.gz" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-tar-duplicate.log" \
  'duplicate or colliding archive member'
assert_verify_refused \
  "$test_root/unsigned-tar-traversal.tar.gz" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-tar-traversal.log" \
  'unsafe archive member'
assert_verify_refused \
  "$test_root/unsigned-tar-absolute.tar.gz" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-tar-absolute.log" \
  'unsafe archive member'
assert_verify_refused \
  "$test_root/unsigned-tar-symlink-parent.tar.gz" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-tar-symlink-parent.log" \
  'duplicate or colliding archive member'
assert_verify_refused \
  "$test_root/unsigned-tar-wrapped.tar.gz" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-tar-wrapped.log" \
  'unsigned archive root'

# Neither container: refused on the magic bytes, not on the file name.
printf 'not an archive\n' >"$test_root/unsigned-bogus.tar.gz"
assert_verify_refused \
  "$test_root/unsigned-bogus.tar.gz" \
  "$test_root/detached.zip" \
  "$test_root/unsigned-bogus.log" \
  'neither a zip nor a'
