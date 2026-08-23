#!/usr/bin/env bash
# Decide which files a tagged release publishes, and prove each one is the
# file the signed manifest says it is.
#
# `artifact-manifest.json` records a hash for every artifact, and
# `docs/verification.md` promises that hash is checkable with nothing but
# `sha256sum`. That promise is only real if the file is published *and* is
# the file the hash covers. The two are produced by different workflows —
# `artifacts.yml` builds and measures, `tinfoil-build.yml` attaches to the
# tagged release — so identity across that seam is checked here rather than
# assumed. Whatever this prints is what gets uploaded; anything it cannot
# account for fails the release.
#
# The manifest is the authority, not the directory: every row that records a
# user-checkable file hash must have its file present and matching. A file
# in the directory that no row covers is simply not published — it has no
# attested identity, and an unattested asset beside attested ones is worse
# than a missing one.
#
# Which rows are publishable is a table, not a guess:
#
#   * `nix` rows carry `archiveSha256`, the hash of the flake-built
#     `.tar.gz`. Published as `<key>.tar.gz`.
#   * `file` rows naming a Debian package carry the `.deb`'s own `sha256` —
#     the deb *is* the byte stream, with no archive/envelope indirection.
#     Published as `<key>.deb`.
#   * `eidola-gui-macos-universal-zip` is deliberately **not** published.
#     It is the *unsigned* container, recorded so a verifier can reproduce
#     the recipe; the macOS installable is the Developer ID-signed zip,
#     whose hash lives in the human attestation (docs/verification.md).
#   * `oci` rows are pulled from a registry by digest, not downloaded.
#
# Any other row shape is a hard failure. A new artifact class must be a
# decision about publication, made by whoever adds it, rather than an
# asset that quietly never appears.
#
# Published names are derived from the manifest key so a downloaded file
# and the row a user checks it against cannot be mismatched by eye. The
# same rule names the files `scripts/artifact-manifest.sh --artifact-dir`
# writes, which is what makes this comparison a real check rather than a
# rename.
#
# Usage:
#   scripts/release-assets.sh verify --manifest PATH --dir DIR [--list PATH]
#
#   --list PATH   Write the verified files' paths, one per line, to PATH
#                 (they are printed to stdout regardless).

set -euo pipefail

MANIFEST=""
DIR=""
LIST=""

usage() {
  sed -n '2,50p' "${BASH_SOURCE[0]}"
}

COMMAND="${1:-}"
case "$COMMAND" in
  verify) shift ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "error: expected the 'verify' command" >&2
    usage >&2
    exit 2
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --dir)
      DIR="$2"
      shift 2
      ;;
    --list)
      LIST="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MANIFEST" || -z "$DIR" ]]; then
  echo "error: --manifest and --dir are required" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "error: no such manifest: $MANIFEST" >&2
  exit 2
fi
if [[ ! -d "$DIR" ]]; then
  echo "error: no such directory: $DIR" >&2
  exit 2
fi
if ! command -v python3 > /dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 2
fi

# One grader, as in scripts/check-manifest-determinism.sh, and for the same
# reasons: it reads and reports, it hashes with the stdlib, and it is the
# interpreter every platform here already has. It announces that it ran, so
# an empty report cannot be mistaken for a dead script.
out="$(
  # shellcheck disable=SC2016 # the grader is Python source, not shell
  python3 -c '
import hashlib
import json
import os
import sys

manifest_path, directory = sys.argv[1], sys.argv[2]

with open(manifest_path, "rb") as handle:
    manifest = json.load(handle)

problems = []
published = []

NOT_PUBLISHED = {
    "eidola-gui-macos-universal-zip": (
        "the unsigned macOS container; the installable is the Developer "
        "ID-signed zip, whose hash lives in the human attestation"
    ),
}


def expected(name, entry):
    """(filename, `sha256:`-prefixed hash) for a publishable row, or None."""
    kind = entry.get("type")
    if kind == "oci":
        return None
    if name in NOT_PUBLISHED:
        return None
    if kind == "nix":
        return name + ".tar.gz", entry.get("archiveSha256")
    if kind == "file" and "-deb-" in name:
        return name + ".deb", entry.get("sha256")
    problems.append(
        "artifacts.{}: type {} is not in the publication table — a new "
        "artifact class needs a deliberate decision about whether a "
        "release publishes it (scripts/release-assets.sh)".format(
            name, json.dumps(kind)
        )
    )
    return None


artifacts = manifest.get("artifacts")
if not isinstance(artifacts, dict) or not artifacts:
    problems.append("manifest records no artifacts")
    artifacts = {}

for name in sorted(artifacts):
    entry = artifacts[name]
    if not isinstance(entry, dict):
        problems.append("artifacts.{}: not an object".format(name))
        continue
    plan = expected(name, entry)
    if plan is None:
        continue
    filename, want = plan
    if not isinstance(want, str) or not want.startswith("sha256:"):
        problems.append(
            "artifacts.{}: no sha256 to check the published file "
            "against".format(name)
        )
        continue
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        problems.append(
            "artifacts.{}: the manifest records this artifact, but {} is "
            "not in the release inputs — the signed manifest would point "
            "at a file nobody can download".format(name, filename)
        )
        continue
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    got = "sha256:" + digest.hexdigest()
    if got != want:
        problems.append(
            "artifacts.{}: {} hashes to {}, but the manifest records {} — "
            "the file built is not the file measured".format(
                name, filename, got, want
            )
        )
        continue
    published.append(path)

for line in problems:
    print("PROBLEM " + line)
for path in published:
    print("PUBLISH " + path)
print("@graded")
' "$MANIFEST" "$DIR"
)"

if [[ "$out" != *"@graded"* ]]; then
  echo "error: the release-asset grader did not run to completion" >&2
  exit 2
fi

problems=0
files=()
while IFS= read -r line; do
  case "$line" in
    "PROBLEM "*)
      echo "  ${line#PROBLEM }" >&2
      problems=$((problems + 1))
      ;;
    "PUBLISH "*) files+=("${line#PUBLISH }") ;;
  esac
done <<< "$out"

if [[ "$problems" -gt 0 ]]; then
  echo "error: $problems release asset(s) could not be verified against $MANIFEST" >&2
  exit 1
fi
if [[ "${#files[@]}" -eq 0 ]]; then
  echo "error: no release assets to publish — the manifest records none, which cannot be right" >&2
  exit 1
fi

printf '%s\n' "${files[@]}"
if [[ -n "$LIST" ]]; then
  printf '%s\n' "${files[@]}" > "$LIST"
fi
echo "Verified ${#files[@]} release asset(s) against $MANIFEST." >&2
