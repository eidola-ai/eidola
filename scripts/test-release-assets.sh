#!/usr/bin/env bash
# Regressions for `scripts/release-assets.sh` — the check that stands
# between a signed manifest and the files a release attaches.
#
# That check only ever runs on a real `v*` tag, which is the worst place to
# discover it is wrong: a release is not a thing you re-run casually. So
# every branch of it is driven here, over synthesized manifests and files,
# on every PR:
#
#   * the happy path publishes exactly the rows the table covers, under
#     the names derived from their manifest keys;
#   * a file whose bytes changed after measurement is rejected — the one
#     failure the whole cross-workflow seam exists to catch;
#   * a recorded artifact with no file is rejected, because a signed
#     manifest pointing at an undownloadable file is the gap this closes;
#   * an artifact class nobody put in the publication table is rejected
#     rather than silently skipped;
#   * `oci` rows and the unsigned macOS container are skipped by
#     construction, and their absence from the directory is not an error.
#
# It also drives `wait-budget`, the other half of the same release path:
# the publisher waits on the artifact workflow with no clock of its own, so
# its `timeout-minutes` is the only bound, and that number lives in a
# different file from the timeouts it has to cover. Checked here against
# the real committed workflows, plus the two ways the relationship can
# break.
#
# Bash + python3 only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY="$REPO_ROOT/scripts/release-assets.sh"

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

failures=0

note() { echo "  $*"; }

fail() {
  echo "  FAIL: $*" >&2
  failures=$((failures + 1))
}

# Build a case directory: `$1/manifest.json` + `$1/dir` holding the files
# named by the caller. Every file's recorded hash is computed from the bytes
# actually written, so only the deliberate mutations below can diverge.
sha_of() {
  python3 -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print("sha256:" + h.hexdigest())
' "$1"
}

make_case() {
  local case_dir="$1"
  mkdir -p "$case_dir/dir"
  printf 'nix archive bytes\n' > "$case_dir/dir/eidola-gui-linux-nix-amd64.tar.gz"
  printf 'amd64 debian package bytes\n' > "$case_dir/dir/eidola-gui-linux-deb-amd64.deb"
  printf 'arm64 debian package bytes\n' > "$case_dir/dir/eidola-gui-linux-deb-arm64.deb"
  python3 - "$case_dir" "$(sha_of "$case_dir/dir/eidola-gui-linux-nix-amd64.tar.gz")" \
    "$(sha_of "$case_dir/dir/eidola-gui-linux-deb-amd64.deb")" \
    "$(sha_of "$case_dir/dir/eidola-gui-linux-deb-arm64.deb")" <<'PY'
import json
import sys

case_dir, nix_sha, amd_sha, arm_sha = sys.argv[1:5]
manifest = {
    "schema_version": 3,
    "enclave": {},
    "artifacts": {
        # Pulled by digest, never downloaded — must not be looked for.
        "eidola-server": {
            "type": "oci",
            "platform": "linux/amd64",
            "digest": "sha256:" + "a" * 64,
        },
        "eidola-gui-linux-nix-amd64": {
            "type": "nix",
            "platform": "linux/amd64",
            "narHash": "sha256-aa",
            "archiveSha256": nix_sha,
        },
        "eidola-gui-linux-deb-amd64": {
            "type": "file",
            "platform": "linux/amd64",
            "sha256": amd_sha,
        },
        "eidola-gui-linux-deb-arm64": {
            "type": "file",
            "platform": "linux/arm64",
            "sha256": arm_sha,
        },
        # Recorded, deliberately not published — and deliberately absent
        # from the directory, so a rule that merely tolerated it would not
        # look the same as one that skips it.
        "eidola-gui-macos-universal-zip": {
            "type": "file",
            "platform": "darwin/universal",
            "sha256": "sha256:" + "b" * 64,
        },
    },
}
with open(case_dir + "/manifest.json", "w") as handle:
    json.dump(manifest, handle, indent=2)
PY
}

run_case() {
  local case_dir="$1"
  "$VERIFY" verify --manifest "$case_dir/manifest.json" --dir "$case_dir/dir" \
    > "$case_dir/stdout" 2> "$case_dir/stderr"
}

echo "── happy path: exactly the publishable rows, named after their keys ──"
make_case "$TMP_ROOT/ok"
if run_case "$TMP_ROOT/ok"; then
  expected="eidola-gui-linux-deb-amd64.deb
eidola-gui-linux-deb-arm64.deb
eidola-gui-linux-nix-amd64.tar.gz"
  actual="$(sed 's|.*/||' "$TMP_ROOT/ok/stdout" | sort)"
  if [[ "$actual" == "$expected" ]]; then
    note "publishes: $(tr '\n' ' ' < "$TMP_ROOT/ok/stdout" | sed 's|[^ ]*/||g')"
  else
    fail "unexpected publication list:"$'\n'"$actual"
  fi
else
  fail "the happy path did not verify:"$'\n'"$(cat "$TMP_ROOT/ok/stderr")"
fi

echo "── a file that changed after it was measured is rejected ──"
make_case "$TMP_ROOT/tampered"
printf 'amd64 debian package bytes (tampered)\n' \
  > "$TMP_ROOT/tampered/dir/eidola-gui-linux-deb-amd64.deb"
if run_case "$TMP_ROOT/tampered"; then
  fail "a file whose bytes no longer match its recorded hash was published"
else
  if grep -q "is not the file measured" "$TMP_ROOT/tampered/stderr"; then
    note "rejected, naming the mismatch"
  else
    fail "rejected, but not for the hash mismatch:"$'\n'"$(cat "$TMP_ROOT/tampered/stderr")"
  fi
fi

echo "── a recorded artifact with no file is rejected ──"
make_case "$TMP_ROOT/missing"
rm "$TMP_ROOT/missing/dir/eidola-gui-linux-deb-arm64.deb"
if run_case "$TMP_ROOT/missing"; then
  fail "a manifest row with no published file was accepted"
else
  # The message, not just the row name: a grader that *crashed* on the
  # absent file would also mention it, in a traceback, and that is a
  # different (and much worse) behavior than reporting it.
  if grep -q "eidola-gui-linux-deb-arm64: the manifest records this artifact" \
    "$TMP_ROOT/missing/stderr"; then
    note "rejected, naming the row"
  else
    fail "rejected, but not for the missing file:"$'\n'"$(cat "$TMP_ROOT/missing/stderr")"
  fi
fi

echo "── an artifact class outside the publication table is rejected ──"
make_case "$TMP_ROOT/unknown"
python3 - "$TMP_ROOT/unknown/manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    manifest = json.load(handle)
manifest["artifacts"]["eidola-gui-linux-rpm-amd64"] = {
    "type": "file",
    "platform": "linux/amd64",
    "sha256": "sha256:" + "c" * 64,
}
with open(path, "w") as handle:
    json.dump(manifest, handle, indent=2)
PY
if run_case "$TMP_ROOT/unknown"; then
  fail "an artifact class nobody decided about was silently not published"
else
  if grep -q "publication table" "$TMP_ROOT/unknown/stderr"; then
    note "rejected, asking for a decision"
  else
    fail "rejected, but not for the missing table entry:"$'\n'"$(cat "$TMP_ROOT/unknown/stderr")"
  fi
fi

echo "── a manifest that records nothing publishes nothing, loudly ──"
make_case "$TMP_ROOT/empty"
python3 - "$TMP_ROOT/empty/manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    manifest = json.load(handle)
manifest["artifacts"] = {}
with open(path, "w") as handle:
    json.dump(manifest, handle, indent=2)
PY
if run_case "$TMP_ROOT/empty"; then
  fail "an empty manifest produced a successful release upload"
else
  note "rejected"
fi

echo "── the publisher outlasts the workflow it waits on ──"
ARTIFACTS_WORKFLOW="$REPO_ROOT/.github/workflows/artifacts.yml"
RELEASE_WORKFLOW="$REPO_ROOT/.github/workflows/tinfoil-build.yml"

budget() {
  "$VERIFY" wait-budget \
    --artifacts-workflow "$1" --release-workflow "$2" > "$3" 2>&1
}

if budget "$ARTIFACTS_WORKFLOW" "$RELEASE_WORKFLOW" "$TMP_ROOT/budget"; then
  note "$(grep 'longest declared path' "$TMP_ROOT/budget")"
else
  fail "the committed workflows do not satisfy the wait budget:"$'\n'"$(cat "$TMP_ROOT/budget")"
fi

echo "── a producer timeout bump the publisher cannot cover is rejected ──"
# The failure this pair exists to prevent: someone raises a build job's
# timeout, and the publisher silently starts giving up on valid runs.
sed 's/^    timeout-minutes: 180$/    timeout-minutes: 900/' \
  "$ARTIFACTS_WORKFLOW" > "$TMP_ROOT/artifacts-bumped.yml"
if cmp -s "$ARTIFACTS_WORKFLOW" "$TMP_ROOT/artifacts-bumped.yml"; then
  fail "the fixture edit changed nothing — the timeout it targets moved"
elif budget "$TMP_ROOT/artifacts-bumped.yml" "$RELEASE_WORKFLOW" "$TMP_ROOT/bumped"; then
  fail "a producer that outlives the publisher was accepted"
else
  note "rejected, naming the minimum: $(grep -o 'at least [0-9]*' "$TMP_ROOT/bumped" | head -1)"
fi

echo "── a publisher timeout GitHub will not honor is rejected ──"
# Patience beyond the hosted-runner limit is a promise nobody keeps, so it
# must not be the way the check above gets satisfied.
sed 's/^    timeout-minutes: 330$/    timeout-minutes: 3000/' \
  "$RELEASE_WORKFLOW" > "$TMP_ROOT/release-overlong.yml"
if cmp -s "$RELEASE_WORKFLOW" "$TMP_ROOT/release-overlong.yml"; then
  fail "the fixture edit changed nothing — the publisher timeout moved"
elif budget "$ARTIFACTS_WORKFLOW" "$TMP_ROOT/release-overlong.yml" "$TMP_ROOT/overlong"; then
  fail "a timeout beyond the hosted-runner limit was accepted"
else
  note "rejected"
fi

if [[ "$failures" -gt 0 ]]; then
  echo "release-asset verification harness: $failures failure(s)" >&2
  exit 1
fi
echo "release-asset verification harness: all cases behaved as documented."
