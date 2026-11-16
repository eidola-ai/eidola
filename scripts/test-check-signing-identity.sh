#!/usr/bin/env bash
# Teeth for scripts/check-signing-identity.sh, over a throwaway git history
# whose releases pin the CI identity the way releases/trust/ does. Bash,
# git and python3 only; no network.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
check="$root/scripts/check-signing-identity.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

repo="$work/repo"
base="https://github.com/eidola-ai/eidola/.github/workflows"
old="$base/tinfoil-build.yml"
new="$base/sign-release.yml"

git init -q "$repo"
git -C "$repo" config user.email test@example.invalid
git -C "$repo" config user.name test
git -C "$repo" config commit.gpgsign false
git -C "$repo" config tag.gpgsign false
git -C "$repo" commit -q --allow-empty -m "before any release"

# Commit a trust root pinning PATTERN, and tag that commit with each TAG.
release() {
  local pattern="$1"
  shift
  mkdir -p "$repo/releases/trust"
  printf '{"expected_ci_identity_pattern": "%s"}\n' "$pattern" \
    > "$repo/releases/trust/trust-constants.json"
  git -C "$repo" add -A
  # The tags in the message keep two otherwise identical commits apart.
  git -C "$repo" commit -q --allow-empty -m "pin $pattern for $*"
  for tag in "$@"; do
    git -C "$repo" tag "$tag"
  done
}

failures=0
expect() {
  local want="$1" name="$2" identity="$3" tag="$4" got=pass
  "$check" "$identity" "$tag" "$repo" > "$work/out" 2>&1 || got=fail
  if [ "$got" = "$want" ]; then
    echo "ok: $name ($got)"
  else
    echo "FAIL: $name: expected $want, got $got" >&2
    sed 's/^/  /' "$work/out" >&2
    failures=$((failures + 1))
  fi
}

expect pass "the first release has no client to satisfy" \
  "$old@refs/tags/v0.1.0" v0.1.0

release "$old@refs/tags/v*" v0.1.0
# The release that moves the pin: its own tree names the new workflow, and
# its clients will require it, but it is checked by v0.1.0's clients.
release "$new@refs/tags/v*" v0.1.1

expect pass "the pin-moving release is signed by the old workflow" \
  "$old@refs/tags/v0.1.1" v0.1.1
expect fail "signing moved in the same release as the pin" \
  "$new@refs/tags/v0.1.1" v0.1.1
expect pass "the release after it is signed by the new workflow" \
  "$new@refs/tags/v0.1.2" v0.1.2
expect fail "and the old workflow can no longer sign" \
  "$old@refs/tags/v0.1.2" v0.1.2
expect fail "an identity bound to another tag" \
  "$new@refs/tags/v0.1.1" v0.1.2
expect fail "another repository's workflow of the same name" \
  "https://github.com/someone-else/eidola/.github/workflows/sign-release.yml@refs/tags/v0.1.2" v0.1.2
expect fail "a tag that is not a release" \
  "$new@refs/tags/latest" latest

# Versions order numerically: v0.1.10 follows v0.1.9, though it sorts
# before it as text, so v0.1.11's clients-to-satisfy are v0.1.10's.
release "$old@refs/tags/v*" v0.1.9
release "$new@refs/tags/v*" v0.1.10
expect pass "the previous release is the highest version, not the last in text order" \
  "$new@refs/tags/v0.1.11" v0.1.11
# And a pre-release sorts below its release: v0.1.10-rc1 is checked by
# v0.1.9's clients, which pin the old workflow.
expect pass "a pre-release is checked by the release below it" \
  "$old@refs/tags/v0.1.10-rc1" v0.1.10-rc1

# Pre-release identifiers compare by precedence, numbers as numbers:
# rc.10 follows rc.9, so v1.0.0's clients-to-satisfy are rc.10's.
release "$old@refs/tags/v*" v1.0.0-rc.9
release "$new@refs/tags/v*" v1.0.0-rc.10
expect pass "rc.10, not rc.9, precedes the release" \
  "$new@refs/tags/v1.0.0" v1.0.0
expect fail "so the workflow rc.9 pinned may not sign it" \
  "$old@refs/tags/v1.0.0" v1.0.0

# Tags that are not releases are not previous releases, wherever they sit.
release "$old@refs/tags/v*" immut-test-1
expect pass "a non-release tag in the history is ignored" \
  "$new@refs/tags/v1.0.1" v1.0.1

# A release on another line is not in this one's history.
git -C "$repo" checkout -q -b other HEAD~1
release "$old@refs/tags/v*" v1.5.0
git -C "$repo" checkout -q -
expect pass "a release outside the tag's history is ignored" \
  "$new@refs/tags/v2.0.0" v2.0.0

# A previous release without trust constants cannot be satisfied knowingly.
git -C "$repo" rm -q -r releases
git -C "$repo" commit -q -m "drop the trust root"
git -C "$repo" tag v0.2.0
expect fail "a previous release with no readable pin" \
  "$new@refs/tags/v0.2.1" v0.2.1

if [ "$failures" -gt 0 ]; then
  echo "check-signing-identity: $failures failure(s)" >&2
  exit 1
fi
echo "check-signing-identity: all cases behaved as documented."
