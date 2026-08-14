#!/bin/bash
# Narrow independent check: signapple must reproduce the arm64 slice and may
# differ on x86_64 only by its measured __LINKEDIT vmsize arithmetic.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$REPO_ROOT/scripts/fixtures/apple-roundtrip/synthetic-universal"
SIGNAPPLE="$(nix build "$REPO_ROOT#signapple" --no-link --print-out-paths)/bin/signapple"
DIFF_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/eidola-apple-differential.XXXXXX")"
trap 'rm -rf "$DIFF_ROOT"' EXIT

cp -R "$FIXTURE/settled/Fixture.app" "$DIFF_ROOT/Fixture.app"
chmod -R u+w "$DIFF_ROOT/Fixture.app"
"$SIGNAPPLE" apply --no-verify \
  "$DIFF_ROOT/Fixture.app" "$FIXTURE/detached/Fixture.app" >/dev/null

classification="$(python3 "$REPO_ROOT/scripts/apple_linkedit_diff.py" \
  "$FIXTURE/signed/Fixture.app/Contents/MacOS/Fixture" \
  "$DIFF_ROOT/Fixture.app/Contents/MacOS/Fixture" | head -1)"
if [[ "$classification" != "linkedit-vmsize-only" ]]; then
  echo "unexpected signapple differential: $classification" >&2
  exit 1
fi

lipo "$FIXTURE/signed/Fixture.app/Contents/MacOS/Fixture" \
  -thin arm64 -output "$DIFF_ROOT/expected-arm64"
lipo "$DIFF_ROOT/Fixture.app/Contents/MacOS/Fixture" \
  -thin arm64 -output "$DIFF_ROOT/applied-arm64"
cmp "$DIFF_ROOT/expected-arm64" "$DIFF_ROOT/applied-arm64"
cmp "$FIXTURE/signed/Fixture.app/Contents/_CodeSignature/CodeResources" \
  "$DIFF_ROOT/Fixture.app/Contents/_CodeSignature/CodeResources"

echo "signapple independently reproduced the arm64 slice and bundle seal"
