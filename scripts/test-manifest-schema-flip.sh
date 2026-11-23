#!/usr/bin/env bash
# The emit side of the next manifest schema, run before any release emits it.
#
# Emitting a new `artifact-manifest.json` schema is one assignment in
# `scripts/artifact-manifest.sh` (`MANIFEST_SCHEMA_VERSION`), made at
# release time once a release accepting that schema has shipped
# (releases/README.md → Rotating document schema versions). Everything that
# assignment turns on — the rows each producer composes from the schema that
# records them, and the artifact set `check-complete` requires — is code that
# otherwise first runs on the release that needs it. So it runs here, on
# every PR, against a copy of the script with the constant flipped:
#
#   * the flip is exactly one line, and the committed script differs from
#     the flipped copy in that line alone;
#   * as committed (below the flip), the mobile producer records nothing;
#   * flipped, the mobile producer records the fixture bundle and APK
#     (scripts/fixtures/manifest-flip/: a small bundle, stand-in APK bytes),
#     and the script's own `verify-full` composes them with every other row
#     the schema requires (fixture values, `rows.json`) and a fixture
#     enclave, holds the result to the complete artifact set, and requires
#     it to equal the committed `good-schema-6-emitted.json`, byte for byte;
#   * that manifest passes `scripts/check-manifest-determinism.sh`, and
#     `scripts/release-assets.sh` publishes the bundle container the
#     producer wrote, under the name its row has.
#
# The same committed manifest is read by app-core's claims check (no claims
# change at schema 6, and the bundle row parses) and by `release-tool`
# (`updates::tests`, `updater::mobile::row::tests`, `apple_verify::tests`),
# so the bytes the emitter writes are the bytes the readers are tested on.
# `--update` rewrites it after a deliberate change to the producers or the
# fixtures.
#
# Usage: scripts/test-manifest-schema-flip.sh [--update]
# Needs bash, jq and python3; builds nothing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/artifact-manifest.sh"
FIXTURES="$REPO_ROOT/scripts/fixtures/manifest-flip"
EMITTED="$REPO_ROOT/scripts/fixtures/manifest-determinism/good-schema-6-emitted.json"
FLIP_TO=6
UPDATE=0
case "${1:-}" in
  "") ;;
  --update) UPDATE=1 ;;
  *)
    echo "usage: $0 [--update]" >&2
    exit 2
    ;;
esac

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT
failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

echo "── the flip is one line ──"
pattern='^MANIFEST_SCHEMA_VERSION=[0-9]+$'
if [[ "$(grep -cE "$pattern" "$SCRIPT")" -ne 1 ]]; then
  echo "FAIL: $SCRIPT must assign MANIFEST_SCHEMA_VERSION on exactly one line" >&2
  exit 1
fi
current="$(grep -E "$pattern" "$SCRIPT" | cut -d= -f2)"
FLIPPED="$TMP_ROOT/artifact-manifest.sh"
sed -E "s/$pattern/MANIFEST_SCHEMA_VERSION=${FLIP_TO}/" "$SCRIPT" > "$FLIPPED"
chmod +x "$FLIPPED"
changed="$(diff "$SCRIPT" "$FLIPPED" | grep -c '^[<>]' || true)"
if [[ "$current" -ne "$FLIP_TO" && "$changed" -ne 2 ]]; then
  fail "flipping the constant changed $changed lines, not one"
fi
echo "emitted today: $current; flipped copy: $FLIP_TO"

# Both copies run from the repository (they find it with git), so the
# producers they call are this tree's.
cd "$REPO_ROOT"

if [[ "$current" -lt "$FLIP_TO" ]]; then
  echo "── as committed, the mobile producer records nothing ──"
  "$SCRIPT" record-mobile --bundle-dir "$FIXTURES/bundle" --apk "$FIXTURES/unsigned.apk" \
    --output "$TMP_ROOT/below.json" 2> "$TMP_ROOT/below.err"
  if [[ "$(jq -c .artifacts "$TMP_ROOT/below.json")" != "{}" ]]; then
    fail "at schema $current the mobile partial records rows: $(jq -c .artifacts "$TMP_ROOT/below.json")"
  fi
fi

echo "── flipped, the producers compose a complete schema-${FLIP_TO} manifest ──"
"$FLIPPED" record-mobile --bundle-dir "$FIXTURES/bundle" --apk "$FIXTURES/unsigned.apk" \
  --output "$TMP_ROOT/mobile.json" --artifact-dir "$TMP_ROOT/assets" 2> "$TMP_ROOT/mobile.err"
for key in eidola-mobile-bundle eidola-mobile-android-apk; do
  if [[ "$(jq -r --arg k "$key" '.artifacts | has($k)' "$TMP_ROOT/mobile.json")" != "true" ]]; then
    fail "the flipped mobile partial has no $key row"
  fi
done

# `verify-full` is the composition CI runs: merge the partials, hold the
# result to the artifact set the schema requires, and compare it with a
# manifest. The enclave block is supplied rather than measured (measuring
# downloads the platform's image), from a fixture, so a release that moves
# the real enclave does not move this file.
enclave="$(jq -c 'del(.schema_version)' "$FIXTURES/server-enclave.json")"
target="$EMITTED"
if [[ "$UPDATE" -eq 1 ]]; then
  target="$TMP_ROOT/emitted.json"
  : > "$target"
fi
if ! ENCLAVE_MEASUREMENTS="$enclave" "$FLIPPED" verify-full \
  --partial "$FIXTURES/rows.json" \
  --partial "$TMP_ROOT/mobile.json" \
  --manifest "$target" \
  --server-enclave "$FIXTURES/server-enclave.json" \
  --output "$TMP_ROOT/composed.json" > "$TMP_ROOT/verify.out" 2>&1; then
  if [[ "$UPDATE" -eq 1 ]] && ! grep -q "carry the artifact set" "$TMP_ROOT/verify.out"; then
    cp "$TMP_ROOT/composed.json" "$EMITTED"
    echo "wrote $EMITTED"
  else
    cat "$TMP_ROOT/verify.out" >&2
    fail "verify-full refused the flipped composition (or it is not $EMITTED)"
  fi
fi
if [[ "$UPDATE" -eq 0 ]] && ! cmp -s "$TMP_ROOT/composed.json" "$EMITTED"; then
  fail "the composed manifest is not byte for byte $EMITTED (run with --update after a deliberate change)"
fi

echo "── the emitted manifest passes the determinism gate ──"
if ! "$REPO_ROOT/scripts/check-manifest-determinism.sh" --manifest "$TMP_ROOT/composed.json" \
  --workflow "$REPO_ROOT/.github/workflows/artifacts.yml" > "$TMP_ROOT/determinism.out" 2>&1; then
  cat "$TMP_ROOT/determinism.out" >&2
  fail "check-manifest-determinism.sh refused the flipped composition"
fi

echo "── the bundle container the producer wrote is the asset its row publishes ──"
jq '{schema_version, artifacts: {"eidola-mobile-bundle": .artifacts["eidola-mobile-bundle"]}}' \
  "$TMP_ROOT/composed.json" > "$TMP_ROOT/bundle-only.json"
if ! "$REPO_ROOT/scripts/release-assets.sh" verify --manifest "$TMP_ROOT/bundle-only.json" \
  --dir "$TMP_ROOT/assets" > "$TMP_ROOT/assets.out" 2>&1; then
  cat "$TMP_ROOT/assets.out" >&2
  fail "release-assets.sh refused the container the producer wrote"
elif ! grep -q 'eidola-mobile-bundle.tar$' "$TMP_ROOT/assets.out"; then
  fail "release-assets.sh did not publish eidola-mobile-bundle.tar"
fi

if [[ "$failures" -gt 0 ]]; then
  echo "$failures schema-flip check(s) failed" >&2
  exit 1
fi
echo "OK: flipping to schema ${FLIP_TO} composes a manifest every gate and reader accepts"
