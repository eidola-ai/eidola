#!/usr/bin/env bash
# Regressions for the macOS shipping container's zip recipe
# (`mkShippingZip` in flake.nix).
#
# The recipe's whole job is to be a *function of the payload*: the same
# tree must produce the same bytes on any machine, and the bytes must still
# describe the tree it was given. Three properties carry that, and each
# fails silently:
#
#   * `-y` stores symlinks as symlinks. Without it Info-ZIP follows them
#     and writes copies — a bundle containing a framework comes back with
#     `Versions/Current` as a directory of duplicated files, which is a
#     different (and invalid) bundle. Nothing errors.
#   * mtimes, entry order and the extra fields carrying uid/gid have to be
#     pinned, or the same tree hashes differently on the next run.
#   * modes have to be normalized. zip records them, so without that the
#     hash is a function of whoever's umask made the tree — and re-zipping
#     a reconstructed bundle is exactly what an external verifier does.
#
# So this asserts the flake's recipe still declares what it must, then
# exercises the same recipe on a fixture bundle shaped like the thing that
# breaks: a framework's `Current`/`Foo` symlink pair, zipped twice and then
# again under a restrictive umask. The final negative case runs the recipe
# *without* `-y` and requires it to lose the symlink — the reason the flag
# is not optional.
#
# Runs on macOS or Linux with Info-ZIP `zip`/`unzip`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLAKE="$REPO_ROOT/flake.nix"

for tool in zip unzip; do
  if ! command -v "$tool" > /dev/null 2>&1; then
    echo "error: $tool is required (Info-ZIP)" >&2
    exit 1
  fi
done

# ── the flake still asks for what this harness proves out ────────────────
# Read the `mkShippingZip` block alone, never the whole file: other
# derivations in flake.nix set SOURCE_DATE_EPOCH (to 0, which zip cannot
# even represent), so a file-wide match would report a declaration this
# recipe does not have.
recipe="$(awk '/^ *mkShippingZip =/ { collecting = 1 }
               collecting { print }
               collecting && /^ *'"''"';$/ { exit }' "$FLAKE")"
if [[ -z "$recipe" ]]; then
  echo "FAIL: no mkShippingZip block found in flake.nix — has the recipe moved?" >&2
  exit 1
fi

invocation="$(printf '%s\n' "$recipe" | grep -E 'find \. -mindepth 1 \| LC_ALL=C sort \| zip ' || true)"
if [[ -z "$invocation" ]]; then
  echo "FAIL: mkShippingZip no longer runs a sorted find into zip" >&2
  exit 1
fi
for flag in -X -y; do
  if [[ "$invocation" != *" $flag "* ]]; then
    echo "FAIL: flake.nix's shipping zip no longer passes $flag:" >&2
    echo "  $invocation" >&2
    exit 1
  fi
done
echo "ok: mkShippingZip passes -X and -y"

# The stamp, the zone and the mode set are hash inputs, so the derivation
# must state them rather than inherit whatever stdenv or the caller's umask
# happens to give it.
for pinned in 'SOURCE_DATE_EPOCH = "315532800"' 'TZ = "UTC"' 'chmod -R u=rwX,go=rX'; do
  if [[ "$recipe" != *"$pinned"* ]]; then
    echo "FAIL: mkShippingZip no longer declares $pinned" >&2
    exit 1
  fi
done
echo "ok: mkShippingZip declares SOURCE_DATE_EPOCH, TZ, and the exact mode set"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ── a payload shaped like the case that breaks ───────────────────────────
app="$WORK/payload/Eidola.app"
mkdir -p "$app/Contents/MacOS" \
  "$app/Contents/Resources" \
  "$app/Contents/Frameworks/Foo.framework/Versions/A"
printf 'main\n' > "$app/Contents/MacOS/Eidola"
chmod 755 "$app/Contents/MacOS/Eidola"
printf 'plist\n' > "$app/Contents/Info.plist"
printf 'resource\n' > "$app/Contents/Resources/data.bin"
printf 'framework\n' > "$app/Contents/Frameworks/Foo.framework/Versions/A/Foo"
ln -s A "$app/Contents/Frameworks/Foo.framework/Versions/Current"
ln -s Versions/Current/Foo "$app/Contents/Frameworks/Foo.framework/Foo"

# `-t` rather than `-d @epoch`: BSD touch has no @epoch form, and both
# accept this stamp. TZ is fixed because zip records DOS *local* time —
# the same reason the derivation sets it.
export TZ=UTC LC_ALL=C
normalize_and_zip() {
  local tree="$1" out="$2"
  shift 2
  # The same two normalizations the derivation performs, in the same order:
  # modes to an exact set (zip records them; a umask would otherwise reach
  # the hash) and every mtime to the pinned stamp.
  chmod -R u=rwX,go=rX "$tree"
  find "$tree" -exec touch -h -t 198001010000.00 {} +
  (cd "$tree" && find . -mindepth 1 | LC_ALL=C sort | zip -q "$@" -@ "$out")
}

# ── determinism: same tree, two runs, same bytes ─────────────────────────
cp -R "$WORK/payload" "$WORK/payload-2"
normalize_and_zip "$WORK/payload" "$WORK/one.zip" -X -y
normalize_and_zip "$WORK/payload-2" "$WORK/two.zip" -X -y
if ! cmp -s "$WORK/one.zip" "$WORK/two.zip"; then
  echo "FAIL: two runs of the recipe over the same tree differ" >&2
  exit 1
fi
echo "ok: the recipe is byte-identical across runs ($(wc -c < "$WORK/one.zip" | tr -d ' ') bytes)"

# ── the verifier's umask must not reach the hash ─────────────────────────
# The documented check re-zips a reconstructed tree and compares the file
# against a published hash. That comparison is only meaningful if the
# recipe is a function of the payload, and zip records Unix modes — so a
# stricter umask on the verifier's machine must produce the same bytes.
(
  umask 077
  cp -R "$WORK/payload" "$WORK/payload-strict"
)
normalize_and_zip "$WORK/payload-strict" "$WORK/strict.zip" -X -y
if ! cmp -s "$WORK/one.zip" "$WORK/strict.zip"; then
  echo "FAIL: the same payload zipped under umask 077 differs — a mode is reaching the hash" >&2
  exit 1
fi
echo "ok: the same payload under umask 077 produces the same bytes"

# ── round trip: the bundle that comes back is the bundle that went in ────
mkdir -p "$WORK/out"
unzip -qq "$WORK/one.zip" -d "$WORK/out"
back="$WORK/out/Eidola.app/Contents/Frameworks/Foo.framework"

if [[ ! -L "$back/Versions/Current" ]]; then
  echo "FAIL: Versions/Current came back as a $( [[ -d "$back/Versions/Current" ]] && echo directory || echo file), not a symlink" >&2
  exit 1
fi
target="$(readlink "$back/Versions/Current")"
if [[ "$target" != "A" ]]; then
  echo "FAIL: Versions/Current points at '$target', expected 'A'" >&2
  exit 1
fi
if [[ ! -L "$back/Foo" || "$(readlink "$back/Foo")" != "Versions/Current/Foo" ]]; then
  echo "FAIL: the framework's Foo symlink did not survive the round trip" >&2
  exit 1
fi
if [[ ! -x "$WORK/out/Eidola.app/Contents/MacOS/Eidola" ]]; then
  echo "FAIL: the executable bit did not survive the round trip" >&2
  exit 1
fi
echo "ok: symlinks and the executable bit survive zip -X -y / unzip"

# ── the negative case that makes -y load-bearing ─────────────────────────
cp -R "$WORK/payload" "$WORK/payload-3"
normalize_and_zip "$WORK/payload-3" "$WORK/no-y.zip" -X
mkdir -p "$WORK/out-no-y"
unzip -qq "$WORK/no-y.zip" -d "$WORK/out-no-y"
if [[ -L "$WORK/out-no-y/Eidola.app/Contents/Frameworks/Foo.framework/Versions/Current" ]]; then
  echo "FAIL: dropping -y still preserved the symlink — this harness no longer proves anything" >&2
  exit 1
fi
echo "ok: without -y the symlink is silently replaced (the regression -y prevents)"

echo "PASS: shipping-zip recipe regressions"
