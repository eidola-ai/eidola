#!/usr/bin/env bash
# Regressions for the macOS shipping container's recipe
# (`scripts/pack-shipping-zip.sh`, which `mkShippingZip` in flake.nix runs).
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
# So this drives the script itself over a fixture bundle shaped like the
# thing that breaks — a framework's `Current`/`Foo` symlink pair — twice,
# and again under a restrictive umask. It also holds the flake to running
# that script rather than restating it, because two copies of a byte-exact
# recipe are two recipes. The final negative case reruns the packing
# *without* `-y` and requires it to lose the symlink.
#
# Runs on macOS or Linux with Info-ZIP `zip`/`unzip`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLAKE="$REPO_ROOT/flake.nix"
PACK="$REPO_ROOT/scripts/pack-shipping-zip.sh"

for tool in zip unzip; do
  if ! command -v "$tool" > /dev/null 2>&1; then
    echo "error: $tool is required (Info-ZIP)" >&2
    exit 1
  fi
done

# ── one recipe: the derivation runs the script, it does not restate it ───
recipe="$(awk '/^ *mkShippingZip =/ { collecting = 1 }
               collecting { print }
               collecting && /^ *'"''"';$/ { exit }' "$FLAKE")"
if [[ -z "$recipe" ]]; then
  echo "FAIL: no mkShippingZip block found in flake.nix — has the recipe moved?" >&2
  exit 1
fi
if [[ "$recipe" != *"scripts/pack-shipping-zip.sh"* ]]; then
  echo "FAIL: mkShippingZip no longer runs scripts/pack-shipping-zip.sh" >&2
  exit 1
fi
if [[ "$recipe" == *" zip "* || "$recipe" == *"touch -"* ]]; then
  echo "FAIL: mkShippingZip has grown its own copy of the packing recipe — there must be one" >&2
  echo "  $recipe" >&2
  exit 1
fi
echo "ok: the derivation runs the packing script rather than restating it"

# ── the script still asks for what this harness proves out ───────────────
invocation="$(grep -E 'find \. -mindepth 1 \| LC_ALL=C sort \| zip ' "$PACK" || true)"
if [[ -z "$invocation" ]]; then
  echo "FAIL: the packing script no longer runs a sorted find into zip" >&2
  exit 1
fi
for flag in -X -y; do
  if [[ "$invocation" != *" $flag "* ]]; then
    echo "FAIL: the packing script no longer passes $flag:" >&2
    echo "  $invocation" >&2
    exit 1
  fi
done
echo "ok: the packing script passes -X and -y"

# The stamp, the zone and the mode set are hash inputs, so the script must
# state them rather than inherit whatever the caller's shell provides.
for pinned in 'chmod -R u=rwX,go=rX' 'touch -h -t 198001010000.00' 'TZ=UTC'; do
  if ! grep -qF "$pinned" "$PACK"; then
    echo "FAIL: the packing script no longer pins: $pinned" >&2
    exit 1
  fi
done
echo "ok: the packing script pins the mode set, the stamp and the zone"

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

# ── determinism: same tree, two runs, same bytes ─────────────────────────
cp -R "$WORK/payload" "$WORK/payload-2"
"$PACK" "$WORK/payload" "$WORK/one.zip"
"$PACK" "$WORK/payload-2" "$WORK/two.zip"
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
"$PACK" "$WORK/payload-strict" "$WORK/strict.zip"
if ! cmp -s "$WORK/one.zip" "$WORK/strict.zip"; then
  echo "FAIL: the same payload zipped under umask 077 differs — a mode is reaching the hash" >&2
  exit 1
fi
echo "ok: the same payload under umask 077 produces the same bytes"

# ── the script packs a tree the caller names, wherever it came from ──────
# The point of the script existing: a verifier's reconstructed tree is not
# the Nix payload and is not in the store.
mkdir -p "$WORK/elsewhere"
cp -R "$WORK/payload/Eidola.app" "$WORK/elsewhere/Eidola.app"
"$PACK" "$WORK/elsewhere" "$WORK/elsewhere.zip"
if ! cmp -s "$WORK/one.zip" "$WORK/elsewhere.zip"; then
  echo "FAIL: the same bundle packed from a different directory differs" >&2
  exit 1
fi
echo "ok: an arbitrary tree with the same contents packs to the same bytes"

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
echo "ok: symlinks and the executable bit survive the round trip"

# ── the negative case that makes -y load-bearing ─────────────────────────
export TZ=UTC LC_ALL=C
cp -R "$WORK/payload" "$WORK/payload-3"
chmod -R u=rwX,go=rX "$WORK/payload-3"
find "$WORK/payload-3" -exec touch -h -t 198001010000.00 {} +
(cd "$WORK/payload-3" && find . -mindepth 1 | LC_ALL=C sort | zip -q -X -@ "$WORK/no-y.zip")
mkdir -p "$WORK/out-no-y"
unzip -qq "$WORK/no-y.zip" -d "$WORK/out-no-y"
if [[ -L "$WORK/out-no-y/Eidola.app/Contents/Frameworks/Foo.framework/Versions/Current" ]]; then
  echo "FAIL: dropping -y still preserved the symlink — this harness no longer proves anything" >&2
  exit 1
fi
echo "ok: without -y the symlink is silently replaced (the regression -y prevents)"

echo "PASS: shipping-zip recipe regressions"
