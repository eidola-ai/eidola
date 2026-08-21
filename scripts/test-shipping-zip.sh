#!/usr/bin/env bash
# Regressions for the macOS shipping container's zip recipe
# (`mkShippingZip` in flake.nix).
#
# The recipe's whole job is to be a *function of the payload*: the same
# tree must produce the same bytes on any machine, and the bytes must still
# describe the tree it was given. Two properties carry that, and both fail
# silently:
#
#   * `-y` stores symlinks as symlinks. Without it Info-ZIP follows them
#     and writes copies — a bundle containing a framework comes back with
#     `Versions/Current` as a directory of duplicated files, which is a
#     different (and invalid) bundle. Nothing errors.
#   * mtimes, entry order and the extra fields carrying uid/gid have to be
#     pinned, or the same tree hashes differently on the next run.
#
# So this asserts the flake's invocation still carries the flags, then
# exercises the recipe on a fixture bundle shaped like the thing that
# breaks: a framework's `Current`/`Foo` symlink pair. The final negative
# case runs the same recipe *without* `-y` and requires it to lose the
# symlink — the reason the flag is not optional.
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

# ── the flake still asks for the flags this harness proves out ───────────
invocation="$(grep -E 'find \. -mindepth 1 \| LC_ALL=C sort \| zip ' "$FLAKE" || true)"
if [[ -z "$invocation" ]]; then
  echo "FAIL: no shipping-zip invocation found in flake.nix — has the recipe moved?" >&2
  exit 1
fi
for flag in -X -y; do
  if [[ "$invocation" != *" $flag "* ]]; then
    echo "FAIL: flake.nix's shipping zip no longer passes $flag:" >&2
    echo "  $invocation" >&2
    exit 1
  fi
done
echo "ok: flake.nix's shipping zip passes -X and -y"

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
