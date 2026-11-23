#!/usr/bin/env bash
# Builds the app bundle the shells embed: eidola-mobile for wasm32-unknown-unknown on the pinned
# stable toolchain, bound by the wasm-bindgen CLI at exactly the lockfile's version, plus the page
# that loads it. No inline script anywhere (the shells' CSP refuses them).
#   build-bundle.sh [--self-test | --dev-backend URL [--dev-model NAME]] [--out DIR]
# --self-test    the self-test bundle (feature self-test, its own page); a release bundle has none of it
# --dev-backend  a development bundle (feature dev): the default agent answers from the
#                OpenAI-compatible server at URL (tools/openai_stub.py), and every probe is
#                mirrored into the page for the UI tests; never a release bundle
# --dev-model    the model id that server answers as (default: stub)
# --out          output directory (default build/bundle, build/bundle-self-test or build/bundle-dev)
# Prints the bundle digest (shell-contract.md → Bundle digest) and the sizes. CARGO_TARGET_DIR is
# honoured; bundle-repro.sh builds twice in fresh ones and compares.
set -euo pipefail
export LC_ALL=C TZ=UTC
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CRATE="$ROOT/crates/eidola-mobile"
SELF_TEST=false
DEV_BACKEND=
DEV_MODEL=stub
OUT=
while (($#)); do
  case "$1" in
    --self-test) SELF_TEST=true; shift ;;
    --dev-backend) DEV_BACKEND="$2"; shift 2 ;;
    --dev-model) DEV_MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
if $SELF_TEST && [[ -n "$DEV_BACKEND" ]]; then echo "--self-test and --dev-backend are separate bundles" >&2; exit 2; fi
if $SELF_TEST; then DEFAULT_OUT="$HERE/build/bundle-self-test"
elif [[ -n "$DEV_BACKEND" ]]; then DEFAULT_OUT="$HERE/build/bundle-dev"
else DEFAULT_OUT="$HERE/build/bundle"; fi
OUT="${OUT:-$DEFAULT_OUT}"
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# The toolchain rust-toolchain.toml pins, with the wasm target (not listed there: the Nix toolchain
# hashes that file).
TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$ROOT/rust-toolchain.toml")"
rustup target list --installed --toolchain "$TOOLCHAIN" | grep -qx wasm32-unknown-unknown ||
  rustup target add wasm32-unknown-unknown --toolchain "$TOOLCHAIN"

# The wasm-bindgen CLI must be the version of the wasm-bindgen crate in Cargo.lock, exactly.
WB_VERSION="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$ROOT/Cargo.lock")"
WB="$HERE/.tools/wasm-bindgen-$WB_VERSION/bin/wasm-bindgen"
if [[ ! -x "$WB" ]]; then
  cargo "+$TOOLCHAIN" install wasm-bindgen-cli --version "=$WB_VERSION" --locked \
    --root "$HERE/.tools/wasm-bindgen-$WB_VERSION"
fi
[[ "$("$WB" --version)" == "wasm-bindgen $WB_VERSION" ]] || { echo "$WB is not $WB_VERSION" >&2; exit 1; }

# The `mobile` profile (Cargo.toml): fat LTO, one codegen unit, no incremental state. turso's
# query engine needs a larger shadow stack than wasm-ld's default 1 MiB, so the stack is 8 MiB.
# Paths are remapped so the bytes do not depend on where the checkout or the cargo home is. These
# flags are added to .cargo/config.toml's (cargo joins config arrays); a RUSTFLAGS variable would
# replace them all, so none may be set.
[[ -z "${RUSTFLAGS:-}${CARGO_ENCODED_RUSTFLAGS:-}" ]] || { echo "unset RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS" >&2; exit 1; }
# The standard library's sources map back to where its own paths point (/rustc/<commit>): with
# rust-src installed, generic std code would otherwise carry the local toolchain's path.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
SYSROOT="$(rustc "+$TOOLCHAIN" --print sysroot)"
COMMIT="$(rustc "+$TOOLCHAIN" -vV | sed -n 's/^commit-hash: //p')"
# The target directory too, last so it wins inside the checkout: build scripts' generated code
# (`include!` from OUT_DIR) carries its path into panic locations.
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
FLAGS="['-Clink-arg=-zstack-size=8388608', '--remap-path-prefix=$ROOT=/eidola', '--remap-path-prefix=$CARGO_HOME_DIR=/cargo', '--remap-path-prefix=$SYSROOT/lib/rustlib/src/rust=/rustc/$COMMIT', '--remap-path-prefix=$TARGET_DIR=/target']"
FEATURES=()
$SELF_TEST && FEATURES=(--features self-test)
[[ -n "$DEV_BACKEND" ]] && FEATURES=(--features dev)
(cd "$ROOT" && cargo "+$TOOLCHAIN" build --locked -p eidola-mobile --target wasm32-unknown-unknown \
  --profile mobile "${FEATURES[@]}" --config "target.wasm32-unknown-unknown.rustflags=$FLAGS")
WASM="$TARGET_DIR/wasm32-unknown-unknown/mobile/eidola_mobile.wasm"

rm -rf "$OUT"
mkdir -p "$OUT"
"$WB" "$WASM" --out-dir "$OUT" --target web --no-typescript --remove-name-section --remove-producers-section
if $SELF_TEST; then
  cp "$CRATE/web/self-test/index.html" "$CRATE/web/self-test/selftest.js" "$OUT/"
elif [[ -n "$DEV_BACKEND" ]]; then
  cp "$CRATE/web/index.html" "$CRATE/web/app.css" "$CRATE/web/dev/boot.js" "$OUT/"
  # The palette follows the device's light/dark switch, untinted, so the UI tests' screenshots
  # show each palette whatever the hour.
  python3 -c 'import json, sys; print(json.dumps({"base_url": sys.argv[1], "model": sys.argv[2], "appearance": "system", "tint": "off"}))' \
    "$DEV_BACKEND" "$DEV_MODEL" >"$OUT/dev-backend.json"
else
  cp "$CRATE/web/index.html" "$CRATE/web/app.css" "$CRATE/web/boot.js" "$OUT/"
fi
chmod -R u=rwX,go=rX "$OUT"

WASM_OUT="$OUT/eidola_mobile_bg.wasm"
echo "bundle: $OUT"
echo "wasm:   $(wc -c <"$WASM_OUT" | tr -d ' ') bytes, gzip -9 $(gzip -9n -c "$WASM_OUT" | wc -c | tr -d ' ')$(
  command -v brotli >/dev/null && echo ", brotli $(brotli -q 11 -c "$WASM_OUT" | wc -c | tr -d ' ')")"
echo "digest: $(python3 "$HERE/tools/bundle_digest.py" "$OUT")"
