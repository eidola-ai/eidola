#!/usr/bin/env bash
# Builds the app bundle the shells embed: eidola-mobile for wasm32-unknown-unknown on the pinned
# stable toolchain, bound by the wasm-bindgen CLI at exactly the lockfile's version, plus the page
# that loads it. No inline script anywhere (the shells' CSP refuses them).
#   build-bundle.sh [--self-test] [--out DIR]
# --self-test  the self-test bundle (feature self-test, its own page); a release bundle has none of it
# --out        output directory (default build/bundle, or build/bundle-self-test)
# Prints the bundle digest (shell-contract.md → Bundle digest) and the sizes. CARGO_TARGET_DIR is
# honoured; bundle-repro.sh builds twice in fresh ones and compares.
set -euo pipefail
export LC_ALL=C TZ=UTC
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CRATE="$ROOT/crates/eidola-mobile"
SELF_TEST=false
OUT=
while (($#)); do
  case "$1" in
    --self-test) SELF_TEST=true; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
$SELF_TEST && DEFAULT_OUT="$HERE/build/bundle-self-test" || DEFAULT_OUT="$HERE/build/bundle"
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
FLAGS="['-Clink-arg=-zstack-size=8388608', '--remap-path-prefix=$ROOT=/eidola', '--remap-path-prefix=$CARGO_HOME_DIR=/cargo', '--remap-path-prefix=$SYSROOT/lib/rustlib/src/rust=/rustc/$COMMIT']"
FEATURES=()
$SELF_TEST && FEATURES=(--features self-test)
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
(cd "$ROOT" && cargo "+$TOOLCHAIN" build --locked -p eidola-mobile --target wasm32-unknown-unknown \
  --profile mobile "${FEATURES[@]}" --config "target.wasm32-unknown-unknown.rustflags=$FLAGS")
WASM="$TARGET_DIR/wasm32-unknown-unknown/mobile/eidola_mobile.wasm"

rm -rf "$OUT"
mkdir -p "$OUT"
"$WB" "$WASM" --out-dir "$OUT" --target web --no-typescript --remove-name-section --remove-producers-section
if $SELF_TEST; then
  cp "$CRATE/web/self-test/index.html" "$CRATE/web/self-test/selftest.js" "$OUT/"
else
  cp "$CRATE/web/index.html" "$CRATE/web/boot.js" "$OUT/"
fi
chmod -R u=rwX,go=rX "$OUT"

WASM_OUT="$OUT/eidola_mobile_bg.wasm"
echo "bundle: $OUT"
echo "wasm:   $(wc -c <"$WASM_OUT" | tr -d ' ') bytes, gzip -9 $(gzip -9n -c "$WASM_OUT" | wc -c | tr -d ' ')$(
  command -v brotli >/dev/null && echo ", brotli $(brotli -q 11 -c "$WASM_OUT" | wc -c | tr -d ' ')")"
echo "digest: $(python3 "$HERE/tools/bundle_digest.py" "$OUT")"
