#!/usr/bin/env bash
# Builds the release app bundle: the one a release records and publishes and the release APK embeds.
# It is built in the pinned StageX toolchain (oci/eidola-mobile-bundle/Containerfile, the bake target
# `mobile-bundle`), every build stage on linux/amd64, so its bytes do not depend on the host.
#   build-bundle-release.sh [--out DIR] [--builder NAME] [--no-cache] [--check | --update]
# --out       output directory (default build/bundle-release); replaced
# --builder   the buildx builder (default: the current one)
# --no-cache  build every stage again rather than reuse cached layers
# --check     fail unless the digest is the one bundle-release.digest records
# --update    write the digest to bundle-release.digest
# Prints the bundle digest (shell-contract.md → Bundle digest). Needs Docker with buildx; on a host
# that is not x86-64 the stages run emulated, which is slow but gives the same bytes.
#
# bundle-release.digest is the digest this tree's release bundle has, whatever host builds it: CI
# rebuilds it on another machine and compares (.github/workflows/mobile-bundle-release.yml). It
# moves with anything the bundle compiles: the app's crates and their dependencies, the trust root
# and model catalog app-core compiles in, the toolchain image. Update it with --update.
#
# build-bundle.sh is the fast development build on this machine's own toolchain. Its bundle is not
# this one: rustup's Rust and StageX's are the same release built differently (a different LLVM
# patch release, a standard library built by another build system, another version string in every
# crate's hash), so the code differs; see crates/eidola-mobile/AGENTS.md → The release bundle.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="$HERE/build/bundle-release"
BAKE_ARGS=()
RECORD=
while (($#)); do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --check) RECORD=check; shift ;;
    --update) RECORD=update; shift ;;
    --builder) BAKE_ARGS+=(--builder "$2"); shift 2 ;;
    --no-cache) BAKE_ARGS+=(--no-cache); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$(dirname "$OUT")"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

# Bake refuses to write outside its working directory unless the destination is granted, and the
# grant is exactly one fresh directory, moved into place once the build has succeeded.
STAGE="$(mktemp -d "$(dirname "$OUT")/.bundle-release.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
(cd "$ROOT" && docker buildx bake mobile-bundle "${BAKE_ARGS[@]}" \
  --allow "fs.write=$STAGE" --set "mobile-bundle.output=type=local,dest=$STAGE/bundle")
chmod -R u=rwX,go=rX "$STAGE/bundle"
rm -rf "$OUT"
mv "$STAGE/bundle" "$OUT"

WASM_OUT="$OUT/eidola_mobile_bg.wasm"
echo "bundle: $OUT"
echo "wasm:   $(wc -c <"$WASM_OUT" | tr -d ' ') bytes, gzip -9 $(gzip -9n -c "$WASM_OUT" | wc -c | tr -d ' ')"
DIGEST="$(python3 -I "$HERE/tools/bundle_digest.py" "$OUT")"
echo "digest: $DIGEST"
case "$RECORD" in
  check)
    WANT="$(tr -d '[:space:]' <"$HERE/bundle-release.digest")"
    if [[ "$DIGEST" != "$WANT" ]]; then
      echo "the release bundle's digest is $DIGEST, but bundle-release.digest records $WANT" >&2
      (cd "$OUT" && find . -type f | LC_ALL=C sort | while read -r f; do
        python3 -I -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(), sys.argv[1])' "$f"
      done) >&2
      echo "If a change to its inputs is why, rebuild it (just mobile-bundle-release --update) and commit the file." >&2
      exit 1
    fi
    echo "matches bundle-release.digest" ;;
  update)
    echo "$DIGEST" >"$HERE/bundle-release.digest"
    echo "wrote bundle-release.digest" ;;
esac
