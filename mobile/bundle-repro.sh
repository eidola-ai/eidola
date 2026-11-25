#!/usr/bin/env bash
# The bundle's reproducibility check: two clean builds (each in a fresh target directory) must give
# the same files, byte for byte, and so the same bundle digest.
#   bundle-repro.sh [--self-test]
# Prints the digest. Builds from scratch twice, so it takes a few minutes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
for x in a b; do
  CARGO_TARGET_DIR="$T/target-$x" "$HERE/build-bundle.sh" "$@" --out "$T/$x" | tee "$T/$x.txt"
done
diff -r "$T/a" "$T/b"
A="$(sed -n 's/^digest: //p' "$T/a.txt")"
B="$(sed -n 's/^digest: //p' "$T/b.txt")"
[[ -n "$A" && "$A" == "$B" ]] || { echo "digests differ: $A $B" >&2; exit 1; }
echo "bundle reproduced: $A"
