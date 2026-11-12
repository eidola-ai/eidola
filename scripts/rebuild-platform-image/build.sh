#!/usr/bin/env bash
# Runs inside the builder container as `builder`. Args: list of nix attrs to build.
set -Eeuo pipefail
export USER=builder PATH="$HOME/.nix-profile/bin:$PATH"
tag=v0.15.3-rc.1
commit=99ec9ace36e52b38381e97216f56fc720d87a21f
src="$HOME/cvmimage"
if [ ! -d "$src" ]; then
  git -c credential.helper= clone --quiet --branch "$tag" https://github.com/tinfoilsh/cvmimage.git "$src"
fi
cd "$src"
[ "$(git rev-parse HEAD)" = "$commit" ] || { echo "wrong commit"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "dirty tree"; exit 1; }
for attr in "$@"; do
  out="result-${attr//./-}"
  echo "=== $(date -u +%FT%TZ) building $attr"
  start=$(date +%s)
  nix-build -I . -A "$attr" -o "$out" --cores "${CORES:-0}" --max-jobs "${JOBS:-4}"
  echo "=== $(date -u +%FT%TZ) built $attr in $(( $(date +%s) - start ))s -> $(readlink "$out")"
done
