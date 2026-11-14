#!/usr/bin/env bash
# Runs the provider's image/release-igvm.sh unmodified over the rebuilt outputs.
# jq/coreutils/gawk come from the release's pinned nixpkgs (the provider's CI
# uses the runner's host tools here). Output: /out/release.
set -Eeuo pipefail
export USER=builder PATH="$HOME/.nix-profile/bin:$PATH"
cd "$HOME/cvmimage"
tools=$(nix-build --no-out-link -I . -E '
  let l = builtins.fromJSON (builtins.readFile ./nixpkgs.lock.json);
      p = import (builtins.fetchTarball { inherit (l) url sha256; }) { config = {}; overlays = []; };
  in p.symlinkJoin { name = "release-tools"; paths = [ p.jq p.coreutils p.gawk p.bash ]; }')
export PATH="$tools/bin:$PATH"
jq --version
rm -rf /out/release
bash image/release-igvm.sh v0.15.3-rc.1 /out/release result-shipping-image result-variants-gpu-shipping-image result-cvm-compiler/bin/cvmc
cd /out/release && sha256sum -c /pub/tinfoil-inference-v0.15.3-rc.1-checksums.sha256
cmp tinfoil-inference-v0.15.3-rc.1-checksums.sha256 /pub/tinfoil-inference-v0.15.3-rc.1-checksums.sha256 && echo "checksums file byte-identical"
