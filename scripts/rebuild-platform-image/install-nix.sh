#!/usr/bin/env bash
# Run as `builder` with /nix owned by builder. Same pins as cvmimage nix/install.sh.
set -Eeuo pipefail
version=2.35.1
sha256=c3fe29778acaa93b5095ee66e36f11ec7c6a284c40970a24cc83ac4f04809db3
store_path=/nix/store/q6yfdws28aj556jlz5yayaggiddmb0b5-nix-2.35.1
release="nix-$version-x86_64-linux"
if [ -x "$store_path/bin/nix" ]; then echo "nix already installed"; exit 0; fi
tmp="$(mktemp -d)"
curl --fail --location --proto '=https' --tlsv1.2 -A curl/8 \
  --output "$tmp/$release.tar.xz" \
  "https://releases.nixos.org/nix/nix-$version/$release.tar.xz"
echo "$sha256  $tmp/$release.tar.xz" | sha256sum --check --strict
tar --extract --xz --file "$tmp/$release.tar.xz" --directory "$tmp"
"$tmp/$release/install" --no-daemon --yes --no-channel-add --no-modify-profile
rm -rf "$tmp"
nix="$HOME/.nix-profile/bin/nix"
[ "$(readlink -f "$nix")" = "$store_path/bin/nix" ] || { echo bad store path; exit 1; }
show() { "$nix" --extra-experimental-features nix-command config show "$1"; }
for kv in sandbox=true sandbox-fallback=false restrict-eval=true \
          allowed-uris=https://github.com/NixOS/nixpkgs/archive/; do
  [ "$(show "${kv%%=*}")" = "${kv#*=}" ] || { echo "bad setting $kv"; exit 1; }
done
"$nix" --version
