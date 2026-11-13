#!/usr/bin/env bash
# One Rust version everywhere: the one the pinned StageX release builds the
# release images with. Fails unless
#   - oci/pallet-rust records exactly one `image` and one `rust` line;
#   - every `FROM stagex/pallet-rust...` line in oci/*/Containerfile names
#     that image (so the Containerfiles agree with each other); and
#   - rust-toolchain.toml's channel is that Rust version, so the native
#     toolchain that lints and tests the tree is the compiler that builds the
#     gateway, the cli and the inference node.
# The record itself is held to the image by the node's image build, which
# compares it with `rustc --version` inside the pallet.
#
#   scripts/check-rust-toolchain.sh [ROOT]   (default: the repository root)
#
# Bash and POSIX tools only; no network, no Docker.
set -euo pipefail

root="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
record="$root/oci/pallet-rust"
toolchain="$root/rust-toolchain.toml"
status=0
fail() { echo "error: $*" >&2; status=1; }

field() {
  local values
  values="$(grep -E "^$1 " "$record" | cut -d' ' -f2- || true)"
  if [ "$(printf '%s\n' "$values" | grep -c .)" -ne 1 ]; then
    echo "error: $record must have exactly one \`$1\` line" >&2
    exit 1
  fi
  printf '%s\n' "$values"
}

[ -f "$record" ] || { echo "error: $record is missing" >&2; exit 1; }
image="$(field image)"
rust="$(field rust)"
printf '%s\n' "$rust" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || fail "$record: \`rust\` is not a version: $rust"
printf '%s\n' "$image" | grep -Eq '^stagex/pallet-rust:[^@[:space:]]+@sha256:[0-9a-f]{64}$' \
  || fail "$record: \`image\` is not a tag and digest pin: $image"

found=0
for containerfile in "$root"/oci/*/Containerfile; do
  while read -r from ref _; do
    found=$((found + 1))
    [ "$ref" = "$image" ] \
      || fail "${containerfile#"$root/"} pins $ref, but oci/pallet-rust records $image"
  done < <(grep -E '^FROM[[:space:]]+stagex/pallet-rust[:@]' "$containerfile" || true)
done
[ "$found" -gt 0 ] || fail "no Containerfile pins stagex/pallet-rust; this check matched nothing"

channel="$(sed -nE 's/^channel[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/p' "$toolchain")"
[ "$(printf '%s\n' "$channel" | grep -c .)" -eq 1 ] \
  || fail "rust-toolchain.toml must set exactly one channel"
[ "$channel" = "$rust" ] \
  || fail "rust-toolchain.toml's channel is $channel, but the pinned StageX pallet-rust is Rust $rust (oci/pallet-rust)"

if [ "$status" -eq 0 ]; then
  echo "ok: Rust $rust everywhere ($found pallet-rust pins, rust-toolchain.toml)"
fi
exit "$status"
