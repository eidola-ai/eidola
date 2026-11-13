#!/usr/bin/env bash
# Teeth for scripts/check-rust-toolchain.sh: it passes on a copy of this tree
# and fails on each kind of drift it exists to catch. Bash and POSIX tools
# only; no network, no Docker.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
check="$root/scripts/check-rust-toolchain.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fresh() {
  rm -rf "$work/tree"
  mkdir -p "$work/tree/oci"
  cp "$root/rust-toolchain.toml" "$work/tree/"
  cp "$root/oci/pallet-rust" "$work/tree/oci/"
  for containerfile in "$root"/oci/*/Containerfile; do
    dir="$work/tree/oci/$(basename "$(dirname "$containerfile")")"
    mkdir -p "$dir"
    cp "$containerfile" "$dir/"
  done
}

# Rewrite a file in place with a sed expression (BSD and GNU sed alike).
edit() {
  sed -E "$2" "$1" >"$1.new"
  mv "$1.new" "$1"
}

failures=0
expect() {
  local want="$1" name="$2" got=pass
  "$check" "$work/tree" >/dev/null 2>"$work/err" || got=fail
  if [ "$got" = "$want" ]; then
    echo "ok: $name ($got)"
  else
    echo "FAIL: $name: expected $want, got $got" >&2
    sed 's/^/  /' "$work/err" >&2
    failures=$((failures + 1))
  fi
}

fresh
expect pass "this tree"

fresh
edit "$work/tree/rust-toolchain.toml" 's/^channel = "[^"]*"/channel = "1.999.0"/'
expect fail "the toolchain moved without the StageX release"

fresh
edit "$work/tree/oci/pallet-rust" 's/^rust .*/rust 1.999.0/'
expect fail "the recorded StageX Rust moved without the toolchain"

fresh
edit "$work/tree/oci/eidola-server-gateway/Containerfile" 's/(FROM stagex\/pallet-rust:[^@]*@sha256:)[0-9a-f]{64}/\10000000000000000000000000000000000000000000000000000000000000000/'
expect fail "one Containerfile pins another pallet-rust"

fresh
printf 'rust 1.97.1\n' >>"$work/tree/oci/pallet-rust"
expect fail "the record states the Rust version twice"

fresh
rm "$work/tree/oci/pallet-rust"
expect fail "the record is missing"

fresh
for containerfile in "$work"/tree/oci/*/Containerfile; do
  edit "$containerfile" '/^FROM stagex\/pallet-rust/d'
done
expect fail "no Containerfile pins pallet-rust"

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
