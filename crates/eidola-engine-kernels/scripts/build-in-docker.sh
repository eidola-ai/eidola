#!/usr/bin/env bash
# Build the kernel set on a GPU-less Linux builder: Nix inside a pinned
# nixos/nix container. Works on any Docker host, including Apple Silicon
# (linux/arm64 natively, linux/amd64 under emulation); cubins do not depend on
# the builder's CPU architecture, which is exactly what building on both
# checks.
#
#   scripts/build-in-docker.sh [--platform linux/arm64|linux/amd64] [--out DIR]
#                              [--check] [--verify | --update]
#
#   --platform  builder architecture (default: the Docker host's)
#   --out       copy the build output here (default: target/engine-kernels/<arch>)
#   --check     after building, rebuild the derivation from scratch and fail
#               unless the second build is byte-identical (nix-build --check)
#   --verify    fail unless the built manifest equals the committed
#               kernels.manifest.json byte for byte
#   --update    replace the committed kernels.manifest.json with the built one
#
# The Nix store lives in a named volume per architecture
# (eidola-kernels-nix-<arch>) so the multi-gigabyte toolkit download
# happens once.
set -euo pipefail

# nixos/nix 2.32.4, multi-arch index digest.
image="nixos/nix:2.32.4@sha256:0d9c872db1ca2f3eaa4a095baa57ed9b72c09d53a0905a4428813f61f0ea98db"

crate_dir=$(cd "$(dirname "$0")/.." && pwd)
repo_dir=$(cd "$crate_dir/../.." && pwd)

platform=""
out=""
check=0
mode=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --platform) platform=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --check) check=1; shift ;;
    --verify) mode=verify; shift ;;
    --update) mode=update; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z $platform ]]; then
  case $(docker info --format '{{.Architecture}}') in
    aarch64 | arm64) platform=linux/arm64 ;;
    *) platform=linux/amd64 ;;
  esac
fi
arch=${platform#linux/}
out=${out:-$repo_dir/target/engine-kernels/$arch}
mkdir -p "$out"
out=$(cd "$out" && pwd)

# The container is the isolation boundary: Nix's own sandbox needs privileges
# a default container lacks, and its seccomp filter cannot be loaded under
# amd64 emulation on an arm64 host. Inputs are still fixed by hash, and the
# build script makes output independent of the build directory.
# (Set in nix.conf: on the command line filter-syscalls does not reach every
# build Nix starts.)
nix_conf='sandbox = false\nfilter-syscalls = false\nmax-jobs = 1\ncores = 0\n'
check_cmd=":"
if [[ $check -eq 1 ]]; then
  check_cmd="nix-build --check /src/nix"
fi

docker run --rm --platform "$platform" \
  -v "eidola-kernels-nix-$arch:/nix" \
  -v "$crate_dir:/src:ro" \
  -v "$out:/out" \
  "$image" sh -euc "
    printf '$nix_conf' >>/etc/nix/nix.conf
    nix-build /src/nix -o /tmp/result
    $check_cmd
    rm -rf /out/cubin /out/fatbin /out/inspect /out/manifest.json
    cp -rL /tmp/result/. /out/
    chmod -R u+w /out
    readlink -f /tmp/result > /out/store-path
  "
echo "build output: $out"

committed="$crate_dir/kernels.manifest.json"
case $mode in
  verify)
    if cmp -s "$out/manifest.json" "$committed"; then
      echo "manifest matches kernels.manifest.json"
    else
      echo "built manifest differs from kernels.manifest.json:" >&2
      diff -u "$committed" "$out/manifest.json" >&2 || true
      exit 1
    fi
    ;;
  update)
    cp "$out/manifest.json" "$committed"
    echo "updated $committed"
    ;;
esac
