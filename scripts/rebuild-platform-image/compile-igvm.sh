#!/usr/bin/env bash
# compile-igvm.sh <cvmc> <kernel> <initrd> <roothash-file-or-hex> <outdir> <artifact_version>
# Same cvmc invocation as image/release-igvm.sh at v0.15.3-rc.1.
set -Eeuo pipefail
cvmc="$1" kernel="$2" initrd="$3" root="$4" out="$5" ver="$6"
[ -f "$root" ] && root="$(tr -d '\n' < "$root")"
mkdir -p "$out"
cmdline="root=/dev/mapper/root roothash=$root pci=nocrs pcie_ports=compat"
for p in snp tdx; do
  "$cvmc" build --platform "$p" --kernel "$kernel" --initramfs "$initrd" \
    --cmdline "$cmdline" --output "$out/tinfoil-$p-$ver.igvm"
  "$cvmc" measure "$out/tinfoil-$p-$ver.igvm"
done
sha256sum "$out"/*
