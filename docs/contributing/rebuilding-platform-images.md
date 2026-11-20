# Rebuilding the platform's boot images

The gateway and every inference engine boot an image built by the hosting platform, from [`tinfoilsh/cvmimage`](https://github.com/tinfoilsh/cvmimage). Eidola pins each image by hash (`cvm-version` in a `tinfoil-config.yml`) and does not build it. This page is the procedure for rebuilding a pinned release from the platform's public source and comparing the result with what is pinned, and the record of the last time it was done. Rebuild before a release is attested and whenever a `cvm-version` changes, then update the note above `cvm-version` in each config that names the release.

The scripts are in [`scripts/rebuild-platform-image/`](../../scripts/rebuild-platform-image/) and name release `v0.15.4-rc.2`; a new release needs its tag, commit and Nix pins changed there.

## Rebuilt releases

Every release rebuilt so far, with the SHA-256 of each release manifest (the CPU one is what the gateway's `cvm-version` pins, the GPU one what an engine's pins). A test fixture captured under a release names it here (`tinfoil-verifier`'s `the_gateway_capture_record_is_the_release_it_names`), so keep a row when its release is no longer pinned.

| Release | Rebuilt | CPU manifest SHA-256 | GPU manifest SHA-256 | Result |
|---|---|---|---|---|
| v0.15.4-rc.2 | 2026-11-20 | `689c945f62d8f8507d5051dfdc97abc019e7ea5c9cc153557fe8b6850db45dc2` | `de67844398b970d5bbddd9ab42f7cded933b49d268ca0b7f8e86861b2e1f2094` | every file byte-identical |
| v0.15.3-rc.1 | 2026-11-11 | `ddd2e76a629df1bd4ec507441a387ac5db010ebc1bda3e372756e095df8692eb` | `1421cbb45dc428f6270d106d8eb68faab3ce7ca02ff2fbb2dc0968605004579f` | every file byte-identical |

## Last rebuild: v0.15.4-rc.2, 2026-11-20

**Every file in the release rebuilt byte-identical.** That is all 17 entries of `tinfoil-inference-v0.15.4-rc.2-checksums.sha256` (both variants' kernel, initrd, raw disk, both SEV-SNP and TDX IGVM images and their cvmc manifests, both release manifests, and `cvmc`), and the checksums file itself, compared with `cmp`.

| Artifact | SHA-256 (published = rebuilt) |
|---|---|
| `tinfoil-snp-v0.15.4-rc.2.igvm` (pinned `igvm_sha256`) | `112501c5b3fa891cf05c96841903b000bd012bfa13cdc780692ce8f625470216` |
| `tinfoil-inference-v0.15.4-rc.2-manifest.json` (pinned in the gateway's `cvm-version`) | `689c945f62d8f8507d5051dfdc97abc019e7ea5c9cc153557fe8b6850db45dc2` |
| `tinfoil-inference-v0.15.4-rc.2+gpu-manifest.json` (pinned in the engines' `cvm-version`) | `de67844398b970d5bbddd9ab42f7cded933b49d268ca0b7f8e86861b2e1f2094` |
| `tinfoil-tdx-v0.15.4-rc.2.igvm` | `460aae05ee9d877d6a0a21e17520085fbb44e048d08dd905058081761362bfc5` |
| `tinfoil-snp-v0.15.4-rc.2+gpu.igvm` | `c29b3fa99cea7d6fa03a6e757a371363b31a5487bb23059189d9c1f5a5699a6b` |
| `tinfoil-tdx-v0.15.4-rc.2+gpu.igvm` | `9286ab924473ed6cad59bd7e22dd7833adb93034bb0e8886c1318c66ba8a61f2` |
| `tinfoil-inference-v0.15.4-rc.2.vmlinuz` | `e71c23ddd20d70972dd215bb188b35bce907c291c10ae5070683818832b57915` |
| `tinfoil-inference-v0.15.4-rc.2+gpu.vmlinuz` | `3819a7d90aa695fc70c54c10716c038f637990192eb3b6693a73a4b060705c82` |
| initrd (identical for both variants, and to v0.15.3-rc.1's) | `c12763b2875338ddcc542cc9144408be0c8124792141465dec5c2e52aaeeeffa` |
| `tinfoil-inference-v0.15.4-rc.2.raw` | `baff9c8040bd4fd647f35bbaae114ee3ddefe77bbccbc86dfe0928a58f6fc564` |
| `tinfoil-inference-v0.15.4-rc.2+gpu.raw` | `99e2945cfd16e78ea19eae2a2856170f797305b98ca0021fab4c650faf6d04e4` |
| `cvmc-v0.15.4-rc.2` | `00e439af8d52d8ae303ac917f852044acb316344061c511f085a2c15c4ba05a0` |

The launch values in the rebuilt manifests are the published ones: for the CPU image SEV-SNP `cea56f9f…c015` and TDX MRTD `e3d165d3…017d`, for the GPU image SEV-SNP `49ed764e…5df0` and MRTD `5d4ccd0d…4de5`. `measure-enclave`'s own measurement reproduces all four from the images (its known-answer tests). The CPU dm-verity root hash is `3fcb9f57…970e`.

What changed from v0.15.3-rc.1 in the measured images: the boot processor's VMSA turns on Secure TSC (SEV_FEATURES `0x201`), `cvmc` appends `no-kvmclock` to every command line beside `no5lvl`, two kernel patches refuse host-supplied or host-adjusted time (`kernel-ignore-host-paravirtual-features.patch`, `kernel-require-protected-time.patch`), and the guest's Go programs add a boot time anchor and a bounded connection drain. The SEV-SNP guest policy stays `0x30133`. The Nix and nixpkgs pins, the package locks and the kernel source are unchanged.

## Source

- Repository: `https://github.com/tinfoilsh/cvmimage`
- Tag `v0.15.4-rc.2`: annotated tag object `cb1e3df5fec99e7d362f8e109e9e3ace6c556be1`, commit `0663ad9e0e72dd675a9846fd2e1d831109b3637b` (2026-11-18). The tag is not GPG-signed (GitHub reports it unsigned), so the commit hash is the pin.
- Provider docs read first: `README.md`, `docs/build.md` (section "Auditing the build"), `docs/launch.md` (section "Reproducible build"), `.github/workflows/release.yml` and `image/release-igvm.sh`.
- Pins at that commit: Nix 2.35.1 (`nix/nix-version`; tarball SHA-256 `c3fe2977…9db3`; store path `/nix/store/q6yfdws28aj556jlz5yayaggiddmb0b5-nix-2.35.1`), and nixpkgs `f49e816f5626998d0b70d713bd2cdc70d2207c56` (`sha256-MrPIcRrted6hW4zEgBLTM2iZBy6yjKU3FTW6YBX+XLw=`).

## Host used

- An Apple-silicon Mac running Docker Desktop 29.9.3. Its Linux VM had 16 GB of memory, and "Use Rosetta for x86_64/amd64 emulation" was on. Every build step ran as `linux/amd64` under Rosetta.
- No `--privileged`, no Docker socket, no home directory or credentials mounted. The containers got only the scratch directory, named volumes for `/nix` and the builder's home, and the default network. They need the network for substitutes and fixed-output fetches.

## Steps

Set `R` to an empty working directory outside the repository, and `S` to the repository's `scripts/rebuild-platform-image` directory (`Dockerfile`, `install-nix.sh`, `build.sh`, `compile-igvm.sh` and `release.sh`). Upstream code runs only inside the containers, which mount `R`, `S` read-only and two named volumes.

### 0. Baseline: fetch the published release and check it against the pins

```sh
V=v0.15.4-rc.2; mkdir -p $R/published && cd $R/published
for f in tinfoil-inference-$V-manifest.json tinfoil-inference-$V%2Bgpu-manifest.json tinfoil-inference-$V-checksums.sha256; do
  curl -sfL -A curl/8 -O "https://github.com/tinfoilsh/cvmimage/releases/download/$V/$f"; done
mv "tinfoil-inference-$V%2Bgpu-manifest.json" "tinfoil-inference-$V+gpu-manifest.json"
for f in cvmc-$V tinfoil-inference-$V.{initrd,vmlinuz,raw} tinfoil-inference-$V+gpu.{initrd,vmlinuz,raw} \
         tinfoil-{snp,tdx}-$V.igvm{,.manifest.json} tinfoil-{snp,tdx}-$V+gpu.igvm{,.manifest.json}; do
  curl -sfL -A curl/8 -o "$f" "https://images.tinfoil.sh/cvm/$(printf %s "$f" | sed 's/+/%2B/g')"; done
shasum -a 256 tinfoil-inference-$V-manifest.json    # must be 689c945f…5dc2 (the gateway's cvm-version pin)
sed 's#  \./#  #' tinfoil-inference-$V-checksums.sha256 | shasum -a 256 -c   # all OK; snp igvm = 112501c5…0216
```

The download is about 4.4 GB, mostly the two 2.17 GB `.raw` disks.

### 1. Builder image and pinned Nix

```sh
cd $S
docker build --platform linux/amd64 -t cvmrepro-builder:1 .
#   FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
#   + curl, xz-utils, ca-certificates, git (apt, unpinned: used only to fetch the checksum-verified Nix tarball and the source)
#   /etc/nix/nix.conf: sandbox=true, sandbox-fallback=false, restrict-eval=true,
#   allowed-uris=https://github.com/NixOS/nixpkgs/archive/, filter-syscalls=false (see Deviations)
docker volume create cvmrepro-nix
docker run --rm --platform linux/amd64 -v cvmrepro-nix:/nix debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc chown 1000:1000 /nix
SEC="--security-opt seccomp=unconfined --security-opt systempaths=unconfined"
docker run --rm --platform linux/amd64 $SEC -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder \
  -u builder cvmrepro-builder:1 install-nix.sh
```

`install-nix.sh` follows `nix/install.sh`: it fetches Nix 2.35.1, checks the tarball against `c3fe2977…`, installs it, and asserts the store path and the four settings. Two changes: it installs single-user (`--no-daemon`), and it gets past the pre-existing-Nix refusal by checking the store path instead.

### 2. Build (inside the container, as `builder`)

`build.sh` clones the tag into the container's home volume and refuses to continue unless HEAD is `0663ad9e…` and the tree is clean. It then runs `nix-build -I . -A <attr>` for each argument it is given.

```sh
docker run --rm --platform linux/amd64 --memory 12g --cpus 12 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v $S:/scripts:ro \
  -u builder cvmrepro-builder:1 /scripts/build.sh cvm-compiler shipping-image
docker run --rm --platform linux/amd64 --memory 14g --cpus 12 -e JOBS=1 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v $S:/scripts:ro \
  -u builder cvmrepro-builder:1 /scripts/build.sh variants.gpu.shipping-image
```

### 3. Compile the IGVM images and the manifests with the provider's own script

`release.sh` runs `bash image/release-igvm.sh v0.15.4-rc.2 /out/release result-shipping-image result-variants-gpu-shipping-image result-cvm-compiler/bin/cvmc` without changes. The jq, coreutils and gawk it uses come from the pinned nixpkgs (jq 1.8.2). The script then checks the output against the published checksums file, and `cmp`s the regenerated checksums file with the published one.

```sh
docker run --rm --platform linux/amd64 --memory 8g $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v $R/out:/out \
  -v $R/published:/pub:ro -v $S:/scripts:ro \
  -u builder cvmrepro-builder:1 /scripts/release.sh
```

`compile-igvm.sh` is a smaller check that can run on its own with `--network none`. It is the same `cvmc build` and `cvmc measure` invocation over any kernel, initrd and root hash. Run with v0.15.3-rc.1's rebuilt cvmc over that release's *published* kernel and initrd, it also reproduced that release's `8fe54c46…`; it was not rerun for v0.15.4-rc.2, whose whole release step reproduced.

## Time and disk (this run)

| Step | Wall clock |
|---|---|
| `cvm-compiler` (cold store) | 3 min 31 s |
| CPU `shipping-image` (kernel, Go 1.27.3 from source, runtime, initrd, rootfs, erofs+verity) | 13 min 53 s |
| GPU `shipping-image` (re-uses the CPU's Go/initrd; GPU kernel, nvidia-open modules, nvattest) | 42 min 14 s (with `JOBS=1` from the start) |
| `release-igvm.sh` (4 IGVM compiles, manifests, checksums) | 47 s |
| **Total, from builder image to verdict** | about 1 h 2 min |

- Nix store volume: 14.8 GB. Builder home volume (source and result links): 0.2 GB.
- Working directory: about 8.5 GB, mostly four 2.17 GB `.raw` disks (two published, two rebuilt).
- Peak memory: on the v0.15.3-rc.1 rebuild the GPU kernel's final `vmlinux` link went past 12 GB while other jobs were running, and a 4-job GPU build ran out of memory. Give the container at least 14 GB and build the GPU variant one derivation at a time (`JOBS=1`), as above.

## Deviations from the provider's procedure

1. **Single-user Nix in a container rather than `nix/install.sh --daemon`.** The script needs systemd and refuses an existing `/nix`. The tarball, checksum, store path and `nix.conf` settings are the same.
2. **`filter-syscalls = false`.** Under Rosetta, Nix cannot load its x86_64 seccomp BPF filter ("unable to load seccomp BPF program: Invalid argument"). That filter blocks setuid/setgid and xattr/ACL creation inside builds. It is hardening, and nothing it blocks is recorded in a NAR. The outputs matched anyway. On a native x86_64 host, leave the setting at its default.
3. **Container flags `seccomp=unconfined` and `systempaths=unconfined`**, without `--privileged` and without added capabilities. The Nix sandbox needs them: Docker's seccomp profile refuses `unshare(CLONE_NEWUSER)`, and the masked `/proc` refuses the sandbox's proc mount.
4. **jq for `release-igvm.sh` taken from the pinned nixpkgs.** The provider's workflow uses whatever jq the runner has. The manifest bytes depend on jq's output format, and jq 1.8.2 reproduced them.
5. **x86_64 under Rosetta emulation**, not a native x86_64 host.
6. **No repeated build (`--check`) and no cross-host repeat** beyond this single rebuild, which was itself a cross-host comparison against the provider's CI.

## What this does and does not establish

What it establishes: the published bytes are what this commit's Nix graph produces. The SEV-SNP image the gateway pins (`112501c5…`) is a function of public source plus the hash-pinned external inputs listed below, and so is everything else in the release. A reproducible build proves this much and no more.

What it still trusts:

- **The Hydra binary cache (cache.nixos.org).** No substituters were disabled, matching the provider's configuration, so 481 store paths came signed from the cache (482 for v0.15.3-rc.1): compilers, binutils, systemd-repart, erofs-utils and the bootstrap Go, among others. The measured outputs were built locally: the kernel, nvidia-open modules, Go 1.27.3, the tinfoil Go binaries, initrd, rootfs, disk images, cvmc and nvattest. A cache-free rebuild (`--option substituters ''`) would remove this trust, but would take far longer under emulation.
- **Binary archives in the measured rootfs, pinned by hash but not built from source:**
  - Ubuntu 26.04 debs from snapshot.ubuntu.com@20261027 (libc6, ca-certificates, e2fsprogs, iproute2, nftables and similar; 93 CPU and 97 GPU lock entries).
  - `busybox-static` from Ubuntu.
  - Docker's prebuilt static binaries (`dockerd`, `containerd`, `containerd-shim-runc-v2`, `runc` from download.docker.com `docker-29.6.2.tgz`), in both variants.
  - GPU only: NVIDIA's userspace libraries, fabric manager, persistenced, container toolkit and the GPU firmware blobs (NVIDIA CUDA apt repo, 595.71.05).
  - The kernel is built from Ubuntu's `linux-source-7.0.0` deb, which is source.
  - The NVIDIA kernel modules are built from the open-source `nvidia-open` tree.
- **Network during the build:** only fixed-output derivations, which declare their hash before fetching, and the pinned nixpkgs tarball (restrict-eval with one allowed URI). The sandbox was on.
- **Release tooling outside Nix:** the workflow's `release-igvm.sh` runs on the runner with the runner's jq and sha256sum.

Not checked here: the Sigstore build-provenance attestations on the release files (`scripts/artifact-manifest.sh --verify-attestations` covers those), and a GPU IGVM run on hardware.

## Cleaning up

```sh
docker volume rm cvmrepro-nix cvmrepro-home && docker image rm cvmrepro-builder:1
```
