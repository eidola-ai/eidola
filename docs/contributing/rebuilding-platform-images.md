# Rebuilding the platform's boot images

The gateway and every inference engine boot an image built by the hosting platform, from [`tinfoilsh/cvmimage`](https://github.com/tinfoilsh/cvmimage). Eidola pins each image by hash (`cvm-version` in a `tinfoil-config.yml`) and does not build it. This page is the procedure for rebuilding a pinned release from the platform's public source and comparing the result with what is pinned, and the record of the last time it was done. Rebuild before a release is attested and whenever a `cvm-version` changes, then update the note above `cvm-version` in each config that names the release.

The scripts are in [`scripts/rebuild-platform-image/`](../../scripts/rebuild-platform-image/) and name release `v0.15.3-rc.1`; a new release needs its tag, commit and Nix pins changed there.

## Last rebuild: v0.15.3-rc.1, 2026-11-11

**Every file in the release rebuilt byte-identical.** That is all 17 entries of `tinfoil-inference-v0.15.3-rc.1-checksums.sha256` (both variants' kernel, initrd, raw disk, both SEV-SNP and TDX IGVM images and their cvmc manifests, both release manifests, and `cvmc`), and the checksums file itself, compared with `cmp`.

| Artifact | SHA-256 (published = rebuilt) |
|---|---|
| `tinfoil-snp-v0.15.3-rc.1.igvm` (pinned `igvm_sha256`) | `8fe54c46ceb111b7cc41f12f60bd479211870c7bc5186c8566d71f286ad0d5ac` |
| `tinfoil-inference-v0.15.3-rc.1-manifest.json` (pinned in `cvm-version`) | `ddd2e76a629df1bd4ec507441a387ac5db010ebc1bda3e372756e095df8692eb` |
| `tinfoil-inference-v0.15.3-rc.1+gpu-manifest.json` | `1421cbb45dc428f6270d106d8eb68faab3ce7ca02ff2fbb2dc0968605004579f` |
| `tinfoil-tdx-v0.15.3-rc.1.igvm` | `ee12b6eefe090fb098818d593524651651a0b1aa3f7309fe7b485693fcde9099` |
| `tinfoil-snp-v0.15.3-rc.1+gpu.igvm` | `eacdbedae52b674e1fb71e3860aec72af85f4a55c51304f718ec4f47ce9c683a` |
| `tinfoil-tdx-v0.15.3-rc.1+gpu.igvm` | `f5c91b2b4ba764dc63a3a4c86a11c572afc4f0554a68a48cba8b42148435e615` |
| `tinfoil-inference-v0.15.3-rc.1.vmlinuz` | `2f16816ae0b14481bd56aba5b9d0046f188a35e3fd957b58b1885d6abb9edd80` |
| `tinfoil-inference-v0.15.3-rc.1+gpu.vmlinuz` | `965f632d95c74338ffa1f4c4ff5a5d82f8b0158a7cc7be3c1fd44c1504e8a1e0` |
| initrd (identical for both variants) | `c12763b2875338ddcc542cc9144408be0c8124792141465dec5c2e52aaeeeffa` |
| `tinfoil-inference-v0.15.3-rc.1.raw` | `55c990d76a8c206d24305a750aaf091083882fa305c05f75f5cbf61982e86d59` |
| `tinfoil-inference-v0.15.3-rc.1+gpu.raw` | `90f527685ad9d388175344953a11980a2970b76adbafa8dfce0daca1320610ec` |
| `cvmc-v0.15.3-rc.1` | `4ccaadaba53612b47ff11b0818f296dad7c643ae7427477813c5bbde4485fc73` |

The launch values that `cvmc measure` reports for the rebuilt images are the same as the manifest's: SEV-SNP `d712705d…6858` and TDX MRTD `cc56391d…577d` for the CPU image. The CPU dm-verity root hash is `7d1c8345…c4e9`.

## Source

- Repository: `https://github.com/tinfoilsh/cvmimage`
- Tag `v0.15.3-rc.1`: annotated tag object `0ddf1e5dc132da682f6f386e9df9427c7e88d2b6`, commit `99ec9ace36e52b38381e97216f56fc720d87a21f` (2026-11-03). The tag is not GPG-signed, so the commit hash is the pin.
- Provider docs read first: `README.md`, `docs/build.md` (section "Auditing the build"), `docs/launch.md` (section "Reproducible build"), `.github/workflows/release.yml` and `image/release-igvm.sh`.
- Pins at that commit: Nix 2.35.1 (`nix/nix-version`; tarball SHA-256 `c3fe2977…9db3`; store path `/nix/store/q6yfdws28aj556jlz5yayaggiddmb0b5-nix-2.35.1`), and nixpkgs `f49e816f5626998d0b70d713bd2cdc70d2207c56` (`sha256-MrPIcRrted6hW4zEgBLTM2iZBy6yjKU3FTW6YBX+XLw=`).

## Host used

- An Apple-silicon Mac running Docker Desktop 29.9.3. Its Linux VM had 16 GB of memory, and "Use Rosetta for x86_64/amd64 emulation" was on. Every build step ran as `linux/amd64` under Rosetta.
- No `--privileged`, no Docker socket, no home directory or credentials mounted. The containers got only the scratch directory, named volumes for `/nix` and the builder's home, and the default network. They need the network for substitutes and fixed-output fetches.

## Steps

Set `R` to an empty working directory outside the repository, and `S` to the repository's `scripts/rebuild-platform-image` directory (`Dockerfile`, `install-nix.sh`, `build.sh`, `compile-igvm.sh` and `release.sh`). Upstream code runs only inside the containers, which mount `R`, `S` read-only and two named volumes.

### 0. Baseline: fetch the published release and check it against the pins

```sh
V=v0.15.3-rc.1; mkdir -p $R/published && cd $R/published
for f in tinfoil-inference-$V-manifest.json tinfoil-inference-$V%2Bgpu-manifest.json tinfoil-inference-$V-checksums.sha256; do
  curl -sfL -A curl/8 -O "https://github.com/tinfoilsh/cvmimage/releases/download/$V/$f"; done
mv "tinfoil-inference-$V%2Bgpu-manifest.json" "tinfoil-inference-$V+gpu-manifest.json"
for f in cvmc-$V tinfoil-inference-$V.{initrd,vmlinuz,raw} tinfoil-inference-$V+gpu.{initrd,vmlinuz,raw} \
         tinfoil-{snp,tdx}-$V.igvm{,.manifest.json} tinfoil-{snp,tdx}-$V+gpu.igvm{,.manifest.json}; do
  curl -sfL -A curl/8 -o "$f" "https://images.tinfoil.sh/cvm/$(printf %s "$f" | sed 's/+/%2B/g')"; done
shasum -a 256 tinfoil-inference-$V-manifest.json    # must be ddd2e76a…92eb (the cvm-version pin)
sed 's#  \./#  #' tinfoil-inference-$V-checksums.sha256 | shasum -a 256 -c   # all OK; snp igvm = 8fe54c46…d5ac
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

`build.sh` clones the tag into the container's home volume and refuses to continue unless HEAD is `99ec9ace…` and the tree is clean. It then runs `nix-build -I . -A <attr>` for each argument it is given.

```sh
docker run --rm --platform linux/amd64 --memory 12g --cpus 12 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v $S:/scripts:ro \
  -u builder cvmrepro-builder:1 /scripts/build.sh cvm-compiler shipping-image
docker run --rm --platform linux/amd64 --memory 14g --cpus 12 -e JOBS=1 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v $S:/scripts:ro \
  -u builder cvmrepro-builder:1 /scripts/build.sh variants.gpu.shipping-image
```

### 3. Compile the IGVM images and the manifests with the provider's own script

`release.sh` runs `bash image/release-igvm.sh v0.15.3-rc.1 /out/release result-shipping-image result-variants-gpu-shipping-image result-cvm-compiler/bin/cvmc` without changes. The jq, coreutils and gawk it uses come from the pinned nixpkgs (jq 1.8.2). The script then checks the output against the published checksums file, and `cmp`s the regenerated checksums file with the published one.

```sh
docker run --rm --platform linux/amd64 --memory 8g $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v $R/out:/out \
  -v $R/published:/pub:ro -v $S:/scripts:ro \
  -u builder cvmrepro-builder:1 /scripts/release.sh
```

`compile-igvm.sh` is a smaller check that can run on its own with `--network none`. It is the same `cvmc build` and `cvmc measure` invocation over any kernel, initrd and root hash. Run with the rebuilt cvmc over the *published* kernel and initrd, it also reproduced `8fe54c46…`.

## Time and disk (this run)

| Step | Wall clock |
|---|---|
| `cvm-compiler` (cold store) | 2 min 13 s |
| CPU `shipping-image` (kernel, Go 1.27.3 from source, runtime, initrd, rootfs, erofs+verity) | 15 min |
| GPU `shipping-image` (re-uses the CPU's Go/initrd; GPU kernel, nvidia-open modules, nvattest) | 34 min (with `JOBS=1`; the first attempt with 4 jobs ran out of memory at 10 min) |
| `release-igvm.sh` (4 IGVM compiles, manifests, checksums) | 47 s |
| **Total, from clone to verdict** | about 1 h 15 min, including the failed attempt |

- Nix store volume: 18.1 GB. Builder home volume (source and result links): 0.2 GB.
- Working directory: about 8.5 GB, mostly four 2.17 GB `.raw` disks (two published, two rebuilt).
- Peak memory: the GPU kernel's final `vmlinux` link went past 12 GB while other jobs were running. Give the container at least 14 GB, or build one derivation at a time.

## Deviations from the provider's procedure

1. **Single-user Nix in a container rather than `nix/install.sh --daemon`.** The script needs systemd and refuses an existing `/nix`. The tarball, checksum, store path and `nix.conf` settings are the same.
2. **`filter-syscalls = false`.** Under Rosetta, Nix cannot load its x86_64 seccomp BPF filter ("unable to load seccomp BPF program: Invalid argument"). That filter blocks setuid/setgid and xattr/ACL creation inside builds. It is hardening, and nothing it blocks is recorded in a NAR. The outputs matched anyway. On a native x86_64 host, leave the setting at its default.
3. **Container flags `seccomp=unconfined` and `systempaths=unconfined`**, without `--privileged` and without added capabilities. The Nix sandbox needs them: Docker's seccomp profile refuses `unshare(CLONE_NEWUSER)`, and the masked `/proc` refuses the sandbox's proc mount.
4. **jq for `release-igvm.sh` taken from the pinned nixpkgs.** The provider's workflow uses whatever jq the runner has. The manifest bytes depend on jq's output format, and jq 1.8.2 reproduced them.
5. **x86_64 under Rosetta emulation**, not a native x86_64 host.
6. **No repeated build (`--check`) and no cross-host repeat** beyond this single rebuild, which was itself a cross-host comparison against the provider's CI.

## What this does and does not establish

What it establishes: the published bytes are what this commit's Nix graph produces. The SEV-SNP image Eidola pins (`8fe54c46…`) is a function of public source plus the hash-pinned external inputs listed below, and so is everything else in the release. A reproducible build proves this much and no more.

What it still trusts:

- **The Hydra binary cache (cache.nixos.org).** No substituters were disabled, matching the provider's configuration, so 482 store paths came signed from the cache: compilers, binutils, systemd-repart, erofs-utils and the bootstrap Go, among others. The measured outputs were built locally: the kernel, nvidia-open modules, Go 1.27.3, the tinfoil Go binaries, initrd, rootfs, disk images, cvmc and nvattest. A cache-free rebuild (`--option substituters ''`) would remove this trust, but would take far longer under emulation.
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
