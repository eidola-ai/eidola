# Rebuilding the platform's boot images

The gateway and every inference engine boot confidential-VM images built by our hosting platform from [`tinfoilsh/cvmimage`](https://github.com/tinfoilsh/cvmimage). Eidola pins each image by hash (`cvm-version` in each `tinfoil-config.yml`) rather than building it in our own release pipeline.

This runbook documents how to independently rebuild a pinned `cvmimage` release from its public source, confirm byte-for-byte parity with the published files, and record the result.

**When to run this:** Before attesting a release and whenever any `cvm-version` pin changes. After verifying a new release, update the rebuild comment above `cvm-version` in every `tinfoil-config.yml` that pins it (note that editing a config file changes its SHA-256 and therefore its measured `HOST_DATA` / `MRCONFIGID`), prepend a row to **Rebuilt releases** below, and replace the **Last rebuild log** section at the bottom.

Helper scripts live in [`scripts/rebuild-platform-image/`](../../scripts/rebuild-platform-image/) (`Dockerfile`, `install-nix.sh`, `build.sh`, `compile-igvm.sh`, and `release.sh`). When bumping to a new `cvmimage` release, update its tag, commit hash, and Nix pins in those scripts first.

## Rebuilt releases

Every `cvmimage` release rebuilt so far, with the SHA-256 of its CPU release manifest (pinned by the gateway's `cvm-version`) and GPU release manifest (pinned by GPU engines' `cvm-version`).

> [!IMPORTANT]
> **Do not delete older rows.** `crates/tinfoil-verifier`'s `the_gateway_capture_record_is_the_release_it_names` test parses this table directly to assert that every captured gateway fixture belongs to a rebuilt release listed here.

| Release | Rebuilt | CPU manifest SHA-256 | GPU manifest SHA-256 | Result |
| --- | --- | --- | --- | --- |
| v0.15.4-rc.2 | 2026-11-20 | `689c945f62d8f8507d5051dfdc97abc019e7ea5c9cc153557fe8b6850db45dc2` | `de67844398b970d5bbddd9ab42f7cded933b49d268ca0b7f8e86861b2e1f2094` | every file byte-identical |
| v0.15.3-rc.1 | 2026-11-11 | `ddd2e76a629df1bd4ec507441a387ac5db010ebc1bda3e372756e095df8692eb` | `1421cbb45dc428f6270d106d8eb68faab3ce7ca02ff2fbb2dc0968605004579f` | every file byte-identical |

## Procedure

### Prerequisites

- **Docker** (native `linux/amd64`, or Apple Silicon with Docker Desktop + Rosetta x86_64 emulation enabled).
- **Memory:** Allocate at least **16 GB** to the Docker VM (14 GB+ for the GPU build container — the GPU kernel's final `vmlinux` link exceeds 12 GB under parallel jobs, so the GPU build runs with `JOBS=1`).
- **Disk:** About **25 GB** free (~15 GB for the Nix store volume and ~8.5 GB in the scratch directory for published and rebuilt `.raw` disks).
- **Isolation:** Do not pass `--privileged`, do not mount the Docker socket, and do not mount your home directory or credentials. Containers receive only the scratch directory, named volumes for `/nix` and the builder's home, and network access for Nix substitutes and fixed-output fetches.

Set `V` to the `cvmimage` release tag, `R` to an empty working directory outside the repository, and `S` to the repository's `scripts/rebuild-platform-image` directory.

### 0. Fetch the published release and check it against our pins

```sh
V=v0.15.4-rc.2
mkdir -p "$R/published" && cd "$R/published"

for f in \
  "tinfoil-inference-$V-manifest.json" \
  "tinfoil-inference-$V%2Bgpu-manifest.json" \
  "tinfoil-inference-$V-checksums.sha256"
do
  curl -sfL -A curl/8 -O "https://github.com/tinfoilsh/cvmimage/releases/download/$V/$f"
done
mv "tinfoil-inference-$V%2Bgpu-manifest.json" "tinfoil-inference-$V+gpu-manifest.json"

for f in \
  "cvmc-$V" \
  "tinfoil-inference-$V."{initrd,vmlinuz,raw} \
  "tinfoil-inference-$V+gpu."{initrd,vmlinuz,raw} \
  "tinfoil-"{snp,tdx}"-$V.igvm"{,.manifest.json} \
  "tinfoil-"{snp,tdx}"-$V+gpu.igvm"{,.manifest.json}
do
  curl -sfL -A curl/8 -o "$f" "https://images.tinfoil.sh/cvm/$(printf %s "$f" | sed 's/+/%2B/g')"
done

# Confirm the manifest hash matches the gateway's cvm-version pin and all checksums pass:
shasum -a 256 "tinfoil-inference-$V-manifest.json"
sed 's#  \./#  #' "tinfoil-inference-$V-checksums.sha256" | shasum -a 256 -c
```

The download is about 4.4 GB (mostly the two ~2.17 GB `.raw` disks).

### 1. Build the builder image and install pinned Nix

`Dockerfile` starts from a digest-pinned `debian:bookworm-slim` image with `curl`, `xz-utils`, `ca-certificates`, and `git`, and writes `/etc/nix/nix.conf` (`sandbox=true`, `sandbox-fallback=false`, `restrict-eval=true`, `allowed-uris=https://github.com/NixOS/nixpkgs/archive/`). `install-nix.sh` follows upstream's `nix/install.sh`, verifying the Nix tarball checksum, installing single-user (`--no-daemon`), and asserting the expected Nix store path and settings.

```sh
cd "$S"
docker build --platform linux/amd64 -t cvmrepro-builder:1 .
docker volume create cvmrepro-nix
docker run --rm --platform linux/amd64 -v cvmrepro-nix:/nix \
  debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc \
  chown 1000:1000 /nix

SEC="--security-opt seccomp=unconfined --security-opt systempaths=unconfined"
docker run --rm --platform linux/amd64 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder \
  -u builder cvmrepro-builder:1 install-nix.sh
```

### 2. Build `cvm-compiler` and both shipping images

`build.sh` clones the `cvmimage` tag into the builder's home volume, refuses to continue unless `HEAD` matches the exact expected commit hash and the tree is clean, and runs `nix-build -I . -A <attr>` for each target.

```sh
# CPU variant + cvmc compiler
docker run --rm --platform linux/amd64 --memory 12g --cpus 12 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v "$S:/scripts:ro" \
  -u builder cvmrepro-builder:1 /scripts/build.sh cvm-compiler shipping-image

# GPU variant (JOBS=1 and >=14 GB RAM to avoid OOM during vmlinux link)
docker run --rm --platform linux/amd64 --memory 14g --cpus 12 -e JOBS=1 $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder -v "$S:/scripts:ro" \
  -u builder cvmrepro-builder:1 /scripts/build.sh variants.gpu.shipping-image
```

### 3. Compile IGVM images and compare against published checksums

`release.sh` runs upstream's unmodified `image/release-igvm.sh` using `jq`, `coreutils`, and `gawk` from the pinned nixpkgs, verifies every rebuilt output against the published checksums file, and compares the regenerated checksums file byte for byte with `cmp`.

```sh
docker run --rm --platform linux/amd64 --memory 8g $SEC \
  -v cvmrepro-nix:/nix -v cvmrepro-home:/home/builder \
  -v "$R/out:/out" -v "$R/published:/pub:ro" -v "$S:/scripts:ro" \
  -u builder cvmrepro-builder:1 /scripts/release.sh
```

*(For a smaller, offline check with `--network none`, `compile-igvm.sh` runs `cvmc build` and `cvmc measure` standalone over any given kernel, initrd, and root hash.)*

### 4. Clean up

```sh
docker volume rm cvmrepro-nix cvmrepro-home && docker image rm cvmrepro-builder:1
```

## Deviations and host notes

1. **Single-user Nix in a container instead of `nix/install.sh --daemon`.** Upstream's installer expects systemd and refuses a pre-existing `/nix`. The Nix tarball, checksum, store path, and `nix.conf` settings match upstream.
2. **`filter-syscalls = false` under Rosetta.** On Apple Silicon under Rosetta, Nix cannot load its `x86_64` seccomp BPF filter (`Invalid argument`). That filter blocks setuid/setgid and xattr/ACL creation inside builds — hardening that does not affect NAR output. On a native `x86_64` Linux host, leave `filter-syscalls` at its default (`true`).
3. **Container flags `seccomp=unconfined` and `systempaths=unconfined`** (without `--privileged` or added capabilities). Nix's build sandbox requires `unshare(CLONE_NEWUSER)` and `/proc` mounts, which Docker's default seccomp profile and masked `/proc` reject.
4. **Pinned nixpkgs `jq` for `release-igvm.sh`.** Upstream's CI workflow uses whatever `jq` sits on the runner; manifest bytes depend on `jq`'s JSON formatting, which nixpkgs's `jq 1.8.2` reproduced identically.
5. **Single cross-host rebuild.** Running this once independently compares against the provider's CI build; `--check` repeated builds are not run here.

## What a rebuild does and does not establish

**What it establishes:** The published bytes are the deterministic output of that commit's Nix graph — public source plus the hash-pinned external inputs listed below.

**What it still trusts:**

- **The NixOS binary cache (`cache.nixos.org`).** Substituters stay enabled to match upstream's build configuration (~480 signed store paths come from the cache: compilers, binutils, `systemd-repart`, `erofs-utils`, bootstrap Go, etc.). The measured guest outputs themselves are built locally: kernel, `nvidia-open` modules, Go toolchain, Tinfoil Go binaries, initrd, rootfs, disk images, `cvmc`, and `nvattest`. Passing `--option substituters ''` would eliminate cache trust at a steep wall-clock cost.
- **Hash-pinned binary archives inside the measured rootfs (not built from source):**
  - Ubuntu 26.04 `.deb` packages from `snapshot.ubuntu.com` (`libc6`, `ca-certificates`, `e2fsprogs`, `iproute2`, `nftables`, `busybox-static`, etc.).
  - Prebuilt static Docker binaries (`dockerd`, `containerd`, `containerd-shim-runc-v2`, `runc`).
  - GPU variant only: NVIDIA userspace libraries, fabric manager, `persistenced`, container toolkit, and GPU firmware blobs (from NVIDIA's CUDA apt repo).
  - *(The kernel itself is built from Ubuntu's `linux-source` deb, and NVIDIA kernel modules are built from the open-source `nvidia-open` tree.)*
- **Network during the build:** Only fixed-output derivations (which declare their hash before fetching) and the pinned nixpkgs tarball (`restrict-eval` with one allowed URI); Nix sandboxing remains on.
- **Release tooling outside Nix:** Upstream's workflow runs `release-igvm.sh` using the runner's `jq` and `sha256sum`.

**Checked separately:** Sigstore build-provenance attestations on the release files are verified by `scripts/artifact-manifest.sh --verify-attestations`, and `measure-enclave`'s known-answer tests verify that Eidola's independent measurement calculation reproduces all four launch digests from the IGVM images.

## Last rebuild log: v0.15.4-rc.2 (2026-11-20)

**Result:** All 17 entries in `tinfoil-inference-v0.15.4-rc.2-checksums.sha256` — both variants' kernel, initrd, raw disk, SEV-SNP and TDX IGVM images and their `cvmc` manifests, both release manifests, and `cvmc` — plus the checksums file itself rebuilt **byte-identical** (`cmp`).

- **Source:** `https://github.com/tinfoilsh/cvmimage`, annotated tag `v0.15.4-rc.2` (tag object `cb1e3df5fec99e7d362f8e109e9e3ace6c556be1`, commit `0663ad9e0e72dd675a9846fd2e1d831109b3637b`, 2026-11-18; tag is unsigned so the commit hash is the pin).
- **Upstream pins:** Nix 2.35.1 (tarball SHA-256 `c3fe2977…9db3`, store path `/nix/store/q6yfdws28aj556jlz5yayaggiddmb0b5-nix-2.35.1`); nixpkgs `f49e816f5626998d0b70d713bd2cdc70d2207c56` (`sha256-MrPIcRrted6hW4zEgBLTM2iZBy6yjKU3FTW6YBX+XLw=`); Ubuntu snapshot `20261027` (93 CPU / 97 GPU lock entries); `docker-29.6.2.tgz`; `linux-source-7.0.0`; NVIDIA CUDA apt `595.71.05`; 481 Hydra cache paths.
- **Host:** Apple Silicon Mac, Docker Desktop 29.9.3 (16 GB Linux VM, Rosetta `x86_64` emulation enabled).
- **Wall-clock time (~1 h 2 min total):** `cvm-compiler` (cold store) 3m 31s · CPU `shipping-image` 13m 53s · GPU `shipping-image` (`JOBS=1`) 42m 14s · `release-igvm.sh` 47s. Disk: 14.8 GB Nix store volume, 0.2 GB builder home, ~8.5 GB scratch directory.
- **Changes from `v0.15.3-rc.1`:** Boot-processor VMSA enables Secure TSC (`SEV_FEATURES 0x201`), `cvmc` appends `no-kvmclock` beside `no5lvl`, two kernel patches refuse host-supplied or host-adjusted time (`kernel-ignore-host-paravirtual-features.patch`, `kernel-require-protected-time.patch`), and guest Go programs add a boot-time anchor and bounded connection drain. SEV-SNP guest policy (`0x30133`), Nix/nixpkgs pins, package locks, and kernel source are unchanged.
- **Launch values (matched published & `measure-enclave`):** CPU SEV-SNP `cea56f9f…c015`, CPU TDX MRTD `e3d165d3…017d`, GPU SEV-SNP `49ed764e…5df0`, GPU TDX MRTD `5d4ccd0d…4de5`, CPU dm-verity root `3fcb9f57…970e`.

| Artifact | SHA-256 (published = rebuilt) |
| --- | --- |
| `tinfoil-snp-v0.15.4-rc.2.igvm` (pinned `igvm_sha256`) | `112501c5b3fa891cf05c96841903b000bd012bfa13cdc780692ce8f625470216` |
| `tinfoil-inference-v0.15.4-rc.2-manifest.json` (gateway `cvm-version` pin) | `689c945f62d8f8507d5051dfdc97abc019e7ea5c9cc153557fe8b6850db45dc2` |
| `tinfoil-inference-v0.15.4-rc.2+gpu-manifest.json` (engines `cvm-version` pin) | `de67844398b970d5bbddd9ab42f7cded933b49d268ca0b7f8e86861b2e1f2094` |
| `tinfoil-tdx-v0.15.4-rc.2.igvm` | `460aae05ee9d877d6a0a21e17520085fbb44e048d08dd905058081761362bfc5` |
| `tinfoil-snp-v0.15.4-rc.2+gpu.igvm` | `c29b3fa99cea7d6fa03a6e757a371363b31a5487bb23059189d9c1f5a5699a6b` |
| `tinfoil-tdx-v0.15.4-rc.2+gpu.igvm` | `9286ab924473ed6cad59bd7e22dd7833adb93034bb0e8886c1318c66ba8a61f2` |
| `tinfoil-inference-v0.15.4-rc.2.vmlinuz` | `e71c23ddd20d70972dd215bb188b35bce907c291c10ae5070683818832b57915` |
| `tinfoil-inference-v0.15.4-rc.2+gpu.vmlinuz` | `3819a7d90aa695fc70c54c10716c038f637990192eb3b6693a73a4b060705c82` |
| `initrd` (identical for both variants and to `v0.15.3-rc.1`) | `c12763b2875338ddcc542cc9144408be0c8124792141465dec5c2e52aaeeeffa` |
| `tinfoil-inference-v0.15.4-rc.2.raw` | `baff9c8040bd4fd647f35bbaae114ee3ddefe77bbccbc86dfe0928a58f6fc564` |
| `tinfoil-inference-v0.15.4-rc.2+gpu.raw` | `99e2945cfd16e78ea19eae2a2856170f797305b98ca0021fab4c650faf6d04e4` |
| `cvmc-v0.15.4-rc.2` | `00e439af8d52d8ae303ac917f852044acb316344061c511f085a2c15c4ba05a0` |
