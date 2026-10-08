# measure-enclave — Agent Development Guide

The `measure-enclave` binary pre-computes the hardware attestation measurements a legitimate Tinfoil Container will produce. The measurement is a deterministic function of:

1. OVMF firmware (pinned from `tinfoilsh/edk2`)
2. CVM kernel + initrd (versioned from `tinfoilsh/cvmimage`, hash-verified)
3. Kernel command line (embeds dm-verity roothash + SHA-256 of the workspace `tinfoil-config.yml`)
4. vCPU count and type

Uses `sev` (with `crypto_nossl` — pure Rust, no OpenSSL) for SEV-SNP launch digest computation and `tdx-measure` for TDX RTMR1/RTMR2 runtime measurements. Both work natively on macOS. Output JSON matches the Tinfoil deployment manifest predicate (`snp-tdx-multiplatform/v1`): `{snp_measurement, tdx_measurement: {rtmr1, rtmr2}, cmdline}`.

**The measurement flow:** source → deterministic OCI build → server digest → `tinfoil-config.yml` (with digest) → cmdline (with config hash) → measurement → `releases/trust/server-enclave.json` → the cli build embeds it as its trust root → cli OCI digest / desktop `narHash` + `archiveSha256` → `artifact-manifest.json`. The `server-enclave.json` step breaks the otherwise-circular self-reference that would occur if the cli build COPYed the manifest containing its own digest; isolating the enclave fields gives the cli build a stable input while the manifest regenerates. All values are committed and verified by CI (see `.github/AGENTS.md`). Payload / archive vocabulary: `docs/verification.md`.

CVM artifacts are cached locally at `~/.cache/eidola/cvm/`. Both mutable-tag downloads are pinned by committed SHA-256 constants in `scripts/artifact-manifest.sh`: `OVMF_SHA256` pins the OVMF.fd asset, and `CVM_MANIFEST_VERSION`/`CVM_MANIFEST_SHA256` pin the CVM release manifest (which transitively pins kernel, initrd, and the dm-verity roothash). Cache hits are re-hashed, not trusted — a poisoned cache entry is discarded and re-fetched, and a fresh download that still mismatches fails hard. Bumping `cvm-version` in `tinfoil-config.yml` requires updating the manifest pin (the script refuses a version with no matching pin); an inline `cvm-version: X@sha256:HEX` pin in the config (the tinfoilsh/measure-image-action syntax) takes precedence. `ovmf-version` is deliberately absent from the config: Tinfoil's deploy infrastructure does not support the field yet, so the OVMF pin lives script-side only. Pass `--verify-attestations` to `scripts/artifact-manifest.sh` (used by CI) to additionally verify CVM manifest provenance via Sigstore (`gh attestation verify --deny-self-hosted-runners`); this fails hard on verification failure.

Because the config hash is bound into the measurement, **any change to `tinfoil-config.yml` produces a different enclave measurement** — regeneration is `just update-manifest` (release time only; see the top-level AGENTS.md conventions).

## TDX launch identity, IGVM model (`measure-tdx-igvm`)

A second binary in this crate emits the launch half of a verifier TDX pin (`tinfoil_verifier::TdxPin`) for the platform provider's IGVM launch model (cvmimage v0.15+). That model has no OVMF and no measured boot after launch, so its TDX identity is two registers rather than per-shape RTMRs:

- **MRTD** is recomputed from the release's TDX IGVM image (`tdx_igvm::mrtd_from_igvm`). It is a clean-room implementation of the TDX module's accumulation over `igvm`-parsed directives in file order. Measured page data gets `MEM.PAGE.ADD` plus sixteen 256-byte `MR.EXTEND` records. Unmeasured page data and inserted parameter areas get the add record only. Shared pages are skipped. Directives outside that model (large pages, non-normal page types, VP contexts) are refused rather than guessed at. Verified against the published v0.15.0-rc8 image: it reproduces the manifest's MRTD `5cfedb59…`. The `published_release_mrtd_known_answer` test reruns that check when `TDX_IGVM` / `TDX_IGVM_MRTD` point at a downloaded image. The image is not committed: it is ~9 MB of kernel and initramfs.
- **MRCONFIGID** is `sha256(tinfoil-config.yml bytes) ‖ 16 zero bytes`.

`tdx_igvm::release_pin` ties the computation to a release and to the deployment. The config's `cvm-version` must name the manifest's release (`X` ↔ `vX`). When it carries the inline `X@sha256:<hex>` form, that hash must also be the manifest's. Otherwise a pin could combine one release's MRTD with a config that launches another, and it could never attest. The release manifest (`tinfoil-inference-<version>-manifest.json`) must match a pinned SHA-256, and its `igvm.format_version` must be 1. The image must hash to the manifest's `igvm.tdx`. The manifest's `tdx_launch` RTMRs must all be zero, and the recomputed MRTD must equal its `tdx_launch.mrtd`. Any disagreement is an error. The output (`cvm_version`, `igvm_sha256`, `mrtd`, `mrconfigid`) carries no machine policy (MR_SEAM, attributes, TCB floors): those describe the host, not the image.

This tool does not touch `releases/trust/server-enclave.json` or `scripts/artifact-manifest.sh`; the server's own enclave stays on the kernel-cmdline model until it moves to an IGVM release.
