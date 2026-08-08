# measure-enclave — Agent Development Guide

The `measure-enclave` binary pre-computes the hardware attestation measurements a legitimate Tinfoil Container will produce. The measurement is a deterministic function of:

1. OVMF firmware (pinned from `tinfoilsh/edk2`)
2. CVM kernel + initrd (versioned from `tinfoilsh/cvmimage`, hash-verified)
3. Kernel command line (embeds dm-verity roothash + SHA-256 of the workspace `tinfoil-config.yml`)
4. vCPU count and type

Uses `sev` (with `crypto_nossl` — pure Rust, no OpenSSL) for SEV-SNP launch digest computation and `tdx-measure` for TDX RTMR1/RTMR2 runtime measurements. Both work natively on macOS. Output JSON matches the Tinfoil deployment manifest predicate (`snp-tdx-multiplatform/v1`): `{snp_measurement, tdx_measurement: {rtmr1, rtmr2}, cmdline}`.

**The measurement flow:** source → deterministic OCI build → server digest → `tinfoil-config.yml` (with digest) → cmdline (with config hash) → measurement → `releases/trust/server-enclave.json` → the cli build embeds it as its trust root → cli OCI/macOS narHash → `artifact-manifest.json`. The `server-enclave.json` step breaks the otherwise-circular self-reference that would occur if the cli build COPYed the manifest containing its own digest; isolating the enclave fields gives the cli build a stable input while the manifest regenerates. All values are committed and verified by CI (see `.github/AGENTS.md`).

CVM artifacts are cached locally at `~/.cache/eidola/cvm/`. Both mutable-tag downloads are pinned by committed SHA-256 constants in `scripts/artifact-manifest.sh`: `OVMF_SHA256` pins the OVMF.fd asset, and `CVM_MANIFEST_VERSION`/`CVM_MANIFEST_SHA256` pin the CVM release manifest (which transitively pins kernel, initrd, and the dm-verity roothash). Cache hits are re-hashed, not trusted — a poisoned cache entry is discarded and re-fetched, and a fresh download that still mismatches fails hard. Bumping `cvm-version` in `tinfoil-config.yml` requires updating the manifest pin (the script refuses a version with no matching pin); an inline `cvm-version: X@sha256:HEX` pin in the config (the tinfoilsh/measure-image-action syntax) takes precedence. `ovmf-version` is deliberately absent from the config: Tinfoil's deploy infrastructure does not support the field yet, so the OVMF pin lives script-side only. Pass `--verify-attestations` to `scripts/artifact-manifest.sh` (used by CI) to additionally verify CVM manifest provenance via Sigstore (`gh attestation verify --deny-self-hosted-runners`); this fails hard on verification failure.

Because the config hash is bound into the measurement, **any change to `tinfoil-config.yml` produces a different enclave measurement** — regeneration is `just update-manifest` (release time only; see the top-level AGENTS.md conventions).
