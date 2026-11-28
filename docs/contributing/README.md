# Contributor runbooks

Operational runbooks and procedures for people working on Eidola — active contributors, new contributors getting oriented, release officers, and technical readers auditing how the project operates.

If you are looking for conceptual docs, user guides, or privacy guarantees, start one level up in [`docs/`](../README.md).

## Runbooks in this directory

| Runbook | Who it is for | What it covers |
| --- | --- | --- |
| [`release-attestant-yubikey.md`](release-attestant-yubikey.md) | Release officers | Provisioning a YubiKey 5 as a **human-presence** release-attestation signing key (touch required), capturing Yubico hardware provenance, and signing releases via `just release-attest`. |
| [`release-apple-signing-yubikey.md`](release-apple-signing-yubikey.md) | Release & CI maintainers | Policy, custody boundaries, and hand-off requirements for the separate **unattended** Apple Developer ID CI signing YubiKey (no touch). |
| [`rebuilding-platform-images.md`](rebuilding-platform-images.md) | Maintainers & auditors | Rebuilding Tinfoil's confidential-VM boot images (`cvmimage`) from public source, verifying byte-for-byte parity with pinned releases, and recording each rebuild. |

## Quick orientation for contributors

Before opening a pull request:

- **Branch flow:** Target `next` for all feature, bugfix, and dependency PRs — never commit directly to `main`. `main` is promotion-only (fast-forwarded from `next` via `/promote` after the full artifact-verification chain passes). See [`.github/AGENTS.md`](../../.github/AGENTS.md).
- **Licensing and CLA:** Every Git author and committer email in a PR must have a matching entry in [`CLA-SIGNERS.txt`](../../CLA-SIGNERS.txt). See [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`LICENSE.md`](../../LICENSE.md).
- **Local checks:** Run `just check` (clippy, rustfmt, markdown linting, `cargo-deny`, and manifest-determinism checks) and `just test` before requesting review.
- **Privacy review:** Read [`docs/privacy-guarantees.md`](../privacy-guarantees.md) before changing anything on the client, gateway, or engine paths, and call out any change that touches an invariant in your PR description.

## Related contributor docs across the repository

- **[`CONTRIBUTING.md`](../../CONTRIBUTING.md)** — CLA signing, per-component licensing, and Git history permanence.
- **[`README.md`](../../README.md#developing)** — Local dev prerequisites (`rustup`, `direnv`, `docker`), container vs. host-mode server loops, local client trust overrides (`just client-local`), GUI/CLI builds, and website builds.
- **[`releases/README.md`](../../releases/README.md)** — Trust-root build inputs, attestant-key rotation, CI workflow rotation, template updates, and document schema rotations.
- **[`deploy/engine/README.md`](../../deploy/engine/README.md)** — Committed inference-engine deployment configs, platform-endorsement vendoring, and launch-identity pinning.
- **[`crates/eidola-billing-cli/README.md`](../../crates/eidola-billing-cli/README.md)** — Billing receiver/worker deployment and private support CLI playbook.
- **[`AGENTS.md`](../../AGENTS.md)** — Workspace architecture map and index of area-specific engineering docs (`crates/*/AGENTS.md`).
