# Known gaps

Every piece of the Eidola trust chain that is intentionally deferred is catalogued here. Each gap closes a specific class of attack that is already constrained by other parts of the chain — but they are real and worth understanding. Reading this page is the fastest way to see what Eidola does not yet defend against.

The cryptographic-verifier gaps are also noted at the top of [`crates/eidola-app-core/src/updater/ci_sigstore/mod.rs`](../crates/eidola-app-core/src/updater/ci_sigstore/mod.rs) and `rekor.rs`, and the install-side gap is at the `TODO (step 5)` marker on `verify_each_artifact_hash` in [`crates/eidola-app-core/src/updater/mod.rs`](../crates/eidola-app-core/src/updater/mod.rs). The same two cryptographic-verifier gaps apply to the server-side runtime upstream-measurement resolver ([`crates/eidola-server/src/upstream_trust/sigstore.rs`](../crates/eidola-server/src/upstream_trust/sigstore.rs)), which reuses the same Fulcio/Rekor primitives.

## Cryptographic verifier

### SCT verification in the Fulcio leaf certificate

**What it would catch.** A malicious or compromised Fulcio issuing certificates for OIDC identities it shouldn't — the Signed Certificate Timestamp proves the cert was logged in a public CT log.

**What constrains it today.** The OIDC-identity match and the Fulcio chain walk are the primary binding. The SCT check is defense-in-depth on top of those, not a single point of failure.

### Rekor checkpoint signature verification

**What it would catch.** The Rekor instance silently forking a side-tree just for our entries — the inclusion proof we compute is mathematically valid but roots to a tree the public never sees.

**What constrains it today.** The Signed Entry Timestamp already requires the Rekor public key to vouch for our specific entry. The checkpoint adds independence from private forks by verifying against the publicly-witnessed log head.

### Artifact-hash check at install time

**What it would catch.** A tampered binary download — the *manifest* is signed and content-verified, but the actual binary bytes that would run are not yet hashed against the manifest's declared digests.

**What constrains it today.** The verifier already proves `artifact-manifest.json` itself is authentic and unmodified. The install-time hash check lives naturally in a to-be-implemented install / atomic-replace step, once we know which platform's artifact the user is downloading. The verifier code is in place; only the wiring to the download path is deferred.

### Multi-hop / fast-forward continuity

**What it would catch.** A client that skips multiple releases (e.g. v1.0 → v1.5, missing v1.1–v1.4). Today the continuity gate requires strict equality between `release.previous_release.git_commit` and the installed commit, so an out-of-date client must update through every release in order.

**What constrains it today.** Strictly sequential is the safer floor. Relaxing to "fast-forward reachable via GitHub commits API" is a small follow-up that's only worth doing once the release cadence makes sequential updates painful in practice.

## Operational

### Install / atomic-replace

**Current behavior.** `eidola update` runs the full verification pipeline and prints the verified attestation prose, but does not download or swap the binary.

**Future.** Step 5: download the artifact for the user's platform from `artifact-manifest.json`, hash-verify, atomic-replace, restart. Platform-specific (CLI = file swap; macOS GUI = staged swap on next launch).

### Single-attestant policy

**Current behavior.** `MIN_HUMAN_ATTESTATIONS` (embedded in the client, sourced from `releases/trust/trust-constants.json`) is `1` in current releases — only one engineer needs to attest for a release to verify.

**Future.** Once a co-attestant key is provisioned and added to `trusted_attestant_fingerprints`, bumping `min_human_attestations` to `2` (in a release signed under the *current* threshold) makes every subsequent release require independent corroboration. The verifier already supports arbitrary M-of-N; the second key just hasn't been generated yet.

### First-install downgrade

**Current behavior.** A fresh client (no prior installed `git_commit`) bypasses continuity, so an adversary serving an internally-consistent *older* `release.json` could route them onto a real-but-stale release.

**Mitigations today.** None that the client can enforce. A first install is exactly the case where the client has nothing to compare against, so the surface "did you download a current release or a stale-but-internally-consistent one?" lives outside the client's reach today.

**Future.** A *freshness anchor* is something a fresh install *can* validate at the moment of download, without needing prior state. The general shape: every release embeds (or references) a recent timestamped artifact from a public, append-only system that an attacker cannot retroactively forge.

### Anonymity-set size

**Current behavior.** An anonymous credit token is unlinkable to its issuance only *within its anonymity set*: the accounts that received tokens under the same `(issuer_key, domain_separator)` during that key's issuance window ([privacy-guarantees.md](privacy-guarantees.md) §2.3). The partition parameters — weekly issuance epochs, a single non-rotating domain separator, an acceptance window one full epoch beyond issuance — are compile-time constants, but the *population* inside a partition is a deployment fact the code cannot control. Early in the deployment, or during a quiet week, a window may contain few active accounts; in the degenerate one-account case the cryptography still holds, but the operator could attribute spends by elimination.

**What constrains it today.** Temporal decoupling (§2.4) keeps issuance and redemption timestamps from being forced near-equal, the server persists no per-redemption timestamps, and nothing in the protocol subdivides the partition further. What nothing can do is conjure a crowd.

**Future.** The set grows with the user base. If population per window stays small, the epoch length can be raised (a release-visible constant change) to trade key-rotation hygiene for coarser partitions.

### Multi-jurisdiction attestant distribution

**Current behavior.** Attestants share a small operational surface and may share a jurisdiction.

**Future.** Distributing pinned attestants across hardware vendors, custody arrangements, and jurisdictions raises the cost of coordinated legal compulsion — the central concern of the no-coercion attestation claims in [privacy-guarantees.md §6.4](privacy-guarantees.md#6-release-integrity) and the bounded claim in [§8.7](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise). This is a matter of organizational rollout, not engineering, but it is named here because it is part of what the guarantee depends on.

## Hardware

### Trust in confidential-compute vendors

**Current behavior.** Eidola trusts AMD, Intel, and NVIDIA to issue genuine attestation chains. A vendor issuing fraudulent attestations for an enclave that does not in fact provide confidential compute would defeat that layer of the chain.

**Mitigations today.** Limited. The use of WebPKI for our TLS certificate provides a defense in depth, ensuring that an outside party issuing a fraudulent attestation must also produce or obtain a fraudulent WebPKI certificate. However, this provides little resistance in the case of a malicious insider. Generally, we accept hardware vendor trust as residual.

**Future.** Open hardware roots like OpenTitan reduce the scope of vendors the trust chain depends on. This is an industry-wide direction, not an Eidola-specific roadmap item, but it is the long-term mitigation for this residual trust.

### TLS-key exfiltration and channel binding

**Current behavior.** Everything the per-handshake verifier proves rides on the enclave-held TLS key: the hardware report commits to the exact endorsed crypto-material section containing the peer certificate's SPKI hash, together with the nonce and device-evidence hash, and the inline attestation shares the request's connection ([privacy-guarantees.md](privacy-guarantees.md) §4.3). It does not commit TLS exporter or other session-specific key material. An adversary who exfiltrated that private key from the enclave could terminate TLS outside it and relay fresh attestations from a real one — an active MITM that fails no client-side check.

**What constrains it today.** The key is generated and held only in enclave memory by the attestation shim, which is part of the measured boot image (§4.1, §5.1) — there is no export interface, so exfiltration requires a defect in that hash-pinned image, not any code path in this repository. The image being measurement-bound means the code that would have to leak it is fixed and inspectable.

**Future.** Stronger channel binding upstream — e.g. attestation freshness bound into the TLS key-exchange transcript rather than to a long-lived per-boot key — would shrink the value of a leaked key to a single session. This is shim-side (Tinfoil) work we track rather than control.

### TDX acceptance

**Current behavior.** The verifier refuses Intel TDX attestations outright — only SEV-SNP presentations are accepted, on both attested paths (client → Eidola server, Eidola server → inference upstream). This is deliberate fail-closed behavior, not an omission: the policy checks TDX acceptance requires are incomplete. MRTD — the only register measured by the TDX module itself, covering the virtual firmware — and RTMR0 are not policy-checked, and RTMR1/RTMR2, the values the trust root records, are guest-extendable: any firmware on a genuine TDX machine can replay the published digests into them. Since the attestation document's author selects the platform branch, accepting TDX on RTMR1/2 alone would let an operator sidestep the SEV-SNP measurement pin entirely.

**What constrains it today.** The refusal itself. Both live paths run SEV-SNP, so refusal costs nothing; if the platform provider moves a path to TDX, that path fails closed until proper support ships — the correct failure mode for this system. The trust root continues to record RTMR1/RTMR2 as the honest statement of what the release would measure; recording is not acceptance.

**Future.** Proper acceptance needs MRTD + RTMR0 reference values — host-side inputs the platform provider would have to publish for our container shape — plus MR_SEAM/XFAM pins and a platform (FMSPC) allowlist for parity with the SNP path's structural generation pin. The dormant verification plumbing is kept wired so the change reduces to policy plus reference values.

## Network / metadata

Eidola does not defend against an adversary observing network metadata (connection patterns, packet sizes, timing). Content is protected by TLS terminated inside the attested enclave; metadata is visible to network observers. There are really two distinct gaps here that share infrastructure but answer different questions for the user:

### Passive traffic analysis

**What it would catch.** Connection patterns, packet sizes, timing — even with TLS confidentiality, these can reveal a great deal (which model you used, the rough shape of conversations, when you are active).

**Mitigations today.** User-side: route Eidola through Tor. Eidola's protocol is plain HTTPS, so this works without modification.

**Future.** We consider this in-scope as an Eidola problem to address, but do not yet have a committed plan. Explored directions include offering a Tor hidden service endpoint and partnering with independent organizations to provide oblivious HTTP (oHTTP) or MASQUE/CONNECT-style transports that decouple network identity from request content.

### Network identity as a linking factor

**What it would catch.** Even a single connection to Eidola from a unique IP is itself an identity signal: an observer (or Eidola's own network logs, were they to exist) can correlate "a connection from IP X" with the account billed at approximately the same time, undermining the unlinkability invariants in [privacy-guarantees.md §2](privacy-guarantees.md#2-unlinkability) at the transport layer rather than at the application layer.

**Mitigations today.** User-side: use Tor, or a reputable VPN provider like Mullvad. Both break the direct IP↔account correlation by inserting a third party that doesn't share data with Eidola.

**Future.** Same direction as above (oHTTP, MASQUE, partner relays). The Eidola-side mitigation here is partnering with an independent organization whose role is to terminate the network connection so that no single party — Eidola included — sees both the network identity and the account it corresponds to.

## Inference upstream

Inference does not run in the Eidola server enclave. `inference.tinfoil.sh` is a *router* enclave operated by the upstream provider (currently Tinfoil) that reverse-proxies each request to a **separate per-model GPU enclave**. That structure shapes the two gaps below.

### The inference code is trusted at the provider's bar, not reviewed or pinned by Eidola

**Current behavior.** The Eidola server attests the *router* enclave on every handshake — genuine confidential-compute hardware, running a measurement that corresponds to a Sigstore-signed release of the expected repository at the expected tag. But two things sit outside Eidola's own review:

- **The downstream fan-out is not chained.** The router's attestation covers only the router; it does not fold in the attestations of the per-model enclaves it forwards to. The router trusts each model enclave via "the latest Sigstore-signed release of that model's repo" (the same trust model as `tinfoilsh/tinfoil-go` / `tinfoil-rs`), and Eidola does not independently attest those enclaves. So the code *actually running inference* — which necessarily sees cleartext prompts and completions — is trusted at the provider's release-signing bar, not re-derived by Eidola.
- **Eidola tracks the latest signed release rather than a reviewed pin.** The server resolves the router measurement at runtime and adopts any new release whose Sigstore provenance verifies (see [upstream.md](upstream.md#what-pins-the-upstream-measurement)) — there is no human-review gate the way there is for Eidola's *own* releases. This is deliberate, not an oversight: pinning-and-reviewing the router measurement bought little rigor while the downstream model enclaves (which also see cleartext) remain trust-latest regardless, so Eidola matches the provider's actual model here instead of performing review theater. Eidola previously carried a pinned-and-PR-reviewed router measurement; that path was removed for this reason.

**What constrains it today.** The model enclaves are themselves confidential-compute enclaves, so a genuine one seals cleartext against its own operator exactly as Eidola's does; every measurement the router or Eidola trusts must still correspond to a Sigstore-signed release of the expected repository; and Eidola's per-handshake, nonce-fresh verifier still runs on the connection it opens. The residual gap is specifically that Eidola does not itself review or pin the inference code — it trusts the provider's release-signing not to ship a malicious model build — and that while retroactive transparency and auditability are preserved, the system does not "fail closed" and a malicious version could be published, deployed, and exploited without an opportunity for the end user to prevent it.

### Upstream-provider trust-discipline mismatch

**Current behavior.** Even for the code Eidola *does* verify (the router) and the model code the provider signs, the upstream release pipeline does not match the discipline applied to Eidola's own releases:

- Tinfoil's builds are **not source-bootstrapped reproducible** in the StageX sense. They are hermetic and provenance-attested through GitHub's CI attestation, which is rigorous, but shaped differently than Eidola's.
- Tinfoil does **not yet ship per-release human attestations under named legal identities** the way Eidola releases do.

A user's chain of trust at the inference layer therefore ends at Tinfoil's release discipline, which is non-trivially different from Eidola's.

**Future.** The intent is to close both gaps by ending the inference-layer trust chain where the rest of Eidola's does. There are two paths, and either one suffices:

- **Primary — self-host inference.** Run the models on hardware Eidola controls, such as GPU-enabled Tinfoil containers, built and released through Eidola's own source-bootstrapped, human-attested flow, so Eidola owns the inference measurement, can *statically pin* it, and reviews the downstream code directly. This is the long-term goal regardless, since it also brings reproducibility and human attestation to the inference layer. Its cost is high fixed GPU spend that is hard to justify before there is meaningful demand — so it is not the immediate move.
- **Alternative — the provider adopts static-pinned upstreams.** If Tinfoil itself moves its router from "trust the latest signed release" to *statically pinning* the specific model-enclave measurements it will forward to, then the downstream-review gap closes without Eidola self-hosting: Eidola would simply keep using Tinfoil's inference and pin the reviewed set. This is the cheaper outcome, and if it materializes it is the preferred long-term solution.

In the meantime, the runtime-resolution scheme above is the honest interim: it claims exactly the assurance the provider's own model provides, and no more.

## Build chain opacity

### Non-source-bootstrapped components in the trust chain

**What it would catch.** Build-pipeline subversion in a stage we don't fully source-bootstrap.

**Current behavior.** Several components of the Eidola build chain are pinned by hash and used reproducibly, but are themselves not fully source-bootstrapped:

- **macOS Nix builds.** Hermetic and reproducible (`narHash` / `archiveSha256` pinning of the unsigned payload), but rely on the Apple SDK / Xcode toolchain as opaque inputs. Cross-compiling macOS binaries from Linux is not viable today, so macOS releases must be built on macOS. The Apple envelope (Developer ID, notarization) is non-reproducible by construction and is bound only in the human attestation; see [verification.md](verification.md) and [apple-distribution.md](apple-distribution.md).
- **Linux GUI Nix build.** Hermetic and reproducible (`narHash` / `archiveSha256`) from the open nixpkgs toolchain, but not source-bootstrapped in the StageX sense. The Nix installable is a glibc dynamic binary with nixpkgs Mesa referenced via `VK_ADD_DRIVER_FILES` (those driver bytes are not in the archive). A desktop GUI that talks to host Vulkan must be glibc; the musl-by-design StageX pipeline cannot produce it. Separately, that artifact is **Nix-shaped**: its archive is a store-path serialization whose binary resolves its libraries out of `/nix/store`, so it is not a download a non-Nix user can extract and run — the Debian package is the installable for everyone else, and it carries no GPU bytes at all ([verification.md](verification.md#the-two-linux-installables)).
- **`cvmimage` and OVMF firmware.** Pinned by hash, but their build chains do not match Eidola's source-bootstrapping discipline. Their contents are bound into the server's enclave measurement, so they cannot be changed silently — but the original build chain is more trusted than we ideally want.

**What constrains it today.** Each of these has digest pinning and provenance verification at the import boundary (committed sha256 pins for OVMF and the `cvmimage` release manifest — with Sigstore provenance verified on top — and `narHash` / `archiveSha256` for Nix outputs), so silent substitution is detectable. The gap is that the upstream *builders* of those artifacts are trusted to a degree we don't fully audit.

**Future.** This is a long-term direction matched to ecosystem progress: source-bootstrapped macOS toolchains, reproducible CVM/firmware builds. We follow the relevant ecosystems and will adopt as they mature. Until then, this is an unavoidable residual.
