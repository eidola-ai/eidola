# Known gaps

Every piece of the Eidola trust chain that is intentionally deferred is catalogued here. Each gap closes a specific class of attack that is already constrained by other parts of the chain — but they are real and worth understanding. Reading this page is the fastest way to see what Eidola does not yet defend against.

The cryptographic-verifier gaps are also noted at the top of [`crates/eidola-app-core/src/updater/ci_sigstore/mod.rs`](../crates/eidola-app-core/src/updater/ci_sigstore/mod.rs) and `rekor.rs`, and the install-side gap is described where the verifier hands off to [`crates/eidola-app-core/src/updater/install/`](../crates/eidola-app-core/src/updater/install/mod.rs).

## Cryptographic verifier

### SCT verification in the Fulcio leaf certificate

**What it would catch.** A malicious or compromised Fulcio issuing certificates for OIDC identities it shouldn't — the Signed Certificate Timestamp proves the cert was logged in a public CT log.

**What constrains it today.** The OIDC-identity match and the Fulcio chain walk are the primary binding. The SCT check is defense-in-depth on top of those, not a single point of failure.

### Rekor checkpoint signature verification

**What it would catch.** The Rekor instance silently forking a side-tree just for our entries — the inclusion proof we compute is mathematically valid but roots to a tree the public never sees.

**What constrains it today.** The Signed Entry Timestamp already requires the Rekor public key to vouch for our specific entry. The checkpoint adds independence from private forks by verifying against the publicly-witnessed log head.

### Artifact-hash check at install time

**What it would catch.** A tampered binary download — the *manifest* is signed and content-verified, but the actual binary bytes that would run are not yet hashed against the manifest's declared digests.

**What constrains it today.** The check itself now exists: `updater::install::stage` hashes each downloaded container against the value the verified manifest records and refuses on any difference, leaving no staged tree. What is still missing is the wiring that builds an install plan from a verified release — which waits on the attestation carrying the detached material's hash and the signing identity to expect.

### Multi-hop / fast-forward continuity

**What it would catch.** A client that skips multiple releases (e.g. v1.0 → v1.5, missing v1.1–v1.4). Today the continuity gate requires strict equality between `release.previous_release.git_commit` and the installed commit, so an out-of-date client must update through every release in order.

**What constrains it today.** Strictly sequential is the safer floor. Relaxing to "fast-forward reachable via GitHub commits API" is a small follow-up that's only worth doing once the release cadence makes sequential updates painful in practice.

## Operational

### Install / atomic-replace

**Current behavior.** `eidola update` runs the full verification pipeline and prints the verified attestation prose. Downloading and verifying an update is implemented — `updater::install::stage` fetches the payload, hashes it against the manifest, reconstructs the signed macOS app from the detached material, and checks that the result claims the identity the attestation names — but it stops at a *staged* bundle. Nothing replaces the running application.

**Future.** Promotion: swapping a staged bundle for the installed one, and relaunching. That is a product decision rather than a missing mechanism, because the answer differs by install location — `updater::install::promotion_readiness` reports whether this process could do it without privileges it should not be acquiring on its own. The CLI stays out of scope (no TCC need, and a bare Mach-O cannot be stapled).

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

## Anonymous credits

The decisions behind anonymous payment, and the reasons for them, are in [anonymous-credits.md](anonymous-credits.md). These are the residuals it names.

### Issuance-to-first-spend timing

**Current behavior.** When a request needs a credit token and the wallet holds none that covers it, the client converts credits from the account and spends the new token immediately. The database records each issuance with its account and time, and sees each spend's nullifier as it is written. Whoever can watch the database, or the server's traffic, can therefore pair an issuance with the spend that follows it a moment later, linking the first spend of each token to its account. A prompt-cache key on that request then links the rest of its conversation lineage ([privacy-guarantees.md §2.7](privacy-guarantees.md#2-unlinkability)). Later spends of the same token's successors are not linked this way.

**What constrains it today.** Allocations are large enough (about $1.00) that a token pays for many requests, so issuances are infrequent; tokens can be converted ahead of time by hand (`eidola account allocate`) and are then spent whenever they are needed.

**Future.** Provision ahead of need: keep a token in reserve and replace it at a time unrelated to any request, so issuance and spending are decoupled by default rather than only when the user arranges it.

### A lost issuance response strands its allocation

**Current behavior.** The server debits the account and then returns the signed token. Issuance cannot be repeated: if the response is lost (a dropped connection, a client crash), the account has paid for a token the client never obtained.

**What constrains it today.** Each automatic allocation is about $1.00, and nothing else is lost.

**Future.** Idempotent issuance: the client names each issuance request, and the server keeps the response on the account side so the same request can be answered again. The response is already known to the issuer, so keeping it does not help link the token's spends.

### A crash mid-spend strands the token's remaining value

**Current behavior.** A spend records its nullifier before the request is served and stores the refund when it completes. If the gateway stops between the two (a crash, a restart), the nullifier is recorded with no refund. Recovery then answers that the refund is not yet available, indefinitely, and the token's whole remaining balance is lost, not just the request's charge.

**What constrains it today.** The window is the duration of one request on one instance. The client keeps the spend's state and keeps retrying recovery until the key stops accepting.

**Future.** Let recovery return the remaining balance of a spend whose refund never arrived: a refund of nothing returns the token's balance less the charge, so the server keeps everything it was paid. Telling an abandoned spend from one still in flight without a per-spend timestamp in the database is the design question.

### Issuer key records are not authenticated against the database

**Current behavior.** Issuer private keys are stored encrypted, but each key's windows (`issue_from`, `issue_until`, `accept_until`) are plain columns, and a gateway decides whether to create a new key from what the database reports. A database that rewrote a key's windows could extend or shorten how long its tokens are accepted. One that hid keys from a gateway instance could make that instance create a key of its own, so that the clients it serves form a smaller anonymity set.

**What constrains it today.** Each instance caches its keys and asks the database only when its cache holds no current or next key, so a hidden key can split the set at most at an instance's start or a rotation, and only by instance. The client checks that every key it uses is named by the hash of its public key, and that every token and refund it receives is under that key with the request context every client under the key shares. It cannot check that other clients were given the same key.

**Future.** Bind each key's windows into the encryption of its private key, so a row whose windows were changed does not decrypt, and publish the sequence of keys somewhere a client can compare against.

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

**Current behavior.** The client → Eidola server path accepts SEV-SNP only, and refuses an Intel TDX presentation before authenticating it. Acceptance is driven by the caller's pins, never by the document: every pinned entry names its platform, evidence is checked only against entries of its own platform, and evidence from a platform with no entry is refused. The client builds its pins from release records, and a record pins its SEV-SNP launch digest only — the RTMR1/RTMR2 values the trust root records are guest-extendable (any firmware on a genuine TDX machine can replay published digests into them), so they are recorded and never accepted. A document author therefore cannot select the TDX branch to sidestep the SEV-SNP pin.

The verifier accepts TDX for a caller that pins an explicit TDX entry. The one such caller is the server's connection to its own engine deployments, whose hosts are Intel TDX ([engine.md](engine.md#what-pins-the-engine)), and those pins are TDX-only by the same construction. Such a pin follows the platform provider's IGVM launch model, in which the guest is a single firmware-less image with no measured boot after launch. The provider ships that model only in pre-releases so far; its stable releases, and the server's own enclave, still boot firmware with a kernel command line. Each release publishes a CPU image and a GPU image (`X+gpu`) with different MRTDs, and an engine deployment must name the GPU image (`engine_deployment::check_cvm_image`):

- **Launch identity.** MRTD, which the TDX module measures over every page of the image and which is fixed per image release, and MRCONFIGID, the SHA-256 of the deployment's config followed by sixteen zero bytes, which the guest refuses to boot without. RTMR0–RTMR3 must all be zero. `measure-enclave` recomputes MRTD from the release's published image, checked against the release manifest, and MRCONFIGID from the config bytes.
- **Machine policy and machines**, for the report fields an image cannot fix. The TDX module (MR_SEAM), exact TD attributes (debug off, `SEPT_VE_DISABLE` on — a debug relaunch of the pinned image has the same MRTD) and XFAM, a TEE TCB SVN floor, the QE vendor, a floor on the Intel collateral's evaluation-data number, and an allowlist of individual machines by PPID: the Platform Provisioning ID in the host's Intel-signed PCK certificate, which names one registered platform. All of these are derived from the provider's signed platform endorsements (below). Intel's platform, QE and TDX-module TCB statuses must all be `UpToDate`.
- **Quote verification.** The PCK chain verifies to the pinned Intel SGX root with both Intel CRLs consulted. TCB Info and QE Identity carry valid Intel signatures and are inside their validity windows. The quote and QE report signatures verify. All of it runs offline, against the Intel PCS responses the document carries.

`REPORT_DATA` binding and the envelope checks are the same as on the SEV-SNP path. An entry may also declare how many GPUs its enclave runs with. The document must then carry exactly that many NVIDIA GPU evidence items, each labelled with this handshake's nonce.

**What constrains it today.** Only a caller that pins a TDX entry can accept TDX: the server's engine connections, and no engine deployment is pinned yet. Recording RTMR1/RTMR2 in the trust root remains an honest statement of what a release would measure, not acceptance. For an engine pin:

- **GPU evidence is counted, not appraised.** The CPU measurement vouches for boot code that refuses to start unless the GPUs attest. The GPU image's measured boot checks the GPUs' PCI device IDs, runs NVIDIA's `nvattest` with its local verifier on each GPU (measurements against NVIDIA's reference values, and the certificate chains and revocation status of the report and reference values), requires exactly `gpus` reports, and only then sets the ready state that lets CUDA work reach the GPUs; any failure ends the boot. The engine, before it opens a GPU, then requires NVML to report confidential computing on, DevTools mode off, the production environment and that ready state, which NVIDIA's verifier does not check as such (`EIDOLA_ENGINE_GPU_CC=required`, which the gateway's build requires of every pin). On each handshake the Eidola server counts the items and checks each one's `nonce` member, which the measured attestation shim writes beside NVIDIA's report; the quote's `REPORT_DATA` covers the whole device section. NVIDIA's certificate chain, the SPDM report signature, the nonce inside that report and the reference measurements are not verified by the server, nor by the provider's own client. So the GPUs are appraised inside the guest at boot, not per handshake, and that boot depends on reaching NVIDIA's reference-value and revocation services.
- **Machine policy is the provider's signed statement, as of a committed release.** The provider publishes Sigstore-signed platform endorsements (`platform-vX.Y.Z` releases of its image repository) that map each of its machines, TDX ones by PPID, to a named policy (TDX module, TD attributes, XFAM, TEE TCB SVN floor, QE vendor, collateral floor). An engine deployment names one policy of one release. That release's file and its attestation bundle are committed, with the file's hash, and `measure-enclave` and every server build verify them offline against the committed Sigstore trusted root: signed by the provider's release workflow at exactly that tag in its repository, on a GitHub-hosted runner, logged in Rekor, about a file with that hash. The pin's policy, and its PPID allowlist of every machine the release maps to that policy, are derived from the file, and a build whose pin differs is refused. No network is read to do it. What this leaves:
  - The values are the provider's. The pin accepts exactly the machines and floors the provider endorses; Eidola checks that the provider signed them, not that they are the right ones.
  - They are as of the committed release. A release the provider publishes later (a raised floor, a machine retired) applies only once it is committed, and the provider's own client also asks for a recent signed freshness statement about the release it applies, which a compiled-in pin cannot do: its release is fixed when the server is built. A scheduled check reports a newer release, but adopting it is a reviewed change and a new server build.
  - The Sigstore verification omits the Fulcio leaf's SCT and the Rekor checkpoint signature, as the client's update verification does ([SCT verification](#sct-verification-in-the-fulcio-leaf-certificate), [Rekor checkpoint signature verification](#rekor-checkpoint-signature-verification)).
  - The PCK certificate's platform flags — dynamic platform, cached keys and SMT enabled — are hand-supplied, because the endorsements do not carry them. Each must be stated in the pin, and a host whose flag differs is refused. A sample multi-package TDX host asserts all three, and accepting SMT in particular admits cross-thread side channels ([threat-model.md](threat-model.md)).

  There is no platform-family (FMSPC) allowlist: the family is a fixed property of each listed machine, so it would refuse nothing the PPID allowlist admits.
- **Collateral can be replayed within its window.** As with the AMD CRL, a relay can serve older Intel-signed collateral (CRLs, TCB Info, QE Identity) until its signed validity ends. That delays recognition of a newer revocation or TCB recovery; the evaluation-data floor bounds how old it may be.

**Future.** Measure and pin the first engine deployment, on a stable release of the IGVM line once there is one. Appraise GPU evidence against NVIDIA's roots and reference values. Adopt the SEV-SNP half of the same launch model (`HOST_DATA` and the ID block) when the server's own enclave moves to it.

## Network / metadata

Eidola does not defend against an adversary observing network metadata (connection patterns, packet sizes, timing). Content is protected by TLS terminated inside the attested enclave; metadata is visible to network observers. There are really two distinct gaps here that share infrastructure but answer different questions for the user:

### Passive traffic analysis

**What it would catch.** Connection patterns, packet sizes, timing — even with TLS confidentiality, these can reveal a great deal (which model you used, the rough shape of conversations, when you are active).

**Mitigations today.** User-side: route Eidola through Tor. Eidola's protocol is plain HTTPS, so this works without modification. Server-side, for the response only: the Eidola server writes a streamed answer as fixed-size frames at a fixed interval and pads every other response (a blocking answer, an error, a refusal) to a size bucket, so an observer sees how long a response ran rather than the size and timing of its tokens ([server.md § Response shape](server.md#response-shape)). Connection patterns, request sizes and the total duration of each response remain visible.

**Future.** We consider this in-scope as an Eidola problem to address, but do not yet have a committed plan. Explored directions include offering a Tor hidden service endpoint and partnering with independent organizations to provide oblivious HTTP (oHTTP) or MASQUE/CONNECT-style transports that decouple network identity from request content.

### Network identity as a linking factor

**What it would catch.** Even a single connection to Eidola from a unique IP is itself an identity signal: an observer (or Eidola's own network logs, were they to exist) can correlate "a connection from IP X" with the account billed at approximately the same time, undermining the unlinkability invariants in [privacy-guarantees.md §2](privacy-guarantees.md#2-unlinkability) at the transport layer rather than at the application layer.

**Mitigations today.** User-side: use Tor, or a reputable VPN provider like Mullvad. Both break the direct IP↔account correlation by inserting a third party that doesn't share data with Eidola.

**Future.** Same direction as above (oHTTP, MASQUE, partner relays). The Eidola-side mitigation here is partnering with an independent organization whose role is to terminate the network connection so that no single party — Eidola included — sees both the network identity and the account it corresponds to.

## Build chain opacity

### Non-source-bootstrapped components in the trust chain

**What it would catch.** Build-pipeline subversion in a stage we don't fully source-bootstrap.

**Current behavior.** Several components of the Eidola build chain are pinned by hash and used reproducibly, but are themselves not fully source-bootstrapped:

- **macOS Nix builds.** Hermetic and reproducible (`narHash` / `archiveSha256` pinning of the unsigned payload), but rely on the Apple SDK / Xcode toolchain as opaque inputs. Cross-compiling macOS binaries from Linux is not viable today, so macOS releases must be built on macOS. The Apple envelope (Developer ID, notarization) is non-reproducible by construction and is bound only in the human attestation; see [verification.md](verification.md) and [apple-distribution.md](apple-distribution.md).
- **Linux GUI Nix build.** Hermetic and reproducible (`narHash` / `archiveSha256`) from the open nixpkgs toolchain, but not source-bootstrapped in the StageX sense. The Nix installable is a glibc dynamic binary with nixpkgs Mesa referenced via `VK_ADD_DRIVER_FILES` (those driver bytes are not in the archive). A desktop GUI that talks to host Vulkan must be glibc; the musl-by-design StageX pipeline cannot produce it. Separately, that artifact is **Nix-shaped**: its archive is a store-path serialization whose binary resolves its libraries out of `/nix/store`, so it is not a download a non-Nix user can extract and run — the Debian package is the installable for everyone else, and it carries no GPU bytes at all ([verification.md](verification.md#the-two-linux-installables)).
- **`cvmimage` and OVMF firmware.** Pinned by hash, but not source-bootstrapped. The server's enclave boots OVMF with a `cvmimage` kernel and initramfs, bound into its measurement. An engine deployment boots a `cvmimage` IGVM image with no firmware, bound into its pin (MRTD); its GPU variant carries the NVIDIA driver, CUDA userspace and GPU firmware inside that measurement. The provider builds `cvmimage` hermetically with Nix from pinned nixpkgs and documents the build as byte-for-byte reproducible, with an audit procedure, but Ubuntu packages and NVIDIA's driver, CUDA and firmware enter it as hash-pinned binary archives, and Eidola has not rebuilt a release to confirm it. So they cannot be changed silently, but the original build chain is more trusted than we ideally want.

**What constrains it today.** Each of these has digest pinning and provenance verification at the import boundary (committed sha256 pins for OVMF and the `cvmimage` release manifest — with Sigstore provenance verified on top — and `narHash` / `archiveSha256` for Nix outputs), so silent substitution is detectable. The gap is that the upstream *builders* of those artifacts are trusted to a degree we don't fully audit.

**Future.** This is a long-term direction matched to ecosystem progress: source-bootstrapped macOS toolchains, source-bootstrapped CVM/firmware builds. Rebuilding a pinned `cvmimage` release and recording the result is within reach now. We follow the relevant ecosystems and will adopt as they mature. Until then, this is an unavoidable residual.
