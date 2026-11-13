# Known gaps

Every piece of the Eidola trust chain that is deferred is catalogued here. Each entry names an attack that other parts of the chain constrain but do not close.

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

**Today every client is a fresh client.** No build knows the commit it was released from, and none can learn it at build time: the release commit records the binaries' `narHash` in `artifact-manifest.json`, so a binary embedding that commit could never reach a fixed point (the same reason the manifest is excluded from the build's source). The installed commit has to come from the updater recording what it installed, which waits on promotion (Install / atomic-replace, below). Until then the continuity stage is skipped on every check, and the version gate (`latest > installed`, against the compile-time workspace version) is the only ordering the client enforces.

**Mitigations today.** None that the client can enforce. A first install is exactly the case where the client has nothing to compare against, so the surface "did you download a current release or a stale-but-internally-consistent one?" lives outside the client's reach today.

**Future.** A *freshness anchor* is something a fresh install *can* validate at the moment of download, without needing prior state. The general shape: every release embeds (or references) a recent timestamped artifact from a public, append-only system that an attacker cannot retroactively forge.

### Anonymity-set size

**Current behavior.** An anonymous credit token is unlinkable to its issuance only *within its anonymity set*: the accounts that received tokens under the same `(issuer_key, domain_separator)` during that key's issuance window ([privacy-guarantees.md](privacy-guarantees.md) §2.3). The partition parameters — weekly issuance epochs, a single non-rotating domain separator, an acceptance window one full epoch beyond issuance — are compile-time constants, but the *population* inside a partition is a deployment fact the code cannot control. Early in the deployment, or during a quiet week, a window may contain few active accounts; in the degenerate one-account case the cryptography still holds, but the operator could attribute spends by elimination.

**What constrains it today.** Temporal decoupling (§2.4) keeps issuance and redemption timestamps from being forced near-equal, the gateway persists no per-redemption timestamps, and nothing in the protocol subdivides the partition further. None of that enlarges a small population.

**Two-input requests narrow it.** A request may present two tokens ([anonymous-credits.md § Spending several tokens in one request](anonymous-credits.md#spending-several-tokens-in-one-request)), and tokens presented together are known to come from one client. Two under one key place the request among the accounts issued at least two tokens under that key; one under each of two keys, among the accounts issued under both. The client issues two tokens under each key it provisions under, so the first set is every account that provisioned under the key rather than the heavy users; an account that only ever converted just in time is not in it, and the second set is the accounts that provisioned in both epochs.

**Future.** The set grows with the user base. If population per window stays small, the epoch length can be raised (a release-visible constant change) to trade key-rotation hygiene for coarser partitions.

### Spend timing at the database

**Current behavior.** A spend writes its nullifier row twice: the nullifier with a provisional sealed refund when the request is accepted, and the final sealed refund when the request ends ([gateway.md § Unlinkability](gateway.md#unlinkability)). A request with two inputs writes both rows in one transaction, and both final refunds as it ends, so the database also sees which two nullifiers belong to one request: its input count and key set, never an amount. The row carries no timestamp, but Postgres runs outside the enclave, so whoever can watch it live — or read its write-ahead log, a replica, or a backup, whose commit records are timed — can see when each write happened, and so how long each request took. Duration follows the length of the answer, so this is a coarse, per-request content-size signal in the operator's hands, and the same database holds per-account issuance timestamps in the credit ledger.

Requests that follow one another closely are a second signal. The client asks for a bounded amount of output at a time and continues a longer answer with a follow-up request as soon as the first ends, so a long answer is a run of spends, each beginning as the one before it finishes. A tool-calling turn is such a run too, and so is a request refused for its declared cached prefix followed by the same request sent again without one. When few requests are in flight, whoever watches the writes can tell that the spends of a run came from one client, although nothing in the rows says so. If the client pays each with the change of the one before, the run is also consecutive spends of one token.

**What constrains it today.** The refund amount itself is stored sealed under a key that needs the client's own secret, at a fixed length; nothing written names an account; and the timing is the signal a network observer of the gateway already has from each response's total duration ([Passive traffic analysis](#passive-traffic-analysis)). The database is treated as adversarial, so this is a real gap rather than a theoretical one.

**Future.** Decouple the second write from completion: hold completed refunds in enclave memory and write them in batches on a fixed interval, so a write says only which interval a request finished in. Batching nullifier inserts from different requests on a fixed interval would likewise blur which rows belong to one request, and which requests follow one another. Recording the nullifier only at completion would hide the start too, but gives up refusing a double spend before it is served once more than one gateway instance shares the database, and is not planned.

### Credit left in a credential when its key lapses

**Current behavior.** Each anonymous credential is issued under one issuer key, and the change from every spend is issued under that same key, so a credential's whole remaining value lives and dies with the key. The gateway accepts a key's credentials until its `accept_until` — one full epoch after the key stops issuing ([gateway.md § Anonymity set](gateway.md#anonymity-set)) — and the client spends a credential until shortly before that deadline (`CREDENTIAL_SPEND_MARGIN`, a few minutes, so a spend begun at the edge still lands inside the window; a spend already in flight stays recoverable until the deadline itself). Whatever is still in the credential after that is lost: it was debited from the account when the credential was issued, and nothing exchanges it for a credential under a newer key.

**What constrains it today.** Every credential gets at least one full epoch of spendable life, however late in its key's issuing window it was minted, and automatic provisioning allocates a modest amount at a time ([`DEFAULT_ALLOCATION_CREDITS`](../crates/eidola-app-core/src/lib.rs)), so the exposure is bounded by one allocation's unspent change per key. A token too small for the next request is drained into it with another token, under a newer key where the older key holds no other, so a residual lapses only if no request reaches it before its key stops accepting. Issuance ahead of need adds at most two allocations to what is held at a time, keeps no buffer for a wallet that has not spent within a week, and prefers a key's early life, when its credentials have the longest to be spent ([anonymous-credits.md § Issuance timing](anonymous-credits.md#issuance-timing)). Issuance is refused outright under a key whose spend deadline has already passed, so the account is never debited into a credential that could not be spent.

**Future.** A rollover: exchanging what remains in an expiring credential for a fresh one under the current key, without that exchange linking the two. That needs its own unlinkability analysis and is not designed yet.

### Multi-jurisdiction attestant distribution

**Current behavior.** Attestants share a small operational surface and may share a jurisdiction.

**Future.** Distributing pinned attestants across hardware vendors, custody arrangements, and jurisdictions raises the cost of coordinated legal compulsion — the central concern of the no-coercion attestation claims in [privacy-guarantees.md §6.4](privacy-guarantees.md#6-release-integrity) and the bounded claim in [§8.7](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise). This is a matter of organizational rollout, not engineering, but it is named here because it is part of what the guarantee depends on.

## Anonymous credits

The decisions behind anonymous payment, and the reasons for them, are in [anonymous-credits.md](anonymous-credits.md). These are the residuals it names, beside two listed above: [Spend timing at the database](#spend-timing-at-the-database) and [Credit left in a credential when its key lapses](#credit-left-in-a-credential-when-its-key-lapses).

### Issuance-to-first-spend timing

**Current behavior.** The app keeps a small buffer of credit tokens issued ahead of need, at random intervals unrelated to any request, so a token's first spend normally comes long after its issuance ([anonymous-credits.md § Issuance timing](anonymous-credits.md#issuance-timing)). When nothing held covers a request, the client still converts credits and spends the new token immediately: on a fresh install, on the first request after a week without spending (no buffer is kept for an idle wallet), from a CLI used while the app is not running, when the balance is below one standard allocation, or for a request larger than what is held. The token converted is then the one standard size, or the remaining balance when that is less, never one sized to the request; a request that one such token beside the largest held one would not pay is refused before anything is converted ([anonymous-credits.md § Issuance sizing](anonymous-credits.md#issuance-sizing)). The database records each issuance with its account and time and sees each spend's nullifier as it is written, so whoever can watch it, or the gateway's traffic, can pair such an issuance with the spend that follows it a moment later, linking that token's first spend to its account. A prompt-cache key on that request then links the rest of its conversation lineage ([privacy-guarantees.md §2.7](privacy-guarantees.md#2-unlinkability)).

A background issuance is timed too: it follows, by one random interval of 10 to 50 minutes, the spend that took the buffer below its threshold. That says some spend in the preceding hour drained a token, which every spend in that window shares; it does not pair the issuance with a spend.

**What constrains it today.** Allocations are large enough (about $1.00) that a token pays for many requests, so even the fallback is rare for an active user; tokens can be converted ahead of time by hand (`eidola account allocate`).

**Future.** A buffer for the CLI-only case, which needs a long-lived process the CLI does not have.

### A crash mid-spend costs the request's hold

**Current behavior.** A spend records its nullifier, with a provisional refund of nothing stored on the same row in the same statement, before the request is served, and replaces that with the final refund when the request ends ([anonymous-credits.md § Provisional refunds, the live set, and ending a request early](anonymous-credits.md#provisional-refunds-the-live-set-and-ending-a-request-early)). If the gateway stops in between (a crash, a restart), the provisional refund is the spend's last: as soon as the gateway is back, recovery finds no request serving the spend, calls the provisional refund final, and the client gets back the token's balance less the hold. The hold itself, which the request may or may not have used, is lost. The same is true, for the request's partial return only, when a client recovers a spend that is still running on a *different* live gateway instance: that instance's live set is not the one recovery asks.

**What constrains it today.** The window is the duration of one request on one instance, and the client's wait is the gateway's restart time. Production runs one gateway instance, so the cross-instance case is at most a deploy overlap. The client keeps the spend's state and keeps retrying recovery until the key stops accepting.

**Future.** Charging a crashed request what it actually used would need its usage to outlive the process, which the gateway deliberately keeps nowhere outside enclave memory. Running several instances side by side would need recovery to ask the instance that holds the spend, without the database learning which requests are in flight.

### A drained balance is visible to the measured code

**Current behavior.** A request paid from two tokens drains the first: its share is that token's whole remaining balance, exactly ([anonymous-credits.md § Draining a token](anonymous-credits.md#draining-a-token)). The spend proof hides a token's balance; a drain gives that up for the drained token, to the gateway's measured code, for the length of the request. A drained token whose share is not fully charged returns as a successor holding exactly the change, a value the gateway computed, so a later drain of that successor shows a share equal to a refund the gateway issued.

**What constrains it today.** The measured server code keeps no share, charge or refund amount anywhere outside enclave memory: refunds are stored sealed under a key only the client holds, at a fixed length, and none of these values is logged or exported. With nothing retained, no one can match a drained balance to an earlier charge or refund. The guarantee for drained tokens therefore rests on the measured code, not on the cryptography.

**Future.** Rounding a drain down to a public step would reveal the balance only to that precision and leave a remainder the gateway does not know, at the cost of stranding less than the step. How coarse a step would matter depends on the population, so it is deferred.

### A client's first sight of an epoch's key

**Current behavior.** Issuer keys sit on a public calendar (one per seven-day epoch, with windows anyone can compute), each key's windows are bound into the encryption of its private key so that the database cannot edit them, and each client records the key it first sees for every epoch and refuses any other for that epoch ([anonymous-credits.md § Key rotation and partitioning](anonymous-credits.md#key-rotation-and-partitioning)). What remains is the first sight: a client with no record for an epoch (a new install, or one that fetched no key listing since the previous epoch began) accepts the key it is shown. A gateway, or a database that made a gateway mint a second key by hiding rows from it, could show that client a key nobody else has, and the client would have nothing to compare it with. A gateway brought up on an empty database is the honest form of the same thing: it mints new keys, which new clients take and existing clients refuse for up to two epochs.

**What constrains it today.** Every listing carries the next epoch's key, so a client that issues at least once an epoch records each key an epoch before it issues; singling it out means starting that early and keeping every listing it fetches consistent. A gateway refuses to mint a second key for an epoch it already holds one for, and refuses to issue at all while it holds two, so a hidden row splits the set only at an instance's start, and only for clients without a record. The client checks that every key it uses is named by the hash of its public key, and that every token and refund it receives is under that key with the request context every client under the key shares.

**Future.** Publish the sequence of keys somewhere clients can compare against without trusting the gateway that serves it (a transparency log, or keys derived from the enclave's measured state), so that a client's first sight is checked too.

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

**Current behavior.** The client → gateway path accepts SEV-SNP only, and refuses an Intel TDX presentation before authenticating it. Acceptance is driven by the caller's pins, never by the document: every pinned entry names its platform, evidence is checked only against entries of its own platform, and evidence from a platform with no entry is refused. The client's pin for the gateway is a SEV-SNP entry only — the launch digest of the platform release image, HOST_DATA (the config's SHA-256) and the guest policy — and the trust root records no TDX identity for the gateway. A document author therefore cannot select the TDX branch to sidestep the SEV-SNP pin.

The verifier accepts TDX for a caller that pins an explicit TDX entry. The one such caller is the gateway's connection to its own engine deployments, whose hosts are Intel TDX ([engine.md](engine.md#what-pins-the-engine)), and those pins are TDX-only by the same construction. Such a pin follows the platform provider's IGVM launch model, in which the guest is a single firmware-less image with no measured boot after launch. The provider ships that model only in pre-releases so far; the gateway's own enclave is on it too, on the SEV-SNP side. Each release publishes a CPU image and a GPU image (`X+gpu`) with different MRTDs, and an engine deployment must name the GPU image (`engine_deployment::check_cvm_image`):

- **Launch identity.** MRTD, which the TDX module measures over every page of the image and which is fixed per image release, and MRCONFIGID, the SHA-256 of the deployment's config followed by sixteen zero bytes, which the guest refuses to boot without. RTMR0–RTMR3 must all be zero. `measure-enclave` recomputes MRTD from the release's published image, checked against the release manifest, and MRCONFIGID from the config bytes. The release manifest is committed, pinned by the config, and every gateway build requires the pin's MRTD to be the one it states.
- **Machine policy and machines**, for the report fields an image cannot fix. The TDX module (MR_SEAM), exact TD attributes (debug off, `SEPT_VE_DISABLE` on — a debug relaunch of the pinned image has the same MRTD) and XFAM, a TEE TCB SVN floor, the QE vendor, a floor on the Intel collateral's evaluation-data number, and an allowlist of individual machines by PPID: the Platform Provisioning ID in the host's Intel-signed PCK certificate, which names one registered platform. All of these are derived from the provider's signed platform endorsements (below). Intel's platform, QE and TDX-module TCB statuses must all be `UpToDate`.
- **Quote verification.** The PCK chain verifies to the pinned Intel SGX root with both Intel CRLs consulted. TCB Info and QE Identity carry valid Intel signatures and are inside their validity windows. The quote and QE report signatures verify. All of it runs offline, against the Intel PCS responses the document carries.

`REPORT_DATA` binding and the envelope checks are the same as on the SEV-SNP path. An entry may also declare how many GPUs its enclave runs with. The document must then carry exactly that many NVIDIA GPU evidence items, each labelled with this handshake's nonce.

**What constrains it today.** Only a caller that pins a TDX entry can accept TDX: the gateway's engine connections, and no engine deployment is pinned yet. For an engine pin:

- **GPU evidence is counted, not appraised.** The CPU measurement vouches for boot code that refuses to start unless the GPUs attest. The GPU image's measured boot checks the GPUs' PCI device IDs, runs NVIDIA's `nvattest` with its local verifier on each GPU (measurements against NVIDIA's reference values, and the certificate chains and revocation status of the report and reference values), requires exactly `gpus` reports, and only then sets the ready state that lets CUDA work reach the GPUs; any failure ends the boot. The engine, before it opens a GPU, then requires NVML to report confidential computing on, DevTools mode off, the production environment and that ready state, which NVIDIA's verifier does not check as such (`EIDOLA_ENGINE_GPU_CC=required`, which the gateway's build requires of every pin). On each handshake the gateway counts the items and checks each one's `nonce` member, which the measured attestation shim writes beside NVIDIA's report; the quote's `REPORT_DATA` covers the whole device section. NVIDIA's certificate chain, the SPDM report signature, the nonce inside that report and the reference measurements are not verified by the gateway, nor by the provider's own client. So the GPUs are appraised inside the guest at boot, not per handshake, and that boot depends on reaching NVIDIA's reference-value and revocation services.
- **Machine policy is the provider's signed statement, as of a committed release.** The provider publishes Sigstore-signed platform endorsements (`platform-vX.Y.Z` releases of its image repository) that map each of its machines, TDX ones by PPID, to a named policy (TDX module, TD attributes, XFAM, TEE TCB SVN floor, QE vendor, collateral floor). An engine deployment names one policy of one release. That release's file and its attestation bundle are committed, with the file's hash, and `measure-enclave` and every gateway build verify them offline against the committed Sigstore trusted root: signed by the provider's release workflow at exactly that tag in its repository, on a GitHub-hosted runner, logged in Rekor, about a file with that hash. The pin's policy, and its PPID allowlist of every machine the release maps to that policy, are derived from the file, and a build whose pin differs is refused. No network is read to do it. What this leaves:
  - The values are the provider's. The pin accepts exactly the machines and floors the provider endorses; Eidola checks that the provider signed them, not that they are the right ones.
  - They are as of the committed release. A release the provider publishes later (a raised floor, a machine retired) applies only once it is committed, and the provider's own client also asks for a recent signed freshness statement about the release it applies, which a compiled-in pin cannot do: its release is fixed when the gateway is built. A scheduled check reports a newer release, but adopting it is a reviewed change and a new gateway build.
  - The Sigstore verification omits the Fulcio leaf's SCT and the Rekor checkpoint signature, as the client's update verification does ([SCT verification](#sct-verification-in-the-fulcio-leaf-certificate), [Rekor checkpoint signature verification](#rekor-checkpoint-signature-verification)).
  - The PCK certificate's platform flags — dynamic platform, cached keys and SMT enabled — are hand-supplied, because the endorsements do not carry them. Each must be stated in the pin, and a host whose flag differs is refused. A sample multi-package TDX host asserts all three, and accepting SMT in particular admits cross-thread side channels ([threat-model.md](threat-model.md)).

  There is no platform-family (FMSPC) allowlist: the family is a fixed property of each listed machine, so it would refuse nothing the PPID allowlist admits.
- **Collateral can be replayed within its window.** As with the AMD CRL, a relay can serve older Intel-signed collateral (CRLs, TCB Info, QE Identity) until its signed validity ends. That delays recognition of a newer revocation or TCB recovery; the evaluation-data floor bounds how old it may be.

**Future.** Measure and pin the first engine deployment, on a stable release of the IGVM line once there is one. Appraise GPU evidence against NVIDIA's roots and reference values. On the gateway's SEV-SNP pin, appraise the machine as the TDX path does: the provider's signed platform endorsements also carry SEV-SNP policies (PLATFORM_INFO, launch-TCB and firmware floors) and an allowlist of chips, which the client does not read yet, so any genuine chip at the TCB floor is accepted. The images carry no ID block, so there is none to pin.

## Network / metadata

Eidola does not defend against an adversary observing network metadata (connection patterns, packet sizes, timing). Content is protected by TLS terminated inside the attested enclave; metadata is visible to network observers. There are really two distinct gaps here that share infrastructure but answer different questions for the user:

### Passive traffic analysis

**What it would catch.** Connection patterns, packet sizes, timing — even with TLS confidentiality, these can reveal a great deal (which model you used, the rough shape of conversations, when you are active).

**Mitigations today.** User-side: route Eidola through Tor. Eidola's protocol is plain HTTPS, so this works without modification. Server-side, for the response only: the gateway writes a streamed answer to its client as fixed-size frames at a fixed interval and pads every other response (an error, a refusal) to a size bucket, and an engine shapes its answer to the gateway the same way, so an observer of either connection sees how long a response ran rather than the size and timing of its tokens ([gateway.md § Response shape](gateway.md#response-shape)). Connection patterns, request sizes and the total duration of each response remain visible, and an engine's host can still observe the engine's own activity.

Request sizes say more than they seem to. A request carries the conversation so far, so the size of each request less the one before it is the size of what was added: the previous answer and the new message together, for a new turn. When the client continues a long answer by itself, the follow-up request adds only the text generated so far, so its size gives that text's exact length in bytes, where the response's own shape gave only its duration. Padding request bodies to size buckets would close this and is not done today.

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
- **Inference engine image.** Built reproducibly with Nix, inside a pinned container and outside Nix's own sandbox, from the official prebuilt Rust release and nixpkgs' glibc, not source-bootstrapped. The engine loads NVIDIA's driver libraries at runtime, which requires a glibc dynamic binary that the StageX pipeline cannot produce. Its GPU kernels are compiled ahead of time with NVIDIA's closed CUDA toolkit: the compiled kernels are reproducible and every image build checks them against the committed `kernels.manifest.json`, but the compiler itself is opaque. The NVIDIA driver is not in the image; the platform's GPU boot image supplies it, inside that image's measurement.
- **`cvmimage`.** Pinned by hash, but not source-bootstrapped. The gateway's enclave boots a `cvmimage` SEV-SNP IGVM image with no firmware, bound into its pin (the launch digest). An engine deployment boots a `cvmimage` IGVM image with no firmware, bound into its pin (MRTD); its GPU variant carries the NVIDIA driver, CUDA userspace and GPU firmware inside that measurement. The provider builds `cvmimage` hermetically with Nix from pinned nixpkgs and documents the build as byte-for-byte reproducible, with an audit procedure, but Ubuntu packages, Docker's container runtime (`dockerd`, `containerd`, `runc`), and NVIDIA's driver, CUDA and firmware enter it as hash-pinned binary archives, and the build takes its compilers and tools from the NixOS binary cache. Eidola rebuilds each release it pins and compares it byte for byte with the published one ([procedure and last result](contributing/rebuilding-platform-images.md)), which confirms the image follows from the source and those inputs, not what the inputs contain. So they cannot be changed silently, but the original build chain is more trusted than we ideally want.

**What constrains it today.** Each of these has digest pinning and provenance verification at the import boundary (the `cvmimage` release manifest pinned by sha256 in the committed config — with Sigstore provenance verified on top — and `narHash` / `archiveSha256` for Nix outputs), so silent substitution is detectable. The gap is that the upstream *builders* of those artifacts are trusted to a degree we don't fully audit.

**Future.** Rebuilding a pinned `cvmimage` release and recording the result is within reach now. Source-bootstrapped macOS toolchains and CVM image builds depend on ecosystem progress.
