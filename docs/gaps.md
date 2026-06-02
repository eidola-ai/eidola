# Known Gaps

Every piece of the Eidola trust chain that is intentionally deferred is
catalogued here. Each gap closes a specific class of attack that is
already constrained by other parts of the chain — but they are real and
worth understanding. Reading this page is the fastest way to see what Eidola
does not yet defend against.

The cryptographic-verifier gaps are also noted at the top of
[`crates/eidola-app-core/src/updater/ci_sigstore/mod.rs`](../crates/eidola-app-core/src/updater/ci_sigstore/mod.rs)
and `rekor.rs`, and the install-side gap is at the `TODO (step 5)`
marker on `verify_each_artifact_hash` in
[`crates/eidola-app-core/src/updater/mod.rs`](../crates/eidola-app-core/src/updater/mod.rs).

## Cryptographic verifier

### SCT verification in the Fulcio leaf certificate

**What it would catch.** A malicious or compromised Fulcio issuing
certificates for OIDC identities it shouldn't — the Signed
Certificate Timestamp proves the cert was logged in a public CT log.

**What constrains it today.** The OIDC-identity match and the
Fulcio chain walk are the primary binding. The SCT check is
defense-in-depth on top of those, not a single point of failure.

### Rekor checkpoint signature verification

**What it would catch.** The Rekor instance silently forking a
side-tree just for our entries — the inclusion proof we compute is
mathematically valid but roots to a tree the public never sees.

**What constrains it today.** The Signed Entry Timestamp already
requires the Rekor public key to vouch for our specific entry. The
checkpoint adds independence from private forks by verifying against
the publicly-witnessed log head.

### Artifact-hash check at install time

**What it would catch.** A tampered binary download — the *manifest*
is signed and content-verified, but the actual binary bytes that
would run are not yet hashed against the manifest's declared
digests.

**What constrains it today.** The verifier already proves
`artifact-manifest.json` itself is authentic and unmodified. The
install-time hash check lives naturally in a to-be-implemented install /
atomic-replace step, once we know which platform's artifact the user is
downloading. The verifier code is in place; only the wiring to the
download path is deferred.

### Multi-hop / fast-forward continuity

**What it would catch.** A client that skips multiple releases (e.g.
v1.0 → v1.5, missing v1.1–v1.4). Today the continuity gate requires
strict equality between `release.previous_release.git_commit` and
the installed commit, so an out-of-date client must update through
every release in order.

**What constrains it today.** Strictly sequential is the safer
floor. Relaxing to "fast-forward reachable via GitHub commits API"
is a small follow-up that's only worth doing once the release
cadence makes sequential updates painful in practice.

## Operational

### Install / atomic-replace

**Current behavior.** `eidola update` runs the full verification
pipeline and prints the verified attestation prose, but does not
download or swap the binary.

**Future.** Step 5: download the artifact for the user's platform
from `artifact-manifest.json`, hash-verify, atomic-replace, restart.
Platform-specific (CLI = file swap; macOS GUI = staged swap on next
launch).

### Single-attestant policy

**Current behavior.** `MIN_HUMAN_ATTESTATIONS` (embedded in the
client, sourced from `releases/trust/trust-constants.json`) is `1`
in current releases — only one engineer needs to attest for a
release to verify.

**Future.** Once a co-attestant key is provisioned and added to
`trusted_attestant_fingerprints`, bumping `min_human_attestations`
to `2` (in a release signed under the *current* threshold) makes
every subsequent release require independent corroboration. The
verifier already supports arbitrary M-of-N; the second key just
hasn't been generated yet.

### First-install downgrade

**Current behavior.** A fresh client (no prior installed
`git_commit`) bypasses continuity, so an adversary serving an
internally-consistent *older* `release.json` could route them onto
a real-but-stale release.

**Mitigations today.** None that the client can enforce. A first
install is exactly the case where the client has nothing to
compare against, so the surface "did you download a current
release or a stale-but-internally-consistent one?" lives outside
the client's reach today.

**Future.** A *freshness anchor* is something a fresh install
*can* validate at the moment of download, without needing prior
state. The general shape: every release embeds (or references) a
recent timestamped artifact from a public, append-only system
that an attacker cannot retroactively forge.

### Multi-jurisdiction attestant distribution

**Current behavior.** Attestants share a small operational surface
and may share a jurisdiction.

**Future.** Distributing pinned attestants across hardware vendors,
custody arrangements, and jurisdictions raises the cost of
coordinated legal compulsion — the central concern of the
no-coercion attestation claims in
[privacy-guarantees.md §6.4](privacy-guarantees.md#6-release-integrity)
and the bounded claim in
[§8.7](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise).
This is a matter of organizational rollout, not engineering, but it
is named here because it is part of what the guarantee depends on.

## Hardware

### Trust in confidential-compute vendors

**Current behavior.** Eidola trusts AMD, Intel, and NVIDIA to issue
genuine attestation chains. A vendor issuing fraudulent attestations
for an enclave that does not in fact provide confidential compute
would defeat that layer of the chain.

**Mitigations today.** Limited. The use of WebPKI for our TLS
certificate provides a defense in depth, ensuring that an outside
party issuing a fraudulent attestation must also product or obtain
a fraudulent WebPKI certificate. However, this provides little
resistance in the case of a malicious insider. Generally, we accept
hardware vendor trust as residual.

**Future.** Open hardware roots like OpenTitan reduce the scope of
vendors the trust chain depends on. This is an industry-wide
direction, not an Eidola-specific roadmap item, but it is the
long-term mitigation for this residual trust.

## Network / metadata

### Traffic analysis

**Current behavior.** Eidola does not defend against an adversary
observing network metadata (connection patterns, packet sizes,
timing). Content is protected by TLS terminated inside the
attested enclave; metadata is visible to network observers.

There are really two distinct gaps here that share infrastructure
but answer different questions for the user:

#### Passive traffic analysis

**What it would catch.** Connection patterns, packet sizes,
timing — even with TLS confidentiality, these can reveal a great
deal (which model you used, the rough shape of conversations,
when you are active).

**Mitigations today.** User-side: route Eidola through Tor.
Eidola's protocol is plain HTTPS, so this works without
modification.

**Future.** We consider this in-scope as an Eidola problem to
address, but do not yet have a committed plan. Explored
directions include offering a Tor hidden service endpoint and
partnering with independent organizations to provide oblivious
HTTP (oHTTP) or MASQUE/CONNECT-style transports that decouple
network identity from request content.

#### Network identity as a linking factor

**What it would catch.** Even a single connection to Eidola
from a unique IP is itself an identity signal: an observer (or
Eidola's own network logs, were they to exist) can correlate
"a connection from IP X" with the account billed at
approximately the same time, undermining the unlinkability
invariants in
[privacy-guarantees.md §2](privacy-guarantees.md#2-unlinkability)
at the transport layer rather than at the application layer.

**Mitigations today.** User-side: use Tor, or a reputable VPN
provider like Mullvad. Both break the direct IP↔account
correlation by inserting a third party that doesn't share data
with Eidola.

**Future.** Same direction as above (oHTTP, MASQUE, partner
relays). The Eidola-side mitigation here is partnering with an
independent organization whose role is to terminate the
network connection so that no single party — Eidola included
— sees both the network identity and the account it
corresponds to.

## Inference upstream

### Upstream-provider trust-discipline mismatch

**Current behavior.** Inference runs in a separately-attested
enclave operated by the upstream provider (currently Tinfoil).
Tinfoil's release pipeline is robust — signed measurements,
Sigstore provenance, public source — but it does not yet match
the discipline applied to Eidola's own releases. Specifically:

- Tinfoil's builds are **not source-bootstrapped reproducible**
  in the StageX sense. They are hermetic and provenance-attested
  through GitHub's CI attestation, which is rigorous, but
  shaped differently than Eidola's.
- Tinfoil does **not yet ship per-release human attestations
  under named legal identities** the way Eidola releases do.

A user's chain of trust at the inference layer therefore ends at
Tinfoil's release discipline, which is non-trivially different
from Eidola's.

**Future.** Bring the inference pipeline into this repo (still
running on Tinfoil's infrastructure), so the same
source-bootstrapping + human-attestation discipline applies
end-to-end. The inference enclave would then be built and
released through Eidola's own release flow rather than
trust-bridged through a separate pipeline.

## Build chain opacity

### Non-source-bootstrapped components in the trust chain

**What it would catch.** Build-pipeline subversion in a stage we
don't fully source-bootstrap.

**Current behavior.** Several components of the Eidola build
chain are pinned by hash and used reproducibly, but are
themselves not fully source-bootstrapped:

- **macOS Nix builds.** Hermetic and reproducible (`narHash`
  pinning), but rely on the Apple SDK / Xcode toolchain as
  opaque inputs. Cross-compiling macOS binaries from Linux is
  not viable today, so macOS releases must be built on macOS.
- **`cvmimage` and OVMF firmware.** Pinned by hash, but their
  build chains do not match Eidola's source-bootstrapping
  discipline. Their contents are bound into the server's
  enclave measurement, so they cannot be changed silently — but
  the original build chain is more trusted than we ideally
  want.

**What constrains it today.** Each of these has digest pinning
and provenance verification at the import boundary (Sigstore
provenance for `cvmimage`, narHash for Nix outputs, committed
hashes for OVMF), so silent substitution is detectable. The
gap is that the upstream *builders* of those artifacts are
trusted to a degree we don't fully audit.

**Future.** This is a long-term direction matched to ecosystem
progress: source-bootstrapped macOS toolchains, reproducible
CVM/firmware builds. We follow the relevant ecosystems and will
adopt as they mature. Until then, this is an unavoidable residual.
