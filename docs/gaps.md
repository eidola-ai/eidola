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

EDIT: verify that this is correct. It probably is, but my recollection
is fuzzy on this topic.

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

**Mitigations today.** The client surfaces `released_at` to the user
before approving an install, and a public release-cadence statement
makes "the latest release is older than the cadence" a question the
community can ask in the open.

EDIT: This mitigation doesn't make sense to me: the client doesn't
mediate a fresh install. A user installs this directly by downloading
and executing it (via the web or some package manager). We currently
don't have a public release cadence, but will in the future.

**Future.** Ship a freshness anchor: a witness checkpoint, a
Bitcoin-block reference, or a co-signed signed-tree-head. Any of
these closes the first-install gap; choosing among them is partly a
matter of which witness ecosystem matures fastest.

EDIT: I'm not 100% sure how these help, for the same reason. We
might need to discuss.

### Multi-jurisdiction attestant distribution

**Current behavior.** Attestants share a small operational surface
and may share a jurisdiction.

**Future.** Distributing pinned attestants across hardware vendors,
custody arrangements, and jurisdictions raises the cost of
coordinated legal compulsion — the central concern of guarantee
[G6](privacy-guarantees.md#g6-no-compelled-subversion-without-disclosure).
This is a matter of organizational rollout, not engineering, but it
is named here because it is part of what the guarantee depends on.

## Hardware

### Trust in confidential-compute vendors

**Current behavior.** Eidola trusts AMD, Intel, and NVIDIA to issue
genuine attestation chains. A vendor issuing fraudulent attestations
for an enclave that does not in fact provide confidential compute
would defeat that layer of the chain.

**Mitigations today.** Multi-vendor coverage (SEV-SNP + TDX, with
NVIDIA confidential compute on the upstream inference layer) limits
exposure to a single vendor compromise. The attestation chain itself
is cryptographically witnessed; a forgery would need the vendor's
hardware-bound signing key.

EDIT: I'm not sure if this is a real mitigation or if it actually
increases the surface area. By supporting *both* AMD and Intel, a
compromise of *either* root cert (for example) would allow a forged
attestion to be accepted. As far as I am concerned, this is not
mitigated, just accepted. Push back if I'm incorrect about this.

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

**Mitigations today.** Users who need metadata privacy can layer
Eidola behind Tor or a similar anonymity network. Eidola's protocol
is plain HTTPS, so this works without modification.

**Future.** No active work on a metadata-privacy story is committed;
this is named explicitly so readers know it is out of scope and
have to make their own provision if they need it. See
[threat-model.md#a2-a-passive-network-observer](threat-model.md#a2-a-passive-network-observer)
for the explicit residual.

EDIT: we *do* consider this in-scope, but do not have a concrete plan to address it. Explored approaches include offering a Tor hidden service and partnering with other independent organizations to improve unlinkability through use of oHTTP, MASQUE/CONNECT, etc. There are probably actually 2 distinct threats here: passive traffic analysis and network identity as a linking factor. Current guidance is to use Tor to address the first, and to use Tor or a well reputed VPN provider like Mullvad to address the second. We might want to tackle these separately.

## Inference upstream

### Sigstore re-verification of upstream measurements at runtime

**Current behavior.** `releases/trust/tinfoil-enclaves.json` is
populated by a workflow that verifies upstream provenance via
Sigstore before opening a PR. At runtime, the server checks the
upstream enclave's measurement against the static allowed list it
was built with. It does not re-verify the upstream's Sigstore
provenance on every request.

**What constrains it today.** The static allowed list is itself a
release-gated value: a new upstream measurement only becomes
trusted after a normal source change, which carries the human
attestation. The Sigstore verification step happens at PR-creation
time, not runtime.

**Future.** Continuous re-verification at runtime would catch a
hypothetical case where a measurement passed PR review (perhaps via
a now-rotated key) but is no longer verifiable today. This is
defense-in-depth on an already-gated path.

EDIT: I don't think we'll ever add continuous re-verification. This
is static by design. However, we *would* like to run our own inference
directly in our server, still hosted on Tinfoil's infrastructure. The primary risk with inference upstream is that *their* builds are very rigorous, but don't
adhere to all the strict properties that we have, including source-
bootstrapped reproducible builds and human release attestations.
