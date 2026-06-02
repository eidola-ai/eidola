# Threat Model

This document names who Eidola defends against, who it does not, and
what is left as residual trust. It is deliberately complete on the
*scope* side — knowing what we **do not** protect against is part of
knowing what protection means.

For each adversary, the question is the same: *given the
[privacy guarantees](privacy-guarantees.md), what can this adversary
do, and what stops them?*

## Adversaries we defend against

### A1. A curious or malicious Eidola operator

**Capability:** Operates Eidola's account-management infrastructure
and could in principle observe its logs, databases, and network.

**What stops them:** The boundary between the *linked* account
surface and the *unlinked* inference surface is enforced at the type
level in the server, and the inference surface uses anonymous
credentials (Privacy Pass ACT). The operator can see *that* a
credential was redeemed, never *whose* credential it was. Inference
content never reaches the operator's logging surface at all. Telemetry
on the inference path is restricted to model name, token counts, status,
and latency.

**Residual exposure:** The operator could attempt to deploy a server
binary that violates these properties. The client refuses to talk to
any server whose attestation does not match the measurement pinned in
the client build, so deploying a violating server requires the
operator to also ship a violating *client release*. That path is
blocked by guarantees [G3](privacy-guarantees.md#g3-no-silent-code-change),
[G5](privacy-guarantees.md#g5-no-backdoor), and
[G6](privacy-guarantees.md#g6-no-compelled-subversion-without-disclosure).

### A2. A passive network observer

**Capability:** Sees all packets between the user's device and Eidola's
servers, including TLS metadata (SNI, IP, timing).

**What stops them:** TLS confidentiality and integrity. Beyond that,
the server's TLS certificate carries the enclave attestation in its
SAN, so even an attacker with a valid WebPKI cert for the same
hostname cannot impersonate the server — the client checks the
attestation, not just the cert chain.

EDIT: the TLS certificate for tinfoil enclaves do *not* contain their
attestation in a SAN, do they? Our verifier doesn't use this mechanism.
Is this relevant to a passive network observer anyway?

**Residual exposure:** Network metadata (the fact that a connection
occurred, its size, its timing) remains visible. Eidola does not
defend against traffic analysis. Users who need that property need
to layer Eidola behind Tor or a similar anonymity network. See
[gaps.md](gaps.md) for ongoing work.

### A3. An active network attacker (MITM)

**Capability:** Can intercept, modify, and re-route traffic between
the user's device and Eidola.

**What stops them:** Mutual binding between the TLS handshake and the
enclave attestation. The client issues an inline
`GET /.well-known/tinfoil-attestation` over the same TCP+TLS
connection as the real request, and verifies that the attestation's
`report_data` field is bound to `sha256(SPKI(peer_cert))`. An
attacker presenting a valid certificate for the same hostname
without the matching attestation fails the check; an attacker
presenting a stale attestation document fails the SPKI binding.

**Residual exposure:** An attacker who has compromised both the TLS
private key inside the enclave *and* obtained a valid attestation
document for that key has defeated this. The TLS key is sealed
inside the enclave by the confidential-compute runtime; defeating
this requires defeating the hardware. Tinfoil is also adding
per-handshake nonces in `report_data` upstream; once that lands,
even key exfiltration no longer suffices.

EDIT: Add a new section on a malicious infrastructure operator. It's
a threat that Tinfoil, who provides our infrastructure, impersonates
us running an instance with different configuration. This is mitigated
by committing every relevant configuration property in the attested
config, including hashes of all injected secrets that are enforced on
boot. This makes it impossible for them to change anything. Further,
our client:server pairing prevents rollback attacks where a malicious
Tinfoil operator or Eidola employee who could deploy with a valid
WebPKI cert on a real enclave using a real signed old Eidola server
version with a known vulnerability from ever being accepted by a
client that has already been updated.

### A4. An attacker who compromises the Eidola release pipeline

**Capability:** Has gained write access to the GitHub repo, the CI
system, or the OIDC identity used to sign artifact manifests.

**What stops them:** The release pipeline's signature alone is not
sufficient for the client to accept a release. The client *also*
requires a [human attestation](releases.md) signed by a pinned
attestant from hardware under their exclusive physical control,
posted to a public transparency log (Rekor). An attacker who controls
CI cannot mint that signature. An attacker who has additionally
compromised an attestant's hardware-bound key has crossed a higher
bar; the verifier enforces a minimum number of independent
attestations, raising the cost as more attestants come online.

**Residual exposure:** A compromise of every pinned attestant
simultaneously, sufficient to extract or coerce each
hardware-bound signing key, defeats this. The mitigation is
distribution of attestants across hardware tokens, jurisdictions,
and physical custody, plus a public minimum threshold pinned in
the *prior* client so the bar cannot be lowered by the compromise
itself.

### A5. A compromised dependency (supply chain)

**Capability:** Has subverted a third-party crate, container base
image, or build tool that Eidola depends on.

**What stops them:** Source-bootstrapped reproducible builds
(StageX on Linux, hermetic Nix on macOS), pinned and digest-verified
container images, and an explicit dependency surface. Every release
attestation includes a claim that the attestant has personally
reproduced the build from source. A divergence between CI's build
output and a reproducer's build output is detectable as a hash
mismatch on the signed manifest.

EDIT: we might also reference our source-committed artifact-manifest
approach here.

**Residual exposure:** A dependency compromise that occurred *and*
was incorporated *and* was reviewed by all attestants without being
caught is not detected by this mechanism. Defense-in-depth comes
from minimal dependency surface and explicit diff review in the attestation
flow.

EDIT: we need to be more realistic about this; tooling is a huge part
of the "defense in depth" and no single human or even AI read every
single line of code in every single production and build pipeline dep. The
most novel thing we do here is full reproducibility with StageX/nix, which eliminates an
entire category of "undefined" risks, but so much else comes from CVE
monitoring (via dependabot, etc). And to note, nix on macos is not "fully
source bootstrapped" in the way stagex is: the latter literally starts
with hand-written assembly, builds a minimal c compiler, uses it to build
the full c compiler, uses it to build c++, uses it to build rust, etc. We
*must* compile for macos on macos, and with that comes opacity. The real
takeaway is that we are at the bleeding edge of best practices here, and
to the absolute highest extent possible, dependencies are explicit,
pinned, and minimal. We probably don't need to say all this here, but
maybe some? Much of these aren't novel at all and are more like the
corporate policies I'd put in a SOC2. I've been trying to minimize
emphesizing "policy" points that are non-structural, at risk of diluting
the impact of the truly important pieces. Let's think about an approach
to that issue: do we consolidate "non-structural policies" to some other
place? That wouldn't read well... I'm not really sure how to approach this?


### A6. A legally-compelled Eidola engineer

**Capability:** A court order, technical capability notice, or other
legal compulsion directs an engineer to weaken the privacy
guarantees or introduce a backdoor.

**What stops them:** Guarantee [G6](privacy-guarantees.md#g6-no-compelled-subversion-without-disclosure)
requires the attestant to sign, under their legal identity, that they
are *not* under such compulsion. A coerced engineer who is also
gagged must either (a) decline to sign, breaking the release and
sending a public signal, or (b) sign falsely and incur the legal
exposure they were trying to avoid. As attestant counts grow, the
required-threshold check forces the adversary to compel multiple
independent engineers, multiplying the legal and operational risk
on the adversary's side.

**Residual exposure:** A jurisdiction able to compel every pinned
attestant in coordinated secrecy defeats this silently. The
mitigation is multi-jurisdiction attestant distribution; this is
[ongoing work](gaps.md).

## Adversaries we do not defend against

### N1. A compromised local environment

If the user's device is compromised — malicious app with sufficient
privilege, OS-level surveillance, malware in the Eidola binary's
host process — most user-facing privacy properties fall, regardless
of what Eidola does. The client's fail-safe defaults and embedded
trust root limit *what code runs*, but they cannot prevent an
already-trusted process from observing the user. This is named
explicitly in [G1's "what this does not guarantee"](privacy-guarantees.md#g1-inference-content-confidentiality).

### N2. A hardware-manufacturer forgery

If AMD, Intel, or NVIDIA were to issue fraudulent attestation chains
(for example, signing measurements for an enclave that does not in
fact provide confidential compute), the corresponding layer of the
verification chain falls. This is residual trust we currently
accept; the long-term mitigation is open hardware roots like
OpenTitan, tracked in [gaps.md](gaps.md).

### N3. A traffic-analysis adversary

An adversary observing network metadata — connection patterns, packet
sizes, timing — can infer a great deal even when content is
encrypted. Eidola does not defend against this. Users who need
metadata privacy should layer Eidola behind Tor or a similar
anonymity network.

### N4. A user who voluntarily leaks

A user who pastes their email into a prompt, copies inference output
to a third-party service, or screenshots their chat history has
exited Eidola's privacy boundary by their own action. Eidola does
not redact, classify, or filter user input.

## Residual trust, named explicitly

Every system has a foundation it cannot verify from inside itself.
Ours is named so a reader can decide whether the foundation is
acceptable for their threat model:

| Trust anchor | What we rely on it for | What lowers the exposure |
|---|---|---|
| **Confidential compute vendors** (AMD, Intel, NVIDIA) | The attestation chain proves real enclave execution | Multi-vendor coverage (SEV-SNP + TDX), future OpenTitan-style roots |
| **WebPKI** (Let's Encrypt) | The TLS cert presented by the server is genuinely the one issued for the hostname | The cert is bound to the enclave attestation by the client; a forged WebPKI cert alone is not enough |
| **Sigstore Rekor** (Linux Foundation) | The transparency log entry for a release attestation is genuine and never removed | Inclusion proofs are verified locally; checkpoint signatures are a [gap](gaps.md) |
| **GitHub OIDC** (Fulcio identity binding) | The release-signing CI workflow ran under the identity it claims | Pinned identity pattern + tag; manual escape via [rotation](../releases/README.md#rotating-the-ci-signing-workflow) |
| **The user's prior client binary** | Embedded trust root has not been silently subverted before install | Public release record + signed continuity check between releases |
| **The user's hardware and OS** | Process isolation, key storage, code execution integrity | Outside Eidola's scope; named in [N1](#n1-a-compromised-local-environment) |

EDIT: Our Rekor checks would fail safe, right? If rekor.sigstore.dev is down, we can't verify a new build; or perhaps we're susceptible to WebPKI compromised MITM? Double check the actual gap here.

EDIT: OIDC compromise is mitigated by the human signing. Is the rotation scheme relevant here?

Each of these is a place where a sufficiently motivated and capable
adversary could break the chain. They are not weaknesses we are
hiding; they are the cost of building software at all. Where we have
in-progress mitigations, they are in [gaps.md](gaps.md).

EDIT: Somewhere we need to address potential vulnerabilities of confidential compute hardware itself, including things like side channel attacks, as a residual risk.
