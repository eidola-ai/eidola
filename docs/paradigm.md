# How to think about Eidola

Eidola exists because privacy and autonomy in AI are easy to *claim* and
hard to *prove*. Most "private AI" products ask you to trust a vendor's
intentions. Eidola is structured so you don't have to.

The operating paradigm is **user sovereignty**: the end user runs
software they control, on hardware they choose, talking to systems whose
properties they can independently verify. Where the user relies on
infrastructure provided by us, multiple layers ensure that they keep the
ability to verify exactly what is running, and to hold Eidola accountable
to the [privacy guarantees](privacy-guarantees.md) it publishes.

Three principles flow from that paradigm. They show up everywhere in the
design, and they are the lens through which the rest of these docs are
written.

## 1. The client is sovereign

The client is the user's entry point and the arbiter of trust. Every
decision about *what to run*, *what to trust*, and *whether to talk to a
given server* is made locally, against values that were compiled into
the binary before it shipped.

A given client binary trusts **exactly one server build**. The trust
root — the measurements, identity patterns, fingerprints, and policy
constants — is embedded at compile time. Every Eidola release is a
coordinated rebuild of clients *and* server so that their values
correspond. There is no runtime trust handoff, and no API the client
calls to "discover" who to trust. See [client.md](client.md).

The client is also designed to **fail safe**: if anything in the
verification chain cannot be confirmed, the connection is refused rather
than downgraded. There is no quiet fallback to an unverified path.

## 2. Code is the trust boundary, not policy

Privacy guarantees in Eidola are properties of the *running code*, not
of an operator's stated policy. The user can verify which code is
running because:

- **The source is public and reproducible.** Anyone can rebuild the
  released binaries from the committed source and bit-for-bit reproduce
  what we shipped.
- **Releases are signed by humans, attesting under their own legal
  identity.** Every release ships with a signed
  [attestation](releases.md) recording that a named engineer reproduced
  the manifest from source, reviewed the diff against the prior
  release, and is not under compulsion to weaken the published
  guarantees. Their fingerprint is pinned in the prior client.
- **Servers run in confidential compute** (currently AMD SEV-SNP, Intel
  TDX, and NVIDIA confidential compute), and the client verifies the
  enclave's hardware attestation on every TLS handshake. The
  measurement it checks against is the one compiled into that client
  build. See [server.md](server.md) and [upstream.md](upstream.md).

This makes guarantees auditable. You don't have to trust that we *say*
your chat history isn't logged — you can verify that the running code
has no path to log it.

## 3. Maximum transparency, including what we don't yet defend against

Trust is built on what is *not* claimed as much as on what is. Each
component of Eidola's trust chain has at least one residual assumption
— the hardware vendor isn't issuing fake attestations, the WebPKI CAs
aren't issuing wildcard certs to attackers, the prior client's pinned
trust root hasn't been quietly subverted before you installed it.
These are real, they are named explicitly in the
[threat model](threat-model.md), and they have known mitigations
where we have them and known [gaps](gaps.md) where we don't.

The same principle applies to capabilities we *intend* to ship but
haven't yet. The verifier already does serious work, but several
defenses are deferred — first-install downgrade protection, Rekor
checkpoint verification, artifact-hash check at install time. They are
written down in one place, with what they would catch and why we
believe the rest of the chain still holds without them.

## Why the audience matters

Eidola has two audiences who need to read this differently.

For the **technically curious user** — someone who has used a few AI
products, is uneasy about where their data goes, and wants enough
mental scaffolding to evaluate Eidola against the alternatives — these
docs offer the design without requiring you to follow every link.
[privacy-guarantees.md](privacy-guarantees.md) is the contract. The
component pages explain how the design upholds it.

For the **technical reader doing due diligence** — security
engineers, privacy researchers, the natural tech-expert in any social
group whose opinion the first audience will rely on — every claim
links to source. Where we cite an enclave measurement, you can read
the code that computes it. Where we describe an attestation flow, you
can read the verifier that walks it. Where we acknowledge a gap, you
can read the issue and the workaround.

This is intentional. We can only be trusted to the extent that we are
checkable, and the deepest layer of that check is the source.

## Where to read next

- [Privacy guarantees](privacy-guarantees.md) — the contract.
- [Threat model](threat-model.md) — who you're trusting and who
  you're not.
- [The client](client.md) — how local sovereignty is implemented.
- [The server](server.md) — what runs in confidential compute, and
  what is deliberately kept apart from it.
- [Inference upstream](upstream.md) — where models actually run.
- [Releases](releases.md) — how a new binary becomes trustable.
- [Known gaps](gaps.md) — what we don't yet defend against.
- [Trust root](trust-root.md) — the technical specification, for
  spot-checking rigor.
