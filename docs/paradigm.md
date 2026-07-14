# How to think about Eidola

Eidola exists because privacy and autonomy in AI are easy to *claim* and hard to *prove*. Most "private AI" products ask you to trust a vendor's intentions. Eidola is structured so you don't have to.

The operating paradigm is **user sovereignty**: the end user runs software they control, on hardware they choose, talking to systems whose properties they can independently verify. Where the user relies on infrastructure provided by us, multiple layers ensure that they keep the ability to verify exactly what is running, and to hold Eidola accountable to the [privacy guarantees](privacy-guarantees.md) it publishes.

Three principles flow from that paradigm. They show up everywhere in the design, and they are the lens through which the rest of these docs are written.

## 1. The client is sovereign

Eidola is structured like a traditional installed application: a self-contained piece of software you put on a device you control, running locally, with its own data. Where it differs from a classic app is that *some* of its capabilities require remote compute that no consumer device can perform alone. The trust architecture in this document is what lets us extend the self-contained-app model to cover those remote interactions without giving up the integrity and confidentiality properties people used to take for granted on their own machines.

The client is the user's entry point and the arbiter of trust. Every decision about *what to run*, *what to trust*, and *whether to talk to a given server* is made locally, against values that were compiled into the binary before it shipped.

A given client binary trusts **exactly one server build**. The trust root — the measurements, identity patterns, fingerprints, and policy constants — is embedded at compile time. Every Eidola release is a coordinated rebuild of clients *and* server so that their values correspond. There is no runtime trust handoff. See [client.md](client.md).

The client is also designed to **fail safe**: if anything in the verification chain cannot be confirmed, the connection is refused rather than downgraded. There is no quiet fallback to an unverified path.

Your app's data — chat history, drafts, accounts — lives on your device. When a request does need remote computation, it is sent only to confidential-compute enclaves, with TLS terminated inside the enclave so that no host, operator, or network observer — us included — can read it in transit, and no system writes it to storage or logs.

The most meaningful limitation today is provenance. The models themselves run in enclaves fully managed by our upstream provider (Tinfoil). These make the same confidentiality promises as the rest of Eidola, but are held to a category-weaker discipline than our own: they are not yet reproducible and human-attested to the same standard, and their exact version is not pinned transitively from your client the way Eidola's server is. Eidola verifies the upstream's hardware attestation on every connection, but does not yet re-derive or pin the code running inside it. Hosting inference ourselves — on the same reproducible, human-attested, client-pinned footing as the rest of Eidola — is the intended resolution; it waits on enough real usage to justify the GPU cost. The full picture is in the [threat model](threat-model.md), [inference upstream](upstream.md), and [known gaps](gaps.md).

## 2. Code is the trust boundary, not policy

Privacy guarantees in Eidola are properties of the *running code*, not of an operator's stated policy. The user can verify which code is running because:

- **The source is public and reproducible.** Anyone can rebuild the released binaries from the committed source and bit-for-bit reproduce what we shipped. The expected values are stored under version control alongside the source and tested in CI, making this property structurally enforced and easily auditable.
- **Releases are signed by humans, attesting under their own legal identity.** Every release ships with a signed [attestation](releases.md) recording that a named engineer reproduced the manifest from source, reviewed the diff against the prior release, and is not under compulsion to weaken the published guarantees. Their fingerprint is pinned in the prior client.
- **Servers run in confidential compute** (currently AMD SEV-SNP, Intel TDX, and NVIDIA confidential compute), and the client verifies the enclave's hardware attestation on every TLS handshake. The measurement it checks against is the one compiled into that client build. See [server.md](server.md) and [upstream.md](upstream.md).

This makes guarantees auditable. You don't have to trust that we *say* your chat history isn't logged — you can verify that the running code has no path to log it.

## 3. Maximum transparency, including what we don't yet defend against

Eidola's residual trust assumptions and deferred defenses are catalogued in the [threat model](threat-model.md) and [known gaps](gaps.md). If you find a threat scenario that isn't addressed and you think should be, open an issue or PR — that's how this list gets better.

## Who are these documents for?

Eidola has two audiences who need to read this differently.

For the **technically curious user** — someone who has used a few AI products, is uneasy about where their data goes, and wants enough mental scaffolding to evaluate Eidola against the alternatives — these docs offer the design without requiring you to follow every link. Read [Privacy guarantees](privacy-guarantees.md) for an enumerated list of Eidola's privacy properties. The component pages explain how the design upholds it.

For the **technical reader doing due diligence** — security engineers, privacy researchers, etc — every claim is realized in this same repo. We can only be trusted to the extent that we are accountable, and the deepest layer of that check is the source.

A note on these docs themselves: they are a map, not the territory. They will be incomplete, and they may drift from the code over time. The source is the source of truth. If you catch a divergence, please open an issue or PR.

## Where to read next

- [Privacy guarantees](privacy-guarantees.md) — the contract.
- [Threat model](threat-model.md) — who you're trusting and who you're not.
- [The client](client.md) — how local sovereignty is implemented.
- [The server](server.md) — what runs in confidential compute, and what is deliberately kept apart from it.
- [Inference upstream](upstream.md) — where models actually run.
- [Releases](releases.md) — how a new binary becomes trustable.
- [Known gaps](gaps.md) — what we don't yet defend against.
- [Trust root](trust-root.md) — the technical specification, for spot-checking rigor.
