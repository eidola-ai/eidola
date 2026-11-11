# How to think about Eidola

Privacy claims about an AI product are easy to make and hard to check. Eidola is built so that its claims can be checked, and so that they do not depend on the operator's intentions.

The operating paradigm is **user sovereignty**: the user runs software they control, on hardware they choose, talking to systems whose properties they can verify. Where the user relies on infrastructure we provide, they can verify what is running on it and hold Eidola to the [privacy guarantees](privacy-guarantees.md) it publishes.

Three principles follow, and the rest of these docs assume them.

## 1. The client is sovereign

Eidola is an installed application: software on a device you control, running locally, with its own data. Some of what it does needs remote compute that no consumer device can supply. The trust architecture extends the installed-app model to those remote interactions without giving up the integrity and confidentiality a user has on their own machine.

Every decision about what to run, what to trust, and whether to talk to a given server is made locally, against values compiled into the client before it shipped.

A client binary trusts **exactly one gateway build**, and that build fixes the inference engines it accepts. The trust root (measurements, identity patterns, fingerprints, and policy constants) is embedded at compile time. Every release rebuilds the clients, the gateway and its engine pins together so that their values correspond, and nothing hands trust over at runtime. See [client.md](client.md).

The client **fails safe**: if anything in the verification chain cannot be confirmed, the connection is refused. There is no fallback to an unverified path.

Your data (chat history, drafts, accounts) lives on your device. A request that needs remote computation is sent only to confidential-compute enclaves, with TLS terminated inside the enclave, so that no host, operator, or network observer, us included, can read it in transit, and nothing writes it to storage or logs.

## 2. Code is the trust boundary, not policy

Privacy guarantees in Eidola are properties of the *running code*, not of an operator's stated policy. The user can verify which code is running because:

- **The source is public and the builds are reproducible.** Anyone can rebuild the released binaries from the committed source and get the same bytes. The expected hashes are committed beside the source and re-derived in CI.
- **Servers run in confidential compute.** The gateway runs in an AMD SEV-SNP enclave, and the inference engines in Intel TDX confidential VMs with NVIDIA confidential-computing GPUs. The client verifies the gateway's hardware attestation on every TLS handshake, and the gateway verifies each engine's the same way. Each checks against measurements compiled into its own build. See [gateway.md](gateway.md) and [engine.md](engine.md).
- **Releases are attested by named people.** Every release [carries a signed attestation](releases.md), in wording the previous version of the app fixes and verifies, that the release meets the versioned properties in [Privacy guarantees](privacy-guarantees.md). Each attestant's key fingerprint is pinned in the prior client, so the chain of authorized attestants is unbroken across releases.

Together these make the guarantees auditable: whether chat history is logged is a question about the running code, which can be read, and a named person has signed that the code matches the claims.

## 3. Limits are stated

Eidola's residual trust assumptions and deferred defenses are catalogued in the [threat model](threat-model.md) and [known gaps](gaps.md). If a threat scenario is missing, open an issue or a pull request.

## Who these documents are for

A **technically curious user** who wants to evaluate Eidola against the alternatives can read [Privacy guarantees](privacy-guarantees.md) for the enumerated properties, and the component pages for how the design upholds them.

A **reader doing due diligence** (a security engineer, a privacy researcher) can check every claim against the source in this repository.

These docs may be incomplete, and may drift from the code. The code is the source of truth. If you find a divergence, please open an issue or a pull request.

## Where to read next

- [Privacy guarantees](privacy-guarantees.md) — the specific commitments in each release.
- [Threat model](threat-model.md) — who you're trusting and who you're not.
- [The client](client.md) — the trust root and what the client refuses.
- [The gateway](gateway.md) — what runs in confidential compute, and what is deliberately kept apart from it.
- [Inference engine](engine.md) — where models run, and how that layer is pinned.
- [Releases](releases.md) — how a new binary becomes trustable.
- [Verification](verification.md) — payload, archive, envelope, installable, and the hashes a user checks.
- [Known gaps](gaps.md) — what we don't yet defend against.
- [Trust root](trust-root.md) — the technical specification.
