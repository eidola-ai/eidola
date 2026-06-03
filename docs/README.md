# Eidola Documentation

These docs describe how Eidola works, what it commits to, and what it doesn't yet defend against. They are written for a technical reader who wants to understand the design without necessarily reading every line of source — and who, in turn, can vouch for it to friends and family who aren't going to read it themselves.

## Start here

1. **[Paradigm](paradigm.md)** — How to think about Eidola. The user-sovereignty lens that everything else assumes.
2. **[Privacy guarantees](privacy-guarantees.md)** — The contract. Enumerated, durable commitments that a release engineer signs against every release.
3. **[Threat model](threat-model.md)** — Who Eidola defends against, who it doesn't, and what is left as residual trust.

## Design pieces

4. **[The client](client.md)** — Fail-safe by design, embedded trust root, per-handshake attestation.
5. **[The server](server.md)** — Linked vs. unlinked surfaces, anonymous credentials, what runs in confidential compute.
6. **[Inference upstream](upstream.md)** — Where models actually run, and how that layer is verified.

## Release flow

7. **[Releases](releases.md)** — How a new client+server bundle becomes trustable. CI signature plus human attestation, both on the same transparency log.
8. **[Trust root: technical specification](trust-root.md)** — What's pinned at compile time, how schema versions work, how the verifier walks the chain.

## What's missing

9. **[Known gaps](gaps.md)** — Every piece of the trust chain that is intentionally deferred, with what it would catch and what constrains it today.

## For contributors

Contributor-facing READMEs live alongside the code they describe. Start with the top-level [`README.md`](../README.md) for the project landing page and dev setup, [`AGENTS.md`](../AGENTS.md) for the architecture overview, and [`releases/README.md`](../releases/README.md) for release-pipeline operations.
