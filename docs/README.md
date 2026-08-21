# Documentation

These docs describe how Eidola works, what it commits to, and what it doesn't yet defend against. They are written for a technical reader who wants to understand the design — and who, in turn, can vouch for it to less technical friends and family.

## Start here

1. **[Paradigm](paradigm.md)** — How to think about Eidola. The user-sovereignty lens that everything else assumes.
2. **[Privacy guarantees](privacy-guarantees.md)** — Enumerated, durable commitments that a release officer signs against every release.
3. **[Threat model](threat-model.md)** — Who Eidola defends against, who it doesn't, and what is left as residual trust.

## Design pieces

4. **[The client](client.md)** — Fail-safe by design, embedded trust root, per-handshake attestation.
5. **[The server](server.md)** — Linked vs. unlinked surfaces, anonymous credentials, what runs in confidential compute.
6. **[Inference upstream](upstream.md)** — Where models actually run, and how that layer is verified.

## Release flow

7. **[Releases](releases.md)** — How a new release becomes trustable. CI signature plus human attestation, both on the same transparency log.
8. **[Verification](verification.md)** — Payload, archive, envelope, installable: the hashes a user checks, and what the CI-signed manifest records.
9. **[Trust root](trust-root.md)** — The technical specification: what's pinned at compile time, how schema versions work, how the verifier walks the chain.
10. **[Apple distribution](apple-distribution.md)** — The macOS envelope: Developer ID, notarization, and why those bytes never enter the manifest.

## What's missing

11. **[Known gaps](gaps.md)** — Every piece of the trust chain that is intentionally deferred, with what it would catch and what constrains it today.

## Operations

12. **[Infrastructure and vendors](vendors.md)** — The complete list of third parties that process data for Eidola, and exactly what each can see. Referenced by the [privacy policy](https://www.eidola.ai/privacy/).

## For contributors

Contributor-facing READMEs live alongside the code they describe. Start with the top-level [`README.md`](../README.md) for the project landing page and dev setup, and [`releases/README.md`](../releases/README.md) for release-pipeline operations. [`AGENTS.md`](../AGENTS.md) is intended for and almost entirely maintained by coding agents; while it is not written for a human audience *per se*, it contains a useful architecture overview. Task-oriented runbooks (e.g. provisioning the release signing YubiKey) live in [`contributing/`](contributing/README.md).
