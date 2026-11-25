# Documentation

These docs describe how Eidola works, what it commits to, and what it does not yet defend against. They are written for a technical reader who wants to check the design.

## Start here

1. **[Paradigm](paradigm.md)** — The principles the design follows.
2. **[Privacy guarantees](privacy-guarantees.md)** — The enumerated commitments a release attestant signs against for every release.
3. **[Threat model](threat-model.md)** — Who Eidola defends against, who it does not, and what is left as residual trust.

## Design pieces

4. **[Architecture](architecture.md)** — Every part and every connection between them, with what each connection reveals and what protects it.
5. **[The client](client.md)** — Embedded trust root, per-handshake attestation, and refusal when verification fails.
6. **[The mobile app](mobile.md)** — The bundle and the native shell: what each platform lets you verify, and what the shell is trusted with.
7. **[The gateway](gateway.md)** — Linked and unlinked surfaces, anonymous credentials, telemetry, and what runs in confidential compute.
8. **[The engine](engine.md)** — Where models run, and how that layer is pinned and verified.
9. **[Model identity](model-identity.md)** — The weights hash that names a model, and how to recompute it.
10. **[Anonymous credits](anonymous-credits.md)** — How payment stays unlinkable from use, one decision at a time.

## Release flow

11. **[Releases](releases.md)** — How a release becomes trustable: a CI signature plus human attestation, both on the same transparency log.
12. **[Verification](verification.md)** — Payload, archive, envelope, installable: the hashes a user checks, and what the CI-signed manifest records.
13. **[Trust root](trust-root.md)** — What is pinned at compile time, how schema versions work, and how the verifier walks the chain.
14. **[Apple distribution](apple-distribution.md)** — The macOS envelope: Developer ID, notarization, and why those bytes never enter the manifest.

## What's missing

15. **[Known gaps](gaps.md)** — Every piece of the trust chain that is deferred, with what it would catch and what constrains it today.

## Operations

16. **[Infrastructure and vendors](vendors.md)** — The third parties that process data for Eidola, and what each can see. Referenced by the [privacy policy](https://www.eidola.ai/privacy/).

## For contributors

Contributor-facing READMEs live beside the code they describe. Start with the top-level [`README.md`](../README.md) for dev setup, and [`releases/README.md`](../releases/README.md) for release-pipeline operations. [`AGENTS.md`](../AGENTS.md) is written for and mostly maintained by coding agents, but contains a useful architecture overview. Task-oriented runbooks (such as provisioning the release signing YubiKey) live in [`contributing/`](contributing/README.md).
