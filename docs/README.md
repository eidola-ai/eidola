# Documentation

Eidola is a private client for running open-weight AI models — locally on your own hardware, on servers you bring yourself, or on Eidola's confidential-compute infrastructure.

These docs explain how to use Eidola, how it works under the hood, what it commits to, and where its limits are.

## Start here

- **[How to think about Eidola](paradigm.md)** — The core mental model: user sovereignty, why Eidola provides compute rather than proprietary AI, and why structure matters more than policy.

## User guide

- **[Getting started](guide/getting-started.md)** — Installing Eidola on desktop or mobile, creating an anonymous account (or skipping straight to local models), and sending your first prompt.
- **[Running hosted models](guide/hosted-models.md)** — Buying credits, how the local wallet works, choosing models, and following tags versus pinning weights.
- **[Local & external models](guide/local-models.md)** — Running models on your own device without an account, or connecting your own server.
- **[Command line & other tools](guide/tools-and-cli.md)** — Using the bundled `eidola` CLI and connecting other apps through the local inference proxy.

## How it works

- **[Architecture](architecture.md)** — The system from end to end: your device, the gateway, Eidola's inference engine, and what every connection reveals.
- **[Anonymous credits](anonymous-credits.md)** — How payment stays cryptographically unlinkable from usage.
- **[Model identity](model-identity.md)** — Why model names aren't enough, how weights hashes work, and how to check them yourself.
- **[Verifiable releases](releases.md)** — How a release becomes trustworthy: reproducible builds, CI signatures, named human attestations, and the compiled-in trust root.
- **[Apps & platform verification](apps.md)** — How each platform (Linux, macOS, Android, and iOS) delivers and verifies the app on your device.

## Trust & limits

- **[Privacy guarantees](privacy-guarantees.md)** — The enumerated commitments signed into every release.
- **[Threat model](threat-model.md)** — Who Eidola defends against, who it does not, and what residual trust remains.
- **[Known gaps](gaps.md)** — What isn't closed yet, why it matters, and what constrains it today.
- **[Infrastructure & vendors](vendors.md)** — The third parties involved in operating Eidola, and what each can see.

## For contributors

Contributor setup, architecture notes, and operational runbooks live in the [GitHub repository](https://github.com/eidola-ai/eidola) — start with [`README.md`](../README.md), [`releases/README.md`](../releases/README.md), and [`docs/contributing/`](contributing/README.md).
