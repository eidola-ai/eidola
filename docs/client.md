# The client

The client decides which server to talk to, which release to install, and what to refuse. Those decisions are made locally, against values compiled into it.

This page explains how. For the mechanical detail of what is pinned and how, see [trust-root.md](trust-root.md). For how a download is hashed, see [verification.md](verification.md).

## Fail-safe by design

The client refuses to proceed when verification cannot complete. There is no degraded mode, no trust-on-first-use fallback, and no prompt asking the user whether to ignore a failed attestation. If the client cannot verify, the connection does not happen.

The cases:

- **Hardware attestation failure** (the gateway's confidential-compute measurement does not match the client's pinned value, or the attestation chain fails to verify) ⇒ the TLS handshake completes but the request is rejected before any application data is sent.
- **Trust-root continuity failure** (the next release's claimed prior commit does not match the installed commit) ⇒ self-update refuses.
- **Insufficient attestations** (a release ships with fewer independent human attestations than `MIN_HUMAN_ATTESTATIONS` pinned in the current client) ⇒ self-update refuses.
- **Schema-version mismatch** (a release document is shaped in a version the client does not understand) ⇒ self-update refuses. Schema versions are integers with no compatible minor versions, so the client never accepts a release with fields it does not know to enforce.

## Trust root, embedded at compile time

Every client binary carries a fixed **trust root** that determines:

- Which gateway build it is willing to talk to (by enclave measurement, not by hostname or operator promise).
- Which release attestants it considers valid (by hardware-bound signing-key fingerprint).
- What CI workflow identity is permitted to sign release manifests (by Fulcio OIDC identity pattern).
- What schema versions of release and attestation documents it can parse.
- What templates the verifier re-renders to check attestation prose.

These values are generated into the binary by `build.rs` at compile time from files committed in the source tree. There is no runtime API that can update them, and no network call that can change them. A running client trusts what it was built with.

The relevant files live in `releases/trust/`. Their roles are documented in [`releases/README.md`](../releases/README.md), and the exact constant-by-constant breakdown is in [trust-root.md](trust-root.md#whats-pinned).

## One release pairs exactly one client with one server

A given client binary trusts **exactly one** gateway launch identity: that of the gateway build in the same release. There is no "minimum version" or "any of N" floor; it is one specific identity.

A floor would let a vulnerability in an older gateway build be exploited against a newer client that still accepts it. With a single pinned identity, the client talks to a different gateway build only when the user installs a new client release, which itself goes through release attestation.

Client and gateway upgrades therefore ship together as one release. The gateway pins the inference engines it accepts in the same way ([engine.md](engine.md#what-pins-the-engine)), so one client release fixes the whole serving path.

## Per-handshake attestation, no caching

Every new TCP+TLS handshake to the gateway triggers a fresh attestation verification. There is no "verified once, trusted for N minutes" cache.

The mechanics: the client's reqwest connector wraps the inner TLS connector. After the TLS handshake completes, the connector generates a fresh random nonce and issues an inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?nonce=<hex>` over the **same TCP+TLS connection**. The response is parsed strictly; the verifier hashes the exact decoded crypto-material and device-evidence section bytes, recomputes the domain-separated `REPORT_DATA` from those hashes and the nonce, and compares the endorsed TLS SPKI fingerprint with the peer certificate. It then enforces report field hygiene — most importantly that the guest policy forbids DEBUG, which the launch measurement does not cover — and verifies the self-contained AMD VCEK chain and report signature. The carried CRL is accepted only as a complete direct AMD CRL with a matching ARK identity and signature, supported extensions and signature parameters, a signed `thisUpdate <= now < nextUpdate` interval, and neither the ASK nor VCEK serial listed. All of this runs offline. SEV-SNP is the only platform the client accepts for the gateway; a TDX presentation is refused ([gaps.md](gaps.md#tdx-acceptance)). The verifier then checks the launch measurement, configuration hash and guest policy against the pinned launch identity, enforces the TCB floor, requires the hardware report's signed `REPORT_DATA` to equal the recomputed value, and only then yields the connection to the application layer.

Subsequent HTTP requests on a pooled keepalive connection inherit the binding to the TLS key that was attested when the connection was first established. A new connection ⇒ a new attestation.

Using the same connection preserves ordinary load-balancer affinity: the backend selected for that connection serves both the attestation and application request. This is an operational property, not TLS-session channel binding; the residual for an exfiltrated TLS key is documented in [Known gaps](gaps.md#tls-key-exfiltration-and-channel-binding).

## What the client does locally, by design

Some operations stay local even though a server could perform them:

- **Conversation history** is stored in a local Turso database in the user's application support directory. It is never uploaded.
- **Anonymous credentials** are held by the client. It generates each token's secrets, has the gateway sign a commitment to them, and spends the token on inference requests. The gateway never sees a token it can pair with the account that requested it ([anonymous-credits.md](anonymous-credits.md)).
- **Trust-chain verification** for self-update runs entirely in the client. The verifier fetches release artifacts, hashes them locally, and checks every signature locally against the embedded trust root. No server-side "is this release safe?" call exists.

## Prefix-cache keys

The inference engine can reuse the computation behind a prompt prefix it has already processed, which makes each turn of a long conversation cheaper and faster. Reuse is scoped by a `cache_key` the client sends: requests carrying the same key may share cached computation, and no other requests do. Because a key links the requests it rides on, the client treats it as the thing to minimize ([Privacy guarantees §2.7](privacy-guarantees.md#2-unlinkability)):

- **When one is sent.** Only for a model whose catalog entry declares an engine prefix cache (`prompt_cache` with `supported: true` and both retention bounds). Any other model gets a request byte-for-byte the one it would get if keys did not exist. Requests made through the client's local inference proxy never carry one.
- **What it is.** 32 bytes from the operating system's random number generator, scoped to one *lineage* — one participant answering in one space, that conversation's branches and regenerations included — and to the server endpoint and trust configuration it was created under. It is never sent for another space, another participant, or to another endpoint or trust configuration.
- **When it rotates.** A fresh key replaces it before any request that would come after the lineage has been idle for the declared idle TTL since its last request that may have been delivered, after the key reaches the declared maximum age, after the participant's model changes or the weights its model name resolves to change, or when the clock reads earlier than the key's recorded times. The key is judged at the moment the request body is written to an attested connection, so a slow connection or a suspended process cannot carry it past those bounds; a request that failed before anything was sent does not count as a use.
- **Where it lives.** Only in the local database. It is overwritten there before being replaced or deleted, and deleted when its conversation is archived, its participant leaves or is retired, or the endpoint or trust configuration changes, including when an update to the client changes them. It is never logged or displayed, and the request record keeps it as withheld.

## Two surfaces: GUI and CLI, one core

The client ships as a native GUI app (macOS and Linux/Wayland) and a CLI, both built on one shared core crate (`crates/eidola-app-core/`). Both run the same verifier code against the same embedded trust root, with the same fail-safe behavior.

## Configuration overrides

A user can override the client's *connection identity* — where it connects and what it accepts there: the `eidola` backend row's `base_url`, trusted measurements, and hardware ARK/ASK CAs in the local database, set via `eidola configure` on the CLI or Settings → Backends → Eidola in the GUI, plus the release-discovery feed (`update_feed_override`), set by editing `config.toml`. This exists for development and for advanced users running their own server. The overrides do *not* lower verification rigor — the client still verifies the attestation against whatever measurement is configured, and release-verification policy (attestant fingerprints, attestation threshold, schema versions, claim templates, Sigstore roots) has no override surface at all.

Against Eidola's own deployment no overrides are needed, and the embedded values are used. The [privacy guarantees](privacy-guarantees.md) do not extend to a client with overrides set.
