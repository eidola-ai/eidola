# The Client

The client is the user's entry point and the **arbiter of trust** for
every interaction with Eidola. Where most products treat the client as
a thin shell around server-side decisions, Eidola treats it as the
locus of authority: the client decides what servers to talk to, what
code to run, and what to refuse.

This page explains how that authority is structured. For the
mechanical detail of what is pinned and how, see
[trust-root.md](trust-root.md).

## Fail-safe by design

The client refuses to proceed when verification cannot complete.
There is no degraded mode, no "trust on first use" fallback, no
prompt asking the user whether to ignore a failed attestation.
Inability to verify ⇒ the connection does not happen.

This shows up in several places:

- **Hardware attestation failure** (the server's confidential-compute
  measurement does not match the client's pinned value, or the
  attestation chain fails to verify) ⇒ the TLS handshake completes
  but the request is rejected before any application data is sent.
- **Trust-root continuity failure** (the next release's claimed prior
  commit does not match the installed commit) ⇒ self-update refuses.
- **Insufficient attestations** (a release ships with fewer
  independent human attestations than `MIN_HUMAN_ATTESTATIONS`
  pinned in the current client) ⇒ self-update refuses.
- **Schema-version mismatch** (a release document is shaped in a
  version the client does not understand) ⇒ self-update refuses.
  Schema versions are integer-versioned with no semver "compatible
  minor" — every change is fully breaking by contract, so the client
  never silently accepts a release with new fields it does not know
  to enforce.

## Trust root, embedded at compile time

Every client binary carries a fixed **trust root** that determines:

- Which server build it is willing to talk to (by enclave
  measurement, not by hostname or operator promise).
- Which release attestants it considers valid (by hardware-bound
  signing-key fingerprint).
- What CI workflow identity is permitted to sign release manifests
  (by Fulcio OIDC identity pattern).
- What schema versions of release and attestation documents it can
  parse.
- What templates the verifier re-renders to check attestation prose.

These values are generated into the binary by `build.rs` at compile
time from files committed in the source tree. There is no runtime API
that can update them, and no network call that can change them. A
running client trusts what it was built with.

The relevant files live in `releases/trust/`. Their roles are
documented in [`releases/README.md`](../releases/README.md), and the
exact constant-by-constant breakdown is in
[trust-root.md](trust-root.md#whats-pinned).

## One release pairs exactly one client with one server

A given client binary trusts **exactly one** server enclave
measurement: the one for the server build in the same release. There
is no "minimum version" or "any of N" floor; it is one specific
measurement.

This is intentional. A floor would mean that a vulnerability in an
older server version could be exploited against a newer client that
still accepts it. By pinning to a single measurement, the only way
the client talks to a different server build is by the user
installing a new client release — which itself goes through the
release attestation process.

For the operator, this means client and server upgrades ship
together as a single coordinated release. For the user, this means
the question "is my client talking to a server I trust?" reduces to
"does my installed client version match the release whose attestation
I verified?" — and the answer is built into the binary.

## Per-handshake attestation, no caching

Every new TCP+TLS handshake to the server triggers a fresh attestation
verification. There is no "verified once, trusted for N minutes" cache.

The mechanics: the client's reqwest connector wraps the inner TLS
connector. After the TLS handshake completes, the connector issues an
inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?v=3` over the
**same TCP+TLS connection**. The response is parsed inline; the
attestation chain (AMD VCEK → ASK → ARK for SEV-SNP, or the TDX
equivalent) is verified; the measurement is checked against
`ALLOWED_MEASUREMENTS`; the TCB policy floor is enforced; the
`report_data` field is checked against `sha256(SPKI(peer_cert))` to
bind the attestation to *this specific TLS key*; and only then is
the connection yielded to the application layer.

Subsequent HTTP requests on a pooled keepalive connection inherit the
binding to the TLS key that was attested when the connection was
first established. A new connection ⇒ a new attestation.

The same-connection guarantee makes this safe behind load balancers:
whatever backend the LB routes you to is the backend whose
attestation you verify on that connection.

## What the client does locally, by design

Some operations stay local even when they could be moved to the
server, because moving them would create privacy or autonomy
costs:

- **Conversation history** is stored in a local Turso (libSQL)
  database in the user's application support directory. It is never
  uploaded. (Sync is a future feature that, if it ships, will be
  end-to-end encrypted with keys under the user's control.)
- **Anonymous credential issuance** is split across the client and
  server: the client generates the blinding factor, holds the
  unblinded token, and submits it on each inference request. The
  server never sees the unblinded token paired with the account
  that requested it.
- **Trust-chain verification** for self-update runs entirely in the
  client. The verifier fetches release artifacts, hashes them
  locally, and checks every signature locally against the embedded
  trust root. No server-side "is this release safe?" call exists.

## Two surfaces: GUI and CLI, one core

The Eidola client ships as a native macOS GUI app (gpui) and a
cross-platform CLI binary, both built on the same shared core crate
(`crates/eidola-app-core/`). The split exists for UX reasons, not
trust reasons:

- The GUI is the friendly surface — chat window, account, balance.
- The CLI is the scriptable, headless surface for power users and
  CI integration.

Both surfaces run the same verifier code, against the same embedded
trust root, with the same fail-safe behavior. The decision a user
makes about which to install is purely about ergonomics.

## Configuration overrides

A user can override the embedded `base_url` and trusted measurements
via `~/Library/Application Support/eidola/config.toml`. This exists
for development and for advanced users running their own server. The
overrides do *not* lower verification rigor — the client still
verifies the attestation against whatever measurement is configured.

In production use against Eidola's deployment, no overrides are
needed; the embedded values are what's used.
