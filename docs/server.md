# The Server

The Eidola server is an OpenAI-compatible proxy that sits between the
client and an upstream inference provider. It also runs the
account and billing surface. Its design is dominated by one structural
decision: the **linked** account surface and the **unlinked**
inference surface are kept disjoint, in code and at runtime.

## Linked vs. unlinked

The server's HTTP surface splits into two layers with separate
authentication mechanisms:

| Surface | Auth | Sees account identity? | Sees inference content? |
|---|---|---|---|
| **Public** | none | n/a | no |
| **Linked** (account, billing) | HTTP Basic — account UUID + secret | yes | no |
| **Unlinked** (inference) | Anonymous credentials (Privacy Pass ACT) | **no** | only as opaque bytes in transit |

The split is enforced at the type level. The `BasicAuth` extractor
and the `TokenAuth` extractor are different types; an inference
endpoint takes only `TokenAuth`, and an account endpoint takes only
`BasicAuth`. There is no path through which Basic auth could be
accepted on an inference endpoint without a code change visible in a
diff.

This means a single Eidola server process, observing only its own
inputs, cannot connect a particular inference request to the account
that funded it. The most it can see on the inference path is *that*
an anonymous credential was redeemed against a particular issuer
key; it cannot map back to *which* account requested that credential
from the issuance flow.

## Unlinkability

The unlinkability property is achieved with **Privacy Pass
Anonymous Credentials Tokens (ACT)**, per
`draft-schlesinger-privacypass-act-01`.

The flow:

1. **Issuance.** The authenticated client (Basic auth, linked
   surface) requests credentials. It generates blinding factors
   locally and sends blinded token requests. The server signs the
   blinded requests with its issuer private key and returns blinded
   signatures. The server learns *that* this account requested
   N credentials, never *which specific tokens* it received.
2. **Redemption.** When the client makes an inference request, it
   unblinds a token and presents it on the unlinked surface. The
   server verifies the token signature against the issuer public
   key. The token contains no account identifier, and the
   blind-signature construction guarantees the server cannot
   correlate this token with any individual issuance request from
   step 1.

The issuer key is stored encrypted at rest in Postgres using a
`CREDENTIAL_MASTER_KEY` that is injected into the server enclave as
a Tinfoil secret. If this key were compromised, new ACTs could be
forged, but the unlinkability property remains.

Domain separation is baked into the credential construction
(`ACT-v1:eidola:inference:production:<date>`) to prevent
cross-deployment correlation if the issuer key were ever reused.

## Anonymity set

The unlinkability invariants in
[privacy-guarantees.md §2](privacy-guarantees.md#2-unlinkability)
are meaningful only to the extent that each token's anonymity set is
large and the issuance/redemption policy doesn't accidentally
re-introduce a linkable identifier. The
server's issuance and key-rotation policies are tuned specifically
for these properties.

**Anonymity set = users sharing the same issuer key + domain
separator.** Every ACT token redeemed against a given issuer key
is, by the math of the blind-signature scheme, indistinguishable
from every other token issued under that key. The size of the set
is the number of distinct accounts that received at least one
token from that key during its issuance window.

**Issuer keys rotate on a ~7-day epoch.** Each key has an
`issue_from` timestamp and an `issue_until` timestamp; while
active, that key signs new credentials. After `issue_until` the
key stops issuing, but tokens already issued under it remain
redeemable until `accept_until` (a grace period beyond the
issuance window). The dual-window design means at any moment
multiple keys are concurrently spendable, but only one key is
actively *issuing*. This is intentional:

- It bounds the lifetime of any single issued token, so revocation
  by retirement (rather than per-token blacklisting) is the
  primary mechanism.
- It gives users a meaningful redemption window that doesn't
  require them to redeem the moment a token is issued. Tokens can
  be requested in advance, held client-side, and spent later
  without timing-correlating issuance to redemption.

**The domain separator does *not* rotate on the same schedule —
deliberately.** The anonymity set is the intersection of "users
sharing the same key" *and* "users sharing the same domain
separator." Rotating the domain separator would shrink the
anonymity set without any compensating gain (nullifiers, which
prevent double-spending, are partitioned by issuer key, not by
domain separator). The domain separator only changes on protocol
upgrades or deployment-identity changes.

**Cross-device and batched issuance.** Because tokens are spendable
across the full `accept_until` window and don't carry per-device
binding, a user with multiple devices on a single account can have
each device hold its own tokens issued under the same key — all
contributing to the same anonymity set. The same flexibility
covers JIT issuance scenarios where a device requests tokens on
demand.

**Why this prevents timing correlation between linked and
unlinked requests.** Issuance happens on the authenticated
(linked) account surface. Redemption happens on the anonymous
(unlinked) inference surface. If issuance and redemption were
forced to be near-simultaneous, an observer of both surfaces
could correlate "account X issued at time T, anonymous token
redeemed at time T+ε" with high confidence. The
batched-and-deferred-redemption policy decouples those timestamps
by design: the issuance request is a separate, asynchronous
event from any individual redemption.

What this policy does *not* defend against is a small total
anonymity set during the early life of the deployment (when only
a few users are issuing under a given key). That is named in
[gaps.md#anonymity-set-size](gaps.md) as an early-stage residual.

## What runs in confidential compute

The server runs inside a **Tinfoil Container** on confidential-compute
hardware (AMD SEV-SNP, with TDX support tracked in measurements).
The relevant properties:

- **TLS termination is inside the enclave.** The Tinfoil shim
  generates TLS certificates whose Subject Alternative Names encode
  the attestation hash and an HPKE public key. The certificate is
  issued by a public CA via ACME, so any client can validate the
  chain; the *binding to the enclave* is what the Eidola verifier
  checks beyond the basic WebPKI chain.
- **Secrets are sealed into the enclave.** Both
  `CREDENTIAL_MASTER_KEY` and `DATABASE_PASSWORD` are Tinfoil
  secrets, decrypted only inside the verified enclave. They are not
  visible to the host, the orchestrator, or any operator.
- **The enclave measurement is deterministic from source.** The
  client's pinned measurement is computed from the same OVMF, kernel,
  initrd, and `tinfoil-config.yml` that the production enclave is
  built from. See [trust-root.md](trust-root.md#whats-pinned).

The server is `FROM scratch`, statically linked musl, runs as
non-root, and ships no shell or package manager. The attack surface
inside the enclave is limited to the server binary itself.

## What the server is *not* doing

Several things that a typical AI proxy might do are deliberately
absent:

- **No session caching, no request memory, no learned state per
  account.** The server is request-based. Two inference requests
  with the same content produce two independent upstream calls.
  Caching across requests would create a correlation surface; it is
  not implemented.
- **No content-based logging.** Inference request bodies and response
  bodies are not logged, persisted, or tee'd into observability
  systems. Telemetry on the inference path is limited to model name,
  token counts, status code, and latency.
- **No "ask the operator to approve this request" path.** There is no
  human-review queue, no flagging system, no operator interface for
  inspecting inference traffic. The server's job on the inference
  path is to route and account for usage; no operator-visible
  branches exist.

## Telemetry: scope and boundary

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, the server exports OpenTelemetry
traces, metrics, and logs. The privacy boundary is enforced in the
telemetry layer:

- **Inference (unlinked) spans** contain only model name, token
  counts, status, and latency. Never account identifiers, credential
  data, or message content.
- **Account (linked) spans** may include `account_id` when relevant
  to the operation (creating an account, allocating credentials).
  They do not include inference content (none flows through them).
- Routing between the two regimes is done in middleware
  (`crates/eidola-server/src/middleware.rs`), which classifies the
  route before creating the span.

The same boundary applies to stdout logging.

## Inference is proxied, not performed

The Eidola server is **not** the inference engine. Models run in a
separate confidential-compute deployment operated by the upstream
inference provider (currently Tinfoil), with its own attestation.
The Eidola server's role on the inference path is:

1. Verify the anonymous credential.
2. Open an attested HTTPS connection to the inference upstream.
3. Stream the request through, stream the response back.
4. Record the per-request token counts for accounting.

This means **two layers of confidential compute** protect the
inference content: the Eidola server enclave (which sees the content
only in transit, never logged) and the inference upstream enclave
(which actually performs the inference). The client verifies the
attestation of the Eidola server directly on every handshake. For
the upstream layer, see [upstream.md](upstream.md).

## Where to read the code

| Subsystem | File |
|---|---|
| Anonymous credentials (Privacy Pass ACT) | `crates/eidola-server/src/credentials.rs` |
| Inference proxying | `crates/eidola-server/src/chat.rs` |
| Auth extractors | `crates/eidola-server/src/auth.rs` |
| Linked/unlinked routing | `crates/eidola-server/src/middleware.rs` |
| Telemetry boundary | `crates/eidola-server/src/telemetry.rs` |
| OpenAPI surface (tags = linked / unlinked / public) | `crates/eidola-server/src/api_doc.rs` |
