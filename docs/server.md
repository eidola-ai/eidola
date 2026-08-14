# The server

The Eidola server is an OpenAI-compatible proxy that sits between the client and an upstream inference provider. It also runs the account and billing surface. Its design is dominated by one structural decision: the **linked** account surface and the **unlinked** inference surface are kept disjoint, in code and at runtime.

## Linked vs. unlinked

The server's HTTP surface splits into two layers with separate authentication mechanisms:

| Surface | Auth | Sees account identity? | Sees inference content? |
|---|---|---|---|
| **Public** | none | n/a | no |
| **Linked** (account, billing) | HTTP Basic — account UUID + secret | yes | no |
| **Unlinked** (inference) | Anonymous credentials (Privacy Pass ACT) | **no** | only as opaque bytes in transit |

The split is enforced at the type level. The `BasicAuth` extractor and the `TokenAuth` extractor are different types; an inference endpoint takes only `TokenAuth`, and an account endpoint takes only `BasicAuth`. There is no path through which Basic auth could be accepted on an inference endpoint without a code change visible in a diff.

This means a single Eidola server process, observing only its own inputs, cannot connect a particular inference request to the account that funded it. The most it can see on the inference path is *that* an anonymous credential was redeemed against a particular issuer key; it cannot map back to *which* account requested that credential from the issuance flow.

## Unlinkability

The unlinkability property is achieved with **Privacy Pass Anonymous Credentials Tokens (ACT)**, per `draft-schlesinger-privacypass-act-01`.

The flow:

1. **Issuance.** The authenticated client (Basic auth, linked surface) requests credentials. It generates blinding factors locally and sends blinded token requests. The server signs the blinded requests with its issuer private key and returns blinded signatures. The server learns *that* this account requested N credentials, never *which specific tokens* it received.
2. **Redemption.** When the client makes an inference request, it unblinds a token and presents it on the unlinked surface. The server verifies the token signature against the issuer public key. The token contains no account identifier, and the blind-signature construction guarantees the server cannot correlate this token with any individual issuance request from step 1.

The issuer key is stored encrypted at rest in Postgres using a `CREDENTIAL_MASTER_KEY` that is injected into the server enclave as a Tinfoil secret. If this key were compromised, new ACTs could be forged, but the unlinkability property remains.

Domain separation is baked into the credential construction (`ACT-v1:eidola:inference:production:<date>`) to prevent cross-deployment correlation if the issuer key were ever reused.

## Anonymity set

The unlinkability invariants in [privacy-guarantees.md §2](privacy-guarantees.md#2-unlinkability) are meaningful only to the extent that each token's anonymity set is large and the issuance/redemption policy doesn't accidentally re-introduce a linkable identifier. The server's issuance and key-rotation policies are tuned specifically for these properties.

**Anonymity set = users sharing the same issuer key + domain separator.** Every ACT token redeemed against a given issuer key is, by the math of the blind-signature scheme, indistinguishable from every other token issued under that key. The size of the set is the number of distinct accounts that received at least one token from that key during its issuance window.

**Issuer keys rotate on a ~7-day epoch.** Each key has an `issue_from` timestamp and an `issue_until` timestamp; while active, that key signs new credentials. After `issue_until` the key stops issuing, but tokens already issued under it remain redeemable until `accept_until` (a grace period beyond the issuance window). The dual-window design means at any moment multiple keys are concurrently spendable, but only one key is actively *issuing*. This is intentional:

- It bounds the lifetime of any single issued token, so revocation by retirement (rather than per-token blacklisting) is the primary mechanism.
- It gives users a meaningful redemption window that doesn't require them to redeem the moment a token is issued. Tokens can be requested in advance, held client-side, and spent later without timing-correlating issuance to redemption.

**The domain separator does *not* rotate on the same schedule — deliberately.** The anonymity set is the intersection of "users sharing the same key" *and* "users sharing the same domain separator." Rotating the domain separator would shrink the anonymity set without any compensating gain (nullifiers, which prevent double-spending, are partitioned by issuer key, not by domain separator). The domain separator only changes on protocol upgrades or deployment-identity changes.

**Cross-device and batched issuance.** Because tokens are spendable across the full `accept_until` window and don't carry per-device binding, a user with multiple devices on a single account can have each device hold its own tokens issued under the same key — all contributing to the same anonymity set. The same flexibility covers JIT issuance scenarios where a device requests tokens on demand.

**Why this prevents timing correlation between linked and unlinked requests.** Issuance happens on the authenticated (linked) account surface. Redemption happens on the anonymous (unlinked) inference surface. If issuance and redemption were forced to be near-simultaneous, an observer of both surfaces could correlate "account X issued at time T, anonymous token redeemed at time T+ε" with high confidence. The batched-and-deferred-redemption policy decouples those timestamps by design: the issuance request is a separate, asynchronous event from any individual redemption.

What this policy does *not* defend against is a small total anonymity set during the early life of the deployment (when only a few users are issuing under a given key). That is named in [gaps.md § Anonymity-set size](gaps.md#anonymity-set-size) as an early-stage residual.

## What runs in confidential compute

The server runs inside a **Tinfoil Container** on confidential-compute hardware (AMD SEV-SNP; TDX values are recorded in the trust root but TDX presentations are refused by the verifier — see [gaps.md](gaps.md#tdx-acceptance)). The relevant properties:

- **TLS termination is inside the enclave.** The Tinfoil shim's self-contained v3 attestation endorses the TLS certificate's SPKI and an HPKE public key; its fresh hardware report binds those exact endorsed bytes and the client's nonce. The certificate is issued by a public CA via ACME, so any client can validate the chain; the *binding to the enclave* is what the Eidola verifier checks beyond the basic WebPKI chain.
- **Secrets are sealed into the enclave.** Both `CREDENTIAL_MASTER_KEY` and `DATABASE_PASSWORD` are Tinfoil secrets, decrypted only inside the verified enclave. They are not visible to the host, the orchestrator, or any operator.
- **The enclave measurement is deterministic from source.** The client's pinned measurement is computed from the same OVMF, kernel, initrd, and `tinfoil-config.yml` that the production enclave is built from. See [trust-root.md](trust-root.md#whats-pinned).

The server is `FROM scratch`, statically linked musl, runs as non-root, and ships no shell or package manager. The attack surface inside the enclave is limited to the server binary itself.

## What the server is *not* doing

Several things that a typical AI proxy might do are deliberately absent:

- **No session caching, no request memory, no learned state per account.** The server is request-based. Two inference requests with the same content produce two independent upstream calls. Caching across requests would create a correlation surface; it is not implemented.
- **No content-based logging.** Inference request bodies and response bodies are not logged, persisted, or tee'd into observability systems. Telemetry is limited to generic, unidentifiable fields and emits no per-request records at all — spans are aggregated into metrics in process and never exported.
- **No "ask the operator to approve this request" path.** There is no human-review queue, no flagging system, no operator interface for inspecting inference traffic. The server's job on the inference path is to route and account for usage; no operator-visible branches exist.

## Telemetry: scope and boundary

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, the server exports OpenTelemetry metrics and logs. It does **not** export traces unless directed by the client.

**Spans are recorded everywhere and exported only on request.** Tracing runs across the whole server: the HTTP request, each database round trip, each upstream call. Every span goes to a processor that converts it into histogram observations and drops it. A second, exporting processor is also registered, but it drops unsampled spans. The sampler returns `RecordOnly` for every trace that was not explicitly authorized, so in ordinary operation the export path carries nothing.

A span is a per-request record by construction — a trace id, a wall-clock start, a nanosecond duration — and a request's duration is not independent of its content, since a response's wall time varies with its length. Aggregating in process answers the operational question ("this route spends 300 ms in the database") without retaining a record that would answer "what did *this* request do."

No account identifier can reach the telemetry vendor, because nothing per-request does.

What is exported:

- **Metrics.** Counters and histograms aggregated over the export interval. Content-derived quantities appear here and only here — token totals keyed by model — never attributed to an individual request. Every label value comes from a fixed set: the requested model id is resolved against the known catalog, span names against a tracked list, Stripe event types against the handled list, and anything unrecognized collapses to a shared `other` bucket. Since traces never leave, these carry the whole observability load, and are instrumented accordingly: per-operation duration derived from spans; time-to-first-token, stream duration, largest inter-chunk gap and sustained output rate for inference; a stream-outcome counter (`done`, `client_disconnect`, `upstream_error`, `channel_closed`); and a webhook-outcome counter, which matters because Stripe originates those requests and we can never opt one into closer inspection. Output *rate* is the signal to alert on — dividing tokens by generation time cancels the response-length term and leaves the throughput term, which both sharpens the signal and carries less about any individual request than the duration it derives from.
- **Logs.** Error and warning events. A `ServerError` has two renderings and they are deliberately different: the client gets the full detail on its own attested connection, and the log gets a redacted one. Fields that could carry request-derived or upstream-authored values, and the upstream's own error string (outside our control, and known to quote token counts and request fragments) are never logged.

Bucket boundaries are set explicitly on every histogram, because the SDK default ladder is scaled for milliseconds while every duration instrument here records seconds — on the defaults, essentially all traffic lands in the first bucket and the resulting percentiles are meaningless.

One rule governs adding a span anywhere: `#[instrument(skip_all)]`. The attribute captures every function argument as a span field by default, which would put nullifiers, spend proofs and account ids into span attributes. Aggregation makes that survivable, not acceptable — and under client-directed tracing, an authorized trace would carry those fields all the way to the vendor.

### Client-directed tracing

A request that carries a W3C `traceparent` with the sampled flag set is traced at full granularity: middleware installs it as a sampled remote parent, which promotes the root span and, through it, every child span in the request. Nothing else marks a request for export.

The sampled flag is honored rather than assumed. Per [W3C Trace Context](https://www.w3.org/TR/trace-context/) it reports whether the caller recorded the request; it is a hint about the caller's own decision, not an instruction, and a receiver is free to sample independently. Honoring it costs nothing and means a client whose tracing is off does not cause an export here.

The trace id comes from the client rather than the server. The client knows it before it sends, so a request that times out, hangs, or dies mid-stream can still be found — and those are the failures most worth tracing.

A consequence to keep in view is that the id is caller-supplied: a caller reusing one across requests asserts that those requests are related, which on the anonymous surface is a self-linking primitive.

`traceparent` is parsed directly by Eidola code rather than by installing a `TextMapPropagator`. A global propagator would also inject context into the server's *outbound* calls, putting the trace id on requests to the inference upstream, Stripe, and GitHub. As written, no outbound request carries trace context, and a traced request's spans go only to the OTLP endpoint.

Inbound headers are counted by outcome (`sampled`, `not_sampled`, `malformed`).

## Inference is proxied, not performed

The Eidola server is **not** the inference engine. Models run in a separate confidential-compute deployment operated by the upstream inference provider (Tinfoil), with its own attestation. The Eidola server's role on the inference path is:

1. Verify the anonymous credential.
2. Open an attested HTTPS connection to the inference upstream.
3. Stream the request through, stream the response back.
4. Record the per-request token counts for accounting.

This means **two layers of confidential compute** protect the inference content: the Eidola server enclave (which sees the content only in transit, never logged) and the upstream provider's confidential-compute enclaves that perform the inference. The upstream is itself a *router* enclave that forwards to a separate per-model enclave; the Eidola server attests the router it connects to on every handshake, and how far that verification reaches (and where it stops) is detailed in [upstream.md](upstream.md) and [gaps.md § Inference upstream](gaps.md#inference-upstream). The client verifies the attestation of the Eidola server directly on every handshake.

## Where to read the code

| Subsystem | File |
|---|---|
| Anonymous credentials (Privacy Pass ACT) | `crates/eidola-server/src/credentials.rs` |
| Inference proxying | `crates/eidola-server/src/chat.rs` |
| Auth extractors | `crates/eidola-server/src/auth.rs` |
| Linked/unlinked routing | `crates/eidola-server/src/middleware.rs` |
| Telemetry boundary | `crates/eidola-server/src/telemetry.rs` |
| OpenAPI surface (tags = linked / unlinked / public) | `crates/eidola-server/src/api_doc.rs` |
