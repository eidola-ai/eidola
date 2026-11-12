# The gateway

The gateway is the part of Eidola's service that clients connect to. It runs the account and billing surface, and is an OpenAI-compatible front for Eidola's inference engines, routing each inference request to one of them. Its design is dominated by one structural decision: the **linked** account surface and the **unlinked** inference surface are kept disjoint, in code and at runtime.

## Linked vs. unlinked

The gateway's HTTP surface splits into two layers with separate authentication mechanisms:

| Surface | Auth | Sees account identity? | Sees inference content? |
|---|---|---|---|
| **Public** | none | n/a | no |
| **Linked** (account, billing) | HTTP Basic — account UUID + secret | yes | no |
| **Unlinked** (inference) | Anonymous credentials (Privacy Pass ACT) | **no** | only as opaque bytes in transit |

The split is structural. The two surfaces take different kinds of credential by different transports: an account endpoint takes HTTP Basic credentials in a header, and an inference endpoint takes anonymous credentials in the request body and refuses a request that carries an `Authorization` header. No endpoint accepts both, and there is no path through which an account credential could be accepted on an inference endpoint without a code change visible in a diff.

So the gateway, observing its own inputs, cannot connect an inference request to the account that paid for it. On the inference path it sees that an anonymous credential was redeemed under a particular issuer key, and cannot tell which account that credential was issued to.

## Unlinkability

Unlinkability comes from **Privacy Pass Anonymous Credit Tokens (ACT)**, per [draft-schlesinger-privacypass-act](https://datatracker.ietf.org/doc/draft-schlesinger-privacypass-act/). This section is a summary; [anonymous-credits.md](anonymous-credits.md) records every decision around the scheme and why it was made.

The flow:

1. **Issuance.** The authenticated client (Basic auth, linked surface) requests credentials. It generates blinding factors locally and sends blinded token requests. The gateway signs the blinded requests with its issuer private key and returns blinded signatures. The gateway learns *that* this account requested N credentials, never *which specific tokens* it received.
2. **Redemption.** When the client makes an inference request, it unblinds a token and presents it on the unlinked surface. The gateway verifies the token signature against the issuer public key. The token contains no account identifier, and the blind-signature construction guarantees the gateway cannot correlate this token with any individual issuance request from step 1.
3. **Refund.** Redeeming a token uses it up. The request holds a worst-case charge against it, and the gateway answers with a refund the client turns into a fresh token carrying the rest of the balance plus the part of the hold the request did not use.

   To stop a token being spent twice, its nullifier is recorded in Postgres before the request is served, with a provisional refund stored beside it in the same write. The final refund replaces that when the request ends. A client whose response was lost can fetch its refund again (`POST /v1/credentials/refund`), and a gateway crash mid-request costs the request's hold and never the rest of the token.

   Postgres is outside the enclave, and a refund's amount is a function of how long the prompt and the answer were. So the refund is stored **sealed under a key only the client can supply**. With each spend the client sends a refund recovery key, derived from secret state it already keeps for that spend. The gateway derives the encryption key from it in enclave memory, encrypts the refund, and keeps the client's key nowhere. The stored refund is the same length whatever the amount, and a spend without a key is refused before it is taken. What the database holds for a spend is the issuer key it was made under, its nullifier, and a ciphertext. The row is written twice, so someone watching the database can see when each write happened ([gaps.md](gaps.md#spend-timing-at-the-database)).

The issuer key is stored encrypted at rest in Postgres using a `CREDENTIAL_MASTER_KEY` that is injected into the gateway enclave as a Tinfoil secret, with the key's id, public key, domain separator and windows bound into the encryption as associated data, so a row whose plain columns were edited does not decrypt and is refused. If this key were compromised, new ACTs could be forged, but the unlinkability property remains.

Domain separation is baked into the credential construction (`ACT-v1:eidola:inference:production:<date>`) to prevent cross-deployment correlation if the issuer key were ever reused.

Account billing recovery runs separately from the gateway. The authenticated `POST /v1/account/reconcile` records a durable request and returns `202`; `GET` on the same path reports the requested and completed generations and the last completed result. Neither route calls Stripe or accepts a caller-chosen balance change. Independently deployable webhook receivers and billing workers share PostgreSQL with live gateway versions and the private support CLI. Their linked-account work may run outside confidential compute; ACT issuance and signing remain inside the gateway. See [payment reversals](anonymous-credits.md#payment-reversals).

## Anonymity set

The unlinkability invariants in [privacy-guarantees.md §2](privacy-guarantees.md#2-unlinkability) mean as much as each token's anonymity set is large, and hold only if the issuance and redemption policy introduces no linkable identifier of its own. The issuance and key-rotation policies are chosen for both.

**Anonymity set = users sharing the same issuer key + domain separator.** Every ACT token redeemed against a given issuer key is, by the math of the blind-signature scheme, indistinguishable from every other token issued under that key. The size of the set is the number of distinct accounts that received at least one token from that key during its issuance window.

**Issuer keys rotate on a 7-day epoch of a public calendar.** Epoch `n` starts at Unix time `n × 604800` (Thursday, 00:00 UTC); its key issues for that epoch and is accepted for one more, so each key's `issue_from`, `issue_until` and `accept_until` are a function of its epoch that anyone can compute, and there is exactly one key per epoch. Those windows are bound into the encryption of the key's private half, so the database that stores it cannot change them, and every client refuses a key listing with any key off the calendar (compared byte for byte), any epoch with two keys or key with two epochs, or any key other than the one it already saw for that epoch ([anonymous-credits.md § Key rotation and partitioning](anonymous-credits.md#key-rotation-and-partitioning)). While a key's issuing window covers now, that key signs new credentials. After `issue_until` the key stops issuing, but tokens already issued under it remain redeemable until `accept_until` (a grace period beyond the issuance window). The client counts a token's spendable life from `accept_until`, not `issue_until`, so a token issued in the last moments of its key's issuance window is still spendable for the whole grace period; what is left in a token when its key lapses is lost ([gaps.md § Credit left in a credential when its key lapses](gaps.md#credit-left-in-a-credential-when-its-key-lapses)). So at any moment more than one key is spendable, but only one is issuing. Two things follow:

- Every token has a bounded lifetime, so keys are retired rather than tokens revoked one by one.
- A token need not be redeemed the moment it is issued. It can be requested in advance, held by the client, and spent later, which keeps issuance and redemption from being correlated by time.

`accept_until` is a hard deadline. A spend that arrives at or after it is refused, even if the gateway still holds the key in memory. A spend accepted just before it still finishes, so its refund can be signed and recovered for a short retention margin afterwards. Nullifiers under the key, the record of which tokens were already spent, are deleted only after that margin. So no token can be spent twice by arriving after its nullifier was pruned. A refund is always issued under the spent token's own key, so it lapses at the same `accept_until`, and no spend can move value into a later epoch.

**The domain separator does not rotate with the keys.** The anonymity set is the users sharing the same key *and* the same domain separator, so rotating the separator would shrink the set for no gain: nullifiers, which prevent double spending, are partitioned by issuer key. The separator changes only on a protocol upgrade or a change of deployment identity.

**Issuance ahead of need.** Issuance happens on the linked surface and redemption on the unlinked one. If the two were always near-simultaneous, an observer of both could pair an account's issuance with the redemption that follows it. The client therefore keeps a small buffer of tokens issued ahead of need, at times unrelated to any request ([anonymous-credits.md § Issuance timing](anonymous-credits.md#issuance-timing)). The cases where it still issues and spends at once are in [gaps.md](gaps.md#issuance-to-first-spend-timing).

This policy does not help when few accounts issue under a key, as early in the deployment ([gaps.md § Anonymity-set size](gaps.md#anonymity-set-size)).

## What runs in confidential compute

The gateway runs inside a **Tinfoil Container** on confidential-compute hardware (AMD SEV-SNP; TDX values are recorded in the trust root but the client refuses a TDX presentation of the gateway — see [gaps.md](gaps.md#tdx-acceptance)). The relevant properties:

- **TLS termination is inside the enclave.** The Tinfoil shim's self-contained v3 attestation endorses the TLS certificate's SPKI and an HPKE public key; its fresh hardware report binds those exact endorsed bytes and the client's nonce. The certificate is issued by a public CA via ACME, so any client can validate the chain; the *binding to the enclave* is what the Eidola verifier checks beyond the basic WebPKI chain.
- **Secrets are sealed into the enclave.** Both `CREDENTIAL_MASTER_KEY` and `DATABASE_PASSWORD` are Tinfoil secrets, decrypted only inside the verified enclave. They are not visible to the host, the orchestrator, or any operator.
- **The enclave measurement is a function of source.** The client's pinned launch identity is computed from the platform's hash-pinned release image (its SEV-SNP launch digest) and the committed `tinfoil-config.yml` (its SHA-256, which the enclave reports as HOST_DATA). See [trust-root.md](trust-root.md#whats-pinned).

The gateway image is `FROM scratch` with a statically linked binary, runs as non-root, and ships no shell or package manager.

## What the gateway does not do

- **No session caching, no request memory, no learned state per account.** The gateway is request-based. Two inference requests with the same content produce two independent engine calls. The gateway caches nothing across requests, because a cache is a correlation surface. The one cache on the inference path is the engine's prefix cache, which is scoped by a key the client chooses and rotates ([privacy-guarantees.md §2.6](privacy-guarantees.md#2-unlinkability), [§3.7](privacy-guarantees.md#3-content)).
- **No content-based logging.** Inference request bodies and response bodies are not logged, persisted, or tee'd into observability systems. Telemetry carries no per-request records: spans are aggregated into metrics in process and exported only when the client asks ([below](#client-directed-tracing)).
- **No operator review.** There is no human-review queue, no flagging system, and no operator interface for inspecting inference traffic.

## Telemetry: scope and boundary

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, the gateway exports OpenTelemetry metrics and logs. It does **not** export traces unless directed by the client.

**Spans are recorded everywhere and exported only on request.** Tracing runs across the whole gateway: the HTTP request, each database round trip, each outbound call. Every span goes to a processor that converts it into histogram observations and drops it. A second, exporting processor is also registered, but it drops unsampled spans. The sampler returns `RecordOnly` for every trace that was not explicitly authorized, so in ordinary operation the export path carries nothing.

A span is a per-request record (a trace id, a start time, a duration), and a request's duration depends on its content, since a response's wall time varies with its length. Aggregating in process answers operational questions such as where a route spends its time without keeping a record of what any one request did. Nothing per-request reaches the telemetry vendor, so no account identifier can.

What is exported:

- **Metrics.** Counters and histograms aggregated over the export interval. Content-derived quantities appear here and only here — token totals keyed by model — never attributed to an individual request. Every label value comes from a fixed set: the requested model reference is resolved to a catalog row's weights hash (never the tag the client sent), span names against a tracked list, and anything unrecognized collapses to a shared `other` bucket. The instruments are per-operation duration derived from spans; time to first token, stream duration, largest inter-chunk gap and sustained output rate for inference; and a stream-outcome counter. Stripe webhook delivery belongs to the separate billing receiver and is not part of gateway telemetry.
- **Logs.** Error and warning events. An internal error has two renderings: the client gets the full detail on its own attested connection, and the log gets a redacted one. Fields that could carry request-derived values, and error strings authored outside Eidola's code, are never logged.

No span captures a function's arguments as fields, so nullifiers, spend proofs and account ids cannot reach a span attribute, including under client-directed tracing.

### Client-directed tracing

A request that carries a W3C `traceparent` with the sampled flag set is traced at full granularity: middleware installs it as a sampled remote parent, which promotes the root span and, through it, every child span in the request. Nothing else marks a request for export.

The sampled flag is honored rather than assumed. Per [W3C Trace Context](https://www.w3.org/TR/trace-context/) it reports whether the caller recorded the request; it is a hint about the caller's own decision, not an instruction, and a receiver is free to sample independently. Honoring it costs nothing and means a client whose tracing is off does not cause an export here.

The trace id comes from the client, which knows it before it sends, so a request that times out or dies mid-stream can still be found. Because the id is caller-supplied, a caller that reuses one across requests links those requests itself.

`traceparent` is parsed directly by Eidola code rather than by installing a `TextMapPropagator`. A global propagator would also inject context into the gateway's *outbound* calls, putting the trace id on requests to the inference engines, Stripe, and the website the terms feed is read from. As written, no outbound request carries trace context, and a traced request's spans go only to the OTLP endpoint.

Inbound headers are counted by outcome (`sampled`, `not_sampled`, `malformed`).

## Inference is proxied, not performed

The gateway does not run models. Models run on Eidola's inference engines, in separate confidential-compute deployments, each attested and pinned in the gateway's own build. The gateway's role on the inference path is:

1. Verify the anonymous credential.
2. Open an attested HTTPS connection to an engine deployment pinned for the requested model.
3. Send the request, and stream the response back, re-framed so that what a network observer sees does not follow the tokens (see [Response shape](#response-shape)).
4. Charge the credential for the tokens used, and return the change.

Inference content is therefore cleartext in two enclaves: the gateway's, in transit, and the engine's confidential VM, which performs the inference. The client verifies the attestation of the gateway on every handshake, and the gateway verifies the engine's on every handshake, against the deployments its own build pins. What that verification covers, and where it stops, is detailed in [engine.md](engine.md). One deployment is pinned and none is running yet, so today the gateway lists one model and refuses its requests, with a full refund.

### Response shape

Encrypted traffic still has a size and a timing. A streamed answer relayed chunk by chunk puts one encrypted record on the wire per token, whose sizes follow the tokens' lengths and whose spacing follows the model's pace; published attacks recover topics and text from that. So the gateway writes a streamed answer as frames of one fixed size (2 KiB) at a fixed interval (every 50 ms), from the moment the stream opens until it ends. Each frame carries whatever output is waiting and is filled out with padding that every server-sent-events reader ignores. Every other response the gateway sends (a blocking answer, an error, the refusal of a malformed request, a 404) is padded to the next power-of-two size, at least 4 KiB, and a refusal never quotes what the request contained. An observer on the network path learns how long a response ran (which tracks its length) and a padded response's size bucket, not the size or timing of any token. The engine shapes its answer to the gateway the same way, with frames of 4 KiB every 20 ms and every other response padded, so the connection between the two enclaves shows how long a response ran and not its tokens; the gateway takes the answer out of the engine's frames, reads what it bills from it, and re-frames it for the client on its own interval. Neither hides the engine's own activity from its host. The reasoning and its costs are in `crates/eidola-server-gateway/AGENTS.md` and `crates/eidola-server-engine/AGENTS.md`.

## Where to read the code

| Subsystem | File |
|---|---|
| Anonymous credentials (Privacy Pass ACT) | `crates/eidola-server-gateway/src/credentials.rs` |
| Inference proxying | `crates/eidola-server-gateway/src/handlers.rs`, `crates/eidola-server-gateway/src/engine_router/` |
| Authentication | `crates/eidola-server-gateway/src/auth.rs` |
| Linked/unlinked routing | `crates/eidola-server-gateway/src/middleware.rs` |
| Telemetry boundary | `crates/eidola-server-gateway/src/telemetry.rs` |
| OpenAPI surface (tags = linked / unlinked / public) | `crates/eidola-server-gateway/src/api_doc.rs` |
