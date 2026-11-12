# Privacy guarantees

This document enumerates the privacy and integrity properties Eidola commits to. It is the referent for several claims in the release attestation schema (see [releases.md](releases.md)): a release attestation signed under this document's content hash asserts that the release does not weaken any item below and that no known code path violates them.

Each item is stated as an invariant. Contributors maintain these invariants when changing code and release attestants walk a diff against them before signing.

Every item is a permanent property of a specific release. Most are inherent attributes of the attested source: properties you can check against the release's exact code, while the rest are facts fixed at release time by a signed attestation claim. If a promise could be broken *after* a release ships — ongoing operational conduct, the future behavior of a third party — it does not belong here; it belongs in the [privacy policy](https://www.eidola.ai/privacy/), where it binds us as conduct rather than as code. Where an item is enforced architecturally — typed routing, blind-signature math, build-time pinning — it names the mechanism; where none is named, the enforcement is simply that the fact is as described.

> [!IMPORTANT]
> These invariants describe an attested, **generally-available release of Eidola** running under the built-in trust root. They do not extend to development builds, contributor-installed test versions, or any scenario where configuration overrides have been set (see [Client](client.md#configuration-overrides) and [Trust root](trust-root.md#whats-pinned)).

---

## 1. Identity and authorization

**1.1.** Every endpoint of the gateway belongs to exactly one of three surfaces: `linked`, `unlinked`, or `public`. The two authenticated surfaces take different kinds of credentials, and no endpoint accepts both, so nothing presented on one surface is ever presented on the other. Which surface an endpoint belongs to is fixed in its code rather than by configuration or routing, and moving one requires a code change visible in a diff. (See [Gateway](gateway.md#linked-vs-unlinked).)

**1.2.** `unlinked` endpoints accept only anonymous credential tokens ([Privacy Pass ACT](https://datatracker.ietf.org/doc/draft-schlesinger-privacypass-act/)), and refuse a request that carries any other credential. They never receive, derive, persist, or log any identifier that ties a request to its issuance transaction, to the account it was issued to, or to other requests from the same client, beyond what the client itself sends to link them: a prefix-cache key (§2.6), credentials presented together (§2.9), or a trace context it asks to have exported (§3.3).

**1.3.** `linked` endpoints never accept an anonymous credential. Those authenticated as the account holder accept the account's identifier and secret and nothing else. Two are authenticated otherwise, of necessity: account creation, which mints the credential it would otherwise require, and the Stripe webhook, which is authenticated by HMAC signature over the request body. None of them receives or emits inference request or response content.

**1.4.** No personally identifiable information is requested or accepted at account creation. Email, phone, name, address, and government identifiers are never collected by Eidola's API or stored on Eidola's servers. Payment runs through Stripe; what Stripe collects and retains is conduct outside the attested source, covered by the [privacy policy](https://www.eidola.ai/privacy/) and bounded in §8.4.

## 2. Unlinkability

**2.1.** *Issuance ↔ redemption.* An ACT presented at an `unlinked` endpoint is cryptographically unlinkable, by the blind-signature construction, to the issuance transaction that produced it. With full access to its own database, the gateway cannot answer "which account paid for *this* inference request."

**2.2.** *Redemption ↔ redemption.* ACTs presented across different requests are cryptographically unlinkable to each other. Eidola's servers cannot answer "which inference requests came from the same account."

**2.3.** The anonymity set for a given token is the set of accounts that received at least one token under the same `(issuer_key, domain_separator)` during the issuer key's issuance window. The partition parameters — the issuance-epoch length, the fixed non-rotating domain separator, and the acceptance window — are compile-time constants chosen to keep this partition as coarse as billing correctness allows. How many accounts actually populate a window is a deployment fact the code cannot promise; the small-population residual is disclosed in [gaps.md](gaps.md#anonymity-set-size). (See [gateway.md](gateway.md#anonymity-set).)

**2.4.** Issuance and redemption are temporally decoupled: tokens remain redeemable across an acceptance window that extends beyond their issuance window, so the issuance timestamp on the linked surface and the redemption timestamp on the unlinked surface are not forced to be near-equal.

**2.5.** No identifier carried on the inference path (credential bytes, request context, nullifier) is correlatable with any record on the linked surface. The two surfaces share no in-process state and no persistence path beyond the ACT issuance and redemption protocol itself. (Notwithstanding network-layer signals — IP address, packet timing — which are out of scope; see [gaps.md](gaps.md#network-identity-as-a-linking-factor).)

**2.6.** *Prefix-cache keys, server side.* A request may carry a cache key: a random value the client chooses and rotates, which scopes the reuse of work already done on a prompt's prefix. Requests carrying the same key may share that work; no other requests do. A key links only the requests the client sends it with. Eidola's servers never write a key, or any value derived from one, to a log or to durable storage: both exist only in enclave memory. The key itself is held only while a request carrying it is in flight; one-way values derived from it are kept for as long as the cached work is (§3.7). A key is sent only to an engine its request is tried on, and requests sharing a key try the same engines in the same order. Which engine carries traffic at a given moment is observable outside the enclaves, so to an observer of the connections between the gateway and the engines, requests sharing a key are linkable by where they go as well as by when.

**2.7.** *Prefix-cache keys, client side.* The client sends a cache key only for a model declared to have a prefix cache, with a stated limit on how long an idle entry is kept and on how old one may grow. A key is random and belongs to one participant answering in one space, under the server endpoint and trust configuration it was created for; it is never sent anywhere else. The client replaces it before any request that would come after the declared idle limit or past the declared maximum age. What a key lets anyone who sees it link is therefore the requests of one such conversation: no two consecutive ones further apart than the idle limit, and all within the maximum age of the first. (See [Client](client.md#prefix-cache-keys).)

**2.8.** *What a spend leaves in the database.* Spending an anonymous credential leaves one record: the credential's nullifier, so it can never be spent again, and the key it was issued under; and the change owed back to the client, encrypted so that only the client can read it. Nothing else about the request is stored with it: not its content, its model, its cost, or how much was refunded. The record is written when the request is accepted and rewritten once when it ends, and a request is never allowed to run past a fixed maximum time. Anyone who can watch the database can see when those writes happen, and so roughly how long each paid request took. (See [gaps.md](gaps.md#spend-timing-at-the-database).)

**2.9.** *Paying one request with more than one credential.* A request may be paid from a small number of credentials at once, so that a credential too small to pay for a request by itself is used up rather than left to expire. Credentials presented together are known to come from the same client. That places the request among the accounts issued at least that many credentials under each key presented, and the client provisions credentials so that this is every account that provisioned under that key. A credential used up this way discloses its remaining balance to the gateway's measured code for the duration of the request; that code keeps no per-request amount. The database sees which of its records belong to one such request (§2.8), never an amount. (See [anonymous-credits.md](anonymous-credits.md#spending-several-tokens-in-one-request) and [gaps.md](gaps.md#a-drained-balance-is-visible-to-the-measured-code).)

## 3. Content

**3.1.** Inference request and response content (prompts, attachments, model outputs, tool inputs and tool results) is never written to durable storage on Eidola's servers.

**3.2.** Inference content is never included in logs, telemetry, traces, error reports, or crash dumps, nor in any derived form that could meaningfully identify a request or link it to other requests — including content hashes, content lengths at request granularity, or per-request metadata beyond what is needed to bill, route, or operate the request.

**3.3.** Telemetry is limited to generic, unidentifiable fields as aggregate counters and histograms, keyed only by low-cardinality attributes fixed in the code (route template, model, operation, status, outcome), unless a request explicitly asks to be traced. An aggregate covering an interval in which a single request fell describes that request. Diagnostic logs are written per-event, but carry no content, no value derived from content, and no span context by which to group them back into a request. Values that originate outside Eidola, such as a Stripe event type, are resolved against a fixed list before use, and anything unrecognized collapses to a shared bucket. The model a request names is resolved, as the request arrives, to the weights hash of a model Eidola sells; from then on logs, telemetry, routing and billing know the request's model only by that hash, never by the name the client used.

**3.4.** Eidola service handlers do not persist or emit client IP addresses, user-agent strings, TLS fingerprints, or other network-layer identifiers. Network infrastructure outside the enclave (CDNs, load balancers, ISPs, the user's own network path) may log such identifiers and is out of scope for this invariant; the application-level promise is that Eidola code does not re-introduce them into its own observability or persistence surfaces.

**3.5.** Inference content is never cleartext on the wire or anywhere outside the ephemeral memory of confidential-compute enclaves: (a) the gateway's enclave, while the request is routed, and (b) the enclave of the inference engine that serves it, a confidential VM with confidential-computing GPUs, which the gateway attests on every new connection against the engine deployments pinned in its own measured build. Every link is TLS terminated inside the respective enclave, so no host, orchestrator, or network observer has cleartext access in transit. Eidola attests every enclave on the inference path itself (the client attests the gateway, and the gateway attests the engine); none is trusted on a third party's attestation of it. The GPUs are vouched for by the measured guest, which admits them only after their vendor's verifier has passed inside it and which the engine checks again before use, rather than by the gateway's own appraisal of their evidence. (See [engine.md](engine.md) and [gaps.md § TDX acceptance](gaps.md#tdx-acceptance).)

**3.6.** Eidola's servers are request-based: on the inference path, there is no cross-connection cache persisted outside ephemeral enclave memory, and no per-account learned state. There is no operator-facing interface for inspecting, reviewing, approving, flagging, or replaying inference traffic.

**3.7.** *What an inference engine keeps.* An engine holds a request's content, and everything computed from it, only in enclave memory; none of it is logged or written to durable storage. When a request without a cache key (§2.6) ends, the engine keeps nothing of it for later use: what it computed is erased or released, though released memory is not all overwritten at once and can hold remnants inside the enclave until it is reused. For a request with a cache key, the engine keeps the computed state of the prompt and answer, for no longer than the limits declared for the model, and only a request presenting the same key can reuse or read it. Between requests that state is encrypted under a key derived from the client's, which the engine does not keep, so the engine cannot read an idle entry by itself. What it keeps to find the entry again is a one-way value that reveals neither the key nor the prompt.

**3.8.** *The shape of an answer.* A network observer between the client and the gateway cannot see the size or timing of an answer's individual tokens. A streamed answer is written in frames of one fixed size at a fixed interval whatever it contains, and every other response, refusals included, is padded to a size bucket. The observer still learns when a response began, how long it took (which tracks the answer's length and how predictable its content was), its size to within a bucket, and everything about the request, which is not padded. This covers the client's connection only: the gateway's connection to an engine carries the stream as the engine produces it, and the engine's host can observe that connection and the engine's own timing (§8.5).

## 4. Transport and attestation

**4.1.** TLS is terminated inside the gateway's enclave by the attestation shim, which is part of the measured boot image (§5.1). The private key is generated and held only in enclave memory; no operator, host, or orchestrator interface can export it. This is a behavior of the pinned image itself, not of any external secret store. The residual — a defect in that image leaking the key would undermine the per-handshake verification built on it — is disclosed in [gaps.md](gaps.md#tls-key-exfiltration-and-channel-binding).

**4.2.** The client re-verifies the gateway's hardware attestation on every new TCP+TLS handshake. There is no "verified once" cache; policy changes (TCB floor, allowed measurements) take effect on the next handshake. (See [client.md](client.md#per-handshake-attestation-no-caching).)

**4.3.** The attestation report is bound to the expected peer cert: its `REPORT_DATA` commits to the exact endorsed crypto-material section containing `sha256(SPKI(peer_cert))`, plus the verifier's fresh nonce and the device-evidence section. The inline attestation rides the *same* TCP+TLS connection as the subsequent application request, preserving ordinary load-balancer affinity. It does not commit TLS exporter or other session-specific key material; the residual for an exfiltrated TLS key is explicit in [gaps.md](gaps.md#tls-key-exfiltration-and-channel-binding).

**4.4.** A TCB policy floor is enforced on every attestation. A report whose launch measurement, configuration hash, or guest policy does not match the gateway launch identity pinned in the client is rejected. The client accepts only SEV-SNP attestations of the gateway: a TDX presentation is refused outright rather than checked under a weaker policy, so the attestation document's author cannot select a softer platform branch. (See [gaps.md](gaps.md#tdx-acceptance).)

**4.5.** The same per-handshake verification discipline applies to the gateway's outbound connections to the inference engines, each checked against the engine deployments pinned for its model (§5.2). (See [engine.md](engine.md#per-connection-verification).)

**4.6.** Verification is fail-safe. There is no degraded mode, no trust-on-first-use fallback, no user prompt to ignore a failed attestation. Inability to verify ⇒ the connection does not happen. (See [client.md](client.md#fail-safe-by-design).)

## 5. Measurement and configuration binding

**5.1.** *What is measured.* The gateway and every inference engine run in enclaves whose hardware reports a launch identity with each attestation: the measurement of the boot image the enclave starts from, and the hash of its configuration. The configuration names the exact image the enclave runs, its arguments and its environment. Both parts are fixed by the release's source, and anyone can recompute them from it. Nothing about what an enclave runs, or how, can change without changing its launch identity. The boot images are the hosting platform's, pinned by hash but not built by Eidola (§8.9), and the processor count and memory a host actually provides to the gateway are not measured.

**5.2.** *What is pinned.* Each client release pins the launch identity of exactly one gateway build, and refuses to connect on any difference. That gateway build in turn pins the launch identity of every engine it accepts, and sends inference content only to an engine whose attestation matches a pin for the requested model. The engine pins are part of the gateway's own measured image, so the client's one pin fixes the whole inference path. Nothing adds to either set at runtime, a gateway whose pins disagree with the engines' committed configurations does not build, and a change to the gateway or to any engine reaches users only through a new client release.

**5.3.** *Model weights.* A model is identified by the hash of its weights, never by its name, and the gateway's pin for an engine includes the weights it serves. For each model it sells, Eidola publishes that hash, where the weights come from, and the hash of each file, so anyone can fetch the files and recompute it. Every request names the weights it expects, and an engine serving any other refuses it. A client can ask for exact weights by hash and is refused rather than served others, and one answer never comes from two sets of weights. The client shows the user which weights their choice names, and tells them when those change or are retired. The hash pins the weights, not the engine's exact behaviour. (See [engine.md](engine.md#what-pins-the-engine) and [model-identity.md](model-identity.md).)

**5.4.** Hardware attestations are verified offline, against vendor roots pinned in the verifier: AMD's for the gateway, Intel's for the engines. The certificates and revocation lists needed arrive with the attestation, from a party that is not trusted, and are accepted only when signed by the pinned vendor and inside their signed validity period. An operator who withholds or corrupts them can only make verification fail. An engine's host must also be reported up to date by Intel. Because the verifier asks no vendor service for anything newer, an operator can replay older vendor-signed revocation data until its signed validity ends, which delays recognition of a newer revocation by that long.

## 6. Release integrity

**6.1.** Every released binary is bit-reproducible from public source. The only pull requests merged into `main` are through the promotion of the verified integration branch; the promotion pipeline re-derives `artifact-manifest.json` and refuses to advance if the result differs from the committed copy, and every other PR into `main` is refused outright.

**6.2.** Every release carries at least `MIN_HUMAN_ATTESTATIONS` independent human attestations conforming to the schema pinned in the client trust root. Each attestation is signed via `cosign sign-blob` under a hardware-bound key whose `sha256(PKIX SubjectPublicKeyInfo DER)` matches a fingerprint in `TRUSTED_ATTESTANT_FINGERPRINTS`, and is recorded in the Sigstore Rekor transparency log as a `hashedrekord` v0.0.1 entry.

**6.3.** Every human release attestation contains positive, prose-equal claims that the attestant: (a) personally reproduced `artifact-manifest.json` from the source commit on hardware under exclusive physical control, (b) reviewed the source-level diff against the prior release, (c) is not aware of any backdoor, covert surveillance mechanism, or undisclosed data path in the release, (d) is not aware of any change that causes the code to fail to deliver these guarantees, and (e) confirms this document does not weaken, narrow, or remove any item that was in effect at the prior release. The verifier re-renders each claim from a pinned template and rejects any character mismatch. (See [releases.md](releases.md#what-the-engineer-attests-to).)

**6.4.** Every release attestation contains positive, prose-equal claims that the attestant is **not** subject to legal compulsion that has caused the release to weaken any guarantee, is **not** subject to a gag order preventing truthful attestation, is **not** coerced, and is signing of their own volition with a hardware-held key under their exclusive physical control.

**6.5.** The client trust root (gateway launch identity, attestant fingerprints, CI identity pattern, supported schema versions, attestation-claim templates, Sigstore trusted root) is embedded at build time from committed source files. Release-verification policy — which keys, how many attestations, which schemas, which claim prose — has no override surface of any kind. Connection identity — which server to contact, which measurement and CA set to accept for it, and which endpoint serves release discovery — is mutable only through the explicit configuration surface described in [client.md](client.md#configuration-overrides), which the scope note above already excludes from these invariants; any value left unset falls back to the compiled-in pin. (See [trust-root.md](trust-root.md#whats-pinned).)

**6.6.** `MIN_HUMAN_ATTESTATIONS` is pinned in the *currently-installed* client, not in the incoming release. A coerced single attestant cannot lower the bar by shipping a release that requires fewer signatures.

**6.7.** Self-update requires that the incoming release's `previous_release.git_commit` equal the currently-installed `git_commit`. Stale-release substitution and rollback to a known-bad past release both fail this check.

> [!WARNING]
> **TODO, blocks release.** §6.7 is not enforced as written: no build can embed its own commit, so the client has no installed commit to compare against. Today an older or equal release is refused by version comparison only. This item must be reworded to the property actually enforced before any release is attested. See [#467](https://github.com/eidola-ai/eidola/issues/467).

**6.8.** Release-document `schema_version` values are integers with no semver tolerance. The verifier refuses a release document or attestation whose version is outside its pinned supported set. An artifact manifest with a version outside that set, or with a field the verifier does not recognize, is never accepted silently: it is shown to the user as a release whose claims differ from what the installed client expects, and is not trusted unless the user explicitly accepts it. (See [trust-root.md](trust-root.md#schema-versions-explicit-and-breaking).)

## 7. Source, build, and operational discipline

**7.1.** All client, gateway and inference engine code, build configuration, and release tooling live in a single repository, public at the moment of release: one git commit identifies the complete first-party source of a release. Dependencies — including temporarily maintained forks, themselves public — may sit outside that tree but are pinned from within it by version and hash (§7.4).

**7.2.** *Builds.* Every artifact of a release is built reproducibly, in an environment pinned in the same commit as its source, so the hashes a release publishes bind both the source and the tools that built it. The gateway and CLI images are built with a source-bootstrapped toolchain (StageX). The desktop apps and the inference engine's image are built with Nix from pinned inputs, which are not source-bootstrapped. The engine's image has two further limits: it is built inside a pinned container and not in Nix's own sandbox, and its GPU kernels are compiled with NVIDIA's closed CUDA toolkit, a compiler that cannot be inspected. Every build of that image checks the compiled kernels against hashes committed in the source.

**7.3.** *What the published hashes cover.* The artifact manifest records a hash for every image and every desktop payload a release produces, and everything functional inside them is covered, whith two caveats: the GPU driver a desktop app uses, which the user's device already supplies; and Apple's signature and notarization on a signed macOS app, which are not reproducible and are bound by the human release attestation instead. The GPU driver an engine loads is not in Eidola's image either, but it is part of the platform's boot image, and so of the engine's pinned launch identity (§5.1, §8.9). (See [verification.md](verification.md) and [apple-distribution.md](apple-distribution.md).)

**7.4.** Source dependencies are pinned by version and hash. Updates are explicit commits.

**7.5.** Logging and telemetry destinations are part of the attested configuration (§5.1). Changing a destination is a release event with a fresh human attestation.

**7.6.** No feature is added whose privacy depends on operator trustworthiness when a comparable feature with cryptographic enforcement is implementable. New items follow the introduction's enforcement-disclosure convention.

**7.7.** Any feature whose existence would let an operator answer the question "did account X ever do Y", for anything done on the `unlinked` surface, is a violation of this document, regardless of operator intent or cited rationale.

---

## 8. Bounded claims (what this document does not promise)

**8.1.** Eidola does not promise resistance to a local adversary observing the user's device — keyloggers, compromised endpoints, malicious peripherals, OS-level surveillance, or another process with sufficient privilege. Local conversation history stored by the client is no more or less private than any other file on the user's device.

**8.2.** Eidola does not promise anything about what a model's weights retain from their training; that is the model author's domain. Nor does it promise that an inference engine holds nothing between requests: a request that carries a cache key leaves a prefix-cache entry behind for a bounded time, on the terms of §3.7.

**8.3.** Eidola does not promise unforgeability of ACTs from a compromised issuer key. Forgery-enabled service abuse is an operator-borne financial loss; it is never permitted to become a user-borne privacy loss, because unlinkability (§2) survives.

**8.4.** Eidola does not promise anonymity against Stripe with respect to payment metadata. The boundary Eidola enforces is between payment metadata and service usage (§1.4). Stripe's own retention and Eidola's retention of Stripe-collected data are out of scope.

**8.5.** Eidola does not promise defense against traffic analysis. Network metadata (the fact that a connection occurred, its size, timing, originating IP) is visible to network observers, and what §3.8 hides of an answer's shape is all that is hidden. Users who need more should layer Eidola behind Tor or a similar anonymity network. (See [gaps.md](gaps.md#network--metadata).)

**8.6.** Eidola does not promise the absence of bugs. An undisclosed vulnerability is not a backdoor in this document's sense (§6.3), but it could be exploited as if it were. The promise is the absence of *intent* to subvert, not the absence of error.

**8.7.** Eidola does not promise defense against coordinated legal compulsion of *every* pinned attestant simultaneously, under credible secrecy. The mitigation is multi-jurisdiction attestant distribution, named in [gaps.md](gaps.md#multi-jurisdiction-attestant-distribution).

**8.8.** Eidola does not promise that confidential-compute hardware vendors (AMD, Intel, NVIDIA) cannot issue fraudulent attestations. That is residual trust we currently accept; see [gaps.md](gaps.md#trust-in-confidential-compute-vendors).

**8.9.** Eidola does not promise more of its hosting than its pins establish (§5.1, §5.2). In particular:

- **The boot images.** The gateway and every engine boot the platform's images. They are pinned by hash and measured, rebuilt by Eidola to ensure reproducibility, but are not source-bootstrapped, and come from a release line the platform still marks pre-release.
- **The gateway's host.** The client requires a genuine AMD processor at or above fixed firmware floors, but accepts any such machine. The current generation of confidential-computing hardware is vulnerable to attacks that need physical control of the machine, such as observing its memory bus, and the client cannot tell a gateway in the platform's data centre from one on a machine its operator can reach. Engines are held to machines the platform has endorsed.
- **The GPUs.** The measured boot image admits them only after their vendor's verifier has passed inside the enclave, and the engine checks their confidential-computing mode before using them, but the gateway does not itself appraise their evidence.

(See [gaps.md § TDX acceptance](gaps.md#tdx-acceptance) and [§ Build chain opacity](gaps.md#build-chain-opacity).)

---

## How this document evolves

Changes are **append-only in spirit.** Subsequent releases may add items, narrow scope where doing so does not remove a promise, or correct ambiguous wording. They may not remove or weaken any item that was in effect at the prior release without breaking the update flow and flagging the discrepancy to the user. Weakening includes downgrading an item's enforcement mechanism — replacing a structural check with reviewed discipline, say — even when the item's text is unchanged.

Strengthening goes through the normal release flow. Weakening requires the attestant to be unable to sign `privacy_guarantees_not_weakened` truthfully; the release notes must call out the weakening explicitly, and users would have to opt into such a release out of band. The verifier enforces the structural side: a release whose attestation lacks `privacy_guarantees_not_weakened` will fail.

## How to use this document

**Contributors.** Before opening a release PR, read this document in full. Any diff that affects an item above must be called out in the PR description, with a justification and any proposed amendment.

**Release attestants.** When reviewing a release, walk this document item by item against the diff between the previous and current release commits. The `code_delivers_guarantees`, `no_known_backdoor`, and `privacy_guarantees_not_weakened` claims in the release attestation are positive statements that this walk has been completed.

**External reviewers and citers.** Every item carries a stable `§X.Y` identifier. Cite this file by durable hash (git commit hash or file content hash) and item number; the numbers MAY change across releases, although new items will generally be appended to preserve identification.

---

*This document is versioned by content hash. The hash referenced by a given release attestation is the version this document had at that release. Prior versions are reachable via git history.*
