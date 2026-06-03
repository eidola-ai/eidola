# Privacy Guarantees

This document enumerates the privacy and integrity properties Eidola
commits to. It is the referent for the
`privacy_guarantees_not_weakened`, `code_delivers_guarantees`, and
`no_known_backdoor` claims in the release attestation schema (see
[releases.md](releases.md)): a release attestation signed under this
document's content hash asserts that the release does not weaken any
item below and that no known code path violates them.

Each item is stated as an invariant. Contributors maintain these
invariants when changing code. Release attestants walk a diff against
them before signing.

Items are labelled **[S]** (structurally enforced — broken only by
code that defeats the architecture, e.g. typed routing, blind-signature
math, or build-time pinning) or **[P]** (policy — broken by code that
violates stated discipline).

These invariants describe what you get when you install and run a
**generally-available release of Eidola** — the app downloaded from
the public release channel, running on hardware whose
confidential-compute attestation verifies under the trust root that
release was built with. They do not extend to development builds,
contributor-installed test versions, or any scenario where
configuration overrides have been set (see
[client.md](client.md#configuration-overrides) and
[trust-root.md](trust-root.md#whats-pinned)).

---

## 1. Identity and authorization

**1.1.** **[S]** Every server endpoint is classified `linked`,
`unlinked`, or `public`. The classification is bound to the handler in
code and checked in middleware before the handler executes. (See
[server.md](server.md#linked-vs-unlinked).)

**1.2.** **[S]** `unlinked` endpoints accept only anonymous credential
tokens (Privacy Pass ACT, `draft-schlesinger-privacypass-act-01`).
They never receive, derive, persist, or log any identifier that ties
a request to its issuance transaction, to the account it was issued
to, or to other requests from the same client.

**1.3.** **[S]** `linked` endpoints accept only the HTTP Basic
`(account_uuid, account_secret)` bearer pair. They never accept ACTs,
and they never receive or emit inference request or response content.

**1.4.** **[S]** The two authentication surfaces are disjoint at the
Rust type level: the `BasicAuth` and `TokenAuth` extractors are
distinct types, and an endpoint may take only one. Cross-acceptance
requires a code change visible in a diff.

**1.5.** **[P]** No personally identifiable information is requested
or accepted at account creation. Email, phone, name, address, and
government identifiers are never collected or stored by Eidola.
Stripe's own retention is governed by Stripe and is out of scope
(see §8.4).

## 2. Unlinkability

**2.1.** **[S]** *Issuance ↔ redemption.* An ACT presented at an
`unlinked` endpoint is cryptographically unlinkable, by the
blind-signature construction, to the issuance transaction that
produced it. With full access to its own database, the server cannot
answer "which account paid for *this* inference request."

**2.2.** **[S]** *Redemption ↔ redemption.* ACTs presented across
different requests are cryptographically unlinkable to each other.
The server cannot answer "which inference requests came from the
same account."

**2.3.** **[S]** The anonymity set for a given token is the set of
accounts that received at least one token under the same
`(issuer_key, domain_separator)` during the issuer key's issuance
window. Issuance and key-rotation policies are tuned to keep this
set as sufficiently large. (See [server.md](server.md#anonymity-set).)

**2.4.** **[S]** Issuance and redemption are temporally decoupled:
tokens remain redeemable across an acceptance window that extends
beyond their issuance window, so the issuance timestamp on the linked
surface and the redemption timestamp on the unlinked surface are not
forced to be near-equal.

**2.5.** **[S]** No identifier carried on the inference path
(credential bytes, request context, nullifier) is correlatable with
any record on the linked surface. The two surfaces share no
in-process state and no persistence path beyond the ACT issuance and
redemption protocol itself. (Notwithstanding network-layer signals —
IP address, packet timing — which are out of scope; see
[gaps.md](gaps.md#network-identity-as-a-linking-factor).)

## 3. Content

**3.1.** **[S]** Inference request and response content (prompts,
attachments, model outputs, tool inputs and tool results) is never
written to durable storage on Eidola-controlled infrastructure.

**3.2.** **[S]** Inference content is never included in logs,
telemetry, traces, error reports, or crash dumps, nor in any
derived form that could meaningfully identify a request or link it
to other requests — including content hashes, content lengths at
request granularity, or per-request metadata beyond what is
needed to bill, route, or operate the request. Aggregate counters
(e.g. tokens-by-model totals) and request-shaped operational
fields (status code, latency, route) are not in scope: they do not
encode content and do not bind to any account identifier.

**3.3.** **[P]** Telemetry on the inference path is limited to model
name, token counts, status code, and latency. The classifier that
splits telemetry between the linked and unlinked surfaces runs in
middleware before the span is created. (See
[server.md](server.md#telemetry-scope-and-boundary).)

**3.4.** **[S]** Eidola service handlers do not persist or emit
client IP addresses, user-agent strings, TLS fingerprints, or other
network-layer identifiers. Network infrastructure outside the
enclave (CDNs, load balancers, ISPs, the user's own network path)
may log such identifiers and is out of scope for this invariant;
the application-level promise is that Eidola code does not
re-introduce them into its own observability or persistence
surfaces.

**3.5.** **[S]** Inference content is never cleartext on the wire.
It is decrypted only in the ephemeral memory of (a) the Eidola
server enclave, while being routed to the upstream, and (b) the
upstream inference enclave, whose attestation the Eidola server
verifies per-handshake. Every link between client, server enclave,
and upstream enclave is TLS terminated inside the respective
enclave; no operator, host, orchestrator, or network observer has
cleartext access at any point. (See [upstream.md](upstream.md).)

**3.6.** **[P]** The Eidola server is request-based: on the inference path, there is no cross-connection cache persisted outside ephemeral enclave memory, and no per-account learned state. There is no operator-facing interface for inspecting, reviewing, approving, flagging, or replaying inference traffic.

## 4. Transport and server attestation

**4.1.** **[S]** TLS is terminated inside the Eidola server enclave.
The TLS private key is sealed to the enclave by the confidential-
compute runtime; no operator, host, or orchestrator has access to it.

**4.2.** **[S]** The client re-verifies the server's hardware
attestation on every new TCP+TLS handshake. There is no "verified
once" cache; policy changes (TCB floor, allowed measurements) take
effect on the next handshake. (See
[client.md](client.md#per-handshake-attestation-no-caching).)

**4.3.** **[S]** The attestation report's is
checked to match the expected peer cert. The inline attestation rides the *same* TCP+TLS
connection as the subsequent application request, so attestation and
request share one HTTP lifecycle and the LB-routed backend that
served the attestation is the one that serves the request.

**4.4.** **[S]** A TCB policy floor is enforced on every attestation.
Measurements outside `ALLOWED_MEASUREMENTS` are rejected.

**4.5.** **[S]** The same per-handshake verification discipline
applies to the Eidola server's outbound connections to the inference
upstream. (See [upstream.md](upstream.md#per-connection-verification).)

**4.6.** **[S]** Each client release pins **exactly one** server-
enclave measurement. There is no minimum-version floor and no
`any of N` list; a different server build requires a different
client release. (See
[client.md](client.md#one-release-pairs-exactly-one-client-with-one-server).)

**4.7.** **[S]** Verification is fail-safe. There is no degraded
mode, no trust-on-first-use fallback, no user prompt to ignore a
failed attestation. Inability to verify ⇒ the connection does not
happen. (See [client.md](client.md#fail-safe-by-design).)

## 5. Server measurement and configuration binding

**5.1.** **[S]** The server-enclave measurement is a deterministic
function of source: OVMF firmware (pinned), CVM kernel + initrd
(pinned), the kernel command line (which embeds the SHA-256 of
`tinfoil-config.yml`), and the vCPU count and type. Any change to the
attested boot path produces a different measurement, which the client
refuses to connect to. TODO: <https://github.com/tinfoilsh/measure-image-action/pull/48>

**5.2.** **[S]** The full server runtime configuration —
image digest, argument list, environment variable schema, and
hashes of all measured secrets — lives in `tinfoil-config.yml` and
is therefore bound into the measurement via §5.1. Configuration
changes are release events.

**5.3.** **[S]** Secrets that allow access to persisted state inside the
enclave (`CREDENTIAL_MASTER_KEY`, `DATABASE_PASSWORD`) are injected
as Tinfoil secrets bound to the enclave measurement. A different
measurement cannot retrieve them; the server image itsle has no
intrinsic ability to access its own persisted state outside the attested
boot path.

**5.4.** **[S]** The client trust root pins exactly one upstream-
inference enclave-measurement set per release
(`releases/trust/tinfoil-enclaves.json`, baked into the server
binary at compile time). Updates go through PR review with Sigstore
provenance verification before merge. (See
[upstream.md](upstream.md#what-pins-the-upstream-measurement).)

**5.5.** **[S]** Hardware-attestation collateral that the operator
could plausibly poison (AMD KDS CRLs for SEV-SNP, Intel PCS
collateral — TCB info, QE identity, PCK CRLs — for TDX) is fetched
by the verifier directly from the hardware vendor in production
mode; the operator is never a relay for its own collateral.

## 6. Release integrity

**6.1.** **[S]** Every released binary is bit-reproducible from
public source. CI re-derives `artifact-manifest.json` on every PR to
`main` and refuses to merge if the result differs from the committed copy;
reproducibility is a *merge invariant*, not just a release-time
property.

**6.2.** **[S]** Every release carries at least `MIN_HUMAN_ATTESTATIONS`
independent human attestations conforming to the schema pinned in
the client trust root. Each attestation is signed via `cosign
sign-blob` under a hardware-bound key whose
`sha256(PKIX SubjectPublicKeyInfo DER)` matches a fingerprint in
`TRUSTED_ATTESTANT_FINGERPRINTS`, and is recorded in the Sigstore
Rekor transparency log as a `hashedrekord` v0.0.1 entry.

**6.3.** **[S]** Every human release attestation contains positive,
prose-equal claims that the attestant: (a) personally reproduced
`artifact-manifest.json` from the source commit on hardware under
exclusive physical control, (b) reviewed the source-level diff
against the prior release, (c) is not aware of any backdoor, covert
surveillance mechanism, or undisclosed data path in the release,
(d) is not aware of any change that causes the code to fail to
deliver these guarantees, and (e) confirms this document does not
weaken, narrow, or remove any item that was in effect at the prior
release. The verifier re-renders each claim from a pinned template
and rejects any character mismatch. (See
[releases.md](releases.md#what-the-engineer-attests-to).)

**6.4.** **[S]** Every release attestation contains positive,
prose-equal claims that the attestant is **not** subject to legal
compulsion that has caused the release to weaken any guarantee, is
**not** subject to a gag order preventing truthful attestation,
is **not** coerced, and is signing of their own volition with a
hardware-held key under their exclusive physical control.

**6.5.** **[S]** The client trust root (server-enclave measurement,
attestant fingerprints, CI identity pattern, supported schema
versions, attestation-claim templates, Sigstore trusted root) is
embedded at build time from committed source files. There is no
runtime API to mutate the trust root or alter policy. (See
[trust-root.md](trust-root.md#whats-pinned).)

**6.6.** **[S]** `MIN_HUMAN_ATTESTATIONS` is pinned in the
*currently-installed* client, not in the incoming release. A
coerced single attestant cannot lower the bar by shipping a release
that requires fewer signatures.

**6.7.** **[S]** Self-update requires that the incoming release's
`previous_release.git_commit` equal the currently-installed
`git_commit`. Stale-release substitution and rollback to a known-bad
past release both fail this check.

**6.8.** **[S]** Release-document `schema_version` values are
integers with no semver tolerance. The verifier refuses to parse any
document outside its pinned supported set; new fields cannot be
silently accepted. (See
[trust-root.md](trust-root.md#schema-versions-explicit-and-breaking).)

## 7. Source, build, and operational discipline

**7.1.** **[P]** All client code, server code, build configuration,
and release tooling are published in a public monorepo.

**7.2.** **[S]** Build environments are pinned and reproducible:
StageX (source-bootstrapped) for Linux OCI images, Nix flake
(hermetic, narHash-pinned) for the macOS universal-binary builds.
Build-environment hashes flow into the artifact manifest.

**7.3.** **[S]** Source dependencies are pinned by version and hash.
Updates are explicit commits.

**7.4.** **[P]** Logging and telemetry destinations are part of the
attested configuration (§5.2). Changing a destination is a release
event with a fresh human attestation.

**7.5.** **[P]** No feature is added whose privacy depends on
operator trustworthiness when a comparable feature with
cryptographic enforcement is implementable. When a policy-only
feature is added, it is labelled **[P]** in this document.

**7.6.** **[P]** Any feature whose existence would let an operator
answer the question "did account X ever do Y" is a violation of
this document, regardless of operator intent or cited rationale.

---

## 8. Bounded claims (what this document does not promise)

**8.1.** Eidola does not promise resistance to a local adversary
observing the user's device — keyloggers, compromised endpoints,
malicious peripherals, OS-level surveillance, or another process
with sufficient privilege. Local conversation history stored by
the client is no more or less private than any other file on the
user's device.

**8.2.** Eidola does not promise that inference models will not
retain content in weights, activations, or KV caches during a
request. That is the model author's domain. Eidola promises only
that *its* infrastructure does not retain content (§3).

**8.3.** Eidola does not promise unforgeability of ACTs from
a compromised issuer key. Forgery-enabled service abuse is an
operator-borne financial loss; it is never permitted to become
a user-borne privacy loss, because unlinkability (§2) survives.

**8.4.** Eidola does not promise anonymity against Stripe with
respect to payment metadata. The boundary Eidola enforces is
between payment metadata and service usage (§1.5). Stripe's own
retention and Eidola's retention of Stripe-collected data are out
of scope.

**8.5.** Eidola does not promise defense against traffic analysis.
Network metadata (the fact that a connection occurred, its size,
timing, originating IP) is visible to network observers. Users
who need that property should layer Eidola behind Tor or a similar
anonymity network. (See [gaps.md](gaps.md#traffic-analysis).)

**8.6.** Eidola does not promise the absence of bugs. An
undisclosed vulnerability is not a backdoor in this document's
sense (§6.3), but it could be exploited as if it were. The promise
is the absence of *intent* to subvert, not the absence of error.

**8.7.** Eidola does not promise defense against coordinated legal
compulsion of *every* pinned attestant simultaneously, under
credible secrecy. The mitigation is multi-jurisdiction attestant
distribution, named in
[gaps.md](gaps.md#multi-jurisdiction-attestant-distribution).

**8.8.** Eidola does not promise that confidential-compute hardware
vendors (AMD, Intel, NVIDIA) cannot issue fraudulent attestations.
That is residual trust we currently accept; see
[gaps.md](gaps.md#trust-in-confidential-compute-vendors).

---

## How this document evolves

Changes are **append-only in spirit.** Subsequent releases may
add items, narrow scope where doing so does not remove a promise,
or correct ambiguous wording. They may not remove or weaken any
item that was in effect at the prior release.

Strengthening goes through the normal release flow. Weakening
requires the attestant to be unable to sign
`privacy_guarantees_not_weakened` truthfully; the release notes
must call out the weakening explicitly, and users would have to
opt into such a release out of band. The verifier enforces the
structural side: a release whose attestation lacks
`privacy_guarantees_not_weakened` will fail.

## How to use this document

**Contributors.** Before opening a release PR, read this document
in full. Any diff that affects an item above must be called out in
the PR description, with a justification and any proposed
amendment.

**Release attestants.** When reviewing a release, walk this
document item by item against the diff between the previous and
current release commits. The `code_delivers_guarantees`,
`no_known_backdoor`, and `privacy_guarantees_not_weakened` claims
in the release attestation are positive statements that this walk
has been completed.

**External reviewers and citers.** Every item carries a stable
`§X.Y` identifier. Cite this file by durable hash (git commit hash
or file content hash) and item number; the numbers MAY change across
releases, although new items will generally be appended to preserve
identification.

---

*This document is versioned by content hash. The hash referenced
by a given release attestation is the version this document had
at that release. Prior versions are reachable via git history.*
