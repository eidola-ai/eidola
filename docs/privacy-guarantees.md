# Privacy Guarantees

This document contains explicit promises about Eidola. Each numbered guarantee below is a
property that Eidola commits to deliver, and that a release engineer
signs against under their legal identity at every release (see
[releases.md](releases.md) for the mechanics).

These guarantees describe what you get when you install and run a
**generally-available release of Eidola** — the app downloaded from
the public release channel, running on hardware whose
confidential-compute attestation verifies under the trust root
that release was built with. They do not extend to development
builds, contributor-installed test versions, or any scenario where
you have intentionally bypassed the release process (for example,
running your own server with a local override). See
[trust-root.md](trust-root.md#whats-pinned) for what's pinned and
[client.md](client.md#configuration-overrides) for how overrides
work.

## Guarantees

### G1. Inference content confidentiality

**The content of your AI inference requests and responses (prompts,
attachments, model outputs) is never available in cleartext to any
party other than the confidential-compute enclave performing the
inference.**

- The Eidola client establishes TLS to the inference upstream through
  a connection whose TLS key is bound to a hardware attestation
  report. The client refuses the connection if the attestation does
  not verify against the measurements pinned in the client build.
  See [client.md](client.md) and [upstream.md](upstream.md).
- The Eidola server does not log, store, or transmit inference
  request bodies, response bodies, or any content derived from them
  beyond what is required to route the request to the upstream
  inference enclave and the response back to the client.
- Telemetry on the inference path is limited to model name, token
  counts, status, and latency. It does not contain message content,
  account identifiers, or credential material.

**What this does not guarantee:**

- The content of *local* state on the user's device (chat history,
  drafts, cached responses) is the user's responsibility to protect.
  Eidola's client stores conversation history locally; it is no more
  or less private than any other file on the user's device.
- Network metadata (the fact that you connected to an Eidola endpoint
  at a given time, from a given IP) is visible to your network path.
  Eidola does not commit to defending against traffic analysis.

### G2. Account–inference unlinkability

**Eidola cannot link any inference request to the account that paid
for it.**

- Inference endpoints authenticate with **anonymous credentials**
  (Privacy Pass ACT tokens), not with account credentials. The server
  verifies that the credential was legitimately issued without
  learning which account it was issued to.
- Account endpoints (balance, allocation, billing) use HTTP Basic
  auth tied to an account UUID. Inference endpoints reject Basic auth
  outright. The two authentication surfaces are disjoint at the type
  level in the server.
- No identifier carried on the inference path (credential, request
  context, token) can be correlated with the account that requested
  the credential, because the credential is unlinkable by
  construction. See [server.md#unlinkability](server.md#unlinkability).

**What this guarantee actually means in practice.** Eidola persists
billing-related metadata (credential issuance records, accounting
events) because we need it to charge accounts. The way the system
is constructed, that persisted metadata does not let us answer
"which account paid for *this* inference request" or "which
inference requests came from the same account," even with full
access to our own database. See
[server.md#unlinkability](server.md#unlinkability) and
[server.md#anonymity-set](server.md#anonymity-set) for the
specific properties of the issuance protocol that make this true.

### G3. No silent code change

**Your Eidola client will not begin running code that has not been
released through the public, signed release process.**

- The client trust root is **embedded at compile time** from
  committed source files. There is no runtime API that updates the
  trust root, alters policy constants, or changes the set of
  trusted attestant fingerprints.
- Self-update (when implemented) requires a signed release
  attestation from a pinned attestant, verified locally before the
  binary is replaced.
- The client trusts exactly one server build per release. A server
  upgrade and a client upgrade ship together; clients reject server
  attestations that do not match their embedded measurement.

**What this does not guarantee:**

- A user who manually installs an unsigned build, or who installs a
  build outside the published release channel, has bypassed this
  guarantee by their own action.
- A first install — a fresh device with no prior pinned trust root
  — inherits whatever trust root is in the binary they downloaded.
  See [gaps.md](gaps.md#first-install-downgrade) for the residual
  exposure.

### G4. Verifiability

**Every claim in this document is verifiable against published
source and signed artifacts.**

- The source for the client, server, and all build inputs lives
  in a public monorepo.
- A copy of `artifact-manifest.json` — the digests of every
  released artifact and the server enclave measurement — is
  **committed in the repo root**, alongside the source it
  describes. CI re-derives the manifest from source on every PR
  and refuses to merge if the result differs. This makes
  reproducibility a *merge invariant*: any commit on `main`
  reproduces a specific manifest, and the published release is
  what that manifest names.
- Released binaries are reproducible: anyone can rebuild from
  source and confirm bit-for-bit equality with what the release
  ships.
- Each release ships with a signed manifest binding the source
  commit, the artifact digests, the enclave measurements, and a
  human attestation, all to a single transparency-log entry.
- The verifier the client uses to walk this chain is open source
  and documented in [trust-root.md](trust-root.md).

### G5. No backdoor

**The released Eidola code contains no covert channel, hidden data
exfiltration path, or surveillance mechanism not described in this
document or the user-facing documentation it references.**

- Each release attestation includes an explicit claim, signed under
  the engineer's legal identity, that they are not aware of any
  backdoor or covert mechanism in the release.
- The engineer's review is informed by a personal diff review against
  the prior release, on hardware under their exclusive physical
  control.

**What this does not guarantee:**

- An undisclosed vulnerability is not a backdoor in this sense, but
  it could be exploited by an attacker as if it were. Eidola does not
  promise the absence of bugs; it promises the absence of *intent*
  to subvert.
- A compromised hardware vendor (issuing fake attestations) or a
  compromised dependency that we did not catch in review could
  reintroduce a covert path despite this claim. See
  [threat-model.md](threat-model.md).
- **A bad-faith Eidola is in scope, not denied.** We could in
  principle sign an attestation falsely claiming no backdoor.
  Two things bound this: every claim in the attestation is
  independently verifiable against the published source (a
  reviewer can find a divergence), and the engineer signs under
  their legal identity in a public, append-only transparency
  log. The defense is verifiability plus legal accountability,
  not "trust us."

### G6. No compelled subversion without disclosure

**Eidola will not weaken these guarantees in response to legal
compulsion without that fact being inferable from the public release
record.**

- Each release attestation includes a claim, signed under the
  engineer's legal identity, that they are not currently subject
  to legal compulsion that has caused, or that requires them to
  cause, this release to weaken any published guarantee.
- A separate claim attests that the engineer is not subject to a
  gag order or other restriction that prevents them from
  truthfully making any claim in this attestation. (Gag orders
  don't compel weakening; they constrain disclosure, which is a
  different surface and needs its own claim.)
- The attestant signs from hardware under their exclusive control
  and posts the signature to a public transparency log.
- A coerced engineer who is forbidden from making the compulsion
  or gag claim truthfully must either (a) decline to sign,
  breaking the release, or (b) sign falsely and incur the
  disclosed legal exposure.
- The minimum number of independent attestants required for a
  release (`MIN_HUMAN_ATTESTATIONS`) is pinned in the **prior**
  client, so a coerced single engineer cannot lower the bar by
  shipping a release that requires fewer signatures.

**What this does not guarantee:**

- A jurisdiction able to compel *every* pinned attestant
  simultaneously, with credible secrecy, defeats this guarantee
  silently. The mitigation is distribution of attestants across
  jurisdictions and a public minimum threshold; that work is ongoing
  (see [gaps.md](gaps.md)).
- An out-of-band compromise (an attestant's signing key extracted
  from hardware without their knowledge) is a different attack
  surface, addressed by the key being hardware-bound and
  fingerprint-pinned, not by this guarantee.

## How this document evolves

Changes to this document are **append-only in spirit**: subsequent
releases can add guarantees, narrow scope where doing so does not
remove a promise, or correct ambiguous wording. They cannot remove,
weaken, or narrow a guarantee that was in effect at the prior
release. The verifier enforces this at the attestation level: a
release whose attestation lacks the `privacy_guarantees_not_weakened`
claim will fail.

If a future release needs to weaken a guarantee — for example, if a
discovered vulnerability requires a fallback that exposes
previously-shielded data — the release notes will say so explicitly,
the attestant will be unable to sign `privacy_guarantees_not_weakened`,
and the release will not pass the standard verifier. Users would
have to opt into such a release by an out-of-band mechanism.

For the technical history of how this document is hashed, pinned, and
checked, see [trust-root.md](trust-root.md).
