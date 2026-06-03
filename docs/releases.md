# Releases

A release is the *unit of trust*. When a user is running Eidola release N, what they are trusting is the entire bundle that comprises release N — client, server, build inputs, measurements, and the signatures attesting to them. This page explains how a new release becomes trustable. For the technical specification, see [trust-root.md](trust-root.md).

## What a release contains

A single release ships:

- A **client binary** for each supported platform (macOS GUI app, CLI, etc.) with a trust root embedded at compile time.
- A **server image** built reproducibly from the same source commit, whose confidential-compute measurement matches the value embedded in the client.
- A signed **artifact manifest** (`artifact-manifest.json`) recording the digests of every artifact and the enclave measurement.
- One or more **human attestations**, each signed under a pinned engineer's hardware-bound key, with their Sigstore Rekor inclusion proof.

The client carries enough information to verify all of this locally before installing an update.

## Two cryptographic systems, one transparency log

Eidola uses two signing systems in coordination. Both ride the same Sigstore Rekor transparency log via the same entry shape (`hashedrekord` v0.0.1).

| Surface | Signature | Identity binding |
|---|---|---|
| CI signs the manifest | Sigstore bundle | Fulcio keyless cert tied to the GitHub OIDC workflow identity |
| Engineer signs a release attestation | `cosign sign-blob` against a hardware-held key (YubiKey-PIV, KMS, etc.) | sha256(PKIX SubjectPublicKeyInfo) matches a fingerprint pinned in the client |

The CI side gives us "this artifact came from the release workflow on this tag." The engineer side gives us "a named human, signing under their legal identity, attests to the properties this release claims." Neither alone is sufficient; both are required.

## What the engineer attests to

Every release attestation is a structured JSON document where the engineer makes specific claims under their legal identity. The full template is at `releases/schema/attestation-templates-v1.json`; the claims include:

- `no_compelled_subversion` — the engineer is **not currently subject to any legal order, technical capability notice, or other legal compulsion** that has caused (or that requires them to cause) the release to weaken a privacy or security guarantee, introduce or preserve a backdoor or covert surveillance mechanism, or misrepresent any of these properties.
- `no_gag_preventing_attestation` — the engineer is **not subject to a gag order or nondisclosure obligation** that prevents them from truthfully making any claim in this attestation.
- `no_coercion` — the engineer has **not been threatened or coerced** by any party to alter the release, falsify any claim, or conceal any material fact about the privacy or security properties of the release.
- `signing_freely` — the engineer is signing **of their own volition**, with a key whose private component exists only on hardware under their exclusive physical control.
- `manifest_reproduced` — the engineer **personally reproduced** `artifact-manifest.json` from the source commit on hardware under their exclusive physical control, and confirmed bit-for-bit equality with CI's output.
- `diff_reviewed` — the engineer **personally inspected the source-level diff** between the prior release commit and this release commit.
- `privacy_guarantees_not_weakened` — the version of [`privacy-guarantees.md`](privacy-guarantees.md) at release time **does not weaken, narrow, or remove** any privacy or security guarantee that was stated in the prior release's version. (Required only when a prior release exists.)
- `code_delivers_guarantees` — based on the diff review, the engineer is **not aware of any change** in this release that causes the code to fail to deliver the privacy and security guarantees stated in `privacy-guarantees.md`.
- `no_known_backdoor` — based on the diff review, the engineer is **not aware of any backdoor**, covert surveillance mechanism, or undisclosed data path in this release that transmits user data in a manner not described in `privacy-guarantees.md` or the user-facing documentation it references.

These claims are recorded verbatim in the attestation document, hashed into the Sigstore Rekor transparency log, and verified by the client during self-update. The verifier re-renders each claim from a pinned template and rejects any attestation whose claim text does not match character-for-character.

## How the client verifies a release

When the user runs `eidola update`, the client:

1. **Downloads** the release index, manifest, and attestation bundles from the published source.
2. **Verifies CI's manifest signature** against the pinned Fulcio identity pattern and the embedded Sigstore trusted root.
3. **Verifies each human attestation**: cosign signature against the pinned attestant fingerprint, Sigstore bundle integrity, Rekor inclusion proof.
4. **Counts independent attestations** and fails if fewer than `MIN_HUMAN_ATTESTATIONS` (pinned in the *current* client) have verified.
5. **Re-renders each claim** from the pinned template and checks character equality with the attestation's recorded claim text.
6. **Checks continuity**: the new release's `previous_release.git_commit` must equal the currently-installed `git_commit`.
7. **Surfaces the verified prose** to the user before approving the install.

If any step fails, the update is rejected. There is no override prompt.

## Why each piece is necessary

A reader might ask: why is CI's signature not enough? Why is a single engineer's attestation not enough?

- **CI alone is not enough.** A compromise of GitHub, the OIDC flow, or the workflow definition gives an attacker the CI signature. But CI cannot mint a `cosign sign-blob` from an engineer's hardware-held key. The human attestation is the defense-in-depth against pipeline compromise.
- **A single human alone is not enough.** A single attestant is one legal target. The minimum-attestant policy (`MIN_HUMAN_ATTESTATIONS`) is pinned in the *prior* client, so a coerced engineer cannot lower the threshold by shipping a release that requires fewer signatures. **Today** `MIN_HUMAN_ATTESTATIONS` is `1` — the framework supports arbitrary M-of-N, but only one attestant key is currently provisioned. As additional attestants are onboarded, the threshold will be raised through the same release process that any other trust-constants change uses. See [gaps.md](gaps.md#single-attestant-policy).
- **Sigstore Rekor is not enough.** Sigstore proves that *a signature exists in a public log*. It does not prove that the signer was acting freely, that the source matches the release, or that the release does not weaken guarantees. The structured claims are what add that semantic content.

## Schema versions: every change is breaking

Every release document carries an integer `schema_version`. The client refuses to parse any document outside its supported set. There is no "compatible minor" tolerance. This is deliberate: silent acceptance of new fields would let a future release add a claim that older clients ignored, weakening the contract for those clients without their knowledge.

Schema rotations themselves go through a release: a new client release adds the new schema version to its supported set, while still signing releases under the old schema. Only after rolled-out clients accept both versions does the engineer start signing under the new schema.

## Release continuity

The client's update path requires that each release's claimed `previous_release.git_commit` equals the currently-installed commit. This rules out two adversary moves:

1. **Stale-release substitution.** An attacker serving a real but older release cannot route an updating client onto it; the continuity check fails.
2. **Rollback to a known-bad past release.** Even if a past release was later discovered to have a vulnerability and was superseded, the continuity check prevents an attacker from walking an updating client *backwards* into it. The only release the verifier will accept is the one whose `previous_release.git_commit` matches the installed commit.

The first-install case (no prior installed commit) is the residual gap; see [gaps.md](gaps.md#first-install-downgrade).

## For the technical specification

For exact pinning, signing-system formats, schema document shapes, and the build-input vs. build-output story (why `artifact-manifest.json` and `server-enclave.json` are separate files), see [trust-root.md](trust-root.md).

For the operational side — how to actually cut a release, rotate an attestant key, or update the Sigstore trusted root — see [`releases/README.md`](../releases/README.md).
