# Trust Root: Technical Specification

This is the spec for spot-checking rigor. For the narrative version of how releases become trustable, see [releases.md](releases.md); for the day-to-day operational procedures (rotating keys, updating Sigstore roots, etc.) see [`releases/README.md`](../releases/README.md).

The trust root is the set of values compiled into the Eidola client at build time that determines every trust decision it will make at runtime. The client's verifier consults these values when checking a server attestation, when verifying a release, and when accepting or rejecting a self-update.

## What's pinned

Generated into the client at compile time by `crates/eidola-app-core/build.rs`, surfaced via [`eidola_app_core::trust_root`](../crates/eidola-app-core/src/trust_root.rs):

| Constant | Source | Purpose |
| --- | --- | --- |
| `SERVER_URL` | derived from `releases/trust/server-enclave.json` | `gateway-<hash>.eidola.containers.tinfoil.sh`, where `<hash>` ties the URL to a server measurement |
| `SERVER_SNP_MEASUREMENT` | `releases/trust/server-enclave.json` → `snp_measurement` | SEV-SNP launch measurement of the paired server enclave |
| `SERVER_TDX_RTMR1` / `SERVER_TDX_RTMR2` | `releases/trust/server-enclave.json` → `tdx_measurement` | TDX runtime measurements of the paired server enclave |
| `TRUSTED_ATTESTANT_FINGERPRINTS` | `releases/trust/trust-constants.json` | `sha256(PKIX SubjectPublicKeyInfo DER)` in hex, for each authorized human-attestant key |
| `MIN_HUMAN_ATTESTATIONS` | `releases/trust/trust-constants.json` | Minimum independently-verified human attestations a release must carry. Pinned here, not in `release.json`, so a forged index cannot lower it |
| `EXPECTED_CI_IDENTITY_PATTERN` | `releases/trust/trust-constants.json` | Fulcio cert SAN pattern the release-signing workflow's OIDC identity must match |
| `EXPECTED_CI_ISSUER` | `releases/trust/trust-constants.json` | OIDC issuer (`https://token.actions.githubusercontent.com`) |
| `SUPPORTED_RELEASE_SCHEMA_VERSIONS` | `releases/trust/trust-constants.json` | Integer `schema_version` values of `release.json` this client will parse |
| `SUPPORTED_ATTESTATION_SCHEMA_VERSIONS` | `releases/trust/trust-constants.json` | Integer `schema_version` values of `attestation.json` this client will parse |
| `UPDATE_DISCOVERY_URL` | `releases/trust/trust-constants.json` | Where to look for the next release (GitHub releases API) |
| `ATTESTATION_TEMPLATES_JSON` | `releases/schema/attestation-templates-v1.json` | Pinned claim templates the verifier re-renders during equality checks |
| `SIGSTORE_TRUSTED_ROOT_JSON` | `releases/trust/sigstore-trusted-root.json` | Pinned Sigstore tlog / Fulcio / CT log keys with validity windows |

`config.toml` overrides (`base_url`, `trusted_measurements`) take precedence at runtime — set them to point a build at a local server or alternate enclave. With overrides unset, the pinned values are what get used.

## Why the enclave block lives in its own file

The enclave block (`snp_measurement`, `tdx_measurement.rtmr1`, `tdx_measurement.rtmr2`, `cmdline`) lives in `releases/trust/server-enclave.json`, separate from `artifact-manifest.json`. The reason is build reproducibility.

`artifact-manifest.json` records the eidola-cli OCI digest and the eidola-cli-macos-universal narHash among other artifacts. If the cli build COPYed (Docker) or filtered-in (Nix) the manifest as a build input, every regeneration of the manifest would also be an input to the build it's describing — a self-reference that produces a different digest on every run instead of converging.

`server-enclave.json` is the minimum slice of the manifest the cli build needs, so it can be COPYed without dragging the cli's own digest into the build context. CI re-asserts the consistency: `scripts/artifact-manifest.sh verify-full` recomputes the enclave block from `tinfoil-config.yml` and rejects the build if either `server-enclave.json` or `artifact-manifest.json`'s `enclave` field disagrees with it.

## Schema versions: explicit and breaking

Each release document carries an explicit `schema_version` (positive integer). The supported version sets are pinned in `trust-constants.json` so the verifier rejects any document outside the set; bumping a schema is itself a release-gated trust event.

**There is no semver-style "backwards-compatible minor bump."** Each integer denotes a distinct, all-or-nothing shape that a verifier either understands fully or refuses outright. This deliberately avoids the security weakening that would happen if old clients silently tolerated a new claim or field without enforcing it.

(Product versions — `release.version`, `release.previous_release.version` — remain semver strings, since those *do* benefit from ordering and matching the Rust/Cargo ecosystem.)

Each document's shape is owned by the Rust `serde` types shared between the release-tool and the verifier — there is no separately-maintained JSON Schema file. Drift between signing and verifying is impossible because both sides deserialize from the same struct definitions.

| Document | Shape (source of truth) | Notes |
| --- | --- | --- |
| `artifact-manifest.json` | format owned by `scripts/artifact-manifest.sh` | `schema_version: 1`. Records OCI digests, the macOS narHash, and a denormalized copy of the enclave block. Signed by CI as a Sigstore bundle (Fulcio keyless, OIDC). |
| `releases/trust/server-enclave.json` | format owned by `scripts/artifact-manifest.sh`, consumed as raw JSON in `eidola-app-core/build.rs` | `schema_version: 1`. Holds just the enclave block (snp/tdx measurement + cmdline) so the cli build doesn't drag its own digest into its build context. |
| `releases/trust/tinfoil-enclaves.json` | format owned by `.github/workflows/update-measurements.yml`, consumed as raw JSON in `eidola-server/build.rs` | `schema_version: 1`. Allowed upstream Tinfoil inference-enclave measurements the server's outbound verifier accepts. One entry per Tinfoil release, with provenance metadata (built\_at, artifact digest, Rekor log index); the workflow keeps the most recent two for rolling deploys. |
| `release.json` | `eidola_attestation::ReleaseIndex` — `crates/eidola-attestation/src/trust_shapes.rs` | Unsigned URL-only index; cross-checked via referenced documents (see caveat below) |
| `attestation.json` | `updater::human_attestation::AttestationProse` — `crates/eidola-app-core/src/updater/human_attestation.rs` | Signed by the attestant via `cosign sign-blob` (local PEM, PKCS#11 URI, or any KMS URI cosign supports), logged to Rekor as a `hashedrekord` v0.0.1 entry with a PKIX SubjectPublicKeyInfo (ECDSA-P256/P384 or Ed25519) in `signature.publicKey.content` |
| `trust-constants.json` | `eidola_attestation::TrustConstants` — `crates/eidola-attestation/src/trust_shapes.rs` | Pinned trust values baked into the verifier at build time |
| Templates | `releases/schema/attestation-templates-v1.json` (data, not a schema) | Pinned claim templates the verifier re-renders during equality checks |

## `release.json` is a pure URL index — no hashes, no policy

`release.json` is an *index*: URLs only. Hashes (the manifest's, each attestation's, each Sigstore bundle's) live in the Sigstore bundles themselves — the bundle signs the hash of what it certifies. The verifier downloads each file, computes its hash, and asks the bundle whether that hash was signed. Putting expected hashes in `release.json` would just add more fields to keep in sync without strengthening any binding.

For the same reason, `release.json` does **not** carry `expected_identity` / `expected_issuer` / `rekor_log_index` / Rekor key material — those are pinned in the client's embedded trust root or inherent in the Sigstore bundle. Echoing them in `release.json` would let an adversary downgrade trust by handing the client a tampered index.

Policy values (minimum-attestation threshold, allowed identities, allowed schema versions) follow the same rule. They live exclusively in the *embedded* trust root of the previous client (see [`MIN_HUMAN_ATTESTATIONS`](#whats-pinned) and friends). Otherwise an attacker who produced a single forged attestation could also forge a `release.json` that lowered the threshold to 1.

## Signing systems: split by surface

Two cryptographic systems carry the trust chain, dispatched by the verifier based on which document is being verified:

| Surface | Signature | Identity binding | Transparency |
| --- | --- | --- | --- |
| CI signs `artifact-manifest.json` | Sigstore bundle | Fulcio keyless cert — OIDC identity matches `EXPECTED_CI_IDENTITY_PATTERN` | Rekor inclusion proof embedded in the bundle |
| Engineer signs `attestation-<id>.json` | `cosign sign-blob --key <ref>` — `<ref>` is a local PEM, PKCS#11 URI (YubiKey-PIV / SmartCard), or any cosign KMS URI | `sha256(PKIX SubjectPublicKeyInfo DER)` matches `TRUSTED_ATTESTANT_FINGERPRINTS` | Posted to Rekor as a `hashedrekord` v0.0.1 entry (the entry kind that survives Rekor v2 — `rekord` and SSH PKI are being retired); inclusion proof saved in `attestation-<id>.bundle.json` |
| Engineer signs the git tag | SSH signature (separate SSH key in the engineer's git config) | OpenSSH wire-format SHA-256 fingerprint — *not* the same as the cosign SPKI fingerprint above, even if the underlying private key is shared | Implicit via the repo |

The CI side uses Sigstore because Fulcio's keyless OIDC binding is *the* mechanism that makes "this signature came from the release workflow on a specific tag" cryptographically meaningful — no other system offers that.

The engineer side uses `cosign sign-blob` with a hardware-held key because:

- `cosign` is the canonical Sigstore client; its `hashedrekord` v0.0.1 output is the one Rekor entry shape that's guaranteed to survive Rekor v2 (the upcoming tile-based rewrite drops `rekord` and all non-x509 PKIs including SSH).
- `--key` accepts a PEM, a PKCS#11 URI, or any cosign-supported KMS URI, so the hardware-backing options range from YubiKey-PIV (the recommended path) through HSMs and cloud KMS backends, without us having to wire each one specifically. The underlying key must be ECDSA-P256, ECDSA-P384, or Ed25519 — the updater's `verify_blob_signature_with_spki` dispatches on the SPKI's AlgorithmIdentifier OID and rejects everything else (RSA, ECDSA-P521, …). `release-tool attest` cross-checks the algorithm before signing via the same classifier (`updater::human_attestation::classify_attestant_spki_algorithm`), so a release can't be published in a shape the updater would reject.
- The verifier shares the bulk of its code path with the CI side: both parse the same `hashedrekord` body shape and the same Sigstore Bundle v0.3 wrapper; only the trust-pinning step differs (fingerprint for human, Fulcio chain + OIDC identity for CI).

Both paths ride the same Sigstore Rekor transparency log via the same entry kind (`hashedrekord` v0.0.1). On the CI side the public key in the body is a Fulcio leaf certificate; on the human side it's a PKIX SubjectPublicKeyInfo (the attestant's own key). The verifier shares its body parsing, Rekor SET signature verification, and Merkle inclusion-proof verification between the two paths — see [`crates/eidola-app-core/src/updater/rekor_verify.rs`](../crates/eidola-app-core/src/updater/rekor_verify.rs).

## Unsigned `release.json` — known caveat, with mitigation

`release.json` is published unsigned on the GitHub release. The verification chain holds because:

- Each artifact and attestation it points to is independently signed.
- The CI Sigstore bundle binds the manifest hash, which is what's actually downloaded.
- Each attestation's content is cross-checked against `release.json`'s `version`, `git_commit`, and `previous_release.git_commit` — so an adversary can't substitute a stale attestation for a different release.
- Trusted attestant fingerprints are checked against the client-embedded set, not anything in `release.json`.

The protection that *doesn't* hold without a signed `release.json` is **first-install downgrade**. A fresh client with no prior installed version has no continuity check to anchor against, so an adversary serving an internally-consistent older `release.json` could route the client onto a real-but-stale release. See [gaps.md](gaps.md#first-install-downgrade) for ongoing mitigations.

## Where each piece lives

Almost everything under `releases/` is a **build input** — pinned data the client and server compile against (the one exception is `trust/attestant-provenance/`, informational auditor-facing evidence that no build or client reads). `artifact-manifest.json` at the repo root is the **build output** — a record of what was actually produced, signed by CI. They live in different places on purpose: files under `releases/` are bulk-copied/filtered into builds as a unit, while `artifact-manifest.json` is deliberately kept out of every build context to prevent self-reference cycles (it records the eidola-cli OCI digest and macOS narHash that the cli build would otherwise see in its own input).

```text
releases/
  README.md                             # contributor README: per-file detail + rotation procedures
  schema/
    attestation-templates-v1.json       # pinned claim templates
  trust/
    trust-constants.json                # non-derivable trust values (input)
    sigstore-trusted-root.json          # upstream Sigstore TrustedRoot snapshot (input)
    server-enclave.json                 # paired-server enclave measurement (input — projection of artifact-manifest.json's enclave block, materialized as its own file so the cli build context can COPY it without dragging the manifest in)
    tinfoil-enclaves.json               # allowed upstream Tinfoil inference-enclave measurements (input — server's build.rs reads this)
    attestant-provenance/               # informational hardware-attestation evidence for pinned attestant keys (NOT a build input — no build.rs or client reads it; auditor-facing only)
artifact-manifest.json                  # full deployment record (output, signed by CI)
crates/eidola-app-core/
  build.rs                              # generator: server-enclave.json + trust-constants.json + … → trust_root.gen.rs
  src/trust_root.rs                     # exposes the generated constants
crates/eidola-server/
  build.rs                              # generator: tinfoil-enclaves.json → measurements.gen.rs
  src/measurements.rs                   # exposes the generated ALLOWED static
```

The generator (`build.rs`) reads `releases/trust/server-enclave.json` — never the per-artifact digests. The chain that invalidates the pin: server source changes → server image digest changes → `tinfoil-config.yml` changes → kernel cmdline changes → enclave measurement changes → `server-enclave.json` changes → client rebuilds with the new pin. Because the client build context never reads `artifact-manifest.json`, regenerating the manifest after a client build doesn't trigger another client rebuild, so `just update-manifest` reaches a fixed point in a single run.

## Known gaps

The verifier has several deferred capabilities. They are catalogued in one place so a reader can see what is not yet defended against without needing to grep source comments. See [gaps.md](gaps.md).

## Acknowledgements

`sigstore-trusted-root.json` is a verbatim copy of the upstream Sigstore TrustedRoot. The Sigstore verification approach (CI side) is adapted from [tinfoil-rs](https://github.com/tinfoilsh/tinfoil-rs), which in turn adapts verification modules from [sigstore-rs](https://github.com/sigstore/sigstore-rs) (Apache 2.0). The human attestation path runs through the same Sigstore Bundle v0.3

- `hashedrekord` v0.0.1 machinery as the CI path; the cosign-side
signing flow comes from [cosign](https://github.com/sigstore/cosign) (Apache 2.0).
