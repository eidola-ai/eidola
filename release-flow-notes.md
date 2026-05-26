# Eidola Release Verification — First Iteration

**Status:** Initial design, schema v1.0.0
**Scope:** Mechanical release flow, client verification flow, file schemas, and pinned attestation templates.

## Design summary

Eidola releases are bundles of reproducibly-built artifacts whose authenticity rests on two cryptographic signatures plus one structured human attestation:

1. **CI signature** (Fulcio keyless via GitHub Actions OIDC) over the canonical `artifact-manifest.json`.
2. **Human attestation** (hardware-backed key, hashedrekord) over a typed JSON document that includes first-person sworn statements about the signer's state, the review they personally performed, and what they are aware of.

Both signatures are logged to public rekor. The client trust root is a small set of values embedded in the binary at build time. Verification is purely cryptographic plus exact string and structural equality — no natural-language parsing on the verification path.

The attestation schema is the legal protection: each claim is a first-person sworn statement that a coerced engineer would have to either omit (failing verification) or assert falsely (perjury). Schema and statement templates are pinned by version in the client; weakening requires an attested schema change.

## What is and isn't in v1

**In:** Reproducible builds (already established), `artifact-manifest.json` CI invariant (already established), Fulcio CI signing (already established), hardware-backed human attestation, structured attestation JSON with pinned templates, `release.json` discovery manifest, rekor log indices explicit in `release.json`, schema versioning on every document, explicit CI identity pinning, server URL and TEE measurements embedded in client (lockstep release with server).

**Deferred (planned, not blocking v1):**

| Deferred item | Why deferred | What's at risk in the meantime |
|---|---|---|
| Log checkpoint + consistency proofs | Witness ecosystem still maturing | Rekor's inclusion proof is the floor; a rekor-side rewrite would be detectable by third-party monitors |
| Freshness anchors (Bitcoin block etc.) | Implementation cost | Stalled-release attack; mitigated socially by a public cadence commitment |
| M-of-N attestation | Solo project today | Single-attestant pressure point; schema is list-shaped so adding attestants is non-breaking |
| AI-driven diff scan in client | Independent feature | Reviewer (human or otherwise) must read diffs unaided |
| TUF-managed trust root for our keys | Cost vs benefit at current scale | Trust root updates ship as embedded-binary changes; rotation requires a release signed under the current root |
| Independent third-party builds (step 1.3 in original plan) | No suitable third party yet | No external reproduction crosscheck |
| Separate channels (stable/beta) | Single channel sufficient | N/A |

## Repository layout

```
/artifact-manifest.json           # existing; produced reproducibly, CI-enforced invariant
/releases/
  schema/
    release-v1.0.0.schema.json    # JSON Schema for release.json
    attestation-v1.0.0.schema.json # JSON Schema for attestation.json
    attestation-templates-v1.0.0.json # pinned statement templates
  v0.5.0/
    attestation-mike-prince.json  # committed pre-release; signed post-merge
```

Release assets on GitHub (not committed to git for v1):
```
release.json
artifact-manifest.json
artifact-manifest.json.sigstore           # CI bundle
attestation-mike-prince.json              # exact bytes of committed file
attestation-mike-prince.json.sigstore     # human bundle
<all built artifacts and their checksums>
```

## Publish flow

1. **Open release PR** against `main`:
   - Bump versions in source
   - Add `releases/vX.Y.Z/attestation-<attestant-id>.json` with all fields populated *except* placeholder `null` values for any field that depends on the merge commit hash. (None today, since the attestation references `git_commit` of the merged state — see step 2.)
   - Review focuses on the attestation prose: every claim matches the pinned template, every substitution value is correct.
2. **Merge `--ff-only`** so the PR's HEAD commit becomes the release commit. Substitute the now-known `git_commit` into the attestation if it wasn't pinned at PR time. (Alternative: reference the tree hash instead, which is known at PR time. v1 uses commit hash for human familiarity.)
3. **Sign and push tag** `vX.Y.Z` with hardware-backed git signing key.
4. **CI builds and signs** (automated, triggered by tag):
   - Reproducible build runs, produces `artifact-manifest.json`
   - CI verifies the manifest matches the invariant baked into `main`
   - `cosign sign-blob artifact-manifest.json --bundle artifact-manifest.json.sigstore --yes` using Fulcio keyless via the workflow's OIDC token
   - All artifacts and the bundle uploaded to GitHub release as assets
   - Rekor log index extracted from bundle, stored for step 6
5. **Human reproduction and attestation** (off-CI, on the attestant's hardware):
   - Fetch the CI-built `artifact-manifest.json`
   - Run the reproducible build locally on hardware under exclusive physical control
   - Confirm bit-for-bit equality with the CI manifest
   - Review the diff against the previous release
   - Sign the committed `attestation-<attestant-id>.json`: `cosign sign-blob attestation-mike-prince.json --key <yubikey-pkcs11-ref> --bundle attestation-mike-prince.json.sigstore --yes`
   - Upload `attestation-mike-prince.json` (verbatim from git) and `attestation-mike-prince.json.sigstore` as release assets
6. **Finalize**:
   - Generate `release.json` with all rekor log indices and bundle hashes filled in
   - Upload `release.json` as release asset
   - Mark the GitHub release as "latest"

The git copy of the attestation and the release-asset copy must be byte-identical; CI can enforce this as a release-time invariant.

## Verification flow

Client maintains: last-verified version, last-verified `git_commit`, last-installed artifact hashes.

1. **Discover.** Fetch `https://api.github.com/repos/eidola-ai/eidola/releases/latest` (with fallback to `https://raw.githubusercontent.com/eidola-ai/eidola/main/releases/latest/release.json` if the API is unreachable). Read `release.json` from assets. Network transport rides the same Tor configuration as the rest of the client.
2. **Schema.** Reject if `release.json.schema_version` is not in the client's supported set. Reject if required fields are missing or malformed.
3. **Continuity.** `release.version` is strictly greater than installed version per semver. `release.previous_release.git_commit`, when present, matches the client's last-installed `git_commit` or is reachable via fast-forward through GitHub's commits API.
4. **Fetch.** Retrieve referenced assets: `artifact-manifest.json`, both `.sigstore` bundles, all attestation JSON files. Verify each asset's SHA-256 matches the hash claimed in `release.json`.
5. **Verify CI signature.** Parse the CI sigstore bundle. Validate the Fulcio certificate chain against the Sigstore TUF root. Confirm:
   - Certificate SAN matches the embedded `expected_ci_identity_pattern` (e.g. `https://github.com/eidola-ai/eidola/.github/workflows/release.yml@refs/tags/v*`)
   - Certificate issuer matches the embedded `expected_ci_issuer` (`https://token.actions.githubusercontent.com`)
   - Signed payload hash equals `release.artifacts.manifest_sha256`
   - Bundle's rekor inclusion proof is valid
   - Bundle's rekor log index equals `release.signatures.ci.rekor_log_index`
6. **Verify human attestations.** For each attestation in `release.signatures.human_attestations`:
   - Parse the sigstore bundle
   - Extract the signing public key; SHA-256 hash it; confirm match with `key_fingerprint_sha256` in `release.json`
   - Confirm the fingerprint is in the embedded `trusted_attestant_fingerprints` set
   - Verify the signature over the attestation JSON bytes
   - Verify the rekor inclusion proof and log index match `release.json`
7. **Validate attestation content.** For each attestation:
   - `schema_version` is in the client's supported set
   - `release_version`, `git_commit`, `artifact_manifest_sha256` match `release.json`
   - `attestant.key_fingerprint_sha256` matches the key used to sign
   - `attestant_statement` exactly equals the schema's pinned `attestant_statement_template` with `{name}`, `{key_fingerprint_sha256}`, and `{jurisdiction}` substituted from `attestant` fields
   - For each required claim in the attestation template manifest, the claim is present, and `claim.statement` equals `template.format(**claim.fields)`, and each substituted field value matches the corresponding `release.json` field where applicable (see cross-checks in template manifest)
8. **Policy.** At least `policy.min_ci_signatures` CI signatures verified (v1: 1). At least `policy.min_human_attestations` human attestations verified (v1: 1).
9. **Manifest and artifacts.** Fetch `artifact-manifest.json`. Verify hash matches. Fetch artifacts listed in the manifest. Verify each hash.
10. **Present.** Show the user: version, git commit, attestation text verbatim (from `attestant_statement` + each claim's `statement`), and a one-line summary of verification status. Await approval.
11. **Install.** On approval, install/switch. Persist new `version` and `git_commit` as the last-verified state.

All verification steps must pass. Any failure → reject the update, surface the reason in user-facing diagnostics, do not retry silently.

## JSON Schema: `release.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://eidola.ai/schema/release-v1.0.0.json",
  "title": "Eidola Release Manifest",
  "type": "object",
  "required": [
    "schema_version", "version", "git_commit", "released_at",
    "artifacts", "signatures"
  ],
  "additionalProperties": false,
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "1.0.0"
    },
    "version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?$"
    },
    "git_commit": {
      "type": "string",
      "pattern": "^[0-9a-f]{40}$"
    },
    "git_tag": {
      "type": "string",
      "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?$"
    },
    "released_at": {
      "type": "string",
      "format": "date-time"
    },
    "previous_release": {
      "type": "object",
      "required": ["version", "git_commit"],
      "additionalProperties": false,
      "properties": {
        "version": { "type": "string" },
        "git_commit": { "type": "string", "pattern": "^[0-9a-f]{40}$" }
      }
    },
    "artifacts": {
      "type": "object",
      "required": ["manifest_sha256", "manifest_url"],
      "additionalProperties": false,
      "properties": {
        "manifest_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "manifest_url": { "type": "string", "format": "uri" }
      }
    },
    "signatures": {
      "type": "object",
      "required": ["ci", "human_attestations", "policy"],
      "additionalProperties": false,
      "properties": {
        "ci": {
          "type": "object",
          "required": [
            "type", "bundle_url", "bundle_sha256",
            "rekor_log_index", "expected_identity",
            "expected_issuer", "signed_payload_sha256"
          ],
          "additionalProperties": false,
          "properties": {
            "type": { "type": "string", "const": "sigstore-fulcio" },
            "bundle_url": { "type": "string", "format": "uri" },
            "bundle_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
            "rekor_log_index": { "type": "integer", "minimum": 0 },
            "expected_identity": { "type": "string", "format": "uri" },
            "expected_issuer": { "type": "string", "format": "uri" },
            "signed_payload_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
          }
        },
        "human_attestations": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": [
              "attestant_id", "attestation_url", "attestation_sha256",
              "bundle_url", "bundle_sha256",
              "rekor_log_index", "key_fingerprint_sha256"
            ],
            "additionalProperties": false,
            "properties": {
              "attestant_id": { "type": "string", "pattern": "^[a-z0-9-]+$" },
              "attestation_url": { "type": "string", "format": "uri" },
              "attestation_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
              "bundle_url": { "type": "string", "format": "uri" },
              "bundle_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
              "rekor_log_index": { "type": "integer", "minimum": 0 },
              "key_fingerprint_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
            }
          }
        },
        "policy": {
          "type": "object",
          "required": ["min_ci_signatures", "min_human_attestations"],
          "additionalProperties": false,
          "properties": {
            "min_ci_signatures": { "type": "integer", "minimum": 1 },
            "min_human_attestations": { "type": "integer", "minimum": 1 }
          }
        }
      }
    }
  }
}
```

## JSON Schema: `attestation.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://eidola.ai/schema/attestation-v1.0.0.json",
  "title": "Eidola Human Release Attestation",
  "type": "object",
  "required": [
    "schema_version", "release_version", "git_commit",
    "artifact_manifest_sha256", "attestant",
    "attested_at", "attestant_statement", "claims"
  ],
  "additionalProperties": false,
  "properties": {
    "schema_version": { "type": "string", "const": "1.0.0" },
    "release_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?$"
    },
    "git_commit": { "type": "string", "pattern": "^[0-9a-f]{40}$" },
    "artifact_manifest_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
    "attestant": {
      "type": "object",
      "required": ["id", "name", "key_fingerprint_sha256", "jurisdiction"],
      "additionalProperties": false,
      "properties": {
        "id": { "type": "string", "pattern": "^[a-z0-9-]+$" },
        "name": { "type": "string", "minLength": 1 },
        "key_fingerprint_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "jurisdiction": { "type": "string", "minLength": 1 }
      }
    },
    "attested_at": { "type": "string", "format": "date-time" },
    "previous_release_git_commit": {
      "type": "string",
      "pattern": "^[0-9a-f]{40}$"
    },
    "privacy_guarantees_doc_sha256": {
      "type": "string",
      "pattern": "^[0-9a-f]{64}$"
    },
    "attestant_statement": { "type": "string", "minLength": 1 },
    "claims": {
      "type": "object",
      "required": ["state_of_self", "process_of_review", "bounded_knowledge"],
      "additionalProperties": false,
      "properties": {
        "state_of_self": {
          "type": "object",
          "required": ["no_compulsion", "no_coercion", "signing_freely"],
          "additionalProperties": false,
          "properties": {
            "no_compulsion": { "$ref": "#/$defs/claim_no_fields" },
            "no_coercion":   { "$ref": "#/$defs/claim_no_fields" },
            "signing_freely":{ "$ref": "#/$defs/claim_no_fields" }
          }
        },
        "process_of_review": {
          "type": "object",
          "required": ["manifest_reproduced", "diff_reviewed"],
          "additionalProperties": false,
          "properties": {
            "manifest_reproduced": { "$ref": "#/$defs/claim_with_fields" },
            "diff_reviewed":        { "$ref": "#/$defs/claim_with_fields" }
          }
        },
        "bounded_knowledge": {
          "type": "object",
          "required": ["no_known_privacy_weakening", "no_known_backdoor"],
          "additionalProperties": false,
          "properties": {
            "no_known_privacy_weakening": { "$ref": "#/$defs/claim_with_fields" },
            "no_known_backdoor":           { "$ref": "#/$defs/claim_no_fields" }
          }
        }
      }
    }
  },
  "$defs": {
    "claim_no_fields": {
      "type": "object",
      "required": ["statement"],
      "additionalProperties": false,
      "properties": {
        "statement": { "type": "string", "minLength": 1 }
      }
    },
    "claim_with_fields": {
      "type": "object",
      "required": ["statement", "fields"],
      "additionalProperties": false,
      "properties": {
        "statement": { "type": "string", "minLength": 1 },
        "fields": {
          "type": "object",
          "additionalProperties": { "type": "string" }
        }
      }
    }
  }
}
```

## Attestation templates (`attestation-templates-v1.0.0.json`)

This file is the **single source of truth** for both the signing script and the client verifier. The signing script renders statements from templates; the client verifier reconstructs expected statements from the same templates and compares. Both sides MUST use this file exactly as committed.

```json
{
  "schema_version": "1.0.0",
  "attestant_statement_template": {
    "template": "I, {name}, holder of the private key with fingerprint sha256:{key_fingerprint_sha256}, affirm under penalty of perjury under the laws of the jurisdiction of {jurisdiction} that the following claims are true and correct to the best of my knowledge.",
    "substitutions": ["name", "key_fingerprint_sha256", "jurisdiction"],
    "substitution_sources": {
      "name": "attestation.attestant.name",
      "key_fingerprint_sha256": "attestation.attestant.key_fingerprint_sha256",
      "jurisdiction": "attestation.attestant.jurisdiction"
    }
  },
  "claims": {
    "state_of_self.no_compulsion": {
      "template": "I am not currently subject to any legal order, gag, technical capability notice, or other compulsion related to this release or to Eidola generally.",
      "substitutions": [],
      "substitution_sources": {},
      "cross_checks": {}
    },
    "state_of_self.no_coercion": {
      "template": "I have not been threatened or coerced by any party in connection with this release.",
      "substitutions": [],
      "substitution_sources": {},
      "cross_checks": {}
    },
    "state_of_self.signing_freely": {
      "template": "I am signing this attestation of my own volition, on hardware under my exclusive physical control.",
      "substitutions": [],
      "substitution_sources": {},
      "cross_checks": {}
    },
    "process_of_review.manifest_reproduced": {
      "template": "I have personally reproduced artifact-manifest.json (sha256 {artifact_manifest_sha256}) from git commit {git_commit} on hardware under my exclusive physical control, and confirmed bit-for-bit equality with the CI-produced manifest.",
      "substitutions": ["artifact_manifest_sha256", "git_commit"],
      "substitution_sources": {
        "artifact_manifest_sha256": "attestation.artifact_manifest_sha256",
        "git_commit": "attestation.git_commit"
      },
      "cross_checks": {
        "artifact_manifest_sha256": "release.artifacts.manifest_sha256",
        "git_commit": "release.git_commit"
      }
    },
    "process_of_review.diff_reviewed": {
      "template": "I have personally reviewed the diff between git commit {previous_release_git_commit} (the prior release) and git commit {git_commit} (this release).",
      "substitutions": ["previous_release_git_commit", "git_commit"],
      "substitution_sources": {
        "previous_release_git_commit": "attestation.previous_release_git_commit",
        "git_commit": "attestation.git_commit"
      },
      "cross_checks": {
        "previous_release_git_commit": "release.previous_release.git_commit",
        "git_commit": "release.git_commit"
      }
    },
    "bounded_knowledge.no_known_privacy_weakening": {
      "template": "Based on the review described above, I am not aware of any change in this release that weakens the privacy guarantees stated in PRIVACY-GUARANTEES.md (sha256 {privacy_guarantees_doc_sha256}) as compared with the prior release.",
      "substitutions": ["privacy_guarantees_doc_sha256"],
      "substitution_sources": {
        "privacy_guarantees_doc_sha256": "attestation.privacy_guarantees_doc_sha256"
      },
      "cross_checks": {}
    },
    "bounded_knowledge.no_known_backdoor": {
      "template": "Based on the review described above, I am not aware of any backdoor, covert surveillance mechanism, or undisclosed data exfiltration path in the code that comprises this release.",
      "substitutions": [],
      "substitution_sources": {},
      "cross_checks": {}
    }
  }
}
```

### Template processing rules

For each entry in `attestant_statement_template` and `claims`:

1. **Render** (signing side, also recomputed by the verifier): substitute each placeholder `{key}` in `template` with the value found at `substitution_sources[key]`. Substitution is literal string replacement; no escaping, no formatting transforms.
2. **Equality check** (verifier): `expected = render(template, substitution_sources)`. Reject unless `claim.statement == expected`.
3. **Field presence check** (verifier): for each `key` in `substitutions`, the claim's `fields` object must contain `key` with the exact value used in rendering. (`claim_no_fields` claims have no `fields` object.)
4. **Cross-check** (verifier): for each `(field, release_path)` in `cross_checks`, the substitution value used must equal the value at `release_path` in `release.json`.

Path notation: `attestation.x.y` refers to `attestation_json["x"]["y"]`; `release.x.y` refers to `release_json["x"]["y"]`. Resolver is a simple dotted-path lookup; no JSONPath, no expressions.

### Adding, modifying, or removing a claim

Any change to this file requires a new `schema_version`. Clients pin the schema versions they accept; a release using an unrecognized schema is rejected. Deprecating a claim requires shipping clients that accept both old and new schemas before any release uses only the new schema. This is the mechanism that prevents a coerced release from silently weakening required claims.

## Worked example: `attestation-mike-prince.json` (illustrative)

```json
{
  "schema_version": "1.0.0",
  "release_version": "0.5.0",
  "git_commit": "9c3a0000000000000000000000000000000000ab",
  "artifact_manifest_sha256": "f4d1000000000000000000000000000000000000000000000000000000000000",
  "attestant": {
    "id": "mike-prince",
    "name": "Mike Prince",
    "key_fingerprint_sha256": "7e3b000000000000000000000000000000000000000000000000000000000000",
    "jurisdiction": "the State of California, United States"
  },
  "attested_at": "2026-05-20T17:28:00Z",
  "previous_release_git_commit": "5e1f0000000000000000000000000000000000cd",
  "privacy_guarantees_doc_sha256": "b2c4000000000000000000000000000000000000000000000000000000000000",
  "attestant_statement": "I, Mike Prince, holder of the private key with fingerprint sha256:7e3b000000000000000000000000000000000000000000000000000000000000, affirm under penalty of perjury under the laws of the jurisdiction of the State of California, United States that the following claims are true and correct to the best of my knowledge.",
  "claims": {
    "state_of_self": {
      "no_compulsion": {
        "statement": "I am not currently subject to any legal order, gag, technical capability notice, or other compulsion related to this release or to Eidola generally."
      },
      "no_coercion": {
        "statement": "I have not been threatened or coerced by any party in connection with this release."
      },
      "signing_freely": {
        "statement": "I am signing this attestation of my own volition, on hardware under my exclusive physical control."
      }
    },
    "process_of_review": {
      "manifest_reproduced": {
        "statement": "I have personally reproduced artifact-manifest.json (sha256 f4d1000000000000000000000000000000000000000000000000000000000000) from git commit 9c3a0000000000000000000000000000000000ab on hardware under my exclusive physical control, and confirmed bit-for-bit equality with the CI-produced manifest.",
        "fields": {
          "artifact_manifest_sha256": "f4d1000000000000000000000000000000000000000000000000000000000000",
          "git_commit": "9c3a0000000000000000000000000000000000ab"
        }
      },
      "diff_reviewed": {
        "statement": "I have personally reviewed the diff between git commit 5e1f0000000000000000000000000000000000cd (the prior release) and git commit 9c3a0000000000000000000000000000000000ab (this release).",
        "fields": {
          "previous_release_git_commit": "5e1f0000000000000000000000000000000000cd",
          "git_commit": "9c3a0000000000000000000000000000000000ab"
        }
      }
    },
    "bounded_knowledge": {
      "no_known_privacy_weakening": {
        "statement": "Based on the review described above, I am not aware of any change in this release that weakens the privacy guarantees stated in PRIVACY-GUARANTEES.md (sha256 b2c4000000000000000000000000000000000000000000000000000000000000) as compared with the prior release.",
        "fields": {
          "privacy_guarantees_doc_sha256": "b2c4000000000000000000000000000000000000000000000000000000000000"
        }
      },
      "no_known_backdoor": {
        "statement": "Based on the review described above, I am not aware of any backdoor, covert surveillance mechanism, or undisclosed data exfiltration path in the code that comprises this release."
      }
    }
  }
}
```

## Client trust root

Embedded in the client binary at build time. Changes require a release signed under the current trust root.

| Field | Example value | Purpose |
|---|---|---|
| `update_discovery_url` | `https://api.github.com/repos/eidola-ai/eidola/releases/latest` | Where to look for new releases |
| `update_discovery_fallback_url` | `https://raw.githubusercontent.com/eidola-ai/eidola/main/releases/latest/release.json` | Fallback if API unreachable |
| `rekor_url` | `https://rekor.sigstore.dev` | For inclusion proof verification |
| `sigstore_tuf_root_url` | `https://tuf-repo-cdn.sigstore.dev` | Source for Sigstore TUF metadata |
| `expected_ci_identity_pattern` | `https://github.com/eidola-ai/eidola/.github/workflows/release.yml@refs/tags/v*` | Pins which workflow can produce CI signatures |
| `expected_ci_issuer` | `https://token.actions.githubusercontent.com` | Pins the OIDC issuer |
| `trusted_attestant_fingerprints` | `["7e3b…"]` | SHA-256 fingerprints of authorized human attestant pubkeys |
| `supported_release_schema_versions` | `["1.0.0"]` | Acceptable `release.json` schema versions |
| `supported_attestation_schema_versions` | `["1.0.0"]` | Acceptable `attestation.json` schema versions |
| `attestation_template_manifest` | (full contents of `attestation-templates-v1.0.0.json`) | The pinned templates and rules |
| `server_url` | `https://api.eidola.ai` | Paired server endpoint for this client version |
| `server_tee_measurement` | (measurement value) | Expected TEE measurement of paired server |
| `webpki_root_source` | `system` | Use OS trust store for TLS (not for signature verification) |

WebPKI is used only for transport TLS. Cryptographic trust in releases derives entirely from the embedded trust root above, not from WebPKI.

## Operational notes

- **Bus factor.** Provision the YubiKey in duplicate at v1 setup; store the spare in a safe-deposit box. Losing the only attestant key means no further releases until clients ship with a new fingerprint, and shipping a new fingerprint requires a release, which requires the key.
- **Release cadence commitment.** Make a public, dated statement of the maximum interval between releases. This is the v1 substitute for cryptographic freshness — a stalled release becomes a question the community can ask publicly.
- **CI identity migration.** Changes to workflow name, default branch, or repo path break the embedded `expected_ci_identity_pattern`. Treat these as breaking changes that require a coordinated trust-root rollout.
- **TEE measurement rotation.** Tinfoil base image and kernel updates change the measurement. Each rotation is a normal release event with a fresh human attestation. Run old and new server endpoints in parallel during transitions so clients on the prior version continue to verify against their embedded measurement until they update.
