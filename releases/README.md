# releases/

This directory holds the **build inputs** that pin Eidola's trust root: the values that get compiled into every client binary at release time and that determine every subsequent trust decision the client makes at runtime.

For the conceptual model — what these files are for and why they are structured this way — see [`docs/trust-root.md`](../docs/trust-root.md). For the privacy and security properties they enforce, see [`docs/privacy-guarantees.md`](../docs/privacy-guarantees.md).

This file is for contributors who need to *modify* something here.

## Files

| Path | Purpose | Consumed by |
| --- | --- | --- |
| `schema/attestation-templates-v1.json` | Pinned claim templates for human release attestations. Schema-versioned; the verifier re-renders each claim from these templates and rejects attestations whose claim text does not match | `crates/eidola-app-core/build.rs` |
| `trust/trust-constants.json` | Non-derivable trust values: pinned attestant fingerprints, CI identity pattern, minimum-attestation count, supported schema versions, update-discovery URL | `crates/eidola-app-core/build.rs` |
| `trust/sigstore-trusted-root.json` | Snapshot of the upstream Sigstore `TrustedRoot` (Fulcio CAs, Rekor public keys, CT log keys, TSAs) | `crates/eidola-app-core/build.rs` |
| `trust/server-enclave.json` | The paired server enclave measurement (SEV-SNP launch digest, TDX RTMR1/RTMR2, kernel cmdline). Materialized as its own file so the cli build can COPY it without dragging the full manifest into its build context | `crates/eidola-app-core/build.rs` |
| `trust/tinfoil-enclaves.json` | Allowed upstream Tinfoil inference-enclave measurements (one entry per Tinfoil release, with provenance metadata; the workflow keeps the most recent two for rolling deploys) | `crates/eidola-server/build.rs` |
| `trust/attestant-provenance/` | **Informational only** — optional hardware-attestation evidence (e.g. YubiKey-PIV certs) that a pinned attestant fingerprint is a real on-device, policy-constrained key. See its [`README.md`](trust/attestant-provenance/README.md) | nothing — auditor-facing |

Most files here are **build inputs**. The corresponding **build output** — `artifact-manifest.json` at the repo root — is signed by CI and records the digests of what was actually produced. The two are kept separate to prevent build-context self-reference (see [`docs/trust-root.md`](../docs/trust-root.md#why-the-enclave-block-lives-in-its-own-file)).

## Rotation procedures

Every rotation below ships as a normal release signed under the **current** trust root. The new client binary carries the new values; old clients keep running against the old values until the user accepts the update.

### Rotating an attestant key

1. Generate a new signing key in your hardware-backed store of choice (YubiKey-PIV via PKCS#11, a cloud KMS supported by cosign, etc.). The key must be ECDSA-P256, ECDSA-P384, or Ed25519 — the updater's `verify_blob_signature_with_spki` rejects anything else.
2. Compute the key's sha256 SPKI fingerprint. For a YubiKey, `cargo run -p release-tool -- pkcs11 list` prints it directly (no PIN). For other key types:

   ```bash
   cosign public-key --key <key-ref> > new-attestant.pem
   openssl pkey -pubin -in new-attestant.pem -outform DER \
     | shasum -a 256 | awk '{print $1}'
   ```

3. Open a release PR that adds the new fingerprint to `releases/trust/trust-constants.json` (`trusted_attestant_fingerprints`). **Keep the old fingerprint** during the overlap window so prior releases remain verifiable.
4. Cut a release signed by the **current** attestant key. The new client binary embeds both fingerprints.
5. After the overlap window has passed, open another release PR removing the old fingerprint. Sign with the new key.
6. *(Optional, informational.)* Keep [`trust/attestant-provenance/`](trust/attestant-provenance/README.md) in lockstep with the pinned set: commit the new key's hardware-provenance bundle (`release-tool provenance capture` for a YubiKey) in the **same PR that pins its fingerprint** (step 3), and remove the old key's bundle in the **same PR that unpins it** (step 5). The retired key's evidence stays in git history; `release-tool provenance check` fails on a bundle left behind for an unpinned fingerprint.

### Rotating the CI signing workflow

The trust root pins `https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v*`. Changes to the workflow file path or repo path break this pattern. Treat as a coordinated rotation:

1. Update `releases/trust/trust-constants.json` with the new pattern.
2. Cut a release signed by the **current** workflow under the **current** pattern. You can't change the workflow path in the same commit that introduces the new pattern — the next release's CI would sign under the new path, which clients with only the old pattern would reject.
3. After clients have updated, rename or move the workflow. The next release's CI signs under the new path; clients accept it because they already embed the new pattern.

### Rotating schema versions

Any change to `attestation-templates-v1.json`, the release schema, or the attestation schema requires bumping `schema_version` to the next integer (never a "patch" or "minor" — every change is fully breaking by contract):

1. Copy `attestation-templates-v1.json` → `attestation-templates-v2.json`, make the change. Same for the companion schema file if the structural shape changes.
2. Update `trust-constants.json`: `supported_attestation_schema_versions` lists both `1` and `2`.
3. Cut a release. Clients now accept both versions. Engineer continues signing schema-`1` attestations.
4. Once in-the-wild clients have updated, cut another release where the engineer signs schema-`2` attestations. `1` can be removed from the supported list in a later release.

This is the mechanism that prevents a coerced release from silently weakening a required claim: weakening it requires a schema bump, which itself requires a release under the current schema.

### Rotating the Sigstore trusted root

`sigstore-trusted-root.json` is a snapshot of Sigstore's upstream `TrustedRoot` (Fulcio CAs, Rekor public keys, CT log keys, TSAs). It rotates rarely. To refresh:

1. Pull the latest from Sigstore's public TUF repo (`https://tuf-repo-cdn.sigstore.dev/`) or copy from an audited downstream snapshot.
2. Diff carefully against the existing file — every added or removed entry should match a public Sigstore announcement.
3. Commit and cut a release. New trust material takes effect on next client update.

### Rotating the server URL pattern or hash length

`server_url_template` and `server_url_hash_length` in `trust-constants.json` control how `SERVER_URL` is derived from the enclave measurement. Changing either changes every future URL, so:

1. Decide the new template and length.
2. Update `trust-constants.json`.
3. Cut a release. The new client embeds the new URL, which the server deployment must serve under (configure Tinfoil's container DNS accordingly before publishing the release).

## Updating the upstream inference enclave list

`trust/tinfoil-enclaves.json` is updated by the `.github/workflows/update-measurements.yml` workflow, which:

1. Pulls the latest measurements from the upstream's published release feed.
2. Verifies the provenance via Sigstore (`gh attestation verify --deny-self-hosted-runners`) against the expected repository identity.
3. Opens a PR that adds the new measurement and removes any measurements older than the rolling-deploy window (currently keeping the most recent two).

A new upstream measurement does not silently become trusted: it goes through the same review-and-merge process as any other source change, and the resulting Eidola server build embeds the new list.
