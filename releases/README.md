# releases/

This directory holds the **build inputs** that pin Eidola's trust root: the values that get compiled into every client binary at release time and that determine every subsequent trust decision the client makes at runtime.

For the conceptual model — what these files are for and why they are structured this way — see [`docs/trust-root.md`](../docs/trust-root.md). For the privacy and security properties they enforce, see [`docs/privacy-guarantees.md`](../docs/privacy-guarantees.md).

This file is for contributors who need to *modify* something here.

## Files

| Path | Purpose | Consumed by |
| --- | --- | --- |
| `schema/attestation-templates.json` | Pinned claim templates for human release attestations. The verifier re-renders each claim from these templates and rejects attestations whose claim text does not match; `schema_version` (inside the file) tracks the file format, pinned by `SUPPORTED_SCHEMA_VERSION` in `crates/eidola-attestation` | `crates/eidola-app-core/build.rs`, `release-tool attest` (as committed at the previous release tag) |
| `trust/trust-constants.json` | Non-derivable trust values: pinned attestant fingerprints, CI identity pattern, minimum-attestation count, supported schema versions, update-discovery URL | `crates/eidola-app-core/build.rs` |
| `trust/sigstore-trusted-root.json` | Snapshot of the upstream Sigstore `TrustedRoot` (Fulcio CAs, Rekor public keys, CT log keys, TSAs) | `crates/eidola-app-core/build.rs` (updater) and `crates/eidola-server/build.rs` (runtime upstream-measurement resolver) |
| `trust/server-enclave.json` | The paired server enclave measurement (SEV-SNP launch digest, TDX RTMR1/RTMR2, kernel cmdline). Materialized as its own file so the cli build can COPY it without dragging the full manifest into its build context | `crates/eidola-app-core/build.rs` |
| `trust/attestant-provenance/` | **Informational only** — optional hardware-attestation evidence (e.g. YubiKey-PIV certs) that a pinned attestant fingerprint is a real on-device, policy-constrained key. See its [`README.md`](trust/attestant-provenance/README.md) | nothing — auditor-facing |

Most files here are **build inputs**. The corresponding **build output** — `artifact-manifest.json` at the repo root — is signed by CI and records the digests of what was actually produced (OCI `digest`, Nix `narHash` + `archiveSha256`). The two are kept separate to prevent build-context self-reference (see [`docs/trust-root.md`](../docs/trust-root.md#why-the-enclave-block-lives-in-its-own-file)). How to check a published download: [`docs/verification.md`](../docs/verification.md).

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

### Changing attestation templates

Templates are pinned by embedding: each installed client re-renders every claim from *its own* copy of `schema/attestation-templates.json` and rejects any character mismatch, so the copy that matters for release N is the one committed at release N−1. `release-tool attest` therefore renders and signs from the templates as committed at the **previous** release tag (via `git show`), never the working tree (`--templates <path>` exists as a deliberate-chain-break escape hatch):

1. Edit `schema/attestation-templates.json` — claim prose, adding or removing claims. `schema_version` stays put; it versions the file *format*, not the claim text.
2. Cut a release as normal. Its attestation is automatically rendered from the previous release's templates, so installed clients verify it, while the new binaries embed the changed templates.
3. The next release's attestation is signed under the changed templates.

A change to the file *format* (field shape) additionally bumps the file's `schema_version` and `SUPPORTED_SCHEMA_VERSION` in `crates/eidola-attestation` in the same commit, and must keep the previous release's templates loadable for the transition release's signing pass. A test in `eidola-attestation` loads the committed file and asserts the claim-ID set, and `eidola-app-core`'s build script loads it through the same verifier code path, so file/loader drift fails the build rather than every release verification.

This ordering is what prevents a coerced release from silently weakening a required claim: the release that changes claim text is itself attested under the unchanged prior text, and the change sits in the public release diff for a full release cycle before any attestation is signed under it.

### Rotating document schema versions

`release.json` and `attestation.json` carry integer `schema_version` values pinned by the supported sets in `trust-constants.json` (never a "patch" or "minor" — every change is fully breaking by contract):

1. Make the change; ship a release whose `supported_release_schema_versions` / `supported_attestation_schema_versions` list both `1` and `2`. The engineer continues signing schema-`1` documents.
2. Once in-the-wild clients have updated, cut another release where the engineer signs schema-`2` documents. `1` can be removed from the supported list in a later release.

#### `artifact-manifest.json` — the same rotation, with no human in it

The manifest carries a `schema_version` too, and rotates under the same accept-before-emit rule — but nobody chooses to emit it: CI regenerates the manifest from source on every run. The two sides are therefore *two files*, and they must move in different releases:

| Side | Where | Effect |
| --- | --- | --- |
| Accept | `SUPPORTED_MANIFEST_SCHEMA_VERSIONS` in `crates/eidola-app-core/src/updates.rs` | which manifest shapes a shipped client will read as authentic |
| Emit | `MANIFEST_SCHEMA_VERSION` in `scripts/artifact-manifest.sh`, plus whatever new rows the version adds | what every subsequent CI run produces |

1. Land the accept side alone: add the new version to the supported set, teach `attested_claims` / `describe_artifact` the new shape, and make the new rows tolerated-when-absent so releases still emitting the old version stay clean. Ship it.
2. Once in-the-wild clients carry that build, land the emit side: bump `MANIFEST_SCHEMA_VERSION` and record the new rows. The committed `artifact-manifest.json` moves with the next `just update-manifest`.

Landing them together is the failure this ordering exists to prevent: every installed client would meet a manifest shape it does not know and report `ClaimsChanged` — an authentic release that looks like a threat-model change, for everyone at once.

**Currently mid-rotation:** clients accept `2` and `3`; CI still emits `2`. Schema `3` adds the macOS unsigned shipping zip (`eidola-gui-macos-universal-zip`, `type: "file"` — the `sha256` of the container a macOS download arrives in, built by `nix build .#eidola-gui-macos-universal-zip`). Step 2 above is what turns it on.

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

## Upstream inference enclave measurements

There is no pinned upstream-measurement file here anymore. The Eidola server resolves the allowed upstream inference-enclave measurement **at runtime** from the provider's latest release and verifies its Sigstore provenance (against `sigstore-trusted-root.json`) before trusting it — see `crates/eidola-server/src/upstream_trust` and [`docs/upstream.md`](../docs/upstream.md#what-pins-the-upstream-measurement). Nothing to rotate in this directory for upstream measurements; keep `sigstore-trusted-root.json` current (below) so that verification keeps working.
