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

A claim that substitutes a field the attestation document does not yet carry is therefore two rotations at once, and they are already in step with each other: the claim takes effect one release after it is committed, and a document schema is accepted one release before it is emitted, so both flip at the same release. Wire the emit side to the templates rather than to a flag, so the two cannot be moved independently — `crates/release-tool/src/attest.rs` does this for the macOS signing block.

This ordering is what prevents a coerced release from silently weakening a required claim: the release that changes claim text is itself attested under the unchanged prior text, and the change sits in the public release diff for a full release cycle before any attestation is signed under it.

### Rotating document schema versions

`release.json` and `attestation.json` carry integer `schema_version` values pinned by the supported sets in `trust-constants.json` (never a "patch" or "minor" — every change is fully breaking by contract):

1. Make the change; ship a release whose `supported_release_schema_versions` / `supported_attestation_schema_versions` list both `1` and `2`. The engineer continues signing schema-`1` documents.
2. Once in-the-wild clients have updated, cut another release where the engineer signs schema-`2` documents. `1` can be removed from the supported list in a later release.

**Currently mid-rotation:** clients accept `release.json` at `1` and `2`, and `release-tool` still emits `1`. Schema `2` adds the artifact index the installer downloads from (`artifacts`, keyed by manifest artifact key, plus `apple_signature_bundle` where one was published) — URLs only, since this document is unsigned. Step 2 is what turns it on, in `crates/release-tool/src/attest.rs`. Until it flips, `attest` publishes the macOS signing outputs as release assets but records no URL for them, because `apple_signature_bundle` at schema `1` is a shape the verifier refuses — publishing and indexing are deliberately two releases apart.

**Also mid-rotation:** clients accept `attestation.json` at `1` and `2`. Schema `2` adds the macOS signing block — `apple_shipped_artifact_sha256`, `apple_signature_bundle_sha256`, `apple_team_id`, `apple_signing_identifier` — the key-dependent values `artifact-manifest.json` may never hold. `apple_team_id` is present but may be `null`: a signature with no Developer ID behind it names no team, and saying so is not the same as omitting it. The verifier holds a document to the schema it declares in both directions, so a schema-`1` attestation carrying any of those keys is refused exactly as a schema-`2` one omitting them is.

This rotation's emit side is not a separate decision, because it is welded to a template change. The claim `apple_signature_reconstructs` substitutes those fields, and the templates that bind are the *previous* release's — so `release-tool attest` emits schema `2` exactly when the templates it renders from declare that claim, which is the release after the one that commits it. That is the same release by which every client has had a build accepting schema `2`. Landing the two together in one release would be the failure both orderings exist to prevent, and the tool refuses the combination rather than relying on the operator: if the binding templates declare the claim, `attest` will not sign without the macOS signing outputs to check it against, and if they do not, it records no Apple fields even when those outputs are supplied.

**A third ordering crosses these two, and it is why publishing and attesting are separate gates.** The claim names "the unsigned build recorded in `artifact-manifest.json`", and the row it means (`eidola-gui-macos-universal-zip`) arrives with *manifest* schema 3 — an accept-before-emit rotation with its own timetable, described above. So there are real releases that have signing outputs worth publishing and nothing yet to say about them. `release-tool attest` therefore asks two questions instead of one:

| Question | What it gates | When it must hold |
| --- | --- | --- |
| Does the reconstruction hold? | whether the assets may be published at all | whenever the outputs are supplied |
| Does the unsigned build match the manifest row? | whether the claim may be affirmed | only where the claim is affirmed |

A present row is always checked — a mismatch is wrong whether or not anything is claimed about it — but an absent row is fatal only to the claim. Requiring it unconditionally would have meant the first release able to publish the signed macOS artifact could not publish it, waiting on a manifest flip that says nothing about whether the artifact is genuine. Publishing an unattested asset is the honest state during a transition, and it is the same state the `release.json` entry is in one paragraph above: the bytes are there, and nothing claims anything about them yet.

Note the consequence a claim addition always has, this one included: from the release that signs under the new templates, a client older than the templates change rejects the attestation — it carries a claim that client's pinned template manifest does not declare, which is the anti-coercion rule in `verify_attestation_content` doing its job. Such a client reports the release as unverifiable rather than installing it, and its user updates by hand. That is the deliberate cost of pinning claim text by embedding.

#### `artifact-manifest.json` — the same rotation, with no human in it

The manifest carries a `schema_version` too, and rotates under the same accept-before-emit rule — but nobody chooses to emit it: CI regenerates the manifest from source on every run. The two sides are therefore *two files*, and they must move in different releases:

| Side | Where | Effect |
| --- | --- | --- |
| Accept | `SUPPORTED_MANIFEST_SCHEMA_VERSIONS` in `crates/eidola-app-core/src/updates.rs` | which manifest shapes a shipped client will read as authentic |
| Emit | `MANIFEST_SCHEMA_VERSION` in `scripts/artifact-manifest.sh`, plus whatever new rows the version adds | what every subsequent CI run produces |

1. Land the accept side alone: add the new version to the supported set, teach `attested_claims` / `describe_artifact` the new shape, and make the new rows tolerated-when-absent so releases still emitting the old version stay clean. Ship it.
2. Once in-the-wild clients carry that build, land the emit side: bump `MANIFEST_SCHEMA_VERSION` and record the new rows. The committed `artifact-manifest.json` moves with the next `just update-manifest`.

Landing them together is the failure this ordering exists to prevent: every installed client would meet a manifest shape it does not know and report `ClaimsChanged` — an authentic release that looks like a threat-model change, for everyone at once.

**Currently mid-rotation:** clients accept `2` and `3`; CI still emits `2`. Schema `3` changes the artifact set in three ways:

- adds the macOS unsigned shipping zip (`eidola-gui-macos-universal-zip`, `type: "file"` — the `sha256` of the container a macOS download arrives in, built by `nix build .#eidola-gui-macos-universal-zip`);
- adds the Debian packages (`eidola-gui-linux-deb-amd64`, `…-arm64`, `type: "file"` — the `sha256` of the `.deb` itself, which is byte-reproducible and needs no archive indirection);
- **renames** the Nix Linux installable's key from `eidola-gui-linux-amd64` to `eidola-gui-linux-nix-amd64`, because Linux now has two installables and the old key implied it had one.

A rename is not an addition, so it is two schema-conditional rows in `updates.rs` rather than one: `EXPECTED_ARTIFACTS_SINCE_SCHEMA_3` for the arriving key and `EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2` for the retiring one. Each spelling is expected exactly under the schema that records it — a schema-2 manifest that drops the old key, and a schema-3 manifest that keeps it, both read as `ClaimsChanged`. Both lists tighten by themselves when a version leaves the supported set.

Step 2 above is one assignment — `MANIFEST_SCHEMA_VERSION` in `scripts/artifact-manifest.sh` — but it is not unconditional. The flake attributes and the CI jobs already carry the schema-3 shape, and the Linux keys and deb rows are gated on that number alone; what the assignment additionally requires is that *every* producer emit its schema-3 rows, because the generator refuses to write a manifest missing a row its own schema promises (`scripts/artifact-manifest.sh check-complete`, run by `just check` and `rust-checks`). Today the macOS unsigned shipping zip is built by CI but not yet recorded, so the flip will refuse until that row is emitted — by design: a manifest short of a row its schema requires is read as `ClaimsChanged` by every installed client, which is precisely what the rotation exists to avoid.

Regeneration host matters too. A Linux host builds only its own architecture, so at the moment of the flip it cannot supply the other architecture's `.deb` and has no committed row to copy one from; `just update-manifest` says so and stops. The macOS path builds both Linux architectures in containers, and CI composes the full set from its per-architecture jobs.

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
