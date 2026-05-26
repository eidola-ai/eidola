# Eidola's Trust Model

Eidola's operating paradigm is designed to maximize user soveriegnty and
agency. The end-user is ultimately in control, running software that they
control and can modify, in environments that they choose. To the extent that
the end user chooses to use remote compute provided by us, multiple layers
ensure the end-user retains the ability to verify precisely what they are
running, and hold Eidola accountible to its unequivocal privacy commitments.

This document describes these mechanisms and the risks they mitigate.

## A Strong Client

In Eidola's model, the client is the user's entrypoint, the arbitor of trust
signals, and the coordinator of data flow. It is designed to "fail safe" when
the integrity of external systems cannot be verified. By design, a client
version trusts exactly one server version. This trust root is **embedded at
build time** — every Eidola release is a coordinated rebuild of *all* artifacts
(clients + server) where the client's trusted measurements correspond to the
server in the same release.

## Trust Roots

### Source Repository: full verifiability
- Our monorepo in git (integrity, immutability, etc)
- Fully reproducible builds
  - Fully source bootstrapped on linux using StageX
  - Hermetic builds on macOS using Tart and nix
  - Self-contained reproducibility invariant (artifact-manifest.json)

### Eidola & Contributors
- Individually signed git commits: accountibility more than trust or verifiability
- Signed human release attestations: accountibility more than trust or verifiability

### Infrastructure Providers
- GitHub OIDC (for signing builds via fulcio): minor trust signal for authenticity
- The Linux Foundation (rekor.sigstore.dev): verifiable append-only log for accountibility

### Hardware Manufacturers
- Whatever hardware the end user uses must be trusted
- AMD/Intel/Nvidia confidential compute: could produce/sign fake enclaves
  - In the future, OpenTitan will reduce the scope of required trust

### Local Environment
- Local operating system
- Other installed apps (particularly non-sandboxed, privileged apps)
- Physical environment

### WebPKI roots
- LetsEncrypt: one trust signal

## What's pinned

Generated into the client at compile time by
`crates/eidola-app-core/build.rs`, surfaced via
[`eidola_app_core::trust_root`](../crates/eidola-app-core/src/trust_root.rs):

| Constant                                | Source                                              | Purpose                                                                                            |
| --------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `SERVER_URL`                            | derived from `artifact-manifest.json` enclave block | `gateway-<hash>.eidola.containers.tinfoil.sh`, where `<hash>` ties the URL to a server measurement |
| `SERVER_SNP_MEASUREMENT`                | `artifact-manifest.json` `enclave.snp_measurement`  | SEV-SNP launch measurement of the paired server enclave                                            |
| `SERVER_TDX_RTMR1` / `SERVER_TDX_RTMR2` | `artifact-manifest.json` `enclave.tdx_measurement`  | TDX runtime measurements of the paired server enclave                                              |
| `TRUSTED_ATTESTANT_FINGERPRINTS`        | `trust-constants.json`                              | SHA-256 fingerprints of authorized human-attestant pubkeys                                         |
| `EXPECTED_CI_IDENTITY_PATTERN`          | `trust-constants.json`                              | Fulcio cert SAN pattern the release-signing workflow's OIDC identity must match                    |
| `EXPECTED_CI_ISSUER`                    | `trust-constants.json`                              | OIDC issuer (`https://token.actions.githubusercontent.com`)                                        |
| `SUPPORTED_RELEASE_SCHEMA_VERSIONS`     | `trust-constants.json`                              | Versions of `release.json` this client will parse                                                  |
| `SUPPORTED_ATTESTATION_SCHEMA_VERSIONS` | `trust-constants.json`                              | Versions of `attestation.json` this client will parse                                              |
| `UPDATE_DISCOVERY_URL`                  | `trust-constants.json`                              | Where to look for the next release (GitHub releases API)                                           |
| `ATTESTATION_TEMPLATES_JSON`            | `releases/schema/attestation-templates-v1.0.0.json` | Pinned claim templates the verifier re-renders during equality checks                              |
| `SIGSTORE_TRUSTED_ROOT_JSON`            | `releases/trust/sigstore-trusted-root.json`         | Pinned Sigstore tlog / Fulcio / CT log keys with validity windows                                  |

`config.toml` overrides (`base_url`, `trusted_measurements`) take precedence
at runtime — set them to point a build at a local server or alternate
enclave. With overrides unset, the pin is what gets used.

## Releases

Each JSON document carries an explicit `schema_version` (string, semver
form). The supported version sets are pinned in `trust-constants.json` so
the verifier rejects any document outside the set; bumping a schema is
itself a release-gated trust event.

| Document                | Schema                                                | Notes                                                                       |
| ----------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------- |
| `artifact-manifest.json` | (no JSON Schema file; format owned by `scripts/artifact-manifest.sh`) | `schema_version: "1.0.0"`                                                   |
| `release.json`          | `releases/schema/release-v1.0.0.schema.json`          | Unsigned index; cross-checked via referenced documents (see caveat below)   |
| `attestation.json`      | `releases/schema/attestation-v1.0.0.schema.json`      | Signed by the attestant's hardware key                                      |
| Templates               | `releases/schema/attestation-templates-v1.0.0.json`   | Source of truth for both the release-tool and the verifier's equality check |

### `release.json` carries no hashes

`release.json` is an *index*: URLs only. Hashes (the manifest's, each
attestation's, each Sigstore bundle's) live in the Sigstore bundles
themselves — the bundle signs the hash of what it certifies. The verifier
downloads each file, computes its hash, and asks the bundle whether that
hash was signed. Putting expected hashes in `release.json` would just add
more fields to keep in sync without strengthening any binding.

For the same reason, `release.json` does **not** carry
`expected_identity` / `expected_issuer` / `rekor_log_index` / Rekor key
material — those are pinned in the client's embedded trust root or
inherent in the Sigstore bundle. Echoing them in `release.json` would let
an adversary downgrade trust by handing the client a tampered index.

### Attestation templates: flat snake_case keys

Both the `claims` object in `attestation.json` and the `claims` object in
`attestation-templates-v1.0.0.json` use a flat set of snake_case keys
(`no_compulsion`, `manifest_reproduced`, …). The verifier walks every
template entry, renders it from the substitution `sources`, and rejects
unless the matching claim's `statement` is character-for-character equal.

The templates file is the single source of truth for both signing and
verification: the release-tool renders prose from it; the client verifier
re-renders from the same pinned bytes and compares. Both sides MUST use
the file exactly as committed; the pinned-bytes constant
`ATTESTATION_TEMPLATES_JSON` is what the verifier sees.

## Unsigned `release.json` — known caveat, with mitigation

`release.json` is published unsigned on the GitHub release. The
verification chain holds because:

- Each artifact and attestation it points to is independently signed.
- The CI Sigstore bundle binds the manifest hash, which is what's actually
  downloaded.
- Each attestation's content is cross-checked against `release.json`'s
  `version`, `git_commit`, and `previous_release.git_commit` — so an
  adversary can't substitute a stale attestation for a different release.
- Trusted attestant fingerprints are checked against the client-embedded
  set, not anything in `release.json`.

The protection that *doesn't* hold without a signed `release.json` is
**first-install downgrade**. A fresh client with no prior installed
version has no continuity check to anchor against, so an adversary serving
an internally-consistent older `release.json` could route the client onto
a real-but-stale release. Mitigations for v1:

1. The client surfaces `released_at` to the user before approving an
   install, so a suspiciously old release is visible.
2. A public release-cadence statement makes "the latest release is older
   than the cadence" a question the community can ask publicly.
3. Once a freshness anchor (witness checkpoint, Bitcoin block reference,
   or co-signed signed-tree-head) ships, the first-install gap closes.
   These are deferred from v1 deliberately, not forgotten.

## Where each piece lives

```
releases/
  schema/
    release-v1.0.0.schema.json
    attestation-v1.0.0.schema.json
    attestation-templates-v1.0.0.json     # pinned claim templates
  trust/
    trust-constants.json                  # non-derivable trust values
    sigstore-trusted-root.json            # upstream Sigstore TrustedRoot snapshot
  TRUST-ROOT.md                           # this doc
artifact-manifest.json                    # source of truth for measurements
crates/eidola-app-core/
  build.rs                                # generator
  src/trust_root.rs                       # exposes the generated constants
```

The generator (`build.rs`) reads `artifact-manifest.json`'s `enclave`
block **only** — never the per-artifact digests. The client's own narHash
appearing in the manifest is therefore not a build-cache input, so
regenerating the manifest after a client build doesn't trigger another
client rebuild. The chain that *does* invalidate the pin: server source
changes → server image digest changes → `tinfoil-config.yml` changes →
kernel cmdline changes → enclave measurement changes → manifest's
`enclave` block changes → client rebuilds with the new pin.

## Rotation procedures

Every rotation below ships as a normal release signed under the **current**
trust root. The new client binary carries the new values; old clients keep
running against the old values until the user accepts the update.

### Rotating an attestant key

1. Generate the new YubiKey-resident keypair.
2. Compute the SHA-256 fingerprint of the new pubkey (DER-encoded SPKI).
3. Open a release PR that adds the new fingerprint to
   `releases/trust/trust-constants.json` (`trusted_attestant_fingerprints`).
   **Keep the old fingerprint** during the overlap window so old releases
   continue to verify against monitoring tools that pinned them.
4. Cut a release signed by the **current** attestant key. This release's
   binary embeds both fingerprints.
5. After the overlap window has passed, open another release PR removing
   the old fingerprint. Sign with the new key.

### Rotating the CI signing workflow

The trust root pins
`https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-release.yml@refs/tags/v*`.
Changes to the workflow file path or repo path break this pattern. Treat
as a coordinated rotation:

1. Update `releases/trust/trust-constants.json` with the new pattern.
2. Cut a release signed by the **current** workflow under the **current**
   pattern. You can't change the workflow path in the same commit that
   introduces the new pattern — the next release's CI would sign under
   the new path, which clients with only the old pattern would reject.
3. After clients have updated, rename or move the workflow. The next
   release's CI signs under the new path; clients accept it because they
   already embed the new pattern.

### Rotating schema versions

Any change to `attestation-templates-v1.0.0.json`, the release schema, or
the attestation schema requires bumping `schema_version`:

1. Copy `attestation-templates-v1.0.0.json` →
   `attestation-templates-v1.1.0.json`, make the change. Same for the
   companion schema file if the structural shape changes.
2. Update `trust-constants.json`:
   `supported_attestation_schema_versions` lists both `"1.0.0"` and
   `"1.1.0"`.
3. Cut a release. Clients now accept both. Engineer continues signing
   `1.0.0` attestations.
4. Once in-the-wild clients have updated, cut another release where the
   engineer signs `1.1.0` attestations. `1.0.0` can be removed from the
   supported list in a later release.

This is the mechanism that prevents a coerced release from silently
weakening a required claim: weakening it requires a schema bump, which
itself requires a release under the current schema.

### Rotating the Sigstore trusted root

`sigstore-trusted-root.json` is a snapshot of Sigstore's upstream
`TrustedRoot` (Fulcio CAs, Rekor public keys, CT log keys, TSAs). It
rotates rarely. To refresh:

1. Pull the latest from Sigstore's public TUF repo
   (`https://tuf-repo-cdn.sigstore.dev/`) or copy from an audited
   downstream snapshot.
2. Diff carefully against the existing file — every added or removed
   entry should match a public Sigstore announcement.
3. Commit and cut a release. New trust material takes effect on next
   client update.

### Rotating the server URL pattern or hash length

`server_url_template` and `server_url_hash_length` in
`trust-constants.json` control how `SERVER_URL` is derived from the
enclave measurement. Changing either changes every future URL, so:

1. Decide the new template and length.
2. Update `trust-constants.json`.
3. Cut a release. The new client embeds the new URL, which the server
   deployment must serve under (configure Tinfoil's container DNS
   accordingly before publishing the release).

## Acknowledgements

`sigstore-trusted-root.json` is a verbatim copy of the upstream Sigstore
TrustedRoot. The schema and verification approach is adapted from
[tinfoil-rs](https://github.com/tinfoilsh/tinfoil-rs), which in turn
adapts verification modules from
[sigstore-rs](https://github.com/sigstore/sigstore-rs) (Apache 2.0).
