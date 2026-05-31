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
| `SERVER_URL`                            | derived from `releases/trust/server-enclave.json`   | `gateway-<hash>.eidola.containers.tinfoil.sh`, where `<hash>` ties the URL to a server measurement |
| `SERVER_SNP_MEASUREMENT`                | `releases/trust/server-enclave.json` `snp_measurement` | SEV-SNP launch measurement of the paired server enclave                                         |
| `SERVER_TDX_RTMR1` / `SERVER_TDX_RTMR2` | `releases/trust/server-enclave.json` `tdx_measurement` | TDX runtime measurements of the paired server enclave                                           |
| `TRUSTED_ATTESTANT_FINGERPRINTS`        | `trust-constants.json`                              | `sha256(OpenSSH wire-format pubkey)` in hex, for each authorized human-attestant key               |
| `MIN_HUMAN_ATTESTATIONS`                | `trust-constants.json`                              | Minimum independently-verified human attestations a release must carry (pinned here, not in `release.json`, so a forged index cannot lower it) |
| `EXPECTED_CI_IDENTITY_PATTERN`          | `trust-constants.json`                              | Fulcio cert SAN pattern the release-signing workflow's OIDC identity must match                    |
| `EXPECTED_CI_ISSUER`                    | `trust-constants.json`                              | OIDC issuer (`https://token.actions.githubusercontent.com`)                                        |
| `SUPPORTED_RELEASE_SCHEMA_VERSIONS`     | `trust-constants.json`                              | Integer `schema_version` values of `release.json` this client will parse                           |
| `SUPPORTED_ATTESTATION_SCHEMA_VERSIONS` | `trust-constants.json`                              | Integer `schema_version` values of `attestation.json` this client will parse                       |
| `UPDATE_DISCOVERY_URL`                  | `trust-constants.json`                              | Where to look for the next release (GitHub releases API)                                           |
| `ATTESTATION_TEMPLATES_JSON`            | `releases/schema/attestation-templates-v1.json`     | Pinned claim templates the verifier re-renders during equality checks                              |
| `SIGSTORE_TRUSTED_ROOT_JSON`            | `releases/trust/sigstore-trusted-root.json`         | Pinned Sigstore tlog / Fulcio / CT log keys with validity windows                                  |

The enclave block lives in its own file, **separate from `artifact-manifest.json`**,
for a single reason: build reproducibility. `artifact-manifest.json` records the
eidola-cli OCI digest and the eidola-cli-macos-universal narHash among other
artifacts. If the cli build COPYed (Docker) or filtered-in (Nix) the manifest as
a build input, every regeneration of the manifest would also be an input to the
build it's describing — a self-reference that produces a different digest on
every run instead of converging. `server-enclave.json` is the minimum slice of
the manifest that the cli build needs, so it can be COPYed without dragging the
cli's own digest into the build context. CI re-asserts the consistency:
`scripts/artifact-manifest.sh verify-full` recomputes the enclave block from
`tinfoil-config.yml` and rejects the build if either `server-enclave.json` or
`artifact-manifest.json`'s `enclave` field disagrees with it.

`config.toml` overrides (`base_url`, `trusted_measurements`) take precedence
at runtime — set them to point a build at a local server or alternate
enclave. With overrides unset, the pin is what gets used.

## Releases

Each JSON document carries an explicit `schema_version` (positive
integer). The supported version sets are pinned in `trust-constants.json`
so the verifier rejects any document outside the set; bumping a schema
is itself a release-gated trust event. **There is no semver-style
"backwards-compatible minor bump"**: each integer denotes a distinct,
all-or-nothing shape that a verifier either understands fully or
refuses outright. This deliberately avoids the security weakening that
would happen if old clients silently tolerated a new claim or field
without enforcing it. (Product versions — `release.version`,
`release.previous_release.version` — remain semver strings, since those
*do* benefit from ordering and matching the Rust/Cargo ecosystem.)

Each document's shape is owned by the Rust `serde` types shared between the release-tool and the verifier — there is no separately-maintained JSON Schema file. Drift between signing and verifying is impossible because both sides deserialize from the same struct definitions.

| Document                 | Shape (source of truth)                                                                                   | Notes                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `artifact-manifest.json` | format owned by `scripts/artifact-manifest.sh`                                                            | `schema_version: 1`. Records OCI digests, the macOS narHash, and a denormalized copy of the enclave block. Signed by CI as a Sigstore bundle (Fulcio keyless, OIDC). |
| `releases/trust/server-enclave.json` | format owned by `scripts/artifact-manifest.sh`, consumed as raw JSON in `eidola-app-core/build.rs` | `schema_version: 1`. Holds just the enclave block (snp/tdx measurement + cmdline) so the cli build doesn't drag its own digest into its build context. |
| `releases/trust/tinfoil-enclaves.json` | format owned by `.github/workflows/update-measurements.yml`, consumed as raw JSON in `eidola-server/build.rs` | `schema_version: 1`. Allowed upstream Tinfoil inference-enclave measurements the server's outbound verifier accepts. One entry per Tinfoil release, with provenance metadata (built\_at, artifact digest, Rekor log index); the workflow keeps the most recent two for rolling deploys. |
| `release.json`           | `eidola_attestation::ReleaseIndex` — `crates/eidola-attestation/src/trust_shapes.rs`                      | Unsigned URL-only index; cross-checked via referenced documents (see caveat below)          |
| `attestation.json`       | `updater::human_attestation::AttestationProse` — `crates/eidola-app-core/src/updater/human_attestation.rs` | Signed by the attestant via SSH (`ssh-keygen -Y sign -n file`), logged to Rekor as a `rekord` v0.0.1 entry with `signature.format=ssh` |
| `trust-constants.json`   | `eidola_attestation::TrustConstants` — `crates/eidola-attestation/src/trust_shapes.rs`                    | Pinned trust values baked into the verifier at build time                                   |
| Templates                | `releases/schema/attestation-templates-v1.json` (data, not a schema)                                      | Pinned claim templates the verifier re-renders during equality checks                       |

### `release.json` is a pure URL index — no hashes, no policy

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

Policy values (minimum-attestation threshold, allowed identities,
allowed schema versions) follow the same rule. They live exclusively in
the *embedded* trust root of the previous client (see
[`MIN_HUMAN_ATTESTATIONS`](#whats-pinned) and friends). Otherwise an
attacker who produced a single forged attestation could also forge a
`release.json` that lowered the threshold to 1.

### Signing systems: split by surface

Two cryptographic systems carry the trust chain, dispatched by the verifier
based on which document is being verified:

| Surface | Signature | Identity binding | Transparency |
| --- | --- | --- | --- |
| CI signs `artifact-manifest.json` | Sigstore bundle | Fulcio keyless cert — OIDC identity matches `EXPECTED_CI_IDENTITY_PATTERN` | Rekor inclusion proof embedded in the bundle |
| Engineer signs `attestation-<id>.json` | SSH signature (`ssh-keygen -Y sign`, namespace `"file"` — forced by Rekor's SSH PKI verifier) | `sha256(SSH wire-format pubkey)` matches `TRUSTED_ATTESTANT_FINGERPRINTS` | Posted to Rekor as a `rekord` v0.0.1 entry with `signature.format=ssh`; Rekor canonicalizes `data.content` away, leaving only `data.hash` in the public log; inclusion proof saved in `attestation-<id>.bundle.json` |
| Engineer signs the git tag | SSH signature (same key) | Same fingerprint | Implicit via the repo |

The CI side uses Sigstore because Fulcio's keyless OIDC binding is *the*
mechanism that makes "this signature came from the release workflow on a
specific tag" cryptographically meaningful — no other system offers that.

The engineer side uses SSH because:
- The hardware-backing options are broader (Secretive's Secure Enclave,
  YubiKey-SK, 1Password agent, FIDO2 resident keys) — no specific brand
  required of future contributors.
- The signature format is dramatically simpler than the Sigstore bundle —
  the verifier code path is small, well-audited, pure-Rust via the
  `ssh-key` crate.
- The same key signs git tags, commits, and attestations, so an engineer
  has one identity surface.

Both ride the same Sigstore Rekor transparency log via different entry
kinds (`hashedrekord` with x509+ECDSA for CI; `rekord` with
`signature.format=ssh` for human). The verifier shares its Rekor
inclusion-proof + signed-tree-head verification code between the two
paths.

### Attestation templates: flat snake_case keys

Both the `claims` object in `attestation.json` and the `claims` object in
`attestation-templates-v1.json` use a flat set of snake_case keys
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

Everything under `releases/` is a **build input** — pinned data the
client and server compile against. `artifact-manifest.json` at the
repo root is the **build output** — a record of what was actually
produced, signed by CI. They live in different places on purpose:
files under `releases/` are bulk-copied/filtered into builds as a
unit, while `artifact-manifest.json` is deliberately kept out of
every build context to prevent self-reference cycles (it records
the eidola-cli OCI digest and macOS narHash that the cli build
would otherwise see in its own input).

```
releases/
  schema/
    attestation-templates-v1.json         # pinned claim templates
  trust/
    trust-constants.json                  # non-derivable trust values (input)
    sigstore-trusted-root.json            # upstream Sigstore TrustedRoot snapshot (input)
    server-enclave.json                   # paired-server enclave measurement (input — projection of artifact-manifest.json's enclave block, materialized as its own file so the cli build context can COPY it without dragging the manifest in)
    tinfoil-enclaves.json                 # allowed upstream Tinfoil inference-enclave measurements (input — server's build.rs reads this)
  TRUST-ROOT.md                           # this doc
artifact-manifest.json                    # full deployment record (output, signed by CI)
crates/eidola-app-core/
  build.rs                                # generator: server-enclave.json + trust-constants.json + … → trust_root.gen.rs
  src/trust_root.rs                       # exposes the generated constants
crates/eidola-server/
  build.rs                                # generator: tinfoil-enclaves.json → measurements.gen.rs
  src/measurements.rs                     # exposes the generated ALLOWED static
```

The generator (`build.rs`) reads `releases/trust/server-enclave.json` —
never the per-artifact digests. The chain that invalidates the pin:
server source changes → server image digest changes → `tinfoil-config.yml`
changes → kernel cmdline changes → enclave measurement changes →
`server-enclave.json` changes → client rebuilds with the new pin. Because
the client build context never reads `artifact-manifest.json`, regenerating
the manifest after a client build doesn't trigger another client rebuild,
so `just update-manifest` reaches a fixed point in a single run.

## Rotation procedures

Every rotation below ships as a normal release signed under the **current**
trust root. The new client binary carries the new values; old clients keep
running against the old values until the user accepts the update.

### Rotating an attestant key

1. Generate a new SSH keypair in your hardware-backed store of choice
   (Secretive / Secure Enclave on macOS, FIDO2-SK YubiKey, 1Password agent,
   …). Confirm the agent is reachable via `SSH_AUTH_SOCK` and `ssh-add -L`
   lists the new identity.
2. Compute the new fingerprint:
   ```
   awk '{print $2}' <path-to-new-pubkey.pub> \
     | base64 -d | shasum -a 256 | awk '{print $1}'
   ```
3. Open a release PR that adds the new fingerprint to
   `releases/trust/trust-constants.json` (`trusted_attestant_fingerprints`).
   **Keep the old fingerprint** during the overlap window so prior
   releases remain verifiable.
4. Cut a release signed by the **current** attestant key. The new client
   binary embeds both fingerprints.
5. After the overlap window has passed, open another release PR removing
   the old fingerprint. Sign with the new key.

### Rotating the CI signing workflow

The trust root pins
`https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v*`.
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

Any change to `attestation-templates-v1.json`, the release schema, or
the attestation schema requires bumping `schema_version` to the next
integer (never a "patch" or "minor" — every change is fully breaking by
contract):

1. Copy `attestation-templates-v1.json` → `attestation-templates-v2.json`,
   make the change. Same for the companion schema file if the structural
   shape changes.
2. Update `trust-constants.json`: `supported_attestation_schema_versions`
   lists both `1` and `2`.
3. Cut a release. Clients now accept both versions. Engineer continues
   signing schema-`1` attestations.
4. Once in-the-wild clients have updated, cut another release where the
   engineer signs schema-`2` attestations. `1` can be removed from the
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

## Known gaps

A consolidated list of every piece of the trust chain that's
intentionally deferred. The verifier is still load-bearing without
these — each one closes a specific class of attack that's already
constrained by other parts of the chain — but they're all worth
landing as follow-ups.

### Cryptographic verifier

| Gap | What it would catch | Why it's deferred |
| --- | --- | --- |
| **SCT (Signed Certificate Timestamp) verification** in the Fulcio leaf cert | A malicious or compromised Fulcio issuing certs for identities it shouldn't — the SCT proves the cert was logged in a public CT log | The OIDC-identity match + Fulcio chain walk are the primary binding; this is defense-in-depth |
| **Rekor checkpoint signature** verification | The Rekor instance forking a side-tree just for us (the inclusion proof we compute is mathematically valid but roots to a tree the public never sees) | The SET already requires the Rekor key to vouch for the entry; checkpoint adds independence-from-private-forks |
| **Artifact-hash check at install time** | A tampered binary download — the *manifest* is signed and content-verified, but the bytes we'd run aren't yet hashed against the manifest's declared digests | Lives naturally in step 5 (install/replace), once we know which platform's artifact the user is downloading. The verifier already proves `artifact-manifest.json` itself is authentic |
| **Multi-hop / fast-forward continuity** | A client that skips multiple releases (e.g. v1.0 → v1.5, missing v1.1–v1.4) — today the continuity gate requires strict equality between `release.previous_release.git_commit` and the installed commit, so an out-of-date client must update through every release in order | Strictly sequential is the safer floor; relaxing to "fast-forward reachable via GitHub commits API" is a small follow-up that's only worth doing once the cadence makes sequential updates painful |

These are noted at the top of
[`crates/eidola-app-core/src/updater/ci_sigstore/mod.rs`](../crates/eidola-app-core/src/updater/ci_sigstore/mod.rs)
and `rekor.rs` (the two crypto-side gaps) and at the
`TODO (step 5)` marker on `verify_each_artifact_hash` in
[`crates/eidola-app-core/src/updater/mod.rs`](../crates/eidola-app-core/src/updater/mod.rs)
(the install-side gap) so anyone reading the verifier code sees them up front.

### Operational

| Gap | Current behavior | Future |
| --- | --- | --- |
| **Install / atomic-replace** | `eidola update` runs the full verification pipeline and prints the verified attestation prose, but does not download or swap the binary | Step 5: download the artifact for the user's platform from `artifact-manifest.json`, hash-verify, atomic-replace + restart. Platform-specific (CLI = file swap; macOS GUI = staged swap on next launch). |
| **Single-attestant policy** | `MIN_HUMAN_ATTESTATIONS` (embedded in the client, sourced from `releases/trust/trust-constants.json`) is `1` in current releases — only one engineer needs to attest for a release to verify | Once a co-attestant key is provisioned and added to `trusted_attestant_fingerprints`, bumping `min_human_attestations` to `2` (in a release signed under the *current* threshold) makes every subsequent release require independent corroboration. The verifier already supports arbitrary M-of-N; we just haven't generated the second key yet. |
| **First-install downgrade** | A fresh client (no prior installed `git_commit`) bypasses continuity, so an adversary serving an internally-consistent older `release.json` could route them onto a real-but-stale release | Mitigations today: client surfaces `released_at`, and we make a public release-cadence statement. Permanent fix: ship a freshness anchor (witness checkpoint, Bitcoin-block reference, or co-signed signed-tree-head). See the [Unsigned `release.json`](#unsigned-releasejson--known-caveat-with-mitigation) section above. |

## Acknowledgements

`sigstore-trusted-root.json` is a verbatim copy of the upstream Sigstore
TrustedRoot. The Sigstore verification approach (CI side) is adapted from
[tinfoil-rs](https://github.com/tinfoilsh/tinfoil-rs), which in turn
adapts verification modules from
[sigstore-rs](https://github.com/sigstore/sigstore-rs) (Apache 2.0). The
SSH signature verification (human side) uses the
[`ssh-key`](https://github.com/RustCrypto/SSH/tree/master/ssh-key) crate
from RustCrypto.
