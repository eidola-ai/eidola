//! Update *notification* flow — "a verified newer release exists; here's
//! the door to it."
//!
//! This module answers one question at launch and about once an hour after:
//! **is there a newer release of Eidola, and can this binary prove it's
//! real?** It never downloads or installs anything; installing is
//! [`crate::updater`]'s job.
//!
//! ## What "verified" means here
//!
//! Exactly what it means to the installer. A release is announced only after
//! [`crate::updater::verify_release`] accepts it: CI's signature over
//! `artifact-manifest.json`, under the pinned workflow identity and bound to
//! this release's tag, **and** at least `MIN_HUMAN_ATTESTATIONS` independent
//! human attestations, each signed by a pinned attestant key, logged in
//! Rekor, re-rendered character for character from the pinned templates, and
//! naming the sha256 of that same manifest. A release that passes CI's
//! check alone is `Unverifiable`. CI is one party that can be compromised,
//! and the attestation exists for that case (`docs/releases.md`).
//!
//! All of it is anchored in the trust root compiled into this build
//! ([`crate::trust_root`]): the Sigstore roots, the CI identity and issuer,
//! the attestant keys, the threshold, the templates, and the schemas this
//! build reads. No trust anchor is fetched from the network.
//!
//! ## Where releases are found
//!
//! `<releases>/latest/download/release.json`, where `<releases>` is the
//! trust root's `UPDATE_RELEASES_URL` (or the `update_feed` override in
//! dev). That is the `release.json` of the GitHub release marked `latest`.
//! The marker is a pointer, not evidence. Anyone who can edit the release
//! listing can move it, and that changes only *which* release goes through
//! the checks above. A 404 means nothing is marked latest, or the latest
//! release has no `release.json`. Neither is an update, and no client
//! check can tell either one from a withheld release.
//!
//! These are GitHub's download redirects, not its REST API. The API allows
//! sixty unauthenticated requests an hour per address, shared by every
//! client behind one NAT. The redirects have no such limit.
//!
//! **The version is read before anything is verified**
//! ([`crate::updater::gate_release`]). A release that is not newer is not
//! judged at all, so a current release this build cannot parse never turns
//! into a warning just by being current. Once a release claims to be newer,
//! every failure to verify it is a security state.
//!
//! ## Failure-mode matrix (product decisions — encoded in [`UpdateCheckResult`])
//!
//! | State | Meaning | Behavior |
//! |---|---|---|
//! | [`UpdateCheckResult::CheckFailed`] | Nothing could be fetched (offline, a server error, a rate limit), or the `release.json` found was unreadable before it claimed any version — not a security signal | Quiet. "Couldn't check (offline?)" in the Updates window; retry next cycle. Never alarms. A standing `Unverifiable`/`ClaimsChanged` state is **not** overwritten by a later `CheckFailed` (see [`UpdateState::absorb`]). |
//! | [`UpdateCheckResult::UpToDate`] | No newer `latest` release | Includes "no release is marked latest" (HTTP 404) and "latest is older/equal". |
//! | [`UpdateCheckResult::UpdateAvailable`] | `latest` is newer and passes every check `crate::updater::verify_release` makes | "Eidola vX.Y.Z is available", with who attested it. One action: open the release page. |
//! | [`UpdateCheckResult::Unverifiable`] | A newer `latest` fails verification: a missing or bad signature, the wrong identity, CI's identity bound to another tag, a missing, untrusted, malformed or unbound human attestation, too few of them, or an index that does not hold to its schema or continue from the installed commit | **Possible fake / channel compromise.** Hard, visible security state — never silent, never offers the artifact. `reason` states exactly what failed. Persists until a later check finds a verifiable `latest`. |
//! | [`UpdateCheckResult::ClaimsChanged`] | Verifies cryptographically, but the attested claims differ structurally from what this build expects | **Authentic, but the threat model changed.** Surfaces a side-by-side of expected vs attested claims; default NOT trusted. An explicit "treat as update" action ([`crate::AppCore::accept_changed_claims`]) records the user's choice. |
//!
//! ## Expected claims — the concrete set
//!
//! "Expected claims" are made concrete by [`expected_claims`]: claim
//! *types and structure*, never values (digests and measurements
//! legitimately change every release). Derived from the embedded trust
//! root plus the `artifact-manifest.json` schema this build was built
//! against:
//!
//! | Claim key | Expected value | Derivation |
//! |---|---|---|
//! | `manifest.schema_version` | `2 or 3 or 4 or 5` | [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`] — the manifest shapes this module's parser understands; a jump outside the set is a release-gated trust event (`docs/trust-root.md`) |
//! | `enclave.cvm_version` | `platform release with its manifest pinned (VERSION@sha256)` | the embedded trust root pins the paired server's SEV-SNP launch identity in the IGVM launch model (`trust_root::SERVER_SNP_MEASUREMENT` and friends), which is the launch of one release image |
//! | `enclave.igvm_sha256` | `SEV-SNP IGVM image hash (32-byte hex)` | ditto: the image the digest is computed over |
//! | `enclave.sev_snp.measurement` | `SEV-SNP launch measurement (48-byte hex)` | ditto, `trust_root::SERVER_SNP_MEASUREMENT` |
//! | `enclave.sev_snp.host_data` | `SEV-SNP HOST_DATA, the config hash (32-byte hex)` | ditto, `trust_root::SERVER_SNP_HOST_DATA`: what binds the server's config, since the digest is the same for every deployment of an image |
//! | `enclave.sev_snp.guest_policy` | `SEV-SNP guest policy (0x-prefixed hex)` | ditto, `trust_root::SERVER_SNP_GUEST_POLICY` |
//! | `artifacts.eidola-cli` | `oci (linux/amd64)` | [`EXPECTED_ARTIFACTS`] — the artifact set every accepted schema records |
//! | `artifacts.eidola-gui-macos-universal` | `nix (darwin/universal)` | ditto |
//! | `artifacts.eidola-postgres` | `oci (linux/amd64)` | ditto |
//! | `artifacts.eidola-server` | `oci (linux/amd64)` | ditto |
//! | `artifacts.eidola-gui-linux-amd64` | `nix (linux/amd64) or absent` | [`EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2`] — the Linux Nix installable's schema-2 key. Schema 3 records it as `…-nix-amd64` instead, so a manifest declaring 3 must *not* carry it |
//! | `artifacts.eidola-cli-macos-universal` | `nix (darwin/universal) or absent` | [`EXPECTED_ARTIFACTS_THROUGH_SCHEMA_4`] — the standalone macOS CLI archive, recorded through schema 4. Every GUI payload carries the CLI built beside it, so schema 5 records no separate macOS CLI and a manifest declaring 5 must *not* carry it |
//! | `artifacts.eidola-gui-linux-deb-amd64` | `file (linux/amd64) or absent` | [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`] — the Debian package a user installs with `apt install ./…`; the `.deb` itself is the hashed byte stream |
//! | `artifacts.eidola-gui-linux-deb-arm64` | `file (linux/arm64) or absent` | ditto, 64-bit ARM |
//! | `artifacts.eidola-gui-linux-nix-amd64` | `nix (linux/amd64) or absent` | ditto — the Linux Nix installable under the name schema 3 gives it |
//! | `artifacts.eidola-gui-macos-universal-zip` | `file (darwin/universal) or absent` | ditto — the macOS unsigned shipping zip |
//! | `artifacts.eidola-server-engine` | `oci (linux/amd64) or absent` | [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_4`] — the inference node's image, the one every Eidola-hosted engine deployment pins by digest |
//! | `artifacts.eidola-cli-linux-static-amd64` | `file (linux/amd64) or absent` | [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_5`] — the static Linux CLI binary, the one file of the `eidola-cli` image, published on its own so a host binary needs no container runtime |
//! | `artifacts.eidola-cli-linux-static-arm64` | `file (linux/arm64) or absent` | ditto, 64-bit ARM: the same client cross-compiled for `aarch64-unknown-linux-musl` by StageX's toolchain, taken out of the image that build produces |
//!
//! The `or absent` rows are the schema-conditional half: absent from the
//! older manifests this build also accepts, required of any manifest
//! declaring the schema that introduced them — and, for the retiring row,
//! the reverse. The
//! retiring row is bounded on both sides, because a row that stops being
//! recorded also started somewhere: below the oldest schema this build
//! accepts it gets no verdict at all, since the artifact set of a shape
//! this module never understood is not something it can report on.
//!
//! An attested manifest produces its own claim list via
//! [`attested_claims`]: a claim disappears when the field is absent, gains
//! a `present but malformed …` value when its shape is wrong, and
//! unrecognized manifest fields / artifact entries / artifact types /
//! artifact-row fields surface as extra claims. Row fields are an exact
//! set per type, never a minimum — an added per-row field is a claim
//! nobody reviewed. Any delta between the two lists — missing,
//! extra, or changed — is `ClaimsChanged`.
//!
//! Whether a field or a whole row is *required* can depend on the schema
//! the manifest declares. `archiveSha256` arrived with schema 2 and is
//! mandatory from there on — it is the only artifact hash a user can check
//! without Nix, so a schema-2 manifest that drops it has weakened the claim
//! and reads as `ClaimsChanged`. Schema-1 rows legitimately have none. Whole
//! rows arrive the same way with schema 3 — and one *leaves* the same way,
//! because schema 3 renames the Linux Nix installable's key rather than
//! adding to it. A rename is two schema-conditional rows, one arriving and
//! one retiring, so that each spelling is expected exactly under the schema
//! that records it. Schema 5 moves the command-line client the same way
//! without a rename: the standalone macOS CLI archive, which every GUI
//! payload makes redundant, retires, and the static Linux CLI arrives as a
//! file of its own for each architecture — one retiring row and two
//! arriving rows, each held to the schema a manifest declares.
//!
//! That two-sided tolerance is the shape of every manifest schema rotation:
//! a build **accepts** the new schema one release before any release
//! **emits** it (`releases/README.md`), so between those releases this
//! module must read the old and the new shape as equally authentic — while
//! still refusing a manifest that declares the new schema and omits what
//! that schema promises, *or* records a row the schema it declares does
//! not have yet. Tolerance runs in one direction only: the shape a
//! manifest announces is the shape it is held to.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::trust_root;
use crate::updater;

/// `artifact-manifest.json` `schema_version` values this build can parse
/// structurally. Distinct from `trust_root::SUPPORTED_RELEASE_SCHEMA_VERSIONS`
/// (which pins `release.json`, the install pipeline's index); this pins the
/// manifest shape [`attested_claims`] walks. A version outside this set is a
/// *claims change* (authentic but unintelligible to this build), not a
/// verification failure.
pub const SUPPORTED_MANIFEST_SCHEMA_VERSIONS: &[u32] = &[2, 3, 4, 5];

/// The schema the current artifact set was introduced at:
/// [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`] rows become required at it, and
/// [`EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2`] rows retire at it. Named once so
/// requirement, retirement and tolerance cannot drift apart.
const ARTIFACT_SET_SCHEMA: u64 = 3;

/// The schema the inference node's image arrives at:
/// [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_4`] rows become required at it.
const ENGINE_IMAGE_SCHEMA: u64 = 4;

/// The schema the command-line client's artifact set takes its current shape
/// at: [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_5`] rows become required at it, and
/// [`EXPECTED_ARTIFACTS_THROUGH_SCHEMA_4`] rows retire at it.
const CLI_ARTIFACT_SET_SCHEMA: u64 = 5;

/// The artifact entries every accepted `artifact-manifest.json` records, as
/// `(name, type, platform)`. Structure only — digests / narHashes /
/// archiveSha256 are values and legitimately change every release.
pub const EXPECTED_ARTIFACTS: &[(&str, &str, &str)] = &[
    ("eidola-cli", "oci", "linux/amd64"),
    ("eidola-gui-macos-universal", "nix", "darwin/universal"),
    ("eidola-postgres", "oci", "linux/amd64"),
    // The gateway image (`ghcr.io/eidola-ai/eidola-server-gateway`). The key
    // is deliberately not the crate's name: changing it is an artifact-set
    // rotation like the Linux Nix row's, not a rename in place.
    ("eidola-server", "oci", "linux/amd64"),
];

/// Artifact entries schema 3 adds:
///
/// * the macOS unsigned shipping zip, whose `sha256` is the identity of the
///   container a signed release is built from;
/// * the Linux Nix installable under its narrowed name — `eidola-gui-linux-nix-amd64`
///   rather than `eidola-gui-linux-amd64`, which used to imply "the Linux
///   GUI" and is now one of two Linux installables (the schema-2 spelling
///   retires in [`EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2`]);
/// * the Debian packages, one per architecture. Each is `type: "file"`
///   because the `.deb` a user downloads *is* the byte stream its hash
///   covers — no Nix and no unpacking between the download and
///   `sha256sum` — the same reading the shipping zip gets.
///
/// Kept separate from [`EXPECTED_ARTIFACTS`] because this build accepts both
/// schemas: a schema-2 manifest legitimately has none of these rows, and only
/// a manifest declaring schema 3 or later is required to carry them. When 2
/// leaves [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`] they become unconditionally
/// required, with no further edit here.
pub const EXPECTED_ARTIFACTS_SINCE_SCHEMA_3: &[(&str, &str, &str)] = &[
    ("eidola-gui-linux-deb-amd64", "file", "linux/amd64"),
    ("eidola-gui-linux-deb-arm64", "file", "linux/arm64"),
    ("eidola-gui-linux-nix-amd64", "nix", "linux/amd64"),
    ("eidola-gui-macos-universal-zip", "file", "darwin/universal"),
];

/// Artifact entries schema 3 *retires*: the Linux Nix installable's old key,
/// which schema 3 records as `eidola-gui-linux-nix-amd64` instead.
///
/// The mirror image of [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`], and it exists
/// for the same reason: a rename is not an addition, so accepting both
/// schemas means accepting both spellings — each only under the schema that
/// records it. A schema-2 manifest without this row has dropped an artifact;
/// a schema-3 manifest carrying it has recorded one its own schema does not
/// have. Symmetric self-tightening: the row disappears from the expected set
/// entirely once 2 leaves [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`], with no
/// further edit here.
pub const EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2: &[(&str, &str, &str)] =
    &[("eidola-gui-linux-amd64", "nix", "linux/amd64")];

/// Artifact entries schema 4 adds: the inference node's OCI image
/// (`ghcr.io/eidola-ai/eidola-server-engine`), the image every Eidola-hosted
/// engine deployment pins by digest in its measured config. The gateway's
/// own measurement already binds those pins; this row is what ties the
/// digest to a reproducible build of the release's source, the way
/// `eidola-server` does for the gateway.
///
/// The same tolerance as [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`], one schema
/// later: absent from every manifest declaring an older schema, required
/// of any declaring 4 or later, and unconditional once no older schema is
/// accepted.
pub const EXPECTED_ARTIFACTS_SINCE_SCHEMA_4: &[(&str, &str, &str)] =
    &[("eidola-server-engine", "oci", "linux/amd64")];

/// Artifact entries schema 5 adds: the static Linux CLI, published as the
/// one file it is, for each architecture. The amd64 binary is the one the
/// `eidola-cli` image carries; the arm64 binary is the same client
/// cross-compiled by the same StageX toolchain into an image of its own
/// that is never published (`oci/eidola-cli-arm64/Containerfile`). Each is
/// extracted from its image and hashed (`scripts/artifact-manifest.sh
/// extract-cli --arch …`), so a user installing a host binary needs neither
/// a container runtime nor a registry. `type: "file"` for the reason the
/// `.deb` is: the download *is* the byte stream its hash covers. The keys
/// say `static` because the CLI also ships inside every Linux GUI payload,
/// and an unqualified `eidola-cli-linux-<arch>` would imply it ships once.
///
/// The same tolerance as [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`], two schemas
/// later.
pub const EXPECTED_ARTIFACTS_SINCE_SCHEMA_5: &[(&str, &str, &str)] = &[
    ("eidola-cli-linux-static-amd64", "file", "linux/amd64"),
    ("eidola-cli-linux-static-arm64", "file", "linux/arm64"),
];

/// Artifact entries schema 5 *retires*: the standalone macOS CLI archive.
/// Every GUI payload carries the CLI built from the same source as the app
/// (`Contents/MacOS/eidola-cli` in the macOS `.app`), so a separate macOS
/// build is a further copy of the same client that could only drift from
/// it.
///
/// The same two-sided reading as [`EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2`]: a
/// manifest declaring schema 2, 3 or 4 without this row has dropped an
/// artifact, one declaring 5 with it has recorded an artifact its schema
/// does not have, and the row leaves the expected set once 4 leaves
/// [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`].
pub const EXPECTED_ARTIFACTS_THROUGH_SCHEMA_4: &[(&str, &str, &str)] =
    &[("eidola-cli-macos-universal", "nix", "darwin/universal")];

/// One slice of the artifact set, as `(name, type, platform)` rows.
type ArtifactRows = &'static [(&'static str, &'static str, &'static str)];

/// Every arriving slice of the artifact set, with the schema it is recorded
/// from. One table, so the expected side and the attested side cannot
/// disagree about where a row starts.
const ARRIVING_ARTIFACTS: &[(u64, ArtifactRows)] = &[
    (ARTIFACT_SET_SCHEMA, EXPECTED_ARTIFACTS_SINCE_SCHEMA_3),
    (ENGINE_IMAGE_SCHEMA, EXPECTED_ARTIFACTS_SINCE_SCHEMA_4),
    (CLI_ARTIFACT_SET_SCHEMA, EXPECTED_ARTIFACTS_SINCE_SCHEMA_5),
];

/// Every retiring slice, with the first schema that no longer records it.
const RETIRING_ARTIFACTS: &[(u64, ArtifactRows)] = &[
    (ARTIFACT_SET_SCHEMA, EXPECTED_ARTIFACTS_THROUGH_SCHEMA_2),
    (CLI_ARTIFACT_SET_SCHEMA, EXPECTED_ARTIFACTS_THROUGH_SCHEMA_4),
];

/// How often, on average, the background poll re-checks while the app is
/// running.
pub const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// The wait before the next background check: [`POLL_INTERVAL`], moved by up
/// to a quarter of it either way, drawn fresh each time.
///
/// Every request to the releases host reveals that an Eidola client is
/// running at an address. A fixed hour would also let that host line the
/// requests up into a launch time and a run length per address. Drawn
/// waits don't, and the jitter keeps clients that start together from
/// staying in step.
pub fn next_poll_delay(rng: &mut impl rand_core::RngCore) -> std::time::Duration {
    let base = POLL_INTERVAL.as_millis() as u64;
    let spread = base / 2;
    std::time::Duration::from_millis(base - spread / 2 + rng.next_u64() % (spread + 1))
}

// ---------------------------------------------------------------------------
// Result types — the typed mirror of the failure-mode matrix
// ---------------------------------------------------------------------------

/// Outcome of one update check. See the module docs for the matrix each
/// variant encodes. Serializable so the last check persists across runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UpdateCheckResult {
    /// No newer `latest` release: either the marked-latest release is our
    /// version or older, or no release is marked `latest` at all
    /// (`latest_version: None`).
    UpToDate { latest_version: Option<String> },
    /// A newer `latest` release exists and every cryptographic check
    /// passed against the embedded trust root. The one action is opening
    /// `release.release_url` in the browser.
    UpdateAvailable { release: VerifiedRelease },
    /// A newer `latest` release exists but could **not** be
    /// cryptographically verified — possible fake or channel compromise.
    /// `reason` states exactly what failed. The UI must surface this
    /// loudly and must never link to the artifact.
    Unverifiable {
        version: String,
        tag: String,
        reason: String,
    },
    /// The release verified cryptographically, but its attested claims
    /// differ structurally from what this build expects. The user makes
    /// the trust call from the side-by-side; default is NOT trusted.
    ClaimsChanged {
        release: VerifiedRelease,
        comparison: ClaimsComparison,
    },
    /// The check itself failed (network, API, malformed feed). Not a
    /// security signal — shown quietly, retried next cycle.
    CheckFailed { message: String },
}

/// A release that passed every check [`crate::updater::verify_release`]
/// makes — CI's signature bound to its tag, and the human attestations the
/// embedded policy requires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedRelease {
    /// Semver, without the leading `v`.
    pub version: String,
    /// The git tag (e.g. `v0.0.9`), cross-checked against the Fulcio
    /// cert's `@refs/tags/…` identity suffix.
    pub tag: String,
    /// The commit the release was built from, as its attestations name it.
    pub git_commit: String,
    /// The release page, derived from the releases location and the tag.
    pub release_url: Option<String>,
    /// When the release says it was cut (`release.json`'s `released_at`).
    pub published_at: Option<String>,
    /// The Fulcio cert SAN identity CI's manifest signature verified under.
    pub ci_identity: String,
    /// Rekor transparency-log index of CI's manifest signature, for
    /// independent lookup.
    pub rekor_log_index: u64,
    /// Every human attestation that verified, in the order the release
    /// lists them.
    pub attestants: Vec<ReleaseAttestant>,
    /// Hex sha256 of the verified manifest bytes — the durable handle an
    /// accepted claims-change is recorded against.
    pub manifest_sha256: String,
    /// True when this release's claims differed from the expected set but
    /// the user explicitly chose "treat as update" for this exact
    /// manifest (matched by version + manifest hash).
    pub claims_accepted: bool,
}

/// Who attested a release, as their verified attestation states it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseAttestant {
    /// The attestant's name, from the signed preamble.
    pub name: String,
    /// The jurisdiction whose perjury laws they signed under.
    pub jurisdiction: String,
    /// Rekor log index of their signature, for independent lookup.
    pub rekor_log_index: u64,
}

/// One structural claim — `key` identifies it, `value` describes its
/// type/shape (never a digest or measurement value).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub key: String,
    pub value: String,
}

/// Side-by-side material for the claims-changed state: the full expected
/// and attested claim lists plus the computed differences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimsComparison {
    pub expected: Vec<Claim>,
    pub attested: Vec<Claim>,
    /// Rows where the two sides differ. Empty deltas never reach the UI —
    /// an empty diff is `UpdateAvailable`.
    pub deltas: Vec<ClaimDelta>,
}

/// One differing claim row. `None` on a side means the claim is absent
/// there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDelta {
    pub key: String,
    pub expected: Option<String>,
    pub attested: Option<String>,
}

/// One completed check with its wall-clock time, persisted and surfaced
/// to UIs ("last checked 2h ago").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateCheckSnapshot {
    pub checked_at_ms: i64,
    pub result: UpdateCheckResult,
}

/// The user's recorded "treat as update" decision for one exact manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptedClaims {
    pub version: String,
    pub manifest_sha256: String,
    pub accepted_at_ms: i64,
}

// ---------------------------------------------------------------------------
// Persisted state — last check + accepted claims choice
// ---------------------------------------------------------------------------

/// The rules a persisted verdict was reached under. A verdict is only as
/// good as the checks behind it, so state written under any other rules is
/// discarded on load ([`load_state`]) rather than shown, held against a
/// later failure, or promoted by an acceptance. Raise this whenever what
/// "verified" means for a release changes.
///
/// `1`: CI's signature bound to the release's tag, **and** the human
/// attestations the embedded policy requires ([`crate::updater::verify_release`]).
/// State with no stamp predates that and holds verdicts reached on CI's
/// signature alone.
pub const VERDICT_RULES: u32 = 1;

/// On-disk update-check state (`<data_dir>/update-state.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateState {
    /// The [`VERDICT_RULES`] `last` and `accepted` were reached under;
    /// absent (0) in state written before the stamp existed.
    #[serde(default)]
    pub verdict_rules: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<UpdateCheckSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<AcceptedClaims>,
}

impl Default for UpdateState {
    fn default() -> Self {
        Self {
            verdict_rules: VERDICT_RULES,
            last: None,
            accepted: None,
        }
    }
}

impl UpdateState {
    /// Fold a fresh check result into the state. The one non-trivial rule:
    /// a `CheckFailed` does **not** overwrite a standing `Unverifiable` or
    /// `ClaimsChanged` — those persist until a later check actually
    /// *completes* (silence or an offline blip must not clear the one
    /// signal that matters).
    pub fn absorb(&mut self, snapshot: UpdateCheckSnapshot) {
        let standing_alert = matches!(
            self.last.as_ref().map(|s| &s.result),
            Some(UpdateCheckResult::Unverifiable { .. })
                | Some(UpdateCheckResult::ClaimsChanged { .. })
        );
        if standing_alert && matches!(snapshot.result, UpdateCheckResult::CheckFailed { .. }) {
            return;
        }
        self.last = Some(snapshot);
    }
}

/// Path of the persisted update state inside the app data directory.
pub fn state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("update-state.json")
}

/// Load persisted state. Missing or corrupt files, and state reached under
/// rules other than [`VERDICT_RULES`], yield the default: the next check
/// reaches a fresh verdict, and nothing is shown or accepted before it.
pub fn load_state(data_dir: &Path) -> UpdateState {
    let Ok(bytes) = std::fs::read(state_path(data_dir)) else {
        return UpdateState::default();
    };
    match serde_json::from_slice::<UpdateState>(&bytes) {
        Ok(state) if state.verdict_rules == VERDICT_RULES => state,
        _ => UpdateState::default(),
    }
}

/// Persist the state, creating the data dir as needed.
pub fn save_state(data_dir: &Path, state: &UpdateState) -> Result<(), AppError> {
    std::fs::create_dir_all(data_dir).map_err(|e| AppError::Update {
        message: format!("creating data dir for update state: {e}"),
    })?;
    let bytes = serde_json::to_vec_pretty(state).map_err(|e| AppError::Update {
        message: format!("serializing update state: {e}"),
    })?;
    std::fs::write(state_path(data_dir), bytes).map_err(|e| AppError::Update {
        message: format!("writing update state: {e}"),
    })
}

// ---------------------------------------------------------------------------
// Check context — trust pins resolved once, injectable for fixture tests
// ---------------------------------------------------------------------------

/// Everything one check needs besides the HTTP client. Production callers
/// use [`CheckContext::new`], which resolves every pin from the embedded
/// trust root; tests override individual fields to drive specific matrix
/// rows (the crypto still runs for real — only the pins move).
#[derive(Debug, Clone)]
pub struct CheckContext {
    /// The releases location (`<url>/latest/download/release.json` is the
    /// latest release's index). Resolved from
    /// `Config::update_releases_url()`.
    pub releases_url: String,
    /// The running binary's semver.
    pub installed_version: String,
    /// The trust pins every check runs against.
    pub pins: updater::ReleasePins,
    /// The expected structural claim set (see module docs).
    pub expected_claims: Vec<Claim>,
}

/// What a completed check found, **before** any decision that depends on
/// persisted state.
///
/// A check is a network round trip and a pile of cryptography, and the user
/// can act during it: "treat as update" may land while the request that will
/// report on that very manifest is still in the air. Deciding acceptance
/// inside the check means deciding it against the state as it was when the
/// request went out, and a check that then reports `ClaimsChanged` for a
/// manifest the user has just accepted puts the warning and the button back
/// while the acceptance sits recorded beside it. So the network phase ends
/// here, holding what it *found*, and [`classify`] turns that into a verdict
/// against the acceptance that exists when it lands.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckOutcome {
    /// A verdict no acceptance can change: nothing to update to, nothing
    /// verifiable, or nothing reached at all.
    Settled(UpdateCheckResult),
    /// The release verified and its claims were compared. Whether that
    /// comparison is *tolerated* is the only open question, and it is a
    /// question about persisted state.
    Compared {
        release: VerifiedRelease,
        comparison: ClaimsComparison,
    },
}

/// Turn what a check found into the verdict to show, given the acceptance on
/// record **now**.
///
/// Pure, and deliberately so: this is the whole decision that depends on
/// state, it needs nothing async, and keeping it out of the network phase is
/// what lets a caller hold it until it has the current state in hand. An
/// empty diff is an ordinary update regardless of acceptance; a non-empty one
/// is tolerated only by an acceptance naming this exact version *and*
/// manifest hash.
pub fn classify(outcome: CheckOutcome, accepted: Option<&AcceptedClaims>) -> UpdateCheckResult {
    let (mut release, comparison) = match outcome {
        CheckOutcome::Settled(result) => return result,
        CheckOutcome::Compared {
            release,
            comparison,
        } => (release, comparison),
    };
    release.claims_accepted = !comparison.deltas.is_empty()
        && accepted.is_some_and(|a| {
            a.version == release.version
                && a.manifest_sha256
                    .eq_ignore_ascii_case(&release.manifest_sha256)
        });
    if comparison.deltas.is_empty() || release.claims_accepted {
        UpdateCheckResult::UpdateAvailable { release }
    } else {
        UpdateCheckResult::ClaimsChanged {
            release,
            comparison,
        }
    }
}

impl CheckContext {
    /// Production context: every pin from the embedded trust root, claims
    /// from [`expected_claims`].
    pub fn new(releases_url: impl Into<String>, installed_version: impl Into<String>) -> Self {
        Self {
            releases_url: releases_url.into(),
            installed_version: installed_version.into(),
            pins: updater::ReleasePins::embedded(),
            expected_claims: expected_claims(),
        }
    }
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// Run one check and classify it against `accepted` — [`check_outcome`] and
/// [`classify`] in one step, for a caller with nothing to reconcile against.
/// Infallible by design: every failure mode maps to an
/// [`UpdateCheckResult`] variant per the matrix in the module docs.
///
/// A caller that persists state should use the two halves instead and take
/// the acceptance from the state it is about to write, since anything read
/// before the round trip is a fact about a moment the user may have moved on
/// from. See `Inner::run_update_check`.
pub async fn check_for_update(
    client: &reqwest::Client,
    ctx: &CheckContext,
    accepted: Option<&AcceptedClaims>,
) -> UpdateCheckResult {
    classify(check_outcome(client, ctx).await, accepted)
}

pub async fn check_outcome(client: &reqwest::Client, ctx: &CheckContext) -> CheckOutcome {
    let settled = CheckOutcome::Settled;
    let fetcher = updater::Fetcher::Network(client.clone());

    // ── discover: the release marked `latest` ───────────────────────────
    let release_json = match fetcher.fetch_latest_release_json(&ctx.releases_url).await {
        Ok(Some(bytes)) => bytes,
        Ok(None) => {
            return settled(UpdateCheckResult::UpToDate {
                latest_version: None,
            });
        }
        // Discovery reached nothing it could read, so nothing claimed an
        // update. A body over the bound is the one `Rejected` discovery can
        // produce, and it carries no version either.
        Err(e) => {
            return settled(UpdateCheckResult::CheckFailed {
                message: e.message().to_string(),
            });
        }
    };

    // ── version gate, then the index's own shape ────────────────────────
    // Continuity is not checked here: no build knows the commit it was
    // released from (`docs/gaps.md` → First-install downgrade).
    let release = match updater::gate_release(&release_json, &ctx.installed_version, None) {
        Ok(updater::ReleaseGate::Newer(release)) => *release,
        Ok(updater::ReleaseGate::NotNewer { version }) => {
            return settled(UpdateCheckResult::UpToDate {
                latest_version: Some(version),
            });
        }
        Err(updater::GateError {
            newer_version: None,
            reason,
        }) => return settled(UpdateCheckResult::CheckFailed { message: reason }),
        Err(updater::GateError {
            newer_version: Some(version),
            reason,
        }) => {
            return settled(UpdateCheckResult::Unverifiable {
                tag: format!("v{version}"),
                version,
                reason,
            });
        }
    };

    // From here on, a newer release has been claimed — every failure to
    // verify it is a security state, and only an unreachable file is not.
    let evidence = match updater::verify_release(&fetcher, &release, &ctx.pins).await {
        Ok(evidence) => evidence,
        Err(updater::ReleaseCheckError::Unreachable(message)) => {
            return settled(UpdateCheckResult::CheckFailed { message });
        }
        Err(updater::ReleaseCheckError::Rejected(reason)) => {
            return settled(UpdateCheckResult::Unverifiable {
                version: release.version.clone(),
                tag: release.git_tag.clone(),
                reason,
            });
        }
    };

    // ── claims: structural comparison against the expected set ──────────
    let manifest_value: serde_json::Value =
        serde_json::from_slice(&evidence.manifest_bytes).unwrap_or(serde_json::Value::Null);
    let attested = attested_claims(&manifest_value);
    let deltas = compare_claims(&ctx.expected_claims, &attested);

    let summary = evidence.summary;
    let release = VerifiedRelease {
        release_url: Some(updater::release_page_url(
            &ctx.releases_url,
            &summary.git_tag,
        )),
        version: summary.version,
        tag: summary.git_tag,
        git_commit: summary.git_commit,
        published_at: Some(summary.released_at),
        ci_identity: evidence.ci_identity,
        rekor_log_index: evidence.ci_rekor_log_index,
        attestants: summary
            .attestations
            .iter()
            .map(|a| ReleaseAttestant {
                name: a.attestant_name.clone(),
                jurisdiction: a.jurisdiction.clone(),
                rekor_log_index: a.rekor_log_index,
            })
            .collect(),
        manifest_sha256: evidence.manifest_sha256,
        // Not yet knowable here: acceptance is a fact about persisted state,
        // and this function is the half that must not read it. `classify`
        // fills it in.
        claims_accepted: false,
    };

    CheckOutcome::Compared {
        release,
        comparison: ClaimsComparison {
            expected: ctx.expected_claims.clone(),
            attested,
            deltas,
        },
    }
}

// ---------------------------------------------------------------------------
// Claims — expected set, attested extraction, comparison
// ---------------------------------------------------------------------------

/// Rendered as an alternative in an expected claim's value when a row is
/// legitimately missing from an older manifest schema this build still
/// accepts. [`compare_claims`] reads it on the attested side's absence.
const ABSENT_CLAIM: &str = "absent";

const CVM_VERSION_CLAIM_VALUE: &str = "platform release with its manifest pinned (VERSION@sha256)";
const IGVM_CLAIM_VALUE: &str = "SEV-SNP IGVM image hash (32-byte hex)";
const SNP_CLAIM_VALUE: &str = "SEV-SNP launch measurement (48-byte hex)";
const HOST_DATA_CLAIM_VALUE: &str = "SEV-SNP HOST_DATA, the config hash (32-byte hex)";
const GUEST_POLICY_CLAIM_VALUE: &str = "SEV-SNP guest policy (0x-prefixed hex)";

/// The enclave block's claims: the server's SEV-SNP launch identity in the
/// IGVM launch model, and what names its image. `(key, value)` in the order
/// they are listed.
const ENCLAVE_CLAIMS: &[(&str, &str)] = &[
    ("enclave.cvm_version", CVM_VERSION_CLAIM_VALUE),
    ("enclave.igvm_sha256", IGVM_CLAIM_VALUE),
    ("enclave.sev_snp.measurement", SNP_CLAIM_VALUE),
    ("enclave.sev_snp.host_data", HOST_DATA_CLAIM_VALUE),
    ("enclave.sev_snp.guest_policy", GUEST_POLICY_CLAIM_VALUE),
];

/// The structural claim set this build expects an authentic release
/// manifest to carry. See the module docs for the full table and the
/// derivation of each row.
pub fn expected_claims() -> Vec<Claim> {
    expected_claims_for(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)
}

/// Does this build accept any manifest schema satisfying `pred`?
fn accepts_schema(supported: &[u32], pred: impl Fn(u64) -> bool) -> bool {
    supported.iter().any(|v| pred(u64::from(*v)))
}

/// The oldest manifest schema this build accepts — the lower bound of the
/// range every retiring row ([`RETIRING_ARTIFACTS`]) is recorded in.
fn oldest_supported_schema() -> u64 {
    SUPPORTED_MANIFEST_SCHEMA_VERSIONS
        .iter()
        .map(|v| u64::from(*v))
        .min()
        .unwrap_or(0)
}

/// [`expected_claims`] over an arbitrary supported-schema set, so the
/// self-tightening the two schema-conditional artifact lists promise is
/// testable at the sets this build will actually reach.
fn expected_claims_for(supported: &[u32]) -> Vec<Claim> {
    let mut claims = Vec::new();
    claims.push(Claim {
        key: "manifest.schema_version".into(),
        value: supported
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" or "),
    });
    // Enclave claims, derived from the embedded trust root: this build pins
    // the paired server's SEV-SNP launch identity in the IGVM launch model
    // (digest, HOST_DATA, guest policy, and the release image they come
    // from), so an authentic future manifest is expected to keep stating
    // exactly that. A different launch model, or a field disappearing, is
    // the "threat model changed" signal the claims comparison exists to
    // catch.
    if !trust_root::SERVER_SNP_MEASUREMENT.is_empty() {
        for (key, value) in ENCLAVE_CLAIMS {
            claims.push(Claim {
                key: (*key).into(),
                value: (*value).into(),
            });
        }
    }
    for (name, ty, platform) in EXPECTED_ARTIFACTS {
        claims.push(Claim {
            key: format!("artifacts.{name}"),
            value: format!("{ty} ({platform})"),
        });
    }
    // The two schema-conditional halves of the artifact set, and the one
    // rule they share: a row is *expected* only while some accepted schema
    // records it, and *tolerated absent* while some accepted schema does
    // not. `" or absent"` is what makes acceptance real — without it every
    // release emitting the other schema would read as a claims change on
    // that row.
    let mut push_conditional = |rows: &[(&str, &str, &str)], recorded: bool, tolerated: bool| {
        if !recorded {
            return;
        }
        for (name, ty, platform) in rows {
            let value = format!("{ty} ({platform})");
            claims.push(Claim {
                key: format!("artifacts.{name}"),
                value: if tolerated {
                    format!("{value} or {ABSENT_CLAIM}")
                } else {
                    value
                },
            });
        }
    };
    // Retiring rows: recorded only by the older schemas, so they leave the
    // expected set outright once no accepted schema records them.
    for (retired_at, rows) in RETIRING_ARTIFACTS {
        push_conditional(
            rows,
            accepts_schema(supported, |v| v < *retired_at),
            accepts_schema(supported, |v| v >= *retired_at),
        );
    }
    // Arriving rows, the mirror: tolerated absent while an older accepted
    // schema still lacks them.
    for (since, rows) in ARRIVING_ARTIFACTS {
        push_conditional(
            rows,
            accepts_schema(supported, |v| v >= *since),
            accepts_schema(supported, |v| v < *since),
        );
    }
    claims
}

/// Extract the structural claims an attested manifest actually makes.
/// Values describe shape, never content; unknown fields and unrecognized
/// artifact types surface as extra claims so they show up in the diff.
pub fn attested_claims(manifest: &serde_json::Value) -> Vec<Claim> {
    let Some(obj) = manifest.as_object() else {
        return vec![Claim {
            key: "manifest".into(),
            value: "not a JSON object".into(),
        }];
    };

    let mut claims = Vec::new();

    // Captured before the artifact walk: from schema 2 on, `archiveSha256`
    // is a *required* field on Nix rows, and whether its absence is a
    // claims change depends on which schema the manifest declares.
    let schema_version = obj.get("schema_version").and_then(|v| v.as_u64());

    if let Some(v) = obj.get("schema_version") {
        claims.push(Claim {
            key: "manifest.schema_version".into(),
            value: match v.as_u64() {
                Some(n) => n.to_string(),
                None => "present but not an integer".into(),
            },
        });
    }

    if let Some(enclave) = obj.get("enclave") {
        if let Some(enclave_obj) = enclave.as_object() {
            if let Some(v) = enclave_obj.get("cvm_version") {
                claims.push(Claim {
                    key: "enclave.cvm_version".into(),
                    value: describe_cvm_version(v),
                });
            }
            if let Some(v) = enclave_obj.get("igvm_sha256") {
                claims.push(Claim {
                    key: "enclave.igvm_sha256".into(),
                    value: describe_hex(v, 32, IGVM_CLAIM_VALUE),
                });
            }
            if let Some(snp) = enclave_obj.get("sev_snp") {
                if let Some(snp_obj) = snp.as_object() {
                    if let Some(v) = snp_obj.get("measurement") {
                        claims.push(Claim {
                            key: "enclave.sev_snp.measurement".into(),
                            value: describe_hex(v, 48, SNP_CLAIM_VALUE),
                        });
                    }
                    if let Some(v) = snp_obj.get("host_data") {
                        claims.push(Claim {
                            key: "enclave.sev_snp.host_data".into(),
                            value: describe_hex(v, 32, HOST_DATA_CLAIM_VALUE),
                        });
                    }
                    if let Some(v) = snp_obj.get("guest_policy") {
                        claims.push(Claim {
                            key: "enclave.sev_snp.guest_policy".into(),
                            value: match v.as_str() {
                                Some(s)
                                    if s.len() > 2
                                        && s.starts_with("0x")
                                        && s[2..].chars().all(|c| c.is_ascii_hexdigit()) =>
                                {
                                    GUEST_POLICY_CLAIM_VALUE.into()
                                }
                                Some(_) => {
                                    "present but malformed (expected 0x-prefixed hex)".into()
                                }
                                None => "present but not a string".into(),
                            },
                        });
                    }
                    for key in snp_obj.keys() {
                        if !matches!(key.as_str(), "measurement" | "host_data" | "guest_policy") {
                            claims.push(Claim {
                                key: format!("enclave.sev_snp.{key}"),
                                value: "unrecognized field".into(),
                            });
                        }
                    }
                } else {
                    claims.push(Claim {
                        key: "enclave.sev_snp".into(),
                        value: "present but not an object".into(),
                    });
                }
            }
            for key in enclave_obj.keys() {
                if !matches!(key.as_str(), "cvm_version" | "igvm_sha256" | "sev_snp") {
                    claims.push(Claim {
                        key: format!("enclave.{key}"),
                        value: "unrecognized field".into(),
                    });
                }
            }
        } else {
            claims.push(Claim {
                key: "enclave".into(),
                value: "present but not an object".into(),
            });
        }
    }

    if let Some(artifacts) = obj.get("artifacts") {
        if let Some(artifacts_obj) = artifacts.as_object() {
            // A manifest with no `schema_version` at all is held to the
            // newest shape: the field's absence is already a delta of its
            // own, and reading the rest under the *older* shape would let a
            // manifest escape a requirement by dropping the field.
            let records_from = |since: u64| schema_version.is_none_or(|v| v >= since);
            // The retiring rows are bounded on *both* sides, unlike the
            // arriving ones: a row that stops being recorded also started
            // being recorded somewhere. Below the oldest schema this build
            // accepts, this module does not know what the artifact set was
            // — the captured schema-1 release predates the Linux GUI
            // entirely — so it says so once, on `manifest.schema_version`,
            // rather than reporting a row missing from a shape it never
            // understood.
            let records_through = |retired_at: u64| {
                schema_version.is_some_and(|v| v >= oldest_supported_schema() && v < retired_at)
            };
            let declared =
                || schema_version.map_or_else(|| "«missing»".into(), |v: u64| v.to_string());
            for (name, entry) in artifacts_obj {
                let boundary = |table: &[(u64, ArtifactRows)]| {
                    table
                        .iter()
                        .find(|(_, rows)| rows.iter().any(|(n, _, _)| n == name))
                        .map(|(schema, _)| *schema)
                };
                let arrives = boundary(ARRIVING_ARTIFACTS);
                let retires = boundary(RETIRING_ARTIFACTS);
                claims.push(Claim {
                    key: format!("artifacts.{name}"),
                    // A row the declared schema does not record is a claim
                    // made outside the shape it was announced under —
                    // whether the row arrived too early or outlived its
                    // schema. Tolerating either would make the rotation
                    // one-sided: a release could start emitting the new
                    // shape without bumping the version, and neither the
                    // gate on the committed manifest nor this verifier
                    // would say a word.
                    value: if let Some(since) = arrives.filter(|since| !records_from(*since)) {
                        format!(
                            "present, but schema {} records it only from \
                             schema {since} on",
                            declared()
                        )
                    // `records_from`, not `!records_through`: the row is a
                    // claim out of shape once its schema has *retired* it,
                    // which is not the same as any schema that does not
                    // record it — a schema older than this build accepts
                    // does neither, and gets no verdict here.
                    } else if let Some(retired_at) =
                        retires.filter(|retired_at| records_from(*retired_at))
                    {
                        format!(
                            "present, but schema {} records it only through \
                             schema {}",
                            declared(),
                            retired_at - 1
                        )
                    } else {
                        describe_artifact(entry, schema_version)
                    },
                });
            }
            // A row a manifest's own declared schema requires but does not
            // carry is a claim it *stopped* making — so it has to appear in
            // the attested list, or the expected side's "or absent"
            // tolerance would silently excuse it.
            let mut note_missing = |rows: &[(&str, &str, &str)], recorded: bool, note: String| {
                if !recorded {
                    return;
                }
                for (name, _, _) in rows {
                    if !artifacts_obj.contains_key(*name) {
                        claims.push(Claim {
                            key: format!("artifacts.{name}"),
                            value: note.clone(),
                        });
                    }
                }
            };
            for (since, rows) in ARRIVING_ARTIFACTS {
                note_missing(
                    rows,
                    records_from(*since),
                    format!("missing (recorded from schema {since} on)"),
                );
            }
            for (retired_at, rows) in RETIRING_ARTIFACTS {
                note_missing(
                    rows,
                    records_through(*retired_at),
                    format!("missing (recorded through schema {})", retired_at - 1),
                );
            }
        } else {
            claims.push(Claim {
                key: "artifacts".into(),
                value: "present but not an object".into(),
            });
        }
    }

    for key in obj.keys() {
        if !matches!(key.as_str(), "schema_version" | "enclave" | "artifacts") {
            claims.push(Claim {
                key: format!("manifest.{key}"),
                value: "unrecognized field".into(),
            });
        }
    }

    claims
}

fn describe_hex(v: &serde_json::Value, bytes: usize, ok: &str) -> String {
    match v.as_str() {
        Some(s) if s.len() == bytes * 2 && s.chars().all(|c| c.is_ascii_hexdigit()) => ok.into(),
        Some(_) => format!("present but malformed (expected {} hex chars)", bytes * 2),
        None => "present but not a string".into(),
    }
}

fn describe_cvm_version(v: &serde_json::Value) -> String {
    match v.as_str() {
        Some(s) => match s.split_once("@sha256:") {
            Some((version, pin))
                if !version.is_empty()
                    && pin.len() == 64
                    && pin.chars().all(|c| c.is_ascii_hexdigit()) =>
            {
                CVM_VERSION_CLAIM_VALUE.into()
            }
            _ => "present but malformed (expected VERSION@sha256:<64 hex>)".into(),
        },
        None => "present but not a string".into(),
    }
}

/// Describe one artifact entry's *shape*. `schema_version` is the version
/// the enclosing manifest declares, which decides whether a field that only
/// exists from a given schema on is required or merely absent.
///
/// A row's fields are an exact set, not a minimum. An unrecognized field is
/// a structural change to what the release is claiming — the same reading
/// this module already gives an unrecognized top-level or enclave field,
/// and the same rule `scripts/check-manifest-determinism.sh` enforces on
/// the committed manifest. A verifier that accepted extras would let an
/// authentic release grow a per-row claim (an Apple hash, a Team ID) that
/// no user ever reviewed.
fn describe_artifact(entry: &serde_json::Value, schema_version: Option<u64>) -> String {
    let Some(obj) = entry.as_object() else {
        return "present but not an object".into();
    };
    let platform = obj
        .get("platform")
        .and_then(|p| p.as_str())
        .unwrap_or("missing platform");
    let described = describe_artifact_fields(obj, platform, schema_version);
    // `archiveSha256` is the one field whose presence is legitimate in some
    // accepted schemas and absent in others, so it is allowed on any `nix`
    // row; requiredness is judged above, by schema.
    let allowed: &[&str] = match obj.get("type").and_then(|t| t.as_str()) {
        Some("oci") => &["type", "platform", "digest"],
        Some("nix") => &["type", "platform", "narHash", "archiveSha256"],
        Some("file") => &["type", "platform", "sha256"],
        _ => return described,
    };
    let extra: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| !allowed.contains(k))
        .collect();
    if extra.is_empty() {
        described
    } else {
        format!("{described} — unrecognized field(s): {}", extra.join(", "))
    }
}

/// The per-type field checks: what each artifact type must carry, and in
/// what shape. Split from the key-set check above so each reads on its own.
fn describe_artifact_fields(
    obj: &serde_json::Map<String, serde_json::Value>,
    platform: &str,
    schema_version: Option<u64>,
) -> String {
    match obj.get("type").and_then(|t| t.as_str()) {
        Some("oci") => {
            let digest_ok = obj
                .get("digest")
                .and_then(|d| d.as_str())
                .is_some_and(|d| d.starts_with("sha256:"));
            if digest_ok {
                format!("oci ({platform})")
            } else {
                format!("oci ({platform}) — malformed: missing sha256 digest")
            }
        }
        Some("nix") => {
            let nar_ok = obj
                .get("narHash")
                .and_then(|d| d.as_str())
                .is_some_and(|d| d.starts_with("sha256-"));
            // `archiveSha256` arrived with schema 2 and is mandatory there:
            // it is the *only* hash a user can check without Nix, so a
            // schema-2 manifest that silently drops it has weakened the
            // claim and must surface as a claims change. Schema-1 rows
            // (the captured v0.0.8 fixture) legitimately have none, so
            // absence is only a defect from schema 2 on.
            let archive_required = schema_version.is_none_or(|v| v >= 2);
            let archive_ok = if obj.contains_key("archiveSha256") {
                is_sha256_hex(obj.get("archiveSha256"))
            } else {
                !archive_required
            };
            if nar_ok && archive_ok {
                format!("nix ({platform})")
            } else if !nar_ok {
                format!("nix ({platform}) — malformed: missing narHash")
            } else {
                format!("nix ({platform}) — malformed: expected sha256: hex archiveSha256")
            }
        }
        // One published file, identified by the hash of its bytes — no Nix
        // and no unpacking between the download and the check. The macOS
        // unsigned shipping zip is the first.
        Some("file") => {
            if is_sha256_hex(obj.get("sha256")) {
                format!("file ({platform})")
            } else {
                format!("file ({platform}) — malformed: expected sha256: hex sha256")
            }
        }
        Some(other) => format!("unrecognized type `{other}` ({platform})"),
        None => "missing type".into(),
    }
}

/// `sha256:` + 64 lowercase hex, the form every file hash in the manifest
/// takes.
fn is_sha256_hex(v: Option<&serde_json::Value>) -> bool {
    v.and_then(|d| d.as_str()).is_some_and(|s| {
        s.starts_with("sha256:")
            && s.len() == "sha256:".len() + 64
            && s["sha256:".len()..].chars().all(|c| c.is_ascii_hexdigit())
    })
}

/// Does an attested claim value satisfy an expected one?
///
/// Expected values are usually a single literal shape, but a few are a set
/// of acceptable alternatives rendered as `"a or b"` — today only
/// `manifest.schema_version`, built from
/// [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`]. Splitting on `" or "` is what
/// makes that set real: under plain equality a list of length > 1 could
/// never match anything, so every release would report `ClaimsChanged` on
/// that row forever. A single-element expected value has no `" or "` and
/// falls through to exact equality unchanged.
fn claim_matches(expected: &str, attested: &str) -> bool {
    expected == attested || expected.split(" or ").any(|alt| alt == attested)
}

/// Diff two claim lists by key. Rows appear in expected order first, then
/// extra attested claims in their manifest order.
pub fn compare_claims(expected: &[Claim], attested: &[Claim]) -> Vec<ClaimDelta> {
    let mut deltas = Vec::new();
    for exp in expected {
        let att = attested.iter().find(|a| a.key == exp.key);
        match att {
            Some(a) if claim_matches(&exp.value, &a.value) => {}
            Some(a) => deltas.push(ClaimDelta {
                key: exp.key.clone(),
                expected: Some(exp.value.clone()),
                attested: Some(a.value.clone()),
            }),
            // An expected value may list `absent` as an alternative — a row
            // a newer manifest schema introduced, which an older schema this
            // build still accepts does not carry. Nothing else may go
            // missing silently.
            None if claim_matches(&exp.value, ABSENT_CLAIM) => {}
            None => deltas.push(ClaimDelta {
                key: exp.key.clone(),
                expected: Some(exp.value.clone()),
                attested: None,
            }),
        }
    }
    for att in attested {
        if !expected.iter().any(|e| e.key == att.key) {
            deltas.push(ClaimDelta {
                key: att.key.clone(),
                expected: None,
                attested: Some(att.value.clone()),
            });
        }
    }
    deltas
}

// ---------------------------------------------------------------------------
// Tests — claim extraction/comparison and state persistence. The full
// matrix (network + crypto) is exercised in `tests/updates_check.rs`
// against wiremock-served fixture release feeds.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// An RNG that always draws the same word, so the delay's arithmetic can
    /// be read at its ends.
    struct Fixed(u64);

    impl rand_core::RngCore for Fixed {
        fn next_u32(&mut self) -> u32 {
            self.0 as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(0);
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            dest.fill(0);
            Ok(())
        }
    }

    #[test]
    fn the_poll_delay_stays_within_a_quarter_of_the_interval() {
        let interval = POLL_INTERVAL.as_millis() as u64;
        let spread = interval / 2;
        let shortest = next_poll_delay(&mut Fixed(0)).as_millis() as u64;
        let longest = next_poll_delay(&mut Fixed(spread)).as_millis() as u64;
        assert_eq!(shortest, interval - interval / 4);
        assert_eq!(longest, interval + interval / 4);
        for word in [1, spread / 2, spread + 1, u64::MAX] {
            let ms = next_poll_delay(&mut Fixed(word)).as_millis() as u64;
            assert!((shortest..=longest).contains(&ms), "{word} drew {ms}ms");
        }
    }

    fn current_manifest() -> serde_json::Value {
        serde_json::json!({
            "artifacts": {
                "eidola-cli": {"digest": "sha256:aa", "platform": "linux/amd64", "type": "oci"},
                "eidola-cli-macos-universal": {
                    "archiveSha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "narHash": "sha256-aa",
                    "platform": "darwin/universal",
                    "type": "nix"
                },
                "eidola-gui-linux-amd64": {
                    "archiveSha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "narHash": "sha256-cc",
                    "platform": "linux/amd64",
                    "type": "nix"
                },
                "eidola-gui-macos-universal": {
                    "archiveSha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "narHash": "sha256-bb",
                    "platform": "darwin/universal",
                    "type": "nix"
                },
                "eidola-postgres": {"digest": "sha256:bb", "platform": "linux/amd64", "type": "oci"},
                "eidola-server": {"digest": "sha256:cc", "platform": "linux/amd64", "type": "oci"}
            },
            "enclave": {
                "cvm_version": format!("0.15.3-rc.1@sha256:{}", "d".repeat(64)),
                "igvm_sha256": "e".repeat(64),
                "sev_snp": {
                    "guest_policy": "0x30133",
                    "host_data": "b".repeat(64),
                    "measurement": "a".repeat(96)
                }
            },
            "schema_version": 2
        })
    }

    #[test]
    fn expected_claims_documented_set() {
        let claims = expected_claims();
        let keys: Vec<&str> = claims.iter().map(|c| c.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "manifest.schema_version",
                "enclave.cvm_version",
                "enclave.igvm_sha256",
                "enclave.sev_snp.measurement",
                "enclave.sev_snp.host_data",
                "enclave.sev_snp.guest_policy",
                "artifacts.eidola-cli",
                "artifacts.eidola-gui-macos-universal",
                "artifacts.eidola-postgres",
                "artifacts.eidola-server",
                "artifacts.eidola-gui-linux-amd64",
                "artifacts.eidola-cli-macos-universal",
                "artifacts.eidola-gui-linux-deb-amd64",
                "artifacts.eidola-gui-linux-deb-arm64",
                "artifacts.eidola-gui-linux-nix-amd64",
                "artifacts.eidola-gui-macos-universal-zip",
                "artifacts.eidola-server-engine",
                "artifacts.eidola-cli-linux-static-amd64",
                "artifacts.eidola-cli-linux-static-arm64",
            ]
        );
    }

    #[test]
    fn schema_conditional_rows_tighten_when_a_schema_leaves_the_supported_set() {
        // The property both conditional lists promise: no edit here when a
        // version is dropped. Accepting only schema 3 must make its rows
        // unconditional and must drop the retired key outright — a client
        // that no longer reads schema 2 has no business expecting a row
        // only schema 2 records.
        let only_3 = expected_claims_for(&[3]);
        let value = |claims: &[Claim], key: &str| {
            claims
                .iter()
                .find(|c| c.key == key)
                .map(|c| c.value.clone())
        };
        assert_eq!(
            value(&only_3, "artifacts.eidola-gui-linux-nix-amd64").as_deref(),
            Some("nix (linux/amd64)")
        );
        assert_eq!(
            value(&only_3, "artifacts.eidola-gui-linux-deb-arm64").as_deref(),
            Some("file (linux/arm64)")
        );
        assert_eq!(value(&only_3, "artifacts.eidola-gui-linux-amd64"), None);

        // And the far side of the same rotation, for completeness: a build
        // that accepted only schema 2 expects only the old spelling.
        let only_2 = expected_claims_for(&[2]);
        assert_eq!(
            value(&only_2, "artifacts.eidola-gui-linux-amd64").as_deref(),
            Some("nix (linux/amd64)")
        );
        assert_eq!(value(&only_2, "artifacts.eidola-gui-linux-nix-amd64"), None);
        assert_eq!(
            value(&only_2, "artifacts.eidola-gui-macos-universal-zip"),
            None
        );
        assert_eq!(value(&only_2, "artifacts.eidola-server-engine"), None);

        // The engine image's own boundary, one schema later: a build that
        // accepts only schema 3 does not expect it, one that accepts 3 and 4
        // tolerates its absence, and one that accepts only 4 requires it.
        assert_eq!(value(&only_3, "artifacts.eidola-server-engine"), None);
        assert_eq!(
            value(
                &expected_claims_for(&[3, 4]),
                "artifacts.eidola-server-engine"
            )
            .as_deref(),
            Some("oci (linux/amd64) or absent")
        );
        let only_4 = expected_claims_for(&[4]);
        assert_eq!(
            value(&only_4, "artifacts.eidola-server-engine").as_deref(),
            Some("oci (linux/amd64)")
        );
        assert_eq!(
            value(&only_4, "artifacts.eidola-gui-linux-nix-amd64").as_deref(),
            Some("nix (linux/amd64)")
        );
        assert_eq!(value(&only_4, "artifacts.eidola-gui-linux-amd64"), None);

        // The command-line client's boundary, at schema 5: one row retires
        // and two arrive there, so each tightens from its own side.
        assert_eq!(
            value(&only_4, "artifacts.eidola-cli-macos-universal").as_deref(),
            Some("nix (darwin/universal)")
        );
        let four_and_five = expected_claims_for(&[4, 5]);
        assert_eq!(
            value(&four_and_five, "artifacts.eidola-cli-macos-universal").as_deref(),
            Some("nix (darwin/universal) or absent")
        );
        let only_5 = expected_claims_for(&[5]);
        assert_eq!(value(&only_5, "artifacts.eidola-cli-macos-universal"), None);
        for (key, platform) in [
            ("artifacts.eidola-cli-linux-static-amd64", "linux/amd64"),
            ("artifacts.eidola-cli-linux-static-arm64", "linux/arm64"),
        ] {
            assert_eq!(value(&only_4, key), None, "{key}");
            assert_eq!(
                value(&four_and_five, key).as_deref(),
                Some(format!("file ({platform}) or absent").as_str()),
                "{key}"
            );
            assert_eq!(
                value(&only_5, key).as_deref(),
                Some(format!("file ({platform})").as_str()),
                "{key}"
            );
        }
        assert_eq!(
            value(&only_5, "artifacts.eidola-server-engine").as_deref(),
            Some("oci (linux/amd64)")
        );
        assert_eq!(
            value(&only_5, "artifacts.eidola-cli").as_deref(),
            Some("oci (linux/amd64)")
        );
    }

    /// A schema-3 manifest — the shape a later release will emit. It adds
    /// the macOS unsigned shipping zip and both Debian packages, and moves
    /// the Linux Nix installable to its narrowed key; it must read as a
    /// clean update, not a claims change.
    fn schema_3_manifest() -> serde_json::Value {
        let mut manifest = current_manifest();
        manifest["schema_version"] = serde_json::json!(3);
        let artifacts = manifest["artifacts"].as_object_mut().unwrap();
        artifacts.remove("eidola-gui-linux-amd64");
        artifacts.insert(
            "eidola-gui-linux-nix-amd64".into(),
            serde_json::json!({
                "archiveSha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "narHash": "sha256-cc",
                "platform": "linux/amd64",
                "type": "nix"
            }),
        );
        for (key, platform, hash) in [
            ("eidola-gui-linux-deb-amd64", "linux/amd64", "e"),
            ("eidola-gui-linux-deb-arm64", "linux/arm64", "f"),
        ] {
            artifacts.insert(
                key.into(),
                serde_json::json!({
                    "platform": platform,
                    "sha256": format!("sha256:{}", hash.repeat(64)),
                    "type": "file"
                }),
            );
        }
        artifacts.insert(
            "eidola-gui-macos-universal-zip".into(),
            serde_json::json!({
                "platform": "darwin/universal",
                "sha256": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "type": "file"
            }),
        );
        manifest
    }

    /// A schema-4 manifest: schema 3's shape plus the inference node's image.
    fn schema_4_manifest() -> serde_json::Value {
        let mut manifest = schema_3_manifest();
        manifest["schema_version"] = serde_json::json!(4);
        manifest["artifacts"]["eidola-server-engine"] =
            serde_json::json!({"digest": "sha256:dd", "platform": "linux/amd64", "type": "oci"});
        manifest
    }

    /// A schema-5 manifest: schema 4's shape without the standalone macOS
    /// CLI archive, and with the static Linux CLI, for each architecture, as
    /// a file of its own.
    fn schema_5_manifest() -> serde_json::Value {
        let mut manifest = schema_4_manifest();
        manifest["schema_version"] = serde_json::json!(5);
        let artifacts = manifest["artifacts"].as_object_mut().unwrap();
        artifacts.remove("eidola-cli-macos-universal");
        for (key, platform) in [
            ("eidola-cli-linux-static-amd64", "linux/amd64"),
            ("eidola-cli-linux-static-arm64", "linux/arm64"),
        ] {
            artifacts.insert(
                key.into(),
                serde_json::json!({
                    "platform": platform,
                    "sha256": format!("sha256:{}", "9".repeat(64)),
                    "type": "file"
                }),
            );
        }
        manifest
    }

    /// The command-line client's rows as a schema-4 and a schema-5 manifest
    /// record them, for the tests that move one across the boundary.
    fn macos_cli_row() -> serde_json::Value {
        schema_4_manifest()["artifacts"]["eidola-cli-macos-universal"].clone()
    }

    fn static_cli_row(key: &str) -> serde_json::Value {
        schema_5_manifest()["artifacts"][key].clone()
    }

    /// The static Linux CLI keys schema 5 adds, one per architecture.
    fn static_cli_keys() -> impl Iterator<Item = &'static str> {
        EXPECTED_ARTIFACTS_SINCE_SCHEMA_5
            .iter()
            .map(|(key, _, _)| *key)
    }

    #[test]
    fn schema_5_adds_a_static_cli_for_each_architecture() {
        assert_eq!(
            EXPECTED_ARTIFACTS_SINCE_SCHEMA_5,
            [
                ("eidola-cli-linux-static-amd64", "file", "linux/amd64"),
                ("eidola-cli-linux-static-arm64", "file", "linux/arm64"),
            ]
        );
    }

    #[test]
    fn schema_5_manifest_with_its_own_rows_is_not_a_claims_change() {
        let deltas = compare_claims(&expected_claims(), &attested_claims(&schema_5_manifest()));
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn schema_5_manifest_missing_a_static_cli_is_a_delta() {
        // Each architecture's binary is promised on its own: the other one
        // being present does not cover for it.
        for key in static_cli_keys() {
            let mut manifest = schema_5_manifest();
            manifest["artifacts"].as_object_mut().unwrap().remove(key);
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{key}: unexpected deltas: {deltas:#?}");
            assert_eq!(deltas[0].key, format!("artifacts.{key}"));
            assert_eq!(
                deltas[0].attested.as_deref(),
                Some("missing (recorded from schema 5 on)"),
                "{key}"
            );
        }
    }

    #[test]
    fn a_static_cli_before_schema_5_is_a_delta() {
        // Emission without the bump, at every older schema this build reads,
        // for each architecture's binary.
        for key in static_cli_keys() {
            for (mut manifest, declared) in [
                (current_manifest(), 2),
                (schema_3_manifest(), 3),
                (schema_4_manifest(), 4),
            ] {
                manifest["artifacts"][key] = static_cli_row(key);
                let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
                assert_eq!(deltas.len(), 1, "{key}, schema {declared}: {deltas:#?}");
                assert_eq!(deltas[0].key, format!("artifacts.{key}"));
                assert_eq!(
                    deltas[0].attested.as_deref(),
                    Some(
                        format!("present, but schema {declared} records it only from schema 5 on")
                            .as_str()
                    ),
                    "{key}"
                );
            }
        }
    }

    #[test]
    fn a_static_cli_under_the_other_architecture_is_a_delta() {
        // The platform is part of the claim: an arm64 key recording an
        // amd64 binary is not the row schema 5 promises.
        let mut manifest = schema_5_manifest();
        manifest["artifacts"]["eidola-cli-linux-static-arm64"]["platform"] =
            serde_json::json!("linux/amd64");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-cli-linux-static-arm64");
    }

    #[test]
    fn the_macos_cli_at_schema_5_is_a_delta() {
        // Schema 5 retires the row, so a manifest declaring 5 that still
        // carries it has recorded an artifact its own schema does not have —
        // and the static Linux CLI beside it does not excuse it.
        let mut manifest = schema_5_manifest();
        manifest["artifacts"]["eidola-cli-macos-universal"] = macos_cli_row();
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-cli-macos-universal");
        assert_eq!(
            deltas[0].attested.as_deref(),
            Some("present, but schema 5 records it only through schema 4")
        );
    }

    #[test]
    fn a_manifest_before_schema_5_missing_the_macos_cli_is_a_delta() {
        // The "or absent" on the expected side covers schema 5 having
        // retired the row, not an older schema having lost it.
        for (mut manifest, declared) in [
            (current_manifest(), 2),
            (schema_3_manifest(), 3),
            (schema_4_manifest(), 4),
        ] {
            manifest["artifacts"]
                .as_object_mut()
                .unwrap()
                .remove("eidola-cli-macos-universal");
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "schema {declared}: deltas {deltas:#?}");
            assert_eq!(deltas[0].key, "artifacts.eidola-cli-macos-universal");
            assert_eq!(
                deltas[0].attested.as_deref(),
                Some("missing (recorded through schema 4)"),
                "schema {declared}"
            );
        }
    }

    #[test]
    fn schema_5_manifest_missing_an_older_arriving_row_is_a_delta() {
        // Schema 5 is schema 4's set with the CLI rows moved, not instead of
        // it: every row that arrived earlier is still promised.
        for (since, rows) in [
            (3, EXPECTED_ARTIFACTS_SINCE_SCHEMA_3),
            (4, EXPECTED_ARTIFACTS_SINCE_SCHEMA_4),
        ] {
            for (row, _, _) in rows {
                let mut manifest = schema_5_manifest();
                manifest["artifacts"].as_object_mut().unwrap().remove(*row);
                let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
                assert_eq!(deltas.len(), 1, "{row}: unexpected deltas: {deltas:#?}");
                assert_eq!(
                    deltas[0].attested.as_deref(),
                    Some(format!("missing (recorded from schema {since} on)").as_str()),
                    "{row}"
                );
            }
        }
    }

    #[test]
    fn the_cli_rows_swapped_without_the_bump_are_each_a_delta() {
        // The whole schema-5 artifact set under schema 4's number: the
        // retirement and each arrival read as a claim out of shape, so no
        // part of the rotation can be emitted alone.
        let mut manifest = schema_5_manifest();
        manifest["schema_version"] = serde_json::json!(4);
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        let mut got: Vec<(&str, Option<&str>)> = deltas
            .iter()
            .map(|d| (d.key.as_str(), d.attested.as_deref()))
            .collect();
        got.sort();
        assert_eq!(
            got,
            [
                (
                    "artifacts.eidola-cli-linux-static-amd64",
                    Some("present, but schema 4 records it only from schema 5 on")
                ),
                (
                    "artifacts.eidola-cli-linux-static-arm64",
                    Some("present, but schema 4 records it only from schema 5 on")
                ),
                (
                    "artifacts.eidola-cli-macos-universal",
                    Some("missing (recorded through schema 4)")
                ),
            ]
        );
    }

    #[test]
    fn schema_4_manifest_with_its_own_rows_is_not_a_claims_change() {
        let deltas = compare_claims(&expected_claims(), &attested_claims(&schema_4_manifest()));
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn schema_4_manifest_missing_the_engine_image_is_a_delta() {
        let mut manifest = schema_4_manifest();
        manifest["artifacts"]
            .as_object_mut()
            .unwrap()
            .remove("eidola-server-engine");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-server-engine");
        assert_eq!(
            deltas[0].attested.as_deref(),
            Some("missing (recorded from schema 4 on)")
        );
    }

    #[test]
    fn the_engine_image_before_schema_4_is_a_delta() {
        // Emission without the bump, at both older schemas this build reads.
        for (mut manifest, declared) in [(current_manifest(), 2), (schema_3_manifest(), 3)] {
            manifest["artifacts"]["eidola-server-engine"] =
                schema_4_manifest()["artifacts"]["eidola-server-engine"].clone();
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "schema {declared}: deltas {deltas:#?}");
            assert_eq!(deltas[0].key, "artifacts.eidola-server-engine");
            assert_eq!(
                deltas[0].attested.as_deref(),
                Some(
                    format!("present, but schema {declared} records it only from schema 4 on")
                        .as_str()
                )
            );
        }
    }

    #[test]
    fn schema_4_manifest_missing_a_schema_3_row_is_a_delta() {
        // Schema 4 is schema 3 plus a row, not instead of it.
        for (row, _, _) in EXPECTED_ARTIFACTS_SINCE_SCHEMA_3 {
            let mut manifest = schema_4_manifest();
            manifest["artifacts"].as_object_mut().unwrap().remove(*row);
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{row}: unexpected deltas: {deltas:#?}");
            assert_eq!(
                deltas[0].attested.as_deref(),
                Some("missing (recorded from schema 3 on)"),
                "{row}"
            );
        }
    }

    #[test]
    fn schema_3_manifest_with_its_own_rows_is_not_a_claims_change() {
        let deltas = compare_claims(&expected_claims(), &attested_claims(&schema_3_manifest()));
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn schema_2_manifest_without_the_schema_3_rows_is_not_a_claims_change() {
        // Accept-before-emit: this build accepts schema 3, while releases
        // keep emitting schema 2 until the release after. If the new row
        // were required unconditionally, every release in that window would
        // read as ClaimsChanged — the failure mode the rotation exists to
        // avoid.
        let manifest = current_manifest();
        assert_eq!(manifest["schema_version"], serde_json::json!(2));
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn schema_3_manifest_missing_a_schema_3_row_is_a_delta() {
        // The other half of the tolerance: a manifest that declares schema 3
        // has promised each row, so dropping one is a weakened claim rather
        // than an older shape.
        for (row, _, _) in EXPECTED_ARTIFACTS_SINCE_SCHEMA_3 {
            let mut manifest = schema_3_manifest();
            manifest["artifacts"].as_object_mut().unwrap().remove(*row);
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{row}: unexpected deltas: {deltas:#?}");
            assert_eq!(deltas[0].key, format!("artifacts.{row}"));
            assert_eq!(
                deltas[0].attested.as_deref(),
                Some("missing (recorded from schema 3 on)"),
                "{row}"
            );
        }
    }

    #[test]
    fn schema_3_rows_present_before_schema_3_are_deltas() {
        // The other side of the rotation. Accept-before-emit means a
        // release must not start recording a row until the version says
        // so — emission without the bump has to be as loud as the bump
        // without emission, or "accepted but not yet emitted" is a claim
        // nothing enforces.
        for (row, _, _) in EXPECTED_ARTIFACTS_SINCE_SCHEMA_3 {
            let mut manifest = current_manifest();
            manifest["artifacts"][*row] = schema_3_manifest()["artifacts"][*row].clone();
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{row}: unexpected deltas: {deltas:#?}");
            assert_eq!(deltas[0].key, format!("artifacts.{row}"));
            assert_eq!(
                deltas[0].attested.as_deref(),
                Some("present, but schema 2 records it only from schema 3 on"),
                "{row}"
            );
        }
    }

    #[test]
    fn the_retired_linux_key_at_schema_3_is_a_delta() {
        // The rename's own accept-before-emit edge, and the reason the
        // retiring row exists at all: schema 3 records the Linux Nix
        // installable as `…-nix-amd64`, so a manifest declaring 3 that
        // still carries the old key has recorded an artifact its own schema
        // does not have. Without this the rename would be readable as a
        // pure addition, and a manifest could quietly claim both.
        let mut manifest = schema_3_manifest();
        manifest["artifacts"]["eidola-gui-linux-amd64"] = serde_json::json!({
            "archiveSha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "narHash": "sha256-cc",
            "platform": "linux/amd64",
            "type": "nix"
        });
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-linux-amd64");
        assert_eq!(
            deltas[0].attested.as_deref(),
            Some("present, but schema 3 records it only through schema 2")
        );
    }

    #[test]
    fn retired_keys_get_no_verdict_below_the_oldest_accepted_schema() {
        // A retiring row is bounded on both sides. The captured schema-1
        // release predates the Linux GUI entirely, so neither its absence
        // nor its presence there says anything about the rename — and the
        // same holds for the macOS CLI row schema 5 retires. The only
        // honest complaint about such a manifest is its schema version, and
        // piling row verdicts on top would make every captured old release
        // read as several problems instead of one.
        for present in [false, true] {
            let mut manifest = current_manifest();
            manifest["schema_version"] = serde_json::json!(1);
            if !present {
                let artifacts = manifest["artifacts"].as_object_mut().unwrap();
                artifacts.remove("eidola-gui-linux-amd64");
                artifacts.remove("eidola-cli-macos-universal");
            }
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "present={present}: deltas {deltas:#?}");
            assert_eq!(deltas[0].key, "manifest.schema_version");
        }
    }

    #[test]
    fn schema_2_manifest_missing_the_retired_linux_key_is_a_delta() {
        // And its mirror: while schema 2 is still accepted, a schema-2
        // manifest that drops the row has dropped an artifact. The "or
        // absent" tolerance covers schema 3 having renamed it, not schema 2
        // having lost it.
        let mut manifest = current_manifest();
        manifest["artifacts"]
            .as_object_mut()
            .unwrap()
            .remove("eidola-gui-linux-amd64");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-linux-amd64");
        assert_eq!(
            deltas[0].attested.as_deref(),
            Some("missing (recorded through schema 2)")
        );
    }

    #[test]
    fn unrecognized_field_on_any_artifact_row_is_a_delta() {
        // Rows are exact key sets on this side too, for every type — the
        // rule the determinism gate applies to the committed manifest. An
        // extra per-row field is how an unreviewed claim would arrive.
        for (row, field, expected) in [
            (
                "eidola-gui-macos-universal-zip",
                "appleTeamId",
                "file (darwin/universal) — unrecognized field(s): appleTeamId",
            ),
            (
                "eidola-gui-macos-universal",
                "appleSignatureSha256",
                "nix (darwin/universal) — unrecognized field(s): appleSignatureSha256",
            ),
            (
                "eidola-server",
                "notarizationTicket",
                "oci (linux/amd64) — unrecognized field(s): notarizationTicket",
            ),
        ] {
            let mut manifest = schema_3_manifest();
            manifest["artifacts"][row][field] = serde_json::json!("x");
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{row}: unexpected deltas: {deltas:#?}");
            assert_eq!(deltas[0].key, format!("artifacts.{row}"));
            assert_eq!(deltas[0].attested.as_deref(), Some(expected), "{row}");
        }
    }

    #[test]
    fn malformed_file_sha256_is_a_delta() {
        // Every `file` row's whole claim is one hash of one published byte
        // stream — the shipping zip and both Debian packages alike — so a
        // hash that isn't one is the row saying nothing.
        for (row, ty, _) in EXPECTED_ARTIFACTS_SINCE_SCHEMA_3
            .iter()
            .chain(EXPECTED_ARTIFACTS_SINCE_SCHEMA_5)
        {
            if *ty != "file" {
                continue;
            }
            let mut manifest = schema_5_manifest();
            manifest["artifacts"][*row]["sha256"] = serde_json::json!("sha256:nope");
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{row}: unexpected deltas: {deltas:#?}");
            assert_eq!(deltas[0].key, format!("artifacts.{row}"));
            assert!(
                deltas[0].attested.as_deref().unwrap().contains("malformed"),
                "{row}: got {deltas:?}"
            );
        }
    }

    #[test]
    fn current_manifest_shape_matches_expected_claims() {
        let attested = attested_claims(&current_manifest());
        let deltas = compare_claims(&expected_claims(), &attested);
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn committed_workspace_manifest_matches_expected_claims() {
        // The real artifact-manifest.json at the workspace root must match
        // the expected claim set — if this fails, either the manifest
        // schema moved without updating `expected_claims`, or vice versa.
        let bytes = include_bytes!("../../../artifact-manifest.json");
        let manifest: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn schema_version_jump_is_a_delta() {
        // Built on the newest shape this build knows, so the schema row is
        // the only thing that can differ.
        let mut manifest = schema_5_manifest();
        manifest["schema_version"] = serde_json::json!(99);
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "manifest.schema_version");
        assert_eq!(deltas[0].expected.as_deref(), Some("2 or 3 or 4 or 5"));
        assert_eq!(deltas[0].attested.as_deref(), Some("99"));
    }

    #[test]
    fn missing_archive_sha256_is_still_nix_on_schema_1() {
        // Schema-1 manifests predate archiveSha256, so its absence there is
        // not a defect: a captured old release must not read as a claims
        // change on that axis.
        let mut manifest = current_manifest();
        manifest["schema_version"] = serde_json::json!(1);
        manifest["artifacts"]["eidola-gui-macos-universal"]
            .as_object_mut()
            .unwrap()
            .remove("archiveSha256");
        // Only the schema_version row differs from this build's expectation.
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "manifest.schema_version");
    }

    #[test]
    fn missing_archive_sha256_is_a_delta_on_schema_2() {
        // From schema 2 on, archiveSha256 is the only hash a user can check
        // without Nix. A manifest that drops it has weakened the claim and
        // must not pass silently.
        let mut manifest = current_manifest();
        manifest["artifacts"]["eidola-gui-macos-universal"]
            .as_object_mut()
            .unwrap()
            .remove("archiveSha256");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-macos-universal");
        assert!(
            deltas[0]
                .attested
                .as_deref()
                .unwrap()
                .contains("archiveSha256"),
            "got: {deltas:?}"
        );
    }

    #[test]
    fn multi_version_schema_support_matches_any_listed_version() {
        // `expected_claims` renders SUPPORTED_MANIFEST_SCHEMA_VERSIONS as
        // "a or b". That set is only real if the comparison honors it —
        // under plain equality a list of length > 1 would make
        // manifest.schema_version a permanent ClaimsChanged for every
        // release. Guarding the comparison directly keeps the affordance
        // usable the next time a version is added.
        assert!(claim_matches("1 or 2", "1"));
        assert!(claim_matches("1 or 2", "2"));
        assert!(!claim_matches("1 or 2", "3"));
        assert!(claim_matches("2", "2"));
        assert!(!claim_matches("2", "1"));
    }

    #[test]
    fn malformed_archive_sha256_is_a_delta() {
        let mut manifest = current_manifest();
        manifest["artifacts"]["eidola-gui-macos-universal"]["archiveSha256"] =
            serde_json::json!("not-a-digest");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-macos-universal");
        assert!(
            deltas[0]
                .attested
                .as_deref()
                .unwrap()
                .contains("archiveSha256"),
            "got: {deltas:?}"
        );
    }

    #[test]
    fn disappearing_enclave_platform_is_a_delta() {
        let mut manifest = current_manifest();
        manifest["enclave"]
            .as_object_mut()
            .unwrap()
            .remove("sev_snp");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        let keys: Vec<&str> = deltas.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "enclave.sev_snp.measurement",
                "enclave.sev_snp.host_data",
                "enclave.sev_snp.guest_policy"
            ]
        );
        assert!(
            deltas.iter().all(|d| d.attested.is_none()),
            "claims should be absent"
        );
    }

    /// A manifest from before the IGVM launch model (the captured v0.0.8
    /// release's enclave block) states a different server identity: every
    /// field it carries is unrecognized and every field this build pins is
    /// missing, which is the "threat model changed" answer, not an update.
    #[test]
    fn a_kernel_cmdline_enclave_block_is_a_changed_threat_model() {
        let mut manifest = current_manifest();
        manifest["enclave"] = serde_json::json!({
            "cmdline": "readonly=on root=/dev/mapper/root",
            "snp_measurement": "a".repeat(96),
            "tdx_measurement": {"rtmr1": "b".repeat(96), "rtmr2": "c".repeat(96)}
        });
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        for key in [
            "enclave.cvm_version",
            "enclave.sev_snp.host_data",
            "enclave.sev_snp.measurement",
        ] {
            assert!(
                deltas.iter().any(|d| d.key == key && d.attested.is_none()),
                "{key}: {deltas:?}"
            );
        }
        for key in [
            "enclave.cmdline",
            "enclave.snp_measurement",
            "enclave.tdx_measurement",
        ] {
            assert!(
                deltas.iter().any(|d| d.key == key && d.expected.is_none()),
                "{key}: {deltas:?}"
            );
        }
    }

    #[test]
    fn an_unrecognized_sev_snp_field_is_a_delta() {
        let mut manifest = current_manifest();
        manifest["enclave"]["sev_snp"]["vmpl"] = serde_json::json!(0);
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "enclave.sev_snp.vmpl");
    }

    #[test]
    fn unrecognized_artifact_type_is_a_delta() {
        let mut manifest = current_manifest();
        manifest["artifacts"]["eidola-server"]["type"] = serde_json::json!("flatpak");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "artifacts.eidola-server");
        assert!(
            deltas[0].attested.as_deref().unwrap().contains("flatpak"),
            "got: {deltas:?}"
        );
    }

    #[test]
    fn extra_artifact_entry_is_a_delta() {
        let mut manifest = current_manifest();
        manifest["artifacts"]["eidola-android"] = serde_json::json!({
            "digest": "sha256:dd", "platform": "android/arm64", "type": "oci"
        });
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "artifacts.eidola-android");
        assert!(deltas[0].expected.is_none());
    }

    #[test]
    fn unrecognized_top_level_field_is_a_delta() {
        let mut manifest = current_manifest();
        manifest["install_hooks"] = serde_json::json!(["curl | sh"]);
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "manifest.install_hooks");
        assert_eq!(deltas[0].attested.as_deref(), Some("unrecognized field"));
    }

    #[test]
    fn malformed_measurement_is_a_delta() {
        for (field, value) in [
            ("measurement", serde_json::json!("deadbeef")),
            ("host_data", serde_json::json!("a".repeat(96))),
            ("guest_policy", serde_json::json!("196915")),
        ] {
            let mut manifest = current_manifest();
            manifest["enclave"]["sev_snp"][field] = value;
            let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
            assert_eq!(deltas.len(), 1, "{field}");
            assert!(
                deltas[0].attested.as_deref().unwrap().contains("malformed"),
                "{field}: {deltas:?}"
            );
        }
        let mut manifest = current_manifest();
        manifest["enclave"]["cvm_version"] = serde_json::json!("0.15.3-rc.1");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert!(deltas[0].attested.as_deref().unwrap().contains("malformed"));
    }

    #[test]
    fn non_object_manifest_is_all_deltas() {
        let attested = attested_claims(&serde_json::json!("just a string"));
        assert_eq!(attested.len(), 1);
        assert_eq!(attested[0].key, "manifest");
        let deltas = compare_claims(&expected_claims(), &attested);
        // Every expected claim missing — except the rows whose absence an
        // older accepted schema makes legitimate — plus the one "not a JSON
        // object" extra.
        let tolerated = expected_claims()
            .iter()
            .filter(|c| c.value.ends_with(ABSENT_CLAIM))
            .count();
        assert_eq!(deltas.len(), expected_claims().len() - tolerated + 1);
    }

    #[test]
    fn update_state_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = UpdateState::default();
        state.absorb(UpdateCheckSnapshot {
            checked_at_ms: 1234,
            result: UpdateCheckResult::UpToDate {
                latest_version: Some("0.0.8".into()),
            },
        });
        state.accepted = Some(AcceptedClaims {
            version: "0.2.0".into(),
            manifest_sha256: "ab".repeat(32),
            accepted_at_ms: 5678,
        });
        save_state(dir.path(), &state).unwrap();
        let loaded = load_state(dir.path());
        assert_eq!(loaded, state);
    }

    /// A standing claims-changed verdict with an acceptance recorded for
    /// it — the state that can promote itself to an available update.
    fn state_with_accepted_claims_change() -> UpdateState {
        let release = VerifiedRelease {
            version: "9.9.9".into(),
            tag: "v9.9.9".into(),
            git_commit: "c".repeat(40),
            release_url: None,
            published_at: None,
            ci_identity: "x".into(),
            rekor_log_index: 1,
            attestants: vec![ReleaseAttestant {
                name: "Ada Example".into(),
                jurisdiction: "Somewhere".into(),
                rekor_log_index: 2,
            }],
            manifest_sha256: "ab".repeat(32),
            claims_accepted: false,
        };
        UpdateState {
            last: Some(UpdateCheckSnapshot {
                checked_at_ms: 1,
                result: UpdateCheckResult::ClaimsChanged {
                    release,
                    comparison: ClaimsComparison {
                        expected: vec![],
                        attested: vec![],
                        deltas: vec![ClaimDelta {
                            key: "x".into(),
                            expected: Some("a".into()),
                            attested: None,
                        }],
                    },
                },
            }),
            accepted: Some(AcceptedClaims {
                version: "9.9.9".into(),
                manifest_sha256: "ab".repeat(32),
                accepted_at_ms: 2,
            }),
            ..UpdateState::default()
        }
    }

    #[test]
    fn state_reached_under_the_current_rules_loads() {
        let dir = tempfile::tempdir().unwrap();
        let state = state_with_accepted_claims_change();
        save_state(dir.path(), &state).unwrap();
        assert_eq!(load_state(dir.path()), state);
    }

    /// State written before verdicts required a human attestation holds
    /// releases CI's signature alone vouched for. None of it may be shown,
    /// held against a later failure, or promoted by its acceptance.
    #[test]
    fn state_from_before_human_attestation_is_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let mut legacy = serde_json::to_value(state_with_accepted_claims_change()).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.remove("verdict_rules").unwrap();
        let release = object["last"]["result"]["release"].as_object_mut().unwrap();
        release.remove("attestants").unwrap();
        release.remove("git_commit").unwrap();
        std::fs::write(state_path(dir.path()), serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert_eq!(load_state(dir.path()), UpdateState::default());
    }

    /// The stamp alone decides it: a release shaped like today's, under
    /// rules other than today's, is still discarded.
    #[test]
    fn state_stamped_with_other_rules_is_discarded() {
        let dir = tempfile::tempdir().unwrap();
        for rules in [0, VERDICT_RULES + 1] {
            let state = UpdateState {
                verdict_rules: rules,
                ..state_with_accepted_claims_change()
            };
            save_state(dir.path(), &state).unwrap();
            assert_eq!(load_state(dir.path()), UpdateState::default(), "{rules}");
        }
    }

    #[test]
    fn load_state_defaults_on_missing_or_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_state(dir.path()), UpdateState::default());
        std::fs::write(state_path(dir.path()), b"{{{not json").unwrap();
        assert_eq!(load_state(dir.path()), UpdateState::default());
    }

    #[test]
    fn check_failed_does_not_clear_standing_unverifiable() {
        let mut state = UpdateState::default();
        let alert = UpdateCheckSnapshot {
            checked_at_ms: 1,
            result: UpdateCheckResult::Unverifiable {
                version: "9.9.9".into(),
                tag: "v9.9.9".into(),
                reason: "signature not from the pinned release identity".into(),
            },
        };
        state.absorb(alert.clone());
        state.absorb(UpdateCheckSnapshot {
            checked_at_ms: 2,
            result: UpdateCheckResult::CheckFailed {
                message: "offline".into(),
            },
        });
        assert_eq!(
            state.last,
            Some(alert),
            "offline blip must not clear the alert"
        );

        // A *completed* later check does clear it.
        let ok = UpdateCheckSnapshot {
            checked_at_ms: 3,
            result: UpdateCheckResult::UpToDate {
                latest_version: Some("9.9.9".into()),
            },
        };
        state.absorb(ok.clone());
        assert_eq!(state.last, Some(ok));
    }

    #[test]
    fn check_failed_does_not_clear_standing_claims_changed() {
        let mut state = UpdateState::default();
        let release = VerifiedRelease {
            version: "9.9.9".into(),
            tag: "v9.9.9".into(),
            git_commit: "c".repeat(40),
            release_url: None,
            published_at: None,
            ci_identity: "x".into(),
            rekor_log_index: 1,
            attestants: vec![],
            manifest_sha256: "ab".repeat(32),
            claims_accepted: false,
        };
        let alert = UpdateCheckSnapshot {
            checked_at_ms: 1,
            result: UpdateCheckResult::ClaimsChanged {
                release,
                comparison: ClaimsComparison {
                    expected: vec![],
                    attested: vec![],
                    deltas: vec![ClaimDelta {
                        key: "x".into(),
                        expected: Some("a".into()),
                        attested: None,
                    }],
                },
            },
        };
        state.absorb(alert.clone());
        state.absorb(UpdateCheckSnapshot {
            checked_at_ms: 2,
            result: UpdateCheckResult::CheckFailed {
                message: "offline".into(),
            },
        });
        assert_eq!(state.last, Some(alert));
    }

    #[test]
    fn check_failed_overwrites_benign_states() {
        let mut state = UpdateState::default();
        state.absorb(UpdateCheckSnapshot {
            checked_at_ms: 1,
            result: UpdateCheckResult::UpToDate {
                latest_version: None,
            },
        });
        let failed = UpdateCheckSnapshot {
            checked_at_ms: 2,
            result: UpdateCheckResult::CheckFailed {
                message: "offline".into(),
            },
        };
        state.absorb(failed.clone());
        assert_eq!(state.last, Some(failed));
    }

    #[test]
    fn result_serde_round_trip() {
        let result = UpdateCheckResult::Unverifiable {
            version: "1.2.3".into(),
            tag: "v1.2.3".into(),
            reason: "bad".into(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: UpdateCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, result);
    }
}
