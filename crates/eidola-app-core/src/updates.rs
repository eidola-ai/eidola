//! Update *notification* flow — "a verified newer release exists; here's
//! the door to it."
//!
//! This module answers one question on launch and every ~6 hours: **is
//! there a newer release of Eidola, and can this binary prove it's real?**
//! It never downloads or installs anything — v1's ceiling is "verified
//! update available → open the release page in the browser." (The full
//! self-update *install* pipeline — `release.json`, human attestations,
//! template equality — lives in [`crate::updater`] and remains the path a
//! future in-app install will take.)
//!
//! ## Source of truth
//!
//! Only the GitHub release **marked `latest`** counts. That marker is
//! applied by the release officer's tooling *after* the human attestation
//! is signed and uploaded (see `.github/workflows/tinfoil-build.yml` — CI
//! deliberately does not mark its release `latest`), so it is the
//! human-attested signal. A `v*` release that exists but isn't `latest` is
//! simply not an update yet. The GitHub `releases/latest` API endpoint
//! returns exactly the release carrying that marker (drafts and
//! prereleases never qualify), and returns 404 when no release carries it
//! — which this module maps to "up to date", not an error.
//!
//! Per release, CI publishes two assets exactly for this verifier:
//! `artifact-manifest.json` and `artifact-manifest.json.sigstore` (a
//! `cosign sign-blob` Sigstore bundle, Fulcio keyless under the
//! `tinfoil-build.yml` workflow's OIDC identity).
//!
//! ## Verification anchor
//!
//! All verification is anchored **exclusively** in the embedded trust root
//! ([`crate::trust_root`]): the pinned Sigstore `TrustedRoot` snapshot
//! (Fulcio CAs + Rekor keys), the pinned CI identity pattern + OIDC
//! issuer, and the schema versions this build was built against. No trust
//! anchor is ever fetched from the network. The cryptographic work is
//! [`crate::updater::ci_sigstore::verify_ci_signature_with`] — full Fulcio
//! chain walk, identity match, ECDSA signature over the manifest hash,
//! Rekor SET + Merkle inclusion proof.
//!
//! On top of the bundle verification, the Fulcio cert's SAN identity is
//! required to end in `@refs/tags/<tag>` for the *same tag* the release
//! claims — so an authentic old manifest can't be replayed under a newer
//! tag to fake an update.
//!
//! ## Failure-mode matrix (product decisions — encoded in [`UpdateCheckResult`])
//!
//! | State | Meaning | Behavior |
//! |---|---|---|
//! | [`UpdateCheckResult::CheckFailed`] | Network/API failure — not a security signal | Quiet. "Couldn't check (offline?)" in the Updates window; retry next cycle. Never alarms. A standing `Unverifiable`/`ClaimsChanged` state is **not** overwritten by a later `CheckFailed` (see [`UpdateState::absorb`]). |
//! | [`UpdateCheckResult::UpToDate`] | No newer `latest` release | Includes "no release is marked latest" (HTTP 404) and "latest is older/equal". |
//! | [`UpdateCheckResult::UpdateAvailable`] | `latest` verifies, version > ours | "Eidola vX.Y.Z is available — cryptographically verified." One action: open the release page. |
//! | [`UpdateCheckResult::Unverifiable`] | `latest` fails cryptographic verification (missing/bad bundle, wrong identity, manifest hash mismatch, tag binding mismatch) | **Possible fake / channel compromise.** Hard, visible security state — never silent, never offers the artifact. `reason` states exactly what failed. Persists until a later check finds a verifiable `latest`. |
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
//! | `manifest.schema_version` | `2 or 3` | [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`] — the manifest shapes this module's parser understands; a jump outside the set is a release-gated trust event (`docs/trust-root.md`) |
//! | `enclave.snp_measurement` | `SEV-SNP launch measurement (48-byte hex)` | the embedded trust root pins a SEV-SNP measurement (`trust_root::SERVER_SNP_MEASUREMENT`), so the paired server's SEV-SNP platform must keep being attested |
//! | `enclave.tdx_measurement.rtmr1` | `TDX runtime measurement (48-byte hex)` | ditto, `trust_root::SERVER_TDX_RTMR1` |
//! | `enclave.tdx_measurement.rtmr2` | `TDX runtime measurement (48-byte hex)` | ditto, `trust_root::SERVER_TDX_RTMR2` |
//! | `enclave.cmdline` | `kernel command line (non-empty)` | manifest schema 2 — the cmdline binds the tinfoil-config hash into the measurement |
//! | `artifacts.eidola-cli` | `oci (linux/amd64)` | [`EXPECTED_ARTIFACTS`] — the artifact set every accepted schema records |
//! | `artifacts.eidola-cli-macos-universal` | `nix (darwin/universal)` | ditto |
//! | `artifacts.eidola-gui-linux-amd64` | `nix (linux/amd64)` | ditto |
//! | `artifacts.eidola-gui-macos-universal` | `nix (darwin/universal)` | ditto |
//! | `artifacts.eidola-postgres` | `oci (linux/amd64)` | ditto |
//! | `artifacts.eidola-server` | `oci (linux/amd64)` | ditto |
//! | `artifacts.eidola-gui-macos-universal-zip` | `file (darwin/universal) or absent` | [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`] — the macOS unsigned shipping zip. Absent from the schema-2 manifests this build also accepts; required of any manifest declaring schema 3 |
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
//! and reads as `ClaimsChanged`. Schema-1 rows legitimately have none. The
//! macOS unsigned shipping zip's row arrives the same way with schema 3.
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
use sha2::{Digest, Sha256};

use crate::error::AppError;
use crate::trust_root;
use crate::updater::ci_sigstore;

/// `artifact-manifest.json` `schema_version` values this build can parse
/// structurally. Distinct from `trust_root::SUPPORTED_RELEASE_SCHEMA_VERSIONS`
/// (which pins `release.json`, the install pipeline's index); this pins the
/// manifest shape [`attested_claims`] walks. A version outside this set is a
/// *claims change* (authentic but unintelligible to this build), not a
/// verification failure.
pub const SUPPORTED_MANIFEST_SCHEMA_VERSIONS: &[u32] = &[2, 3];

/// The schema at which [`EXPECTED_ARTIFACTS_SINCE_SCHEMA_3`] rows become
/// required. Named once so the tolerance and the requirement cannot drift.
const SCHEMA_WITH_MACOS_SHIPPING_ZIP: u64 = 3;

/// The artifact entries every accepted `artifact-manifest.json` records, as
/// `(name, type, platform)`. Structure only — digests / narHashes /
/// archiveSha256 are values and legitimately change every release.
pub const EXPECTED_ARTIFACTS: &[(&str, &str, &str)] = &[
    ("eidola-cli", "oci", "linux/amd64"),
    ("eidola-cli-macos-universal", "nix", "darwin/universal"),
    ("eidola-gui-linux-amd64", "nix", "linux/amd64"),
    ("eidola-gui-macos-universal", "nix", "darwin/universal"),
    ("eidola-postgres", "oci", "linux/amd64"),
    ("eidola-server", "oci", "linux/amd64"),
];

/// Artifact entries schema 3 adds: the macOS unsigned shipping zip, whose
/// `sha256` is the identity of the container a signed release is built from
/// (`type: "file"` — one published file, one hash, no Nix).
///
/// Kept separate from [`EXPECTED_ARTIFACTS`] because this build accepts both
/// schemas: a schema-2 manifest legitimately has no such row, and only a
/// manifest declaring schema 3 or later is required to carry it. When 2
/// leaves [`SUPPORTED_MANIFEST_SCHEMA_VERSIONS`] the row becomes
/// unconditionally required, with no further edit here.
pub const EXPECTED_ARTIFACTS_SINCE_SCHEMA_3: &[(&str, &str, &str)] =
    &[("eidola-gui-macos-universal-zip", "file", "darwin/universal")];

/// How often the background poll re-checks while the app is running.
pub const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

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

/// A release whose `artifact-manifest.json` Sigstore bundle verified
/// against the embedded trust root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedRelease {
    /// Semver, without the leading `v`.
    pub version: String,
    /// The git tag (e.g. `v0.0.9`), cross-checked against the Fulcio
    /// cert's `@refs/tags/…` identity suffix.
    pub tag: String,
    /// The GitHub release page — the only artifact-adjacent link v1 offers.
    pub release_url: Option<String>,
    pub published_at: Option<String>,
    /// The Fulcio cert SAN identity the bundle verified under.
    pub ci_identity: String,
    /// Rekor transparency-log index, for independent lookup.
    pub rekor_log_index: u64,
    /// Hex sha256 of the verified manifest bytes — the durable handle an
    /// accepted claims-change is recorded against.
    pub manifest_sha256: String,
    /// True when this release's claims differed from the expected set but
    /// the user explicitly chose "treat as update" for this exact
    /// manifest (matched by version + manifest hash).
    pub claims_accepted: bool,
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

/// On-disk update-check state (`<data_dir>/update-state.json`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UpdateState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<UpdateCheckSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<AcceptedClaims>,
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

/// Load persisted state; missing or corrupt files yield the default.
pub fn load_state(data_dir: &Path) -> UpdateState {
    let Ok(bytes) = std::fs::read(state_path(data_dir)) else {
        return UpdateState::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
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
    /// Full URL of the latest-release endpoint (GitHub `releases/latest`
    /// shape). Resolved from `Config::update_feed_url()`.
    pub feed_url: String,
    /// The running binary's semver.
    pub installed_version: String,
    /// Fulcio SAN identity glob the bundle must match.
    pub ci_identity_pattern: String,
    /// OIDC issuer the Fulcio cert must record.
    pub ci_issuer: String,
    /// The expected structural claim set (see module docs).
    pub expected_claims: Vec<Claim>,
    /// A previously recorded "treat as update" decision, if any.
    pub accepted: Option<AcceptedClaims>,
}

impl CheckContext {
    /// Production context: every pin from the embedded trust root, claims
    /// from [`expected_claims`].
    pub fn new(feed_url: impl Into<String>, installed_version: impl Into<String>) -> Self {
        Self {
            feed_url: feed_url.into(),
            installed_version: installed_version.into(),
            ci_identity_pattern: trust_root::EXPECTED_CI_IDENTITY_PATTERN.to_string(),
            ci_issuer: trust_root::EXPECTED_CI_ISSUER.to_string(),
            expected_claims: expected_claims(),
            accepted: None,
        }
    }
}

// ---------------------------------------------------------------------------
// GitHub releases/latest — minimal subset we consume
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GhLatestRelease {
    tag_name: String,
    #[serde(default)]
    html_url: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

#[derive(Debug, Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// Run one update check. Infallible by design: every failure mode maps to
/// an [`UpdateCheckResult`] variant per the matrix in the module docs.
pub async fn check_for_update(client: &reqwest::Client, ctx: &CheckContext) -> UpdateCheckResult {
    // ── discover: the release marked `latest` ───────────────────────────
    let release = match fetch_latest_release(client, &ctx.feed_url).await {
        Ok(Some(r)) => r,
        // 404: no release is marked `latest` (the human-attested marker
        // hasn't been applied to anything). Not an update, not an error.
        Ok(None) => {
            return UpdateCheckResult::UpToDate {
                latest_version: None,
            };
        }
        Err(message) => return UpdateCheckResult::CheckFailed { message },
    };

    // ── version gate: compare semver against the `latest` tag ───────────
    let tag = release.tag_name.clone();
    let version_str = tag.strip_prefix('v').unwrap_or(&tag).to_string();
    let latest = match semver::Version::parse(&version_str) {
        Ok(v) => v,
        Err(e) => {
            // A malformed tag is a feed anomaly, not a crypto verdict —
            // we never reached any signed material.
            return UpdateCheckResult::CheckFailed {
                message: format!("latest release tag `{tag}` is not semver: {e}"),
            };
        }
    };
    let installed = match semver::Version::parse(&ctx.installed_version) {
        Ok(v) => v,
        Err(e) => {
            return UpdateCheckResult::CheckFailed {
                message: format!(
                    "installed version `{}` is not semver: {e}",
                    ctx.installed_version
                ),
            };
        }
    };
    if latest <= installed {
        return UpdateCheckResult::UpToDate {
            latest_version: Some(version_str),
        };
    }

    // From here on, a newer `latest` exists — every failure is a security
    // state, never silence.
    let unverifiable = |reason: String| UpdateCheckResult::Unverifiable {
        version: version_str.clone(),
        tag: tag.clone(),
        reason,
    };

    // ── fetch the two verifier assets ────────────────────────────────────
    let manifest_bytes = match fetch_listed_asset(client, &release, "artifact-manifest.json").await
    {
        Ok(bytes) => bytes,
        Err(FetchAssetError::Security(reason)) => return unverifiable(reason),
        Err(FetchAssetError::Transient(message)) => {
            return UpdateCheckResult::CheckFailed { message };
        }
    };
    let bundle_bytes =
        match fetch_listed_asset(client, &release, "artifact-manifest.json.sigstore").await {
            Ok(bytes) => bytes,
            Err(FetchAssetError::Security(reason)) => return unverifiable(reason),
            Err(FetchAssetError::Transient(message)) => {
                return UpdateCheckResult::CheckFailed { message };
            }
        };

    // ── cryptographic verification, anchored in the embedded trust root ─
    let trust = match crate::updater::trust::load() {
        Ok(t) => t,
        Err(e) => {
            // The embedded trust root failing to parse is a build defect,
            // but surfacing it as Unverifiable keeps it loud.
            return unverifiable(format!("embedded Sigstore trust root failed to load: {e}"));
        }
    };
    let verified = match ci_sigstore::verify_ci_signature_with(
        &manifest_bytes,
        &bundle_bytes,
        &trust,
        &ctx.ci_identity_pattern,
        &ctx.ci_issuer,
    ) {
        Ok(v) => v,
        Err(e) => return unverifiable(e.to_string()),
    };

    // ── tag binding: the identity must be for *this* tag ────────────────
    // The Fulcio SAN ends in `@refs/tags/<tag>`; requiring it to equal the
    // release's tag prevents replaying an authentic older manifest under a
    // newer tag to fabricate an "update".
    match verified.ci_identity.rsplit_once("@refs/tags/") {
        Some((_, signed_tag)) if signed_tag == tag => {}
        Some((_, signed_tag)) => {
            return unverifiable(format!(
                "the manifest signature is from the pinned release identity, but for tag \
                 `{signed_tag}`, not this release's tag `{tag}` — an authentic file appears \
                 to have been replayed under a different release"
            ));
        }
        None => {
            return unverifiable(format!(
                "the signing identity `{}` is not bound to a release tag (`@refs/tags/…`)",
                verified.ci_identity
            ));
        }
    }

    let manifest_sha256 = hex(&Sha256::digest(&manifest_bytes));

    // ── claims: structural comparison against the expected set ──────────
    let manifest_value: serde_json::Value = match serde_json::from_slice(&manifest_bytes) {
        Ok(v) => v,
        Err(_) => serde_json::Value::Null,
    };
    let attested = attested_claims(&manifest_value);
    let deltas = compare_claims(&ctx.expected_claims, &attested);

    let claims_accepted = !deltas.is_empty()
        && ctx.accepted.as_ref().is_some_and(|a| {
            a.version == version_str && a.manifest_sha256.eq_ignore_ascii_case(&manifest_sha256)
        });

    let release = VerifiedRelease {
        version: version_str.clone(),
        tag: tag.clone(),
        release_url: release.html_url.clone(),
        published_at: release.published_at.clone(),
        ci_identity: verified.ci_identity,
        rekor_log_index: verified.rekor_log_index,
        manifest_sha256,
        claims_accepted,
    };

    if deltas.is_empty() || claims_accepted {
        UpdateCheckResult::UpdateAvailable { release }
    } else {
        UpdateCheckResult::ClaimsChanged {
            release,
            comparison: ClaimsComparison {
                expected: ctx.expected_claims.clone(),
                attested,
                deltas,
            },
        }
    }
}

/// Fetch and parse the latest-release endpoint. `Ok(None)` = HTTP 404 (no
/// release marked latest). `Err` = transient network/API failure message.
async fn fetch_latest_release(
    client: &reqwest::Client,
    feed_url: &str,
) -> Result<Option<GhLatestRelease>, String> {
    let resp = client
        .get(feed_url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("GET {feed_url}: {e}"))?;
    let status = resp.status();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reading latest-release response: {e}"))?;
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(format!("GET {feed_url} → HTTP {status}"));
    }
    let release: GhLatestRelease =
        serde_json::from_slice(&bytes).map_err(|e| format!("parsing latest-release JSON: {e}"))?;
    Ok(Some(release))
}

enum FetchAssetError {
    /// The release should have this file but doesn't (absent from the
    /// listing, or listed but 404) — a security state for a newer release.
    Security(String),
    /// Transport-level or server-side trouble — quiet retry.
    Transient(String),
}

async fn fetch_listed_asset(
    client: &reqwest::Client,
    release: &GhLatestRelease,
    name: &str,
) -> Result<Vec<u8>, FetchAssetError> {
    let Some(asset) = release.assets.iter().find(|a| a.name == name) else {
        return Err(FetchAssetError::Security(format!(
            "the release has no `{name}` asset — the verification material this client \
             requires is missing"
        )));
    };
    let resp = client
        .get(&asset.browser_download_url)
        .header("Accept", "application/octet-stream")
        .send()
        .await
        .map_err(|e| FetchAssetError::Transient(format!("GET {name}: {e}")))?;
    let status = resp.status();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| FetchAssetError::Transient(format!("reading {name}: {e}")))?;
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(FetchAssetError::Security(format!(
            "the release lists `{name}` but the file is gone (HTTP 404)"
        )));
    }
    if !status.is_success() {
        return Err(FetchAssetError::Transient(format!(
            "{name} returned HTTP {status}"
        )));
    }
    Ok(bytes.to_vec())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

// ---------------------------------------------------------------------------
// Claims — expected set, attested extraction, comparison
// ---------------------------------------------------------------------------

/// Rendered as an alternative in an expected claim's value when a row is
/// legitimately missing from an older manifest schema this build still
/// accepts. [`compare_claims`] reads it on the attested side's absence.
const ABSENT_CLAIM: &str = "absent";

const SNP_CLAIM_VALUE: &str = "SEV-SNP launch measurement (48-byte hex)";
const TDX_CLAIM_VALUE: &str = "TDX runtime measurement (48-byte hex)";
const CMDLINE_CLAIM_VALUE: &str = "kernel command line (non-empty)";

/// The structural claim set this build expects an authentic release
/// manifest to carry. See the module docs for the full table and the
/// derivation of each row.
pub fn expected_claims() -> Vec<Claim> {
    let mut claims = Vec::new();
    claims.push(Claim {
        key: "manifest.schema_version".into(),
        value: SUPPORTED_MANIFEST_SCHEMA_VERSIONS
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" or "),
    });
    // Enclave platform claims, derived from the embedded trust root: this
    // build pins SEV-SNP + TDX measurements for the paired server, so an
    // authentic future manifest is expected to keep attesting both
    // platforms. A platform disappearing is exactly the "threat model
    // changed" signal the claims comparison exists to catch.
    if !trust_root::SERVER_SNP_MEASUREMENT.is_empty() {
        claims.push(Claim {
            key: "enclave.snp_measurement".into(),
            value: SNP_CLAIM_VALUE.into(),
        });
    }
    if !trust_root::SERVER_TDX_RTMR1.is_empty() {
        claims.push(Claim {
            key: "enclave.tdx_measurement.rtmr1".into(),
            value: TDX_CLAIM_VALUE.into(),
        });
    }
    if !trust_root::SERVER_TDX_RTMR2.is_empty() {
        claims.push(Claim {
            key: "enclave.tdx_measurement.rtmr2".into(),
            value: TDX_CLAIM_VALUE.into(),
        });
    }
    claims.push(Claim {
        key: "enclave.cmdline".into(),
        value: CMDLINE_CLAIM_VALUE.into(),
    });
    for (name, ty, platform) in EXPECTED_ARTIFACTS {
        claims.push(Claim {
            key: format!("artifacts.{name}"),
            value: format!("{ty} ({platform})"),
        });
    }
    // Rows a later schema introduced, while an earlier schema this build
    // still accepts legitimately lacks them. `" or absent"` is what makes
    // the acceptance real: without it every release still emitting the
    // older schema would read as a claims change on this row.
    let tolerate_absence = SUPPORTED_MANIFEST_SCHEMA_VERSIONS
        .iter()
        .any(|v| u64::from(*v) < SCHEMA_WITH_MACOS_SHIPPING_ZIP);
    for (name, ty, platform) in EXPECTED_ARTIFACTS_SINCE_SCHEMA_3 {
        let value = format!("{ty} ({platform})");
        claims.push(Claim {
            key: format!("artifacts.{name}"),
            value: if tolerate_absence {
                format!("{value} or {ABSENT_CLAIM}")
            } else {
                value
            },
        });
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
            if let Some(snp) = enclave_obj.get("snp_measurement") {
                claims.push(Claim {
                    key: "enclave.snp_measurement".into(),
                    value: describe_hex96(snp, SNP_CLAIM_VALUE),
                });
            }
            if let Some(tdx) = enclave_obj.get("tdx_measurement") {
                if let Some(tdx_obj) = tdx.as_object() {
                    for rtmr in ["rtmr1", "rtmr2"] {
                        if let Some(v) = tdx_obj.get(rtmr) {
                            claims.push(Claim {
                                key: format!("enclave.tdx_measurement.{rtmr}"),
                                value: describe_hex96(v, TDX_CLAIM_VALUE),
                            });
                        }
                    }
                    for key in tdx_obj.keys() {
                        if key != "rtmr1" && key != "rtmr2" {
                            claims.push(Claim {
                                key: format!("enclave.tdx_measurement.{key}"),
                                value: "unrecognized field".into(),
                            });
                        }
                    }
                } else {
                    claims.push(Claim {
                        key: "enclave.tdx_measurement".into(),
                        value: "present but not an object".into(),
                    });
                }
            }
            if let Some(cmdline) = enclave_obj.get("cmdline") {
                claims.push(Claim {
                    key: "enclave.cmdline".into(),
                    value: match cmdline.as_str() {
                        Some(s) if !s.trim().is_empty() => CMDLINE_CLAIM_VALUE.into(),
                        Some(_) => "present but empty".into(),
                        None => "present but not a string".into(),
                    },
                });
            }
            for key in enclave_obj.keys() {
                if !matches!(
                    key.as_str(),
                    "snp_measurement" | "tdx_measurement" | "cmdline"
                ) {
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
            let records_since_3 =
                schema_version.is_none_or(|v| v >= SCHEMA_WITH_MACOS_SHIPPING_ZIP);
            for (name, entry) in artifacts_obj {
                let since_3 = EXPECTED_ARTIFACTS_SINCE_SCHEMA_3
                    .iter()
                    .any(|(n, _, _)| n == name);
                claims.push(Claim {
                    key: format!("artifacts.{name}"),
                    // A row the declared schema does not record yet is a
                    // claim made outside the shape it was announced under.
                    // Tolerating it would make the rotation one-sided: a
                    // release could start emitting the row without bumping
                    // the version, and neither the gate on the committed
                    // manifest nor this verifier would say a word.
                    value: if since_3 && !records_since_3 {
                        format!(
                            "present, but schema {} records it only from \
                             schema {SCHEMA_WITH_MACOS_SHIPPING_ZIP} on",
                            schema_version.map_or_else(|| "«missing»".into(), |v| v.to_string())
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
            if records_since_3 {
                for (name, _, _) in EXPECTED_ARTIFACTS_SINCE_SCHEMA_3 {
                    if !artifacts_obj.contains_key(*name) {
                        claims.push(Claim {
                            key: format!("artifacts.{name}"),
                            value: format!(
                                "missing (recorded from schema \
                                 {SCHEMA_WITH_MACOS_SHIPPING_ZIP} on)"
                            ),
                        });
                    }
                }
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

fn describe_hex96(v: &serde_json::Value, ok: &str) -> String {
    match v.as_str() {
        Some(s) if s.len() == 96 && s.chars().all(|c| c.is_ascii_hexdigit()) => ok.into(),
        Some(_) => "present but malformed (expected 96 hex chars)".into(),
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
                "cmdline": "readonly=on root=/dev/mapper/root",
                "snp_measurement": "a".repeat(96),
                "tdx_measurement": {"rtmr1": "b".repeat(96), "rtmr2": "c".repeat(96)}
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
                "enclave.snp_measurement",
                "enclave.tdx_measurement.rtmr1",
                "enclave.tdx_measurement.rtmr2",
                "enclave.cmdline",
                "artifacts.eidola-cli",
                "artifacts.eidola-cli-macos-universal",
                "artifacts.eidola-gui-linux-amd64",
                "artifacts.eidola-gui-macos-universal",
                "artifacts.eidola-postgres",
                "artifacts.eidola-server",
                "artifacts.eidola-gui-macos-universal-zip",
            ]
        );
    }

    /// A schema-3 manifest — the shape a later release will emit. It adds
    /// the macOS unsigned shipping zip and nothing else, and must read as a
    /// clean update, not a claims change.
    fn schema_3_manifest() -> serde_json::Value {
        let mut manifest = current_manifest();
        manifest["schema_version"] = serde_json::json!(3);
        manifest["artifacts"]["eidola-gui-macos-universal-zip"] = serde_json::json!({
            "platform": "darwin/universal",
            "sha256": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "type": "file"
        });
        manifest
    }

    #[test]
    fn schema_3_manifest_with_the_zip_row_is_not_a_claims_change() {
        let deltas = compare_claims(&expected_claims(), &attested_claims(&schema_3_manifest()));
        assert!(deltas.is_empty(), "unexpected deltas: {deltas:#?}");
    }

    #[test]
    fn schema_2_manifest_without_the_zip_row_is_not_a_claims_change() {
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
    fn schema_3_manifest_missing_the_zip_row_is_a_delta() {
        // The other half of the tolerance: a manifest that declares schema 3
        // has promised the row, so dropping it is a weakened claim rather
        // than an older shape.
        let mut manifest = schema_3_manifest();
        manifest["artifacts"]
            .as_object_mut()
            .unwrap()
            .remove("eidola-gui-macos-universal-zip");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-macos-universal-zip");
        assert!(
            deltas[0].attested.as_deref().unwrap().contains("missing"),
            "got: {deltas:?}"
        );
    }

    #[test]
    fn zip_row_present_before_schema_3_is_a_delta() {
        // The other side of the rotation. Accept-before-emit means a
        // release must not start recording the row until the version says
        // so — emission without the bump has to be as loud as the bump
        // without emission, or "accepted but not yet emitted" is a claim
        // nothing enforces.
        let mut manifest = schema_3_manifest();
        manifest["schema_version"] = serde_json::json!(2);
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-macos-universal-zip");
        assert_eq!(
            deltas[0].attested.as_deref(),
            Some("present, but schema 2 records it only from schema 3 on")
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
    fn malformed_zip_sha256_is_a_delta() {
        let mut manifest = schema_3_manifest();
        manifest["artifacts"]["eidola-gui-macos-universal-zip"]["sha256"] =
            serde_json::json!("sha256:nope");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1, "unexpected deltas: {deltas:#?}");
        assert_eq!(deltas[0].key, "artifacts.eidola-gui-macos-universal-zip");
        assert!(
            deltas[0].attested.as_deref().unwrap().contains("malformed"),
            "got: {deltas:?}"
        );
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
        let mut manifest = schema_3_manifest();
        manifest["schema_version"] = serde_json::json!(99);
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "manifest.schema_version");
        assert_eq!(deltas[0].expected.as_deref(), Some("2 or 3"));
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
            .remove("snp_measurement");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].key, "enclave.snp_measurement");
        assert!(deltas[0].attested.is_none(), "claim should be absent");
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
        let mut manifest = current_manifest();
        manifest["enclave"]["snp_measurement"] = serde_json::json!("deadbeef");
        let deltas = compare_claims(&expected_claims(), &attested_claims(&manifest));
        assert_eq!(deltas.len(), 1);
        assert!(
            deltas[0].attested.as_deref().unwrap().contains("malformed"),
            "got: {deltas:?}"
        );
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
            release_url: None,
            published_at: None,
            ci_identity: "x".into(),
            rekor_log_index: 1,
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
