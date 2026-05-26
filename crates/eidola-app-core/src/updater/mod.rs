//! Self-update verifier — fetches `release.json` from the embedded
//! `UPDATE_DISCOVERY_URL`, validates every signed artifact it references
//! against the embedded trust root, and surfaces the attestation prose to
//! the user for approval before swapping the running binary.
//!
//! ## Pipeline
//!
//! ```text
//!   discover    → fetch release.json from GitHub releases API   [step 4b]
//!   schema      → release.json.schema_version supported          [step 4b]
//!   continuity  → release.version > installed; previous matches  [step 4b]
//!   fetch       → download artifact-manifest.json + both bundles + each attestation
//!   verify-ci   → tinfoil-rs sigstore verification               [step 4c]
//!   verify-human→ SSH signature + Rekor hashedrekord per attestation [step 4d]
//!   templates   → render each template; require character-exact match [step 4e]
//!   cross-check → resolved substitution values match release.x.y paths [step 4e]
//!   policy      → ≥ min_human_attestations human attestations verified [step 4e]
//!   manifest    → fetch artifact-manifest.json artifacts; verify each hash [step 4e]
//!   present     → return ReleaseSummary to the UI for user approval
//!   install     → (step 5, deferred) download + swap the new binary
//! ```

use serde::Deserialize;

use crate::error::AppError;
use crate::trust_root;

pub mod ci_sigstore;
pub mod trust;

// ---------------------------------------------------------------------------
// GitHub releases API — minimal subset we consume
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GhRelease {
    #[allow(dead_code)]
    tag_name: String,
    assets: Vec<GhReleaseAsset>,
}

#[derive(Debug, Deserialize)]
struct GhReleaseAsset {
    name: String,
    browser_download_url: String,
}

// ---------------------------------------------------------------------------
// release.json types — match releases/schema/release-v1.schema.json
// ---------------------------------------------------------------------------

/// Top-level shape of a release index. The verifier deserializes this from
/// the bytes fetched at `UPDATE_DISCOVERY_URL`'s `release.json` asset.
#[derive(Debug, Clone, Deserialize)]
pub struct ReleaseIndex {
    pub schema_version: u32,
    pub version: String,
    pub git_commit: String,
    pub git_tag: String,
    pub released_at: String,
    #[serde(default)]
    pub previous_release: Option<PreviousRelease>,
    pub artifact_manifest: ArtifactManifestRef,
    pub human_attestations: Vec<HumanAttestationRef>,
    pub policy: ReleasePolicy,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreviousRelease {
    pub version: String,
    pub git_commit: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArtifactManifestRef {
    pub url: String,
    pub sigstore_bundle_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HumanAttestationRef {
    pub attestant_id: String,
    pub url: String,
    pub bundle_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReleasePolicy {
    pub min_human_attestations: u32,
}

// ---------------------------------------------------------------------------
// Verifier output — what the UI displays to the user before install
// ---------------------------------------------------------------------------

/// What [`check_for_update`] returns once verification succeeds. The UI
/// shows the user the prose in `attestations` and waits for approval
/// before invoking install.
#[derive(Debug, Clone)]
pub struct ReleaseSummary {
    pub version: String,
    pub git_commit: String,
    pub git_tag: String,
    pub released_at: String,
    pub previous_release: Option<PreviousRelease>,
    pub attestations: Vec<VerifiedAttestation>,
}

/// One verified human attestation, ready to render. The verifier guarantees:
/// every `statement` in `claims` is character-for-character equal to the
/// rendered output of its pinned template.
#[derive(Debug, Clone)]
pub struct VerifiedAttestation {
    pub attestant_id: String,
    pub attestant_name: String,
    pub jurisdiction: String,
    pub attested_at: String,
    pub attestant_statement: String,
    pub claims: Vec<VerifiedClaim>,
}

#[derive(Debug, Clone)]
pub struct VerifiedClaim {
    pub claim_id: String,
    pub statement: String,
}

// ---------------------------------------------------------------------------
// Entry point — partially implemented (steps 4b through 4f flesh this out)
// ---------------------------------------------------------------------------

/// Fetch the latest release index, run the verification pipeline, and
/// return a [`ReleaseSummary`] ready for the UI to display.
///
/// Returns `Ok(None)` if no update is available (the latest release matches
/// or is older than the currently-installed version). Returns `Err` if the
/// latest release fails any pipeline stage — the UI surfaces the error and
/// the installed version stays put.
pub async fn check_for_update(
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<Option<ReleaseSummary>, AppError> {
    let client = build_http_client()?;

    let release_json_bytes = fetch_release_json(&client).await?;
    let release = match parse_and_gate_release(
        &release_json_bytes,
        installed_version,
        installed_git_commit,
    ) {
        Ok(r) => r,
        Err(NoUpdateOrError::NoUpdate) => return Ok(None),
        Err(NoUpdateOrError::Error(e)) => return Err(e),
    };

    let trust = trust::load()?;

    // ── verify CI side ───────────────────────────────────────────────────
    let manifest_bytes = fetch_url(
        &client,
        &release.artifact_manifest.url,
        "artifact-manifest.json",
    )
    .await?;
    let bundle_bytes = fetch_url(
        &client,
        &release.artifact_manifest.sigstore_bundle_url,
        "artifact-manifest.json.sigstore",
    )
    .await?;
    let _verified_ci = ci_sigstore::verify_ci_signature(&manifest_bytes, &bundle_bytes, &trust)?;

    // TODO (4d): verify each human attestation (SSH + rekor proof).
    // TODO (4e): render templates and require character-for-character
    //            equality with the signed `statement`; cross-check resolved
    //            values against release.git_commit / .previous_release.git_commit;
    //            policy check on min_human_attestations.
    // TODO (4e): verify each artifact hash inside artifact-manifest.json.

    Err(AppError::Update {
        message: format!(
            "release {} ({}) passes discover/continuity/CI-structural; human-attestation + \
             template-equality + artifact-hash stages are not yet implemented in this build",
            release.git_tag, release.git_commit,
        ),
    })
}

/// Outcome wrapper for the discover→continuity stages: either we have a
/// parsed `ReleaseIndex` to feed to the crypto stages, or we hit the "no
/// newer release" signal (collapses to `Ok(None)` upstream), or we hit a
/// hard failure.
#[derive(Debug)]
enum NoUpdateOrError {
    NoUpdate,
    Error(AppError),
}

impl From<AppError> for NoUpdateOrError {
    fn from(e: AppError) -> Self {
        NoUpdateOrError::Error(e)
    }
}

fn parse_and_gate_release(
    release_json_bytes: &[u8],
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<ReleaseIndex, NoUpdateOrError> {
    let release: ReleaseIndex = serde_json::from_slice(release_json_bytes).map_err(|e| {
        NoUpdateOrError::Error(AppError::Update {
            message: format!("release.json failed to parse: {e}"),
        })
    })?;

    if !trust_root::SUPPORTED_RELEASE_SCHEMA_VERSIONS.contains(&release.schema_version) {
        return Err(NoUpdateOrError::Error(AppError::Update {
            message: format!(
                "release.json schema_version `{}` is not in the supported set {:?} \
                 — this client cannot parse the release safely; install a newer client.",
                release.schema_version,
                trust_root::SUPPORTED_RELEASE_SCHEMA_VERSIONS,
            ),
        }));
    }

    match compare_versions(&release.version, installed_version)? {
        VersionCompare::OlderOrEqual => return Err(NoUpdateOrError::NoUpdate),
        VersionCompare::Newer => {}
    }

    check_continuity(&release, installed_git_commit)?;

    Ok(release)
}

enum VersionCompare {
    Newer,
    OlderOrEqual,
}

fn compare_versions(latest: &str, installed: &str) -> Result<VersionCompare, NoUpdateOrError> {
    let latest_v = semver::Version::parse(latest).map_err(|e| {
        NoUpdateOrError::Error(AppError::Update {
            message: format!("release.json `version` is not valid semver ({latest}): {e}"),
        })
    })?;
    let installed_v = semver::Version::parse(installed).map_err(|e| {
        NoUpdateOrError::Error(AppError::Update {
            message: format!("installed version is not valid semver ({installed}): {e}"),
        })
    })?;
    Ok(if latest_v > installed_v {
        VersionCompare::Newer
    } else {
        VersionCompare::OlderOrEqual
    })
}

fn check_continuity(
    release: &ReleaseIndex,
    installed_git_commit: Option<&str>,
) -> Result<(), NoUpdateOrError> {
    // First install (no prior commit) bypasses continuity. There's no
    // installed state to anchor against, so we trust the release purely
    // on the cryptographic verification stages (and the user's review of
    // the attestation prose at install time).
    let Some(installed) = installed_git_commit else {
        return Ok(());
    };

    let Some(prev) = release.previous_release.as_ref() else {
        return Err(NoUpdateOrError::Error(AppError::Update {
            message: format!(
                "client has installed git_commit `{installed}` but the new release.json \
                 declares no `previous_release` — cannot verify continuity"
            ),
        }));
    };

    // Strict equality. v1 doesn't attempt fast-forward / multi-hop
    // reachability — that requires querying GitHub for commit ancestry,
    // and the simpler "must update through each release in order" model is
    // sufficient until the cadence makes it painful.
    if !prev.git_commit.eq_ignore_ascii_case(installed) {
        return Err(NoUpdateOrError::Error(AppError::Update {
            message: format!(
                "continuity failure: new release.previous_release.git_commit is `{}`, \
                 but the installed git_commit is `{}`. The new release is not a direct \
                 successor; either you have an old client that skipped a release, or \
                 someone served a forked release.json.",
                prev.git_commit, installed
            ),
        }));
    }

    Ok(())
}

/// Fetch the latest `release.json` asset from the GitHub releases API.
async fn fetch_release_json(client: &reqwest::Client) -> Result<Vec<u8>, AppError> {
    let resp = client
        .get(trust_root::UPDATE_DISCOVERY_URL)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| AppError::Update {
            message: format!("GET {}: {e}", trust_root::UPDATE_DISCOVERY_URL),
        })?;
    let status = resp.status();
    let body_bytes = resp.bytes().await.map_err(|e| AppError::Update {
        message: format!("reading response body: {e}"),
    })?;
    if !status.is_success() {
        return Err(AppError::Update {
            message: format!(
                "GET {} → HTTP {} ({})",
                trust_root::UPDATE_DISCOVERY_URL,
                status,
                String::from_utf8_lossy(&body_bytes).trim()
            ),
        });
    }
    let gh: GhRelease = serde_json::from_slice(&body_bytes).map_err(|e| AppError::Update {
        message: format!("parsing GitHub release JSON: {e}"),
    })?;

    let asset = gh
        .assets
        .iter()
        .find(|a| a.name == "release.json")
        .ok_or_else(|| AppError::Update {
            message: "latest GitHub release has no `release.json` asset — incomplete release"
                .to_string(),
        })?;

    fetch_url(client, &asset.browser_download_url, "release.json").await
}

/// Fetch a single URL's body bytes. Errors carry the file label so the
/// caller doesn't have to reformat them.
async fn fetch_url(client: &reqwest::Client, url: &str, label: &str) -> Result<Vec<u8>, AppError> {
    let resp = client
        .get(url)
        .header("Accept", "application/octet-stream")
        .send()
        .await
        .map_err(|e| AppError::Update {
            message: format!("GET {label} ({url}): {e}"),
        })?;
    let status = resp.status();
    let bytes = resp.bytes().await.map_err(|e| AppError::Update {
        message: format!("reading {label} body: {e}"),
    })?;
    if !status.is_success() {
        return Err(AppError::Update {
            message: format!(
                "{label} ({url}) returned HTTP {} ({})",
                status,
                String::from_utf8_lossy(&bytes).trim()
            ),
        });
    }
    Ok(bytes.to_vec())
}

fn build_http_client() -> Result<reqwest::Client, AppError> {
    // Idempotent — AppCore::new normally installs this; for standalone
    // updater callers (tests, future binaries) this is the safety net.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(crate::load_native_root_store())
        .with_no_client_auth();

    reqwest::Client::builder()
        .tls_backend_preconfigured(tls_config)
        .user_agent(concat!("eidola-app-core/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| AppError::Update {
            message: format!("constructing HTTPS client: {e}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unwrap_err(r: Result<ReleaseIndex, NoUpdateOrError>) -> AppError {
        match r {
            Err(NoUpdateOrError::Error(e)) => e,
            Err(NoUpdateOrError::NoUpdate) => panic!("expected Error, got NoUpdate"),
            Ok(_) => panic!("expected Error, got Ok"),
        }
    }

    fn is_no_update(r: &Result<ReleaseIndex, NoUpdateOrError>) -> bool {
        matches!(r, Err(NoUpdateOrError::NoUpdate))
    }

    fn release_json(version: &str, previous_commit: Option<&str>) -> Vec<u8> {
        let prev = match previous_commit {
            Some(c) => {
                format!(r#""previous_release": {{"version": "0.0.1", "git_commit": "{c}"}},"#)
            }
            None => String::new(),
        };
        format!(
            r#"{{
                "schema_version": 1,
                "version": "{version}",
                "git_commit": "9c3a000000000000000000000000000000000001",
                "git_tag": "v{version}",
                "released_at": "2026-05-26T17:00:00Z",
                {prev}
                "artifact_manifest": {{
                    "url": "https://example/m.json",
                    "sigstore_bundle_url": "https://example/m.json.sigstore"
                }},
                "human_attestations": [{{
                    "attestant_id": "mike-prince",
                    "url": "https://example/a.json",
                    "bundle_url": "https://example/a.bundle.json"
                }}],
                "policy": {{ "min_human_attestations": 1 }}
            }}"#
        )
        .into_bytes()
    }

    #[test]
    fn no_update_when_latest_equals_installed() {
        let bytes = release_json("1.0.0", None);
        let r = parse_and_gate_release(&bytes, "1.0.0", None);
        assert!(is_no_update(&r));
    }

    #[test]
    fn no_update_when_latest_older_than_installed() {
        let bytes = release_json("0.9.0", None);
        let r = parse_and_gate_release(&bytes, "1.0.0", None);
        assert!(is_no_update(&r));
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let bytes = br#"{
            "schema_version": 99,
            "version": "1.1.0",
            "git_commit": "9c3a000000000000000000000000000000000001",
            "git_tag": "v1.1.0",
            "released_at": "2026-05-26T17:00:00Z",
            "artifact_manifest": {"url":"x","sigstore_bundle_url":"x"},
            "human_attestations": [{"attestant_id":"x","url":"x","bundle_url":"x"}],
            "policy": {"min_human_attestations": 1}
        }"#
        .to_vec();
        let err = unwrap_err(parse_and_gate_release(&bytes, "1.0.0", None));
        let msg = format!("{err}");
        assert!(msg.contains("schema_version"), "got: {msg}");
    }

    #[test]
    fn rejects_invalid_semver_in_release() {
        let bytes = release_json("not.semver.at.all", None);
        let err = unwrap_err(parse_and_gate_release(&bytes, "1.0.0", None));
        let msg = format!("{err}");
        assert!(msg.contains("semver"), "got: {msg}");
    }

    #[test]
    fn first_install_bypasses_continuity() {
        // No installed git_commit + release with previous_release ⇒ OK;
        // continuity stage skipped, returns the parsed ReleaseIndex
        // ready for the crypto stages.
        let bytes = release_json("1.1.0", Some("5e1f000000000000000000000000000000000002"));
        let release = parse_and_gate_release(&bytes, "1.0.0", None)
            .expect("first install with prior release should pass gating");
        assert_eq!(release.version, "1.1.0");
    }

    #[test]
    fn continuity_passes_when_previous_matches_installed() {
        let installed_commit = "5e1f000000000000000000000000000000000002";
        let bytes = release_json("1.1.0", Some(installed_commit));
        let release = parse_and_gate_release(&bytes, "1.0.0", Some(installed_commit))
            .expect("matching continuity should pass gating");
        assert_eq!(release.version, "1.1.0");
        assert_eq!(
            release.previous_release.unwrap().git_commit,
            installed_commit
        );
    }

    #[test]
    fn continuity_fails_when_previous_differs_from_installed() {
        let bytes = release_json("1.1.0", Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        let err = unwrap_err(parse_and_gate_release(
            &bytes,
            "1.0.0",
            Some("5e1f000000000000000000000000000000000002"),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("continuity"), "got: {msg}");
    }

    #[test]
    fn continuity_fails_when_release_has_no_previous_and_client_does() {
        let bytes = release_json("1.1.0", None);
        let err = unwrap_err(parse_and_gate_release(
            &bytes,
            "1.0.0",
            Some("5e1f000000000000000000000000000000000002"),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("cannot verify continuity"), "got: {msg}");
    }

    #[test]
    fn release_index_deserializes_minimal_form() {
        let bytes = release_json("0.5.0", None);
        let parsed: ReleaseIndex = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.schema_version, 1);
        assert_eq!(parsed.version, "0.5.0");
        assert!(parsed.previous_release.is_none());
        assert_eq!(parsed.human_attestations.len(), 1);
        assert_eq!(parsed.policy.min_human_attestations, 1);
    }

    #[test]
    fn release_index_with_previous_release() {
        let bytes = release_json("0.5.0", Some("5e1f000000000000000000000000000000000002"));
        let parsed: ReleaseIndex = serde_json::from_slice(&bytes).unwrap();
        let prev = parsed.previous_release.expect("expected previous_release");
        assert_eq!(prev.git_commit, "5e1f000000000000000000000000000000000002");
    }
}
