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
//!   verify-human→ cosign-signed hashedrekord (PKIX SPKI in body) per attestation [step 4d]
//!   templates   → render each template; require character-exact match [step 4e]
//!   cross-check → resolved substitution values match release.x.y paths [step 4e]
//!   policy      → ≥ trust_root::MIN_HUMAN_ATTESTATIONS human attestations verified [step 4e]
//!   manifest    → fetch artifact-manifest.json artifacts; verify each hash [step 4e]
//!   present     → return ReleaseSummary to the UI for user approval
//!   install     → (step 5, deferred) download + swap the new binary
//! ```
//!
//! The minimum-attestation threshold lives in the *embedded* trust root
//! ([`trust_root::MIN_HUMAN_ATTESTATIONS`]), **not** in `release.json` —
//! otherwise an adversary who produced a single forged attestation could
//! also forge a `release.json` that lowered the threshold to 1.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;

use crate::error::AppError;
use crate::trust_root;

pub use eidola_attestation::{
    ArtifactManifestRef, HumanAttestationRef, PreviousRelease, ReleaseIndex,
};

pub mod ci_sigstore;
pub mod human_attestation;
mod merkle;
mod rekor_verify;
pub mod sigstore_bundle;
pub mod trust;

// ---------------------------------------------------------------------------
// Fetcher abstraction — production fetches from GitHub, dev fetches from disk
// ---------------------------------------------------------------------------

/// Where the verifier reads release bytes from.
///
/// `Network` is the production path: hit GitHub releases for `release.json`,
/// then fetch each referenced asset over HTTPS. `Fixtures` is a dev-only
/// loop-tightener: read the same byte sequence from a local directory, so
/// the verifier can be re-run against captured release bytes without
/// re-tagging on GitHub each iteration.
///
/// The crypto stages are identical in both modes — only the byte source
/// differs. Specifically, the `Fixtures` path does not skip any signature,
/// Rekor, or inclusion-proof verification; the whole point is that the
/// fixtures path exercises the same verifier code.
pub enum Fetcher {
    Network(reqwest::Client),
    Fixtures(PathBuf),
}

impl Fetcher {
    /// Construct the default network fetcher (rustls + native roots).
    pub fn network() -> Result<Self, AppError> {
        Ok(Fetcher::Network(build_http_client()?))
    }

    /// Read release bytes from the given directory. URLs in `release.json`
    /// are mapped to filenames by taking the URL's last path component, so
    /// captured downloads can be dropped in unrenamed.
    pub fn fixtures(dir: impl Into<PathBuf>) -> Self {
        Fetcher::Fixtures(dir.into())
    }

    /// Fetch `release.json` itself. In `Network` mode this hits the GitHub
    /// releases API at `trust_root::UPDATE_DISCOVERY_URL`. In `Fixtures`
    /// mode it reads `<dir>/release.json` directly — the GitHub releases
    /// lookup is the one stage the fixtures path bypasses.
    async fn fetch_release_json(&self) -> Result<Vec<u8>, AppError> {
        match self {
            Fetcher::Network(client) => fetch_release_json_network(client).await,
            Fetcher::Fixtures(dir) => read_fixture(dir, "release.json"),
        }
    }

    /// Fetch a referenced asset by URL.
    async fn fetch_url(&self, url: &str, label: &str) -> Result<Vec<u8>, AppError> {
        match self {
            Fetcher::Network(client) => fetch_url_network(client, url, label).await,
            Fetcher::Fixtures(dir) => {
                let name = url_to_filename(url).ok_or_else(|| AppError::Update {
                    message: format!(
                        "fixtures mode: could not extract filename from URL `{url}` (for {label})"
                    ),
                })?;
                read_fixture(dir, &name)
            }
        }
    }
}

fn url_to_filename(url: &str) -> Option<String> {
    // Strip query/fragment, then take the last `/`-separated segment.
    let no_query = url.split('?').next().unwrap_or(url);
    let no_frag = no_query.split('#').next().unwrap_or(no_query);
    let last = no_frag.rsplit('/').next()?;
    if last.is_empty() {
        None
    } else {
        Some(last.to_string())
    }
}

fn read_fixture(dir: &Path, name: &str) -> Result<Vec<u8>, AppError> {
    let path = dir.join(name);
    std::fs::read(&path).map_err(|e| AppError::Update {
        message: format!("reading fixture {}: {e}", path.display()),
    })
}

// ---------------------------------------------------------------------------
// Verbose tracing — diagnostic eprintln! at each pipeline stage
// ---------------------------------------------------------------------------

/// Options that flow through the verifier independently of the byte source.
#[derive(Debug, Clone, Copy, Default)]
pub struct VerifyOptions {
    /// When set, emit one `eprintln!` per pipeline stage with a millisecond
    /// timestamp relative to verifier start. Intended for the `eidola
    /// update --verbose` dev loop — see module docs.
    pub verbose: bool,
}

struct Tracer {
    enabled: bool,
    start: Instant,
}

impl Tracer {
    fn new(verbose: bool) -> Self {
        Self {
            enabled: verbose,
            start: Instant::now(),
        }
    }

    fn log(&self, stage: &str, msg: impl AsRef<str>) {
        if !self.enabled {
            return;
        }
        let ms = self.start.elapsed().as_millis();
        eprintln!("[{:>5}ms] {}: {}", ms, stage, msg.as_ref());
    }
}

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
// release.json types are defined in `eidola-attestation::trust_shapes` and
// re-exported above so both the signing side (release-tool) and the
// verifier side share one source of truth for the on-disk shape.
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
///
/// - The cosign-emitted blob signature over the attestation bytes is
///   valid under the attestant's pubkey (ECDSA-P256 / ECDSA-P384 /
///   Ed25519 — dispatched on the SPKI's algorithm OID).
/// - The signer's pubkey fingerprint (`fingerprint_hex` =
///   `sha256(PKIX SubjectPublicKeyInfo DER)`) is in this client's
///   pinned `TRUSTED_ATTESTANT_FINGERPRINTS`.
/// - The signature was logged in Sigstore Rekor (`rekor_log_index`,
///   SET-signed by a pinned Rekor key, with a valid inclusion proof).
/// - The attestation's `release_version` / `git_commit` /
///   `previous_release_git_commit` match the release index's
///   corresponding fields (so this attestation actually pertains to
///   *this* release, not a different one).
/// - Every `statement` in `claims` is character-for-character equal to
///   the rendered output of its pinned template, with every declared
///   `cross_check` field also equal to the corresponding `release.x.y`
///   path.
#[derive(Debug, Clone)]
pub struct VerifiedAttestation {
    pub attestant_id: String,
    pub attestant_name: String,
    pub jurisdiction: String,
    pub fingerprint_hex: String,
    pub rekor_log_index: u64,
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
///
/// Production callers should use this entry point — it constructs the
/// default network `Fetcher` and emits no diagnostics. The dev loop uses
/// [`check_for_update_with`] with a `Fetcher::Fixtures` and/or
/// `VerifyOptions { verbose: true }`.
pub async fn check_for_update(
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<Option<ReleaseSummary>, AppError> {
    let fetcher = Fetcher::network()?;
    check_for_update_with(
        &fetcher,
        VerifyOptions::default(),
        installed_version,
        installed_git_commit,
    )
    .await
}

/// Verifier entry point with explicit byte source and options.
///
/// Same pipeline as [`check_for_update`]; only the byte source and
/// diagnostic verbosity differ. See [`Fetcher`] and [`VerifyOptions`].
pub async fn check_for_update_with(
    fetcher: &Fetcher,
    opts: VerifyOptions,
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<Option<ReleaseSummary>, AppError> {
    let tracer = Tracer::new(opts.verbose);

    // ── discover ─────────────────────────────────────────────────────────
    tracer.log("discover", "reading release.json");
    let release_json_bytes = fetcher.fetch_release_json().await?;
    tracer.log(
        "discover",
        format!("release.json ({} bytes)", release_json_bytes.len()),
    );

    // ── schema + continuity ─────────────────────────────────────────────
    let release = match parse_and_gate_release(
        &release_json_bytes,
        installed_version,
        installed_git_commit,
    ) {
        Ok(r) => r,
        Err(NoUpdateOrError::NoUpdate) => {
            tracer.log(
                "schema",
                format!("installed version {installed_version} is current; no update"),
            );
            return Ok(None);
        }
        Err(NoUpdateOrError::Error(e)) => return Err(e),
    };
    tracer.log(
        "schema",
        format!(
            "schema_version={} version={} -> ok",
            release.schema_version, release.version
        ),
    );
    match (&release.previous_release, installed_git_commit) {
        (Some(prev), _) => tracer.log(
            "continuity",
            format!(
                "previous_release.version={} git_commit={}",
                prev.version, prev.git_commit
            ),
        ),
        (None, None) => tracer.log("continuity", "first install; no prior commit to anchor"),
        (None, Some(_)) => { /* parse_and_gate_release already returned Err */ }
    }

    let trust = trust::load()?;
    tracer.log("trust", "embedded Sigstore trust root loaded");

    // ── verify CI side ───────────────────────────────────────────────────
    tracer.log(
        "fetch",
        format!(
            "artifact-manifest.json <- {}",
            release.artifact_manifest.url
        ),
    );
    let manifest_bytes = fetcher
        .fetch_url(&release.artifact_manifest.url, "artifact-manifest.json")
        .await?;
    tracer.log(
        "fetch",
        format!("artifact-manifest.json ({} bytes)", manifest_bytes.len()),
    );
    tracer.log(
        "fetch",
        format!(
            "artifact-manifest.json.sigstore <- {}",
            release.artifact_manifest.sigstore_bundle_url
        ),
    );
    let bundle_bytes = fetcher
        .fetch_url(
            &release.artifact_manifest.sigstore_bundle_url,
            "artifact-manifest.json.sigstore",
        )
        .await?;
    tracer.log(
        "fetch",
        format!(
            "artifact-manifest.json.sigstore ({} bytes)",
            bundle_bytes.len()
        ),
    );

    tracer.log(
        "verify-ci",
        "verifying Fulcio chain + Rekor SET + inclusion",
    );
    let _verified_ci = ci_sigstore::verify_ci_signature(&manifest_bytes, &bundle_bytes, &trust)?;
    tracer.log("verify-ci", "ok");

    // ── verify each human attestation (signature + content) ─────────────
    let mut verified_attestations: Vec<VerifiedAttestation> =
        Vec::with_capacity(release.human_attestations.len());
    for human in &release.human_attestations {
        let att_label = format!("attestation-{}.json", human.attestant_id);
        tracer.log("fetch", format!("{att_label} <- {}", human.url));
        let attestation_bytes = fetcher.fetch_url(&human.url, &att_label).await?;
        tracer.log(
            "fetch",
            format!("{att_label} ({} bytes)", attestation_bytes.len()),
        );

        let bundle_label = format!("attestation-{}.bundle.json", human.attestant_id);
        tracer.log("fetch", format!("{bundle_label} <- {}", human.bundle_url));
        let bundle_bytes = fetcher.fetch_url(&human.bundle_url, &bundle_label).await?;
        tracer.log(
            "fetch",
            format!("{bundle_label} ({} bytes)", bundle_bytes.len()),
        );

        tracer.log(
            "verify-human",
            format!(
                "verifying {} (cosign blob signature + Rekor + templates)",
                human.attestant_id
            ),
        );
        let verified = human_attestation::verify_human_attestation(
            &attestation_bytes,
            &bundle_bytes,
            &release,
            &trust,
        )?;
        // The attestant_id in release.json must match what the
        // attestation prose says — defends against a release manifest
        // listing one attestant_id but pointing to a different person's
        // signed prose.
        if verified.attestant_id != human.attestant_id {
            return Err(AppError::Update {
                message: format!(
                    "release.human_attestations[].attestant_id `{}` ≠ attestation prose \
                     attestant.id `{}`",
                    human.attestant_id, verified.attestant_id
                ),
            });
        }
        tracer.log(
            "verify-human",
            format!(
                "{} ok (fingerprint sha256:{}, rekor logIndex={})",
                verified.attestant_id, verified.fingerprint_hex, verified.rekor_log_index
            ),
        );
        verified_attestations.push(verified);
    }

    // Policy: minimum number of independently-verified attestations. The
    // threshold is pinned in the *embedded* trust root rather than in
    // `release.json` so an adversary who controls the index can't lower
    // the bar — see module docs and `docs/trust-root.md`.
    if (verified_attestations.len() as u32) < trust_root::MIN_HUMAN_ATTESTATIONS {
        return Err(AppError::Update {
            message: format!(
                "release verified only {} human attestation(s); embedded policy requires ≥{}",
                verified_attestations.len(),
                trust_root::MIN_HUMAN_ATTESTATIONS,
            ),
        });
    }
    tracer.log(
        "policy",
        format!(
            "{} attestation(s) verified (>= {} required)",
            verified_attestations.len(),
            trust_root::MIN_HUMAN_ATTESTATIONS
        ),
    );

    // TODO (step 5): `verify_each_artifact_hash` — fetch
    // `artifact-manifest.json`, walk its `artifacts` table, and for each
    // artifact this client is about to install (today: just the
    // platform-appropriate CLI/GUI binary), download it and verify its
    // hash against the manifest's declared digest. The manifest itself
    // was already CI-signature-verified above, so its declared hashes
    // are trustworthy — this stage just turns "manifest authentic" into
    // "downloaded bytes authentic." Lives in step 5 because the natural
    // home is the install/replace flow, where we already need to be
    // downloading the binary anyway. See `docs/gaps.md` for the
    // audit-side description.

    tracer.log("present", format!("release {} verified", release.version));

    Ok(Some(ReleaseSummary {
        version: release.version,
        git_commit: release.git_commit,
        git_tag: release.git_tag,
        released_at: release.released_at,
        previous_release: release.previous_release,
        attestations: verified_attestations,
    }))
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
async fn fetch_release_json_network(client: &reqwest::Client) -> Result<Vec<u8>, AppError> {
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

    fetch_url_network(client, &asset.browser_download_url, "release.json").await
}

/// Fetch a single URL's body bytes. Errors carry the file label so the
/// caller doesn't have to reformat them.
async fn fetch_url_network(
    client: &reqwest::Client,
    url: &str,
    label: &str,
) -> Result<Vec<u8>, AppError> {
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
                }}]
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
            "human_attestations": [{"attestant_id":"x","url":"x","bundle_url":"x"}]
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
    }

    #[test]
    fn min_human_attestations_lives_in_embedded_trust_root() {
        // Threshold is pinned at build time from trust-constants.json, not
        // read from release.json — so a forged release index cannot lower
        // it. The current pin is 1; bumping it is a release-gated event.
        // (Static `const`-block assertion would compile-fail; a runtime
        // check keeps the lint quiet while still pinning the invariant.)
        let pinned = trust_root::MIN_HUMAN_ATTESTATIONS;
        assert!(pinned >= 1, "MIN_HUMAN_ATTESTATIONS must be ≥1");
    }

    #[test]
    fn release_index_with_previous_release() {
        let bytes = release_json("0.5.0", Some("5e1f000000000000000000000000000000000002"));
        let parsed: ReleaseIndex = serde_json::from_slice(&bytes).unwrap();
        let prev = parsed.previous_release.expect("expected previous_release");
        assert_eq!(prev.git_commit, "5e1f000000000000000000000000000000000002");
    }
}
