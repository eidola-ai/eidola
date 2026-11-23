//! Self-update verifier — fetches the latest `release.json` from the
//! releases location pinned in the trust root (`UPDATE_RELEASES_URL`),
//! validates every signed artifact it references against the embedded trust
//! root, and surfaces the attestation prose to the user for approval before
//! swapping the running binary.
//!
//! ## Pipeline
//!
//! ```text
//!   discover    → GET <releases>/latest/download/release.json (404 = nothing marked latest)
//!   version     → release.version > installed, read before anything else
//!   schema      → release.json.schema_version supported, shape held to it
//!   continuity  → previous_release matches the installed commit, when one is known
//!   fetch       → artifact-manifest.json + its bundle + each attestation + its bundle
//!   verify-ci   → Fulcio chain, pinned identity, Rekor; identity bound to this release's tag
//!   verify-human→ cosign-signed hashedrekord (PKIX SPKI in body) per attestation
//!   templates   → render each template; require character-exact match
//!   cross-check → resolved substitution values match release.x.y paths
//!   bind        → every attestation names the sha256 of the manifest fetched
//!   policy      → ≥ MIN_HUMAN_ATTESTATIONS independent attestations verified
//!   present     → return ReleaseSummary to the UI for user approval
//!   install     → `install::stage` downloads and hashes; promotion is separate
//! ```
//!
//! [`verify_release`] is the stages from `fetch` to `policy`, and it is the
//! only definition of "this release is real" in the client: the update
//! notification (`crate::updates`) and the install path both run it, so a
//! release can never be announced on less evidence than installing it
//! requires.
//!
//! **Where the latest release is found is not a trust decision.** The
//! `latest` marker, the URLs in `release.json`, and `release.json` itself are
//! all unsigned, and anyone who can edit the release listing can change
//! them. What they can do with that is limited to pointing at a different
//! release, which then has to pass every stage above, or pointing at
//! nothing. Withholding an update is the one thing a discovery channel can
//! always do, and no client-side check can tell a withheld release from no
//! release.
//!
//! Discovery uses GitHub's download redirects rather than its REST API:
//! the API allows sixty unauthenticated requests an hour per address, which
//! every client behind one NAT shares, and the redirects carry no such limit.
//!
//! The minimum-attestation threshold lives in the *embedded* trust root
//! ([`trust_root::MIN_HUMAN_ATTESTATIONS`]), **not** in `release.json` —
//! otherwise an adversary who produced a single forged attestation could
//! also forge a `release.json` that lowered the threshold to 1.

use crate::rt::Instant;
use std::path::{Path, PathBuf};

use sha2::Digest;

use crate::error::AppError;
use crate::trust_root;

pub use eidola_attestation::{
    ArtifactManifestRef, HumanAttestationRef, PreviousRelease, ReleaseIndex,
};

pub mod ci_sigstore;
pub mod human_attestation;
#[cfg(feature = "desktop")]
pub mod install;
pub mod install_kind;
pub mod installed;
mod merkle;
pub mod mobile;
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
    Network(crate::transport::Client),
    Fixtures(PathBuf),
}

impl Fetcher {
    /// Construct the default network fetcher (rustls + native roots).
    pub fn network() -> Result<Self, AppError> {
        Ok(Fetcher::Network(build_http_client(
            &crate::transport::Transport::default(),
        )?))
    }

    /// Read release bytes from the given directory. URLs in `release.json`
    /// are mapped to filenames by taking the URL's last path component, so
    /// captured downloads can be dropped in unrenamed.
    pub fn fixtures(dir: impl Into<PathBuf>) -> Self {
        Fetcher::Fixtures(dir.into())
    }

    /// Fetch the `release.json` of the release marked latest under
    /// `releases_url` (`<releases_url>/latest/download/release.json`).
    /// `Ok(None)` when there is none to fetch: GitHub answers 404 both when
    /// nothing is marked latest and when the release marked latest has no
    /// `release.json`, and neither is an update. In `Fixtures` mode it reads
    /// `<dir>/release.json`, and a missing file is likewise `None`.
    pub async fn fetch_latest_release_json(
        &self,
        releases_url: &str,
    ) -> Result<Option<Vec<u8>>, ReleaseCheckError> {
        let result = match self {
            Fetcher::Network(client) => {
                let url = latest_release_json_url(releases_url);
                fetch_url_network(client, &url, "release.json", MAX_DOCUMENT_BYTES).await
            }
            Fetcher::Fixtures(dir) => read_fixture(dir, "release.json"),
        };
        match result {
            Ok(bytes) => Ok(Some(bytes)),
            Err(FetchFailure::Missing(_)) => Ok(None),
            Err(other) => Err(other.into()),
        }
    }

    /// Fetch the `release.json` of the release tagged `tag` under
    /// `releases_url` (`<releases_url>/download/<tag>/release.json`), the
    /// way [`discover_next_release`] walks a release's history. In
    /// `Fixtures` mode it reads `<dir>/<tag>/release.json`. A missing file
    /// is a refusal: the release that named this one said it was there.
    pub async fn fetch_release_json(
        &self,
        releases_url: &str,
        tag: &str,
    ) -> Result<Vec<u8>, ReleaseCheckError> {
        let result = match self {
            Fetcher::Network(client) => {
                let url = release_json_url(releases_url, tag);
                fetch_url_network(client, &url, "release.json", MAX_DOCUMENT_BYTES).await
            }
            Fetcher::Fixtures(dir) => read_fixture(&dir.join(tag), "release.json"),
        };
        result.map_err(ReleaseCheckError::from)
    }

    /// Fetch a referenced asset by URL.
    ///
    /// Bounded at [`MAX_DOCUMENT_BYTES`]: every URL reached this way comes
    /// out of `release.json`, which is unsigned and therefore chosen by
    /// whoever served it.
    async fn fetch_url(&self, url: &str, label: &str) -> Result<Vec<u8>, ReleaseCheckError> {
        self.fetch_url_bounded(url, label, MAX_DOCUMENT_BYTES).await
    }

    /// Fetch a referenced asset with an explicit ceiling, for callers whose
    /// artifacts are larger than a signed document.
    pub async fn fetch_url_bounded(
        &self,
        url: &str,
        label: &str,
        max_bytes: u64,
    ) -> Result<Vec<u8>, ReleaseCheckError> {
        let result = match self {
            Fetcher::Network(client) => fetch_url_network(client, url, label, max_bytes).await,
            Fetcher::Fixtures(dir) => match url_to_filename(url) {
                Some(name) => read_fixture(dir, &name),
                None => Err(FetchFailure::Refused(format!(
                    "fixtures mode: could not extract filename from URL `{url}` (for {label})"
                ))),
            },
        };
        result.map_err(ReleaseCheckError::from)
    }
}

/// Where the `release.json` of the release marked latest is downloaded
/// from, under a releases location.
pub fn latest_release_json_url(releases_url: &str) -> String {
    format!(
        "{}/latest/download/release.json",
        releases_url.trim_end_matches('/')
    )
}

/// Where the `release.json` of the release tagged `tag` is downloaded from,
/// under a releases location.
pub fn release_json_url(releases_url: &str, tag: &str) -> String {
    format!(
        "{}/download/{tag}/release.json",
        releases_url.trim_end_matches('/')
    )
}

/// The human-readable page for one release tag, under a releases location.
pub fn release_page_url(releases_url: &str, tag: &str) -> String {
    format!("{}/tag/{tag}", releases_url.trim_end_matches('/'))
}

/// Why a release could not be read or could not be believed.
///
/// The two are different facts and a caller has to be able to tell them
/// apart. A release that could not be *reached* says nothing about whether
/// it is real: the network was down, a server failed, a rate limit applied,
/// and the right response is to try again later without alarm. A release
/// whose bytes arrived and failed a check, or that names a file which is not
/// there, is evidence: either the release is broken or someone has
/// tampered with it, and a client that has been told an update exists must
/// say so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseCheckError {
    /// The bytes could not be fetched. Not a security signal.
    Unreachable(String),
    /// The bytes were fetched, or were missing where the release said they
    /// would be, and the release does not verify.
    Rejected(String),
}

impl ReleaseCheckError {
    pub fn message(&self) -> &str {
        match self {
            ReleaseCheckError::Unreachable(m) | ReleaseCheckError::Rejected(m) => m,
        }
    }
}

impl std::fmt::Display for ReleaseCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl From<ReleaseCheckError> for AppError {
    fn from(e: ReleaseCheckError) -> Self {
        AppError::Update {
            message: e.message().to_string(),
        }
    }
}

/// Every verification stage reports through `AppError::Update`, and every
/// one of them is a judgement about bytes that arrived. Only the fetch layer
/// can say a failure was the network's, and it says so through
/// [`FetchFailure`] instead of this conversion.
impl From<AppError> for ReleaseCheckError {
    fn from(e: AppError) -> Self {
        match e {
            AppError::Update { message } => ReleaseCheckError::Rejected(message),
            other => ReleaseCheckError::Rejected(other.to_string()),
        }
    }
}

/// How one fetch failed, before a caller decides what that means.
///
/// `Missing` is kept apart from `Unreachable` because the two callers read it
/// differently: for discovery, a missing `release.json` means nothing is
/// marked latest; for a file a release names, a missing file is a release
/// that does not hold together.
#[derive(Debug)]
pub(crate) enum FetchFailure {
    /// Transport failure, a timeout, or a status that says the server could
    /// not answer (a 5xx, a rate limit, an authorization wall).
    Unreachable(String),
    /// The server answered that there is no such file (404 or 410).
    Missing(String),
    /// The server answered with something this will not read: a body over
    /// the bound, or a request that could not be formed.
    Refused(String),
}

impl FetchFailure {
    fn message(&self) -> &str {
        match self {
            FetchFailure::Unreachable(m) | FetchFailure::Missing(m) | FetchFailure::Refused(m) => m,
        }
    }
}

impl std::fmt::Display for FetchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl From<FetchFailure> for ReleaseCheckError {
    fn from(f: FetchFailure) -> Self {
        match f {
            FetchFailure::Unreachable(m) => ReleaseCheckError::Unreachable(m),
            FetchFailure::Missing(m) | FetchFailure::Refused(m) => ReleaseCheckError::Rejected(m),
        }
    }
}

impl From<FetchFailure> for AppError {
    fn from(f: FetchFailure) -> Self {
        AppError::Update {
            message: f.message().to_string(),
        }
    }
}

/// The trust pins one verification runs against.
///
/// Production uses [`ReleasePins::embedded`], every value from the trust
/// root compiled into this build. The struct exists so tests can move a pin
/// and watch the real cryptography refuse — the verification code is the
/// same either way, and nothing outside a test constructs anything else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleasePins {
    /// Fulcio SAN identity glob CI's manifest signature must match.
    pub ci_identity_pattern: String,
    /// OIDC issuer the Fulcio certificate must record.
    pub ci_issuer: String,
    /// `sha256(SPKI DER)` of every key whose attestation counts.
    pub trusted_attestant_fingerprints: Vec<String>,
    /// How many independent attestations a release needs.
    pub min_human_attestations: u32,
}

impl ReleasePins {
    pub fn embedded() -> Self {
        Self {
            ci_identity_pattern: trust_root::EXPECTED_CI_IDENTITY_PATTERN.to_string(),
            ci_issuer: trust_root::EXPECTED_CI_ISSUER.to_string(),
            trusted_attestant_fingerprints: trust_root::TRUSTED_ATTESTANT_FINGERPRINTS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            min_human_attestations: trust_root::MIN_HUMAN_ATTESTATIONS,
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

fn read_fixture(dir: &Path, name: &str) -> Result<Vec<u8>, FetchFailure> {
    let path = dir.join(name);
    std::fs::read(&path).map_err(|e| {
        let message = format!("reading fixture {}: {e}", path.display());
        if e.kind() == std::io::ErrorKind::NotFound {
            FetchFailure::Missing(message)
        } else {
            FetchFailure::Unreachable(message)
        }
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
    /// The manifest hash the attestant swore to inside the signed prose.
    /// The updater binds this to the sha256 of the manifest it actually
    /// fetched (see `verify_release`) — the template `cross_checks` only
    /// bind prose values to `release.json`, which is unsigned.
    pub artifact_manifest_sha256: String,
    /// What the attestant swore about the macOS signature material, when
    /// the attestation's schema records it. `None` for a schema that
    /// predates those fields.
    pub apple: Option<AppleSignatureAttestation>,
    /// What the attestant swore about the signed Android APK, when the
    /// attestation's schema records it. `None` for a schema that predates
    /// those fields.
    pub android: Option<AndroidSignatureAttestation>,
    pub fingerprint_hex: String,
    pub rekor_log_index: u64,
    pub attested_at: String,
    pub attestant_statement: String,
    pub claims: Vec<VerifiedClaim>,
}

/// The key-dependent facts about a signed macOS release, as a named human
/// recorded them.
///
/// These are the values [`install::InstallPlan`] needs and the manifest may
/// never hold: the envelope's hash, the installable's hash, and the
/// identity the reconstruction must claim once the envelope is applied.
/// The manifest is a function of source; a signature is a function of a
/// key, so this is the layer that can carry it — see `docs/trust-root.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppleSignatureAttestation {
    /// Lowercase hex sha256 of the container a browser downloads.
    pub shipped_artifact_sha256: String,
    /// Lowercase hex sha256 of the published detached signature material.
    pub signature_bundle_sha256: String,
    /// `None` where the signature names no team — an ad-hoc signature has
    /// no Developer ID behind it, and saying so is not the same as
    /// omitting the fact.
    pub team_id: Option<String>,
    pub signing_identifier: String,
}

/// The key-dependent facts about the Android APK a release publishes, as a
/// named human recorded them.
///
/// The manifest records the *unsigned* APK, a function of source; the APK
/// a user installs is that file signed, and a signature is a function of a
/// key, so its hash and the certificate it is signed with belong here. A
/// signed APK reduces to the unsigned one once its signing block is
/// stripped (`mobile/android/tools/strip_apk_signature.py`), and the
/// certificate's SHA-256 is what Android, and tools such as AppVerifier,
/// show for an installed app — the identity a user compares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AndroidSignatureAttestation {
    /// Lowercase hex sha256 of the signed APK the release publishes.
    pub signed_apk_sha256: String,
    /// Lowercase hex sha256 of the DER signing certificate (APK Signature
    /// Scheme v2/v3), as `apksigner verify --print-certs` prints it.
    pub signing_certificate_sha256: String,
}

#[derive(Debug, Clone)]
pub struct VerifiedClaim {
    pub claim_id: String,
    pub statement: String,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Fetch the latest release index, run the verification pipeline, and
/// return a [`ReleaseSummary`] ready for the UI to display.
///
/// Returns `Ok(None)` if no update is available (nothing is marked latest,
/// or the latest release matches or is older than the installed version).
/// Returns `Err` if the latest release fails any pipeline stage — the UI
/// surfaces the error and the installed version stays put.
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

    // ── discover, version, schema, continuity ────────────────────────────
    tracer.log("discover", "reading release.json");
    let discovery = discover_next_release(
        fetcher,
        trust_root::UPDATE_RELEASES_URL,
        installed_version,
        installed_git_commit,
    )
    .await
    .map_err(|e| AppError::from(e.error))?;
    let release = match discovery {
        Discovery::NothingMarked => {
            tracer.log("discover", "no release is marked latest; no update");
            return Ok(None);
        }
        Discovery::NotNewer { version } => {
            tracer.log(
                "version",
                format!("latest is {version}; installed {installed_version} is current"),
            );
            return Ok(None);
        }
        Discovery::Next {
            release,
            latest_version,
        } => {
            if release.version != latest_version {
                tracer.log(
                    "discover",
                    format!(
                        "latest is {latest_version}; {} is the next release after the installed one",
                        release.version
                    ),
                );
            }
            *release
        }
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
        (None, Some(_)) => { /* gate_release already refused */ }
    }

    let evidence =
        verify_release_traced(fetcher, &release, &ReleasePins::embedded(), &tracer).await?;
    tracer.log("present", format!("release {} verified", release.version));
    Ok(Some(evidence.summary))
}

/// What reading a `release.json` against the installed version found,
/// before any signed material is fetched.
#[derive(Debug)]
pub enum ReleaseGate {
    /// The release is not newer than the one installed. Nothing about it is
    /// checked: a client has no business judging a release it would not
    /// install, and a release this build cannot parse (a schema from after
    /// it) must not turn into a warning merely by being current.
    NotNewer { version: String },
    /// A newer release whose index parsed, holds to the schema it declares,
    /// names the tag its version implies, and continues from the installed
    /// commit when one is known. Its signed material is still unchecked.
    Newer(Box<ReleaseIndex>),
}

/// Why a `release.json` could not be gated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateError {
    /// The version the document claims, when it was readable and newer than
    /// the installed one. `None` means the document never got that far, and
    /// so never claimed an update.
    pub newer_version: Option<String>,
    pub reason: String,
}

/// Read a `release.json` against the installed version.
///
/// **The version is read first, and alone.** Whether a release is newer is
/// the only question worth asking of a release that is not, so the
/// document's own schema, shape and continuity are checked only once it
/// claims to be an update. From that point every refusal names the version,
/// because a document that claims an update and does not hold together is
/// the thing a client must surface.
pub fn gate_release(
    release_json_bytes: &[u8],
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<ReleaseGate, GateError> {
    let (value, version) = match read_version(release_json_bytes, installed_version)? {
        (_, version, VersionCompare::OlderOrEqual) => {
            return Ok(ReleaseGate::NotNewer { version });
        }
        (value, version, VersionCompare::Newer) => (value, version),
    };

    let refuse = |reason: String| GateError {
        newer_version: Some(version.clone()),
        reason,
    };

    let release: ReleaseIndex = serde_json::from_value(value)
        .map_err(|e| refuse(format!("release.json failed to parse: {e}")))?;

    if !trust_root::SUPPORTED_RELEASE_SCHEMA_VERSIONS.contains(&release.schema_version) {
        return Err(refuse(format!(
            "release.json schema_version `{}` is not in the supported set {:?} \
             — this client cannot parse the release safely; install a newer client.",
            release.schema_version,
            trust_root::SUPPORTED_RELEASE_SCHEMA_VERSIONS,
        )));
    }
    check_schema_shape(&release).map_err(|e| refuse(update_message(e)))?;

    // The tag is what CI's signing identity is bound to, so the version a
    // client is told about and the tag it checks must be one fact.
    let expected_tag = format!("v{}", release.version);
    if release.git_tag != expected_tag {
        return Err(refuse(format!(
            "release.json names version `{}` but tag `{}`; a release's tag is `{expected_tag}`",
            release.version, release.git_tag
        )));
    }

    check_continuity(&release, installed_git_commit).map_err(|e| refuse(update_message(e)))?;

    Ok(ReleaseGate::Newer(Box::new(release)))
}

/// A `release.json`'s `version`, read alone and compared with the installed
/// one. The refusal carries no version: a document that cannot say what it
/// is has not claimed to be an update.
fn read_version(
    release_json_bytes: &[u8],
    installed_version: &str,
) -> Result<(serde_json::Value, String, VersionCompare), GateError> {
    let unreadable = |reason: String| GateError {
        newer_version: None,
        reason,
    };
    let value: serde_json::Value = serde_json::from_slice(release_json_bytes)
        .map_err(|e| unreadable(format!("release.json failed to parse: {e}")))?;
    let version = value
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| unreadable("release.json has no string `version`".to_string()))?
        .to_string();
    let compared =
        compare_versions(&version, installed_version).map_err(|e| unreadable(update_message(e)))?;
    Ok((value, version, compared))
}

/// What discovery found: whether there is a release to offer, and which.
#[derive(Debug)]
pub enum Discovery {
    /// Nothing is marked latest.
    NothingMarked,
    /// The release marked latest is not newer than the one installed.
    NotNewer { version: String },
    /// The release to verify and offer next, gated
    /// ([`ReleaseGate::Newer`]) and with its signed material unchecked.
    /// `latest_version` is the release marked latest, which is this one
    /// unless the installed release is more than one behind it.
    Next {
        release: Box<ReleaseIndex>,
        latest_version: String,
    },
}

/// Why discovery stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryError {
    /// The version the release marked latest claims, once it has claimed to
    /// be newer than the installed one. `None` means nothing claimed an
    /// update.
    pub newer_version: Option<String>,
    pub error: ReleaseCheckError,
}

/// How many releases [`discover_next_release`] reads back through before it
/// gives up. A client this far behind updates by a manual install.
pub const MAX_RELEASE_HOPS: usize = 64;

/// The release this client should be offered next.
///
/// **A release is checked by the clients of the release before it**, against
/// the pins *they* embed (`releases/README.md` → Rotating the CI signing
/// workflow). So when the installed commit is known, the release to offer
/// is the one whose `previous_release` is that commit: the latest when the
/// install is one release behind, and otherwise the first release after the
/// installed one along the latest's history. A client several releases
/// behind takes them one at a time, and each is verified under the pins of
/// the release it replaces, which is the only chain of pins that holds
/// across a rotation.
///
/// The history is read back from the latest through each `release.json`'s
/// `previous_release`, by tag, through the same download redirects
/// discovery uses. Those documents are unsigned and are used to *find* the
/// next release, never to believe it: the release found is gated with the
/// installed commit (strict continuity) and then has to pass
/// [`verify_release`], whose human attestation names the previous commit it
/// was sworn to. Only the fields the walk needs are read from the releases
/// in between, so a later release in a schema this build cannot parse does
/// not stop a client from taking the one it can.
///
/// With no installed commit known (a fresh install, or one replaced behind
/// the updater's back) there is nothing to anchor continuity to, and the
/// release offered is the oldest one newer than the installed version along
/// the latest's history, gated without a commit. That is still the release
/// whose clients' pins are the ones this build holds, so a fresh install of
/// an older release steps across a pin rotation instead of reporting the
/// rotated release as unverifiable.
pub async fn discover_next_release(
    fetcher: &Fetcher,
    releases_url: &str,
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<Discovery, DiscoveryError> {
    let latest_bytes = match fetcher.fetch_latest_release_json(releases_url).await {
        Ok(Some(bytes)) => bytes,
        Ok(None) => return Ok(Discovery::NothingMarked),
        Err(error) => {
            return Err(DiscoveryError {
                newer_version: None,
                error,
            });
        }
    };
    let latest_version = match read_version(&latest_bytes, installed_version) {
        Ok((_, version, VersionCompare::OlderOrEqual)) => {
            return Ok(Discovery::NotNewer { version });
        }
        Ok((_, version, VersionCompare::Newer)) => version,
        Err(GateError {
            newer_version,
            reason,
        }) => {
            return Err(DiscoveryError {
                newer_version,
                error: ReleaseCheckError::Rejected(reason),
            });
        }
    };
    // From here the latest has claimed an update, so every refusal names it.
    let claimed = |error: ReleaseCheckError| DiscoveryError {
        newer_version: Some(latest_version.clone()),
        error,
    };
    let refuse = |reason: String| claimed(ReleaseCheckError::Rejected(reason));
    // The release found, held to its schema and to continuity from the
    // installed commit when one is known.
    let offer = |bytes: &[u8]| match gate_release(bytes, installed_version, installed_git_commit) {
        Ok(ReleaseGate::Newer(release)) => Ok(Discovery::Next {
            release,
            latest_version: latest_version.clone(),
        }),
        Ok(ReleaseGate::NotNewer { version }) => Err(refuse(format!(
            "release {version} was found as the next release but is not newer than the \
             installed version {installed_version}"
        ))),
        Err(GateError { reason, .. }) => Err(refuse(reason)),
    };

    let mut bytes = latest_bytes;
    for _ in 0..MAX_RELEASE_HOPS {
        let link = ReleaseLink::read(&bytes).map_err(refuse)?;
        let Some((prev_version, prev_commit)) = link.previous else {
            return match installed_git_commit {
                // The oldest release there is, and newer than this build.
                None => offer(&bytes),
                Some(installed_commit) => Err(refuse(format!(
                    "release {} declares no previous release, so the history of the latest \
                     release {latest_version} does not reach the installed commit \
                     `{installed_commit}`",
                    link.version
                ))),
            };
        };
        if installed_git_commit.is_some_and(|c| prev_commit.eq_ignore_ascii_case(c)) {
            return offer(&bytes);
        }
        let older = compare_versions(&prev_version, installed_version)
            .map_err(|e| refuse(update_message(e)))?;
        if matches!(older, VersionCompare::OlderOrEqual) {
            return match installed_git_commit {
                // This release is the oldest one newer than the installed
                // version.
                None => offer(&bytes),
                // The history went past the installed release without
                // passing through it: a fork, or a release.json describing
                // someone else's history.
                Some(installed_commit) => Err(refuse(format!(
                    "continuity failure: the history of the latest release {latest_version} \
                     passes release {prev_version} (commit `{prev_commit}`) without reaching \
                     the installed commit `{installed_commit}` ({installed_version}). Either \
                     the installed release is not one of this line's, or someone served a \
                     forked release.json."
                ))),
            };
        }
        let tag = format!("v{prev_version}");
        let fetched = fetcher
            .fetch_release_json(releases_url, &tag)
            .await
            .map_err(&claimed)?;
        let found = ReleaseLink::read(&fetched).map_err(refuse)?;
        if found.version != prev_version || !found.git_commit.eq_ignore_ascii_case(&prev_commit) {
            return Err(refuse(format!(
                "release {} names its previous release as {prev_version} at commit \
                 `{prev_commit}`, but the release.json under {tag} is {} at commit `{}`",
                link.version, found.version, found.git_commit
            )));
        }
        bytes = fetched;
    }
    Err(refuse(format!(
        "the installed release is more than {MAX_RELEASE_HOPS} releases behind the latest \
         ({latest_version}); install the latest release manually"
    )))
}

/// The fields of a `release.json` that place it in a history, read without
/// holding the document to any schema.
struct ReleaseLink {
    version: String,
    git_commit: String,
    previous: Option<(String, String)>,
}

impl ReleaseLink {
    fn read(bytes: &[u8]) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| format!("release.json failed to parse: {e}"))?;
        let string = |v: &serde_json::Value, name: &str| -> Result<String, String> {
            v.get(name)
                .and_then(|s| s.as_str())
                .map(str::to_string)
                .ok_or_else(|| format!("release.json has no string `{name}`"))
        };
        let previous = match value.get("previous_release") {
            None | Some(serde_json::Value::Null) => None,
            Some(prev) => Some((string(prev, "version")?, string(prev, "git_commit")?)),
        };
        Ok(Self {
            version: string(&value, "version")?,
            git_commit: string(&value, "git_commit")?,
            previous,
        })
    }
}

/// What a release that verified is made of: the summary a person approves,
/// and the signed manifest that every artifact hash comes from.
#[derive(Debug, Clone)]
pub struct ReleaseEvidence {
    pub summary: ReleaseSummary,
    /// The CI-signed `artifact-manifest.json`, exactly as fetched.
    pub manifest_bytes: Vec<u8>,
    /// Lowercase hex sha256 of `manifest_bytes`, which every counted
    /// attestation names.
    pub manifest_sha256: String,
    /// The Fulcio SAN identity CI's signature verified under, bound to this
    /// release's tag.
    pub ci_identity: String,
    /// Rekor log index of CI's manifest signature.
    pub ci_rekor_log_index: u64,
}

/// Verify a gated release: CI's signature over the manifest, bound to the
/// release's tag, and at least `pins.min_human_attestations` independent
/// human attestations over that same manifest.
///
/// This is the whole of what makes a release real to this client. Both the
/// update notification and the install path call it, and neither adds a
/// shortcut around it.
pub async fn verify_release(
    fetcher: &Fetcher,
    release: &ReleaseIndex,
    pins: &ReleasePins,
) -> Result<ReleaseEvidence, ReleaseCheckError> {
    verify_release_traced(fetcher, release, pins, &Tracer::new(false)).await
}

async fn verify_release_traced(
    fetcher: &Fetcher,
    release: &ReleaseIndex,
    pins: &ReleasePins,
    tracer: &Tracer,
) -> Result<ReleaseEvidence, ReleaseCheckError> {
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
    let verified_ci = ci_sigstore::verify_ci_signature_with(
        &manifest_bytes,
        &bundle_bytes,
        &trust,
        &pins.ci_identity_pattern,
        &pins.ci_issuer,
    )?;

    // The Fulcio SAN ends in `@refs/tags/<tag>`. Requiring that tag to be
    // this release's stops an authentic manifest from an older release
    // being offered under a newer one.
    match verified_ci.ci_identity.rsplit_once("@refs/tags/") {
        Some((_, signed_tag)) if signed_tag == release.git_tag => {}
        Some((_, signed_tag)) => {
            return Err(ReleaseCheckError::Rejected(format!(
                "the manifest signature is from the pinned release identity, but for tag \
                 `{signed_tag}`, not this release's tag `{}` — an authentic file appears \
                 to have been replayed under a different release",
                release.git_tag
            )));
        }
        None => {
            return Err(ReleaseCheckError::Rejected(format!(
                "the signing identity `{}` is not bound to a release tag (`@refs/tags/…`)",
                verified_ci.ci_identity
            )));
        }
    }
    tracer.log(
        "verify-ci",
        format!(
            "ok ({}, bound to {})",
            verified_ci.ci_identity, release.git_tag
        ),
    );

    let manifest_sha256_hex = human_attestation::hex_encode(&sha2::Sha256::digest(&manifest_bytes));

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
        let verified = human_attestation::verify_human_attestation_with(
            &attestation_bytes,
            &bundle_bytes,
            release,
            &trust,
            &pins.trusted_attestant_fingerprints,
        )?;
        // The attestant_id in release.json must match what the
        // attestation prose says — defends against a release manifest
        // listing one attestant_id but pointing to a different person's
        // signed prose.
        if verified.attestant_id != human.attestant_id {
            return Err(ReleaseCheckError::Rejected(format!(
                "release.human_attestations[].attestant_id `{}` ≠ attestation prose \
                 attestant.id `{}`",
                human.attestant_id, verified.attestant_id
            )));
        }
        // The attestant swears to a specific manifest hash inside the
        // signed prose; bind it to the manifest actually fetched and
        // CI-verified above, so a compromised pipeline can't pair a
        // legitimate human attestation with a *different* CI-signed
        // manifest for the same version and commit.
        if !verified
            .artifact_manifest_sha256
            .eq_ignore_ascii_case(&manifest_sha256_hex)
        {
            return Err(ReleaseCheckError::Rejected(format!(
                "attestation `{}` attests artifact-manifest.json sha256 `{}`, but the \
                 fetched manifest hashes to `{}`",
                verified.attestant_id, verified.artifact_manifest_sha256, manifest_sha256_hex
            )));
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

    // Independence: each counted attestation must be signed by a distinct
    // hardware key. `release.json` is unsigned and attacker-suppliable, so
    // without this check an index listing the same attestation twice (or
    // one key signing under two attestant ids) would count twice toward
    // the threshold below.
    let mut seen_fingerprints = std::collections::HashSet::new();
    for att in &verified_attestations {
        if !seen_fingerprints.insert(att.fingerprint_hex.to_ascii_lowercase()) {
            return Err(ReleaseCheckError::Rejected(format!(
                "release lists multiple attestations signed by the same key \
                 (fingerprint sha256:{}); attestations must be independent",
                att.fingerprint_hex
            )));
        }
    }

    // Policy: minimum number of independently-verified attestations. The
    // threshold is pinned in the *embedded* trust root rather than in
    // `release.json` so an adversary who controls the index can't lower
    // the bar — see module docs and `docs/trust-root.md`.
    if (verified_attestations.len() as u32) < pins.min_human_attestations {
        return Err(ReleaseCheckError::Rejected(format!(
            "release verified only {} human attestation(s); embedded policy requires ≥{}",
            verified_attestations.len(),
            pins.min_human_attestations,
        )));
    }
    tracer.log(
        "policy",
        format!(
            "{} attestation(s) verified (>= {} required)",
            verified_attestations.len(),
            pins.min_human_attestations
        ),
    );

    // Turning "manifest authentic" into "downloaded bytes authentic" is
    // [`install::stage`]'s job, not this function's: it hashes each
    // downloaded artifact against what the manifest here declares, and
    // the natural place to do that is where the bytes are being fetched
    // anyway. This function stops at verified evidence because the user
    // approves a release *before* anything is downloaded.

    Ok(ReleaseEvidence {
        summary: ReleaseSummary {
            version: release.version.clone(),
            git_commit: release.git_commit.clone(),
            git_tag: release.git_tag.clone(),
            released_at: release.released_at.clone(),
            previous_release: release.previous_release.clone(),
            attestations: verified_attestations,
        },
        manifest_bytes,
        manifest_sha256: manifest_sha256_hex,
        ci_identity: verified_ci.ci_identity,
        ci_rekor_log_index: verified_ci.rekor_log_index,
    })
}

/// The text of an `AppError` from one of the gate's checks, without the
/// variant's own prefix.
fn update_message(e: AppError) -> String {
    match e {
        AppError::Update { message } => message,
        other => other.to_string(),
    }
}

/// The gate's answer in the shape the unit tests below read: a release to
/// verify, "no update", or a refusal.
#[cfg(test)]
#[derive(Debug)]
enum NoUpdateOrError {
    NoUpdate,
    Error(AppError),
}

#[cfg(test)]
fn parse_and_gate_release(
    release_json_bytes: &[u8],
    installed_version: &str,
    installed_git_commit: Option<&str>,
) -> Result<ReleaseIndex, NoUpdateOrError> {
    match gate_release(release_json_bytes, installed_version, installed_git_commit) {
        Ok(ReleaseGate::Newer(release)) => Ok(*release),
        Ok(ReleaseGate::NotNewer { .. }) => Err(NoUpdateOrError::NoUpdate),
        Err(e) => Err(NoUpdateOrError::Error(AppError::Update {
            message: e.reason,
        })),
    }
}

/// The schema a release document *declares* is the shape it is held to.
///
/// Both directions matter, and for the same reason a manifest is held to
/// its own declared schema (`releases/README.md`, and `crate::updates`):
/// a rotation ships acceptance one release before emission, so between
/// those releases this client reads the old shape and the new one as
/// equally authentic — but a document that declares the old version while
/// carrying the new fields, or declares the new one and omits what it
/// promises, is not a shape either release ever produced. Tolerating
/// either would make the version number decorative.
fn check_schema_shape(release: &ReleaseIndex) -> Result<(), AppError> {
    const ARTIFACT_INDEX_SINCE: u32 = 2;

    let refuse = |message: String| AppError::Update { message };

    if release.schema_version < ARTIFACT_INDEX_SINCE {
        if release.artifacts.is_some() {
            return Err(refuse(format!(
                "release.json declares schema_version `{}` but carries an `artifacts` \
                 index, which releases record only from schema {ARTIFACT_INDEX_SINCE} on",
                release.schema_version,
            )));
        }
        if release.apple_signature_bundle.is_some() {
            return Err(refuse(format!(
                "release.json declares schema_version `{}` but carries \
                 `apple_signature_bundle`, which releases record only from schema \
                 {ARTIFACT_INDEX_SINCE} on",
                release.schema_version,
            )));
        }
    } else if release.artifacts.is_none() {
        // `apple_signature_bundle` stays optional at schema 2: a release
        // with no signed macOS installable publishes none.
        return Err(refuse(format!(
            "release.json declares schema_version `{}` but carries no `artifacts` \
             index, which every release from schema {ARTIFACT_INDEX_SINCE} on records",
            release.schema_version,
        )));
    }

    Ok(())
}

enum VersionCompare {
    Newer,
    OlderOrEqual,
}

fn compare_versions(latest: &str, installed: &str) -> Result<VersionCompare, AppError> {
    let latest_v = semver::Version::parse(latest).map_err(|e| AppError::Update {
        message: format!("release.json `version` is not valid semver ({latest}): {e}"),
    })?;
    let installed_v = semver::Version::parse(installed).map_err(|e| AppError::Update {
        message: format!("installed version is not valid semver ({installed}): {e}"),
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
) -> Result<(), AppError> {
    // First install (no prior commit) bypasses continuity. There's no
    // installed state to anchor against, so we trust the release purely
    // on the cryptographic verification stages (and the user's review of
    // the attestation prose at install time).
    let Some(installed) = installed_git_commit else {
        return Ok(());
    };

    let Some(prev) = release.previous_release.as_ref() else {
        return Err(AppError::Update {
            message: format!(
                "client has installed git_commit `{installed}` but the new release.json \
                 declares no `previous_release` — cannot verify continuity"
            ),
        });
    };

    // Strict equality. v1 doesn't attempt fast-forward / multi-hop
    // reachability — that requires querying GitHub for commit ancestry,
    // and the simpler "must update through each release in order" model is
    // sufficient until the cadence makes it painful.
    if !prev.git_commit.eq_ignore_ascii_case(installed) {
        return Err(AppError::Update {
            message: format!(
                "continuity failure: new release.previous_release.git_commit is `{}`, \
                 but the installed git_commit is `{}`. The new release is not a direct \
                 successor; either you have an old client that skipped a release, or \
                 someone served a forked release.json.",
                prev.git_commit, installed
            ),
        });
    }

    Ok(())
}

/// Fetch a single URL's body bytes. Errors carry the file label so the
/// caller doesn't have to reformat them.
/// As much of a failed response as is worth repeating back.
///
/// An error body is a diagnostic, and a hostile one is whatever the server
/// felt like sending — up to the ceiling the read allows. Interpolating it
/// whole would allocate a second copy of it, and a lossy conversion of
/// invalid UTF-8 grows: every bad byte becomes a three-byte replacement.
/// So it is cut first, on a character boundary, and cut again if the
/// conversion inflates it.
const ERROR_BODY_EXCERPT: usize = 1024;

fn excerpt(bytes: &[u8]) -> String {
    let head = &bytes[..bytes.len().min(ERROR_BODY_EXCERPT)];
    let mut text = String::from_utf8_lossy(head).trim().to_string();
    if text.len() > ERROR_BODY_EXCERPT {
        // The lossy conversion inflated it; cut on a character boundary.
        let mut end = ERROR_BODY_EXCERPT;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    if bytes.len() > head.len() {
        text.push('…');
    }
    text
}

/// How long a transfer may say nothing before it counts as dead.
///
/// Shorter than the model downloader's equivalent because these fetches
/// sit behind something a person is waiting on, and because the bodies are
/// small enough that a live connection has no reason to go quiet this long.
pub(crate) const FETCH_STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// The most any signed release document may be. Manifests and
/// attestations are kilobytes; this is a ceiling, not a size.
pub(crate) const MAX_DOCUMENT_BYTES: u64 = 16 * 1024 * 1024;

/// Fetch a URL, refusing a body larger than `max_bytes`.
///
/// The bound is applied while reading rather than after: `Content-Length`
/// is a claim by the same server that supplies the body, so it is used
/// only to refuse early, never to decide when to stop. Every URL that
/// reaches here came from an unsigned index, so a hostile one is in scope.
///
/// No error text here carries the URL. `reqwest` attaches it to the error
/// it returns and prints it in `Display`, and a redirect can leave a
/// credential in it, so errors go through `request_error_text` — the one
/// path a `crate::transport::Error` becomes text through in this crate.
async fn fetch_url_network(
    client: &crate::transport::Client,
    url: &str,
    label: &str,
    max_bytes: u64,
) -> Result<Vec<u8>, FetchFailure> {
    fetch_url_network_with(
        client,
        url,
        "application/octet-stream",
        label,
        max_bytes,
        FETCH_STALL_TIMEOUT,
    )
    .await
}

/// The body of [`fetch_url_network`], with the silence bound supplied —
/// which is how a test drives it in milliseconds rather than minutes.
async fn fetch_url_network_with(
    client: &crate::transport::Client,
    url: &str,
    accept: &str,
    label: &str,
    max_bytes: u64,
    stall: std::time::Duration,
) -> Result<Vec<u8>, FetchFailure> {
    use futures_util::StreamExt;

    // Connecting is bounded on the client and each body read is bounded
    // below, which leaves the gap between them: a server can complete the
    // handshake and then never send a status line. `send()` resolves when
    // the *headers* arrive, so that wait needs the same silence bound —
    // and not a whole-request timeout, which would cut off a large
    // container that is still arriving.
    let resp =
        match crate::rt::timeout(stall, client.get(url).header("Accept", accept).send()).await {
            Ok(sent) => sent.map_err(|e| {
                let malformed = e.is_builder();
                let message = format!("GET {label}: {}", crate::error::request_error_text(e));
                if malformed {
                    FetchFailure::Refused(message)
                } else {
                    FetchFailure::Unreachable(message)
                }
            })?,
            Err(_) => {
                return Err(FetchFailure::Unreachable(format!(
                    "{label} sent no response headers within {stall:?}"
                )));
            }
        };
    let status = resp.status();

    if let Some(claimed) = resp.content_length()
        && claimed > max_bytes
    {
        return Err(FetchFailure::Refused(format!(
            "{label} is {claimed} bytes, more than the {max_bytes} this will read"
        )));
    }

    let mut bytes: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    loop {
        // A server that accepts the connection and then says nothing is
        // indistinguishable from a slow one, right up until it never
        // finishes. The bound is per read rather than overall: a large
        // container on a slow link keeps arriving, and only silence is
        // grounds to give up.
        let next = match crate::rt::timeout(stall, stream.next()).await {
            Ok(next) => next,
            Err(_) => {
                return Err(FetchFailure::Unreachable(format!(
                    "{label} stopped sending for {stall:?}"
                )));
            }
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|e| {
            FetchFailure::Unreachable(format!(
                "reading {label} body: {}",
                crate::error::request_error_text(e)
            ))
        })?;
        if bytes.len() as u64 + chunk.len() as u64 > max_bytes {
            return Err(FetchFailure::Refused(format!(
                "{label} is larger than the {max_bytes} bytes this will read"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }

    if !status.is_success() {
        let message = format!("{label} returned HTTP {} ({})", status, excerpt(&bytes));
        // A 404 or 410 is the server answering that the file is not there.
        // Anything else is the server failing to answer — an outage, a rate
        // limit, a wall in front of it — which says nothing about the file.
        return Err(
            if status == http::StatusCode::NOT_FOUND || status == http::StatusCode::GONE {
                FetchFailure::Missing(message)
            } else {
                FetchFailure::Unreachable(message)
            },
        );
    }
    Ok(bytes)
}

pub(crate) fn build_http_client(
    transport: &crate::transport::Transport,
) -> Result<crate::transport::Client, AppError> {
    crate::transport::update_client(transport)
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

    /// The fetch path, against the threat model it actually has: every URL
    /// it is given comes out of an unsigned index, so "hostile" is the
    /// ordinary case rather than the exotic one.
    mod fetching {
        use super::*;

        #[tokio::test(flavor = "multi_thread")]
        async fn refuses_a_body_larger_than_the_ceiling() {
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::any())
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(vec![0u8; 4096]))
                .mount(&server)
                .await;

            let client = build_http_client(&crate::transport::Transport::default()).unwrap();
            let err = fetch_url_network(&client, &format!("{}/big", server.uri()), "a payload", 64)
                .await
                .expect_err("an oversized body must be refused");
            let msg = format!("{err}");
            assert!(
                msg.contains("larger than") || msg.contains("more than"),
                "got: {msg}"
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn refuses_an_oversized_body_before_reading_it() {
            // The claim is the server's, so it is only ever grounds to
            // refuse early — never grounds to stop counting.
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::any())
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(vec![0u8; 4096]))
                .mount(&server)
                .await;

            let client = build_http_client(&crate::transport::Transport::default()).unwrap();
            let err =
                fetch_url_network(&client, &format!("{}/big", server.uri()), "a payload", 100)
                    .await
                    .expect_err("an oversized body must be refused");
            assert!(format!("{err}").contains("4096") || format!("{err}").contains("100"));
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn a_credential_in_the_url_stays_out_of_the_error() {
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::any())
                .respond_with(wiremock::ResponseTemplate::new(404).set_body_string("nope"))
                .mount(&server)
                .await;

            let url = format!(
                "{}/asset.zip?X-Amz-Signature=SUPERSECRETTOKEN",
                server.uri()
            );
            let err = fetch_url_network(&client_for_test(), &url, "a payload", 1024)
                .await
                .expect_err("a 404 is an error");
            let msg = format!("{err}");
            assert!(
                !msg.contains("SUPERSECRETTOKEN"),
                "a redirect can leave a credential in a URL, and this text is rendered: {msg}"
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn a_credential_stays_out_of_a_transport_error_too() {
            // Nothing listening: the failure comes from reqwest itself,
            // which attaches the URL to its own `Display`.
            let url = "http://127.0.0.1:1/asset.zip?X-Amz-Signature=SUPERSECRETTOKEN";
            let err = fetch_url_network(&client_for_test(), url, "a payload", 1024)
                .await
                .expect_err("a closed port is an error");
            let msg = format!("{err}");
            assert!(!msg.contains("SUPERSECRETTOKEN"), "got: {msg}");
            assert!(!msg.contains("127.0.0.1"), "got: {msg}");
        }

        /// A body with **no** `Content-Length`, which is the case the
        /// running count exists for: the early check is a courtesy the
        /// server may decline to make possible, so the bound that matters
        /// is the one applied while reading.
        #[tokio::test(flavor = "multi_thread")]
        async fn refuses_an_unbounded_chunked_body() {
            use tokio::io::AsyncWriteExt;

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                if let Ok((mut socket, _)) = listener.accept().await {
                    // Read the request first; hyper will not accept a
                    // response that arrives before it has sent one.
                    use tokio::io::AsyncReadExt;
                    let mut scratch = [0u8; 2048];
                    let _ = socket.read(&mut scratch).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                        .await;
                    // Keep sending; a reader without a bound never stops.
                    let chunk = format!("{:x}\r\n{}\r\n", 4096, "A".repeat(4096));
                    for _ in 0..64 {
                        if socket.write_all(chunk.as_bytes()).await.is_err() {
                            return;
                        }
                    }
                }
            });

            let url = format!("http://{addr}/asset.zip");
            let err = fetch_url_network(&client_for_test(), &url, "a payload", 1024)
                .await
                .expect_err("a chunked body past the ceiling must be refused");
            let msg = format!("{err}");
            assert!(msg.contains("larger than"), "got: {msg}");
        }

        /// Accept the connection, then say nothing. Without a bound this
        /// never returns — the failure mode is a hang, so the assertion
        /// that matters is that it comes back at all.
        #[tokio::test(flavor = "multi_thread")]
        async fn refuses_a_transfer_that_goes_quiet() {
            use tokio::io::AsyncWriteExt;

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                if let Ok((mut socket, _)) = listener.accept().await {
                    use tokio::io::AsyncReadExt;
                    let mut scratch = [0u8; 2048];
                    let _ = socket.read(&mut scratch).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                        .await;
                    // Headers, then silence.
                    std::future::pending::<()>().await;
                }
            });

            let url = format!("http://{addr}/asset.zip");
            let err = fetch_url_network_with(
                &client_for_test(),
                &url,
                "application/octet-stream",
                "a payload",
                1024,
                std::time::Duration::from_millis(200),
            )
            .await
            .expect_err("a silent transfer must not be waited on forever");
            assert!(format!("{err}").contains("stopped sending"), "got: {err}");
        }

        /// Headers that never arrive. The handshake completes, so the
        /// connect bound is satisfied and the body bound never gets a
        /// chance to apply — this is the gap between the two.
        #[tokio::test(flavor = "multi_thread")]
        async fn refuses_a_server_that_never_sends_headers() {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                if let Ok((socket, _)) = listener.accept().await {
                    // Accept, read nothing, answer nothing, hold it open.
                    let _held = socket;
                    std::future::pending::<()>().await;
                }
            });

            let url = format!("http://{addr}/asset.zip");
            let err = fetch_url_network_with(
                &client_for_test(),
                &url,
                "application/octet-stream",
                "a payload",
                1024,
                std::time::Duration::from_millis(200),
            )
            .await
            .expect_err("a server that never answers must not be waited on forever");
            assert!(
                format!("{err}").contains("no response headers"),
                "got: {err}"
            );
        }

        /// A failed response is a diagnostic. Repeating a hostile one back
        /// whole is a second allocation of whatever it felt like sending —
        /// and invalid UTF-8 grows on the way.
        #[tokio::test(flavor = "multi_thread")]
        async fn quotes_only_an_excerpt_of_a_failure_body() {
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::any())
                .respond_with(
                    wiremock::ResponseTemplate::new(500).set_body_bytes(vec![0xffu8; 200_000]),
                )
                .mount(&server)
                .await;

            let err = fetch_url_network_with(
                &client_for_test(),
                &format!("{}/asset.zip", server.uri()),
                "application/octet-stream",
                "a payload",
                1_000_000,
                std::time::Duration::from_secs(5),
            )
            .await
            .expect_err("HTTP 500 is an error");
            let msg = format!("{err}");
            assert!(msg.contains("500"), "got: {msg}");
            assert!(
                msg.len() < 4096,
                "a failure message must not carry the body back: {} bytes",
                msg.len()
            );
        }

        fn client_for_test() -> crate::transport::Client {
            build_http_client(&crate::transport::Transport::default()).unwrap()
        }
    }

    /// Both halves of the artifact-index rotation, on the shape a release
    /// declares rather than on the shape this build would prefer.
    ///
    /// Between the release that accepts schema 2 and the release that
    /// emits it, both shapes are authentic — but a document that declares
    /// one and carries the other is neither, and reading it as whichever
    /// it resembles would make the version number decorative.
    mod artifact_index_rotation {
        use super::*;

        fn release(schema: u32, extra: &str) -> Vec<u8> {
            format!(
                r#"{{
                    "schema_version": {schema},
                    "version": "1.1.0",
                    "git_commit": "9c3a000000000000000000000000000000000001",
                    "git_tag": "v1.1.0",
                    "released_at": "2026-05-26T17:00:00Z",
                    "artifact_manifest": {{"url":"x","sigstore_bundle_url":"x"}},
                    "human_attestations": [{{"attestant_id":"x","url":"x","bundle_url":"x"}}]
                    {extra}
                }}"#
            )
            .into_bytes()
        }

        const INDEX: &str = r#", "artifacts": {"eidola-gui-macos-universal-zip": {"url": "https://example.com/p.zip"}}"#;

        #[test]
        fn schema_1_without_an_index_is_the_shape_releases_emit_today() {
            let release = parse_and_gate_release(&release(1, ""), "1.0.0", None)
                .expect("today's emitted shape must keep verifying");
            assert!(release.artifacts.is_none());
        }

        #[test]
        fn schema_2_with_an_index_is_accepted_before_anything_emits_it() {
            let release = parse_and_gate_release(&release(2, INDEX), "1.0.0", None)
                .expect("the shape the next rotation emits must already verify");
            assert_eq!(release.artifacts.unwrap().len(), 1);
        }

        #[test]
        fn schema_1_carrying_an_index_is_refused() {
            let err = unwrap_err(parse_and_gate_release(&release(1, INDEX), "1.0.0", None));
            let msg = format!("{err}");
            assert!(msg.contains("only from schema 2 on"), "got: {msg}");
        }

        #[test]
        fn schema_2_without_an_index_is_refused() {
            let err = unwrap_err(parse_and_gate_release(&release(2, ""), "1.0.0", None));
            let msg = format!("{err}");
            assert!(msg.contains("carries no `artifacts`"), "got: {msg}");
        }

        #[test]
        fn schema_1_carrying_signature_material_is_refused() {
            let extra = r#", "apple_signature_bundle": {"url": "https://example.com/s.zip"}"#;
            let err = unwrap_err(parse_and_gate_release(&release(1, extra), "1.0.0", None));
            let msg = format!("{err}");
            assert!(msg.contains("apple_signature_bundle"), "got: {msg}");
        }

        #[test]
        fn schema_2_may_omit_signature_material() {
            // A release with no signed macOS installable publishes none;
            // that is an absence, not an omission.
            parse_and_gate_release(&release(2, INDEX), "1.0.0", None)
                .expect("a release without an envelope must still verify");
        }
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
