//! Runtime resolution of the allowed Tinfoil inference-enclave measurements.
//!
//! # Why this exists (and why it's temporary)
//!
//! Tinfoil's `inference.tinfoil.sh` is a *router* enclave that reverse-
//! proxies to separate per-model GPU enclaves; the router itself trusts
//! those downstream enclaves via "latest Sigstore-signed release of the
//! repo" (see `tinfoil-go`/`tinfoil-rs`). So statically pinning the
//! router's measurement and requiring a human PR review before it can
//! change buys little rigor over what Tinfoil's own clients do — while
//! costing us a fail-closed outage window every time Tinfoil ships a
//! router release (their sign→deploy lag is ~zero; our review→rebuild→
//! promote→deploy lag is hours).
//!
//! This module matches Tinfoil's actual trust model: it resolves the
//! *latest* router release at runtime and verifies its Sigstore DSSE
//! attestation end-to-end ([`sigstore`]) — the same cryptographic bar the
//! client's update verifier holds — then feeds the resulting measurement
//! into our own [`tinfoil_verifier`] (which still performs the superior
//! per-handshake, nonce-fresh, held-connection attestation on every
//! request). The verifier's `allowed_measurements` are baked in at client
//! construction, so to change the set *without* touching `tinfoil-verifier`
//! we rebuild the `reqwest::Client` and hot-swap it via [`ArcSwap`].
//!
//! When we self-host inference we delete this module and hand
//! `attesting_client` a statically pinned measurement set instead.
//!
//! # Failure posture
//!
//! - **Boot:** one synchronous resolution of the *latest* release acts as the
//!   readiness gate. There is no static fallback — if that measurement can't
//!   be resolved and verified at boot (GitHub unreachable, bad signature, …)
//!   the server refuses to start. The initial allowed set also folds in the
//!   *previous* published release (best-effort, non-fatal) so a cold start
//!   during a rolling deploy still attests the draining old enclave —
//!   matching the refresh path's rolling window of 2.
//! - **Refresh:** a resolution or client-rebuild failure keeps the current
//!   set — we never clear trust or widen it on error. A new measurement is
//!   only adopted after the replacement client is built successfully.

pub mod sigstore;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use tinfoil_verifier::EnclaveMeasurement;
use tracing::{error, info, warn};

use sigstore::{TrustError, TrustedRoot, VerifiedMeasurement};

/// Builds a fresh attesting [`reqwest::Client`] for a given allowed
/// measurement set. Supplied by `main.rs` so all the `attesting_client`
/// wiring (TLS roots, ATC, enclave repo, telemetry observers) stays there
/// and this module doesn't take a telemetry dependency. Returns the client
/// or a human-readable error string.
pub type AttestingClientFactory = Arc<
    dyn Fn(
            Vec<EnclaveMeasurement>,
        )
            -> Pin<Box<dyn Future<Output = std::result::Result<reqwest::Client, String>> + Send>>
        + Send
        + Sync,
>;

/// Default refresh cadence. Bounds the race window between Tinfoil cutting a
/// release and deploying the new enclave: if the new enclave is deployed
/// before we've picked up its measurement, requests fail closed until the
/// next tick. 10 minutes keeps that window small while staying well within
/// GitHub's unauthenticated rate limit (~3 calls/refresh).
const DEFAULT_REFRESH_SECS: u64 = 600;

/// Holds the current attesting client and the state needed to refresh it.
/// Shared as `Arc<UpstreamTrust>`; the backend holds a clone of the inner
/// [`ArcSwap`] cell so request paths read the current client lock-free.
pub struct UpstreamTrust {
    client: Arc<ArcSwap<reqwest::Client>>,
    state: tokio::sync::Mutex<ResolverState>,
}

struct ResolverState {
    repo: String,
    github: reqwest::Client,
    factory: AttestingClientFactory,
    trust: TrustedRoot,
    /// The measurement currently at the head of the allowed set (the last
    /// successfully resolved release). Always set once `bootstrap` returns —
    /// there is no fallback, so boot resolves this or the server crashes.
    /// On each change the *prior* value of this field is kept alongside the
    /// new one in the rebuilt client's allowed set (a rolling window of 2)
    /// so an in-flight rolling deploy — old + new router enclaves both live —
    /// still attests during the overlap.
    latest: VerifiedMeasurement,
}

impl UpstreamTrust {
    /// Resolve + verify the latest release's measurement and build the
    /// initial attesting client. There is no fallback: if the *latest*
    /// measurement can't be resolved, this returns `Err` and the server
    /// refuses to start.
    ///
    /// The initial allowed set is a rolling window of 2 — the latest release
    /// plus the immediately-previous published release — mirroring the
    /// refresh path (`refresh_once`). Without the previous entry, a cold
    /// start that lands mid rolling-deploy (when the still-draining previous
    /// router enclave may answer the readiness probe) would fail attestation
    /// and abort startup. The previous entry is *best-effort*: unlike the
    /// latest (the readiness gate), a missing or unverifiable previous
    /// release does not fail boot — we start with the single latest
    /// measurement and the next successful refresh re-establishes the window.
    pub async fn bootstrap(
        repo: String,
        factory: AttestingClientFactory,
    ) -> std::result::Result<Arc<Self>, String> {
        let github = build_github_client()?;
        let trust = sigstore::load_trusted_root().map_err(|e| e.to_string())?;

        let latest_tag = latest_tag(&github, &repo)
            .await
            .map_err(|e| format!("resolving latest upstream release tag at boot: {e}"))?;
        let latest = resolve_tag(&github, &repo, &latest_tag, &trust)
            .await
            .map_err(|e| format!("resolving upstream measurement at boot: {e}"))?;
        info!(
            tag = %latest_tag,
            tag_identity = %latest.ci_identity,
            rekor_log_index = latest.rekor_log_index,
            "Resolved Tinfoil upstream measurement from latest release attestation"
        );

        // Fold in the previous published release so the readiness probe (and
        // early requests) still attest against a router enclave that hasn't
        // finished draining. Best-effort — see the doc comment.
        let mut allowed = vec![to_enclave(&latest)];
        match resolve_previous(&github, &repo, &latest_tag, &trust).await {
            Ok(Some((prev_tag, prev))) => {
                info!(
                    tag = %prev_tag,
                    tag_identity = %prev.ci_identity,
                    "Including previous release measurement in initial allowed set (rolling-deploy overlap)"
                );
                allowed.push(to_enclave(&prev));
            }
            Ok(None) => {
                info!(
                    "No previous published release to include; starting with latest measurement only"
                );
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "Could not resolve previous release measurement at boot; starting with latest only"
                );
            }
        }

        let client = (factory)(allowed)
            .await
            .map_err(|e| format!("building initial attesting client: {e}"))?;

        Ok(Arc::new(Self {
            client: Arc::new(ArcSwap::from_pointee(client)),
            state: tokio::sync::Mutex::new(ResolverState {
                repo,
                github,
                factory,
                trust,
                latest,
            }),
        }))
    }

    /// The cell the backend reads on every request. Cloning the `Arc` is
    /// cheap; `load()` on it is lock-free.
    pub fn client_cell(&self) -> Arc<ArcSwap<reqwest::Client>> {
        self.client.clone()
    }

    /// Spawn the periodic refresh task. Interval comes from
    /// `TINFOIL_MEASUREMENT_REFRESH_SECS` (default hourly).
    pub fn spawn_refresh(self: Arc<Self>) {
        let interval = std::env::var("TINFOIL_MEASUREMENT_REFRESH_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(DEFAULT_REFRESH_SECS));

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // Skip the immediate first tick — bootstrap already resolved.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                self.refresh_once().await;
            }
        });
    }

    /// One refresh cycle: resolve latest, and if the measurement changed,
    /// build a new client over the rolling {new, previous} set and swap it
    /// in. Any failure leaves the current client untouched.
    async fn refresh_once(&self) {
        let mut state = self.state.lock().await;

        let resolved = match resolve_latest(&state.github, &state.repo, &state.trust).await {
            Ok(m) => m,
            Err(e) => {
                warn!(error = %e, "Upstream measurement refresh failed; keeping current set");
                return;
            }
        };

        if same_measurement(&state.latest, &resolved) {
            return; // unchanged
        }

        // Rolling window of 2: the new head plus the prior head, so an
        // in-progress rolling deploy (old + new enclaves both serving)
        // still attests during the overlap.
        let prior = state.latest.clone();
        let allowed = vec![to_enclave(&resolved), to_enclave(&prior)];

        // Build the replacement client BEFORE mutating state or swapping,
        // so a build failure leaves the working client in place.
        let new_client = match (state.factory)(allowed).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "Rebuilding attesting client for new measurement failed; keeping current client");
                return;
            }
        };

        self.client.store(Arc::new(new_client));
        info!(
            tag_identity = %resolved.ci_identity,
            rekor_log_index = resolved.rekor_log_index,
            "Adopted new Tinfoil upstream measurement and swapped attesting client"
        );
        state.latest = resolved;
    }
}

fn same_measurement(a: &VerifiedMeasurement, b: &VerifiedMeasurement) -> bool {
    a.snp_measurement == b.snp_measurement && a.rtmr1 == b.rtmr1 && a.rtmr2 == b.rtmr2
}

fn to_enclave(m: &VerifiedMeasurement) -> EnclaveMeasurement {
    EnclaveMeasurement {
        snp_measurement: m.snp_measurement.clone(),
        tdx_measurement: tinfoil_verifier::TdxMeasurement {
            rtmr1: m.rtmr1.clone(),
            rtmr2: m.rtmr2.clone(),
        },
    }
}

/// A plain HTTPS client (Mozilla WebPKI roots) for GitHub. Uses the
/// process-global rustls `CryptoProvider` installed in `main` — separate
/// from the attesting client (which speaks only to the enclave).
fn build_github_client() -> std::result::Result<reqwest::Client, String> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    reqwest::Client::builder()
        .use_preconfigured_tls(tls)
        .user_agent("eidola-server-measurement-resolver")
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("building GitHub HTTPS client: {e}"))
}

/// Resolve and verify the latest router release's measurement:
/// latest tag → `tinfoil.hash` digest → attestation bundle → full Sigstore
/// DSSE verification (pinned to `repo` + that exact tag) → measurement from
/// the signed in-toto predicate.
async fn resolve_latest(
    github: &reqwest::Client,
    repo: &str,
    trust: &TrustedRoot,
) -> std::result::Result<VerifiedMeasurement, TrustError> {
    let tag = latest_tag(github, repo).await?;
    resolve_tag(github, repo, &tag, trust).await
}

/// GitHub's "latest" published (non-draft, non-prerelease) release tag.
async fn latest_tag(
    github: &reqwest::Client,
    repo: &str,
) -> std::result::Result<String, TrustError> {
    let trust_err = |m: String| TrustError(m);
    let release: serde_json::Value = github
        .get(format!(
            "https://api.github.com/repos/{repo}/releases/latest"
        ))
        .header("accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| trust_err(format!("fetching latest release: {e}")))?
        .error_for_status()
        .map_err(|e| trust_err(format!("latest release request returned error: {e}")))?
        .json()
        .await
        .map_err(|e| trust_err(format!("parsing latest release JSON: {e}")))?;
    release
        .get("tag_name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| trust_err("latest release JSON has no tag_name".into()))
}

/// The most recent published release whose tag precedes `latest_tag` — i.e.
/// the release that was serving before the latest one, and which may still be
/// draining during a rolling deploy. Returns its resolved + verified
/// measurement, or `None` if the repo has only a single published release.
///
/// GitHub's `GET /releases` list is ordered newest-first by `created_at`;
/// after dropping drafts/prereleases (which `releases/latest` also ignores)
/// and the `latest_tag` entry itself, the first remaining release is the
/// previous one.
async fn resolve_previous(
    github: &reqwest::Client,
    repo: &str,
    latest_tag: &str,
    trust: &TrustedRoot,
) -> std::result::Result<Option<(String, VerifiedMeasurement)>, TrustError> {
    let trust_err = |m: String| TrustError(m);
    let releases: serde_json::Value = github
        .get(format!(
            "https://api.github.com/repos/{repo}/releases?per_page=10"
        ))
        .header("accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| trust_err(format!("listing releases: {e}")))?
        .error_for_status()
        .map_err(|e| trust_err(format!("releases list request returned error: {e}")))?
        .json()
        .await
        .map_err(|e| trust_err(format!("parsing releases list JSON: {e}")))?;
    let list = releases
        .as_array()
        .ok_or_else(|| trust_err("releases list JSON is not an array".into()))?;

    let Some(previous_tag) = select_previous_tag(list, latest_tag) else {
        return Ok(None);
    };
    let previous_tag = previous_tag.to_string();
    let measurement = resolve_tag(github, repo, &previous_tag, trust).await?;
    Ok(Some((previous_tag, measurement)))
}

/// Pick the previous published release tag from a newest-first GitHub
/// `/releases` list: the first release that is neither a draft nor a
/// prerelease and whose tag isn't `latest_tag`. Pure (no I/O) so the
/// filtering rules are unit-testable.
fn select_previous_tag<'a>(releases: &'a [serde_json::Value], latest_tag: &str) -> Option<&'a str> {
    releases
        .iter()
        .filter(|r| {
            !r.get("draft").and_then(|v| v.as_bool()).unwrap_or(false)
                && !r
                    .get("prerelease")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
        })
        .filter_map(|r| r.get("tag_name").and_then(|v| v.as_str()))
        .find(|tag| *tag != latest_tag)
}

/// Resolve and verify a specific release tag's measurement:
/// `tinfoil.hash` digest → attestation bundle → full Sigstore DSSE
/// verification (pinned to `repo` + `tag`) → measurement from the signed
/// in-toto predicate.
async fn resolve_tag(
    github: &reqwest::Client,
    repo: &str,
    tag: &str,
    trust: &TrustedRoot,
) -> std::result::Result<VerifiedMeasurement, TrustError> {
    let trust_err = |m: String| TrustError(m);

    // 2. `tinfoil.hash` — the sha256 of tinfoil-deployment.json (the
    //    attestation subject). We look up the attestation by this digest and
    //    later confirm the signed subject digest matches it.
    let digest = github
        .get(format!(
            "https://github.com/{repo}/releases/download/{tag}/tinfoil.hash"
        ))
        .send()
        .await
        .map_err(|e| trust_err(format!("fetching tinfoil.hash: {e}")))?
        .error_for_status()
        .map_err(|e| trust_err(format!("tinfoil.hash request returned error: {e}")))?
        .text()
        .await
        .map_err(|e| trust_err(format!("reading tinfoil.hash body: {e}")))?
        .trim()
        .to_ascii_lowercase();
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(trust_err(format!(
            "tinfoil.hash is not a 64-char sha256 hex digest: `{digest}`"
        )));
    }

    // 3. Attestation bundle(s) for that digest.
    let attestations: serde_json::Value = github
        .get(format!(
            "https://api.github.com/repos/{repo}/attestations/sha256:{digest}"
        ))
        .header("accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| trust_err(format!("fetching attestation bundle: {e}")))?
        .error_for_status()
        .map_err(|e| trust_err(format!("attestation request returned error: {e}")))?
        .json()
        .await
        .map_err(|e| trust_err(format!("parsing attestation JSON: {e}")))?;

    let list = attestations
        .get("attestations")
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
        .ok_or_else(|| trust_err(format!("no attestations returned for sha256:{digest}")))?;

    // 4. Accept the first attestation that verifies end-to-end.
    let mut last_err: Option<TrustError> = None;
    for att in list {
        let Some(bundle) = att.get("bundle") else {
            continue;
        };
        let bundle_bytes = match serde_json::to_vec(bundle) {
            Ok(b) => b,
            Err(e) => {
                last_err = Some(trust_err(format!("re-serializing bundle: {e}")));
                continue;
            }
        };
        match sigstore::verify_release_attestation(&bundle_bytes, repo, tag, &digest, trust) {
            Ok(m) => return Ok(m),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| trust_err("no verifiable attestation bundle found".into())))
}

#[cfg(test)]
mod tests {
    use super::select_previous_tag;
    use serde_json::json;

    #[test]
    fn picks_the_second_published_release() {
        let list = vec![
            json!({ "tag_name": "v0.0.115" }),
            json!({ "tag_name": "v0.0.114" }),
            json!({ "tag_name": "v0.0.113" }),
        ];
        assert_eq!(select_previous_tag(&list, "v0.0.115"), Some("v0.0.114"));
    }

    #[test]
    fn skips_drafts_and_prereleases_ahead_of_the_previous() {
        // Newest-first list where the entries between latest and the real
        // previous are a draft and a prerelease — both must be skipped.
        let list = vec![
            json!({ "tag_name": "v0.0.116", "draft": true }),
            json!({ "tag_name": "v0.0.115" }),
            json!({ "tag_name": "v0.0.115-rc1", "prerelease": true }),
            json!({ "tag_name": "v0.0.114" }),
        ];
        assert_eq!(select_previous_tag(&list, "v0.0.115"), Some("v0.0.114"));
    }

    #[test]
    fn returns_none_for_a_single_published_release() {
        let list = vec![json!({ "tag_name": "v0.0.115" })];
        assert_eq!(select_previous_tag(&list, "v0.0.115"), None);
    }

    #[test]
    fn returns_none_when_only_prereleases_precede_latest() {
        let list = vec![
            json!({ "tag_name": "v0.0.115" }),
            json!({ "tag_name": "v0.0.114-rc1", "prerelease": true }),
        ];
        assert_eq!(select_previous_tag(&list, "v0.0.115"), None);
    }

    #[test]
    fn empty_list_yields_none() {
        assert_eq!(select_previous_tag(&[], "v0.0.115"), None);
    }
}
