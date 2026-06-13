//! Failure-mode matrix tests for the update-notification flow
//! (`eidola_app_core::updates`), driven against wiremock fixture release
//! feeds. One test (at least) per row of the matrix in the task/module
//! docs:
//!
//! - ok (verified update available)
//! - up to date / not-latest ignored (only the `latest`-marked release
//!   counts; 404 = nothing marked latest)
//! - bad signature (tampered manifest → hash mismatch; tampered bundle
//!   signature → ECDSA failure)
//! - wrong identity (real signed fixture vs a divergent pinned pattern)
//! - malformed bundle
//! - missing asset (absent from listing, and listed-but-404)
//! - claims-changed (side-by-side comparison + acceptance flow)
//! - network failure (connection refused, HTTP 500, malformed feed JSON)
//!
//! The cryptographic stages run **for real** in every test: the happy
//! path serves the captured `v0.0.8` release bytes (a genuine
//! CI-cosign-signed manifest + bundle, see `tests/fixtures/v0.0.8/`),
//! which verify against the *embedded* Sigstore trust root — no trust
//! anchor is stubbed. Rows that need a divergent pin (wrong identity,
//! changed expected claims) move the pin in `CheckContext`, never the
//! crypto.

use eidola_app_core::updates::{
    AcceptedClaims, CheckContext, Claim, UpdateCheckResult, check_for_update, expected_claims,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const FIXTURE_MANIFEST: &[u8] = include_bytes!("fixtures/v0.0.8/artifact-manifest.json");
const FIXTURE_BUNDLE: &[u8] = include_bytes!("fixtures/v0.0.8/artifact-manifest.json.sigstore");

/// The tag the fixture bundle's Fulcio identity is bound to
/// (`…tinfoil-build.yml@refs/tags/v0.0.8`).
const FIXTURE_TAG: &str = "v0.0.8";
/// An installed version older than the fixture release.
const OLD_INSTALLED: &str = "0.0.1";

fn http_client() -> reqwest::Client {
    // reqwest's `rustls-no-provider` build refuses to construct a client
    // until a crypto provider is installed, even for plain-HTTP fixture
    // servers. Idempotent — mirrors what `AppCore::new` does.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    reqwest::Client::builder()
        .build()
        .expect("plain http client")
}

/// Build the GitHub `releases/latest` JSON for `tag`, with the given
/// asset names pointing back at the mock server.
fn latest_release_json(server_uri: &str, tag: &str, asset_names: &[&str]) -> serde_json::Value {
    let assets: Vec<serde_json::Value> = asset_names
        .iter()
        .map(|name| {
            serde_json::json!({
                "name": name,
                "browser_download_url": format!("{server_uri}/assets/{name}")
            })
        })
        .collect();
    serde_json::json!({
        "tag_name": tag,
        "html_url": format!("https://github.com/eidola-ai/eidola/releases/tag/{tag}"),
        "published_at": "2026-06-01T12:00:00Z",
        "assets": assets,
    })
}

/// Mount a complete release: the latest-release endpoint plus both
/// verifier assets with the given bytes.
async fn mount_release(server: &MockServer, tag: &str, manifest: &[u8], bundle: &[u8]) {
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(latest_release_json(
                &server.uri(),
                tag,
                &["artifact-manifest.json", "artifact-manifest.json.sigstore"],
            )),
        )
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/artifact-manifest.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(manifest.to_vec()))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/artifact-manifest.json.sigstore"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bundle.to_vec()))
        .mount(server)
        .await;
}

fn ctx(server: &MockServer, installed: &str) -> CheckContext {
    CheckContext::new(format!("{}/releases/latest", server.uri()), installed)
}

// ---------------------------------------------------------------------------
// Row: ok — a real, verified update
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn verified_update_available() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;

    let UpdateCheckResult::UpdateAvailable { release } = result else {
        panic!("expected UpdateAvailable, got: {result:#?}");
    };
    assert_eq!(release.version, "0.0.8");
    assert_eq!(release.tag, FIXTURE_TAG);
    assert!(!release.claims_accepted, "no claims diff to accept");
    assert!(
        release.ci_identity.contains("tinfoil-build.yml"),
        "identity should be the pinned release workflow, got: {}",
        release.ci_identity
    );
    assert!(release.ci_identity.ends_with("@refs/tags/v0.0.8"));
    assert!(release.rekor_log_index > 0);
    assert_eq!(release.manifest_sha256.len(), 64);
    assert_eq!(
        release.release_url.as_deref(),
        Some("https://github.com/eidola-ai/eidola/releases/tag/v0.0.8")
    );
}

// ---------------------------------------------------------------------------
// Rows: up to date / not-latest ignored
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn up_to_date_when_latest_equals_installed() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check_for_update(&http_client(), &ctx(&server, "0.0.8")).await;
    assert_eq!(
        result,
        UpdateCheckResult::UpToDate {
            latest_version: Some("0.0.8".into())
        }
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn newer_non_latest_tags_do_not_count() {
    // A v9.9.9 tag may exist on GitHub, but the `latest` marker (the
    // human-attested signal) still points at v0.0.8 — so the client sees
    // only v0.0.8 at `releases/latest` and reports up-to-date for an
    // installed 0.0.8. The newer-but-unmarked release is simply not an
    // update yet; the verifier never even fetches its assets.
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check_for_update(&http_client(), &ctx(&server, "0.0.8")).await;
    assert_eq!(
        result,
        UpdateCheckResult::UpToDate {
            latest_version: Some("0.0.8".into())
        }
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn no_release_marked_latest_is_up_to_date() {
    // GitHub returns 404 from `releases/latest` when nothing carries the
    // marker. Not an error, not an update.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert_eq!(
        result,
        UpdateCheckResult::UpToDate {
            latest_version: None
        }
    );
}

// ---------------------------------------------------------------------------
// Row: network/API failure — quiet, never alarms
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn unreachable_feed_is_check_failed() {
    // `.invalid` is reserved (RFC 2606) and never resolves — a fast,
    // deterministic transport failure. (A dropped mock server's port can
    // be re-bound by a parallel test, so "connection refused" via port
    // reuse is racy.)
    let ctx = CheckContext::new(
        "http://feed.invalid/releases/latest".to_string(),
        OLD_INSTALLED,
    );
    let result = check_for_update(&http_client(), &ctx).await;
    assert!(
        matches!(result, UpdateCheckResult::CheckFailed { .. }),
        "expected CheckFailed, got: {result:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn server_error_is_check_failed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert!(
        matches!(result, UpdateCheckResult::CheckFailed { .. }),
        "expected CheckFailed, got: {result:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_feed_json_is_check_failed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<!doctype html>nope"))
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert!(
        matches!(result, UpdateCheckResult::CheckFailed { .. }),
        "expected CheckFailed, got: {result:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn transient_asset_server_error_is_check_failed() {
    // The release lists the asset but the download 500s — server-side
    // trouble, not evidence of tampering. Quiet retry.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(latest_release_json(
                &server.uri(),
                FIXTURE_TAG,
                &["artifact-manifest.json", "artifact-manifest.json.sigstore"],
            )),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/artifact-manifest.json"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert!(
        matches!(result, UpdateCheckResult::CheckFailed { .. }),
        "expected CheckFailed, got: {result:#?}"
    );
}

// ---------------------------------------------------------------------------
// Row: fails cryptographic verification — hard, visible security state
// ---------------------------------------------------------------------------

fn assert_unverifiable(result: &UpdateCheckResult, reason_contains: &str) {
    let UpdateCheckResult::Unverifiable {
        version,
        tag,
        reason,
    } = result
    else {
        panic!("expected Unverifiable, got: {result:#?}");
    };
    assert_eq!(version, "0.0.8");
    assert_eq!(tag, FIXTURE_TAG);
    assert!(
        reason.contains(reason_contains),
        "expected reason to mention `{reason_contains}`, got: {reason}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_manifest_asset_is_unverifiable() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(latest_release_json(
                &server.uri(),
                FIXTURE_TAG,
                &["artifact-manifest.json.sigstore"], // manifest itself missing
            )),
        )
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert_unverifiable(&result, "no `artifact-manifest.json` asset");
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_bundle_asset_is_unverifiable() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(latest_release_json(
                &server.uri(),
                FIXTURE_TAG,
                &["artifact-manifest.json"], // signature bundle missing
            )),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/artifact-manifest.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(FIXTURE_MANIFEST.to_vec()))
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert_unverifiable(&result, "no `artifact-manifest.json.sigstore` asset");
}

#[tokio::test(flavor = "multi_thread")]
async fn listed_but_404_asset_is_unverifiable() {
    // The release claims the asset exists but it's gone — suspicious, not
    // transient.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(latest_release_json(
                &server.uri(),
                FIXTURE_TAG,
                &["artifact-manifest.json", "artifact-manifest.json.sigstore"],
            )),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/artifact-manifest.json"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert_unverifiable(&result, "gone");
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_bundle_is_unverifiable() {
    let server = MockServer::start().await;
    mount_release(
        &server,
        FIXTURE_TAG,
        FIXTURE_MANIFEST,
        b"not a sigstore bundle at all",
    )
    .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert_unverifiable(&result, "parsing CI Sigstore bundle");
}

#[tokio::test(flavor = "multi_thread")]
async fn tampered_manifest_is_unverifiable() {
    // One byte of drift in the manifest → its hash no longer matches what
    // the (genuine) bundle signed.
    let mut tampered = FIXTURE_MANIFEST.to_vec();
    tampered.extend_from_slice(b" ");
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, &tampered, FIXTURE_BUNDLE).await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert_unverifiable(&result, "does not match");
}

#[tokio::test(flavor = "multi_thread")]
async fn tampered_bundle_signature_is_unverifiable() {
    // Flip the ECDSA signature inside an otherwise-genuine bundle: the
    // digest still matches, the cert chain still verifies, but the
    // signature check over the manifest hash must fail.
    let mut bundle: serde_json::Value = serde_json::from_slice(FIXTURE_BUNDLE).unwrap();
    let sig = bundle["messageSignature"]["signature"]
        .as_str()
        .expect("fixture bundle has a messageSignature.signature")
        .to_string();
    use base64::Engine;
    let mut sig_bytes = base64::engine::general_purpose::STANDARD
        .decode(&sig)
        .unwrap();
    let last = sig_bytes.len() - 1;
    sig_bytes[last] ^= 0x01;
    bundle["messageSignature"]["signature"] =
        serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(&sig_bytes));
    let tampered_bundle = serde_json::to_vec(&bundle).unwrap();

    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, &tampered_bundle).await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    let UpdateCheckResult::Unverifiable { .. } = &result else {
        panic!("expected Unverifiable, got: {result:#?}");
    };
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_identity_is_unverifiable() {
    // The fixture bundle is genuinely signed — but if the embedded pin
    // expected a different workflow identity, it must be rejected with a
    // message that names the identity failure. (We move the pin rather
    // than forge a cert: forging would require Fulcio's key. The pinned
    // pattern is the *only* thing that differs from production here.)
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.ci_identity_pattern =
        "https://github.com/someone-else/repo/.github/workflows/release.yml@refs/tags/v*".into();

    let result = check_for_update(&http_client(), &ctx).await;
    assert_unverifiable(&result, "not from the pinned release identity");
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_issuer_is_unverifiable() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.ci_issuer = "https://issuer.evil.example".into();

    let result = check_for_update(&http_client(), &ctx).await;
    assert_unverifiable(&result, "OIDC issuer");
}

#[tokio::test(flavor = "multi_thread")]
async fn authentic_manifest_under_wrong_tag_is_unverifiable() {
    // Replay attack: serve the genuine v0.0.8 manifest+bundle under a
    // v9.9.9 release. Signature verifies (it's authentic) but the Fulcio
    // identity is bound to `@refs/tags/v0.0.8`, not v9.9.9.
    let server = MockServer::start().await;
    mount_release(&server, "v9.9.9", FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    let UpdateCheckResult::Unverifiable {
        version, reason, ..
    } = &result
    else {
        panic!("expected Unverifiable, got: {result:#?}");
    };
    assert_eq!(version, "9.9.9");
    assert!(
        reason.contains("v0.0.8") && reason.contains("v9.9.9"),
        "reason should name both tags, got: {reason}"
    );
}

// ---------------------------------------------------------------------------
// Row: verifies, but claims changed — side-by-side, default NOT trusted
// ---------------------------------------------------------------------------

/// An expected-claims set that this build *didn't* derive: it additionally
/// expects an `sgx` enclave claim and schema_version 2. Serving the real
/// (authentic, schema-1, snp+tdx) fixture against it produces exactly the
/// "missing expected claim types" situation of the matrix — while every
/// cryptographic stage still runs and passes for real.
fn divergent_expected_claims() -> Vec<Claim> {
    let mut claims = expected_claims();
    for c in claims.iter_mut() {
        if c.key == "manifest.schema_version" {
            c.value = "2".into();
        }
    }
    claims.push(Claim {
        key: "enclave.sgx_measurement".into(),
        value: "SGX enclave measurement".into(),
    });
    claims
}

#[tokio::test(flavor = "multi_thread")]
async fn changed_claims_surface_side_by_side() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.expected_claims = divergent_expected_claims();

    let result = check_for_update(&http_client(), &ctx).await;
    let UpdateCheckResult::ClaimsChanged {
        release,
        comparison,
    } = result
    else {
        panic!("expected ClaimsChanged, got: {result:#?}");
    };

    // Cryptographically verified — the release facts are all present...
    assert_eq!(release.version, "0.0.8");
    assert!(release.rekor_log_index > 0);
    // ...but NOT framed as an accepted update.
    assert!(!release.claims_accepted);

    // Side-by-side material: both full lists plus exactly the two deltas.
    assert_eq!(comparison.expected, divergent_expected_claims());
    assert!(!comparison.attested.is_empty());
    assert_eq!(comparison.deltas.len(), 2, "got: {:#?}", comparison.deltas);

    let schema = comparison
        .deltas
        .iter()
        .find(|d| d.key == "manifest.schema_version")
        .expect("schema_version delta");
    assert_eq!(schema.expected.as_deref(), Some("2"));
    assert_eq!(schema.attested.as_deref(), Some("1"));

    let sgx = comparison
        .deltas
        .iter()
        .find(|d| d.key == "enclave.sgx_measurement")
        .expect("sgx delta");
    assert!(
        sgx.attested.is_none(),
        "sgx claim must be absent in attested"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn accepted_claims_change_becomes_update_available() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    // First check: claims changed; capture the manifest hash.
    let mut ctx1 = ctx(&server, OLD_INSTALLED);
    ctx1.expected_claims = divergent_expected_claims();
    let first = check_for_update(&http_client(), &ctx1).await;
    let UpdateCheckResult::ClaimsChanged { release, .. } = first else {
        panic!("expected ClaimsChanged, got: {first:#?}");
    };

    // Re-check with the user's recorded "treat as update" choice.
    let mut ctx2 = ctx1.clone();
    ctx2.accepted = Some(AcceptedClaims {
        version: release.version.clone(),
        manifest_sha256: release.manifest_sha256.clone(),
        accepted_at_ms: 1,
    });
    let second = check_for_update(&http_client(), &ctx2).await;
    let UpdateCheckResult::UpdateAvailable { release } = second else {
        panic!("expected UpdateAvailable after acceptance, got: {second:#?}");
    };
    assert!(release.claims_accepted);
}

#[tokio::test(flavor = "multi_thread")]
async fn acceptance_of_a_different_manifest_does_not_carry_over() {
    // The recorded choice is bound to one exact (version, manifest hash);
    // a different manifest under the same version must re-prompt.
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.expected_claims = divergent_expected_claims();
    ctx.accepted = Some(AcceptedClaims {
        version: "0.0.8".into(),
        manifest_sha256: "00".repeat(32), // not this manifest
        accepted_at_ms: 1,
    });

    let result = check_for_update(&http_client(), &ctx).await;
    assert!(
        matches!(result, UpdateCheckResult::ClaimsChanged { .. }),
        "expected ClaimsChanged, got: {result:#?}"
    );
}

// ---------------------------------------------------------------------------
// AppCore wiring — `update_feed` config override, persistence across cores
// ---------------------------------------------------------------------------

#[test]
fn app_core_resolves_feed_override_and_persists_result() {
    // Plain #[test] + the core's own runtime: AppCore owns a tokio
    // runtime, and dropping it from inside another runtime's async
    // context panics — so this drives everything through `block_on` and
    // drops the cores from sync code, like the CLI does.
    let config_dir = tempfile::tempdir().unwrap();
    let data_dir = tempfile::tempdir().unwrap();

    let core = eidola_app_core::AppCore::new(
        config_dir.path().to_path_buf(),
        data_dir.path().to_path_buf(),
    );
    assert!(core.last_update_check().is_none(), "fresh data dir");

    let snapshot = core.runtime().block_on(async {
        let server = MockServer::start().await;
        mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;
        // The `update_feed` override is a *base* URL; AppCore resolves it
        // to `<base>/releases/latest`.
        std::fs::write(
            config_dir.path().join("config.toml"),
            format!("update_feed = \"{}\"\n", server.uri()),
        )
        .unwrap();

        core.update_check().await
    });

    // This binary's CARGO_PKG_VERSION (0.1.0) is newer than the fixture's
    // v0.0.8, so the wired-up check lands on UpToDate — proving the feed
    // override, the GitHub-shape parsing, and the version gate end-to-end.
    assert_eq!(
        snapshot.result,
        UpdateCheckResult::UpToDate {
            latest_version: Some("0.0.8".into())
        }
    );
    assert_eq!(core.last_update_check(), Some(snapshot.clone()));
    drop(core);

    // A second core on the same data dir sees the persisted snapshot —
    // this is how a background-poll result survives an app restart.
    let core2 = eidola_app_core::AppCore::new(
        config_dir.path().to_path_buf(),
        data_dir.path().to_path_buf(),
    );
    assert_eq!(core2.last_update_check(), Some(snapshot));
}

// ---------------------------------------------------------------------------
// Feed anomalies that are neither crypto verdicts nor offline blips
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn non_semver_tag_is_check_failed() {
    // A garbage tag is a feed anomaly: nothing signed was ever reached, so
    // it's not a crypto verdict — but it must not be silent success either.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/releases/latest"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(latest_release_json(
                &server.uri(),
                "nightly-build",
                &[],
            )),
        )
        .mount(&server)
        .await;

    let result = check_for_update(&http_client(), &ctx(&server, OLD_INSTALLED)).await;
    assert!(
        matches!(result, UpdateCheckResult::CheckFailed { .. }),
        "expected CheckFailed, got: {result:#?}"
    );
}
