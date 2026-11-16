//! Failure-mode matrix tests for the update-notification flow
//! (`eidola_app_core::updates`), driven against wiremock fixture releases.
//! One test (at least) per row of the matrix in the module docs:
//!
//! - ok (verified update available: CI signature *and* human attestation)
//! - up to date (404 = nothing marked latest; latest not newer)
//! - CI side fails (tampered manifest, tampered signature, malformed
//!   bundle, wrong identity or issuer, manifest replayed under another tag)
//! - human side fails (none at all, an untrusted key, a tampered document,
//!   one sworn to another commit, the same attestation counted twice,
//!   fewer than the threshold)
//! - the index does not hold together (missing files, a tag that is not
//!   its version, an unsupported schema on a newer release)
//! - claims-changed (side-by-side comparison + acceptance flow)
//! - network failure (unresolvable host, HTTP 500, unreadable index)
//!
//! The cryptographic stages run **for real** in every test. The release is
//! the captured `v0.0.8`: its genuine CI-cosign-signed manifest and bundle,
//! and a human attestation over that same manifest signed by a throwaway
//! key and logged in the public Rekor (`tests/fixtures/v0.0.8/`). Both
//! verify against the *embedded* Sigstore trust root, so no trust anchor
//! is stubbed. The one pin a test context moves by default is the set of
//! trusted attestant keys, to the throwaway key's fingerprint. Rows that
//! need a different pin (wrong identity, changed expected claims, a higher
//! threshold) move that pin in `CheckContext`, never the crypto.

use eidola_app_core::updates::{
    AcceptedClaims, CheckContext, Claim, ReleaseAttestant, UpdateCheckResult, UpdateCheckSnapshot,
    UpdateState, attested_claims, check_for_update, check_outcome, classify, compare_claims,
    expected_claims,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const FIXTURE_MANIFEST: &[u8] = include_bytes!("fixtures/v0.0.8/artifact-manifest.json");
const FIXTURE_BUNDLE: &[u8] = include_bytes!("fixtures/v0.0.8/artifact-manifest.json.sigstore");
/// A schema-2 attestation of the v0.0.8 release, naming the sha256 of
/// `FIXTURE_MANIFEST`, signed by a throwaway key (`attestation-throwaway.pub`)
/// and logged in the public Rekor.
const FIXTURE_ATTESTATION: &[u8] = include_bytes!("fixtures/v0.0.8/attestation-throwaway.json");
const FIXTURE_ATTESTATION_BUNDLE: &[u8] =
    include_bytes!("fixtures/v0.0.8/attestation-throwaway.bundle.json");
/// `sha256(SPKI DER)` of the throwaway key.
const THROWAWAY_FINGERPRINT: &str =
    "9f2103792a8007816eb963d1b284d139ac3cf69f3d6ab7cb93878bb654d70915";

/// The tag the fixture bundle's Fulcio identity is bound to
/// (`…tinfoil-build.yml@refs/tags/v0.0.8`).
const FIXTURE_TAG: &str = "v0.0.8";
/// The commit v0.0.8 was released from, and the one before it, as the
/// release and its attestation name them.
const FIXTURE_COMMIT: &str = "bb7773fafbaa90d40e7c7634ff7f690b54dd9be9";
const FIXTURE_PREVIOUS_COMMIT: &str = "7ba323817820a04948ffc70e78955201c96c733f";
/// An installed version older than the fixture release.
const OLD_INSTALLED: &str = "0.0.1";

/// Where a fixture release's index is served, under the releases location.
const LATEST: &str = "/latest/download/release.json";

fn http_client() -> reqwest::Client {
    // reqwest's `rustls-no-provider` build refuses to construct a client
    // until a crypto provider is installed, even for plain-HTTP fixture
    // servers. Idempotent — mirrors what `AppCore::new` does.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    reqwest::Client::builder()
        .build()
        .expect("plain http client")
}

/// The bytes a fixture release serves. `None` leaves the file unserved,
/// which the mock answers with a 404.
#[derive(Clone)]
struct Files {
    manifest: Option<Vec<u8>>,
    bundle: Option<Vec<u8>>,
    attestation: Option<Vec<u8>>,
    attestation_bundle: Option<Vec<u8>>,
}

impl Files {
    fn genuine() -> Self {
        Self {
            manifest: Some(FIXTURE_MANIFEST.to_vec()),
            bundle: Some(FIXTURE_BUNDLE.to_vec()),
            attestation: Some(FIXTURE_ATTESTATION.to_vec()),
            attestation_bundle: Some(FIXTURE_ATTESTATION_BUNDLE.to_vec()),
        }
    }
}

/// A schema-1 `release.json` for `tag`, the shape `release-tool` writes,
/// with every URL pointing back at the mock server under the tag.
fn release_json(server_uri: &str, tag: &str) -> serde_json::Value {
    let version = tag.strip_prefix('v').unwrap_or(tag);
    let base = format!("{server_uri}/download/{tag}");
    serde_json::json!({
        "schema_version": 1,
        "version": version,
        "git_commit": FIXTURE_COMMIT,
        "git_tag": tag,
        "released_at": "2026-05-31T23:53:54Z",
        "previous_release": { "version": "0.0.6", "git_commit": FIXTURE_PREVIOUS_COMMIT },
        "artifact_manifest": {
            "url": format!("{base}/artifact-manifest.json"),
            "sigstore_bundle_url": format!("{base}/artifact-manifest.json.sigstore"),
        },
        "human_attestations": [{
            "attestant_id": "throwaway",
            "url": format!("{base}/attestation-throwaway.json"),
            "bundle_url": format!("{base}/attestation-throwaway.bundle.json"),
        }],
    })
}

/// Serve `index` as the latest release's `release.json`, and `files` under
/// `tag`. `delay` holds the index back, for the ordering tests.
async fn mount_release_with(
    server: &MockServer,
    tag: &str,
    index: serde_json::Value,
    files: Files,
    delay: Option<std::time::Duration>,
) {
    let mut response = ResponseTemplate::new(200).set_body_json(index);
    if let Some(delay) = delay {
        response = response.set_delay(delay);
    }
    Mock::given(method("GET"))
        .and(path(LATEST))
        .respond_with(response)
        .mount(server)
        .await;
    for (name, bytes) in [
        ("artifact-manifest.json", files.manifest),
        ("artifact-manifest.json.sigstore", files.bundle),
        ("attestation-throwaway.json", files.attestation),
        (
            "attestation-throwaway.bundle.json",
            files.attestation_bundle,
        ),
    ] {
        if let Some(bytes) = bytes {
            Mock::given(method("GET"))
                .and(path(format!("/download/{tag}/{name}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
                .mount(server)
                .await;
        }
    }
}

/// Mount a complete release under `tag` with the given CI files and the
/// genuine attestation.
async fn mount_release(server: &MockServer, tag: &str, manifest: &[u8], bundle: &[u8]) {
    let files = Files {
        manifest: Some(manifest.to_vec()),
        bundle: Some(bundle.to_vec()),
        ..Files::genuine()
    };
    mount_release_with(server, tag, release_json(&server.uri(), tag), files, None).await;
}

/// Serve a fixed status for the latest release's index.
async fn mount_index_status(server: &MockServer, status: u16) {
    Mock::given(method("GET"))
        .and(path(LATEST))
        .respond_with(ResponseTemplate::new(status))
        .mount(server)
        .await;
}

/// Expected claims as of the captured v0.0.8 fixture release, which predates
/// the Linux GUI artifact (`eidola-gui-linux-amd64`, added with the Nix Linux
/// release pipeline) and still carries `artifact-manifest.json` schema 1
/// (no `archiveSha256`). The matrix tests exercise the check *mechanics*
/// against the real captured release, so the context pins the fixture-era
/// expected set — otherwise every row would collapse into `ClaimsChanged`.
/// (A real pre-linux-gui / schema-1 client checking a post-linux-gui /
/// schema-2 release seeing `ClaimsChanged` is by design: a new artifact
/// class or manifest shape is a threat-model change the user reviews once.)
///
/// The fixture's enclave block is likewise of the kernel-cmdline launch
/// model, which this build does not pin: it reads that block's three
/// members as unrecognized fields and expects the IGVM-model members in
/// their place. The fixture-era set states that reading of the block, so
/// the matrix rows stay about their own deltas
/// (`a_kernel_cmdline_enclave_block_is_a_changed_threat_model` in
/// `updates.rs` is the test of the enclave change itself).
fn fixture_expected_claims() -> Vec<Claim> {
    let mut claims: Vec<Claim> = expected_claims()
        .into_iter()
        .filter(|c| !c.key.starts_with("artifacts.eidola-gui-linux"))
        .filter(|c| !c.key.starts_with("enclave."))
        .map(|mut c| {
            if c.key == "manifest.schema_version" {
                c.value = "1".into();
            }
            c
        })
        .collect();
    for key in [
        "enclave.cmdline",
        "enclave.snp_measurement",
        "enclave.tdx_measurement",
    ] {
        claims.push(Claim {
            key: key.into(),
            value: "unrecognized field".into(),
        });
    }
    claims
}

fn ctx(server: &MockServer, installed: &str) -> CheckContext {
    let mut ctx = CheckContext::new(server.uri(), installed);
    ctx.expected_claims = fixture_expected_claims();
    ctx.pins.trusted_attestant_fingerprints = vec![THROWAWAY_FINGERPRINT.to_string()];
    ctx
}

async fn check(server: &MockServer) -> UpdateCheckResult {
    check_for_update(&http_client(), &ctx(server, OLD_INSTALLED), None).await
}

// ---------------------------------------------------------------------------
// Row: ok — a real, verified update
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn verified_update_available() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check(&server).await;

    let UpdateCheckResult::UpdateAvailable { release } = result else {
        panic!("expected UpdateAvailable, got: {result:#?}");
    };
    assert_eq!(release.version, "0.0.8");
    assert_eq!(release.tag, FIXTURE_TAG);
    assert_eq!(release.git_commit, FIXTURE_COMMIT);
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
        Some(format!("{}/tag/v0.0.8", server.uri()).as_str())
    );
    let [attestant] = release.attestants.as_slice() else {
        panic!("one attestation was served: {:#?}", release.attestants);
    };
    assert_eq!(
        attestant,
        &ReleaseAttestant {
            name: "Throwaway Test Key".into(),
            jurisdiction: "the Test Jurisdiction".into(),
            rekor_log_index: attestant.rekor_log_index,
        }
    );
    assert!(attestant.rekor_log_index > 0);
}

// ---------------------------------------------------------------------------
// Rows: up to date
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn up_to_date_when_latest_equals_installed() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check_for_update(&http_client(), &ctx(&server, "0.0.8"), None).await;
    assert_eq!(
        result,
        UpdateCheckResult::UpToDate {
            latest_version: Some("0.0.8".into())
        }
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_release_that_is_not_newer_is_not_judged() {
    // Nothing behind the index is served, and the index declares a schema
    // this build does not read. A release that is not an update is not
    // checked, so neither fact can turn it into a warning.
    let server = MockServer::start().await;
    let mut index = release_json(&server.uri(), FIXTURE_TAG);
    index["schema_version"] = 99.into();
    let nothing = Files {
        manifest: None,
        bundle: None,
        attestation: None,
        attestation_bundle: None,
    };
    mount_release_with(&server, FIXTURE_TAG, index, nothing, None).await;

    let result = check_for_update(&http_client(), &ctx(&server, "0.0.8"), None).await;
    assert_eq!(
        result,
        UpdateCheckResult::UpToDate {
            latest_version: Some("0.0.8".into())
        }
    );
    let fetched: Vec<String> = server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(|r| r.url.path().to_string())
        .collect();
    assert_eq!(fetched, [LATEST], "only the index is read");
}

#[tokio::test(flavor = "multi_thread")]
async fn no_release_marked_latest_is_up_to_date() {
    // GitHub's `latest/download/<file>` answers 404 when nothing carries the
    // marker, or the release that does has no such file. Not an error, not
    // an update.
    let server = MockServer::start().await;
    mount_index_status(&server, 404).await;

    assert_eq!(
        check(&server).await,
        UpdateCheckResult::UpToDate {
            latest_version: None
        }
    );
}

// ---------------------------------------------------------------------------
// Row: network failure — quiet, never alarms
// ---------------------------------------------------------------------------

fn assert_check_failed(result: &UpdateCheckResult) {
    assert!(
        matches!(result, UpdateCheckResult::CheckFailed { .. }),
        "expected CheckFailed, got: {result:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn unreachable_feed_is_check_failed() {
    // `.invalid` is reserved (RFC 2606) and never resolves — a fast,
    // deterministic transport failure. (A dropped mock server's port can
    // be re-bound by a parallel test, so "connection refused" via port
    // reuse is racy.)
    let ctx = CheckContext::new("http://feed.invalid", OLD_INSTALLED);
    assert_check_failed(&check_for_update(&http_client(), &ctx, None).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn server_error_is_check_failed() {
    let server = MockServer::start().await;
    mount_index_status(&server, 500).await;
    assert_check_failed(&check(&server).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rate_limit_is_check_failed() {
    // A 403 or 429 in front of the index is the host declining to answer,
    // which says nothing about any release.
    for status in [403, 429] {
        let server = MockServer::start().await;
        mount_index_status(&server, status).await;
        assert_check_failed(&check(&server).await);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unreadable_index_is_check_failed() {
    // An index that never says what version it is never claimed an update.
    for body in [
        serde_json::json!("<!doctype html>nope"),
        serde_json::json!({ "schema_version": 1 }),
        serde_json::json!({ "version": "nightly-build" }),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(LATEST))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        assert_check_failed(&check(&server).await);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_error_on_any_file_is_check_failed() {
    // A newer release was claimed, but a file it names could not be fetched:
    // server-side trouble, not evidence of tampering. Quiet retry.
    for broken in [
        "artifact-manifest.json",
        "artifact-manifest.json.sigstore",
        "attestation-throwaway.json",
        "attestation-throwaway.bundle.json",
    ] {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("/download/{FIXTURE_TAG}/{broken}")))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;
        let result = check(&server).await;
        assert!(
            matches!(result, UpdateCheckResult::CheckFailed { .. }),
            "a 500 on {broken} must be quiet, got: {result:#?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Rows: a newer release that does not verify — hard, visible security state
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

async fn check_files(files: Files) -> UpdateCheckResult {
    let server = MockServer::start().await;
    let index = release_json(&server.uri(), FIXTURE_TAG);
    mount_release_with(&server, FIXTURE_TAG, index, files, None).await;
    check(&server).await
}

#[tokio::test(flavor = "multi_thread")]
async fn a_missing_file_is_unverifiable() {
    // The release names a file that is not there — a release that does not
    // hold together, not a transient failure.
    for (name, files) in [
        (
            "artifact-manifest.json",
            Files {
                manifest: None,
                ..Files::genuine()
            },
        ),
        (
            "artifact-manifest.json.sigstore",
            Files {
                bundle: None,
                ..Files::genuine()
            },
        ),
        (
            "attestation-throwaway.json",
            Files {
                attestation: None,
                ..Files::genuine()
            },
        ),
        (
            "attestation-throwaway.bundle.json",
            Files {
                attestation_bundle: None,
                ..Files::genuine()
            },
        ),
    ] {
        let result = check_files(files).await;
        assert_unverifiable(&result, "HTTP 404");
        let UpdateCheckResult::Unverifiable { reason, .. } = &result else {
            unreachable!()
        };
        assert!(reason.contains(name), "names {name}: {reason}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_bundle_is_unverifiable() {
    let result = check_files(Files {
        bundle: Some(b"not a sigstore bundle at all".to_vec()),
        ..Files::genuine()
    })
    .await;
    assert_unverifiable(&result, "parsing CI Sigstore bundle");
}

#[tokio::test(flavor = "multi_thread")]
async fn tampered_manifest_is_unverifiable() {
    // One byte of drift in the manifest → its hash no longer matches what
    // the (genuine) bundle signed.
    let mut tampered = FIXTURE_MANIFEST.to_vec();
    tampered.extend_from_slice(b" ");
    let result = check_files(Files {
        manifest: Some(tampered),
        ..Files::genuine()
    })
    .await;
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

    let result = check_files(Files {
        bundle: Some(serde_json::to_vec(&bundle).unwrap()),
        ..Files::genuine()
    })
    .await;
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
    ctx.pins.ci_identity_pattern =
        "https://github.com/someone-else/repo/.github/workflows/release.yml@refs/tags/v*".into();

    let result = check_for_update(&http_client(), &ctx, None).await;
    assert_unverifiable(&result, "not from the pinned release identity");
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_issuer_is_unverifiable() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.pins.ci_issuer = "https://issuer.evil.example".into();

    let result = check_for_update(&http_client(), &ctx, None).await;
    assert_unverifiable(&result, "OIDC issuer");
}

#[tokio::test(flavor = "multi_thread")]
async fn authentic_manifest_under_wrong_tag_is_unverifiable() {
    // Replay attack: serve the genuine v0.0.8 manifest+bundle under a
    // v9.9.9 release. Signature verifies (it's authentic) but the Fulcio
    // identity is bound to `@refs/tags/v0.0.8`, not v9.9.9.
    let server = MockServer::start().await;
    mount_release(&server, "v9.9.9", FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let result = check(&server).await;
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
// Rows: CI's signature verifies, and the human side does not
// ---------------------------------------------------------------------------
//
// Each of these releases carries a manifest CI genuinely signed, under the
// pinned identity and bound to the right tag. That is everything a
// compromised release pipeline can produce, and none of it is enough.

#[tokio::test(flavor = "multi_thread")]
async fn a_release_ci_signed_and_nobody_attested_is_unverifiable() {
    let server = MockServer::start().await;
    let mut index = release_json(&server.uri(), FIXTURE_TAG);
    index["human_attestations"] = serde_json::json!([]);
    mount_release_with(&server, FIXTURE_TAG, index, Files::genuine(), None).await;

    assert_unverifiable(&check(&server).await, "requires ≥1");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_attestation_by_a_key_this_build_does_not_trust_is_unverifiable() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    // The embedded set, which does not hold the throwaway key.
    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.pins = eidola_app_core::updater::ReleasePins::embedded();

    let result = check_for_update(&http_client(), &ctx, None).await;
    assert_unverifiable(&result, THROWAWAY_FINGERPRINT);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tampered_attestation_is_unverifiable() {
    // One claim's wording changed after signing: the signature over the
    // document no longer holds.
    let text = String::from_utf8(FIXTURE_ATTESTATION.to_vec()).unwrap();
    let tampered = text.replacen("not aware of any backdoor", "aware of no backdoor", 1);
    assert_ne!(tampered, text, "the fixture has to carry the phrase");
    let result = check_files(Files {
        attestation: Some(tampered.into_bytes()),
        ..Files::genuine()
    })
    .await;
    let UpdateCheckResult::Unverifiable { .. } = &result else {
        panic!("expected Unverifiable, got: {result:#?}");
    };
}

#[tokio::test(flavor = "multi_thread")]
async fn an_attestation_of_another_commit_is_unverifiable() {
    // The index says the release was built from a different commit than the
    // one the attestant swore to. The attestation is genuine and the manifest
    // genuine; what the attestant vouched for is not what this index claims.
    let server = MockServer::start().await;
    let mut index = release_json(&server.uri(), FIXTURE_TAG);
    index["git_commit"] = "0".repeat(40).into();
    mount_release_with(&server, FIXTURE_TAG, index, Files::genuine(), None).await;

    assert_unverifiable(&check(&server).await, "≠ release.git_commit");
}

#[tokio::test(flavor = "multi_thread")]
async fn one_attestation_listed_twice_counts_once() {
    let server = MockServer::start().await;
    let mut index = release_json(&server.uri(), FIXTURE_TAG);
    let entry = index["human_attestations"][0].clone();
    index["human_attestations"] = serde_json::json!([entry.clone(), entry]);
    mount_release_with(&server, FIXTURE_TAG, index, Files::genuine(), None).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.pins.min_human_attestations = 2;
    let result = check_for_update(&http_client(), &ctx, None).await;
    assert_unverifiable(&result, "must be independent");
}

#[tokio::test(flavor = "multi_thread")]
async fn fewer_attestations_than_the_threshold_is_unverifiable() {
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.pins.min_human_attestations = 2;
    let result = check_for_update(&http_client(), &ctx, None).await;
    assert_unverifiable(&result, "requires ≥2");
}

// ---------------------------------------------------------------------------
// Rows: a newer index that does not hold together
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn a_tag_that_is_not_the_version_is_unverifiable() {
    let server = MockServer::start().await;
    let mut index = release_json(&server.uri(), FIXTURE_TAG);
    index["git_tag"] = "v0.0.9".into();
    mount_release_with(&server, FIXTURE_TAG, index, Files::genuine(), None).await;

    assert_unverifiable(&check(&server).await, "a release's tag is `v0.0.8`");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_newer_release_this_build_cannot_read_is_unverifiable() {
    // Unlike the not-newer case above: this one claims to be an update.
    let server = MockServer::start().await;
    let mut index = release_json(&server.uri(), FIXTURE_TAG);
    index["schema_version"] = 99.into();
    mount_release_with(&server, FIXTURE_TAG, index, Files::genuine(), None).await;

    assert_unverifiable(&check(&server).await, "install a newer client");
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
    // Built on the fixture-era baseline (not the full expected_claims()) so
    // the fixture-vs-divergent comparison yields exactly the two synthetic
    // deltas below, not an extra "missing Linux GUI artifact" row.
    let mut claims = fixture_expected_claims();
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

    let result = check_for_update(&http_client(), &ctx, None).await;
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
    let first = check_for_update(&http_client(), &ctx1, None).await;
    let UpdateCheckResult::ClaimsChanged { release, .. } = first else {
        panic!("expected ClaimsChanged, got: {first:#?}");
    };

    // Re-check with the user's recorded "treat as update" choice.
    let accepted = AcceptedClaims {
        version: release.version.clone(),
        manifest_sha256: release.manifest_sha256.clone(),
        accepted_at_ms: 1,
    };
    let second = check_for_update(&http_client(), &ctx1, Some(&accepted)).await;
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
    let elsewhere = AcceptedClaims {
        version: "0.0.8".into(),
        manifest_sha256: "00".repeat(32), // not this manifest
        accepted_at_ms: 1,
    };

    let result = check_for_update(&http_client(), &ctx, Some(&elsewhere)).await;
    assert!(
        matches!(result, UpdateCheckResult::ClaimsChanged { .. }),
        "expected ClaimsChanged, got: {result:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_acceptance_landing_during_a_check_is_not_undone_by_it() {
    // "Treat as update" is one keystroke, and a check is a network round trip
    // — so the acceptance can land while the request that will report on that
    // very manifest is still in the air. A check that decided acceptance when
    // it *started* answers about a moment the user has moved on from: it
    // reports `ClaimsChanged` for a manifest now on record as accepted, and
    // `absorb` installs it over the `UpdateAvailable` the acceptance just
    // wrote. The warning and the button come straight back, with the
    // acceptance still sitting in the state beside them.
    //
    // So the round trip ends in a `CheckOutcome` that has decided nothing,
    // and one and the same completed check reads either way depending on what
    // is on record when it lands.
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;
    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.expected_claims = divergent_expected_claims();

    let outcome = check_outcome(&http_client(), &ctx).await;

    // What the in-flight check would have concluded, had it been asked before
    // the user acted.
    let before = classify(outcome.clone(), None);
    let UpdateCheckResult::ClaimsChanged { release, .. } = &before else {
        panic!("expected ClaimsChanged with nothing accepted, got: {before:#?}");
    };

    // The acceptance lands mid-flight, and rewrites the standing snapshot.
    let mut state = UpdateState::default();
    state.absorb(UpdateCheckSnapshot {
        checked_at_ms: 1,
        result: before.clone(),
    });
    let accepted = AcceptedClaims {
        version: release.version.clone(),
        manifest_sha256: release.manifest_sha256.clone(),
        accepted_at_ms: 2,
    };
    state.accepted = Some(accepted.clone());

    // Now the check lands, and is classified against the state as it is.
    let after = classify(outcome.clone(), state.accepted.as_ref());
    state.absorb(UpdateCheckSnapshot {
        checked_at_ms: 3,
        result: after,
    });

    let Some(UpdateCheckSnapshot {
        result: UpdateCheckResult::UpdateAvailable { release },
        ..
    }) = &state.last
    else {
        panic!(
            "the completing check must not resurrect a warning the user has \
             already answered, got: {:#?}",
            state.last
        );
    };
    assert!(release.claims_accepted, "and it says why it is an update");
    assert_eq!(
        state.accepted.as_ref().map(|a| &a.manifest_sha256),
        Some(&accepted.manifest_sha256),
        "the acceptance itself is untouched"
    );

    // The other direction, from the same completed check: an acceptance that
    // names a different manifest tolerates nothing here.
    let elsewhere = AcceptedClaims {
        manifest_sha256: "00".repeat(32),
        ..accepted
    };
    assert!(
        matches!(
            classify(outcome, Some(&elsewhere)),
            UpdateCheckResult::ClaimsChanged { .. }
        ),
        "acceptance is bound to one exact manifest"
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
    )
    .expect("open core");
    assert!(core.last_update_check().is_none(), "fresh data dir");

    let snapshot = core.runtime().block_on(async {
        let server = MockServer::start().await;
        mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;
        // The `update_feed` override is a releases location; AppCore reads
        // `<it>/latest/download/release.json`.
        std::fs::write(
            config_dir.path().join("config.toml"),
            format!("update_feed = \"{}\"\n", server.uri()),
        )
        .unwrap();

        core.update_check().await
    });

    // This binary's CARGO_PKG_VERSION is newer than the fixture's v0.0.8,
    // so the wired-up check lands on UpToDate — proving the feed override,
    // the index read, and the version gate end-to-end.
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
    )
    .expect("reopen core after the first released its database lock");
    assert_eq!(core2.last_update_check(), Some(snapshot));
}

#[test]
fn a_standing_alert_survives_a_check_that_started_before_it() {
    // Two checks overlap — an app's poll and a command-line client's, or a
    // poll and a manual one. The slower started first, so anything it copied
    // up front is a picture of the state *before* the other check found a
    // security alert; writing that picture back is how the alert disappears.
    // Only one thing may cross a network wait, and it is not the state.
    let config_dir = tempfile::tempdir().unwrap();
    let data_dir = tempfile::tempdir().unwrap();
    let core = eidola_app_core::AppCore::new(
        config_dir.path().to_path_buf(),
        data_dir.path().to_path_buf(),
    )
    .expect("open core");

    let feed = |uri: &str| {
        std::fs::write(
            config_dir.path().join("config.toml"),
            format!("update_feed = \"{uri}\"\n"),
        )
        .unwrap();
    };

    core.runtime().block_on(async {
        // The check that finishes last: a feed that fails, slowly.
        let slow = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(LATEST))
            .respond_with(ResponseTemplate::new(500).set_delay(std::time::Duration::from_secs(3)))
            .mount(&slow)
            .await;
        // The check that finishes first: the genuine bundle replayed under a
        // release newer than this build, whose Fulcio identity names another
        // tag — a real `Unverifiable`, which is exactly the state that must
        // outlive a later failure.
        let alerting = MockServer::start().await;
        mount_release(&alerting, "v9.9.9", FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

        feed(&slow.uri());
        let slow_check = core.update_check();
        tokio::pin!(slow_check);
        // Let it read the feed and get its request out, then leave it waiting.
        tokio::select! {
            _ = &mut slow_check => panic!("the slow feed answered too early to prove anything"),
            _ = request_arrived(&slow) => {}
        }

        feed(&alerting.uri());
        let alert = core.update_check().await;
        assert!(
            matches!(alert.result, UpdateCheckResult::Unverifiable { .. }),
            "the fixture has to raise a real alert: {:#?}",
            alert.result
        );

        let failed = slow_check.await;
        assert!(
            matches!(failed.result, UpdateCheckResult::Unverifiable { .. }),
            "the later-finishing check must fold into the state as it is now, \
             where the alert stands — not into the copy it started from: {:#?}",
            failed.result
        );
    });

    assert!(
        matches!(
            core.last_update_check().map(|s| s.result),
            Some(UpdateCheckResult::Unverifiable { .. })
        ),
        "and the alert is what persists"
    );
}

#[test]
fn a_newer_alert_is_not_replaced_by_an_older_check_that_finished_after_it() {
    // The sibling of the standing-alert case, and the one `absorb` cannot
    // reach: this older check does not *fail*, it succeeds with a perfectly
    // valid `UpToDate` — a cached feed answer, or simply the state of the
    // world before the release that raised the alarm was marked latest.
    // `absorb` holds an alert against a later failure; nothing about an
    // `UpToDate` says it is stale. Only start order does.
    let config_dir = tempfile::tempdir().unwrap();
    let data_dir = tempfile::tempdir().unwrap();
    let core = eidola_app_core::AppCore::new(
        config_dir.path().to_path_buf(),
        data_dir.path().to_path_buf(),
    )
    .expect("open core");

    let feed = |uri: &str| {
        std::fs::write(
            config_dir.path().join("config.toml"),
            format!("update_feed = \"{uri}\"\n"),
        )
        .unwrap();
    };

    core.runtime().block_on(async {
        // Started first, finishes last, and answers honestly: this build is
        // newer than the release the feed still calls latest.
        let slow = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(LATEST))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(release_json(&slow.uri(), FIXTURE_TAG))
                    .set_delay(std::time::Duration::from_secs(3)),
            )
            .mount(&slow)
            .await;
        // Started second, finishes first, and finds a genuine alert.
        let alerting = MockServer::start().await;
        mount_release(&alerting, "v9.9.9", FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

        feed(&slow.uri());
        let slow_check = core.update_check();
        tokio::pin!(slow_check);
        tokio::select! {
            _ = &mut slow_check => panic!("the slow feed answered too early to prove anything"),
            _ = request_arrived(&slow) => {}
        }

        feed(&alerting.uri());
        let alert = core.update_check().await;
        assert!(
            matches!(alert.result, UpdateCheckResult::Unverifiable { .. }),
            "the fixture has to raise a real alert: {:#?}",
            alert.result
        );

        let stale = slow_check.await;
        assert!(
            matches!(stale.result, UpdateCheckResult::Unverifiable { .. }),
            "a check that started earlier must not undo a newer one's alert with \
             a picture from before it — and is told what stands: {:#?}",
            stale.result
        );
    });

    assert!(
        matches!(
            core.last_update_check().map(|s| s.result),
            Some(UpdateCheckResult::Unverifiable { .. })
        ),
        "and the alert is what persists"
    );
}

#[test]
fn a_quick_failure_does_not_silence_a_slower_check_that_saw_something() {
    // The other edge of the start-order rule. A check that *failed* reached
    // no feed and drew no conclusion, so it is not a newer observation — it
    // is no observation. Counting it as one lets a network blip that started
    // a moment later throw away a real security alert for having begun first,
    // which is worse than the staleness the ordering exists to prevent.
    let config_dir = tempfile::tempdir().unwrap();
    let data_dir = tempfile::tempdir().unwrap();
    let core = eidola_app_core::AppCore::new(
        config_dir.path().to_path_buf(),
        data_dir.path().to_path_buf(),
    )
    .expect("open core");

    let feed = |uri: &str| {
        std::fs::write(
            config_dir.path().join("config.toml"),
            format!("update_feed = \"{uri}\"\n"),
        )
        .unwrap();
    };

    core.runtime().block_on(async {
        // Started first, finishes last, and actually saw the feed: the
        // genuine bundle replayed under a newer tag.
        let slow = MockServer::start().await;
        let index = release_json(&slow.uri(), "v9.9.9");
        let delay = Some(std::time::Duration::from_secs(3));
        mount_release_with(&slow, "v9.9.9", index, Files::genuine(), delay).await;
        // Started second, finishes first, and reached nothing at all.
        let offline = MockServer::start().await;
        mount_index_status(&offline, 500).await;

        feed(&slow.uri());
        let slow_check = core.update_check();
        tokio::pin!(slow_check);
        tokio::select! {
            _ = &mut slow_check => panic!("the slow feed answered too early to prove anything"),
            _ = request_arrived(&slow) => {}
        }

        feed(&offline.uri());
        let blip = core.update_check().await;
        assert!(
            matches!(blip.result, UpdateCheckResult::CheckFailed { .. }),
            "the fixture has to actually fail to prove anything: {:#?}",
            blip.result
        );

        let alert = slow_check.await;
        assert!(
            matches!(alert.result, UpdateCheckResult::Unverifiable { .. }),
            "a check that reached nothing cannot make a real verdict old: {:#?}",
            alert.result
        );
    });

    assert!(
        matches!(
            core.last_update_check().map(|s| s.result),
            Some(UpdateCheckResult::Unverifiable { .. })
        ),
        "and the alert is what persists"
    );
}

/// Resolves once `server` has received a request: the check that sent it has
/// read the feed from the config and is now waiting on the answer.
async fn request_arrived(server: &MockServer) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while server
        .received_requests()
        .await
        .unwrap_or_default()
        .is_empty()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the check never sent its request"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

// ---------------------------------------------------------------------------
// The command-line client's artifact rows across the schema-5 rotation
// ---------------------------------------------------------------------------

/// The manifest this tree emits today, at whatever schema it declares.
const WORKSPACE_MANIFEST: &[u8] = include_bytes!("../../../artifact-manifest.json");

fn workspace_manifest() -> serde_json::Value {
    serde_json::from_slice(WORKSPACE_MANIFEST).expect("the committed manifest is JSON")
}

/// The static Linux CLI rows schema 5 adds, one per architecture, as
/// `(key, platform)`.
const STATIC_CLI_ROWS: [(&str, &str); 2] = [
    ("eidola-cli-linux-static-amd64", "linux/amd64"),
    ("eidola-cli-linux-static-arm64", "linux/arm64"),
];

fn cli_deltas(manifest: &serde_json::Value) -> Vec<(String, Option<String>)> {
    compare_claims(&expected_claims(), &attested_claims(manifest))
        .into_iter()
        .filter(|d| d.key.starts_with("artifacts.eidola-cli"))
        .map(|d| (d.key, d.attested))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_authentic_release_recording_the_macos_cli_reads_cleanly_on_its_cli_rows() {
    // The real signed release, checked against this build's *own* expected
    // set rather than the fixture-era one: it records the standalone macOS
    // CLI archive, which schema 5 retires, so this is the verified path's
    // proof that accepting schema 5 did not stop this build reading a
    // release that still records the row. Other rows differ (the fixture
    // predates the current enclave and schema); the CLI rows must not.
    let server = MockServer::start().await;
    mount_release(&server, FIXTURE_TAG, FIXTURE_MANIFEST, FIXTURE_BUNDLE).await;

    let mut ctx = ctx(&server, OLD_INSTALLED);
    ctx.expected_claims = expected_claims();
    let result = check_for_update(&http_client(), &ctx, None).await;
    let UpdateCheckResult::ClaimsChanged { comparison, .. } = result else {
        panic!("expected the fixture to verify and differ elsewhere, got: {result:#?}");
    };
    assert!(
        comparison
            .attested
            .iter()
            .any(|c| c.key == "artifacts.eidola-cli-macos-universal"),
        "the fixture has to record the retiring row to prove anything"
    );
    let cli: Vec<_> = comparison
        .deltas
        .iter()
        .filter(|d| d.key.starts_with("artifacts.eidola-cli"))
        .collect();
    assert!(cli.is_empty(), "CLI rows read as changed: {cli:#?}");
}

#[test]
fn this_build_tolerates_both_sides_of_the_cli_rotation() {
    // What a shipped client shows for these rows: each is expected, and
    // each may be absent, because this build reads a manifest from either
    // side of schema 5.
    let expected = expected_claims();
    let value = |key: &str| {
        expected
            .iter()
            .find(|c| c.key == key)
            .map(|c| c.value.as_str())
    };
    assert_eq!(
        value("artifacts.eidola-cli-macos-universal"),
        Some("nix (darwin/universal) or absent")
    );
    assert_eq!(
        value("artifacts.eidola-cli-linux-static-amd64"),
        Some("file (linux/amd64) or absent")
    );
    assert_eq!(
        value("artifacts.eidola-cli-linux-static-arm64"),
        Some("file (linux/arm64) or absent")
    );
    // The image row keeps its meaning: one linux/amd64 image, whatever
    // other architectures the static binary is published for.
    assert_eq!(value("artifacts.eidola-cli"), Some("oci (linux/amd64)"));
}

#[test]
fn the_committed_manifest_is_held_to_its_own_cli_rows() {
    // The manifest releases emit now, through the public surface: clean as
    // committed, and refusing either half of the CLI rotation applied
    // without the schema that records it.
    let manifest = workspace_manifest();
    let declared = manifest["schema_version"]
        .as_u64()
        .expect("an integer schema");
    assert!(cli_deltas(&manifest).is_empty());

    if declared < 5 {
        let mut dropped = manifest.clone();
        dropped["artifacts"]
            .as_object_mut()
            .unwrap()
            .remove("eidola-cli-macos-universal");
        assert_eq!(
            cli_deltas(&dropped),
            [(
                "artifacts.eidola-cli-macos-universal".to_string(),
                Some("missing (recorded through schema 4)".to_string())
            )]
        );

        for (key, platform) in STATIC_CLI_ROWS {
            let mut early = manifest.clone();
            early["artifacts"][key] = serde_json::json!({
                "platform": platform,
                "sha256": format!("sha256:{}", "9".repeat(64)),
                "type": "file"
            });
            assert_eq!(
                cli_deltas(&early),
                [(
                    format!("artifacts.{key}"),
                    Some(format!(
                        "present, but schema {declared} records it only from schema 5 on"
                    ))
                )]
            );
        }
    } else {
        let mut kept = manifest.clone();
        kept["artifacts"]["eidola-cli-macos-universal"] = serde_json::json!({
            "archiveSha256": format!("sha256:{}", "a".repeat(64)),
            "narHash": "sha256-aa",
            "platform": "darwin/universal",
            "type": "nix"
        });
        assert_eq!(
            cli_deltas(&kept),
            [(
                "artifacts.eidola-cli-macos-universal".to_string(),
                Some(format!(
                    "present, but schema {declared} records it only through schema 4"
                ))
            )]
        );

        for (key, _) in STATIC_CLI_ROWS {
            let mut dropped = manifest.clone();
            dropped["artifacts"].as_object_mut().unwrap().remove(key);
            assert_eq!(
                cli_deltas(&dropped),
                [(
                    format!("artifacts.{key}"),
                    Some("missing (recorded from schema 5 on)".to_string())
                )]
            );
        }
    }
}
