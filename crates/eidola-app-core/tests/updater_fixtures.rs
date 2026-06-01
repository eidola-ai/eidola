//! Integration tests for the updater's `FixturesFetcher` mode.
//!
//! The fixtures path exists so the verifier pipeline can be re-run against
//! captured release bytes without re-tagging on GitHub each iteration.
//! These tests exercise the dev-mode plumbing — they prove the fixtures
//! `Fetcher` reads the right file at each pipeline stage, and that the
//! verifier reaches the cryptographic stages with on-disk bytes — but they
//! do **not** attempt to construct a fully-passing crypto run. That would
//! require real Fulcio certs, Rekor entries, cosign-emitted blob
//! signatures, etc., none of which we want to invent here.
//!
//! TODO: capture a real `v0.0.1` release set into
//! `tests/fixtures/v0.0.1/` once the first signed release lands. With real
//! bytes, this file can additionally assert the full pipeline reaches the
//! `present` stage and yields a `ReleaseSummary`.

use std::path::Path;

use eidola_app_core::updater::{self, Fetcher, VerifyOptions};

/// Build a minimal `release.json` fixture that parses cleanly and advances
/// past the discover/schema/continuity stages, then references an
/// `artifact-manifest.json` and `artifact-manifest.json.sigstore` from the
/// same fixtures dir (so the URL→filename mapping is exercised).
fn write_minimal_fixture(dir: &Path) {
    let release_json = r#"{
        "schema_version": 1,
        "version": "9.9.9",
        "git_commit": "9c3a000000000000000000000000000000000001",
        "git_tag": "v9.9.9",
        "released_at": "2026-05-26T17:00:00Z",
        "artifact_manifest": {
            "url": "https://example.com/v9.9.9/artifact-manifest.json",
            "sigstore_bundle_url": "https://example.com/v9.9.9/artifact-manifest.json.sigstore"
        },
        "human_attestations": [{
            "attestant_id": "test-attestant",
            "url": "https://example.com/v9.9.9/attestation-test-attestant.json",
            "bundle_url": "https://example.com/v9.9.9/attestation-test-attestant.json.sigstore"
        }]
    }"#;
    std::fs::write(dir.join("release.json"), release_json).unwrap();

    // Garbage but non-empty — the goal is to make the verifier reach the CI
    // sigstore stage and reject these bytes there, proving the fetch
    // plumbing routed them correctly.
    std::fs::write(dir.join("artifact-manifest.json"), b"not a real manifest").unwrap();
    std::fs::write(
        dir.join("artifact-manifest.json.sigstore"),
        b"not a real sigstore bundle",
    )
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fixtures_fetcher_reaches_verify_ci_with_local_bytes() {
    let dir = tempfile::tempdir().unwrap();
    write_minimal_fixture(dir.path());

    let fetcher = Fetcher::fixtures(dir.path());
    let opts = VerifyOptions { verbose: false };

    // "9.9.9" is newer than any installed version we'd pass; first-install
    // mode (no installed_git_commit) bypasses continuity.
    let result = updater::check_for_update_with(&fetcher, opts, "0.0.1", None).await;

    let err = result.expect_err("expected verifier to fail at verify-ci with garbage bundle");
    let msg = format!("{err}");
    // The CI Sigstore stage should reject "not a real sigstore bundle" with
    // a JSON parse failure — that's the stage we want to have reached.
    assert!(
        msg.contains("Sigstore bundle") || msg.contains("sigstore") || msg.contains("bundle"),
        "expected verify-ci-stage error, got: {msg}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn fixtures_fetcher_no_update_when_version_not_newer() {
    let dir = tempfile::tempdir().unwrap();
    write_minimal_fixture(dir.path());

    let fetcher = Fetcher::fixtures(dir.path());
    let opts = VerifyOptions::default();

    // Same version as the fixture ⇒ no update; verifier returns Ok(None)
    // *before* touching any of the bogus manifest/bundle bytes.
    let summary = updater::check_for_update_with(&fetcher, opts, "9.9.9", None)
        .await
        .expect("same-version path should be Ok(None)");
    assert!(summary.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn fixtures_fetcher_errors_when_release_json_missing() {
    let dir = tempfile::tempdir().unwrap();
    // Deliberately do not write release.json.

    let fetcher = Fetcher::fixtures(dir.path());
    let opts = VerifyOptions::default();

    let err = updater::check_for_update_with(&fetcher, opts, "0.0.1", None)
        .await
        .expect_err("missing release.json should fail at the discover stage");
    let msg = format!("{err}");
    assert!(
        msg.contains("release.json"),
        "expected discover-stage error mentioning release.json, got: {msg}"
    );
}
