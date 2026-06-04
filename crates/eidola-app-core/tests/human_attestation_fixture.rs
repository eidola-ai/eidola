//! Real-bytes fixture test for the human-attestation verifier.
//!
//! The fixture under `tests/fixtures/human_attestation/` was produced
//! with a *throwaway* `cosign generate-key-pair` keypair (no
//! passphrase, immediately discarded), then `cosign sign-blob
//! --bundle …` against a synthetic `attestation.json`. The cosign-
//! produced Sigstore Bundle v0.3 is what `release-tool attest`
//! uploads to the GitHub release.
//!
//! The throwaway key's fingerprint is intentionally *not* in
//! `TRUSTED_ATTESTANT_FINGERPRINTS`, so the verifier rejects it at
//! the identity-pinning step. That rejection is what this test
//! asserts — it proves every step *before* the rejection succeeded
//! (bundle parse, body parse, hash binding, sig binding, pubkey
//! extraction, fingerprint computation), without requiring us to
//! ship a real attestant key into the test trust root.
//!
//! When a real engineer-signed attestation lands on a tagged
//! release, capture its files into `tests/fixtures/v0.0.X/` and add a
//! parallel test that asserts full `Ok`, mirroring
//! `ci_sigstore_fixture.rs`.

use eidola_app_core::updater::{self, ReleaseIndex, trust};

const ATTESTATION: &[u8] = include_bytes!("fixtures/human_attestation/attestation.json");
const BUNDLE: &[u8] = include_bytes!("fixtures/human_attestation/bundle.json");

/// Fingerprint (sha256 of PKIX SubjectPublicKeyInfo DER) of the
/// throwaway key the fixture was signed with. Regenerate this
/// constant when refreshing the fixture.
const EXPECTED_FINGERPRINT_HEX: &str =
    "579b5ebe6cab42779fa288c2a1dd8afdd7a0082c28ba45b91aaea5be7a075183";

fn synthetic_release() -> ReleaseIndex {
    // Mirrors the synthetic attestation prose under
    // `tests/fixtures/human_attestation/attestation.json` — the verifier
    // cross-checks every field, so these must match.
    serde_json::from_str(
        r#"{
            "schema_version": 1,
            "version": "0.99.99",
            "git_commit": "9c3a000000000000000000000000000000000001",
            "git_tag": "v0.99.99",
            "released_at": "2026-05-31T00:00:00Z",
            "previous_release": {
                "version": "0.99.98",
                "git_commit": "9c3a000000000000000000000000000000000000"
            },
            "artifact_manifest": {
                "url": "https://example.com/artifact-manifest.json",
                "sigstore_bundle_url": "https://example.com/artifact-manifest.json.sigstore"
            },
            "human_attestations": [{
                "attestant_id": "throwaway",
                "url": "https://example.com/attestation-throwaway.json",
                "bundle_url": "https://example.com/attestation-throwaway.bundle.json"
            }]
        }"#,
    )
    .unwrap()
}

#[test]
fn cosign_bundle_parses_and_pubkey_derived_but_fingerprint_rejected() {
    let release = synthetic_release();
    let trust = trust::load().expect("loading embedded sigstore trust root");

    let err =
        updater::human_attestation::verify_human_attestation(ATTESTATION, BUNDLE, &release, &trust)
            .expect_err(
                "throwaway key is not in TRUSTED_ATTESTANT_FINGERPRINTS — verify must reject",
            );

    let msg = format!("{err}");
    // Identity rejection is the expected failure mode. If we see any
    // earlier-stage error (bundle parse, body parse, hash binding, sig
    // binding, pubkey extraction), the verifier broke for cosign output.
    assert!(
        msg.contains("TRUSTED_ATTESTANT_FINGERPRINTS"),
        "expected fingerprint-rejection error, got: {msg}"
    );
    // And it must surface *the right fingerprint* — proving the SPKI
    // extraction + sha256 fingerprint computation walked all the way
    // to the correct key.
    assert!(
        msg.contains(EXPECTED_FINGERPRINT_HEX),
        "expected error to surface fingerprint {EXPECTED_FINGERPRINT_HEX}, got: {msg}"
    );
}
