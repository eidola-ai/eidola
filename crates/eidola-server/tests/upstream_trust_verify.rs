//! End-to-end verification of a real, captured Tinfoil router release
//! attestation against the compile-time-embedded Sigstore trusted root.
//!
//! Fixture: `fixtures/router-attestation-v0.0.115.json` is the
//! `.attestations[0].bundle` object GitHub served for
//! `tinfoilsh/confidential-model-router@v0.0.115` (subject digest =
//! `tinfoil.hash`). The bundle is self-contained (Fulcio chain, DSSE
//! envelope, Rekor SET + inclusion proof), so verification is fully offline
//! — no network, no time-dependence beyond the cert validity window baked
//! into the fixture.
//!
//! If this test ever fails after refreshing `sigstore-trusted-root.json`,
//! the snapshot no longer covers the fixture's Fulcio CA / Rekor key era —
//! capture a fresh fixture from a current release rather than loosening the
//! verifier.

use eidola_server::upstream_trust::sigstore::{
    TrustError, load_trusted_root, verify_release_attestation,
};

const REPO: &str = "tinfoilsh/confidential-model-router";
const TAG: &str = "v0.0.115";
const DIGEST: &str = "d6494131e21aaf44bc2c6f2cb7148d217ddd21a83616c486485dcb9f4ac23a5d";
const EXPECTED_SNP: &str = "2d334538b2abab3e51c0af976162b522ac6ba3433383c8cb6b8fecc2eec79321cf9305d6fea42861ce64b54e004124d0";
const EXPECTED_RTMR1: &str = "46658ae5655794d3ea0130e2d425aa002f224c7a47c1eb1792f656d79f808aac6006ce84d71ee24d97c3eea42c867e51";
const EXPECTED_RTMR2: &str = "f1ffdea22cb5ed4a5d8bc332eef7bdd63d1a938e2c474067b26667b1b74a0f5764d431bcda4da7dcc4ad46b0940d287e";

const BUNDLE: &[u8] = include_bytes!("fixtures/router-attestation-v0.0.115.json");

fn verify(
    repo: &str,
    tag: &str,
    digest: &str,
) -> Result<eidola_server::upstream_trust::sigstore::VerifiedMeasurement, TrustError> {
    let trust = load_trusted_root().expect("embedded trusted root parses");
    verify_release_attestation(BUNDLE, repo, tag, digest, &trust)
}

#[test]
fn verifies_real_release_and_extracts_measurement() {
    let m = verify(REPO, TAG, DIGEST).expect("real release attestation must verify");
    assert_eq!(m.snp_measurement, EXPECTED_SNP);
    assert_eq!(m.rtmr1, EXPECTED_RTMR1);
    assert_eq!(m.rtmr2, EXPECTED_RTMR2);
    assert_eq!(m.subject_digest_hex, DIGEST);
    // The SAN identity must be a GitHub Actions workflow in the expected repo
    // at the expected tag.
    assert!(
        m.ci_identity
            .starts_with(&format!("https://github.com/{REPO}/.github/workflows/")),
        "unexpected CI identity: {}",
        m.ci_identity
    );
    assert!(m.ci_identity.ends_with(&format!("@refs/tags/{TAG}")));
}

#[test]
fn rejects_wrong_tag() {
    // The tag pin is our tightening over Tinfoil's own clients: an authentic
    // attestation for a *different* tag must not verify under this tag.
    let err = verify(REPO, "v0.0.114", DIGEST).unwrap_err();
    assert!(
        err.to_string().contains("does not match expected identity"),
        "got: {err}"
    );
}

#[test]
fn rejects_wrong_repo() {
    let err = verify("attacker/confidential-model-router", TAG, DIGEST).unwrap_err();
    assert!(
        err.to_string().contains("does not match expected identity"),
        "got: {err}"
    );
}

#[test]
fn rejects_wrong_subject_digest() {
    // A bundle whose signed subject digest doesn't equal the release's
    // tinfoil.hash (e.g. the attacker served a bundle for a different
    // artifact) must be rejected.
    let wrong = "0".repeat(64);
    let err = verify(REPO, TAG, &wrong).unwrap_err();
    assert!(err.to_string().contains("subject digest"), "got: {err}");
}

#[test]
fn rejects_tampered_bundle() {
    // Flip a byte in the base64 payload region; the DSSE signature (or the
    // rekor payloadHash binding) must fail.
    let mut tampered = BUNDLE.to_vec();
    // Find the `"payload":"` value and corrupt a character well inside it.
    let needle = b"\"payload\":\"";
    let pos = tampered
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("bundle has a payload field")
        + needle.len()
        + 8;
    tampered[pos] = if tampered[pos] == b'A' { b'B' } else { b'A' };

    let trust = load_trusted_root().unwrap();
    let err = verify_release_attestation(&tampered, REPO, TAG, DIGEST, &trust).unwrap_err();
    // Depending on where the flip lands it fails at base64-decode, the DSSE
    // signature, the statement parse, or the rekor payload-hash binding —
    // all acceptable rejections.
    let msg = err.to_string();
    assert!(!msg.is_empty(), "tampered bundle must be rejected");
}
