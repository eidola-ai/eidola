//! Real-bytes fixture test for the CI Sigstore verifier.
//!
//! The unit tests in `ci_sigstore::tests` use a `minimal_bundle` with a
//! placeholder `canonicalizedBody: "AAA="` that never reaches the
//! `hashedrekord` body parser. That left a real schema-drift bug
//! invisible: `HashedRekordBody.api_version` was missing
//! `#[serde(rename = "apiVersion")]`, so any actual Rekor entry would
//! fail to deserialize with `missing field `api_version``. This fixture
//! covers that gap by driving `verify_ci_signature` end-to-end against
//! captured release bytes from a real `v*` tag.
//!
//! When iterating on the verifier, edit-compile-test against this file:
//!
//!     cargo test -p eidola-app-core --test ci_sigstore_fixture
//!
//! No network, no Rekor writes, no `release-tool` rebuild loop.
//!
//! To refresh or add fixtures:
//!
//!     gh release download v0.0.X -R eidola-ai/eidola \
//!       -p artifact-manifest.json -p artifact-manifest.json.sigstore \
//!       -D crates/eidola-app-core/tests/fixtures/v0.0.X/

use eidola_app_core::updater::{ci_sigstore, trust};

#[test]
fn verifies_v0_0_8_release_bundle() {
    let manifest = include_bytes!("fixtures/v0.0.8/artifact-manifest.json");
    let bundle = include_bytes!("fixtures/v0.0.8/artifact-manifest.json.sigstore");

    let trust = trust::load().expect("loading embedded sigstore trust root");
    let verified = ci_sigstore::verify_ci_signature(manifest, bundle, &trust)
        .expect("verify_ci_signature against real v0.0.8 fixture");

    // The identity/issuer assertions are the same ones the embedded trust
    // root globs against; reasserting them here catches any silent
    // weakening of those checks.
    assert!(
        verified.ci_identity.contains("tinfoil-build.yml"),
        "expected CI identity to reference the tinfoil-build.yml workflow, got: {}",
        verified.ci_identity
    );
    assert_eq!(
        verified.ci_issuer,
        "https://token.actions.githubusercontent.com"
    );
    assert!(
        verified.rekor_log_index > 0,
        "rekor log index should be set"
    );
}
