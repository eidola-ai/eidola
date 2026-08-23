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

// ---------------------------------------------------------------------------
// Install — download, verify, reconstruct, inspect
// ---------------------------------------------------------------------------
//
// These run on every platform. Reconstruction is pure Rust, so a Linux CI
// machine proves exactly what a Mac does; only promoting a staged bundle
// into a running application is macOS-shaped, and that is deliberately not
// what this module does.
//
// The fixture is the committed synthetic universal app the reconstruction
// crate is graded against. Its unsigned input and its signed output differ
// in a way the assertions lean on: the input carries no hardened-runtime
// flag and the output does, so a test that finds the flag set has watched
// the composition actually happen rather than watching a payload pass
// through untouched.

use eidola_app_core::updater::install::{
    self, ExpectedSignature, InstallError, InstallPlan, RemoteFile,
};

fn apple_fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fixtures/apple-roundtrip/synthetic-universal")
}

/// Pack a directory's *contents* into a zip, the way the shipping recipe
/// does: entries relative to the directory, modes carrying the executable
/// bit, deflate.
fn zip_dir(src: &Path) -> Vec<u8> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let mut stack = vec![src.to_path_buf()];
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    files.push(path);
                }
            }
        }
        files.sort();
        for path in files {
            let relative = path.strip_prefix(src).unwrap();
            let mode = file_mode(&path);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(mode);
            writer
                .start_file(relative.to_string_lossy().to_string(), options)
                .unwrap();
            std::io::Write::write_all(&mut writer, &std::fs::read(&path).unwrap()).unwrap();
        }
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

#[cfg(unix)]
fn file_mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[cfg(not(unix))]
fn file_mode(_path: &Path) -> u32 {
    0o644
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The two containers a macOS install downloads, written into a fixtures
/// directory the way `Fetcher::fixtures` reads them (URL last segment ->
/// filename).
struct Published {
    dir: tempfile::TempDir,
    payload_sha256: String,
    envelope_sha256: String,
}

fn publish() -> Published {
    let dir = tempfile::tempdir().unwrap();
    let payload = zip_dir(&apple_fixtures().join("settled"));
    let envelope = zip_dir(&apple_fixtures().join("detached"));
    let payload_sha256 = sha256_hex(&payload);
    let envelope_sha256 = sha256_hex(&envelope);
    std::fs::write(dir.path().join("payload.zip"), &payload).unwrap();
    std::fs::write(dir.path().join("envelope.zip"), &envelope).unwrap();
    Published {
        dir,
        payload_sha256,
        envelope_sha256,
    }
}

fn plan(published: &Published) -> InstallPlan {
    InstallPlan {
        version: "9.9.9".to_string(),
        bundle_name: "Fixture.app".to_string(),
        payload: RemoteFile {
            url: "https://example.com/v9.9.9/payload.zip".to_string(),
            sha256: published.payload_sha256.clone(),
        },
        envelope: Some(RemoteFile {
            url: "https://example.com/v9.9.9/envelope.zip".to_string(),
            sha256: published.envelope_sha256.clone(),
        }),
        expected_signature: Some(ExpectedSignature {
            team_id: None,
            identifier: "ai.eidola.fixture".to_string(),
            hardened_runtime: true,
        }),
    }
}

/// Every file under `root`, relative, with its bytes.
fn tree(root: &Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn stages_a_reconstructed_bundle_that_matches_the_signed_release() {
    let published = publish();
    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");

    let staged = install::stage(
        &Fetcher::fixtures(published.dir.path()),
        &plan(&published),
        &root,
    )
    .await
    .expect("the fixture release should install");

    // The composition happened, and it produced exactly the bytes the
    // release publishes as its signed app.
    assert_eq!(
        tree(staged.bundle()),
        tree(&apple_fixtures().join("signed/Fixture.app")),
        "the staged bundle should be byte-identical to the signed release"
    );

    let facts = staged.signature().expect("signature facts");
    assert_eq!(facts.identifier, "ai.eidola.fixture");
    assert!(
        facts.hardened_runtime,
        "the unsigned input carries no hardened-runtime flag; finding it set is how this test \
         knows reconstruction ran"
    );

    // Nothing was installed — only staged.
    // Under the *resolved* root: on macOS the temporary directory lives
    // below a symlinked `/var`, so the pinned path is spelled differently
    // than the one this test asked for.
    assert!(staged.bundle().starts_with(root.canonicalize().unwrap()));
    staged.discard().unwrap();
    assert!(!root.exists(), "discarding removes the staged tree");
}

/// The three refusals, each asserted to leave nothing behind.
async fn refuses_leaving_nothing(
    mutate: impl FnOnce(&Published, &mut InstallPlan),
    expect: impl FnOnce(&InstallError) -> bool,
    what: &str,
) {
    let published = publish();
    let mut plan = plan(&published);
    mutate(&published, &mut plan);

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");

    let error = install::stage(&Fetcher::fixtures(published.dir.path()), &plan, &root)
        .await
        .expect_err(what);
    assert!(expect(&error), "{what}: unexpected error {error}");
    assert!(
        !root.exists(),
        "{what}: a refused install must leave no staging tree"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_payload_that_is_not_what_the_manifest_records() {
    refuses_leaving_nothing(
        |_, plan| plan.payload.sha256 = "00".repeat(32),
        |e| matches!(e, InstallError::HashMismatch { label, .. } if label == "update payload"),
        "a payload whose hash is not the manifest's",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_signature_material_that_is_not_what_the_attestation_records() {
    refuses_leaving_nothing(
        |_, plan| plan.envelope.as_mut().unwrap().sha256 = "11".repeat(32),
        |e| matches!(e, InstallError::HashMismatch { label, .. } if label == "signature material"),
        "signature material whose hash is not the attestation's",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_corrupted_signature_material_that_hashes_to_its_own_bytes() {
    // The sharper corruption case: the container is damaged *and* the plan
    // names the damaged container's hash, so the hash gate passes and the
    // refusal has to come from reconstruction itself.
    let published = publish();
    let path = published.dir.path().join("envelope.zip");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 200;
    bytes[last] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();

    let mut plan = plan(&published);
    plan.envelope.as_mut().unwrap().sha256 = sha256_hex(&bytes);

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");
    let error = install::stage(&Fetcher::fixtures(published.dir.path()), &plan, &root)
        .await
        .expect_err("corrupted signature material should be refused");
    assert!(
        matches!(
            error,
            InstallError::Archive { .. } | InstallError::Reconstruct(_)
        ),
        "unexpected error: {error}"
    );
    assert!(
        !root.exists(),
        "a refused install must leave no staging tree"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_bundle_whose_claims_disagree_with_the_attestation() {
    refuses_leaving_nothing(
        |_, plan| {
            plan.expected_signature.as_mut().unwrap().team_id = Some("XXXXXXXXXX".to_string());
        },
        |e| matches!(e, InstallError::SignatureMismatch { field, .. } if *field == "Team ID"),
        "a bundle claiming a different Team ID than the attestation names",
    )
    .await;

    refuses_leaving_nothing(
        |_, plan| {
            plan.expected_signature.as_mut().unwrap().identifier = "ai.eidola.other".to_string();
        },
        |e| {
            matches!(e, InstallError::SignatureMismatch { field, .. } if *field == "bundle identifier")
        },
        "a bundle claiming a different identifier than the attestation names",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_plan_that_names_an_identity_it_would_never_check() {
    // The other direction: an identity the attestation requires, with no
    // signature material to apply. Nothing is composed, `inspect` never
    // runs, and the requirement would evaporate silently — an app staged
    // as if it had been checked against a Team ID nobody ever read.
    let published = publish();
    let mut plan = plan(&published);
    plan.envelope = None;

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");
    let error = install::stage(&Fetcher::fixtures(published.dir.path()), &plan, &root)
        .await
        .expect_err("an identity with nothing to check it against should be refused");
    assert!(matches!(error, InstallError::PlanIncomplete), "{error}");
    assert!(!root.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_plan_that_would_apply_signatures_without_checking_them() {
    let published = publish();
    let mut plan = plan(&published);
    plan.expected_signature = None;

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");
    let error = install::stage(&Fetcher::fixtures(published.dir.path()), &plan, &root)
        .await
        .expect_err("an unchecked identity should be refused");
    assert!(matches!(error, InstallError::PlanIncomplete));
    assert!(!root.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_staging_path_that_is_already_something_and_leaves_it_alone() {
    // A staging path can collide with a directory the caller never meant
    // to hand over. Refusing it is half the answer; the other half is not
    // deleting it on the way out, which is only structurally true if the
    // cleanup can run exclusively on a directory this call created.
    let published = publish();
    let root = tempfile::tempdir().unwrap();
    let occupied = root.path().join("not-ours");
    std::fs::create_dir(&occupied).unwrap();
    std::fs::write(occupied.join("irreplaceable.txt"), b"years of work").unwrap();

    let error = install::stage(
        &Fetcher::fixtures(published.dir.path()),
        &plan(&published),
        &occupied,
    )
    .await
    .expect_err("an occupied staging path should be refused");

    assert!(matches!(error, InstallError::Staging { .. }), "{error}");
    assert!(
        occupied.join("irreplaceable.txt").exists(),
        "refusing a staging path must never delete what was already there"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_bundle_name_that_would_reach_outside_the_staging_tree() {
    // Reconstruction *writes*. A plan that could name a path rather than a
    // directory would hand it somewhere this module promised never to go.
    for name in [
        "../../../outside",
        "/tmp/eidola-install-escape",
        "a/b",
        "..",
    ] {
        let published = publish();
        let mut p = plan(&published);
        p.bundle_name = name.to_string();

        let staging = tempfile::tempdir().unwrap();
        let root = staging.path().join("9.9.9");
        let error = install::stage(&Fetcher::fixtures(published.dir.path()), &p, &root)
            .await
            .expect_err("a bundle name that is not one component should be refused");
        assert!(
            matches!(&error, InstallError::BundleMissing { expected } if expected == name),
            "`{name}`: {error}"
        );
        assert!(!root.exists());
        assert!(
            !Path::new("/tmp/eidola-install-escape").exists(),
            "nothing outside the staging tree may be created"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_members_that_the_filesystem_would_fold_together() {
    // `A.app/x` and `a.app/x` are one file on a case-insensitive volume —
    // the macOS default — so accepting both means the bytes that land are
    // whichever the extractor wrote last.
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);
        for name in ["A.app/Contents/Info.plist", "a.app/Contents/Info.plist"] {
            writer.start_file(name.to_string(), options).unwrap();
            std::io::Write::write_all(&mut writer, b"body").unwrap();
        }
        writer.finish().unwrap();
    }
    let payload = cursor.into_inner();

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("payload.zip"), &payload).unwrap();

    let plan = InstallPlan {
        version: "9.9.9".to_string(),
        bundle_name: "A.app".to_string(),
        payload: RemoteFile {
            url: "https://example.com/v9.9.9/payload.zip".to_string(),
            sha256: sha256_hex(&payload),
        },
        envelope: None,
        expected_signature: None,
    };

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");
    let error = install::stage(&Fetcher::fixtures(dir.path()), &plan, &root)
        .await
        .expect_err("case-equivalent members should be refused");
    assert!(
        matches!(&error, InstallError::Archive { reason, .. } if reason.contains("same file")),
        "{error}"
    );
    assert!(!root.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn creates_the_staging_root_private_whatever_the_umask() {
    // Every hash this module checks would be checkable-then-replaceable if
    // the tree were world-writable: another local user could swap a
    // verified payload before it was used.
    use std::os::unix::fs::PermissionsExt;

    let published = publish();
    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");

    // SAFETY: `umask` reads and sets process state; it cannot fail.
    let previous = unsafe { libc::umask(0o000) };
    let staged = install::stage(
        &Fetcher::fixtures(published.dir.path()),
        &plan(&published),
        &root,
    )
    .await;
    unsafe { libc::umask(previous) };

    let staged = staged.expect("the fixture release should install");
    let mode = std::fs::metadata(&root).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o700,
        "the staging root must not be reachable by others"
    );
    staged.discard().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_a_staging_path_whose_parent_does_not_exist() {
    // The caller picks where staging happens, so the directory holding it
    // is the caller's. Creating it would be the same ownership mistake as
    // deleting one we did not create, pointed the other way.
    let published = publish();
    let root = tempfile::tempdir().unwrap();
    let deep = root.path().join("a/b/c/staging");

    let error = install::stage(
        &Fetcher::fixtures(published.dir.path()),
        &plan(&published),
        &deep,
    )
    .await
    .expect_err("a missing parent should be refused");

    assert!(matches!(error, InstallError::Staging { .. }), "{error}");
    assert!(
        !root.path().join("a").exists(),
        "refusing must not leave ancestors behind"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abandoned_install_takes_its_staging_tree_with_it() {
    // An install can end by never finishing: a caller that times out, or a
    // quit that races the download, drops the future mid-flight. Cleanup
    // written as an error arm never runs then.
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(vec![0u8; 64])
                .set_delay(std::time::Duration::from_secs(30)),
        )
        .mount(&server)
        .await;

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");
    let fetcher = Fetcher::network().unwrap();
    let mut plan = plan(&publish());
    plan.payload.url = format!("{}/payload.zip", server.uri());
    plan.envelope = None;
    plan.expected_signature = None;

    let outcome = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        install::stage(&fetcher, &plan, &root),
    )
    .await;

    assert!(
        outcome.is_err(),
        "the install should still have been running"
    );
    assert!(
        !root.exists(),
        "an abandoned install must not leave a staging tree for a later run to find"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_bundle_name_is_matched_exactly_not_resolved_by_the_filesystem() {
    // A container whose bundle directory differs from the plan's name only
    // by a spelling the filesystem folds. `Path::is_dir` would resolve it;
    // the plan's exact-name requirement would go unenforced and the path
    // handed to reconstruction would be an alias.
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);
        writer
            .start_file("fixture.app/Contents/Info.plist".to_string(), options)
            .unwrap();
        std::io::Write::write_all(&mut writer, b"body").unwrap();
        writer.finish().unwrap();
    }
    let payload = cursor.into_inner();

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("payload.zip"), &payload).unwrap();

    let plan = InstallPlan {
        version: "9.9.9".to_string(),
        bundle_name: "Fixture.app".to_string(),
        payload: RemoteFile {
            url: "https://example.com/v9.9.9/payload.zip".to_string(),
            sha256: sha256_hex(&payload),
        },
        envelope: None,
        expected_signature: None,
    };

    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");
    let error = install::stage(&Fetcher::fixtures(dir.path()), &plan, &root)
        .await
        .expect_err("a bundle that is not the named one should be refused");
    assert!(
        matches!(&error, InstallError::BundleMissing { expected } if expected == "Fixture.app"),
        "{error}"
    );
    assert!(!root.exists());
}
