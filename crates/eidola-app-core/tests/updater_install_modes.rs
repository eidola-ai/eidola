//! The one test that has to change the process umask, alone in its own
//! binary.
//!
//! `umask` is process-wide, so a test that changes it changes it for every
//! test running beside it — and around an `await`, that includes work on
//! every other runtime thread. Two such tests interleaving can also leave
//! the process holding the wrong mask for good. Cargo gives each
//! integration file its own process, which is the only isolation that
//! actually holds; the mutex below is for the day this file gains a
//! second test.

use std::path::Path;

use eidola_app_core::updater::Fetcher;
use eidola_app_core::updater::install::{self, ExpectedSignature, InstallPlan, RemoteFile};

/// Serializes anything in this file that touches the process umask.
///
/// Async-aware, because the mask has to stay set across the install's
/// awaits — a `std` guard held over one blocks the runtime thread rather
/// than yielding it.
static UMASK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn apple_fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fixtures/apple-roundtrip/synthetic-universal")
}

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
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(file_mode(&path));
            writer
                .start_file(relative.to_string_lossy().to_string(), options)
                .unwrap();
            std::io::Write::write_all(&mut writer, &std::fs::read(&path).unwrap()).unwrap();
        }
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

fn file_mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

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

#[tokio::test(flavor = "multi_thread")]
async fn every_path_in_a_staged_bundle_carries_the_packers_modes() {
    // Reconstruction creates the signing directory and the seal itself,
    // with ordinary creates that take the umask — after the unpacked tree
    // was already normalized. A staged bundle should not carry two mode
    // rules depending on which step wrote each path.
    use std::os::unix::fs::PermissionsExt;

    let published = publish();
    let staging = tempfile::tempdir().unwrap();
    let root = staging.path().join("9.9.9");

    let _serialized = UMASK.lock().await;
    // SAFETY: `umask` reads and sets process state; it cannot fail.
    let previous = unsafe { libc::umask(0o077) };
    let staged = install::stage(
        &Fetcher::fixtures(published.dir.path()),
        &plan(&published),
        &root,
    )
    .await;
    unsafe { libc::umask(previous) };
    let staged = staged.expect("the fixture release should install");

    // Everything reconstruction wrote, not just everything unpacked.
    let seal = staged.bundle().join("Contents/_CodeSignature");
    assert!(seal.is_dir(), "the fixture reconstructs a bundle seal");

    let mut checked = 0;
    let mut stack = vec![staged.bundle().to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            if path.is_dir() {
                assert_eq!(mode, 0o755, "{}", path.display());
                stack.push(path);
            } else {
                assert!(
                    mode == 0o644 || mode == 0o755,
                    "{} is {mode:o}",
                    path.display()
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 3, "the walk should have seen the bundle");
    staged.discard().unwrap();
}
