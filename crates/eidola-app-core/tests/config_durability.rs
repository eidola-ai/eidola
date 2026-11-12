//! `config.toml` holds the account secret — the only credential for the
//! account's balance and subscription, which the server cannot reissue — next
//! to preferences that are rewritten on every zoom step or appearance change.
//! Every setter is a read-modify-write of the whole file, so these pin that no
//! preference write can cost the secret: not a hand edit the parser rejects,
//! not a reader that lands mid-write, and not another local user.

use eidola_app_core::AppCore;
use eidola_app_core::config::{AppearanceSetting, Config};
use eidola_app_core::error::AppError;

fn make_core() -> (AppCore, tempfile::TempDir) {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    (AppCore::new(config_dir, data_dir).expect("open core"), dir)
}

fn config_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("config.toml")
}

/// A hand edit the parser refuses (here an unquoted string) must never read
/// as "no config at all" to a setter, or the next preference write saves
/// those defaults over the file — the account secret with them.
#[test]
fn a_typo_elsewhere_in_config_never_erases_the_account() {
    let (core, dir) = make_core();
    core.set_account_credentials("acct-1".into(), "s3cret-never-lost".into())
        .unwrap();
    let path = config_path(&dir);
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str("language = en\n");
    std::fs::write(&path, &text).unwrap();

    // Refusing is fine; overwriting is not.
    let _ = core.set_appearance(AppearanceSetting::Day);
    let _ = core.set_font_scale(1.25);

    let after = std::fs::read_to_string(&path).unwrap();
    assert!(
        after.contains("s3cret-never-lost"),
        "a preference write replaced an unreadable config and lost the secret:\n{after}"
    );
}

/// An unreadable config shows first-run onboarding, and linking an account
/// there must not write over the one already held.
#[test]
fn an_unreadable_config_refuses_new_credentials() {
    let (core, dir) = make_core();
    core.set_account_credentials("acct-1".into(), "s3cret-never-lost".into())
        .unwrap();
    let path = config_path(&dir);
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str("font_scale = big\n");
    std::fs::write(&path, &text).unwrap();

    let linked = core.set_account_credentials("acct-2".into(), "other".into());
    let after = std::fs::read_to_string(&path).unwrap();
    assert!(
        linked.is_err() && after.contains("s3cret-never-lost"),
        "linking over an unreadable config replaced the held account:\n{after}"
    );
}

/// The refusal is shown — a settings alert, an unsaved account's reason — and
/// the parser's own text quotes the failing line. A sync write cut off inside
/// the secret fails on exactly that line, so the refusal names where, never
/// what.
#[test]
fn refusing_an_unreadable_config_never_quotes_the_secret() {
    let (core, dir) = make_core();
    let path = config_path(&dir);
    std::fs::write(
        &path,
        "account_id = \"acct-1\"\naccount_secret = \"s3cret-cut-off-mid-wri",
    )
    .unwrap();
    let refused = core
        .set_appearance(AppearanceSetting::Day)
        .expect_err("an unreadable config refuses the write");
    let text = format!("{refused} {refused:?}");
    assert!(
        !text.contains("s3cret-cut-off") && text.contains("line 2"),
        "the refusal quotes the secret, or loses its position: {text}"
    );

    // A value parsed into the wrong field is quoted by the message itself.
    std::fs::write(&path, "font_scale = \"s3cret-in-the-wrong-field\"\n").unwrap();
    let refused = core
        .set_appearance(AppearanceSetting::Day)
        .expect_err("an unreadable config refuses the write");
    let text = format!("{refused} {refused:?}");
    assert!(
        !text.contains("s3cret-in-the-wrong-field"),
        "the refusal quotes a value: {text}"
    );
}

/// A write that truncates in place lets a reader landing between the
/// truncate and the write see no account at all (and a crash there leaves it
/// so). The file is only ever replaced whole.
#[test]
fn a_reader_never_sees_a_half_written_config() {
    let (core, dir) = make_core();
    core.set_account_credentials("acct-1".into(), "s3cret-never-lost".into())
        .unwrap();
    let path = config_path(&dir);

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader = {
        let (path, stop) = (path.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut torn = 0usize;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if Config::load_from(&path).account_secret.is_none() {
                    torn += 1;
                }
            }
            torn
        })
    };
    for i in 0..300 {
        let scale = if i % 2 == 0 { 1.25 } else { 1.5 };
        core.set_font_scale(scale).unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let torn = reader.join().unwrap();
    assert_eq!(torn, 0, "{torn} reads saw a config with no account secret");
}

#[cfg(unix)]
#[test]
fn the_config_holding_the_secret_is_private_to_its_owner() {
    use std::os::unix::fs::PermissionsExt;
    let (core, dir) = make_core();
    core.set_account_credentials("acct-1".into(), "s3cret".into())
        .unwrap();
    let mode = std::fs::metadata(config_path(&dir))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o077, 0, "config.toml is mode {:o}", mode & 0o777);
}

/// Two setters at once — a zoom step on the GUI thread against a background
/// account link, say. Each save publishes only its own bytes, but each is a
/// read-modify-write of the whole file, so without one lock from load to
/// rename the later rename wins with the earlier snapshot, and the field the
/// other setter wrote is gone — the account secret, if that was the field.
#[test]
fn concurrent_setters_lose_no_update() {
    let (core, dir) = make_core();
    std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..200 {
                core.set_language(Some(format!("lang-{i}"))).unwrap();
            }
        });
        s.spawn(|| {
            for i in 0..200 {
                core.set_font_scale(1.0 + (i % 50) as f32 / 100.0).unwrap();
            }
            core.set_account_credentials("acct-1".into(), "s3cret-never-lost".into())
                .unwrap();
        });
    });

    let saved = Config::load_for_update(&config_path(&dir)).unwrap();
    assert_eq!(saved.language_override.as_deref(), Some("lang-199"));
    assert_eq!(saved.font_scale_override, Some(1.49));
    assert_eq!(saved.account_secret.as_deref(), Some("s3cret-never-lost"));
}

/// A crash between a save's create and its rename leaves its temporary
/// beside the profile, holding the secret. The next update sweeps it, under
/// the lock no other save of the file can be in flight without, and leaves
/// anything that only resembles one alone.
#[test]
fn a_crashed_saves_temporary_is_swept_by_the_next_update() {
    let (core, dir) = make_core();
    core.set_account_credentials("acct-1".into(), "s3cret".into())
        .unwrap();
    let stale = dir
        .path()
        .join(format!("config.toml.{}.tmp", "0123456789abcdef".repeat(2)));
    std::fs::write(&stale, "account_secret = \"s3cret\"\n").unwrap();
    let kept = ["config.toml.notes.tmp", "config.toml.bak", "other.toml.tmp"];
    for name in kept {
        std::fs::write(dir.path().join(name), "").unwrap();
    }

    core.set_font_scale(1.25).unwrap();

    assert!(
        !stale.exists(),
        "the crashed save's temporary was left behind"
    );
    for name in kept {
        assert!(dir.path().join(name).exists(), "{name} was swept");
    }
    let saved = Config::load_for_update(&config_path(&dir)).unwrap();
    assert_eq!(saved.account_secret.as_deref(), Some("s3cret"));
}

/// `POST /v1/account` that changes the profile under the create before it
/// answers — what a hand edit or a profile-sync write does while the request
/// is in flight — and then creates the account. The server keeps only the
/// secret's hash, so from its 201 on the response is the only copy.
struct CreateAfter {
    config: std::path::PathBuf,
    rewrite_to: &'static str,
}

impl wiremock::Respond for CreateAfter {
    fn respond(&self, _: &wiremock::Request) -> wiremock::ResponseTemplate {
        std::fs::write(&self.config, self.rewrite_to).unwrap();
        wiremock::ResponseTemplate::new(201).set_body_json(serde_json::json!({
            "account_id": "00000000-0000-0000-0000-00000000000c",
            "secret": "only-copy-of-the-new-secret",
            "created_at": "2025-01-01T00:00:00Z",
        }))
    }
}

/// `POST /v1/account/terms` — records the `Authorization` each submission
/// went out under, and accepts it.
struct RecordTerms(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl wiremock::Respond for RecordTerms {
    fn respond(&self, request: &wiremock::Request) -> wiremock::ResponseTemplate {
        let auth = request
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        self.0.lock().unwrap().push(auth);
        wiremock::ResponseTemplate::new(204)
    }
}

/// The one document the reader is shown and accepts.
fn presented_terms() -> eidola_app_core::TermsDocument {
    eidola_app_core::TermsDocument {
        document: "terms_of_service".into(),
        version: 3,
        url: "https://example.invalid/terms_of_service/".into(),
        sha256: "a".repeat(64),
    }
}

/// HTTP Basic credentials as they go on the wire.
fn basic(id: &str, secret: &str) -> String {
    use base64::Engine as _;
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("{id}:{secret}"))
    )
}

/// Create an account, accepting [`presented_terms`], against a server whose
/// create rewrites `config.toml` to `rewrite_to` first; returns the create's
/// outcome, the file after it, the `Authorization` of each terms
/// submission, and any recovery file the create left beside the profile.
fn create_while_config_becomes(rewrite_to: &'static str) -> CreateOutcome {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let client = reqwest::Client::builder().build().unwrap();
    let core =
        AppCore::with_test_http_client(dir.path().to_path_buf(), dir.path().join("data"), client)
            .expect("open core");
    let path = config_path(&dir);
    let submissions = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let server = core.runtime().block_on(async {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/account"))
            .respond_with(CreateAfter {
                config: path.clone(),
                rewrite_to,
            })
            .mount(&server)
            .await;
        let doc = presented_terms();
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/terms"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "documents": [{
                        "document": doc.document,
                        "version": doc.version,
                        "url": doc.url,
                        "sha256": doc.sha256,
                    }],
                })),
            )
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/account/terms"))
            .respond_with(RecordTerms(submissions.clone()))
            .mount(&server)
            .await;
        server
    });
    core.runtime()
        .block_on(core.set_base_url(server.uri()))
        .expect("set base url");
    let created = core
        .runtime()
        .block_on(core.account_create(vec![presented_terms()]));
    let after = std::fs::read_to_string(&path).unwrap();
    let submitted = submissions.lock().unwrap().clone();
    let recovery = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("account-recovery-"))
        })
        .map(|file| {
            #[cfg(unix)]
            let mode = {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::metadata(&file).unwrap().permissions().mode() & 0o777
            };
            // Only Unix has the bits; elsewhere the assertion holds trivially.
            #[cfg(not(unix))]
            let mode = 0o600;
            (file.clone(), Config::load_from(&file), mode)
        });
    CreateOutcome {
        created,
        after,
        submitted,
        recovery,
    }
}

/// What [`create_while_config_becomes`] observed.
struct CreateOutcome {
    created: Result<eidola_app_core::AccountCreateResult, AppError>,
    /// `config.toml` afterwards.
    after: String,
    /// The `Authorization` of each terms submission.
    submitted: Vec<String>,
    /// A recovery file beside the profile: its path, what it loads as, and
    /// its permission bits.
    recovery: Option<(std::path::PathBuf, Config, u32)>,
}

/// The strict reload refuses a config that became unparseable while the
/// create was in flight — rightly, it may hold another account — but the
/// account already exists by then. The refusal must hand the credential back,
/// not drop it with an ordinary config error.
#[test]
fn a_created_account_survives_a_config_that_broke_in_flight() {
    let broken = "font_scale = big\n";
    let CreateOutcome {
        created,
        after,
        submitted,
        recovery,
    } = create_while_config_becomes(broken);
    // The credential outlives a caller that drops the error (a window closed
    // mid-create): it is on disk, owner-only, in the profile's own shape.
    let (file, saved, mode) = recovery.expect("no recovery file was written");
    assert_eq!(
        (
            saved.account_id.as_deref(),
            saved.account_secret.as_deref(),
            mode
        ),
        (
            Some("00000000-0000-0000-0000-00000000000c"),
            Some("only-copy-of-the-new-secret"),
            0o600
        ),
        "the recovery file does not hold the credential privately"
    );
    match created {
        Err(AppError::AccountNotSaved { account, .. }) => {
            assert_eq!(account.recovery_file.as_deref(), Some(file.as_path()));
            assert_eq!(account.account_id, "00000000-0000-0000-0000-00000000000c");
            assert_eq!(account.secret.expose(), "only-copy-of-the-new-secret");
            let terms = &account.terms;
            // The snapshot the reader accepted is recorded for the account it
            // was accepted for, so linking it later does not come back gated.
            assert!(
                matches!(terms, eidola_app_core::TermsAcceptance::Complete),
                "the accepted terms were not recorded: {terms:?}"
            );
        }
        other => panic!("the created credential was not handed back: {other:?}"),
    }
    assert_eq!(after, broken, "the unreadable config was written over");
    assert_eq!(
        submitted,
        vec![basic(
            "00000000-0000-0000-0000-00000000000c",
            "only-copy-of-the-new-secret"
        )],
        "the accepted terms did not go out as the new account"
    );
}

/// The preflight's "no account yet" ran before the request. An account linked
/// while it was in flight is seen only under the write lock, and is never
/// overwritten: its secret would be the one lost. The new one comes back.
#[test]
fn an_account_linked_while_creating_is_never_overwritten() {
    let linked = "account_id = \"acct-1\"\naccount_secret = \"s3cret-never-lost\"\n";
    let CreateOutcome {
        created,
        after,
        submitted,
        recovery,
    } = create_while_config_becomes(linked);
    assert!(recovery.is_some(), "no recovery file was written");
    // Accepted for the account being created, never for the one that was
    // linked meanwhile.
    assert_eq!(
        submitted,
        vec![basic(
            "00000000-0000-0000-0000-00000000000c",
            "only-copy-of-the-new-secret"
        )],
        "the accepted terms went out under another account"
    );
    let err = match created {
        Err(e @ AppError::AccountNotSaved { .. }) => e,
        other => panic!("the create wrote over a linked account: {other:?}"),
    };
    assert!(
        after.contains("s3cret-never-lost") && !after.contains("only-copy-of-the-new-secret"),
        "the linked account was replaced:\n{after}"
    );
    // The secret travels for the surface to show, never into a log line.
    let AppError::AccountNotSaved { account, .. } = &err else {
        unreachable!()
    };
    assert_eq!(account.secret.expose(), "only-copy-of-the-new-secret");
    for printed in [format!("{err}"), format!("{err:?}")] {
        assert!(
            !printed.contains("only-copy-of-the-new-secret"),
            "the error prints its secret: {printed}"
        );
    }
}
