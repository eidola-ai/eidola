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

/// A hand edit the parser refuses (here an unquoted string) used to read as
/// "no config at all", and the next preference write then saved those
/// defaults over the file — the account secret with them.
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

/// Same hole, worse door: an unreadable config shows first-run onboarding,
/// and linking an account there must not write over the one already held.
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

/// `fs::write` truncates in place, so a reader landing between the truncate
/// and the write sees no account at all (and a crash there leaves it so). The
/// file must only ever be replaced whole.
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

/// Create an account against a server whose create rewrites `config.toml` to
/// `rewrite_to` first; returns the create's outcome and the file after it.
fn create_while_config_becomes(
    rewrite_to: &'static str,
) -> (
    Result<eidola_app_core::AccountCreateResult, AppError>,
    String,
) {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let client = reqwest::Client::builder().build().unwrap();
    let core =
        AppCore::with_test_http_client(dir.path().to_path_buf(), dir.path().join("data"), client)
            .expect("open core");
    let path = config_path(&dir);
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
        server
    });
    core.runtime()
        .block_on(core.set_base_url(server.uri()))
        .expect("set base url");
    let created = core.runtime().block_on(core.account_create(Vec::new()));
    let after = std::fs::read_to_string(&path).unwrap();
    (created, after)
}

/// The strict reload refuses a config that became unparseable while the
/// create was in flight — rightly, it may hold another account — but the
/// account already exists by then. The refusal must hand the credential back,
/// not drop it with an ordinary config error.
#[test]
fn a_created_account_survives_a_config_that_broke_in_flight() {
    let broken = "font_scale = big\n";
    let (created, after) = create_while_config_becomes(broken);
    match created {
        Err(AppError::AccountNotSaved {
            account_id, secret, ..
        }) => {
            assert_eq!(account_id, "00000000-0000-0000-0000-00000000000c");
            assert_eq!(secret.expose(), "only-copy-of-the-new-secret");
        }
        other => panic!("the created credential was not handed back: {other:?}"),
    }
    assert_eq!(after, broken, "the unreadable config was written over");
}

/// The preflight's "no account yet" ran before the request. An account linked
/// while it was in flight is seen only under the write lock, and is never
/// overwritten: its secret would be the one lost. The new one comes back.
#[test]
fn an_account_linked_while_creating_is_never_overwritten() {
    let linked = "account_id = \"acct-1\"\naccount_secret = \"s3cret-never-lost\"\n";
    let (created, after) = create_while_config_becomes(linked);
    let err = match created {
        Err(e @ AppError::AccountNotSaved { .. }) => e,
        other => panic!("the create wrote over a linked account: {other:?}"),
    };
    assert!(
        after.contains("s3cret-never-lost") && !after.contains("only-copy-of-the-new-secret"),
        "the linked account was replaced:\n{after}"
    );
    // The secret travels for the surface to show, never into a log line.
    let AppError::AccountNotSaved { secret, .. } = &err else {
        unreachable!()
    };
    assert_eq!(secret.expose(), "only-copy-of-the-new-secret");
    for printed in [format!("{err}"), format!("{err:?}")] {
        assert!(
            !printed.contains("only-copy-of-the-new-secret"),
            "the error prints its secret: {printed}"
        );
    }
}
