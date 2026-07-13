//! Integration tests for the local-inference domain: model downloads (over a
//! wiremock HTTP fixture), deletion, and — via the chat harness — routing
//! `local/<slug>` turns to a loopback engine with **no credential spend, no
//! ACT header, and no Wallet emissions**. The engine itself is faked through
//! the `test_register_loaded_local_model` seam; the real `llama-server`
//! subprocess lifecycle is exercised manually / end-to-end (it needs the
//! binary and a multi-GB model, neither of which belongs in CI).

mod chat_harness;

use chat_harness::{ChatBehavior, MockConfig, with_account};
use eidola_app_core::changes::Change;
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatStreamEvent, LocalModelStatus};

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// Drain all currently-available bus messages (non-blocking).
fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(c) => out.push(c),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                panic!("test receiver lagged by {n}");
            }
        }
    }
    out
}

/// A bare core (no chat mock) with a plain HTTP client for download tests.
fn bare_core() -> (AppCore, tempfile::TempDir) {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().expect("tempdir");
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    let client = reqwest::Client::builder().build().expect("client");
    (
        AppCore::with_test_http_client(config_dir, data_dir, client),
        dir,
    )
}

/// Poll `local_models_state` until `pred` holds (or panic after ~10s).
fn wait_for_state(
    core: &AppCore,
    pred: impl Fn(&eidola_app_core::LocalModelsState) -> bool,
) -> eidola_app_core::LocalModelsState {
    for _ in 0..200 {
        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        if pred(&state) {
            return state;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("local_models_state never reached the expected condition");
}

// ===========================================================================
// Downloads
// ===========================================================================

#[test]
fn download_persists_model_and_emits() {
    run(|| {
        let (core, dir) = bare_core();
        let mut rx = core.subscribe_changes();

        let gguf_bytes = vec![0x47u8; 64 * 1024]; // arbitrary payload
        let mock = core.runtime().block_on(async {
            let mock = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .and(wiremock::matchers::path("/repo/tiny-test-model.gguf"))
                .respond_with(
                    wiremock::ResponseTemplate::new(200).set_body_bytes(gguf_bytes.clone()),
                )
                .mount(&mock)
                .await;
            mock
        });

        let id = core
            .runtime()
            .block_on(
                core.download_local_model(format!("{}/repo/tiny-test-model.gguf", mock.uri())),
            )
            .expect("download starts");
        assert_eq!(id, "local/tiny-test-model");

        let state = wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "tiny-test-model" && m.status == LocalModelStatus::Available)
        });
        let model = state
            .models
            .iter()
            .find(|m| m.slug == "tiny-test-model")
            .unwrap();
        assert_eq!(model.id, "local/tiny-test-model");
        assert_eq!(model.size_bytes, Some(64 * 1024));
        assert_eq!(model.last_error, None);
        assert!(
            model
                .source_url
                .as_deref()
                .is_some_and(|u| u.ends_with("/repo/tiny-test-model.gguf")),
            "sidecar records the source URL: {:?}",
            model.source_url
        );

        // The file (and its sidecar) landed under <data_dir>/models, and no
        // .part file remains.
        let models_dir = dir.path().join("data").join("models");
        assert!(models_dir.join("tiny-test-model.gguf").is_file());
        assert!(models_dir.join("tiny-test-model.gguf.meta.json").is_file());
        assert!(!models_dir.join("tiny-test-model.gguf.part").exists());

        // Start + completion each emitted LocalModels; nothing else moved.
        let changes = drain(&mut rx);
        assert!(
            changes
                .iter()
                .filter(|c| **c == Change::LocalModels)
                .count()
                >= 2,
            "got {changes:?}"
        );
        assert!(
            changes.iter().all(|c| *c == Change::LocalModels),
            "downloads must not touch other domains: {changes:?}"
        );
    });
}

#[test]
fn failed_download_surfaces_error_and_cleans_up() {
    run(|| {
        let (core, dir) = bare_core();

        let mock = core.runtime().block_on(async {
            let mock = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .respond_with(wiremock::ResponseTemplate::new(404))
                .mount(&mock)
                .await;
            mock
        });

        core.runtime()
            .block_on(core.download_local_model(format!("{}/gone.gguf", mock.uri())))
            .expect("download starts");

        let state = wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "gone" && m.last_error.is_some())
        });
        let model = state.models.iter().find(|m| m.slug == "gone").unwrap();
        assert!(
            model.last_error.as_deref().unwrap().contains("HTTP 404"),
            "got {:?}",
            model.last_error
        );

        let models_dir = dir.path().join("data").join("models");
        assert!(!models_dir.join("gone.gguf").exists());
        assert!(!models_dir.join("gone.gguf.part").exists());
    });
}

#[test]
fn download_rejects_bad_urls_and_duplicates() {
    run(|| {
        let (core, dir) = bare_core();

        // Non-.gguf and non-http are typed LocalModel errors.
        for bad in ["https://example.com/model.bin", "not a url", ""] {
            let err = core
                .runtime()
                .block_on(core.download_local_model(bad.into()))
                .expect_err("must reject");
            assert!(matches!(err, AppError::LocalModel { .. }), "got {err:?}");
        }

        // An already-downloaded file refuses a second download.
        let models_dir = dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("dupe.gguf"), b"gguf").unwrap();
        let err = core
            .runtime()
            .block_on(core.download_local_model("https://example.com/dupe.gguf".into()))
            .expect_err("must reject duplicate");
        assert!(err.to_string().contains("already downloaded"), "got {err}");
    });
}

// ===========================================================================
// Delete
// ===========================================================================

#[test]
fn delete_removes_files_and_emits() {
    run(|| {
        let (core, dir) = bare_core();
        let models_dir = dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("doomed.gguf"), b"gguf").unwrap();
        std::fs::write(models_dir.join("doomed.gguf.meta.json"), b"{}").unwrap();

        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        assert!(state.models.iter().any(|m| m.slug == "doomed"));

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.delete_local_model("local/doomed".into()))
            .expect("delete");

        assert!(!models_dir.join("doomed.gguf").exists());
        assert!(!models_dir.join("doomed.gguf.meta.json").exists());
        let changes = drain(&mut rx);
        assert_eq!(changes, vec![Change::LocalModels]);

        // Deleting a loaded model is refused.
        std::fs::write(models_dir.join("held.gguf"), b"gguf").unwrap();
        core.test_register_loaded_local_model("held", 1);
        let err = core
            .runtime()
            .block_on(core.delete_local_model("held".into()))
            .expect_err("refuse while loaded");
        assert!(err.to_string().contains("unload"), "got {err}");
    });
}

// ===========================================================================
// Local chat turns — the credential-free path through run_turn
// ===========================================================================

#[test]
fn local_blocking_chat_has_no_spend_no_auth_no_wallet() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        // NOTE: no `with_account` — local inference must work with zero
        // onboarding (no account, no balance, no credentials).
        core.test_register_loaded_local_model("test-model", mock.port());
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("Hello local".into(), "local/test-model".into(), None))
            .expect("local chat succeeds");

        assert_eq!(result.content, "Hello from the mock.");
        assert_eq!(result.model, "local/test-model");
        assert_eq!(result.credits_charged, 0);

        // The upstream request carried no Authorization header and the
        // refund endpoint was never consulted.
        assert_eq!(mock.chat_hits(), 1);
        assert_eq!(mock.chat_auths(), vec![false]);
        assert_eq!(mock.refund_hits(), 0);
        // The request body carried the local model id.
        assert_eq!(
            mock.chat_bodies()[0]["model"],
            serde_json::json!("local/test-model")
        );

        // Emissions: post's SpaceIndex + Space, then run_turn's Space +
        // Record. Never Wallet, never LocalModels.
        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(
            changes.contains(&Change::Space(result.space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Wallet),
            "local turns must not touch the wallet: {changes:?}"
        );

        // Persistence: the inference action records the local model id and
        // no consumed credits (None, not zero).
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(result.space_id.clone()))
            .expect("tree");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("inference node");
        assert_eq!(inference.model.as_deref(), Some("local/test-model"));
        assert_eq!(inference.credits_consumed, None);

        // The request row exists with no credential nonce (Record stays
        // honest about the free turn).
        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].response_status, Some(200));
    });
}

#[test]
fn local_streaming_chat_streams_and_persists_without_wallet() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        core.test_register_loaded_local_model("test-model", mock.port());
        let mut rx = core.subscribe_changes();

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let result = core
            .runtime()
            .block_on(core.chat_stream("Stream local".into(), "local/test-model".into(), None, tx))
            .expect("local chat_stream succeeds");

        assert_eq!(result.content, "Hello from the stream.");
        assert_eq!(result.credits_charged, 0);

        let mut got_content_delta = false;
        while let Ok(ev) = events_rx.try_recv() {
            if matches!(ev, ChatStreamEvent::ContentDelta(_)) {
                got_content_delta = true;
            }
        }
        assert!(got_content_delta, "content deltas must be forwarded");

        assert_eq!(mock.chat_auths(), vec![false]);
        assert_eq!(mock.refund_hits(), 0, "SSE recovery must not run locally");

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Wallet),
            "local turns must not touch the wallet: {changes:?}"
        );
    });
}

#[test]
fn local_chat_with_unloaded_model_is_typed_error() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        with_account(&core);

        let err = core
            .runtime()
            .block_on(core.chat("Hi".into(), "local/never-loaded".into(), None))
            .expect_err("must fail");
        assert!(
            matches!(err.root(), AppError::LocalModel { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("not loaded"), "got {err}");

        // The post persisted before routing failed — the saved thought
        // survives (post-first contract), with no inference row.
        let spaces = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");
        assert_eq!(spaces.len(), 1);
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(spaces[0].id.clone()))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    });
}
