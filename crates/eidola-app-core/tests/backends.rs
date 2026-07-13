//! Integration tests for the backend registry: the seeded singletons, the
//! external-backend CRUD lifecycle (+ `Change::Backends` emissions), and —
//! via the chat harness — routing turns through an `openai` backend (Bearer
//! key, **no credential spend, no Wallet emissions**, `backend_id` recorded)
//! and a `llamacpp` backend (loopback engine keyed by backend).

mod chat_harness;

use chat_harness::MockConfig;
use eidola_app_core::changes::Change;
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, BackendKind, BackendUpdate, NewBackend};

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// Drain all currently-available bus messages (non-blocking).
fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
    }
    out
}

/// A bare core (no chat mock) for registry-only tests.
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

fn openai_backend(id: &str, base_url: &str, api_key: Option<&str>) -> NewBackend {
    NewBackend {
        id: id.into(),
        kind: BackendKind::OpenAi,
        display_name: String::new(),
        base_url: Some(base_url.into()),
        api_key: api_key.map(String::from),
        models_dir: None,
        model_overrides: None,
    }
}

// ===========================================================================
// Registry
// ===========================================================================

#[test]
fn singleton_backends_are_seeded_and_ordered_first() {
    run(|| {
        let (core, _dir) = bare_core();
        let backends = core.runtime().block_on(core.list_backends()).expect("list");
        assert_eq!(backends.len(), 2);
        assert_eq!(backends[0].id, "eidola");
        assert_eq!(backends[0].kind, BackendKind::Eidola);
        assert!(backends[0].enabled);
        assert_eq!(backends[1].id, "local");
        assert_eq!(backends[1].kind, BackendKind::Local);
        assert!(backends[1].enabled);
    });
}

#[test]
fn external_backend_lifecycle_add_update_disable_remove_revive() {
    run(|| {
        let (core, _dir) = bare_core();
        // Touch the DB once so the seed emissions (none) are settled before
        // subscribing.
        core.runtime().block_on(core.list_backends()).unwrap();
        let mut rx = core.subscribe_changes();

        // Add.
        let added = core
            .runtime()
            .block_on(core.add_backend(openai_backend(
                "my-vllm",
                "http://127.0.0.1:1/",
                Some("sk-test"),
            )))
            .expect("add");
        assert_eq!(added.display_name, "my-vllm", "empty name defaults to id");
        assert_eq!(
            added.base_url.as_deref(),
            Some("http://127.0.0.1:1"),
            "trailing slash trimmed"
        );
        assert!(added.has_api_key);
        assert_eq!(drain(&mut rx), vec![Change::Backends]);

        // Duplicate id refused.
        let err = core
            .runtime()
            .block_on(core.add_backend(openai_backend("my-vllm", "http://x/", None)))
            .expect_err("duplicate");
        assert!(err.to_string().contains("already exists"), "got {err}");
        assert_eq!(drain(&mut rx), vec![]);

        // Update: pin models, rename.
        core.runtime()
            .block_on(core.update_backend(
                "my-vllm".into(),
                BackendUpdate {
                    display_name: Some("My vLLM box".into()),
                    model_overrides: Some(Some(vec!["qwen3-8b".into()])),
                    ..BackendUpdate::default()
                },
            ))
            .expect("update");
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        let mine = listed.iter().find(|b| b.id == "my-vllm").unwrap();
        assert_eq!(mine.display_name, "My vLLM box");
        assert_eq!(
            mine.model_overrides.as_deref(),
            Some(&["qwen3-8b".into()][..])
        );

        // Pinned models come back qualified, without any HTTP fetch.
        let models = core
            .runtime()
            .block_on(core.backend_models("my-vllm".into()))
            .expect("models");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "qwen3-8b@my-vllm");

        // Disable / enable.
        core.runtime()
            .block_on(core.set_backend_enabled("my-vllm".into(), false))
            .expect("disable");
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        assert!(!listed.iter().find(|b| b.id == "my-vllm").unwrap().enabled);

        // Remove (soft) — gone from the listing.
        core.runtime()
            .block_on(core.remove_backend("my-vllm".into()))
            .expect("remove");
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        assert!(!listed.iter().any(|b| b.id == "my-vllm"));

        // Re-adding the same id revives it with the new configuration.
        let revived = core
            .runtime()
            .block_on(core.add_backend(openai_backend("my-vllm", "http://127.0.0.1:2", None)))
            .expect("revive");
        assert_eq!(revived.base_url.as_deref(), Some("http://127.0.0.1:2"));
        assert!(!revived.has_api_key);
        assert!(revived.enabled);
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        assert!(listed.iter().any(|b| b.id == "my-vllm"));
    });
}

#[test]
fn add_backend_validates_ids_and_kind_requirements() {
    run(|| {
        let (core, _dir) = bare_core();
        let add = |new: NewBackend| core.runtime().block_on(core.add_backend(new));

        // Reserved and malformed ids.
        for bad in ["eidola", "local", "Has-Caps", "with space", "with@at", ""] {
            let err = add(openai_backend(bad, "http://x", None)).expect_err(bad);
            assert!(matches!(err, AppError::Config { .. }), "{bad}: {err}");
        }
        // openai needs a base URL.
        let err = add(NewBackend {
            id: "no-url".into(),
            kind: BackendKind::OpenAi,
            display_name: String::new(),
            base_url: None,
            api_key: None,
            models_dir: None,
            model_overrides: None,
        })
        .expect_err("missing url");
        assert!(err.to_string().contains("base URL"), "got {err}");
        // llamacpp needs a models dir.
        let err = add(NewBackend {
            id: "no-dir".into(),
            kind: BackendKind::LlamaCpp,
            display_name: String::new(),
            base_url: None,
            api_key: None,
            models_dir: None,
            model_overrides: None,
        })
        .expect_err("missing dir");
        assert!(err.to_string().contains("models directory"), "got {err}");
        // Built-in kinds cannot be added.
        let err = add(NewBackend {
            id: "fake-eidola".into(),
            kind: BackendKind::Eidola,
            display_name: String::new(),
            base_url: None,
            api_key: None,
            models_dir: None,
            model_overrides: None,
        })
        .expect_err("built-in kind");
        assert!(err.to_string().contains("built in"), "got {err}");

        // Singletons refuse update/remove but allow enable/disable.
        let err = core
            .runtime()
            .block_on(core.remove_backend("eidola".into()))
            .expect_err("remove eidola");
        assert!(err.to_string().contains("built in"), "got {err}");
        core.runtime()
            .block_on(core.set_backend_enabled("eidola".into(), false))
            .expect("disable eidola");
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        assert!(!listed.iter().find(|b| b.id == "eidola").unwrap().enabled);
    });
}

// ===========================================================================
// Routing through backends (chat harness)
// ===========================================================================

#[test]
fn openai_backend_chat_sends_bearer_no_spend_and_records_backend() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        // NOTE: no `with_account` — external inference must work with zero
        // onboarding, exactly like local.
        core.runtime()
            .block_on(core.add_backend(openai_backend(
                "my-vllm",
                &mock.base_url,
                Some("sk-test-key"),
            )))
            .expect("add backend");
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("Hello external".into(), "qwen3-8b@my-vllm".into(), None))
            .expect("external chat succeeds");

        assert_eq!(result.content, "Hello from the mock.");
        // The canonical qualified id is what's reported and recorded…
        assert_eq!(result.model, "qwen3-8b@my-vllm");
        assert_eq!(result.credits_charged, 0);

        // …while the wire body carries the bare model the server expects,
        // and the Authorization header is the backend's key, not an ACT
        // spend token.
        assert_eq!(
            mock.chat_bodies()[0]["model"],
            serde_json::json!("qwen3-8b")
        );
        assert_eq!(
            mock.chat_auth_values(),
            vec![Some("Bearer sk-test-key".to_string())]
        );
        assert_eq!(mock.refund_hits(), 0);

        // No Wallet emissions — nothing was spent.
        let changes = drain(&mut rx);
        assert!(
            !changes.contains(&Change::Wallet),
            "external turns must not touch the wallet: {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");

        // The inference action records the qualified id and no charge; the
        // request row records the backend.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(result.space_id.clone()))
            .expect("tree");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("inference node");
        assert_eq!(inference.model.as_deref(), Some("qwen3-8b@my-vllm"));
        assert_eq!(inference.credits_consumed, None);

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        let detail = core
            .runtime()
            .block_on(core.request_detail(requests[0].id.clone()))
            .expect("detail")
            .expect("row");
        assert_eq!(detail.backend_id.as_deref(), Some("my-vllm"));
        assert_eq!(detail.backend_display_name.as_deref(), Some("my-vllm"));
    });
}

#[test]
fn llamacpp_backend_routes_to_its_own_engine() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let models_dir = tempfile::tempdir().expect("models dir");
        std::fs::write(models_dir.path().join("tiny.gguf"), b"gguf").unwrap();

        core.runtime()
            .block_on(core.add_backend(NewBackend {
                id: "my-box".into(),
                kind: BackendKind::LlamaCpp,
                display_name: "My box".into(),
                base_url: None,
                api_key: None,
                models_dir: Some(models_dir.path().display().to_string()),
                model_overrides: None,
            }))
            .expect("add backend");

        // The scan sees the user's file under the qualified id.
        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        let external = state
            .external
            .iter()
            .find(|b| b.backend_id == "my-box")
            .expect("external backend section");
        assert_eq!(external.models.len(), 1);
        assert_eq!(external.models[0].id, "tiny@my-box");

        // Register a fake ready engine for it and chat through it.
        core.test_register_loaded_local_model("my-box", "tiny", mock.port());
        let result = core
            .runtime()
            .block_on(core.chat("Hello box".into(), "tiny@my-box".into(), None))
            .expect("llamacpp chat succeeds");
        assert_eq!(result.model, "tiny@my-box");
        assert_eq!(result.credits_charged, 0);
        // Engine turns carry no Authorization at all, and the body model is
        // the qualified id (== the engine's --alias).
        assert_eq!(mock.chat_auths(), vec![false]);
        assert_eq!(
            mock.chat_bodies()[0]["model"],
            serde_json::json!("tiny@my-box")
        );

        // Deleting a user-owned file through Eidola is refused.
        let err = core
            .runtime()
            .block_on(core.delete_local_model("tiny@my-box".into()))
            .expect_err("refuse external delete");
        assert!(err.to_string().contains("managed by you"), "got {err}");
    });
}

#[test]
fn disabled_backend_refuses_turns_with_typed_error() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        core.runtime()
            .block_on(core.set_backend_enabled("eidola".into(), false))
            .expect("disable");

        let err = core
            .runtime()
            .block_on(core.chat("Hi".into(), "gemma4-31b".into(), None))
            .expect_err("must refuse");
        assert!(
            matches!(err.root(), AppError::NotConfigured { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("disabled"), "got {err}");

        // Unknown backend ids are refused distinctly.
        let err = core
            .runtime()
            .block_on(core.chat("Hi".into(), "m@nowhere".into(), None))
            .expect_err("must refuse");
        assert!(err.to_string().contains("no backend named"), "got {err}");
    });
}

#[test]
fn openai_backend_models_come_from_listing_when_not_pinned() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        core.runtime()
            .block_on(core.add_backend(openai_backend("ext", &mock.base_url, None)))
            .expect("add");
        let models = core
            .runtime()
            .block_on(core.backend_models("ext".into()))
            .expect("models");
        // The mock lists gemma4-31b; through this backend it is qualified.
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemma4-31b@ext");
        // Generic listings publish no pricing — honest zeros.
        assert_eq!(models[0].context_length, 0);
        assert!(models[0].request_credits.is_none());
    });
}
