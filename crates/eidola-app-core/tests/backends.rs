//! Integration tests for the backend registry: the seeded singletons, the
//! external-backend CRUD lifecycle (+ `Change::Backends` emissions), and —
//! via the chat harness — routing turns through an `openai` backend (Bearer
//! key, **no credential spend, no Wallet emissions**, `backend_id` recorded)
//! and a `llamacpp` backend (loopback engine keyed by backend).

mod chat_harness;

use chat_harness::MockConfig;
use eidola_app_core::changes::{Change, ChangeEvent};
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, BackendKind, BackendUpdate, NewBackend};

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// Drain all currently-available bus messages (non-blocking).
fn drain(rx: &mut tokio::sync::broadcast::Receiver<ChangeEvent>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c.change);
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
        AppCore::with_test_http_client(config_dir, data_dir, client).expect("open core"),
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
        engine_path: None,
        auto_start: true,
    }
}

/// A `llamacpp` backend over `models_dir`, with an optional explicit engine
/// path and the auto-start flag.
fn llamacpp_backend(
    id: &str,
    models_dir: &str,
    engine_path: Option<&str>,
    auto_start: bool,
) -> NewBackend {
    NewBackend {
        id: id.into(),
        kind: BackendKind::LlamaCpp,
        display_name: String::new(),
        base_url: None,
        api_key: None,
        models_dir: Some(models_dir.into()),
        model_overrides: None,
        engine_path: engine_path.map(String::from),
        auto_start,
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
fn eidola_row_owns_connection_and_trust_bundle() {
    run(|| {
        let (core, _dir) = bare_core();
        let mut rx = core.subscribe_changes();

        // Defaults: the embedded pin, no overrides.
        let trust = core.runtime().block_on(core.eidola_trust()).unwrap();
        assert!(!trust.base_url_is_override);
        assert_eq!(trust.base_url, trust.base_url_pin);
        assert!(!trust.trusted_measurements_are_override);
        assert_eq!(trust.trusted_measurements.len(), 1);
        assert!(!trust.has_hardware_root_ca);

        // Base URL override → Backends emission, honest override flag.
        core.runtime()
            .block_on(core.set_base_url("https://staging.example/v1".into()))
            .unwrap();
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let trust = core.runtime().block_on(core.eidola_trust()).unwrap();
        assert!(trust.base_url_is_override);
        assert_eq!(trust.base_url, "https://staging.example/v1");

        // Revert to pin.
        core.runtime()
            .block_on(core.clear_base_url_override())
            .unwrap();
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let trust = core.runtime().block_on(core.eidola_trust()).unwrap();
        assert!(!trust.base_url_is_override);

        // Trust a measurement (idempotent), then untrust back to pin.
        let snp = "a".repeat(96);
        let r1 = "b".repeat(96);
        let r2 = "c".repeat(96);
        let added = core
            .runtime()
            .block_on(core.trust_measurement(snp.clone(), r1.clone(), r2.clone()))
            .unwrap();
        assert!(added);
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let again = core
            .runtime()
            .block_on(core.trust_measurement(snp.clone(), r1, r2))
            .unwrap();
        assert!(!again, "idempotent: no second write");
        assert_eq!(drain(&mut rx), vec![]);
        let trust = core.runtime().block_on(core.eidola_trust()).unwrap();
        assert!(trust.trusted_measurements_are_override);
        assert_eq!(trust.trusted_measurements.len(), 1);
        assert_eq!(trust.trusted_measurements[0].snp, snp);

        let removed = core
            .runtime()
            .block_on(core.untrust_measurement(snp.clone()))
            .unwrap();
        assert!(removed);
        assert_eq!(drain(&mut rx), vec![Change::Backends]);
        let trust = core.runtime().block_on(core.eidola_trust()).unwrap();
        assert!(
            !trust.trusted_measurements_are_override,
            "emptying the override list reverts to the pin"
        );

        // A malformed base URL never lands (validated before write).
        let err = core
            .runtime()
            .block_on(core.set_base_url("not-a-url".into()))
            .expect_err("bad url");
        assert!(matches!(err, AppError::Config { .. }), "{err}");
        assert_eq!(drain(&mut rx), vec![]);
    });
}

#[test]
fn update_backend_per_kind_field_validation() {
    run(|| {
        let (core, _dir) = bare_core();

        // `local` is built in — it refuses every update.
        let err = core
            .runtime()
            .block_on(core.update_backend(
                "local".into(),
                BackendUpdate {
                    base_url: Some(Some("http://x".into())),
                    ..BackendUpdate::default()
                },
            ))
            .expect_err("local update");
        assert!(err.to_string().contains("built in"), "got {err}");

        // `eidola` refuses non-trust (external) fields.
        let err = core
            .runtime()
            .block_on(core.update_backend(
                "eidola".into(),
                BackendUpdate {
                    api_key: Some(Some("k".into())),
                    ..BackendUpdate::default()
                },
            ))
            .expect_err("eidola external field");
        assert!(
            err.to_string().contains("connection and trust"),
            "got {err}"
        );

        // An `openai` backend refuses the trust bundle.
        core.runtime()
            .block_on(core.add_backend(openai_backend("box", "http://x", None)))
            .unwrap();
        let err = core
            .runtime()
            .block_on(core.update_backend(
                "box".into(),
                BackendUpdate {
                    hardware_root_ca: Some(Some("pem".into())),
                    ..BackendUpdate::default()
                },
            ))
            .expect_err("openai trust field");
        assert!(err.to_string().contains("only to the eidola"), "got {err}");
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
            engine_path: None,
            auto_start: true,
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
            engine_path: None,
            auto_start: true,
        })
        .expect_err("missing dir");
        assert!(err.to_string().contains("models directory"), "got {err}");
        // engine path / auto-start are llamacpp-only.
        let err = add(NewBackend {
            engine_path: Some("/usr/bin/llama-server".into()),
            ..openai_backend("openai-engine", "http://x", None)
        })
        .expect_err("engine path on openai");
        assert!(err.to_string().contains("llama.cpp backends"), "got {err}");
        let err = add(NewBackend {
            auto_start: false,
            ..openai_backend("openai-noauto", "http://x", None)
        })
        .expect_err("auto-start off on openai");
        assert!(err.to_string().contains("llama.cpp backends"), "got {err}");
        // Built-in kinds cannot be added.
        let err = add(NewBackend {
            id: "fake-eidola".into(),
            kind: BackendKind::Eidola,
            display_name: String::new(),
            base_url: None,
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
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

#[test]
fn llamacpp_engine_path_and_auto_start_persist_and_revive() {
    run(|| {
        let (core, _dir) = bare_core();

        // Add with an explicit engine path and auto-start disabled.
        let added = core
            .runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                "/Users/me/models",
                Some("/opt/llama-server"),
                false,
            )))
            .expect("add");
        assert_eq!(added.engine_path.as_deref(), Some("/opt/llama-server"));
        assert!(!added.auto_start);

        // The columns round-trip through the listing.
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        let mine = listed.iter().find(|b| b.id == "my-box").unwrap();
        assert_eq!(mine.engine_path.as_deref(), Some("/opt/llama-server"));
        assert!(!mine.auto_start);

        // Update flips auto-start and clears the engine path.
        core.runtime()
            .block_on(core.update_backend(
                "my-box".into(),
                BackendUpdate {
                    auto_start: Some(true),
                    engine_path: Some(None),
                    ..BackendUpdate::default()
                },
            ))
            .expect("update");
        let listed = core.runtime().block_on(core.list_backends()).unwrap();
        let mine = listed.iter().find(|b| b.id == "my-box").unwrap();
        assert!(mine.auto_start);
        assert_eq!(mine.engine_path, None);

        // engine_path / auto_start are refused on a non-llamacpp backend.
        core.runtime()
            .block_on(core.add_backend(openai_backend("ext", "http://x", None)))
            .expect("add openai");
        let err = core
            .runtime()
            .block_on(core.update_backend(
                "ext".into(),
                BackendUpdate {
                    auto_start: Some(false),
                    ..BackendUpdate::default()
                },
            ))
            .expect_err("auto-start on openai");
        assert!(err.to_string().contains("llama.cpp backends"), "got {err}");

        // Soft-remove, then revive with fresh engine settings.
        core.runtime()
            .block_on(core.remove_backend("my-box".into()))
            .expect("remove");
        let revived = core
            .runtime()
            .block_on(core.add_backend(llamacpp_backend("my-box", "/Users/me/models", None, false)))
            .expect("revive");
        assert_eq!(revived.engine_path, None);
        assert!(!revived.auto_start);
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
                display_name: "My box".into(),
                ..llamacpp_backend(
                    "my-box",
                    &models_dir.path().display().to_string(),
                    None,
                    true,
                )
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
fn llamacpp_auto_start_disabled_refuses_request_without_spawning() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let models_dir = tempfile::tempdir().expect("models dir");
        std::fs::write(models_dir.path().join("tiny.gguf"), b"gguf").unwrap();

        // auto-start OFF, and no engine registered/loaded.
        core.runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &models_dir.path().display().to_string(),
                None,
                false,
            )))
            .expect("add backend");

        let err = core
            .runtime()
            .block_on(core.chat("Hi".into(), "tiny@my-box".into(), None))
            .expect_err("must refuse a request-triggered load");
        // Typed refusal following the disabled-backend pattern; nothing was
        // spawned (the model stays Available, never Loading/Loaded).
        assert!(matches!(err, AppError::NotConfigured { .. }), "got {err:?}");
        assert!(err.to_string().contains("auto-start"), "got {err}");

        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        let external = state
            .external
            .iter()
            .find(|b| b.backend_id == "my-box")
            .expect("external backend section");
        assert!(!external.auto_start);
        assert!(
            matches!(
                external.models[0].status,
                eidola_app_core::LocalModelStatus::Available
            ),
            "no engine should have spawned: {:?}",
            external.models[0].status
        );
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
        assert!(matches!(err, AppError::NotConfigured { .. }), "got {err:?}");
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
        // The mock's catalog (gemma4-31b plus the router-test model) comes
        // back qualified through this backend.
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["gemma4-31b@ext", "router-remote@ext"],
            "every listed model is qualified with the backend id"
        );
        // Generic listings publish no pricing — honest zeros.
        assert_eq!(models[0].context_length, 0);
        assert!(models[0].request_credits.is_none());
    });
}

/// An engine may not outlive the backend row that gave it its meaning.
///
/// Engines are keyed by `(backend_id, slug)` and removal is a *soft* remove —
/// re-adding the same id revives the row. So an engine left registered under
/// a removed backend is inherited by its revival: point the revived backend at
/// a different models directory that happens to hold the same file name, and
/// the next load of `<slug>@<id>` finds the old ready entry, returns
/// immediately, and every turn runs against the previous file while Settings
/// describes the new directory.
#[test]
fn removing_a_backend_retires_its_engines_so_a_revival_cannot_inherit_them() {
    run(|| {
        let (core, dir) = bare_core();
        let old_models = dir.path().join("old-models");
        let new_models = dir.path().join("new-models");
        std::fs::create_dir_all(&old_models).unwrap();
        std::fs::create_dir_all(&new_models).unwrap();
        // The same file name in both directories — the trap a revival springs.
        std::fs::write(old_models.join("tiny.gguf"), b"old").unwrap();
        std::fs::write(new_models.join("tiny.gguf"), b"new").unwrap();

        core.runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &old_models.display().to_string(),
                Some("/usr/bin/false"),
                true,
            )))
            .expect("add backend");
        core.test_register_loaded_local_model("my-box", "tiny", 51234);
        // A managed-store engine stands by to prove the sweep is scoped.
        core.test_register_loaded_local_model("local", "keep", 51235);

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.remove_backend("my-box".into()))
            .expect("remove");

        let running: Vec<String> = core.running_engines().into_iter().map(|e| e.id).collect();
        assert_eq!(
            running,
            vec!["keep@local".to_string()],
            "the removed backend's engine must be gone, and only it"
        );
        let emitted = drain(&mut rx);
        assert!(
            emitted.contains(&Change::LocalModels),
            "retiring an engine is a local-models change: {emitted:?}"
        );

        // Revive the id over a different directory. The engine that served
        // the *old* directory's `tiny.gguf` must not answer for the new one:
        // the load has to start a real engine, which `/usr/bin/false` makes
        // fail honestly rather than silently succeeding against port 51234.
        core.runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &new_models.display().to_string(),
                Some("/usr/bin/false"),
                true,
            )))
            .expect("revive backend");
        let err = core
            .runtime()
            .block_on(core.load_local_model("tiny@my-box".into()))
            .expect_err("the revived backend must load its own file, not inherit an engine");
        assert!(err.to_string().contains("exited during load"), "got {err}");
    });
}

/// The same rule, for the two other ways a row stops meaning what it meant:
/// repointing it at another models directory, and disabling it (a backend
/// that may not serve a turn has no business holding gigabytes).
#[test]
fn repointing_or_disabling_a_backend_retires_its_engines() {
    run(|| {
        let (core, dir) = bare_core();
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();

        core.runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &first.display().to_string(),
                None,
                true,
            )))
            .expect("add backend");
        core.test_register_loaded_local_model("my-box", "tiny", 51236);

        core.runtime()
            .block_on(core.update_backend(
                "my-box".into(),
                BackendUpdate {
                    models_dir: Some(Some(second.display().to_string())),
                    ..Default::default()
                },
            ))
            .expect("repoint");
        assert!(
            core.running_engines().is_empty(),
            "an engine started from the old directory may not survive the repoint"
        );

        // An update that leaves the directory alone keeps the engine.
        core.test_register_loaded_local_model("my-box", "tiny", 51237);
        core.runtime()
            .block_on(core.update_backend(
                "my-box".into(),
                BackendUpdate {
                    display_name: Some("My Box".into()),
                    ..Default::default()
                },
            ))
            .expect("rename");
        assert_eq!(core.running_engines().len(), 1, "a rename changes nothing");

        core.runtime()
            .block_on(core.set_backend_enabled("my-box".into(), false))
            .expect("disable");
        assert!(
            core.running_engines().is_empty(),
            "a disabled backend keeps no engines"
        );
    });
}

/// A disabled backend serves nothing and **starts** nothing — the built-in
/// `local` singleton included.
///
/// Disabling retires the backend's engines, and the chat path already refuses
/// a disabled backend. The explicit verb has to hold the same line or the
/// guarantee is decorative: `eidola model load tiny@local` (and the Load
/// button behind it) would start another `llama-server` a moment after the
/// disable stopped one, leaving a disabled backend holding gigabytes.
/// Managing files is a different thing and stays open.
#[test]
fn a_disabled_backend_starts_no_engine_even_when_asked_explicitly() {
    run(|| {
        let (core, dir) = bare_core();
        let models_dir = dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("tiny.gguf"), b"gguf").unwrap();
        // `/usr/bin/false` spawns and exits at once, so a load that got past
        // the gate reports the engine's own failure — which is what
        // distinguishes "refused" from "spawned and died".
        core.set_llama_server_path(Some("/usr/bin/false".into()))
            .unwrap();

        core.runtime()
            .block_on(core.set_backend_enabled("local".into(), false))
            .expect("disable");

        let err = core
            .runtime()
            .block_on(core.load_local_model("tiny@local".into()))
            .expect_err("a disabled backend must not start an engine");
        assert!(matches!(err, AppError::NotConfigured { .. }), "got {err:?}");
        assert!(err.to_string().contains("disabled"), "got {err}");
        assert!(core.running_engines().is_empty(), "nothing was spawned");

        // Re-enabling restores the verb (the load now reaches the engine and
        // fails on its own terms).
        core.runtime()
            .block_on(core.set_backend_enabled("local".into(), true))
            .expect("enable");
        let err = core
            .runtime()
            .block_on(core.load_local_model("tiny@local".into()))
            .expect_err("/usr/bin/false exits immediately");
        assert!(err.to_string().contains("exited during load"), "got {err}");
    });
}

/// A re-added backend starts clean: it does not inherit the error its
/// predecessor's engine left behind.
///
/// Removal is a *soft* remove and re-adding the id revives the row, so a
/// standing engine failure keyed on `(backend, slug)` outlives the backend
/// that earned it and reappears on a configuration that may have nothing to do
/// with it — a different directory, a different `llama-server`. Retiring a
/// backend's engines therefore retires what those engines had to say.
#[test]
fn a_re_added_backend_does_not_inherit_its_predecessors_error() {
    run(|| {
        let (core, dir) = bare_core();
        let models = dir.path().join("box-models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("tiny.gguf"), b"gguf").unwrap();
        let add = || {
            core.runtime().block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &models.display().to_string(),
                // Spawns and exits immediately: a real, honest load failure.
                Some("/usr/bin/false"),
                true,
            )))
        };
        let last_error = || {
            let state = core
                .runtime()
                .block_on(core.local_models_state())
                .expect("state");
            state
                .external
                .iter()
                .find(|b| b.backend_id == "my-box")
                .expect("backend section")
                .models
                .iter()
                .find(|m| m.slug == "tiny")
                .expect("model row")
                .last_error
                .clone()
        };

        add().expect("add");
        core.runtime()
            .block_on(core.load_local_model("tiny@my-box".into()))
            .expect_err("/usr/bin/false exits during load");
        assert!(
            last_error().is_some_and(|e| e.contains("exited during load")),
            "precondition: the failed load is reported"
        );

        core.runtime()
            .block_on(core.remove_backend("my-box".into()))
            .expect("remove");
        add().expect("re-add");
        assert_eq!(
            last_error(),
            None,
            "a backend re-added under the same id starts with no standing error"
        );
    });
}

/// Retiring a backend that has no live engine but *does* have a standing
/// failure still changes what the local-model snapshot shows — so it has to
/// say so on the bus.
///
/// Retirement has two effects: it stops engines and it forgets those engines'
/// reports. A subscriber that refreshes the snapshot on its documented
/// invalidation would otherwise go on rendering an error that is gone.
#[test]
fn retiring_a_backend_that_only_has_a_failure_still_invalidates() {
    run(|| {
        let (core, dir) = bare_core();
        let models = dir.path().join("box-models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("tiny.gguf"), b"gguf").unwrap();
        core.runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &models.display().to_string(),
                // Spawns and exits immediately: a standing load failure, and
                // no engine left behind to be retired.
                Some("/usr/bin/false"),
                true,
            )))
            .expect("add");
        core.runtime()
            .block_on(core.load_local_model("tiny@my-box".into()))
            .expect_err("/usr/bin/false exits during load");
        assert!(
            core.running_engines().is_empty(),
            "precondition: the failed load left no engine — only its report"
        );

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.remove_backend("my-box".into()))
            .expect("remove");

        let emitted = drain(&mut rx);
        assert!(
            emitted.contains(&Change::LocalModels),
            "forgetting the report is a local-models change: {emitted:?}"
        );
    });
}

/// ...and a retirement that changes nothing stays silent. Widening the
/// condition to "either effect happened" must not become "emit always": a
/// spurious invalidation on every disable is its own regression.
#[test]
fn retiring_a_backend_with_nothing_to_retire_emits_nothing() {
    run(|| {
        let (core, dir) = bare_core();
        let models = dir.path().join("box-models");
        std::fs::create_dir_all(&models).unwrap();
        core.runtime()
            .block_on(core.add_backend(llamacpp_backend(
                "my-box",
                &models.display().to_string(),
                None,
                true,
            )))
            .expect("add");

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.set_backend_enabled("my-box".into(), false))
            .expect("disable");

        let emitted = drain(&mut rx);
        assert_eq!(
            emitted,
            vec![Change::Backends],
            "no engine, no report: the registry change is the only news"
        );
    });
}

/// A configuration write and the cleanup that belongs to it are **one**
/// operation, so a newer write cannot land between them.
///
/// Otherwise an older disable's cleanup outlives a newer enable: the disable
/// commits, the enable commits over it, a load registers an engine the *final*
/// configuration authorises — and then the disable's cleanup arrives and stops
/// it. The load has already reported success, and the backend is enabled, so
/// nothing about the end state explains the engine that vanished.
#[test]
fn an_older_disables_cleanup_cannot_retire_an_enabled_backends_engine() {
    run(|| {
        let (core, _dir) = bare_core();
        let core = std::sync::Arc::new(core);
        // Widen the gap between the write and its cleanup; the race is real,
        // this only makes it reachable on purpose.
        core.test_pause_before_backend_cleanup(std::time::Duration::from_millis(600));

        let disable = core.runtime().spawn({
            let core = core.clone();
            async move { core.set_backend_enabled("local".into(), false).await }
        });
        // The disable has committed and is on its way to its cleanup.
        std::thread::sleep(std::time::Duration::from_millis(150));

        // A newer write says the backend is enabled after all...
        core.runtime()
            .block_on(core.set_backend_enabled("local".into(), true))
            .expect("enable");
        // ...and a load registers the engine that configuration authorises.
        core.test_register_loaded_local_model("local", "tiny", 51999);

        core.runtime()
            .block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(5), disable).await
            })
            .expect("the disable must settle")
            .expect("join")
            .expect("disable");

        let backends = core.runtime().block_on(core.list_backends()).expect("list");
        let local = backends
            .iter()
            .find(|b| b.id == "local")
            .expect("local row");
        assert!(local.enabled, "the newer write is the one that stands");
        assert_eq!(
            core.running_engines().len(),
            1,
            "an engine the current configuration authorises may not be retired \
             by an operation the configuration has already moved past"
        );
    });
}
