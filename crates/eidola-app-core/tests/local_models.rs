//! Integration tests for the local-inference domain: model downloads (over a
//! wiremock HTTP fixture), deletion, and — via the chat harness — routing
//! `<slug>@local` turns to a loopback engine with **no credential spend, no
//! ACT header, and no Wallet emissions**. The engine itself is faked through
//! the `test_register_loaded_local_model` seam; the real `llama-server`
//! subprocess lifecycle is exercised manually / end-to-end (it needs the
//! binary and a multi-GB model, neither of which belongs in CI).

mod chat_harness;

use chat_harness::{ChatBehavior, MockConfig, with_account};
use eidola_app_core::changes::{Change, ChangeEvent};
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatStreamEvent, LocalModelStatus};

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// Drain all currently-available bus messages (non-blocking).
fn drain(rx: &mut tokio::sync::broadcast::Receiver<ChangeEvent>) -> Vec<Change> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(c) => out.push(c.change),
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
        AppCore::with_test_http_client(config_dir, data_dir, client).expect("open core"),
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

/// A direct model link's query is its authorization — signed S3/CDN links
/// carry `?token=…` — so the request must go out with it. Only the *path*
/// decides what the file is called and which transfer it identifies, which is
/// why the query can be stripped for inspection and still ride along to the
/// fetch.
#[test]
fn a_signed_download_url_is_fetched_with_its_authorization() {
    run(|| {
        let (core, dir) = bare_core();
        let mock = core.runtime().block_on(async {
            let mock = wiremock::MockServer::start().await;
            // The object is served *only* to a request that presents the
            // token; anything else is the 403 a stripped signature earns.
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .and(wiremock::matchers::path("/models/signed-model.gguf"))
                .and(wiremock::matchers::query_param("token", "s3cret"))
                .respond_with(
                    wiremock::ResponseTemplate::new(200).set_body_bytes(vec![0x47u8; 2048]),
                )
                .mount(&mock)
                .await;
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .respond_with(wiremock::ResponseTemplate::new(403))
                .mount(&mock)
                .await;
            mock
        });

        let url = format!("{}/models/signed-model.gguf?token=s3cret", mock.uri());
        core.runtime()
            .block_on(core.download_local_model(url))
            .expect("download starts");

        let state = wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "signed-model" && m.status == LocalModelStatus::Available)
        });
        let model = state
            .models
            .iter()
            .find(|m| m.slug == "signed-model")
            .unwrap();
        assert_eq!(model.last_error, None, "the signed URL downloads");
        assert_eq!(model.size_bytes, Some(2048));
        // What is recorded beside the file is provenance, not a credential:
        // the token is short-lived, says nothing about where the model came
        // from, and has no business on disk.
        assert_eq!(
            model.source_url.as_deref(),
            Some(format!("{}/models/signed-model.gguf", mock.uri()).as_str()),
            "the sidecar records the URL without its query"
        );
        let sidecar = dir.path().join("data/models/signed-model.gguf.meta.json");
        let written = std::fs::read_to_string(&sidecar).expect("sidecar");
        assert!(
            !written.contains("s3cret"),
            "no credential may be written beside the model: {written}"
        );
    });
}

/// A server that answers with a `Content-Length` it never satisfies: headers,
/// a first chunk, then silence with the socket held open — the shape of a
/// transfer that dies mid-response. Returns its port.
fn stalling_download_server(rt: &tokio::runtime::Runtime) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    rt.spawn(async move {
        let mut held = Vec::new();
        while let Ok((mut sock, _)) = listener.accept().await {
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf).await;
            let _ = sock
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\n\r\n")
                .await;
            let _ = sock.write_all(&[0x47u8; 1024]).await;
            let _ = sock.flush().await;
            // Held, and silent from here.
            held.push(sock);
        }
    });
    port
}

/// Cancel must reach a transfer that is blocked on the network.
///
/// The shared HTTP client sets no request timeout, so a server that stops
/// talking mid-response leaves the read pending indefinitely. A cancellation
/// that is only *checked between chunks* never runs: the partial file stays,
/// the entry stays in the download map, and every retry is refused as already
/// downloading — the transfer becomes unstoppable and unrepeatable at once.
#[test]
fn cancelling_a_stalled_download_interrupts_the_blocked_read() {
    run(|| {
        let (core, dir) = bare_core();
        let port = stalling_download_server(core.runtime());
        let url = format!("http://127.0.0.1:{port}/stalled.gguf");

        core.runtime()
            .block_on(core.download_local_model(url.clone()))
            .expect("download starts");
        // Bytes have arrived and the next read is blocked on a silent server.
        wait_for_state(&core, |s| {
            s.models.iter().any(|m| {
                m.slug == "stalled"
                    && matches!(m.status, LocalModelStatus::Downloading { received, .. }
                        if received > 0)
            })
        });

        core.runtime()
            .block_on(core.cancel_local_model_download("stalled@local".into()))
            .expect("cancel");

        // The transfer actually ends: no row, no partial file, and no
        // failure — a cancellation is not an error.
        wait_for_state(&core, |s| !s.models.iter().any(|m| m.slug == "stalled"));
        assert!(
            !dir.path().join("data/models/stalled.gguf.part").exists(),
            "the partial file is removed"
        );

        // And the slug is free again, so the user can try once more.
        core.runtime()
            .block_on(core.download_local_model(url))
            .expect("a cancelled download can be retried");
        core.runtime()
            .block_on(core.cancel_local_model_download("stalled@local".into()))
            .expect("cancel the retry too");
    });
}

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
        assert_eq!(id, "tiny-test-model@local");

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
        assert_eq!(model.id, "tiny-test-model@local");
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

/// A failed download leaves a row standing for an error, not for a file:
/// the listing must keep it (the error is the point) while the *selectable*
/// surface must not offer an id whose only possible outcome is a load
/// failure. Its status is `Available` — the same status a scanned idle file
/// carries — so `on_disk` is what has to be read.
#[test]
fn a_failed_download_is_listed_but_never_selectable() {
    run(|| {
        let (core, _dir) = bare_core();

        let mock = core.runtime().block_on(async {
            let mock = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .respond_with(wiremock::ResponseTemplate::new(500))
                .mount(&mock)
                .await;
            mock
        });

        core.runtime()
            .block_on(core.download_local_model(format!("{}/ghost.gguf", mock.uri())))
            .expect("download starts");

        let state = wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "ghost" && m.last_error.is_some())
        });
        let row = state.models.iter().find(|m| m.slug == "ghost").unwrap();
        assert!(!row.on_disk, "nothing was ever written for this row");
        assert_eq!(
            row.status,
            LocalModelStatus::Available,
            "status alone would read as a usable file"
        );

        let offered = core
            .runtime()
            .block_on(core.backend_models("local".into()))
            .expect("local backend lists");
        assert!(
            !offered.iter().any(|m| m.id == "ghost@local"),
            "a row with no file behind it must not be offered: {offered:?}"
        );
    });
}

/// The row a failed download leaves behind is the *only* thing that remembers
/// where it was fetching from, so it has to carry it: with no file and no URL
/// the surface showing it has nothing to offer but the error text. And
/// dismissing the report is its own verb — it forgets the failure, and the
/// synthesized row goes with it.
#[test]
fn a_failed_download_remembers_its_url_and_can_be_dismissed() {
    run(|| {
        let (core, _dir) = bare_core();

        let mock = core.runtime().block_on(async {
            let mock = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .respond_with(wiremock::ResponseTemplate::new(500))
                .mount(&mock)
                .await;
            mock
        });
        let url = format!("{}/wisp.gguf", mock.uri());

        core.runtime()
            .block_on(core.download_local_model(url.clone()))
            .expect("download starts");

        let state = wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "wisp" && m.last_error.is_some())
        });
        let row = state.models.iter().find(|m| m.slug == "wisp").unwrap();
        assert!(!row.on_disk, "nothing was ever written for this row");
        assert_eq!(
            row.source_url.as_deref(),
            Some(url.as_str()),
            "the row must carry what a retry re-runs"
        );

        // Dismissing forgets the report; the row was only ever the report.
        core.runtime()
            .block_on(core.dismiss_local_model_failure("wisp@local".into()))
            .expect("dismiss");
        let state = wait_for_state(&core, |s| !s.models.iter().any(|m| m.slug == "wisp"));
        assert!(
            !state.models.iter().any(|m| m.slug == "wisp"),
            "the dismissed failure left a row behind: {:?}",
            state.models
        );

        // Idempotent: a second window dismissing the same report is not an
        // error.
        core.runtime()
            .block_on(core.dismiss_local_model_failure("wisp@local".into()))
            .expect("dismissing nothing is not a failure");
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
            .block_on(core.delete_local_model("doomed@local".into()))
            .expect("delete");

        assert!(!models_dir.join("doomed.gguf").exists());
        assert!(!models_dir.join("doomed.gguf.meta.json").exists());
        let changes = drain(&mut rx);
        assert_eq!(changes, vec![Change::LocalModels]);

        // Deleting a loaded model is refused.
        std::fs::write(models_dir.join("held.gguf"), b"gguf").unwrap();
        core.test_register_loaded_local_model("local", "held", 1);
        let err = core
            .runtime()
            .block_on(core.delete_local_model("held".into()))
            .expect_err("refuse while loaded");
        assert!(err.to_string().contains("unload"), "got {err}");
    });
}

/// The scan accepts any case variant of `.gguf`, so the slug→file mapping
/// must find the actual file rather than synthesizing a lowercase
/// extension — on a case-sensitive filesystem `Mixed.GgUf` would otherwise
/// be advertised but never loadable or deletable (codex finding, PR #216).
#[test]
fn mixed_case_extension_scans_and_deletes() {
    run(|| {
        let (core, dir) = bare_core();
        let models_dir = dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("Mixed.GgUf"), b"gguf").unwrap();

        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        let model = state
            .models
            .iter()
            .find(|m| m.slug == "Mixed")
            .expect("mixed-case file is scanned under its stripped slug");
        assert_eq!(model.id, "Mixed@local");
        assert_eq!(model.file_name, "Mixed.GgUf");

        core.runtime()
            .block_on(core.delete_local_model("Mixed@local".into()))
            .expect("delete resolves the actual file name");
        assert!(!models_dir.join("Mixed.GgUf").exists());
    });
}

// ===========================================================================
// Pinning + the on-demand load's eviction pass
// ===========================================================================

const GIB: u64 = 1 << 30;

#[test]
fn pin_state_round_trips_and_requires_a_loaded_engine() {
    run(|| {
        let (core, _dir) = bare_core();
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("tiny.gguf"), b"gguf").unwrap();
        core.test_register_loaded_local_model("local", "tiny", 1);
        let mut rx = core.subscribe_changes();

        core.runtime()
            .block_on(core.set_local_model_pinned("tiny@local".into(), true))
            .expect("pin");
        assert_eq!(drain(&mut rx), vec![Change::LocalModels]);
        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        assert!(matches!(
            state.models[0].status,
            eidola_app_core::LocalModelStatus::Loaded { pinned: true, .. }
        ));

        core.runtime()
            .block_on(core.set_local_model_pinned("tiny@local".into(), false))
            .expect("unpin");
        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        assert!(matches!(
            state.models[0].status,
            eidola_app_core::LocalModelStatus::Loaded { pinned: false, .. }
        ));

        // Pinning an engine that isn't loaded is refused honestly.
        let err = core
            .runtime()
            .block_on(core.set_local_model_pinned("absent@local".into(), true))
            .expect_err("must refuse");
        assert!(err.to_string().contains("not loaded"), "got {err}");
    });
}

/// The on-demand load evicts LRU idle engines to make room — and never
/// touches pinned ones. The load itself is pointed at `/usr/bin/false`
/// (spawns, exits immediately) so the test exercises the eviction pass on
/// any machine without a real llama-server; the load's failure is expected
/// and honest.
#[test]
fn on_demand_load_evicts_lru_idle_but_not_pinned() {
    run(|| {
        let (core, _dir) = bare_core();
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        // A "5 GiB" model by declared footprint; the file itself is tiny,
        // so pin the budget + register fixtures with explicit footprints.
        std::fs::write(models_dir.join("wanted.gguf"), vec![0u8; 64]).unwrap();
        // Fixture files so the scan lists the fake engines' models.
        std::fs::write(models_dir.join("pinned-old.gguf"), b"gguf").unwrap();
        std::fs::write(models_dir.join("idle-new.gguf"), b"gguf").unwrap();
        core.set_llama_server_path(Some("/usr/bin/false".into()))
            .unwrap();
        core.test_set_memory_budget(8 * GIB);
        // Loaded pool: an old pinned engine (4 GiB) + a newer idle one
        // (3 GiB). The new model (~1 GiB overhead + tiny file) needs the
        // idle engine's memory; the pinned one must survive despite being
        // the better LRU candidate.
        core.test_register_engine("local", "pinned-old", 1, 4 * GIB, true, 100);
        core.test_register_engine("local", "idle-new", 2, 3 * GIB, false, 900);

        let err = core
            .runtime()
            .block_on(core.load_local_model("wanted@local".into()))
            .expect_err("/usr/bin/false exits — the load fails after eviction");
        assert!(err.to_string().contains("exited during load"), "got {err}");

        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        let status_of = |slug: &str| {
            state
                .models
                .iter()
                .find(|m| m.slug == slug)
                .map(|m| m.status.clone())
        };
        // The idle engine was LRU-evicted (back to Available); the pinned
        // one survived despite being older.
        assert!(
            matches!(
                status_of("idle-new"),
                Some(eidola_app_core::LocalModelStatus::Available)
            ),
            "the idle engine must have been LRU-evicted: {state:?}"
        );
        assert!(
            matches!(
                status_of("pinned-old"),
                Some(eidola_app_core::LocalModelStatus::Loaded { pinned: true, .. })
            ),
            "the pinned engine must survive eviction: {state:?}"
        );
    });
}

/// When even evicting every idle engine can't make room, the load is
/// refused *without unloading anything* — a pointless eviction would
/// punish the user twice.
#[test]
fn on_demand_load_refuses_without_evicting_when_pins_hold_the_memory() {
    run(|| {
        let (core, _dir) = bare_core();
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("wanted.gguf"), vec![0u8; 64]).unwrap();
        std::fs::write(models_dir.join("idle-small.gguf"), b"gguf").unwrap();
        core.set_llama_server_path(Some("/usr/bin/false".into()))
            .unwrap();
        core.test_set_memory_budget(8 * GIB);
        // 7.5 GiB pinned + idle 0.2 GiB: evicting the idle engine still
        // can't fit the ~1 GiB requirement.
        core.test_register_engine("local", "pinned-big", 1, 7 * GIB + GIB / 2, true, 100);
        core.test_register_engine("local", "idle-small", 2, GIB / 5, false, 900);

        let err = core
            .runtime()
            .block_on(core.load_local_model("wanted@local".into()))
            .expect_err("must refuse");
        assert!(err.to_string().contains("pinned or in-use"), "got {err}");
        // Nothing was unloaded: the idle engine still serves.
        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        assert!(
            state.models.iter().any(|m| m.slug == "idle-small"
                && matches!(m.status, eidola_app_core::LocalModelStatus::Loaded { .. })),
            "a refused load must not evict anyone: {state:?}"
        );
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
        core.test_register_loaded_local_model("local", "test-model", mock.port());
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("Hello local".into(), "test-model@local".into(), None))
            .expect("local chat succeeds");

        assert_eq!(result.content, "Hello from the mock.");
        assert_eq!(result.model, "test-model@local");
        assert_eq!(result.credits_charged, 0);

        // The upstream request carried no Authorization header and the
        // refund endpoint was never consulted.
        assert_eq!(mock.chat_hits(), 1);
        assert_eq!(mock.chat_auths(), vec![false]);
        assert_eq!(mock.refund_hits(), 0);
        // The request body carried the local model id.
        assert_eq!(
            mock.chat_bodies()[0]["model"],
            serde_json::json!("test-model@local")
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
        assert_eq!(inference.model.as_deref(), Some("test-model@local"));
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
        core.test_register_loaded_local_model("local", "test-model", mock.port());
        let mut rx = core.subscribe_changes();

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let result = core
            .runtime()
            .block_on(core.chat_stream("Stream local".into(), "test-model@local".into(), None, tx))
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
fn local_chat_with_missing_model_is_typed_error() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        with_account(&core);

        // A request against an unloaded model *auto-loads* it; here the
        // model file doesn't exist, so the load itself fails honestly.
        let err = core
            .runtime()
            .block_on(core.chat("Hi".into(), "never-loaded@local".into(), None))
            .expect_err("must fail");
        assert!(matches!(err, AppError::LocalModel { .. }), "got {err:?}");
        assert!(err.to_string().contains("no model file"), "got {err}");

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

/// Quit-time teardown must reach every **live** engine, including one the
/// model snapshot cannot see.
///
/// `local_models_state` is reconstructed by *scanning* the model directories
/// and consulting the engine map only to decorate a `.gguf` it already found
/// — so an engine whose backing file was renamed or deleted mid-session is
/// absent from that snapshot while its subprocess is still running. A
/// teardown written as a loop over the snapshot would silently leave it
/// behind, which on macOS means orphaning it to launchd for good.
#[test]
fn shutdown_reaches_an_engine_the_model_scan_cannot_see() {
    run(|| {
        let (core, _dir) = bare_core();
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        // A loaded engine whose file was removed after it started. No
        // `.gguf` is written at all — the same state a rename produces.
        core.test_register_engine("local", "ghost", 1, GIB, false, 100);

        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        assert!(
            !state.models.iter().any(|m| m.slug == "ghost"),
            "precondition: the scan-based snapshot cannot see this engine"
        );

        assert_eq!(
            core.shutdown_engines(),
            1,
            "the registry-based teardown signals it anyway"
        );
        assert_eq!(
            core.shutdown_engines(),
            0,
            "the registry is drained, so teardown is idempotent"
        );
    });
}

/// The read-only sibling of the test above, and the same defect class.
///
/// Any surface that *reports* running engines has the problem the teardown
/// had: `local_models_state` is a directory scan, so an engine whose backing
/// `.gguf` is gone reads as "nothing running" while the subprocess holds its
/// memory. `running_engines` is the registry read that answers honestly, and
/// it carries enough identity (`id`, `slug`, `port`) to name the engine even
/// though the display name died with the sidecar beside the file.
#[test]
fn running_engines_reports_an_engine_the_model_scan_cannot_see() {
    run(|| {
        let (core, _dir) = bare_core();
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        core.test_register_engine("local", "ghost", 4321, GIB, false, 100);

        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        assert!(
            !state.models.iter().any(|m| m.slug == "ghost"),
            "precondition: the scan-based snapshot cannot see this engine"
        );

        let running = core.running_engines();
        assert_eq!(running.len(), 1, "the registry sees it");
        let engine = &running[0];
        assert_eq!(engine.id, "ghost@local", "the join key against the listing");
        assert_eq!(engine.slug, "ghost", "the name that survives the file");
        assert_eq!(engine.backend_id, "local");
        assert_eq!(engine.port, 4321);
        assert!(engine.ready);

        // And it is genuinely read-only: asking twice changes nothing, unlike
        // the draining teardown next door.
        assert_eq!(core.running_engines().len(), 1);
        assert_eq!(core.shutdown_engines(), 1);
        assert!(core.running_engines().is_empty(), "drained");
    });
}

/// The registry read is ordered, because the menu it feeds is re-read every
/// time the user opens it. A `HashMap`'s iteration order changes between
/// runs, which would reshuffle a readout that had not changed.
#[test]
fn running_engines_is_ordered_by_id() {
    run(|| {
        let (core, _dir) = bare_core();
        for slug in ["zeta", "alpha", "mid"] {
            core.test_register_engine("local", slug, 1, GIB, false, 100);
        }
        core.test_register_engine("mine", "alpha", 1, GIB, false, 100);

        let ids: Vec<String> = core.running_engines().into_iter().map(|e| e.id).collect();
        assert_eq!(
            ids,
            vec!["alpha@local", "alpha@mine", "mid@local", "zeta@local"],
            "stable across runs, and the backend disambiguates a shared name"
        );
    });
}

/// The quit-time drain must stay **silent** on the invalidation bus.
///
/// Every other engine transition emits `Change::LocalModels` because
/// something is watching. Here nothing is: the process is exiting. Worse,
/// the GUI's app-lifetime bus bridge is a foreground task gpui keeps driving
/// through its bounded shutdown block — *after* `App::shutdown` sets
/// `quitting` — so a dispatch would reach `LocalModelsStore::refresh` →
/// `cx.spawn` → gpui's "Can't spawn on main thread after on_app_quit" panic,
/// turning a clean quit into a crash on exactly the machines that have an
/// engine loaded.
#[test]
fn the_shutdown_drain_emits_nothing() {
    run(|| {
        let (core, _dir) = bare_core();
        core.test_register_engine("local", "loaded", 1, GIB, false, 100);
        let mut rx = core.subscribe_changes();

        assert_eq!(core.shutdown_engines(), 1, "an engine was signalled");
        assert!(
            drain(&mut rx).is_empty(),
            "the drain must not emit — nobody can render it, and the GUI \
             bridge would dispatch it into gpui's quitting state"
        );
    });
}

/// A load that arrives after the quit-time drain must not spawn.
///
/// `load_local_model` does real async work before it reserves — backend
/// lookup, port pick, `fs::metadata` — so a quit landing mid-load would
/// otherwise let the load resume *past* a drain that saw an empty registry,
/// reserve, and spawn a subprocess into a process about to `exit()`. The
/// shutdown latch is set inside the same lock the reservation takes, so the
/// refusal happens at the write point rather than on captured state.
#[test]
fn a_load_after_the_shutdown_drain_is_refused_and_spawns_nothing() {
    run(|| {
        let (core, _dir) = bare_core();
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("wanted.gguf"), vec![0u8; 64]).unwrap();
        // `/usr/bin/false` exits immediately, so a load that *did* get past
        // the latch would fail with the engine's own message — which is what
        // distinguishes "refused before spawning" from "spawned and died".
        core.set_llama_server_path(Some("/usr/bin/false".into()))
            .unwrap();

        assert_eq!(core.shutdown_engines(), 0, "nothing is loaded yet");

        let err = core
            .runtime()
            .block_on(core.load_local_model("wanted@local".into()))
            .expect_err("a load after the drain must be refused");
        assert!(err.to_string().contains("shutting down"), "got {err}");

        // Nothing was reserved, so nothing is there to tear down or to
        // leave behind as a warming entry.
        assert_eq!(core.shutdown_engines(), 0);
        let state = core
            .runtime()
            .block_on(core.local_models_state())
            .expect("state");
        let wanted = state
            .models
            .iter()
            .find(|m| m.slug == "wanted")
            .expect("the file is still on disk");
        assert!(
            matches!(wanted.status, LocalModelStatus::Available),
            "no engine entry was created: {:?}",
            wanted.status
        );
    });
}

/// A fake `llama-server` that ignores its arguments and stays alive, so a
/// load stays in its warming loop for the duration of a test.
fn write_sleeping_engine(dir: &std::path::Path) -> String {
    let path = dir.join("fake-llama-server");
    std::fs::write(&path, b"#!/bin/sh\nexec sleep 30\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path.to_string_lossy().into_owned()
}

/// The supervisor's own shutdown arm must be silent during a quit, exactly
/// like the drain that signalled it.
///
/// Silencing `shutdown_all_engines` alone left this second emitter wide open:
/// the drain only *signals*, and the supervisor emits right after killing its
/// child — into the same window that panics. The GUI's bus bridge is a
/// foreground task gpui keeps driving through its bounded shutdown block
/// *after* `App::shutdown` sets `quitting`, so that dispatch reaches
/// `cx.spawn` → "Can't spawn on main thread after on_app_quit".
#[test]
fn the_supervisors_shutdown_arm_emits_nothing() {
    run(|| {
        let (core, _dir) = bare_core();
        let core = std::sync::Arc::new(core);
        let models_dir = _dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("sleeper.gguf"), b"gguf").unwrap();
        core.set_llama_server_path(Some(write_sleeping_engine(_dir.path())))
            .unwrap();

        let handle = core.runtime().spawn({
            let core = core.clone();
            async move { core.load_local_model("sleeper@local".into()).await }
        });
        // Wait until the engine is registered and warming.
        wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "sleeper" && matches!(m.status, LocalModelStatus::Loading))
        });

        // Subscribe *after* the load's own emissions, so anything we see is
        // the shutdown path's.
        let mut rx = core.subscribe_changes();
        assert_eq!(
            core.shutdown_engines(),
            1,
            "the warming engine is signalled"
        );

        // Let the supervisor run its shutdown arm to completion.
        let outcome = core
            .runtime()
            .block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(5), handle).await
            })
            .expect("the supervisor must finish promptly")
            .expect("join");
        assert!(outcome.is_err(), "a cancelled load reports failure");

        assert!(
            drain(&mut rx).is_empty(),
            "no path in the supervisor may emit once the shutdown latch is set"
        );
    });
}

/// A warming engine's in-flight `/health` probe must not swallow the
/// shutdown signal.
///
/// The probe used to be awaited *outside* the supervisor's `select!`, so the
/// oneshot was simply not polled for its duration — and warming is exactly
/// when `/health` can hang rather than refuse (the socket is accepted while a
/// multi-gigabyte model loads, and this client sets no request timeout). A
/// quit landing in that window sent its signal into a receiver nobody was
/// watching, the drain's grace expired, `exit()` followed, and the child
/// outlived the process.
///
/// The hang is reproduced deterministically by pointing the injected HTTP
/// client at a proxy that accepts connections and never answers.
/// A core whose HTTP client is pointed at a proxy that accepts connections
/// and never answers, with a sleeping fake engine already configured: every
/// `/health` probe hangs for as long as the test cares to watch.
///
/// The returned runtime owns the proxy and must outlive the core.
fn core_with_a_hanging_health_endpoint(
    dir: &std::path::Path,
) -> (std::sync::Arc<AppCore>, tokio::runtime::Runtime) {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

    // A listener that accepts and never responds. Sockets are held so the
    // connection stays open rather than being closed under the client.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    rt.spawn(async move {
        let mut held = Vec::new();
        while let Ok((sock, _)) = listener.accept().await {
            held.push(sock);
        }
    });

    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::http(format!("http://127.0.0.1:{proxy_port}")).unwrap())
        .build()
        .expect("client");
    let core =
        AppCore::with_test_http_client(dir.to_path_buf(), dir.join("data"), client).expect("core");

    let models_dir = dir.join("data").join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(models_dir.join("sleeper.gguf"), b"gguf").unwrap();
    core.set_llama_server_path(Some(write_sleeping_engine(dir)))
        .unwrap();
    (std::sync::Arc::new(core), rt)
}

#[test]
fn a_hanging_health_probe_does_not_swallow_the_shutdown_signal() {
    run(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let (core, rt) = core_with_a_hanging_health_endpoint(dir.path());

        let handle = core.runtime().spawn({
            let core = core.clone();
            async move { core.load_local_model("sleeper@local".into()).await }
        });
        wait_for_state(&core, |s| {
            s.models
                .iter()
                .any(|m| m.slug == "sleeper" && matches!(m.status, LocalModelStatus::Loading))
        });
        // Give the loop time to enter a probe and hang in it.
        std::thread::sleep(std::time::Duration::from_millis(600));

        assert_eq!(core.shutdown_engines(), 1);
        let outcome = core
            .runtime()
            .block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(5), handle).await
            })
            .expect("the signal must be observed even mid-probe")
            .expect("join");
        assert!(outcome.is_err(), "a cancelled load reports failure");

        drop(core);
        rt.shutdown_background();
    });
}

/// The five-minute readiness budget must bind **each probe**, not merely the
/// gaps between them.
///
/// A `/health` endpoint that accepts the request and never answers is exactly
/// what a warming engine can look like (the socket is accepted while a
/// multi-gigabyte model loads, and this client sets no request timeout). With
/// the deadline checked only at the top of the loop, such a probe left the
/// supervisor waiting on the shutdown signal alone: the documented budget
/// never expired, the load never failed, and every caller joining the warming
/// engine waited for a load that would never end.
#[test]
fn a_hanging_health_probe_still_hits_the_readiness_deadline() {
    run(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let (core, rt) = core_with_a_hanging_health_endpoint(dir.path());
        core.test_set_engine_ready_timeout(std::time::Duration::from_millis(1500));

        let handle = core.runtime().spawn({
            let core = core.clone();
            async move { core.load_local_model("sleeper@local".into()).await }
        });
        let err = core
            .runtime()
            .block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(20), handle).await
            })
            .expect("the readiness deadline must fire even inside a probe")
            .expect("join")
            .expect_err("a load that never became ready must fail");
        assert!(
            err.to_string().contains("did not become ready within 1s"),
            "got {err}"
        );
        assert!(
            core.running_engines().is_empty(),
            "the timed-out load leaves no entry behind"
        );

        drop(core);
        rt.shutdown_background();
    });
}

/// A retirement must bind a load that is already in flight.
///
/// `load_local_model` reads its backend's configuration and then awaits —
/// model-file lookup, engine resolution, port pick, `fs::metadata` — before it
/// reserves. A retirement landing in that window sweeps an engine map the load
/// has not written to yet, so sweeping alone guarantees nothing: the load
/// resumes, registers, and later turns lease an engine spawned from the
/// configuration that was just retired. The load therefore carries the epoch
/// it read its configuration under and the reservation validates it, under the
/// same lock the sweep takes.
///
/// Driven here through disabling the `local` singleton; removal and repointing
/// retire through the very same sweep.
#[test]
fn a_load_in_flight_when_its_backend_is_retired_registers_nothing() {
    run(|| {
        let (core, dir) = bare_core();
        let core = std::sync::Arc::new(core);
        let models_dir = dir.path().join("data").join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("sleeper.gguf"), b"gguf").unwrap();
        // A fake engine that stays alive keeps a registration visible: an
        // engine that exited would tidy itself away and hide the defect.
        core.set_llama_server_path(Some(write_sleeping_engine(dir.path())))
            .unwrap();
        core.test_set_engine_ready_timeout(std::time::Duration::from_millis(1500));
        // Widen the window the retirement has to race; the race itself is
        // real, this only makes it reachable on purpose.
        core.test_pause_before_engine_reserve(std::time::Duration::from_millis(800));

        let handle = core.runtime().spawn({
            let core = core.clone();
            async move { core.load_local_model("sleeper@local".into()).await }
        });
        // The load has read its configuration and is inside the pause.
        std::thread::sleep(std::time::Duration::from_millis(200));
        core.runtime()
            .block_on(core.set_backend_enabled("local".into(), false))
            .expect("disable");

        // From the moment the retirement returns, no engine for that backend
        // may ever appear — including one reserved by a load that read the
        // old configuration.
        for _ in 0..50 {
            let running = core.running_engines();
            assert!(
                running.is_empty(),
                "a retired backend must never acquire an engine: {running:?}"
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        let outcome = core
            .runtime()
            .block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(5), handle).await
            })
            .expect("the load must settle")
            .expect("join");
        let err = outcome.expect_err("the load must be refused, not silently registered");
        assert!(err.to_string().contains("changed while"), "got {err}");
    });
}
