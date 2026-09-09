//! The local inference proxy, end to end over a real HTTP connection.
//!
//! Everything here speaks HTTP/1.1 through an in-memory duplex into
//! `proxy::http::serve_connection` — no socket, no port, no privileges, which
//! is exactly what the transport-agnostic split is for (`crate::ipc`'s shape).
//! The other end of the app is the chat harness's in-process upstream, so a
//! proxied completion goes through the real route, the real pricing contract
//! and the real credential machinery.
//!
//! What each test is for:
//!
//! - **Authentication is the whole gate.** A proxy with no live keys refuses
//!   everything rather than running open, a wrong key is refused, and a revoked
//!   one stops working. All three answer the same `401`, because which of them
//!   it was is not something a caller holding the wrong key is owed.
//! - **Exposure is a permission, checked before anything opens.** A model on a
//!   backend the reader did not tick is refused by name — ahead of any engine
//!   start and ahead of any credential.
//! - **A proxied turn pays and records, and creates nothing.** The Record gains
//!   the exchange with `action_id` unset; the Library gains no conversation.
//! - **The downstream never sees a credential.** The `refund` Eidola answers
//!   with is wallet material this app consumes; it is stripped on the way out.

mod chat_harness;

use chat_harness::{ChatBehavior, MODEL, MockConfig, RefundMode, core_for, with_account};
use eidola_app_core::AppCore;
use eidola_app_core::ipc::Shutdown;
use eidola_app_core::proxy::{ProxySettingsUpdate, http};
use std::sync::Arc;

/// Run an async test body on a dedicated OS thread — `AppCore` owns its own
/// tokio runtime and dropping it while another is active on the same thread
/// panics. Mirrors `tests/chat_path.rs`.
fn run<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().unwrap();
}

/// One HTTP exchange over an in-memory duplex.
///
/// `Connection: close` is sent so hyper ends the connection after answering and
/// the read below terminates on EOF — the simplest honest client for a test
/// that is about the *answers*, not about keep-alive.
async fn exchange(core: &Arc<AppCore>, request: &str) -> (u16, String) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (client, server) = tokio::io::duplex(1024 * 1024);
    let serving = tokio::spawn(http::serve_connection(
        Arc::clone(core),
        server,
        Shutdown::default(),
    ));

    let (mut reader, mut writer) = tokio::io::split(client);
    writer
        .write_all(request.as_bytes())
        .await
        .expect("write the request");
    writer.flush().await.expect("flush");

    let mut raw = Vec::new();
    reader.read_to_end(&mut raw).await.expect("read the answer");
    let _ = serving.await;

    let text = String::from_utf8_lossy(&raw).to_string();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or_else(|| panic!("no status line in {text:?}"));
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default();
    (status, body)
}

fn get(path: &str, key: Option<&str>) -> String {
    let auth = key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{auth}\r\n")
}

fn post(path: &str, key: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
         Authorization: Bearer {key}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// A body with a downstream `Authorization` and a `traceparent` on the wire —
/// the two headers the doctrine says must never be forwarded.
fn probing_post(path: &str, key: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
         Authorization: Bearer {key}\r\n\
         traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01\r\n\
         tracestate: vendor=whatever\r\n\
         X-Tool-Session: a-linking-primitive\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// Expose the `eidola` backend and mint one key; hand back the key.
fn armed(core: &AppCore) -> String {
    core.runtime().block_on(async {
        core.set_proxy_backend_exposed("eidola".to_string(), true)
            .await
            .expect("expose eidola");
        core.create_proxy_key("a tool".to_string())
            .await
            .expect("mint a key")
            .key
    })
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

#[test]
fn a_proxy_with_no_keys_refuses_everything() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let core = Arc::new(core);
        let runtime = core.runtime();
        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", None)));
        assert_eq!(
            status, 401,
            "\"no keys yet\" and \"everyone welcome\" are not the same state"
        );
        assert!(body.contains("invalid_api_key"), "{body}");

        // And a fabricated key is refused in the same breath and the same
        // words, so nothing here distinguishes a wrong key from no key at all.
        let (status, _) = runtime.block_on(exchange(&core, &get("/v1/models", Some("eid-nope"))));
        assert_eq!(status, 401);
    });
}

#[test]
fn a_revoked_key_stops_working_and_a_live_one_does_not() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, _) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200, "a live key is admitted");

        let id = runtime
            .block_on(core.proxy_keys())
            .expect("keys")
            .first()
            .expect("one key")
            .id
            .clone();
        assert!(
            runtime.block_on(core.revoke_proxy_key(id)).expect("revoke"),
            "the row was live, so revoking it changed something"
        );

        let (status, _) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 401, "the key the reader took away no longer works");
    });
}

#[test]
fn a_keys_first_use_is_announced_and_its_second_is_not() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let runtime = core.runtime();
        let mut rx = core.subscribe_changes();
        // The exposure and the mint above have already been announced.
        while rx.try_recv().is_ok() {}

        assert!(
            runtime
                .block_on(core.authenticate_proxy_key(key.clone()))
                .expect("auth")
        );
        let announced: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|e| e.change)
            .collect();
        assert!(
            announced.contains(&eidola_app_core::changes::Change::Proxy),
            "the pane says \"never used\" until something tells it otherwise: {announced:?}"
        );

        // **And exactly once.** The pane renders used-or-never and nothing
        // finer, so every later authentication moves no rendered value — while
        // announcing each one would put a bus event on every proxied request,
        // costing every window two reads and a listener reconcile.
        assert!(
            runtime
                .block_on(core.authenticate_proxy_key(key))
                .expect("auth")
        );
        let announced: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|e| e.change)
            .collect();
        assert!(
            announced.is_empty(),
            "a key already used announces nothing: {announced:?}"
        );
    });
}

#[test]
fn a_latched_process_starts_no_work_for_a_key_it_would_admit() {
    run(|| {
        // The latch is asked twice — once before the request is looked at, and
        // again on the last line before dispatch, because authentication is a
        // real await the latch can be thrown across. **That interleaving
        // cannot be scheduled from a test**: the stretch from the auth landing
        // to the dispatch has no seam to land on, which is the same honest
        // limit `crates/eidola-gui/src/ipc.rs` records for its own sweep. What
        // *is* testable is the outcome the second ask exists to produce — a
        // latched process serves a perfectly good key nothing at all, and
        // starts no billed work doing it.
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let shutdown = Shutdown::default();
        shutdown.latch();
        let answer = runtime.block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (client, server) = tokio::io::duplex(64 * 1024);
            let serving = tokio::spawn(http::serve_connection(
                Arc::clone(&core),
                server,
                shutdown.clone(),
            ));
            let (mut reader, mut writer) = tokio::io::split(client);
            let request = post(
                "/v1/chat/completions",
                &key,
                &format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#),
            );
            writer.write_all(request.as_bytes()).await.expect("write");
            writer.flush().await.expect("flush");
            let mut raw = Vec::new();
            reader.read_to_end(&mut raw).await.expect("read");
            let _ = serving.await;
            String::from_utf8_lossy(&raw).to_string()
        });
        assert!(answer.contains("503"), "{answer}");
        assert_eq!(
            mock.chat_hits(),
            0,
            "nothing went upstream, so nothing was billed"
        );
    });
}

#[test]
fn the_surface_is_three_routes_and_nothing_else() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        // Everything the app *has* that is not stateless inference has no
        // route here, and that is a capability statement rather than a gap.
        for path in [
            "/v1/spaces",
            "/v1/account",
            "/v1/wallet/credentials",
            "/v1/embeddings",
            "/",
        ] {
            let (status, _) = runtime.block_on(exchange(&core, &get(path, Some(&key))));
            assert_eq!(status, 404, "`{path}` must not be reachable");
        }
    });
}

// ---------------------------------------------------------------------------
// What the proxy offers
// ---------------------------------------------------------------------------

#[test]
fn only_the_backends_the_reader_ticked_are_offered() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        // A key, but nothing exposed yet.
        let key = core
            .runtime()
            .block_on(core.create_proxy_key("a tool".into()))
            .expect("mint")
            .key;
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200);
        let listing: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(
            listing["data"].as_array().map(|a| a.len()),
            Some(0),
            "an empty exposure set offers nothing: {body}"
        );

        // Naming a model anyway is refused by name, ahead of everything.
        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#),
            ),
        ));
        assert_eq!(status, 404, "{body}");
        assert!(body.contains("model_not_found"), "{body}");

        // Ticking the backend is what makes it reachable.
        runtime
            .block_on(core.set_proxy_backend_exposed("eidola".into(), true))
            .expect("expose");
        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200);
        assert!(body.contains(MODEL), "{body}");
    });
}

#[test]
fn only_loaded_means_a_request_cannot_start_an_engine() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = core.runtime().block_on(async {
            core.set_proxy_backend_exposed("local".to_string(), true)
                .await
                .expect("expose local");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);
        let runtime = core.runtime();

        // **The exposure setting is a permission to start an engine**, so it is
        // enforced where the engine would start and not only in the listing: a
        // tool that names a model the listing withheld must not be able to
        // start it anyway.
        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"nothing-loaded@local","messages":[{"role":"user","content":"hi"}]}"#,
            ),
        ));
        assert_eq!(status, 404, "{body}");
        assert!(body.contains("model_not_found"), "{body}");
    });
}

// ---------------------------------------------------------------------------
// A proxied turn
// ---------------------------------------------------------------------------

#[test]
fn a_proxied_completion_pays_records_and_creates_no_conversation() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let spaces_before = runtime
            .block_on(core.list_spaces(true))
            .expect("spaces")
            .len();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &probing_post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}]}}"#
                ),
            ),
        ));
        assert_eq!(status, 200, "{body}");
        let answer: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert!(
            answer["choices"][0]["message"]["content"]
                .as_str()
                .is_some_and(|c| !c.is_empty()),
            "an answer came back: {body}"
        );

        // **The credential artifact never reaches downstream.**
        assert!(
            answer.get("refund").is_none(),
            "a `refund` is wallet material, not part of any OpenAI response: {body}"
        );
        // The model is named as the caller named it.
        assert_eq!(answer["model"], MODEL);

        // **It paid.** The harness mints real credentials, so a completed
        // remote call has taken a hold and settled it.
        assert!(mock.chat_hits() >= 1, "the upstream was actually reached");
        // The blocking answer carries its refund inline, so settlement is read
        // off the wallet rather than off the recovery endpoint: nothing is left
        // `spending`, which is what "the hold settled rather than being
        // abandoned" means.
        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(
            wallet.iter().any(|c| c.state == "spent"),
            "the credential this turn spent settled into its successor: {wallet:?}"
        );
        assert!(
            !wallet.iter().any(|c| c.state == "spending"),
            "and nothing was stranded mid-spend: {wallet:?}"
        );

        // **It recorded.** The exchange is in the Record, attached to no
        // action — a proxied request produces nothing persistable at the
        // semantic layer, and the request row is nullable on `action_id`
        // precisely so it still lands.
        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let completion = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the proxied exchange is in the Record");
        let detail = runtime
            .block_on(core.request_detail(completion.id.clone()))
            .expect("detail")
            .expect("the row");
        assert_eq!(detail.action_id, None, "a proxied request has no action");
        assert_eq!(detail.space_id, None, "and therefore no conversation");
        assert_eq!(detail.backend_id.as_deref(), Some("eidola"));
        assert!(
            detail.credential_nonce.is_some(),
            "the credential it spent is recorded beside it"
        );

        // **And it created nothing.** No space, at the semantic layer or
        // anywhere else.
        assert_eq!(
            runtime
                .block_on(core.list_spaces(true))
                .expect("spaces")
                .len(),
            spaces_before,
            "a proxied request creates no conversation"
        );
    });
}

#[test]
fn no_downstream_header_reaches_the_upstream() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, _) = runtime.block_on(exchange(
            &core,
            &probing_post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],
                        "temperature":0.5,"logit_bias":{{"1":2}},"x-vendor":"anything"}}"#
                ),
            ),
        ));
        assert_eq!(status, 200);

        // The downstream sent `Authorization`, `traceparent`, `tracestate` and
        // a vendor header. What went upstream carries the app's own ACT and
        // nothing of the tool's.
        let auths = mock.chat_auth_values();
        assert_eq!(auths.len(), 1, "one upstream call");
        let sent_auth = auths[0].clone().expect("the ACT this app spends");
        assert!(
            !sent_auth.contains(&key),
            "a downstream key authenticates the tool to the proxy and is never forwarded"
        );

        // And the body is an allowlist too: the fields Eidola accepts travel,
        // the ones it does not are dropped rather than turning one unfamiliar
        // key from an SDK into a 400 for the whole request.
        let bodies = mock.chat_bodies();
        let sent = bodies.first().expect("one body").as_object().expect("obj");
        assert!(sent.contains_key("temperature"), "{sent:?}");
        assert!(!sent.contains_key("logit_bias"), "{sent:?}");
        assert!(!sent.contains_key("x-vendor"), "{sent:?}");
    });
}

#[test]
fn a_streaming_completion_forwards_the_upstreams_own_events() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"stream me"}}],"stream":true}}"#
                ),
            ),
        ));
        assert_eq!(status, 200, "{body}");
        assert!(
            body.contains("data:"),
            "the answer is server-sent events: {body}"
        );
        assert!(
            body.contains("[DONE]"),
            "and it ends where the upstream did"
        );
        assert!(
            !body.contains("refund"),
            "no credential material travels downstream: {body}"
        );
        assert!(mock.refund_hits() >= 1, "the streaming hold settled");

        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        assert!(
            requests.iter().any(|r| r.path == "/v1/chat/completions"),
            "a streamed exchange lands in the Record too"
        );
    });
}

#[test]
fn a_streaming_refund_that_arrives_in_band_is_the_one_that_settles() {
    run(|| {
        // **The one case where the in-band token is the only copy.** The
        // server sends its terminal metadata event carrying the refund even
        // when its own best-effort persistence of that token failed — and that
        // failure is exactly what makes the recovery endpoint unable to
        // answer. `RefundMode::Fail` is that server: recovery 500s, and the
        // event still carries a usable token. Discarding it and asking
        // recovery would strand the credential in `spending` for good.
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::StreamingWithMetadataRefund,
            refund: RefundMode::Fail,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":true}}"#
                ),
            ),
        ));
        assert_eq!(status, 200, "{body}");

        // The metadata event still reaches the tool — it is Eidola's own
        // metadata and a standard client ignores it — but stripped of the one
        // thing that may not travel.
        assert!(
            body.contains("eidola.chat.completion.metadata"),
            "the event itself is forwarded: {body}"
        );
        assert!(
            !body.contains("\"refund\""),
            "no credential material travels downstream: {body}"
        );

        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(
            wallet.iter().any(|c| c.state == "spent"),
            "the in-band token settled the hold: {wallet:?}"
        );
        assert!(
            !wallet.iter().any(|c| c.state == "spending"),
            "nothing is stranded mid-spend: {wallet:?}"
        );
        // And recovery was asked for nothing, because nothing was missing.
        assert_eq!(
            mock.refund_hits(),
            0,
            "recovery is the fallback for an absent token, not the first move"
        );
    });
}

#[test]
fn a_stream_that_carries_no_refund_still_falls_back_to_recovery() {
    run(|| {
        // The other half of the same rule, so the fix is not "always in-band":
        // a stream with no metadata event settles through recovery exactly as
        // it did before.
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, _) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":true}}"#
                ),
            ),
        ));
        assert_eq!(status, 200);
        assert!(
            mock.refund_hits() >= 1,
            "with no token in the stream, recovery is what settles it"
        );
    });
}

#[test]
fn a_successful_answer_that_is_not_json_is_a_gateway_failure() {
    run(|| {
        // A truncated response or an intermediary's HTML page arriving with a
        // `2xx` must not reach a tool as an apparent success carrying a
        // fabricated body.
        let (_mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkNonJsonBody,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#),
            ),
        ));
        assert_eq!(status, 502, "{body}");
        assert!(body.contains("not JSON"), "{body}");

        // The exchange is still in the Record — the whole point of keeping raw
        // bodies is the failure nobody can otherwise diagnose.
        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let completion = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded");
        assert_eq!(completion.response_status, Some(200));
    });
}

// ---------------------------------------------------------------------------
// Refusals a caller has to be able to read
// ---------------------------------------------------------------------------

#[test]
fn a_malformed_request_is_refused_and_the_connection_survives_it() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post("/v1/chat/completions", &key, "not json"),
        ));
        assert_eq!(status, 400, "{body}");
        assert!(body.contains("invalid_request_error"), "{body}");

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post("/v1/chat/completions", &key, r#"{"messages":[]}"#),
        ));
        assert_eq!(status, 400, "a body with no model is unanswerable: {body}");

        // The proxy is still serving.
        let (status, _) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200);
    });
}

#[test]
fn a_latched_shutdown_answers_nothing_more() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        // The abort a socket close performs lands only at an await point, and
        // the stretch from a request arriving to its dispatch passes only
        // awaits that are typically already ready — so the latch is what stops
        // a connection already inside from starting billed work.
        let shutdown = Shutdown::default();
        shutdown.latch();

        let answer = runtime.block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (client, server) = tokio::io::duplex(64 * 1024);
            let serving = tokio::spawn(http::serve_connection(
                Arc::clone(&core),
                server,
                shutdown.clone(),
            ));
            let (mut reader, mut writer) = tokio::io::split(client);
            writer
                .write_all(get("/v1/models", Some(&key)).as_bytes())
                .await
                .expect("write");
            writer.flush().await.expect("flush");
            let mut raw = Vec::new();
            reader.read_to_end(&mut raw).await.expect("read");
            let _ = serving.await;
            String::from_utf8_lossy(&raw).to_string()
        });
        assert!(
            answer.contains("503"),
            "a latched connection says the process is going away: {answer}"
        );
    });
}

// ---------------------------------------------------------------------------
// The settings the pane writes
// ---------------------------------------------------------------------------

#[test]
fn the_settings_round_trip_and_refuse_what_cannot_be_bound() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let runtime = core.runtime();

        let initial = runtime.block_on(core.proxy_settings()).expect("settings");
        assert!(!initial.enabled, "off until the reader turns it on");
        assert!(initial.is_loopback(), "and loopback until they move it");
        assert_eq!(initial.live_key_count, 0);

        let moved = runtime
            .block_on(core.update_proxy_settings(ProxySettingsUpdate {
                enabled: Some(true),
                bind_address: Some("0.0.0.0".into()),
                bind_port: Some(9911),
                ..Default::default()
            }))
            .expect("write");
        assert!(moved.enabled);
        assert_eq!(moved.bind_port, 9911);
        assert!(
            !moved.is_loopback(),
            "the pane's warning band reads exactly this"
        );

        // A refusal leaves zero trace: validation runs before the first write.
        assert!(
            runtime
                .block_on(core.update_proxy_settings(ProxySettingsUpdate {
                    bind_address: Some("localhost".into()),
                    ..Default::default()
                }))
                .is_err(),
            "a name is refused rather than resolved"
        );
        let after = runtime.block_on(core.proxy_settings()).expect("settings");
        assert_eq!(after.bind_address, "0.0.0.0", "nothing moved");
    });
}

#[test]
fn two_settings_writes_each_move_only_their_own_column() {
    run(|| {
        // **Two controls used before the first settles are two writes**, and a
        // read-modify-write of the whole row lets the last one restore the
        // other's old value — the reader watches a switch they flipped flip
        // back. Each write names its own column and nothing else, so the two
        // compose whatever order they land in.
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let runtime = core.runtime();

        let (enabled, exposure) = runtime.block_on(async {
            tokio::join!(
                core.update_proxy_settings(ProxySettingsUpdate {
                    enabled: Some(true),
                    ..Default::default()
                }),
                core.update_proxy_settings(ProxySettingsUpdate {
                    local_exposure: Some(eidola_app_core::proxy::LocalExposure::Downloaded),
                    ..Default::default()
                }),
            )
        });
        enabled.expect("enable");
        exposure.expect("exposure");

        let settled = runtime.block_on(core.proxy_settings()).expect("settings");
        assert!(settled.enabled, "the enable survived the exposure write");
        assert_eq!(
            settled.local_exposure,
            eidola_app_core::proxy::LocalExposure::Downloaded,
            "and the exposure survived the enable"
        );
        assert_eq!(
            settled.bind_address, "127.0.0.1",
            "a column nothing named keeps its value"
        );
    });
}

#[test]
fn a_key_is_shown_once_and_stored_only_as_a_digest() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let runtime = core.runtime();

        let minted = runtime
            .block_on(core.create_proxy_key("my editor".into()))
            .expect("mint");
        assert!(minted.key.starts_with(eidola_app_core::proxy::KEY_PREFIX));
        assert!(minted.key.starts_with(&minted.info.prefix));

        let listed = runtime.block_on(core.proxy_keys()).expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].label, "my editor");
        assert_eq!(
            listed[0].prefix, minted.info.prefix,
            "enough to tell rows apart"
        );
        assert!(
            listed[0].prefix.len() < minted.key.len(),
            "and never the whole key — there is no second chance to show it"
        );

        // The digest is what authenticates, so the key itself still works
        // while nothing on disk can reproduce it.
        assert!(
            runtime
                .block_on(core.authenticate_proxy_key(minted.key.clone()))
                .expect("auth")
        );
        assert!(
            !runtime
                .block_on(core.authenticate_proxy_key(minted.info.prefix.clone()))
                .expect("auth"),
            "what the pane shows is not a key"
        );

        // A key needs a name, so a reader can tell later which tool holds it.
        assert!(
            runtime
                .block_on(core.create_proxy_key("  ".into()))
                .is_err(),
            "a blank name is refused"
        );
    });
}

/// REGRESSION: **a stream that never opened can still carry its refund.**
///
/// The server spends the credential before it dispatches, so a streaming
/// request that fails after the nullifier is recorded — request validation,
/// `send_stream`, a spend-proof re-encode — answers with a refund-bearing JSON
/// error body rather than an SSE stream. Persisting that token for recovery is
/// best-effort there, so when it fails the in-band copy is the only one: this
/// branch passed `None` to settlement, recovery answered nothing, and the
/// credential stayed `spending` for good.
///
/// Third door, one rule: the refund the server *hands* us settles, and
/// recovery is what absence falls back to. `RefundMode::Fail` is the server
/// whose persistence failed, so a passing test cannot be recovery in disguise.
#[test]
fn a_pre_stream_failure_still_settles_from_the_refund_it_carried() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::Non2xxWithRefund(503),
            refund: RefundMode::Fail,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":true}}"#
                ),
            ),
        ));
        assert_eq!(status, 503, "the upstream's own status is passed through");
        assert!(
            !body.contains("\"refund\""),
            "no credential material travels downstream, on this arm either: {body}"
        );

        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(
            wallet.iter().any(|c| c.state == "spent"),
            "the token in the error body settled the hold: {wallet:?}"
        );
        assert!(
            !wallet.iter().any(|c| c.state == "spending"),
            "a failed stream must not strand a spent credential: {wallet:?}"
        );
        assert_eq!(
            mock.refund_hits(),
            0,
            "recovery is the fallback for an absent token, not the first move"
        );
    });
}

/// REGRESSION: **the Record keeps a bounded body, and says when it did.**
///
/// Every upstream chunk was retained until the stream ended so it could be
/// written to a `request` row. An authenticated caller names its own ceiling
/// against whatever backend is exposed, so one request could cost this process
/// the whole answer in memory and then the same bytes again in the database.
///
/// Capping alone would be the worse bug: a Record row holding the first
/// megabyte of a larger answer and claiming to be whole is a trail that lies,
/// which is the one thing it may never be. So the seal states both numbers, in
/// the payload, in a form no upstream sends by accident — and the delivery
/// downstream is untouched, because what the cap bounds is retention.
#[test]
fn an_oversized_answer_is_recorded_as_the_truncation_it_is() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkStreamingOversized,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":true}}"#
                ),
            ),
        ));
        assert_eq!(status, 200, "the answer is delivered in full");
        assert!(
            body.len() > 1_200_000,
            "delivery is not what the cap bounds: {} bytes",
            body.len()
        );

        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let row = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is in the Record");
        let detail = runtime
            .block_on(core.request_detail(row.id.clone()))
            .expect("detail")
            .expect("the row");
        let recorded = detail.response_body.expect("a recorded body");
        assert!(
            recorded.len() < 1_200_000,
            "what is retained is bounded: {} bytes",
            recorded.len()
        );
        let text = String::from_utf8_lossy(&recorded);
        assert!(
            text.contains("this Record entry keeps the first"),
            "and a partial says it is one: {}",
            &text[text.len().saturating_sub(300)..]
        );
    });
}

/// REGRESSION: **the registry is authoritative for engine membership; the scan
/// only decorates.**
///
/// `lease_engine` reads the in-memory registry and never touches the
/// filesystem, so an engine whose backing `.gguf` was renamed or deleted — or
/// whose model directory stopped being readable — stays ready and stays
/// serviceable: `open_proxy_route` leases it before the exposure guard is even
/// consulted. `backend_models` derives its candidates from a *directory scan*,
/// so the model was missing from `/v1/models` while the proxy went on answering
/// requests for it. A capability statement that hides a capability is the one
/// thing this surface must not be.
///
/// The fixture is exactly that state and nothing else: a ready engine in the
/// registry, an empty models directory, `Loaded` exposure.
#[test]
fn a_ready_engine_whose_file_vanished_is_still_offered() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = core.runtime().block_on(async {
            core.set_proxy_backend_exposed("local".to_string(), true)
                .await
                .expect("expose local");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        // A ready engine the scan can never find — there is no file behind it,
        // which is precisely the state the status menu already names
        // "(file missing)" and the route already serves.
        core.test_register_loaded_local_model("local", "orphaned", 5199);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200, "{body}");
        assert!(
            body.contains("orphaned@local"),
            "a model the proxy will serve has to be in its listing: {body}"
        );

        // And nothing else was invented: an engine-backed backend under
        // `Loaded` offers what is running and no more.
        let listed = body.matches("\"id\"").count();
        assert_eq!(listed, 1, "one ready engine, one entry: {body}");
    });
}
