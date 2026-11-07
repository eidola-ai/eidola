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

use chat_harness::{
    ChatBehavior, FLAT_MODEL, FLAT_PRICE, MODEL, MockConfig, RefundMode, STREAM_WIRE_MODEL,
    core_for, with_account,
};
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
        // This is the **door**: a process that had already begun teardown when
        // the request arrived refuses it before authenticating, before reading
        // a body, before anything. The latch thrown *during* one of those is
        // the dispatch check's business and has its own test
        // (`a_latch_thrown_while_the_body_uploads_starts_no_work`); what this
        // pins is that a latched process serves a perfectly good key nothing at
        // all, and starts no billed work doing it.
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

/// REGRESSION: **a flat-priced model is held at its flat price, blocking and
/// streaming.**
///
/// The route copied the catalog's token rates and dropped `per_request`, so a
/// model the server prices per request — zero token rates, the price flat — was
/// refused locally as a zero charge; with nonzero rates it would have held a
/// sum the server's own worst case does not compute and been refused there.
/// The hold is read off the spend proof that reached the wire.
/// The Record's completion rows, newest first.
fn completion_rows(core: &AppCore) -> Vec<eidola_app_core::RequestInfo> {
    core.runtime()
        .block_on(core.list_requests(50, 0))
        .expect("record")
        .into_iter()
        .filter(|r| r.path == "/v1/chat/completions")
        .collect()
}

/// Ask for one completion, blocking and then streaming, and return what each
/// was answered with.
fn ask_both_ways(core: &Arc<AppCore>, key: &str, model: &str) -> [(u16, String); 2] {
    [false, true].map(|stream| {
        core.runtime().block_on(exchange(
            core,
            &post(
                "/v1/chat/completions",
                key,
                &format!(
                    r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"stream":{stream}}}"#
                ),
            ),
        ))
    })
}

/// Every row a refusal before sending must be: one per ask, attached to no
/// transport and no credential, saying it was refused before anything left.
#[track_caller]
fn assert_refused_before_sending(core: &AppCore, asks: usize, model: &str) {
    let rows = completion_rows(core);
    assert_eq!(
        rows.len(),
        asks,
        "one row per refused completion, blocking and streaming alike: {rows:#?}"
    );
    for row in rows {
        assert_eq!(row.response_status, None, "nothing was sent: {row:?}");
        assert_eq!(row.transport, None, "no transport carried it: {row:?}");
        assert_eq!(row.credential_nonce, None, "nothing was spent: {row:?}");
        let error = row.error.clone().unwrap_or_default();
        assert!(
            error.contains("refused before anything was sent"),
            "the row says what happened: {error}"
        );
        let detail = core
            .runtime()
            .block_on(core.request_detail(row.id.clone()))
            .expect("detail")
            .expect("a recorded row");
        let body =
            String::from_utf8_lossy(detail.request_body.as_deref().unwrap_or_default()).to_string();
        assert!(
            body.contains(model),
            "the row keeps what the tool asked for: {body}"
        );
    }
}

/// REGRESSION: **a completion refused before it is sent is still in the
/// Record.**
///
/// Every exit between authentication and the first recorded exchange was a
/// bare `?` — target resolution, the route's opening, the hold — so a tool
/// that asked and was refused left nothing behind, while the Record's own
/// contract is one row per proxied completion. One door now records whatever
/// the preflight refuses; these are its exits from each stage, both
/// transports.
#[test]
fn a_completion_refused_before_sending_is_still_recorded() {
    // Resolution: a backend the reader never exposed.
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        for (status, body) in ask_both_ways(&core, &key, "nothing@local") {
            assert_eq!(status, 404, "{body}");
        }
        assert_refused_before_sending(&core, 2, "nothing@local");
    });

    // The route's opening: an exposed engine backend this proxy may not start.
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
        for (status, body) in ask_both_ways(&core, &key, "nothing-loaded@local") {
            assert_eq!(status, 404, "{body}");
        }
        assert_refused_before_sending(&core, 2, "nothing-loaded@local");
    });

    // The hold: a billed route with no account to spend from.
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        for (status, body) in ask_both_ways(&core, &key, MODEL) {
            assert!(status >= 400, "{status} {body}");
        }
        assert_eq!(mock.chat_hits(), 0, "nothing reached the upstream");
        assert_refused_before_sending(&core, 2, MODEL);
    });
}

#[test]
fn a_flat_priced_model_is_held_at_its_flat_price() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            list_flat_model: true,
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
                    r#"{{"model":"{FLAT_MODEL}","messages":[{{"role":"user","content":"hello"}}]}}"#
                ),
            ),
        ));
        assert_eq!(status, 200, "a flat price is not a zero charge: {body}");
        assert_eq!(
            mock.chat_charges(),
            vec![Some(FLAT_PRICE as u128)],
            "the hold is the flat price, whatever the request holds"
        );
    });
}

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

/// A stream the Eidola server padded (fixed-size frames of comment events,
/// trailing whitespace, an event carried across frames) reaches the local tool
/// with its padding, which every server-sent-events reader ignores and which
/// hides nothing on this machine; it settles from its in-band refund; and the
/// Record keeps its events without the padding, saying how much it left out.
#[test]
fn a_padded_stream_is_forwarded_whole_and_recorded_without_its_padding() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::StreamingPadded,
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
        for piece in ["Hello ", "from the ", "stream.", "data: [DONE]"] {
            assert!(body.contains(piece), "{piece} is forwarded: {body}");
        }
        assert!(body.contains("\n:"), "the padding is forwarded too");
        assert!(
            !body.contains("\"refund\""),
            "no credential travels downstream"
        );

        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(wallet.iter().any(|c| c.state == "spent"), "{wallet:?}");
        assert!(!wallet.iter().any(|c| c.state == "spending"), "{wallet:?}");
        assert_eq!(mock.refund_hits(), 0, "settled in-band");

        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let row = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is in the Record");
        let detail = runtime
            .block_on(core.request_detail(row.id.clone()))
            .expect("detail")
            .expect("the row");
        let text = String::from_utf8(detail.response_body.expect("a recorded body")).unwrap();
        assert!(!text.contains("\n:"), "no padding is kept: {text}");
        for piece in ["Hello ", "from the ", "stream.", "data: [DONE]"] {
            assert!(text.contains(piece), "{piece} is kept: {text}");
        }
        assert!(text.contains("bytes of stream padding"), "{text}");
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
        // **And the refusal is in the row that made it.** The upstream's `200`
        // is the upstream's claim; a row carrying only that shows a request the
        // caller was refused as the success it was not.
        assert!(
            completion
                .error
                .as_deref()
                .is_some_and(|e| e.contains("not JSON")),
            "the row says what this app would not accept: {:?}",
            completion.error
        );
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

/// REGRESSION: **the latch is asked where the spending starts, not where the
/// request arrives.**
///
/// This function grew an `await` between the check and the dispatch twice —
/// authentication, then `Limited::collect()` reading a body a slow client is
/// still uploading — and each time the gap re-opened: a request that had passed
/// every check resumed after teardown began and went on to spend. A third point
/// check would only move the next gap, so the authoritative question is asked
/// inside `completions_response`, on the last line before either dispatch.
///
/// **And this interleaving *is* schedulable**, unlike the authentication one
/// the earlier round could only pin by outcome: the body arrives in two writes
/// and the latch is thrown between them, which is exactly a slow uploader
/// riding into a teardown.
#[test]
fn a_latch_thrown_while_the_body_uploads_starts_no_work() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let shutdown = Shutdown::default();
        let answer = runtime.block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (client, server) = tokio::io::duplex(64 * 1024);
            let serving = tokio::spawn(http::serve_connection(
                Arc::clone(&core),
                server,
                shutdown.clone(),
            ));
            let (mut reader, mut writer) = tokio::io::split(client);

            let body =
                format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#);
            // Headers and the first half of the body: the request is admitted,
            // authenticated, and then parked inside `Limited::collect()`.
            let (head, tail) = body.split_at(body.len() / 2);
            writer
                .write_all(
                    format!(
                        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
                         Connection: close\r\nAuthorization: Bearer {key}\r\n\
                         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{head}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .expect("write the head");
            writer.flush().await.expect("flush");
            // Let the connection reach the body read before teardown begins.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            shutdown.latch();

            writer
                .write_all(tail.as_bytes())
                .await
                .expect("write the tail");
            writer.flush().await.expect("flush");
            let mut raw = Vec::new();
            let _ = reader.read_to_end(&mut raw).await;
            let _ = serving.await;
            String::from_utf8_lossy(&raw).to_string()
        });

        assert!(
            answer.contains("503"),
            "a process that has begun teardown serves nothing, however far the request had got: \
             {answer}"
        );
        assert_eq!(
            mock.chat_hits(),
            0,
            "and it starts no billed work on the way to saying so"
        );
    });
}

/// REGRESSION: **a socket that says nothing must not hold a slot for ever.**
///
/// Admission happens before authentication — the key is in a header nobody has
/// sent yet — so a peer that opens a connection and writes nothing occupies one
/// of the listener's slots with no credential and no request. hyper *has* a
/// thirty-second default here and it is **inert without a timer**: `Time::Empty`
/// turns the default into `None` and logs "timeout has default, but no timer
/// set", so the builder has to be given one.
///
/// The property is that the connection ends on its own. Its slot is then reaped
/// by the listener's own `try_join_next` sweep, which is what lets the next
/// caller in.
#[test]
fn a_connection_that_says_nothing_is_reaped() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let core = Arc::new(core);
        let runtime = core.runtime();

        // The seam moves the number, never the mechanism — the same builder,
        // the same timer, the same arming site.
        http::set_header_read_timeout_for_test(150);
        let ended = runtime.block_on(async {
            let (client, server) = tokio::io::duplex(1024);
            let serving = tokio::spawn(http::serve_connection(
                Arc::clone(&core),
                server,
                Shutdown::default(),
            ));
            // A peer that holds its end open and writes nothing at all.
            let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), serving).await;
            drop(client);
            // **Ended, and ended cleanly.** The deadline is armed only when the
            // builder has been given a timer, and hyper *panics* on a
            // configured timeout with none (`common::time::Time::check`) — a
            // panic inside the connection task also "ends" it, so a test asking
            // no more than "did the future resolve" would pass with the timer
            // removed and every real connection taken down by it.
            matches!(outcome, Ok(Ok(())))
        });
        http::set_header_read_timeout_for_test(0);

        assert!(
            ended,
            "a silent connection must give its slot back — cleanly — rather than hold it against \
             every legitimate client"
        );
    });
}

/// REGRESSION: **a `200` is not an answer until it is the shape that was asked
/// for.**
///
/// Opening downstream commits the response to `200 text/event-stream` and the
/// head cannot be taken back — so a backend that ignored `stream: true` and
/// answered a normal JSON completion, or an intermediary that answered an HTML
/// page, had its body forwarded as an unterminated SSE fragment under a status
/// saying everything went well. That is the blocking transport's malformed-2xx
/// rule read on the other transport, where the wrong shape is *not* JSON.
///
/// **And it is a fifth arm of the refund class**: a backend that ignored the
/// flag may well have answered with a whole completion, refund and all — the
/// credential is spent either way, so the body is read for its token before
/// this fails. `OkBlocking` answers JSON whatever transport asked, which is
/// precisely the fixture.
#[test]
fn a_streamed_ask_answered_with_json_is_a_gateway_failure() {
    run(|| {
        // `RefundMode::Succeed` so the body the backend sent really carries a
        // token — the point being that it is *that* one which settles. With
        // recovery working too, the load-bearing assertion is that it was
        // never asked.
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
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
        assert_eq!(
            status, 502,
            "a client must meet a gateway failure, not an apparent success it cannot parse: {body}"
        );
        assert!(
            !body.contains("data:"),
            "and nothing was framed as server-sent events: {body}"
        );

        // The exchange is still evidence, and the credential still settles from
        // the token that body carried.
        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(
            wallet.iter().any(|c| c.state == "spent"),
            "the completion the backend did send carried the refund: {wallet:?}"
        );
        assert!(
            !wallet.iter().any(|c| c.state == "spending"),
            "nothing is stranded mid-spend: {wallet:?}"
        );
        assert_eq!(
            mock.refund_hits(),
            0,
            "the token the body carried settled the hold; recovery is for its absence"
        );
        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let completion = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("and the exchange is in the Record");
        assert_eq!(completion.response_status, Some(200));
        assert!(
            completion
                .error
                .as_deref()
                .is_some_and(|e| e.contains("server-sent events")),
            "recorded as the refusal it was, not as the 200 the backend claimed: {:?}",
            completion.error
        );
    });
}

/// REGRESSION: **every exit past the hold settles it and records the
/// exchange** — including the one building the request opened.
///
/// Making the header set exact made the build fallible, which put a new early
/// return between `acquire_spend` and the send: a `?` there leaves a credential
/// `spending` with nothing in the Record to say why. The reachable arm is an
/// **external** backend, whose key is user-typed and can carry a value no
/// header may hold; it spends nothing, so what is held here is the other half
/// of the same arm — the refusal is recorded rather than returned bare, which
/// is what proves the exit is taken instead of `?`.
#[test]
fn a_request_that_cannot_be_built_is_still_recorded() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = core.runtime().block_on(async {
            core.add_backend(eidola_app_core::NewBackend {
                id: "acme".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: "Acme".into(),
                base_url: Some("http://127.0.0.1:1".into()),
                // A newline cannot travel in a header value, so the request
                // refuses at the build — after the route is open.
                api_key: Some("bad\nkey".into()),
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            })
            .await
            .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}]}"#,
            ),
        ));
        assert_eq!(
            status, 400,
            "the caller is told what could not be sent: {body}"
        );

        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let refused = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("an exit past the route's opening is recorded, never returned bare");
        assert_eq!(
            refused.response_status, None,
            "nothing was sent, so there is no status to claim"
        );
        assert!(
            refused.error.is_some(),
            "and the row says what happened: {:?}",
            refused.error
        );

        // **No transport is claimed for a request nothing sent** — and the
        // destination it would have used is still on the record, in the row's
        // own words rather than on a connection that was never opened.
        let detail = runtime
            .block_on(core.request_detail(refused.id.clone()))
            .expect("detail")
            .expect("a recorded row");
        assert_eq!(
            detail.base_url, None,
            "a build that failed opened no connection to name"
        );
        assert_eq!(detail.transport, None);
        let error = detail.error.expect("an error");
        assert!(
            error.contains("nothing was sent") && error.contains("http://127.0.0.1:1"),
            "the row still says where it would have gone: {error}"
        );
        assert!(
            !body.contains("127.0.0.1:1"),
            "the caller is not told the reader's configuration: {body}"
        );
    });
}

/// REGRESSION: **a pass-through sends the options the caller sent, and no
/// others.**
///
/// `stream_options: {"include_usage": true}` was added to every streaming
/// request on a route that does not bill — and both zero-spend route kinds
/// carry no pricing, so "does not bill" swept up **external** OpenAI backends
/// alongside engine-backed ones. On a strict OpenAI-compatible server that does
/// not implement the optional field, that rejects every proxied stream; on a
/// lenient one it emits a usage event nobody asked for. Nothing on this path
/// reads `usage` — a billed route settles from the refund the server hands
/// back — so the question was never about billing at all.
///
/// Both halves are asserted, because "send nothing" would be the other bug:
/// the caller who *does* ask still gets the field.
#[test]
fn a_streamed_ask_carries_only_the_options_the_caller_sent() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..Default::default()
        });
        let key = core.runtime().block_on(async {
            core.add_backend(eidola_app_core::NewBackend {
                id: "acme".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: "Acme".into(),
                base_url: Some(mock.base_url.clone()),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            })
            .await
            .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            ),
        ));
        assert_eq!(status, 200, "{body}");
        let sent = mock.chat_bodies();
        let first = sent.first().expect("the upstream saw a request");
        assert!(
            first.get("stream_options").is_none(),
            "an option the caller never sent is not this proxy's to add: {first}"
        );

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}"#,
            ),
        ));
        assert_eq!(status, 200, "{body}");
        let sent = mock.chat_bodies();
        let second = sent.get(1).expect("the upstream saw the second request");
        assert_eq!(
            second.get("stream_options"),
            Some(&serde_json::json!({"include_usage": true})),
            "and the option the caller did send travels: {second}"
        );
    });
}

/// REGRESSION: **exposure is granted to a backend, not to a name.**
///
/// Removal is soft (`request.backend_id` keeps a resolvable target) and
/// `insert_backend` revives a row of the same id with every configuration
/// column overwritten — so an exposure row that outlived the removal made the
/// replacement exposed on arrival, and any holder of a proxy key could send
/// prompts to a destination the reader had never ticked. Invisible in between,
/// too: the listing joins `removed_at IS NULL`, so the standing permission is
/// unseeable for exactly as long as it is unattached.
#[test]
fn a_removed_backend_takes_its_exposure_with_it() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let external = |url: &str, key: Option<&str>| eidola_app_core::NewBackend {
            id: "acme".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: "Acme".into(),
            base_url: Some(url.into()),
            api_key: key.map(str::to_string),
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        };
        core.runtime().block_on(async {
            core.add_backend(external("https://first.example", None))
                .await
                .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            let settings = core.proxy_settings().await.expect("settings");
            assert!(settings.backends.iter().any(|b| b == "acme"));

            core.remove_backend("acme".to_string())
                .await
                .expect("remove");
            let settings = core.proxy_settings().await.expect("settings");
            assert!(
                !settings.exposed_ids.iter().any(|b| b == "acme"),
                "the permission ends with the thing it was about, rather than \
                 standing where nobody can see it: {settings:?}"
            );

            // The same id, a different destination and a different key: the
            // reader ticks it again or nothing reaches it.
            core.add_backend(external("https://second.example", Some("k")))
                .await
                .expect("re-add");
            let settings = core.proxy_settings().await.expect("settings");
            assert!(
                !settings.backends.iter().any(|b| b == "acme"),
                "a replacement is not exposed on arrival — this is what the \
                 route reads: {settings:?}"
            );
            assert!(!settings.exposed_ids.iter().any(|b| b == "acme"));
        });
    });
}

/// REGRESSION: **"and nothing else" includes the headers the *builder* adds.**
///
/// `plain_http_client` installs `User-Agent: eidola-app-core/<version>`, so
/// every proxied completion to a local engine or an external backend carried a
/// version fingerprint outside the enumerated set — while `for_record` showed
/// the reader the enumeration as the whole request. The proxy's routes build a
/// client of their own.
///
/// The core here is an **ordinary** one, deliberately: the harness's injected
/// client would answer for `proxy_client` too, and what is being held is the
/// client the route really builds.
#[test]
fn a_proxied_completion_carries_no_user_agent() {
    run(|| {
        // A one-shot capture server standing in for an external backend.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let sink = Arc::clone(&captured);
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().expect("accept");
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => break,
                }
            }
            let text = String::from_utf8_lossy(&head).to_string();
            // Consume the request body too, or answering and closing over a
            // client still writing resets the connection.
            let length: usize = text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())?
                })
                .unwrap_or(0);
            let mut rest = vec![0u8; length];
            let _ = stream.read_exact(&mut rest);
            *sink.lock().expect("captured head") = text;
            let body = br#"{"id":"x","object":"chat.completion","choices":[]}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(body);
        });

        let dir = tempfile::tempdir().expect("tempdir");
        let core = AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("core");
        let key = core.runtime().block_on(async {
            core.add_backend(eidola_app_core::NewBackend {
                id: "capture".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: "A capture server".into(),
                base_url: Some(format!("http://127.0.0.1:{port}")),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            })
            .await
            .expect("add the capture backend");
            core.set_proxy_backend_exposed("capture".to_string(), true)
                .await
                .expect("expose it");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);

        let (status, body) = core.runtime().block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"m@capture","messages":[{"role":"user","content":"hi"}]}"#,
            ),
        ));
        assert_eq!(status, 200, "{body}");

        let head = captured.lock().expect("captured head").clone();
        let lower = head.to_ascii_lowercase();
        let names: Vec<String> = lower
            .lines()
            .skip(1)
            .filter_map(|line| Some(line.split_once(':')?.0.trim().to_string()))
            .filter(|name| !name.is_empty())
            .collect();
        // Exactly the allowlist for a non-streaming request, plus HTTP's own
        // framing. Asserted as a **set**, because every way this claim was
        // false was a header nobody here wrote: the client's `User-Agent`, the
        // client's `Accept: */*` — which on a streaming request would have sat
        // beside the one the allowlist names — and a second `Content-Type`,
        // since `RequestBuilder::header` appends and `json()` had already set
        // one.
        assert_eq!(
            names,
            vec!["content-type", "accept", "host", "content-length"],
            "the proxy sends the set it enumerates and nothing else: {head}"
        );
        // And the `Accept` is the one the allowlist names, not the `*/*`
        // reqwest inserts where a request states none — the name alone cannot
        // tell the two apart, which is what made the header easy to miss.
        assert!(
            lower.contains("accept: application/json"),
            "the blocking transport says what it will take back: {head}"
        );
    });
}

/// REGRESSION: **one wedged backend must not take the listing with it.**
///
/// `plain_http_client` sets no request timeout, so a backend that accepts the
/// connection and then says nothing left the catalog `await` outstanding for
/// ever: `.ok()` only isolates a future that *resolves*, so `/v1/models` never
/// answered at all and every healthy backend's models went with it. Awaiting
/// them one after another made the endpoint's latency the sum of every
/// backend's besides — the same defect measured in seconds rather than in
/// forever — so they are asked at once, each with its own deadline.
#[test]
fn a_backend_that_never_answers_does_not_take_the_listing_with_it() {
    run(|| {
        // A listener that accepts and then says nothing, for ever.
        let wedged = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let wedged_port = wedged.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            let mut held = Vec::new();
            while let Ok((stream, _)) = wedged.accept() {
                held.push(stream);
            }
        });

        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = core.runtime().block_on(async {
            core.add_backend(eidola_app_core::NewBackend {
                id: "wedged".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: "A wedged server".into(),
                base_url: Some(format!("http://127.0.0.1:{wedged_port}")),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            })
            .await
            .expect("add the wedged backend");
            core.set_proxy_backend_exposed("eidola".to_string(), true)
                .await
                .expect("expose eidola");
            core.set_proxy_backend_exposed("wedged".to_string(), true)
                .await
                .expect("expose the wedged one");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);
        let runtime = core.runtime();

        eidola_app_core::proxy::route::set_model_list_timeout_for_test(300);
        // **The wait is bounded here too**, because the defect's own shape is
        // that the endpoint never answers: without a per-backend deadline this
        // test would hang rather than fail, and a hang says nothing.
        let answered = runtime.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                exchange(&core, &get("/v1/models", Some(&key))),
            )
            .await
        });
        eidola_app_core::proxy::route::set_model_list_timeout_for_test(0);

        let (status, body) = answered.expect(
            "the listing answers on its own deadline rather than waiting out a wedged backend",
        );
        assert_eq!(status, 200, "the listing answers: {body}");
        assert!(
            body.contains(MODEL),
            "a healthy backend's models are not lost to an unhealthy one's silence: {body}"
        );
    });
}

/// REGRESSION: **the catalog deadline belongs to the read, not to the
/// listing.**
///
/// Opening a route reads the catalog again — pricing has to be known before a
/// hold can be computed — and that fetch sat outside every bound: an endpoint
/// that accepted the connection and then said nothing held the completion, and
/// the proxy connection behind it, for ever, without the chat request ever
/// being made.
#[test]
fn a_backend_that_stalls_on_pricing_does_not_hold_the_completion() {
    run(|| {
        // A listener that accepts and then says nothing, for ever.
        let wedged = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let wedged_port = wedged.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            let mut held = Vec::new();
            while let Ok((stream, _)) = wedged.accept() {
                held.push(stream);
            }
        });

        let (_mock, core, _dir) = core_for(MockConfig::default());
        with_account(&core);
        let key = armed(&core);
        core.runtime()
            .block_on(core.set_base_url(format!("http://127.0.0.1:{wedged_port}")))
            .expect("point the eidola backend at the wedged listener");
        let core = Arc::new(core);
        let runtime = core.runtime();

        eidola_app_core::proxy::route::set_model_list_timeout_for_test(300);
        // Bounded here too, because the defect's own shape is a wait that never
        // ends: without the deadline this hangs rather than fails, and a hang
        // says nothing.
        let answered = runtime.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                exchange(
                    &core,
                    &post(
                        "/v1/chat/completions",
                        &key,
                        &format!(
                            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#
                        ),
                    ),
                ),
            )
            .await
        });
        eidola_app_core::proxy::route::set_model_list_timeout_for_test(0);

        let (status, body) = answered
            .expect("the completion answers on the catalog's deadline rather than waiting out a silent endpoint");
        assert_eq!(status, 502, "{body}");
        assert!(
            body.contains("catalog"),
            "and it says what did not answer: {body}"
        );
    });
}

/// REGRESSION: **a proxied prompt does not follow a redirect.**
///
/// reqwest's default policy follows up to ten and replays a cloneable body on
/// `307`/`308`, so an exposed external backend could answer
/// `/v1/chat/completions` with a `Location` pointing anywhere and be handed the
/// whole prompt at an origin the reader never exposed — with the Record still
/// naming the backend that was configured. The `Authorization` is stripped
/// cross-origin by reqwest; the body is not, and the body is what matters here.
#[test]
fn a_proxied_completion_does_not_follow_a_redirect() {
    run(|| {
        // The destination the redirect points at: it records anything it is
        // handed, and must be handed nothing.
        let elsewhere = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let elsewhere_port = elsewhere.local_addr().expect("addr").port();
        let reached = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let saw = Arc::clone(&reached);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = elsewhere.accept() {
                use std::io::{Read, Write};
                saw.store(true, std::sync::atomic::Ordering::SeqCst);
                let mut scratch = [0u8; 1024];
                let _ = stream.read(&mut scratch);
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}");
            }
        });

        // The exposed backend, which answers only with a redirect.
        let backend = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let backend_port = backend.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            while let Ok((mut stream, _)) = backend.accept() {
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => break,
                    }
                }
                let text = String::from_utf8_lossy(&head).to_string();
                let length: usize = text
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                let mut rest = vec![0u8; length];
                let _ = stream.read_exact(&mut rest);
                let _ = write!(
                    stream,
                    "HTTP/1.1 307 Temporary Redirect\r\n\
                     Location: http://127.0.0.1:{elsewhere_port}/v1/chat/completions\r\n\
                     Content-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });

        // An ordinary core: the harness's injected client would answer for the
        // one the route builds, and the policy *is* what is being held.
        let dir = tempfile::tempdir().expect("tempdir");
        let core = AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("core");
        let key = core.runtime().block_on(async {
            core.add_backend(eidola_app_core::NewBackend {
                id: "acme".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: "Acme".into(),
                base_url: Some(format!("http://127.0.0.1:{backend_port}")),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            })
            .await
            .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);

        let (status, body) = core.runtime().block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"m@acme","messages":[{"role":"user","content":"a private prompt"}]}"#,
            ),
        ));
        assert_eq!(
            status, 307,
            "the upstream's own answer reaches the caller rather than being chased: {body}"
        );
        assert!(
            !reached.load(std::sync::atomic::Ordering::SeqCst),
            "and the prompt never left for an origin the reader did not expose"
        );
    });
}

/// REGRESSION: **a body that never finishes arriving does not hold a slot.**
///
/// `Limited::collect()` bounds the byte count and hyper's deadline ends with
/// the head, so a caller that declared a body under the cap and then sent it
/// arbitrarily slowly occupied a connection slot having authenticated and
/// started nothing — every slot, given a key and a loop.
#[test]
fn a_body_that_stops_arriving_ends_the_request() {
    run(|| {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        http::set_body_read_timeout_for_test(200);
        let outcome = runtime.block_on(async {
            let (client, server) = tokio::io::duplex(64 * 1024);
            let serving = tokio::spawn(http::serve_connection(
                Arc::clone(&core),
                server,
                Shutdown::default(),
            ));
            let (mut reader, mut writer) = tokio::io::split(client);
            // A head promising a body, and a body that stops after a few bytes.
            let head = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
                 Authorization: Bearer {key}\r\nContent-Type: application/json\r\n\
                 Content-Length: 4096\r\n\r\n"
            );
            writer.write_all(head.as_bytes()).await.expect("head");
            writer
                .write_all(b"{\"model\":")
                .await
                .expect("a little body");
            writer.flush().await.expect("flush");

            // Bounded, because the defect's shape is a wait that never ends.
            let answered = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                let mut raw = Vec::new();
                reader.read_to_end(&mut raw).await.expect("read the answer");
                raw
            })
            .await;
            // **Aborted, never awaited.** Without the deadline the connection
            // task is still waiting on a body that will never finish, so
            // awaiting it here would turn this test into a hang — and a hang
            // says nothing. The answer above is what is being asserted.
            serving.abort();
            let _ = serving.await;
            answered
        });
        http::set_body_read_timeout_for_test(0);

        let raw =
            outcome.expect("the request ends on its own deadline rather than waiting for ever");
        let text = String::from_utf8_lossy(&raw).to_string();
        assert!(
            text.starts_with("HTTP/1.1 408"),
            "the caller is told its body never arrived: {text}"
        );
    });
}

/// REGRESSION: **the frame accumulator has a ceiling of its own.**
///
/// A stream is bounded in what it retains and in what it may queue for the
/// caller, and neither bounds the buffer events are assembled in: a backend
/// that never terminates an event grows it until the process dies, with the
/// Record's cap and the delivery queue both looking healthy. What passes the
/// ceiling is not an event this app can forward, so the stream ends and the row
/// says why.
#[test]
fn an_event_that_never_ends_is_refused_rather_than_accumulated() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::StreamingUnterminatedFlood,
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
        // The head was committed before a byte of the body arrived, so what the
        // caller meets is a stream that ends — the honest ending for a refusal
        // this app makes mid-stream.
        assert_eq!(status, 200, "{body}");
        assert!(
            !body.contains("xxxx"),
            "and the bytes this app refused are not forwarded either: {}",
            &body[..body.len().min(200)]
        );

        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let completion = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded");
        assert!(
            completion
                .error
                .as_deref()
                .is_some_and(|e| e.contains("ceiling")),
            "the row says the stream ended on this app's ceiling: {:?}",
            completion.error
        );
    });
}

/// REGRESSION: **the frame ceiling holds for an event that arrives whole.**
/// Measured only on the residual after draining, an oversized event whose
/// terminating blank line arrived in the same read was drained whole and
/// forwarded. It is now refused before it is drained, and never forwarded.
#[test]
fn an_oversized_complete_event_is_refused_rather_than_forwarded() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::StreamingOneOversizedEvent,
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
        assert_eq!(
            status, 200,
            "the head was committed before the event arrived"
        );
        assert!(
            !body.contains("zzzz"),
            "the refused event is not forwarded: {} bytes",
            body.len()
        );
        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        let completion = requests
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded");
        assert!(
            completion
                .error
                .as_deref()
                .is_some_and(|e| e.contains("ceiling")),
            "{:?}",
            completion.error
        );
    });
}

/// REGRESSION: **the request column is bounded too, and it is the durable
/// half.**
///
/// A caller may send the whole allowed body and repeat it, and every exchange
/// wrote the reconstructed prompt down in full — a few dozen calls adding a
/// gigabyte to the profile database and its WAL, with nothing pruning `request`
/// rows to take it back. The prompt still travels upstream whole; what is
/// bounded is what is kept, and a row that keeps less says so.
#[test]
fn an_enormous_prompt_is_recorded_as_the_truncation_it_is() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig::default());
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        // Comfortably past the retention cap, and nowhere near the request
        // ceiling the HTTP surface enforces.
        let prompt = "x".repeat(2 * 1024 * 1024);
        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{prompt}"}}]}}"#
                ),
            ),
        ));
        assert_eq!(status, 200, "{}", &body[..body.len().min(200)]);

        // The whole prompt reached the backend: what is bounded is the row.
        let seen = mock
            .chat_bodies()
            .first()
            .expect("the upstream saw a request")
            .to_string();
        assert!(
            seen.len() > 1024 * 1024,
            "the prompt travels upstream whole: {} bytes",
            seen.len()
        );

        let id = runtime
            .block_on(core.list_requests(20, 0))
            .expect("record")
            .iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded")
            .id
            .clone();
        let detail = runtime
            .block_on(core.request_detail(id))
            .expect("detail")
            .expect("a recorded row");
        let recorded = detail.request_body.expect("a request body");
        assert!(
            recorded.len() < seen.len(),
            "the row keeps less than travelled: {} of {} bytes",
            recorded.len(),
            seen.len()
        );
        let text = String::from_utf8_lossy(&recorded);
        assert!(
            text.contains("keeps the first"),
            "and a partial says it is one: {}",
            &text[text.len().saturating_sub(200)..]
        );
        // **And it says which side it truncated.** A shared wording had this
        // column claiming it kept part of a "response" whose remainder "was
        // delivered" — the other side's story told about the bytes this app
        // sent, on the one trail whose value is that it describes itself.
        let tail = &text[text.len().saturating_sub(300)..];
        assert!(
            tail.contains("-byte request"),
            "a truncated request says it is a request: {tail}"
        );
        assert!(
            !tail.contains("sent upstream"),
            "and claims no send — a row is also written for a request never sent: {tail}"
        );
        assert!(
            !tail.contains("response") && !tail.contains("delivered"),
            "and never borrows the response's words: {tail}"
        );
    });
}

/// **A test-only seam is compiled only for tests — the storage as much as the
/// setter.**
///
/// `#[doc(hidden)]` gates documentation and nothing else, so a `pub fn` over a
/// process-global atomic is a live API in every release build: a dependent
/// could move this proxy's admission or catalog deadline at runtime, and an
/// admission deadline set enormous is exactly how a peer that never held a key
/// fills every connection slot. The house rule is the non-default
/// `test-support` feature, where a release build provably contains no path
/// rather than an undocumented one — the rule the attestation seams already
/// obey (`only_a_spawn_can_mint_a_capability` pins the same thing lexically,
/// for the same reason).
///
/// The seams are **enumerated** rather than merely checked: a scan that only
/// tested what it happened to find would pass on a file with none, and the
/// point is that this is a closed set somebody has looked at.
#[test]
fn the_proxys_deadline_seams_are_compiled_only_for_tests() {
    const GATE: &str = "#[cfg(feature = \"test-support\")]";
    let mut found: Vec<String> = Vec::new();
    for (file, source) in [
        ("proxy/http.rs", include_str!("../src/proxy/http.rs")),
        ("proxy/route.rs", include_str!("../src/proxy/route.rs")),
    ] {
        let production = source
            .split_once("\n#[cfg(test)]\nmod tests {")
            .map_or(source, |(before, _)| before);
        let lines: Vec<&str> = production.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            // A seam is either the setter or the override it writes to.
            let name = if let Some(rest) = trimmed.strip_prefix("pub fn ") {
                rest.split('(').next().unwrap_or_default()
            } else if let Some(rest) = trimmed.strip_prefix("static ") {
                rest.split(':').next().unwrap_or_default()
            } else {
                continue;
            };
            if !(name.ends_with("_for_test") || name.ends_with("_TIMEOUT_MS")) {
                continue;
            }
            found.push(format!("{file}::{name}"));
            let gated = lines[..i].iter().rev().take(4).any(|l| l.trim() == GATE);
            assert!(
                gated,
                "{file}::{name} is a documented test-only seam over process-global state and \
                 must carry `{GATE}`, or a release build carries it too"
            );
        }
    }
    assert_eq!(
        found,
        vec![
            "proxy/http.rs::HEADER_READ_TIMEOUT_MS",
            "proxy/http.rs::set_header_read_timeout_for_test",
            "proxy/http.rs::BODY_READ_TIMEOUT_MS",
            "proxy/http.rs::set_body_read_timeout_for_test",
            "proxy/route.rs::MODEL_LIST_TIMEOUT_MS",
            "proxy/route.rs::set_model_list_timeout_for_test",
        ],
        "the seam set is closed: a new one joins this list deliberately, gated"
    );
}

/// **The permission is read with the row it is about, at the moment of use.**
///
/// A settings snapshot names backend *ids*, and an id is not an incarnation:
/// `remove_backend` soft-deletes the row and drops its exposure, and re-adding
/// the same id revives it with a different base URL and a different key, not
/// exposed. So authorizing from the snapshot and then resolving the live row
/// asks two questions about two different things — with a whole request's
/// network and engine latency in between, which is why this stages the gap
/// rather than racing it.
///
/// What the window sees here is the withdrawal by its **effect**, which is what
/// a removal leaves behind: no exposure row for that id. The other half of the
/// incarnation story — that a removal really does take the permission with it,
/// so a revived row comes back unexposed — is `db::exposed_backend`'s own
/// assertion below.
#[test]
fn a_permission_withdrawn_mid_request_is_not_authorized_by_the_snapshot() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let mut window = core.test_open_proxy_resolve_window();
        let asking = {
            let core = Arc::clone(&core);
            let key = key.clone();
            runtime.spawn(async move {
                exchange(
                    &core,
                    &post(
                        "/v1/chat/completions",
                        &key,
                        &format!(
                            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#
                        ),
                    ),
                )
                .await
            })
        };

        let (status, body) = runtime.block_on(async {
            let resume = window.recv().await.expect("the request reaches the window");
            // The reader takes the permission back while the request is in
            // flight, exactly as a remove-and-re-add would.
            core.set_proxy_backend_exposed("eidola".to_string(), false)
                .await
                .expect("withdraw");
            let _ = resume.send(());
            asking.await.expect("the request finishes")
        });

        assert_eq!(
            status, 404,
            "a snapshot taken before the withdrawal must not authorize the request: {body}"
        );
        assert!(body.contains("model_not_found"), "{body}");
    });
}

/// REGRESSION: **the permission to start an engine is read where the engine
/// starts, not where the request arrived.**
///
/// Local exposure is the setting that says whether a downstream tool may cost
/// this machine a subprocess and its memory. It was read into a `ProxySettings`
/// snapshot before the backend was even resolved, so a reader who withdrew it
/// while a request was in flight still had `llama-server` started on their
/// behalf — the request-time twin of the backend-exposure defect, on the
/// setting whose whole purpose is to bound what a request may spend locally.
///
/// The window is the same seam, staged at the same point: the resolve pauses,
/// the reader narrows the setting, and the branch below decides on what is true
/// when it decides. `Loaded` refuses by name (`404`); the stale snapshot would
/// have gone on to `load_local_model`, which fails differently (`503`) because
/// there is no file behind the model — so the two answers cannot be confused.
#[test]
fn a_local_exposure_withdrawn_mid_request_starts_no_engine() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let core = Arc::new(core);
        let runtime = core.runtime();
        let key = runtime.block_on(async {
            core.update_proxy_settings(ProxySettingsUpdate {
                local_exposure: Some(eidola_app_core::proxy::LocalExposure::Downloaded),
                ..Default::default()
            })
            .await
            .expect("open the permission");
            core.set_proxy_backend_exposed("local".to_string(), true)
                .await
                .expect("expose local");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });

        let mut window = core.test_open_proxy_resolve_window();
        let asking = {
            let core = Arc::clone(&core);
            runtime.spawn(async move {
                exchange(
                    &core,
                    &post(
                        "/v1/chat/completions",
                        &key,
                        r#"{"model":"absent@local","messages":[{"role":"user","content":"hi"}]}"#,
                    ),
                )
                .await
            })
        };

        let (status, body) = runtime.block_on(async {
            let resume = window.recv().await.expect("the request reaches the window");
            // The reader narrows the permission while the request is in flight.
            core.update_proxy_settings(ProxySettingsUpdate {
                local_exposure: Some(eidola_app_core::proxy::LocalExposure::Loaded),
                ..Default::default()
            })
            .await
            .expect("withdraw the permission");
            let _ = resume.send(());
            asking.await.expect("the request finishes")
        });

        assert_eq!(
            status, 404,
            "a snapshot taken before the withdrawal must not start an engine: {body}"
        );
        assert!(body.contains("model_not_found"), "{body}");
    });
}

/// REGRESSION: **a backend's auto-start is read where the engine would start,
/// and so is its exposure.**
///
/// `auto_start` came from the row captured at resolution, and turning it off
/// neither bumps the backend's epoch nor retires anything — so a request that
/// had resolved before the reader withdrew it still started `llama-server`.
/// The proxy's exposure of the backend is the same kind of permission and was
/// carried the same way. Both are now one read at the start decision; the
/// spawn counter says whether a subprocess was started at all.
#[test]
fn a_backend_permission_withdrawn_mid_request_starts_no_engine() {
    for withdraw_exposure in [false, true] {
        run(move || {
            let (_mock, core, dir) = core_for(MockConfig::default());
            let models = dir.path().join("models");
            std::fs::create_dir_all(&models).expect("models dir");
            std::fs::write(models.join("m.gguf"), b"gguf").expect("model file");
            let engine = write_exiting_engine(dir.path());
            let key = core.runtime().block_on(async {
                core.update_proxy_settings(ProxySettingsUpdate {
                    local_exposure: Some(eidola_app_core::proxy::LocalExposure::Downloaded),
                    ..Default::default()
                })
                .await
                .expect("open the permission");
                core.add_backend(engine_backend("acme", &models, &engine))
                    .await
                    .expect("add");
                core.set_proxy_backend_exposed("acme".to_string(), true)
                    .await
                    .expect("expose");
                core.create_proxy_key("a tool".to_string())
                    .await
                    .expect("mint")
                    .key
            });
            let core = Arc::new(core);
            let runtime = core.runtime();

            let mut window = core.test_open_proxy_authorized_window();
            let asking = {
                let core = Arc::clone(&core);
                runtime.spawn(async move {
                    exchange(
                        &core,
                        &post(
                            "/v1/chat/completions",
                            &key,
                            r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}]}"#,
                        ),
                    )
                    .await
                })
            };
            let (status, body) = runtime.block_on(async {
                // Past resolution: the row is in hand. Withdraw while the
                // request waits between its authorization and the start.
                let resume = window.recv().await.expect("the authorized request waits");
                if withdraw_exposure {
                    core.set_proxy_backend_exposed("acme".to_string(), false)
                        .await
                        .expect("withdraw the exposure");
                } else {
                    core.update_backend(
                        "acme".to_string(),
                        eidola_app_core::BackendUpdate {
                            auto_start: Some(false),
                            ..Default::default()
                        },
                    )
                    .await
                    .expect("turn auto-start off");
                }
                let _ = resume.send(());
                asking.await.expect("the request finishes")
            });
            assert!(status >= 400, "{status} {body}");
            assert_eq!(
                core.test_engine_spawn_count(),
                0,
                "no subprocess was started after the permission was withdrawn \
                 (exposure withdrawn: {withdraw_exposure}): {status} {body}"
            );
        });
    }
}

/// A `llama-server` stand-in that starts and exits at once: enough for the
/// spawn counter to see a process, and nothing to wait for afterwards.
fn write_exiting_engine(dir: &std::path::Path) -> String {
    let path = dir.join("fake-llama-server");
    std::fs::write(&path, b"#!/bin/sh\nexit 0\n").expect("write engine");
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    path.to_string_lossy().into_owned()
}

/// A `llamacpp` backend with its own models directory and engine.
fn engine_backend(
    id: &str,
    models_dir: &std::path::Path,
    engine: &str,
) -> eidola_app_core::NewBackend {
    eidola_app_core::NewBackend {
        id: id.into(),
        kind: eidola_app_core::BackendKind::LlamaCpp,
        display_name: "Acme".into(),
        base_url: None,
        api_key: None,
        models_dir: Some(models_dir.to_string_lossy().into_owned()),
        model_overrides: None,
        engine_path: Some(engine.to_string()),
        auto_start: true,
    }
}

/// REGRESSION: **the proxy reads an event's data as one value, so a payload
/// split across `data:` fields is still rewritten — and its refund still
/// settles.**
///
/// Parsing field by field saw fragments that are each not JSON and forwarded
/// the event untouched: the chunk kept the upstream's unqualified model name,
/// and the metadata event's refund was never found. `RefundMode::Fail` makes
/// that refund the only copy, so a parser that misses it strands the credential.
#[test]
fn a_payload_split_across_data_fields_is_rewritten_and_settles() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::StreamingSplitDataFields,
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
        assert!(
            !body.contains(STREAM_WIRE_MODEL),
            "every split chunk's model was rewritten: {body}"
        );
        assert!(
            !body.contains("\"refund\""),
            "and no credential material travels downstream: {body}"
        );
        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(
            !wallet.iter().any(|c| c.state == "spending"),
            "the split refund was found and settled the hold: {wallet:?}"
        );
        assert_eq!(mock.refund_hits(), 0, "in band, not by recovery");
    });
}

/// One streamed ask whose upstream sent a head and then reset before the body;
/// returns what the caller saw and what the Record row says.
fn a_cut_pre_stream_body(status: u16, content_type: &'static str) -> (u16, String, Option<String>) {
    let (_mock, core, _dir) = core_for(MockConfig {
        chat: ChatBehavior::HeadThenCut {
            status,
            content_type,
        },
        ..Default::default()
    });
    with_account(&core);
    let key = armed(&core);
    let core = Arc::new(core);
    let runtime = core.runtime();
    let (seen, body) = runtime.block_on(exchange(
        &core,
        &post(
            "/v1/chat/completions",
            &key,
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":true}}"#
            ),
        ),
    ));
    let row = runtime
        .block_on(core.list_requests(20, 0))
        .expect("record")
        .into_iter()
        .find(|r| r.path == "/v1/chat/completions")
        .expect("the exchange is recorded");
    let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
    assert!(
        !wallet.iter().any(|c| c.state == "spending"),
        "a failed read still settles the hold: {wallet:?}"
    );
    assert_cut_prefix_recorded(&core, &row.id);
    (seen, body, row.error)
}

/// **What arrived before the reset is part of the exchange**, so the Record
/// keeps it: `HeadThenCut` sends a head, the first bytes of a JSON body and
/// then nothing. A row holding an empty body would describe an exchange in
/// which no bytes came back.
fn assert_cut_prefix_recorded(core: &AppCore, id: &str) {
    let detail = core
        .runtime()
        .block_on(core.request_detail(id.to_string()))
        .expect("detail")
        .expect("the row");
    let recorded = detail.response_body.unwrap_or_default();
    assert!(
        recorded.starts_with(br#"{"error":"#),
        "the prefix that arrived is recorded: {:?}",
        String::from_utf8_lossy(&recorded)
    );
}

/// The blocking transport's twin: a body cut short keeps what arrived.
#[test]
fn a_blocking_body_cut_short_keeps_what_arrived() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::HeadThenCut {
                status: 200,
                content_type: "application/json",
            },
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
        assert_ne!(status, 200, "{body}");
        let row = runtime
            .block_on(core.list_requests(20, 0))
            .expect("record")
            .into_iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded");
        assert!(row.error.is_some());
        assert_cut_prefix_recorded(&core, &row.id);
    });
}

/// REGRESSION: **a non-2xx whose body could not be read is a read failure, not
/// the upstream's status with nothing behind it.**
///
/// The pre-stream error body was read with `unwrap_or_default()`, so an upstream
/// that reset after its head became a clean read of an empty body: the caller
/// was handed the upstream's `503` as though that were the whole story, and the
/// Record row carried no error. The blocking transport already treats this as
/// the transport failure it is.
#[test]
fn a_pre_stream_error_body_cut_short_is_reported_as_the_read_failure() {
    run(|| {
        let (status, body, error) = a_cut_pre_stream_body(503, "application/json");
        assert_ne!(
            status, 503,
            "the upstream's status is not the answer when its body never arrived: {body}"
        );
        assert!(
            error.is_some(),
            "and the Record says the read failed rather than storing a silent empty body"
        );
    });
}

/// The same defect on the other pre-stream arm: a `2xx` that is not a stream,
/// cut short, was recorded as only the shape refusal — the read failure that
/// actually ended it went nowhere.
#[test]
fn a_non_stream_body_cut_short_is_reported_as_the_read_failure() {
    run(|| {
        let (status, body, error) = a_cut_pre_stream_body(200, "application/json");
        assert_ne!(status, 200, "{body}");
        let error = error.expect("the row carries an error");
        assert!(
            !error.contains("server-sent events"),
            "the row names the read failure, not a shape verdict on bytes that never came: {error}"
        );
    });
}

/// One proxied exchange whose completion opened a connection of its own and
/// then failed before the response head; returns the attestation its Record
/// row is attached to.
fn failed_send_records_against(streaming: bool) -> Option<String> {
    let (_mock, core, _dir) = core_for(MockConfig {
        chat: ChatBehavior::DropBeforeResponse,
        ..Default::default()
    });
    with_account(&core);
    let key = armed(&core);
    core.test_plant_proxy_completion_handshake("completion-handshake");
    let core = Arc::new(core);
    let runtime = core.runtime();

    let (status, body) = runtime.block_on(exchange(
        &core,
        &post(
            "/v1/chat/completions",
            &key,
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":{streaming}}}"#
            ),
        ),
    ));
    assert_ne!(status, 200, "the send failed: {body}");

    let id = runtime
        .block_on(core.list_requests(20, 0))
        .expect("record")
        .into_iter()
        .find(|r| r.path == "/v1/chat/completions")
        .expect("a failed send is still recorded")
        .id;
    runtime
        .block_on(core.request_detail(id))
        .expect("detail")
        .expect("a recorded row")
        .attestation_hash
}

/// REGRESSION: **a send that failed is recorded against the connection it
/// actually opened** — on the blocking transport.
///
/// The handshake completes before a byte of the request is written, so a
/// completion that dies before the response head can already have verified a
/// fresh enclave the catalog fetch never reached. The flush ran only after a
/// response arrived, so the failure row named the catalog's connection and the
/// fresh handshake sat unrecorded in the observer: the Record said the prompt
/// went somewhere it did not, on exactly the exchange someone opens the Record
/// to look at.
#[test]
fn a_failed_blocking_send_is_recorded_against_its_own_connection() {
    run(|| {
        assert_eq!(
            failed_send_records_against(false).as_deref(),
            Some("completion-handshake"),
            "the row names the connection the prompt went down"
        );
    });
}

/// The streaming twin of the arm above: the same exit, written once more on the
/// other transport, so it is pinned once more.
#[test]
fn a_failed_streaming_send_is_recorded_against_its_own_connection() {
    run(|| {
        assert_eq!(
            failed_send_records_against(true).as_deref(),
            Some("completion-handshake"),
            "the row names the connection the prompt went down"
        );
    });
}

/// REGRESSION: **the catalog scanned is the exposed row's, not whatever the id
/// names by the time the scan runs.**
///
/// `/v1/models` read each exposed id and then scanned by id all over again, so a
/// remove-and-re-add between the two sent the catalog request to the
/// replacement's URL — with the replacement's key — and handed its answer to the
/// caller: a destination the reader never exposed, contacted on a proxy
/// caller's say-so. The replacement here is a second server, so whether it was
/// ever asked is a count, not an inference.
#[test]
fn a_backend_replaced_mid_listing_is_never_asked_for_its_catalog() {
    run(|| {
        let (authorized, core, _dir) = core_for(MockConfig::default());
        let replacement = core
            .runtime()
            .block_on(chat_harness::start(MockConfig::default()));
        let external = |url: &str| eidola_app_core::NewBackend {
            id: "acme".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: "Acme".into(),
            base_url: Some(url.into()),
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        };
        let key = core.runtime().block_on(async {
            core.add_backend(external(&authorized.base_url))
                .await
                .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);
        let runtime = core.runtime();

        let mut window = core.test_open_proxy_authorized_window();
        let listing = {
            let core = Arc::clone(&core);
            runtime.spawn(async move { exchange(&core, &get("/v1/models", Some(&key))).await })
        };
        let (status, body) = runtime.block_on(async {
            let resume = window.recv().await.expect("the listing reaches its scan");
            core.remove_backend("acme".to_string())
                .await
                .expect("remove");
            core.add_backend(external(&replacement.base_url))
                .await
                .expect("re-add a different backend under the same name");
            let _ = resume.send(());
            listing.await.expect("the listing finishes")
        });

        assert_eq!(status, 200, "{body}");
        assert!(
            authorized.models_hits() >= 1,
            "the scan ran, against the row that was exposed"
        );
        assert_eq!(
            replacement.models_hits(),
            0,
            "a destination nobody exposed was never contacted"
        );
    });
}

/// REGRESSION: **the listing offers only engines the lease would take.**
///
/// `/v1/models` matched a registry snapshot taken *before* its authorizing read
/// to the authorized row by backend id alone, so a remove-and-re-add between
/// the two published the retired incarnation's engine — a model in the
/// capability statement that every request for it was refused. The listing now
/// asks the lease's own predicate for the incarnation it authorized.
#[test]
fn a_listing_offers_only_the_engines_of_the_incarnation_it_authorized() {
    run(|| {
        let (mock, core, dir) = core_for(MockConfig::default());
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).expect("models dir");
        let engine = write_exiting_engine(dir.path());
        let key = core.runtime().block_on(async {
            core.add_backend(engine_backend("acme", &models, &engine))
                .await
                .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        core.test_register_loaded_local_model("acme", "m", mock.port());
        let core = Arc::new(core);
        let runtime = core.runtime();

        // Control: the authorized incarnation's ready engine is listed.
        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("m@acme"), "a ready engine is offered: {body}");

        let mut window = core.test_open_proxy_authorized_window();
        let listing = {
            let core = Arc::clone(&core);
            let key = key.clone();
            runtime.spawn(async move { exchange(&core, &get("/v1/models", Some(&key))).await })
        };
        let (status, body) = runtime.block_on(async {
            let resume = window.recv().await.expect("the listing reaches its scan");
            core.remove_backend("acme".to_string())
                .await
                .expect("remove");
            core.add_backend(engine_backend("acme", &models, &engine))
                .await
                .expect("re-add under the same name");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose the replacement");
            // The replacement loads the same slug: a ready engine under the
            // same key, belonging to an incarnation this listing never read.
            core.test_register_loaded_local_model("acme", "m", mock.port());
            let _ = resume.send(());
            listing.await.expect("the listing finishes")
        });
        drop(window);
        assert_eq!(status, 200, "{body}");
        assert!(
            !body.contains("m@acme"),
            "a retired incarnation's engine is not offered: {body}"
        );

        // And a listing that authorizes the replacement offers the
        // replacement's engine: each listing answers for the incarnation it
        // read, exactly as the lease does.
        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200, "{body}");
        assert!(
            body.contains("m@acme"),
            "the authorized incarnation's engine is offered: {body}"
        );
    });
}

/// REGRESSION: **the Record says where a plain route's prompt went, after
/// anything is done to the backend that sent it.**
///
/// An attested route answers "where did this go?" through the connection row
/// its handshake writes. An external or engine route wrote none, so the request
/// row's only path to a URL was its `backend_id` — a key to a row the reader
/// can edit or replace under the same name — and the trail lost the answer the
/// moment that happened. The row is now written as the route opens: the URL,
/// `clearnet`, and no attestation, which is the truthful statement that nothing
/// was verified.
#[test]
fn a_plain_routes_destination_survives_the_backend_being_replaced() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            ..Default::default()
        });
        let external = |url: &str| eidola_app_core::NewBackend {
            id: "acme".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: "Acme".into(),
            base_url: Some(url.into()),
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        };
        let key = core.runtime().block_on(async {
            core.add_backend(external(&mock.base_url))
                .await
                .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}]}"#,
            ),
        ));
        assert_eq!(status, 200, "{body}");

        // The reader replaces the backend under the same name.
        runtime.block_on(async {
            core.remove_backend("acme".to_string())
                .await
                .expect("remove");
            core.add_backend(external("https://somewhere-else.example"))
                .await
                .expect("re-add");
        });

        let id = runtime
            .block_on(core.list_requests(20, 0))
            .expect("record")
            .into_iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded")
            .id;
        let detail = runtime
            .block_on(core.request_detail(id))
            .expect("detail")
            .expect("a recorded row");
        assert_eq!(
            detail.base_url.as_deref(),
            Some(mock.base_url.as_str()),
            "the Record still says where the prompt went"
        );
        assert_eq!(detail.transport.as_deref(), Some("clearnet"));
        assert_eq!(
            detail.attestation_hash, None,
            "and claims no verification that never happened"
        );
    });
}

/// The engine-route twin: an engine's address is a port this process picked,
/// and the next start picks another, so only a row written as the route opens
/// can say where a prompt went.
#[test]
fn an_engine_routes_destination_is_recorded() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            ..Default::default()
        });
        let key = core.runtime().block_on(async {
            core.set_proxy_backend_exposed("local".to_string(), true)
                .await
                .expect("expose local");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        // A ready engine answering at the mock's address.
        core.test_register_loaded_local_model("local", "engine", mock.port());
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(
            &core,
            &post(
                "/v1/chat/completions",
                &key,
                r#"{"model":"engine@local","messages":[{"role":"user","content":"hi"}]}"#,
            ),
        ));
        assert_eq!(status, 200, "{body}");

        let id = runtime
            .block_on(core.list_requests(20, 0))
            .expect("record")
            .into_iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded")
            .id;
        let detail = runtime
            .block_on(core.request_detail(id))
            .expect("detail")
            .expect("a recorded row");
        assert_eq!(
            detail.base_url.as_deref(),
            Some(format!("http://127.0.0.1:{}", mock.port()).as_str()),
            "the engine's address at the time is in the Record"
        );
        assert_eq!(detail.attestation_hash, None);
    });
}

/// **A model listing writes no Record row — and the pane says so.**
///
/// A pin rather than a regression: the Proxy pane's copy claims the Record for
/// *completions* and tells the reader, beside the backend list, that a listing
/// is not recorded (and that for a server they added it fetches that server's
/// catalog with its key). That sentence is only true while this is, so a change
/// that starts recording listings fails here and has to move the copy with it.
#[test]
fn a_model_listing_writes_no_record_row() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig::default());
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();

        let (status, body) = runtime.block_on(exchange(&core, &get("/v1/models", Some(&key))));
        assert_eq!(status, 200, "{body}");
        let requests = runtime.block_on(core.list_requests(20, 0)).expect("record");
        assert!(
            requests.iter().all(|r| !r.path.contains("models")),
            "a listing leaves no request row: {:?}",
            requests.iter().map(|r| &r.path).collect::<Vec<_>>()
        );
    });
}

/// REGRESSION: **the incarnation a request was authorized against is the one
/// whose engine may start.**
///
/// Exposure now authorizes a backend *row* rather than an id, but the load that
/// followed resolved its configuration by id all over again — so a
/// remove-and-re-add landing between the two put a different models directory
/// and a different `llama-server` behind the same name, and the subprocess
/// started was the **replacement's**, holding a prompt leased to a destination
/// nobody had exposed. (The replacement is not even exposed here: it never
/// needed to be, which is the point.)
///
/// The fixture makes the two incarnations tell themselves apart by what they
/// can serve: only the replacement's directory holds the model file. A load
/// acting on the authorized row therefore cannot get as far as a process, and
/// the spawn counter is the witness — a child killed microseconds after `fork`
/// leaves no other trace.
#[test]
fn a_backend_replaced_mid_request_never_gets_its_engine_started() {
    run(|| {
        let (_mock, core, dir) = core_for(MockConfig::default());
        let authorized = dir.path().join("authorized-models");
        let replacement = dir.path().join("replacement-models");
        std::fs::create_dir_all(&authorized).expect("authorized dir");
        std::fs::create_dir_all(&replacement).expect("replacement dir");
        std::fs::write(replacement.join("m.gguf"), b"gguf").expect("model file");
        let engine = write_exiting_engine(dir.path());

        let key = core.runtime().block_on(async {
            core.update_proxy_settings(ProxySettingsUpdate {
                local_exposure: Some(eidola_app_core::proxy::LocalExposure::Downloaded),
                ..Default::default()
            })
            .await
            .expect("open the permission");
            core.add_backend(engine_backend("acme", &authorized, &engine))
                .await
                .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        core.test_set_engine_ready_timeout(std::time::Duration::from_millis(1500));
        let core = Arc::new(core);
        let runtime = core.runtime();

        let mut window = core.test_open_proxy_authorized_window();
        let asking = {
            let core = Arc::clone(&core);
            runtime.spawn(async move {
                exchange(
                    &core,
                    &post(
                        "/v1/chat/completions",
                        &key,
                        r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}]}"#,
                    ),
                )
                .await
            })
        };

        let (status, body) = runtime.block_on(async {
            let resume = window
                .recv()
                .await
                .expect("the authorized request reaches the engine start");
            core.remove_backend("acme".to_string())
                .await
                .expect("remove");
            core.add_backend(engine_backend("acme", &replacement, &engine))
                .await
                .expect("re-add a different backend under the same name");
            let _ = resume.send(());
            asking.await.expect("the request finishes")
        });

        assert_ne!(status, 200, "the request cannot be answered: {body}");
        assert_eq!(
            core.test_engine_spawn_count(),
            0,
            "a replacement nobody exposed must never have its engine started"
        );
        assert!(
            core.running_engines().is_empty(),
            "and nothing was left registered"
        );
    });
}

/// Stage the lease regressions: an exposed `llamacpp` backend authorizes a
/// request, and while it waits past its authorization the backend is removed,
/// re-added under the same name, and `replace` puts an engine for the same
/// slug under the replacement. Returns what the caller saw, how long it took,
/// and how many requests reached the replacement's engine.
fn a_replacement_engine_under_the_same_slug(
    replace: impl FnOnce(&AppCore, u16),
) -> (u16, String, std::time::Duration, u64) {
    let (mock, core, dir) = core_for(MockConfig {
        chat: ChatBehavior::OkBlocking,
        ..Default::default()
    });
    let models = dir.path().join("models");
    std::fs::create_dir_all(&models).expect("models dir");
    std::fs::write(models.join("m.gguf"), b"gguf").expect("model file");
    let engine = write_exiting_engine(dir.path());
    let key = core.runtime().block_on(async {
        core.update_proxy_settings(ProxySettingsUpdate {
            local_exposure: Some(eidola_app_core::proxy::LocalExposure::Downloaded),
            ..Default::default()
        })
        .await
        .expect("open the permission");
        core.add_backend(engine_backend("acme", &models, &engine))
            .await
            .expect("add");
        core.set_proxy_backend_exposed("acme".to_string(), true)
            .await
            .expect("expose");
        core.create_proxy_key("a tool".to_string())
            .await
            .expect("mint")
            .key
    });
    // A join that waits must still end, so a hang reads as a failure.
    core.test_set_engine_ready_timeout(std::time::Duration::from_secs(30));
    let core = Arc::new(core);
    let runtime = core.runtime();

    let mut window = core.test_open_proxy_authorized_window();
    let asking = {
        let core = Arc::clone(&core);
        runtime.spawn(async move {
            exchange(
                &core,
                &post(
                    "/v1/chat/completions",
                    &key,
                    r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}]}"#,
                ),
            )
            .await
        })
    };
    let started = std::time::Instant::now();
    let (status, body) = runtime.block_on(async {
        let resume = window.recv().await.expect("the authorized request waits");
        core.remove_backend("acme".to_string())
            .await
            .expect("remove");
        core.add_backend(engine_backend("acme", &models, &engine))
            .await
            .expect("re-add under the same name");
        replace(&core, mock.port());
        let _ = resume.send(());
        tokio::time::timeout(std::time::Duration::from_secs(10), asking)
            .await
            .expect("the request must end rather than wait on another incarnation's engine")
            .expect("the request finishes")
    });
    (status, body, started.elapsed(), mock.chat_hits())
}

/// REGRESSION: **a ready engine is leased only to the incarnation that loaded
/// it.**
///
/// The authorized row and its epoch reached the engine *start*, but the lease
/// in front of it asked `(backend, slug)` alone — so a remove-and-re-add whose
/// replacement had loaded the same slug left a ready engine under that key, and
/// the request leased it: the prompt went to an engine belonging to a backend
/// nobody exposed. "A leased engine is already running" was true; whose engine
/// it was had not been asked.
#[test]
fn a_ready_engine_of_the_replacement_is_never_leased() {
    run(|| {
        let (status, body, _, hits) = a_replacement_engine_under_the_same_slug(|core, port| {
            core.test_register_loaded_local_model("acme", "m", port);
        });
        assert_ne!(status, 200, "the request cannot be served: {body}");
        assert_eq!(
            hits, 0,
            "no prompt reached the engine of a backend nobody exposed"
        );
    });
}

/// The warming-engine shortcut, held to the same rule: a load already in flight
/// under this slug is joined only if it is the authorized incarnation's. A join
/// by slug waited on the replacement's engine — here one that never becomes
/// ready, so the uncured path is a wait the test bounds, and the cured path a
/// refusal at once.
#[test]
fn a_warming_engine_of_the_replacement_is_never_joined() {
    run(|| {
        let (status, body, took, hits) = a_replacement_engine_under_the_same_slug(|core, port| {
            core.test_register_warming_local_model("acme", "m", port);
        });
        assert_ne!(status, 200, "{body}");
        assert!(
            took < std::time::Duration::from_secs(5),
            "refused at once rather than waiting on another incarnation's load: {took:?}"
        );
        assert_eq!(hits, 0);
    });
}

/// REGRESSION: **the epoch that vouches for a request's configuration is read
/// where that configuration was authorized.**
///
/// The sibling of the test above, with the other half isolated: both
/// incarnations can serve the model, so nothing distinguishes them by *what*
/// they hold — only *when* their identity was taken. A load that reads its own
/// epoch reads it after the retirement and so vouches for a configuration the
/// reader has already withdrawn; the epoch read beside the authorized row moves
/// under it, and `reserve_engine` refuses inside the critical section the sweep
/// takes.
#[test]
fn an_authorized_incarnation_retired_mid_request_starts_nothing() {
    run(|| {
        let (_mock, core, dir) = core_for(MockConfig::default());
        let authorized = dir.path().join("authorized-models");
        let replacement = dir.path().join("replacement-models");
        std::fs::create_dir_all(&authorized).expect("authorized dir");
        std::fs::create_dir_all(&replacement).expect("replacement dir");
        // Both can serve it, so only the incarnation's identity separates them.
        std::fs::write(authorized.join("m.gguf"), b"gguf").expect("model file");
        std::fs::write(replacement.join("m.gguf"), b"gguf").expect("model file");
        let engine = write_exiting_engine(dir.path());

        let key = core.runtime().block_on(async {
            core.update_proxy_settings(ProxySettingsUpdate {
                local_exposure: Some(eidola_app_core::proxy::LocalExposure::Downloaded),
                ..Default::default()
            })
            .await
            .expect("open the permission");
            core.add_backend(engine_backend("acme", &authorized, &engine))
                .await
                .expect("add");
            core.set_proxy_backend_exposed("acme".to_string(), true)
                .await
                .expect("expose");
            core.create_proxy_key("a tool".to_string())
                .await
                .expect("mint")
                .key
        });
        core.test_set_engine_ready_timeout(std::time::Duration::from_millis(1500));
        let core = Arc::new(core);
        let runtime = core.runtime();

        let mut window = core.test_open_proxy_authorized_window();
        let asking = {
            let core = Arc::clone(&core);
            runtime.spawn(async move {
                exchange(
                    &core,
                    &post(
                        "/v1/chat/completions",
                        &key,
                        r#"{"model":"m@acme","messages":[{"role":"user","content":"hi"}]}"#,
                    ),
                )
                .await
            })
        };

        let (status, body) = runtime.block_on(async {
            let resume = window
                .recv()
                .await
                .expect("the authorized request reaches the engine start");
            core.remove_backend("acme".to_string())
                .await
                .expect("remove");
            core.add_backend(engine_backend("acme", &replacement, &engine))
                .await
                .expect("re-add a different backend under the same name");
            let _ = resume.send(());
            asking.await.expect("the request finishes")
        });

        assert_ne!(status, 200, "the request cannot be answered: {body}");
        assert_eq!(
            core.test_engine_spawn_count(),
            0,
            "an incarnation retired since it was authorized starts nothing"
        );
        assert!(
            core.running_engines().is_empty(),
            "and nothing was left registered"
        );
    });
}

/// **A removal takes the exposure with it, so the read that authorizes finds
/// nothing on a revived row.**
///
/// The join is what makes the two one question; this is the join answering.
#[test]
fn a_revived_backend_is_not_exposed_by_the_permission_its_predecessor_held() {
    run(|| {
        let (_mock, core, dir) = core_for(MockConfig::default());
        let data_dir = dir.path().join("data");
        let runtime = core.runtime();
        runtime.block_on(async {
            core.set_proxy_backend_exposed("eidola".to_string(), true)
                .await
                .expect("expose");
            let db = eidola_app_core::db::open(&data_dir).await.expect("open");
            let conn = eidola_app_core::db::connect(&db).await.expect("connect");
            assert!(
                eidola_app_core::db::exposed_backend(&conn, "eidola")
                    .await
                    .expect("read")
                    .is_some(),
                "an exposed live backend is what the read is for"
            );
            assert!(
                eidola_app_core::db::exposed_backend(&conn, "local")
                    .await
                    .expect("read")
                    .is_none(),
                "a live backend nobody ticked is not authorized"
            );
        });
    });
}

/// **A body the ceiling stopped is refused whatever the fragment parses as.**
///
/// "A truncated body does not parse" is true of most oversized JSON and false
/// of the case that matters: a complete object followed by enough whitespace to
/// cross the ceiling parses perfectly, because trailing whitespace is valid. So
/// the caller was handed a `200` carrying that object while the Record row
/// beside it said the read had stopped at this app's ceiling — one exchange
/// described two ways, and a partial answer taken for a whole one.
///
/// Which is exactly the `whole_text` rule (`peer_read`) reaching the one
/// surface that *keeps* what it read: the exchange is still recorded, the hold
/// still settles, and only the answer becomes the gateway failure it is.
#[test]
fn an_answer_past_the_read_ceiling_is_refused_rather_than_parsed() {
    run(|| {
        let (_mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlockingPaddedPastCeiling,
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
        assert_ne!(
            status, 200,
            "a read this app stopped is not an answer to hand on: {body}"
        );
        assert!(
            body.contains("ceiling"),
            "and the refusal says what stopped it: {body}"
        );

        // The exchange is still evidence, and the row says the same thing the
        // caller was told.
        let recorded = runtime
            .block_on(core.list_requests(20, 0))
            .expect("record")
            .into_iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded");
        assert!(
            recorded
                .error
                .as_deref()
                .is_some_and(|e| e.contains("ceiling")),
            "the row carries this app's own refusal: {:?}",
            recorded.error
        );
    });
}

/// REGRESSION: **the ceiling refuses whatever the status**, on both
/// transports. An oversized non-2xx was passed through as the upstream's own
/// status with a message read off the fragment, and the row carried no
/// refusal; the pre-stream arm also parsed the fragment for a refund.
#[test]
fn a_non_2xx_answer_past_the_read_ceiling_is_refused_too() {
    for stream in [false, true] {
        run(move || {
            let (_mock, core, _dir) = core_for(MockConfig {
                chat: ChatBehavior::Non2xxPaddedPastCeiling(500),
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
                        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":{stream}}}"#
                    ),
                ),
            ));
            assert_ne!(
                status, 500,
                "stream={stream}: not the fragment's status: {body}"
            );
            assert!(body.contains("ceiling"), "stream={stream}: {body}");
            let recorded = runtime
                .block_on(core.list_requests(20, 0))
                .expect("record")
                .into_iter()
                .find(|r| r.path == "/v1/chat/completions")
                .expect("the exchange is recorded");
            assert!(
                recorded
                    .error
                    .as_deref()
                    .is_some_and(|e| e.contains("ceiling")),
                "stream={stream}: {:?}",
                recorded.error
            );
        });
    }
}

/// REGRESSION: **a `200` that parses is not yet an answer.**
///
/// Syntax was the whole test: `{}`, `null` and an upstream's error document all
/// parse, so each went downstream as a success — with this app's own canonical
/// `model` inserted into it, which makes the fabrication read *more* like a
/// completion than the raw body did. The floor is the turn path's own: a body
/// with no `choices` array carries nothing this app's chat would call an answer.
///
/// The refund half is the other assertion, and it is why the settlement runs
/// before the refusal: the credential was spent upstream either way, so a body
/// too shapeless to answer with can still hold the only copy of its successor.
/// `RefundMode::Fail` makes that the *only* copy — recovery cannot answer — so a
/// cure that refused before reading the token would strand the credential.
#[test]
fn a_success_that_is_not_a_completion_is_refused_rather_than_forwarded() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlockingNotACompletion,
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
                &format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#),
            ),
        ));
        assert_ne!(
            status, 200,
            "a body with no completion in it is not an answer to hand on: {body}"
        );
        assert!(
            body.contains("not a chat completion"),
            "and the refusal says what was wrong with it: {body}"
        );

        // The exchange is evidence, and the row says what the caller was told.
        let recorded = runtime
            .block_on(core.list_requests(20, 0))
            .expect("record")
            .into_iter()
            .find(|r| r.path == "/v1/chat/completions")
            .expect("the exchange is recorded");
        assert!(
            recorded
                .error
                .as_deref()
                .is_some_and(|e| e.contains("not a chat completion")),
            "the row carries this app's own refusal: {:?}",
            recorded.error
        );

        // The hold settled from the token the refused body carried.
        let wallet = runtime.block_on(core.wallet_lifecycle()).expect("wallet");
        assert!(
            wallet.iter().any(|c| c.state == "spent"),
            "the refund rode the body this app refused: {wallet:?}"
        );
        assert!(
            !wallet.iter().any(|c| c.state == "spending"),
            "a refusal must not strand a spent credential: {wallet:?}"
        );
        assert_eq!(
            mock.refund_hits(),
            0,
            "recovery is the fallback for an absent token, not the first move"
        );
    });
}

/// **A bare-`\r` stream is split into its events, not accumulated into one.**
///
/// The event-stream format takes `\r\n`, `\n` and a bare `\r`, and any two in a
/// row end an event — so recognising only `\n\n` and `\r\n\r\n` never split such
/// a stream at all. On this path that is worse than slow: the frame accumulator
/// has a ceiling of its own, so a long enough stream is refused outright, and
/// until then nothing streams and the per-event model rewrite never runs.
///
/// The assertion is the rewrite, because it is per event: it can only be true
/// if the events were separated.
#[test]
fn a_stream_framed_with_bare_carriage_returns_is_split_into_its_events() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkStreamingBareCarriageReturns,
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
        assert!(body.contains("[DONE]"), "the stream ran to its end: {body}");
        // **Separately** is the claim, and the count of `data:` cannot make it:
        // one undivided frame carries every event's text too. What differs is
        // the *framing* — each forwarded event is its own HTTP chunk — so the
        // chunk headers are what the assertion reads.
        let chunks = body
            .split("\r\n")
            .filter(|line| {
                !line.is_empty()
                    && *line != "0"
                    && line.chars().all(|c| c.is_ascii_hexdigit())
                    && u64::from_str_radix(line, 16).is_ok()
            })
            .count();
        assert!(
            chunks >= 3,
            "the events were forwarded one at a time rather than accumulated \
             into one frame — {chunks} chunk(s) in: {body}"
        );
        assert!(
            !body.contains("refund"),
            "no credential material travels downstream: {body}"
        );
        // **And the fields inside each event were read, not just its edges.**
        // Every payload names the model in the upstream's own spelling behind
        // an `id:` line; a parser that splits events at the right byte and then
        // walks their fields with `str::lines()` sees one line with no `data:`
        // prefix, finds no payload, and forwards the event untouched — which is
        // silent, and is also how a metadata event's refund goes missing.
        assert!(
            !body.contains(STREAM_WIRE_MODEL),
            "every chunk's model was rewritten, so every chunk's fields were \
             parsed: {body}"
        );
        assert!(
            body.matches(&format!("\"model\":\"{MODEL}\"")).count() >= 3,
            "and the rewrite is per event: {body}"
        );
        assert!(mock.refund_hits() >= 1, "the streaming hold settled");
    });
}

/// REGRESSION: **a proxied completion carries no prefix-cache key — not one of
/// this app's, and not one its caller wrote.**
///
/// The proxy's turn has no lineage: no space, no participant, nothing that
/// says two requests continue one conversation, so there is nothing for an
/// app-minted key to be scoped to. And a downstream tool's `cache_key` is a
/// body field outside the allowlist for the reason `traceparent` is a header
/// outside its own: a value the caller picks and never rotates, linking every
/// request it rides on for as long as the tool likes. So even for a model whose
/// engine caches prefixes, the upstream sees no key either way.
#[test]
fn a_proxied_completion_carries_no_cache_key_even_when_its_caller_sends_one() {
    run(|| {
        let (mock, core, _dir) = core_for(MockConfig {
            chat: ChatBehavior::OkBlocking,
            declared_prompt_cache: Some(chat_harness::supported_prompt_cache()),
            ..Default::default()
        });
        with_account(&core);
        let key = armed(&core);
        let core = Arc::new(core);
        let runtime = core.runtime();
        let caller_key = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";

        for body in [
            format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}]}}"#),
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"cache_key":"{caller_key}"}}"#
            ),
        ] {
            let (status, answer) =
                runtime.block_on(exchange(&core, &post("/v1/chat/completions", &key, &body)));
            assert_eq!(status, 200, "{answer}");
        }

        let sent = mock.chat_bodies();
        assert_eq!(sent.len(), 2);
        for body in &sent {
            assert!(body.get("cache_key").is_none(), "{body}");
        }
        for raw in mock.chat_raw_bodies() {
            let text = String::from_utf8(raw).unwrap();
            assert!(!text.contains(caller_key), "{text}");
        }
    });
}
