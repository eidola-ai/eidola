//! End-to-end tests for `AppCore::chat` / `chat_stream` against an in-process
//! mock upstream (see `chat_harness`). These pay down two waves of debt:
//!
//! * **Wave-3 failure-path bus emissions** — the exit-point table in
//!   `tests/bus.rs` was asserted-by-inspection only. Each typed-failure test
//!   here asserts BOTH the returned error AND the emitted `Change`s, turning a
//!   table row into an executed test.
//! * **Wave-1 auto-provisioning** — `ensure_spendable_credential` had only a
//!   pure-decision unit test; `auto_provisioning_*` here drives the empty-wallet
//!   + funded-account path all the way through a successful chat.
//!
//! Determinism: the mock is in-process over loopback HTTP with no real network
//! and no attestation handshake (the `with_test_http_client` seam). The whole
//! suite runs in well under a second.

mod chat_harness;

use chat_harness::{ChatBehavior, MODEL, MockConfig, MockServer, RefundMode, with_account};
use eidola_app_core::changes::Change;
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatStreamEvent};

// ---------------------------------------------------------------------------
// Harness: run an async test body on a dedicated OS thread.
//
// `AppCore` owns its own multi-thread tokio runtime; dropping it while another
// runtime is active on the same thread panics. We run each test body — and the
// `AppCore` Drop — on a plain OS thread that builds the mock + core inside its
// own runtime (`AppCore::runtime().block_on`). Mirrors `tests/bus.rs`.
// ---------------------------------------------------------------------------

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

fn space_changes(changes: &[Change]) -> Vec<&Change> {
    changes
        .iter()
        .filter(|c| matches!(c, Change::Space(_)))
        .collect()
}

/// Build a mock + a core wired to it (see `chat_harness::core_for`). Callers
/// add an account via `with_account` when they want the auto-provisioning path.
fn setup(config: MockConfig) -> (MockServer, AppCore, tempfile::TempDir) {
    chat_harness::core_for(config)
}

// ===========================================================================
// Happy path — blocking chat
// ===========================================================================

#[test]
fn blocking_chat_persists_and_emits() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("How do tides work?".into(), MODEL.into(), None))
            .expect("chat should succeed");

        // Returned usage + charge.
        assert_eq!(result.input_tokens, Some(11));
        assert_eq!(result.output_tokens, Some(5));
        assert!(result.credits_charged > 0);
        assert_eq!(result.content, "Hello from the mock.");

        let changes = drain(&mut rx);
        // SpaceIndex (new space + auto-title), Space(id), Wallet, Record all
        // emitted on success. Wallet appears twice: once at spend start, once
        // on final success.
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(
            changes.contains(&Change::Space(result.space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(changes.contains(&Change::Record), "got {changes:?}");

        // Persistence: space, user + assistant turns, request row.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(result.space_id.clone()))
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "How do tides work?");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "Hello from the mock.");

        let spaces = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces[0].title.as_deref(), Some("How do tides work?"));

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].path, "/v1/chat/completions");
        assert_eq!(requests[0].response_status, Some(200));

        // The inline refund recovered a successor credential — no recovery hit.
        assert_eq!(mock.refund_hits(), 0);
        assert_eq!(mock.chat_hits(), 1);
    });
}

#[test]
fn blocking_chat_into_existing_space_does_not_emit_space_index_again() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        // First turn creates + auto-titles the space.
        let first = core
            .runtime()
            .block_on(core.chat("First question".into(), MODEL.into(), None))
            .expect("first chat");

        let mut rx = core.subscribe_changes();
        // Second turn into the same (titled) space.
        let second = core
            .runtime()
            .block_on(core.chat(
                "Second question".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
            ))
            .expect("second chat");
        assert_eq!(second.space_id, first.space_id);

        let changes = drain(&mut rx);
        // Not a new space and not auto-titled → no SpaceIndex this time.
        assert!(
            !changes.contains(&Change::SpaceIndex),
            "second turn into a titled space must not emit SpaceIndex; got {changes:?}"
        );
        assert!(changes.contains(&Change::Space(first.space_id.clone())));
        assert!(changes.contains(&Change::Record));

        // Four messages now (2 turns × user+assistant).
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(first.space_id))
            .expect("messages");
        assert_eq!(messages.len(), 4);
    });
}

#[test]
fn blocking_chat_recovers_refund_when_no_inline_refund() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkBlockingNoInlineRefund,
            ..MockConfig::default()
        });
        with_account(&core);

        let result = core
            .runtime()
            .block_on(core.chat("hello".into(), MODEL.into(), None))
            .expect("chat should succeed");
        assert_eq!(result.content, "Hello from the mock.");

        // No inline refund → the recovery endpoint was consulted.
        assert!(mock.refund_hits() >= 1);

        // A successor credential exists and is active/spendable.
        let creds = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet");
        assert!(
            !creds.is_empty(),
            "a recovered successor credential should be active"
        );
    });
}

// ===========================================================================
// Happy path — streaming chat
// ===========================================================================

#[test]
fn streaming_chat_delivers_deltas_and_persists() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();

        let result = core.runtime().block_on(async {
            // Collect events concurrently with the stream completing.
            let collector = async {
                let mut content = String::new();
                let mut reasoning = String::new();
                while let Some(ev) = events_rx.recv().await {
                    match ev {
                        ChatStreamEvent::ContentDelta(t) => content.push_str(&t),
                        ChatStreamEvent::ReasoningDelta(t) => reasoning.push_str(&t),
                    }
                }
                (content, reasoning)
            };
            let chat = core.chat_stream("stream me".into(), MODEL.into(), None, tx);
            let (res, (content, reasoning)) = tokio::join!(chat, collector);
            (res, content, reasoning)
        });

        let (res, content, reasoning) = result;
        let res = res.expect("stream should complete");

        assert_eq!(content, "Hello from the stream.");
        assert_eq!(reasoning, "thinking…");
        assert_eq!(res.content, "Hello from the stream.");
        assert_eq!(res.input_tokens, Some(11));
        assert_eq!(res.output_tokens, Some(5));

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Space(res.space_id.clone())));
        assert!(changes.contains(&Change::Wallet));
        assert!(changes.contains(&Change::Record));

        // Streaming always goes through the recovery endpoint for its refund.
        assert!(mock.refund_hits() >= 1);

        let messages = core
            .runtime()
            .block_on(core.get_space_messages(res.space_id))
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content, "Hello from the stream.");
    });
}

// ===========================================================================
// Auto-provisioning (wave-1 debt)
// ===========================================================================

#[test]
fn auto_provisioning_empty_wallet_funded_account_succeeds() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        // Wallet starts empty.
        let before = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet");
        assert!(before.is_empty(), "wallet should start empty");

        let mut rx = core.subscribe_changes();
        let result = core
            .runtime()
            .block_on(core.chat("provision me".into(), MODEL.into(), None))
            .expect("chat should auto-provision and succeed");
        assert_eq!(result.content, "Hello from the mock.");

        // Allocation emits Wallet + Account transparently; chat then emits its
        // own Space/Wallet/Record. Account is only emitted by the allocate path.
        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Account),
            "auto-allocation should emit Account; got {changes:?}"
        );
        assert!(changes.contains(&Change::Wallet));

        // After the spend, a successor credential remains.
        let after = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet");
        assert!(
            !after.is_empty(),
            "a successor credential should remain after the spend"
        );
    });
}

// ===========================================================================
// Typed failure: pre-space errors leave zero durable trace, emit nothing
// ===========================================================================

#[test]
fn no_account_leaves_zero_trace_and_no_emissions() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        // NO account configured and empty wallet → NoAccount before any space.
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("hi".into(), MODEL.into(), None))
            .expect_err("should fail with NoAccount");
        assert!(matches!(err.root(), AppError::NoAccount), "got {err:?}");
        // Pre-space error stays unwrapped (no space id to adopt).
        assert_eq!(err.chat_space_id(), None);

        let changes = drain(&mut rx);
        assert!(
            changes.is_empty(),
            "NoAccount must emit nothing; got {changes:?}"
        );

        // Zero durable trace: no space row was inserted.
        let spaces = core
            .runtime()
            .block_on(core.list_spaces(true))
            .expect("spaces");
        assert!(spaces.is_empty(), "no orphan space; got {spaces:?}");
    });
}

#[test]
fn insufficient_balance_leaves_zero_trace_and_no_emissions() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            balance: 1, // cannot cover the charge
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("hi".into(), MODEL.into(), None))
            .expect_err("should fail with InsufficientBalance");
        assert!(
            matches!(err.root(), AppError::InsufficientBalance { .. }),
            "got {err:?}"
        );
        assert_eq!(err.chat_space_id(), None);

        let changes = drain(&mut rx);
        // The balance fetch + allocate path does not commit anything; only the
        // (failed) allocation could emit. No durable chat write happened, so no
        // Space/Record. Account/Wallet are only emitted on a *successful*
        // allocation.
        assert!(
            !changes.contains(&Change::Record),
            "no Record on pre-space failure; got {changes:?}"
        );
        assert!(space_changes(&changes).is_empty(), "got {changes:?}");

        let spaces = core
            .runtime()
            .block_on(core.list_spaces(true))
            .expect("spaces");
        assert!(spaces.is_empty(), "no orphan space; got {spaces:?}");
    });
}

// ===========================================================================
// Typed failure: network error after send (connection dropped)
// ===========================================================================

#[test]
fn network_error_after_send_emits_user_turn_and_wraps_space_id() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::DropBeforeResponse,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("dropped".into(), MODEL.into(), None))
            .expect_err("should fail on dropped connection");

        // Wrapped with the persisted space id (the user turn committed).
        let space_id = err
            .chat_space_id()
            .expect("network-error arm must carry the space id")
            .to_string();
        // Underlying error is a transport/network error, not a server error.
        assert!(
            matches!(err.root(), AppError::Network { .. }),
            "got {:?}",
            err.root()
        );

        let changes = drain(&mut rx);
        // User turn committed → Space(id) + SpaceIndex (new space) emitted.
        // Wallet was emitted at spend start. No Record (request row not written).
        assert!(
            changes.contains(&Change::Space(space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Record),
            "no Record before request row; got {changes:?}"
        );

        // The committed user turn is durable and discoverable by the wrapped id.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    });
}

// ===========================================================================
// Typed failure: non-2xx response with error body
// ===========================================================================

#[test]
fn non_2xx_emits_record_and_space_and_persists_request_row() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx should fail");

        let space_id = err
            .chat_space_id()
            .expect("space id on non-2xx")
            .to_string();
        match err.root() {
            AppError::Server { status, .. } => assert_eq!(*status, 500),
            other => panic!("expected Server(500), got {other:?}"),
        }

        let changes = drain(&mut rx);
        // The full set: Wallet (spend start), Space(id), SpaceIndex (new space),
        // and Record (request row committed in the non-2xx arm).
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id.clone())));
        assert!(changes.contains(&Change::SpaceIndex));
        assert!(changes.contains(&Change::Wallet));

        // Request row persisted with the 500 status.
        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].response_status, Some(500));
    });
}

#[test]
fn streaming_non_2xx_emits_record_and_space() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(503),
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, _events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let err = core
            .runtime()
            .block_on(core.chat_stream("boom".into(), MODEL.into(), None, tx))
            .expect_err("non-2xx stream should fail");

        let space_id = err.chat_space_id().expect("space id").to_string();
        match err.root() {
            AppError::Server { status, .. } => assert_eq!(*status, 503),
            other => panic!("expected Server(503), got {other:?}"),
        }

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id)));
        assert!(changes.contains(&Change::SpaceIndex));
        assert!(changes.contains(&Change::Wallet));

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].response_status, Some(503));
    });
}

// ===========================================================================
// Typed failure: mid-SSE abort (server closes the stream mid-events)
// ===========================================================================

#[test]
fn mid_sse_abort_emits_user_turn_and_wraps_space_id() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::StreamingMidAbort,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, _events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let err = core
            .runtime()
            .block_on(core.chat_stream("stream me".into(), MODEL.into(), None, tx))
            .expect_err("mid-stream abort should fail");

        let space_id = err
            .chat_space_id()
            .expect("mid-SSE abort must carry the space id")
            .to_string();
        assert!(
            matches!(err.root(), AppError::Network { .. }),
            "got {:?}",
            err.root()
        );

        let changes = drain(&mut rx);
        // User turn committed before the stream began reading → Space + SpaceIndex
        // + Wallet, but no Record (request row not written on mid-stream failure).
        assert!(
            changes.contains(&Change::Space(space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Record),
            "no Record on mid-SSE abort; got {changes:?}"
        );

        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    });
}

// ===========================================================================
// Refund-recovery variants: succeed vs fail on the non-2xx path
// ===========================================================================

#[test]
fn non_2xx_with_refund_recovery_emits_wallet_for_successor() {
    run(|| {
        // Pre-fund the wallet with a large credential so the spend does NOT
        // auto-provision (keeps Account out of the picture) and the only Wallet
        // emissions come from spend-start + recovered successor.
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            refund: RefundMode::Succeed,
            ..MockConfig::default()
        });
        with_account(&core);

        let _ = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx fails");

        // The non-2xx arm consulted the recovery endpoint and minted a successor.
        assert!(mock.refund_hits() >= 1);
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        // At least one credential should now be active (the recovered successor).
        assert!(
            lifecycle.iter().any(|c| c.state == "active"),
            "a recovered successor should be active; got {lifecycle:?}"
        );
    });
}

#[test]
fn non_2xx_with_failed_refund_recovery_still_errors_and_emits_record() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            refund: RefundMode::Fail,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx fails");
        let space_id = err.chat_space_id().expect("space id").to_string();

        // Recovery was attempted but failed (500 from the refund endpoint).
        assert!(mock.refund_hits() >= 1);

        let changes = drain(&mut rx);
        // Even with no recovered successor, the non-2xx arm still emits the
        // request-row Record + Space + SpaceIndex.
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id)));
        assert!(changes.contains(&Change::SpaceIndex));

        // The spending credential never recovered → it stays in `spending`.
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        assert!(
            lifecycle.iter().any(|c| c.state == "spending"),
            "the unspent credential should remain in spending; got {lifecycle:?}"
        );
    });
}
