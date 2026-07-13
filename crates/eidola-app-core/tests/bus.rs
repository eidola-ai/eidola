//! Invalidation-bus tests.
//!
//! Verifies the core invariants of the bus:
//! - Every durable commit emits the correct domain change(s).
//! - Errors never emit.
//! - Two independent subscribers both receive the same events.
//!
//! Operations that require HTTP (account_allocate, chat) are tested at the
//! `Inner` db-helper level via `AppCore`'s sync/async surface where possible;
//! full-HTTP paths are covered by other test suites (updates_check.rs uses
//! wiremock). The bus itself doesn't care about the write mechanism — only
//! that the emit calls are placed correctly, which is what these tests assert.
//!
//! ## Error-path emission coverage
//!
//! `chat` / `chat_stream` are now `post` + `run_turn(Reply)` (wave 5.2b). `post`
//! persists the user turn FIRST (no credential) and emits `Change::Space(id)` +
//! `Change::SpaceIndex` (new space / auto-title). `run_turn` then does the
//! request: `Change::Wallet` at spend start, `Change::Record` (+`Space`) on
//! non-2xx, `Space`+`Wallet`+`Record` on success, and re-signals `Space` on its
//! error exits — it never emits `SpaceIndex` (post owns that). Every `run_turn`
//! error wraps the (always-persisted) space id. The exit-point map below lists
//! the UNION of emissions across the post + run_turn pair:
//!
//! The rows marked **`chat_path.rs`** are now *executed* against the in-process
//! mock-upstream harness (`tests/chat_harness/`), which drives the real `chat` /
//! `chat_stream` HTTP paths via the `with_test_http_client` seam and asserts
//! BOTH the typed/wrapped error AND the emitted `Change`s — turning these from
//! asserted-by-inspection into regression-gated.
//!
//! | Exit point | Writes committed | Emissions | Tested here |
//! |---|---|---|---|
//! | Funding failure at request time (`NoAccount`, `InsufficientBalance`, zero-charge) | The posted user turn (post committed it) | `Space(id)`, `SpaceIndex` (from post); error wraps the space id | `chat_path.rs` (`no_account_persists_post_*`, `insufficient_balance_persists_post_*`) — root() still routes onboarding; the post survives |
//! | `chat`/`chat_stream` — `insert_pre_credential_refund` succeeds, later step fails | Credential in `spending` state | `Wallet` | `chat_path.rs` (every post-send failure test asserts `Wallet`; the failed-recovery test asserts the credential stays `spending`) |
//! | `chat`/`chat_stream` — network-error arm (`send` `Err`), `process_refund` `Ok` | Successor credential + user turn | `Wallet`, `Space(id)`, `SpaceIndex`? | `chat_path.rs` (`network_error_after_send_*`; the non-2xx-with-recovery test covers the recovered-successor `Wallet`) |
//! | `chat`/`chat_stream` — network-error arm, no refund recovered | User turn | `Space(id)`, `SpaceIndex`? | `chat_path.rs` (`network_error_after_send_*`) |
//! | `chat` (Ok arm) — `flush_attestations` / `resp.text()` / response JSON parse fails | User turn | `Space(id)`, `SpaceIndex`? | Partial — the `resp.text()` failure on a dropped connection is exercised by `network_error_after_send_*` (reqwest may surface the drop in either arm); both arms emit the same user-turn set |
//! | `chat` — refund-from-body `process_refund` fails | User turn | `Space(id)`, `SpaceIndex`? | No — needs a malformed inline refund (low value: identical emission set to the tested arms) |
//! | `chat_stream` (Ok arm) — `flush_attestations` fails | User turn | `Space(id)`, `SpaceIndex`? | No — `flush_attestations` is a no-op under the no-attestation seam |
//! | `chat_stream` — mid-SSE read failure (`chunk` `Err`) | User turn | `Space(id)`, `SpaceIndex`? | `chat_path.rs` (`mid_sse_abort_*`) |
//! | `chat` — non-2xx response, after `insert_request` | Space, user-message, request rows | `Space(id)`, `SpaceIndex`?, `Record` | `chat_path.rs` (`non_2xx_emits_record_and_space_*`) |
//! | `chat_stream` — non-2xx response, after `insert_request` inside that branch | Space, user-message, request rows | `Space(id)`, `SpaceIndex`?, `Record`; `Wallet` if refund recovered | `chat_path.rs` (`streaming_non_2xx_*`, `non_2xx_with_refund_recovery_*`, `non_2xx_with_failed_refund_recovery_*`) |
//!
//! `SpaceIndex?` = emitted by `post` when the listing changed (new space /
//! auto-title); `run_turn` never emits it. Plain `?` on intervening local-DB
//! action/content/antecedent inserts stays *unemitted* — internal-consistency
//! (kill-`-9`-class) failures, not durable partial state to reconcile.
//!
//! **Local turns (`local/<slug>` models).** `prepare_turn` routes these to the
//! loopback llama.cpp engine with `TurnPrep.spend = None`: no credential is
//! provisioned, no `Authorization` header is sent, and every `Wallet` emission
//! above is skipped (spend-start, refund-recovery, and the success emission are
//! all gated on `spend`). The rows otherwise apply unchanged — same
//! `Space`/`SpaceIndex`/`Record` placement, same `ChatFailed` wrapping after
//! the post persists. A local model that is not loaded fails in the routing
//! step (typed `AppError::LocalModel`, pre-`wrap`, post surviving) — executed
//! in `tests/local_models.rs` (`local_blocking_chat_has_no_spend_no_auth_no_wallet`,
//! `local_streaming_chat_streams_and_persists_without_wallet`,
//! `local_chat_with_unloaded_model_is_typed_error`). The local-domain
//! lifecycle emissions (`Change::LocalModels` on download start / throttled
//! progress / completion / failure / delete / engine load / ready / unload /
//! crash) are asserted there too and never touch the chat-domain rows.
//!
//! **Failure-path id adoption (item C).** `post` persists the space before
//! `run_turn` runs, so every `run_turn` error is wrapped as
//! `AppError::ChatFailed { space_id }` (its `Display` defers to the source). A
//! blank GUI `Space` (id=`None`) thus learns its persisted id even on a funding
//! failure. Only post's own pre-persist errors (e.g. an empty prompt) stay
//! unwrapped. Unit-tested in `error.rs` (`chat_failed_display_defers_to_source`,
//! `root_unwraps_*`, `chat_space_id_only_on_wrapper`).
//!
//! The happy-path tests below confirm the success-path emissions remain intact
//! and that the shared infrastructure (bus capacity, multi-subscriber delivery)
//! works. The full chat HTTP paths — happy-path persistence/emission and the
//! error-path emission rows above — live in `tests/chat_path.rs` on top of the
//! `tests/chat_harness/` mock upstream; chat-path changes must extend that
//! harness.

use eidola_app_core::{AppCore, changes::Change};

fn make_core() -> (AppCore, tempfile::TempDir) {
    // A single crypto-provider install is idempotent across tests.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    (AppCore::new(config_dir, data_dir), dir)
}

// ---------------------------------------------------------------------------
// Helper: drain all messages currently available on a receiver (non-blocking).
// ---------------------------------------------------------------------------

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(c) => out.push(c),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                panic!(
                    "test receiver lagged by {n} — increase BUS_CAPACITY or slow down test writes"
                );
            }
        }
    }
    out
}

// ===========================================================================
// Config domain
// ===========================================================================

#[test]
fn config_write_emits_config() {
    let (core, _dir) = make_core();
    let mut rx = core.subscribe_changes();

    core.set_base_url("https://example.com".into()).unwrap();
    let changes = drain(&mut rx);
    assert!(
        changes.contains(&Change::Config),
        "set_base_url should emit Config; got {changes:?}"
    );
}

#[test]
fn set_default_model_emits_config() {
    let (core, _dir) = make_core();
    let mut rx = core.subscribe_changes();

    core.set_default_model("kimi-k2-6".into()).unwrap();
    let changes = drain(&mut rx);
    assert!(
        changes.contains(&Change::Config),
        "set_default_model should emit Config; got {changes:?}"
    );
}

#[test]
fn clear_base_url_override_emits_config() {
    let (core, _dir) = make_core();
    core.set_base_url("https://example.com".into()).unwrap();

    let mut rx = core.subscribe_changes();
    core.clear_base_url_override().unwrap();
    let changes = drain(&mut rx);
    assert!(
        changes.contains(&Change::Config),
        "clear_base_url_override should emit Config; got {changes:?}"
    );
}

#[test]
fn set_account_credentials_emits_config() {
    let (core, _dir) = make_core();
    let mut rx = core.subscribe_changes();

    core.set_account_credentials("id123".into(), "secret456".into())
        .unwrap();
    let changes = drain(&mut rx);
    assert!(
        changes.contains(&Change::Config),
        "set_account_credentials should emit Config; got {changes:?}"
    );
}

#[test]
fn reset_account_emits_config() {
    let (core, _dir) = make_core();
    core.set_account_credentials("id123".into(), "secret456".into())
        .unwrap();

    let mut rx = core.subscribe_changes();
    core.reset_account().unwrap();
    let changes = drain(&mut rx);
    assert!(
        changes.contains(&Change::Config),
        "reset_account should emit Config; got {changes:?}"
    );
}

#[test]
fn config_write_failure_does_not_emit() {
    let (core, _dir) = make_core();
    // set_default_model rejects empty strings — no write, no emit.
    let mut rx = core.subscribe_changes();
    let _ = core.set_default_model("   ".into()); // returns Err
    let changes = drain(&mut rx);
    assert!(
        changes.is_empty(),
        "failed config write must not emit; got {changes:?}"
    );
}

// ---------------------------------------------------------------------------
// Helper: run an async closure in a dedicated OS thread.
// AppCore owns its own tokio runtime; dropping it while another tokio
// runtime is active on the same thread panics. The solution is to run the
// entire test body — including the Drop of AppCore — on a plain OS thread
// that itself calls block_on via AppCore's runtime (AppCore::new spins up
// the runtime; async AppCore methods .await it from any context). We expose
// a sync shim rather than #[tokio::test] for all async AppCore tests.
// ---------------------------------------------------------------------------

fn run_in_thread<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

// ===========================================================================
// SpaceIndex domain
// ===========================================================================

#[test]
fn create_space_emits_space_index() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();

        core.runtime()
            .block_on(core.create_space(Some("My Space".into())))
            .unwrap();
        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::SpaceIndex),
            "create_space should emit SpaceIndex; got {changes:?}"
        );
    });
}

#[test]
fn archive_space_emits_space_index() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let space = core.runtime().block_on(core.create_space(None)).unwrap();

        let mut rx = core.subscribe_changes();
        let archived = core
            .runtime()
            .block_on(core.archive_space(space.id.clone()))
            .unwrap();
        assert!(archived);

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::SpaceIndex),
            "archive_space should emit SpaceIndex; got {changes:?}"
        );
    });
}

#[test]
fn archive_space_no_emit_when_space_does_not_exist() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();

        // archive_space on an unknown id returns Ok(false) — no write, no emit.
        let result = core
            .runtime()
            .block_on(core.archive_space("no-such-id".into()))
            .unwrap();
        assert!(!result);
        let changes = drain(&mut rx);
        assert!(
            changes.is_empty(),
            "archive_space(unknown) must not emit; got {changes:?}"
        );
    });
}

#[test]
fn rename_space_emits_space_index() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let space = core.runtime().block_on(core.create_space(None)).unwrap();

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.rename_space(space.id, "New Title".into()))
            .unwrap();

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::SpaceIndex),
            "rename_space should emit SpaceIndex; got {changes:?}"
        );
    });
}

#[test]
fn rename_space_no_emit_on_failure() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();

        // Renaming a non-existent space returns an error.
        let result = core
            .runtime()
            .block_on(core.rename_space("no-such-id".into(), "Irrelevant".into()));
        assert!(result.is_err());

        let changes = drain(&mut rx);
        assert!(
            changes.is_empty(),
            "rename_space error must not emit; got {changes:?}"
        );
    });
}

// ===========================================================================
// post() — save a thought without requesting a response (wave 5)
//
// post() is the save side of the save-vs-request split: no credential, no
// HTTP, so it's exercised directly here rather than through the chat harness.
// Contract: emit Change::Space(id) on every successful post (content changed)
// + Change::SpaceIndex only when the listing changed (new space or auto-title);
// no emit on a failed (empty) post.
// ===========================================================================

#[test]
fn post_new_space_emits_space_and_index_and_persists() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.post("Why is the sky blue?".into(), None))
            .unwrap();
        assert!(result.is_new_space);

        let changes = drain(&mut rx);
        assert!(
            changes.iter().any(|c| matches!(c, Change::Space(_))),
            "post should emit Space(id); got {changes:?}"
        );
        assert!(
            changes.contains(&Change::SpaceIndex),
            "post creating a space should emit SpaceIndex; got {changes:?}"
        );

        // The thought is durably persisted as a user turn.
        let msgs = core
            .runtime()
            .block_on(core.get_space_messages(result.space_id))
            .unwrap();
        assert_eq!(msgs.len(), 1, "post should persist exactly one user turn");
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "Why is the sky blue?");
    });
}

#[test]
fn second_post_emits_space_but_not_index() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        // First post creates (and auto-titles) the space.
        let first = core
            .runtime()
            .block_on(core.post("First thought".into(), None))
            .unwrap();

        // A second post into the same space changes content but not the
        // library listing — Space without SpaceIndex.
        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.post("Second thought".into(), Some(first.space_id.clone())))
            .unwrap();

        let changes = drain(&mut rx);
        assert!(
            changes.iter().any(|c| matches!(c, Change::Space(_))),
            "second post should emit Space(id); got {changes:?}"
        );
        assert!(
            !changes.contains(&Change::SpaceIndex),
            "second post must not emit SpaceIndex (listing unchanged); got {changes:?}"
        );

        let msgs = core
            .runtime()
            .block_on(core.get_space_messages(first.space_id))
            .unwrap();
        assert_eq!(msgs.len(), 2);
    });
}

#[test]
fn empty_post_errors_and_does_not_emit() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();

        let result = core.runtime().block_on(core.post("   ".into(), None));
        assert!(result.is_err(), "empty post should error");

        let changes = drain(&mut rx);
        assert!(
            changes.is_empty(),
            "failed post must not emit; got {changes:?}"
        );
    });
}

#[test]
fn post_requires_no_account_or_credential() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        // No account configured and no credentials in the wallet: posting
        // still succeeds (only a response request would need funding).
        let result = core
            .runtime()
            .block_on(core.post("A thought needs no funding".into(), None));
        assert!(
            result.is_ok(),
            "post must succeed with no account/credential; got {result:?}"
        );
    });
}

// ===========================================================================
// edit_post() — append a generation (human-side collaborative edit, wave 5)
//
// Exercises the 5.1 generation machinery end-to-end (supersedes chain +
// item_current resolution): an edit appends a new generation, the default view
// resolves to it, the prior generation is preserved, and the listing counts the
// item once (editing does not inflate message_count).
// ===========================================================================

#[test]
fn edit_post_appends_generation_replacing_default_view() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let first = core
            .runtime()
            .block_on(core.post("draft one".into(), None))
            .unwrap();

        let mut rx = core.subscribe_changes();
        let edited = core
            .runtime()
            .block_on(core.edit_post(first.action_id.clone(), "draft two".into()))
            .unwrap();

        // Same item, a new action (a generation, not a replacement-in-place).
        assert_eq!(edited.item_id, first.item_id);
        assert_ne!(edited.action_id, first.action_id);

        let changes = drain(&mut rx);
        assert!(
            changes.iter().any(|c| matches!(c, Change::Space(_))),
            "edit should emit Space(id); got {changes:?}"
        );
        assert!(
            changes.contains(&Change::SpaceIndex),
            "edit may change the listing snippet; got {changes:?}"
        );

        // The default view resolves to the current generation only.
        let msgs = core
            .runtime()
            .block_on(core.get_space_messages(first.space_id.clone()))
            .unwrap();
        assert_eq!(
            msgs.len(),
            1,
            "edit replaces in the default view, not appends"
        );
        assert_eq!(msgs[0].content, "draft two");

        // The listing counts the item once — editing must not inflate the count.
        let spaces = core.runtime().block_on(core.list_spaces(false)).unwrap();
        assert_eq!(spaces.len(), 1);
        assert_eq!(
            spaces[0].message_count, 1,
            "edit must not inflate message_count; got {}",
            spaces[0].message_count
        );
    });
}

#[test]
fn edit_post_keeps_tree_position_and_rethreads_replies() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        // A question with a reply threaded onto it (post links to the tail).
        let first = core
            .runtime()
            .block_on(core.post("original question".into(), None))
            .unwrap();
        let second = core
            .runtime()
            .block_on(core.post("a reply".into(), Some(first.space_id.clone())))
            .unwrap();

        // Edit the *first* post — a new generation of its item, created later
        // than everything else in the space.
        let edited = core
            .runtime()
            .block_on(core.edit_post(first.action_id.clone(), "edited question".into()))
            .unwrap();

        // The render tree: the edited post stays exactly where the item has
        // always been (the root, first), and the reply re-threads under the
        // edit via item identity — not dangling as a sibling root branch (the
        // pre-fix rendering: the edit floated to the end as a new branch).
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(first.space_id.clone()))
            .unwrap();
        assert_eq!(tree.len(), 2, "two posts render; got {tree:#?}");
        assert_eq!(tree[0].action_id, edited.action_id, "edit stays in place");
        assert_eq!(tree[0].parent_action_id, None);
        assert_eq!(tree[0].generation, 1);
        assert_eq!(tree[0].blocks[0].text.as_deref(), Some("edited question"));
        assert_eq!(tree[1].action_id, second.action_id);
        assert_eq!(
            tree[1].parent_action_id.as_deref(),
            Some(edited.action_id.as_str()),
            "the reply re-threads under the item's current tip"
        );
        assert_eq!(tree[1].depth, 0, "the spine stays flat — no branch");
        assert!(!tree[1].is_branch);

        // The upstream-context view keeps the item's position too — the model
        // must see the edited text where the original stood, not appended.
        let msgs = core
            .runtime()
            .block_on(core.get_space_messages(first.space_id.clone()))
            .unwrap();
        let contents: Vec<&str> = msgs.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, vec!["edited question", "a reply"]);
    });
}

#[test]
fn edit_post_on_unknown_action_errors_without_emit() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.edit_post("no-such-action".into(), "x".into()));
        assert!(result.is_err());

        let changes = drain(&mut rx);
        assert!(
            changes.is_empty(),
            "failed edit must not emit; got {changes:?}"
        );
    });
}

// ===========================================================================
// UpdateState domain
// ===========================================================================

#[test]
fn accept_changed_claims_emits_update_state() {
    let (core, _dir) = make_core();
    let mut rx = core.subscribe_changes();

    // accept_changed_claims always persists state (even with no prior check).
    core.accept_changed_claims("v1.2.3".into(), "abc123".into())
        .unwrap();
    let changes = drain(&mut rx);
    assert!(
        changes.contains(&Change::UpdateState),
        "accept_changed_claims should emit UpdateState; got {changes:?}"
    );
}

// ===========================================================================
// Two-subscriber test
// ===========================================================================

#[test]
fn two_subscribers_both_receive() {
    let (core, _dir) = make_core();
    let mut rx1 = core.subscribe_changes();
    let mut rx2 = core.subscribe_changes();

    core.set_base_url("https://example.com".into()).unwrap();

    let c1 = drain(&mut rx1);
    let c2 = drain(&mut rx2);

    assert!(
        c1.contains(&Change::Config),
        "subscriber 1 should receive Config; got {c1:?}"
    );
    assert!(
        c2.contains(&Change::Config),
        "subscriber 2 should receive Config; got {c2:?}"
    );
}

#[test]
fn two_subscribers_both_receive_async() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx1 = core.subscribe_changes();
        let mut rx2 = core.subscribe_changes();

        core.runtime()
            .block_on(core.create_space(Some("test".into())))
            .unwrap();

        let c1 = drain(&mut rx1);
        let c2 = drain(&mut rx2);

        assert!(
            c1.contains(&Change::SpaceIndex),
            "subscriber 1 should receive SpaceIndex; got {c1:?}"
        );
        assert!(
            c2.contains(&Change::SpaceIndex),
            "subscriber 2 should receive SpaceIndex; got {c2:?}"
        );
    });
}

// ===========================================================================
// Multiple domains from one operation
// ===========================================================================

#[test]
fn set_account_credentials_followed_by_reset_emits_config_each_time() {
    let (core, _dir) = make_core();
    let mut rx = core.subscribe_changes();

    core.set_account_credentials("id1".into(), "sec1".into())
        .unwrap();
    core.reset_account().unwrap();

    let changes = drain(&mut rx);
    let config_count = changes.iter().filter(|c| **c == Change::Config).count();
    assert_eq!(
        config_count, 2,
        "each config write emits once; got {changes:?}"
    );
}

// ===========================================================================
// Deduplication sanity: subscribe after writes receives nothing
// ===========================================================================

#[test]
fn late_subscriber_does_not_see_past_events() {
    let (core, _dir) = make_core();

    core.set_base_url("https://example.com".into()).unwrap();

    // Subscribe AFTER the write.
    let mut rx = core.subscribe_changes();
    let changes = drain(&mut rx);
    assert!(
        changes.is_empty(),
        "late subscriber must not see prior events; got {changes:?}"
    );
}
