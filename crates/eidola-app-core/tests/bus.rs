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
//! The partial-failure emissions in `chat` and `chat_stream` require an HTTP
//! server fixture to exercise end-to-end. The rule (`docs/architecture/state.md`,
//! "every write emits"): **every explicit error exit AFTER the user-action
//! commit emits `Change::Space(id)` plus `Change::SpaceIndex` when
//! `is_new_space || auto_titled`**, mirroring the non-2xx arm — and *also*
//! `Change::Record` once the request row is committed. The complete exit-point
//! map (both functions now reorder the new-space row insert to AFTER credential
//! resolution, so the rows below exist only once spendability is known):
//!
//! The rows marked **`chat_path.rs`** are now *executed* against the in-process
//! mock-upstream harness (`tests/chat_harness/`), which drives the real `chat` /
//! `chat_stream` HTTP paths via the `with_test_http_client` seam and asserts
//! BOTH the typed/wrapped error AND the emitted `Change`s — turning these from
//! asserted-by-inspection into regression-gated.
//!
//! | Exit point | Writes committed | Emissions | Tested here |
//! |---|---|---|---|
//! | Pre-space failure (config, `NoAccount`, `InsufficientBalance`, zero-charge, `ensure_spendable_credential`) | **none** (new-space insert is deferred) | none — pure error, no orphan space | bus unit tests assert "no emit on error"; `NoAccount` / `InsufficientBalance` executed in `chat_path.rs` |
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
//! `SpaceIndex?` = emitted only when `is_new_space || auto_titled`. Plain `?` on
//! the intervening local-DB action/content/antecedent inserts stays *unemitted*
//! — those are internal-consistency (kill-`-9`-class) failures, not durable
//! partial state a subscriber needs to reconcile.
//!
//! **Failure-path id adoption (item C).** Every error returned *after* the
//! new-space row is persisted is wrapped as `AppError::ChatFailed { space_id }`
//! (its `Display` defers to the source, so messages don't regress). This lets a
//! blank GUI `Space` (id=`None`) learn its persisted id on failure even though
//! no `ChatResult` was produced. Pre-space errors stay unwrapped. Unit-tested in
//! `error.rs` (`chat_failed_display_defers_to_source`, `root_unwraps_*`,
//! `chat_space_id_only_on_wrapper`).
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
