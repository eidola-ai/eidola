//! **Deciding at the write means reserving the writer.**
//!
//! Several of app-core's transactions read before they write: they check a
//! scope, a membership, a `left_at`, and choose what to write from what they
//! found (task 36's promotion guard, task 37's promote-or-join grant, the
//! notebook-owner removal guard). That shape is only worth anything if the read
//! and the write are one act against one snapshot — and a *deferred* `BEGIN`
//! (what plain `BEGIN` means here as in SQLite) reserves nothing.
//!
//! Measured on turso at our pin, two live connections on one `Database`
//! (`AppCore::db_conn` mints a fresh connection per call, so this is an
//! in-process race, not merely a cross-process one):
//!
//! * `BEGIN` — A reads, B writes and commits without blocking A, and A's write
//!   comes back `BusySnapshot("database snapshot is stale, rollback and retry
//!   the transaction")`. `busy_timeout` cannot rescue it: a stale snapshot is
//!   not something you can wait out.
//! * `BEGIN IMMEDIATE` — A reserves the writer; B waits on its own `BEGIN
//!   IMMEDIATE` for exactly A's hold, then acquires and reads A's committed
//!   state, deciding against *that*.
//!
//! So every transaction goes through `db::begin_write`. This file pins the
//! consequence at the layer the finding is about (Codex review, PR #280).

use eidola_app_core::db;

/// The concurrent-window case the grant's transaction exists for: a promotion
/// holds the writer while a grant starts. The grant must **wait** and then take
/// the join branch against the promoted row — never fail with a stale-snapshot
/// error, and never write a second promotion.
#[test]
fn a_grant_that_meets_a_writer_waits_and_decides_against_what_it_finds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("runtime");

    runtime.block_on(async move {
        let database = db::open(dir.path()).await.expect("open");
        let setup = db::connect(&database).await.expect("connect");

        db::insert_space(&setup, "home", Some("Home"), "unlinked", 1)
            .await
            .expect("home space");
        db::insert_space(&setup, "here", Some("Here"), "unlinked", 1)
            .await
            .expect("destination space");
        db::insert_participant(
            &setup,
            "agent",
            "space",
            Some("home"),
            None,
            "agent",
            "Mara",
            None,
            None,
            "explicit",
            "member",
            None,
            1,
        )
        .await
        .expect("a space-owned agent");

        // A competing writer — another window's promotion — holds the writer
        // for a beat, exactly as a real transaction does between its first
        // statement and its commit.
        let holder = db::connect(&database).await.expect("connect");
        holder
            .execute("BEGIN IMMEDIATE", ())
            .await
            .expect("the competing writer reserves");
        holder
            .execute(
                "UPDATE participant SET scope = 'global', owner_space_id = NULL WHERE id = 'agent'",
                (),
            )
            .await
            .expect("the promotion's own write");

        let granting = db::connect(&database).await.expect("connect");
        let grant = tokio::spawn(async move {
            let started = std::time::Instant::now();
            let decision = db::grant_space_membership_tx(
                &granting,
                "here",
                "agent",
                "observer",
                "notebook-id",
                2,
            )
            .await;
            (decision, started.elapsed())
        });

        // Let the grant reach its `BEGIN IMMEDIATE` and block there, then let
        // the promotion land.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        holder
            .execute("COMMIT", ())
            .await
            .expect("the promotion commits");

        let (decision, waited) = grant.await.expect("the grant task");
        let decision = decision.expect(
            "the grant waited for the writer rather than failing on a stale snapshot — \
             a deferred BEGIN answers BusySnapshot here",
        );
        assert_eq!(
            decision,
            db::GrantDecision::Joined,
            "it decided against the row it found: already global, so plain membership"
        );
        assert!(
            waited >= std::time::Duration::from_millis(150),
            "it really did wait for the other writer (waited {waited:?})"
        );

        // One promotion, one membership — the join did not promote again, and
        // the notebook the promoting branch would have minted does not exist.
        let member = db::space_participants(&setup, "here")
            .await
            .expect("roster")
            .into_iter()
            .find(|m| m.participant_id == "agent")
            .expect("a member of the destination");
        assert_eq!(member.role, "observer");
        assert!(
            db::get_space(&setup, "notebook-id")
                .await
                .expect("read")
                .is_none(),
            "the join branch mints no notebook"
        );
    });
}
