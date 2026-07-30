//! Loud DB contention: the process-lifetime lock on the local database.
//!
//! Turso is single-writer. Two processes opening one `eidola.db` (the open GUI
//! app plus an `eidola …` CLI invocation is the everyday case) used to contend
//! for the file with no honest signal. `AppCore` now takes an exclusive
//! advisory lock on a sidecar lockfile at open, so the second opener is refused
//! at construction with a typed [`AppError::DatabaseInUse`] naming the holder.
//!
//! These tests exercise that from the `AppCore` surface (the contract callers
//! see) plus the lockfile mechanics that back the error message. `flock`
//! conflicts across separate `open()`s in the *same* process too — each `open`
//! is its own file description — so a second `AppCore` here is refused exactly
//! as a second process would be, which is what makes the contract testable
//! in-process at all.

use eidola_app_core::{AppCore, db, error::AppError};

fn dirs() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    (dir, config_dir, data_dir)
}

/// The single-opener path is unchanged: construction succeeds and leaves the
/// lockfile next to the database, stamped with this process's pid.
#[test]
fn a_single_open_succeeds_and_stamps_the_lockfile() {
    let (_dir, config_dir, data_dir) = dirs();

    let core = AppCore::new(config_dir, data_dir.clone()).expect("first open");

    let lock_path = data_dir.join(db::LOCK_FILE_NAME);
    assert!(
        lock_path.exists(),
        "the lockfile must sit next to the database at {}",
        lock_path.display()
    );
    let recorded = std::fs::read_to_string(&lock_path).expect("read lockfile");
    assert_eq!(
        recorded.trim(),
        std::process::id().to_string(),
        "the holder writes its own pid so the next contender can name it"
    );

    drop(core);
}

/// The contention case: a second open on the same data dir is refused with the
/// typed variant, and the message names the holder's pid.
#[test]
fn a_second_open_on_one_data_dir_is_refused_and_names_the_holder() {
    let (_dir, config_dir, data_dir) = dirs();

    let _held = AppCore::new(config_dir.clone(), data_dir.clone()).expect("first open");

    // `AppCore` isn't `Debug`, so unwrap the Result by hand.
    let err = match AppCore::new(config_dir, data_dir) {
        Ok(_) => panic!("second open must be refused while the first is held"),
        Err(e) => e,
    };

    let pid = match &err {
        AppError::DatabaseInUse { pid, .. } => *pid,
        other => panic!("expected AppError::DatabaseInUse, got {other:?}"),
    };
    assert_eq!(
        pid,
        Some(std::process::id()),
        "the refusal carries the holder's pid"
    );
    // The rendered message is what the CLI prints and the GUI would show, so
    // it must name the pid on its own — not only via the structured field.
    let rendered = err.to_string();
    assert!(
        rendered.contains(&std::process::id().to_string()),
        "the message must name the holder pid, got: {rendered}"
    );
}

/// Contention is keyed on the *data* dir (where the database lives), not the
/// config dir — two cores with separate configs still collide on one database.
#[test]
fn contention_is_keyed_on_the_data_dir_not_the_config_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().join("data");

    let _held = AppCore::new(dir.path().join("config-a"), data_dir.clone()).expect("first open");
    match AppCore::new(dir.path().join("config-b"), data_dir) {
        Ok(_) => panic!("same database, different config dir, must still be refused"),
        Err(e) => assert!(matches!(e, AppError::DatabaseInUse { .. }), "{e:?}"),
    }
}

/// Disjoint data dirs never contend — the property test parallelism relies on
/// (every test builds its core over its own tempdir).
#[test]
fn separate_data_dirs_do_not_contend() {
    let (_dir_a, config_a, data_a) = dirs();
    let (_dir_b, config_b, data_b) = dirs();

    let _a = AppCore::new(config_a, data_a).expect("core a");
    let _b = AppCore::new(config_b, data_b).expect("core b");
}

/// Dropping the core releases the lock, so the next run reopens the same data
/// dir cleanly. This is the *restart* path — and, because an advisory lock is
/// held by the open file description, the same thing the kernel does for us
/// when a process crashes without dropping anything.
#[test]
fn the_lock_releases_on_drop_so_a_restart_reopens() {
    let (_dir, config_dir, data_dir) = dirs();

    let core = AppCore::new(config_dir.clone(), data_dir.clone()).expect("first open");
    drop(core);

    let reopened = AppCore::new(config_dir, data_dir).expect("reopen after the holder dropped");
    drop(reopened);
}

/// A crash cannot wedge the lock: the lockfile surviving on disk (with a stale
/// pid in it) means nothing on its own — only a live holder's open descriptor
/// does. Simulated by leaving a lockfile behind with a bogus pid and no holder.
#[test]
fn a_leftover_lockfile_with_a_stale_pid_does_not_wedge_the_lock() {
    let (_dir, config_dir, data_dir) = dirs();

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join(db::LOCK_FILE_NAME), "999999").unwrap();

    let core = AppCore::new(config_dir, data_dir).expect("a stale lockfile must not block startup");
    drop(core);
}

/// The pid is advisory, never authoritative: an unreadable/garbage pid still
/// refuses the second opener — it just says less.
#[test]
fn an_unparseable_pid_still_refuses_but_reports_no_pid() {
    let (_dir, _config_dir, data_dir) = dirs();

    let held = db::DbLock::acquire(&data_dir).expect("acquire");
    // Overwrite the stamped pid behind the holder's back (a holder killed
    // mid-write leaves the same observable state).
    std::fs::write(held.path(), "not-a-pid").unwrap();

    match db::DbLock::acquire(&data_dir) {
        Err(AppError::DatabaseInUse { pid, message }) => {
            assert_eq!(pid, None, "an unparseable pid is reported honestly as none");
            assert!(
                message.contains("another Eidola process"),
                "the refusal is still named honestly: {message}"
            );
        }
        other => panic!("expected DatabaseInUse, got {other:?}"),
    }
}
