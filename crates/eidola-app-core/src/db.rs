use std::path::{Path, PathBuf};

use turso::{Builder, Connection, Database, Value};

use crate::error::AppError;
use crate::subspaces::{
    MAX_LIVE_SUBSPACES_PER_OWNER, MAX_SPAWN_DEPTH, MAX_SUBAGENTS_PER_SPAWN, SpawnRefusal,
};

const SCHEMA: &str = include_str!("../schema/schema.sql");

/// Name of the sidecar advisory lockfile held next to `eidola.db` for the
/// lifetime of an `AppCore`. See [`DbLock`].
pub const LOCK_FILE_NAME: &str = "eidola.db.lock";

/// Current schema version. `schema.sql` is the whole baseline — there are no
/// incremental migrations (the pre-release history was collapsed for the
/// Participants v1 fresh start; there are no real users). A database whose
/// `user_version` is neither `0` (fresh) nor this value is from an
/// incompatible build and [`initialize`] refuses to open it (delete the dev
/// database; see the error text). Bump this on every fresh-start reset so
/// stale databases are detected rather than silently limping.
const LATEST_VERSION: i64 = 9;

/// Well-known id of the shared human "User" participant — the single
/// participant row joined into every space (agent participants are per-space
/// instances; the human is the one shared identity). Seeded idempotently at
/// every DB open; stable so `INSERT OR IGNORE` re-seeds are no-ops and human
/// actions across all spaces reference one row.
pub const HUMAN_PARTICIPANT_ID: &str = "00000000-0000-7000-8000-000000000001";

/// Well-known id of the harness itself as a participant — a **global**,
/// `kind = 'system'` row that authors the actions Eidola writes on its own
/// behalf rather than on anyone's: today the branch summaries of
/// [`crate::summaries`]. It is deliberately never *referenced* into a space,
/// so it is not a member anywhere: it never appears in a participant list,
/// never gets planned a turn, never authors a post. Attributing these to the
/// human "User" or to an agent participant would be a lie in the Record.
pub const SYSTEM_PARTICIPANT_ID: &str = "00000000-0000-7000-8000-000000000002";

/// Well-known id of the seeded "Default" template's single agent participant
/// (seed-only; user-created template participants get fresh UUIDv7s). Written
/// once, when the template row is created — the id is a stable handle for
/// tests, **not** an idempotency key: re-seeding by id is exactly the bug
/// [`ensure_default_participants`] documents.
const DEFAULT_TEMPLATE_AGENT_ID: &str = "00000000-0000-7000-8000-000000000011";

/// The generic system prompt the seeded default agent participant carries.
const DEFAULT_AGENT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer clearly and concisely.";

/// A readable default label derived from a model reference (e.g.
/// `gemma4-31b` → `Gemma4 31b`). Only a *default* — the user edits it — so
/// title-casing hyphen tokens is enough; it never needs to be canonical.
pub fn default_agent_label(model_ref: &str) -> String {
    // Drop any `@backend` suffix, then title-case hyphen/underscore tokens.
    let model = model_ref
        .rsplit_once('@')
        .map(|(m, _)| m)
        .unwrap_or(model_ref);
    let words: Vec<String> = model
        .split(['-', '_'])
        .filter(|s| !s.is_empty())
        .map(|tok| {
            let mut chars = tok.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect();
    if words.is_empty() {
        model.to_string()
    } else {
        words.join(" ")
    }
}

/// An exclusive advisory lock on `<data_dir>/eidola.db.lock`, held for the
/// lifetime of the owning `AppCore`.
///
/// **Why.** Turso is single-writer. Two processes opening the same
/// `eidola.db` (an open GUI app plus a `eidola …` CLI invocation is the
/// everyday case) contend for the file with no honest signal — writes fail or
/// misbehave silently. The lock makes that state loud: the second opener is
/// refused at construction with [`AppError::DatabaseInUse`] naming the holder.
///
/// **Why `flock` and not a pidfile-existence check.** An advisory lock is held
/// by the *open file description*, so the kernel drops it when the process
/// exits — however it exits. A crash therefore cannot wedge a stale lock and
/// lock out every future run; the file's mere existence means nothing. Do not
/// replace this with "does the file exist / is that pid alive" logic — that
/// property is the whole reason for the mechanism.
///
/// The lockfile's *contents* are only the holder's pid, written after the lock
/// is taken, purely so the refusal message can name a process. It is advisory
/// (never authoritative): a contender that reads a truncated/absent pid still
/// refuses, it just says less.
///
/// Note `flock` conflicts across separate `open()`s in the *same* process too
/// (each `open` is its own file description), so a second `AppCore` on one data
/// dir inside one process is refused exactly like a second process — which is
/// what the tests rely on.
pub struct DbLock {
    /// Holding the `File` open holds the lock; dropping it closes the
    /// descriptor and releases. The lockfile itself is left on disk (an
    /// unlocked file is meaningless, and unlinking it would race a contender
    /// that already opened it).
    _file: std::fs::File,
    path: PathBuf,
}

impl DbLock {
    /// Take the exclusive lock for `data_dir`, creating the directory and the
    /// lockfile if needed. Fails with [`AppError::DatabaseInUse`] if another
    /// process (or another `AppCore` in this one) already holds it.
    pub fn acquire(data_dir: &Path) -> Result<Self, AppError> {
        use fs4::{FileExt, TryLockError};

        std::fs::create_dir_all(data_dir).map_err(|e| AppError::Database {
            message: format!("failed to create data directory: {e}"),
        })?;
        let path = data_dir.join(LOCK_FILE_NAME);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| AppError::Database {
                message: format!("failed to open database lockfile {}: {e}", path.display()),
            })?;

        match FileExt::try_lock(&file) {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                let pid = read_lock_pid(&path);
                let who = match pid {
                    Some(pid) => format!("another Eidola process (pid {pid})"),
                    None => "another Eidola process".to_string(),
                };
                return Err(AppError::DatabaseInUse {
                    pid,
                    message: format!(
                        "{who} has this database open ({}). \
                         Quit it and try again.",
                        data_dir.join("eidola.db").display()
                    ),
                });
            }
            Err(TryLockError::Error(e)) => {
                return Err(AppError::Database {
                    message: format!("failed to lock database ({}): {e}", path.display()),
                });
            }
        }

        // We hold the lock — record who, for the next contender's message.
        // Best-effort: a lockfile we can't write is still a lock we hold, and
        // the contender's refusal is correct with or without a pid.
        let _ = write_lock_pid(&file);

        Ok(Self { _file: file, path })
    }

    /// Path of the lockfile this guard holds (diagnostics/tests).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for DbLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DbLock").field("path", &self.path).finish()
    }
}

/// Stamp our pid into the (already locked) lockfile.
fn write_lock_pid(file: &std::fs::File) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut file = file;
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    write!(file, "{}", std::process::id())?;
    file.flush()
}

/// Read the holder's pid from a lockfile. `None` for an empty/garbage file —
/// the holder may have been mid-write, which is honest to report as "unknown"
/// rather than guessed at.
fn read_lock_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Opens (or creates) the local database at `data_dir/eidola.db` and runs any
/// pending migrations.
pub async fn open(data_dir: &Path) -> Result<Database, AppError> {
    let path = data_dir.join("eidola.db");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| AppError::Database {
            message: format!("failed to create data directory: {e}"),
        })?;
    }

    let db = Builder::new_local(path.to_string_lossy().as_ref())
        .build()
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to open database: {e}"),
        })?;

    let conn = connect(&db).await?;
    initialize(&conn).await?;

    Ok(db)
}

/// Open a **write** transaction — `BEGIN IMMEDIATE`, reserving the writer
/// before the first read.
///
/// Every transaction in this module reads before it writes: it checks a scope,
/// a membership, a `left_at`, and decides what to write from what it found.
/// That is the whole point of deciding at the write — and a *deferred* `BEGIN`
/// (what plain `BEGIN` means, here as in SQLite) does not reserve anything, so
/// two connections could both read the old state and the second one's write
/// would be refused rather than serialized. Measured on turso at our pin, with
/// two live connections on one `Database` (`AppCore::db_conn` mints a fresh one
/// per call, so this is an in-process race, not just a cross-process one):
///
/// ```text
/// BEGIN            A reads scope=space
///                  B begins, writes scope=global, commits   (A is not blocked)
///                  A writes -> Err(BusySnapshot("database snapshot is stale"))
/// BEGIN IMMEDIATE  A reserves the writer, reads, writes
///                  B waits on its own BEGIN IMMEDIATE (328ms, the hold)
///                  …then acquires and reads scope=global — the join branch
/// ```
///
/// `busy_timeout` does not rescue the deferred case: a stale snapshot cannot be
/// waited out, only retried, and the whole decision would have to be re-taken.
/// Reserving up front converts the race into an ordinary wait — the loser reads
/// the winner's committed state and decides against *that*, which is what the
/// grant's promote-or-join branch (and #279's promote / retire / remove
/// guards) were always meant to do (Codex review, PR #280).
pub(crate) async fn begin_write(conn: &Connection) -> Result<(), AppError> {
    conn.execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(AppError::db)?;
    Ok(())
}

/// Open a connection with FK enforcement enabled. turso defaults
/// `foreign_keys` **OFF** (every `REFERENCES`/composite FK is then unenforced
/// documentation), and the scope-owned participant model relies on the tuple
/// FKs + CHECKs actually firing — so enable it on **every** connection, since
/// the pragma is per-connection. See `turso_enforcement_smoke` for the proof
/// that this build enforces single/composite FKs, MATCH-SIMPLE NULL-skip, and
/// CHECKs once it's on.
pub async fn connect(db: &Database) -> Result<Connection, AppError> {
    let conn = db.connect().map_err(AppError::db)?;
    conn.execute("PRAGMA foreign_keys = ON", ())
        .await
        .map_err(AppError::db)?;
    // turso is single-writer, and defaults to erroring immediately ("database
    // is locked") when a second connection tries to write concurrently.
    // Participants v1 (wave 2) drives concurrent turns — a multi-participant
    // fan-out runs several `respond_stream_as` at once, each writing its own
    // rows — so give every connection a bounded busy wait: a concurrent writer
    // blocks until the current write commits instead of failing. The
    // wallet-level `spend_gate` already serializes the *credential* step; this
    // covers the remaining per-turn row writes (posts, inference rows).
    conn.execute("PRAGMA busy_timeout = 5000", ())
        .await
        .map_err(AppError::db)?;
    Ok(conn)
}

/// Initialize the database. A fresh (`user_version == 0`) database gets
/// `schema.sql` applied directly and stamped at [`LATEST_VERSION`]. There are
/// no incremental migrations (the Participants v1 fresh start reset the
/// baseline), so a database already at [`LATEST_VERSION`] just re-seeds its
/// idempotent defaults; any *other* non-zero version is from an incompatible
/// pre-release build and is refused with an honest "delete your dev database"
/// error rather than limping against a schema it doesn't match.
async fn initialize(conn: &Connection) -> Result<(), AppError> {
    let version = get_user_version(conn).await?;

    if version == 0 {
        conn.execute_batch(SCHEMA)
            .await
            .map_err(|e| AppError::Database {
                message: format!("schema init failed: {e}"),
            })?;
        set_user_version(conn, LATEST_VERSION).await?;
    } else if version != LATEST_VERSION {
        return Err(AppError::Database {
            message: format!(
                "your local Eidola database is from an incompatible build \
                 (schema v{version}; this build expects v{LATEST_VERSION}). \
                 There are no real users yet, so there is no migration — \
                 delete your dev database and restart: \
                 ~/Library/Application Support/eidola/eidola.db \
                 (or the eidola.db in your data directory)."
            ),
        });
    }

    // The idempotent defaults exist on every database — fresh installs and
    // already-current ones alike. They simply run on every open.
    ensure_default_backends(conn).await?;
    ensure_default_participants(conn).await?;

    Ok(())
}

/// Seed the shared human "User" participant and the "Default" space template
/// (with its single agent participant) if missing. Runs on every open.
///
/// **A seed makes a *fresh* database usable; it never re-asserts state over a
/// user's later edits.** The distinction a seed must draw is "this row has
/// never existed" vs. "the user removed it", and `INSERT OR IGNORE` on a
/// well-known id only draws it for rows nothing can hard-delete. That holds for
/// the globals and the template row itself — participants soft-remove, spaces
/// leave by reference, and the built-in template refuses removal, so the row
/// always survives and the re-seed is a genuine no-op.
///
/// It did **not** hold for the template's owned agent: `update_template`
/// replaces a template's owned participants wholesale (hard delete + re-insert
/// with fresh ids), so after *any* save to the Default template — a removal, an
/// addition, even a plain rename — nothing sat at the well-known id and the
/// next open injected the seeded agent back into the template, and from there
/// into every new space (task 41). So that row is seeded **only in the same
/// call that creates the template**: after the first open there is no path here
/// that can touch a template's participants at all, which makes the
/// re-injection unrepresentable rather than guarded.
///
/// (Nothing here ever needs to backfill an existing database: schema changes
/// bump [`LATEST_VERSION`] and old databases are refused, not migrated.)
///
/// **Partial-seed safety.** `conn.execute` autocommits, so a process exit
/// between two seed writes leaves the earlier one committed. Every seed here is
/// independently keyed on its own well-known id and therefore self-heals on the
/// next open — *except* the template + agent pair, whose second row is keyed on
/// the first's creation. That group, and only that group, is written in one
/// transaction ([`ensure_default_template_tx`]).
async fn ensure_default_participants(conn: &Connection) -> Result<(), AppError> {
    let now = crate::now_ms();

    // The one shared human "User" — a GLOBAL participant, referenced into every
    // instantiated space (instantiate_template ensures the reference).
    conn.execute(
        "INSERT OR IGNORE INTO participant \
         (id, scope, kind, label, notify_policy, role, created_at) \
         VALUES (?1, 'global', 'human', 'User', 'explicit', 'owner', ?2)",
        (
            Value::Text(HUMAN_PARTICIPANT_ID.to_string()),
            Value::Integer(now),
        ),
    )
    .await
    .map_err(AppError::db)?;

    // Eidola itself, for the actions it writes on its own behalf (branch
    // summaries). Global, but referenced into no space — see the const.
    conn.execute(
        "INSERT OR IGNORE INTO participant \
         (id, scope, kind, label, notify_policy, role, created_at) \
         VALUES (?1, 'global', 'system', 'Eidola', 'explicit', 'observer', ?2)",
        (
            Value::Text(SYSTEM_PARTICIPANT_ID.to_string()),
            Value::Integer(now),
        ),
    )
    .await
    .map_err(AppError::db)?;

    // The "Default" template and its agent — created together or not at all.
    ensure_default_template_tx(conn, now).await
}

/// Create the "Default" template **and** its owned agent, atomically.
///
/// The two rows are one unit because the second is keyed on the first's
/// *creation*, and `conn.execute` autocommits: without a transaction, a process
/// exit (or a failing second write) between them would commit the template row
/// alone and consume the creation signal — every later open would then see zero
/// changes and skip the agent seed **permanently**, leaving new spaces with no
/// assistant.
///
/// It is a transaction rather than a repair heuristic on purpose. The
/// post-crash state ("template row, no owned agents, no user edits") is
/// **indistinguishable** from a user who deliberately emptied the template, so
/// any code that repaired it would resurrect exactly what
/// [`ensure_default_participants`] exists to stop. Rolling back makes the state
/// unrepresentable instead: there is nothing to detect, so nothing to
/// misinterpret, and the seed simply stays retryable on the next open.
async fn ensure_default_template_tx(conn: &Connection, now: i64) -> Result<(), AppError> {
    begin_write(conn).await?;
    match ensure_default_template_tx_body(conn, now).await {
        Ok(()) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(())
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn ensure_default_template_tx_body(conn: &Connection, now: i64) -> Result<(), AppError> {
    // The template row (must exist before its owned agent's owner FK resolves).
    // The rows-changed count is the creation signal: 0 means the template was
    // already here, and everything about it from that moment on is the user's.
    let template_created = conn
        .execute(
            "INSERT OR IGNORE INTO space_template \
             (id, title, cascade_limit, created_at) \
             VALUES (?1, 'Default', 4, ?2)",
            (
                Value::Text(crate::config::DEFAULT_TEMPLATE_ID.to_string()),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?
        > 0;
    if !template_created {
        return Ok(());
    }

    // …owning a single template-scoped agent participant (label from the
    // model's friendly name, model = the compiled DEFAULT_MODEL, a generic
    // prompt, notify 'human') — written once, with the template, and never
    // again. See `ensure_default_participants`.
    conn.execute(
        "INSERT INTO participant \
         (id, scope, owner_template_id, kind, label, model_ref, system_prompt, \
          notify_policy, role, created_at) \
         VALUES (?1, 'template', ?2, 'agent', ?3, ?4, ?5, 'human', 'member', ?6)",
        (
            Value::Text(DEFAULT_TEMPLATE_AGENT_ID.to_string()),
            Value::Text(crate::config::DEFAULT_TEMPLATE_ID.to_string()),
            Value::Text(default_agent_label(crate::config::DEFAULT_MODEL)),
            Value::Text(crate::config::DEFAULT_MODEL.to_string()),
            Value::Text(DEFAULT_AGENT_SYSTEM_PROMPT.to_string()),
            Value::Integer(now),
        ),
    )
    .await
    .map_err(AppError::db)?;

    Ok(())
}

/// Insert the `eidola` and `local` singleton backend rows if they are
/// missing. Runs on every open (idempotent), so both the fresh-install and
/// migration paths — and any database from before backends existed — end up
/// with them. Never overwrites: a user's enabled/display choices persist. The
/// seed-vs-user-edit hazard [`ensure_default_participants`] documents cannot
/// arise here — a singleton backend is enable/disable only, never removable, so
/// an absent row really is a fresh database.
async fn ensure_default_backends(conn: &Connection) -> Result<(), AppError> {
    let now = crate::now_ms();
    conn.execute(
        "INSERT OR IGNORE INTO backend (id, kind, display_name, enabled, created_at, updated_at) \
         VALUES ('eidola', 'eidola', 'Eidola', 1, ?1, ?1)",
        (Value::Integer(now),),
    )
    .await
    .map_err(AppError::db)?;
    conn.execute(
        "INSERT OR IGNORE INTO backend (id, kind, display_name, enabled, created_at, updated_at) \
         VALUES ('local', 'local', 'Local', 1, ?1, ?1)",
        (Value::Integer(now),),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

async fn get_user_version(conn: &Connection) -> Result<i64, AppError> {
    let mut stmt = conn
        .prepare("PRAGMA user_version")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let row = rows
        .next()
        .await
        .map_err(AppError::db)?
        .ok_or_else(|| AppError::Database {
            message: "no user_version row".into(),
        })?;
    row.get::<i64>(0).map_err(AppError::db)
}

async fn set_user_version(conn: &Connection, version: i64) -> Result<(), AppError> {
    conn.execute(&format!("PRAGMA user_version = {version}"), ())
        .await
        .map_err(AppError::db)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 0 — Wallet: Issuer key operations
// ---------------------------------------------------------------------------

pub async fn upsert_issuer_key(
    conn: &Connection,
    id: &str,
    params_hash: &str,
    public_key_data: &[u8],
    params_data: &[u8],
    expires_at: i64,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT OR IGNORE INTO issuer_key (id, params_hash, public_key_data, params_data, expires_at, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (
            Value::Text(id.to_string()),
            Value::Text(params_hash.to_string()),
            Value::Blob(public_key_data.to_vec()),
            Value::Blob(params_data.to_vec()),
            Value::Integer(expires_at),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to upsert issuer key: {e}"),
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 0 — Wallet: Pre-credential operations
// ---------------------------------------------------------------------------

pub async fn insert_pre_credential_issuance(
    conn: &Connection,
    id: &str,
    issuer_key_id: &str,
    data: &[u8],
    credits: i64,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO pre_credential (id, type, credential_nonce, issuer_key_id, data, credits, spend_amount, created_at) \
         VALUES (?1, 'issuance', NULL, ?2, ?3, ?4, NULL, ?5)",
        (
            Value::Text(id.to_string()),
            Value::Text(issuer_key_id.to_string()),
            Value::Blob(data.to_vec()),
            Value::Integer(credits),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert pre_credential: {e}"),
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_pre_credential_refund(
    conn: &Connection,
    id: &str,
    credential_nonce: &str,
    issuer_key_id: &str,
    data: &[u8],
    spend_amount: i64,
    spend_proof_data: &[u8],
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO pre_credential (id, type, credential_nonce, issuer_key_id, data, credits, spend_amount, spend_proof_data, created_at) \
         VALUES (?1, 'refund', ?2, ?3, ?4, NULL, ?5, ?6, ?7)",
        (
            Value::Text(id.to_string()),
            Value::Text(credential_nonce.to_string()),
            Value::Text(issuer_key_id.to_string()),
            Value::Blob(data.to_vec()),
            Value::Integer(spend_amount),
            Value::Blob(spend_proof_data.to_vec()),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert refund pre_credential: {e}"),
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 0 — Wallet: Credential operations
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub async fn insert_credential(
    conn: &Connection,
    nonce: &str,
    pre_credential_id: &str,
    issuer_key_id: &str,
    data: &[u8],
    credits: i64,
    generation: i64,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO credential (nonce, pre_credential_id, issuer_key_id, data, credits, generation, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        (
            Value::Text(nonce.to_string()),
            Value::Text(pre_credential_id.to_string()),
            Value::Text(issuer_key_id.to_string()),
            Value::Blob(data.to_vec()),
            Value::Integer(credits),
            Value::Integer(generation),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert credential: {e}"),
    })?;
    Ok(())
}

pub struct SpendableCredential {
    pub nonce: String,
    pub issuer_key_id: String,
    pub data: Vec<u8>,
    pub credits: i64,
    pub generation: i64,
    pub public_key_data: Vec<u8>,
}

pub async fn find_spendable_credential(
    conn: &Connection,
    min_credits: i64,
) -> Result<Option<SpendableCredential>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT c.nonce, c.issuer_key_id, c.data, c.credits, c.generation, ik.public_key_data \
             FROM credential_lifecycle cl \
             JOIN credential c ON c.nonce = cl.nonce \
             JOIN issuer_key ik ON ik.id = c.issuer_key_id \
             WHERE cl.state = 'active' AND c.credits >= ?1 \
             ORDER BY c.credits ASC \
             LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query([min_credits]).await.map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(SpendableCredential {
            nonce: row.get::<String>(0).map_err(AppError::db)?,
            issuer_key_id: row.get::<String>(1).map_err(AppError::db)?,
            data: row.get::<Vec<u8>>(2).map_err(AppError::db)?,
            credits: row.get::<i64>(3).map_err(AppError::db)?,
            generation: row.get::<i64>(4).map_err(AppError::db)?,
            public_key_data: row.get::<Vec<u8>>(5).map_err(AppError::db)?,
        })),
    }
}

pub struct CredentialRow {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub created_at: i64,
    pub state: String,
}

pub async fn list_active_credentials(conn: &Connection) -> Result<Vec<CredentialRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT nonce, credits, generation, created_at, state \
             FROM credential_lifecycle WHERE state = 'active' \
             ORDER BY created_at",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(CredentialRow {
            nonce: row.get::<String>(0).map_err(AppError::db)?,
            credits: row.get::<i64>(1).map_err(AppError::db)?,
            generation: row.get::<i64>(2).map_err(AppError::db)?,
            created_at: row.get::<i64>(3).map_err(AppError::db)?,
            state: row.get::<String>(4).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

pub struct SpendingCredentialRow {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub created_at: i64,
    pub spend_amount: i64,
    pub pre_credential_id: String,
    pub pre_refund_data: Vec<u8>,
    pub spend_proof_data: Vec<u8>,
    pub issuer_key_id: String,
    pub public_key_data: Vec<u8>,
}

pub async fn list_spending_credentials(
    conn: &Connection,
) -> Result<Vec<SpendingCredentialRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT c.nonce, c.credits, c.generation, c.created_at, \
                    pc.spend_amount, pc.id, pc.data, pc.spend_proof_data, \
                    pc.issuer_key_id, ik.public_key_data \
             FROM credential_lifecycle cl \
             JOIN credential c ON c.nonce = cl.nonce \
             JOIN pre_credential pc ON pc.credential_nonce = c.nonce AND pc.type = 'refund' \
             JOIN issuer_key ik ON ik.id = pc.issuer_key_id \
             WHERE cl.state = 'spending' \
             ORDER BY c.created_at",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(SpendingCredentialRow {
            nonce: row.get::<String>(0).map_err(AppError::db)?,
            credits: row.get::<i64>(1).map_err(AppError::db)?,
            generation: row.get::<i64>(2).map_err(AppError::db)?,
            created_at: row.get::<i64>(3).map_err(AppError::db)?,
            spend_amount: row.get::<i64>(4).map_err(AppError::db)?,
            pre_credential_id: row.get::<String>(5).map_err(AppError::db)?,
            pre_refund_data: row.get::<Vec<u8>>(6).map_err(AppError::db)?,
            spend_proof_data: row.get::<Vec<u8>>(7).map_err(AppError::db)?,
            issuer_key_id: row.get::<String>(8).map_err(AppError::db)?,
            public_key_data: row.get::<Vec<u8>>(9).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Provider operations
// ---------------------------------------------------------------------------

pub async fn ensure_provider(
    conn: &Connection,
    name: &str,
    kind: &str,
    created_at: i64,
) -> Result<String, AppError> {
    let mut stmt = conn
        .prepare("SELECT id FROM provider WHERE name = ?1 AND kind = ?2 LIMIT 1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(name.to_string()), Value::Text(kind.to_string())))
        .await
        .map_err(AppError::db)?;
    if let Some(row) = rows.next().await.map_err(AppError::db)? {
        return row.get::<String>(0).map_err(AppError::db);
    }
    drop(rows);
    drop(stmt);

    let id = uuid::Uuid::now_v7().to_string();
    conn.execute(
        "INSERT INTO provider (id, name, kind, created_at) VALUES (?1, ?2, ?3, ?4)",
        (
            Value::Text(id.clone()),
            Value::Text(name.to_string()),
            Value::Text(kind.to_string()),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert provider: {e}"),
    })?;
    Ok(id)
}

// ---------------------------------------------------------------------------
// Layer 1 — Backends: configured inference destinations
// ---------------------------------------------------------------------------

/// One row of the `backend` table. `model_overrides` stays the raw JSON
/// text here; `backends::BackendInfo` parses it for consumers.
#[derive(Clone, Debug)]
pub struct BackendRow {
    pub id: String,
    pub kind: String,
    pub display_name: String,
    pub enabled: bool,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub models_dir: Option<String>,
    pub model_overrides: Option<String>,
    /// `llamacpp` only: explicit `llama-server` path; `None` = discover it.
    pub engine_path: Option<String>,
    /// `llamacpp` only: may a request auto-start an engine? The `local`
    /// backend always auto-starts regardless.
    pub auto_start: bool,
    /// `eidola` only: JSON array of enclave-measurement overrides; `None`
    /// = the single build measurement pinned in the trust root.
    pub trusted_measurements: Option<String>,
    /// `eidola` only: PEM ARK certificate override; `None` = vendor chain.
    pub hardware_root_ca: Option<String>,
    /// `eidola` only: PEM ASK certificate override; `None` = vendor chain.
    pub hardware_intermediate_ca: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub removed_at: Option<i64>,
}

fn backend_row_from(row: &turso::Row) -> Result<BackendRow, AppError> {
    Ok(BackendRow {
        id: row.get::<String>(0).map_err(AppError::db)?,
        kind: row.get::<String>(1).map_err(AppError::db)?,
        display_name: row.get::<String>(2).map_err(AppError::db)?,
        enabled: row.get::<i64>(3).map_err(AppError::db)? != 0,
        base_url: row.get::<Option<String>>(4).map_err(AppError::db)?,
        api_key: row.get::<Option<String>>(5).map_err(AppError::db)?,
        models_dir: row.get::<Option<String>>(6).map_err(AppError::db)?,
        model_overrides: row.get::<Option<String>>(7).map_err(AppError::db)?,
        engine_path: row.get::<Option<String>>(8).map_err(AppError::db)?,
        auto_start: row.get::<i64>(9).map_err(AppError::db)? != 0,
        trusted_measurements: row.get::<Option<String>>(10).map_err(AppError::db)?,
        hardware_root_ca: row.get::<Option<String>>(11).map_err(AppError::db)?,
        hardware_intermediate_ca: row.get::<Option<String>>(12).map_err(AppError::db)?,
        created_at: row.get::<i64>(13).map_err(AppError::db)?,
        updated_at: row.get::<i64>(14).map_err(AppError::db)?,
        removed_at: row.get::<Option<i64>>(15).map_err(AppError::db)?,
    })
}

const BACKEND_COLUMNS: &str = "id, kind, display_name, enabled, base_url, api_key, \
     models_dir, model_overrides, engine_path, auto_start, trusted_measurements, \
     hardware_root_ca, hardware_intermediate_ca, created_at, updated_at, removed_at";

/// List backends, soft-removed rows excluded. Singletons first (eidola,
/// then local), then externals in creation order — the stable presentation
/// order both UIs use.
pub async fn list_backends(conn: &Connection) -> Result<Vec<BackendRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {BACKEND_COLUMNS} FROM backend WHERE removed_at IS NULL \
             ORDER BY CASE kind WHEN 'eidola' THEN 0 WHEN 'local' THEN 1 ELSE 2 END, \
             created_at, id"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut backends = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        backends.push(backend_row_from(&row)?);
    }
    Ok(backends)
}

/// Fetch one backend by id — including soft-removed rows (callers decide
/// whether `removed_at` matters; the chat router treats removed as absent,
/// while `insert_backend` uses the row to revive).
pub async fn get_backend(conn: &Connection, id: &str) -> Result<Option<BackendRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {BACKEND_COLUMNS} FROM backend WHERE id = ?1"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        Some(row) => Ok(Some(backend_row_from(&row)?)),
        None => Ok(None),
    }
}

/// Insert a new external backend, or **revive** a soft-removed row with the
/// same id (fully overwriting its configuration — forensic `request` rows
/// keep pointing at the same id, which is again a live backend of that
/// name). Fails if a live row already holds the id.
pub async fn insert_backend(conn: &Connection, row: &BackendRow) -> Result<(), AppError> {
    if let Some(existing) = get_backend(conn, &row.id).await? {
        if existing.removed_at.is_none() {
            return Err(AppError::Config {
                message: format!("a backend named `{}` already exists", row.id),
            });
        }
        conn.execute(
            "UPDATE backend SET kind = ?2, display_name = ?3, enabled = ?4, base_url = ?5, \
             api_key = ?6, models_dir = ?7, model_overrides = ?8, engine_path = ?9, \
             auto_start = ?10, trusted_measurements = ?11, hardware_root_ca = ?12, \
             hardware_intermediate_ca = ?13, updated_at = ?14, removed_at = NULL WHERE id = ?1",
            (
                Value::Text(row.id.clone()),
                Value::Text(row.kind.clone()),
                Value::Text(row.display_name.clone()),
                Value::Integer(row.enabled as i64),
                opt_text(&row.base_url),
                opt_text(&row.api_key),
                opt_text(&row.models_dir),
                opt_text(&row.model_overrides),
                opt_text(&row.engine_path),
                Value::Integer(row.auto_start as i64),
                opt_text(&row.trusted_measurements),
                opt_text(&row.hardware_root_ca),
                opt_text(&row.hardware_intermediate_ca),
                Value::Integer(row.updated_at),
            ),
        )
        .await
        .map_err(AppError::db)?;
        return Ok(());
    }
    conn.execute(
        "INSERT INTO backend (id, kind, display_name, enabled, base_url, api_key, \
         models_dir, model_overrides, engine_path, auto_start, trusted_measurements, \
         hardware_root_ca, hardware_intermediate_ca, created_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        (
            Value::Text(row.id.clone()),
            Value::Text(row.kind.clone()),
            Value::Text(row.display_name.clone()),
            Value::Integer(row.enabled as i64),
            opt_text(&row.base_url),
            opt_text(&row.api_key),
            opt_text(&row.models_dir),
            opt_text(&row.model_overrides),
            opt_text(&row.engine_path),
            Value::Integer(row.auto_start as i64),
            opt_text(&row.trusted_measurements),
            opt_text(&row.hardware_root_ca),
            opt_text(&row.hardware_intermediate_ca),
            Value::Integer(row.created_at),
            Value::Integer(row.updated_at),
        ),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

/// Flip a backend's enabled flag. Returns whether a live row was updated.
pub async fn set_backend_enabled(
    conn: &Connection,
    id: &str,
    enabled: bool,
    now: i64,
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE backend SET enabled = ?2, updated_at = ?3 \
             WHERE id = ?1 AND removed_at IS NULL",
            (
                Value::Text(id.to_string()),
                Value::Integer(enabled as i64),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

/// Update an external backend's configuration fields (each `Some` replaces;
/// the inner `Option` clears/sets nullable columns). Returns whether a live
/// row was updated.
#[allow(clippy::too_many_arguments)]
pub async fn update_backend_config(
    conn: &Connection,
    id: &str,
    display_name: Option<&str>,
    base_url: Option<Option<&str>>,
    api_key: Option<Option<&str>>,
    models_dir: Option<Option<&str>>,
    model_overrides: Option<Option<&str>>,
    engine_path: Option<Option<&str>>,
    auto_start: Option<bool>,
    trusted_measurements: Option<Option<&str>>,
    hardware_root_ca: Option<Option<&str>>,
    hardware_intermediate_ca: Option<Option<&str>>,
    now: i64,
) -> Result<bool, AppError> {
    // Build the SET list dynamically; every branch binds positionally.
    let mut sets: Vec<String> = vec!["updated_at = ?2".into()];
    let mut params: Vec<Value> = vec![Value::Text(id.to_string()), Value::Integer(now)];
    let bind = |expr: &str, v: Value, params: &mut Vec<Value>, sets: &mut Vec<String>| {
        params.push(v);
        sets.push(format!("{expr} = ?{}", params.len()));
    };
    if let Some(name) = display_name {
        bind(
            "display_name",
            Value::Text(name.to_string()),
            &mut params,
            &mut sets,
        );
    }
    if let Some(url) = base_url {
        bind(
            "base_url",
            match url {
                Some(u) => Value::Text(u.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(key) = api_key {
        bind(
            "api_key",
            match key {
                Some(k) => Value::Text(k.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(dir) = models_dir {
        bind(
            "models_dir",
            match dir {
                Some(d) => Value::Text(d.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(overrides) = model_overrides {
        bind(
            "model_overrides",
            match overrides {
                Some(o) => Value::Text(o.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(path) = engine_path {
        bind(
            "engine_path",
            match path {
                Some(p) => Value::Text(p.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(auto) = auto_start {
        bind(
            "auto_start",
            Value::Integer(auto as i64),
            &mut params,
            &mut sets,
        );
    }
    if let Some(measurements) = trusted_measurements {
        bind(
            "trusted_measurements",
            match measurements {
                Some(m) => Value::Text(m.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(ca) = hardware_root_ca {
        bind(
            "hardware_root_ca",
            match ca {
                Some(c) => Value::Text(c.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    if let Some(ca) = hardware_intermediate_ca {
        bind(
            "hardware_intermediate_ca",
            match ca {
                Some(c) => Value::Text(c.to_string()),
                None => Value::Null,
            },
            &mut params,
            &mut sets,
        );
    }
    let sql = format!(
        "UPDATE backend SET {} WHERE id = ?1 AND removed_at IS NULL",
        sets.join(", ")
    );
    let n = conn.execute(&sql, params).await.map_err(AppError::db)?;
    Ok(n > 0)
}

/// Soft-remove a backend (forensic rows keep their FK target). Returns
/// whether a live row was removed.
pub async fn remove_backend(conn: &Connection, id: &str, now: i64) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE backend SET removed_at = ?2, updated_at = ?2 \
             WHERE id = ?1 AND removed_at IS NULL",
            (Value::Text(id.to_string()), Value::Integer(now)),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Attestation operations
// ---------------------------------------------------------------------------

pub async fn upsert_attestation(
    conn: &Connection,
    hash: &str,
    doc: &[u8],
    pcr_digest: Option<&str>,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT OR IGNORE INTO attestation (hash, doc, pcr_digest, created_at) \
         VALUES (?1, ?2, ?3, ?4)",
        (
            Value::Text(hash.to_string()),
            Value::Blob(doc.to_vec()),
            match pcr_digest {
                Some(d) => Value::Text(d.to_string()),
                None => Value::Null,
            },
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to upsert attestation: {e}"),
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Connection operations
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub async fn insert_connection(
    conn: &Connection,
    id: &str,
    provider_id: &str,
    base_url: &str,
    transport: &str,
    attestation_hash: Option<&str>,
    opened_at: i64,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO connection (id, provider_id, base_url, transport, attestation_hash, opened_at, closed_at, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7)",
        (
            Value::Text(id.to_string()),
            Value::Text(provider_id.to_string()),
            Value::Text(base_url.to_string()),
            Value::Text(transport.to_string()),
            match attestation_hash {
                Some(h) => Value::Text(h.to_string()),
                None => Value::Null,
            },
            Value::Integer(opened_at),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert connection: {e}"),
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Request operations
// ---------------------------------------------------------------------------

pub struct Request {
    pub id: String,
    pub connection_id: Option<String>,
    pub action_id: Option<String>,
    pub method: String,
    pub path: String,
    pub request_headers: Option<String>,
    pub request_body: Option<Vec<u8>>,
    pub response_status: Option<i64>,
    pub response_headers: Option<String>,
    pub response_body: Option<Vec<u8>>,
    pub request_at: i64,
    pub response_at: Option<i64>,
    pub duration_ms: Option<i64>,
    pub error: Option<String>,
    pub credential_nonce: Option<String>,
    pub created_at: i64,
    /// The configured backend this request was routed through, if any.
    pub backend_id: Option<String>,
}

pub async fn insert_request(conn: &Connection, entry: &Request) -> Result<(), AppError> {
    // 17 parameters — beyond turso's tuple IntoParams impls, so a Vec.
    let params: Vec<Value> = vec![
        Value::Text(entry.id.clone()),
        opt_text(&entry.connection_id),
        opt_text(&entry.action_id),
        Value::Text(entry.method.clone()),
        Value::Text(entry.path.clone()),
        opt_text(&entry.request_headers),
        match &entry.request_body {
            Some(b) => Value::Blob(b.clone()),
            None => Value::Null,
        },
        match entry.response_status {
            Some(s) => Value::Integer(s),
            None => Value::Null,
        },
        opt_text(&entry.response_headers),
        match &entry.response_body {
            Some(b) => Value::Blob(b.clone()),
            None => Value::Null,
        },
        Value::Integer(entry.request_at),
        match entry.response_at {
            Some(t) => Value::Integer(t),
            None => Value::Null,
        },
        match entry.duration_ms {
            Some(d) => Value::Integer(d),
            None => Value::Null,
        },
        opt_text(&entry.error),
        opt_text(&entry.credential_nonce),
        Value::Integer(entry.created_at),
        opt_text(&entry.backend_id),
    ];
    conn.execute(
        "INSERT INTO request (id, connection_id, action_id, method, path, \
         request_headers, request_body, response_status, response_headers, response_body, \
         request_at, response_at, duration_ms, error, credential_nonce, created_at, backend_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
        params,
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert request: {e}"),
    })?;
    Ok(())
}

fn opt_text(v: &Option<String>) -> Value {
    match v {
        Some(s) => Value::Text(s.clone()),
        None => Value::Null,
    }
}

fn opt_str(v: Option<&str>) -> Value {
    match v {
        Some(s) => Value::Text(s.to_string()),
        None => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Participant operations
// ---------------------------------------------------------------------------

/// Find-or-create a **global** participant deduped by (kind, label). Used for
/// the shared library identities (and by tests). Scope-owned per-space and
/// per-template participants are never created here — they are minted with an
/// explicit owner via [`insert_participant`].
pub async fn ensure_participant(
    conn: &Connection,
    kind: &str,
    label: &str,
    provider_id: Option<&str>,
    created_at: i64,
) -> Result<String, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id FROM participant \
             WHERE scope = 'global' AND kind = ?1 AND label = ?2 LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(kind.to_string()),
            Value::Text(label.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    if let Some(row) = rows.next().await.map_err(AppError::db)? {
        return row.get::<String>(0).map_err(AppError::db);
    }
    drop(rows);
    drop(stmt);

    let id = uuid::Uuid::now_v7().to_string();
    insert_participant(
        conn,
        &id,
        "global",
        None,
        None,
        kind,
        label,
        None,
        None,
        "explicit",
        "member",
        provider_id,
        created_at,
    )
    .await?;
    Ok(id)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Space operations
// ---------------------------------------------------------------------------

pub async fn insert_space(
    conn: &Connection,
    id: &str,
    title: Option<&str>,
    linkability: &str,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO space (id, parent_space_id, title, linkability, created_at) \
         VALUES (?1, NULL, ?2, ?3, ?4)",
        (
            Value::Text(id.to_string()),
            match title {
                Some(t) => Value::Text(t.to_string()),
                None => Value::Null,
            },
            Value::Text(linkability.to_string()),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert space: {e}"),
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: the pristineness stamp and the disposal of untouched
// spaces.
//
// A space's window opens the space, so an abandoned new window (and every
// launch that opens a blank one) leaves a durable empty conversation behind.
// These two doors reap them, under one rule: **reap only what is provably
// pristine; when unsure, keep.** A wrongly-kept orphan costs a Library row; a
// wrongly-reaped space costs someone's work, so every mechanism below is
// arranged so that the safe error is the one a mistake makes.
//
// Pristine is two legs, both asked inside the disposal's own transaction:
//   * no `action` row in the space — every post, inference, trace, decision,
//     memory revision and branch summary is its own witness, which is why the
//     action write path carries no stamp; and
//   * `space.touched_at IS NULL` — nothing has changed the space's
//     configuration footprint (its own row, its `space_participant` rows, the
//     `participant` rows it owns) since it was instantiated.
//
// The stamp is written by the db-layer write functions themselves, so the
// enumeration lives where the writes live rather than in a caller's memory;
// `pristine_spaces_stamp_ledger` (tests/reap_pristine.rs) scans this module for
// statements against those three tables and fails on one that is not accounted
// for. `instantiate_template` is the one door that *un*-stamps: it writes
// `touched_at` last, inside its own transaction, because a fresh instantiation
// is pristine by definition.
// ---------------------------------------------------------------------------

/// Mark a space as touched — the first write that changes its configuration
/// footprint wins, so the value names the moment it stopped being untouched
/// and every later write is a cheap no-op.
///
/// An unknown id strikes nothing, deliberately: a caller writing into a space
/// that does not exist has a bigger problem than the stamp, and the write it
/// is about will fail on its own terms.
pub async fn touch_space(conn: &Connection, space_id: &str, now: i64) -> Result<(), AppError> {
    conn.execute(
        "UPDATE space SET touched_at = ?2 WHERE id = ?1 AND touched_at IS NULL",
        (Value::Text(space_id.to_string()), Value::Integer(now)),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

/// Mark the space a **space-owned** participant belongs to, deriving the space
/// inside the statement rather than from anything the caller captured — the
/// module's derive-authority-at-the-write idiom. A global or template-owned
/// participant owns no space, so the statement strikes nothing.
pub async fn touch_space_of_participant(
    conn: &Connection,
    participant_id: &str,
    now: i64,
) -> Result<(), AppError> {
    conn.execute(
        "UPDATE space SET touched_at = ?2 \
         WHERE touched_at IS NULL \
           AND id = (SELECT owner_space_id FROM participant WHERE id = ?1)",
        (Value::Text(participant_id.to_string()), Value::Integer(now)),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

/// Every space that currently looks pristine — the sweep's candidate read.
///
/// Advisory only: each candidate is re-decided inside
/// [`discard_space_if_pristine`]'s own transaction, so a write landing between
/// this read and that transaction keeps its space.
pub async fn pristine_space_ids(conn: &Connection) -> Result<Vec<String>, AppError> {
    let mut rows = conn
        .query(PRISTINE_SPACE_IDS_SQL, ())
        .await
        .map_err(AppError::db)?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        ids.push(row.get::<String>(0).map_err(AppError::db)?);
    }
    Ok(ids)
}

/// The pristineness predicate, as one query over the whole table.
///
/// A **notebook is excluded unconditionally**: it exists only for the agent
/// that owns it, is the residence of that agent's `core` memory, and is
/// created by a promotion — a deliberate act, whose freshly-minted space would
/// otherwise be the most obviously pristine thing in the database. The stamp
/// covers it too (`insert_notebook_space` writes one), so this clause is the
/// second lock on the one mistake with the highest cost.
///
/// The `parent_space_id` clause is no longer hypothetical: an agent can spawn
/// a sub-space (see [`spawn_subspace_tx`]). It protects the **parent**, and a
/// parent is over-protected rather than under-protected — spawning requires
/// the owner to be mid-turn in the parent, so a space that owns a sub-space
/// has actions and was never reapable anyway. The **sub-space** is kept by the
/// first leg instead: its brief commits in the spawning transaction, so it is
/// born with a post and is never pristine for a single instant.
const PRISTINE_SPACE_IDS_SQL: &str = "\
SELECT s.id FROM space s \
WHERE s.touched_at IS NULL \
  AND s.notebook_participant_id IS NULL \
  AND NOT EXISTS (SELECT 1 FROM action a WHERE a.space_id = s.id) \
  AND NOT EXISTS (SELECT 1 FROM memory_block m WHERE m.space_id = s.id) \
  AND NOT EXISTS (SELECT 1 FROM space c WHERE c.parent_space_id = s.id) \
  AND NOT EXISTS ( \
        SELECT 1 FROM participant p \
        WHERE p.owner_space_id = s.id \
          AND (EXISTS (SELECT 1 FROM action a2 WHERE a2.participant_id = p.id) \
            OR EXISTS (SELECT 1 FROM memory_block m2 WHERE m2.owner_participant_id = p.id)))";

/// The raw row counts of a space's whole footprint — `(space, membership,
/// owned participants)`. The test seam behind
/// `AppCore::test_space_footprint`; see it for why counting rows rather than
/// reading a roster is the only way to hold a delete to its scope.
#[doc(hidden)]
pub async fn space_footprint_counts(
    conn: &Connection,
    space_id: &str,
) -> Result<(i64, i64, i64), AppError> {
    let mut rows = conn
        .query(
            "SELECT \
               (SELECT COUNT(*) FROM space WHERE id = ?1), \
               (SELECT COUNT(*) FROM space_participant WHERE space_id = ?1), \
               (SELECT COUNT(*) FROM participant \
                 WHERE scope = 'space' AND owner_space_id = ?1)",
            (Value::Text(space_id.to_string()),),
        )
        .await
        .map_err(AppError::db)?;
    let row = rows
        .next()
        .await
        .map_err(AppError::db)?
        .ok_or_else(|| AppError::Database {
            message: "footprint count returned no row".into(),
        })?;
    Ok((
        row.get::<i64>(0).map_err(AppError::db)?,
        row.get::<i64>(1).map_err(AppError::db)?,
        row.get::<i64>(2).map_err(AppError::db)?,
    ))
}

/// Delete a space **if it is provably pristine**, and say whether it did.
///
/// This is the app's first real delete (archival is the only other removal, and
/// it is soft), so its scope is stated exactly. Inside one `BEGIN IMMEDIATE`
/// transaction it re-asks the whole predicate — never trusting the caller's
/// earlier read, which is what closes the window where a write lands between a
/// check and a delete — and then removes, in FK order:
///
/// 1. the space's `space_participant` rows (memberships *of this space*: the
///    referenced globals stay, they are the shared library);
/// 2. the `participant` rows the space **owns** (`scope = 'space'`), which by
///    the predicate have authored nothing and own no memory anywhere;
/// 3. the `space` row.
///
/// Nothing shared is reachable from there. The full inbound-edge graph of
/// `space(id)` is `space.parent_space_id`, `participant.owner_space_id`,
/// `space_participant.space_id`, `action.space_id`, `memory_block.space_id`
/// and `space_capability.space_id` — `action`/`memory_block` are empty by the
/// predicate's first leg, `parent_space_id` is checked explicitly, and the
/// membership and owned-participant rows are what this deletes.
/// **`space_capability` is empty for anything reapable and is deliberately not
/// deleted**: the only writer of a capability row is the sub-space spawn, and
/// that transaction writes the sub-space's brief in the same breath — so a
/// space with capabilities always has an action and is refused by the first
/// leg. If that ever stopped being true the FK would abort this transaction
/// and the space would be kept, which is the direction this whole feature errs
/// in; deleting the rows to be safe would err the other way, taking a grant
/// with a space nobody proved was empty. The inbound edges of a
/// space-owned `participant` are `action` and `memory_block` (both checked;
/// `space_participant` / `space_template_participant` are CHECK-pinned to
/// globals and `space.notebook_participant_id` names only a global). FK
/// enforcement is on for every connection, so anything this reasoning missed
/// aborts the transaction and the space is kept — which is the direction the
/// whole feature errs in.
pub(crate) async fn discard_space_if_pristine(
    conn: &Connection,
    space_id: &str,
) -> Result<bool, AppError> {
    begin_write(conn).await?;
    match discard_space_if_pristine_body(conn, space_id).await {
        Ok(discarded) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(discarded)
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn discard_space_if_pristine_body(
    conn: &Connection,
    space_id: &str,
) -> Result<bool, AppError> {
    let sql = format!("{PRISTINE_SPACE_IDS_SQL} AND s.id = ?1");
    let mut rows = conn
        .query(&sql, (Value::Text(space_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    if rows.next().await.map_err(AppError::db)?.is_none() {
        return Ok(false);
    }

    for stmt in [
        "DELETE FROM space_participant WHERE space_id = ?1",
        "DELETE FROM participant WHERE scope = 'space' AND owner_space_id = ?1",
        "DELETE FROM space WHERE id = ?1",
    ] {
        conn.execute(stmt, (Value::Text(space_id.to_string()),))
            .await
            .map_err(AppError::db)?;
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: agent-spawned sub-spaces.
//
// A sub-space is an ordinary space with `parent_space_id` set and **no human
// member**: the room an agent opens to delegate work, holding the spawning
// (owning) agent and the sub-agents it delegates to. Everything that makes it
// one is written by `spawn_subspace_tx` in a single transaction — the space
// row, the memberships, the brief that opens it, and the capability snapshot —
// because a sub-space missing any of those is not a sub-space.
//
// Ownership needs no column: the owner is definitionally a member, so its
// `role = 'owner'` membership row carries it (the notebook lesson in reverse
// — a notebook needed a column because its consumers ask "whose" *without*
// membership). `parent_space_id`, documented navigational since the schema was
// written and until now always NULL, is the other half.
// ---------------------------------------------------------------------------

/// One sub-space, as its parent and its owner see it.
#[derive(Clone, Debug)]
pub struct SubspaceRow {
    pub id: String,
    pub parent_space_id: String,
    /// The `role = 'owner'` member — the agent that spawned it.
    pub owner_participant_id: String,
    /// The post in the parent the delegation was opened from, when the spawn
    /// named one. It is where the report attaches, so the answer lands on the
    /// branch the work was asked for on. `None` for a spawn that named none.
    pub parent_action_id: Option<String>,
    pub title: Option<String>,
    pub created_at: i64,
    pub archived_at: Option<i64>,
}

/// One capability a space holds (see the `space_capability` table).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpaceCapabilityRow {
    pub name: String,
    /// Capability-specific JSON; `{}` for a bare grant.
    pub config: String,
}

/// The `role = 'owner'` membership, as one SQL fragment shared by every read
/// that resolves a sub-space's owner — so "who owns this" cannot be answered
/// two ways.
///
/// **The uniqueness this reads is enforced at the writes, not assumed here.**
/// A sub-space's owner membership cannot be ended
/// ([`remove_space_participant_tx`] refuses it in the leave's own `WHERE`, the
/// notebook owner's rule applied to the other structural membership) and no
/// door may grant a *second* `role = 'owner'` into a sub-space
/// ([`subspace_owner_of`] is asked inside both granting transactions). So this
/// names one row by construction; the `left_at IS NULL` filter and the total
/// ordering are the belt to that brace — with the guards in place a departed
/// owner is unrepresentable, and if one ever appeared this would answer "no
/// owner" (a sub-space that stops being one, which every consumer already
/// handles) rather than resurrect a membership that ended, and two owners
/// would answer the *older* one rather than whichever the query planner
/// reached first.
const SUBSPACE_OWNER_SQL: &str = "\
(SELECT r.participant_id FROM space_participant r \
  WHERE r.space_id = s.id AND r.role = 'owner' AND r.left_at IS NULL \
  ORDER BY r.joined_at ASC, r.participant_id ASC LIMIT 1)";

/// The live owner of `space_id` **if it is a sub-space** — `None` for an
/// ordinary space, and `None` for a sub-space with no live owner membership
/// (unrepresentable while the guards hold; see [`SUBSPACE_OWNER_SQL`]).
///
/// This is the predicate both granting transactions ask before writing a
/// `role = 'owner'` membership. It is asked **inside** them because "does this
/// room already have an owner" is exactly the kind of fact a concurrent spawn
/// changes, and a caller that read it earlier would be deciding from a
/// snapshot rather than from the state its own write lands on.
pub async fn subspace_owner_of(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<String>, AppError> {
    let sql = format!(
        "SELECT {SUBSPACE_OWNER_SQL} FROM space s \
         WHERE s.id = ?1 AND s.parent_space_id IS NOT NULL"
    );
    let mut rows = conn
        .query(&sql, (Value::Text(space_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(row.get::<Option<String>>(0).map_err(AppError::db)?),
    }
}

/// Refuse a `role = 'owner'` membership into a sub-space that already has an
/// owner — the guard both granting doors share.
///
/// A sub-space's owner is minted with the room and is the whole of what
/// `parent_space_id` + one membership row encode: who is answerable for the
/// delegation, whose live-room quota it counts against, and who the report
/// goes to. A second owner would make all three arbitrary. The remedy for a
/// delegation someone regrets is archiving it, exactly as it is for the quota.
async fn refuse_second_subspace_owner(
    conn: &Connection,
    space_id: &str,
    role: &str,
    joining: &str,
) -> Result<(), AppError> {
    if role != "owner" {
        return Ok(());
    }
    match subspace_owner_of(conn, space_id).await? {
        Some(owner) if owner != joining => Err(AppError::Config {
            message: "this conversation was opened by an agent that already owns it; another \
                      participant can join it, but not as its owner"
                .into(),
        }),
        _ => Ok(()),
    }
}

fn subspace_row(row: &turso::Row) -> Result<SubspaceRow, AppError> {
    Ok(SubspaceRow {
        id: row.get::<String>(0).map_err(AppError::db)?,
        parent_space_id: row.get::<String>(1).map_err(AppError::db)?,
        owner_participant_id: row.get::<String>(2).map_err(AppError::db)?,
        parent_action_id: row.get::<Option<String>>(3).map_err(AppError::db)?,
        title: row.get::<Option<String>>(4).map_err(AppError::db)?,
        created_at: row.get::<i64>(5).map_err(AppError::db)?,
        archived_at: row.get::<Option<i64>>(6).map_err(AppError::db)?,
    })
}

async fn subspace_rows(
    conn: &Connection,
    where_clause: &str,
    param: Option<&str>,
) -> Result<Vec<SubspaceRow>, AppError> {
    let sql = format!(
        "SELECT s.id, s.parent_space_id, {SUBSPACE_OWNER_SQL}, s.parent_action_id, s.title, \
                s.created_at, s.archived_at \
         FROM space s \
         WHERE s.parent_space_id IS NOT NULL AND {SUBSPACE_OWNER_SQL} IS NOT NULL \
           AND {where_clause} \
         ORDER BY s.created_at ASC"
    );
    let mut rows = match param {
        Some(p) => conn.query(&sql, (Value::Text(p.to_string()),)).await,
        None => conn.query(&sql, ()).await,
    }
    .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(subspace_row(&row)?);
    }
    Ok(out)
}

/// Every sub-space spawned from `parent_space_id`, oldest first (archived
/// included — a parent's history of delegations does not shorten).
pub async fn subspaces_of(
    conn: &Connection,
    parent_space_id: &str,
) -> Result<Vec<SubspaceRow>, AppError> {
    subspace_rows(conn, "s.parent_space_id = ?1", Some(parent_space_id)).await
}

/// Every **live** (non-archived) sub-space this participant owns — the set the
/// per-owner spawn guard counts, and the enumeration a turn driver needs to
/// find the rooms it is responsible for.
pub async fn live_subspaces_owned_by(
    conn: &Connection,
    owner_participant_id: &str,
) -> Result<Vec<SubspaceRow>, AppError> {
    subspace_rows(
        conn,
        &format!("s.archived_at IS NULL AND {SUBSPACE_OWNER_SQL} = ?1"),
        Some(owner_participant_id),
    )
    .await
}

/// Read one space **as a sub-space**: `None` when it has no parent or no owner
/// row, i.e. when it is an ordinary space. The read behind "who do I report to,
/// and where" for a sub-space id.
pub async fn subspace(conn: &Connection, space_id: &str) -> Result<Option<SubspaceRow>, AppError> {
    Ok(subspace_rows(conn, "s.id = ?1", Some(space_id))
        .await?
        .pop())
}

/// Every **live** sub-space, whoever owns it — what a turn driver enumerates
/// when it starts, to pick up the rooms a previous run left mid-delegation.
///
/// Ordered oldest first, so a restart resumes delegations in the order they
/// were opened rather than in whatever order the planner reached them.
pub async fn live_subspaces(conn: &Connection) -> Result<Vec<SubspaceRow>, AppError> {
    subspace_rows(conn, "s.archived_at IS NULL", None).await
}

/// Whether `space_id` is a sub-space that still takes work: it has a parent, it
/// has an owner, and it is not archived.
///
/// One row read on the space's primary key. It is the question asked of every
/// `Change::Space` a driver hears, so it is deliberately narrower than
/// [`subspace`] — no owner join, no columns to build a row out of.
pub async fn is_live_subspace(conn: &Connection, space_id: &str) -> Result<bool, AppError> {
    let mut rows = conn
        .query(
            "SELECT 1 FROM space \
             WHERE id = ?1 AND parent_space_id IS NOT NULL AND archived_at IS NULL",
            (Value::Text(space_id.to_string()),),
        )
        .await
        .map_err(AppError::db)?;
    Ok(rows.next().await.map_err(AppError::db)?.is_some())
}

/// How many turns have been taken in `space_id` — the count a per-delegation
/// turn budget is spent against.
///
/// **Derived, never stored.** A turn ends in exactly one durable act: the
/// `inference` it persists, or the `decision` the decline checkpoint writes
/// instead. Counting those rows is counting turns, and the answer survives a
/// restart because the rows do — an in-memory tally would reset the meter every
/// time the process came back, which is the one way a budget can be escaped.
/// A turn that failed wrote neither and is deliberately not counted: nothing
/// was produced and nothing was persisted to produce it from. Driven turns
/// keep that true by writing no generation on failure; a human's *blocking*
/// verb in a room writes a `status = 'error'` generation instead, so the
/// terminal-status filter is what keeps the sentence true of every writer.
/// Superseded generations stay counted on purpose — each was a turn that ran.
pub async fn turns_taken_in_space(conn: &Connection, space_id: &str) -> Result<i64, AppError> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM action \
             WHERE space_id = ?1 AND action_type IN ('inference', 'decision') \
               AND status IN ('complete', 'cancelled')",
            (Value::Text(space_id.to_string()),),
        )
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        Some(row) => row.get::<i64>(0).map_err(AppError::db),
        None => Ok(0),
    }
}

/// `(id, text)` of an action's first quotable ([`QUOTABLE_BLOCK_TYPE`]) content
/// block — what a driver quotes when it attaches a post to a turn it is about
/// to run. `None` for an action with no text block (a decline, a trace).
pub async fn first_quotable_block(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<(String, String)>, AppError> {
    let mut rows = conn
        .query(
            "SELECT id, text_content FROM content_block \
             WHERE action_id = ?1 AND block_type = ?2 AND text_content IS NOT NULL \
             ORDER BY ordinal ASC LIMIT 1",
            (
                Value::Text(action_id.to_string()),
                Value::Text(QUOTABLE_BLOCK_TYPE.to_string()),
            ),
        )
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some((
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<String>(1).map_err(AppError::db)?,
        ))),
    }
}

/// Whether `participant_id` has already quoted `antecedent_action_id` in
/// `space_id` — the read that answers "has this delegation already been
/// reported?"
///
/// **This is what keeps a delegation's lifecycle derived** rather than stored.
/// A driver's terminal act is a turn in the parent quoting the delegated room's
/// last post; asking whether that edge exists asks whether the room's *current*
/// tail has been reported, which is true after a report, false again the moment
/// somebody posts into the room, and identical after a restart. A status column
/// would have to be written, kept in step with continuation, and believed.
///
/// **Only a generation the parent actually shows answers it.** Current,
/// because a superseded generation shows nobody anything — and in a terminal
/// status and a post type, the same predicate `get_space_tree_data` renders
/// by, because "current" alone is not "visible": a report regenerated against
/// a failing upstream persists a **current `status = 'error'`** generation
/// carrying the report's replicated edges (`persist_turn`), which the
/// transcript hides — counting that edge would settle the room on a report
/// the parent cannot show, durably and across restarts. A report regenerated
/// *successfully* carries its edges forward (`prepare_turn`'s replication, the
/// rule `edit_post` has always followed), so the tip still answers; the
/// visibility join is what makes the driver's belief and the reader's
/// footnote the same fact rather than two that can drift, and it errs toward
/// reporting again rather than toward silence.
///
/// **And the owner is the item's original author, not the tip's.** A
/// successful regeneration can mint a new agent (`TurnSelector::Model` when
/// no seated member matches the picked model), and the edges travel with the
/// item: asking the tip's `participant_id` would treat that re-wording as
/// somebody else's quote and the next walk would post a duplicate. The origin
/// generation (`supersedes_action_id IS NULL`) is who opened the item; a
/// fresh quote by anyone else is a different item and still does not settle.
pub async fn has_reference_from(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    antecedent_action_id: &str,
) -> Result<bool, AppError> {
    let sql = format!(
        "SELECT 1 FROM action_antecedent aa \
         JOIN action a ON a.id = aa.action_id \
         JOIN item_current ic ON ic.current_action_id = a.id \
         JOIN action origin ON origin.item_id = a.item_id \
           AND origin.supersedes_action_id IS NULL \
         WHERE aa.relation = 'reference' AND aa.antecedent_action_id = ?1 \
           AND a.space_id = ?2 AND origin.participant_id = ?3 \
           AND a.status IN ('complete', 'cancelled') \
           AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
         LIMIT 1"
    );
    let mut rows = conn
        .query(
            &sql,
            (
                Value::Text(antecedent_action_id.to_string()),
                Value::Text(space_id.to_string()),
                Value::Text(participant_id.to_string()),
            ),
        )
        .await
        .map_err(AppError::db)?;
    Ok(rows.next().await.map_err(AppError::db)?.is_some())
}

/// The most recent **post** `participant_id` wrote in `space_id` **replying
/// directly to `antecedent_action_id`** — the owner's own answer to the post a
/// delegation was opened from, which is where that delegation's report belongs.
///
/// A direct reply rather than any descendant, because the relationship is
/// exact: a spawn happens inside a turn, and that turn persists its answer as a
/// reply to the very post it was answering. Newest — by **commit order**, the
/// one ordering a writer's clock cannot contradict (see [`action_watermark`])
/// — and **a generation the parent shows** (current, terminal status, post
/// type — the transcript's predicate, as settlement reads it): an answer
/// whose regeneration failed leaves a current `status = 'error'` tip, and a
/// superseded generation that matched here would end the wait and put the
/// report beneath an answer rendering resolves to nothing — at the
/// conversation root, not under the owner's visible word. With the item's tip
/// hidden nothing of that item matches, so the wait goes on waiting, which is
/// the honest reading: the visible answer is gone until a regeneration lands
/// or the grace alarm ends the wait against the anchor.
///
/// **And the owner is the item's original author, not the tip's** — the same
/// join [`has_reference_from`] uses, for the same reason. A successful
/// regeneration can mint a new agent (`TurnSelector::Model` when no seated
/// member matches the picked model) and the reply edge travels with the item,
/// so asking the tip's `participant_id` would miss the owner's own re-worded
/// answer and wait out the grace alarm, attaching the report to the anchor as
/// a sibling of a word the parent already shows.
///
/// **`after_row` is the line that says *which* answer.** An anchored spawn
/// happens inside the owning agent's turn, so the answer the report belongs
/// under is the one that turn is still to write — and any answer of that owner
/// to that anchor which is already committed is a *different* answer: the
/// generation a regeneration is in the middle of replacing, or a prior reply to
/// the same post. Accepting one of those ends the wait at once and reports
/// beneath an answer the delegation did not come from, while the answer it did
/// come from is still on the wire; if the regeneration in flight then fails,
/// its item's tip is a hidden `error` and the report is left hanging under a
/// generation the parent no longer shows. So a caller that knows a turn is
/// behind the delegation passes the commit-order line the room was opened at
/// ([`subspace_opened_at_row`]) and only answers committed after it are
/// candidates. `None` keeps the unrestricted rule, which is the honest one for
/// a caller with no turn behind it: the newest answer is then the best guess
/// available.
///
/// **It is a `rowid` and not a clock** for the reason [`action_watermark`]
/// gives: every writer samples `now_ms()` above its own transaction, so a
/// timestamp cannot order two writes that raced, while `rowid` is assigned at
/// insert under serialized writers and `action` is append-only — nothing here
/// ever deletes a row, so the sequence cannot be reused.
pub async fn last_reply_by_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    antecedent_action_id: &str,
    after_row: Option<i64>,
) -> Result<Option<String>, AppError> {
    let sql = format!(
        "SELECT a.id FROM action a \
         JOIN action_antecedent aa \
           ON aa.action_id = a.id AND aa.relation = 'reply' \
         JOIN item_current ic ON ic.current_action_id = a.id \
         JOIN action origin ON origin.item_id = a.item_id \
           AND origin.supersedes_action_id IS NULL \
         WHERE a.space_id = ?1 AND origin.participant_id = ?2 \
           AND aa.antecedent_action_id = ?3 \
           AND a.rowid > ?4 \
           AND a.status IN ('complete', 'cancelled') \
           AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY a.rowid DESC LIMIT 1"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([
            Value::Text(space_id.to_string()),
            Value::Text(participant_id.to_string()),
            Value::Text(antecedent_action_id.to_string()),
            // No line means every row is after it. `rowid` starts at 1.
            Value::Integer(after_row.unwrap_or(0)),
        ])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// The commit-order line a delegated room was opened at: the `rowid` of its
/// **brief**, which [`spawn_subspace_tx`] writes in the same transaction as the
/// room itself and which is therefore the first action the room has.
///
/// **No new state records this** — the fact was already durable. Every action
/// committed before the spawn has a lower `rowid` and every action committed
/// after it a higher one (writers are serialized and `action` is append-only),
/// so the brief *is* the pre-spawn watermark, and it reads the same after a
/// restart as before one because it reads rows. `MIN` rather than a stored
/// brief id for the same reason: the room's own first action is the brief by
/// construction, so there is nothing to keep in step.
pub async fn subspace_opened_at_row(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<i64>, AppError> {
    let mut stmt = conn
        .prepare("SELECT MIN(rowid) FROM action WHERE space_id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(space_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => row.get::<Option<i64>>(0).map_err(AppError::db),
    }
}

/// The most recent — by **commit order** — **post** `participant_id` wrote in
/// `space_id`, or `None` when they have written none. The fallback a report
/// takes when its spawn named no anchor — see [`last_reply_by_participant`]
/// for the anchored path, including why only a generation the parent shows
/// may answer (the same `item_current` + visibility predicate) and why the
/// owner is the item's original author, not the tip's.
pub async fn last_post_by_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
) -> Result<Option<String>, AppError> {
    let sql = format!(
        "SELECT a.id FROM action a \
         JOIN item_current ic ON ic.current_action_id = a.id \
         JOIN action origin ON origin.item_id = a.item_id \
           AND origin.supersedes_action_id IS NULL \
         WHERE a.space_id = ?1 AND origin.participant_id = ?2 \
           AND a.status IN ('complete', 'cancelled') \
           AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY a.rowid DESC LIMIT 1"
    );
    let mut rows = conn
        .query(
            &sql,
            (
                Value::Text(space_id.to_string()),
                Value::Text(participant_id.to_string()),
            ),
        )
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// How deep a space sits in the `parent_space_id` chain: 0 for a space nobody
/// spawned, 1 for a sub-space of one of those, and so on.
///
/// Cycle-guarded. The column is a self-referencing FK with no cycle
/// constraint, and this walk is a spawn-time guard — the one place a cycle
/// would turn into a hang rather than a wrong answer.
pub async fn space_depth(conn: &Connection, space_id: &str) -> Result<i64, AppError> {
    let mut depth = 0;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cursor = space_id.to_string();
    while seen.insert(cursor.clone()) {
        let mut rows = conn
            .query(
                "SELECT parent_space_id FROM space WHERE id = ?1",
                (Value::Text(cursor.clone()),),
            )
            .await
            .map_err(AppError::db)?;
        let Some(row) = rows.next().await.map_err(AppError::db)? else {
            break;
        };
        match row.get::<Option<String>>(0).map_err(AppError::db)? {
            Some(parent) => {
                depth += 1;
                cursor = parent;
            }
            None => break,
        }
    }
    Ok(depth)
}

/// The capabilities a space holds, by name.
pub async fn space_capabilities(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<SpaceCapabilityRow>, AppError> {
    let mut rows = conn
        .query(
            "SELECT name, config FROM space_capability WHERE space_id = ?1 ORDER BY name ASC",
            (Value::Text(space_id.to_string()),),
        )
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(SpaceCapabilityRow {
            name: row.get::<String>(0).map_err(AppError::db)?,
            config: row.get::<String>(1).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// Test-only seam: seed a capability on a space. See
/// `AppCore::test_grant_space_capability` for why the attenuation gate cannot
/// be tested in both directions without one.
///
/// **Compiled out of release builds, not merely hidden from the docs.** The
/// read seams beside it (`space_footprint_counts`, `space_is_archived`) are
/// `#[doc(hidden)]` alone because a read cannot forge state; this one *mints a
/// capability*, which is the one thing the whole attenuation model exists to
/// make impossible outside a spawn. `#[doc(hidden)]` gates documentation and
/// nothing else, so a release-built dependent could link it and hand itself a
/// grant no parent held.
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub(crate) async fn test_insert_space_capability(
    conn: &Connection,
    space_id: &str,
    name: &str,
    config: &str,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO space_capability (space_id, name, config) VALUES (?1, ?2, ?3)",
        (
            Value::Text(space_id.to_string()),
            Value::Text(name.to_string()),
            Value::Text(config.to_string()),
        ),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

/// Whether a participant row carries a model of its own — the configuration a
/// **new** space sees, since nothing but the notify policy is overridden into
/// one. An agent without one is skipped by every planner
/// (`Inner::mechanical_plan`), so seating it schedules nothing.
fn has_model(p: &ParticipantRow) -> bool {
    p.model_ref.as_deref().is_some_and(|m| !m.trim().is_empty())
}

/// Everything a spawn needs, with every id already minted by the caller so the
/// outcome can be reported without re-reading what was just written.
///
/// Crate-private with [`spawn_subspace_tx`]: it is that write's argument, and
/// an argument type reachable from outside is the write reachable from outside.
pub(crate) struct SubspacePlan<'a> {
    pub space_id: &'a str,
    pub parent_space_id: &'a str,
    pub owner_participant_id: &'a str,
    pub title: Option<&'a str>,
    pub brief: &'a str,
    pub brief_action_id: &'a str,
    pub brief_item_id: &'a str,
    /// The post in the parent this delegation is being opened from — the turn's
    /// own target, which the caller knows and this door cannot derive. Written
    /// to `space.parent_action_id` and validated to be a post the parent
    /// **currently shows** (current generation, terminal status, post type).
    pub parent_action_id: Option<&'a str>,
    /// Global agents to seat beside the owner, deduped and in requested order.
    pub participant_ids: &'a [String],
    /// Capability names requested; each must already be held by the parent,
    /// and each is **copied** from the parent's row rather than composed here.
    pub capabilities: &'a [String],
    pub now: i64,
}

/// Mint a sub-space, or say why not — one `BEGIN IMMEDIATE` transaction that
/// **decides every guard at the write**.
///
/// Each guard reads state a concurrent window can change (a membership, a
/// retirement, another spawn, an archival), so asking them before the writer is
/// reserved would let two spawns pass a limit only one of them may. The whole
/// question therefore lives inside the transaction, and a refusal rolls back to
/// exactly nothing.
///
/// The guards, in the order a caller would want them explained:
///
/// 1. the parent exists;
/// 2. the spawner is a **live global agent** and a live member of the parent.
///    Global because a space-owned participant cannot be referenced into
///    another space at all (the pinned scope echo makes that declarative) — so
///    the rule that lets an agent own a room elsewhere is the same rule that
///    lets it join one;
/// 3. the new space's depth would not exceed [`MAX_SPAWN_DEPTH`];
/// 4. the owner holds fewer than [`MAX_LIVE_SUBSPACES_PER_OWNER`] live
///    sub-spaces (archiving one frees a slot);
/// 5. every requested capability is one the **parent** already holds, its
///    config copied verbatim — the monotonicity check, which composes down a
///    chain for free because each sub-space's snapshot is the only thing its
///    own children can be granted from;
/// 6. every requested sub-agent is a live global agent (the
///    `add_global_participant` rule, asked here so the refusal is the spawn's).
///
/// Then, in one commit: the space row (parent, the parent's `cascade_limit`
/// and `router_model`, **born stamped** — an agent minted this deliberately,
/// exactly as a promotion mints a notebook), the owner's `role = 'owner'`
/// membership, each sub-agent as a `member` with `override_notify_policy =
/// 'all'` (what makes a human-less room live: the seeded policies only fire on
/// human posts, and the override is per-membership, so the agents' own global
/// rows are untouched), the brief as the first post, and the capability rows.
///
/// **No human membership row.** This is a new creation door, not
/// `instantiate_template` — which joins the shared human unconditionally, and
/// whose template agents have nothing to do with a delegation.
///
/// **Crate-private, like every other raw write in this module.** The guards
/// above are the ones that can only be decided against live state; the ones
/// that decide against the *caller's own values* — a brief that is empty or
/// nothing but whitespace — sit in front of the transaction, in
/// `crate::Inner::spawn_subspace`, because a value cannot be raced and a
/// message belongs where a caller can read it rather than after a rollback.
/// That split is only sound while this door is unreachable from outside: a
/// dependent holding it could commit a room opened by a blank brief, which
/// would then plan and bill turns off a post that says nothing. Privatizing is
/// the same cure the membership writers took, and for the same reason —
/// repeating the check here would put one rule in two places to drift apart.
pub(crate) async fn spawn_subspace_tx(
    conn: &Connection,
    plan: &SubspacePlan<'_>,
) -> Result<Result<String, SpawnRefusal>, AppError> {
    begin_write(conn).await?;
    match spawn_subspace_tx_body(conn, plan).await {
        Ok(Ok(title)) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(Ok(title))
        }
        Ok(Err(refusal)) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Ok(Err(refusal))
        }
        Err(e) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn spawn_subspace_tx_body(
    conn: &Connection,
    plan: &SubspacePlan<'_>,
) -> Result<Result<String, SpawnRefusal>, AppError> {
    macro_rules! refuse {
        ($r:expr) => {
            return Ok(Err($r))
        };
    }

    // (1) the parent, and the two settings a sub-space inherits from it:
    // `cascade_limit` (attenuation-flavoured — a delegation never gets a
    // looser runaway guard than the room it was delegated from) and
    // `router_model` (a panel wants routing exactly as much as its parent
    // did).
    // **Exists, is still open, and what it passes down — one read.** The
    // parent being *archived* refuses a spawn for the same reason an archived
    // space takes no new turns: a room minted under a closed conversation is
    // work nobody asked to continue, and nothing about archival makes the
    // parent row or the owner's membership stop looking live. It is reachable
    // rather than theoretical — a turn already past `prepare_turn` runs to
    // completion by design, so an archival landing mid-turn leaves the owner's
    // spawn call still to come. Read here rather than inherited from that turn
    // for the same reason refinement re-reads it: two calls, and the archival
    // can land between them.
    let Some(parent) = get_space(conn, plan.parent_space_id).await? else {
        refuse!(SpawnRefusal::UnknownParent {
            space_id: plan.parent_space_id.to_string(),
        });
    };
    if parent.archived_at.is_some() {
        refuse!(SpawnRefusal::ParentArchived {
            space_id: plan.parent_space_id.to_string(),
        });
    }
    let parent_cascade_limit = parent.cascade_limit;
    let parent_router_model = parent.router_model;

    // (2) the spawner. **Its base config is what the sub-space will see** —
    // the spawn copies no overrides, so a model supplied by an override in the
    // parent does not travel — which is why the model is read from the row
    // rather than from the parent's effective view.
    let owner = match get_participant(conn, plan.owner_participant_id).await? {
        Some(p)
            if p.scope == "global"
                && p.kind == "agent"
                && p.removed_at.is_none()
                && is_space_member(conn, plan.parent_space_id, plan.owner_participant_id)
                    .await? =>
        {
            p
        }
        _ => refuse!(SpawnRefusal::SpawnerNotEligible {
            participant_id: plan.owner_participant_id.to_string(),
        }),
    };
    if !has_model(&owner) {
        refuse!(SpawnRefusal::NoModelConfigured {
            label: owner.label.clone(),
        });
    }

    // (2b) the anchor, when one was named: the post in the parent this
    // delegation is being opened from. Decided at the write like every other
    // guard — and it must be a generation the parent **shows**, not merely a
    // row that was once a post there. A superseded wording, or a failed
    // regeneration's hidden `error` tip, would let the room run and bill, then
    // wait for a reply nobody can write to that id; after grace the report
    // would attach to an item whose tip is not renderable and land at the
    // conversation root. The transcript's own predicate is the whole of
    // "somewhere here to report back to".
    if let Some(anchor) = plan.parent_action_id
        && !is_visible_post_in_space(conn, plan.parent_space_id, anchor).await?
    {
        refuse!(SpawnRefusal::AnchorNotInParent {
            action_id: anchor.to_string(),
        });
    }

    // (2c) somewhere for the report to land. **Refused here rather than
    // discovered there**, because the asymmetry is total: a refused spawn costs
    // nothing at all, while a delegation that runs its turns and then finds no
    // post to reply to has spent real money on work nobody will ever be told
    // about. An anchor is itself such a post (guard 2b proved it), so this only
    // ever bites a caller with no turn behind it opening a delegation from a
    // conversation where nothing has been said.
    if plan.parent_action_id.is_none()
        && last_action_in_space(conn, plan.parent_space_id)
            .await?
            .is_none()
    {
        refuse!(SpawnRefusal::NothingToReportTo);
    }

    // (3) depth.
    let depth = space_depth(conn, plan.parent_space_id).await? + 1;
    if depth > MAX_SPAWN_DEPTH {
        refuse!(SpawnRefusal::TooDeep {
            depth,
            limit: MAX_SPAWN_DEPTH,
        });
    }

    // (4) live sub-spaces per owner.
    let live = live_subspaces_owned_by(conn, plan.owner_participant_id).await?;
    if live.len() as i64 >= MAX_LIVE_SUBSPACES_PER_OWNER {
        refuse!(SpawnRefusal::TooManyLiveSubspaces {
            live: live.len() as i64,
            limit: MAX_LIVE_SUBSPACES_PER_OWNER,
        });
    }

    // (5) attenuation. The parent's snapshot is the whole of what may be
    // granted, so a chain attenuates without anyone walking it: what B may
    // give C is what B was given, which was a subset of what A held.
    let held = space_capabilities(conn, plan.parent_space_id).await?;
    let mut granted: Vec<SpaceCapabilityRow> = Vec::new();
    for name in plan.capabilities {
        match held.iter().find(|c| &c.name == name) {
            Some(c) => granted.push(c.clone()),
            None => refuse!(SpawnRefusal::CapabilityNotHeld {
                name: name.to_string(),
            }),
        }
    }

    // (6) how many sub-agents. Every seat is written with
    // `override_notify_policy = 'all'`, so each one answers every post in the
    // room and each of those answers notifies all the others: the work a
    // spawn schedules grows with the square of the roster and the cascade
    // guard is the only thing that ever stops it. A panel is the point, but a
    // panel is small.
    if plan.participant_ids.len() as i64 > MAX_SUBAGENTS_PER_SPAWN {
        refuse!(SpawnRefusal::TooManySubagents {
            requested: plan.participant_ids.len() as i64,
            limit: MAX_SUBAGENTS_PER_SPAWN,
        });
    }

    // (7) the sub-agents themselves — the same base-config rule as the owner:
    // an agent with no model of its own is skipped by every planner, so
    // seating one would report a spawn that scheduled nothing.
    for id in plan.participant_ids {
        match get_participant(conn, id).await? {
            Some(p) if p.scope == "global" && p.kind == "agent" && p.removed_at.is_none() => {
                if !has_model(&p) {
                    refuse!(SpawnRefusal::NoModelConfigured {
                        label: p.label.clone(),
                    });
                }
            }
            _ => refuse!(SpawnRefusal::ParticipantNotEligible {
                participant_id: id.to_string(),
            }),
        }
    }

    // The title the room carries in the Library. An explicit one, or the
    // brief's opening line, has already been chosen by the caller; what is
    // left is a brief that yields no line at all (pure markdown scaffolding,
    // say). A row with neither a title nor a snippet — briefs are not what
    // `first_user_text` reads — would be unrecognizable, and refusing the
    // spawn would punish a model for its formatting, so the room takes the
    // name of the agent answerable for it.
    //
    // **The owner's label alone, not a sentence about it.** A title is
    // persisted, and a persisted string can never pass through the
    // presentation layer's translations — so anything this crate writes there
    // is read as-is in every language. A label is a *name*: it is already
    // whatever its owner called it, it is locale-neutral by nature, and it
    // says the one thing a reader scanning the Library needs (whose room this
    // is). A phrase like "Delegated by …" would have been English copy stored
    // in the database, which is exactly what the layering keeps out of here —
    // this crate ships no user-facing strings.
    let title = plan
        .title
        .map(str::to_string)
        .unwrap_or_else(|| owner.label.clone());

    // ---- writes -----------------------------------------------------------
    //
    // `touched_at` is written **in the insert**, so this function never calls
    // `touch_space` and has no stamp ordering to get wrong. A sub-space is born
    // stamped for the reason a notebook is: an agent minted it on purpose, and
    // the brief below means it could never have been reaped anyway.
    conn.execute(
        "INSERT INTO space \
         (id, parent_space_id, parent_action_id, title, linkability, cascade_limit, \
          router_model, created_at, touched_at) \
         VALUES (?1, ?2, ?7, ?3, 'unlinked', ?4, ?5, ?6, ?6)",
        (
            Value::Text(plan.space_id.to_string()),
            Value::Text(plan.parent_space_id.to_string()),
            Value::Text(title.clone()),
            Value::Integer(parent_cascade_limit),
            opt_str(parent_router_model.as_deref()),
            Value::Integer(plan.now),
            opt_str(plan.parent_action_id),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert sub-space: {e}"),
    })?;

    // **The owner's policy is written, not inherited.** Leaving the override
    // NULL means the room adopts whatever that agent's global row happens to
    // say, and a shared agent configured `notify_policy = 'all'` is ordinary —
    // it would then be scheduled by the first helper's answer, and its own
    // answer would wake every notify-all helper again, so a spawn's work grows
    // with the square of the roster until the cascade guard stops it. The
    // sub-agents are `all` on purpose (nothing else would ever wake them in a
    // room with no human); the owner must not be.
    //
    // `'human'` rather than `'explicit'`, and the difference only shows once
    // somebody joins: both are silent among agents, but `'human'` means the
    // agent answerable for the delegation answers *the human who came to look
    // at it*, which is what that agent is for, while `'explicit'` would leave
    // it deaf to them. It is also the seeded default, so the owner behaves like
    // an ordinary agent toward people and stays quiet among its helpers.
    // Neither choice is how the owner's own turns get driven — the
    // orchestration layer plans those deliberately through its own channel —
    // so this decides one thing only: who a post in this room wakes.
    insert_participant_ref(
        conn,
        "space_participant",
        "space_id",
        plan.space_id,
        plan.owner_participant_id,
        "owner",
        plan.now,
        &ParticipantRefRow {
            participant_id: plan.owner_participant_id.to_string(),
            role: "owner".to_string(),
            joined_at: plan.now,
            override_label: None,
            override_model_ref: None,
            override_system_prompt: None,
            override_notify_policy: Some("human".to_string()),
        },
        false,
    )
    .await?;
    for id in plan.participant_ids {
        insert_participant_ref(
            conn,
            "space_participant",
            "space_id",
            plan.space_id,
            id,
            "member",
            plan.now,
            &ParticipantRefRow {
                participant_id: id.to_string(),
                role: "member".to_string(),
                joined_at: plan.now,
                override_label: None,
                override_model_ref: None,
                override_system_prompt: None,
                // The one override a spawn writes, and the reason a human-less
                // room is not inert: the seeded `human` policy fires only on a
                // human-authored post, and there will never be one here.
                override_notify_policy: Some("all".to_string()),
            },
            false,
        )
        .await?;
    }

    insert_action(
        conn,
        &ActionEntry {
            id: plan.brief_action_id.to_string(),
            space_id: plan.space_id.to_string(),
            participant_id: plan.owner_participant_id.to_string(),
            item_id: plan.brief_item_id.to_string(),
            supersedes_action_id: None,
            action_type: BRIEF_ACTION_TYPE.to_string(),
            status: "complete".to_string(),
            intent: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            credits_consumed: None,
            created_at: plan.now,
        },
    )
    .await?;
    insert_text_content_block(
        conn,
        &uuid::Uuid::now_v7().to_string(),
        plan.brief_action_id,
        0,
        "text",
        plan.brief,
    )
    .await?;

    for c in &granted {
        conn.execute(
            "INSERT INTO space_capability (space_id, name, config) VALUES (?1, ?2, ?3)",
            (
                Value::Text(plan.space_id.to_string()),
                Value::Text(c.name.clone()),
                Value::Text(c.config.clone()),
            ),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to insert space capability: {e}"),
        })?;
    }

    Ok(Ok(title))
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Participants (scope-owned) + space templates
//
// Every participant row has exactly one scope: 'global' (the shared library —
// today "User"), 'space' (owned by one space), or 'template' (owned by one
// template). The config columns live ONLY on `participant`. Reference tables
// (space_participant / space_template_participant) point at globals only (the
// pinned `participant_scope='global'` echo + composite FK make that
// declarative) and carry per-membership overrides. A space's/template's
// participant set = its owned rows ∪ its referenced globals; the effective
// config of a referenced global is COALESCE(override, global config).
// ---------------------------------------------------------------------------

/// A participant's own row — the config lives here (scope-owned).
#[derive(Clone, Debug)]
pub struct ParticipantRow {
    pub id: String,
    pub scope: String,
    pub owner_space_id: Option<String>,
    pub owner_template_id: Option<String>,
    pub kind: String,
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
    pub role: String,
    pub removed_at: Option<i64>,
}

/// The effective (override-resolved) view of one member of a space or template:
/// owned rows contribute their own config; referenced globals contribute
/// `COALESCE(override, global config)`. `source` is "owned" | "referenced";
/// `scope` is the underlying participant row's scope (needed for the composite
/// echo when recording actions or copying references).
#[derive(Clone, Debug)]
pub struct EffectiveParticipantRow {
    pub participant_id: String,
    pub scope: String,
    pub source: String,
    pub kind: String,
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
    pub role: String,
}

/// One reference row (a membership pointing at a global), with its overrides —
/// the unit the projections copy between spaces and templates.
#[derive(Clone, Debug)]
pub struct ParticipantRefRow {
    pub participant_id: String,
    pub role: String,
    pub joined_at: i64,
    pub override_label: Option<String>,
    pub override_model_ref: Option<String>,
    pub override_system_prompt: Option<String>,
    pub override_notify_policy: Option<String>,
}

impl ParticipantRefRow {
    fn from_row(row: &turso::Row) -> Result<Self, AppError> {
        Ok(Self {
            participant_id: row.get::<String>(0).map_err(AppError::db)?,
            role: row.get::<String>(1).map_err(AppError::db)?,
            joined_at: row.get::<i64>(2).map_err(AppError::db)?,
            override_label: row.get::<Option<String>>(3).map_err(AppError::db)?,
            override_model_ref: row.get::<Option<String>>(4).map_err(AppError::db)?,
            override_system_prompt: row.get::<Option<String>>(5).map_err(AppError::db)?,
            override_notify_policy: row.get::<Option<String>>(6).map_err(AppError::db)?,
        })
    }
}

// --- participant table -----------------------------------------------------

/// Insert a participant row with an explicit scope + owner. `provider_id` is
/// transport/forensic linkage (not a mirrored config column); copied instances
/// carry `None`.
///
/// A `scope = 'space'` row **stamps its owner space** — a space's agent roster
/// is part of its configuration footprint, and both doors that grow it (the
/// roster's Add, and the model picker's mint of an agent for an unmatched
/// model) are changes to the space. Instantiation uses this primitive too and
/// takes its stamp back at the end of its own transaction, which is where the
/// birth-vs-change distinction belongs (see [`instantiate_template`]).
#[allow(clippy::too_many_arguments)]
pub async fn insert_participant(
    conn: &Connection,
    id: &str,
    scope: &str,
    owner_space_id: Option<&str>,
    owner_template_id: Option<&str>,
    kind: &str,
    label: &str,
    model_ref: Option<&str>,
    system_prompt: Option<&str>,
    notify_policy: &str,
    role: &str,
    provider_id: Option<&str>,
    created_at: i64,
) -> Result<(), AppError> {
    // **Before the insert, not after.** turso autocommits each statement, so a
    // stamp written afterwards leaves a window in which the new participant is
    // durable and its space still reads untouched — long enough for a disposal
    // to take the space and the row together, after which the stamp updates
    // nothing and the roster change is simply gone. Marking first can only
    // over-mark (a failed insert leaves a space nothing changed looking
    // touched), which costs a listing row.
    if let Some(space_id) = owner_space_id.filter(|_| scope == "space") {
        touch_space(conn, space_id, created_at).await?;
    }
    conn.execute(
        "INSERT INTO participant \
         (id, scope, owner_space_id, owner_template_id, kind, label, model_ref, \
          system_prompt, notify_policy, role, provider_id, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        (
            Value::Text(id.to_string()),
            Value::Text(scope.to_string()),
            opt_str(owner_space_id),
            opt_str(owner_template_id),
            Value::Text(kind.to_string()),
            Value::Text(label.to_string()),
            opt_str(model_ref),
            opt_str(system_prompt),
            Value::Text(notify_policy.to_string()),
            Value::Text(role.to_string()),
            opt_str(provider_id),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert participant: {e}"),
    })?;
    Ok(())
}

/// Fetch one participant's own row by id.
pub async fn get_participant(
    conn: &Connection,
    id: &str,
) -> Result<Option<ParticipantRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, scope, owner_space_id, owner_template_id, kind, label, \
                    model_ref, system_prompt, notify_policy, role, removed_at \
             FROM participant WHERE id = ?1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(ParticipantRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            scope: row.get::<String>(1).map_err(AppError::db)?,
            owner_space_id: row.get::<Option<String>>(2).map_err(AppError::db)?,
            owner_template_id: row.get::<Option<String>>(3).map_err(AppError::db)?,
            kind: row.get::<String>(4).map_err(AppError::db)?,
            label: row.get::<String>(5).map_err(AppError::db)?,
            model_ref: row.get::<Option<String>>(6).map_err(AppError::db)?,
            system_prompt: row.get::<Option<String>>(7).map_err(AppError::db)?,
            notify_policy: row.get::<String>(8).map_err(AppError::db)?,
            role: row.get::<String>(9).map_err(AppError::db)?,
            removed_at: row.get::<Option<i64>>(10).map_err(AppError::db)?,
        })),
    }
}

/// Update a participant's own config columns (label, model_ref, system_prompt,
/// notify_policy, role). Editing a global edits it everywhere; editing an owned
/// row edits that space/template only. Each `Some` replaces; the inner `Option`
/// clears/sets the two nullable columns.
///
/// **Liveness rides the `WHERE`, so retirement wins terminally** (Codex review,
/// PR #279). Save and Retire are independent operations in independent slots,
/// and a `bridge`d core call runs to completion even after its gpui task is
/// replaced — so a Save started in another window can commit *after* a
/// retirement. Updating by `id` alone let that stale write rename a retired
/// participant, and since the trail renders an author's byline by joining this
/// row, the record retirement promises to leave alone changed silently
/// underneath it. Asking `removed_at` here rather than in a caller's earlier
/// read is what makes the answer terminal: no interleaving can put a live read
/// in front of a write that lands after the retirement.
///
/// `false` therefore means "nothing to set, or nothing live to set it on"; the
/// caller distinguishes them (the read only *explains* a write that struck
/// nothing — it never decides one).
///
/// **The pristineness stamp is written first, and unconditionally.** turso
/// autocommits each statement, so stamping after a successful update leaves a
/// window in which the edit is durable and the space still reads as untouched
/// — precisely the state a reaper must never see. Stamping first can only
/// over-mark (a refused write leaves a space that nothing changed looking
/// touched), which costs a Library row; the other order costs the edit.
pub async fn update_participant_config(
    conn: &Connection,
    id: &str,
    write: &PersonaWrite,
    now: i64,
) -> Result<bool, AppError> {
    let mut sets: Vec<String> = Vec::new();
    let mut params: Vec<Value> = vec![Value::Text(id.to_string())];
    if let Some(l) = &write.label {
        params.push(Value::Text(l.clone()));
        sets.push(format!("label = ?{}", params.len()));
    }
    if let Some(m) = &write.model_ref {
        params.push(opt_str(m.as_deref()));
        sets.push(format!("model_ref = ?{}", params.len()));
    }
    if let Some(s) = &write.system_prompt {
        params.push(opt_str(s.as_deref()));
        sets.push(format!("system_prompt = ?{}", params.len()));
    }
    if let Some(n) = &write.notify_policy {
        params.push(Value::Text(n.clone()));
        sets.push(format!("notify_policy = ?{}", params.len()));
    }
    if let Some(r) = &write.role {
        params.push(Value::Text(r.clone()));
        sets.push(format!("role = ?{}", params.len()));
    }
    if sets.is_empty() {
        return Ok(false);
    }
    // Liveness, plus whatever the caller believed it was editing.
    let mut premise = String::from(" AND removed_at IS NULL");
    match &write.premise {
        ScopePremise::Any => {}
        ScopePremise::Global => premise.push_str(" AND scope = 'global'"),
        ScopePremise::SpaceOwned { space_id } => {
            params.push(Value::Text(space_id.clone()));
            premise.push_str(&format!(
                " AND scope = 'space' AND owner_space_id = ?{}",
                params.len()
            ));
        }
    }
    let sql = format!(
        "UPDATE participant SET {} WHERE id = ?1{premise}",
        sets.join(", ")
    );
    touch_space_of_participant(conn, id, now).await?;
    let n = conn.execute(&sql, params).await.map_err(AppError::db)?;
    Ok(n > 0)
}

/// Whether a participant row exists and has not been retired/removed — the
/// liveness question a write asks **inside its own transaction** when it cannot
/// carry the predicate in its statement (see `memory::Inner::remember`).
pub async fn participant_is_live(conn: &Connection, id: &str) -> Result<bool, AppError> {
    Ok(get_participant(conn, id)
        .await?
        .is_some_and(|p| p.removed_at.is_none()))
}

/// Soft-remove a participant row (global: library soft-remove; owned:
/// left/deactivated). The row survives so `action.participant_id` references
/// stay resolvable.
///
/// **Private, like every other raw membership write in this module.** Retiring
/// a shared agent is not one statement: it has to archive the spaces that
/// existed only because of that agent — its notebook, and every sub-space it
/// owned — or it leaves rooms the Library still lists whose owner is absent
/// from their own roster. [`retire_participant_tx`] is that operation, and
/// this is one third of it. Exported, it would have been a way to perform the
/// first third alone.
async fn soft_remove_participant(conn: &Connection, id: &str, now: i64) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE participant SET removed_at = ?2 WHERE id = ?1 AND removed_at IS NULL",
            (Value::Text(id.to_string()), Value::Integer(now)),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

/// Owned participants of a space (`scope='space'`, not removed).
pub async fn list_space_owned_participants(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<ParticipantRow>, AppError> {
    owned_participants(conn, "owner_space_id", space_id).await
}

/// Owned participants of a template (`scope='template'`, not removed).
pub async fn list_template_owned_participants(
    conn: &Connection,
    template_id: &str,
) -> Result<Vec<ParticipantRow>, AppError> {
    owned_participants(conn, "owner_template_id", template_id).await
}

async fn owned_participants(
    conn: &Connection,
    owner_col: &str,
    owner_id: &str,
) -> Result<Vec<ParticipantRow>, AppError> {
    let sql = format!(
        "SELECT id, scope, owner_space_id, owner_template_id, kind, label, \
                model_ref, system_prompt, notify_policy, role, removed_at \
         FROM participant WHERE {owner_col} = ?1 AND removed_at IS NULL \
         ORDER BY created_at, id"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(owner_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(ParticipantRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            scope: row.get::<String>(1).map_err(AppError::db)?,
            owner_space_id: row.get::<Option<String>>(2).map_err(AppError::db)?,
            owner_template_id: row.get::<Option<String>>(3).map_err(AppError::db)?,
            kind: row.get::<String>(4).map_err(AppError::db)?,
            label: row.get::<String>(5).map_err(AppError::db)?,
            model_ref: row.get::<Option<String>>(6).map_err(AppError::db)?,
            system_prompt: row.get::<Option<String>>(7).map_err(AppError::db)?,
            notify_policy: row.get::<String>(8).map_err(AppError::db)?,
            role: row.get::<String>(9).map_err(AppError::db)?,
            removed_at: row.get::<Option<i64>>(10).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// Hard-delete a template's owned participants (used to replace the set on
/// update). Safe: template-owned rows are never referenced (references point at
/// globals only, and actions can't reference template scope).
pub async fn delete_template_owned_participants(
    conn: &Connection,
    template_id: &str,
) -> Result<(), AppError> {
    conn.execute(
        "DELETE FROM participant WHERE scope = 'template' AND owner_template_id = ?1",
        (Value::Text(template_id.to_string()),),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

// --- promotion (task 36): space-owned agent → global identity --------------

/// A **validated** persona to write onto a participant's own config columns —
/// the resolved form of a `ParticipantUpdate` (label trimmed and checked,
/// notify policy checked, the two nullable columns already reduced to
/// set-or-clear). Produced once in `lib.rs` so an ordinary edit and a promotion
/// that carries one cannot drift on what an update *means*.
#[derive(Clone, Debug, Default)]
pub struct PersonaWrite {
    pub label: Option<String>,
    pub model_ref: Option<Option<String>>,
    pub system_prompt: Option<Option<String>>,
    pub notify_policy: Option<String>,
    /// The participant's `role` column — only the seeding paths set it.
    pub role: Option<String>,
    /// **What the caller believed it was editing.** Liveness is not the whole
    /// premise of a config write: an editor opened on a *space-owned* row was
    /// composed against a persona that promotion moves out from under it, and a
    /// write that carries only `removed_at IS NULL` will happily republish the
    /// old values to every space the agent has since joined. Carried into the
    /// statement, the premise makes that stale write strike nothing (Codex
    /// review, PR #279).
    pub premise: ScopePremise,
}

/// The scope a config write was composed against.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ScopePremise {
    /// No premise — the write is about a row by id alone.
    #[default]
    Any,
    /// This space's own row (the inspector's owned-agent editor).
    SpaceOwned { space_id: String },
    /// A shared identity (the Agents pane; the inspector's "Everyone" mode).
    Global,
}

impl PersonaWrite {
    /// Whether it would write nothing at all (every column left alone).
    pub fn is_empty(&self) -> bool {
        self.label.is_none()
            && self.model_ref.is_none()
            && self.system_prompt.is_none()
            && self.notify_policy.is_none()
    }

    /// Apply it, reporting whether any row changed. Every caller writes through
    /// here, so "a persona write" is one statement builder
    /// ([`update_participant_config`]) with one set of semantics.
    pub async fn apply(
        &self,
        conn: &Connection,
        participant_id: &str,
        now: i64,
    ) -> Result<bool, AppError> {
        update_participant_config(conn, participant_id, self, now).await
    }
}

/// A membership to write **inside** another transaction — task 37's grant.
///
/// The blocked-follow → grant → retry loop's middle step is ordinary
/// membership, and for a space-owned agent it can only follow a promotion. Two
/// calls would be two transactions: the promotion could land and the grant be
/// refused, leaving an agent shared (irreversibly — promotion is one-way) for a
/// grant the reader asked for and did not get. So the grant travels *with* the
/// promotion, as an argument, exactly as the persona does.
#[derive(Clone, Debug)]
pub struct MembershipGrant<'a> {
    pub space_id: &'a str,
    pub role: &'a str,
}

/// What [`promote_participant_tx`] writes — one struct rather than nine
/// positional arguments.
pub struct Promotion<'a> {
    pub participant_id: &'a str,
    pub home_space_id: &'a str,
    pub role: &'a str,
    pub notebook_space_id: &'a str,
    pub notebook_title: &'a str,
    /// A membership in **another** space to grant in the same transaction (task
    /// 37's "Share this agent and add it to *A* as an observer"). The caller has
    /// already established that the space exists and is not the home space (a
    /// second insert on the same PK would fail the whole transaction).
    pub grant: Option<MembershipGrant<'a>>,
    /// A persona to adopt **in the promoting transaction**, if the caller is
    /// carrying one (the GUI's "Share this agent…" promotes what its editor is
    /// showing). Inside the same transaction *and behind the same guard*, so a
    /// promotion that loses a race takes the persona write back with it.
    pub persona: Option<&'a PersonaWrite>,
    pub now: i64,
}

/// Promote a space-owned agent to a **global** identity, in place and in one
/// transaction. Same row, same id — so authorship, provenance and memory
/// continuity are structural rather than stitched. Copy-projection was
/// rejected: it fragments past posts, memory blocks and reference edges across
/// two identities, which is the opposite of what promotion exists to create.
///
/// Four writes, all or nothing:
///
/// 1. `owner_space_id → NULL`, `scope → 'global'` on the participant row. The
///    pinned `participant_scope` echo on every past `action` and every
///    `memory_block` follows via `ON UPDATE CASCADE` — see the schema comments
///    there and `turso_enforcement_smoke` case (e).
/// 2. The optional [`Promotion::persona`], onto the row that statement just
///    proved was still ours to promote.
/// 3. A `space_participant` reference row for the **former owner space**, with
///    NULL overrides. Ownership no longer implies membership, and NULL
///    overrides mean the effective config (`COALESCE(override, config)`) is
///    byte-identical to what it was: the space's persona is preserved exactly.
/// 4. The agent's private **notebook space** (`space.notebook_participant_id`),
///    referenced into by the agent itself, hidden from the Library listing.
///
/// The caller has already validated *what* may be promoted (kind, scope, the
/// shared "User", removal) and what the persona may say; this is the mechanics —
/// **plus the one check that cannot be done ahead of the transaction**, below.
pub(crate) async fn promote_participant_tx(
    conn: &Connection,
    promotion: &Promotion<'_>,
) -> Result<(), AppError> {
    begin_write(conn).await?;
    match promote_participant_tx_body(conn, promotion).await {
        Ok(()) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(())
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn promote_participant_tx_body(
    conn: &Connection,
    promotion: &Promotion<'_>,
) -> Result<(), AppError> {
    let participant_id = promotion.participant_id;
    // Sharing an agent changes the space that owned it — it loses an owned
    // participant and gains a reference to a shared one. Stamped explicitly
    // rather than left to the membership insert below, because the flip is
    // what the change *is*, and `touch_space_of_participant` cannot see it
    // afterwards (the statement nulls the owner it would have derived from).
    touch_space(conn, promotion.home_space_id, promotion.now).await?;
    // Both key columns move in one statement: the three-way scope/owner CHECK
    // must hold at statement end, and the cascade fires off the (id, scope)
    // parent key.
    //
    // **This statement is the transaction's own gate**, not a repeat of the
    // caller's. The caller read the row a moment ago; another window sharing or
    // removing the same agent in between is what the predicate catches, and
    // catching it *here* is what lets everything after it — the persona above
    // all — roll back as one. `removed_at` rides the same predicate for the
    // same reason.
    let n = conn
        .execute(
            "UPDATE participant SET scope = 'global', owner_space_id = NULL \
             WHERE id = ?1 AND scope = 'space' AND removed_at IS NULL",
            (Value::Text(participant_id.to_string()),),
        )
        .await
        .map_err(AppError::db)?;
    if n == 0 {
        return Err(AppError::Database {
            message: format!(
                "participant {participant_id} was no longer a live space-owned row to promote"
            ),
        });
    }
    if let Some(persona) = promotion.persona {
        persona.apply(conn, participant_id, promotion.now).await?;
    }
    insert_space_participant(
        conn,
        promotion.home_space_id,
        participant_id,
        promotion.role,
        promotion.now,
    )
    .await?;
    insert_notebook_space(
        conn,
        promotion.notebook_space_id,
        participant_id,
        promotion.notebook_title,
        promotion.now,
    )
    .await?;
    insert_space_participant(
        conn,
        promotion.notebook_space_id,
        participant_id,
        "owner",
        promotion.now,
    )
    .await?;
    // Task 37's grant. Inside the transaction the scope flip has just proved
    // this row live and space-owned, so the membership cannot be joined onto a
    // retired agent and a refusal anywhere above takes the grant back with it.
    if let Some(grant) = &promotion.grant {
        insert_space_participant(
            conn,
            grant.space_id,
            participant_id,
            grant.role,
            promotion.now,
        )
        .await?;
    }
    Ok(())
}

/// Create an agent's private notebook space. A real space in every respect —
/// only `notebook_participant_id` distinguishes it, which is what the Library
/// listing filters on and what the agent-management surface will look it up by.
pub async fn insert_notebook_space(
    conn: &Connection,
    id: &str,
    participant_id: &str,
    title: &str,
    created_at: i64,
) -> Result<(), AppError> {
    // Born stamped: a notebook exists because a human shared an agent, which
    // is as deliberate a gesture as naming a conversation. The disposal
    // predicate excludes notebooks outright as well — two locks on the one
    // mistake that would cost an agent the residence of its core memory.
    conn.execute(
        "INSERT INTO space (id, parent_space_id, title, linkability, \
                            notebook_participant_id, created_at, touched_at) \
         VALUES (?1, NULL, ?2, 'unlinked', ?3, ?4, ?4)",
        (
            Value::Text(id.to_string()),
            Value::Text(title.to_string()),
            Value::Text(participant_id.to_string()),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert notebook space: {e}"),
    })?;
    Ok(())
}

/// The notebook space of a global agent, if it has one.
pub async fn notebook_space_for(
    conn: &Connection,
    participant_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare("SELECT id FROM space WHERE notebook_participant_id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(participant_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// One row of the **global agent library** — a shared identity plus the door to
/// its notebook (see [`list_global_agents`]).
#[derive(Clone, Debug)]
pub struct GlobalAgentRow {
    pub id: String,
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
    /// The agent's private notebook space, `None` for a global agent that has
    /// none. Only promotion creates one, and it does so in the same
    /// transaction as the scope flip — so a `None` here is a row from some
    /// future path that mints globals another way, never a half-promoted one.
    pub notebook_space_id: Option<String>,
}

/// The live **global agents** — the shared agent library (task 36).
///
/// `kind = 'agent'` is what keeps the two seeded non-agent globals out: the
/// shared human ("User") and Eidola-the-system are global rows too, and neither
/// is a colleague anyone manages. The notebook is **joined here rather than
/// looked up per row**, because the only consumer is a roster that offers the
/// notebook door on every line.
///
/// Ordered by the label the reader sees (id breaking ties), so the management
/// surface and any future listing agree on an order a person can predict.
pub async fn list_global_agents(conn: &Connection) -> Result<Vec<GlobalAgentRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT p.id, p.label, p.model_ref, p.system_prompt, p.notify_policy, s.id \
             FROM participant p \
             LEFT JOIN space s ON s.notebook_participant_id = p.id \
             WHERE p.scope = 'global' AND p.kind = 'agent' AND p.removed_at IS NULL \
             ORDER BY p.label, p.id",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(GlobalAgentRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            label: row.get::<String>(1).map_err(AppError::db)?,
            model_ref: row.get::<Option<String>>(2).map_err(AppError::db)?,
            system_prompt: row.get::<Option<String>>(3).map_err(AppError::db)?,
            notify_policy: row.get::<String>(4).map_err(AppError::db)?,
            notebook_space_id: row.get::<Option<String>>(5).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// What [`grant_space_membership_tx`] did — the decision it took *inside* the
/// transaction, reported so the caller can build its outcome and decide what to
/// announce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrantDecision {
    /// The row was space-owned: promoted, and the membership written in the
    /// same transaction. Carries what the promotion minted.
    Promoted {
        home_space_id: String,
        notebook_space_id: String,
    },
    /// The row was already a shared global: membership joined (or revived).
    Joined,
    /// It was already a live member of this space — nothing to do. A competing
    /// promotion that granted this very destination lands here.
    AlreadyAMember,
}

/// **Grant `participant_id` membership of `space_id`, promoting first if — and
/// only if — the row is still space-owned.** Task 37's grant, with the branch
/// moved inside the transaction.
///
/// The picker's snapshot cannot decide this. It records whether a candidate was
/// shared when the *list* landed, and another window sharing that agent before
/// the reader confirms makes the snapshot a lie: a caller branching on it asked
/// for a promotion of an already-global row, which is refused — for a
/// membership that could simply have been added, and (when the competing
/// promotion granted this very space) about a state that already held (Codex
/// review, PR #280). So the scope is read where the write happens, under the
/// same `BEGIN`, and each branch still carries its own premise into its own
/// statement: `promote_participant_tx_body`'s `WHERE scope = 'space'` and
/// `ensure_space_participant`'s `WHERE EXISTS (… removed_at IS NULL)`.
///
/// The caller's up-front refusals (unknown space, unknown/retired participant,
/// not an agent, the shared human, a template-scoped row) still belong to the
/// caller: they are typed errors a reader can act on, and finding them out in
/// here would be a rollback where a message belongs.
pub(crate) async fn grant_space_membership_tx(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    notebook_space_id: &str,
    now: i64,
) -> Result<GrantOutcome, AppError> {
    begin_write(conn).await?;
    let body = grant_space_membership_tx_body(
        conn,
        space_id,
        participant_id,
        role,
        notebook_space_id,
        now,
    )
    .await;
    match body {
        Ok(outcome) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(outcome)
        }
        Err(e) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

/// Join a global into a space and answer with the membership **as of the
/// commit** — the insert-or-revive plus the read that describes it, in one
/// transaction.
///
/// `add_global_participant` used to run the join and then read the roster
/// outside any transaction, which is the same window the grant had: a removal
/// or retirement landing in between made the read find nothing, and the call
/// reported a failure for a join that had committed (Codex review, PR #280).
/// `None` means the join struck nothing — the liveness premise expired — with
/// nothing written and nothing to announce.
pub(crate) async fn join_space_participant_tx(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    now: i64,
) -> Result<Option<(bool, EffectiveParticipantRow)>, AppError> {
    begin_write(conn).await?;
    let body = async {
        // A sub-space has exactly one owner and it is minted with the room.
        refuse_second_subspace_owner(conn, space_id, role, participant_id).await?;
        let joined = ensure_space_participant(conn, space_id, participant_id, role, now).await?;
        let member = space_participants(conn, space_id)
            .await?
            .into_iter()
            .find(|m| m.participant_id == participant_id);
        Ok::<_, AppError>(member.map(|m| (joined, m)))
    }
    .await;
    match body {
        Ok(found) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(found)
        }
        Err(e) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

/// What the grant did **and what it left behind** — the membership as of the
/// commit point.
///
/// The caller's answer is built here rather than from a roster read *after* the
/// transaction, because between that commit and that read another window can
/// remove the membership or retire the agent: the read then finds nothing and
/// the call returns an error about an operation that committed — and, for a
/// space-owned candidate, committed an **irreversible promotion** (Codex
/// review, PR #280). That is precisely the failure-message-beside-a-committed-
/// write state this transaction exists to prevent, reintroduced one line after
/// it. A result that describes the commit point cannot be overtaken by what
/// happens next.
#[derive(Clone, Debug)]
pub struct GrantOutcome {
    pub decision: GrantDecision,
    /// The membership as the transaction wrote it — effective label, role and
    /// overrides, read inside the same `BEGIN IMMEDIATE`.
    pub member: EffectiveParticipantRow,
}

async fn grant_space_membership_tx_body(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    notebook_space_id: &str,
    now: i64,
) -> Result<GrantOutcome, AppError> {
    let row = get_participant(conn, participant_id)
        .await?
        .ok_or_else(|| AppError::NotConfigured {
            message: format!("participant not found: {participant_id}"),
        })?;
    if row.removed_at.is_some() {
        return Err(AppError::Config {
            message: format!("{} has been retired and cannot rejoin a space", row.label),
        });
    }
    // A sub-space has exactly one owner and it is minted with the room. Asked
    // before the already-a-member shortcut so the refusal is about the role the
    // caller asked for rather than silently satisfied by a lesser membership.
    refuse_second_subspace_owner(conn, space_id, role, participant_id).await?;
    // Already a live member: satisfied. This is the whole point of deciding
    // here — the competing promotion may have granted this very space, and an
    // operation that reports failure about a state that already holds is
    // telling the reader something untrue.
    let already = space_participants(conn, space_id)
        .await?
        .into_iter()
        .find(|m| m.participant_id == participant_id);
    if let Some(member) = already {
        return Ok(GrantOutcome {
            decision: GrantDecision::AlreadyAMember,
            member,
        });
    }
    let decision = match row.scope.as_str() {
        "global" => {
            ensure_space_participant(conn, space_id, participant_id, role, now).await?;
            GrantDecision::Joined
        }
        "space" => {
            let home_space_id = row
                .owner_space_id
                .clone()
                .ok_or_else(|| AppError::Internal {
                    message: "space-owned participant has no owner space".into(),
                })?;
            // The promotion writes the home membership itself, so a grant
            // naming the home space is satisfied by it rather than written
            // twice (a second insert on the same key fails the transaction) —
            // `promote_participant`'s own rule, applied here.
            let grant = (space_id != home_space_id).then_some(MembershipGrant { space_id, role });
            let notebook_title = format!("{} — notebook", row.label);
            promote_participant_tx_body(
                conn,
                &Promotion {
                    participant_id,
                    home_space_id: &home_space_id,
                    role: &row.role,
                    notebook_space_id,
                    notebook_title: &notebook_title,
                    grant,
                    persona: None,
                    now,
                },
            )
            .await?;
            GrantDecision::Promoted {
                home_space_id,
                notebook_space_id: notebook_space_id.to_string(),
            }
        }
        other => {
            return Err(AppError::Config {
                message: format!(
                    "{} belongs to a space template, not a space, so it cannot be given \
                     membership (scope: {other})",
                    row.label
                ),
            });
        }
    };
    // Read **inside** the transaction: the membership this write just made, as
    // of the commit point. A row missing here is a genuine failure of the write
    // (and rolls back with it), never the race a post-commit read invents.
    let member = space_participants(conn, space_id)
        .await?
        .into_iter()
        .find(|m| m.participant_id == participant_id)
        .ok_or_else(|| AppError::Internal {
            message: format!("membership for {participant_id} vanished inside its own write"),
        })?;
    Ok(GrantOutcome { decision, member })
}

/// One candidate for task 37's grant — an agent a reader could give membership
/// of a space (see [`list_grantable_agents`]).
#[derive(Clone, Debug)]
pub struct GrantableAgentRow {
    pub id: String,
    pub label: String,
    /// Already a shared identity, so the grant is plain membership. `false` is
    /// a space-owned agent, whose grant is promotion + membership in one
    /// transaction ([`promote_participant_tx`] carrying a [`MembershipGrant`]).
    pub shared: bool,
    /// The title of the space that owns it (`None` when shared, or when its
    /// home space is untitled) — what names the agent to a reader who knows it
    /// from somewhere else.
    pub home_space_title: Option<String>,
}

/// The agents that could be granted membership of `space_id`, as the viewer
/// may see them (task 37's grant picker).
///
/// Three exclusions, each a rule rather than a nicety:
///
/// - **Already a member** — of either kind (a space's own agent, or a global
///   already referenced in). Offering a grant that would change nothing is the
///   fake affordance the honest-states rule exists to prevent.
/// - **Not an agent, or retired** — the shared human and Eidola-the-system are
///   globals nobody grants, and a retired agent cannot rejoin a space
///   (`add_global_participant` refuses one; the picker must not offer it).
/// - **Owned by a space the viewer does not take part in** — the listing is a
///   read like any other, so it obeys the same ACL: without this it would
///   announce the existence of agents (and, through `home_space_title`, of
///   conversations) in spaces the viewer has no part in, which is exactly the
///   leak rule 4 exists to close. A global has no home space to leak, so it is
///   listed unconditionally.
///
/// Ordered by label, the name the reader sees.
pub async fn list_grantable_agents(
    conn: &Connection,
    space_id: &str,
    viewer_participant_id: &str,
) -> Result<Vec<GrantableAgentRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT p.id, p.label, p.scope = 'global', s.title \
             FROM participant p \
             LEFT JOIN space s ON s.id = p.owner_space_id \
             WHERE p.kind = 'agent' AND p.removed_at IS NULL \
               AND p.scope IN ('global', 'space') \
               AND COALESCE(p.owner_space_id, '') <> ?1 \
               AND NOT EXISTS ( \
                   SELECT 1 FROM space_participant r \
                   WHERE r.participant_id = p.id AND r.space_id = ?1 \
                     AND r.left_at IS NULL \
               ) \
               AND ( \
                   p.scope = 'global' \
                   OR EXISTS ( \
                       SELECT 1 FROM participant v \
                       WHERE v.id = ?2 AND v.owner_space_id = p.owner_space_id \
                         AND v.removed_at IS NULL \
                   ) \
                   OR EXISTS ( \
                       SELECT 1 FROM space_participant vr \
                       JOIN participant v ON v.id = vr.participant_id \
                       WHERE vr.participant_id = ?2 \
                         AND vr.space_id = p.owner_space_id \
                         AND vr.left_at IS NULL AND v.removed_at IS NULL \
                   ) \
               ) \
             ORDER BY p.label, p.id",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(space_id.to_string()),
            Value::Text(viewer_participant_id.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        let shared = row.get::<i64>(2).map_err(AppError::db)? != 0;
        out.push(GrantableAgentRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            label: row.get::<String>(1).map_err(AppError::db)?,
            shared,
            home_space_title: row.get::<Option<String>>(3).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// What a retirement did: whether it retired anything, and how many spaces the
/// **Library lists** it archived on the way (the notebook is never one — the
/// listing excludes notebooks in both its branches — so this counts sub-spaces
/// and is what decides whether the caller announces a listing change).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Retirement {
    pub retired: bool,
    /// How many spaces the Library lists were archived — what the `SpaceIndex`
    /// emission is decided from. Never counts the notebook.
    pub listed_spaces_archived: i64,
    /// **Every space this retirement closed**, notebook included, so the
    /// caller can do what every archival door does: announce each one and end
    /// any wait it was holding. A count cannot say *which*, and the rooms are
    /// exactly the things a delegation can be registered against.
    pub archived_spaces: Vec<String>,
}

/// Retire a global agent: the soft-remove **and** the archival of every space
/// that exists only because of it — its notebook, and any sub-space it still
/// owns — in one transaction ("archival tied to retirement").
///
/// **The notebook** belongs here because it exists only for the agent: a
/// retired agent whose notebook stayed open would leave a live space nobody
/// owns, and an archived notebook beside a live agent would take away the
/// residence its next `core` memory block needs.
///
/// **A sub-space it owns belongs here for the first of those reasons, sharpened
/// by what ownership carries.** Retirement soft-removes the participant and
/// deliberately leaves its membership rows standing, and every membership
/// *question* then answers "no" ([`is_space_member`], [`space_participants`]) —
/// so a live sub-space left behind is a room the Library still lists, whose
/// owner is named by [`subspace_owner_of`] and absent from its own roster,
/// whose planning can never reach that agent again, and which still counts
/// against a live-room quota nothing can now spend. No door could repair it
/// either: `refuse_second_subspace_owner` correctly refuses a replacement
/// owner, because ownership is not transferable. Archiving dissolves the
/// question instead of answering it — the room stops being live, so it holds
/// no quota, schedules nothing, and is no longer a Library row promising work
/// that cannot happen. The transcript survives in full: archival is a
/// visibility choice, not a deletion, and neither membership nor the human
/// read bypass filters on it, so everyone who could read the room yesterday
/// still can.
///
/// **Refusing the retirement was the alternative and was rejected.** It would
/// let one stuck delegation make a shared agent unretirable, which is the
/// opposite of what retirement is for, and it would contradict the notebook
/// arm standing beside it in this very function.
///
/// **The agent's own rooms are selected by ownership, and what hangs beneath
/// them goes with them.** The statement selects by *ownership*, not by descent,
/// so it reaches the agent's sub-spaces at every depth and nobody else's rooms
/// *by that clause* — and then [`archive_rooms_under_a_closed_one`] closes what
/// those rooms had delegated onward, whoever owns it. A nested room's owner is
/// not being retired, but its *purpose* is: it exists to serve the conversation
/// above it, and that conversation is closed, so its report is a turn an
/// archived parent refuses and its delegation would stay outstanding forever.
///
/// The participant row itself always survives (soft-remove), so every
/// `action.participant_id` in the trail stays resolvable; retirement is about
/// the *library*, never the history.
pub(crate) async fn retire_participant_tx(
    conn: &Connection,
    participant_id: &str,
    now: i64,
) -> Result<Retirement, AppError> {
    begin_write(conn).await?;
    match retire_participant_tx_body(conn, participant_id, now).await {
        Ok(retirement) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(retirement)
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn retire_participant_tx_body(
    conn: &Connection,
    participant_id: &str,
    now: i64,
) -> Result<Retirement, AppError> {
    if !soft_remove_participant(conn, participant_id, now).await? {
        return Ok(Retirement {
            retired: false,
            listed_spaces_archived: 0,
            archived_spaces: Vec::new(),
        });
    }
    // **Read before written, on both arms**, because the caller needs the ids
    // and not only the count: every door that closes a room announces it and
    // ends any wait it was holding (see [`Retirement::archived_spaces`]), and a
    // rows-affected figure cannot say *which*. The predicate is the same in the
    // select and the update, so the two describe one set — nothing can land in
    // between, the whole body being one `BEGIN IMMEDIATE` transaction.
    const LIVE_NOTEBOOK_OF: &str = "notebook_participant_id = ?1 AND archived_at IS NULL";
    // Owner-scoped, so depth is irrelevant and nobody else's room is reached.
    // The membership predicate is [`SUBSPACE_OWNER_SQL`]'s, minus its
    // tie-break: that fragment names *the* owner where this asks *is an owner*,
    // which is the same set while the write guards hold and the safer of the
    // two if one ever did not.
    const LIVE_ROOMS_OWNED_BY: &str = "archived_at IS NULL AND parent_space_id IS NOT NULL \
               AND EXISTS ( \
                   SELECT 1 FROM space_participant r \
                   WHERE r.space_id = space.id AND r.participant_id = ?1 \
                     AND r.role = 'owner' AND r.left_at IS NULL \
               )";
    let notebook = space_ids_where(conn, LIVE_NOTEBOOK_OF, participant_id).await?;
    let rooms = space_ids_where(conn, LIVE_ROOMS_OWNED_BY, participant_id).await?;
    conn.execute(
        &format!(
            "UPDATE space SET archived_at = ?2, touched_at = COALESCE(touched_at, ?2) \
         WHERE {LIVE_NOTEBOOK_OF}"
        ),
        (Value::Text(participant_id.to_string()), Value::Integer(now)),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to archive the retired agent's notebook: {e}"),
    })?;
    let archived = conn
        .execute(
            &format!(
                "UPDATE space SET archived_at = ?2, touched_at = COALESCE(touched_at, ?2) \
             WHERE {LIVE_ROOMS_OWNED_BY}"
            ),
            (Value::Text(participant_id.to_string()), Value::Integer(now)),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to archive the retired agent's sub-spaces: {e}"),
        })?;
    // And the delegations *those* rooms were themselves running, at any depth
    // and whoever owns them — see [`archive_rooms_under_a_closed_one`]. A room
    // exists to serve the conversation above it, so one that can never report
    // is not a room somebody else is still answerable for; it is work nobody
    // will ever be told the result of. Counted into the listing figure because
    // every sub-space is a Library row.
    let beneath = archive_rooms_under_a_closed_one(conn, now).await?;
    Ok(Retirement {
        retired: true,
        // The *listing* figure counts what the Library shows: every sub-space,
        // and never the notebook, which it has never listed.
        listed_spaces_archived: archived as i64 + beneath.len() as i64,
        // The *announcement* set is every space this closed, notebook
        // included — a room can be opened from one, so a wait can be
        // registered against one.
        archived_spaces: notebook.into_iter().chain(rooms).chain(beneath).collect(),
    })
}

/// The ids of the spaces matching `predicate`, whose single bind parameter is
/// `?1` — the read half of an archival that has to name what it closed.
async fn space_ids_where(
    conn: &Connection,
    predicate: &str,
    bind: &str,
) -> Result<Vec<String>, AppError> {
    let mut stmt = conn
        .prepare(&format!("SELECT id FROM space WHERE {predicate}"))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(bind.to_string()),))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(row.get::<String>(0).map_err(AppError::db)?);
    }
    Ok(out)
}

/// Whether a participant is a member of one space — **owned row ∪ live
/// reference row, and in both cases a live participant**: the same membership
/// definition [`participant_spaces`] and [`space_participants`] read from their
/// two sides, asked of a single space. This is the cross-space ACL (task 37):
/// reference *creation* and reference *following* both gate on it.
///
/// **Membership means live membership.** Retirement soft-removes the
/// participant and deliberately leaves its `space_participant` rows standing —
/// the trail still records where the agent worked — so a question that asked
/// only about the row went on answering "member" for an agent the human had
/// just taken out of service. A turn binds its tools to the responding
/// participant when it starts, so a retirement landing between two rounds left
/// a live `read_post` holding a retired id, still opening the conversations it
/// had belonged to (Codex review, PR #279). What ends availability is therefore
/// the *question*, asked here at the boundary rather than by sweeping `left_at`
/// onto every membership row: one rule that covers any future
/// availability-ending state, that leaves the historical record undisturbed,
/// and that no racing grant can slip past (`add_global_participant` refuses a
/// retired participant, and a row created anyway would still not answer here).
/// `space_participants` already read membership this way; these two had drifted
/// from the definition their own docs claimed. Rendering paths are untouched on
/// purpose — "who wrote this post" must keep naming retired authors.
///
/// Deliberately **not** filtered on `space.archived_at`: archiving is a Library
/// visibility choice, not a departure. A member of an archived space is still a
/// member, and quoting or following into one is exactly as permitted as it was
/// the day before it was archived. (`participant_spaces` filters archived spaces
/// because it renders a *listing*.)
pub async fn is_space_member(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
) -> Result<bool, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT EXISTS ( \
                 SELECT 1 FROM participant p \
                 WHERE p.id = ?2 AND p.owner_space_id = ?1 AND p.removed_at IS NULL \
             ) OR EXISTS ( \
                 SELECT 1 FROM space_participant r \
                 JOIN participant p ON p.id = r.participant_id \
                 WHERE r.participant_id = ?2 AND r.space_id = ?1 \
                   AND r.left_at IS NULL AND p.removed_at IS NULL \
             )",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(space_id.to_string()),
            Value::Text(participant_id.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(false),
        Some(row) => Ok(row.get::<i64>(0).map_err(AppError::db)? != 0),
    }
}

/// Whether a viewer may **read** a space: a live member, or a human reading
/// anything that is not an agent's notebook.
///
/// This is the one place membership is widened, and every edge of the widening
/// is deliberate.
///
/// **Why it exists.** An agent-spawned sub-space has no human member by
/// construction, so under membership alone the rooms an agent opens on the
/// human's behalf — spending the human's account, on work the human asked for
/// — would be the one thing the human could not oversee. Oversight is the
/// justification, so the bypass is scoped to exactly what oversight needs:
///
/// * **Reads only.** Writing into a space still means joining it, so the
///   roster the models see stays truthful about who is in the room.
/// * **Human viewers only** (`kind = 'human'`, live). Every model-facing gate
///   goes on asking [`is_space_member`] — `read_post`'s per-call follow check,
///   the reference-authoring gate, `participant_spaces` — so nothing an agent
///   can reach is affected by this function's existence.
/// * **Never a notebook.** A notebook is an agent's own residence rather than
///   its work: the space the cross-space permission model names as *the* privacy this rule exists for,
///   and the one whose backlinks are already withheld from the human who opens
///   it from Settings. A sub-space is the opposite — delegated work, on the
///   reader's behalf, with nothing private about it. Keeping the notebook
///   clause is what lets both facts stand at once, and it is the whole of the
///   difference between them.
///
/// So: a human may read any conversation their agents are having *with each
/// other*, and no conversation an agent is having *with itself*.
///
/// A retired participant answers `false` here as everywhere else; the shared
/// human cannot be retired, so that clause is for future humans rather than
/// today's one.
pub async fn may_read_space(
    conn: &Connection,
    space_id: &str,
    viewer_participant_id: &str,
) -> Result<bool, AppError> {
    if is_space_member(conn, space_id, viewer_participant_id).await? {
        return Ok(true);
    }
    let mut rows = conn
        .query(
            "SELECT EXISTS ( \
                 SELECT 1 FROM participant \
                 WHERE id = ?2 AND kind = 'human' AND removed_at IS NULL \
             ) AND EXISTS ( \
                 SELECT 1 FROM space \
                 WHERE id = ?1 AND notebook_participant_id IS NULL \
             )",
            (
                Value::Text(space_id.to_string()),
                Value::Text(viewer_participant_id.to_string()),
            ),
        )
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(false),
        Some(row) => Ok(row.get::<i64>(0).map_err(AppError::db)? != 0),
    }
}

/// One post read through a `reference` edge (task 37's membership-gated
/// follow): everything needed to render it in the wire format, for an action
/// that may live in another space — so it is a **concrete generation** read by
/// id, not a tree walk. `space_title` is `None` for an untitled space.
#[derive(Clone, Debug)]
pub struct ReferencedPostRow {
    pub space_id: String,
    pub space_title: Option<String>,
    pub item_id: String,
    /// Effective label of the author in *its own* space (the override applies
    /// where the post was written, which is where it is being read from).
    pub participant_label: String,
    pub action_type: String,
    /// The action's `text` blocks, concatenated in ordinal order (`thinking`
    /// and tool blocks are never part of the readable transcript).
    pub text: String,
    /// This generation's creation time, epoch ms — the followed post's header
    /// stamp. A follow names a *concrete* generation and is never remapped, so
    /// this is that generation's own time.
    pub created_at: i64,
}

/// Read one action as a referenced post. `None` when the action doesn't exist
/// **or is not a post**.
///
/// The post-type filter is defense in depth, not the primary gate: a reference
/// may only be *created* against a post (`crate::Inner::validate_reference_spec`
/// refuses anything else), but a reference edge that predates that gate — or
/// arrives through some future writer below it — must not become a door into a
/// first-person tool trace, a decision, or another participant's memory block.
/// The follow reports "not a readable post" and stops.
///
/// The caller is responsible for the membership gate — this is the read that
/// runs *after* it passes.
pub async fn referenced_post(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<ReferencedPostRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT a.space_id, s.title, a.item_id, \
                    COALESCE(sp.override_label, p.label), a.action_type, a.created_at \
             FROM action a \
             JOIN space s ON s.id = a.space_id \
             JOIN participant p ON p.id = a.participant_id \
             LEFT JOIN space_participant sp \
                    ON sp.space_id = a.space_id AND sp.participant_id = a.participant_id \
             WHERE a.id = ?1 AND a.action_type IN ({POST_ACTION_TYPES_SQL})",
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let Some(row) = rows.next().await.map_err(AppError::db)? else {
        return Ok(None);
    };
    let space_id = row.get::<String>(0).map_err(AppError::db)?;
    let space_title = row.get::<Option<String>>(1).map_err(AppError::db)?;
    let item_id = row.get::<String>(2).map_err(AppError::db)?;
    let participant_label = row.get::<String>(3).map_err(AppError::db)?;
    let action_type = row.get::<String>(4).map_err(AppError::db)?;
    let created_at = row.get::<i64>(5).map_err(AppError::db)?;

    // The readable transcript is `text` blocks only — the same rule both
    // context queries apply, so a persisted `thinking` block stays a
    // render-side disclosure and never travels through a follow.
    let mut stmt = conn
        .prepare(
            "SELECT text_content FROM content_block \
             WHERE action_id = ?1 AND block_type = 'text' \
             ORDER BY ordinal ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut text = String::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        if let Some(t) = row.get::<Option<String>>(0).map_err(AppError::db)? {
            text.push_str(&t);
        }
    }

    Ok(Some(ReferencedPostRow {
        space_id,
        space_title,
        item_id,
        participant_label,
        action_type,
        text,
        created_at,
    }))
}

/// Every space a participant is a member of — **owned rows ∪ live references,
/// and in both cases a live participant** ([`is_space_member`] states why
/// retirement has to be visible here). The same membership definition
/// [`space_participants`] reads from the other side. This is the boundary for
/// everything cross-space: task 36's `list_my_spaces` reaches exactly these and
/// nothing else.
///
/// Rows carry the same activity signals as [`list_spaces`], and notebooks are
/// **included** — an agent's own notebook is one of its spaces.
pub async fn participant_spaces(
    conn: &Connection,
    participant_id: &str,
) -> Result<Vec<SpaceListRow>, AppError> {
    let sql = format!(
        "SELECT s.id, s.title, s.created_at, s.archived_at, \
                COALESCE(MAX(a.created_at), s.created_at) AS last_activity_at, \
                COUNT(a.id) AS message_count \
         FROM space s \
         LEFT JOIN action a ON a.space_id = s.id \
              AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
              AND a.status IN ('complete', 'cancelled') \
              AND NOT EXISTS ( \
                  SELECT 1 FROM action sup WHERE sup.supersedes_action_id = a.id \
              ) \
         WHERE s.archived_at IS NULL \
           AND ( \
               EXISTS ( \
                   SELECT 1 FROM participant p \
                   WHERE p.id = ?1 AND p.owner_space_id = s.id AND p.removed_at IS NULL \
               ) \
               OR EXISTS ( \
                   SELECT 1 FROM space_participant r \
                   JOIN participant p ON p.id = r.participant_id \
                   WHERE r.participant_id = ?1 AND r.space_id = s.id \
                     AND r.left_at IS NULL AND p.removed_at IS NULL \
               ) \
           ) \
         GROUP BY s.id, s.title, s.created_at, s.archived_at \
         ORDER BY last_activity_at DESC"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(participant_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(SpaceListRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            title: row.get::<Option<String>>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            archived_at: row.get::<Option<i64>>(3).map_err(AppError::db)?,
            last_activity_at: row.get::<i64>(4).map_err(AppError::db)?,
            message_count: row.get::<i64>(5).map_err(AppError::db)?,
            // **Deliberately never a parent here.** This read is model-facing
            // (`list_my_spaces`), and membership is its whole boundary: naming
            // the conversation a delegated room was opened from would report a
            // space the asking agent may well not be in, from a listing that
            // exists to report only the ones it is.
            parent: None,
        });
    }
    Ok(out)
}

// --- reference tables (references to globals + overrides) -------------------

/// Reference a global into a space (pinned `participant_scope='global'`, no
/// overrides). The common membership-add.
///
/// **Private** — see the note on [`ensure_space_participant`]: a `role`
/// arriving here is written verbatim, so an exported version was a way to mint
/// a second owner of a sub-space.
async fn insert_space_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    joined_at: i64,
) -> Result<(), AppError> {
    insert_participant_ref(
        conn,
        "space_participant",
        "space_id",
        space_id,
        participant_id,
        role,
        joined_at,
        &ParticipantRefRow {
            participant_id: participant_id.to_string(),
            role: role.to_string(),
            joined_at,
            override_label: None,
            override_model_ref: None,
            override_system_prompt: None,
            override_notify_policy: None,
        },
        false,
    )
    .await
}

/// Ensure a global is referenced into a space — **insert or revive**, on the
/// space's primary key. Used to guarantee "User" joins every instantiated space
/// even if a copied template reference already added it, and to let an agent
/// that left be invited back.
///
/// Answers whether the membership *changed*: a fresh join or a revived one.
/// A membership that was already live is left exactly as it stands, which is
/// what "idempotent" has always meant here.
///
/// **Private, and that is the enforcement.** The rules a membership write has
/// to respect do not live in this statement — they live in the transactions
/// that wrap it ([`join_space_participant_tx`],
/// [`grant_space_membership_tx`]), which refuse a second `role = 'owner'` in a
/// sub-space, and in [`remove_space_participant_tx`], which refuses to end the
/// two structural memberships. Duplicating those checks here would put the
/// same rule in two places to drift apart; withholding the primitive from
/// outside the module makes the guarded doors the only way in, so the invalid
/// state is unrepresentable at the API boundary rather than guarded at each
/// entrance. Every caller is in this file.
async fn ensure_space_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    joined_at: i64,
) -> Result<bool, AppError> {
    // **Liveness rides the write.** The caller reads the participant first for
    // its typed refusals, but a read cannot answer for a write that lands after
    // it: a retirement committing in between left a durable membership joined
    // *after* retirement, plus an invalidation and an `Internal` error where a
    // refusal belonged (Codex review, PR #279). `INSERT … SELECT … WHERE EXISTS`
    // makes the premise part of the statement, so a lost race writes nothing —
    // and it guards the revive too, since a `SELECT` that yields no row can
    // conflict with nothing.
    //
    // **A departure is not an absence.** Leaving is soft (`left_at`), so the row
    // survives on the PK and an insert-only join struck nothing: the caller's
    // roster read then found no member and reported the *live* agent retired —
    // a sentence about the wrong thing, and permanent, since every retry took
    // the same path (Codex review, PR #280). The upsert revives that row and
    // applies the role being asked for, so the offer the picker makes (an agent
    // that left is grantable again) is the one the write honours.
    //
    // The `WHERE left_at IS NOT NULL` on the update is what keeps this a
    // *revive* rather than an overwrite: a live membership is never rewritten,
    // so the template instantiation's "User" (inserted a few lines earlier with
    // the template's own role and overrides) survives the `ensure` that follows
    // it, exactly as `OR IGNORE` made it.
    //
    // The membership stamps the space (see `insert_participant_ref`), before
    // the write and whether or not it changes anything: an over-mark keeps a
    // space, and this door is reached by the invite as well as by birth.
    touch_space(conn, space_id, joined_at).await?;
    let n = conn
        .execute(
            "INSERT INTO space_participant \
             (space_id, participant_id, participant_scope, role, joined_at, \
              override_label, override_model_ref, override_system_prompt, \
              override_notify_policy) \
             SELECT ?1, ?2, 'global', ?3, ?4, NULL, NULL, NULL, NULL \
             WHERE EXISTS ( \
                 SELECT 1 FROM participant WHERE id = ?2 AND removed_at IS NULL \
             ) \
             ON CONFLICT (space_id, participant_id) DO UPDATE SET \
                 participant_scope = 'global', \
                 role = excluded.role, \
                 joined_at = excluded.joined_at, \
                 left_at = NULL, \
                 override_label = NULL, \
                 override_model_ref = NULL, \
                 override_system_prompt = NULL, \
                 override_notify_policy = NULL \
             WHERE space_participant.left_at IS NOT NULL",
            (
                Value::Text(space_id.to_string()),
                Value::Text(participant_id.to_string()),
                Value::Text(role.to_string()),
                Value::Integer(joined_at),
            ),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to insert space_participant: {e}"),
        })?;
    Ok(n > 0)
}

#[allow(clippy::too_many_arguments)]
async fn insert_participant_ref(
    conn: &Connection,
    table: &str,
    owner_col: &str,
    owner_id: &str,
    participant_id: &str,
    role: &str,
    joined_at: i64,
    overrides: &ParticipantRefRow,
    ignore: bool,
) -> Result<(), AppError> {
    // A space's membership is part of its configuration footprint, so a join
    // stamps it (a template's does not — templates are not reaped). Stamped
    // before the insert for the ordering reason `update_participant_config`
    // states; instantiation's own copies are un-stamped wholesale at the end of
    // its transaction, so birth is still pristine.
    if table == "space_participant" {
        touch_space(conn, owner_id, joined_at).await?;
    }
    let verb = if ignore { "INSERT OR IGNORE" } else { "INSERT" };
    let sql = format!(
        "{verb} INTO {table} \
         ({owner_col}, participant_id, participant_scope, role, joined_at, \
          override_label, override_model_ref, override_system_prompt, override_notify_policy) \
         VALUES (?1, ?2, 'global', ?3, ?4, ?5, ?6, ?7, ?8)"
    );
    conn.execute(
        &sql,
        (
            Value::Text(owner_id.to_string()),
            Value::Text(participant_id.to_string()),
            Value::Text(role.to_string()),
            Value::Integer(joined_at),
            opt_text(&overrides.override_label),
            opt_text(&overrides.override_model_ref),
            opt_text(&overrides.override_system_prompt),
            opt_text(&overrides.override_notify_policy),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert {table}: {e}"),
    })?;
    Ok(())
}

/// Update a space membership's overrides (each `Some` replaces; inner `Option`
/// clears/sets). Only meaningful for referenced globals ("override here").
///
/// Same rule as [`update_participant_config`], at the membership level: the
/// write lands only on a **live membership of a live participant** — the
/// definition [`is_space_member`] reads, now asked of a write. A stale
/// "override here" arriving after the member left (or after the agent was
/// retired) is refused rather than parked in columns nothing reads, which is
/// what keeps "your write was refused" from meaning "your write is waiting to
/// reappear".
#[allow(clippy::too_many_arguments)]
pub async fn update_space_participant_override(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    override_label: Option<Option<&str>>,
    override_model_ref: Option<Option<&str>>,
    override_system_prompt: Option<Option<&str>>,
    override_notify_policy: Option<Option<&str>>,
    now: i64,
) -> Result<bool, AppError> {
    let mut sets: Vec<String> = Vec::new();
    let mut params: Vec<Value> = vec![
        Value::Text(space_id.to_string()),
        Value::Text(participant_id.to_string()),
    ];
    let mut policy_param: Option<usize> = None;
    for (col, val) in [
        ("override_label", override_label),
        ("override_model_ref", override_model_ref),
        ("override_system_prompt", override_system_prompt),
        ("override_notify_policy", override_notify_policy),
    ] {
        if let Some(inner) = val {
            params.push(opt_str(inner));
            sets.push(format!("{col} = ?{}", params.len()));
            if col == "override_notify_policy" {
                policy_param = Some(params.len());
            }
        }
    }
    if sets.is_empty() {
        return Ok(false);
    }
    // **A sub-space owner's notify policy may not become agent-triggering, and
    // may not go back to being inherited.** The room's helpers are `all` so
    // that anything they say wakes each other; the owner must never be, or its
    // answer wakes them all again and one spawn's work grows with the square of
    // its roster ([`spawn_subspace_tx`] states the whole argument).
    //
    // Refusing `all` alone would not close it: an inherited (NULL) override
    // resolves to the agent's **global** row, which another door can flip to
    // `all` afterwards — so the hole would reopen through a write that never
    // mentioned this space. Keeping the override *always present and never
    // `all`* makes the global policy irrelevant here by construction
    // (`COALESCE` takes the override every time), which is why this refuses the
    // clear as well as the value, and why no second guard is needed on the
    // config door. What a reader keeps is the choice that matters — `human` or
    // `explicit`; what they lose is "inherit", which is exactly the state that
    // was unsafe.
    //
    // In the statement's own `WHERE`, like the two structural memberships'
    // refusals: a spawn commits the room and this membership together, so a
    // guard that read first could be looking at a space that was ordinary a
    // moment ago.
    let owner_guard = match policy_param {
        Some(k) => format!(
            " AND NOT ( \
                 (?{k} IS NULL OR ?{k} = 'all') \
                 AND role = 'owner' \
                 AND EXISTS ( \
                     SELECT 1 FROM space s \
                     WHERE s.id = ?1 AND s.parent_space_id IS NOT NULL \
                 ) \
             )"
        ),
        None => String::new(),
    };
    let sql = format!(
        "UPDATE space_participant SET {} \
         WHERE space_id = ?1 AND participant_id = ?2 AND left_at IS NULL \
           AND EXISTS ( \
               SELECT 1 FROM participant p WHERE p.id = ?2 AND p.removed_at IS NULL \
           ){owner_guard}",
        sets.join(", ")
    );
    // An override is per-space configuration, so it stamps the space — before
    // the write, and whether or not the write lands (see
    // `update_participant_config` for why that order is the safe one).
    touch_space(conn, space_id, now).await?;
    let n = conn.execute(&sql, params).await.map_err(AppError::db)?;
    Ok(n > 0)
}

/// Reference a global into a template, carrying overrides. Used by the
/// space→template projection.
pub async fn insert_template_participant(
    conn: &Connection,
    template_id: &str,
    r: &ParticipantRefRow,
) -> Result<(), AppError> {
    insert_participant_ref(
        conn,
        "space_template_participant",
        "template_id",
        template_id,
        &r.participant_id,
        &r.role,
        r.joined_at,
        r,
        false,
    )
    .await
}

/// What [`remove_space_participant_tx`] did — decided by the write, not by the
/// caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpaceRemoval {
    /// The participant was still this space's own, so its row was deactivated.
    SoftRemoved,
    /// It was a referenced global (shared, or shared *just now*), so it left the
    /// space and the row stands.
    Left,
    /// Nothing live to end — already removed, already left, or never a member.
    NothingToDo,
    /// The participant owns this space as its **notebook**, and that membership
    /// is structural: the notebook exists only for it, and is the residence of
    /// the `core` memory it writes. Refused rather than obeyed, because nothing
    /// can grant a notebook membership back.
    RefusedNotebookOwner,
    /// The participant **owns this sub-space**, and that membership is
    /// structural for the same kind of reason a notebook owner's is: it is the
    /// whole of what records who is answerable for the delegation, whose
    /// live-room quota it counts against, and who the report goes to. Nothing
    /// can grant a sub-space ownership back, so it is refused rather than
    /// obeyed; archiving the sub-space is the remedy.
    RefusedSubspaceOwner,
}

/// Take a participant out of one space, **deciding at the write** whether that
/// means deactivating this space's own row or ending a membership.
///
/// The two are not interchangeable: a soft-remove retires the participant
/// everywhere (it is what the library's Retire does), while a leave ends one
/// membership. Choosing between them from a `get_participant` read taken a
/// moment earlier is a read-then-write window, and promotion is exactly what
/// walks through it: another window flips the row to `global` in between, and
/// the removal — still believing it holds a space-owned agent — sets
/// `removed_at` on a freshly *shared* agent, retiring it from the library the
/// instant it arrived and orphaning the notebook promotion had just created
/// (retirement archives it; this write knows nothing about it). Both callers
/// then report success (Codex review, PR #279).
///
/// So the ownership test rides **in the soft-remove's own `WHERE`**: it strikes
/// only a row that is still `scope = 'space'`, still owned by *this* space, and
/// still live. Zero rows means the premise no longer holds, and the fallback is
/// the honest one — end the membership, which after a promotion is precisely the
/// reference row promotion left behind. The mirror guard is on the other side
/// (`promote_participant_tx` requires `scope = 'space' AND removed_at IS NULL`),
/// so whichever transaction commits first, the loser refuses cleanly rather than
/// writing through a premise that has expired.
/// What a departure did: which of the two ways it ended, and **every
/// delegation it closed on the way out**.
///
/// The second half is the same rule archival and retirement already apply, said
/// of the third door that can end an agent's part in a conversation: a room an
/// agent opened *for* this conversation exists to serve it, and an agent that
/// is no longer in the conversation can no longer report to it — every report
/// it tried would act as a participant that has left and be refused, forever,
/// against a meter nobody reads and a live-room slot nobody gets back.
pub struct SpaceDeparture {
    pub outcome: SpaceRemoval,
    /// The rooms closed with the departure, so the caller can announce each one
    /// (which is what releases a delegation waiting on it as its parent).
    pub archived_spaces: Vec<String>,
}

pub(crate) async fn remove_space_participant_tx(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    now: i64,
) -> Result<SpaceDeparture, AppError> {
    begin_write(conn).await?;
    match remove_space_participant_tx_body(conn, space_id, participant_id, now).await {
        Ok(outcome) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(outcome)
        }
        Err(e) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn remove_space_participant_tx_body(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    now: i64,
) -> Result<SpaceDeparture, AppError> {
    // Either branch changes the space's roster, and the transaction makes the
    // stamp atomic with whichever one lands; a `NothingToDo` over-marks, which
    // keeps a space rather than losing one.
    touch_space(conn, space_id, now).await?;
    let struck = conn
        .execute(
            "UPDATE participant SET removed_at = ?3 \
             WHERE id = ?1 AND removed_at IS NULL \
               AND scope = 'space' AND owner_space_id = ?2",
            (
                Value::Text(participant_id.to_string()),
                Value::Text(space_id.to_string()),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?;
    if struck > 0 {
        return Ok(SpaceDeparture {
            outcome: SpaceRemoval::SoftRemoved,
            archived_spaces: close_delegations_run_for(conn, space_id, participant_id, now).await?,
        });
    }
    // The two **structural** memberships are guarded **in the leave's own
    // `WHERE`**, for the reason the soft-remove is guarded above: a promotion
    // mints a notebook and its membership in one transaction, and a spawn mints
    // a sub-space and its owner's in one transaction, so a removal that read the
    // space a moment earlier saw an ordinary conversation. The reads below only
    // *explain* a write that struck nothing — they never decide one.
    let left = conn
        .execute(
            "UPDATE space_participant SET left_at = ?3 \
             WHERE space_id = ?1 AND participant_id = ?2 AND left_at IS NULL \
               AND NOT EXISTS ( \
                   SELECT 1 FROM space s \
                   WHERE s.id = ?1 AND s.notebook_participant_id = ?2 \
               ) \
               AND NOT ( \
                   role = 'owner' AND EXISTS ( \
                       SELECT 1 FROM space s \
                       WHERE s.id = ?1 AND s.parent_space_id IS NOT NULL \
                   ) \
               )",
            (
                Value::Text(space_id.to_string()),
                Value::Text(participant_id.to_string()),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?;
    if left > 0 {
        return Ok(SpaceDeparture {
            outcome: SpaceRemoval::Left,
            archived_spaces: close_delegations_run_for(conn, space_id, participant_id, now).await?,
        });
    }
    let refused = |outcome| SpaceDeparture {
        outcome,
        archived_spaces: Vec::new(),
    };
    if notebook_participant_of(conn, space_id).await?.as_deref() == Some(participant_id) {
        return Ok(refused(SpaceRemoval::RefusedNotebookOwner));
    }
    if subspace_owner_of(conn, space_id).await?.as_deref() == Some(participant_id) {
        return Ok(refused(SpaceRemoval::RefusedSubspaceOwner));
    }
    Ok(refused(SpaceRemoval::NothingToDo))
}

/// Archive every live delegation `participant_id` was running **for
/// `space_id`** — the rooms it owns whose parent is the conversation it has
/// just left — and everything beneath them.
///
/// **Two rules about one membership, and they are not in tension.** An owner
/// cannot leave the *room* it owns (`SpaceRemoval::RefusedSubspaceOwner`,
/// guarded in the leave's own `WHERE`): that membership is the whole record of
/// who is answerable for the delegation, and nothing can grant it back. An
/// owner leaving the *parent* is an ordinary departure and is allowed — and it
/// takes the delegations it opened there with it, because their purpose was to
/// serve that conversation and their reports are turns it can no longer take:
/// every one would act as a participant that has left and be refused. The first
/// rule protects a room from losing its owner; the second stops a room outliving
/// the reason it existed. Together they say one thing — a delegation and the
/// conversation it was opened from stand or fall together.
///
/// Scoped by `parent_space_id`, so an agent's rooms under *other* conversations
/// are untouched: leaving one conversation says nothing about another.
async fn close_delegations_run_for(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    now: i64,
) -> Result<Vec<String>, AppError> {
    const OWNED_UNDER_THIS_PARENT: &str = "archived_at IS NULL AND parent_space_id = ?1          AND EXISTS (              SELECT 1 FROM space_participant r              WHERE r.space_id = space.id AND r.participant_id = ?2                AND r.role = 'owner' AND r.left_at IS NULL          )";
    let mut stmt = conn
        .prepare(&format!(
            "SELECT id FROM space WHERE {OWNED_UNDER_THIS_PARENT}"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(space_id.to_string()),
            Value::Text(participant_id.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    let mut archived = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        archived.push(row.get::<String>(0).map_err(AppError::db)?);
    }
    if archived.is_empty() {
        return Ok(archived);
    }
    conn.execute(
        &format!(
            "UPDATE space SET archived_at = ?3, touched_at = COALESCE(touched_at, ?3) \
             WHERE {OWNED_UNDER_THIS_PARENT}"
        ),
        (
            Value::Text(space_id.to_string()),
            Value::Text(participant_id.to_string()),
            Value::Integer(now),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to close the delegations of a departing owner: {e}"),
    })?;
    // And whatever those rooms had delegated onward, at any depth and whoever
    // owns it — the invariant every archival restores.
    archived.extend(archive_rooms_under_a_closed_one(conn, now).await?);
    Ok(archived)
}

/// The participant a space is the **notebook** of, if it is one — the reverse of
/// [`notebook_space_for`]. Read by the removal guard above, and surfaced to the
/// GUI (via `SpaceSettings`) so the roster can withhold a Remove that could only
/// be refused.
pub async fn notebook_participant_of(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare("SELECT notebook_participant_id FROM space WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(space_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(row.get::<Option<String>>(0).map_err(AppError::db)?),
    }
}

/// End a space membership (reference) — set `left_at`. For a referenced global;
/// owned participants are removed via [`soft_remove_participant`].
///
/// **Compiled only into test builds**, which is the strongest form the
/// no-production-caller rule takes in this module: `insert_space` is held to it
/// by an assertion (`the_raw_space_insert_has_no_production_caller`), and this
/// one is held to it by the compiler.
///
/// The real leave is the statement inside [`remove_space_participant_tx`],
/// which carries in its own `WHERE` the two refusals this one has never had: a
/// notebook's owner and a sub-space's owner cannot leave, because nothing can
/// grant either membership back. This is the same write *without* them — so it
/// survives only as an arrangement primitive for this module's own tests,
/// which need a departed membership to revive, and cannot become a second,
/// unguarded way to do a guarded thing.
#[cfg(test)]
async fn leave_space_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    now: i64,
) -> Result<bool, AppError> {
    touch_space(conn, space_id, now).await?;
    let n = conn
        .execute(
            "UPDATE space_participant SET left_at = ?3 \
             WHERE space_id = ?1 AND participant_id = ?2 AND left_at IS NULL",
            (
                Value::Text(space_id.to_string()),
                Value::Text(participant_id.to_string()),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

/// The live reference rows (references to globals) of a space, for copying.
pub async fn list_space_participant_refs(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<ParticipantRefRow>, AppError> {
    list_participant_refs(conn, "space_participant", "space_id", space_id).await
}

/// The live reference rows of a template, for copying.
pub async fn list_template_participant_refs(
    conn: &Connection,
    template_id: &str,
) -> Result<Vec<ParticipantRefRow>, AppError> {
    list_participant_refs(
        conn,
        "space_template_participant",
        "template_id",
        template_id,
    )
    .await
}

async fn list_participant_refs(
    conn: &Connection,
    table: &str,
    owner_col: &str,
    owner_id: &str,
) -> Result<Vec<ParticipantRefRow>, AppError> {
    let sql = format!(
        "SELECT participant_id, role, joined_at, override_label, override_model_ref, \
                override_system_prompt, override_notify_policy \
         FROM {table} WHERE {owner_col} = ?1 AND left_at IS NULL \
         ORDER BY joined_at, participant_id"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(owner_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(ParticipantRefRow::from_row(&row)?);
    }
    Ok(out)
}

// --- unified effective membership view -------------------------------------

/// The effective participants of a space: owned rows ∪ referenced globals, with
/// referenced config resolved via `COALESCE(override, global config)`. Human
/// members (the referenced "User") sort first, then others by id.
pub async fn space_participants(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<EffectiveParticipantRow>, AppError> {
    effective_participants(
        conn,
        "space_participant",
        "space_id",
        "owner_space_id",
        space_id,
    )
    .await
}

/// The effective participants of a template (same shape as [`space_participants`]).
pub async fn template_participants(
    conn: &Connection,
    template_id: &str,
) -> Result<Vec<EffectiveParticipantRow>, AppError> {
    effective_participants(
        conn,
        "space_template_participant",
        "template_id",
        "owner_template_id",
        template_id,
    )
    .await
}

async fn effective_participants(
    conn: &Connection,
    ref_table: &str,
    owner_col: &str,
    participant_owner_col: &str,
    owner_id: &str,
) -> Result<Vec<EffectiveParticipantRow>, AppError> {
    // Two arms: owned rows (own config), then referenced globals
    // (COALESCE(override, config)). Wrapped so ORDER BY can reference the
    // unioned output columns cleanly. `?1`/`?2` are both the owner id.
    let sql = format!(
        "SELECT * FROM ( \
            SELECT p.id AS participant_id, p.scope AS scope, 'owned' AS source, \
                   p.kind AS kind, p.label AS label, p.model_ref AS model_ref, \
                   p.system_prompt AS system_prompt, p.notify_policy AS notify_policy, \
                   p.role AS role \
            FROM participant p \
            WHERE p.{participant_owner_col} = ?1 AND p.removed_at IS NULL \
            UNION ALL \
            SELECT p.id, p.scope, 'referenced', p.kind, \
                   COALESCE(r.override_label, p.label), \
                   COALESCE(r.override_model_ref, p.model_ref), \
                   COALESCE(r.override_system_prompt, p.system_prompt), \
                   COALESCE(r.override_notify_policy, p.notify_policy), \
                   r.role \
            FROM {ref_table} r \
            JOIN participant p ON p.id = r.participant_id AND p.scope = r.participant_scope \
            WHERE r.{owner_col} = ?2 AND r.left_at IS NULL AND p.removed_at IS NULL \
        ) ORDER BY CASE kind WHEN 'human' THEN 0 ELSE 1 END, participant_id"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(owner_id.to_string()),
            Value::Text(owner_id.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(EffectiveParticipantRow {
            participant_id: row.get::<String>(0).map_err(AppError::db)?,
            scope: row.get::<String>(1).map_err(AppError::db)?,
            source: row.get::<String>(2).map_err(AppError::db)?,
            kind: row.get::<String>(3).map_err(AppError::db)?,
            label: row.get::<String>(4).map_err(AppError::db)?,
            model_ref: row.get::<Option<String>>(5).map_err(AppError::db)?,
            system_prompt: row.get::<Option<String>>(6).map_err(AppError::db)?,
            notify_policy: row.get::<String>(7).map_err(AppError::db)?,
            role: row.get::<String>(8).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// Resolve the space's agent participant whose **effective** model_ref matches,
/// returning `(participant_id, scope)` for the composite echo an action needs.
/// The wave-1 seam so an inference records a real per-space (or referenced
/// global) agent while `run_turn` still takes a model string.
pub async fn space_agent_participant_by_model(
    conn: &Connection,
    space_id: &str,
    model_ref: &str,
) -> Result<Option<(String, String)>, AppError> {
    let members = space_participants(conn, space_id).await?;
    Ok(members.into_iter().find_map(|m| {
        if m.kind == "agent" && m.model_ref.as_deref() == Some(model_ref) {
            Some((m.participant_id, m.scope))
        } else {
            None
        }
    }))
}

/// A space's `cascade_limit` (the copied-from-template setting).
pub async fn space_cascade_limit(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<i64>, AppError> {
    let mut stmt = conn
        .prepare("SELECT cascade_limit FROM space WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(space_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<i64>(0).map_err(AppError::db)?)),
    }
}

/// Set a space's `cascade_limit`. Returns whether a row was updated (`false` =
/// no such space).
pub async fn set_space_cascade_limit(
    conn: &Connection,
    space_id: &str,
    cascade_limit: i64,
    now: i64,
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE space SET cascade_limit = ?2, \
             touched_at = COALESCE(touched_at, ?3) WHERE id = ?1",
            (
                Value::Text(space_id.to_string()),
                Value::Integer(cascade_limit),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

/// A space's `router_model` (the may-decline router setting, copied from the
/// template it was instantiated from). `Ok(None)` covers both "no such space"
/// and "the feature is off here" — the caller validates space existence
/// separately, and both answers mean *don't call a router*.
pub async fn space_router_model(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare("SELECT router_model FROM space WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(space_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(row.get::<Option<String>>(0).map_err(AppError::db)?),
    }
}

/// Set (or clear, with `None`) a space's `router_model`. Returns whether a row
/// was updated.
pub async fn set_space_router_model(
    conn: &Connection,
    space_id: &str,
    router_model: Option<&str>,
    now: i64,
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE space SET router_model = ?2, \
             touched_at = COALESCE(touched_at, ?3) WHERE id = ?1",
            (
                Value::Text(space_id.to_string()),
                opt_str(router_model),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

// --- space templates -------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SpaceTemplateRow {
    pub id: String,
    pub title: String,
    pub cascade_limit: i64,
    /// The may-decline router model copied into a space at instantiation.
    /// `None` = the feature is off for spaces born from this template.
    pub router_model: Option<String>,
    pub created_at: i64,
    pub removed_at: Option<i64>,
}

fn space_template_row_from(row: &turso::Row) -> Result<SpaceTemplateRow, AppError> {
    Ok(SpaceTemplateRow {
        id: row.get::<String>(0).map_err(AppError::db)?,
        title: row.get::<String>(1).map_err(AppError::db)?,
        cascade_limit: row.get::<i64>(2).map_err(AppError::db)?,
        router_model: row.get::<Option<String>>(3).map_err(AppError::db)?,
        created_at: row.get::<i64>(4).map_err(AppError::db)?,
        removed_at: row.get::<Option<i64>>(5).map_err(AppError::db)?,
    })
}

/// List live (non-removed) templates; the seeded default's id sorts first.
pub async fn list_space_templates(conn: &Connection) -> Result<Vec<SpaceTemplateRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, title, cascade_limit, router_model, created_at, removed_at \
             FROM space_template WHERE removed_at IS NULL \
             ORDER BY created_at, id",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(space_template_row_from(&row)?);
    }
    Ok(out)
}

/// Fetch one template by id, including soft-removed rows.
pub async fn get_space_template(
    conn: &Connection,
    id: &str,
) -> Result<Option<SpaceTemplateRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, title, cascade_limit, router_model, created_at, removed_at \
             FROM space_template WHERE id = ?1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((Value::Text(id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(space_template_row_from(&row)?)),
    }
}

pub async fn insert_space_template(
    conn: &Connection,
    id: &str,
    title: &str,
    cascade_limit: i64,
    router_model: Option<&str>,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO space_template (id, title, cascade_limit, router_model, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        (
            Value::Text(id.to_string()),
            Value::Text(title.to_string()),
            Value::Integer(cascade_limit),
            opt_str(router_model),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert space_template: {e}"),
    })?;
    Ok(())
}

/// Update a template's own settings (title / cascade_limit). Returns whether a
/// live row was updated.
pub async fn update_space_template(
    conn: &Connection,
    id: &str,
    title: Option<&str>,
    cascade_limit: Option<i64>,
) -> Result<bool, AppError> {
    let mut sets: Vec<String> = Vec::new();
    let mut params: Vec<Value> = vec![Value::Text(id.to_string())];
    if let Some(t) = title {
        params.push(Value::Text(t.to_string()));
        sets.push(format!("title = ?{}", params.len()));
    }
    if let Some(c) = cascade_limit {
        params.push(Value::Integer(c));
        sets.push(format!("cascade_limit = ?{}", params.len()));
    }
    if sets.is_empty() {
        return Ok(false);
    }
    let sql = format!(
        "UPDATE space_template SET {} WHERE id = ?1 AND removed_at IS NULL",
        sets.join(", ")
    );
    let n = conn.execute(&sql, params).await.map_err(AppError::db)?;
    Ok(n > 0)
}

/// Set (or clear, with `None`) a template's `router_model`. Deliberately its
/// own setter rather than another optional parameter on
/// [`update_space_template`]: `Option<&str>` there would be ambiguous between
/// "leave alone" and "clear", and the router setting has exactly two states.
/// Returns whether a live row was updated.
pub async fn set_template_router_model(
    conn: &Connection,
    id: &str,
    router_model: Option<&str>,
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE space_template SET router_model = ?2 \
             WHERE id = ?1 AND removed_at IS NULL",
            (Value::Text(id.to_string()), opt_str(router_model)),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

/// A validated template-participant tuple for the transactional update:
/// `(label, model_ref, system_prompt, notify_policy)` (agents only).
pub type TemplateParticipantInput = (String, Option<String>, Option<String>, String);

/// Update a template's settings **and/or** replace its OWNED participant set,
/// **atomically** in one transaction. turso autocommits each statement, so a
/// plain "DELETE owned; then re-INSERT" (the previous shape) exposed a window:
/// a concurrent `instantiate_template` could observe the template with zero (or
/// partially rebuilt) agents, and an insert error mid-loop left the set
/// destroyed. Wrapping settings-update + delete + re-insert in `BEGIN … COMMIT`
/// closes both — the replacement is all-or-nothing and a reader on another
/// connection never sees the in-between. On any error the transaction is rolled
/// back (prior state intact) and the error propagates; the caller emits
/// `Change::Templates` only after this returns `Ok` (the changes.rs
/// emit-after-commit rule). `participants = None` leaves the participant set
/// untouched; `Some(&[])` clears it.
pub(crate) async fn update_template_tx(
    conn: &Connection,
    id: &str,
    title: Option<&str>,
    cascade_limit: Option<i64>,
    participants: Option<&[TemplateParticipantInput]>,
    now: i64,
) -> Result<(), AppError> {
    begin_write(conn).await?;
    match update_template_tx_body(conn, id, title, cascade_limit, participants, now).await {
        Ok(()) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(())
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn update_template_tx_body(
    conn: &Connection,
    id: &str,
    title: Option<&str>,
    cascade_limit: Option<i64>,
    participants: Option<&[TemplateParticipantInput]>,
    now: i64,
) -> Result<(), AppError> {
    if title.is_some() || cascade_limit.is_some() {
        update_space_template(conn, id, title, cascade_limit).await?;
    }
    if let Some(participants) = participants {
        delete_template_owned_participants(conn, id).await?;
        for (label, model_ref, system_prompt, notify_policy) in participants {
            insert_participant(
                conn,
                &uuid::Uuid::now_v7().to_string(),
                "template",
                None,
                Some(id),
                "agent",
                label,
                model_ref.as_deref(),
                system_prompt.as_deref(),
                notify_policy,
                "member",
                None,
                now,
            )
            .await?;
        }
    }
    Ok(())
}

/// Soft-remove a template. Returns whether a live row was removed.
pub async fn soft_remove_space_template(
    conn: &Connection,
    id: &str,
    now: i64,
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE space_template SET removed_at = ?2 \
             WHERE id = ?1 AND removed_at IS NULL",
            (Value::Text(id.to_string()), Value::Integer(now)),
        )
        .await
        .map_err(AppError::db)?;
    Ok(n > 0)
}

// --- projections (INSERT … SELECT-style row copies) ------------------------

/// Instantiate a template into a **new space**: create the space (copying
/// `cascade_limit` and `router_model`); copy the template's OWNED participants into fresh
/// SPACE-owned rows; copy its reference rows (with overrides) into space
/// references; then ensure the shared human "User" is referenced (as owner).
/// Errors if the template is missing or removed.
///
/// **One transaction, because a space without its participants is not a
/// space.** These are several statements, and turso autocommits each one — so
/// a failure partway (a full disk, a constraint the copied rows violate) used
/// to leave a durable `space` row with some or none of its membership: a
/// conversation nobody could be told about, which every caller's error handling
/// reasonably reads as "nothing was written". `BEGIN IMMEDIATE` makes the
/// refusal zero-state and the success atomic, for every caller of this door —
/// the default-template path `create_space` and `post` take, and the explicit
/// one `create_space_from_template` takes. **The template's liveness is read
/// inside it too**, decided at the write like every other guard in this module,
/// so a template removed a moment ago cannot be instantiated by a check taken
/// before the writer was reserved.
pub(crate) async fn instantiate_template(
    conn: &Connection,
    template_id: &str,
    new_space_id: &str,
    title: Option<&str>,
    linkability: &str,
    now: i64,
) -> Result<(), AppError> {
    begin_write(conn).await?;
    match instantiate_template_body(conn, template_id, new_space_id, title, linkability, now).await
    {
        Ok(()) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(())
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn instantiate_template_body(
    conn: &Connection,
    template_id: &str,
    new_space_id: &str,
    title: Option<&str>,
    linkability: &str,
    now: i64,
) -> Result<(), AppError> {
    let template = get_space_template(conn, template_id)
        .await?
        .ok_or_else(|| AppError::NotConfigured {
            message: format!("space template not found: {template_id}"),
        })?;
    if template.removed_at.is_some() {
        return Err(AppError::NotConfigured {
            message: format!("space template `{template_id}` was removed"),
        });
    }

    conn.execute(
        "INSERT INTO space \
         (id, parent_space_id, title, linkability, cascade_limit, router_model, created_at) \
         VALUES (?1, NULL, ?2, ?3, ?4, ?5, ?6)",
        (
            Value::Text(new_space_id.to_string()),
            opt_str(title),
            Value::Text(linkability.to_string()),
            Value::Integer(template.cascade_limit),
            opt_str(template.router_model.as_deref()),
            Value::Integer(now),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert space: {e}"),
    })?;

    // Template-owned participants → fresh space-owned rows.
    for p in list_template_owned_participants(conn, template_id).await? {
        insert_participant(
            conn,
            &uuid::Uuid::now_v7().to_string(),
            "space",
            Some(new_space_id),
            None,
            &p.kind,
            &p.label,
            p.model_ref.as_deref(),
            p.system_prompt.as_deref(),
            &p.notify_policy,
            &p.role,
            None,
            now,
        )
        .await?;
    }

    // Template reference rows → space reference rows (overrides preserved).
    for r in list_template_participant_refs(conn, template_id).await? {
        insert_participant_ref(
            conn,
            "space_participant",
            "space_id",
            new_space_id,
            &r.participant_id,
            &r.role,
            now,
            &r,
            true, // OR IGNORE — a template that already references "User" won't
                  // collide with the ensure below.
        )
        .await?;
    }

    // The shared human "User" joins every instantiated space (idempotent).
    ensure_space_participant(conn, new_space_id, HUMAN_PARTICIPANT_ID, "owner", now).await?;

    // **Birth is not a change.** Every copy above went through a primitive that
    // stamps the space it writes into, which is what makes the enumeration live
    // where the writes live — so instantiation takes the stamp back here, last,
    // inside the same transaction, and a fresh space is pristine by definition.
    //
    // The one thing an instantiation carries that is *not* birth is a
    // caller-supplied `title`: a human saying what this conversation is for.
    // So a titled creation is born stamped, and an untitled one — the ⌘N door,
    // and "New Space from Template", both of which pass `None` — is not.
    conn.execute(
        "UPDATE space SET touched_at = ?2 WHERE id = ?1",
        (
            Value::Text(new_space_id.to_string()),
            match title {
                Some(_) => Value::Integer(now),
                None => Value::Null,
            },
        ),
    )
    .await
    .map_err(AppError::db)?;
    Ok(())
}

/// Project a space's current participants + settings into a **new template**:
/// copy `cascade_limit` and `router_model`; copy the space's OWNED participants
/// into TEMPLATE-owned rows; copy its reference rows (with overrides) into
/// template references.
pub async fn template_from_space(
    conn: &Connection,
    space_id: &str,
    title: &str,
    new_template_id: &str,
    now: i64,
) -> Result<(), AppError> {
    let cascade_limit =
        space_cascade_limit(conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
    let router_model = space_router_model(conn, space_id).await?;
    insert_space_template(
        conn,
        new_template_id,
        title,
        cascade_limit,
        router_model.as_deref(),
        now,
    )
    .await?;

    // Space-owned participants → fresh template-owned rows.
    for p in list_space_owned_participants(conn, space_id).await? {
        insert_participant(
            conn,
            &uuid::Uuid::now_v7().to_string(),
            "template",
            None,
            Some(new_template_id),
            &p.kind,
            &p.label,
            p.model_ref.as_deref(),
            p.system_prompt.as_deref(),
            &p.notify_policy,
            &p.role,
            None,
            now,
        )
        .await?;
    }

    // Space reference rows → template reference rows (overrides preserved).
    for r in list_space_participant_refs(conn, space_id).await? {
        insert_template_participant(conn, new_template_id, &r).await?;
    }
    Ok(())
}
// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Action operations
// ---------------------------------------------------------------------------

pub struct ActionEntry {
    pub id: String,
    pub space_id: String,
    /// The acting participant. There is deliberately **no** scope field: the
    /// pinned `(participant_id, participant_scope)` echo is derived from this
    /// id inside [`insert_action`]'s own statement, so a caller cannot supply
    /// a scope that has gone stale (see that function).
    pub participant_id: String,
    /// Stable identity shared by every generation of this item. For original
    /// (gen-0) work, mint a fresh UUIDv7; an edit/regeneration reuses the
    /// item_id of the action it supersedes.
    pub item_id: String,
    /// Prior generation this one replaces; `None` for gen 0. (The generation
    /// *number* is derived from this chain, not stored.)
    pub supersedes_action_id: Option<String>,
    pub action_type: String,
    pub status: String,
    pub intent: Option<String>,
    pub model: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub credits_consumed: Option<i64>,
    pub created_at: i64,
}

/// Insert one action.
///
/// **The pinned `participant_scope` echo is derived inside the statement**
/// (`SELECT scope FROM participant WHERE id = ?`), never supplied by the
/// caller. The echo is a *constraint device* that must equal the participant's
/// scope at the instant of the write, and task 36 made that scope mutable: a
/// promotion flips it mid-flight, so any scope a caller captured earlier — in
/// a `TurnPrep` built before an HTTP round trip, in a turn-scoped tool — is a
/// value that can go stale between capture and write. Reading it here, in the
/// same single statement as the insert, makes staleness **unrepresentable**
/// rather than merely unlikely: turso is single-writer, so a promotion cannot
/// interleave *within* the statement, and there is no read-then-write window
/// to lose a race in. A re-resolve before the call would only narrow that
/// window; a lock would have to be held across an HTTP await.
///
/// `participant_scope` is `NOT NULL`, so an unknown participant fails loudly
/// here instead of inserting a NULL echo (which MATCH SIMPLE would then skip).
/// Everything one saved post is made of — see [`post_tx`].
pub(crate) struct PostPlan<'a> {
    pub space_id: &'a str,
    pub participant_id: &'a str,
    pub action_id: &'a str,
    pub item_id: &'a str,
    pub text: &'a str,
    /// The title to derive onto the space, when this post is the one that earns
    /// the space a name. `None` leaves the title alone.
    pub auto_title: Option<&'a str>,
    /// The structural `reply` antecedent at ordinal 0: the explicit branch
    /// target, or the space's tail, or nothing for a space's first post.
    pub reply_to: Option<&'a str>,
    /// Quoted references at ordinals `1..=N` in supplied order — already
    /// validated by the caller, which is where a refusal belongs.
    pub references: &'a [crate::ReferenceSpec],
    /// Join the author into this space as part of writing the post.
    ///
    /// Set for a **delegated room**, where the shared human has no membership
    /// by construction: the first thing they say there joins them. The join
    /// rides the post's own transaction because the roster the models are shown
    /// has to be true of the transcript they read — a post by somebody the
    /// roster omits is a lie about the room, and any gap between the two
    /// writes is a window in which a turn can be planned over exactly that.
    pub join_author: bool,
    pub created_at: i64,
}

/// What [`post_tx`] committed, beyond the post itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PostOutcome {
    /// The space took a derived title from this post.
    pub auto_titled: bool,
    /// The author's membership was written (a fresh join or a revived one) —
    /// what the caller announces on.
    pub joined: bool,
}

/// Write one post — **the action, its text, its reply edge, its quoted
/// references, any title it earns and, where the plan asks for it, the author's
/// own membership, in one `BEGIN IMMEDIATE` transaction.**
///
/// **A post is not its action row.** Written as separate autocommitted
/// statements, the row lands first and its text, its place in the thread and
/// its quotations follow — so for as long as those take, the post exists and
/// says nothing. Every reader keyed on the action is entitled to it in that
/// window, and one of them acts: the sub-space driver's refill takes posts by
/// commit order, so it can plan and *bill* a turn against a post with no words
/// in it, and by the time the writer's own change event arms the room again the
/// driven reply may already have settled it. Rendering has the same exposure
/// with a gentler ending (a blank row in an open window).
///
/// The transaction is what makes "a post" a single event, and it is the same
/// cure `instantiate_template` took for the same class of defect: a space
/// without its participants is not a space, and a post without its words is not
/// a post. Nothing here spends or calls out — the writes are local and
/// adjacent, which is what makes them transaction-able at all — and the caller
/// validates before it opens this, so a refusal still writes nothing and now
/// cannot half-write either.
pub(crate) async fn post_tx(
    conn: &Connection,
    plan: &PostPlan<'_>,
) -> Result<PostOutcome, AppError> {
    begin_write(conn).await?;
    match post_tx_body(conn, plan).await {
        Ok(outcome) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(outcome)
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn post_tx_body(conn: &Connection, plan: &PostPlan<'_>) -> Result<PostOutcome, AppError> {
    // The membership lands **before** the words, so no ordering inside the
    // transaction can show a post whose author the roster does not carry.
    // Insert-or-revive on the space's primary key: a live membership is left
    // exactly as it stands, so this is a no-op for every ordinary space and
    // for the second thing a reader says in a delegated one.
    let joined = if plan.join_author {
        ensure_space_participant(
            conn,
            plan.space_id,
            plan.participant_id,
            crate::MembershipRole::Member.as_str(),
            plan.created_at,
        )
        .await?
    } else {
        false
    };
    insert_action(
        conn,
        &ActionEntry {
            id: plan.action_id.to_string(),
            space_id: plan.space_id.to_string(),
            participant_id: plan.participant_id.to_string(),
            item_id: plan.item_id.to_string(),
            supersedes_action_id: None,
            action_type: "user_input".to_string(),
            status: "complete".to_string(),
            intent: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            credits_consumed: None,
            created_at: plan.created_at,
        },
    )
    .await?;
    insert_text_content_block(
        conn,
        &uuid::Uuid::now_v7().to_string(),
        plan.action_id,
        0,
        "text",
        plan.text,
    )
    .await?;
    let auto_titled = match plan.auto_title {
        Some(title) => update_space_title(conn, plan.space_id, title, plan.created_at).await?,
        None => false,
    };
    if let Some(ante) = plan.reply_to {
        insert_action_antecedent(conn, plan.action_id, ante, 0, "reply").await?;
    }
    for (i, spec) in plan.references.iter().enumerate() {
        insert_reference_antecedent(
            conn,
            plan.action_id,
            &spec.antecedent_action_id,
            (i + 1) as i64,
            spec.content_block_id.as_deref(),
            spec.range_start,
            spec.range_end,
            spec.annotation.as_deref(),
        )
        .await?;
    }
    Ok(PostOutcome {
        auto_titled,
        joined,
    })
}

/// Everything an edit writes, validated by the caller before this is opened.
/// See [`edit_post_tx`].
pub(crate) struct EditPostPlan<'a> {
    pub space_id: &'a str,
    pub participant_id: &'a str,
    /// The new generation's id.
    pub action_id: &'a str,
    pub item_id: &'a str,
    /// The current tip this edit supersedes.
    pub supersedes_action_id: &'a str,
    pub text: &'a str,
    /// The tip's reply edge, replicated.
    pub reply_to: Option<&'a str>,
    /// The surviving reference edges, replicated **at their original
    /// ordinals** (stable across generations — see `ReferenceSpec`'s ordinal
    /// convention).
    pub references: &'a [ReferenceEdgeRow],
    pub created_at: i64,
}

/// Append an edit — the new `user_input` generation, its text, its reply edge
/// and its surviving references — **in one `BEGIN IMMEDIATE` transaction.**
///
/// The same cure, for the same defect, as [`post_tx`]: a post is not its
/// action row, and an edit is sharper still — its new row *supersedes* the
/// tip, so written as separate autocommitted statements the item's current
/// generation says nothing for as long as the tail of writes takes, and a
/// reader keyed on the action (the sub-space driver's refill takes posts by
/// commit order) is entitled to that state and can plan and bill against it.
/// Nothing here spends or calls out; the caller validates before it opens
/// this, so a refused edit still writes nothing and cannot half-write either.
pub(crate) async fn edit_post_tx(
    conn: &Connection,
    plan: &EditPostPlan<'_>,
) -> Result<(), AppError> {
    begin_write(conn).await?;
    match edit_post_tx_body(conn, plan).await {
        Ok(()) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(())
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn edit_post_tx_body(conn: &Connection, plan: &EditPostPlan<'_>) -> Result<(), AppError> {
    insert_action(
        conn,
        &ActionEntry {
            id: plan.action_id.to_string(),
            space_id: plan.space_id.to_string(),
            participant_id: plan.participant_id.to_string(),
            item_id: plan.item_id.to_string(),
            supersedes_action_id: Some(plan.supersedes_action_id.to_string()),
            action_type: "user_input".to_string(),
            status: "complete".to_string(),
            intent: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            credits_consumed: None,
            created_at: plan.created_at,
        },
    )
    .await?;
    insert_text_content_block(
        conn,
        &uuid::Uuid::now_v7().to_string(),
        plan.action_id,
        0,
        "text",
        plan.text,
    )
    .await?;
    if let Some(ante) = plan.reply_to {
        insert_action_antecedent(conn, plan.action_id, ante, 0, "reply").await?;
    }
    for r in plan.references {
        insert_reference_antecedent(
            conn,
            plan.action_id,
            &r.antecedent_action_id,
            r.ordinal,
            r.content_block_id.as_deref(),
            r.range_start,
            r.range_end,
            r.annotation.as_deref(),
        )
        .await?;
    }
    Ok(())
}

pub async fn insert_action(conn: &Connection, entry: &ActionEntry) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO action (id, space_id, participant_id, participant_scope, item_id, \
         supersedes_action_id, supersedes_item_id, action_type, status, \
         intent, model, input_tokens, output_tokens, credits_consumed, created_at) \
         VALUES (?1, ?2, ?3, (SELECT scope FROM participant WHERE id = ?3), \
                 ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        (
            Value::Text(entry.id.clone()),
            Value::Text(entry.space_id.clone()),
            Value::Text(entry.participant_id.clone()),
            Value::Text(entry.item_id.clone()),
            opt_text(&entry.supersedes_action_id),
            // Denormalized supersedes item — always the row's own item (a
            // generation chain never hops items; the schema CHECKs equality
            // and the compound FK proves the referenced pair exists).
            match entry.supersedes_action_id {
                Some(_) => Value::Text(entry.item_id.clone()),
                None => Value::Null,
            },
            Value::Text(entry.action_type.clone()),
            Value::Text(entry.status.clone()),
            opt_text(&entry.intent),
            opt_text(&entry.model),
            match entry.input_tokens {
                Some(t) => Value::Integer(t),
                None => Value::Null,
            },
            match entry.output_tokens {
                Some(t) => Value::Integer(t),
                None => Value::Null,
            },
            match entry.credits_consumed {
                Some(c) => Value::Integer(c),
                None => Value::Null,
            },
            Value::Integer(entry.created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert action: {e}"),
    })?;
    Ok(())
}

pub async fn insert_action_antecedent(
    conn: &Connection,
    action_id: &str,
    antecedent_action_id: &str,
    ordinal: i64,
    relation: &str,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO action_antecedent \
         (action_id, antecedent_action_id, ordinal, relation) \
         VALUES (?1, ?2, ?3, ?4)",
        (
            Value::Text(action_id.to_string()),
            Value::Text(antecedent_action_id.to_string()),
            Value::Integer(ordinal),
            Value::Text(relation.to_string()),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert action_antecedent: {e}"),
    })?;
    Ok(())
}

/// Insert a non-structural `reference` edge with its quote detail columns
/// (`content_block_id` + byte `range` into that block's `text_content`, plus
/// an optional annotation). The schema CHECK requires `range_start`/`range_end`
/// to be both present or both absent, with `0 <= start < end`.
#[allow(clippy::too_many_arguments)]
pub async fn insert_reference_antecedent(
    conn: &Connection,
    action_id: &str,
    antecedent_action_id: &str,
    ordinal: i64,
    content_block_id: Option<&str>,
    range_start: Option<i64>,
    range_end: Option<i64>,
    annotation: Option<&str>,
) -> Result<(), AppError> {
    fn opt_text(v: Option<&str>) -> Value {
        match v {
            Some(s) => Value::Text(s.to_string()),
            None => Value::Null,
        }
    }
    fn opt_int(v: Option<i64>) -> Value {
        match v {
            Some(n) => Value::Integer(n),
            None => Value::Null,
        }
    }
    conn.execute(
        "INSERT INTO action_antecedent \
         (action_id, antecedent_action_id, ordinal, relation, \
          content_block_id, range_start, range_end, annotation) \
         VALUES (?1, ?2, ?3, 'reference', ?4, ?5, ?6, ?7)",
        (
            Value::Text(action_id.to_string()),
            Value::Text(antecedent_action_id.to_string()),
            Value::Integer(ordinal),
            opt_text(content_block_id),
            opt_int(range_start),
            opt_int(range_end),
            opt_text(annotation),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert reference antecedent: {e}"),
    })?;
    Ok(())
}

/// One outgoing `reference` edge of an action, with the referenced content
/// block's text joined in (for snippet resolution). Ordered by `ordinal`.
pub struct ReferenceEdgeRow {
    pub ordinal: i64,
    pub antecedent_action_id: String,
    pub content_block_id: Option<String>,
    pub range_start: Option<i64>,
    pub range_end: Option<i64>,
    pub annotation: Option<String>,
    /// `text_content` of the referenced content block, when the edge carries
    /// a `content_block_id` and that block has text.
    pub block_text: Option<String>,
    /// Action type of the antecedent, and block type of the quoted block (when
    /// there is one). Carried so a reader can apply the quotable rule —
    /// [`is_post_action_type`] + [`QUOTABLE_BLOCK_TYPE`] — *without* the query
    /// dropping rows: `edit_post` replicates whatever edges exist, and a
    /// replication that silently lost one would rewrite history.
    pub antecedent_action_type: String,
    pub block_type: Option<String>,
    /// The quoted post's author, as its **own** space names it:
    /// `COALESCE(space_participant.override_label, participant.label)` joined
    /// on `ant.space_id`. A per-space override is that space's name for the
    /// participant, so the reading space's override would misattribute a
    /// cross-space passage.
    pub antecedent_author_label: String,
}

impl ReferenceEdgeRow {
    /// Whether this edge's quoted passage may be shown or sent — a post's
    /// `text` block and nothing else (see [`crate::Inner::validate_reference_spec`]
    /// for why an edge could be otherwise).
    pub fn is_quotable(&self) -> bool {
        is_post_action_type(&self.antecedent_action_type)
            && self.block_type.as_deref() == Some(QUOTABLE_BLOCK_TYPE)
    }

    /// Whether the edge names a range at all. A range-less reference is a
    /// plain backlink — a pointer to a post, not a quote of one — which
    /// [`crate::ReferenceSpec`] permits deliberately ("both present or both
    /// absent"), so a reader must not describe it as a quote that failed.
    pub fn has_range(&self) -> bool {
        self.range_start.is_some() && self.range_end.is_some()
    }
}

/// The `reference`-relation antecedents of an action, ordinal order. Used by
/// `edit_post` to replicate references onto a new generation and by the
/// upstream-context embed expansion — which also renders each passage's
/// byline, hence the author join (on the **antecedent's** space).
/// How a post's author is **named**, as one SQL fragment every reader shares.
///
/// Two things about it are load-bearing and both are easy to get wrong
/// separately: the label is the one that post's **own space** gives the author
/// (a reading space's override is that space's name for somebody, and using it
/// on a cross-space passage misattributes it), and there is **no liveness
/// filter** — who wrote a post goes on being named after they retire, which is
/// exactly the record retirement promises to leave alone. Expects the author's
/// action to be in scope as `ant`, and binds `p` / `sp`.
const POST_AUTHOR_LABEL_SQL: &str = "COALESCE(sp.override_label, p.label)";

/// The joins [`POST_AUTHOR_LABEL_SQL`] reads from.
const POST_AUTHOR_JOIN_SQL: &str = "JOIN participant p ON p.id = ant.participant_id \
     LEFT JOIN space_participant sp \
       ON sp.space_id = ant.space_id AND sp.participant_id = ant.participant_id";

/// How a post's author is named, for a reader holding the post rather than an
/// edge to it — the same fragment [`reference_antecedents`] joins, asked
/// directly, so a passage cannot be attributed one way in a prompt and another
/// way in the record. `None` for an action that does not exist.
pub async fn post_author_label(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {POST_AUTHOR_LABEL_SQL} FROM action ant {POST_AUTHOR_JOIN_SQL} \
             WHERE ant.id = ?1"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

pub async fn reference_antecedents(
    conn: &Connection,
    action_id: &str,
) -> Result<Vec<ReferenceEdgeRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT aa.ordinal, aa.antecedent_action_id, aa.content_block_id, \
                    aa.range_start, aa.range_end, aa.annotation, cb.text_content, \
                    ant.action_type, cb.block_type, \
                    {POST_AUTHOR_LABEL_SQL} \
             FROM action_antecedent aa \
             JOIN action ant ON ant.id = aa.antecedent_action_id \
             {POST_AUTHOR_JOIN_SQL} \
             LEFT JOIN content_block cb ON cb.id = aa.content_block_id \
             WHERE aa.action_id = ?1 AND aa.relation = 'reference' \
             ORDER BY aa.ordinal ASC",
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(ReferenceEdgeRow {
            ordinal: row.get::<i64>(0).map_err(AppError::db)?,
            antecedent_action_id: row.get::<String>(1).map_err(AppError::db)?,
            content_block_id: row.get::<Option<String>>(2).map_err(AppError::db)?,
            range_start: row.get::<Option<i64>>(3).map_err(AppError::db)?,
            range_end: row.get::<Option<i64>>(4).map_err(AppError::db)?,
            annotation: row.get::<Option<String>>(5).map_err(AppError::db)?,
            block_text: row.get::<Option<String>>(6).map_err(AppError::db)?,
            antecedent_action_type: row.get::<String>(7).map_err(AppError::db)?,
            block_type: row.get::<Option<String>>(8).map_err(AppError::db)?,
            antecedent_author_label: row.get::<String>(9).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// `(action_id, block_type, text_content)` of a content block, or `None` if the
/// block doesn't exist. Used to validate that a [`crate::ReferenceSpec`]'s block
/// belongs to its antecedent action, is quotable at all
/// ([`QUOTABLE_BLOCK_TYPE`]), and that its byte range is honest.
pub async fn content_block_owner_text(
    conn: &Connection,
    content_block_id: &str,
) -> Result<Option<(String, String, Option<String>)>, AppError> {
    let mut stmt = conn
        .prepare("SELECT action_id, block_type, text_content FROM content_block WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(content_block_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some((
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<String>(1).map_err(AppError::db)?,
            row.get::<Option<String>>(2).map_err(AppError::db)?,
        ))),
    }
}

/// `(id, text_content)` of an action's first content block, whatever its type.
/// Test-only support for `AppCore::test_insert_unvalidated_reference`; the
/// production reference path never needs to name a block it wasn't given.
#[doc(hidden)]
#[cfg(feature = "test-support")]
pub async fn first_content_block(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<(String, Option<String>)>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, text_content FROM content_block \
             WHERE action_id = ?1 ORDER BY ordinal ASC LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some((
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<Option<String>>(1).map_err(AppError::db)?,
        ))),
    }
}

/// `(space_id, action_type)` of an action, or `None` if it doesn't exist — what
/// the reference gate needs to decide *where* a quote points and *what kind of
/// thing* it points at, in one read.
pub async fn action_space_and_type(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<(String, String)>, AppError> {
    let mut stmt = conn
        .prepare("SELECT space_id, action_type FROM action WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some((
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<String>(1).map_err(AppError::db)?,
        ))),
    }
}

/// Whether `action_id` is a **post `space_id` currently shows** — current
/// generation, terminal status, post type, the transcript's predicate. The
/// spawn's anchor guard asks this so a delegation cannot be opened from a
/// generation the parent hides.
async fn is_visible_post_in_space(
    conn: &Connection,
    space_id: &str,
    action_id: &str,
) -> Result<bool, AppError> {
    let sql = format!(
        "SELECT 1 FROM action a \
         JOIN item_current ic ON ic.current_action_id = a.id \
         WHERE a.id = ?1 AND a.space_id = ?2 \
           AND a.status IN ('complete', 'cancelled') \
           AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
         LIMIT 1"
    );
    let mut rows = conn
        .query(
            &sql,
            (
                Value::Text(action_id.to_string()),
                Value::Text(space_id.to_string()),
            ),
        )
        .await
        .map_err(AppError::db)?;
    Ok(rows.next().await.map_err(AppError::db)?.is_some())
}

/// The post `action_id`'s item currently shows, or `None` if that item has no
/// visible tip — superseded wording, a hidden `error` generation, a missing
/// row. The walk asks this of every frontier id before planning it, so an
/// The post an **item** currently shows in `space_id`, or `None` when it shows
/// none — the item-keyed twin of [`visible_tip_of_action`], through the same
/// transcript predicate (current generation, terminal status, post type).
///
/// It exists for the one question an action id cannot answer: *has the turn
/// that opened this delegated room written its answer yet?* A turn mints the
/// item its answer will be written under before it makes a single request
/// (`TurnPrep::inf_item_id`), and that item is the only thing that identifies
/// one turn among several the same agent may be running against the same post
/// — the answer's own action id does not exist until the turn ends, and a
/// capped or budget-stopped turn never writes one at all.
///
/// **`after_row` is still required, and the two rules are not redundant.** An
/// item names the turn but not the *generation*: a regeneration's
/// `inf_item_id` is the item it is revising, whose visible post until the turn
/// lands is the answer being replaced. So the item rules out an answer from
/// **another** turn, and the watermark rules out an answer that predates the
/// room; each covers exactly what the other cannot. See
/// [`last_reply_by_participant`] for the watermark's own argument.
pub async fn visible_post_of_item(
    conn: &Connection,
    space_id: &str,
    item_id: &str,
    after_row: Option<i64>,
) -> Result<Option<String>, AppError> {
    let sql = format!(
        "SELECT tip.id FROM item_current ic \
         JOIN action tip ON tip.id = ic.current_action_id \
         WHERE ic.item_id = ?1 AND ic.space_id = ?2 \
           AND tip.rowid > ?3 \
           AND tip.status IN ('complete', 'cancelled') \
           AND tip.action_type IN ({POST_ACTION_TYPES_SQL}) \
         LIMIT 1"
    );
    let mut rows = conn
        .query(
            &sql,
            (
                Value::Text(item_id.to_string()),
                Value::Text(space_id.to_string()),
                Value::Integer(after_row.unwrap_or(0)),
            ),
        )
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// edit or regeneration that landed while a sibling branch was walking is
/// one wording, not two.
pub async fn visible_tip_of_action(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<String>, AppError> {
    let sql = format!(
        "SELECT tip.id FROM action a \
         JOIN item_current ic ON ic.item_id = a.item_id AND ic.space_id = a.space_id \
         JOIN action tip ON tip.id = ic.current_action_id \
         WHERE a.id = ?1 \
           AND tip.status IN ('complete', 'cancelled') \
           AND tip.action_type IN ({POST_ACTION_TYPES_SQL}) \
         LIMIT 1"
    );
    let mut rows = conn
        .query(&sql, (Value::Text(action_id.to_string()),))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// One incoming `reference` edge: a post (elsewhere or in the same space)
/// referencing the queried action. Restricted to referrers that are their
/// item's **current generation** with terminal status — the set whose quotes
/// should highlight the source. The target is the *concrete generation*
/// queried (references never remap to tips).
pub struct IncomingReferenceRow {
    /// The referring post's action id (a current generation).
    pub action_id: String,
    /// The referring post's **item** — its identity across generations.
    ///
    /// The action id is the generation, and an edit mints a new one: editing a
    /// referring post without touching its quote replicates the edge onto a
    /// fresh action, so the backlink is the same backlink while
    /// [`Self::action_id`] is not the same string. Anything holding on to a
    /// row across a reload wants this.
    pub item_id: String,
    /// The referring post's space (references may cross spaces).
    pub space_id: String,
    pub ordinal: i64,
    pub content_block_id: Option<String>,
    pub range_start: Option<i64>,
    pub range_end: Option<i64>,
    pub annotation: Option<String>,
    pub created_at: i64,
    /// The **referring** post's author, as that post's own space names it
    /// (`COALESCE(space_participant.override_label, participant.label)` joined
    /// on `ar.space_id`). Carried for the same reason the outgoing direction
    /// carries [`AntecedentEdgeRow::antecedent_author_label`]: a referrer in
    /// another space cannot be named from anything the quoted space knows.
    pub author_label: String,
    /// The referring post's author's participant **kind**, from the same join.
    /// The label is not renderable on its own — see
    /// [`crate::IncomingReference::author_kind`].
    pub author_kind: String,
    /// The referring space's title, `NULL` for one never named.
    ///
    /// An author does not identify a post: the same participant can quote one
    /// passage from two conversations, and a surface listing backlinks would
    /// then offer two rows that read alike and open different windows. The
    /// title is what tells them apart, and it travels with the row that already
    /// names that space by id. A surface that shows it wants
    /// [`crate::AppCore::references_to_visible_to`], which reports only spaces
    /// the viewer may read.
    pub space_title: Option<String>,
}

/// All current-generation posts referencing `antecedent_action_id` (relation
/// `reference`), with their quoted ranges — the reverse index behind source
/// highlights and click-to-navigate. Pure read.
pub async fn references_to(
    conn: &Connection,
    antecedent_action_id: &str,
) -> Result<Vec<IncomingReferenceRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT aa.action_id, ar.space_id, aa.ordinal, aa.content_block_id, \
                    aa.range_start, aa.range_end, aa.annotation, ar.created_at, \
                    COALESCE(rsp.override_label, rp.label), rp.kind, rs.title, \
                    ar.item_id \
             FROM action_antecedent aa \
             JOIN action_resolved ar ON ar.action_id = aa.action_id \
             JOIN participant rp ON rp.id = ar.participant_id \
             JOIN space rs ON rs.id = ar.space_id \
             LEFT JOIN space_participant rsp \
               ON rsp.space_id = ar.space_id AND rsp.participant_id = ar.participant_id \
             WHERE aa.antecedent_action_id = ?1 \
               AND aa.relation = 'reference' \
               AND ar.action_type IN ({POST_ACTION_TYPES_SQL}) \
               AND ar.is_current = 1 \
               AND ar.status IN ('complete', 'cancelled') \
             ORDER BY ar.created_at ASC, aa.action_id ASC, aa.ordinal ASC"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(antecedent_action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(IncomingReferenceRow {
            action_id: row.get::<String>(0).map_err(AppError::db)?,
            space_id: row.get::<String>(1).map_err(AppError::db)?,
            ordinal: row.get::<i64>(2).map_err(AppError::db)?,
            content_block_id: row.get::<Option<String>>(3).map_err(AppError::db)?,
            range_start: row.get::<Option<i64>>(4).map_err(AppError::db)?,
            range_end: row.get::<Option<i64>>(5).map_err(AppError::db)?,
            annotation: row.get::<Option<String>>(6).map_err(AppError::db)?,
            created_at: row.get::<i64>(7).map_err(AppError::db)?,
            author_label: row.get::<String>(8).map_err(AppError::db)?,
            author_kind: row.get::<String>(9).map_err(AppError::db)?,
            space_title: row.get::<Option<String>>(10).map_err(AppError::db)?,
            item_id: row.get::<String>(11).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// `action.intent` marking a `checkpoint` action as a branch summary (see
/// [`crate::summaries`]).
pub const BRANCH_SUMMARY_INTENT: &str = "branch_summary";

/// Ordinal of a branch summary's `reference` edge to the **branch root** —
/// which branch this summarizes. Any generation of the root resolves to the
/// same item, which is the identity the cache keys on.
pub const BRANCH_SUMMARY_ROOT_ORDINAL: i64 = 1;

/// Ordinal of a branch summary's `reference` edge to the **tip it read** — the
/// concrete generation the summary was current as of, i.e. the cache key.
/// (Ordinal 0 stays reserved for the structural `reply` edge, which a summary
/// does not have: it is not part of the thread.)
pub const BRANCH_SUMMARY_TIP_ORDINAL: i64 = 2;

/// One current-generation branch summary in a space.
#[derive(Clone, Debug)]
pub struct BranchSummaryRow {
    /// Item id of the branch's root post — the branch's stable identity.
    pub branch_item_id: String,
    /// The summary action (the item's current generation).
    pub action_id: String,
    /// The summary's item — superseded, not replaced, when it regenerates.
    pub item_id: String,
    /// The branch tip this summary read. Compare against the branch's current
    /// tip to decide whether it is stale.
    pub summarized_action_id: String,
    pub text: String,
}

/// Every branch summary in a space, at its current generation.
pub async fn current_branch_summaries(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<BranchSummaryRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT root.item_id, s.id, s.item_id, tip_e.antecedent_action_id, cb.text_content \
             FROM action s \
             JOIN action_antecedent root_e \
               ON root_e.action_id = s.id AND root_e.ordinal = {BRANCH_SUMMARY_ROOT_ORDINAL} \
             JOIN action root ON root.id = root_e.antecedent_action_id \
             JOIN action_antecedent tip_e \
               ON tip_e.action_id = s.id AND tip_e.ordinal = {BRANCH_SUMMARY_TIP_ORDINAL} \
             LEFT JOIN content_block cb ON cb.action_id = s.id AND cb.ordinal = 0 \
             WHERE s.space_id = ?1 AND s.action_type = 'checkpoint' AND s.intent = ?2 \
               AND NOT EXISTS (SELECT 1 FROM action x WHERE x.supersedes_action_id = s.id)"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(space_id.to_string()),
            Value::Text(BRANCH_SUMMARY_INTENT.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        let text: Option<String> = row.get(4).map_err(AppError::db)?;
        let Some(text) = text else { continue };
        out.push(BranchSummaryRow {
            branch_item_id: row.get::<String>(0).map_err(AppError::db)?,
            action_id: row.get::<String>(1).map_err(AppError::db)?,
            item_id: row.get::<String>(2).map_err(AppError::db)?,
            summarized_action_id: row.get::<String>(3).map_err(AppError::db)?,
            text,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Agent memory (see [`crate::memory`])
// ---------------------------------------------------------------------------

/// `action_type` of one generation of a memory block. Not a post type, so
/// every render, tree and context query collapses it out for free.
pub const MEMORY_ACTION_TYPE: &str = "memory";

/// A new memory block's identity row, as it is **written**. Its *contents*
/// live on the item's action generations; this is who owns it, what it is
/// called, and where it applies.
#[derive(Clone, Debug)]
pub struct NewMemoryBlock {
    pub item_id: String,
    pub root_action_id: String,
    /// The owning agent. No scope field, for the same reason [`ActionEntry`]
    /// has none — [`insert_memory_block`] derives the echo from this id.
    pub owner_participant_id: String,
    pub name: String,
    /// `core` or `space` — the scope *label*, which is addressing and has
    /// nothing to do with the owner's participant scope.
    pub scope: String,
    /// Residence: the space the block is about.
    pub space_id: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// One block's identity row as it is **read back** (the write shape is
/// [`NewMemoryBlock`]).
#[derive(Clone, Debug)]
pub struct MemoryBlockRow {
    pub item_id: String,
    pub root_action_id: String,
    pub owner_participant_id: String,
    pub owner_scope: String,
    pub name: String,
    /// `core` (loads wherever the owner goes) or `space` (loads in
    /// `space_id` only). Addressing, never ownership.
    pub scope: String,
    /// Residence: the space the block is about, and where its actions live.
    pub space_id: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// One block as a turn loads it: the identity plus the current generation's
/// text.
#[derive(Clone, Debug)]
pub struct MemoryEntryRow {
    pub item_id: String,
    pub name: String,
    pub scope: String,
    pub space_id: String,
    pub action_id: String,
    pub text: String,
    pub updated_at: i64,
}

/// One generation of a block, for inspection.
#[derive(Clone, Debug)]
pub struct MemoryRevisionRow {
    pub action_id: String,
    /// Who wrote *this* generation. The owner is on the block; this is the
    /// author, which is what tells a self-revision from a human correction.
    pub author_participant_id: String,
    pub created_at: i64,
    pub text: String,
}

/// Insert a block's identity row. Like [`insert_action`], the pinned
/// `owner_scope` echo is **derived inside the statement** — [`NewMemoryBlock`]
/// therefore has no scope field to get wrong, and a promotion landing between
/// a revision's action write and this one can no longer strand the action
/// without its block.
pub async fn insert_memory_block(conn: &Connection, row: &NewMemoryBlock) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO memory_block (item_id, root_action_id, owner_participant_id, owner_scope, \
         name, scope, space_id, created_at, updated_at) \
         VALUES (?1, ?2, ?3, (SELECT scope FROM participant WHERE id = ?3), \
                 ?4, ?5, ?6, ?7, ?8)",
        (
            Value::Text(row.item_id.clone()),
            Value::Text(row.root_action_id.clone()),
            Value::Text(row.owner_participant_id.clone()),
            Value::Text(row.name.clone()),
            Value::Text(row.scope.clone()),
            Value::Text(row.space_id.clone()),
            Value::Integer(row.created_at),
            Value::Integer(row.updated_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert memory block: {e}"),
    })?;
    Ok(())
}

/// Record a revision against an existing block: its scope may move (`core` ⇄
/// `space`) and its `updated_at` advances. The contents are the new
/// generation's, so nothing else here changes.
pub async fn touch_memory_block(
    conn: &Connection,
    item_id: &str,
    scope: &str,
    updated_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "UPDATE memory_block SET scope = ?2, updated_at = ?3 WHERE item_id = ?1",
        (
            Value::Text(item_id.to_string()),
            Value::Text(scope.to_string()),
            Value::Integer(updated_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to update memory block: {e}"),
    })?;
    Ok(())
}

fn memory_block_row(row: &turso::Row) -> Result<MemoryBlockRow, AppError> {
    Ok(MemoryBlockRow {
        item_id: row.get::<String>(0).map_err(AppError::db)?,
        root_action_id: row.get::<String>(1).map_err(AppError::db)?,
        owner_participant_id: row.get::<String>(2).map_err(AppError::db)?,
        owner_scope: row.get::<String>(3).map_err(AppError::db)?,
        name: row.get::<String>(4).map_err(AppError::db)?,
        scope: row.get::<String>(5).map_err(AppError::db)?,
        space_id: row.get::<String>(6).map_err(AppError::db)?,
        created_at: row.get::<i64>(7).map_err(AppError::db)?,
        updated_at: row.get::<i64>(8).map_err(AppError::db)?,
    })
}

const MEMORY_BLOCK_COLUMNS: &str = "item_id, root_action_id, owner_participant_id, owner_scope, \
                                    name, scope, space_id, created_at, updated_at";

/// The owner's block of that name, wherever it resides. Names are unique per
/// owner, so this is the upsert key `remember` addresses.
pub async fn memory_block_by_name(
    conn: &Connection,
    owner_participant_id: &str,
    name: &str,
) -> Result<Option<MemoryBlockRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {MEMORY_BLOCK_COLUMNS} FROM memory_block \
             WHERE owner_participant_id = ?1 AND name = ?2"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(owner_participant_id.to_string()),
            Value::Text(name.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(memory_block_row(&row)?)),
    }
}

/// Every block an owner has, newest-updated first — the inspection read (and
/// the block-count budget's denominator).
pub async fn memory_blocks_owned(
    conn: &Connection,
    owner_participant_id: &str,
) -> Result<Vec<MemoryBlockRow>, AppError> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {MEMORY_BLOCK_COLUMNS} FROM memory_block \
             WHERE owner_participant_id = ?1 ORDER BY updated_at DESC, name ASC"
        ))
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(owner_participant_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(memory_block_row(&row)?);
    }
    Ok(out)
}

/// **The loading rule**: the participant's `core` blocks plus the blocks about
/// `space_id`, each at its current generation, core first then by name.
///
/// The order is a function of the data (not of write order), so a turn's bytes
/// only move when the memory itself does.
pub async fn participant_memory(
    conn: &Connection,
    owner_participant_id: &str,
    space_id: &str,
) -> Result<Vec<MemoryEntryRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT m.item_id, m.name, m.scope, m.space_id, ic.current_action_id, \
                    cb.text_content, m.updated_at \
             FROM memory_block m \
             JOIN item_current ic \
               ON ic.space_id = m.space_id AND ic.item_id = m.item_id \
             LEFT JOIN content_block cb \
               ON cb.action_id = ic.current_action_id AND cb.ordinal = 0 \
             WHERE m.owner_participant_id = ?1 \
               AND (m.scope = 'core' OR m.space_id = ?2) \
             ORDER BY CASE m.scope WHEN 'core' THEN 0 ELSE 1 END, m.name ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query((
            Value::Text(owner_participant_id.to_string()),
            Value::Text(space_id.to_string()),
        ))
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        let text: Option<String> = row.get(5).map_err(AppError::db)?;
        let Some(text) = text else { continue };
        out.push(MemoryEntryRow {
            item_id: row.get::<String>(0).map_err(AppError::db)?,
            name: row.get::<String>(1).map_err(AppError::db)?,
            scope: row.get::<String>(2).map_err(AppError::db)?,
            space_id: row.get::<String>(3).map_err(AppError::db)?,
            action_id: row.get::<String>(4).map_err(AppError::db)?,
            text,
            updated_at: row.get::<i64>(6).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// Every generation of a block, oldest first — the revision history the
/// inspector reads (and the authorship trail).
pub async fn memory_revisions(
    conn: &Connection,
    item_id: &str,
) -> Result<Vec<MemoryRevisionRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.id, a.participant_id, a.created_at, cb.text_content \
             FROM action a \
             LEFT JOIN content_block cb ON cb.action_id = a.id AND cb.ordinal = 0 \
             WHERE a.item_id = ?1 ORDER BY a.created_at ASC, a.id ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(item_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(MemoryRevisionRow {
            action_id: row.get::<String>(0).map_err(AppError::db)?,
            author_participant_id: row.get::<String>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            text: row
                .get::<Option<String>>(3)
                .map_err(AppError::db)?
                .unwrap_or_default(),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Content block operations
// ---------------------------------------------------------------------------

pub async fn insert_text_content_block(
    conn: &Connection,
    id: &str,
    action_id: &str,
    ordinal: i64,
    block_type: &str,
    text_content: &str,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO content_block (id, action_id, ordinal, block_type, text_content) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        (
            Value::Text(id.to_string()),
            Value::Text(action_id.to_string()),
            Value::Integer(ordinal),
            Value::Text(block_type.to_string()),
            Value::Text(text_content.to_string()),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert content_block: {e}"),
    })?;
    Ok(())
}

/// Insert a `tool_use` content block — one tool call the model requested.
///
/// The schema requires both `tool_name` and `tool_call_id` on this block type
/// (and forbids `media_data`); the raw argument string is stored verbatim in
/// `data`, so the Record shows exactly what the model asked for, valid JSON or
/// not.
pub async fn insert_tool_use_content_block(
    conn: &Connection,
    id: &str,
    action_id: &str,
    ordinal: i64,
    tool_name: &str,
    tool_call_id: &str,
    arguments: &str,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO content_block \
         (id, action_id, ordinal, block_type, tool_name, tool_call_id, data) \
         VALUES (?1, ?2, ?3, 'tool_use', ?4, ?5, ?6)",
        (
            Value::Text(id.to_string()),
            Value::Text(action_id.to_string()),
            Value::Integer(ordinal),
            Value::Text(tool_name.to_string()),
            Value::Text(tool_call_id.to_string()),
            Value::Text(arguments.to_string()),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert tool_use content_block: {e}"),
    })?;
    Ok(())
}

/// Insert a `tool_result` content block — what the harness handed back for one
/// call. Keyed by `tool_call_id` (the schema requires it and forbids
/// `tool_name` here); `text_content` is exactly the text the model was shown,
/// including an honest error line when the tool failed.
pub async fn insert_tool_result_content_block(
    conn: &Connection,
    id: &str,
    action_id: &str,
    ordinal: i64,
    tool_call_id: &str,
    text_content: &str,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO content_block \
         (id, action_id, ordinal, block_type, tool_call_id, text_content) \
         VALUES (?1, ?2, ?3, 'tool_result', ?4, ?5)",
        (
            Value::Text(id.to_string()),
            Value::Text(action_id.to_string()),
            Value::Integer(ordinal),
            Value::Text(tool_call_id.to_string()),
            Value::Text(text_content.to_string()),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert tool_result content_block: {e}"),
    })?;
    Ok(())
}

/// A raw action row plus its content blocks and structural reply parent —
/// **including the trace types** (`tool_call` / `tool_result`) that the render
/// views deliberately collapse out.
///
/// Exposed for the chat-path tests behind `AppCore::test_space_actions`: the
/// tool loop's durable rows are invisible to `get_space_tree` by design, so
/// asserting on them needs a view that doesn't filter. It is a plain SELECT —
/// no production caller, and nothing here that a future Record surface
/// couldn't reuse.
#[derive(Debug, Clone)]
pub struct RawActionRow {
    pub id: String,
    pub action_type: String,
    /// The acting participant, with the pinned composite echo of its scope.
    /// The echo is a constraint device, not a historical claim — task 36's
    /// in-place promotion rewrites it across an agent's whole trail via
    /// `ON UPDATE CASCADE` while `participant_id`, the identity, never moves.
    pub participant_id: String,
    pub participant_scope: String,
    pub status: String,
    pub model: Option<String>,
    pub credits_consumed: Option<i64>,
    pub reply_to: Option<String>,
    pub blocks: Vec<RawBlockRow>,
}

/// One content block of a [`RawActionRow`].
#[derive(Debug, Clone)]
pub struct RawBlockRow {
    pub block_type: String,
    pub text_content: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub data: Option<String>,
}

/// Every action in a space, oldest first, with its blocks and reply parent.
pub async fn raw_space_actions(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<RawActionRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.id, a.action_type, a.status, a.model, a.credits_consumed, \
                    (SELECT aa.antecedent_action_id FROM action_antecedent aa \
                      WHERE aa.action_id = a.id AND aa.relation = 'reply'), \
                    a.participant_id, a.participant_scope \
             FROM action a WHERE a.space_id = ?1 ORDER BY a.created_at ASC, a.id ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out: Vec<RawActionRow> = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(RawActionRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            action_type: row.get::<String>(1).map_err(AppError::db)?,
            status: row.get::<String>(2).map_err(AppError::db)?,
            model: row.get::<Option<String>>(3).map_err(AppError::db)?,
            credits_consumed: row.get::<Option<i64>>(4).map_err(AppError::db)?,
            reply_to: row.get::<Option<String>>(5).map_err(AppError::db)?,
            participant_id: row.get::<String>(6).map_err(AppError::db)?,
            participant_scope: row.get::<String>(7).map_err(AppError::db)?,
            blocks: Vec::new(),
        });
    }
    for action in out.iter_mut() {
        let mut stmt = conn
            .prepare(
                "SELECT block_type, text_content, tool_name, tool_call_id, data \
                 FROM content_block WHERE action_id = ?1 ORDER BY ordinal ASC",
            )
            .await
            .map_err(AppError::db)?;
        let mut rows = stmt
            .query([Value::Text(action.id.clone())])
            .await
            .map_err(AppError::db)?;
        while let Some(row) = rows.next().await.map_err(AppError::db)? {
            action.blocks.push(RawBlockRow {
                block_type: row.get::<String>(0).map_err(AppError::db)?,
                text_content: row.get::<Option<String>>(1).map_err(AppError::db)?,
                tool_name: row.get::<Option<String>>(2).map_err(AppError::db)?,
                tool_call_id: row.get::<Option<String>>(3).map_err(AppError::db)?,
                data: row.get::<Option<String>>(4).map_err(AppError::db)?,
            });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: System prompt operations
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub async fn upsert_system_prompt(conn: &Connection, text: &str) -> Result<String, AppError> {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.as_bytes());
    let hash: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    conn.execute(
        "INSERT OR IGNORE INTO system_prompt (hash, text) VALUES (?1, ?2)",
        (Value::Text(hash.clone()), Value::Text(text.to_string())),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to upsert system_prompt: {e}"),
    })?;
    Ok(hash)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Context assembly operations
// ---------------------------------------------------------------------------

pub async fn insert_context_assembly(
    conn: &Connection,
    id: &str,
    action_id: &str,
    system_prompt_hash: Option<&str>,
    total_tokens: Option<i64>,
    truncation_applied: bool,
    created_at: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO context_assembly (id, action_id, system_prompt_hash, total_tokens, truncation_applied, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (
            Value::Text(id.to_string()),
            Value::Text(action_id.to_string()),
            match system_prompt_hash {
                Some(h) => Value::Text(h.to_string()),
                None => Value::Null,
            },
            match total_tokens {
                Some(t) => Value::Integer(t),
                None => Value::Null,
            },
            Value::Integer(if truncation_applied { 1 } else { 0 }),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert context_assembly: {e}"),
    })?;
    Ok(())
}

pub async fn insert_context_assembly_action(
    conn: &Connection,
    context_assembly_id: &str,
    action_id: &str,
    position: i64,
) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO context_assembly_action (context_assembly_id, action_id, position) \
         VALUES (?1, ?2, ?3)",
        (
            Value::Text(context_assembly_id.to_string()),
            Value::Text(action_id.to_string()),
            Value::Integer(position),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to insert context_assembly_action: {e}"),
    })?;
    Ok(())
}

/// The actions an inference's context assembly records, in `position` order —
/// the ordered composition of the prompt that produced it.
pub async fn context_assembly_actions(
    conn: &Connection,
    inference_action_id: &str,
) -> Result<Vec<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT caa.action_id \
             FROM context_assembly ca \
             JOIN context_assembly_action caa ON caa.context_assembly_id = ca.id \
             WHERE ca.action_id = ?1 \
             ORDER BY caa.position ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(inference_action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(row.get::<String>(0).map_err(AppError::db)?);
    }
    Ok(out)
}

/// Ordinal of a `decision`'s `reference` edge to the **root of its own turn's
/// trace chain** — which turn declined (see [`crate::decline`]).
///
/// A decision's ordinal 0 is its structural `reply` edge to the post it
/// declines, which is deliberately *not* the trace chain: a decision is about
/// the post, not about the rounds. That leaves the rounds and the decision of
/// one turn with no durable link between them, which matters as soon as a
/// participant declines the same post twice — so the link is written as a real
/// relation, the same way a branch summary carries its keys
/// ([`BRANCH_SUMMARY_ROOT_ORDINAL`]) rather than packing them into a string.
pub const DECLINE_TRACE_ORDINAL: i64 = 1;

/// One trace action of a space — a `tool_call`, `tool_result` or `decision`
/// row with everything the space UI's audit disclosure needs (task 34).
///
/// The render views collapse these types out by design; this is the parallel
/// read that puts them back, keyed off the posts already on screen.
#[derive(Debug, Clone)]
pub struct TraceActionRow {
    pub id: String,
    /// `tool_call` | `tool_result` | `decision`.
    pub action_type: String,
    pub created_at: i64,
    pub participant_id: String,
    /// The acting participant's **effective** label in this space.
    pub participant_label: String,
    /// The structural `reply` antecedent: the previous round for a chained
    /// trace, the post the turn answered for a chain's first round and for
    /// every `decision`.
    pub reply_to: Option<String>,
    /// `reply_to`'s item resolved to its **current generation** — where the
    /// answered post now lives in the rendered tree.
    pub reply_to_current: Option<String>,
    /// The **earliest** inference whose context assembly recorded this action:
    /// the turn that produced it. A later turn of the same participant replays
    /// its own rounds (task 33) and so records them again, which is why the
    /// earliest wins.
    ///
    /// `None` is the gap case the disclosure exists to make visible — a turn
    /// that produced no post at all (a decline, a round-cap exit, a failure)
    /// leaves its trace with no answer to attribute it to.
    pub produced_by: Option<String>,
    /// The raw exchange this round is recorded under — the Record deep link.
    pub request_id: Option<String>,
    /// A `decision`'s `reference` edge to the **root of its own turn's trace
    /// chain** ([`DECLINE_TRACE_ORDINAL`]) — the durable link that says which
    /// turn declined.
    ///
    /// A decision hangs off the *post* it declines, not off the chain (task
    /// 22), so nothing structural would otherwise tie it to the rounds it ran
    /// — and a participant may decline the same post more than once. `None`
    /// on every non-`decision` row (a round's own chain root is reachable by
    /// walking its `reply_to` edges).
    pub turn_root: Option<String>,
    pub blocks: Vec<RawBlockRow>,
}

/// Every trace action in a space, oldest first, with its content blocks.
///
/// One read backs the whole disclosure: attribution (`produced_by`), the gap
/// anchor (`reply_to` / `reply_to_current`), turn identity for a decision
/// (`turn_root`), the payload (`blocks`) and the Record link (`request_id`).
/// Ordering is the actions' own, so a turn's rounds come back in the order
/// they ran.
pub async fn space_trace_rows(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<TraceActionRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.id, a.action_type, a.created_at, a.participant_id, \
                    COALESCE(sp.override_label, p.label), \
                    (SELECT aa.antecedent_action_id FROM action_antecedent aa \
                      WHERE aa.action_id = a.id AND aa.relation = 'reply'), \
                    (SELECT ic.current_action_id \
                       FROM action pa \
                       JOIN item_current ic \
                         ON ic.item_id = pa.item_id AND ic.space_id = pa.space_id \
                      WHERE pa.id = (SELECT aa2.antecedent_action_id FROM action_antecedent aa2 \
                                      WHERE aa2.action_id = a.id AND aa2.relation = 'reply')), \
                    (SELECT ca.action_id FROM context_assembly ca \
                       JOIN context_assembly_action caa ON caa.context_assembly_id = ca.id \
                       JOIN action inf ON inf.id = ca.action_id \
                      WHERE caa.action_id = a.id AND inf.action_type = 'inference' \
                      ORDER BY inf.created_at ASC, inf.id ASC LIMIT 1), \
                    (SELECT r.id FROM request r WHERE r.action_id = a.id \
                      ORDER BY r.created_at ASC LIMIT 1), \
                    (SELECT aa3.antecedent_action_id FROM action_antecedent aa3 \
                      WHERE aa3.action_id = a.id AND aa3.relation = 'reference' \
                        AND aa3.ordinal = ?2) \
             FROM action a \
             JOIN participant p ON p.id = a.participant_id \
             LEFT JOIN space_participant sp \
               ON sp.space_id = a.space_id AND sp.participant_id = a.participant_id \
             WHERE a.space_id = ?1 \
               AND a.action_type IN ('tool_call', 'tool_result', 'decision') \
             ORDER BY a.created_at ASC, a.id ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([
            Value::Text(space_id.to_string()),
            Value::Integer(DECLINE_TRACE_ORDINAL),
        ])
        .await
        .map_err(AppError::db)?;
    let mut out: Vec<TraceActionRow> = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(TraceActionRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            action_type: row.get::<String>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            participant_id: row.get::<String>(3).map_err(AppError::db)?,
            participant_label: row
                .get::<Option<String>>(4)
                .map_err(AppError::db)?
                .unwrap_or_default(),
            reply_to: row.get::<Option<String>>(5).map_err(AppError::db)?,
            reply_to_current: row.get::<Option<String>>(6).map_err(AppError::db)?,
            produced_by: row.get::<Option<String>>(7).map_err(AppError::db)?,
            request_id: row.get::<Option<String>>(8).map_err(AppError::db)?,
            turn_root: row.get::<Option<String>>(9).map_err(AppError::db)?,
            blocks: Vec::new(),
        });
    }
    for action in out.iter_mut() {
        let mut stmt = conn
            .prepare(
                "SELECT block_type, text_content, tool_name, tool_call_id, data \
                 FROM content_block WHERE action_id = ?1 ORDER BY ordinal ASC",
            )
            .await
            .map_err(AppError::db)?;
        let mut rows = stmt
            .query([Value::Text(action.id.clone())])
            .await
            .map_err(AppError::db)?;
        while let Some(row) = rows.next().await.map_err(AppError::db)? {
            action.blocks.push(RawBlockRow {
                block_type: row.get::<String>(0).map_err(AppError::db)?,
                text_content: row.get::<Option<String>>(1).map_err(AppError::db)?,
                tool_name: row.get::<Option<String>>(2).map_err(AppError::db)?,
                tool_call_id: row.get::<Option<String>>(3).map_err(AppError::db)?,
                data: row.get::<Option<String>>(4).map_err(AppError::db)?,
            });
        }
    }
    Ok(out)
}

/// One tool-payload content block of a **trace** action recorded in some
/// inference's context assembly — the rows a later turn of the same
/// participant replays as its own first-person memory (task 33).
pub struct TraceBlockRow {
    /// The `tool_call` / `tool_result` action the block belongs to.
    pub action_id: String,
    /// `tool_call` | `tool_result`.
    pub action_type: String,
    /// `tool_use` | `tool_result`.
    pub block_type: String,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    /// A `tool_result` block's text — what the model was shown.
    pub text_content: Option<String>,
    /// A `tool_use` block's raw arguments string.
    pub data: Option<String>,
}

/// The tool rounds an inference's turn actually ran, in the order they were
/// fed upstream: the `tool_use` / `tool_result` blocks of the `tool_call` and
/// `tool_result` actions recorded in that inference's **context assembly**.
///
/// The assembly is what attributes a trace to its turn. The trace chain hangs
/// off the *post* the turn answered (never off the inference — a capped or
/// declined turn writes no inference), so nothing structural links a chain to
/// the answer it produced; the writing turn, however, recorded exactly its own
/// rounds as fed context. Reading it back is therefore branch-correct,
/// regeneration-correct and abandoned-chain-correct by construction: you see
/// the traces of the turns whose posts you can see.
///
/// Only the tool payload blocks come back. A round's `thinking` and `text`
/// blocks are deliberately excluded — the live loop replays a round as an
/// assistant message with `content: null` and its calls, so replaying anything
/// more would be a shape the model never produced.
pub async fn assembly_trace_blocks(
    conn: &Connection,
    inference_action_id: &str,
) -> Result<Vec<TraceBlockRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.id, a.action_type, cb.block_type, cb.tool_name, cb.tool_call_id, \
                    cb.text_content, cb.data \
             FROM context_assembly ca \
             JOIN context_assembly_action caa ON caa.context_assembly_id = ca.id \
             JOIN action a ON a.id = caa.action_id \
             JOIN content_block cb ON cb.action_id = a.id \
             WHERE ca.action_id = ?1 \
               AND a.action_type IN ('tool_call', 'tool_result') \
               AND cb.block_type IN ('tool_use', 'tool_result') \
             ORDER BY caa.position ASC, cb.ordinal ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(inference_action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(TraceBlockRow {
            action_id: row.get::<String>(0).map_err(AppError::db)?,
            action_type: row.get::<String>(1).map_err(AppError::db)?,
            block_type: row.get::<String>(2).map_err(AppError::db)?,
            tool_name: row.get::<Option<String>>(3).map_err(AppError::db)?,
            tool_call_id: row.get::<Option<String>>(4).map_err(AppError::db)?,
            text_content: row.get::<Option<String>>(5).map_err(AppError::db)?,
            data: row.get::<Option<String>>(6).map_err(AppError::db)?,
        });
    }
    Ok(out)
}

/// Whether a space holds any tool-round trace at all. The one cheap read that
/// keeps a traceless space — the overwhelming majority — from paying a
/// per-spine-inference lookup on every turn.
pub async fn space_has_trace_actions(conn: &Connection, space_id: &str) -> Result<bool, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT 1 FROM action \
             WHERE space_id = ?1 AND action_type IN ('tool_call', 'tool_result') \
             LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    Ok(rows.next().await.map_err(AppError::db)?.is_some())
}

/// Author (`participant_id`) of every current-generation **post** in a space,
/// keyed by action id. The thread map's per-participant annotation needs it
/// and `PostNode` carries only the display label, so it rides alongside the
/// tree materials rather than widening a public DTO.
pub async fn post_authors(
    conn: &Connection,
    space_id: &str,
) -> Result<std::collections::HashMap<String, String>, AppError> {
    let sql = format!(
        "SELECT a.id, a.participant_id \
         FROM action a \
         JOIN item_current ic ON ic.current_action_id = a.id \
         WHERE a.space_id = ?1 AND a.action_type IN ({POST_ACTION_TYPES_SQL})"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut out = std::collections::HashMap::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.insert(
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<String>(1).map_err(AppError::db)?,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Space query operations
// ---------------------------------------------------------------------------

/// A space's own row, as the one statement that reads it answers.
///
/// **Existence, availability and the cascade budget arrive together on
/// purpose.** Every caller that asks "is this space there" is about to do
/// something in it, and whether it is *archived* decides that just as
/// completely as whether it exists — so the two are one read rather than a
/// check a later statement could be interleaved with (see
/// [`get_space`] and the two gates in `crate::Inner::prepare_turn` and
/// `crate::Inner::mechanical_plan`).
pub struct SpaceRow {
    pub id: String,
    pub title: Option<String>,
    pub created_at: i64,
    /// The space's runaway guard, read here so a planner needs one statement
    /// for its whole verdict. [`space_cascade_limit`] remains for the two
    /// callers that want the column alone.
    pub cascade_limit: i64,
    /// When the space was archived, if it has been. **Archived means it takes
    /// no new turns**: not a soft delete, not a departure — every membership
    /// and every read goes on working exactly as before.
    pub archived_at: Option<i64>,
    /// The may-decline router's model (`None` = the router is off), read here
    /// so refinement can decide "is this room still open, and does it route"
    /// in one statement rather than two. [`space_router_model`] remains for the
    /// callers that want the column alone.
    pub router_model: Option<String>,
}

/// One row of the space listing, with the cheap activity signals the UI
/// needs to render a meaningful entry. `last_activity_at` is the max
/// `action.created_at` over **current generations** in the space (falling back
/// to the space's own `created_at` for empty spaces); `message_count` counts
/// terminal (`complete`/`cancelled`) **current-generation** actions — editing
/// an item appends a generation but does not inflate the count.
pub struct SpaceListRow {
    pub id: String,
    pub title: Option<String>,
    pub created_at: i64,
    pub archived_at: Option<i64>,
    pub last_activity_at: i64,
    pub message_count: i64,
    /// The conversation this one was delegated from, when it was — `None` for
    /// an ordinary conversation.
    pub parent: Option<ParentSpace>,
}

/// The conversation a delegated room was opened from: enough to name it and to
/// open it, and nothing else.
///
/// **One value, not two nullable columns.** "Which conversation" and "what it
/// is called" travel together or not at all — a title with no id names
/// somewhere a reader cannot go, and an id with no title is a link with no
/// words — so the pair is the unit and its absence is the ordinary case. The
/// title stays optional inside it because a conversation genuinely need not
/// have one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentSpace {
    pub space_id: String,
    pub title: Option<String>,
}

pub async fn list_spaces(
    conn: &Connection,
    include_archived: bool,
) -> Result<Vec<SpaceListRow>, AppError> {
    // Notebook spaces are excluded unconditionally: this is the Library
    // listing, and an agent's notebook is reached from that agent's
    // management surface, not from the list of the human's conversations.
    // `include_archived` is about archival, an orthogonal axis — so it does
    // not surface notebooks either.
    let filter = if include_archived {
        "WHERE s.notebook_participant_id IS NULL "
    } else {
        "WHERE s.notebook_participant_id IS NULL AND s.archived_at IS NULL "
    };
    // The parent join is a self-join on the navigational `parent_space_id`, and
    // it answers in the **same read** as the listing rather than in a per-row
    // lookup behind it: a listing is virtualized, so a second read per row
    // would put a query inside a scroll. A delegated room's parent always
    // exists (the FK says so) and is deliberately **not** filtered on archival
    // or on being a notebook — a room can be opened from either, and naming
    // where a row came from is not a door into it.
    let sql = format!(
        "SELECT s.id, s.title, s.created_at, s.archived_at, \
                COALESCE(MAX(a.created_at), s.created_at) AS last_activity_at, \
                COUNT(a.id) AS message_count, \
                s.parent_space_id, p.title \
         FROM space s \
         LEFT JOIN space p ON p.id = s.parent_space_id \
         LEFT JOIN action a ON a.space_id = s.id \
              AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
              AND a.status IN ('complete', 'cancelled') \
              AND NOT EXISTS ( \
                  SELECT 1 FROM action sup WHERE sup.supersedes_action_id = a.id \
              ) \
         {filter}\
         GROUP BY s.id, s.title, s.created_at, s.archived_at, s.parent_space_id, p.title \
         ORDER BY last_activity_at DESC"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(SpaceListRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            title: row.get::<Option<String>>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            archived_at: row.get::<Option<i64>>(3).map_err(AppError::db)?,
            last_activity_at: row.get::<i64>(4).map_err(AppError::db)?,
            message_count: row.get::<i64>(5).map_err(AppError::db)?,
            parent: row
                .get::<Option<String>>(6)
                .map_err(AppError::db)?
                .map(|space_id| ParentSpace {
                    space_id,
                    title: row.get::<Option<String>>(7).ok().flatten(),
                }),
        });
    }
    Ok(results)
}

/// First text content block of the first user_input *item* in a space, at its
/// **current generation** — the raw source for the listing snippet shown for
/// untitled spaces. Ordered by `item_id` (UUIDv7, ~item-creation order) so an
/// edit to the first post updates the snippet rather than leaving the original.
pub async fn first_user_text(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT cb.text_content \
             FROM action a \
             JOIN content_block cb ON cb.action_id = a.id \
             WHERE a.space_id = ?1 AND a.action_type = 'user_input' \
               AND cb.block_type = 'text' \
               AND NOT EXISTS ( \
                   SELECT 1 FROM action sup WHERE sup.supersedes_action_id = a.id \
               ) \
             ORDER BY a.item_id ASC, cb.ordinal ASC \
             LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(row.get::<Option<String>>(0).map_err(AppError::db)?),
    }
}

/// Whether a space is archived. Answers `false` for a space that does not
/// exist — the question is "is this one hidden", not "does this one exist".
pub async fn space_is_archived(conn: &Connection, space_id: &str) -> Result<bool, AppError> {
    let mut stmt = conn
        .prepare("SELECT archived_at IS NOT NULL FROM space WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(false),
        Some(row) => Ok(row.get::<i64>(0).map_err(AppError::db)? != 0),
    }
}

pub async fn get_space(conn: &Connection, space_id: &str) -> Result<Option<SpaceRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, title, created_at, cascade_limit, archived_at, router_model \
             FROM space WHERE id = ?1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(SpaceRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            title: row.get::<Option<String>>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            cascade_limit: row.get::<i64>(3).map_err(AppError::db)?,
            archived_at: row.get::<Option<i64>>(4).map_err(AppError::db)?,
            router_model: row.get::<Option<String>>(5).map_err(AppError::db)?,
        })),
    }
}

pub struct SpaceActionRow {
    pub action_id: String,
    /// The action's **item** identity — stable across edits/regenerations.
    /// This is what the upstream message header's handle is derived from
    /// (`crate::post_handle`), so a post keeps its handle when edited.
    pub item_id: String,
    pub action_type: String,
    pub participant_id: String,
    pub participant_kind: String,
    /// The author's **effective** label in this space: a referenced global's
    /// per-membership `override_label` when set, else the participant's own.
    /// Rendered into the upstream message header next to the handle.
    pub participant_label: String,
    pub status: String,
    pub text_content: Option<String>,
    pub block_ordinal: Option<i64>,
    /// Creation time of **this generation** (the item's current tip), epoch ms.
    /// Rendered into the upstream message header's stamp field — see
    /// `crate::post_stamp` for why the tip's time rather than the item's.
    pub created_at: i64,
}

/// Returns every current-generation action in a space with its text content
/// blocks — the flat **whole-space** view backing `get_space_messages`. Turns
/// no longer send this upstream (`get_upstream_context` assembles the
/// branch-scoped thread instead). Filters to terminal statuses and to the
/// *current* generation of each item (via item_current), so superseded
/// generations never enter the context the model sees.
///
/// **Only `text` blocks are joined.** An inference's persisted `thinking`
/// block is the model's own reasoning; it is a render-side disclosure, never
/// part of a conversation's readable text (and never something we replay to a
/// model). The filter rides the `LEFT JOIN`'s `ON` clause so an action with no
/// text block still yields its row.
pub async fn get_space_actions_for_context(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<SpaceActionRow>, AppError> {
    // Ordered by the item's origin (first generation), not the tip's own
    // created_at — an edited early post must occupy its original position in
    // the model's context, not jump to the end (same rule as the render
    // order in `get_space_tree_data`).
    let mut stmt = conn
        .prepare(
            "SELECT a.id, a.item_id, a.action_type, a.participant_id, p.kind, \
                    COALESCE(sp.override_label, p.label), a.status, \
                    cb.text_content, cb.ordinal, a.created_at \
             FROM action a \
             JOIN item_current ic \
               ON ic.current_action_id = a.id \
             JOIN participant p ON p.id = a.participant_id \
             LEFT JOIN space_participant sp \
               ON sp.space_id = a.space_id AND sp.participant_id = a.participant_id \
             JOIN (SELECT space_id, item_id, \
                          MIN(created_at) AS born_at, MIN(id) AS first_action_id \
                   FROM action GROUP BY space_id, item_id) origin \
               ON origin.space_id = a.space_id AND origin.item_id = a.item_id \
             LEFT JOIN content_block cb \
               ON cb.action_id = a.id AND cb.block_type = 'text' \
             WHERE a.space_id = ?1 \
               AND a.status IN ('complete', 'cancelled') \
             ORDER BY origin.born_at ASC, origin.first_action_id ASC, cb.ordinal ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(SpaceActionRow {
            action_id: row.get::<String>(0).map_err(AppError::db)?,
            item_id: row.get::<String>(1).map_err(AppError::db)?,
            action_type: row.get::<String>(2).map_err(AppError::db)?,
            participant_id: row.get::<String>(3).map_err(AppError::db)?,
            participant_kind: row.get::<String>(4).map_err(AppError::db)?,
            participant_label: row.get::<String>(5).map_err(AppError::db)?,
            status: row.get::<String>(6).map_err(AppError::db)?,
            text_content: row.get::<Option<String>>(7).map_err(AppError::db)?,
            block_ordinal: row.get::<Option<i64>>(8).map_err(AppError::db)?,
            created_at: row.get::<i64>(9).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

/// The **upstream** context of an action: its reply ancestry, root-first, each
/// hop resolved to the antecedent **item's current tip** — an edited ancestor
/// contributes its most recent version. Everything downstream of the action
/// (replies to it, later turns) and every sibling branch is excluded. This is
/// the thread a turn sends upstream: a regeneration (`ResponseMode::Revise`)
/// walks from the generation being replaced *exclusive* (the model must not
/// see its own prior output); a fresh reply (`ResponseMode::Reply`) walks from
/// the post being answered *inclusive* — on a linear thread that is the whole
/// conversation, and on a branched space it is exactly the target's branch.
///
/// Like `get_space_actions_for_context`, only `text` blocks are joined: a
/// persisted `thinking` block is a render-side disclosure of one agent's own
/// reasoning and is never replayed upstream.
pub async fn get_upstream_context(
    conn: &Connection,
    target_action_id: &str,
    include_target: bool,
) -> Result<Vec<SpaceActionRow>, AppError> {
    // Walk the reply chain upward (short, local — one query per hop). Each
    // tip carries its own reply edge (edits/regenerations replicate it).
    // Item-tip re-rooting can create logical cycles in *resolved* space even
    // though raw edges only point backward, so guard on revisit.
    let mut chain: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(target_action_id.to_string());
    let mut cursor = target_action_id.to_string();
    if include_target {
        // The target itself reads at its item's current tip too (a reply to a
        // since-edited post answers the latest text).
        let tip = item_tip_of_action(conn, target_action_id)
            .await?
            .unwrap_or_else(|| target_action_id.to_string());
        seen.insert(tip.clone());
        chain.push(tip.clone());
        cursor = tip;
    }
    while let Some(parent_tip) = reply_antecedent_tip(conn, &cursor).await? {
        if !seen.insert(parent_tip.clone()) {
            break;
        }
        chain.push(parent_tip.clone());
        cursor = parent_tip;
    }
    chain.reverse(); // root first — the order the model reads the thread

    let mut results = Vec::new();
    for id in &chain {
        let mut stmt = conn
            .prepare(
                "SELECT a.id, a.item_id, a.action_type, a.participant_id, p.kind, \
                        COALESCE(sp.override_label, p.label), a.status, \
                        cb.text_content, cb.ordinal, a.created_at \
                 FROM action a \
                 JOIN participant p ON p.id = a.participant_id \
                 LEFT JOIN space_participant sp \
                   ON sp.space_id = a.space_id AND sp.participant_id = a.participant_id \
                 LEFT JOIN content_block cb \
                   ON cb.action_id = a.id AND cb.block_type = 'text' \
                 WHERE a.id = ?1 \
                   AND a.status IN ('complete', 'cancelled') \
                 ORDER BY cb.ordinal ASC",
            )
            .await
            .map_err(AppError::db)?;
        let mut rows = stmt
            .query([Value::Text(id.clone())])
            .await
            .map_err(AppError::db)?;
        while let Some(row) = rows.next().await.map_err(AppError::db)? {
            results.push(SpaceActionRow {
                action_id: row.get::<String>(0).map_err(AppError::db)?,
                item_id: row.get::<String>(1).map_err(AppError::db)?,
                action_type: row.get::<String>(2).map_err(AppError::db)?,
                participant_id: row.get::<String>(3).map_err(AppError::db)?,
                participant_kind: row.get::<String>(4).map_err(AppError::db)?,
                participant_label: row.get::<String>(5).map_err(AppError::db)?,
                status: row.get::<String>(6).map_err(AppError::db)?,
                text_content: row.get::<Option<String>>(7).map_err(AppError::db)?,
                block_ordinal: row.get::<Option<i64>>(8).map_err(AppError::db)?,
                created_at: row.get::<i64>(9).map_err(AppError::db)?,
            });
        }
    }
    Ok(results)
}

/// The current tip of `action_id`'s own item. `None` if the action doesn't
/// exist.
async fn item_tip_of_action(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT ic.current_action_id \
             FROM action a \
             JOIN item_current ic \
               ON ic.space_id = a.space_id AND ic.item_id = a.item_id \
             WHERE a.id = ?1 \
             LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// The current tip of the item that `action_id`'s reply antecedent belongs to
/// — one upward hop of the item-identity thread walk. `None` for a root (no
/// reply edge) or malformed data (no tip resolvable).
async fn reply_antecedent_tip(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT ic.current_action_id \
             FROM action_antecedent aa \
             JOIN action ant ON ant.id = aa.antecedent_action_id \
             JOIN item_current ic \
               ON ic.space_id = ant.space_id AND ic.item_id = ant.item_id \
             WHERE aa.action_id = ?1 AND aa.relation = 'reply' \
             LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

// ---------------------------------------------------------------------------
// Space tree — the materials for the threaded-post render (get_space_tree).
//
// Unlike get_space_actions_for_context (which flattens to OpenAI messages for
// the model), this fetches each item's CURRENT generation (item tip) as a
// renderable post, with its participant identity, derived generation number,
// content blocks, and antecedent edges (structural `reply` parent + any
// `reference` links). The Rust side (lib::build_post_tree) assembles these
// into the flattened render-row list. Only post-bearing action types render;
// trace types (request/tool/retrieval/…) are collapsed out here.
// ---------------------------------------------------------------------------

/// Action types that render as posts in the threaded view. Trace types are
/// excluded so requests/tool plumbing don't appear as posts. (Widen this as
/// tool_call/tool_result/error gain a post render.)
///
/// `brief` is the post an agent authors directly rather than by inferring it
/// — the opening post of an agent-spawned sub-space. It is a post on every
/// axis that matters here (it renders, it is replied to, it notifies, it may
/// be quoted), so it belongs to this set; what it is not is a human's input or
/// a model's answer, which is why it is its own type.
pub const POST_ACTION_TYPES_SQL: &str = "'user_input', 'inference', 'brief'";

/// The same set, for the checks that happen in Rust rather than in SQL — the
/// reference gate (what a quote may name) and the read-side filters that back
/// it up. Kept beside [`POST_ACTION_TYPES_SQL`] so the two cannot drift.
pub const POST_ACTION_TYPES: [&str; 3] = ["user_input", "inference", BRIEF_ACTION_TYPE];

/// The opening post of an agent-spawned sub-space — the only action type this
/// module writes outside a turn, and written in exactly one place
/// ([`spawn_subspace_tx`]). Named because planning asks about it by name (a
/// brief is the one post that must schedule somebody; see
/// `crate::Inner::mechanical_plan`), and a rule that spelled the type inline
/// would be a rule that drifts from the writer.
pub const BRIEF_ACTION_TYPE: &str = "brief";

/// Action types an **agent** authors as a post. The role split that renders a
/// transcript for a model is participant-derived (only the responder's own
/// posts are `assistant`), but the display-side view has no responder to ask,
/// so it reads this.
pub const AGENT_POST_ACTION_TYPES: [&str; 2] = ["inference", BRIEF_ACTION_TYPE];

/// Whether an action type is a post an agent authored (see
/// [`AGENT_POST_ACTION_TYPES`]).
pub fn is_agent_post_action_type(action_type: &str) -> bool {
    AGENT_POST_ACTION_TYPES.contains(&action_type)
}

/// Whether an action type renders as a post.
pub fn is_post_action_type(action_type: &str) -> bool {
    POST_ACTION_TYPES.contains(&action_type)
}

/// The only block type a reference may quote, and the only one any context
/// query sends upstream. A `thinking` block is a render-side disclosure and a
/// `tool_use` / `tool_result` block is trace plumbing; neither is transcript.
pub const QUOTABLE_BLOCK_TYPE: &str = "text";

/// One renderable post — an item's current-generation action with its
/// participant identity and derived generation number. Content blocks and
/// antecedent edges come back separately (see [`SpaceTreeData`]).
pub struct PostActionRow {
    pub action_id: String,
    pub item_id: String,
    pub participant_kind: String,
    /// The author as **this** space names them
    /// (`COALESCE(space_participant.override_label, participant.label)`) — the
    /// same effective label every other author-rendering read returns, so a
    /// post carries one byline whether it is read through the tree, the
    /// transcript, the navigation tools or the trace rail. Deliberately not
    /// liveness-filtered: who wrote a post goes on being named after
    /// retirement.
    pub participant_label: String,
    pub action_type: String,
    pub model: Option<String>,
    pub credits_consumed: Option<i64>,
    /// Derived 0-based generation number of this (tip) action. The item's total
    /// generation count is `generation + 1`.
    pub generation: i64,
    pub created_at: i64,
}

/// One content block of a post, in `ordinal` order within its action.
pub struct PostBlockRow {
    pub id: String,
    pub action_id: String,
    pub ordinal: i64,
    pub block_type: String,
    pub text_content: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub data: Option<String>,
}

/// One antecedent edge of a post (its `relation` distinguishes the structural
/// `reply` parent from non-structural `reference` links).
pub struct AntecedentEdgeRow {
    pub action_id: String,
    /// The concrete generation this edge points at — causality, immutable.
    pub antecedent_action_id: String,
    /// The current tip of that generation's *item* — the intended logical
    /// target once edits/regenerations have moved the tip. Reply threading
    /// follows this; `None` only for malformed data (no tip resolvable).
    pub antecedent_current_action_id: Option<String>,
    pub ordinal: i64,
    pub relation: String,
    pub content_block_id: Option<String>,
    pub range_start: Option<i64>,
    pub range_end: Option<i64>,
    pub annotation: Option<String>,
    /// `text_content` of the referenced content block (reference edges with a
    /// `content_block_id` only) — the raw material for snippet resolution.
    pub block_text: Option<String>,
    /// Action type of the antecedent, and block type of the quoted block —
    /// the quotable rule (a post's `text` block and nothing else), applied by
    /// `build_post_tree` when it resolves the snippet. The *edge* still
    /// renders (its existence is public); only the passage is withheld.
    pub antecedent_action_type: String,
    pub block_type: Option<String>,
    /// The quoted post's author, as **its own** space names it
    /// (`COALESCE(space_participant.override_label, participant.label)` joined
    /// on `ant.space_id`). Carried because a reference is the cross-space
    /// mechanism: the reading space cannot name an author it has never met,
    /// and a sibling branch's quote is a model's *only* view of that passage.
    pub antecedent_author_label: String,
    /// The quoted post's author's participant **kind**, from the same join.
    /// The label is not renderable on its own — see
    /// [`crate::PostReference::antecedent_author_kind`].
    pub antecedent_author_kind: String,
}

/// The raw materials for one space's threaded-post render.
pub struct SpaceTreeData {
    pub actions: Vec<PostActionRow>,
    pub blocks: Vec<PostBlockRow>,
    pub edges: Vec<AntecedentEdgeRow>,
}

/// Fetch the current-generation post actions of a space along with their
/// content blocks and antecedent edges. Filters to terminal statuses, current
/// generations (item tips via `is_current`), and post-bearing action types.
pub async fn get_space_tree_data(
    conn: &Connection,
    space_id: &str,
) -> Result<SpaceTreeData, AppError> {
    // Tip post actions, ordered by their **item's origin** (first generation),
    // not the tip's own created_at — an edited post is a *newer action* of the
    // same item and must stay exactly where the item has always been, both
    // here (render order → sibling/root order in the tree) and in the
    // upstream-context view (`get_space_actions_for_context`).
    let action_sql = format!(
        "SELECT ar.action_id, ar.item_id, p.kind, \
                COALESCE(sp.override_label, p.label), ar.action_type, \
                ar.model, ar.credits_consumed, ar.generation, ar.created_at \
         FROM action_resolved ar \
         JOIN participant p ON p.id = ar.participant_id \
         LEFT JOIN space_participant sp \
           ON sp.space_id = ar.space_id AND sp.participant_id = ar.participant_id \
         JOIN (SELECT space_id, item_id, \
                      MIN(created_at) AS born_at, MIN(id) AS first_action_id \
               FROM action GROUP BY space_id, item_id) origin \
           ON origin.space_id = ar.space_id AND origin.item_id = ar.item_id \
         WHERE ar.space_id = ?1 \
           AND ar.is_current = 1 \
           AND ar.status IN ('complete', 'cancelled') \
           AND ar.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY origin.born_at ASC, origin.first_action_id ASC"
    );
    let mut stmt = conn.prepare(&action_sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut actions = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        actions.push(PostActionRow {
            action_id: row.get::<String>(0).map_err(AppError::db)?,
            item_id: row.get::<String>(1).map_err(AppError::db)?,
            participant_kind: row.get::<String>(2).map_err(AppError::db)?,
            participant_label: row.get::<String>(3).map_err(AppError::db)?,
            action_type: row.get::<String>(4).map_err(AppError::db)?,
            model: row.get::<Option<String>>(5).map_err(AppError::db)?,
            credits_consumed: row.get::<Option<i64>>(6).map_err(AppError::db)?,
            generation: row.get::<i64>(7).map_err(AppError::db)?,
            created_at: row.get::<i64>(8).map_err(AppError::db)?,
        });
    }

    // Content blocks of those tip actions, in (action, ordinal) order.
    let block_sql = format!(
        "SELECT cb.id, cb.action_id, cb.ordinal, cb.block_type, cb.text_content, \
                cb.tool_name, cb.tool_call_id, cb.data \
         FROM content_block cb \
         JOIN action_resolved ar ON ar.action_id = cb.action_id \
         WHERE ar.space_id = ?1 \
           AND ar.is_current = 1 \
           AND ar.status IN ('complete', 'cancelled') \
           AND ar.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY cb.action_id ASC, cb.ordinal ASC"
    );
    let mut stmt = conn.prepare(&block_sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut blocks = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        blocks.push(PostBlockRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            action_id: row.get::<String>(1).map_err(AppError::db)?,
            ordinal: row.get::<i64>(2).map_err(AppError::db)?,
            block_type: row.get::<String>(3).map_err(AppError::db)?,
            text_content: row.get::<Option<String>>(4).map_err(AppError::db)?,
            tool_name: row.get::<Option<String>>(5).map_err(AppError::db)?,
            tool_call_id: row.get::<Option<String>>(6).map_err(AppError::db)?,
            data: row.get::<Option<String>>(7).map_err(AppError::db)?,
        });
    }

    // Antecedent edges of those tip actions (reply + reference). Each edge
    // also carries the **current tip of the antecedent's item**: the raw
    // antecedent action id records causality (which concrete generation was
    // replied to), but the *intended* logical flow follows item identity — a
    // reply to a since-edited post threads under the edit, not under a
    // dangling superseded generation (`build_post_tree` uses the resolved id
    // for reply threading; references keep the raw causal id).
    let edge_sql = format!(
        "SELECT aa.action_id, aa.antecedent_action_id, aa.ordinal, aa.relation, \
                aa.range_start, aa.range_end, aa.annotation, \
                ic.current_action_id, aa.content_block_id, qcb.text_content, \
                ant.action_type, qcb.block_type, \
                COALESCE(asp.override_label, ap.label), ap.kind \
         FROM action_antecedent aa \
         JOIN action_resolved ar ON ar.action_id = aa.action_id \
         JOIN action ant ON ant.id = aa.antecedent_action_id \
         JOIN participant ap ON ap.id = ant.participant_id \
         LEFT JOIN space_participant asp \
           ON asp.space_id = ant.space_id AND asp.participant_id = ant.participant_id \
         LEFT JOIN item_current ic \
           ON ic.space_id = ant.space_id AND ic.item_id = ant.item_id \
         LEFT JOIN content_block qcb ON qcb.id = aa.content_block_id \
         WHERE ar.space_id = ?1 \
           AND ar.is_current = 1 \
           AND ar.status IN ('complete', 'cancelled') \
           AND ar.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY aa.action_id ASC, aa.ordinal ASC"
    );
    let mut stmt = conn.prepare(&edge_sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        edges.push(AntecedentEdgeRow {
            action_id: row.get::<String>(0).map_err(AppError::db)?,
            antecedent_action_id: row.get::<String>(1).map_err(AppError::db)?,
            ordinal: row.get::<i64>(2).map_err(AppError::db)?,
            relation: row.get::<String>(3).map_err(AppError::db)?,
            range_start: row.get::<Option<i64>>(4).map_err(AppError::db)?,
            range_end: row.get::<Option<i64>>(5).map_err(AppError::db)?,
            annotation: row.get::<Option<String>>(6).map_err(AppError::db)?,
            antecedent_current_action_id: row.get::<Option<String>>(7).map_err(AppError::db)?,
            content_block_id: row.get::<Option<String>>(8).map_err(AppError::db)?,
            block_text: row.get::<Option<String>>(9).map_err(AppError::db)?,
            antecedent_action_type: row.get::<String>(10).map_err(AppError::db)?,
            block_type: row.get::<Option<String>>(11).map_err(AppError::db)?,
            antecedent_author_label: row.get::<String>(12).map_err(AppError::db)?,
            antecedent_author_kind: row.get::<String>(13).map_err(AppError::db)?,
        });
    }

    Ok(SpaceTreeData {
        actions,
        blocks,
        edges,
    })
}

/// The high-water mark of the `action` table: **no row above this existed when
/// this was read.**
///
/// The boundary a walk divides "already in front of me" from "arrived while I
/// was working" with, and it is `rowid` rather than a clock because a clock
/// cannot answer the question. Every writer here samples `now_ms()` *before*
/// its transaction — `Inner::post` takes it above several awaited validations —
/// so a row's `created_at` can predate a reader that its commit postdates.
/// Anything deciding "did this arrive after me" from that timestamp therefore
/// misses exactly the writes that raced it. `rowid` is assigned by the store at
/// insert, and writers here are serialized (`BEGIN IMMEDIATE`, one writer at a
/// time), so it *is* commit order: a row inserted after this read is above this
/// number, and a row at or below it was already visible.
///
/// Table-wide rather than per space, which is both cheaper (`MAX(rowid)` is a
/// seek to the last row, not a scan of a space's actions) and a stronger claim
/// — a later insert anywhere is above it.
pub async fn action_watermark(conn: &Connection) -> Result<i64, AppError> {
    let mut stmt = conn
        .prepare("SELECT COALESCE(MAX(rowid), 0) FROM action")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(0),
        Some(row) => row.get::<i64>(0).map_err(AppError::db),
    }
}

/// Every **post** in `space_id` written after `since_row` (see
/// [`action_watermark`]), oldest first.
///
/// What a turn driver asks at the end of a walk to find the posts that arrived
/// while it was walking: subtract the ones it served itself and what is left is
/// somebody else's, still unanswered. Ordered by `rowid`, which is the order
/// they were *committed* — the order a reader watching the room saw them
/// appear, and the only order that agrees with the boundary.
///
/// **Current generations only.** An arrival that is edited before this read
/// runs is one post, not two: without the join both generations land on the
/// walk's frontier, and the walk plans and bills a turn against wording its
/// author already replaced — then quotes the retraction into the report
/// beside the edit. The superseding row's own `rowid` is inside any boundary
/// its predecessor's is, so restricting to the tip drops nothing that is
/// still anyone's word. The claim is the refill's, exactly: a post the walk
/// already served and recorded as a **leaf**, then edited, keeps its
/// superseded id among the leaves while the edit arrives here — the report
/// quotes both wordings, which is the honest record of a room whose words
/// changed under the walk, and settlement holds because the current tail is
/// among the quoted ids.
pub async fn posts_in_space_since(
    conn: &Connection,
    space_id: &str,
    since_row: i64,
) -> Result<Vec<String>, AppError> {
    let sql = format!(
        "SELECT a.id FROM action a \
         JOIN item_current ic ON ic.current_action_id = a.id \
         WHERE a.space_id = ?1 AND a.rowid > ?2 \
           AND a.status IN ('complete', 'cancelled') \
           AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY a.rowid ASC"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string()), Value::Integer(since_row)])
        .await
        .map_err(AppError::db)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        out.push(row.get::<String>(0).map_err(AppError::db)?);
    }
    Ok(out)
}

pub async fn last_action_in_space(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<String>, AppError> {
    // The tail is the last **post**, not the last row. Trace actions
    // (`tool_call` / `tool_result`) are written into a space by a tool-calling
    // turn and can be the newest rows in it — a turn that ended at the round
    // cap leaves a `tool_result` as the literal last action. Replying to one
    // would give the next post a parent that no rendered view contains,
    // orphaning it into a second thread root. Restricting to post types keeps
    // "the space's tail" meaning what every reader means by it.
    //
    // **Last means last-committed** ([`action_watermark`]): every writer samples
    // `now_ms()` above its own transaction, so a post stamped early and
    // committed late is not the newest by `created_at` and *is* the room's last
    // word — and picking the other one tells the delegation driver its last
    // word has already been reported, which retires a room holding a post
    // nobody answered. `rowid` also totally orders two posts written in the
    // same millisecond, which `created_at` never did.
    //
    // **And last means the last word the transcript shows** (`item_current`,
    // like every read the delegation lifecycle stands on): a failed
    // regeneration inside the room leaves a current hidden tip, and without
    // the join the *superseded* generation — still the highest matching
    // `rowid` — would come back as the tail, so the driver would walk from,
    // reply beneath, and quote into its report a wording the room no longer
    // shows. With the join, that item contributes nothing and the tail falls
    // back to the newest visible post, which is what a reader would call the
    // room's last word.
    let sql = format!(
        "SELECT a.id FROM action a \
         JOIN item_current ic ON ic.current_action_id = a.id \
         WHERE a.space_id = ?1 AND a.status IN ('complete', 'cancelled') \
           AND a.action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY a.rowid DESC LIMIT 1"
    );
    let mut stmt = conn.prepare(&sql).await.map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// Returns `(item_id, space_id)` for an action, or `None` if it doesn't exist.
/// Used by the generation paths (edit/regenerate) to locate the item an action
/// belongs to.
pub async fn action_item_and_space(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<(String, String)>, AppError> {
    let mut stmt = conn
        .prepare("SELECT item_id, space_id FROM action WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some((
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<String>(1).map_err(AppError::db)?,
        ))),
    }
}

/// Returns the current tip (the action no other action supersedes) of an item.
pub async fn current_tip_of_item(
    conn: &Connection,
    space_id: &str,
    item_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT current_action_id FROM item_current \
             WHERE space_id = ?1 AND item_id = ?2",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([
            Value::Text(space_id.to_string()),
            Value::Text(item_id.to_string()),
        ])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// Returns the `reply`-relation antecedent of an action (its structural thread
/// parent), if any. A new generation replicates this so it keeps the item's
/// place in the thread.
pub async fn reply_antecedent(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT antecedent_action_id FROM action_antecedent \
             WHERE action_id = ?1 AND relation = 'reply' LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// The acting participant of an action: `(participant_id, participant_scope,
/// kind)`. `None` if the action doesn't exist. Used by `plan_notifications`
/// to identify a post's author (to exclude it from the notify set and to
/// resolve the `human`-policy predicate).
/// An action's `action_type`. Used by notification planning to refuse to
/// cascade off a non-post action (a `decision` / `tool_call` / `tool_result`
/// is not something anyone replies to).
pub async fn action_type(conn: &Connection, action_id: &str) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare("SELECT action_type FROM action WHERE id = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

pub async fn action_author(
    conn: &Connection,
    action_id: &str,
) -> Result<Option<(String, String, String)>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.participant_id, a.participant_scope, p.kind \
             FROM action a JOIN participant p ON p.id = a.participant_id \
             WHERE a.id = ?1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some((
            row.get::<String>(0).map_err(AppError::db)?,
            row.get::<String>(1).map_err(AppError::db)?,
            row.get::<String>(2).map_err(AppError::db)?,
        ))),
    }
}

/// The participant *kind* of an action (`human`/`agent`/`tool`/`system`), or
/// `None` if the action doesn't exist.
async fn action_kind(conn: &Connection, action_id: &str) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT p.kind FROM action a \
             JOIN participant p ON p.id = a.participant_id WHERE a.id = ?1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(action_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
}

/// The **cascade depth** of a post: the number of consecutive agent-authored
/// posts in its reply ancestry, counting from the post itself back to (but not
/// including) the most recent human-authored post. A human-authored post has
/// depth 0.
///
/// This is the data-derived cascade guard — no schema state. Each hop resolves
/// to the antecedent item's **current tip** (matching `get_upstream_context`),
/// and the walk is **branch-scoped** (only the target's own reply ancestry, so
/// sibling branches carry independent depths). Authorship is stable across an
/// item's generations (a `user_input` item is always human; an `inference` or
/// `brief` item is always its agent), so an edit or regeneration never changes
/// the count. Note what that means in a sub-space, and is meant to: its brief
/// is agent-authored, so the room opens at depth 1 and the cascade it
/// inherited from its parent bounds the whole delegation from there.
/// Cycle-guarded like `get_upstream_context` (item-tip re-rooting can create
/// logical cycles even though raw edges only point backward).
pub async fn agent_cascade_depth(conn: &Connection, action_id: &str) -> Result<i64, AppError> {
    let mut depth: i64 = 0;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Start at the post's own item tip (a reply to a since-edited post answers
    // the latest generation; authorship is unchanged either way).
    let mut cursor = item_tip_of_action(conn, action_id)
        .await?
        .unwrap_or_else(|| action_id.to_string());
    loop {
        if !seen.insert(cursor.clone()) {
            break;
        }
        match action_kind(conn, &cursor).await? {
            Some(kind) if kind == "agent" => depth += 1,
            // A human (or non-agent) post — or a vanished action — ends the run.
            _ => break,
        }
        match reply_antecedent_tip(conn, &cursor).await? {
            Some(parent) => cursor = parent,
            None => break,
        }
    }
    Ok(depth)
}

/// Returns all action IDs in a space with terminal status, ordered by created_at.
pub async fn space_action_ids(conn: &Connection, space_id: &str) -> Result<Vec<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id FROM action \
             WHERE space_id = ?1 AND status IN ('complete', 'cancelled') \
             ORDER BY created_at ASC",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        ids.push(row.get::<String>(0).map_err(AppError::db)?);
    }
    Ok(ids)
}

/// Archive a space **and every live delegation beneath it**, in one
/// transaction, answering with every space id that was archived (the named one
/// first).
///
/// **A delegation exists to serve the conversation it was opened from**, so
/// closing that conversation closes them: a sub-space under an archived room is
/// a room whose helpers go on being paid to answer a question nobody will ever
/// be told the answer to. Its report is a turn in the archived parent, which
/// that parent refuses at the same gate every other turn meets — so the
/// delegation stays outstanding forever, retrying against its meter, holding a
/// live-room slot of an owner who is not being retired, and waiting on an
/// anchor no post will ever answer. There is no un-archive door anywhere, so
/// none of that is a state anything recovers from.
///
/// **Recursive down `parent_space_id`, and inside the transaction.** The room
/// being closed may hold delegations of its own at any depth, and a nested one
/// may be owned by a *different* agent: that agent is not being retired, but
/// the purpose of its room is gone with the conversation above it, which is the
/// thing that decides this. Doing it in the archival's own transaction is what
/// makes "a live room under a closed one" unrepresentable rather than a state
/// somebody has to notice.
pub(crate) async fn archive_space_tx(
    conn: &Connection,
    space_id: &str,
    archived_at: i64,
) -> Result<Vec<String>, AppError> {
    begin_write(conn).await?;
    match archive_space_tx_body(conn, space_id, archived_at).await {
        Ok(archived) => {
            conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
            Ok(archived)
        }
        Err(e) => {
            // Best-effort rollback; propagate the original error regardless.
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

async fn archive_space_tx_body(
    conn: &Connection,
    space_id: &str,
    archived_at: i64,
) -> Result<Vec<String>, AppError> {
    let changed = conn
        .execute(
            "UPDATE space SET archived_at = ?2, \
             touched_at = COALESCE(touched_at, ?2) \
             WHERE id = ?1 AND archived_at IS NULL",
            (
                Value::Text(space_id.to_string()),
                Value::Integer(archived_at),
            ),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to archive space: {e}"),
        })?;
    if changed == 0 {
        // Nothing was closed here — an unknown space, or one already archived —
        // so there is nothing new beneath it to close either.
        return Ok(Vec::new());
    }
    let mut archived = vec![space_id.to_string()];
    archived.extend(archive_rooms_under_a_closed_one(conn, archived_at).await?);
    Ok(archived)
}

/// Archive every live sub-space whose parent is archived, repeatedly, until
/// none is left — the descent [`archive_space_tx`] and [`retire_participant_tx`]
/// both need, stated as the invariant it restores rather than as a walk.
///
/// **The invariant is "no live delegation under a closed room"**, and asking it
/// of the rows rather than carrying a list of roots is what makes one statement
/// serve both doors: an archival names one space, a retirement names an agent's
/// whole set, and neither has to tell this how deep to go. Each round archives
/// the children of everything closed so far, so a chain of any depth is closed
/// in as many rounds as it is deep; a round that archives nothing ends it, which
/// also makes the loop cycle-safe (the column is a self-referencing FK with no
/// cycle constraint, and a space already archived never matches again).
async fn archive_rooms_under_a_closed_one(
    conn: &Connection,
    archived_at: i64,
) -> Result<Vec<String>, AppError> {
    const LIVE_UNDER_A_CLOSED_ONE: &str = "archived_at IS NULL AND parent_space_id IN \
         (SELECT id FROM space WHERE archived_at IS NOT NULL)";
    let mut archived = Vec::new();
    loop {
        let mut stmt = conn
            .prepare(&format!(
                "SELECT id FROM space WHERE {LIVE_UNDER_A_CLOSED_ONE}"
            ))
            .await
            .map_err(AppError::db)?;
        let mut rows = stmt.query(()).await.map_err(AppError::db)?;
        let mut round = Vec::new();
        while let Some(row) = rows.next().await.map_err(AppError::db)? {
            round.push(row.get::<String>(0).map_err(AppError::db)?);
        }
        if round.is_empty() {
            return Ok(archived);
        }
        conn.execute(
            &format!(
                "UPDATE space SET archived_at = ?1, touched_at = COALESCE(touched_at, ?1) \
                 WHERE {LIVE_UNDER_A_CLOSED_ONE}"
            ),
            (Value::Integer(archived_at),),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to archive the delegations under a closed conversation: {e}"),
        })?;
        archived.extend(round);
    }
}

/// Set a space's title, answering **whether a row took it**.
///
/// The answer is the existence check: a caller that read the space a moment ago
/// and then wrote by id alone would report success for a title that landed
/// nowhere, because a space can stop existing between the two (an untouched one
/// is disposed of when its last window closes). Deciding at the write is the
/// module's rule, and here it costs one `bool`.
pub async fn update_space_title(
    conn: &Connection,
    space_id: &str,
    title: &str,
    now: i64,
) -> Result<bool, AppError> {
    let changed = conn
        .execute(
            "UPDATE space SET title = ?2, touched_at = COALESCE(touched_at, ?3) \
         WHERE id = ?1",
            (
                Value::Text(space_id.to_string()),
                Value::Text(title.to_string()),
                Value::Integer(now),
            ),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to update space title: {e}"),
        })?;
    Ok(changed > 0)
}

// ---------------------------------------------------------------------------
// The Record — read-only queries over the local trail (attestations,
// requests, spend trail). Pure SELECTs over existing tables/views; these
// exist so the GUI's Record window (and future CLI inspection commands) can
// page through the raw local history without loading whole tables.
// ---------------------------------------------------------------------------

/// One row of the attestation listing. `doc_bytes` is the stored document's
/// size; `connection_count` is how many recorded connections presented this
/// attestation.
pub struct AttestationListRow {
    pub hash: String,
    pub pcr_digest: Option<String>,
    pub created_at: i64,
    pub doc_bytes: i64,
    pub connection_count: i64,
}

pub async fn list_attestations(
    conn: &Connection,
    limit: i64,
    offset: i64,
) -> Result<Vec<AttestationListRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.hash, a.pcr_digest, a.created_at, length(a.doc), COUNT(c.id) \
             FROM attestation a \
             LEFT JOIN connection c ON c.attestation_hash = a.hash \
             GROUP BY a.hash, a.pcr_digest, a.created_at \
             ORDER BY a.created_at DESC, a.hash \
             LIMIT ?1 OFFSET ?2",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query([limit, offset]).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(AttestationListRow {
            hash: row.get::<String>(0).map_err(AppError::db)?,
            pcr_digest: row.get::<Option<String>>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            doc_bytes: row.get::<i64>(3).map_err(AppError::db)?,
            connection_count: row.get::<i64>(4).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

/// The full stored attestation document for the detail view.
pub struct AttestationDocRow {
    pub hash: String,
    pub pcr_digest: Option<String>,
    pub created_at: i64,
    pub doc: Vec<u8>,
}

pub async fn get_attestation(
    conn: &Connection,
    hash: &str,
) -> Result<Option<AttestationDocRow>, AppError> {
    let mut stmt = conn
        .prepare("SELECT hash, pcr_digest, created_at, doc FROM attestation WHERE hash = ?1")
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(hash.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(AttestationDocRow {
            hash: row.get::<String>(0).map_err(AppError::db)?,
            pcr_digest: row.get::<Option<String>>(1).map_err(AppError::db)?,
            created_at: row.get::<i64>(2).map_err(AppError::db)?,
            doc: row.get::<Vec<u8>>(3).map_err(AppError::db)?,
        })),
    }
}

/// One row of the request listing — the summary line, without the (possibly
/// large) header/body payloads. Joined against `connection` for transport
/// metadata and the attestation link.
pub struct RequestListRow {
    pub id: String,
    pub method: String,
    pub path: String,
    pub response_status: Option<i64>,
    pub duration_ms: Option<i64>,
    pub request_at: i64,
    pub error: Option<String>,
    pub attempt_number: i64,
    pub credential_nonce: Option<String>,
    pub transport: Option<String>,
    pub base_url: Option<String>,
    pub attestation_hash: Option<String>,
}

pub async fn list_requests(
    conn: &Connection,
    limit: i64,
    offset: i64,
) -> Result<Vec<RequestListRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT r.id, r.method, r.path, r.response_status, r.duration_ms, \
                    r.request_at, r.error, r.attempt_number, r.credential_nonce, \
                    c.transport, c.base_url, c.attestation_hash \
             FROM request r \
             LEFT JOIN connection c ON c.id = r.connection_id \
             ORDER BY r.request_at DESC, r.id DESC \
             LIMIT ?1 OFFSET ?2",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query([limit, offset]).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(RequestListRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            method: row.get::<String>(1).map_err(AppError::db)?,
            path: row.get::<String>(2).map_err(AppError::db)?,
            response_status: row.get::<Option<i64>>(3).map_err(AppError::db)?,
            duration_ms: row.get::<Option<i64>>(4).map_err(AppError::db)?,
            request_at: row.get::<i64>(5).map_err(AppError::db)?,
            error: row.get::<Option<String>>(6).map_err(AppError::db)?,
            attempt_number: row.get::<i64>(7).map_err(AppError::db)?,
            credential_nonce: row.get::<Option<String>>(8).map_err(AppError::db)?,
            transport: row.get::<Option<String>>(9).map_err(AppError::db)?,
            base_url: row.get::<Option<String>>(10).map_err(AppError::db)?,
            attestation_hash: row.get::<Option<String>>(11).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

/// The full stored request/response pair, raw bodies included.
pub struct RequestDetailRow {
    pub id: String,
    pub method: String,
    pub path: String,
    pub request_headers: Option<String>,
    pub request_body: Option<Vec<u8>>,
    pub response_status: Option<i64>,
    pub response_headers: Option<String>,
    pub response_body: Option<Vec<u8>>,
    pub request_at: i64,
    pub response_at: Option<i64>,
    pub duration_ms: Option<i64>,
    pub error: Option<String>,
    pub retry_of_id: Option<String>,
    pub attempt_number: i64,
    pub credential_nonce: Option<String>,
    pub action_id: Option<String>,
    pub transport: Option<String>,
    pub base_url: Option<String>,
    pub attestation_hash: Option<String>,
    pub space_id: Option<String>,
    pub space_title: Option<String>,
    /// The configured backend this request was routed through, if recorded.
    pub backend_id: Option<String>,
    /// The backend's display name at read time (soft-removed rows keep it).
    pub backend_display_name: Option<String>,
}

pub async fn get_request(
    conn: &Connection,
    id: &str,
) -> Result<Option<RequestDetailRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT r.id, r.method, r.path, r.request_headers, r.request_body, \
                    r.response_status, r.response_headers, r.response_body, \
                    r.request_at, r.response_at, r.duration_ms, r.error, \
                    r.retry_of_id, r.attempt_number, r.credential_nonce, r.action_id, \
                    c.transport, c.base_url, c.attestation_hash, \
                    a.space_id, s.title, r.backend_id, b.display_name \
             FROM request r \
             LEFT JOIN connection c ON c.id = r.connection_id \
             LEFT JOIN action a ON a.id = r.action_id \
             LEFT JOIN space s ON s.id = a.space_id \
             LEFT JOIN backend b ON b.id = r.backend_id \
             WHERE r.id = ?1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(RequestDetailRow {
            id: row.get::<String>(0).map_err(AppError::db)?,
            method: row.get::<String>(1).map_err(AppError::db)?,
            path: row.get::<String>(2).map_err(AppError::db)?,
            request_headers: row.get::<Option<String>>(3).map_err(AppError::db)?,
            request_body: row.get::<Option<Vec<u8>>>(4).map_err(AppError::db)?,
            response_status: row.get::<Option<i64>>(5).map_err(AppError::db)?,
            response_headers: row.get::<Option<String>>(6).map_err(AppError::db)?,
            response_body: row.get::<Option<Vec<u8>>>(7).map_err(AppError::db)?,
            request_at: row.get::<i64>(8).map_err(AppError::db)?,
            response_at: row.get::<Option<i64>>(9).map_err(AppError::db)?,
            duration_ms: row.get::<Option<i64>>(10).map_err(AppError::db)?,
            error: row.get::<Option<String>>(11).map_err(AppError::db)?,
            retry_of_id: row.get::<Option<String>>(12).map_err(AppError::db)?,
            attempt_number: row.get::<i64>(13).map_err(AppError::db)?,
            credential_nonce: row.get::<Option<String>>(14).map_err(AppError::db)?,
            action_id: row.get::<Option<String>>(15).map_err(AppError::db)?,
            transport: row.get::<Option<String>>(16).map_err(AppError::db)?,
            base_url: row.get::<Option<String>>(17).map_err(AppError::db)?,
            attestation_hash: row.get::<Option<String>>(18).map_err(AppError::db)?,
            space_id: row.get::<Option<String>>(19).map_err(AppError::db)?,
            space_title: row.get::<Option<String>>(20).map_err(AppError::db)?,
            backend_id: row.get::<Option<String>>(21).map_err(AppError::db)?,
            backend_display_name: row.get::<Option<String>>(22).map_err(AppError::db)?,
        })),
    }
}

/// One row of the `spend_trail` view: credential → request → action → space.
pub struct SpendTrailRow {
    pub credential_nonce: String,
    pub spend_amount: Option<i64>,
    pub credential_state: String,
    pub request_id: String,
    pub method: String,
    pub path: String,
    pub request_at: i64,
    pub duration_ms: Option<i64>,
    pub attempt_number: i64,
    pub action_id: Option<String>,
    pub action_type: Option<String>,
    pub model: Option<String>,
    pub credits_consumed: Option<i64>,
    pub intent: Option<String>,
    pub space_id: Option<String>,
    pub space_title: Option<String>,
    pub linkability: Option<String>,
}

pub async fn list_spend_trail(
    conn: &Connection,
    limit: i64,
    offset: i64,
) -> Result<Vec<SpendTrailRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT credential_nonce, spend_amount, credential_state, request_id, \
                    method, path, request_at, duration_ms, attempt_number, \
                    action_id, action_type, model, credits_consumed, intent, \
                    space_id, space_title, linkability \
             FROM spend_trail \
             ORDER BY request_at DESC, request_id DESC \
             LIMIT ?1 OFFSET ?2",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query([limit, offset]).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(SpendTrailRow {
            credential_nonce: row.get::<String>(0).map_err(AppError::db)?,
            spend_amount: row.get::<Option<i64>>(1).map_err(AppError::db)?,
            credential_state: row.get::<String>(2).map_err(AppError::db)?,
            request_id: row.get::<String>(3).map_err(AppError::db)?,
            method: row.get::<String>(4).map_err(AppError::db)?,
            path: row.get::<String>(5).map_err(AppError::db)?,
            request_at: row.get::<i64>(6).map_err(AppError::db)?,
            duration_ms: row.get::<Option<i64>>(7).map_err(AppError::db)?,
            attempt_number: row.get::<i64>(8).map_err(AppError::db)?,
            action_id: row.get::<Option<String>>(9).map_err(AppError::db)?,
            action_type: row.get::<Option<String>>(10).map_err(AppError::db)?,
            model: row.get::<Option<String>>(11).map_err(AppError::db)?,
            credits_consumed: row.get::<Option<i64>>(12).map_err(AppError::db)?,
            intent: row.get::<Option<String>>(13).map_err(AppError::db)?,
            space_id: row.get::<Option<String>>(14).map_err(AppError::db)?,
            space_title: row.get::<Option<String>>(15).map_err(AppError::db)?,
            linkability: row.get::<Option<String>>(16).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

/// One row per credential, with the lifecycle state computed by the
/// `credential_lifecycle` view (`active` / `spending` / `spent` / `expired`).
pub struct CredentialLifecycleRow {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub created_at: i64,
    pub state: String,
    pub spend_amount: Option<i64>,
}

pub async fn list_credential_lifecycle(
    conn: &Connection,
) -> Result<Vec<CredentialLifecycleRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT nonce, credits, generation, created_at, state, spend_amount \
             FROM credential_lifecycle \
             ORDER BY created_at DESC, nonce",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt.query(()).await.map_err(AppError::db)?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await.map_err(AppError::db)? {
        results.push(CredentialLifecycleRow {
            nonce: row.get::<String>(0).map_err(AppError::db)?,
            credits: row.get::<i64>(1).map_err(AppError::db)?,
            generation: row.get::<i64>(2).map_err(AppError::db)?,
            created_at: row.get::<i64>(3).map_err(AppError::db)?,
            state: row.get::<String>(4).map_err(AppError::db)?,
            spend_amount: row.get::<Option<i64>>(5).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference edge quoting a post's text — what the two edge-level
    /// predicates are asked about. *Whether the quoted post can be named by
    /// handle* is not among them: that is the turn snapshot's answer alone
    /// (`ThreadSnapshot::node_for_action`), so no second definition of it can
    /// live here and drift.
    fn quoting_edge() -> ReferenceEdgeRow {
        ReferenceEdgeRow {
            ordinal: 1,
            antecedent_action_id: "a1".into(),
            content_block_id: Some("cb1".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            block_text: Some("text".into()),
            antecedent_action_type: "user_input".into(),
            block_type: Some("text".into()),
            antecedent_author_label: "Ada".into(),
        }
    }

    /// The quotable rule and the range test are separate questions and the
    /// rendering needs both: a `thinking` block belongs to a post, so the post
    /// stays readable while that passage is withheld; a range-less edge is a
    /// backlink rather than a quote whose range broke.
    #[test]
    fn quotability_and_having_a_range_are_separate_questions() {
        let quoting = quoting_edge();
        assert!(quoting.is_quotable());
        assert!(quoting.has_range());

        let thinking = ReferenceEdgeRow {
            block_type: Some("thinking".into()),
            ..quoting_edge()
        };
        assert!(!thinking.is_quotable(), "only a post's text may be quoted");
        assert!(thinking.has_range());

        for action_type in ["tool_call", "decision", "memory", "checkpoint"] {
            let non_post = ReferenceEdgeRow {
                antecedent_action_type: action_type.into(),
                ..quoting_edge()
            };
            assert!(!non_post.is_quotable(), "{action_type}");
        }

        let backlink = ReferenceEdgeRow {
            content_block_id: None,
            range_start: None,
            range_end: None,
            block_text: None,
            block_type: None,
            ..quoting_edge()
        };
        assert!(!backlink.has_range());
        assert!(!backlink.is_quotable(), "no block, nothing to quote");
    }

    async fn open_memory_fresh() -> Database {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        // Mirror production: initialize (and seed) under FK enforcement, so a
        // seed-order bug would surface here too.
        let conn = connect(&db).await.unwrap();
        initialize(&conn).await.unwrap();
        db
    }

    /// A test connection with FK enforcement on (the scope-owned schema's
    /// constraints only fire with `foreign_keys = ON`).
    async fn fk_conn(db: &Database) -> Connection {
        connect(db).await.unwrap()
    }

    async fn open_memory_migrated() -> Database {
        // Post fresh-start reset there are no incremental migrations —
        // `schema.sql` IS the baseline. Applying it directly must yield the
        // same schema objects `initialize()` produces (initialize also applies
        // schema.sql, then seeds rows; seeds add no schema objects), keeping
        // `migrations_match_schema` a meaningful — if now trivially satisfied —
        // structural check that the schema applies cleanly and consistently.
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();
        conn.execute_batch(SCHEMA).await.unwrap();
        set_user_version(&conn, LATEST_VERSION).await.unwrap();
        db
    }

    async fn list_objects(conn: &Connection) -> Vec<(String, String)> {
        let mut stmt = conn
            .prepare(
                "SELECT type, name FROM sqlite_master \
                 WHERE type IN ('table', 'view', 'index') \
                 AND name NOT LIKE 'sqlite_%' \
                 ORDER BY type, name",
            )
            .await
            .unwrap();
        let mut rows = stmt.query(()).await.unwrap();
        let mut objects = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            objects.push((row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()));
        }
        objects
    }

    async fn table_columns(
        conn: &Connection,
        table: &str,
    ) -> Vec<(String, String, bool, Option<String>, bool)> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info('{table}')"))
            .await
            .unwrap();
        let mut rows = stmt.query(()).await.unwrap();
        let mut cols = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            cols.push((
                row.get::<String>(1).unwrap(),
                row.get::<String>(2).unwrap(),
                row.get::<i64>(3).unwrap() != 0,
                row.get::<Option<String>>(4).unwrap(),
                row.get::<i64>(5).unwrap() != 0,
            ));
        }
        cols.sort_by(|a, b| a.0.cmp(&b.0));
        cols
    }

    async fn index_columns(conn: &Connection, index: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA index_info('{index}')"))
            .await
            .unwrap();
        let mut rows = stmt.query(()).await.unwrap();
        let mut cols = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            cols.push(row.get::<String>(2).unwrap());
        }
        cols.sort();
        cols
    }

    async fn view_sql(conn: &Connection, name: &str) -> String {
        let mut stmt = conn
            .prepare("SELECT sql FROM sqlite_master WHERE type='view' AND name=?1")
            .await
            .unwrap();
        let mut rows = stmt.query([name]).await.unwrap();
        rows.next()
            .await
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap()
    }

    /// Enforcement smoke test — the whole scope-owned design leans on turso
    /// actually enforcing FKs and CHECKs (its default is `foreign_keys = OFF`,
    /// which turns every `REFERENCES` into unenforced documentation). Proves,
    /// under this exact turso build: (a) single-column FK violations error;
    /// (b) composite/tuple FK violations error; (c) MATCH-SIMPLE NULL-skip (a
    /// NULL child column skips the composite FK — must NOT error); (d) CHECK
    /// violations error; (e) `ON UPDATE CASCADE` on a composite FK really
    /// cascades — a parent `(id, scope)` update rewrites the child echo
    /// columns (task 36's in-place promotion rests on exactly this).
    /// Also confirms the pragma is load-bearing (default-off lets a dangling
    /// FK through).
    #[tokio::test]
    async fn turso_enforcement_smoke() {
        let db = Builder::new_local(":memory:").build().await.unwrap();

        // --- default (foreign_keys OFF): a dangling single FK is accepted ----
        let off = db.connect().unwrap();
        off.execute_batch(
            "CREATE TABLE p0 (id TEXT PRIMARY KEY);
             CREATE TABLE c0 (id TEXT PRIMARY KEY, p TEXT REFERENCES p0(id));",
        )
        .await
        .unwrap();
        let dangling = off
            .execute("INSERT INTO c0 (id, p) VALUES ('x', 'nope')", ())
            .await;
        assert!(
            dangling.is_ok(),
            "turso default is foreign_keys=OFF — a dangling FK should be \
             accepted without the pragma (proving the pragma is load-bearing); \
             got {dangling:?}"
        );

        // --- foreign_keys ON: enforcement is real -----------------------------
        let conn = db.connect().unwrap();
        conn.execute("PRAGMA foreign_keys = ON", ()).await.unwrap();
        conn.execute_batch(
            "CREATE TABLE parent (id TEXT PRIMARY KEY, scope TEXT NOT NULL);
             CREATE UNIQUE INDEX ux_parent_id_scope ON parent(id, scope);
             CREATE TABLE child_single (
                 id TEXT PRIMARY KEY,
                 parent_id TEXT NOT NULL REFERENCES parent(id));
             CREATE TABLE child_comp (
                 id TEXT PRIMARY KEY,
                 parent_id TEXT,
                 parent_scope TEXT,
                 FOREIGN KEY (parent_id, parent_scope) REFERENCES parent(id, scope));
             CREATE TABLE checked (
                 id TEXT PRIMARY KEY,
                 v TEXT NOT NULL CHECK (v IN ('a', 'b')));",
        )
        .await
        .unwrap();
        conn.execute("INSERT INTO parent (id, scope) VALUES ('p1', 'global')", ())
            .await
            .unwrap();

        // (a) single-column FK
        assert!(
            conn.execute(
                "INSERT INTO child_single (id, parent_id) VALUES ('c1', 'nope')",
                ()
            )
            .await
            .is_err(),
            "(a) single-column FK violation must error under foreign_keys=ON"
        );
        conn.execute(
            "INSERT INTO child_single (id, parent_id) VALUES ('c2', 'p1')",
            (),
        )
        .await
        .expect("(a) a satisfied single FK must be accepted");

        // (b) composite FK — the id exists but the scope doesn't match the
        // (id, scope) tuple, and a wholly-absent id; both must error.
        assert!(
            conn.execute(
                "INSERT INTO child_comp (id, parent_id, parent_scope) VALUES ('cc1', 'p1', 'space')",
                (),
            )
            .await
            .is_err(),
            "(b) composite FK violation (scope mismatch) must error"
        );
        assert!(
            conn.execute(
                "INSERT INTO child_comp (id, parent_id, parent_scope) VALUES ('cc2', 'nope', 'global')",
                (),
            )
            .await
            .is_err(),
            "(b) composite FK violation (missing id) must error"
        );
        conn.execute(
            "INSERT INTO child_comp (id, parent_id, parent_scope) VALUES ('cc3', 'p1', 'global')",
            (),
        )
        .await
        .expect("(b) a satisfied composite FK must be accepted");

        // (c) MATCH SIMPLE NULL-skip: either NULL half skips the composite FK.
        conn.execute(
            "INSERT INTO child_comp (id, parent_id, parent_scope) VALUES ('cc4', NULL, 'global')",
            (),
        )
        .await
        .expect("(c) NULL parent_id must skip the composite FK (MATCH SIMPLE)");
        conn.execute(
            "INSERT INTO child_comp (id, parent_id, parent_scope) VALUES ('cc5', 'nope', NULL)",
            (),
        )
        .await
        .expect("(c) NULL parent_scope must skip the composite FK (MATCH SIMPLE)");

        // (d) CHECK
        assert!(
            conn.execute("INSERT INTO checked (id, v) VALUES ('k1', 'z')", ())
                .await
                .is_err(),
            "(d) CHECK violation must error"
        );
        conn.execute("INSERT INTO checked (id, v) VALUES ('k2', 'a')", ())
            .await
            .expect("(d) a satisfied CHECK must be accepted");

        // (e) ON UPDATE CASCADE on a composite FK. Task 36 promotes an agent
        // in place — the same participant row flips scope 'space' → 'global' —
        // and the pinned echo on every past `action` (and `memory_block`) must
        // follow declaratively rather than by an app-layer rewrite. Prove the
        // cascade fires on the *scope* half of the tuple, which is the half
        // promotion mutates.
        conn.execute_batch(
            "CREATE TABLE child_cascade (
                 id TEXT PRIMARY KEY,
                 parent_id TEXT NOT NULL,
                 parent_scope TEXT NOT NULL,
                 FOREIGN KEY (parent_id, parent_scope) REFERENCES parent (id, scope)
                     ON UPDATE CASCADE);",
        )
        .await
        .unwrap();
        conn.execute("INSERT INTO parent (id, scope) VALUES ('p2', 'space')", ())
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO child_cascade (id, parent_id, parent_scope) VALUES ('e1', 'p2', 'space')",
            (),
        )
        .await
        .unwrap();
        // A second child on a DIFFERENT parent, to prove the cascade is scoped
        // to the updated row rather than blanket-rewriting the column.
        conn.execute(
            "INSERT INTO child_cascade (id, parent_id, parent_scope) VALUES ('e2', 'p1', 'global')",
            (),
        )
        .await
        .unwrap();
        conn.execute("UPDATE parent SET scope = 'global' WHERE id = 'p2'", ())
            .await
            .expect("(e) the parent scope update itself must succeed");

        let mut stmt = conn
            .prepare("SELECT id, parent_scope FROM child_cascade ORDER BY id")
            .await
            .unwrap();
        let mut rows = stmt.query(()).await.unwrap();
        let mut seen: Vec<(String, String)> = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            seen.push((row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()));
        }
        assert_eq!(
            seen,
            vec![
                ("e1".to_string(), "global".to_string()),
                ("e2".to_string(), "global".to_string()),
            ],
            "(e) ON UPDATE CASCADE must rewrite the echo of the updated parent \
             ('e1' space → global) and leave every other child alone"
        );
    }

    /// The idiom `insert_action` / `insert_memory_block` rely on to make a
    /// stale scope echo unrepresentable: a scalar subquery inside `VALUES`,
    /// reading the parent's *current* scope in the same statement as the
    /// insert. Pins two properties of this turso build — the subquery is
    /// supported and yields the live value, and an unknown parent yields NULL
    /// into a `NOT NULL` column, i.e. a loud error rather than a NULL echo
    /// that MATCH SIMPLE would silently skip.
    #[tokio::test]
    async fn a_scope_echo_can_be_derived_inside_the_insert() {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();
        conn.execute("PRAGMA foreign_keys = ON", ()).await.unwrap();
        conn.execute_batch(
            "CREATE TABLE p (id TEXT PRIMARY KEY, scope TEXT NOT NULL);
             CREATE UNIQUE INDEX ux ON p(id, scope);
             CREATE TABLE c (id TEXT PRIMARY KEY, pid TEXT NOT NULL, pscope TEXT NOT NULL,
                 FOREIGN KEY (pid, pscope) REFERENCES p(id, scope) ON UPDATE CASCADE);",
        )
        .await
        .unwrap();
        conn.execute("INSERT INTO p VALUES ('p1', 'space')", ())
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO c (id, pid, pscope) VALUES (?1, ?2, (SELECT scope FROM p WHERE id = ?2))",
            (Value::Text("c1".into()), Value::Text("p1".into())),
        )
        .await
        .expect("subquery in VALUES must work");
        let mut stmt = conn
            .prepare("SELECT pscope FROM c WHERE id = 'c1'")
            .await
            .unwrap();
        let mut rows = stmt.query(()).await.unwrap();
        assert_eq!(
            rows.next()
                .await
                .unwrap()
                .unwrap()
                .get::<String>(0)
                .unwrap(),
            "space"
        );
        // A missing parent yields NULL -> NOT NULL violation, i.e. a loud error.
        let missing = conn.execute(
            "INSERT INTO c (id, pid, pscope) VALUES (?1, ?2, (SELECT scope FROM p WHERE id = ?2))",
            (Value::Text("c2".into()), Value::Text("nope".into())),
        ).await;
        assert!(
            missing.is_err(),
            "a missing participant must error, not insert NULL: {missing:?}"
        );
    }

    #[tokio::test]
    async fn fresh_install_creates_tables() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();

        assert_eq!(get_user_version(&conn).await.unwrap(), LATEST_VERSION);

        let objects = list_objects(&conn).await;
        let table_names: Vec<&str> = objects
            .iter()
            .filter(|(t, _)| t == "table")
            .map(|(_, n)| n.as_str())
            .collect();
        assert!(table_names.contains(&"issuer_key"));
        assert!(table_names.contains(&"pre_credential"));
        assert!(table_names.contains(&"credential"));
        assert!(table_names.contains(&"provider"));
        assert!(table_names.contains(&"attestation"));
        assert!(table_names.contains(&"connection"));
        assert!(table_names.contains(&"participant"));
        assert!(table_names.contains(&"space"));
        assert!(table_names.contains(&"action"));
        assert!(table_names.contains(&"content_block"));
        assert!(table_names.contains(&"request"));
    }

    #[tokio::test]
    async fn initialize_is_idempotent() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();
        initialize(&conn).await.unwrap();
        assert_eq!(get_user_version(&conn).await.unwrap(), LATEST_VERSION);
    }

    /// **A sub-space's anchor is a post in its parent, and the schema says so.**
    /// `parent_action_id` alone only proves the row exists; the compound FK
    /// `(parent_action_id, parent_space_id) → action(id, space_id)` is what
    /// makes "an action from somewhere else" unrepresentable.
    #[tokio::test]
    async fn a_subspace_anchor_must_belong_to_its_parent() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let author = ensure_participant(&conn, "agent", "Navigator", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "here", None, "unlinked", 1_000)
            .await
            .unwrap();
        insert_space(&conn, "elsewhere", None, "unlinked", 1_000)
            .await
            .unwrap();
        let here_post = add_user_action(&conn, "here", &author, "in the parent", 2_000).await;
        let else_post = add_user_action(&conn, "elsewhere", &author, "somewhere else", 2_000).await;

        conn.execute(
            "INSERT INTO space (id, parent_space_id, parent_action_id, linkability, created_at) \
             VALUES ('ok', 'here', ?1, 'unlinked', 3_000)",
            (Value::Text(here_post.clone()),),
        )
        .await
        .expect("an anchor in the named parent must be accepted");

        let mismatch = conn
            .execute(
                "INSERT INTO space (id, parent_space_id, parent_action_id, linkability, created_at) \
                 VALUES ('bad', 'here', ?1, 'unlinked', 3_000)",
                (Value::Text(else_post),),
            )
            .await;
        assert!(
            mismatch.is_err(),
            "an anchor that lives in another conversation must be refused: {mismatch:?}"
        );
    }

    async fn add_user_action(
        conn: &Connection,
        space_id: &str,
        participant_id: &str,
        text: &str,
        created_at: i64,
    ) -> String {
        let action_id = uuid::Uuid::now_v7().to_string();
        insert_action(
            conn,
            &ActionEntry {
                id: action_id.clone(),
                space_id: space_id.to_string(),
                participant_id: participant_id.to_string(),
                item_id: uuid::Uuid::now_v7().to_string(),
                supersedes_action_id: None,
                action_type: "user_input".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at,
            },
        )
        .await
        .unwrap();
        insert_text_content_block(
            conn,
            &uuid::Uuid::now_v7().to_string(),
            &action_id,
            0,
            "text",
            text,
        )
        .await
        .unwrap();
        action_id
    }

    /// **The room's last word is the last one committed**, which is the whole
    /// of what the delegation driver's lifecycle stands on: *is there work
    /// outstanding* is "has the owner quoted the room's last word back to the
    /// parent", so choosing the wrong post as the last word answers the
    /// question about somebody else's post.
    ///
    /// Staged as it actually happens: a reader's post into a watched room is
    /// stamped (`now_ms()`, above the writer's own transaction) before a
    /// driven reply that commits first, so it is *older* by `created_at` and
    /// *newer* by commit. Ordered by the clock, the tail is the driven reply —
    /// which the owner has already quoted — so the room reads as reported and
    /// retires holding a post nobody answered, and the rearmed pass reads the
    /// same wrong tail and exits again.
    #[tokio::test]
    async fn the_rooms_last_word_is_the_last_committed_not_the_last_stamped() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        let owner = ensure_participant(&conn, "agent", "Navigator", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "parent", None, "unlinked", 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();

        // The room's driven reply, and the owner's report in the parent
        // quoting it — the room is settled on that word.
        let driven = add_user_action(&conn, "room", &owner, "the helper's answer", 9_000).await;
        let report = add_user_action(&conn, "parent", &owner, "reporting back", 9_500).await;
        insert_reference_antecedent(&conn, &report, &driven, 1, None, None, None, None)
            .await
            .unwrap();
        assert!(
            has_reference_from(&conn, "parent", &owner, &driven)
                .await
                .unwrap(),
            "the driven reply is the word the parent has"
        );

        // The reader's post: stamped before that reply, committed after it.
        let watched = add_user_action(&conn, "room", &user, "but what about Friday?", 8_000).await;

        assert_eq!(
            last_action_in_space(&conn, "room")
                .await
                .unwrap()
                .as_deref(),
            Some(watched.as_str()),
            "the last word is the post that landed last, not the one stamped last"
        );
        assert!(
            !has_reference_from(&conn, "parent", &owner, &watched)
                .await
                .unwrap(),
            "and it has not been reported — so the rearmed pass has work to do \
             rather than reading a settled room and exiting"
        );
    }

    /// **A regenerated report is still the owner's quote.** Settlement asks
    /// whether the owner has quoted the room's last word, and a successful
    /// regeneration can change the tip's author (`TurnSelector::Model` mints
    /// when no seated member matches) while carrying the edges forward. Asking
    /// the tip's `participant_id` would treat that re-wording as somebody
    /// else's quote. The origin generation is who opened the item; a fresh
    /// quote by anyone else is a different item and still does not settle.
    #[tokio::test]
    async fn a_report_is_recognized_by_its_original_author() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let owner = ensure_participant(&conn, "agent", "Navigator", None, 1_000)
            .await
            .unwrap();
        let other = ensure_participant(&conn, "agent", "Scribe", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "parent", None, "unlinked", 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();

        let driven = add_user_action(&conn, "room", &owner, "the helper's answer", 9_000).await;
        let item = uuid::Uuid::now_v7().to_string();
        let origin = add_report(
            &conn,
            "parent",
            &owner,
            &item,
            None,
            &driven,
            "reporting back",
            9_500,
        )
        .await;
        assert!(
            has_reference_from(&conn, "parent", &owner, &driven)
                .await
                .unwrap(),
            "the owner's report settles the room"
        );

        // A regeneration by someone else — the current generation's author is
        // not the owner, but the item's origin still is, and the edges
        // travelled with it.
        let _tip = add_report(
            &conn,
            "parent",
            &other,
            &item,
            Some(&origin),
            &driven,
            "re-worded",
            10_000,
        )
        .await;
        assert!(
            has_reference_from(&conn, "parent", &owner, &driven)
                .await
                .unwrap(),
            "a regenerated report is still the owner's quote"
        );

        // A fresh quote by someone else is a different item, not a
        // regeneration of the owner's report.
        let elsewhere = add_user_action(&conn, "room", &owner, "another finding", 9_100).await;
        let stranger = add_user_action(&conn, "parent", &other, "quoting too", 10_500).await;
        insert_reference_antecedent(&conn, &stranger, &elsewhere, 1, None, None, None, None)
            .await
            .unwrap();
        assert!(
            !has_reference_from(&conn, "parent", &owner, &elsewhere)
                .await
                .unwrap(),
            "somebody else's quote does not settle the room"
        );
    }

    /// **An answer regenerated under a new author is still the owner's
    /// answer.** `last_reply_by_participant` is where a report attaches, and
    /// asking the tip's `participant_id` would miss a successful regeneration
    /// that minted a new agent — the wait would run out to grace and the
    /// report would land on the anchor as a sibling of a word the parent
    /// already shows. The origin generation is who opened the item; the tip
    /// is what the parent shows.
    #[tokio::test]
    async fn an_answer_is_recognized_by_its_original_author() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let owner = ensure_participant(&conn, "agent", "Navigator", None, 1_000)
            .await
            .unwrap();
        let other = ensure_participant(&conn, "agent", "Scribe", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "parent", None, "unlinked", 1_000)
            .await
            .unwrap();
        let asked = add_user_action(&conn, "parent", &owner, "what about Friday?", 2_000).await;
        let item = uuid::Uuid::now_v7().to_string();
        let origin = add_reply(
            &conn,
            "parent",
            &owner,
            &item,
            None,
            &asked,
            "Friday looks clear",
            3_000,
        )
        .await;
        assert_eq!(
            last_reply_by_participant(&conn, "parent", &owner, &asked, None)
                .await
                .unwrap()
                .as_deref(),
            Some(origin.as_str()),
            "the owner's own answer"
        );

        let tip = add_reply(
            &conn,
            "parent",
            &other,
            &item,
            Some(&origin),
            &asked,
            "Friday looks clear, re-worded",
            4_000,
        )
        .await;
        assert_eq!(
            last_reply_by_participant(&conn, "parent", &owner, &asked, None)
                .await
                .unwrap()
                .as_deref(),
            Some(tip.as_str()),
            "a regenerated answer is still the owner's reply"
        );
        assert_eq!(
            last_post_by_participant(&conn, "parent", &owner)
                .await
                .unwrap()
                .as_deref(),
            Some(tip.as_str()),
            "and still their last word"
        );
        assert_eq!(
            visible_tip_of_action(&conn, &origin)
                .await
                .unwrap()
                .as_deref(),
            Some(tip.as_str()),
            "a queued origin remaps to the wording the parent shows"
        );
        assert_eq!(
            visible_tip_of_action(&conn, &tip).await.unwrap().as_deref(),
            Some(tip.as_str()),
            "and a current tip remaps to itself"
        );
    }

    #[allow(clippy::too_many_arguments)]
    async fn add_reply(
        conn: &Connection,
        space_id: &str,
        participant_id: &str,
        item_id: &str,
        supersedes: Option<&str>,
        antecedent: &str,
        text: &str,
        created_at: i64,
    ) -> String {
        let action_id = uuid::Uuid::now_v7().to_string();
        insert_action(
            conn,
            &ActionEntry {
                id: action_id.clone(),
                space_id: space_id.to_string(),
                participant_id: participant_id.to_string(),
                item_id: item_id.to_string(),
                supersedes_action_id: supersedes.map(str::to_string),
                action_type: "inference".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at,
            },
        )
        .await
        .unwrap();
        insert_text_content_block(
            conn,
            &uuid::Uuid::now_v7().to_string(),
            &action_id,
            0,
            "text",
            text,
        )
        .await
        .unwrap();
        insert_action_antecedent(conn, &action_id, antecedent, 0, "reply")
            .await
            .unwrap();
        action_id
    }

    #[allow(clippy::too_many_arguments)]
    async fn add_report(
        conn: &Connection,
        space_id: &str,
        participant_id: &str,
        item_id: &str,
        supersedes: Option<&str>,
        quoted: &str,
        text: &str,
        created_at: i64,
    ) -> String {
        let action_id = uuid::Uuid::now_v7().to_string();
        insert_action(
            conn,
            &ActionEntry {
                id: action_id.clone(),
                space_id: space_id.to_string(),
                participant_id: participant_id.to_string(),
                item_id: item_id.to_string(),
                supersedes_action_id: supersedes.map(str::to_string),
                action_type: "inference".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at,
            },
        )
        .await
        .unwrap();
        insert_text_content_block(
            conn,
            &uuid::Uuid::now_v7().to_string(),
            &action_id,
            0,
            "text",
            text,
        )
        .await
        .unwrap();
        insert_reference_antecedent(conn, &action_id, quoted, 1, None, None, None, None)
            .await
            .unwrap();
        action_id
    }

    /// **A post is written whole or not at all.** Its action row, its words,
    /// its place in the thread and its quotations were separate autocommitted
    /// statements, so for as long as the tail of them took, the post existed
    /// and said nothing — a state every reader keyed on the action could see
    /// and the sub-space driver's refill would *act* on, planning and billing a
    /// turn against an empty post. One transaction is what makes the post one
    /// event; the fragment is then not a state anything can observe, which is
    /// the only way to hold a reader to it.
    #[tokio::test]
    async fn a_post_is_written_whole_or_not_at_all() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();
        let first = add_user_action(&conn, "room", &user, "the ask", 2_000).await;

        // A post whose *last* write cannot land: the reference edge names an
        // action that does not exist, which the foreign key refuses. Everything
        // before it — the action, its text, the title, the reply edge — is
        // already written when that happens.
        let doomed = uuid::Uuid::now_v7().to_string();
        let err = post_tx(
            &conn,
            &PostPlan {
                space_id: "room",
                participant_id: &user,
                action_id: &doomed,
                item_id: &uuid::Uuid::now_v7().to_string(),
                text: "about that",
                auto_title: Some("A title it must not keep"),
                reply_to: Some(&first),
                references: &[crate::ReferenceSpec {
                    antecedent_action_id: "no-such-action".into(),
                    content_block_id: None,
                    range_start: None,
                    range_end: None,
                    annotation: None,
                }],
                join_author: false,
                created_at: 3_000,
            },
        )
        .await
        .expect_err("a reference to an action that does not exist cannot be written");
        assert!(matches!(err, AppError::Database { .. }), "{err:?}");

        assert_eq!(
            space_action_ids(&conn, "room").await.unwrap(),
            vec![first.clone()],
            "the half-written post is gone entirely — no action for a reader to find"
        );
        assert!(
            first_content_block(&conn, &doomed).await.unwrap().is_none(),
            "and no orphaned text block"
        );
        assert!(
            get_space(&conn, "room")
                .await
                .unwrap()
                .unwrap()
                .title
                .is_none(),
            "and no title from a post that was never written"
        );

        // The same post, written whole: everything arrives together.
        let good = uuid::Uuid::now_v7().to_string();
        assert!(
            post_tx(
                &conn,
                &PostPlan {
                    space_id: "room",
                    participant_id: &user,
                    action_id: &good,
                    item_id: &uuid::Uuid::now_v7().to_string(),
                    text: "about that",
                    auto_title: Some("Named by its first post"),
                    reply_to: Some(&first),
                    references: &[],
                    join_author: false,
                    created_at: 4_000,
                },
            )
            .await
            .unwrap()
            .auto_titled,
            "the space took the title"
        );
        assert!(
            first_content_block(&conn, &good)
                .await
                .unwrap()
                .is_some_and(|(_, text)| text.as_deref() == Some("about that")),
            "the words are there the moment the action is"
        );
        assert_eq!(
            reply_antecedent(&conn, &good).await.unwrap().as_deref(),
            Some(first.as_str()),
            "and so is its place in the thread"
        );
    }

    /// **An edit lands whole or not at all — and an edit is the sharper case.**
    /// Its new row *supersedes* the tip the moment it lands, so written as
    /// separate autocommitted statements the item's **current** generation
    /// said nothing for as long as the tail of writes took — the same window
    /// [`a_post_is_written_whole_or_not_at_all`] closes, opened over a post
    /// that already had words.
    #[tokio::test]
    async fn an_edit_lands_whole_or_not_at_all() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();
        let first = add_user_action(&conn, "room", &user, "the ask", 2_000).await;
        let (item_id, _) = action_item_and_space(&conn, &first).await.unwrap().unwrap();

        // An edit whose *last* write cannot land: a replicated reference names
        // an action that does not exist, which the foreign key refuses. The
        // new generation and its text are already written when that happens.
        let doomed = uuid::Uuid::now_v7().to_string();
        let err = edit_post_tx(
            &conn,
            &EditPostPlan {
                space_id: "room",
                participant_id: &user,
                action_id: &doomed,
                item_id: &item_id,
                supersedes_action_id: &first,
                text: "about that",
                reply_to: None,
                references: &[ReferenceEdgeRow {
                    ordinal: 1,
                    antecedent_action_id: "no-such-action".into(),
                    content_block_id: None,
                    range_start: None,
                    range_end: None,
                    annotation: None,
                    block_text: None,
                    antecedent_action_type: "user_input".into(),
                    antecedent_author_label: "User".into(),
                    block_type: None,
                }],
                created_at: 3_000,
            },
        )
        .await
        .expect_err("a reference to an action that does not exist cannot be written");
        assert!(matches!(err, AppError::Database { .. }), "{err:?}");

        assert_eq!(
            current_tip_of_item(&conn, "room", &item_id).await.unwrap(),
            Some(first.clone()),
            "the tip is untouched — no current generation with no words"
        );
        assert!(
            first_content_block(&conn, &doomed).await.unwrap().is_none(),
            "and no orphaned text block"
        );

        // The same edit, written whole: the tip moves in one event.
        let good = uuid::Uuid::now_v7().to_string();
        edit_post_tx(
            &conn,
            &EditPostPlan {
                space_id: "room",
                participant_id: &user,
                action_id: &good,
                item_id: &item_id,
                supersedes_action_id: &first,
                text: "about that",
                reply_to: None,
                references: &[],
                created_at: 3_000,
            },
        )
        .await
        .expect("a valid edit lands");
        assert_eq!(
            current_tip_of_item(&conn, "room", &item_id).await.unwrap(),
            Some(good),
            "the edit is the current generation, words and all"
        );
    }

    /// **The refill serves an arrival's current words, once.** An arrival
    /// edited before the refill runs is one post, not two: returning both
    /// generations puts both on the walk's frontier, and the walk plans and
    /// bills a turn against wording its author already replaced — then quotes
    /// the retraction into the report beside the edit.
    #[tokio::test]
    async fn the_refill_serves_only_current_generations() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();
        let boundary = action_watermark(&conn).await.unwrap();

        // A post arrives after the boundary and is edited before the refill.
        let orig = add_user_action(&conn, "room", &user, "first wording", 2_000).await;
        let (item_id, _) = action_item_and_space(&conn, &orig).await.unwrap().unwrap();
        let edit = uuid::Uuid::now_v7().to_string();
        edit_post_tx(
            &conn,
            &EditPostPlan {
                space_id: "room",
                participant_id: &user,
                action_id: &edit,
                item_id: &item_id,
                supersedes_action_id: &orig,
                text: "second wording",
                reply_to: None,
                references: &[],
                created_at: 3_000,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            posts_in_space_since(&conn, "room", boundary).await.unwrap(),
            vec![edit],
            "one arrival, its current words — not the retraction beside them"
        );
    }

    /// **The room's last word is the last word the transcript shows.** A
    /// failed regeneration leaves a current hidden tip; without the
    /// `item_current` join the *superseded* generation — still the highest
    /// matching `rowid` — would come back as the tail, and the driver would
    /// walk from, reply beneath, and quote a wording the room no longer
    /// shows. And the budget meter must not charge that failure: a driven
    /// turn writes nothing on failure, and the terminal-status filter keeps
    /// "a failed turn is uncounted" true of the blocking verbs too.
    #[tokio::test]
    async fn a_hidden_tip_is_neither_the_tail_nor_a_spent_turn() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();
        let first = add_user_action(&conn, "room", &user, "the ask", 2_000).await;
        let second = add_user_action(&conn, "room", &user, "the answer", 3_000).await;
        assert_eq!(
            last_action_in_space(&conn, "room").await.unwrap(),
            Some(second.clone())
        );
        let turns_before = turns_taken_in_space(&conn, "room").await.unwrap();

        // A failed regeneration of the newest post: a current, hidden tip.
        let (item_id, _) = action_item_and_space(&conn, &second)
            .await
            .unwrap()
            .unwrap();
        insert_action(
            &conn,
            &ActionEntry {
                id: uuid::Uuid::now_v7().to_string(),
                space_id: "room".to_string(),
                participant_id: user.clone(),
                item_id,
                supersedes_action_id: Some(second),
                action_type: "inference".to_string(),
                status: "error".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at: 4_000,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            last_action_in_space(&conn, "room").await.unwrap(),
            Some(first),
            "the tail falls back to the newest visible post, not the superseded wording"
        );
        assert_eq!(
            turns_taken_in_space(&conn, "room").await.unwrap(),
            turns_before,
            "and the failure spends nothing against the budget"
        );
    }

    /// **The walk boundary is commit order, and a clock cannot stand in for
    /// it.** Every writer in this crate samples `now_ms()` above its own
    /// transaction — `Inner::post` takes it before several awaited validations
    /// — so a post's `created_at` can predate a reader that its commit
    /// postdates. A refill that asked "written at or after my start time" then
    /// missed exactly the posts that raced it, and missing one is a loss: the
    /// walk drives on, its own newer answer becomes the room's last word, the
    /// report settles the room on that word, and the unserved post sits in a
    /// room that reads as reported.
    #[tokio::test]
    async fn the_refill_boundary_is_commit_order_not_the_writers_clock() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();
        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "room", None, "unlinked", 1_000)
            .await
            .unwrap();
        let early = add_user_action(&conn, "room", &user, "before the walk", 5_000).await;

        // The walk opens here.
        let since = action_watermark(&conn).await.unwrap();
        assert!(since > 0, "the mark names a row that exists");

        // A post whose author stamped it *before* the walk opened and whose
        // insert lands after — the interleaving `Inner::post` makes reachable
        // by sampling its clock above its awaits.
        let raced =
            add_user_action(&conn, "room", &user, "stamped early, landed late", 4_000).await;
        // And an ordinary one after it.
        let later = add_user_action(&conn, "room", &user, "plainly later", 9_000).await;

        let arrived = posts_in_space_since(&conn, "room", since).await.unwrap();
        assert_eq!(
            arrived,
            vec![raced.clone(), later.clone()],
            "both arrivals are served, in the order they were committed"
        );
        assert!(
            !arrived.contains(&early),
            "and what was already in front of the walk is not served twice"
        );

        // The mark moves with the writes, so a second walk opening now sees
        // nothing outstanding.
        let next = action_watermark(&conn).await.unwrap();
        assert!(next > since);
        assert!(
            posts_in_space_since(&conn, "room", next)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn list_spaces_reports_activity_and_excludes_archived() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();

        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();

        // Space A: titled, two actions (latest at t=3000).
        insert_space(&conn, "space-a", Some("Alpha"), "unlinked", 1_000)
            .await
            .unwrap();
        add_user_action(&conn, "space-a", &user, "first question", 2_000).await;
        add_user_action(&conn, "space-a", &user, "follow-up", 3_000).await;

        // Space B: untitled, one action, more recent activity.
        insert_space(&conn, "space-b", None, "unlinked", 1_500)
            .await
            .unwrap();
        add_user_action(&conn, "space-b", &user, "what is a monad?", 4_000).await;

        // Space C: empty (no actions yet).
        insert_space(&conn, "space-c", None, "unlinked", 5_000)
            .await
            .unwrap();

        // Space D: archived.
        insert_space(&conn, "space-d", Some("Old"), "unlinked", 500)
            .await
            .unwrap();
        assert_eq!(
            archive_space_tx(&conn, "space-d", 6_000).await.unwrap(),
            vec!["space-d".to_string()]
        );

        let rows = list_spaces(&conn, false).await.unwrap();
        let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        // Ordered by last activity, most recent first; archived excluded.
        assert_eq!(ids, vec!["space-c", "space-b", "space-a"]);

        let a = rows.iter().find(|r| r.id == "space-a").unwrap();
        assert_eq!(a.title.as_deref(), Some("Alpha"));
        assert_eq!(a.last_activity_at, 3_000);
        assert_eq!(a.message_count, 2);
        assert!(a.archived_at.is_none());

        let b = rows.iter().find(|r| r.id == "space-b").unwrap();
        assert!(b.title.is_none());
        assert_eq!(b.last_activity_at, 4_000);
        assert_eq!(b.message_count, 1);

        // Empty space falls back to its own created_at.
        let c = rows.iter().find(|r| r.id == "space-c").unwrap();
        assert_eq!(c.last_activity_at, 5_000);
        assert_eq!(c.message_count, 0);

        // include_archived = true brings the archived space back.
        let all = list_spaces(&conn, true).await.unwrap();
        assert_eq!(all.len(), 4);
        let d = all.iter().find(|r| r.id == "space-d").unwrap();
        assert_eq!(d.archived_at, Some(6_000));

        // Snippet source: first user text in the space.
        assert_eq!(
            first_user_text(&conn, "space-a").await.unwrap().as_deref(),
            Some("first question")
        );
        assert_eq!(
            first_user_text(&conn, "space-b").await.unwrap().as_deref(),
            Some("what is a monad?")
        );
        assert_eq!(first_user_text(&conn, "space-c").await.unwrap(), None);
    }

    #[tokio::test]
    async fn get_space_tree_data_resolves_tips_filters_trace_and_returns_edges() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();

        let user = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        let agent = ensure_participant(&conn, "agent", "kimi", None, 1_000)
            .await
            .unwrap();

        insert_space(&conn, "space-t", None, "unlinked", 1_000)
            .await
            .unwrap();

        #[allow(clippy::too_many_arguments)]
        async fn mk(
            conn: &Connection,
            id: &str,
            participant: &str,
            item: &str,
            supersedes: Option<&str>,
            ty: &str,
            model: Option<&str>,
            credits: Option<i64>,
            at: i64,
        ) {
            insert_action(
                conn,
                &ActionEntry {
                    id: id.to_string(),
                    space_id: "space-t".to_string(),
                    participant_id: participant.to_string(),
                    item_id: item.to_string(),
                    supersedes_action_id: supersedes.map(String::from),
                    action_type: ty.to_string(),
                    status: "complete".to_string(),
                    intent: None,
                    model: model.map(String::from),
                    input_tokens: None,
                    output_tokens: None,
                    credits_consumed: credits,
                    created_at: at,
                },
            )
            .await
            .unwrap();
        }

        // u1 (gen0) -> i1 (reply) -> then edit u1 to u1b (gen1, supersedes u1).
        mk(&conn, "u1", &user, "iu1", None, "user_input", None, None, 1).await;
        insert_text_content_block(&conn, "cb-u1", "u1", 0, "text", "first")
            .await
            .unwrap();
        mk(
            &conn,
            "i1",
            &agent,
            "ii1",
            None,
            "inference",
            Some("kimi"),
            Some(700),
            2,
        )
        .await;
        insert_text_content_block(&conn, "cb-i1", "i1", 0, "text", "answer")
            .await
            .unwrap();
        insert_action_antecedent(&conn, "i1", "u1", 0, "reply")
            .await
            .unwrap();
        // Edit: u1b supersedes u1, replicating the (absent) reply edge.
        mk(
            &conn,
            "u1b",
            &user,
            "iu1",
            Some("u1"),
            "user_input",
            None,
            None,
            3,
        )
        .await;
        insert_text_content_block(&conn, "cb-u1b", "u1b", 0, "text", "edited")
            .await
            .unwrap();
        // A reference edge from the edited post.
        conn.execute(
            "INSERT INTO action_antecedent \
             (action_id, antecedent_action_id, ordinal, relation, range_start, range_end, annotation) \
             VALUES ('u1b', 'i1', 1, 'reference', 0, 6, 'quoting')",
            (),
        )
        .await
        .unwrap();
        // A trace action (request) that must NOT render as a post.
        mk(&conn, "req1", &user, "ireq", None, "request", None, None, 4).await;

        let data = get_space_tree_data(&conn, "space-t").await.unwrap();

        // Tips only: u1b (not the superseded u1) and i1. No trace action.
        // Ordered by **item origin** (iu1 born at t=1, ii1 at t=2), not the
        // tip's own created_at (u1b at t=3) — the edited post keeps its place.
        let ids: Vec<&str> = data.actions.iter().map(|a| a.action_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["u1b", "i1"],
            "edited item must keep its origin position, not jump to the end"
        );

        let u1b = data.actions.iter().find(|a| a.action_id == "u1b").unwrap();
        assert_eq!(u1b.item_id, "iu1");
        assert_eq!(
            u1b.generation, 1,
            "tip of a once-edited item is generation 1"
        );
        assert_eq!(u1b.participant_kind, "human");

        let i1 = data.actions.iter().find(|a| a.action_id == "i1").unwrap();
        assert_eq!(i1.generation, 0);
        assert_eq!(i1.participant_kind, "agent");
        assert_eq!(i1.model.as_deref(), Some("kimi"));
        assert_eq!(i1.credits_consumed, Some(700));

        // Blocks come back only for the tip actions.
        let block_actions: Vec<&str> = data.blocks.iter().map(|b| b.action_id.as_str()).collect();
        assert!(block_actions.contains(&"u1b"));
        assert!(block_actions.contains(&"i1"));
        assert!(!block_actions.contains(&"u1"), "superseded gen excluded");

        // Edges: i1's reply -> u1 (raw, now non-tip), resolved through item
        // identity to u1b (iu1's current tip) — causality keeps the concrete
        // generation, threading follows the item. And u1b's reference -> i1.
        let reply: Vec<_> = data
            .edges
            .iter()
            .filter(|e| e.relation == "reply")
            .collect();
        assert_eq!(reply.len(), 1);
        assert_eq!(reply[0].action_id, "i1");
        assert_eq!(reply[0].antecedent_action_id, "u1");
        assert_eq!(
            reply[0].antecedent_current_action_id.as_deref(),
            Some("u1b"),
            "a reply to a superseded generation resolves to the item's tip"
        );
        let refs: Vec<_> = data
            .edges
            .iter()
            .filter(|e| e.relation == "reference")
            .collect();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].action_id, "u1b");
        assert_eq!(refs[0].range_end, Some(6));
        assert_eq!(refs[0].annotation.as_deref(), Some("quoting"));
        assert_eq!(
            refs[0].antecedent_current_action_id.as_deref(),
            Some("i1"),
            "an un-superseded target resolves to itself"
        );
    }

    /// The one-root-per-item invariant is enforced by the database, not
    /// convention: a second gen-0 action for an existing item must be
    /// rejected (`idx_one_root_per_item`), so `item_current` can never yield
    /// two tips for one item.
    #[tokio::test]
    async fn duplicate_item_root_is_rejected() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();

        let p = ensure_participant(&conn, "human", "user", None, 1_000)
            .await
            .unwrap();
        insert_space(&conn, "space-r", None, "unlinked", 1_000)
            .await
            .unwrap();

        let entry = |id: &str, supersedes: Option<&str>| ActionEntry {
            id: id.to_string(),
            space_id: "space-r".to_string(),
            participant_id: p.clone(),
            item_id: "item-1".to_string(),
            supersedes_action_id: supersedes.map(String::from),
            action_type: "user_input".to_string(),
            status: "complete".to_string(),
            intent: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            credits_consumed: None,
            created_at: 1_000,
        };

        insert_action(&conn, &entry("a1", None)).await.unwrap();
        // A proper generation (supersedes the root) is fine…
        insert_action(&conn, &entry("a2", Some("a1")))
            .await
            .unwrap();
        // …but a second root for the same item must be rejected.
        let dup = insert_action(&conn, &entry("a3", None)).await;
        assert!(
            dup.is_err(),
            "a second gen-0 for one item must violate idx_one_root_per_item"
        );
    }

    /// Seed a database with one connected story: two attestations, a
    /// connection, three requests (one carrying a credential + action into a
    /// space, one a retry, one errored), and a credential mid-spend. The
    /// fixture all the Record query tests share.
    async fn seed_record_fixture(conn: &Connection) {
        // Wallet: issuer key + issued credential + a pending refund (spending).
        upsert_issuer_key(
            conn,
            "ik-1",
            "ph",
            b"pub",
            b"params",
            99_999_999_999_999,
            1_000,
        )
        .await
        .unwrap();
        insert_pre_credential_issuance(conn, "pc-1", "ik-1", b"pre", 10_000, 1_000)
            .await
            .unwrap();
        insert_credential(conn, "nonce-1", "pc-1", "ik-1", b"cred", 10_000, 0, 1_100)
            .await
            .unwrap();
        insert_pre_credential_refund(
            conn, "pc-2", "nonce-1", "ik-1", b"pre2", 700, b"proof", 2_000,
        )
        .await
        .unwrap();

        // Transport: provider, attestations, connection.
        let provider = ensure_provider(conn, "eidola", "inference", 1_000)
            .await
            .unwrap();
        upsert_attestation(conn, "att-old", b"{\"v\":1}", None, 1_000)
            .await
            .unwrap();
        upsert_attestation(conn, "att-new", b"{\"v\":2}", Some("pcr-abc"), 2_000)
            .await
            .unwrap();
        insert_connection(
            conn,
            "conn-1",
            &provider,
            "https://e.example",
            "clearnet",
            Some("att-new"),
            2_100,
            2_100,
        )
        .await
        .unwrap();

        // Semantic: space + participant + a costed inference action.
        insert_space(conn, "space-1", Some("Tides"), "unlinked", 1_000)
            .await
            .unwrap();
        let agent = ensure_participant(conn, "agent", "eidola", None, 1_000)
            .await
            .unwrap();
        insert_action(
            conn,
            &ActionEntry {
                id: "act-1".into(),
                space_id: "space-1".into(),
                participant_id: agent,
                item_id: "item-1".into(),
                supersedes_action_id: None,
                action_type: "inference".into(),
                status: "complete".into(),
                intent: Some("answer".into()),
                model: Some("gemma4-31b".into()),
                input_tokens: Some(120),
                output_tokens: Some(480),
                credits_consumed: Some(700),
                created_at: 2_200,
            },
        )
        .await
        .unwrap();

        // Requests: chat (credential-bearing, newest), a retry of it, and an
        // older errored one with no connection.
        insert_request(
            conn,
            &Request {
                id: "req-err".into(),
                connection_id: None,
                action_id: None,
                method: "GET".into(),
                path: "/v1/models".into(),
                request_headers: None,
                request_body: None,
                response_status: None,
                response_headers: None,
                response_body: None,
                request_at: 1_500,
                response_at: None,
                duration_ms: None,
                error: Some("connection refused".into()),
                credential_nonce: None,
                created_at: 1_500,
                backend_id: None,
            },
        )
        .await
        .unwrap();
        insert_request(
            conn,
            &Request {
                id: "req-chat".into(),
                connection_id: Some("conn-1".into()),
                action_id: Some("act-1".into()),
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                request_headers: Some("content-type: application/json".into()),
                request_body: Some(b"{\"model\":\"gemma4-31b\"}".to_vec()),
                response_status: Some(200),
                response_headers: Some("content-type: application/json".into()),
                response_body: Some(b"{\"ok\":true}".to_vec()),
                request_at: 2_200,
                response_at: Some(2_900),
                duration_ms: Some(700),
                error: None,
                credential_nonce: Some("nonce-1".into()),
                created_at: 2_200,
                backend_id: Some("eidola".into()),
            },
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE request SET retry_of_id = 'req-err', attempt_number = 2 \
             WHERE id = 'req-chat'",
            (),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn record_attestation_queries() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();
        seed_record_fixture(&conn).await;

        let rows = list_attestations(&conn, 10, 0).await.unwrap();
        assert_eq!(rows.len(), 2);
        // Newest first.
        assert_eq!(rows[0].hash, "att-new");
        assert_eq!(rows[0].pcr_digest.as_deref(), Some("pcr-abc"));
        assert_eq!(rows[0].doc_bytes, 7);
        assert_eq!(rows[0].connection_count, 1);
        assert_eq!(rows[1].hash, "att-old");
        assert_eq!(rows[1].connection_count, 0);

        // Pagination.
        let page2 = list_attestations(&conn, 1, 1).await.unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].hash, "att-old");

        // Detail returns the raw stored document.
        let doc = get_attestation(&conn, "att-new").await.unwrap().unwrap();
        assert_eq!(doc.doc, b"{\"v\":2}");
        assert_eq!(doc.created_at, 2_000);
        assert!(get_attestation(&conn, "missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn record_request_queries() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();
        seed_record_fixture(&conn).await;

        let rows = list_requests(&conn, 10, 0).await.unwrap();
        assert_eq!(rows.len(), 2);
        // Newest first, joined connection metadata present.
        assert_eq!(rows[0].id, "req-chat");
        assert_eq!(rows[0].method, "POST");
        assert_eq!(rows[0].response_status, Some(200));
        assert_eq!(rows[0].duration_ms, Some(700));
        assert_eq!(rows[0].attempt_number, 2);
        assert_eq!(rows[0].credential_nonce.as_deref(), Some("nonce-1"));
        assert_eq!(rows[0].transport.as_deref(), Some("clearnet"));
        assert_eq!(rows[0].attestation_hash.as_deref(), Some("att-new"));
        // Connection-less request still lists, with NULL transport.
        assert_eq!(rows[1].id, "req-err");
        assert_eq!(rows[1].error.as_deref(), Some("connection refused"));
        assert!(rows[1].transport.is_none());

        // Pagination.
        let page2 = list_requests(&conn, 1, 1).await.unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].id, "req-err");

        // Detail carries the raw bodies, headers, and retry linkage.
        let d = get_request(&conn, "req-chat").await.unwrap().unwrap();
        assert_eq!(
            d.request_body.as_deref(),
            Some(b"{\"model\":\"gemma4-31b\"}".as_slice())
        );
        assert_eq!(
            d.response_body.as_deref(),
            Some(b"{\"ok\":true}".as_slice())
        );
        assert_eq!(
            d.request_headers.as_deref(),
            Some("content-type: application/json")
        );
        assert_eq!(d.retry_of_id.as_deref(), Some("req-err"));
        assert_eq!(d.action_id.as_deref(), Some("act-1"));
        assert_eq!(d.base_url.as_deref(), Some("https://e.example"));
        assert!(get_request(&conn, "missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn record_spend_trail_and_lifecycle() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();
        seed_record_fixture(&conn).await;

        // The credential has a pending refund and no successor → spending.
        let trail = list_spend_trail(&conn, 10, 0).await.unwrap();
        assert_eq!(trail.len(), 1, "only the credential-bearing request joins");
        let t = &trail[0];
        assert_eq!(t.credential_nonce, "nonce-1");
        assert_eq!(t.credential_state, "spending");
        assert_eq!(t.spend_amount, Some(700));
        assert_eq!(t.request_id, "req-chat");
        assert_eq!(t.action_id.as_deref(), Some("act-1"));
        assert_eq!(t.model.as_deref(), Some("gemma4-31b"));
        assert_eq!(t.credits_consumed, Some(700));
        assert_eq!(t.space_id.as_deref(), Some("space-1"));
        assert_eq!(t.space_title.as_deref(), Some("Tides"));
        assert_eq!(t.linkability.as_deref(), Some("unlinked"));

        let lc = list_credential_lifecycle(&conn).await.unwrap();
        assert_eq!(lc.len(), 1);
        assert_eq!(lc[0].state, "spending");
        assert_eq!(lc[0].spend_amount, Some(700));

        // Issuing the successor credential settles the spend → spent.
        insert_credential(&conn, "nonce-2", "pc-2", "ik-1", b"cred2", 9_300, 1, 3_000)
            .await
            .unwrap();
        let trail = list_spend_trail(&conn, 10, 0).await.unwrap();
        assert_eq!(trail[0].credential_state, "spent");
        let lc = list_credential_lifecycle(&conn).await.unwrap();
        let states: Vec<(&str, &str)> = lc
            .iter()
            .map(|r| (r.nonce.as_str(), r.state.as_str()))
            .collect();
        assert_eq!(states, vec![("nonce-2", "active"), ("nonce-1", "spent")]);

        // An expired issuer key flips its credentials to expired.
        upsert_issuer_key(&conn, "ik-exp", "ph", b"pub", b"params", 1, 1_000)
            .await
            .unwrap();
        insert_pre_credential_issuance(&conn, "pc-exp", "ik-exp", b"pre", 5, 4_000)
            .await
            .unwrap();
        insert_credential(&conn, "nonce-exp", "pc-exp", "ik-exp", b"c", 5, 0, 4_000)
            .await
            .unwrap();
        let lc = list_credential_lifecycle(&conn).await.unwrap();
        let exp = lc.iter().find(|r| r.nonce == "nonce-exp").unwrap();
        assert_eq!(exp.state, "expired");
    }

    #[tokio::test]
    async fn seeds_human_and_default_template_idempotently() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();

        // Run the seed again — it must be a no-op (no duplicate rows, no
        // error).
        ensure_default_participants(&conn).await.unwrap();

        // The shared human is a GLOBAL participant.
        let you = get_participant(&conn, HUMAN_PARTICIPANT_ID)
            .await
            .unwrap()
            .expect("the shared human is seeded");
        assert_eq!(you.scope, "global");
        assert_eq!(you.kind, "human");
        assert_eq!(you.label, "User");
        assert!(you.owner_space_id.is_none() && you.owner_template_id.is_none());

        let templates = list_space_templates(&conn).await.unwrap();
        assert_eq!(templates.len(), 1, "exactly one seeded template");
        assert_eq!(templates[0].id, crate::config::DEFAULT_TEMPLATE_ID);
        assert_eq!(templates[0].title, "Default");
        assert_eq!(templates[0].cascade_limit, 4);

        // The Default template OWNS its one agent (scope='template').
        let owned = list_template_owned_participants(&conn, crate::config::DEFAULT_TEMPLATE_ID)
            .await
            .unwrap();
        assert_eq!(owned.len(), 1, "single owned agent, not duplicated");
        assert_eq!(owned[0].scope, "template");
        assert_eq!(
            owned[0].owner_template_id.as_deref(),
            Some(crate::config::DEFAULT_TEMPLATE_ID)
        );
        assert_eq!(owned[0].kind, "agent");
        assert_eq!(
            owned[0].model_ref.as_deref(),
            Some(crate::config::DEFAULT_MODEL)
        );
        assert_eq!(owned[0].notify_policy, "human");
    }

    /// The primitive behind task 41: a re-seed must not re-populate a template
    /// whose agents the user has replaced. `update_template_tx` replaces the
    /// owned set by hard-deleting it, so "the seeded id is absent" is the
    /// *normal* state of an edited template — never a reason to write.
    #[tokio::test]
    async fn a_reseed_does_not_repopulate_an_edited_default_template() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;

        // The user edits the Default template: its agents are replaced by one
        // of their own (what `update_template_tx` does).
        update_template_tx(
            &conn,
            crate::config::DEFAULT_TEMPLATE_ID,
            None,
            None,
            Some(&[(
                "Mine".to_string(),
                Some("gemma-4-e2b@local".to_string()),
                None,
                "human".to_string(),
            )]),
            2_000,
        )
        .await
        .unwrap();

        // Next open.
        ensure_default_participants(&conn).await.unwrap();

        let owned = list_template_owned_participants(&conn, crate::config::DEFAULT_TEMPLATE_ID)
            .await
            .unwrap();
        let labels: Vec<&str> = owned.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["Mine"],
            "the seed must not re-inject the default agent over the user's edit"
        );
    }

    /// Creating the Default template and its agent is **atomic**. Without
    /// that, a failure between the two writes would commit the template row
    /// alone and *consume the creation signal* — every later open would report
    /// zero changes and skip the agent seed permanently, leaving new spaces
    /// with no assistant and no way back (the state is indistinguishable from a
    /// user who emptied the template on purpose, which
    /// `a_reseed_does_not_refill_an_emptied_default_template` requires us to
    /// respect). The failure is forced deterministically by squatting on the
    /// seeded agent's id, so the plain `INSERT` hits a PK conflict.
    #[tokio::test]
    async fn a_failed_first_seed_leaves_no_half_created_template() {
        // Schema only — `open_memory_migrated` does not seed.
        let db = open_memory_migrated().await;
        let conn = fk_conn(&db).await;

        insert_participant(
            &conn,
            DEFAULT_TEMPLATE_AGENT_ID,
            "global",
            None,
            None,
            "agent",
            "Squatter",
            None,
            None,
            "explicit",
            "member",
            None,
            1_000,
        )
        .await
        .unwrap();

        ensure_default_participants(&conn)
            .await
            .expect_err("the agent insert must fail on the id conflict");

        assert!(
            get_space_template(&conn, crate::config::DEFAULT_TEMPLATE_ID)
                .await
                .unwrap()
                .is_none(),
            "a half-created template must roll back, not persist without its agent"
        );

        // …and the seed stays retryable: with the conflict gone, the next open
        // creates both rows.
        conn.execute(
            "DELETE FROM participant WHERE id = ?1",
            (Value::Text(DEFAULT_TEMPLATE_AGENT_ID.to_string()),),
        )
        .await
        .unwrap();
        ensure_default_participants(&conn).await.unwrap();

        assert!(
            get_space_template(&conn, crate::config::DEFAULT_TEMPLATE_ID)
                .await
                .unwrap()
                .is_some()
        );
        let owned = list_template_owned_participants(&conn, crate::config::DEFAULT_TEMPLATE_ID)
            .await
            .unwrap();
        assert_eq!(owned.len(), 1, "the retry seeds the agent too");
    }

    /// …and the same for an outright emptied template: absence is a choice.
    #[tokio::test]
    async fn a_reseed_does_not_refill_an_emptied_default_template() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;

        delete_template_owned_participants(&conn, crate::config::DEFAULT_TEMPLATE_ID)
            .await
            .unwrap();
        ensure_default_participants(&conn).await.unwrap();

        assert!(
            list_template_owned_participants(&conn, crate::config::DEFAULT_TEMPLATE_ID)
                .await
                .unwrap()
                .is_empty(),
            "a template the user emptied stays empty"
        );
    }

    #[tokio::test]
    async fn notify_policy_check_rejects_bad_value() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        // A value outside ('explicit','human','all') must violate the CHECK.
        let bad = insert_participant(
            &conn,
            &uuid::Uuid::now_v7().to_string(),
            "global",
            None,
            None,
            "agent",
            "Bad",
            Some("gemma4-31b"),
            None,
            "sometimes",
            "member",
            None,
            1_000,
        )
        .await;
        assert!(bad.is_err(), "notify_policy CHECK must reject 'sometimes'");
    }

    /// The scope-owned model's central invariant, proven by the FK/CHECK
    /// machinery rather than convention: a reference row (space_participant)
    /// may point ONLY at a global. Referencing a SPACE-owned participant must
    /// fail — its (id, scope='space') tuple has no (id, 'global') match, and the
    /// pinned `participant_scope='global'` echo + composite FK reject it.
    #[tokio::test]
    async fn referencing_a_space_owned_participant_is_structurally_impossible() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space(&conn, "space-i", None, "unlinked", 1_000)
            .await
            .unwrap();
        // A space-OWNED agent (scope='space').
        let owned = uuid::Uuid::now_v7().to_string();
        insert_participant(
            &conn,
            &owned,
            "space",
            Some("space-i"),
            None,
            "agent",
            "Owned",
            Some("m"),
            None,
            "explicit",
            "member",
            None,
            1_000,
        )
        .await
        .unwrap();
        // Referencing it as a global membership must be rejected by the DB.
        let smuggle = conn
            .execute(
                "INSERT INTO space_participant \
                 (space_id, participant_id, participant_scope, role, joined_at) \
                 VALUES ('space-i', ?1, 'global', 'member', 1000)",
                (Value::Text(owned.clone()),),
            )
            .await;
        assert!(
            smuggle.is_err(),
            "a space_participant referencing a space-owned participant must \
             violate the composite FK (only globals are referenceable)"
        );
    }

    #[tokio::test]
    async fn instantiate_project_round_trip_preserves_owned_referenced_and_overrides() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;

        // Instantiate the seeded default template into a fresh space.
        instantiate_template(
            &conn,
            crate::config::DEFAULT_TEMPLATE_ID,
            "space-x",
            None,
            "unlinked",
            1_000,
        )
        .await
        .unwrap();

        assert_eq!(
            space_cascade_limit(&conn, "space-x").await.unwrap(),
            Some(4)
        );
        // The may-decline router is OFF by default: the seeded template carries
        // no router model, so neither does a space born from it.
        assert_eq!(
            get_space_template(&conn, crate::config::DEFAULT_TEMPLATE_ID)
                .await
                .unwrap()
                .unwrap()
                .router_model,
            None
        );
        assert_eq!(space_router_model(&conn, "space-x").await.unwrap(), None);

        // Effective members: You (referenced global, owner) + one owned agent.
        let members = space_participants(&conn, "space-x").await.unwrap();
        assert_eq!(members.len(), 2, "You + one owned agent");
        let you = members.iter().find(|m| m.kind == "human").unwrap();
        assert_eq!(you.participant_id, HUMAN_PARTICIPANT_ID);
        assert_eq!(you.source, "referenced");
        assert_eq!(you.scope, "global");
        assert_eq!(you.role, "owner");
        let agent = members.iter().find(|m| m.kind == "agent").unwrap();
        assert_eq!(agent.source, "owned");
        assert_eq!(agent.scope, "space");
        assert_eq!(
            agent.model_ref.as_deref(),
            Some(crate::config::DEFAULT_MODEL)
        );
        assert_ne!(
            agent.participant_id, DEFAULT_TEMPLATE_AGENT_ID,
            "fresh copy"
        );

        // By-model resolution returns (id, scope) for the action echo.
        assert_eq!(
            space_agent_participant_by_model(&conn, "space-x", crate::config::DEFAULT_MODEL)
                .await
                .unwrap(),
            Some((agent.participant_id.clone(), "space".to_string()))
        );

        // Override the referenced global (You) for this space only, and edit the
        // owned agent's own config.
        assert!(
            update_space_participant_override(
                &conn,
                "space-x",
                HUMAN_PARTICIPANT_ID,
                Some(Some("Mike")),
                None,
                None,
                None,
                2_000,
            )
            .await
            .unwrap()
        );
        update_participant_config(
            &conn,
            &agent.participant_id,
            &PersonaWrite {
                label: Some("Justin".into()),
                model_ref: Some(Some("kimi-k2-6".into())),
                system_prompt: Some(Some("Be terse.".into())),
                notify_policy: Some("all".into()),
                ..Default::default()
            },
            2_000,
        )
        .await
        .unwrap();

        // COALESCE resolution: You now reads "Mike" (override), the agent its edit.
        let members = space_participants(&conn, "space-x").await.unwrap();
        let you = members.iter().find(|m| m.kind == "human").unwrap();
        assert_eq!(you.label, "Mike", "override wins via COALESCE");
        let agent = members.iter().find(|m| m.kind == "agent").unwrap();
        assert_eq!(agent.label, "Justin");
        assert_eq!(agent.model_ref.as_deref(), Some("kimi-k2-6"));

        // Turn the may-decline router on for this space, so the projection
        // below carries it exactly like cascade_limit.
        assert!(
            set_space_router_model(&conn, "space-x", Some("router@local"), 2_000)
                .await
                .unwrap()
        );
        assert_eq!(
            space_router_model(&conn, "space-x")
                .await
                .unwrap()
                .as_deref(),
            Some("router@local")
        );

        // Project back into a template: owned agent → template-owned; the You
        // reference (with its override) → a template reference.
        template_from_space(&conn, "space-x", "My Template", "tmpl-x", 2_000)
            .await
            .unwrap();
        let projected = get_space_template(&conn, "tmpl-x").await.unwrap().unwrap();
        assert_eq!(projected.cascade_limit, 4);
        assert_eq!(
            projected.router_model.as_deref(),
            Some("router@local"),
            "template_from_space copies router_model like cascade_limit"
        );

        let owned = list_template_owned_participants(&conn, "tmpl-x")
            .await
            .unwrap();
        assert_eq!(owned.len(), 1, "the edited agent, owned");
        assert_eq!(owned[0].label, "Justin");
        assert_eq!(owned[0].model_ref.as_deref(), Some("kimi-k2-6"));
        assert_eq!(owned[0].system_prompt.as_deref(), Some("Be terse."));
        assert_eq!(owned[0].notify_policy, "all");

        let refs = list_template_participant_refs(&conn, "tmpl-x")
            .await
            .unwrap();
        assert_eq!(refs.len(), 1, "the You reference carried across");
        assert_eq!(refs[0].participant_id, HUMAN_PARTICIPANT_ID);
        assert_eq!(
            refs[0].override_label.as_deref(),
            Some("Mike"),
            "override preserved"
        );

        // Instantiate the projected template → the round-tripped space resolves
        // You to "Mike" (override) and carries the owned edited agent.
        instantiate_template(&conn, "tmpl-x", "space-y", None, "unlinked", 3_000)
            .await
            .unwrap();
        assert_eq!(
            space_router_model(&conn, "space-y")
                .await
                .unwrap()
                .as_deref(),
            Some("router@local"),
            "instantiate_template copies router_model into the new space"
        );
        // …and clearing it back to NULL (the feature off) round-trips.
        assert!(
            set_template_router_model(&conn, "tmpl-x", None)
                .await
                .unwrap()
        );
        assert_eq!(
            get_space_template(&conn, "tmpl-x")
                .await
                .unwrap()
                .unwrap()
                .router_model,
            None
        );
        instantiate_template(&conn, "tmpl-x", "space-z", None, "unlinked", 4_000)
            .await
            .unwrap();
        assert_eq!(space_router_model(&conn, "space-z").await.unwrap(), None);
        let members = space_participants(&conn, "space-y").await.unwrap();
        let you = members.iter().find(|m| m.kind == "human").unwrap();
        assert_eq!(you.label, "Mike");
        let agent = members.iter().find(|m| m.kind == "agent").unwrap();
        assert_eq!(agent.source, "owned");
        assert_eq!(agent.model_ref.as_deref(), Some("kimi-k2-6"));
    }

    /// Override-COALESCE edge cases: NULL inherits, '' overrides to empty.
    #[tokio::test]
    async fn override_coalesce_inherit_and_empty() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space(&conn, "space-o", None, "unlinked", 1_000)
            .await
            .unwrap();
        ensure_space_participant(&conn, "space-o", HUMAN_PARTICIPANT_ID, "owner", 1_000)
            .await
            .unwrap();

        // No override → inherits the global's label "User".
        let m = space_participants(&conn, "space-o").await.unwrap();
        assert_eq!(m[0].label, "User");

        // '' override → effective empty (override to empty, not inherit).
        update_space_participant_override(
            &conn,
            "space-o",
            HUMAN_PARTICIPANT_ID,
            Some(Some("")),
            None,
            None,
            None,
            2_000,
        )
        .await
        .unwrap();
        let m = space_participants(&conn, "space-o").await.unwrap();
        assert_eq!(m[0].label, "", "empty-string override wins over inherit");

        // NULL override → back to inherit.
        update_space_participant_override(
            &conn,
            "space-o",
            HUMAN_PARTICIPANT_ID,
            Some(None),
            None,
            None,
            None,
            2_000,
        )
        .await
        .unwrap();
        let m = space_participants(&conn, "space-o").await.unwrap();
        assert_eq!(m[0].label, "User", "NULL override inherits again");
    }

    #[tokio::test]
    async fn instantiate_removed_template_fails() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space_template(&conn, "t-gone", "Gone", 4, None, 1_000)
            .await
            .unwrap();
        assert!(
            soft_remove_space_template(&conn, "t-gone", 2_000)
                .await
                .unwrap()
        );
        let r = instantiate_template(&conn, "t-gone", "space-z", None, "unlinked", 3_000).await;
        assert!(r.is_err(), "instantiating a removed template must fail");
    }

    /// **A failed instantiation leaves no space at all.** These are several
    /// statements over one database, so before the transaction a failure
    /// partway left a durable `space` row carrying some or none of its
    /// membership — a conversation nobody could be told about, which every
    /// caller reasonably reads as "nothing was written" (the GUI's ⌘N drops its
    /// registry entry on exactly that reading).
    ///
    /// The fault is injected the only way a mid-instantiation failure can be
    /// staged deterministically: a template reference naming a participant that
    /// does not exist, written on a connection with FK enforcement off (turso's
    /// default), so the copy of it into `space_participant` — several
    /// statements *after* the space row and its owned agents — hits the
    /// composite FK. What a full disk or a future constraint would do to the
    /// same door, at a step we can name.
    /// **A write marks its space before it writes, so even a write that never
    /// lands leaves the space kept.**
    ///
    /// This is the observable end of the ordering rule. `conn.execute`
    /// autocommits, so if `insert_participant` stamped *after* its insert there
    /// would be a moment when the new participant is durable and its space
    /// still reads pristine — a disposal interleaving there deletes the space
    /// and the row together, and the stamp that follows finds nothing to
    /// update. Marking first makes that window impossible, and the price is
    /// visible here: a refused insert marks a space nothing changed. Keeping a
    /// listing row is the error worth making.
    /// **A space's existence is decided by the write, not by a read before
    /// it** — staged as the interleaving that makes the difference visible.
    ///
    /// A disposal of an untouched space travels while the Library still lists
    /// its row, so a rename can be issued after the delete is on its way: the
    /// read says the space is there, the delete commits, and the write then
    /// strikes nothing. A door that took its answer from the read reports
    /// success for a title no row took — and the caller's optimistic edit
    /// stands over a database that never agreed to it, with nothing on screen
    /// to say so. `update_space_title` answers whether a row took it, which is
    /// what lets `rename_space` refuse.
    ///
    /// The roster's door is checked in the same breath, from the other
    /// direction: it has no rows-affected to read, and does not need one — the
    /// foreign key is what refuses it, and a refusal is all its caller needs.
    #[tokio::test]
    async fn a_write_into_a_space_that_has_just_gone_refuses_rather_than_reporting_success() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        instantiate_template(
            &conn,
            crate::config::DEFAULT_TEMPLATE_ID,
            "space-race",
            None,
            "unlinked",
            1_000,
        )
        .await
        .unwrap();

        // The read a door would have decided from.
        assert!(
            get_space(&conn, "space-race").await.unwrap().is_some(),
            "the space is there when the door is asked (the premise)"
        );

        // …and the disposal lands between that read and the write.
        assert!(
            discard_space_if_pristine(&conn, "space-race")
                .await
                .unwrap(),
            "the untouched space is disposed of"
        );

        assert!(
            !update_space_title(&conn, "space-race", "Tides", 2_000)
                .await
                .unwrap(),
            "the title write strikes nothing, and says so"
        );

        let err = insert_participant(
            &conn,
            "p-race",
            "space",
            Some("space-race"),
            None,
            "agent",
            "Ada",
            None,
            None,
            "explicit",
            "member",
            None,
            2_000,
        )
        .await
        .expect_err("a roster add into a space that is gone must fail its foreign key");
        assert!(matches!(err, AppError::Database { .. }), "{err:?}");
    }

    #[tokio::test]
    async fn a_failed_participant_insert_still_marks_its_space() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        instantiate_template(
            &conn,
            crate::config::DEFAULT_TEMPLATE_ID,
            "space-mark",
            None,
            "unlinked",
            1_000,
        )
        .await
        .unwrap();
        assert!(
            pristine_space_ids(&conn)
                .await
                .unwrap()
                .contains(&"space-mark".to_string()),
            "a fresh instantiation is pristine (the premise)"
        );

        // The insert fails on the primary key — a stand-in for every way a
        // statement after the stamp can refuse (a full disk, a constraint a
        // future column adds).
        let taken = HUMAN_PARTICIPANT_ID;
        let err = insert_participant(
            &conn,
            taken,
            "space",
            Some("space-mark"),
            None,
            "agent",
            "Ada",
            None,
            None,
            "explicit",
            "member",
            None,
            2_000,
        )
        .await
        .expect_err("inserting a participant at a taken id must fail");
        assert!(matches!(err, AppError::Database { .. }), "{err:?}");

        assert!(
            !pristine_space_ids(&conn)
                .await
                .unwrap()
                .contains(&"space-mark".to_string()),
            "the space is marked anyway — the stamp precedes the statement it \
             covers, so no interleaving can see a durable change beside an \
             untouched space"
        );
    }

    #[tokio::test]
    async fn a_failed_instantiation_leaves_no_space_at_all() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;

        // The seeded default template already owns an agent, so the rollback
        // has copied rows of both kinds to take back.
        let owned_before =
            list_template_owned_participants(&conn, crate::config::DEFAULT_TEMPLATE_ID)
                .await
                .unwrap();
        assert!(
            !owned_before.is_empty(),
            "the default template owns an agent"
        );

        let raw = db.connect().unwrap(); // FKs off — the staging connection
        raw.execute(
            "INSERT INTO space_template_participant \
             (template_id, participant_id, participant_scope, role, joined_at) \
             VALUES (?1, 'no-such-participant', 'global', 'member', 1)",
            (Value::Text(crate::config::DEFAULT_TEMPLATE_ID.to_string()),),
        )
        .await
        .unwrap();

        let err = instantiate_template(
            &conn,
            crate::config::DEFAULT_TEMPLATE_ID,
            "space-partial",
            None,
            "unlinked",
            1_000,
        )
        .await
        .expect_err("copying a reference to a participant that does not exist must fail");
        assert!(matches!(err, AppError::Database { .. }), "{err:?}");

        // Zero state, all three ways it could have been left behind.
        assert!(
            get_space(&conn, "space-partial").await.unwrap().is_none(),
            "the space row is rolled back with everything else"
        );
        assert!(
            space_participants(&conn, "space-partial")
                .await
                .unwrap()
                .is_empty(),
            "and no membership survives it"
        );
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM participant WHERE owner_space_id = ?1",
                (Value::Text("space-partial".to_string()),),
            )
            .await
            .unwrap();
        let owned_after: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(owned_after, 0, "nor any agent copied for it");
    }

    /// PR #221 review comment 2: the template-update replacement is atomic. A
    /// failure mid-replacement must roll back — the owned participant set (and
    /// the settings) are left exactly as they were, so a concurrent reader
    /// never sees a destroyed/partial set and no `Change::Templates` fires
    /// (the caller emits only after this returns `Ok`).
    #[tokio::test]
    async fn update_template_tx_rolls_back_on_failure() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space_template(&conn, "t-atomic", "Orig", 4, None, 1_000)
            .await
            .unwrap();
        for (pid, label, model) in [("a1", "One", "m1"), ("a2", "Two", "m2")] {
            insert_participant(
                &conn,
                pid,
                "template",
                None,
                Some("t-atomic"),
                "agent",
                label,
                Some(model),
                None,
                "explicit",
                "member",
                None,
                1_000,
            )
            .await
            .unwrap();
        }

        // Replacement whose SECOND participant has an invalid notify_policy —
        // its insert violates the CHECK mid-transaction, forcing a rollback.
        let bad: Vec<TemplateParticipantInput> = vec![
            ("New1".into(), Some("nm1".into()), None, "explicit".into()),
            ("New2".into(), Some("nm2".into()), None, "bogus".into()),
        ];
        let r = update_template_tx(
            &conn,
            "t-atomic",
            Some("Renamed"),
            Some(9),
            Some(&bad),
            2_000,
        )
        .await;
        assert!(r.is_err(), "a mid-replacement CHECK violation must fail");

        // Atomicity: owned set unchanged (not destroyed, not partially rebuilt).
        let after = list_template_owned_participants(&conn, "t-atomic")
            .await
            .unwrap();
        let labels: Vec<&str> = after.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(labels, vec!["One", "Two"], "owned set survives rollback");
        // …and the settings rolled back too (one transaction).
        let tmpl = get_space_template(&conn, "t-atomic")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tmpl.title, "Orig");
        assert_eq!(tmpl.cascade_limit, 4);
    }

    #[tokio::test]
    async fn update_template_tx_commits_a_valid_replacement() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space_template(&conn, "t-ok", "Orig", 4, None, 1_000)
            .await
            .unwrap();
        insert_participant(
            &conn,
            "a1",
            "template",
            None,
            Some("t-ok"),
            "agent",
            "Old",
            Some("m"),
            None,
            "explicit",
            "member",
            None,
            1_000,
        )
        .await
        .unwrap();

        let new: Vec<TemplateParticipantInput> =
            vec![("Fresh".into(), Some("nm".into()), None, "human".into())];
        update_template_tx(&conn, "t-ok", Some("Renamed"), Some(7), Some(&new), 2_000)
            .await
            .unwrap();

        let after = list_template_owned_participants(&conn, "t-ok")
            .await
            .unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].label, "Fresh");
        assert_eq!(after[0].notify_policy, "human");
        let tmpl = get_space_template(&conn, "t-ok").await.unwrap().unwrap();
        assert_eq!(tmpl.title, "Renamed");
        assert_eq!(tmpl.cascade_limit, 7);
    }

    #[tokio::test]
    async fn stale_user_version_is_refused() {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();
        // Simulate a database from an older, incompatible schema version.
        set_user_version(&conn, 1).await.unwrap();
        let err = initialize(&conn).await.expect_err("stale version refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("incompatible") && msg.contains("delete"),
            "stale-db error must be honest about deleting the dev database: {msg}"
        );
    }

    #[tokio::test]
    async fn migrations_match_schema() {
        let fresh_db = open_memory_fresh().await;
        let migrated_db = open_memory_migrated().await;
        let fresh = fresh_db.connect().unwrap();
        let migrated = migrated_db.connect().unwrap();

        let fresh_objects = list_objects(&fresh).await;
        let migrated_objects = list_objects(&migrated).await;
        assert_eq!(
            fresh_objects, migrated_objects,
            "schema objects differ:\n  fresh:    {fresh_objects:?}\n  migrated: {migrated_objects:?}",
        );

        for (obj_type, name) in &fresh_objects {
            match obj_type.as_str() {
                "table" => {
                    let fresh_cols = table_columns(&fresh, name).await;
                    let migrated_cols = table_columns(&migrated, name).await;
                    assert_eq!(
                        fresh_cols, migrated_cols,
                        "column mismatch in table '{name}':\n  fresh:    {fresh_cols:?}\n  migrated: {migrated_cols:?}",
                    );
                }
                "index" => {
                    let fresh_cols = index_columns(&fresh, name).await;
                    let migrated_cols = index_columns(&migrated, name).await;
                    assert_eq!(
                        fresh_cols, migrated_cols,
                        "index column mismatch for '{name}':\n  fresh:    {fresh_cols:?}\n  migrated: {migrated_cols:?}",
                    );
                }
                "view" => {
                    let fresh_sql = view_sql(&fresh, name).await;
                    let migrated_sql = view_sql(&migrated, name).await;
                    assert_eq!(
                        fresh_sql, migrated_sql,
                        "view SQL mismatch for '{name}':\n--- schema.sql ---\n{fresh_sql}\n--- migrations ---\n{migrated_sql}",
                    );
                }
                _ => {}
            }
        }
    }

    /// **The removal's decision belongs to its write** (Codex review, PR #279).
    ///
    /// `remove_space_participant` used to read the participant, see a
    /// space-owned row, and *then* soft-remove it. Between those two steps
    /// another window's promotion can commit — and the interleave is played out
    /// literally below, with the same two primitives the old code used. The
    /// stale soft-remove lands on the now-**global** row: the agent is retired
    /// from the library the instant it was shared, its brand-new notebook is
    /// left unarchived (only retirement archives it), and neither caller saw an
    /// error. Promotion is one-way, so there is no way back.
    ///
    /// The cure is not a fresher read but no read at all: the ownership test
    /// rides in the soft-remove's own `WHERE`, so after the same interleave the
    /// new door strikes nothing and ends the membership instead — leaving the
    /// shared agent alive with the reference row promotion created marked
    /// `left_at`.
    #[tokio::test]
    async fn a_removal_decided_before_a_promotion_cannot_retire_the_shared_agent() {
        async fn fixture(conn: &Connection, space: &str, agent: &str) {
            insert_space(conn, space, Some("Home"), "unlinked", 1)
                .await
                .unwrap();
            insert_participant(
                conn,
                agent,
                "space",
                Some(space),
                None,
                "agent",
                "Cartographer",
                None,
                None,
                "explicit",
                "member",
                None,
                1,
            )
            .await
            .unwrap();
        }
        async fn promote(conn: &Connection, space: &str, agent: &str, notebook: &str) {
            promote_participant_tx(
                conn,
                &Promotion {
                    participant_id: agent,
                    home_space_id: space,
                    role: "member",
                    notebook_space_id: notebook,
                    notebook_title: "Cartographer — notebook",
                    persona: None,
                    grant: None,
                    now: 2,
                },
            )
            .await
            .unwrap();
        }

        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;

        // --- the old shape, played out ------------------------------------
        fixture(&conn, "s-old", "p-old").await;
        let read = get_participant(&conn, "p-old").await.unwrap().unwrap();
        assert_eq!(read.scope, "space", "the read that decided");
        promote(&conn, "s-old", "p-old", "nb-old").await;
        // ...and the write it decided on, arriving after the share landed.
        assert!(
            soft_remove_participant(&conn, "p-old", 3).await.unwrap(),
            "the unguarded soft-remove reports success"
        );
        let wrecked = get_participant(&conn, "p-old").await.unwrap().unwrap();
        assert_eq!(wrecked.scope, "global");
        assert!(
            wrecked.removed_at.is_some(),
            "the shared agent was retired by a removal aimed at a space-owned one"
        );
        assert!(
            !space_is_archived(&conn, "nb-old").await.unwrap(),
            "and its notebook was orphaned — nothing archived it"
        );

        // --- the same interleave, through the door that decides at the write
        fixture(&conn, "s-new", "p-new").await;
        let read = get_participant(&conn, "p-new").await.unwrap().unwrap();
        assert_eq!(read.scope, "space");
        promote(&conn, "s-new", "p-new", "nb-new").await;
        assert_eq!(
            remove_space_participant_tx(&conn, "s-new", "p-new", 3)
                .await
                .unwrap()
                .outcome,
            SpaceRemoval::Left,
            "the premise expired, so the removal ends the membership instead"
        );
        let survivor = get_participant(&conn, "p-new").await.unwrap().unwrap();
        assert_eq!(survivor.scope, "global");
        assert!(
            survivor.removed_at.is_none(),
            "the shared agent survives its home space's removal"
        );
        assert!(
            space_participants(&conn, "s-new")
                .await
                .unwrap()
                .iter()
                .all(|p| p.participant_id != "p-new"),
            "and it is no longer a member of the space it left"
        );

        // The ordinary case still soft-removes: a row this space really owns.
        fixture(&conn, "s-owned", "p-owned").await;
        assert_eq!(
            remove_space_participant_tx(&conn, "s-owned", "p-owned", 3)
                .await
                .unwrap()
                .outcome,
            SpaceRemoval::SoftRemoved
        );
        assert!(
            get_participant(&conn, "p-owned")
                .await
                .unwrap()
                .unwrap()
                .removed_at
                .is_some()
        );
        // And a second removal has nothing left to end.
        assert_eq!(
            remove_space_participant_tx(&conn, "s-owned", "p-owned", 4)
                .await
                .unwrap()
                .outcome,
            SpaceRemoval::NothingToDo
        );
    }

    /// **A join decides at its write** (Codex review, PR #279). The caller reads
    /// the participant first — for its typed refusals — but a read cannot answer
    /// for a write that lands after it: a retirement committing in between left
    /// a durable membership row joined *after* retirement. The old shape's two
    /// halves are played out around a retirement below, then the same interleave
    /// goes through the guarded insert.
    #[tokio::test]
    async fn a_join_decided_before_a_retirement_inserts_no_membership() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space(&conn, "s", Some("Home"), "unlinked", 1)
            .await
            .unwrap();
        insert_participant(
            &conn, "g", "global", None, None, "agent", "Ada", None, None, "explicit", "member",
            None, 1,
        )
        .await
        .unwrap();

        // The read that decided: live and global.
        let read = get_participant(&conn, "g").await.unwrap().unwrap();
        assert!(read.removed_at.is_none());
        // ...the retirement...
        assert!(soft_remove_participant(&conn, "g", 2).await.unwrap());
        // ...and the write it decided on, now guarded.
        assert!(
            !ensure_space_participant(&conn, "s", "g", "member", 3)
                .await
                .unwrap(),
            "the premise expired, so the insert strikes nothing"
        );
        assert!(
            space_participants(&conn, "s")
                .await
                .unwrap()
                .iter()
                .all(|m| m.participant_id != "g"),
            "and no membership row was left behind"
        );

        // The ordinary case still joins, and stays idempotent.
        insert_participant(
            &conn, "h", "global", None, None, "agent", "Bo", None, None, "explicit", "member",
            None, 1,
        )
        .await
        .unwrap();
        assert!(
            ensure_space_participant(&conn, "s", "h", "member", 3)
                .await
                .unwrap()
        );
        assert!(
            !ensure_space_participant(&conn, "s", "h", "member", 4)
                .await
                .unwrap(),
            "re-adding an existing member changes nothing"
        );
    }

    /// **A departure is not an absence.** Leaving is soft, so the row stays on
    /// the space's primary key; an insert-only join struck nothing and the
    /// membership could never be given back (Codex review, PR #280). The three
    /// things the upsert has to get right, in one place: the revive, the
    /// premise still riding it, and the live row it must not touch.
    #[tokio::test]
    async fn a_membership_that_left_is_revived_and_a_live_one_is_left_alone() {
        let db = open_memory_fresh().await;
        let conn = fk_conn(&db).await;
        insert_space(&conn, "s", Some("Home"), "unlinked", 1)
            .await
            .unwrap();
        for (id, label) in [("g", "Ada"), ("r", "Cy"), ("t", "User")] {
            insert_participant(
                &conn, id, "global", None, None, "agent", label, None, None, "explicit", "member",
                None, 1,
            )
            .await
            .unwrap();
        }

        // It joined, then left.
        assert!(
            ensure_space_participant(&conn, "s", "g", "member", 2)
                .await
                .unwrap()
        );
        assert!(leave_space_participant(&conn, "s", "g", 3).await.unwrap());
        assert!(
            space_participants(&conn, "s")
                .await
                .unwrap()
                .iter()
                .all(|m| m.participant_id != "g"),
            "a member that left is not a member"
        );

        // Inviting it back revives that row — one membership, the new role.
        assert!(
            ensure_space_participant(&conn, "s", "g", "observer", 4)
                .await
                .unwrap(),
            "the membership came back, so the write reports a change"
        );
        let roster = space_participants(&conn, "s").await.unwrap();
        let back: Vec<_> = roster.iter().filter(|m| m.participant_id == "g").collect();
        assert_eq!(back.len(), 1, "revived, not duplicated");
        assert_eq!(back[0].role, "observer", "the requested role rides it");

        // The premise rides the revive as it rides the insert: a retired
        // participant's departed row stays departed.
        assert!(
            ensure_space_participant(&conn, "s", "r", "member", 5)
                .await
                .unwrap()
        );
        assert!(leave_space_participant(&conn, "s", "r", 6).await.unwrap());
        assert!(soft_remove_participant(&conn, "r", 7).await.unwrap());
        assert!(
            !ensure_space_participant(&conn, "s", "r", "member", 8)
                .await
                .unwrap(),
            "a retired participant cannot be revived into a space either"
        );
        assert!(
            space_participants(&conn, "s")
                .await
                .unwrap()
                .iter()
                .all(|m| m.participant_id != "r")
        );

        // And a **live** membership is not rewritten — what the template
        // instantiation's copied human reference depends on (it is inserted
        // with the template's own role and overrides, and the `ensure` that
        // follows must leave both standing).
        insert_participant_ref(
            &conn,
            "space_participant",
            "space_id",
            "s",
            "t",
            "member",
            9,
            &ParticipantRefRow {
                participant_id: "t".into(),
                role: "member".into(),
                joined_at: 9,
                override_label: Some("Scribe".into()),
                override_model_ref: None,
                override_system_prompt: None,
                override_notify_policy: None,
            },
            true,
        )
        .await
        .unwrap();
        assert!(
            !ensure_space_participant(&conn, "s", "t", "owner", 10)
                .await
                .unwrap(),
            "a live membership is already what it is"
        );
        let kept = space_participants(&conn, "s")
            .await
            .unwrap()
            .into_iter()
            .find(|m| m.participant_id == "t")
            .expect("still a member");
        assert_eq!(kept.role, "member", "its role was not overwritten");
        assert_eq!(
            kept.label, "Scribe",
            "nor were the overrides it was inserted with"
        );
    }

    /// **Deciding at the write means reserving the writer.**
    ///
    /// Several of app-core's transactions read before they write: they check a
    /// scope, a membership, a `left_at`, and choose what to write from what they
    /// found (task 36's promotion guard, task 37's promote-or-join grant, the
    /// notebook-owner removal guard). That shape is only worth anything if the read
    /// and the write are one act against one snapshot — and a *deferred* `BEGIN`
    /// (what plain `BEGIN` means here as in SQLite) reserves nothing.
    ///
    /// Measured on turso at our pin, two live connections on one `Database`
    /// (`AppCore::db_conn` mints a fresh connection per call, so this is an
    /// in-process race, not merely a cross-process one):
    ///
    /// * `BEGIN` — A reads, B writes and commits without blocking A, and A's
    ///   write comes back `BusySnapshot("database snapshot is stale, rollback
    ///   and retry the transaction")`. `busy_timeout` cannot rescue it: a stale
    ///   snapshot is not something you can wait out.
    /// * `BEGIN IMMEDIATE` — A reserves the writer; B waits on its own `BEGIN
    ///   IMMEDIATE` for exactly A's hold, then acquires and reads A's committed
    ///   state, deciding against *that*.
    ///
    /// So every transaction goes through `db::begin_write`. These tests pin the
    /// consequence at the layer the finding is about (Codex review, PR #280).
    ///
    /// **They live beside the writers rather than in `tests/`** because the
    /// transactions they drive are crate-private — an exported one is the write
    /// reachable without its guards (see
    /// `the_raw_db_writers_are_not_exported`), and an integration test is an
    /// external consumer like any other.
    mod tx_contention {
        use crate::db;

        /// **A result that describes the commit point cannot be overtaken.** The grant
        /// answers with the membership it wrote, read inside its own transaction — so a
        /// removal or retirement landing immediately afterwards changes what the roster
        /// says next, but not what the grant reported. Answering from a read *after*
        /// the commit put a failure message beside committed work, including (for a
        /// space-owned candidate) an irreversible promotion — the very state deciding
        /// at the write exists to prevent (Codex review, PR #280).
        #[test]
        fn a_grants_answer_describes_its_commit_not_the_roster_afterwards() {
            let dir = tempfile::tempdir().expect("tempdir");
            let runtime = tokio::runtime::Runtime::new().expect("runtime");

            runtime.block_on(async move {
                let database = db::open(dir.path()).await.expect("open");
                let conn = db::connect(&database).await.expect("connect");
                db::insert_space(&conn, "home", Some("Home"), "unlinked", 1)
                    .await
                    .expect("home");
                db::insert_space(&conn, "here", Some("Here"), "unlinked", 1)
                    .await
                    .expect("destination");
                db::insert_participant(
                    &conn,
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

                let outcome =
                    db::grant_space_membership_tx(&conn, "here", "agent", "observer", "notebook-id", 2)
                        .await
                        .expect("the grant");
                assert!(matches!(
                    outcome.decision,
                    db::GrantDecision::Promoted { .. }
                ));
                assert_eq!(outcome.member.role, "observer");
                assert_eq!(outcome.member.label, "Mara");

                // Another window ends the membership the instant the grant commits —
                // the window a post-commit read would have raced.
                db::remove_space_participant_tx(&conn, "here", "agent", 3)
                    .await
                    .expect("the other window's removal");

                // The roster a caller would have re-read now says nothing about it…
                assert!(
                    db::space_participants(&conn, "here")
                        .await
                        .expect("roster")
                        .iter()
                        .all(|m| m.participant_id != "agent"),
                    "the hazard is real: a read taken now finds no member"
                );
                // …while the grant's own answer still describes what it committed, and
                // the promotion it made is still there to be seen.
                assert_eq!(outcome.member.role, "observer");
                let promoted = db::get_participant(&conn, "agent")
                    .await
                    .expect("read")
                    .expect("the row");
                assert_eq!(
                    promoted.scope, "global",
                    "the irreversible half committed — which is why its call must not have reported failure"
                );
            });
        }

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
                    decision.decision,
                    db::GrantDecision::Joined,
                    "it decided against the row it found: already global, so plain membership"
                );
                assert_eq!(
                    decision.member.role, "observer",
                    "and it answered with the membership as of its own commit"
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
    }
}
