use std::path::{Path, PathBuf};

use turso::{Builder, Connection, Database, Value};

use crate::error::AppError;

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
const LATEST_VERSION: i64 = 5;

/// Well-known id of the shared human "You" participant — the single
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
/// human "You" or to an agent participant would be a lie in the Record.
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

/// Seed the shared human "You" participant and the "Default" space template
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

    // The one shared human "You" — a GLOBAL participant, referenced into every
    // instantiated space (instantiate_template ensures the reference).
    conn.execute(
        "INSERT OR IGNORE INTO participant \
         (id, scope, kind, label, notify_policy, role, created_at) \
         VALUES (?1, 'global', 'human', 'You', 'explicit', 'owner', ?2)",
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
    conn.execute("BEGIN", ()).await.map_err(AppError::db)?;
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
// Layer 2 — Semantic: Participants (scope-owned) + space templates
//
// Every participant row has exactly one scope: 'global' (the shared library —
// today "You"), 'space' (owned by one space), or 'template' (owned by one
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
pub async fn update_participant_config(
    conn: &Connection,
    id: &str,
    label: Option<&str>,
    model_ref: Option<Option<&str>>,
    system_prompt: Option<Option<&str>>,
    notify_policy: Option<&str>,
    role: Option<&str>,
) -> Result<bool, AppError> {
    let mut sets: Vec<String> = Vec::new();
    let mut params: Vec<Value> = vec![Value::Text(id.to_string())];
    if let Some(l) = label {
        params.push(Value::Text(l.to_string()));
        sets.push(format!("label = ?{}", params.len()));
    }
    if let Some(m) = model_ref {
        params.push(opt_str(m));
        sets.push(format!("model_ref = ?{}", params.len()));
    }
    if let Some(s) = system_prompt {
        params.push(opt_str(s));
        sets.push(format!("system_prompt = ?{}", params.len()));
    }
    if let Some(n) = notify_policy {
        params.push(Value::Text(n.to_string()));
        sets.push(format!("notify_policy = ?{}", params.len()));
    }
    if let Some(r) = role {
        params.push(Value::Text(r.to_string()));
        sets.push(format!("role = ?{}", params.len()));
    }
    if sets.is_empty() {
        return Ok(false);
    }
    let sql = format!("UPDATE participant SET {} WHERE id = ?1", sets.join(", "));
    let n = conn.execute(&sql, params).await.map_err(AppError::db)?;
    Ok(n > 0)
}

/// Soft-remove a participant row (global: library soft-remove; owned:
/// left/deactivated). The row survives so `action.participant_id` references
/// stay resolvable.
pub async fn soft_remove_participant(
    conn: &Connection,
    id: &str,
    now: i64,
) -> Result<bool, AppError> {
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

/// Promote a space-owned agent to a **global** identity, in place and in one
/// transaction. Same row, same id — so authorship, provenance and memory
/// continuity are structural rather than stitched. Copy-projection was
/// rejected: it fragments past posts, memory blocks and reference edges across
/// two identities, which is the opposite of what promotion exists to create.
///
/// Three writes, all or nothing:
///
/// 1. `owner_space_id → NULL`, `scope → 'global'` on the participant row. The
///    pinned `participant_scope` echo on every past `action` and every
///    `memory_block` follows via `ON UPDATE CASCADE` — see the schema comments
///    there and `turso_enforcement_smoke` case (e).
/// 2. A `space_participant` reference row for the **former owner space**, with
///    NULL overrides. Ownership no longer implies membership, and NULL
///    overrides mean the effective config (`COALESCE(override, config)`) is
///    byte-identical to what it was: the space's persona is preserved exactly.
/// 3. The agent's private **notebook space** (`space.notebook_participant_id`),
///    referenced into by the agent itself, hidden from the Library listing.
///
/// The caller has already validated *what* may be promoted (kind, scope, the
/// shared "You", removal); this is the mechanics.
pub async fn promote_participant_tx(
    conn: &Connection,
    participant_id: &str,
    home_space_id: &str,
    role: &str,
    notebook_space_id: &str,
    notebook_title: &str,
    now: i64,
) -> Result<(), AppError> {
    conn.execute("BEGIN", ()).await.map_err(AppError::db)?;
    match promote_participant_tx_body(
        conn,
        participant_id,
        home_space_id,
        role,
        notebook_space_id,
        notebook_title,
        now,
    )
    .await
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

async fn promote_participant_tx_body(
    conn: &Connection,
    participant_id: &str,
    home_space_id: &str,
    role: &str,
    notebook_space_id: &str,
    notebook_title: &str,
    now: i64,
) -> Result<(), AppError> {
    // Both key columns move in one statement: the three-way scope/owner CHECK
    // must hold at statement end, and the cascade fires off the (id, scope)
    // parent key.
    let n = conn
        .execute(
            "UPDATE participant SET scope = 'global', owner_space_id = NULL \
             WHERE id = ?1 AND scope = 'space'",
            (Value::Text(participant_id.to_string()),),
        )
        .await
        .map_err(AppError::db)?;
    if n == 0 {
        return Err(AppError::Database {
            message: format!("participant {participant_id} was not a space-owned row to promote"),
        });
    }
    insert_space_participant(conn, home_space_id, participant_id, role, now).await?;
    insert_notebook_space(conn, notebook_space_id, participant_id, notebook_title, now).await?;
    insert_space_participant(conn, notebook_space_id, participant_id, "owner", now).await?;
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
    conn.execute(
        "INSERT INTO space (id, parent_space_id, title, linkability, \
                            notebook_participant_id, created_at) \
         VALUES (?1, NULL, ?2, 'unlinked', ?3, ?4)",
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

/// Whether a participant is a member of one space — **owned row ∪ live
/// reference row**, the same membership definition [`participant_spaces`] and
/// [`space_participants`] read from their two sides, asked of a single space.
/// This is the cross-space ACL (task 37): reference *creation* and reference
/// *following* both gate on it.
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
                 WHERE r.participant_id = ?2 AND r.space_id = ?1 AND r.left_at IS NULL \
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
                    COALESCE(sp.override_label, p.label), a.action_type \
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
    }))
}

/// Every space a participant is a member of — **owned rows ∪ live references**,
/// the same membership definition [`space_participants`] reads from the other
/// side. This is the boundary for everything cross-space: task 36's
/// `list_my_spaces` reaches exactly these and nothing else.
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
                   WHERE r.participant_id = ?1 AND r.space_id = s.id AND r.left_at IS NULL \
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
        });
    }
    Ok(out)
}

// --- reference tables (references to globals + overrides) -------------------

/// Reference a global into a space (pinned `participant_scope='global'`, no
/// overrides). The common membership-add.
pub async fn insert_space_participant(
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

/// Ensure a global is referenced into a space (idempotent — INSERT OR IGNORE on
/// the PK). Used to guarantee "You" joins every instantiated space even if a
/// copied template reference already added it.
pub async fn ensure_space_participant(
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
        true,
    )
    .await
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
pub async fn update_space_participant_override(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    override_label: Option<Option<&str>>,
    override_model_ref: Option<Option<&str>>,
    override_system_prompt: Option<Option<&str>>,
    override_notify_policy: Option<Option<&str>>,
) -> Result<bool, AppError> {
    let mut sets: Vec<String> = Vec::new();
    let mut params: Vec<Value> = vec![
        Value::Text(space_id.to_string()),
        Value::Text(participant_id.to_string()),
    ];
    for (col, val) in [
        ("override_label", override_label),
        ("override_model_ref", override_model_ref),
        ("override_system_prompt", override_system_prompt),
        ("override_notify_policy", override_notify_policy),
    ] {
        if let Some(inner) = val {
            params.push(opt_str(inner));
            sets.push(format!("{col} = ?{}", params.len()));
        }
    }
    if sets.is_empty() {
        return Ok(false);
    }
    let sql = format!(
        "UPDATE space_participant SET {} WHERE space_id = ?1 AND participant_id = ?2",
        sets.join(", ")
    );
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

/// End a space membership (reference) — set `left_at`. For a referenced global;
/// owned participants are removed via [`soft_remove_participant`].
pub async fn leave_space_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    now: i64,
) -> Result<bool, AppError> {
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
/// members (the referenced "You") sort first, then others by id.
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
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE space SET cascade_limit = ?2 WHERE id = ?1",
            (
                Value::Text(space_id.to_string()),
                Value::Integer(cascade_limit),
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
) -> Result<bool, AppError> {
    let n = conn
        .execute(
            "UPDATE space SET router_model = ?2 WHERE id = ?1",
            (Value::Text(space_id.to_string()), opt_str(router_model)),
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
pub async fn update_template_tx(
    conn: &Connection,
    id: &str,
    title: Option<&str>,
    cascade_limit: Option<i64>,
    participants: Option<&[TemplateParticipantInput]>,
    now: i64,
) -> Result<(), AppError> {
    conn.execute("BEGIN", ()).await.map_err(AppError::db)?;
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
/// references; then ensure the shared human "You" is referenced (as owner).
/// Errors if the template is missing or removed.
pub async fn instantiate_template(
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
            true, // OR IGNORE — a template that already references "You" won't
                  // collide with the ensure below.
        )
        .await?;
    }

    // The shared human "You" joins every instantiated space (idempotent).
    ensure_space_participant(conn, new_space_id, HUMAN_PARTICIPANT_ID, "owner", now).await?;
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
}

impl ReferenceEdgeRow {
    /// Whether this edge's quoted passage may be shown or sent — a post's
    /// `text` block and nothing else (see [`crate::Inner::validate_reference_spec`]
    /// for why an edge could be otherwise).
    pub fn is_quotable(&self) -> bool {
        is_post_action_type(&self.antecedent_action_type)
            && self.block_type.as_deref() == Some(QUOTABLE_BLOCK_TYPE)
    }
}

/// The `reference`-relation antecedents of an action, ordinal order. Used by
/// `edit_post` to replicate references onto a new generation and by the
/// upstream-context embed expansion.
pub async fn reference_antecedents(
    conn: &Connection,
    action_id: &str,
) -> Result<Vec<ReferenceEdgeRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT aa.ordinal, aa.antecedent_action_id, aa.content_block_id, \
                    aa.range_start, aa.range_end, aa.annotation, cb.text_content, \
                    ant.action_type, cb.block_type \
             FROM action_antecedent aa \
             JOIN action ant ON ant.id = aa.antecedent_action_id \
             LEFT JOIN content_block cb ON cb.id = aa.content_block_id \
             WHERE aa.action_id = ?1 AND aa.relation = 'reference' \
             ORDER BY aa.ordinal ASC",
        )
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

/// One incoming `reference` edge: a post (elsewhere or in the same space)
/// referencing the queried action. Restricted to referrers that are their
/// item's **current generation** with terminal status — the set whose quotes
/// should highlight the source. The target is the *concrete generation*
/// queried (references never remap to tips).
pub struct IncomingReferenceRow {
    /// The referring post's action id (a current generation).
    pub action_id: String,
    /// The referring post's space (references may cross spaces).
    pub space_id: String,
    pub ordinal: i64,
    pub content_block_id: Option<String>,
    pub range_start: Option<i64>,
    pub range_end: Option<i64>,
    pub annotation: Option<String>,
    pub created_at: i64,
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
                    aa.range_start, aa.range_end, aa.annotation, ar.created_at \
             FROM action_antecedent aa \
             JOIN action_resolved ar ON ar.action_id = aa.action_id \
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

pub struct SpaceRow {
    pub id: String,
    pub title: Option<String>,
    pub created_at: i64,
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
         {filter}\
         GROUP BY s.id, s.title, s.created_at, s.archived_at \
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

pub async fn get_space(conn: &Connection, space_id: &str) -> Result<Option<SpaceRow>, AppError> {
    let mut stmt = conn
        .prepare("SELECT id, title, created_at FROM space WHERE id = ?1")
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
                    cb.text_content, cb.ordinal \
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
                        cb.text_content, cb.ordinal \
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
pub const POST_ACTION_TYPES_SQL: &str = "'user_input', 'inference'";

/// The same set, for the checks that happen in Rust rather than in SQL — the
/// reference gate (what a quote may name) and the read-side filters that back
/// it up. Kept beside [`POST_ACTION_TYPES_SQL`] so the two cannot drift.
pub const POST_ACTION_TYPES: [&str; 2] = ["user_input", "inference"];

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
        "SELECT ar.action_id, ar.item_id, p.kind, p.label, ar.action_type, \
                ar.model, ar.credits_consumed, ar.generation, ar.created_at \
         FROM action_resolved ar \
         JOIN participant p ON p.id = ar.participant_id \
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
                ant.action_type, qcb.block_type \
         FROM action_antecedent aa \
         JOIN action_resolved ar ON ar.action_id = aa.action_id \
         JOIN action ant ON ant.id = aa.antecedent_action_id \
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
        });
    }

    Ok(SpaceTreeData {
        actions,
        blocks,
        edges,
    })
}

/// Returns the ID of the last terminal action in a space (for antecedent linking).
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
    let sql = format!(
        "SELECT id FROM action \
         WHERE space_id = ?1 AND status IN ('complete', 'cancelled') \
           AND action_type IN ({POST_ACTION_TYPES_SQL}) \
         ORDER BY created_at DESC LIMIT 1"
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
/// item's generations (a `user_input` item is always human, an `inference` item
/// always its agent), so an edit or regeneration never changes the count.
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

pub async fn archive_space(
    conn: &Connection,
    space_id: &str,
    archived_at: i64,
) -> Result<bool, AppError> {
    let changed = conn
        .execute(
            "UPDATE space SET archived_at = ?2 WHERE id = ?1 AND archived_at IS NULL",
            (
                Value::Text(space_id.to_string()),
                Value::Integer(archived_at),
            ),
        )
        .await
        .map_err(|e| AppError::Database {
            message: format!("failed to archive space: {e}"),
        })?;
    Ok(changed > 0)
}

pub async fn update_space_title(
    conn: &Connection,
    space_id: &str,
    title: &str,
) -> Result<(), AppError> {
    conn.execute(
        "UPDATE space SET title = ?2 WHERE id = ?1",
        (
            Value::Text(space_id.to_string()),
            Value::Text(title.to_string()),
        ),
    )
    .await
    .map_err(|e| AppError::Database {
        message: format!("failed to update space title: {e}"),
    })?;
    Ok(())
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

    async fn add_user_action(
        conn: &Connection,
        space_id: &str,
        participant_id: &str,
        text: &str,
        created_at: i64,
    ) {
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
        assert!(archive_space(&conn, "space-d", 6_000).await.unwrap());

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

        // "You" is a GLOBAL participant.
        let you = get_participant(&conn, HUMAN_PARTICIPANT_ID)
            .await
            .unwrap()
            .expect("You seeded");
        assert_eq!(you.scope, "global");
        assert_eq!(you.kind, "human");
        assert_eq!(you.label, "You");
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
            )
            .await
            .unwrap()
        );
        update_participant_config(
            &conn,
            &agent.participant_id,
            Some("Justin"),
            Some(Some("kimi-k2-6")),
            Some(Some("Be terse.")),
            Some("all"),
            None,
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
            set_space_router_model(&conn, "space-x", Some("router@local"))
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

        // No override → inherits the global's label "You".
        let m = space_participants(&conn, "space-o").await.unwrap();
        assert_eq!(m[0].label, "You");

        // '' override → effective empty (override to empty, not inherit).
        update_space_participant_override(
            &conn,
            "space-o",
            HUMAN_PARTICIPANT_ID,
            Some(Some("")),
            None,
            None,
            None,
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
        )
        .await
        .unwrap();
        let m = space_participants(&conn, "space-o").await.unwrap();
        assert_eq!(m[0].label, "You", "NULL override inherits again");
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
}
