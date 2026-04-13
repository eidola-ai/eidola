use std::path::PathBuf;

use turso::{Builder, Connection, Database, Value};

const SCHEMA: &str = include_str!("../schema/schema.sql");
const LATEST_VERSION: i64 = 1;

/// Returns the path to the CLI database file.
fn db_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("eidola").join("eidola.db"))
}

/// Opens (or creates) the local database and runs any pending migrations.
pub async fn open() -> Result<Database, String> {
    let path = db_path().ok_or("could not determine data directory")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create data directory: {e}"))?;
    }

    let db = Builder::new_local(path.to_string_lossy().as_ref())
        .build()
        .await
        .map_err(|e| format!("failed to open database: {e}"))?;

    let conn = db
        .connect()
        .map_err(|e| format!("failed to connect: {e}"))?;

    initialize(&conn).await?;

    Ok(db)
}

/// Initializes the database: fresh install gets schema.sql directly,
/// existing databases run incremental migrations.
async fn initialize(conn: &Connection) -> Result<(), String> {
    let version = get_user_version(conn).await?;

    if version == 0 {
        // Fresh database — apply canonical schema directly
        conn.execute_batch(SCHEMA)
            .await
            .map_err(|e| format!("schema init failed: {e}"))?;
        set_user_version(conn, LATEST_VERSION).await?;
    } else {
        // Existing database — run incremental migrations
        migrate(conn, version).await?;
    }

    Ok(())
}

/// Runs forward-only migrations from `current_version` to `LATEST_VERSION`.
async fn migrate(conn: &Connection, current_version: i64) -> Result<(), String> {
    if current_version < 1 {
        conn.execute_batch(MIGRATION_1)
            .await
            .map_err(|e| format!("migration 1 failed: {e}"))?;
        set_user_version(conn, 1).await?;
    }

    Ok(())
}

async fn get_user_version(conn: &Connection) -> Result<i64, String> {
    let mut stmt = conn
        .prepare("PRAGMA user_version")
        .await
        .map_err(|e| format!("failed to query user_version: {e}"))?;
    let mut rows = stmt
        .query(())
        .await
        .map_err(|e| format!("failed to query user_version: {e}"))?;
    let row = rows
        .next()
        .await
        .map_err(|e| format!("failed to read user_version: {e}"))?
        .ok_or("no user_version row")?;
    row.get::<i64>(0)
        .map_err(|e| format!("failed to read user_version value: {e}"))
}

async fn set_user_version(conn: &Connection, version: i64) -> Result<(), String> {
    conn.execute(&format!("PRAGMA user_version = {version}"), ())
        .await
        .map_err(|e| format!("failed to set user_version: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 0 — Wallet: Issuer key operations
// ---------------------------------------------------------------------------

/// Upsert an issuer key (insert or ignore if already exists).
pub async fn upsert_issuer_key(
    conn: &Connection,
    id: &str,
    params_hash: &str,
    public_key_data: &[u8],
    params_data: &[u8],
    expires_at: i64,
    created_at: i64,
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to upsert issuer key: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 0 — Wallet: Pre-credential operations
// ---------------------------------------------------------------------------

/// Insert a pre-credential record for issuance.
pub async fn insert_pre_credential_issuance(
    conn: &Connection,
    id: &str,
    issuer_key_id: &str,
    data: &[u8],
    credits: i64,
    created_at: i64,
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to insert pre_credential: {e}"))?;
    Ok(())
}

/// Insert a pre-credential record for a refund (spend checkpoint).
pub async fn insert_pre_credential_refund(
    conn: &Connection,
    id: &str,
    credential_nonce: &str,
    issuer_key_id: &str,
    data: &[u8],
    spend_amount: i64,
    created_at: i64,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO pre_credential (id, type, credential_nonce, issuer_key_id, data, credits, spend_amount, created_at) \
         VALUES (?1, 'refund', ?2, ?3, ?4, NULL, ?5, ?6)",
        (
            Value::Text(id.to_string()),
            Value::Text(credential_nonce.to_string()),
            Value::Text(issuer_key_id.to_string()),
            Value::Blob(data.to_vec()),
            Value::Integer(spend_amount),
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| format!("failed to insert refund pre_credential: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 0 — Wallet: Credential operations
// ---------------------------------------------------------------------------

/// Insert a completed credential.
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
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to insert credential: {e}"))?;
    Ok(())
}

/// A spendable credential with associated key data.
#[allow(dead_code)]
pub struct SpendableCredential {
    pub nonce: String,
    pub issuer_key_id: String,
    pub data: Vec<u8>,
    pub credits: i64,
    pub generation: i64,
    pub public_key_data: Vec<u8>,
}

/// Find an active credential with at least `min_credits`, returning it with key data.
pub async fn find_spendable_credential(
    conn: &Connection,
    min_credits: i64,
) -> Result<Option<SpendableCredential>, String> {
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
        .map_err(|e| format!("failed to prepare query: {e}"))?;
    let mut rows = stmt
        .query([min_credits])
        .await
        .map_err(|e| format!("failed to query credentials: {e}"))?;
    match rows
        .next()
        .await
        .map_err(|e| format!("failed to read row: {e}"))?
    {
        None => Ok(None),
        Some(row) => Ok(Some(SpendableCredential {
            nonce: row
                .get::<String>(0)
                .map_err(|e| format!("failed to read nonce: {e}"))?,
            issuer_key_id: row
                .get::<String>(1)
                .map_err(|e| format!("failed to read issuer_key_id: {e}"))?,
            data: row
                .get::<Vec<u8>>(2)
                .map_err(|e| format!("failed to read data: {e}"))?,
            credits: row
                .get::<i64>(3)
                .map_err(|e| format!("failed to read credits: {e}"))?,
            generation: row
                .get::<i64>(4)
                .map_err(|e| format!("failed to read generation: {e}"))?,
            public_key_data: row
                .get::<Vec<u8>>(5)
                .map_err(|e| format!("failed to read public_key_data: {e}"))?,
        })),
    }
}

/// A row from the credential_lifecycle view.
#[allow(dead_code)]
pub struct CredentialRow {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub created_at: i64,
    pub state: String,
}

/// List active credentials (not expired, not spent).
pub async fn list_active_credentials(conn: &Connection) -> Result<Vec<CredentialRow>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT nonce, credits, generation, created_at, state \
             FROM credential_lifecycle WHERE state = 'active' \
             ORDER BY created_at",
        )
        .await
        .map_err(|e| format!("failed to prepare query: {e}"))?;
    let mut rows = stmt
        .query(())
        .await
        .map_err(|e| format!("failed to query credentials: {e}"))?;
    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| format!("failed to read row: {e}"))?
    {
        results.push(CredentialRow {
            nonce: row
                .get::<String>(0)
                .map_err(|e| format!("failed to read nonce: {e}"))?,
            credits: row
                .get::<i64>(1)
                .map_err(|e| format!("failed to read credits: {e}"))?,
            generation: row
                .get::<i64>(2)
                .map_err(|e| format!("failed to read generation: {e}"))?,
            created_at: row
                .get::<i64>(3)
                .map_err(|e| format!("failed to read created_at: {e}"))?,
            state: row
                .get::<String>(4)
                .map_err(|e| format!("failed to read state: {e}"))?,
        });
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Provider operations
// ---------------------------------------------------------------------------

/// Find a provider by name, or create one if it doesn't exist. Returns the id.
pub async fn ensure_provider(
    conn: &Connection,
    name: &str,
    kind: &str,
    created_at: i64,
) -> Result<String, String> {
    // Try to find existing
    let mut stmt = conn
        .prepare("SELECT id FROM provider WHERE name = ?1 AND kind = ?2 LIMIT 1")
        .await
        .map_err(|e| format!("failed to prepare provider query: {e}"))?;
    let mut rows = stmt
        .query((Value::Text(name.to_string()), Value::Text(kind.to_string())))
        .await
        .map_err(|e| format!("failed to query provider: {e}"))?;
    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| format!("failed to read provider row: {e}"))?
    {
        return row
            .get::<String>(0)
            .map_err(|e| format!("failed to read provider id: {e}"));
    }
    drop(rows);
    drop(stmt);

    // Create new
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
    .map_err(|e| format!("failed to insert provider: {e}"))?;
    Ok(id)
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Attestation operations
// ---------------------------------------------------------------------------

/// Insert an attestation record (ignore if hash already exists).
pub async fn upsert_attestation(
    conn: &Connection,
    hash: &str,
    doc: &[u8],
    pcr_digest: Option<&str>,
    created_at: i64,
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to upsert attestation: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Connection operations
// ---------------------------------------------------------------------------

/// Insert a new connection record.
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
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to insert connection: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 1 — Transport: Request log operations
// ---------------------------------------------------------------------------

/// A request/response log entry.
pub struct RequestLogEntry {
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
}

/// Insert a request log entry.
pub async fn insert_request_log(conn: &Connection, entry: &RequestLogEntry) -> Result<(), String> {
    conn.execute(
        "INSERT INTO request_log (id, connection_id, action_id, method, path, \
         request_headers, request_body, response_status, response_headers, response_body, \
         request_at, response_at, duration_ms, error, credential_nonce, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        (
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
        ),
    )
    .await
    .map_err(|e| format!("failed to insert request_log: {e}"))?;
    Ok(())
}

fn opt_text(v: &Option<String>) -> Value {
    match v {
        Some(s) => Value::Text(s.clone()),
        None => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Participant operations
// ---------------------------------------------------------------------------

/// Find a participant by kind+label, or create one. Returns the id.
pub async fn ensure_participant(
    conn: &Connection,
    kind: &str,
    label: &str,
    provider_id: Option<&str>,
    created_at: i64,
) -> Result<String, String> {
    let mut stmt = conn
        .prepare("SELECT id FROM participant WHERE kind = ?1 AND label = ?2 LIMIT 1")
        .await
        .map_err(|e| format!("failed to prepare participant query: {e}"))?;
    let mut rows = stmt
        .query((
            Value::Text(kind.to_string()),
            Value::Text(label.to_string()),
        ))
        .await
        .map_err(|e| format!("failed to query participant: {e}"))?;
    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| format!("failed to read participant row: {e}"))?
    {
        return row
            .get::<String>(0)
            .map_err(|e| format!("failed to read participant id: {e}"));
    }
    drop(rows);
    drop(stmt);

    let id = uuid::Uuid::now_v7().to_string();
    conn.execute(
        "INSERT INTO participant (id, kind, label, provider_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
        (
            Value::Text(id.clone()),
            Value::Text(kind.to_string()),
            Value::Text(label.to_string()),
            match provider_id {
                Some(p) => Value::Text(p.to_string()),
                None => Value::Null,
            },
            Value::Integer(created_at),
        ),
    )
    .await
    .map_err(|e| format!("failed to insert participant: {e}"))?;
    Ok(id)
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Space operations
// ---------------------------------------------------------------------------

/// Create a new space.
pub async fn insert_space(
    conn: &Connection,
    id: &str,
    title: Option<&str>,
    linkability: &str,
    created_at: i64,
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to insert space: {e}"))?;
    Ok(())
}

/// Add a participant to a space.
pub async fn insert_space_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    joined_at: i64,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO space_participant (space_id, participant_id, role, joined_at) \
         VALUES (?1, ?2, ?3, ?4)",
        (
            Value::Text(space_id.to_string()),
            Value::Text(participant_id.to_string()),
            Value::Text(role.to_string()),
            Value::Integer(joined_at),
        ),
    )
    .await
    .map_err(|e| format!("failed to insert space_participant: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Action operations
// ---------------------------------------------------------------------------

/// An action to insert.
pub struct ActionEntry {
    pub id: String,
    pub space_id: String,
    pub participant_id: String,
    pub action_type: String,
    pub status: String,
    pub intent: Option<String>,
    pub model: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub credits_consumed: Option<i64>,
    pub created_at: i64,
}

/// Insert an action.
pub async fn insert_action(conn: &Connection, entry: &ActionEntry) -> Result<(), String> {
    conn.execute(
        "INSERT INTO action (id, space_id, participant_id, action_type, status, \
         intent, model, input_tokens, output_tokens, credits_consumed, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        (
            Value::Text(entry.id.clone()),
            Value::Text(entry.space_id.clone()),
            Value::Text(entry.participant_id.clone()),
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
    .map_err(|e| format!("failed to insert action: {e}"))?;
    Ok(())
}

/// Insert an antecedent edge in the action causal graph.
pub async fn insert_action_antecedent(
    conn: &Connection,
    action_id: &str,
    antecedent_action_id: &str,
    ordinal: i64,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO action_antecedent (action_id, antecedent_action_id, ordinal) \
         VALUES (?1, ?2, ?3)",
        (
            Value::Text(action_id.to_string()),
            Value::Text(antecedent_action_id.to_string()),
            Value::Integer(ordinal),
        ),
    )
    .await
    .map_err(|e| format!("failed to insert action_antecedent: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Content block operations
// ---------------------------------------------------------------------------

/// Insert a text content block.
pub async fn insert_text_content_block(
    conn: &Connection,
    id: &str,
    action_id: &str,
    ordinal: i64,
    block_type: &str,
    text_content: &str,
) -> Result<(), String> {
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
    .map_err(|e| format!("failed to insert content_block: {e}"))?;
    Ok(())
}

/// Insert a tool_use content block.
#[allow(dead_code)]
pub async fn insert_tool_use_content_block(
    conn: &Connection,
    id: &str,
    action_id: &str,
    ordinal: i64,
    tool_name: &str,
    tool_call_id: &str,
    data: Option<&str>,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO content_block (id, action_id, ordinal, block_type, tool_name, tool_call_id, data) \
         VALUES (?1, ?2, ?3, 'tool_use', ?4, ?5, ?6)",
        (
            Value::Text(id.to_string()),
            Value::Text(action_id.to_string()),
            Value::Integer(ordinal),
            Value::Text(tool_name.to_string()),
            Value::Text(tool_call_id.to_string()),
            match data {
                Some(d) => Value::Text(d.to_string()),
                None => Value::Null,
            },
        ),
    )
    .await
    .map_err(|e| format!("failed to insert tool_use content_block: {e}"))?;
    Ok(())
}

/// Insert a tool_result content block.
#[allow(dead_code)]
pub async fn insert_tool_result_content_block(
    conn: &Connection,
    id: &str,
    action_id: &str,
    ordinal: i64,
    tool_call_id: &str,
    text_content: Option<&str>,
    data: Option<&str>,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO content_block (id, action_id, ordinal, block_type, tool_call_id, text_content, data) \
         VALUES (?1, ?2, ?3, 'tool_result', ?4, ?5, ?6)",
        (
            Value::Text(id.to_string()),
            Value::Text(action_id.to_string()),
            Value::Integer(ordinal),
            Value::Text(tool_call_id.to_string()),
            match text_content {
                Some(t) => Value::Text(t.to_string()),
                None => Value::Null,
            },
            match data {
                Some(d) => Value::Text(d.to_string()),
                None => Value::Null,
            },
        ),
    )
    .await
    .map_err(|e| format!("failed to insert tool_result content_block: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Migrations
// ---------------------------------------------------------------------------

const MIGRATION_1: &str = include_str!("../schema/schema.sql");

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    async fn open_memory_fresh() -> Database {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();
        initialize(&conn).await.unwrap();
        db
    }

    async fn open_memory_migrated() -> Database {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();
        migrate(&conn, 0).await.unwrap();
        set_user_version(&conn, LATEST_VERSION).await.unwrap();
        db
    }

    /// List all table and view names from sqlite_master, sorted.
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

    /// Dump columns for a table via PRAGMA table_info, sorted by name.
    /// Returns (name, type, notnull, dflt_value, pk) tuples.
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
                row.get::<String>(1).unwrap(),         // name
                row.get::<String>(2).unwrap(),         // type
                row.get::<i64>(3).unwrap() != 0,       // notnull
                row.get::<Option<String>>(4).unwrap(), // dflt_value
                row.get::<i64>(5).unwrap() != 0,       // pk
            ));
        }
        cols.sort_by(|a, b| a.0.cmp(&b.0));
        cols
    }

    /// Dump index columns via PRAGMA index_info, sorted by name.
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

    /// Get the SQL for a view from sqlite_master.
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
        // Layer 0
        assert!(table_names.contains(&"issuer_key"));
        assert!(table_names.contains(&"pre_credential"));
        assert!(table_names.contains(&"credential"));
        // Layer 1
        assert!(table_names.contains(&"provider"));
        assert!(table_names.contains(&"attestation"));
        assert!(table_names.contains(&"connection"));
        // Layer 2
        assert!(table_names.contains(&"participant"));
        assert!(table_names.contains(&"space"));
        assert!(table_names.contains(&"action"));
        assert!(table_names.contains(&"content_block"));
        assert!(table_names.contains(&"request_log"));
    }

    #[tokio::test]
    async fn initialize_is_idempotent() {
        let db = open_memory_fresh().await;
        let conn = db.connect().unwrap();

        initialize(&conn).await.unwrap();
        assert_eq!(get_user_version(&conn).await.unwrap(), LATEST_VERSION);
    }

    #[tokio::test]
    async fn migrations_match_schema() {
        let fresh_db = open_memory_fresh().await;
        let migrated_db = open_memory_migrated().await;
        let fresh = fresh_db.connect().unwrap();
        let migrated = migrated_db.connect().unwrap();

        // 1. Same set of objects (tables, views, indexes)
        let fresh_objects = list_objects(&fresh).await;
        let migrated_objects = list_objects(&migrated).await;
        assert_eq!(
            fresh_objects, migrated_objects,
            "schema objects differ:\n  fresh:    {fresh_objects:?}\n  migrated: {migrated_objects:?}",
        );

        for (obj_type, name) in &fresh_objects {
            match obj_type.as_str() {
                "table" => {
                    // 2. Same columns (name, type, notnull, pk) per table
                    let fresh_cols = table_columns(&fresh, name).await;
                    let migrated_cols = table_columns(&migrated, name).await;
                    assert_eq!(
                        fresh_cols, migrated_cols,
                        "column mismatch in table '{name}':\n  fresh:    {fresh_cols:?}\n  migrated: {migrated_cols:?}",
                    );
                }
                "index" => {
                    // 3. Same index columns
                    let fresh_cols = index_columns(&fresh, name).await;
                    let migrated_cols = index_columns(&migrated, name).await;
                    assert_eq!(
                        fresh_cols, migrated_cols,
                        "index column mismatch for '{name}':\n  fresh:    {fresh_cols:?}\n  migrated: {migrated_cols:?}",
                    );
                }
                "view" => {
                    // 4. Same view SQL
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
