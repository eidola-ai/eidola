use std::path::Path;

use turso::{Builder, Connection, Database, Value};

use crate::error::AppError;

const SCHEMA: &str = include_str!("../schema/schema.sql");
const LATEST_VERSION: i64 = 1;

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

    let conn = db.connect().map_err(AppError::db)?;
    initialize(&conn).await?;

    Ok(db)
}

/// Initialize: fresh install gets schema.sql directly, existing databases run
/// incremental migrations.
async fn initialize(conn: &Connection) -> Result<(), AppError> {
    let version = get_user_version(conn).await?;

    if version == 0 {
        conn.execute_batch(SCHEMA)
            .await
            .map_err(|e| AppError::Database {
                message: format!("schema init failed: {e}"),
            })?;
        set_user_version(conn, LATEST_VERSION).await?;
    } else {
        migrate(conn, version).await?;
    }

    Ok(())
}

async fn migrate(conn: &Connection, current_version: i64) -> Result<(), AppError> {
    if current_version < 1 {
        conn.execute_batch(MIGRATION_1)
            .await
            .map_err(|e| AppError::Database {
                message: format!("migration 1 failed: {e}"),
            })?;
        set_user_version(conn, 1).await?;
    }

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
}

pub async fn insert_request(conn: &Connection, entry: &Request) -> Result<(), AppError> {
    conn.execute(
        "INSERT INTO request (id, connection_id, action_id, method, path, \
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

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Participant operations
// ---------------------------------------------------------------------------

pub async fn ensure_participant(
    conn: &Connection,
    kind: &str,
    label: &str,
    provider_id: Option<&str>,
    created_at: i64,
) -> Result<String, AppError> {
    let mut stmt = conn
        .prepare("SELECT id FROM participant WHERE kind = ?1 AND label = ?2 LIMIT 1")
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
    .map_err(|e| AppError::Database {
        message: format!("failed to insert participant: {e}"),
    })?;
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

pub async fn insert_space_participant(
    conn: &Connection,
    space_id: &str,
    participant_id: &str,
    role: &str,
    joined_at: i64,
) -> Result<(), AppError> {
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
    .map_err(|e| AppError::Database {
        message: format!("failed to insert space_participant: {e}"),
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 2 — Semantic: Action operations
// ---------------------------------------------------------------------------

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

pub async fn insert_action(conn: &Connection, entry: &ActionEntry) -> Result<(), AppError> {
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
) -> Result<(), AppError> {
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
    .map_err(|e| AppError::Database {
        message: format!("failed to insert action_antecedent: {e}"),
    })?;
    Ok(())
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
/// `action.created_at` in the space (falling back to the space's own
/// `created_at` for empty spaces); `message_count` counts terminal
/// (`complete`/`cancelled`) actions.
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
    let filter = if include_archived {
        ""
    } else {
        "WHERE s.archived_at IS NULL "
    };
    let sql = format!(
        "SELECT s.id, s.title, s.created_at, s.archived_at, \
                COALESCE(MAX(a.created_at), s.created_at) AS last_activity_at, \
                COUNT(a.id) AS message_count \
         FROM space s \
         LEFT JOIN action a ON a.space_id = s.id \
              AND a.status IN ('complete', 'cancelled') \
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

/// First text content block of the first user_input action in a space —
/// the raw source for the listing snippet shown for untitled spaces.
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
             ORDER BY a.created_at ASC, cb.ordinal ASC \
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
    pub action_type: String,
    pub participant_kind: String,
    pub status: String,
    pub text_content: Option<String>,
    pub block_ordinal: Option<i64>,
}

/// Returns actions in a space with their text content blocks, suitable for
/// building the OpenAI messages array. Filters to terminal statuses and
/// uses action_resolved to dereference origin references.
pub async fn get_space_actions_for_context(
    conn: &Connection,
    space_id: &str,
) -> Result<Vec<SpaceActionRow>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT ar.action_id, ar.action_type, p.kind, ar.status, \
                    cb.text_content, cb.ordinal \
             FROM action_resolved ar \
             JOIN participant p ON p.id = ar.participant_id \
             LEFT JOIN content_block cb ON cb.action_id = ar.content_source_id \
             WHERE ar.space_id = ?1 \
               AND ar.status IN ('complete', 'cancelled') \
             ORDER BY ar.created_at ASC, cb.ordinal ASC",
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
            action_type: row.get::<String>(1).map_err(AppError::db)?,
            participant_kind: row.get::<String>(2).map_err(AppError::db)?,
            status: row.get::<String>(3).map_err(AppError::db)?,
            text_content: row.get::<Option<String>>(4).map_err(AppError::db)?,
            block_ordinal: row.get::<Option<i64>>(5).map_err(AppError::db)?,
        });
    }
    Ok(results)
}

/// Returns the ID of the last terminal action in a space (for antecedent linking).
pub async fn last_action_in_space(
    conn: &Connection,
    space_id: &str,
) -> Result<Option<String>, AppError> {
    let mut stmt = conn
        .prepare(
            "SELECT id FROM action \
             WHERE space_id = ?1 AND status IN ('complete', 'cancelled') \
             ORDER BY created_at DESC LIMIT 1",
        )
        .await
        .map_err(AppError::db)?;
    let mut rows = stmt
        .query([Value::Text(space_id.to_string())])
        .await
        .map_err(AppError::db)?;
    match rows.next().await.map_err(AppError::db)? {
        None => Ok(None),
        Some(row) => Ok(Some(row.get::<String>(0).map_err(AppError::db)?)),
    }
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
