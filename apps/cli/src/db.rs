use std::path::PathBuf;

use turso::{Builder, Connection, Database};

const SCHEMA: &str = include_str!("../schema.sql");
const LATEST_VERSION: i64 = 1;

/// Returns the path to the CLI database file.
fn db_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("eidolons").join("eidolons.db"))
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
// Migrations
// ---------------------------------------------------------------------------

const MIGRATION_1: &str = "
PRAGMA foreign_keys = ON;

-- Issuer key registry
CREATE TABLE issuer_key (
    key_id          TEXT PRIMARY KEY,
    epoch           TEXT NOT NULL,
    params_hash     TEXT NOT NULL,
    public_key_data BLOB NOT NULL,
    params_data     BLOB NOT NULL,
    expires_at      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- Pre-credential: append-only log of in-flight protocol states
CREATE TABLE pre_credential (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL CHECK (type IN ('issuance', 'refund')),
    credential_id   TEXT REFERENCES credential(id),
    issuer_key_id   TEXT NOT NULL REFERENCES issuer_key(key_id),
    data            BLOB NOT NULL,
    credits         INTEGER,
    spend_amount    INTEGER,
    created_at      TEXT NOT NULL,

    CHECK (
        (type = 'issuance' AND credential_id IS NULL AND spend_amount IS NULL)
        OR
        (type = 'refund'   AND credential_id IS NOT NULL AND spend_amount IS NOT NULL)
    )
);

CREATE UNIQUE INDEX idx_one_spend_per_credential
    ON pre_credential (credential_id)
    WHERE type = 'refund';

-- Credential: immutable materialized CreditToken
CREATE TABLE credential (
    id                  TEXT PRIMARY KEY,
    pre_credential_id   TEXT NOT NULL UNIQUE
                        REFERENCES pre_credential(id),
    issuer_key_id       TEXT NOT NULL
                        REFERENCES issuer_key(key_id),
    data                BLOB NOT NULL,
    nonce               BLOB NOT NULL,
    credits             INTEGER NOT NULL,
    generation          INTEGER NOT NULL DEFAULT 0,
    expires_at          TEXT,
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_credential_fefo
    ON credential (expires_at, created_at)
    WHERE expires_at IS NOT NULL;

-- Lifecycle view
CREATE VIEW credential_lifecycle AS
SELECT
    c.id,
    c.credits,
    c.generation,
    c.expires_at,
    c.created_at,
    c.issuer_key_id,
    CASE
        WHEN c.expires_at IS NOT NULL
             AND c.expires_at < datetime('now')     THEN 'expired'
        WHEN pc_spend.id IS NULL                    THEN 'active'
        WHEN c_next.id IS NULL                      THEN 'spending'
        ELSE                                             'spent'
    END AS state,
    pc_spend.id             AS pending_spend_id,
    pc_spend.spend_amount   AS spend_amount,
    c_next.id               AS successor_id
FROM credential c
LEFT JOIN pre_credential pc_spend
    ON  pc_spend.credential_id = c.id
    AND pc_spend.type = 'refund'
LEFT JOIN credential c_next
    ON  c_next.pre_credential_id = pc_spend.id;
";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    async fn open_memory_fresh() -> Database {
        let db = Builder::new_local(":memory:")
            .build()
            .await
            .unwrap();
        let conn = db.connect().unwrap();
        initialize(&conn).await.unwrap();
        db
    }

    async fn open_memory_migrated() -> Database {
        let db = Builder::new_local(":memory:")
            .build()
            .await
            .unwrap();
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
            objects.push((
                row.get::<String>(0).unwrap(),
                row.get::<String>(1).unwrap(),
            ));
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
                row.get::<String>(1).unwrap(),          // name
                row.get::<String>(2).unwrap(),          // type
                row.get::<i64>(3).unwrap() != 0,        // notnull
                row.get::<Option<String>>(4).unwrap(),  // dflt_value
                row.get::<i64>(5).unwrap() != 0,        // pk
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
        rows.next().await.unwrap().unwrap().get::<String>(0).unwrap()
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
