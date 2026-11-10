//! Database connection pool and query helpers.

use std::time::SystemTime;

use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::credentials::SealedRefundToken;
use crate::error::ServerError;

/// A row from the `account` table.
pub struct AccountRow {
    pub id: Uuid,
    pub secret_hash: String,
    pub stripe_customer_id: Option<String>,
    pub billing_state: String,
    pub created_at: SystemTime,
}

pub use eidola_billing::connection::create_pool;

/// Maximum tolerated clock skew between this server and the database.
/// Anything beyond this is treated as a misconfigured time source and
/// causes time-sensitive operations to fail closed.
pub const MAX_CLOCK_SKEW: std::time::Duration = std::time::Duration::from_secs(10);

/// Log-safe rendering of a `tokio_postgres` error: its `Display` (a fixed
/// string per error kind — "db error", "connection closed", …) plus the
/// SQLSTATE code when one exists. Never its `Debug`: that prints the
/// server-authored message and DETAIL, and Postgres quotes row values
/// there — a unique-violation DETAIL carries the conflicting key's
/// values verbatim. The SQLSTATE alone (23505 unique_violation, 23514
/// check_violation, …) names the failure class from a bounded vocabulary
/// without carrying any value.
fn db_error_summary(e: &tokio_postgres::Error) -> String {
    match e.code() {
        Some(code) => format!("{e} (sqlstate {})", code.code()),
        None => e.to_string(),
    }
}

/// Verify that this server's wall clock and the database's wall clock
/// agree to within `max_skew`. Returns the measured absolute drift on
/// success. Errors if the query fails or the drift exceeds `max_skew`.
///
/// The check brackets `SELECT clock_timestamp()` between two
/// `SystemTime::now()` reads on the server and compares the database
/// time against the *midpoint* of the bracket. This bounds the noise
/// contributed by network latency to RTT/2, which is several orders
/// of magnitude below any reasonable skew threshold even on
/// transcontinental links.
///
/// Use this as a precondition for any code path that writes
/// time-anchored state derived from the server's clock (e.g.,
/// issuer key rotation). Per-request code paths that only consume
/// already-stored validity windows do not need to call this.
pub async fn check_clock_skew(
    pool: &Pool,
    max_skew: std::time::Duration,
) -> Result<std::time::Duration, ServerError> {
    use crate::telemetry::metrics;

    let client = pool.get().await.map_err(|e| {
        metrics::DB_CLOCK_SKEW_CHECK_FAILURES
            .add(1, &[opentelemetry::KeyValue::new("reason", "pool")]);
        ServerError::Internal(format!("db pool error: {e}"))
    })?;

    let t1 = SystemTime::now();
    let row = client
        .query_one("SELECT clock_timestamp()", &[])
        .await
        .map_err(|e| {
            metrics::DB_CLOCK_SKEW_CHECK_FAILURES
                .add(1, &[opentelemetry::KeyValue::new("reason", "query")]);
            ServerError::Internal(format!("clock skew query failed: {}", db_error_summary(&e)))
        })?;
    let t2 = SystemTime::now();

    let db_time: SystemTime = row.get(0);
    let half_rtt = t2.duration_since(t1).unwrap_or_default() / 2;
    let server_mid = t1 + half_rtt;

    // Compute the signed drift (positive = db ahead) for the gauge,
    // and the unsigned magnitude for the threshold comparison.
    let (drift, signed_secs, direction) = match db_time.duration_since(server_mid) {
        Ok(d) => (d, d.as_secs_f64(), "ahead of"),
        Err(e) => {
            let d = e.duration();
            (d, -d.as_secs_f64(), "behind")
        }
    };

    metrics::DB_CLOCK_SKEW_SECONDS.record(signed_secs, &[]);

    if drift > max_skew {
        metrics::DB_CLOCK_SKEW_CHECK_FAILURES
            .add(1, &[opentelemetry::KeyValue::new("reason", "exceeded")]);
        return Err(ServerError::Internal(format!(
            "database clock is {direction} server by {drift:?}, \
             which exceeds the {max_skew:?} maximum tolerated skew. \
             Check NTP/chrony on this host."
        )));
    }

    tracing::debug!("db clock check: {direction} server by {drift:?}");
    Ok(drift)
}

/// Spawn a background task that issues `SELECT 1` against the pool on a
/// fixed interval. This serves two purposes when running against a
/// serverless Postgres like Neon: it keeps the upstream compute from
/// autosuspending during quiet periods, and it ensures `deadpool` always
/// has at least one warm, recently-validated connection so the first
/// real request after an idle stretch does not pay a cold-start penalty.
pub fn spawn_keepalive(pool: Pool, interval: std::time::Duration) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Skip the immediate first tick — startup already touches the pool.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match pool.get().await {
                Ok(client) => {
                    if let Err(e) = client.execute("SELECT 1", &[]).await {
                        tracing::warn!("db keepalive query failed: {e}");
                    }
                }
                Err(e) => tracing::warn!("db keepalive pool checkout failed: {e}"),
            }
        }
    });
}

/// Insert a new account and return its `created_at` timestamp.
pub async fn insert_account(
    pool: &Pool,
    id: Uuid,
    credential_hash: &str,
) -> Result<SystemTime, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one(
            "INSERT INTO account (id, secret_hash) VALUES ($1, $2) RETURNING created_at",
            &[&id, &credential_hash],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!("insert account failed: {}", db_error_summary(&e)))
        })?;

    Ok(row.get::<_, SystemTime>("created_at"))
}

/// Retrieve an account by ID.
#[tracing::instrument(skip_all, name = "db.get_account", err)]
pub async fn get_account_by_id(pool: &Pool, id: Uuid) -> Result<AccountRow, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_opt(
            "SELECT id, secret_hash, stripe_customer_id, billing_state, created_at \
             FROM account WHERE id = $1",
            &[&id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!("query account failed: {}", db_error_summary(&e)))
        })?;

    match row {
        Some(row) => Ok(AccountRow {
            id: row.get("id"),
            secret_hash: row.get("secret_hash"),
            stripe_customer_id: row.get("stripe_customer_id"),
            billing_state: row.get("billing_state"),
            created_at: row.get("created_at"),
        }),
        None => Err(ServerError::NotFound {
            message: "account not found".to_string(),
        }),
    }
}

/// Set the Stripe customer ID on an account (only if currently NULL).
///
/// Returns the customer ID that is now set on the account. If another request
/// raced and set it first, the existing value is returned instead of the
/// provided one.
pub async fn set_stripe_customer_id(
    pool: &Pool,
    id: Uuid,
    customer_id: &str,
) -> Result<String, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows_updated = client
        .execute(
            "UPDATE account SET stripe_customer_id = $1, updated_at = now() \
             WHERE id = $2 AND stripe_customer_id IS NULL",
            &[&customer_id, &id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "update stripe customer failed: {}",
                db_error_summary(&e)
            ))
        })?;

    if rows_updated == 1 {
        return Ok(customer_id.to_string());
    }

    // Another request may have set it — re-read the current value.
    let account = get_account_by_id(pool, id).await?;
    match account.stripe_customer_id {
        Some(existing) => Ok(existing),
        None => Err(ServerError::Internal(
            "failed to set stripe_customer_id".to_string(),
        )),
    }
}

/// Retrieve an account by its Stripe customer ID.
pub async fn get_account_by_stripe_customer(
    pool: &Pool,
    customer_id: &str,
) -> Result<Option<AccountRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_opt(
            "SELECT id, secret_hash, stripe_customer_id, billing_state, created_at \
             FROM account WHERE stripe_customer_id = $1",
            &[&customer_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query account by customer failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(row.map(|row| AccountRow {
        id: row.get("id"),
        secret_hash: row.get("secret_hash"),
        stripe_customer_id: row.get("stripe_customer_id"),
        billing_state: row.get("billing_state"),
        created_at: row.get("created_at"),
    }))
}

/// Test funding uses the same purchase operation as production; it grants a
/// free lot so an issuance fixture needs no real payment-provider object.
#[cfg(test)]
pub(crate) async fn insert_credit_ledger(
    pool: &Pool,
    account_id: Uuid,
    credits: i64,
    _reason: &str,
    source_id: &str,
    _payment_intent: Option<&str>,
    expires_at: Option<SystemTime>,
) -> Result<bool, ServerError> {
    use std::time::{Duration, UNIX_EPOCH};
    let purchased_at = UNIX_EPOCH
        + Duration::from_secs(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
    let purchase = eidola_billing::store::Purchase {
        id: source_id.into(),
        account_id,
        charge_id: None,
        payment_intent: None,
        invoice_id: None,
        currency: "usd".into(),
        paid_minor: 0,
        credits,
        purchased_at,
        expires_at: expires_at.unwrap_or(purchased_at + Duration::from_secs(365 * 86400)),
        kind: "purchase".into(),
    };
    Ok(eidola_billing::store::Store::new(pool.clone())
        .record_purchase(&purchase)
        .await?)
}

// --- Account acceptance / required document queries ---

/// The currently required version of one legal document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequiredDocumentRow {
    pub document: String,
    pub version: i64,
    pub sha256: String,
    pub url: String,
}

/// Outcome of recording an observed document version.
#[derive(Debug, PartialEq, Eq)]
pub enum RecordRequiredOutcome {
    /// A new (document, version) row was recorded; the requirement history
    /// grew (and, if this is the highest version, the gate advanced).
    Recorded,
    /// This exact (document, version, sha256) was already on record — a
    /// no-op that preserves the original `first_required_at`.
    AlreadyRecorded,
    /// A row for this (document, version) exists **with a different
    /// hash** — the published bytes changed without a version increment,
    /// violating the versioning contract CI enforces. Nothing is written;
    /// the stored hash remains authoritative.
    HashConflict { stored_sha256: String },
}

/// Record that a document version was observed as published. Append-only:
/// one row per (document, version), first observation wins, rows are never
/// updated — so the table is a complete history of what was required and
/// since when, and the current requirement (the highest version per
/// document) can never regress no matter how stale an observation is.
pub async fn record_required_document(
    pool: &Pool,
    doc: &RequiredDocumentRow,
) -> Result<RecordRequiredOutcome, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows = client
        .execute(
            "INSERT INTO required_document (document, version, sha256, url) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (document, version) DO NOTHING",
            &[&doc.document, &doc.version, &doc.sha256, &doc.url],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "record required_document failed: {}",
                db_error_summary(&e)
            ))
        })?;

    if rows == 1 {
        return Ok(RecordRequiredOutcome::Recorded);
    }

    let row = client
        .query_one(
            "SELECT sha256 FROM required_document WHERE document = $1 AND version = $2",
            &[&doc.document, &doc.version],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query required_document failed: {}",
                db_error_summary(&e)
            ))
        })?;
    let stored_sha256: String = row.get("sha256");

    if stored_sha256 == doc.sha256 {
        Ok(RecordRequiredOutcome::AlreadyRecorded)
    } else {
        Ok(RecordRequiredOutcome::HashConflict { stored_sha256 })
    }
}

/// One `engine_placement` row: where an operator says an engine deployment
/// of a model can be reached. Availability only, never trust (see
/// `engine_router::placement`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnginePlacementRow {
    pub model_id: String,
    /// The deployment's config hash (`engine-enclaves.json`'s
    /// `config_sha256`).
    pub deployment: String,
    pub base_url: String,
    pub enabled: bool,
}

/// Every `engine_placement` row. Only the named columns are read, so a column
/// a later gateway adds does not disturb this one: the table is shared by
/// every gateway version running against the database, and grows additively.
pub async fn get_engine_placement(pool: &Pool) -> Result<Vec<EnginePlacementRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows = client
        .query(
            "SELECT model_id, deployment, base_url, enabled \
             FROM engine_placement ORDER BY model_id, base_url",
            &[],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query engine_placement failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(rows
        .iter()
        .map(|r| EnginePlacementRow {
            model_id: r.get("model_id"),
            deployment: r.get("deployment"),
            base_url: r.get("base_url"),
            enabled: r.get("enabled"),
        })
        .collect())
}

/// The currently required version of every gated document — the highest
/// recorded version per document. Empty when the acceptance gate is
/// disabled (nothing seeded or polled yet).
pub async fn get_required_documents(pool: &Pool) -> Result<Vec<RequiredDocumentRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows = client
        .query(
            "SELECT DISTINCT ON (document) document, version, sha256, url \
             FROM required_document ORDER BY document, version DESC",
            &[],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query required_document failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(rows
        .iter()
        .map(|r| RequiredDocumentRow {
            document: r.get("document"),
            version: r.get("version"),
            sha256: r.get("sha256"),
            url: r.get("url"),
        })
        .collect())
}

/// Record acceptance of a document version. Append-only and idempotent —
/// re-accepting an already-accepted version is a no-op that preserves the
/// original timestamp.
pub async fn insert_acceptance(
    pool: &Pool,
    account_id: Uuid,
    document: &str,
    version: i64,
    sha256: &str,
) -> Result<(), ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    client
        .execute(
            "INSERT INTO account_acceptance (account_id, document, version, sha256) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT DO NOTHING",
            &[&account_id, &document, &version, &sha256],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "insert acceptance failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(())
}

/// The highest version of each document the account has ever accepted.
pub async fn get_accepted_versions(
    pool: &Pool,
    account_id: Uuid,
) -> Result<Vec<(String, i64)>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows = client
        .query(
            "SELECT document, MAX(version) AS version \
             FROM account_acceptance WHERE account_id = $1 GROUP BY document",
            &[&account_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query acceptances failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(rows
        .iter()
        .map(|r| (r.get("document"), r.get("version")))
        .collect())
}

// --- Issuer Key queries ---

/// A row from the `issuer_key` table.
pub struct IssuerKeyRow {
    pub id: String,
    pub private_key_enc: Vec<u8>,
    pub public_key: Vec<u8>,
    pub domain_separator: String,
    pub issue_from: SystemTime,
    pub issue_until: SystemTime,
    pub accept_until: SystemTime,
}

/// Insert a new issuer key within a serializable transaction to prevent races.
///
/// The `check` callback receives, inside the transaction, every key whose
/// `issue_from` is one of `issue_froms` (the epochs the caller needs), and
/// returns `Ok(Some(key))` to insert or `Ok(None)` to skip. This ensures the
/// "is a new key needed?" check and the insert are atomic.
pub async fn insert_issuer_key_checked<F>(
    pool: &Pool,
    issue_froms: &[SystemTime],
    check: F,
) -> Result<Option<IssuerKeyRow>, ServerError>
where
    F: FnOnce(&[IssuerKeyRow]) -> Result<Option<IssuerKeyRow>, ServerError>,
{
    let mut client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let tx = client
        .build_transaction()
        .isolation_level(tokio_postgres::IsolationLevel::Serializable)
        .start()
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "begin transaction failed: {}",
                db_error_summary(&e)
            ))
        })?;

    // Read the keys for the wanted epochs inside the transaction.
    let rows = tx
        .query(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key WHERE issue_from = ANY($1)",
            &[&issue_froms],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query issuer_keys by epoch failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let present: Vec<IssuerKeyRow> = rows.iter().map(map_issuer_key_row).collect();

    let key = match check(&present)? {
        Some(k) => k,
        None => {
            tx.rollback().await.map_err(|e| {
                ServerError::Internal(format!("rollback failed: {}", db_error_summary(&e)))
            })?;
            return Ok(None);
        }
    };

    tx.execute(
        "INSERT INTO issuer_key \
            (id, private_key_enc, public_key, domain_separator, \
             issue_from, issue_until, accept_until) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
        &[
            &key.id.as_str(),
            &key.private_key_enc.as_slice(),
            &key.public_key.as_slice(),
            &key.domain_separator.as_str(),
            &key.issue_from,
            &key.issue_until,
            &key.accept_until,
        ],
    )
    .await
    .map_err(|e| {
        ServerError::Internal(format!(
            "insert issuer_key failed: {}",
            db_error_summary(&e)
        ))
    })?;

    tx.commit()
        .await
        .map_err(|e| ServerError::Internal(format!("commit failed: {}", db_error_summary(&e))))?;

    Ok(Some(key))
}

/// Retrieve all issuer keys that are still accepted (accept_until > now()),
/// plus any future keys (issue_from > now()). Ordered by issue_from ASC.
#[tracing::instrument(skip_all, name = "db.get_valid_issuer_keys", err)]
pub async fn get_valid_issuer_keys(pool: &Pool) -> Result<Vec<IssuerKeyRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows = client
        .query(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key WHERE accept_until > now() \
             ORDER BY issue_from ASC",
            &[],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query valid issuer_keys failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(rows.iter().map(map_issuer_key_row).collect())
}

fn map_issuer_key_row(r: &tokio_postgres::Row) -> IssuerKeyRow {
    IssuerKeyRow {
        id: r.get("id"),
        private_key_enc: r.get("private_key_enc"),
        public_key: r.get("public_key"),
        domain_separator: r.get("domain_separator"),
        issue_from: r.get("issue_from"),
        issue_until: r.get("issue_until"),
        accept_until: r.get("accept_until"),
    }
}

/// What [`record_credential_issuance`] found or did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssuanceRecord {
    /// Debited now and stored under the idempotency key.
    Issued { ledger_entry_id: Uuid },
    /// An earlier attempt under this idempotency key was debited; this is
    /// what it stored. Nothing was debited now.
    Stored(StoredIssuance),
    /// The named key's issue window has closed by the database clock, read
    /// under the account lock; nothing debited or stored.
    Retired,
    /// Account closure permanently forbids new conversions. Stored retries
    /// are answered before this check, so no paid response is lost.
    Closed,
    /// The balance could not cover the amount; nothing debited or stored.
    Insufficient,
}

/// What [`resolve_issuance_off_current_key`] found, for an issuance naming a
/// key that is not the gateway's current issuing key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OffCurrentKey {
    /// An earlier attempt under this idempotency key was debited.
    Stored(StoredIssuance),
    /// Nothing is stored and the named key has stopped issuing (or is
    /// unknown) by the database clock under the account lock, so nothing can
    /// ever be debited under this idempotency key.
    Retired,
    /// Nothing is stored, but the database still has the named key issuing:
    /// an attempt under this idempotency key may yet debit.
    Issuing,
}

/// A stored issuance response (a `credential_issuance` row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredIssuance {
    pub request_digest: Vec<u8>,
    pub issuer_key_id: String,
    pub credits: i64,
    pub issuance_response: Vec<u8>,
    pub ledger_entry_id: Uuid,
}

fn map_stored_issuance(r: &tokio_postgres::Row) -> StoredIssuance {
    StoredIssuance {
        request_digest: r.get("request_digest"),
        issuer_key_id: r.get("issuer_key_id"),
        credits: r.get("credits"),
        issuance_response: r.get("issuance_response"),
        ledger_entry_id: r.get("ledger_entry_id"),
    }
}

/// The issuance stored under `(account_id, idempotency_key)`, if any. A
/// lock-free read, used only to answer a retry early; the decision that
/// debits is [`record_credential_issuance`]'s, under the account lock.
#[tracing::instrument(skip_all, name = "db.find_credential_issuance", err)]
pub async fn find_credential_issuance(
    pool: &Pool,
    account_id: Uuid,
    idempotency_key: &[u8],
) -> Result<Option<StoredIssuance>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;
    let row = client
        .query_opt(
            "SELECT request_digest, issuer_key_id, credits, issuance_response, ledger_entry_id \
             FROM credential_issuance WHERE account_id = $1 AND idempotency_key = $2",
            &[&account_id, &idempotency_key],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "credential issuance lookup failed: {}",
                db_error_summary(&e)
            ))
        })?;
    Ok(row.as_ref().map(map_stored_issuance))
}

/// Debit an account for a credential issuance and store its response, once
/// per idempotency key.
///
/// One transaction, serialized per account by the account row lock: an
/// issuance already stored under the key is returned and nothing is debited;
/// a key whose issue window has closed by the database clock, read under the
/// lock, is refused ([`IssuanceRecord::Retired`]) however long ago this
/// attempt passed the gateway's current-key check; otherwise the debit
/// (pool-aware, FIFO by expiry, through `debit_account`)
/// and the stored response commit together, so a row exists exactly when the
/// account paid for it.
#[tracing::instrument(skip_all, name = "db.record_credential_issuance", err)]
pub async fn record_credential_issuance(
    pool: &Pool,
    account_id: Uuid,
    idempotency_key: &[u8],
    request_digest: &[u8],
    credential_key_id: &str,
    credits: i64,
    issuance_response: &[u8],
) -> Result<IssuanceRecord, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one(
            "SELECT outcome, request_digest, issuer_key_id, credits, issuance_response, \
             ledger_entry_id FROM record_credential_issuance($1, $2, $3, $4, $5, $6)",
            &[
                &account_id,
                &idempotency_key,
                &request_digest,
                &credential_key_id,
                &credits,
                &issuance_response,
            ],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "credential issuance debit failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let outcome: String = row.get("outcome");
    Ok(match outcome.as_str() {
        "issued" => IssuanceRecord::Issued {
            ledger_entry_id: row.get("ledger_entry_id"),
        },
        "stored" => IssuanceRecord::Stored(map_stored_issuance(&row)),
        "retired" => IssuanceRecord::Retired,
        "closed" => IssuanceRecord::Closed,
        "insufficient" => IssuanceRecord::Insufficient,
        other => {
            return Err(ServerError::Internal(format!(
                "unexpected issuance outcome {other:?}"
            )));
        }
    })
}

/// Decide an issuance that names a key other than the current issuing key,
/// under the account lock [`record_credential_issuance`] takes and on the
/// same database clock, so a refusal is ordered against every attempt under
/// the idempotency key still in flight.
#[tracing::instrument(skip_all, name = "db.resolve_issuance_off_current_key", err)]
pub async fn resolve_issuance_off_current_key(
    pool: &Pool,
    account_id: Uuid,
    idempotency_key: &[u8],
    issuer_key_id: &str,
) -> Result<OffCurrentKey, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;
    let row = client
        .query_one(
            "SELECT outcome, request_digest, issuer_key_id, credits, issuance_response, \
             ledger_entry_id FROM resolve_issuance_off_current_key($1, $2, $3)",
            &[&account_id, &idempotency_key, &issuer_key_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "credential issuance key check failed: {}",
                db_error_summary(&e)
            ))
        })?;
    let outcome: String = row.get("outcome");
    Ok(match outcome.as_str() {
        "stored" => OffCurrentKey::Stored(map_stored_issuance(&row)),
        "retired" => OffCurrentKey::Retired,
        "issuing" => OffCurrentKey::Issuing,
        other => {
            return Err(ServerError::Internal(format!(
                "unexpected issuance key outcome {other:?}"
            )));
        }
    })
}

/// Delete the stored issuance responses of every key retired at least
/// `retain_past_accept` ago: the deadline nullifiers are pruned on.
#[tracing::instrument(skip_all, name = "db.prune_expired_issuances", err)]
pub async fn prune_expired_issuances(
    pool: &Pool,
    retain_past_accept: std::time::Duration,
) -> Result<i64, ServerError> {
    let retain_secs = retain_past_accept.as_secs_f64();
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;
    let row = client
        .query_one(
            "SELECT prune_expired_issuances(make_interval(secs => $1)) AS pruned",
            &[&retain_secs],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "prune_expired_issuances failed: {}",
                db_error_summary(&e)
            ))
        })?;
    Ok(row.get("pruned"))
}

/// Current balance, including the shared billing ledger's outstanding debt.
pub async fn get_available_balance(pool: &Pool, account_id: Uuid) -> Result<i64, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one("SELECT available_balance($1) as balance", &[&account_id])
        .await
        .map_err(|e| {
            ServerError::Internal(format!("balance query failed: {}", db_error_summary(&e)))
        })?;

    Ok(row.get("balance"))
}

/// A single balance pool row.
pub struct BalancePoolRow {
    pub expires_at: Option<SystemTime>,
    pub pool_amount: i64,
    pub source_reason: Option<String>,
}

/// Get balance pools and total available balance for an account.
pub async fn get_balance_pools(
    pool: &Pool,
    account_id: Uuid,
) -> Result<(i64, Vec<BalancePoolRow>), ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let total_row = client
        .query_one("SELECT available_balance($1) as balance", &[&account_id])
        .await
        .map_err(|e| {
            ServerError::Internal(format!("balance query failed: {}", db_error_summary(&e)))
        })?;

    let total: i64 = total_row.get("balance");

    let rows = client
        .query(
            "SELECT cl.expires_at, SUM(cl.delta)::bigint AS pool_amount, p.kind AS source_reason \
             FROM credit_ledger cl LEFT JOIN billing_purchase p ON p.id = cl.purchase_id \
             WHERE cl.account_id = $1 AND (cl.expires_at IS NULL OR cl.expires_at > now()) \
             GROUP BY cl.purchase_id, cl.expires_at, p.kind \
             HAVING SUM(cl.delta) != 0 ORDER BY cl.expires_at NULLS LAST, cl.purchase_id",
            &[&account_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "balance pools query failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let pools = rows
        .iter()
        .map(|row| BalancePoolRow {
            expires_at: row.get("expires_at"),
            pool_amount: row.get("pool_amount"),
            source_reason: row.get("source_reason"),
        })
        .collect();

    Ok((total, pools))
}

/// Retrieve an issuer key by its hex-encoded hash, if it is still retained:
/// `accept_until + retain_past_accept > now()`. The caller decides on its
/// own clock whether the key is usable for what it wants
/// (`credentials::key_usable`).
#[tracing::instrument(skip_all, name = "db.get_issuer_key", err)]
pub async fn get_issuer_key_by_hash(
    pool: &Pool,
    key_hash: &[u8],
    retain_past_accept: std::time::Duration,
) -> Result<Option<IssuerKeyRow>, ServerError> {
    let id = hex::encode(key_hash);
    let retain_secs = retain_past_accept.as_secs_f64();
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_opt(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key \
             WHERE id = $1 AND accept_until + make_interval(secs => $2) > now()",
            &[&id, &retain_secs],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query issuer_key by hash failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(row.as_ref().map(map_issuer_key_row))
}

/// Delete the nullifiers of every key retired at least `retain_past_accept`
/// ago (`accept_until + retain_past_accept < now()`). Returns how many were
/// deleted.
///
/// Called with `credentials::RETENTION_PAST_ACCEPT`, the same margin that
/// bounds every use of a key, so a nullifier outlives anything that could
/// accept a spend of it.
#[tracing::instrument(skip_all, name = "db.prune_expired_nullifiers", err)]
pub async fn prune_expired_nullifiers(
    pool: &Pool,
    retain_past_accept: std::time::Duration,
) -> Result<i64, ServerError> {
    let retain_secs = retain_past_accept.as_secs_f64();
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;
    let row = client
        .query_one(
            "SELECT prune_expired_nullifiers(make_interval(secs => $1)) AS pruned",
            &[&retain_secs],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "prune_expired_nullifiers failed: {}",
                db_error_summary(&e)
            ))
        })?;
    Ok(row.get("pruned"))
}

/// One nullifier row to record: the issuer key id, the nullifier value, and
/// the sealed refund stored beside it.
pub struct NullifierRecord<'a> {
    pub issuer_key_id: &'a str,
    pub value: &'a [u8],
    pub refund_token: &'a SealedRefundToken,
}

/// Atomically record a nullifier with its sealed refund token, in one
/// statement, so no recorded nullifier is ever without a refund. Returns
/// `true` if successfully recorded, `false` if the nullifier was already
/// present (double-spend attempt), in which case nothing is written.
pub async fn record_nullifier(
    pool: &Pool,
    issuer_key_id: &str,
    value: &[u8],
    refund_token: &SealedRefundToken,
) -> Result<bool, ServerError> {
    record_nullifiers(
        pool,
        &[NullifierRecord {
            issuer_key_id,
            value,
            refund_token,
        }],
    )
    .await
}

/// Record every one of a request's nullifiers, each with its sealed refund
/// token, **all or none**: `true` if every row was inserted, `false` if any
/// was already present (a replay, a recovery that raced the request, or the
/// same nullifier twice in `rows`), in which case nothing is written.
///
/// One transaction, one multi-row `INSERT … ON CONFLICT DO NOTHING` whose
/// count is compared with the number of rows: anything short of all of them
/// rolls the transaction back. A concurrent transaction inserting a shared
/// nullifier waits on the first's uncommitted row and then sees it (or, if
/// the first rolled back, inserts it), so two requests sharing an input can
/// never both record it. Rows are inserted sorted by (issuer key id, value),
/// so two requests sharing nullifiers in opposite orders lock them in the
/// same order and cannot deadlock.
///
/// What the database sees is what one request is: its rows commit together.
/// That discloses the request's input count and key set to whoever watches
/// the database, never an amount (`docs/specs/act-multi-spend.md` §9.6).
#[tracing::instrument(skip_all, name = "db.record_nullifiers", err)]
pub async fn record_nullifiers(
    pool: &Pool,
    rows: &[NullifierRecord<'_>],
) -> Result<bool, ServerError> {
    let mut sorted: Vec<&NullifierRecord<'_>> = rows.iter().collect();
    sorted.sort_by(|a, b| (a.issuer_key_id, a.value).cmp(&(b.issuer_key_id, b.value)));
    let key_ids: Vec<&str> = sorted.iter().map(|r| r.issuer_key_id).collect();
    let values: Vec<&[u8]> = sorted.iter().map(|r| r.value).collect();
    let tokens: Vec<&[u8]> = sorted.iter().map(|r| r.refund_token.as_bytes()).collect();

    let mut client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;
    let tx = client.transaction().await.map_err(|e| {
        ServerError::Internal(format!(
            "begin transaction failed: {}",
            db_error_summary(&e)
        ))
    })?;
    let inserted = tx
        .execute(
            "INSERT INTO nullifier (issuer_key_id, value, refund_token) \
             SELECT * FROM unnest($1::text[], $2::bytea[], $3::bytea[]) \
             ON CONFLICT DO NOTHING",
            &[&key_ids, &values, &tokens],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "record_nullifiers failed: {}",
                db_error_summary(&e)
            ))
        })?;
    if inserted != rows.len() as u64 {
        tx.rollback().await.map_err(|e| {
            ServerError::Internal(format!("rollback failed: {}", db_error_summary(&e)))
        })?;
        return Ok(false);
    }
    tx.commit()
        .await
        .map_err(|e| ServerError::Internal(format!("commit failed: {}", db_error_summary(&e))))?;
    Ok(true)
}

/// Store a sealed refund token on an existing nullifier row. Only a
/// [`SealedRefundToken`] is accepted: the CBOR token carries the refund
/// amount and is never written to Postgres in the clear.
#[tracing::instrument(skip_all, name = "db.store_refund_token", err)]
pub async fn store_refund_token(
    pool: &Pool,
    issuer_key_id: &str,
    nullifier_value: &[u8],
    refund_token: &SealedRefundToken,
) -> Result<(), ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    client
        .execute(
            "UPDATE nullifier SET refund_token = $3 \
             WHERE issuer_key_id = $1 AND value = $2",
            &[&issuer_key_id, &nullifier_value, &refund_token.as_bytes()],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "store_refund_token failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(())
}

/// Replace the sealed refund token on a nullifier row with `new`, **only if**
/// it is still exactly `expected`: `true` if it was replaced. A
/// compare-and-swap, so of two writers racing over one row (settlement
/// storing the final refund over the provisional one, and recovery declaring
/// the provisional one final) exactly one wins, and the other sees it lost:
/// under Postgres's row lock the loser re-checks its condition against the
/// winner's row and matches nothing.
#[tracing::instrument(skip_all, name = "db.replace_refund_token", err)]
pub async fn replace_refund_token(
    pool: &Pool,
    issuer_key_id: &str,
    nullifier_value: &[u8],
    expected: &SealedRefundToken,
    new: &SealedRefundToken,
) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;
    let replaced = client
        .execute(
            "UPDATE nullifier SET refund_token = $4 \
             WHERE issuer_key_id = $1 AND value = $2 AND refund_token = $3",
            &[
                &issuer_key_id,
                &nullifier_value,
                &expected.as_bytes(),
                &new.as_bytes(),
            ],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "replace_refund_token failed: {}",
                db_error_summary(&e)
            ))
        })?;
    Ok(replaced > 0)
}

/// Retrieve the sealed refund token stored for a nullifier. Returns `None` if
/// the nullifier does not exist or no refund token has been stored yet; the
/// caller opens what comes back ([`crate::credentials::open_refund_token`]).
#[tracing::instrument(skip_all, name = "db.get_refund_token", err)]
pub async fn get_refund_token(
    pool: &Pool,
    issuer_key_id: &str,
    nullifier_value: &[u8],
) -> Result<Option<SealedRefundToken>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_opt(
            "SELECT refund_token FROM nullifier \
             WHERE issuer_key_id = $1 AND value = $2",
            &[&issuer_key_id, &nullifier_value],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!("get_refund_token failed: {}", db_error_summary(&e)))
        })?;

    Ok(row
        .and_then(|r| r.get::<_, Option<Vec<u8>>>("refund_token"))
        .map(SealedRefundToken::from_stored))
}

/// A single ledger entry row.
pub struct LedgerEntryRow {
    pub id: Uuid,
    pub delta: i64,
    pub reason: String,
    pub expires_at: Option<SystemTime>,
    pub created_at: SystemTime,
    pub credential_key_id: Option<String>,
    pub credential_credits: Option<i64>,
}

/// Whether a reconciled payment has ever funded this account.
#[tracing::instrument(skip_all, name = "db.has_stripe_ledger_history", err)]
pub async fn has_stripe_ledger_history(pool: &Pool, account_id: Uuid) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one(
            "SELECT EXISTS (\
                 SELECT 1 FROM billing_purchase \
                 WHERE account_id = $1\
             )",
            &[&account_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "stripe ledger history query failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(row.get(0))
}

/// Get all ledger entries for an account, sorted by created_at ASC, id ASC.
pub async fn get_ledger_entries(
    pool: &Pool,
    account_id: Uuid,
) -> Result<Vec<LedgerEntryRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows = client
        .query(
            "SELECT id, delta, reason, expires_at, created_at, credential_key_id, credential_credits \
             FROM credit_ledger WHERE account_id = $1 \
             ORDER BY created_at ASC, id ASC",
            &[&account_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("ledger query failed: {}", db_error_summary(&e))))?;

    Ok(rows
        .iter()
        .map(|row| LedgerEntryRow {
            id: row.get("id"),
            delta: row.get("delta"),
            reason: row.get("reason"),
            expires_at: row.get("expires_at"),
            created_at: row.get("created_at"),
            credential_key_id: row.get("credential_key_id"),
            credential_credits: row.get("credential_credits"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalizes_verify_full_sslmode() {
        let normalized = "postgres://user@db.example.com/postgres?sslmode=verify-full"
            .replace("sslmode=verify-full", "sslmode=require");
        assert!(normalized.contains("sslmode=require"));
    }

    // -----------------------------------------------------------------------
    // Integration tests (require running postgres: just db && just db-reset)
    // -----------------------------------------------------------------------

    use super::*;

    /// A throwaway issuer key row: `credit_ledger.credential_key_id` is a
    /// foreign key, and `issuer_key.id` must be the hash of its public key.
    ///
    /// Its windows are all in the past. The row's `private_key_enc` is one
    /// byte, which `credentials::cache_key` rejects, and `get_valid_issuer_keys`
    /// selects every unexpired key regardless of domain — so a fixture that
    /// was still accepted would stop a server started against this database
    /// from issuing until it expired or someone deleted it. Expired, it is
    /// only ever a foreign-key target.
    async fn insert_dummy_issuer_key(pool: &Pool) -> String {
        let public_key = Uuid::new_v4().as_bytes().to_vec();
        let client = pool.get().await.unwrap();
        let row = client
            .query_one(
                "INSERT INTO issuer_key \
                     (id, private_key_enc, public_key, domain_separator, \
                      issue_from, issue_until, accept_until) \
                 VALUES (encode(sha256($1), 'hex'), '\\x00', $1, 'test', \
                         now() - interval '20 days', now() - interval '13 days', \
                         now() - interval '6 days') \
                 RETURNING id",
                &[&public_key],
            )
            .await
            .unwrap();
        row.get("id")
    }

    /// Concurrent issuance must never debit more than the account holds.
    ///
    /// `debit_account` checks `available_balance` and then inserts as separate
    /// statements; without its account row lock, under READ COMMITTED,
    /// requests that overlap each see the balance before the others' debits
    /// commit. This guards the SQL function's own lock directly: the newer
    /// `record_credential_issuance` also locks the account, which would mask
    /// a missing lock here from this regression.
    /// Each round funds a fresh account and fires `CONCURRENCY` issuances
    /// for the whole balance at once; at most one may succeed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore]
    async fn concurrent_issuance_never_overdraws() {
        const CONCURRENCY: usize = 8;
        const ROUNDS: usize = 20;
        const BALANCE: i64 = 1_000_000;

        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = create_pool(&url, None, None).expect("failed to create test pool");
        let key_id = insert_dummy_issuer_key(&pool).await;

        let mut overdrawn = Vec::new();
        for round in 0..ROUNDS {
            let account_id = Uuid::new_v4();
            insert_account(&pool, account_id, "test_hash")
                .await
                .unwrap();
            insert_credit_ledger(
                &pool,
                account_id,
                BALANCE,
                "purchase",
                &format!("evt_test_{}", Uuid::new_v4()),
                None,
                None,
            )
            .await
            .unwrap();

            let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(CONCURRENCY));
            let tasks: Vec<_> = (0..CONCURRENCY)
                .map(|_| {
                    let (pool, key_id, barrier) = (pool.clone(), key_id.clone(), barrier.clone());
                    tokio::spawn(async move {
                        barrier.wait().await;
                        let client = pool.get().await.unwrap();
                        client
                            .query_one(
                                "SELECT debit_account($1, $2, 'credential_issuance', NULL, $3, TRUE) AS ids",
                                &[&account_id, &BALANCE, &key_id],
                            )
                            .await
                            .map(|row| {
                                row.get::<_, Option<Vec<Uuid>>>("ids")
                                    .and_then(|ids| ids.into_iter().next())
                            })
                    })
                })
                .collect();

            let mut succeeded = 0;
            for task in tasks {
                if task.await.unwrap().unwrap().is_some() {
                    succeeded += 1;
                }
            }
            let balance = get_available_balance(&pool, account_id).await.unwrap();
            if succeeded > 1 {
                overdrawn.push((round, succeeded, balance));
            }
        }

        assert!(
            overdrawn.is_empty(),
            "{} of {ROUNDS} rounds issued more than the balance \
             (round, issuances that succeeded, final balance): {overdrawn:?}",
            overdrawn.len()
        );
    }
}
