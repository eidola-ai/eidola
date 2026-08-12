//! Database connection pool and query helpers.

use std::time::SystemTime;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use rustls_pki_types::CertificateDer;
use rustls_pki_types::pem::PemObject;
use tokio_postgres::NoTls;
use tokio_postgres::config::SslMode;
use tokio_postgres_rustls::MakeRustlsConnect;
use uuid::Uuid;

use crate::error::ServerError;

/// A row from the `account` table.
pub struct AccountRow {
    pub id: Uuid,
    pub secret_hash: String,
    pub stripe_customer_id: Option<String>,
    pub created_at: SystemTime,
}

/// Create a connection pool from a PostgreSQL connection string.
pub fn create_pool(
    database_url: &str,
    database_password: Option<&str>,
    database_ssl_cert: Option<&str>,
) -> Result<Pool, String> {
    let normalized_url = database_url.replace("sslmode=verify-full", "sslmode=require");
    let normalized_url = normalized_url.replace("sslmode=verify-ca", "sslmode=require");

    let mut pg_config: tokio_postgres::Config = normalized_url
        .parse()
        .map_err(|e| format!("invalid DATABASE_URL: {}", e))?;

    if let Some(database_password) = database_password.filter(|value| !value.is_empty()) {
        pg_config.password(database_password);
    }

    // `Verified` issues a cheap liveness check before handing a connection
    // out of the pool. This costs one extra round-trip per checkout but is
    // required when the upstream is a serverless Postgres (e.g. Neon) whose
    // compute can autosuspend and silently kill long-lived sockets.
    let manager_config = ManagerConfig {
        recycling_method: RecyclingMethod::Verified,
    };
    let manager = match pg_config.get_ssl_mode() {
        SslMode::Disable => Manager::from_config(pg_config, NoTls, manager_config),
        _ => {
            let tls = MakeRustlsConnect::new(build_tls_config(database_ssl_cert)?);
            Manager::from_config(pg_config, tls, manager_config)
        }
    };

    Pool::builder(manager)
        .max_size(8)
        .build()
        .map_err(|e| format!("failed to build connection pool: {}", e))
}

fn build_tls_config(database_ssl_cert: Option<&str>) -> Result<rustls::ClientConfig, String> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    if let Some(database_ssl_cert) = database_ssl_cert.filter(|value| !value.is_empty()) {
        let certificates = CertificateDer::pem_slice_iter(database_ssl_cert.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("invalid DATABASE_SSL_CERT PEM: {e:?}"))?;

        if certificates.is_empty() {
            return Err("DATABASE_SSL_CERT did not contain any PEM certificates".to_string());
        }

        let (added, ignored) = root_store.add_parsable_certificates(certificates);
        if added == 0 {
            return Err(
                "DATABASE_SSL_CERT did not contain any usable root certificates".to_string(),
            );
        }
        if ignored > 0 {
            tracing::warn!("Ignored {ignored} invalid certificate(s) in DATABASE_SSL_CERT");
        }
    }

    Ok(rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth())
}

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
            "SELECT id, secret_hash, stripe_customer_id, created_at \
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
            "SELECT id, secret_hash, stripe_customer_id, created_at \
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
        created_at: row.get("created_at"),
    }))
}

/// Insert a credit ledger entry. Returns true if inserted, false if duplicate
/// (based on `stripe_event_id` uniqueness).
///
/// `stripe_payment_intent` links payment-originated credits to their Stripe
/// PaymentIntent so a later refund can be matched back to the same pool.
#[tracing::instrument(skip_all, name = "db.insert_credit_ledger", err)]
pub async fn insert_credit_ledger(
    pool: &Pool,
    account_id: Uuid,
    delta: i64,
    reason: &str,
    stripe_event_id: &str,
    stripe_payment_intent: Option<&str>,
    expires_at: Option<SystemTime>,
) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let result = client
        .execute(
            "INSERT INTO credit_ledger \
                 (account_id, delta, reason, stripe_event_id, stripe_payment_intent, expires_at) \
             VALUES ($1, $2, $3, $4, $5, $6) \
             ON CONFLICT (stripe_event_id) DO NOTHING",
            &[
                &account_id,
                &delta,
                &reason,
                &stripe_event_id,
                &stripe_payment_intent,
                &expires_at,
            ],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "insert credit_ledger failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(result == 1)
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
/// The `check` callback receives the latest key (if any) inside the transaction
/// and returns `Ok(Some(key))` to insert or `Ok(None)` to skip. This ensures
/// the "is a new key needed?" check and the insert are atomic.
pub async fn insert_issuer_key_checked<F>(
    pool: &Pool,
    check: F,
) -> Result<Option<IssuerKeyRow>, ServerError>
where
    F: FnOnce(Option<&IssuerKeyRow>) -> Result<Option<IssuerKeyRow>, ServerError>,
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

    // Read the latest key inside the transaction.
    let latest_row = tx
        .query_opt(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key ORDER BY issue_from DESC LIMIT 1",
            &[],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "query latest issuer_key failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let latest = latest_row.as_ref().map(map_issuer_key_row);

    let key = match check(latest.as_ref())? {
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

/// Atomically debit credits from an account for credential issuance.
///
/// Credits are drawn from balance pools in FIFO order (earliest expiry
/// first, permanent last) so each debit row inherits the pool's
/// `expires_at`.  Returns `Some(first_ledger_entry_id)` if the debit
/// succeeded, `None` if the account has insufficient balance.
pub async fn insert_credential_issuance(
    pool: &Pool,
    account_id: Uuid,
    credits: i64,
    credential_key_id: &str,
) -> Result<Option<Uuid>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one(
            "SELECT debit_account($1, $2, 'credential_issuance', NULL, $3, TRUE) AS ids",
            &[&account_id, &credits, &credential_key_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "credential issuance debit failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let ids: Option<Vec<Uuid>> = row.get("ids");
    Ok(ids.and_then(|v| v.into_iter().next()))
}

/// Debit an account for a Stripe-originated event (refund or clawback).
///
/// Credits are drawn from balance pools in FIFO order (earliest expiry
/// first, permanent last).  Any remainder beyond existing pools is placed
/// in the permanent (NULL expiry) pool.  Returns `Ok(true)` if inserted,
/// `Ok(false)` if the `stripe_event_id` was already processed (duplicate).
pub async fn debit_stripe_event(
    pool: &Pool,
    account_id: Uuid,
    amount: i64,
    reason: &str,
    stripe_event_id: &str,
) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one(
            "SELECT debit_account($1, $2, $3, $4, NULL, FALSE) AS ids",
            &[&account_id, &amount, &reason, &stripe_event_id],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "debit_stripe_event failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let ids: Option<Vec<Uuid>> = row.get("ids");
    match ids {
        None => Ok(true), // p_require_balance is FALSE, so NULL is never returned
        Some(v) if v.is_empty() => Ok(false), // duplicate event
        Some(_) => Ok(true),
    }
}

/// Insert a refund debit matched to the original payment's balance pool.
///
/// Looks up the credit entry recorded for `payment_intent` and, if found,
/// inserts the refund debit with that entry's `expires_at` — even when the
/// pool has already expired — so the refund nets against exactly the credits
/// it reverses instead of draining unrelated pools or creating a permanent
/// negative entry. Returns `Ok(None)` when no credit entry carries this
/// payment intent (caller should fall back to `debit_stripe_event`),
/// `Ok(Some(false))` on a duplicate `stripe_event_id`, and `Ok(Some(true))`
/// on success.
pub async fn refund_matched_to_payment_intent(
    pool: &Pool,
    account_id: Uuid,
    amount: i64,
    stripe_event_id: &str,
    payment_intent: &str,
) -> Result<Option<bool>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let original = client
        .query_opt(
            "SELECT expires_at FROM credit_ledger \
             WHERE account_id = $1 AND stripe_payment_intent = $2 AND delta > 0 \
             ORDER BY created_at ASC LIMIT 1",
            &[&account_id, &payment_intent],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "refund pool lookup failed: {}",
                db_error_summary(&e)
            ))
        })?;

    let Some(row) = original else {
        return Ok(None);
    };
    let expires_at: Option<SystemTime> = row.get("expires_at");
    let delta = -amount;

    let result = client
        .execute(
            "INSERT INTO credit_ledger \
                 (account_id, delta, reason, stripe_event_id, stripe_payment_intent, expires_at) \
             VALUES ($1, $2, 'refund', $3, $4, $5) \
             ON CONFLICT (stripe_event_id) DO NOTHING",
            &[
                &account_id,
                &delta,
                &stripe_event_id,
                &payment_intent,
                &expires_at,
            ],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!(
                "matched refund insert failed: {}",
                db_error_summary(&e)
            ))
        })?;

    Ok(Some(result == 1))
}

/// Get the current available balance for an account.
#[tracing::instrument(skip_all, name = "db.get_available_balance", err)]
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
            "SELECT cl.expires_at, SUM(cl.delta)::bigint as pool_amount, \
               (SELECT reason FROM credit_ledger sub \
                WHERE sub.account_id = cl.account_id \
                  AND sub.expires_at IS NOT DISTINCT FROM cl.expires_at AND sub.delta > 0 \
                ORDER BY sub.created_at ASC LIMIT 1) as source_reason \
             FROM credit_ledger cl \
             WHERE cl.account_id = $1 AND (cl.expires_at IS NULL OR cl.expires_at > now()) \
             GROUP BY cl.account_id, cl.expires_at \
             HAVING SUM(cl.delta)::bigint != 0 \
             ORDER BY cl.expires_at NULLS LAST",
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

/// Retrieve an issuer key by its hex-encoded hash (if still valid for acceptance).
#[tracing::instrument(skip_all, name = "db.get_issuer_key", err)]
pub async fn get_issuer_key_by_hash(
    pool: &Pool,
    key_hash: &[u8],
) -> Result<Option<IssuerKeyRow>, ServerError> {
    let id = hex::encode(key_hash);
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_opt(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key WHERE id = $1 AND accept_until > now()",
            &[&id],
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

/// Atomically record a nullifier. Returns `true` if successfully recorded,
/// `false` if the nullifier was already present (double-spend attempt).
#[tracing::instrument(skip_all, name = "db.record_nullifier", err)]
pub async fn record_nullifier(
    pool: &Pool,
    issuer_key_id: &str,
    value: &[u8],
) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let rows_affected = client
        .execute(
            "INSERT INTO nullifier (issuer_key_id, value) \
             VALUES ($1, $2) ON CONFLICT DO NOTHING",
            &[&issuer_key_id, &value],
        )
        .await
        .map_err(|e| {
            ServerError::Internal(format!("record_nullifier failed: {}", db_error_summary(&e)))
        })?;

    Ok(rows_affected > 0)
}

/// Store a CBOR-encoded refund token on an existing nullifier row.
#[tracing::instrument(skip_all, name = "db.store_refund_token", err)]
pub async fn store_refund_token(
    pool: &Pool,
    issuer_key_id: &str,
    nullifier_value: &[u8],
    refund_token: &[u8],
) -> Result<(), ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    client
        .execute(
            "UPDATE nullifier SET refund_token = $3 \
             WHERE issuer_key_id = $1 AND value = $2",
            &[&issuer_key_id, &nullifier_value, &refund_token],
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

/// Retrieve a stored refund token for a nullifier. Returns `None` if the
/// nullifier does not exist or no refund token has been stored yet.
#[tracing::instrument(skip_all, name = "db.get_refund_token", err)]
pub async fn get_refund_token(
    pool: &Pool,
    issuer_key_id: &str,
    nullifier_value: &[u8],
) -> Result<Option<Vec<u8>>, ServerError> {
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

    Ok(row.and_then(|r| r.get::<_, Option<Vec<u8>>>("refund_token")))
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

/// Whether Stripe has ever moved money for this account.
///
/// A `stripe_event_id` is stamped on exactly the ledger rows a Stripe
/// webhook wrote — purchases, subscription renewals, refunds, dispute
/// adjustments — so its presence is the honest test for "this account has a
/// payment history", as distinct from merely having a Stripe customer
/// record. Internal rows (credential issuance and the like) carry no event
/// id and correctly do not count.
#[tracing::instrument(skip_all, name = "db.has_stripe_ledger_history", err)]
pub async fn has_stripe_ledger_history(pool: &Pool, account_id: Uuid) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e}")))?;

    let row = client
        .query_one(
            "SELECT EXISTS (\
                 SELECT 1 FROM credit_ledger \
                 WHERE account_id = $1 AND stripe_event_id IS NOT NULL\
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
}
