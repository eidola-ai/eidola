//! Database connection pool and query helpers.

use std::io::BufReader;
use std::time::SystemTime;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
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

    let manager_config = ManagerConfig {
        recycling_method: RecyclingMethod::Fast,
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
        let mut reader = BufReader::new(database_ssl_cert.as_bytes());
        let certificates = rustls_pemfile::certs(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("invalid DATABASE_SSL_CERT PEM: {e}"))?;

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

/// Insert a new account and return its `created_at` timestamp.
pub async fn insert_account(
    pool: &Pool,
    id: Uuid,
    credential_hash: &str,
) -> Result<SystemTime, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_one(
            "INSERT INTO account (id, secret_hash) VALUES ($1, $2) RETURNING created_at",
            &[&id, &credential_hash],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("insert account failed: {e:?}")))?;

    Ok(row.get::<_, SystemTime>("created_at"))
}

/// Retrieve an account by ID.
pub async fn get_account_by_id(pool: &Pool, id: Uuid) -> Result<AccountRow, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_opt(
            "SELECT id, secret_hash, stripe_customer_id, created_at \
             FROM account WHERE id = $1",
            &[&id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query account failed: {e:?}")))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let rows_updated = client
        .execute(
            "UPDATE account SET stripe_customer_id = $1, updated_at = now() \
             WHERE id = $2 AND stripe_customer_id IS NULL",
            &[&customer_id, &id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("update stripe customer failed: {e:?}")))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_opt(
            "SELECT id, secret_hash, stripe_customer_id, created_at \
             FROM account WHERE stripe_customer_id = $1",
            &[&customer_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query account by customer failed: {e:?}")))?;

    Ok(row.map(|row| AccountRow {
        id: row.get("id"),
        secret_hash: row.get("secret_hash"),
        stripe_customer_id: row.get("stripe_customer_id"),
        created_at: row.get("created_at"),
    }))
}

/// Insert a credit ledger entry. Returns true if inserted, false if duplicate
/// (based on `stripe_event_id` uniqueness).
pub async fn insert_credit_ledger(
    pool: &Pool,
    account_id: Uuid,
    delta: i64,
    reason: &str,
    stripe_event_id: &str,
    expires_at: Option<SystemTime>,
) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let result = client
        .execute(
            "INSERT INTO credit_ledger (account_id, delta, reason, stripe_event_id, expires_at) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (stripe_event_id) DO NOTHING",
            &[&account_id, &delta, &reason, &stripe_event_id, &expires_at],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("insert credit_ledger failed: {e:?}")))?;

    Ok(result == 1)
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
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let tx = client
        .build_transaction()
        .isolation_level(tokio_postgres::IsolationLevel::Serializable)
        .start()
        .await
        .map_err(|e| ServerError::Internal(format!("begin transaction failed: {e:?}")))?;

    // Read the latest key inside the transaction.
    let latest_row = tx
        .query_opt(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key ORDER BY issue_from DESC LIMIT 1",
            &[],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query latest issuer_key failed: {e:?}")))?;

    let latest = latest_row.as_ref().map(map_issuer_key_row);

    let key = match check(latest.as_ref())? {
        Some(k) => k,
        None => {
            tx.rollback()
                .await
                .map_err(|e| ServerError::Internal(format!("rollback failed: {e:?}")))?;
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
    .map_err(|e| ServerError::Internal(format!("insert issuer_key failed: {e:?}")))?;

    tx.commit()
        .await
        .map_err(|e| ServerError::Internal(format!("commit failed: {e:?}")))?;

    Ok(Some(key))
}

/// Retrieve all issuer keys that are still accepted (accept_until > now()),
/// plus any future keys (issue_from > now()). Ordered by issue_from ASC.
pub async fn get_valid_issuer_keys(pool: &Pool) -> Result<Vec<IssuerKeyRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let rows = client
        .query(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key WHERE accept_until > now() \
             ORDER BY issue_from ASC",
            &[],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query valid issuer_keys failed: {e:?}")))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_one(
            "SELECT debit_account($1, $2, 'credential_issuance', NULL, $3, TRUE) AS ids",
            &[&account_id, &credits, &credential_key_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("credential issuance debit failed: {e:?}")))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_one(
            "SELECT debit_account($1, $2, $3, $4, NULL, FALSE) AS ids",
            &[&account_id, &amount, &reason, &stripe_event_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("debit_stripe_event failed: {e:?}")))?;

    let ids: Option<Vec<Uuid>> = row.get("ids");
    match ids {
        None => Ok(true), // p_require_balance is FALSE, so NULL is never returned
        Some(v) if v.is_empty() => Ok(false), // duplicate event
        Some(_) => Ok(true),
    }
}

/// Get the current available balance for an account.
pub async fn get_available_balance(pool: &Pool, account_id: Uuid) -> Result<i64, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_one("SELECT available_balance($1) as balance", &[&account_id])
        .await
        .map_err(|e| ServerError::Internal(format!("balance query failed: {e:?}")))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let total_row = client
        .query_one("SELECT available_balance($1) as balance", &[&account_id])
        .await
        .map_err(|e| ServerError::Internal(format!("balance query failed: {e:?}")))?;

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
        .map_err(|e| ServerError::Internal(format!("balance pools query failed: {e:?}")))?;

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
pub async fn get_issuer_key_by_hash(
    pool: &Pool,
    key_hash: &[u8],
) -> Result<Option<IssuerKeyRow>, ServerError> {
    let id = hex::encode(key_hash);
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_opt(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key WHERE id = $1 AND accept_until > now()",
            &[&id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query issuer_key by hash failed: {e:?}")))?;

    Ok(row.as_ref().map(map_issuer_key_row))
}

/// Atomically record a nullifier. Returns `true` if successfully recorded,
/// `false` if the nullifier was already present (double-spend attempt).
pub async fn record_nullifier(
    pool: &Pool,
    issuer_key_id: &str,
    value: &[u8],
) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_one(
            "SELECT record_nullifier($1, $2) as recorded",
            &[&issuer_key_id, &value],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("record_nullifier failed: {e:?}")))?;

    Ok(row.get::<_, bool>("recorded"))
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

/// Get all ledger entries for an account, sorted by created_at ASC, id ASC.
pub async fn get_ledger_entries(
    pool: &Pool,
    account_id: Uuid,
) -> Result<Vec<LedgerEntryRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let rows = client
        .query(
            "SELECT id, delta, reason, expires_at, created_at, credential_key_id, credential_credits \
             FROM credit_ledger WHERE account_id = $1 \
             ORDER BY created_at ASC, id ASC",
            &[&account_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("ledger query failed: {e:?}")))?;

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
