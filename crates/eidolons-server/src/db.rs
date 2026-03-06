//! Database connection pool and query helpers.

use std::time::SystemTime;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use tokio_postgres::NoTls;
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
pub fn create_pool(database_url: &str) -> Result<Pool, String> {
    let pg_config: tokio_postgres::Config = database_url
        .parse()
        .map_err(|e| format!("invalid DATABASE_URL: {}", e))?;

    let manager_config = ManagerConfig {
        recycling_method: RecyclingMethod::Fast,
    };
    let manager = Manager::from_config(pg_config, NoTls, manager_config);

    Pool::builder(manager)
        .max_size(8)
        .build()
        .map_err(|e| format!("failed to build connection pool: {}", e))
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
    pub id: Uuid,
    pub private_key_enc: Vec<u8>,
    pub public_key: Vec<u8>,
    pub domain_separator: String,
    pub issue_from: SystemTime,
    pub issue_until: SystemTime,
    pub accept_until: SystemTime,
}

/// Retrieve the currently active issuer key (issue_from <= now < issue_until).
pub async fn get_current_issuer_key(pool: &Pool) -> Result<Option<IssuerKeyRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_opt(
            "SELECT id, private_key_enc, public_key, domain_separator, \
                    issue_from, issue_until, accept_until \
             FROM issuer_key WHERE issue_from <= now() AND issue_until > now()",
            &[],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query issuer_key failed: {e:?}")))?;

    Ok(row.map(|r| IssuerKeyRow {
        id: r.get("id"),
        private_key_enc: r.get("private_key_enc"),
        public_key: r.get("public_key"),
        domain_separator: r.get("domain_separator"),
        issue_from: r.get("issue_from"),
        issue_until: r.get("issue_until"),
        accept_until: r.get("accept_until"),
    }))
}

/// Insert a new issuer key. Returns true if inserted, false if a key for the
/// same period already exists (race-safe via ON CONFLICT on issue_from).
pub async fn insert_issuer_key(pool: &Pool, key: &IssuerKeyRow) -> Result<bool, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let result = client
        .execute(
            "INSERT INTO issuer_key \
                (id, private_key_enc, public_key, domain_separator, \
                 issue_from, issue_until, accept_until) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             ON CONFLICT (issue_from) DO NOTHING",
            &[
                &key.id,
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

    Ok(result == 1)
}

/// Retrieve all issuer keys that are still valid for spending (accept_until > now()).
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
             ORDER BY issue_from DESC",
            &[],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query valid issuer_keys failed: {e:?}")))?;

    Ok(rows
        .iter()
        .map(|r| IssuerKeyRow {
            id: r.get("id"),
            private_key_enc: r.get("private_key_enc"),
            public_key: r.get("public_key"),
            domain_separator: r.get("domain_separator"),
            issue_from: r.get("issue_from"),
            issue_until: r.get("issue_until"),
            accept_until: r.get("accept_until"),
        })
        .collect())
}

/// Atomically debit credits from an account for credential issuance.
///
/// Returns `Some(ledger_entry_id)` if the debit succeeded, `None` if
/// the account has insufficient balance. The balance check and debit
/// happen in a single SQL statement to prevent TOCTOU races.
pub async fn insert_credential_issuance(
    pool: &Pool,
    account_id: Uuid,
    credits: i64,
    credential_key_id: Uuid,
) -> Result<Option<Uuid>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {e:?}")))?;

    let row = client
        .query_opt(
            "INSERT INTO credit_ledger \
                (id, account_id, delta, reason, credential_key_id, credential_credits, created_at) \
             SELECT gen_random_uuid(), $1, -$2::bigint, 'credential_issuance', $3, $2::bigint, now() \
             WHERE available_balance($1) >= $2::bigint \
             RETURNING id",
            &[&account_id, &credits, &credential_key_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("credential issuance debit failed: {e:?}")))?;

    Ok(row.map(|r| r.get::<_, Uuid>("id")))
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

/// A single ledger entry row.
pub struct LedgerEntryRow {
    pub id: Uuid,
    pub delta: i64,
    pub reason: String,
    pub expires_at: Option<SystemTime>,
    pub created_at: SystemTime,
    pub credential_key_id: Option<Uuid>,
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
