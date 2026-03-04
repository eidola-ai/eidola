//! Database connection pool and query helpers.

use std::time::SystemTime;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use tokio_postgres::NoTls;
use tokio_postgres::types::ToSql;
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
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    let row = client
        .query_one(
            "INSERT INTO account (id, secret_hash) VALUES ($1, $2) RETURNING created_at",
            &[&id, &credential_hash],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("insert account failed: {}", e)))?;

    Ok(row.get::<_, SystemTime>("created_at"))
}

/// Retrieve an account by ID.
pub async fn get_account_by_id(pool: &Pool, id: Uuid) -> Result<AccountRow, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    let row = client
        .query_opt(
            "SELECT id, secret_hash, stripe_customer_id, created_at \
             FROM account WHERE id = $1",
            &[&id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query account failed: {}", e)))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    let rows_updated = client
        .execute(
            "UPDATE account SET stripe_customer_id = $1, updated_at = now() \
             WHERE id = $2 AND stripe_customer_id IS NULL",
            &[&customer_id, &id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("update stripe customer failed: {}", e)))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    let row = client
        .query_opt(
            "SELECT id, secret_hash, stripe_customer_id, created_at \
             FROM account WHERE stripe_customer_id = $1",
            &[&customer_id],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("query account by customer failed: {}", e)))?;

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
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    let result = client
        .execute(
            "INSERT INTO credit_ledger (account_id, delta, reason, stripe_event_id, expires_at) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (stripe_event_id) DO NOTHING",
            &[&account_id, &delta, &reason, &stripe_event_id, &expires_at],
        )
        .await
        .map_err(|e| ServerError::Internal(format!("insert credit_ledger failed: {}", e)))?;

    Ok(result == 1)
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
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    let total_row = client
        .query_one("SELECT available_balance($1) as balance", &[&account_id])
        .await
        .map_err(|e| ServerError::Internal(format!("balance query failed: {}", e)))?;

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
        .map_err(|e| ServerError::Internal(format!("balance pools query failed: {}", e)))?;

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
    pub token_epoch: Option<String>,
    pub token_credits: Option<i64>,
}

/// Get ledger entries for an account with optional filtering and cursor pagination.
///
/// The cursor is a `(created_at, id)` tuple for keyset pagination, ensuring
/// stable ordering even when multiple entries share the same timestamp.
pub async fn get_ledger_entries(
    pool: &Pool,
    account_id: Uuid,
    reasons: Option<&[String]>,
    cursor: Option<(SystemTime, Uuid)>,
    limit: i64,
) -> Result<Vec<LedgerEntryRow>, ServerError> {
    let client = pool
        .get()
        .await
        .map_err(|e| ServerError::Internal(format!("db pool error: {}", e)))?;

    // Build query dynamically based on optional filters.
    let mut query = String::from(
        "SELECT id, delta, reason, expires_at, created_at, token_epoch, token_credits \
         FROM credit_ledger WHERE account_id = $1",
    );
    let mut params: Vec<Box<dyn ToSql + Sync + Send>> = vec![Box::new(account_id)];
    let mut param_idx = 2u32;

    if let Some(reasons) = reasons {
        query.push_str(&format!(" AND reason = ANY(${})", param_idx));
        params.push(Box::new(reasons.to_vec()));
        param_idx += 1;
    }

    if let Some((cursor_ts, cursor_id)) = cursor {
        query.push_str(&format!(
            " AND (created_at, id) < (${}, ${})",
            param_idx,
            param_idx + 1
        ));
        params.push(Box::new(cursor_ts));
        params.push(Box::new(cursor_id));
        param_idx += 2;
    }

    let _ = param_idx;

    query.push_str(&format!(
        " ORDER BY created_at DESC, id DESC LIMIT {}",
        limit + 1
    ));

    let param_refs: Vec<&(dyn ToSql + Sync)> = params.iter().map(|p| &**p as _).collect();

    let rows = client
        .query(&query, &param_refs)
        .await
        .map_err(|e| ServerError::Internal(format!("ledger query failed: {}", e)))?;

    Ok(rows
        .iter()
        .map(|row| LedgerEntryRow {
            id: row.get("id"),
            delta: row.get("delta"),
            reason: row.get("reason"),
            expires_at: row.get("expires_at"),
            created_at: row.get("created_at"),
            token_epoch: row.get("token_epoch"),
            token_credits: row.get("token_credits"),
        })
        .collect())
}
