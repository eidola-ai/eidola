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
