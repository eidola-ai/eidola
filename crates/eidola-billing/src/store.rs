//! Durable billing state. Every credit effect shares the account row lock with
//! credential issuance. Provider observations are account data, never ACT data.

use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use deadpool_postgres::{Client, ClientWrapper, GenericClient, Pool};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_postgres::Row;
use uuid::Uuid;

use crate::BillingError;
use crate::service::ReconcileReport;

/// Stable command envelope understood by gateway and webhook producers.
pub const INGRESS_PROTOCOL_VERSION: i32 = 1;
/// Billing semantics admitted by this worker / private CLI implementation.
pub const EXECUTION_PROTOCOL_VERSION: i32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Purchase {
    pub id: String,
    pub account_id: Uuid,
    pub charge_id: Option<String>,
    pub payment_intent: Option<String>,
    pub invoice_id: Option<String>,
    pub currency: String,
    pub paid_minor: i64,
    pub credits: i64,
    pub expires_at: SystemTime,
    pub purchased_at: SystemTime,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefundRequest {
    pub id: Uuid,
    pub account_id: Uuid,
    pub purchase_id: String,
    pub credits: i64,
    /// Present only for a cash-only goodwill refund; never inferred by webhook.
    pub cash_minor: Option<i64>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefundOperation {
    pub id: Uuid,
    pub account_id: Uuid,
    pub purchase_id: String,
    pub credits: i64,
    pub amount_minor: i64,
    pub currency: String,
    pub charge_id: String,
    pub state: String,
    pub refund_id: Option<String>,
    pub created_at: SystemTime,
    pub submitted_at: Option<SystemTime>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefundObservation {
    pub id: String,
    pub charge_id: String,
    pub amount_minor: i64,
    pub currency: String,
    pub status: String,
    pub operation_id: Option<Uuid>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisputeObservation {
    pub id: String,
    pub charge_id: String,
    pub amount_minor: i64,
    pub currency: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Debt {
    pub currency: String,
    /// Decimal minor units: pricing an arbitrary number of credits can produce
    /// fractions of a cent. No currency conversion or at-par assumption.
    pub amount_minor: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewIssue {
    pub kind: String,
    pub id: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingEvent {
    pub id: String,
    pub kind: String,
    pub object_id: String,
    pub attempts: i32,
    pub protocol_version: i32,
    pub generation: i64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RepairStatus {
    pub requested_generation: i64,
    pub completed_generation: i64,
    /// The report for completed_generation; it may predate a pending request.
    pub report: Option<ReconcileReport>,
    pub last_error: Option<String>,
}

#[derive(Clone)]
pub struct Store {
    pool: Pool,
}

fn db(e: tokio_postgres::Error) -> BillingError {
    // Server-authored DETAIL can contain row values. Only a SQLSTATE escapes.
    BillingError::Database(
        e.code()
            .map_or("database connection failed", |v| v.code())
            .into(),
    )
}

async fn check_protocol(c: &tokio_postgres::Client, ingress: bool) -> Result<(), BillingError> {
    let row = c
        .query_opt(
            "SELECT ingress_version,execution_version FROM billing_protocol WHERE singleton",
            &[],
        )
        .await
        .map_err(db)?
        .ok_or_else(|| {
            BillingError::ServiceUnavailable("billing protocol is not configured".into())
        })?;
    let supported = if ingress {
        INGRESS_PROTOCOL_VERSION
    } else {
        EXECUTION_PROTOCOL_VERSION
    };
    let actual: i32 = row.get(if ingress {
        "ingress_version"
    } else {
        "execution_version"
    });
    if actual != supported {
        return Err(BillingError::ServiceUnavailable(
            "billing protocol is incompatible".into(),
        ));
    }
    Ok(())
}

fn row_pending_event(r: Row) -> PendingEvent {
    PendingEvent {
        id: r.get("id"),
        kind: r.get("kind"),
        object_id: r.get("object_id"),
        attempts: r.get("attempts"),
        protocol_version: r.get("protocol_version"),
        generation: r.get("generation"),
    }
}

fn row_repair_status(r: Row) -> Result<RepairStatus, BillingError> {
    let report: Option<serde_json::Value> = r.get("report");
    Ok(RepairStatus {
        requested_generation: r.get("generation"),
        completed_generation: r.get("completed_generation"),
        report: report
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| BillingError::Parse("stored repair report is incompatible".into()))?,
        last_error: r.get("last_error"),
    })
}

fn row_purchase(r: Row) -> Purchase {
    Purchase {
        id: r.get("id"),
        account_id: r.get("account_id"),
        charge_id: r.get("charge_id"),
        payment_intent: r.get("payment_intent"),
        invoice_id: r.get("invoice_id"),
        currency: r.get("currency"),
        paid_minor: r.get("paid_minor"),
        credits: r.get("credits"),
        expires_at: r.get("expires_at"),
        purchased_at: r.get("purchased_at"),
        kind: r.get("kind"),
    }
}

fn row_refund(r: Row) -> RefundOperation {
    RefundOperation {
        id: r.get("id"),
        account_id: r.get("account_id"),
        purchase_id: r.get("purchase_id"),
        credits: r.get("credits"),
        amount_minor: r.get("amount_minor"),
        currency: r.get("currency"),
        charge_id: r.get::<_, Option<String>>("charge_id").unwrap_or_default(),
        state: r.get("state"),
        refund_id: r.get("refund_id"),
        created_at: r.get("created_at"),
        submitted_at: r.get("submitted_at"),
    }
}

const REFUND_SELECT: &str = "SELECT r.*, o.account_id, o.created_at, p.charge_id, p.currency \
    FROM billing_refund_operation r JOIN billing_operation o ON o.id = r.id \
    JOIN billing_purchase p ON p.id = r.purchase_id";

async fn lock_account(c: &impl GenericClient, account: Uuid) -> Result<(), BillingError> {
    if c.query_opt(
        "SELECT id FROM account WHERE id = $1 FOR UPDATE",
        &[&account],
    )
    .await
    .map_err(db)?
    .is_none()
    {
        return Err(BillingError::NotFound("account not found".into()));
    }
    Ok(())
}

/// Called under the account row lock, in the same transaction as an operator
/// settlement. A new generation survives completion of an older claimed job.
async fn wake_closure(c: &impl GenericClient, account: Uuid) -> Result<(), BillingError> {
    let state: String = c
        .query_one("SELECT billing_state FROM account WHERE id=$1", &[&account])
        .await
        .map_err(db)?
        .get(0);
    if state == "open" {
        return Ok(());
    }
    let changed = c
        .execute(
            "INSERT INTO billing_event(id,kind,object_id) VALUES($1,'billing.close.resume',$2) \
         ON CONFLICT(id) DO UPDATE SET generation=billing_event.generation+1,completed_at=NULL, \
         next_attempt_at=clock_timestamp(),attempts=0,last_error=NULL \
         WHERE billing_event.kind='billing.close.resume' AND billing_event.object_id=$2 \
         AND billing_event.protocol_version=1",
            &[&format!("local:close:{account}"), &account.to_string()],
        )
        .await
        .map_err(db)?;
    if changed != 1 {
        return Err(BillingError::RequiresReview(
            "closure job is incompatible".into(),
        ));
    }
    Ok(())
}

async fn get_purchase(c: &impl GenericClient, id: &str) -> Result<Purchase, BillingError> {
    c.query_opt("SELECT * FROM billing_purchase WHERE id = $1", &[&id])
        .await
        .map_err(db)?
        .map(row_purchase)
        .ok_or_else(|| BillingError::RequiresReview("payment has no verified purchase".into()))
}

async fn get_refund(c: &impl GenericClient, id: Uuid) -> Result<RefundOperation, BillingError> {
    c.query_opt(&format!("{REFUND_SELECT} WHERE r.id = $1"), &[&id])
        .await
        .map_err(db)?
        .map(row_refund)
        .ok_or_else(|| BillingError::NotFound("refund operation not found".into()))
}

fn validate_purchase(p: &Purchase) -> Result<(), BillingError> {
    if p.id.is_empty()
        || p.credits <= 0
        || p.paid_minor < 0
        || (p.kind == "purchase" && p.expires_at <= p.purchased_at)
        || !matches!(p.kind.as_str(), "purchase" | "subscription_renewal")
        || p.currency.len() != 3
        || !p.currency.bytes().all(|b| b.is_ascii_lowercase())
        || (p.paid_minor > 0 && p.charge_id.as_ref().is_none_or(|s| s.is_empty()))
    {
        return Err(BillingError::Invalid("invalid purchase provenance".into()));
    }
    Ok(())
}

async fn record_purchase(c: &impl GenericClient, p: &Purchase) -> Result<bool, BillingError> {
    validate_purchase(p)?;
    lock_account(c, p.account_id).await?;
    if let Some(existing) = c
        .query_opt("SELECT * FROM billing_purchase WHERE id = $1", &[&p.id])
        .await
        .map_err(db)?
    {
        let mut incoming = p.clone();
        // PostgreSQL timestamps retain microseconds, while SystemTime can
        // carry nanoseconds. Compare the immutable values at stored precision.
        for time in [&mut incoming.expires_at, &mut incoming.purchased_at] {
            let micros = time
                .duration_since(UNIX_EPOCH)
                .map_err(|_| BillingError::Invalid("purchase timestamp predates epoch".into()))?
                .as_micros();
            *time = UNIX_EPOCH
                + std::time::Duration::from_micros(u64::try_from(micros).map_err(|_| {
                    BillingError::Invalid("purchase timestamp out of range".into())
                })?);
        }
        if row_purchase(existing) != incoming {
            return Err(BillingError::RequiresReview(
                "purchase provenance changed".into(),
            ));
        }
        return Ok(false);
    }
    let op = Uuid::new_v4();
    c.execute("INSERT INTO billing_operation(id,account_id,kind,reason) VALUES($1,$2,'purchase','verified payment')",
              &[&op,&p.account_id]).await.map_err(db)?;
    c.execute("INSERT INTO billing_purchase(id,account_id,operation_id,charge_id,payment_intent,invoice_id,\
        currency,paid_minor,credits,expires_at,purchased_at,kind) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
        &[&p.id,&p.account_id,&op,&p.charge_id,&p.payment_intent,&p.invoice_id,&p.currency,&p.paid_minor,
          &p.credits,&p.expires_at,&p.purchased_at,&p.kind]).await.map_err(db)?;
    c.execute("INSERT INTO credit_ledger(account_id,delta,reason,expires_at,purchase_id,billing_operation_id,stripe_payment_intent) \
        VALUES($1,$2,$3,$4,$5,$6,$7)", &[&p.account_id,&p.credits,&p.kind,&p.expires_at,&p.id,&op,&p.payment_intent]).await.map_err(db)?;
    // A dispute notification can arrive before enough purchase provenance was
    // available. Attach the already-recorded cash loss in the same transaction
    // as its eventual grant; the caller may then apply a newer provider state.
    let unmatched_disputes = c
        .query(
            "SELECT id,charge_id,amount_minor,currency,status FROM billing_unmatched_observation \
        WHERE account_id=$1 AND kind='dispute' AND charge_id=$2 ORDER BY id",
            &[&p.account_id, &p.charge_id],
        )
        .await
        .map_err(db)?;
    for d in unmatched_disputes {
        observe_dispute(
            c,
            &DisputeObservation {
                id: d.get(0),
                charge_id: d.get(1),
                amount_minor: d.get(2),
                currency: d.get(3),
                status: d.get(4),
            },
            Some(p.account_id),
        )
        .await?;
    }
    let disputes = c
        .query(
            "SELECT d.id FROM billing_dispute d JOIN billing_purchase p ON p.id=d.purchase_id \
        WHERE p.account_id=$1 AND d.debt_minor>0 ORDER BY d.id",
            &[&p.account_id],
        )
        .await
        .map_err(db)?;
    for d in disputes {
        c.query_one("SELECT billing_collect_debt($1)", &[&d.get::<_, String>(0)])
            .await
            .map_err(db)?;
    }
    Ok(true)
}

async fn observe_refund(
    c: &impl GenericClient,
    r: &RefundObservation,
    account: Option<Uuid>,
) -> Result<(), BillingError> {
    if r.id.is_empty()
        || r.amount_minor <= 0
        || !matches!(
            r.status.as_str(),
            "pending" | "requires_action" | "succeeded" | "failed" | "canceled"
        )
    {
        return Err(BillingError::Invalid("invalid refund observation".into()));
    }
    let purchase = c
        .query_opt(
            "SELECT * FROM billing_purchase WHERE charge_id=$1",
            &[&r.charge_id],
        )
        .await
        .map_err(db)?
        .map(row_purchase);
    let Some(p) = purchase else {
        let account = account.ok_or_else(|| {
            BillingError::RequiresReview("refund purchase is not yet verified".into())
        })?;
        lock_account(c, account).await?;
        unmatched(
            c,
            account,
            "refund",
            &r.id,
            &r.charge_id,
            r.amount_minor,
            &r.currency,
            &r.status,
            r.operation_id,
            "purchase_not_verified",
        )
        .await?;
        let recorded=c.execute("INSERT INTO billing_refund_observation(id,account_id,charge_id,amount_minor,currency,status) VALUES($1,$2,$3,$4,$5,$6) \
            ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status,observed_at=clock_timestamp() \
            WHERE billing_refund_observation.account_id=EXCLUDED.account_id AND billing_refund_observation.charge_id=EXCLUDED.charge_id \
            AND billing_refund_observation.amount_minor=EXCLUDED.amount_minor AND billing_refund_observation.currency=EXCLUDED.currency",
            &[&r.id,&account,&r.charge_id,&r.amount_minor,&r.currency,&r.status]).await.map_err(db)?;
        if recorded != 1 {
            return Err(BillingError::RequiresReview(
                "refund financial identity changed".into(),
            ));
        }
        return Ok(());
    };
    if account.is_some_and(|a| a != p.account_id)
        || p.currency != r.currency
        || r.amount_minor > p.paid_minor
    {
        return Err(BillingError::RequiresReview(
            "refund does not match purchase".into(),
        ));
    }
    lock_account(c, p.account_id).await?;
    if let Some(old) = c.query_opt("SELECT charge_id,amount_minor,currency,operation_id,account_id FROM billing_refund_observation WHERE id=$1", &[&r.id]).await.map_err(db)?
        && (old.get::<_,String>(0) != r.charge_id || old.get::<_,i64>(1) != r.amount_minor
            || old.get::<_,String>(2) != r.currency || old.get::<_,Uuid>(4)!=p.account_id
            || old.get::<_,Option<Uuid>>(3).is_some_and(|old|Some(old)!=r.operation_id)) {
            return Err(BillingError::RequiresReview("refund identity changed".into()));
    }
    let mut binding = None;
    if let Some(id) = r.operation_id {
        let existing = c
            .query_opt(&format!("{REFUND_SELECT} WHERE r.id=$1"), &[&id])
            .await
            .map_err(db)?
            .map(row_refund);
        if let Some(op) = existing {
            if op.account_id != p.account_id
                || op.charge_id != r.charge_id
                || op.amount_minor != r.amount_minor
                || op.currency != r.currency
                || op.refund_id.as_ref().is_some_and(|old| *old != r.id)
            {
                unmatched(
                    c,
                    p.account_id,
                    "refund",
                    &r.id,
                    &r.charge_id,
                    r.amount_minor,
                    &r.currency,
                    &r.status,
                    r.operation_id,
                    "operation_does_not_match",
                )
                .await?;
            } else {
                binding = Some(id);
                let paid_externally: bool = c.query_one(
                    "SELECT EXISTS(SELECT 1 FROM billing_external_refund WHERE operation_id=$1)",
                    &[&id],
                ).await.map_err(db)?.get(0);
                if !paid_externally && !matches!(op.state.as_str(), "released" | "superseded") {
                    c.execute(
                        "UPDATE billing_refund_operation SET refund_id=$2,state=$3 WHERE id=$1",
                        &[&id, &r.id, &r.status],
                    )
                    .await
                    .map_err(db)?;
                } else if !matches!(r.status.as_str(), "failed" | "canceled") {
                    // The operator's payout already consumed this operation's
                    // cash allocation. A conflicting provider return is extra
                    // money, not another observation of that same payout.
                    if paid_externally {
                        binding = None;
                    }
                    unmatched(
                        c,
                        p.account_id,
                        "refund",
                        &r.id,
                        &r.charge_id,
                        r.amount_minor,
                        &r.currency,
                        &r.status,
                        r.operation_id,
                        if paid_externally {
                            "externally_settled_refund_returned_money"
                        } else {
                            "released_refund_returned_money"
                        },
                    )
                    .await?;
                }
                c.execute(
                    "UPDATE billing_event SET completed_at=clock_timestamp(),completed_generation=generation WHERE id=$1",
                    &[&format!("local:refund:{id}")],
                )
                .await
                .map_err(db)?;
            }
        } else {
            unmatched(
                c,
                p.account_id,
                "refund",
                &r.id,
                &r.charge_id,
                r.amount_minor,
                &r.currency,
                &r.status,
                r.operation_id,
                "operation_not_found",
            )
            .await?;
        }
    }
    c.execute("INSERT INTO billing_refund_observation(id,charge_id,amount_minor,currency,status,operation_id,account_id) \
        VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status,operation_id=EXCLUDED.operation_id,observed_at=clock_timestamp()",
        &[&r.id,&r.charge_id,&r.amount_minor,&r.currency,&r.status,&binding,&p.account_id]).await.map_err(db)?;
    if r.operation_id.is_none() || binding.is_some() {
        c.execute("DELETE FROM billing_unmatched_observation WHERE kind='refund' AND id=$1 AND reason='purchase_not_verified'", &[&r.id]).await.map_err(db)?;
    }
    c.query_one("SELECT billing_refresh_charge($1)", &[&p.id])
        .await
        .map_err(db)?;
    Ok(())
}

async fn observe_dispute(
    c: &impl GenericClient,
    d: &DisputeObservation,
    account: Option<Uuid>,
) -> Result<(), BillingError> {
    if d.id.is_empty()
        || d.amount_minor <= 0
        || !matches!(
            d.status.as_str(),
            "needs_response"
                | "under_review"
                | "lost"
                | "won"
                | "warning_needs_response"
                | "warning_under_review"
                | "warning_closed"
        )
    {
        return Err(BillingError::Invalid("invalid dispute observation".into()));
    }
    let purchase = c
        .query_opt(
            "SELECT * FROM billing_purchase WHERE charge_id=$1",
            &[&d.charge_id],
        )
        .await
        .map_err(db)?
        .map(row_purchase);
    let Some(p) = purchase else {
        let account = account.ok_or_else(|| {
            BillingError::RequiresReview("dispute purchase is not yet verified".into())
        })?;
        lock_account(c, account).await?;
        unmatched(
            c,
            account,
            "dispute",
            &d.id,
            &d.charge_id,
            d.amount_minor,
            &d.currency,
            &d.status,
            None,
            "purchase_not_verified",
        )
        .await?;
        return Ok(());
    };
    if account.is_some_and(|a| a != p.account_id)
        || p.currency != d.currency
        || d.amount_minor > p.paid_minor
    {
        return Err(BillingError::RequiresReview(
            "dispute does not match purchase".into(),
        ));
    }
    lock_account(c, p.account_id).await?;
    if let Some(old) = c
        .query_opt(
            "SELECT purchase_id,amount_minor FROM billing_dispute WHERE id=$1",
            &[&d.id],
        )
        .await
        .map_err(db)?
    {
        if old.get::<_, String>(0) != p.id || old.get::<_, i64>(1) != d.amount_minor {
            return Err(BillingError::RequiresReview(
                "dispute identity changed".into(),
            ));
        }
        c.execute(
            "UPDATE billing_dispute SET status=$2 WHERE id=$1",
            &[&d.id, &d.status],
        )
        .await
        .map_err(db)?;
    } else {
        let op = Uuid::new_v4();
        c.execute("INSERT INTO billing_operation(id,account_id,kind,reason) VALUES($1,$2,'dispute','verified payment dispute')", &[&op,&p.account_id]).await.map_err(db)?;
        c.execute("INSERT INTO billing_dispute(id,purchase_id,operation_id,amount_minor,status) VALUES($1,$2,$3,$4,$5)",
            &[&d.id,&p.id,&op,&d.amount_minor,&d.status]).await.map_err(db)?;
    }
    c.query_one("SELECT billing_refresh_charge($1)", &[&p.id])
        .await
        .map_err(db)?;
    c.execute(
        "DELETE FROM billing_unmatched_observation WHERE kind='dispute' AND id=$1",
        &[&d.id],
    )
    .await
    .map_err(db)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn unmatched(
    c: &impl GenericClient,
    account: Uuid,
    kind: &str,
    id: &str,
    charge: &str,
    amount: i64,
    currency: &str,
    status: &str,
    operation: Option<Uuid>,
    reason: &str,
) -> Result<(), BillingError> {
    let count=c.execute("INSERT INTO billing_unmatched_observation(account_id,kind,id,charge_id,amount_minor,currency,status,operation_id,reason) \
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT(kind,id) DO UPDATE SET status=EXCLUDED.status,reason=EXCLUDED.reason \
        WHERE billing_unmatched_observation.account_id=EXCLUDED.account_id AND billing_unmatched_observation.charge_id=EXCLUDED.charge_id \
        AND billing_unmatched_observation.amount_minor=EXCLUDED.amount_minor AND billing_unmatched_observation.currency=EXCLUDED.currency",
        &[&account,&kind,&id,&charge,&amount,&currency,&status,&operation,&reason]).await.map_err(db)?;
    if count != 1 {
        return Err(BillingError::RequiresReview(
            "unmatched financial identity changed".into(),
        ));
    }
    Ok(())
}

async fn apply(
    c: &impl GenericClient,
    account: Uuid,
    purchases: &[Purchase],
    refunds: &[RefundObservation],
    disputes: &[DisputeObservation],
) -> Result<(), BillingError> {
    lock_account(c, account).await?;
    for p in purchases {
        if p.account_id != account {
            return Err(BillingError::Invalid(
                "purchase belongs to another account".into(),
            ));
        }
        record_purchase(c, p).await?;
    }
    for r in refunds {
        observe_refund(c, r, Some(account)).await?;
    }
    for d in disputes {
        observe_dispute(c, d, Some(account)).await?;
    }
    Ok(())
}

impl Store {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }
    pub fn pool(&self) -> &Pool {
        &self.pool
    }
    /// Check executable billing semantics before starting a worker or operator
    /// command. Account execution checks again after acquiring its lock.
    pub async fn check_protocol(&self) -> Result<(), BillingError> {
        let client = self.client().await?;
        check_protocol(&client, false).await
    }
    /// Receiver / gateway producers only depend on the stable command envelope.
    pub async fn check_ingress_protocol(&self) -> Result<(), BillingError> {
        let client = self.client().await?;
        check_protocol(&client, true).await
    }
    /// Retry windows use the database clock that stamped submission, never a
    /// support operator's potentially skewed laptop clock.
    pub(crate) async fn current_time(&self) -> Result<SystemTime, BillingError> {
        Ok(self
            .client()
            .await?
            .query_one("SELECT clock_timestamp()", &[])
            .await
            .map_err(db)?
            .get(0))
    }
    async fn client(&self) -> Result<Client, BillingError> {
        self.pool
            .get()
            .await
            .map_err(|_| BillingError::Database("database pool unavailable".into()))
    }

    pub async fn customer_id(&self, account: Uuid) -> Result<Option<String>, BillingError> {
        self.client()
            .await?
            .query_opt(
                "SELECT stripe_customer_id FROM account WHERE id=$1",
                &[&account],
            )
            .await
            .map_err(db)?
            .map(|r| r.get(0))
            .ok_or_else(|| BillingError::NotFound("account not found".into()))
    }
    pub async fn account_for_customer(&self, customer: &str) -> Result<Option<Uuid>, BillingError> {
        Ok(self
            .client()
            .await?
            .query_opt(
                "SELECT id FROM account WHERE stripe_customer_id=$1",
                &[&customer],
            )
            .await
            .map_err(db)?
            .map(|r| r.get(0)))
    }
    pub async fn get_purchase(&self, id: &str) -> Result<Purchase, BillingError> {
        get_purchase(&self.client().await?, id).await
    }
    pub async fn purchase_for_charge(&self, charge: &str) -> Result<Purchase, BillingError> {
        self.client()
            .await?
            .query_opt(
                "SELECT * FROM billing_purchase WHERE charge_id=$1",
                &[&charge],
            )
            .await
            .map_err(db)?
            .map(row_purchase)
            .ok_or_else(|| BillingError::RequiresReview("payment has no verified purchase".into()))
    }
    pub async fn record_purchase(&self, p: &Purchase) -> Result<bool, BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let result = record_purchase(&tx, p).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }
    pub async fn apply_observations(
        &self,
        account: Uuid,
        purchases: &[Purchase],
        refunds: &[RefundObservation],
        disputes: &[DisputeObservation],
    ) -> Result<(), BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        apply(&tx, account, purchases, refunds, disputes).await?;
        tx.commit().await.map_err(db)
    }
    pub async fn observe_refund(&self, r: &RefundObservation) -> Result<(), BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        observe_refund(&tx, r, None).await?;
        tx.commit().await.map_err(db)
    }
    pub async fn reconcile_dispute(&self, d: &DisputeObservation) -> Result<(), BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        observe_dispute(&tx, d, None).await?;
        tx.commit().await.map_err(db)
    }
    pub async fn get_refund(&self, id: Uuid) -> Result<RefundOperation, BillingError> {
        get_refund(&self.client().await?, id).await
    }

    pub async fn preview_refund(&self, r: &RefundRequest) -> Result<RefundOperation, BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        lock_account(&tx, r.account_id).await?;
        let result = quote(&tx, r).await?;
        tx.rollback().await.map_err(db)?;
        Ok(result)
    }
    pub async fn prepare_refund(&self, r: &RefundRequest) -> Result<RefundOperation, BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        lock_account(&tx, r.account_id).await?;
        if let Ok(existing) = get_refund(&tx, r.id).await {
            let reason: String = tx
                .query_one("SELECT reason FROM billing_operation WHERE id=$1", &[&r.id])
                .await
                .map_err(db)?
                .get(0);
            if existing.account_id != r.account_id
                || existing.purchase_id != r.purchase_id
                || existing.credits != r.credits
                || (r.credits == 0 && r.cash_minor != Some(existing.amount_minor))
                || reason != r.reason
            {
                return Err(BillingError::Conflict(
                    "operation id was used for another request".into(),
                ));
            }
            tx.commit().await.map_err(db)?;
            return Ok(existing);
        }
        let op = quote(&tx, r).await?;
        tx.execute(
            "INSERT INTO billing_operation(id,account_id,kind,reason) VALUES($1,$2,'refund',$3)",
            &[&r.id, &r.account_id, &r.reason],
        )
        .await
        .map_err(db)?;
        let state = if op.amount_minor == 0 {
            "succeeded"
        } else {
            "prepared"
        };
        tx.execute("INSERT INTO billing_refund_operation(id,purchase_id,credits,amount_minor,state) VALUES($1,$2,$3,$4,$5)",
            &[&r.id,&r.purchase_id,&r.credits,&op.amount_minor,&state]).await.map_err(db)?;
        if op.amount_minor > 0 {
            tx.execute("INSERT INTO billing_event(id,kind,object_id) VALUES($1,'billing.refund.resume',$2)", &[&format!("local:refund:{}",r.id),&r.id.to_string()]).await.map_err(db)?;
        }
        if r.credits > 0 {
            tx.execute("INSERT INTO credit_ledger(account_id,delta,reason,purchase_id,billing_operation_id,expires_at) \
                SELECT $1,-$2::BIGINT,'refund',id,$3,expires_at FROM billing_purchase WHERE id=$4",
                &[&r.account_id,&r.credits,&r.id,&r.purchase_id]).await.map_err(db)?;
        }
        let result = get_refund(&tx, r.id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }

    pub async fn mark_submitted(&self, id: Uuid) -> Result<RefundOperation, BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let op = get_refund(&tx, id).await?;
        lock_account(&tx, op.account_id).await?;
        let changed=tx.execute("UPDATE billing_refund_operation SET state='submitting',submitted_at=COALESCE(submitted_at,clock_timestamp()) \
            WHERE id=$1 AND state IN ('prepared','submitting') AND refund_id IS NULL AND amount_minor>0", &[&id]).await.map_err(db)?;
        if changed != 1 {
            return Err(BillingError::Conflict(
                "refund must not be submitted again".into(),
            ));
        }
        let result = get_refund(&tx, id).await?;
        tx.commit().await.map_err(db)?;
        Ok(result)
    }
    pub async fn mark_refund_rejected(&self, id: Uuid) -> Result<(), BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let op = get_refund(&tx, id).await?;
        lock_account(&tx, op.account_id).await?;
        if tx.execute("UPDATE billing_refund_operation SET state='failed' WHERE id=$1 AND state='submitting' AND refund_id IS NULL", &[&id]).await.map_err(db)? != 1 {
            return Err(BillingError::Conflict("refund rejection does not match submission".into()));
        }
        tx.execute(
            "UPDATE billing_event SET completed_at=clock_timestamp(),completed_generation=generation WHERE id=$1",
            &[&format!("local:refund:{id}")],
        )
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)
    }
    pub async fn release_refund(&self, id: Uuid) -> Result<(), BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let first = get_refund(&tx, id).await?;
        lock_account(&tx, first.account_id).await?;
        let op = get_refund(&tx, id).await?;
        if op.state == "released" {
            return Ok(());
        }
        if !matches!(op.state.as_str(), "prepared" | "failed" | "canceled") {
            return Err(BillingError::Conflict(
                "refund outcome is not safely releasable".into(),
            ));
        }
        let expired: bool = tx
            .query_one(
                "SELECT expires_at<=clock_timestamp() FROM billing_purchase WHERE id=$1",
                &[&op.purchase_id],
            )
            .await
            .map_err(db)?
            .get(0);
        if expired && op.credits > 0 {
            return Err(BillingError::RequiresReview(
                "expired refund hold must retain its refund eligibility".into(),
            ));
        }
        if op.credits > 0 {
            tx.execute("INSERT INTO credit_ledger(account_id,delta,reason,purchase_id,billing_operation_id,expires_at) \
                SELECT $1,$2,'manual_adjustment',id,$3,expires_at FROM billing_purchase WHERE id=$4",
                &[&op.account_id,&op.credits,&id,&op.purchase_id]).await.map_err(db)?;
        }
        tx.execute(
            "UPDATE billing_refund_operation SET state='released' WHERE id=$1",
            &[&id],
        )
        .await
        .map_err(db)?;
        tx.execute(
            "UPDATE billing_event SET completed_at=clock_timestamp(),completed_generation=generation WHERE id=$1",
            &[&format!("local:refund:{id}")],
        )
        .await
        .map_err(db)?;
        let debts=tx.query("SELECT d.id FROM billing_dispute d JOIN billing_purchase p ON p.id=d.purchase_id WHERE p.account_id=$1 AND d.debt_minor>0", &[&op.account_id]).await.map_err(db)?;
        for d in debts {
            tx.query_one("SELECT billing_collect_debt($1)", &[&d.get::<_, String>(0)])
                .await
                .map_err(db)?;
        }
        wake_closure(&tx, op.account_id).await?;
        tx.commit().await.map_err(db)
    }

    /// A new Stripe attempt owns the existing held entitlement. No release and
    /// re-reservation, so a processing delay cannot expire its refund value.
    pub async fn retry_refund(
        &self,
        old_id: Uuid,
        new_id: Uuid,
        reason: &str,
    ) -> Result<RefundOperation, BillingError> {
        if old_id == new_id || reason.trim().is_empty() {
            return Err(BillingError::Invalid(
                "retry needs a new id and reason".into(),
            ));
        }
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let first = get_refund(&tx, old_id).await?;
        lock_account(&tx, first.account_id).await?;
        if let Some(row) = tx
            .query_opt(
                "SELECT id FROM billing_refund_operation WHERE replaces=$1",
                &[&old_id],
            )
            .await
            .map_err(db)?
        {
            if row.get::<_, Uuid>(0) != new_id {
                return Err(BillingError::Conflict(
                    "refund already has a replacement attempt".into(),
                ));
            }
            let op = get_refund(&tx, new_id).await?;
            tx.commit().await.map_err(db)?;
            return Ok(op);
        }
        let old = get_refund(&tx, old_id).await?;
        if !matches!(old.state.as_str(), "failed" | "canceled") {
            return Err(BillingError::Conflict(
                "only a definitively failed refund may be retried".into(),
            ));
        }
        tx.execute(
            "INSERT INTO billing_operation(id,account_id,kind,reason) VALUES($1,$2,'refund',$3)",
            &[&new_id, &old.account_id, &reason],
        )
        .await
        .map_err(db)?;
        tx.execute("INSERT INTO billing_refund_operation(id,purchase_id,credits,amount_minor,replaces) VALUES($1,$2,$3,$4,$5)",
            &[&new_id,&old.purchase_id,&old.credits,&old.amount_minor,&old_id]).await.map_err(db)?;
        tx.execute(
            "UPDATE billing_refund_operation SET state='superseded' WHERE id=$1",
            &[&old_id],
        )
        .await
        .map_err(db)?;
        tx.execute(
            "INSERT INTO billing_event(id,kind,object_id) VALUES($1,'billing.refund.resume',$2)",
            &[&format!("local:refund:{new_id}"), &new_id.to_string()],
        )
        .await
        .map_err(db)?;
        let op = get_refund(&tx, new_id).await?;
        tx.commit().await.map_err(db)?;
        Ok(op)
    }

    /// Explicit operator attestation of an alternate payout. It is never
    /// inferred from a webhook and cannot coexist with an ambiguous submission.
    pub async fn confirm_external_refund(
        &self,
        id: Uuid,
        reference: &str,
    ) -> Result<(), BillingError> {
        if reference.trim().is_empty() {
            return Err(BillingError::Invalid(
                "alternate payout needs its reference".into(),
            ));
        }
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let first = get_refund(&tx, id).await?;
        lock_account(&tx, first.account_id).await?;
        if let Some(row) = tx
            .query_opt(
                "SELECT reference FROM billing_external_refund WHERE operation_id=$1",
                &[&id],
            )
            .await
            .map_err(db)?
        {
            if row.get::<_, String>(0) != reference {
                return Err(BillingError::Conflict(
                    "alternate payout reference changed".into(),
                ));
            }
            tx.commit().await.map_err(db)?;
            return Ok(());
        }
        let op = get_refund(&tx, id).await?;
        if !matches!(op.state.as_str(), "prepared" | "failed" | "canceled") {
            return Err(BillingError::Conflict(
                "refund outcome does not permit an alternate payout".into(),
            ));
        }
        tx.execute(
            "INSERT INTO billing_external_refund(operation_id,reference) VALUES($1,$2)",
            &[&id, &reference],
        )
        .await
        .map_err(db)?;
        tx.execute(
            "UPDATE billing_refund_operation SET state='succeeded' WHERE id=$1",
            &[&id],
        )
        .await
        .map_err(db)?;
        tx.execute(
            "UPDATE billing_event SET completed_at=clock_timestamp(),completed_generation=generation WHERE id=$1",
            &[&format!("local:refund:{id}")],
        )
        .await
        .map_err(db)?;
        tx.query_one("SELECT billing_refresh_charge($1)", &[&op.purchase_id])
            .await
            .map_err(db)?;
        wake_closure(&tx, op.account_id).await?;
        tx.commit().await.map_err(db)
    }

    pub async fn list_purchases(&self, account: Uuid) -> Result<Vec<Purchase>, BillingError> {
        Ok(self
            .client()
            .await?
            .query(
                "SELECT * FROM billing_purchase WHERE account_id=$1 ORDER BY purchased_at,id",
                &[&account],
            )
            .await
            .map_err(db)?
            .into_iter()
            .map(row_purchase)
            .collect())
    }
    pub async fn list_refunds(&self, account: Uuid) -> Result<Vec<RefundOperation>, BillingError> {
        Ok(self
            .client()
            .await?
            .query(
                &format!("{REFUND_SELECT} WHERE o.account_id=$1 ORDER BY o.created_at,r.id"),
                &[&account],
            )
            .await
            .map_err(db)?
            .into_iter()
            .map(row_refund)
            .collect())
    }
    pub async fn debts(&self, account: Uuid) -> Result<Vec<Debt>, BillingError> {
        Ok(self.client().await?.query("SELECT p.currency,SUM(d.debt_minor)::TEXT AS amount FROM billing_dispute d \
            JOIN billing_purchase p ON p.id=d.purchase_id WHERE p.account_id=$1 AND d.debt_minor>0 GROUP BY p.currency ORDER BY p.currency", &[&account]).await.map_err(db)?
            .into_iter().map(|r|Debt{currency:r.get(0),amount_minor:r.get(1)}).collect())
    }
    pub async fn account_state(&self, account: Uuid) -> Result<String, BillingError> {
        self.client()
            .await?
            .query_opt("SELECT billing_state FROM account WHERE id=$1", &[&account])
            .await
            .map_err(db)?
            .map(|r| r.get(0))
            .ok_or_else(|| BillingError::NotFound("account not found".into()))
    }
    pub async fn begin_close(&self, account: Uuid) -> Result<(), BillingError> {
        self.begin_close_with_reason(account, "account closure requested")
            .await
    }
    pub async fn begin_close_with_reason(
        &self,
        account: Uuid,
        reason: &str,
    ) -> Result<(), BillingError> {
        if reason.trim().is_empty() {
            return Err(BillingError::Invalid("closure requires a reason".into()));
        }
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        lock_account(&tx, account).await?;
        let count=tx.execute("UPDATE account SET billing_state='closing',billing_closed_at=clock_timestamp() WHERE id=$1 AND billing_state='open'", &[&account]).await.map_err(db)?;
        if count > 0 {
            tx.execute(
                "INSERT INTO billing_operation(account_id,kind,reason) VALUES($1,'closure',$2)",
                &[&account, &reason],
            )
            .await
            .map_err(db)?;
        }
        tx.execute("INSERT INTO billing_event(id,kind,object_id) VALUES($1,'billing.close.resume',$2) ON CONFLICT(id) DO NOTHING", &[&format!("local:close:{account}"),&account.to_string()]).await.map_err(db)?;
        tx.commit().await.map_err(db)
    }
    pub async fn list_refundable_purchases(
        &self,
        account: Uuid,
    ) -> Result<Vec<(Purchase, i64)>, BillingError> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT p.*,SUM(cl.delta)::BIGINT AS remaining FROM billing_purchase p \
            JOIN account a ON a.id=p.account_id JOIN credit_ledger cl ON cl.purchase_id=p.id \
            WHERE p.account_id=$1 AND (p.expires_at>COALESCE(a.billing_closed_at,clock_timestamp()) OR p.purchased_at>=a.billing_closed_at) \
            GROUP BY p.id HAVING SUM(cl.delta)>0 ORDER BY p.expires_at,p.id",
            &[&account],
        )
        .await
        .map_err(db)?
        .into_iter()
        .map(|r| {
            let amount = r.get("remaining");
            (row_purchase(r), amount)
        })
        .collect())
    }
    pub async fn finalize_close(&self, account: Uuid) -> Result<(), BillingError> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        lock_account(&tx, account).await?;
        let unsettled:bool=tx.query_one("SELECT EXISTS(SELECT 1 FROM billing_refund_operation r JOIN billing_operation o ON o.id=r.id \
            WHERE o.account_id=$1 AND r.state NOT IN ('succeeded','released','superseded')) OR EXISTS(SELECT 1 FROM billing_purchase p \
            JOIN account a ON a.id=p.account_id JOIN credit_ledger cl ON cl.purchase_id=p.id WHERE p.account_id=$1 \
            AND (p.expires_at>COALESCE(a.billing_closed_at,clock_timestamp()) OR p.purchased_at>=a.billing_closed_at) GROUP BY p.id HAVING SUM(cl.delta)>0)", &[&account]).await.map_err(db)?.get(0);
        if unsettled {
            return Err(BillingError::Conflict(
                "account still has unsettled refunds".into(),
            ));
        }
        if tx.execute("UPDATE account SET billing_state='closed' WHERE id=$1 AND billing_state IN ('closing','closed')", &[&account]).await.map_err(db)?!=1 {
            return Err(BillingError::Conflict("account closure has not started".into()));
        }
        tx.commit().await.map_err(db)
    }

    pub async fn enqueue_event(
        &self,
        id: &str,
        kind: &str,
        object_id: &str,
    ) -> Result<bool, BillingError> {
        let c = self.client().await?;
        check_protocol(&c, true).await?;
        let inserted=c.execute("INSERT INTO billing_event(id,kind,object_id) VALUES($1,$2,$3) ON CONFLICT(id) DO NOTHING", &[&id,&kind,&object_id]).await.map_err(db)?;
        if inserted == 0 {
            let old = c
                .query_one(
                    "SELECT kind,object_id,protocol_version FROM billing_event WHERE id=$1",
                    &[&id],
                )
                .await
                .map_err(db)?;
            if old.get::<_, String>(0) != kind || old.get::<_, String>(1) != object_id {
                return Err(BillingError::Conflict("event id payload changed".into()));
            }
            if old.get::<_, i32>(2) != INGRESS_PROTOCOL_VERSION {
                return Err(BillingError::Conflict("event protocol changed".into()));
            }
        }
        Ok(inserted > 0)
    }

    /// One row per account bounds storage for repeated repair requests. Each
    /// request advances a generation, so a request arriving during an active
    /// provider read cannot disappear when that earlier read completes.
    pub async fn request_reconciliation(
        &self,
        account: Uuid,
    ) -> Result<RepairStatus, BillingError> {
        let c = self.client().await?;
        check_protocol(&c, true).await?;
        let id = format!("billing.reconcile:{account}");
        let row = c.query_opt(
            "INSERT INTO billing_event(id,kind,object_id) \
             SELECT $1,'billing.account.reconcile',$2 FROM account WHERE id=$3 \
             ON CONFLICT(id) DO UPDATE SET generation=billing_event.generation+1,completed_at=NULL, \
             next_attempt_at=CASE WHEN billing_event.completed_at IS NOT NULL THEN clock_timestamp() ELSE billing_event.next_attempt_at END, \
             attempts=CASE WHEN billing_event.completed_at IS NOT NULL THEN 0 ELSE billing_event.attempts END, \
             last_error=CASE WHEN billing_event.completed_at IS NOT NULL THEN NULL ELSE billing_event.last_error END \
             WHERE billing_event.kind='billing.account.reconcile' AND billing_event.object_id=$2 \
             AND billing_event.protocol_version=1 \
             RETURNING generation,completed_generation,report,last_error",
            &[&id, &account.to_string(), &account],
        ).await.map_err(db)?.ok_or_else(|| BillingError::Conflict("account repair cannot be requested".into()))?;
        row_repair_status(row)
    }

    pub async fn reconciliation_status(&self, account: Uuid) -> Result<RepairStatus, BillingError> {
        let c = self.client().await?;
        check_protocol(&c, true).await?;
        let Some(row) = c.query_opt(
            "SELECT kind,object_id,protocol_version,generation,completed_generation,report,last_error \
             FROM billing_event WHERE id=$1", &[&format!("billing.reconcile:{account}")],
        ).await.map_err(db)? else { return Ok(RepairStatus::default()); };
        if row.get::<_, String>("kind") != "billing.account.reconcile"
            || row.get::<_, String>("object_id") != account.to_string()
            || row.get::<_, i32>("protocol_version") != INGRESS_PROTOCOL_VERSION
        {
            return Err(BillingError::ServiceUnavailable(
                "account repair protocol is incompatible".into(),
            ));
        }
        row_repair_status(row)
    }

    pub async fn pending_events(&self, limit: i64) -> Result<Vec<PendingEvent>, BillingError> {
        let limit = limit.clamp(1, 100);
        Ok(self
            .client()
            .await?
            .query(
                "SELECT id,kind,object_id,attempts,protocol_version,generation FROM billing_event WHERE completed_at IS NULL \
            AND next_attempt_at<=clock_timestamp() ORDER BY next_attempt_at,received_at,id LIMIT $1",
                &[&limit],
            )
            .await
            .map_err(db)?
            .into_iter()
            .map(row_pending_event)
            .collect())
    }

    /// Claim a candidate on a detached session, then re-read its eligibility.
    /// The selection above is only a hint: another process can complete the
    /// item before this process gets here. No SQL transaction spans Stripe I/O.
    pub async fn claim_event(&self, id: &str) -> Result<Option<EventGuard>, BillingError> {
        static LIMIT: OnceLock<Arc<Semaphore>> = OnceLock::new();
        let permit = LIMIT
            .get_or_init(|| Arc::new(Semaphore::new(4)))
            .clone()
            .try_acquire_owned()
            .map_err(|_| BillingError::ServiceUnavailable("billing workers are busy".into()))?;
        let client = Client::take(self.client().await?);
        let acquired: bool = client
            .query_one(
                "SELECT pg_try_advisory_lock(hashtext('eidola-billing-event'),hashtext($1))",
                &[&id],
            )
            .await
            .map_err(db)?
            .get(0);
        if !acquired {
            return Ok(None);
        }
        check_protocol(&client, false).await?;
        let Some(row) = client
            .query_opt(
                "SELECT id,kind,object_id,attempts,protocol_version,generation FROM billing_event \
             WHERE id=$1 AND completed_at IS NULL AND next_attempt_at<=clock_timestamp()",
                &[&id],
            )
            .await
            .map_err(db)?
        else {
            return Ok(None);
        };
        Ok(Some(EventGuard {
            client: Some(client),
            event: row_pending_event(row),
            _permit: permit,
        }))
    }
    pub async fn review_issues(&self, account: Uuid) -> Result<Vec<ReviewIssue>, BillingError> {
        Ok(self.client().await?.query("SELECT kind,id,reason FROM billing_unmatched_observation WHERE account_id=$1 ORDER BY kind,id", &[&account]).await.map_err(db)?
            .into_iter().map(|r|ReviewIssue{kind:r.get(0),id:r.get(1),reason:r.get(2)}).collect())
    }
    pub async fn lock_reconciliation(
        &self,
        account: Uuid,
    ) -> Result<ReconciliationGuard, BillingError> {
        self.lock_account_work(account, false).await
    }

    /// Checkout publishes v1 purchase metadata and coordinates with account
    /// reconciliation, but never executes the worker's evolving billing rules.
    pub async fn lock_checkout(&self, account: Uuid) -> Result<CheckoutGuard, BillingError> {
        Ok(CheckoutGuard {
            inner: self.lock_account_work(account, true).await?,
        })
    }

    async fn lock_account_work(
        &self,
        account: Uuid,
        ingress: bool,
    ) -> Result<ReconciliationGuard, BillingError> {
        static LIMIT: OnceLock<Arc<Semaphore>> = OnceLock::new();
        let permit = LIMIT
            .get_or_init(|| Arc::new(Semaphore::new(4)))
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                BillingError::ServiceUnavailable("billing reconciliation is busy".into())
            })?;
        // Detach so a session lock can never return to the pool, including on
        // future cancellation. A global limit bounds the extra connections.
        let client = Client::take(self.client().await?);
        let key = account.to_string();
        let acquired:bool=client.query_one("SELECT pg_try_advisory_lock(hashtext('eidola-billing-reconciliation'),hashtext($1))", &[&key]).await.map_err(db)?.get(0);
        if !acquired {
            return Err(BillingError::Conflict(
                "account reconciliation is already running".into(),
            ));
        }
        check_protocol(&client, ingress).await?;
        Ok(ReconciliationGuard {
            client: Some(client),
            account,
            _permit: permit,
        })
    }
}

pub struct CheckoutGuard {
    inner: ReconciliationGuard,
}

impl CheckoutGuard {
    pub async fn release(self) -> Result<(), BillingError> {
        self.inner.release().await
    }
}

/// A work item belongs to this database session until completion or drop.
/// If the process/connection disappears, another worker can claim the still
/// pending row. An old guard can never finish through a replacement connection.
pub struct EventGuard {
    client: Option<ClientWrapper>,
    event: PendingEvent,
    _permit: OwnedSemaphorePermit,
}

impl EventGuard {
    pub fn event(&self) -> &PendingEvent {
        &self.event
    }

    pub async fn complete(self, report: Option<&ReconcileReport>) -> Result<(), BillingError> {
        let report = report
            .map(serde_json::to_value)
            .transpose()
            .map_err(|_| BillingError::Parse("repair report serialization failed".into()))?;
        self.client.as_ref().expect("guard not released").execute(
            "UPDATE billing_event SET completed_generation=GREATEST(completed_generation,$2), \
             completed_at=CASE WHEN generation=$2 THEN COALESCE(completed_at,clock_timestamp()) ELSE NULL END, \
             report=COALESCE($3::JSONB,report),last_error=NULL,attempts=0,next_attempt_at=clock_timestamp() \
             WHERE id=$1 AND completed_generation<=$2 AND generation>=$2",
            &[&self.event.id, &self.event.generation, &report],
        ).await.map_err(db)?;
        Ok(())
    }

    pub async fn fail(self, reason: &str) -> Result<(), BillingError> {
        // Only fixed categories enter this column; never a provider response.
        let reason = match reason {
            "requires_review"
            | "provider_unavailable"
            | "database_unavailable"
            | "conflict"
            | "unsupported_kind"
            | "unsupported_protocol" => reason,
            _ => "reconciliation_failed",
        };
        self.client.as_ref().expect("guard not released").execute(
            "UPDATE billing_event SET attempts=LEAST(attempts+1,30),last_error=$2, \
             next_attempt_at=clock_timestamp()+make_interval(secs=>LEAST(3600,5*power(2,LEAST(attempts,10)))::DOUBLE PRECISION) \
             WHERE id=$1 AND completed_at IS NULL",
            &[&self.event.id,&reason],
        ).await.map_err(db)?;
        Ok(())
    }
}

pub struct ReconciliationGuard {
    client: Option<ClientWrapper>,
    account: Uuid,
    _permit: OwnedSemaphorePermit,
}

impl ReconciliationGuard {
    /// Recheck the actual lock connection immediately before provider writes.
    /// An uncertain provider outcome still uses the durable operation identity.
    pub async fn ensure_compatible(&self) -> Result<(), BillingError> {
        check_protocol(self.client.as_ref().expect("guard not released"), false).await
    }

    pub async fn mark_refund_rejected(&mut self, id: Uuid) -> Result<(), BillingError> {
        let tx = self
            .client
            .as_mut()
            .expect("guard not released")
            .transaction()
            .await
            .map_err(db)?;
        lock_account(&tx, self.account).await?;
        let op = get_refund(&tx, id).await?;
        if op.account_id != self.account {
            return Err(BillingError::NotFound(
                "refund not found for account".into(),
            ));
        }
        if tx.execute("UPDATE billing_refund_operation SET state='failed' WHERE id=$1 AND state='submitting' AND refund_id IS NULL", &[&id]).await.map_err(db)?!=1 {
            return Err(BillingError::Conflict("refund rejection does not match submission".into()));
        }
        tx.execute(
            "UPDATE billing_event SET completed_at=clock_timestamp(),completed_generation=generation WHERE id=$1",
            &[&format!("local:refund:{id}")],
        )
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)
    }
    pub async fn observe_refund(&mut self, refund: &RefundObservation) -> Result<(), BillingError> {
        let tx = self
            .client
            .as_mut()
            .expect("guard not released")
            .transaction()
            .await
            .map_err(db)?;
        observe_refund(&tx, refund, Some(self.account)).await?;
        tx.commit().await.map_err(db)
    }
    /// Commit on the very connection holding the provider-read lock. Losing
    /// that connection cannot apply an old snapshot through a different one.
    pub async fn apply_observations(
        &mut self,
        account: Uuid,
        purchases: &[Purchase],
        refunds: &[RefundObservation],
        disputes: &[DisputeObservation],
    ) -> Result<(), BillingError> {
        if account != self.account {
            return Err(BillingError::Invalid(
                "reconciliation account changed".into(),
            ));
        }
        let tx = self
            .client
            .as_mut()
            .expect("guard not released")
            .transaction()
            .await
            .map_err(db)?;
        apply(&tx, account, purchases, refunds, disputes).await?;
        tx.commit().await.map_err(db)
    }
    pub async fn release(mut self) -> Result<(), BillingError> {
        if let Some(client) = self.client.take() {
            client.query_one("SELECT pg_advisory_unlock(hashtext('eidola-billing-reconciliation'),hashtext($1))", &[&self.account.to_string()]).await.map_err(db)?;
        }
        Ok(())
    }
}

async fn quote(c: &impl GenericClient, r: &RefundRequest) -> Result<RefundOperation, BillingError> {
    if r.reason.trim().is_empty()
        || r.credits < 0
        || (r.credits > 0 && r.cash_minor.is_some())
        || (r.credits == 0 && r.cash_minor.is_none_or(|n| n <= 0))
    {
        return Err(BillingError::Invalid(
            "refund needs credits or an explicit cash-only amount and reason".into(),
        ));
    }
    let p = get_purchase(c, &r.purchase_id).await?;
    if p.account_id != r.account_id {
        return Err(BillingError::NotFound(
            "purchase not found for account".into(),
        ));
    }
    let row=c.query_one("SELECT (p.expires_at>COALESCE(a.billing_closed_at,clock_timestamp()) OR COALESCE(p.purchased_at>=a.billing_closed_at,FALSE)) AS eligible, \
        COALESCE((SELECT SUM(delta) FROM credit_ledger WHERE purchase_id=p.id),0)::BIGINT AS remaining, \
        clock_timestamp() AS requested_at FROM billing_purchase p JOIN account a ON a.id=p.account_id WHERE p.id=$1", &[&p.id]).await.map_err(db)?;
    if r.credits > 0
        && (!row.get::<_, bool>("eligible") || row.get::<_, i64>("remaining") < r.credits)
    {
        return Err(BillingError::Conflict(
            "requested credits are expired, converted, or already reserved".into(),
        ));
    }
    let mut amount = if let Some(cash) = r.cash_minor {
        cash
    } else {
        let already: i64 = c
            .query_one(
                "SELECT COALESCE(SUM(credits),0)::BIGINT FROM billing_refund_operation \
            WHERE purchase_id=$1 AND state NOT IN ('released','superseded')",
                &[&p.id],
            )
            .await
            .map_err(db)?
            .get(0);
        let rounded = |credits: i128| {
            (i128::from(p.paid_minor) * credits + i128::from(p.credits) / 2) / i128::from(p.credits)
        };
        i64::try_from(
            rounded(i128::from(already) + i128::from(r.credits)) - rounded(i128::from(already)),
        )
        .map_err(|_| BillingError::Invalid("refund amount exceeds supported range".into()))?
    };
    let allocated:i64=c.query_one("SELECT (COALESCE((SELECT SUM(r.amount_minor) FROM billing_refund_operation r WHERE r.purchase_id=$1 AND r.state NOT IN ('released','superseded')),0) \
        + COALESCE((SELECT SUM(ro.amount_minor) FROM billing_refund_observation ro LEFT JOIN billing_refund_operation prior ON prior.id=ro.operation_id \
        WHERE ro.charge_id=$2 AND (ro.operation_id IS NULL OR prior.state IN ('released','superseded')) \
        AND ro.status IN ('pending','requires_action','succeeded')),0) \
        + COALESCE((SELECT SUM(target_minor) FROM billing_dispute WHERE purchase_id=$1),0))::BIGINT", &[&p.id,&p.charge_id]).await.map_err(db)?.get(0);
    let remaining = p.paid_minor.saturating_sub(allocated).max(0);
    if r.credits > 0 {
        // Cumulative rounding makes a complete credit refund return the whole
        // purchase price. Prior goodwill refunds/chargebacks can leave less
        // cash than the remaining credits' nominal value, including zero.
        amount = amount.min(remaining);
    } else if amount > remaining {
        return Err(BillingError::Conflict(
            "refund would return payment more than once".into(),
        ));
    }
    Ok(RefundOperation {
        id: r.id,
        account_id: r.account_id,
        purchase_id: r.purchase_id.clone(),
        credits: r.credits,
        amount_minor: amount,
        currency: p.currency,
        charge_id: p.charge_id.unwrap_or_default(),
        state: "preview".into(),
        refund_id: None,
        created_at: row.get("requested_at"),
        submitted_at: None,
    })
}
