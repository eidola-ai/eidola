//! Real PostgreSQL tests. Each test creates a unique schema in the explicitly
//! supplied disposable database; none reads or changes another schema's data.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use deadpool_postgres::{Manager, Pool};
use eidola_billing::store::{
    DisputeObservation, Purchase, RefundObservation, RefundRequest, Store,
};
use tokio_postgres::NoTls;
use uuid::Uuid;

struct Fixture {
    pool: Pool,
    store: Store,
    account: Uuid,
    key: String,
}

fn now() -> SystemTime {
    UNIX_EPOCH
        + Duration::from_secs(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
}

async fn fixture() -> Fixture {
    let url = std::env::var("BILLING_TEST_DATABASE_URL")
        .expect("BILLING_TEST_DATABASE_URL must name a disposable PostgreSQL database");
    let (root, connection) = tokio_postgres::connect(&url, NoTls).await.unwrap();
    tokio::spawn(async move { connection.await.unwrap() });
    let schema = format!("billing_test_{}", Uuid::new_v4().simple());
    root.batch_execute(&format!(
        "CREATE SCHEMA {schema}; SET search_path TO {schema}"
    ))
    .await
    .unwrap();
    root.batch_execute(include_str!(
        "../../eidola-server-gateway/schema/schema.sql"
    ))
    .await
    .unwrap();
    let mut config: tokio_postgres::Config = url.parse().unwrap();
    config.options(format!("-c search_path={schema}"));
    let pool = Pool::builder(Manager::new(config, NoTls))
        .max_size(8)
        .build()
        .unwrap();
    let account = Uuid::new_v4();
    root.execute(
        "INSERT INTO account(id,secret_hash,stripe_customer_id) VALUES($1,'test secret hash',$2)",
        &[&account, &format!("cus_{account}")],
    )
    .await
    .unwrap();
    let key:String=root.query_one("INSERT INTO issuer_key(id,private_key_enc,public_key,domain_separator,issue_from,issue_until,accept_until) \
        VALUES(encode(sha256('test public key'::BYTEA),'hex'),'test private key'::BYTEA,'test public key'::BYTEA,'test', \
        now()-interval '1 day',now()+interval '1 day',now()+interval '2 days') RETURNING id", &[]).await.unwrap().get(0);
    Fixture {
        store: Store::new(pool.clone()),
        pool,
        account,
        key,
    }
}

fn purchase(f: &Fixture, name: &str, credits: i64, paid: i64) -> Purchase {
    Purchase {
        id: format!("{name}_{}", Uuid::new_v4()),
        account_id: f.account,
        charge_id: if paid > 0 {
            Some(format!("ch_{}", Uuid::new_v4()))
        } else {
            None
        },
        payment_intent: None,
        invoice_id: None,
        currency: "usd".into(),
        paid_minor: paid,
        credits,
        purchased_at: now(),
        expires_at: now() + Duration::from_secs(86400),
        kind: "purchase".into(),
    }
}

fn refund(f: &Fixture, p: &Purchase, credits: i64, cash: Option<i64>) -> RefundRequest {
    RefundRequest {
        id: Uuid::new_v4(),
        account_id: f.account,
        purchase_id: p.id.clone(),
        credits,
        cash_minor: cash,
        reason: "support case".into(),
    }
}

fn dispute(p: &Purchase, amount: i64, status: &str) -> DisputeObservation {
    DisputeObservation {
        id: format!("dp_{}", Uuid::new_v4()),
        charge_id: p.charge_id.clone().unwrap(),
        amount_minor: amount,
        currency: p.currency.clone(),
        status: status.into(),
    }
}

async fn balance(f: &Fixture) -> i64 {
    f.pool
        .get()
        .await
        .unwrap()
        .query_one("SELECT available_balance($1)", &[&f.account])
        .await
        .unwrap()
        .get(0)
}

async fn issue(f: &Fixture, amount: i64, id: u8) -> String {
    f.pool
        .get()
        .await
        .unwrap()
        .query_one(
            "SELECT outcome FROM record_credential_issuance($1,$2,$3,$4,$5,$6)",
            &[
                &f.account,
                &vec![id; 32],
                &vec![id; 32],
                &f.key,
                &amount,
                &vec![id; 1],
            ],
        )
        .await
        .unwrap()
        .get(0)
}

#[test]
fn gateway_and_library_share_exact_billing_schema() {
    let schema = include_str!("../../eidola-server-gateway/schema/schema.sql");
    let block = schema
        .split("-- BEGIN SHARED BILLING OPERATIONS\n")
        .nth(1)
        .unwrap()
        .split("-- END SHARED BILLING OPERATIONS\n")
        .next()
        .unwrap();
    assert_eq!(block, include_str!("../schema/billing.sql"));
    assert_eq!(schema.matches("FUNCTION debit_account(").count(), 1);
    assert_eq!(schema.matches("FUNCTION available_balance(").count(), 1);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn purchase_identity_not_event_delivery_controls_grants() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    assert!(f.store.record_purchase(&p).await.unwrap());
    assert!(!f.store.record_purchase(&p).await.unwrap());
    let mut changed = p.clone();
    changed.credits = 2000;
    assert!(f.store.record_purchase(&changed).await.is_err());
    assert_eq!(balance(&f).await, 1000);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn equal_expiries_keep_purchase_prices_and_issuance_provenance() {
    let f = fixture().await;
    let a = purchase(&f, "a", 1000, 100);
    let mut b = purchase(&f, "b", 1000, 200);
    b.expires_at = a.expires_at;
    f.store.record_purchase(&a).await.unwrap();
    f.store.record_purchase(&b).await.unwrap();
    assert_eq!(issue(&f, 500, 1).await, "issued");
    let ra = f
        .store
        .prepare_refund(&refund(&f, &a, 500, None))
        .await
        .unwrap();
    let rb = f
        .store
        .prepare_refund(&refund(&f, &b, 1000, None))
        .await
        .unwrap();
    assert_eq!(ra.amount_minor, 50);
    assert_eq!(rb.amount_minor, 200);
    assert_eq!(balance(&f).await, 0);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn refund_reservation_and_issuance_cannot_spend_one_balance_twice() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    f.store.record_purchase(&p).await.unwrap();
    let request = refund(&f, &p, 1000, None);
    let (held, minted) = tokio::join!(f.store.prepare_refund(&request), issue(&f, 1000, 1));
    assert_eq!(held.is_ok(), minted == "insufficient");
    assert!(minted == "issued" || minted == "insufficient");
    assert_eq!(balance(&f).await, 0);
    if held.is_ok() {
        assert_eq!(
            f.store.prepare_refund(&request).await.unwrap().id,
            request.id
        );
    }
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn failed_refunds_keep_holds_and_expired_entitlements_can_retry() {
    let f = fixture().await;
    let mut p = purchase(&f, "purchase", 1000, 100);
    p.expires_at = now() + Duration::from_secs(2);
    f.store.record_purchase(&p).await.unwrap();
    let r = refund(&f, &p, 1000, None);
    f.store.prepare_refund(&r).await.unwrap();
    f.store.mark_submitted(r.id).await.unwrap();
    assert!(f.store.release_refund(r.id).await.is_err());
    let observation = RefundObservation {
        id: "re_failed".into(),
        charge_id: p.charge_id.clone().unwrap(),
        amount_minor: 100,
        currency: "usd".into(),
        status: "failed".into(),
        operation_id: Some(r.id),
    };
    f.store.observe_refund(&observation).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(f.store.release_refund(r.id).await.is_err());
    let next = Uuid::new_v4();
    let retry = f
        .store
        .retry_refund(r.id, next, "retry failed payment")
        .await
        .unwrap();
    assert_eq!(retry.state, "prepared");
    assert_eq!(retry.credits, 1000);
    assert_eq!(retry.amount_minor, 100);
    f.store.observe_refund(&observation).await.unwrap(); // old failure cannot regress supersession
    assert_eq!(f.store.get_refund(r.id).await.unwrap().state, "superseded");
    f.store
        .confirm_external_refund(next, "bank transfer case 123")
        .await
        .unwrap();
    assert_eq!(f.store.get_refund(next).await.unwrap().state, "succeeded");
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn outside_refunds_are_cash_observations_without_credit_clawback() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    f.store.record_purchase(&p).await.unwrap();
    let r = RefundObservation {
        id: "re_external".into(),
        charge_id: p.charge_id.clone().unwrap(),
        amount_minor: 100,
        currency: "usd".into(),
        status: "succeeded".into(),
        operation_id: None,
    };
    f.store.observe_refund(&r).await.unwrap();
    f.store.observe_refund(&r).await.unwrap();
    assert_eq!(balance(&f).await, 1000);
    assert!(
        f.store
            .prepare_refund(&refund(&f, &p, 0, Some(1)))
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .prepare_refund(&refund(&f, &p, 1000, None))
            .await
            .unwrap()
            .amount_minor,
        0
    );
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn partial_refund_then_partial_or_full_dispute_never_deducts_twice() {
    for disputed in [80, 100] {
        let f = fixture().await;
        let p = purchase(&f, "purchase", 1000, 100);
        f.store.record_purchase(&p).await.unwrap();
        let request = refund(&f, &p, 200, None);
        let op = f.store.prepare_refund(&request).await.unwrap();
        let r = RefundObservation {
            id: "re_partial".into(),
            charge_id: p.charge_id.clone().unwrap(),
            amount_minor: 20,
            currency: "usd".into(),
            status: "succeeded".into(),
            operation_id: Some(op.id),
        };
        f.store.observe_refund(&r).await.unwrap();
        let d = dispute(&p, disputed, "needs_response");
        f.store.reconcile_dispute(&d).await.unwrap();
        assert_eq!(balance(&f).await, 0);
        assert!(f.store.debts(f.account).await.unwrap().is_empty());
        f.store.reconcile_dispute(&d).await.unwrap();
        assert_eq!(balance(&f).await, 0);
        let mut won = d;
        won.status = "won".into();
        f.store.reconcile_dispute(&won).await.unwrap();
        assert_eq!(balance(&f).await, 800);
    }
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn pending_goodwill_refund_does_not_reduce_a_chargeback() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    f.store.record_purchase(&p).await.unwrap();
    let r = RefundObservation {
        id: "re_pending".into(),
        charge_id: p.charge_id.clone().unwrap(),
        amount_minor: 20,
        currency: "usd".into(),
        status: "pending".into(),
        operation_id: None,
    };
    f.store.observe_refund(&r).await.unwrap();
    let d = dispute(&p, 100, "needs_response");
    f.store.reconcile_dispute(&d).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    let mut failed = r;
    failed.status = "failed".into();
    f.store.observe_refund(&failed).await.unwrap();
    assert_eq!(balance(&f).await, 0);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn cash_debt_survives_original_expiry_uses_other_lot_prices_and_leaves_free_credits() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    f.store.record_purchase(&p).await.unwrap();
    assert_eq!(issue(&f, 1000, 1).await, "issued");
    let mut d = dispute(&p, 100, "lost");
    f.store.reconcile_dispute(&d).await.unwrap();
    assert_eq!(balance(&f).await, -1000);
    let b = purchase(&f, "premium", 1000, 200);
    f.store.record_purchase(&b).await.unwrap();
    assert_eq!(balance(&f).await, 500);
    assert!(f.store.debts(f.account).await.unwrap().is_empty());
    assert_eq!(issue(&f, 500, 2).await, "issued");
    d.status = "won".into();
    f.store.reconcile_dispute(&d).await.unwrap();
    assert_eq!(balance(&f).await, 500);
    let fresh = fixture().await;
    let p = purchase(&fresh, "payment", 1000, 100);
    fresh.store.record_purchase(&p).await.unwrap();
    issue(&fresh, 1000, 1).await;
    fresh
        .store
        .reconcile_dispute(&dispute(&p, 100, "lost"))
        .await
        .unwrap();
    let free = purchase(&fresh, "free", 77, 0);
    fresh.store.record_purchase(&free).await.unwrap();
    assert_eq!(balance(&fresh).await, 77);
    assert_eq!(issue(&fresh, 77, 2).await, "issued");
    assert_eq!(balance(&fresh).await, -1000);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn reversal_preserves_expiry_and_won_before_created_creates_no_credit() {
    let f = fixture().await;
    let mut p = purchase(&f, "purchase", 1000, 100);
    p.expires_at = now() + Duration::from_secs(2);
    f.store.record_purchase(&p).await.unwrap();
    let won = dispute(&p, 100, "won");
    f.store.reconcile_dispute(&won).await.unwrap();
    assert_eq!(balance(&f).await, 1000);
    let mut d = dispute(&p, 100, "lost");
    f.store.reconcile_dispute(&d).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    tokio::time::sleep(Duration::from_secs(3)).await;
    d.status = "won".into();
    f.store.reconcile_dispute(&d).await.unwrap();
    assert_eq!(balance(&f).await, 0);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn purchase_and_chargeback_apply_atomically_and_validation_rolls_back() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    let d = dispute(&p, 100, "lost");
    f.store
        .apply_observations(f.account, std::slice::from_ref(&p), &[], &[d])
        .await
        .unwrap();
    assert_eq!(balance(&f).await, 0);
    let q = purchase(&f, "invalid", 1000, 100);
    let mut invalid = dispute(&q, 100, "lost");
    invalid.currency = "eur".into();
    assert!(
        f.store
            .apply_observations(f.account, std::slice::from_ref(&q), &[], &[invalid])
            .await
            .is_err()
    );
    assert!(f.store.get_purchase(&q.id).await.is_err());
    assert_eq!(balance(&f).await, 0);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn unmatched_cash_is_recorded_without_blocking_unrelated_purchases() {
    let f = fixture().await;
    let p = purchase(&f, "known", 1000, 100);
    let r = RefundObservation {
        id: "re_unknown".into(),
        charge_id: "ch_unknown".into(),
        amount_minor: 100,
        currency: "usd".into(),
        status: "succeeded".into(),
        operation_id: None,
    };
    f.store
        .apply_observations(f.account, &[p], &[r], &[])
        .await
        .unwrap();
    assert_eq!(balance(&f).await, 1000);
    assert_eq!(
        f.pool
            .get()
            .await
            .unwrap()
            .query_one("SELECT count(*) FROM billing_unmatched_observation", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn closing_freezes_refund_eligibility_and_keeps_idempotent_issuance_recovery() {
    let f = fixture().await;
    let mut p = purchase(&f, "purchase", 1000, 100);
    p.expires_at = now() + Duration::from_secs(2);
    f.store.record_purchase(&p).await.unwrap();
    assert_eq!(issue(&f, 100, 1).await, "issued");
    f.store
        .begin_close_with_reason(f.account, "support closure")
        .await
        .unwrap();
    assert_eq!(issue(&f, 100, 1).await, "stored");
    assert_eq!(issue(&f, 100, 2).await, "closed");
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(
        f.store.list_refundable_purchases(f.account).await.unwrap()[0].1,
        900
    );
    let r = refund(&f, &p, 900, None);
    f.store.prepare_refund(&r).await.unwrap();
    assert!(f.store.finalize_close(f.account).await.is_err());
    f.store
        .confirm_external_refund(r.id, "alternate closure payout")
        .await
        .unwrap();
    f.store.finalize_close(f.account).await.unwrap();
    assert_eq!(f.store.account_state(f.account).await.unwrap(), "closed");
    let late = purchase(&f, "late", 500, 50);
    f.store.record_purchase(&late).await.unwrap();
    assert_eq!(issue(&f, 100, 3).await, "closed");
    assert_eq!(
        f.store.list_refundable_purchases(f.account).await.unwrap()[0].1,
        500
    );
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn guard_cancellation_releases_database_lock_and_pool_capacity() {
    let f = fixture().await;
    let guard = f.store.lock_reconciliation(f.account).await.unwrap();
    assert!(f.store.lock_reconciliation(f.account).await.is_err());
    drop(guard);
    let mut next = None;
    for _ in 0..50 {
        match f.store.lock_reconciliation(f.account).await {
            Ok(g) => {
                next = Some(g);
                break;
            }
            Err(_) => tokio::task::yield_now().await,
        }
    }
    let mut next = next.expect("dropping a guard closes its lock connection");
    let p = purchase(&f, "purchase", 1000, 100);
    next.apply_observations(f.account, &[p], &[], &[])
        .await
        .unwrap();
    assert_eq!(balance(&f).await, 1000);
    next.release().await.unwrap();
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn partial_rounding_is_cumulative_and_can_cancel_credits_after_all_cash_returned() {
    for (credits, paid, expected) in [(2, 1, vec![1, 0]), (3, 100, vec![33, 34, 33])] {
        let f = fixture().await;
        let p = purchase(&f, "tiny", credits, paid);
        f.store.record_purchase(&p).await.unwrap();
        for cash in expected {
            let op = f
                .store
                .prepare_refund(&refund(&f, &p, 1, None))
                .await
                .unwrap();
            assert_eq!(op.amount_minor, cash);
        }
        assert_eq!(balance(&f).await, 0);
    }
    let f = fixture().await;
    let p = purchase(&f, "goodwill", 1000, 100);
    f.store.record_purchase(&p).await.unwrap();
    let op = f
        .store
        .prepare_refund(&refund(&f, &p, 0, Some(100)))
        .await
        .unwrap();
    f.store
        .confirm_external_refund(op.id, "full goodwill payout")
        .await
        .unwrap();
    assert_eq!(balance(&f).await, 1000);
    f.store.begin_close(f.account).await.unwrap();
    let cancel = f
        .store
        .prepare_refund(&refund(&f, &p, 1000, None))
        .await
        .unwrap();
    assert_eq!(cancel.amount_minor, 0);
    assert_eq!(cancel.state, "succeeded");
    f.store.finalize_close(f.account).await.unwrap();
    assert_eq!(balance(&f).await, 0);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn dispute_reversal_services_other_debt_before_restored_credits_can_issue() {
    let f = fixture().await;
    let a = purchase(&f, "a", 1000, 100);
    let b = purchase(&f, "b", 1000, 100);
    f.store.record_purchase(&a).await.unwrap();
    f.store.record_purchase(&b).await.unwrap();
    assert_eq!(issue(&f, 1000, 1).await, "issued");
    let mut d1 = dispute(&a, 100, "lost");
    let mut d2 = dispute(&b, 100, "lost");
    f.store.reconcile_dispute(&d1).await.unwrap();
    f.store.reconcile_dispute(&d2).await.unwrap();
    assert_eq!(balance(&f).await, -1000);
    d1.status = "won".into();
    f.store.reconcile_dispute(&d1).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    assert_eq!(issue(&f, 1, 2).await, "insufficient");
    d2.status = "won".into();
    f.store.reconcile_dispute(&d2).await.unwrap();
    assert_eq!(balance(&f).await, 1000);
    f.store.reconcile_dispute(&d1).await.unwrap();
    f.store.reconcile_dispute(&d2).await.unwrap();
    assert_eq!(balance(&f).await, 1000);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn durable_local_intents_and_provider_retries_survive_process_loss() {
    let f = fixture().await;
    let p = purchase(&f, "purchase", 1000, 100);
    f.store.record_purchase(&p).await.unwrap();
    let op = f
        .store
        .prepare_refund(&refund(&f, &p, 100, None))
        .await
        .unwrap();
    let pending = f.store.pending_events(100).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].object_id, op.id.to_string());
    f.store
        .fail_event(&pending[0].id, "provider_unavailable")
        .await
        .unwrap();
    assert!(f.store.pending_events(100).await.unwrap().is_empty());
    let first = f.store.mark_submitted(op.id).await.unwrap();
    let second = f.store.mark_submitted(op.id).await.unwrap();
    assert_eq!(first.submitted_at, second.submitted_at);
    f.store.mark_refund_rejected(op.id).await.unwrap();
    f.store.release_refund(op.id).await.unwrap();
    assert_eq!(balance(&f).await, 1000);
    f.store.begin_close(f.account).await.unwrap();
    assert_eq!(
        f.store.pending_events(100).await.unwrap()[0].kind,
        "billing.close.resume"
    );
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn chargebacks_offset_closure_refunds_even_after_frozen_credits_expire() {
    let f = fixture().await;
    let mut a = purchase(&f, "a", 1000, 100);
    let mut b = purchase(&f, "b", 1000, 100);
    a.expires_at = now() + Duration::from_secs(2);
    b.expires_at = a.expires_at;
    f.store.record_purchase(&a).await.unwrap();
    f.store.record_purchase(&b).await.unwrap();
    issue(&f, 1000, 1).await;
    f.store.begin_close(f.account).await.unwrap();
    tokio::time::sleep(Duration::from_secs(3)).await;
    f.store
        .reconcile_dispute(&dispute(&a, 100, "lost"))
        .await
        .unwrap();
    assert!(
        f.store
            .list_refundable_purchases(f.account)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(f.store.debts(f.account).await.unwrap().is_empty());
    f.store.finalize_close(f.account).await.unwrap();
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn late_collected_invoices_keep_actual_expiry_and_are_refundable_after_closure() {
    let f = fixture().await;
    let mut expired = purchase(&f, "late_invoice", 1000, 100);
    expired.kind = "subscription_renewal".into();
    expired.expires_at = now() - Duration::from_secs(86400);
    f.store.record_purchase(&expired).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    assert!(
        f.store
            .list_refundable_purchases(f.account)
            .await
            .unwrap()
            .is_empty()
    );
    f.store.begin_close(f.account).await.unwrap();
    f.store.finalize_close(f.account).await.unwrap();
    let mut late = purchase(&f, "late_after_close", 1000, 100);
    late.kind = "subscription_renewal".into();
    late.purchased_at = SystemTime::now();
    late.expires_at = now() - Duration::from_secs(3600);
    f.store.record_purchase(&late).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    assert_eq!(
        f.store.list_refundable_purchases(f.account).await.unwrap()[0].1,
        1000
    );
    let op = f
        .store
        .prepare_refund(&refund(&f, &late, 1000, None))
        .await
        .unwrap();
    assert_eq!(op.amount_minor, 100);
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn dispute_received_before_purchase_is_attached_before_grant_becomes_visible() {
    let f = fixture().await;
    let p = purchase(&f, "delayed", 1000, 100);
    let d = dispute(&p, 100, "lost");
    f.store
        .apply_observations(f.account, &[], &[], &[d])
        .await
        .unwrap();
    assert_eq!(balance(&f).await, 0);
    f.store.record_purchase(&p).await.unwrap();
    assert_eq!(balance(&f).await, 0);
    let pending: i64 = f
        .pool
        .get()
        .await
        .unwrap()
        .query_one(
            "SELECT count(*) FROM billing_unmatched_observation WHERE kind='dispute'",
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(pending, 0);
}
