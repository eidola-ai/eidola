//! Real PostgreSQL transactions with a stateful local Stripe stand-in.
//! Set BILLING_TEST_DATABASE_URL to an isolated test database; each test uses
//! its own schema and never modifies the database's existing application data.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{FromRequest, Request, State};
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Form, Json, Router};
use deadpool_postgres::{Manager, Pool};
use eidola_billing::BillingError;
use eidola_billing::service::BillingService;
use eidola_billing::store::{RefundRequest, Store};
use eidola_billing::stripe::StripeClient;
use serde_json::{Value, json};
use tokio_postgres::NoTls;
use uuid::Uuid;

#[derive(Default)]
struct Provider {
    created: i64,
    purchases: usize,
    refunds: Vec<Value>,
    posts: Vec<(String, String)>,
    lose_next_response: bool,
    reject_refund_posts: bool,
    subscription_active: bool,
    cancellations: usize,
    refund_pending: bool,
    response_gate: Option<Arc<tokio::sync::Notify>>,
    post_started: Arc<tokio::sync::Notify>,
    snapshot_reads: usize,
    snapshot_gate: Option<Arc<tokio::sync::Notify>>,
    snapshot_started: Arc<tokio::sync::Notify>,
}

fn list(data: Vec<Value>) -> Value {
    json!({"data":data,"has_more":false})
}

async fn serve(State(state): State<Arc<Mutex<Provider>>>, request: Request) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let query = request.uri().query().unwrap_or("").to_owned();
    let key = request
        .headers()
        .get("Idempotency-Key")
        .and_then(|s| s.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let form = if method == Method::POST {
        Form::<HashMap<String, String>>::from_request(request, &())
            .await
            .unwrap()
            .0
    } else {
        HashMap::new()
    };
    let (result, response_gate) = {
        let mut provider = state.lock().unwrap();
        let mut response_gate = None;
        let result = match (method.as_str(), path.as_str()) {
        ("GET", "/charges") => {
            provider.snapshot_reads += 1;
            provider.snapshot_started.notify_one();
            response_gate = provider.snapshot_gate.clone();
            list((1..=provider.purchases).map(|n| json!({
            "id":format!("ch_{n}"),"customer":"cus_1","payment_intent":format!("pi_{n}"),
            "amount":1000,"amount_refunded":0,"currency":"usd","paid":true,"captured":true,
            "status":"succeeded","created":provider.created
        })).collect())
        },
        ("GET", "/checkout/sessions") => list((1..=provider.purchases).map(|n| json!({
            "id":format!("cs_{n}"),"customer":"cus_1","mode":"payment","status":"complete","payment_status":"paid",
            "amount_total":1000,"currency":"usd","payment_intent":format!("pi_{n}"),"created":provider.created,
            "metadata":{"eidola_schema":"1","eidola_credits":"2000000","eidola_expiry":"purchase_year","eidola_price":"price_1"}
        })).collect()),
        ("GET", "/invoices" | "/disputes") => list(vec![]),
        ("GET", "/refunds") => {
            let charge = query.split('&').find_map(|v| v.strip_prefix("charge="));
            list(provider.refunds.iter().filter(|r| Some(r["charge"].as_str().unwrap()) == charge).cloned().collect())
        }
        ("GET", "/subscriptions") => list(if provider.subscription_active {
            vec![json!({"id":"sub_1","status":"active","items":{"data":[{"current_period_end":provider.created+3600}]}})]
        } else { vec![] }),
        ("DELETE", "/subscriptions/sub_1") => {
            provider.subscription_active = false;
            provider.cancellations += 1;
            json!({"id":"sub_1","status":"canceled"})
        }
        ("POST", "/refunds") => {
            let operation = form["metadata[eidola_operation]"].clone();
            provider.posts.push((key, operation.clone()));
            if provider.reject_refund_posts {
                return (StatusCode::FORBIDDEN, Json(json!({"error":{"type":"invalid_request_error"}}))).into_response();
            }
            let refund = json!({"id":format!("re_{}",provider.posts.len()),"charge":form["charge"],
                "amount":form["amount"].parse::<i64>().unwrap(),"currency":"usd","status":if provider.refund_pending {"pending"} else {"succeeded"},
                "metadata":{"eidola_operation":operation}});
            provider.refunds.push(refund.clone());
            provider.post_started.notify_one();
            response_gate = provider.response_gate.clone();
            if provider.lose_next_response {
                provider.lose_next_response = false;
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":{"type":"api_error"}}))).into_response();
            }
            refund
        }
        ("GET", path) if path.starts_with("/payment_intents/pi_") => {
            let n = path.strip_prefix("/payment_intents/pi_").unwrap();
            json!({"id":format!("pi_{n}"),"customer":"cus_1","latest_charge":format!("ch_{n}"),
                "status":"succeeded","currency":"usd","amount_received":1000})
        }
        _ => return (StatusCode::NOT_FOUND, Json(json!({"error":{"type":"invalid_request_error"}}))).into_response(),
    };
        (result, response_gate)
    };
    if let Some(gate) = response_gate {
        gate.notified().await;
    }
    Json(result).into_response()
}

struct Fixture {
    account: Uuid,
    schema: String,
    admin: tokio_postgres::Client,
    pool: Pool,
    service: Arc<BillingService>,
    stripe: StripeClient,
    provider: Arc<Mutex<Provider>>,
    http: tokio::task::JoinHandle<()>,
}

impl Fixture {
    async fn new() -> Self {
        let url = std::env::var("BILLING_TEST_DATABASE_URL")
            .expect("set isolated BILLING_TEST_DATABASE_URL");
        let (admin, connection) = tokio_postgres::connect(&url, NoTls).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let schema = format!("billing_service_{}", Uuid::new_v4().simple());
        admin
            .batch_execute(&format!("CREATE SCHEMA {schema}"))
            .await
            .unwrap();
        let mut config: tokio_postgres::Config = url.parse().unwrap();
        config.options(format!("-c search_path={schema},public"));
        config.application_name(&schema);
        let pool = Pool::builder(Manager::new(config, NoTls))
            .max_size(8)
            .build()
            .unwrap();
        let c = pool.get().await.unwrap();
        c.batch_execute(include_str!(
            "../../eidola-server-gateway/schema/schema.sql"
        ))
        .await
        .unwrap();
        let account = Uuid::new_v4();
        c.execute(
            "INSERT INTO account(id,secret_hash,stripe_customer_id) VALUES($1,'test','cus_1')",
            &[&account],
        )
        .await
        .unwrap();
        drop(c);
        let provider = Arc::new(Mutex::new(Provider {
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
                - 100,
            purchases: 1,
            ..Provider::default()
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = Router::new().fallback(serve).with_state(provider.clone());
        let http = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let _ = rustls_rustcrypto::provider().install_default();
        let stripe = StripeClient::new("test".into()).with_base_url(format!("http://{addr}"));
        let service = Arc::new(BillingService::new(pool.clone(), stripe.clone()));
        service.reconcile_account(account).await.unwrap();
        Self {
            account,
            schema,
            admin,
            pool,
            service,
            stripe,
            provider,
            http,
        }
    }

    fn request(&self) -> RefundRequest {
        RefundRequest {
            id: Uuid::new_v4(),
            account_id: self.account,
            purchase_id: "cs_1".into(),
            credits: 1_000_000,
            cash_minor: None,
            reason: "support case".into(),
        }
    }

    async fn balance(&self) -> i64 {
        self.pool
            .get()
            .await
            .unwrap()
            .query_one("SELECT available_balance($1)", &[&self.account])
            .await
            .unwrap()
            .get(0)
    }

    async fn close(self) {
        self.http.abort();
        self.pool.close();
        self.admin
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", self.schema))
            .await
            .unwrap();
    }
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn a_lost_refund_response_is_recovered_without_a_second_payment() {
    let f = Fixture::new().await;
    let request = f.request();
    f.provider.lock().unwrap().lose_next_response = true;
    assert!(f.service.submit_refund(&request).await.is_err());
    assert_eq!(f.balance().await, 1_000_000);
    assert_eq!(
        f.service
            .store()
            .get_refund(request.id)
            .await
            .unwrap()
            .state,
        "submitting"
    );
    let operation = f.service.resume_refund(request.id).await.unwrap();
    assert_eq!(operation.state, "succeeded");
    assert_eq!(f.provider.lock().unwrap().posts.len(), 1);
    assert_eq!(f.balance().await, 1_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn a_rejected_retry_cannot_resolve_an_earlier_unknown_refund() {
    let f = Fixture::new().await;
    let request = f.request();
    f.provider.lock().unwrap().lose_next_response = true;
    assert!(f.service.submit_refund(&request).await.is_err());
    // An indeterminate provider request may materialize its object only during
    // incident recovery. Meanwhile a restricted key can still read objects but
    // refuse a later POST; that refusal says nothing about the first request.
    let delayed = {
        let mut provider = f.provider.lock().unwrap();
        provider.reject_refund_posts = true;
        std::mem::take(&mut provider.refunds)
    };
    assert!(f.service.resume_refund(request.id).await.is_err());
    assert_eq!(
        f.service
            .store()
            .get_refund(request.id)
            .await
            .unwrap()
            .state,
        "submitting"
    );
    assert!(f.service.release_refund(request.id).await.is_err());
    assert!(
        f.service
            .retry_refund(request.id, Uuid::new_v4(), "retry")
            .await
            .is_err()
    );
    assert!(
        f.service
            .confirm_external_refund(request.id, "alternate")
            .await
            .is_err()
    );
    assert_eq!(f.balance().await, 1_000_000);
    assert_eq!(f.provider.lock().unwrap().posts.len(), 2);
    f.provider.lock().unwrap().refunds = delayed;
    assert_eq!(
        f.service.resume_refund(request.id).await.unwrap().state,
        "succeeded"
    );
    assert_eq!(f.provider.lock().unwrap().posts.len(), 2);
    assert_eq!(f.balance().await, 1_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn a_rejected_first_submission_can_release_its_refund_hold() {
    let f = Fixture::new().await;
    let request = f.request();
    f.provider.lock().unwrap().reject_refund_posts = true;
    assert!(matches!(
        f.service.submit_refund(&request).await,
        Err(BillingError::ProviderRejected(_))
    ));
    assert_eq!(
        f.service
            .store()
            .get_refund(request.id)
            .await
            .unwrap()
            .state,
        "failed"
    );
    assert_eq!(f.balance().await, 1_000_000);
    assert_eq!(
        f.service.release_refund(request.id).await.unwrap().state,
        "released"
    );
    assert_eq!(f.balance().await, 2_000_000);
    assert!(f.provider.lock().unwrap().refunds.is_empty());
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn account_repair_finishes_an_approved_intent_but_creates_no_new_one() {
    let f = Fixture::new().await;
    let request = f.request();
    f.service.store().prepare_refund(&request).await.unwrap();
    f.service.reconcile_account(f.account).await.unwrap();
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(f.provider.lock().unwrap().posts.len(), 1);
    assert_eq!(
        f.service
            .store()
            .list_refunds(f.account)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(f.balance().await, 1_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn releasing_a_prepared_refund_never_submits_it_first() {
    let f = Fixture::new().await;
    let request = f.request();
    f.service.store().prepare_refund(&request).await.unwrap();
    let released = f.service.release_refund(request.id).await.unwrap();
    assert_eq!(released.state, "released");
    assert_eq!(f.provider.lock().unwrap().posts.len(), 0);
    assert_eq!(f.balance().await, 2_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn an_uncertain_old_attempt_is_not_resubmitted_after_key_retention() {
    let f = Fixture::new().await;
    let request = f.request();
    f.service.store().prepare_refund(&request).await.unwrap();
    f.service.store().mark_submitted(request.id).await.unwrap();
    f.pool.get().await.unwrap().execute(
        "UPDATE billing_refund_operation SET submitted_at=clock_timestamp()-interval '25 hours' WHERE id=$1", &[&request.id]
    ).await.unwrap();
    assert!(matches!(
        f.service.resume_refund(request.id).await,
        Err(BillingError::RequiresReview(_))
    ));
    assert_eq!(f.provider.lock().unwrap().posts.len(), 0);
    assert_eq!(f.balance().await, 1_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn cash_only_refund_does_not_cancel_account_credits() {
    let f = Fixture::new().await;
    let mut request = f.request();
    request.credits = 0;
    request.cash_minor = Some(300);
    f.service.submit_refund(&request).await.unwrap();
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(f.balance().await, 2_000_000);
    assert_eq!(f.provider.lock().unwrap().posts.len(), 1);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn repair_resumes_closure_and_refunds_payments_arriving_after_it() {
    let f = Fixture::new().await;
    f.provider.lock().unwrap().subscription_active = true;
    f.service.store().begin_close(f.account).await.unwrap();
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(f.provider.lock().unwrap().cancellations, 1);
    assert_eq!(
        f.service.store().account_state(f.account).await.unwrap(),
        "closed"
    );
    assert_eq!(f.balance().await, 0);
    f.provider.lock().unwrap().purchases = 2;
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(f.provider.lock().unwrap().posts.len(), 2);
    assert_eq!(f.balance().await, 0);
    assert_eq!(
        f.service.store().account_state(f.account).await.unwrap(),
        "closed"
    );
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn a_canceled_reconciliation_releases_its_database_lock() {
    let f = Fixture::new().await;
    let store = Store::new(f.pool.clone());
    let guard = store.lock_reconciliation(f.account).await.unwrap();
    assert!(store.lock_reconciliation(f.account).await.is_err());
    drop(guard);
    // Closing a detached PostgreSQL connection is asynchronous. Bound the wait
    // while checking the actual database lock rather than an in-memory flag.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(guard) = store.lock_reconciliation(f.account).await {
                guard.release().await.unwrap();
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn durable_jobs_finish_work_started_by_an_operator_who_disappeared() {
    let f = Fixture::new().await;
    f.provider.lock().unwrap().subscription_active = true;
    f.service.store().begin_close(f.account).await.unwrap();
    let report = f.service.drain_events(32).await.unwrap();
    assert_eq!(report.completed, 1);
    assert_eq!(report.failed, 0);
    assert_eq!(
        f.service.store().account_state(f.account).await.unwrap(),
        "closed"
    );
    assert_eq!(f.provider.lock().unwrap().cancellations, 1);
    assert_eq!(f.provider.lock().unwrap().posts.len(), 1);
    assert_eq!(f.service.drain_events(32).await.unwrap().processed, 0);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn a_known_pending_refund_waits_for_observation_without_background_polling() {
    let f = Fixture::new().await;
    f.provider.lock().unwrap().refund_pending = true;
    let request = f.request();
    assert_eq!(
        f.service.submit_refund(&request).await.unwrap().state,
        "pending"
    );
    assert_eq!(f.service.drain_events(32).await.unwrap().processed, 0);
    f.provider.lock().unwrap().refunds[0]["status"] = json!("succeeded");
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(
        f.service
            .store()
            .get_refund(request.id)
            .await
            .unwrap()
            .state,
        "succeeded"
    );
    assert_eq!(f.provider.lock().unwrap().posts.len(), 1);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn recording_an_alternate_payout_never_creates_a_stripe_refund() {
    let f = Fixture::new().await;
    let request = f.request();
    f.service.store().prepare_refund(&request).await.unwrap();
    let operation = f
        .service
        .confirm_external_refund(request.id, "case-123-payment")
        .await
        .unwrap();
    assert_eq!(operation.state, "succeeded");
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(f.provider.lock().unwrap().posts.len(), 0);
    assert_eq!(f.balance().await, 1_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn an_alternate_payout_survives_reconciliation_and_resumes_closure() {
    let f = Fixture::new().await;
    let mut request = f.request();
    request.credits = 2_000_000;
    f.provider.lock().unwrap().refund_pending = true;
    f.service.submit_refund(&request).await.unwrap();
    f.provider.lock().unwrap().refunds[0]["status"] = json!("failed");
    f.service.close_account(f.account, "closure").await.unwrap();
    let close_id = format!("local:close:{}", f.account);
    let old_job = f
        .service
        .store()
        .claim_event(&close_id)
        .await
        .unwrap()
        .unwrap();
    f.service
        .confirm_external_refund(request.id, "bank-transfer")
        .await
        .unwrap();
    // A worker finishing its earlier generation must not acknowledge the new
    // work recorded by the operator's settlement transaction.
    old_job.complete(None).await.unwrap();
    assert!(
        f.service
            .store()
            .pending_events(100)
            .await
            .unwrap()
            .iter()
            .any(|e| e.id == close_id)
    );
    assert_eq!(f.service.drain_events(100).await.unwrap().failed, 0);
    assert_eq!(
        f.service.store().account_state(f.account).await.unwrap(),
        "closed"
    );
    f.service.reconcile_account(f.account).await.unwrap();
    assert_eq!(
        f.service
            .store()
            .get_refund(request.id)
            .await
            .unwrap()
            .state,
        "succeeded"
    );
    assert!(f.service.release_refund(request.id).await.is_err());
    assert!(
        f.service
            .retry_refund(request.id, Uuid::new_v4(), "duplicate")
            .await
            .is_err()
    );
    assert_eq!(f.provider.lock().unwrap().posts.len(), 1);
    assert_eq!(f.balance().await, 0);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn releasing_a_hold_reopens_a_completed_closure_job() {
    let f = Fixture::new().await;
    let mut request = f.request();
    request.credits = 2_000_000;
    f.provider.lock().unwrap().refund_pending = true;
    f.service.submit_refund(&request).await.unwrap();
    f.provider.lock().unwrap().refunds[0]["status"] = json!("failed");
    f.service.close_account(f.account, "closure").await.unwrap();
    f.service.drain_events(100).await.unwrap();
    assert!(
        f.service
            .store()
            .pending_events(100)
            .await
            .unwrap()
            .is_empty()
    );
    f.service.release_refund(request.id).await.unwrap();
    assert_eq!(f.balance().await, 2_000_000);
    f.provider.lock().unwrap().refund_pending = false;
    assert_eq!(f.service.drain_events(100).await.unwrap().failed, 0);
    assert_eq!(
        f.service.store().account_state(f.account).await.unwrap(),
        "closed"
    );
    assert_eq!(
        f.service
            .store()
            .get_refund(request.id)
            .await
            .unwrap()
            .state,
        "released"
    );
    assert_eq!(f.provider.lock().unwrap().posts.len(), 2);
    assert_eq!(f.balance().await, 0);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn losing_the_database_lock_prevents_an_old_provider_response_from_overwriting_a_new_one() {
    let f = Fixture::new().await;
    let request = f.request();
    let gate = Arc::new(tokio::sync::Notify::new());
    let started = {
        let mut provider = f.provider.lock().unwrap();
        provider.response_gate = Some(gate.clone());
        provider.post_started.clone()
    };
    let service = f.service.clone();
    let id = request.id;
    let old = tokio::spawn(async move { service.submit_refund(&request).await });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    // Simulate a database disconnect while the old caller waits on Stripe.
    // The provider's status then changes and another worker observes it.
    let killed = f.admin.query(
        "SELECT pg_terminate_backend(a.pid) FROM pg_stat_activity a JOIN pg_locks l ON l.pid=a.pid \
         WHERE a.application_name=$1 AND l.locktype='advisory' AND l.granted", &[&f.schema]
    ).await.unwrap();
    assert_eq!(killed.len(), 1);
    f.provider.lock().unwrap().refunds[0]["status"] = json!("failed");
    f.service.reconcile_account(f.account).await.unwrap();
    gate.notify_one();
    assert!(old.await.unwrap().is_err());
    assert_eq!(
        f.service.store().get_refund(id).await.unwrap().state,
        "failed"
    );
    assert_eq!(f.balance().await, 1_000_000);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn separate_workers_do_not_duplicate_a_claim_and_a_later_repair_is_not_lost() {
    let f = Fixture::new().await;
    let second = BillingService::new(f.pool.clone(), f.stripe.clone());
    let gate = Arc::new(tokio::sync::Notify::new());
    let started = Arc::new(tokio::sync::Notify::new());
    let before = {
        let mut provider = f.provider.lock().unwrap();
        provider.snapshot_gate = Some(gate.clone());
        provider.snapshot_started = started.clone();
        provider.snapshot_reads
    };
    f.service
        .store()
        .request_reconciliation(f.account)
        .await
        .unwrap();
    let worker = f.service.clone();
    let first = tokio::spawn(async move { worker.drain_events(32).await });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    assert_eq!(second.drain_events(32).await.unwrap().processed, 0);
    assert_eq!(f.provider.lock().unwrap().snapshot_reads, before + 1);
    // A private operator still uses the account lock, independently of queue
    // ownership. It cannot race a refund through this active provider snapshot.
    assert!(matches!(
        second.submit_refund(&f.request()).await,
        Err(BillingError::Conflict(_))
    ));
    assert!(f.provider.lock().unwrap().posts.is_empty());
    let pending = f
        .service
        .store()
        .request_reconciliation(f.account)
        .await
        .unwrap();
    assert_eq!(
        (pending.requested_generation, pending.completed_generation),
        (2, 0)
    );
    f.provider.lock().unwrap().snapshot_gate = None;
    gate.notify_one();
    assert_eq!(first.await.unwrap().unwrap().completed, 1);
    let pending = f
        .service
        .store()
        .reconciliation_status(f.account)
        .await
        .unwrap();
    assert_eq!(
        (pending.requested_generation, pending.completed_generation),
        (2, 1)
    );
    assert_eq!(second.drain_events(32).await.unwrap().completed, 1);
    let complete = f
        .service
        .store()
        .reconciliation_status(f.account)
        .await
        .unwrap();
    assert_eq!(
        (complete.requested_generation, complete.completed_generation),
        (2, 2)
    );
    assert_eq!(complete.report.unwrap().purchases, 1);
    assert_eq!(f.provider.lock().unwrap().snapshot_reads, before + 2);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn unsupported_work_remains_pending_and_visible_without_provider_calls() {
    let f = Fixture::new().await;
    let before = f.provider.lock().unwrap().snapshot_reads;
    f.service
        .store()
        .enqueue_event("future_work", "billing.future.action", "future_object")
        .await
        .unwrap();
    f.service
        .store()
        .request_reconciliation(f.account)
        .await
        .unwrap();
    let id = format!("billing.reconcile:{}", f.account);
    f.pool
        .get()
        .await
        .unwrap()
        .execute(
            "UPDATE billing_event SET protocol_version=2 WHERE id=$1",
            &[&id],
        )
        .await
        .unwrap();
    // An incompatible record is not an idle account and must not be overwritten.
    assert!(
        f.service
            .store()
            .reconciliation_status(f.account)
            .await
            .is_err()
    );
    assert!(
        f.service
            .store()
            .request_reconciliation(f.account)
            .await
            .is_err()
    );
    let result = f.service.drain_events(32).await.unwrap();
    assert_eq!((result.completed, result.failed, result.review), (0, 2, 2));
    let rows = f.pool.get().await.unwrap().query(
        "SELECT last_error,completed_at IS NULL,completed_generation FROM billing_event ORDER BY last_error", &[],
    ).await.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get::<_, String>(0), "unsupported_kind");
    assert_eq!(rows[1].get::<_, String>(0), "unsupported_protocol");
    for row in rows {
        assert!(row.get::<_, bool>(1));
        assert_eq!(row.get::<_, i64>(2), 0);
    }
    assert_eq!(f.provider.lock().unwrap().snapshot_reads, before);
    f.close().await;
}

#[tokio::test]
#[ignore = "requires isolated BILLING_TEST_DATABASE_URL"]
async fn incompatible_executors_refuse_before_provider_calls_while_v1_producers_continue() {
    let f = Fixture::new().await;
    let before = f.provider.lock().unwrap().snapshot_reads;
    f.pool
        .get()
        .await
        .unwrap()
        .execute("UPDATE billing_protocol SET execution_version=2", &[])
        .await
        .unwrap();
    f.service.store().check_ingress_protocol().await.unwrap();
    f.service
        .store()
        .request_reconciliation(f.account)
        .await
        .unwrap();
    f.service
        .store()
        .enqueue_event("evt_future_executor", "refund.updated", "re_1")
        .await
        .unwrap();
    f.service
        .store()
        .lock_checkout(f.account)
        .await
        .unwrap()
        .release()
        .await
        .unwrap();
    assert!(f.service.store().check_protocol().await.is_err());
    assert!(f.service.drain_events(32).await.is_err());
    assert!(f.service.reconcile_account(f.account).await.is_err());
    assert!(f.service.submit_refund(&f.request()).await.is_err());
    assert!(f.service.cancel_subscriptions(f.account).await.is_err());
    assert!(
        f.service
            .close_account(f.account, "operator request")
            .await
            .is_err()
    );
    assert_eq!(
        f.service.store().account_state(f.account).await.unwrap(),
        "open"
    );
    assert_eq!(f.provider.lock().unwrap().snapshot_reads, before);
    assert!(f.provider.lock().unwrap().posts.is_empty());
    assert_eq!(f.balance().await, 2_000_000);
    f.pool
        .get()
        .await
        .unwrap()
        .execute("UPDATE billing_protocol SET ingress_version=2", &[])
        .await
        .unwrap();
    assert!(
        f.service
            .store()
            .request_reconciliation(f.account)
            .await
            .is_err()
    );
    assert!(
        f.service
            .store()
            .enqueue_event("evt_new", "refund.updated", "re_2")
            .await
            .is_err()
    );
    assert!(f.service.store().lock_checkout(f.account).await.is_err());
    f.close().await;
}
