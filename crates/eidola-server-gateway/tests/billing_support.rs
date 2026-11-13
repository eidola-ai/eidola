//! Billing boundary tests use a separate schema in an explicitly selected test
//! database. Run with BILLING_TEST_DATABASE_URL and --ignored.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use argon2::{Argon2, PasswordHasher};
use axum::{
    body::{Body, Bytes, to_bytes},
    extract::State,
    http::{Method, Request, StatusCode, Uri},
    response::IntoResponse,
};
use base64::Engine;
use deadpool_postgres::{Manager, Pool};
use eidola_server_gateway::{
    AppState, backend::Catalog, credentials::KeyCache, stripe::StripeClient,
};
use serde_json::{Value, json};
use tokio_postgres::NoTls;
use tower::ServiceExt;
use uuid::Uuid;

const ACCOUNT_SECRET: &str = "billing-boundary-secret";

struct TestDb {
    pool: Pool,
    admin: tokio_postgres::Client,
    schema: String,
}

impl TestDb {
    async fn new() -> Self {
        let url = std::env::var("BILLING_TEST_DATABASE_URL")
            .expect("set BILLING_TEST_DATABASE_URL to a dedicated test database");
        let (admin, connection) = tokio_postgres::connect(&url, NoTls).await.unwrap();
        tokio::spawn(async move { connection.await.unwrap() });
        // Extensions are shared by schemas. Install once outside the isolated
        // schema so cleaning one test cannot remove another test's extension.
        admin
            .batch_execute("SELECT pg_advisory_lock(812441); CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public; SELECT pg_advisory_unlock(812441)")
            .await.unwrap();
        let schema = format!("billing_boundary_{}", Uuid::new_v4().simple());
        admin
            .batch_execute(&format!("CREATE SCHEMA {schema}"))
            .await
            .unwrap();
        let mut config: tokio_postgres::Config = url.parse().unwrap();
        config.options(format!("-c search_path={schema},public"));
        let pool = Pool::builder(Manager::new(config, NoTls))
            .max_size(8)
            .build()
            .unwrap();
        pool.get()
            .await
            .unwrap()
            .batch_execute(include_str!("../schema/schema.sql"))
            .await
            .unwrap();
        Self {
            pool,
            admin,
            schema,
        }
    }

    fn state(&self, stripe: Option<StripeClient>) -> AppState {
        AppState::new(
            Catalog::new(None).unwrap(),
            None,
            self.pool.clone(),
            stripe,
            [0; 32],
            KeyCache::default(),
        )
    }

    async fn account(&self, customer: &str, state: &str) -> Uuid {
        let id = Uuid::new_v4();
        let hash = Argon2::default()
            .hash_password(ACCOUNT_SECRET.as_bytes())
            .unwrap()
            .to_string();
        self.pool.get().await.unwrap().execute(
            "INSERT INTO account (id, secret_hash, stripe_customer_id, billing_state) VALUES ($1,$2,$3,$4)",
            &[&id,&hash,&customer,&state],
        ).await.unwrap();
        id
    }

    async fn finish(self) {
        self.pool.close();
        self.admin
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", self.schema))
            .await
            .unwrap();
    }
}

fn router(state: AppState) -> axum::Router {
    eidola_server_gateway::build_router()
        .with_state(state)
        .split_for_parts()
        .0
}

fn auth(account: Uuid) -> String {
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("{account}:{ACCOUNT_SECRET}"))
    )
}

#[derive(Debug, Clone)]
struct ProviderRequest {
    method: Method,
    uri: String,
    body: String,
}

#[derive(Clone)]
struct Provider {
    requests: Arc<Mutex<Vec<ProviderRequest>>>,
    responses: Arc<HashMap<String, Value>>,
}

async fn provider_request(
    State(provider): State<Provider>,
    method: Method,
    uri: Uri,
    body: Bytes,
) -> impl IntoResponse {
    let response = provider.responses.get(uri.path()).cloned();
    provider.requests.lock().unwrap().push(ProviderRequest {
        method,
        uri: uri.to_string(),
        body: String::from_utf8(body.to_vec()).unwrap(),
    });
    match response {
        Some(value) => (StatusCode::OK, axum::Json(value)),
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({"error":{"type":"invalid_request_error"}})),
        ),
    }
}

async fn stripe(responses: HashMap<String, Value>) -> (StripeClient, Provider) {
    let provider = Provider {
        requests: Arc::default(),
        responses: Arc::new(responses),
    };
    let app = axum::Router::new()
        .fallback(provider_request)
        .with_state(provider.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    (
        StripeClient::new("sk_test".into()).with_base_url(format!("http://{address}")),
        provider,
    )
}

fn empty_provider() -> HashMap<String, Value> {
    [
        "/charges",
        "/checkout/sessions",
        "/invoices",
        "/subscriptions",
    ]
    .into_iter()
    .map(|path| (path.into(), json!({"data":[],"has_more":false})))
    .collect()
}

fn repair_request(method: &str, account: Uuid) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri("/v1/account/reconcile")
        .header("authorization", auth(account))
        .body(Body::empty())
        .unwrap()
}

async fn response_json(response: axum::response::Response, status: StatusCode) -> Value {
    let actual = response.status();
    let bytes = to_bytes(response.into_body(), 65536).await.unwrap();
    assert_eq!(actual, status, "{}", String::from_utf8_lossy(&bytes));
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn repair_is_durable_without_stripe_and_survives_terms_closure_and_gateway_restart() {
    let db = TestDb::new().await;
    // A newer executor can retain the same stable ingress protocol. This old
    // gateway must still enqueue/read repairs without executing billing code.
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute("UPDATE billing_protocol SET execution_version=2")
        .await
        .unwrap();
    let own = db.account("cus_own", "open").await;
    let other = db.account("cus_other", "open").await;
    db.pool.get().await.unwrap().execute(
        "INSERT INTO required_document(document,version,sha256,url) VALUES ('terms_of_service',2,$1,'https://example.test/terms')",
        &[&"ab".repeat(32)],
    ).await.unwrap();
    for (index, state) in ["open", "closed"].into_iter().enumerate() {
        db.pool
            .get()
            .await
            .unwrap()
            .execute(
                "UPDATE account SET billing_state=$2 WHERE id=$1",
                &[&own, &state],
            )
            .await
            .unwrap();
        // This gateway has no Stripe client. Neither enqueue nor status reads
        // may perform provider work or require accepted terms/an open account.
        let mut request = repair_request("POST", own);
        *request.uri_mut() = format!("/v1/account/reconcile?account_id={other}&customer=cus_other")
            .parse()
            .unwrap();
        let response = router(db.state(None)).oneshot(request).await.unwrap();
        let status = response_json(response, StatusCode::ACCEPTED).await;
        assert_eq!(status["state"], "pending");
        assert_eq!(status["requested_generation"], index + 1);
        assert_eq!(status["completed_generation"], 0);
        assert!(status["report"].is_null());
        // A fresh application state proves the request is durable, not a
        // process-local notification. Foreign query parameters cannot scope it.
        let response = router(db.state(None))
            .oneshot(repair_request("GET", own))
            .await
            .unwrap();
        assert_eq!(response_json(response, StatusCode::OK).await, status);
    }
    let response = router(db.state(None))
        .oneshot(repair_request("GET", other))
        .await
        .unwrap();
    let status = response_json(response, StatusCode::OK).await;
    assert_eq!(status["state"], "idle");
    assert_eq!(status["requested_generation"], 0);
    assert!(status["report"].is_null());
    let rows = db
        .pool
        .get()
        .await
        .unwrap()
        .query("SELECT kind, object_id FROM billing_event", &[])
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, String>(0), "billing.account.reconcile");
    assert_eq!(rows[0].get::<_, String>(1), own.to_string());
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn repair_rejects_caller_chosen_effects_without_enqueuing_work() {
    let db = TestDb::new().await;
    let own = db.account("cus_own", "open").await;
    let other = db.account("cus_other", "open").await;
    let response = router(db.state(None)).oneshot(Request::builder().method("POST")
        .uri("/v1/account/reconcile")
        .header("authorization", auth(own)).header("content-type", "application/json")
        .body(Body::from(json!({"account_id":other,"customer":"cus_other","credits":999999,"charge_id":"ch_other"}).to_string())).unwrap())
        .await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let count: i64 = db
        .pool
        .get()
        .await
        .unwrap()
        .query_one("SELECT count(*) FROM billing_event", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(count, 0);
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn repair_and_status_require_account_authentication() {
    let db = TestDb::new().await;
    for method in ["POST", "GET"] {
        let response = router(db.state(None))
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri("/v1/account/reconcile")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    let count: i64 = db
        .pool
        .get()
        .await
        .unwrap()
        .query_one("SELECT count(*) FROM billing_event", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(count, 0);
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn failed_durable_repair_write_is_not_accepted() {
    let db = TestDb::new().await;
    let own = db.account("cus_own", "open").await;
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute(
            "ALTER TABLE billing_event ADD CONSTRAINT reject_test_write CHECK (false) NOT VALID",
        )
        .await
        .unwrap();
    let response = router(db.state(None))
        .oneshot(repair_request("POST", own))
        .await
        .unwrap();
    assert!(response.status().is_server_error());
    db.finish().await;
}

fn checkout_request(account: Uuid) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/account/checkout")
        .header("authorization", auth(account))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"price_id":"price_1","success_url":"https://example.test/paid",
                "cancel_url":"https://example.test/canceled"})
            .to_string(),
        ))
        .unwrap()
}

fn checkout_provider() -> HashMap<String, Value> {
    let mut responses = empty_provider();
    responses.insert("/prices/price_1".into(), json!({"id":"price_1","active":true,"currency":"usd",
        "unit_amount":1500,"type":"one_time","recurring":null,"lookup_key":null,
        "product":{"id":"prod_1","name":"Credits","description":null,"metadata":{"credits":"20000000"}}}));
    responses.insert(
        "/checkout/sessions".into(),
        json!({"url":"https://checkout.stripe.test/session"}),
    );
    responses
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn checkout_reports_active_account_work_as_retryable_without_calling_stripe() {
    let db = TestDb::new().await;
    let own = db.account("cus_own", "open").await;
    let (stripe, provider) = stripe(checkout_provider()).await;
    let app = router(db.state(Some(stripe)));
    let guard = eidola_billing::store::Store::new(db.pool.clone())
        .lock_reconciliation(own)
        .await
        .unwrap();

    let response = app.clone().oneshot(checkout_request(own)).await.unwrap();
    let error = response_json(response, StatusCode::SERVICE_UNAVAILABLE).await;
    assert_eq!(error["error"]["type"], "service_unavailable");
    assert!(provider.requests.lock().unwrap().is_empty());

    guard.release().await.unwrap();
    let response = app.oneshot(checkout_request(own)).await.unwrap();
    let checkout = response_json(response, StatusCode::OK).await;
    assert_eq!(
        checkout["checkout_url"],
        "https://checkout.stripe.test/session"
    );
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn checkout_snapshots_the_selected_grant_and_closure_refuses_new_checkout() {
    let db = TestDb::new().await;
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute("UPDATE billing_protocol SET execution_version=2")
        .await
        .unwrap();
    let own = db.account("cus_own", "open").await;
    let (stripe, provider) = stripe(checkout_provider()).await;
    let app = router(db.state(Some(stripe)));
    let response = app.clone().oneshot(checkout_request(own)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = provider.requests.lock().unwrap().clone();
    let create = requests
        .iter()
        .find(|request| request.method == Method::POST && request.uri == "/checkout/sessions")
        .unwrap();
    assert!(
        create
            .body
            .contains("metadata%5Beidola_credits%5D=20000000")
    );
    assert!(
        create
            .body
            .contains("payment_intent_data%5Bmetadata%5D%5Beidola_credits%5D=20000000")
    );
    assert!(
        create
            .body
            .contains("metadata%5Beidola_expiry%5D=purchase_year")
    );
    let calls = requests.len();
    drop(requests);
    db.pool
        .get()
        .await
        .unwrap()
        .execute(
            "UPDATE account SET billing_state='closed' WHERE id=$1",
            &[&own],
        )
        .await
        .unwrap();
    let response = app.oneshot(checkout_request(own)).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(provider.requests.lock().unwrap().len(), calls);
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn late_invoice_payment_records_cash_and_original_expiry_without_spendable_credit() {
    let db = TestDb::new().await;
    let own = db.account("cus_own", "open").await;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let mut responses = empty_provider();
    responses.insert(
        "/charges".into(),
        json!({"has_more":false,"data":[{
        "id":"ch_late","customer":"cus_own","payment_intent":"pi_late",
        "amount":1500,"amount_refunded":0,"currency":"usd","paid":true,
        "captured":true,"status":"succeeded","created":now-60}]}),
    );
    responses.insert("/refunds".into(), json!({"has_more":false,"data":[]}));
    responses.insert("/disputes".into(), json!({"has_more":false,"data":[]}));
    responses.insert(
        "/payment_intents/pi_late".into(),
        json!({"id":"pi_late","customer":"cus_own",
        "latest_charge":"ch_late","status":"succeeded","currency":"usd","amount_received":1500}),
    );
    responses.insert(
        "/invoice_payments".into(),
        json!({"has_more":false,"data":[{
        "id":"inpay_late","invoice":"in_late","status":"paid","amount_paid":1500,"currency":"usd",
        "payment":{"type":"payment_intent","payment_intent":"pi_late"}}]}),
    );
    responses.insert("/invoices".into(), json!({"has_more":false,"data":[{
        "id":"in_late","customer":"cus_own","status":"paid","amount_paid":1500,
        "total":1500,"starting_balance":0,"currency":"usd","billing_reason":"subscription_cycle",
        "parent":{"type":"subscription_details","subscription_details":{"subscription":"sub_late",
            "metadata":{"eidola_schema":"1","eidola_credits":"20000000","eidola_expiry":"billing_period","eidola_price":"price_late"}}},
        "lines":{"has_more":false,"data":[{"id":"il_late","quantity":1,
            "period":{"start":now-2600000,"end":now-3600},
            "pricing":{"type":"price_details","price_details":{"price":"price_late"}},
            "parent":{"type":"subscription_item_details","subscription_item_details":{"proration":false}}}]},
        "status_transitions":{"paid_at":now-60}}]}));
    let (stripe, provider) = stripe(responses).await;
    let response = router(db.state(None))
        .oneshot(repair_request("POST", own))
        .await
        .unwrap();
    let pending = response_json(response, StatusCode::ACCEPTED).await;
    assert_eq!(pending["state"], "pending");
    assert!(provider.requests.lock().unwrap().is_empty());
    // The independently deployed worker executes the financial workflow.
    let result = eidola_billing::service::BillingService::new(db.pool.clone(), stripe)
        .drain_events(32)
        .await
        .unwrap();
    assert_eq!(result.completed, 1);
    let response = router(db.state(None))
        .oneshot(repair_request("GET", own))
        .await
        .unwrap();
    let status = response_json(response, StatusCode::OK).await;
    assert_eq!(status["state"], "complete");
    assert_eq!(status["requested_generation"], 1);
    assert_eq!(status["completed_generation"], 1);
    let report = &status["report"];
    assert_eq!(report["purchases"], 1);
    assert!(
        report["review"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["reason"] == "paid_after_service_period")
    );
    let row = db.pool.get().await.unwrap().query_one(
        "SELECT paid_minor, expires_at <= purchased_at, available_balance(account_id) FROM billing_purchase WHERE id='in_late'", &[]).await.unwrap();
    assert_eq!(row.get::<_, i64>(0), 1500);
    assert!(row.get::<_, bool>(1));
    assert_eq!(row.get::<_, i64>(2), 0);
    let response = router(db.state(None))
        .oneshot(repair_request("POST", own))
        .await
        .unwrap();
    let again = response_json(response, StatusCode::ACCEPTED).await;
    assert_eq!(again["state"], "pending");
    assert_eq!(again["requested_generation"], 2);
    assert_eq!(again["completed_generation"], 1);
    assert_eq!(again["report"], status["report"]);
    db.finish().await;
}
