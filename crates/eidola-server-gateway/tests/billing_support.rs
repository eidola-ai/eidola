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
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{Value, json};
use sha2::Sha256;
use tokio_postgres::NoTls;
use tower::ServiceExt;
use uuid::Uuid;

const WEBHOOK_SECRET: &str = "whsec_billing_boundary_test";
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
            Catalog::new(None, None).unwrap(),
            None,
            self.pool.clone(),
            stripe,
            Some(WEBHOOK_SECRET.into()),
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

fn signature(body: &[u8]) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut mac = Hmac::<Sha256>::new_from_slice(WEBHOOK_SECRET.as_bytes()).unwrap();
    mac.update(timestamp.to_string().as_bytes());
    mac.update(b".");
    mac.update(body);
    format!(
        "t={timestamp},v1={}",
        hex::encode(mac.finalize().into_bytes())
    )
}

fn webhook_request(body: &[u8], valid: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/webhooks/stripe")
        .header("content-type", "application/json")
        .header(
            "stripe-signature",
            if valid {
                signature(body)
            } else {
                "t=0,v1=00".into()
            },
        )
        .body(Body::from(body.to_vec()))
        .unwrap()
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn webhook_acknowledges_only_a_durable_inbox_entry_and_duplicates_do_not_fork_it() {
    let db = TestDb::new().await;
    let app = router(db.state(None)); // Stripe can be down or unconfigured.
    let body = serde_json::to_vec(&json!({"id":"evt_durable", "type":"refund.created",
        "data":{"object":{"id":"re_durable", "amount":500, "customer_email":"never persist this"}}})).unwrap();
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(webhook_request(&body, true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Read from another connection after each acknowledgment. No worker is
        // running, so a non-durable spawn cannot satisfy this assertion.
        let rows = db
            .pool
            .get()
            .await
            .unwrap()
            .query(
                "SELECT id,kind,object_id,completed_at IS NULL FROM billing_event",
                &[],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<_, String>(0), "evt_durable");
        assert_eq!(rows[0].get::<_, String>(1), "refund.created");
        assert_eq!(rows[0].get::<_, String>(2), "re_durable");
        assert!(rows[0].get::<_, bool>(3));
    }
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute(
            "ALTER TABLE billing_event ADD CONSTRAINT reject_test_write CHECK (false) NOT VALID",
        )
        .await
        .unwrap();
    let response = app.oneshot(webhook_request(&body, true)).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "a failed durable write must ask Stripe to retry"
    );
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn invalid_webhook_signature_cannot_enqueue_an_observation() {
    let db = TestDb::new().await;
    let body = serde_json::to_vec(&json!({"id":"evt_forged", "type":"charge.dispute.created",
        "data":{"object":{"id":"dp_forged"}}}))
    .unwrap();
    let response = router(db.state(None))
        .oneshot(webhook_request(&body, false))
        .await
        .unwrap();
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

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn repair_uses_only_the_authenticated_account_and_survives_terms_and_closure() {
    let db = TestDb::new().await;
    let own = db.account("cus_own", "open").await;
    let other = db.account("cus_other", "open").await;
    db.pool.get().await.unwrap().execute(
        "INSERT INTO required_document(document,version,sha256,url) VALUES ('terms_of_service',2,$1,'https://example.test/terms')",
        &[&"ab".repeat(32)],
    ).await.unwrap();
    let (stripe, provider) = stripe(empty_provider()).await;
    let app = router(db.state(Some(stripe)));
    for state in ["open", "closed"] {
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
        let response = app.clone().oneshot(Request::builder().method("POST")
            .uri(format!("/v1/account/reconcile?account_id={other}&customer=cus_other"))
            .header("authorization", auth(own)).header("content-type", "application/json")
            .body(Body::from(json!({"account_id":other,"customer":"cus_other","credits":999999,"charge_id":"ch_other"}).to_string())).unwrap())
            .await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "repair in {state}: {}",
            String::from_utf8_lossy(&body)
        );
        let report: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(report["purchases"], 0);
        assert_eq!(report["refunds"], 0);
        assert_eq!(report["disputes"], 0);
    }
    let requests = provider.requests.lock().unwrap().clone();
    assert!(!requests.is_empty());
    assert!(
        requests
            .iter()
            .all(|request| request.uri.contains("customer=cus_own")),
        "{requests:?}"
    );
    assert!(requests.iter().all(|request| request.method == Method::GET));
    drop(requests);
    let other_state: String = db
        .pool
        .get()
        .await
        .unwrap()
        .query_one("SELECT billing_state FROM account WHERE id=$1", &[&other])
        .await
        .unwrap()
        .get(0);
    assert_eq!(other_state, "open");
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn repair_without_account_authentication_is_refused_before_provider_access() {
    let db = TestDb::new().await;
    let (stripe, provider) = stripe(empty_provider()).await;
    let response = router(db.state(Some(stripe)))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/account/reconcile")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(provider.requests.lock().unwrap().is_empty());
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn checkout_snapshots_the_selected_grant_and_closure_refuses_new_checkout() {
    let db = TestDb::new().await;
    let own = db.account("cus_own", "open").await;
    let mut responses = empty_provider();
    responses.insert("/prices/price_1".into(), json!({"id":"price_1","active":true,"currency":"usd",
        "unit_amount":1500,"type":"one_time","recurring":null,"lookup_key":null,
        "product":{"id":"prod_1","name":"Credits","description":null,"metadata":{"credits":"20000000"}}}));
    responses.insert(
        "/checkout/sessions".into(),
        json!({"url":"https://checkout.stripe.test/session"}),
    );
    let (stripe, provider) = stripe(responses).await;
    let app = router(db.state(Some(stripe)));
    let checkout = || {
        Request::builder()
            .method("POST")
            .uri("/v1/account/checkout")
            .header("authorization", auth(own))
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"price_id":"price_1","success_url":"https://example.test/paid",
            "cancel_url":"https://example.test/canceled"})
                .to_string(),
            ))
            .unwrap()
    };
    let response = app.clone().oneshot(checkout()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = provider.requests.lock().unwrap().clone();
    let create = requests
        .iter()
        .find(|request| request.method == Method::POST)
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
    let response = app.oneshot(checkout()).await.unwrap();
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
    let (stripe, _) = stripe(responses).await;
    let response = router(db.state(Some(stripe)))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/account/reconcile")
                .header("authorization", auth(own))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 65536).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let report: Value = serde_json::from_slice(&bytes).unwrap();
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
    db.finish().await;
}
