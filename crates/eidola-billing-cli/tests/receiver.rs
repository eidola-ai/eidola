//! The separately deployed HTTP boundary against disposable PostgreSQL.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use deadpool_postgres::{Manager, Pool};
use eidola_billing::service::BillingService;
use eidola_billing::store::Store;
use eidola_billing::stripe::StripeClient;
use eidola_billing_cli::runtime::{serve_webhooks, work};
use hmac::{Hmac, KeyInit, Mac};
use reqwest::{Client, StatusCode};
use serde_json::json;
use sha2::Sha256;
use tokio_postgres::NoTls;
use uuid::Uuid;

const SECRET: &str = "whsec_receiver_test";

struct TestDb {
    pool: Pool,
    admin: tokio_postgres::Client,
    schema: String,
}

impl TestDb {
    async fn new() -> Self {
        let _ = rustls_rustcrypto::provider().install_default();
        let url = std::env::var("BILLING_TEST_DATABASE_URL")
            .expect("set BILLING_TEST_DATABASE_URL to a dedicated test database");
        let (admin, connection) = tokio_postgres::connect(&url, NoTls).await.unwrap();
        tokio::spawn(async move { connection.await.unwrap() });
        admin.batch_execute("SELECT pg_advisory_lock(812441); CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public; SELECT pg_advisory_unlock(812441)")
            .await.unwrap();
        let schema = format!("billing_receiver_{}", Uuid::new_v4().simple());
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
            .batch_execute(include_str!(
                "../../eidola-server-gateway/schema/schema.sql"
            ))
            .await
            .unwrap();
        Self {
            pool,
            admin,
            schema,
        }
    }

    async fn finish(self) {
        self.pool.close();
        self.admin
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", self.schema))
            .await
            .unwrap();
    }
}

struct Receiver {
    url: String,
    client: Client,
    stop: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), eidola_billing::BillingError>>,
}

impl Receiver {
    async fn start(pool: Pool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(serve_webhooks(
            listener,
            Store::new(pool),
            SECRET.into(),
            async {
                let _ = stopped.await;
            },
        ));
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        assert_eq!(
            client
                .get(format!("{url}/healthz"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        Self {
            url,
            client,
            stop,
            task,
        }
    }

    async fn post(&self, body: Vec<u8>, signature: String) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/webhooks/stripe", self.url))
            .header("content-type", "application/json")
            .header("stripe-signature", signature)
            .body(body)
            .send()
            .await
            .unwrap()
    }

    async fn finish(self) {
        self.stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

fn signature(body: &[u8], secret: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    signature_at(body, secret, timestamp)
}

fn signature_at(body: &[u8], secret: &str, timestamp: u64) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(timestamp.to_string().as_bytes());
    mac.update(b".");
    mac.update(body);
    format!(
        "t={timestamp},v1={}",
        hex::encode(mac.finalize().into_bytes())
    )
}

fn event(id: &str, kind: &str, object_id: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"id":id,"type":kind,"data":{"object":{
        "id":object_id,"amount":500,"customer_email":"never persist this"
    }}}))
    .unwrap()
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn receiver_acknowledges_committed_identities_and_rejects_conflicting_duplicates() {
    let db = TestDb::new().await;
    // No provider client or Stripe API credentials enter receiver construction.
    let receiver = Receiver::start(db.pool.clone()).await;
    let body = event("evt_durable", "refund.created", "re_durable");
    for _ in 0..2 {
        assert_eq!(
            receiver
                .post(body.clone(), signature(&body, SECRET))
                .await
                .status(),
            StatusCode::OK
        );
        let client = db.pool.get().await.unwrap();
        let rows = client
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
        let stored: serde_json::Value = client
            .query_one("SELECT to_jsonb(e) FROM billing_event e", &[])
            .await
            .unwrap()
            .get(0);
        assert!(!stored.to_string().contains("never persist this"));
        assert!(!stored.to_string().contains("customer_email"));
    }
    let conflicting = event("evt_durable", "refund.created", "re_changed");
    assert_eq!(
        receiver
            .post(conflicting.clone(), signature(&conflicting, SECRET))
            .await
            .status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    receiver.finish().await;
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn receiver_never_acknowledges_an_observation_it_failed_to_persist() {
    let db = TestDb::new().await;
    let receiver = Receiver::start(db.pool.clone()).await;
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute(
            "ALTER TABLE billing_event ADD CONSTRAINT reject_test_write CHECK (false) NOT VALID",
        )
        .await
        .unwrap();
    let body = event("evt_failed", "charge.dispute.created", "dp_failed");
    let response = receiver.post(body.clone(), signature(&body, SECRET)).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        !response
            .text()
            .await
            .unwrap()
            .contains("never persist this")
    );
    receiver.finish().await;
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn receiver_checks_signatures_identifiers_and_body_size_before_inserting() {
    let db = TestDb::new().await;
    let receiver = Receiver::start(db.pool.clone()).await;
    let valid = event("evt_forged", "charge.dispute.created", "dp_forged");
    assert_eq!(
        receiver
            .post(valid.clone(), signature(&valid, "whsec_wrong"))
            .await
            .status(),
        StatusCode::BAD_REQUEST
    );
    let malformed = event("evt_bad", "refund.created", "../refunds/re_other");
    assert_eq!(
        receiver
            .post(malformed.clone(), signature(&malformed, SECRET))
            .await
            .status(),
        StatusCode::BAD_REQUEST
    );
    let oversized = vec![b' '; 2 * 1024 * 1024 + 1];
    assert_eq!(
        receiver
            .post(oversized.clone(), signature(&oversized, SECRET))
            .await
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let ignored = event("evt_ignored", "customer.updated", "cus_ignored");
    assert_eq!(
        receiver
            .post(ignored.clone(), signature(&ignored, SECRET))
            .await
            .status(),
        StatusCode::OK
    );
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
    receiver.finish().await;
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn receiver_accepts_rotating_signatures_and_reports_incompatible_ingress() {
    let db = TestDb::new().await;
    let receiver = Receiver::start(db.pool.clone()).await;
    // Receiver admission remains available while execution protocol changes.
    // The worker, not ingress, must refuse an incompatible financial algorithm.
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute("UPDATE billing_protocol SET execution_version=999")
        .await
        .unwrap();
    let body = event("evt_rotation", "refund.created", "re_rotation");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let matching = signature_at(&body, SECRET, timestamp);
    let other = signature_at(&body, "whsec_rotated", timestamp);
    for (first, second) in [(&matching, &other), (&other, &matching)] {
        let header = format!("{first}, {}", second.split_once(',').unwrap().1);
        assert_eq!(
            receiver.post(body.clone(), header).await.status(),
            StatusCode::OK
        );
    }
    db.pool
        .get()
        .await
        .unwrap()
        .batch_execute("UPDATE billing_protocol SET ingress_version=999")
        .await
        .unwrap();
    assert_eq!(
        receiver
            .client
            .get(format!("{}/healthz", receiver.url))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        receiver
            .post(body.clone(), signature(&body, SECRET))
            .await
            .status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    receiver.finish().await;
    db.finish().await;
}

async fn shutdown_interrupts_startup_protocol_wait(worker: bool) {
    let db = TestDb::new().await;
    let mut locker = db.pool.get().await.unwrap();
    let transaction = locker.transaction().await.unwrap();
    transaction
        .batch_execute("LOCK TABLE billing_protocol IN ACCESS EXCLUSIVE MODE")
        .await
        .unwrap();

    let (stop, stopped) = tokio::sync::oneshot::channel();
    let shutdown = async {
        let _ = stopped.await;
    };
    let task = if worker {
        let stripe =
            StripeClient::new("sk_test_unused".into()).with_base_url("http://127.0.0.1:9".into());
        tokio::spawn(work(
            BillingService::new(db.pool.clone(), stripe),
            32,
            Duration::from_secs(5),
            shutdown,
        ))
    } else {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        tokio::spawn(serve_webhooks(
            listener,
            Store::new(db.pool.clone()),
            SECRET.into(),
            shutdown,
        ))
    };

    // Prove startup reached the blocked database read before signaling, rather
    // than merely exercising an already-ready shutdown future.
    tokio::time::timeout(Duration::from_secs(5), async {
        let relation = format!("{}.billing_protocol", db.schema);
        loop {
            let blocked: bool = db
                .admin
                .query_one(
                    "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE relation=$1::text::regclass \
                     AND mode='AccessShareLock' AND NOT granted)",
                    &[&relation],
                )
                .await
                .unwrap()
                .get(0);
            if blocked {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("runtime never reached the blocked protocol check");
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .expect("shutdown waited for the blocked startup query")
        .unwrap()
        .unwrap();

    // The lock was held throughout the completion assertion above.
    transaction.rollback().await.unwrap();
    drop(locker);
    db.finish().await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn receiver_shutdown_interrupts_a_blocked_startup_protocol_check() {
    shutdown_interrupts_startup_protocol_wait(false).await;
}

#[tokio::test]
#[ignore = "requires BILLING_TEST_DATABASE_URL"]
async fn worker_shutdown_interrupts_a_blocked_startup_protocol_check() {
    shutdown_interrupts_startup_protocol_wait(true).await;
}
