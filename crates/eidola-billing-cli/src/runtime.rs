//! The receiver only authenticates and persists observations. Workers execute
//! the shared billing workflow; neither role requires a gateway instance.

use std::future::{Future, IntoFuture};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use eidola_billing::BillingError;
use eidola_billing::service::{BillingService, DrainReport};
use eidola_billing::store::Store;
use eidola_billing::webhook::{parse_event, verify_signature};
use serde_json::json;
use tokio::net::TcpListener;

use crate::telemetry::metrics;

const SHUTDOWN_GRACE: Duration = Duration::from_secs(30);
const MAX_WEBHOOK_BYTES: usize = 2 * 1024 * 1024;
/// How often the worker reads the backlog it reports as gauges.
const BACKLOG_INTERVAL: Duration = Duration::from_secs(30);
/// Consecutive deliveries failing one way, with none verified between, that
/// read as misconfiguration rather than noise.
const FAILING_STREAK: u32 = 5;

/// The billing database as this process last found it, logged only when that
/// changes: failing calls, refusing this build's protocol, or answering.
pub(crate) struct DatabaseHealth(AtomicU8);

const HEALTHY: u8 = 0;
const FAILING: u8 = 1;
const UNCONFIGURED: u8 = 2;
const INCOMPATIBLE: u8 = 3;

impl DatabaseHealth {
    pub(crate) const fn new() -> Self {
        Self(AtomicU8::new(HEALTHY))
    }

    /// Record what one database call found, logging when the state changes.
    /// Errors that say nothing about the database itself leave it as it was.
    /// Returns whether it logged.
    pub(crate) fn observe(&self, role: &str, result: Result<(), &BillingError>) -> bool {
        let state = match result {
            Ok(()) => HEALTHY,
            Err(error) => match error.protocol_refusal() {
                Some("not_configured") => UNCONFIGURED,
                Some(_) => INCOMPATIBLE,
                None if matches!(error, BillingError::Database(_)) => FAILING,
                None => return false,
            },
        };
        if self.0.swap(state, Ordering::Relaxed) == state {
            return false;
        }
        match (state, result) {
            (HEALTHY, _) => tracing::info!(role, "billing database is answering again"),
            // The error is a fixed description or a SQLSTATE, never row data.
            (FAILING, Err(error)) => tracing::error!(
                role,
                "billing database calls are failing ({error}); billing {role} work waits for it"
            ),
            (UNCONFIGURED, _) => tracing::error!(
                role,
                "billing database names no billing protocol (billing_protocol is empty); \
                 billing {role} work is refused until it does"
            ),
            _ => tracing::error!(
                role,
                "billing database's protocol version is not this build's; billing {role} \
                 work is refused until a compatible build runs"
            ),
        }
        true
    }
}

/// A run of one delivery failure with no verified delivery between, warned
/// once when it reaches [`FAILING_STREAK`] and re-armed by a verified one.
struct FailingStreak {
    failures: AtomicU32,
    warned: AtomicBool,
}

impl FailingStreak {
    const fn new() -> Self {
        Self {
            failures: AtomicU32::new(0),
            warned: AtomicBool::new(false),
        }
    }

    /// Count a failure; returns true when this one completes the streak and
    /// the condition has not been warned since the last verified delivery.
    fn fail(&self) -> bool {
        self.failures.fetch_add(1, Ordering::Relaxed) + 1 >= FAILING_STREAK
            && !self.warned.swap(true, Ordering::Relaxed)
    }

    fn reset(&self) {
        self.failures.store(0, Ordering::Relaxed);
        self.warned.store(false, Ordering::Relaxed);
    }
}

struct ReceiverState {
    store: Store,
    secret: String,
    database: DatabaseHealth,
    bad_signatures: FailingStreak,
    stale_timestamps: FailingStreak,
}

/// The receiver has no provider API client and no account or operator routes.
pub fn webhook_router(store: Store, secret: String) -> Result<Router, BillingError> {
    if secret.is_empty() {
        return Err(BillingError::Invalid(
            "STRIPE_WEBHOOK_SECRET is required".into(),
        ));
    }
    Ok(Router::new()
        .route("/v1/webhooks/stripe", post(stripe_webhook))
        .route("/healthz", get(receiver_health))
        .route_layer(axum::middleware::from_fn(measure))
        .layer(DefaultBodyLimit::max(MAX_WEBHOOK_BYTES))
        .with_state(Arc::new(ReceiverState {
            store,
            secret,
            database: DatabaseHealth::new(),
            bad_signatures: FailingStreak::new(),
            stale_timestamps: FailingStreak::new(),
        })))
}

/// Time each routed request under its matched route, which is fixed in code.
async fn measure(request: Request, next: Next) -> Response {
    let method = match *request.method() {
        Method::GET => "GET",
        Method::POST => "POST",
        _ => "other",
    };
    let route = match request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str)
    {
        Some("/v1/webhooks/stripe") => "/v1/webhooks/stripe",
        Some("/healthz") => "/healthz",
        _ => "other",
    };
    let started = std::time::Instant::now();
    let response = next.run(request).await;
    metrics::http_request(method, route, response.status().as_u16(), started.elapsed());
    response
}

async fn receiver_health(State(state): State<Arc<ReceiverState>>) -> StatusCode {
    let result = state.store.check_ingress_protocol().await;
    state
        .database
        .observe("receiver", result.as_ref().map(|_| ()));
    match result {
        Ok(()) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

async fn stripe_webhook(
    State(state): State<Arc<ReceiverState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) => {
            let outcome = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                "too_large"
            } else {
                "unreadable"
            };
            metrics::webhook_delivery("unknown", outcome);
            return rejection.into_response();
        }
    };
    let Some(signature) = headers
        .get("stripe-signature")
        .and_then(|value| value.to_str().ok())
    else {
        metrics::webhook_delivery("unknown", "missing_signature");
        return receiver_error(StatusCode::BAD_REQUEST, "missing stripe-signature header");
    };
    if let Err(reason) = verify_signature(signature, &body, &state.secret) {
        let outcome = match reason {
            "stale timestamp" => "stale_timestamp",
            "signature mismatch" | "invalid webhook secret" => "bad_signature",
            _ => "malformed_signature",
        };
        metrics::webhook_delivery("unknown", outcome);
        if outcome == "bad_signature" && state.bad_signatures.fail() {
            tracing::warn!(
                "the last {FAILING_STREAK} signed webhook deliveries all failed signature \
                 verification: STRIPE_WEBHOOK_SECRET may not be this endpoint's signing secret"
            );
        }
        if outcome == "stale_timestamp" && state.stale_timestamps.fail() {
            tracing::warn!(
                "the last {FAILING_STREAK} webhook deliveries were all signed more than five \
                 minutes from this host's clock: check its time synchronization"
            );
        }
        return receiver_error(StatusCode::BAD_REQUEST, "invalid webhook signature");
    }
    state.bad_signatures.reset();
    state.stale_timestamps.reset();
    let event = match parse_event(&body) {
        Ok(Some(event)) => event,
        Ok(None) => {
            metrics::webhook_delivery("unsupported", "unsupported_ack");
            return StatusCode::OK.into_response();
        }
        Err(reason) => {
            let outcome = if reason == "only thin events are accepted" {
                "snapshot_refused"
            } else {
                "malformed"
            };
            metrics::webhook_delivery("unknown", outcome);
            return receiver_error(StatusCode::BAD_REQUEST, reason);
        }
    };
    let event_type = eidola_billing::metrics::job_type_label(&event.kind);
    let result = state
        .store
        .enqueue_event(&event.id, &event.kind, &event.object_id)
        .await;
    state
        .database
        .observe("receiver", result.as_ref().map(|_| ()));
    match result {
        Ok(stored) => {
            metrics::webhook_delivery(event_type, if stored { "stored" } else { "duplicate" });
            StatusCode::OK.into_response()
        }
        Err(error) => {
            let outcome = if error.protocol_refusal().is_some() {
                "protocol_refused"
            } else if matches!(error, BillingError::Conflict(_)) {
                "conflict"
            } else {
                "db_failed"
            };
            metrics::webhook_delivery(event_type, outcome);
            receiver_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not persist billing event",
            )
        }
    }
}

fn receiver_error(status: StatusCode, message: &'static str) -> Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

/// Serve until shutdown, then allow active requests up to thirty seconds to
/// finish. Unacknowledged observations remain Stripe's delivery responsibility.
pub async fn serve_webhooks(
    listener: TcpListener,
    store: Store,
    secret: String,
    shutdown: impl Future<Output = ()>,
) -> Result<(), BillingError> {
    tokio::pin!(shutdown);
    tokio::select! {
        biased;
        () = &mut shutdown => return Ok(()),
        result = store.check_ingress_protocol() => result?,
    }
    let router = webhook_router(store, secret)?;
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let _ = stopped.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result.map_err(|_| BillingError::Network("webhook listener failed".into())),
        () = &mut shutdown => {
            let _ = stop.send(());
            match tokio::time::timeout(SHUTDOWN_GRACE, &mut server).await {
                Ok(result) => result.map_err(|_| BillingError::Network("webhook listener failed".into())),
                Err(_) => {
                    tracing::warn!("webhook shutdown grace elapsed; unacknowledged deliveries will retry");
                    Ok(())
                }
            }
        }
    }
}

/// Retry only durable database work. No timer periodically scans Stripe.
pub async fn work(
    service: BillingService,
    batch_size: i64,
    retry_interval: Duration,
    shutdown: impl Future<Output = ()>,
) -> Result<(), BillingError> {
    if !(1..=100).contains(&batch_size)
        || retry_interval.is_zero()
        || retry_interval > Duration::from_secs(3600)
    {
        return Err(BillingError::Invalid(
            "worker requires batch size 1..=100 and retry interval above zero through 3600 seconds"
                .into(),
        ));
    }
    tokio::pin!(shutdown);
    tokio::select! {
        biased;
        () = &mut shutdown => return Ok(()),
        result = service.store().check_protocol() => result?,
    }
    let backlog = tokio::spawn(report_backlog(service.store().clone()));
    let database = DatabaseHealth::new();
    work_loop(
        || {
            let (service, database) = (&service, &database);
            async move {
                let report = service.drain_events(batch_size).await;
                database.observe("worker", report.as_ref().map(|_| ()));
                report
            }
        },
        batch_size,
        retry_interval,
        SHUTDOWN_GRACE,
        shutdown,
    )
    .await;
    backlog.abort();
    Ok(())
}

/// Read the durable backlog on a timer and record it as gauges. Only the
/// database is read; a failed read leaves the last values and is the drain
/// loop's to report.
async fn report_backlog(store: Store) {
    let mut tick = tokio::time::interval(BACKLOG_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        if let Ok(backlog) = store.backlog().await {
            eidola_billing::metrics::backlog(&backlog);
        }
    }
}

async fn work_loop<D, F>(
    mut drain: D,
    batch_size: i64,
    retry_interval: Duration,
    shutdown_grace: Duration,
    shutdown: impl Future<Output = ()>,
) where
    D: FnMut() -> F,
    F: Future<Output = Result<DrainReport, BillingError>>,
{
    tokio::pin!(shutdown);
    loop {
        let started = std::sync::atomic::AtomicBool::new(false);
        let batch = async {
            started.store(true, std::sync::atomic::Ordering::Relaxed);
            drain().await
        };
        tokio::pin!(batch);
        let report = tokio::select! {
            biased;
            () = &mut shutdown => {
                if started.load(std::sync::atomic::Ordering::Relaxed)
                    && tokio::time::timeout(shutdown_grace, &mut batch).await.is_err() {
                    tracing::warn!("billing shutdown grace elapsed; unfinished work remains recoverable");
                }
                return;
            },
            result = &mut batch => result,
        };
        // Jobs, failures and review items are counted where they happen
        // (`eidola_billing::metrics`); the database's state is logged by the
        // drain closure when it changes.
        let another_batch = match report {
            Ok(report) => report.processed == batch_size as usize && report.failed == 0,
            Err(_) => false,
        };
        if another_batch {
            tokio::task::yield_now().await;
        } else {
            tokio::select! {
                biased;
                () = &mut shutdown => return,
                () = tokio::time::sleep(retry_interval) => {},
            }
        }
    }
}

/// Install both handlers before starting a runtime so a signal cannot arrive
/// between accepting work and installing the shutdown path.
pub fn shutdown_signal() -> Result<impl Future<Output = ()>, BillingError> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate()).map_err(|_| {
            BillingError::ServiceUnavailable("could not install SIGTERM handler".into())
        })?;
        let mut interrupt = signal(SignalKind::interrupt()).map_err(|_| {
            BillingError::ServiceUnavailable("could not install SIGINT handler".into())
        })?;
        Ok(async move {
            tokio::select! {
                _ = terminate.recv() => {},
                _ = interrupt.recv() => {},
            }
        })
    }
    #[cfg(not(unix))]
    Ok(async {
        let _ = tokio::signal::ctrl_c().await;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// The database's state is logged when it changes and never while it
    /// holds; an error that says nothing about the database changes nothing.
    #[test]
    fn database_health_logs_only_its_transitions() {
        let health = DatabaseHealth::new();
        let down = BillingError::Database("08006".into());
        let refused = BillingError::ServiceUnavailable("billing protocol is incompatible".into());
        let busy = BillingError::ServiceUnavailable("billing workers are busy".into());
        assert!(!health.observe("worker", Ok(())));
        assert!(health.observe("worker", Err(&down)));
        assert!(!health.observe("worker", Err(&down)));
        assert!(!health.observe("worker", Err(&busy)));
        assert!(health.observe("worker", Err(&refused)));
        assert!(!health.observe("worker", Err(&refused)));
        assert!(health.observe("worker", Ok(())));
        assert!(!health.observe("worker", Ok(())));
    }

    /// A failing streak warns once when it reaches its length, and again only
    /// after a verified delivery has reset it.
    #[test]
    fn a_failing_streak_warns_once_until_reset() {
        let streak = FailingStreak::new();
        for _ in 1..FAILING_STREAK {
            assert!(!streak.fail());
        }
        assert!(streak.fail());
        assert!(!streak.fail());
        streak.reset();
        for _ in 1..FAILING_STREAK {
            assert!(!streak.fail());
        }
        assert!(streak.fail());
    }

    #[tokio::test]
    async fn worker_already_stopping_does_not_start_another_batch() {
        let calls = AtomicUsize::new(0);
        work_loop(
            || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(DrainReport::default())
            },
            32,
            Duration::from_secs(3600),
            Duration::from_secs(30),
            std::future::ready(()),
        )
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn worker_finishes_the_active_batch_on_shutdown_without_starting_another() {
        let calls = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(tokio::sync::Notify::new());
        let finish = Arc::new(tokio::sync::Notify::new());
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn({
            let calls = calls.clone();
            let entered = entered.clone();
            let finish = finish.clone();
            async move {
                work_loop(
                    || async {
                        calls.fetch_add(1, Ordering::SeqCst);
                        entered.notify_one();
                        finish.notified().await;
                        Ok(DrainReport {
                            processed: 32,
                            completed: 32,
                            ..Default::default()
                        })
                    },
                    32,
                    Duration::from_secs(3600),
                    Duration::from_secs(5),
                    async {
                        let _ = stopped.await;
                    },
                )
                .await;
            }
        });
        entered.notified().await;
        stop.send(()).unwrap();
        finish.notify_one();
        tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn worker_shutdown_drops_stuck_work_after_its_grace() {
        struct Running(Arc<AtomicUsize>);
        impl Drop for Running {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(tokio::sync::Notify::new());
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn({
            let dropped = dropped.clone();
            let entered = entered.clone();
            async move {
                work_loop(
                    || async {
                        let _running = Running(dropped.clone());
                        entered.notify_one();
                        std::future::pending::<()>().await;
                        Ok(DrainReport::default())
                    },
                    32,
                    Duration::from_secs(3600),
                    Duration::ZERO,
                    async {
                        let _ = stopped.await;
                    },
                )
                .await;
            }
        });
        entered.notified().await;
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
