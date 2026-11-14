//! The receiver only authenticates and persists observations. Workers execute
//! the shared billing workflow; neither role requires a gateway instance.

use std::future::{Future, IntoFuture};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use eidola_billing::BillingError;
use eidola_billing::service::{BillingService, DrainReport};
use eidola_billing::store::Store;
use eidola_billing::webhook::{parse_event, verify_signature};
use serde_json::json;
use tokio::net::TcpListener;

const SHUTDOWN_GRACE: Duration = Duration::from_secs(30);
const MAX_WEBHOOK_BYTES: usize = 2 * 1024 * 1024;

struct ReceiverState {
    store: Store,
    secret: String,
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
        .layer(DefaultBodyLimit::max(MAX_WEBHOOK_BYTES))
        .with_state(Arc::new(ReceiverState { store, secret })))
}

async fn receiver_health(State(state): State<Arc<ReceiverState>>) -> StatusCode {
    match state.store.check_ingress_protocol().await {
        Ok(()) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

async fn stripe_webhook(
    State(state): State<Arc<ReceiverState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(signature) = headers
        .get("stripe-signature")
        .and_then(|value| value.to_str().ok())
    else {
        return receiver_error(StatusCode::BAD_REQUEST, "missing stripe-signature header");
    };
    if verify_signature(signature, &body, &state.secret).is_err() {
        return receiver_error(StatusCode::BAD_REQUEST, "invalid webhook signature");
    }
    let event = match parse_event(&body) {
        Ok(Some(event)) => event,
        Ok(None) => return StatusCode::OK.into_response(),
        Err(reason) => return receiver_error(StatusCode::BAD_REQUEST, reason),
    };
    match state
        .store
        .enqueue_event(&event.id, &event.kind, &event.object_id)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(_) => {
            tracing::warn!("webhook was not persisted; delivery must retry");
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
    work_loop(
        || service.drain_events(batch_size),
        batch_size,
        retry_interval,
        SHUTDOWN_GRACE,
        shutdown,
    )
    .await;
    Ok(())
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
        let another_batch = match report {
            Ok(report) => {
                if report.failed > 0 {
                    tracing::warn!(failed = report.failed, "billing work remains pending");
                }
                if report.review > 0 {
                    tracing::warn!(
                        review = report.review,
                        "billing work requires support review"
                    );
                }
                report.processed == batch_size as usize && report.failed == 0
            }
            Err(_) => {
                tracing::warn!("billing inbox unavailable or incompatible; work remains pending");
                false
            }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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
