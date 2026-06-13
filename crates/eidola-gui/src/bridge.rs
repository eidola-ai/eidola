//! tokio → gpui bridge helpers.
//!
//! `AppCore` runs on its own tokio multi-thread runtime; gpui's executor is
//! smol-based. The sanctioned bridge is a `tokio::sync::oneshot` (or
//! `mpsc`) channel: the call is spawned on `AppCore::runtime()`, and the
//! receiver — runtime-agnostic — is awaited from gpui's executor inside an
//! entity's own `Task` slot. The stores use `cx.spawn` directly with the
//! [`bridge`] adapter below.
//!
//! This module keeps the small set of *non-store* bridges that the doctrine
//! schedules to become entities in step 3 (the per-`Space` entity owns chat
//! streaming + transcript loads; window-scoped reader entities own the Record
//! fetches). For now they live here as plain free functions taking an
//! `Arc<AppCore>`, with no `Core` god-object wrapper — the views that use them
//! pull the `Arc<AppCore>` out of their store bundle and own the resulting
//! gpui `Task` themselves.

use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AppCore, AttestationDetail, AttestationInfo, ChatResult, ChatStreamEvent, RequestDetail,
    RequestInfo, SpaceMessage, SpendTrailEntry,
};
use tokio::sync::{mpsc, oneshot};

/// Run a future on `AppCore`'s tokio runtime and await its result from gpui.
///
/// The future is produced by `make_fut` (so it can capture an
/// `Arc<AppCore>` and `.await` core methods) and spawned on the runtime; the
/// returned future resolves on the caller's (gpui) executor when the oneshot
/// fires. Cancelling the gpui `Task` that holds this future drops the
/// receiver — the core-side work runs to completion regardless (see the
/// atomicity rules in `docs/architecture/state.md`).
pub async fn bridge<MakeFut, Fut, T>(core: Arc<AppCore>, make_fut: MakeFut) -> Result<T, AppError>
where
    MakeFut: FnOnce(Arc<AppCore>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T, AppError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    core.runtime().handle().clone().spawn(async move {
        let _ = tx.send(make_fut(core).await);
    });
    rx.await.unwrap_or_else(|_| {
        Err(AppError::Internal {
            message: "background task cancelled".into(),
        })
    })
}

// ---------------------------------------------------------------------------
// Chat streaming (becomes the `Space` entity in step 3).
// ---------------------------------------------------------------------------

/// Streaming chat. Spawns the streaming call on the core's tokio runtime and
/// returns an `mpsc` receiver of incremental deltas (closes when the stream
/// ends) plus a `oneshot` receiver for the terminal `ChatResult`. Both are
/// drained from gpui's main thread by `chat::ChatView::spawn_stream`.
pub fn chat_stream(
    core: Arc<AppCore>,
    prompt: String,
    model: String,
    space_id: Option<String>,
) -> (
    mpsc::UnboundedReceiver<ChatStreamEvent>,
    oneshot::Receiver<Result<ChatResult, AppError>>,
) {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (done_tx, done_rx) = oneshot::channel();
    core.runtime().handle().clone().spawn(async move {
        let res = core.chat_stream(prompt, model, space_id, event_tx).await;
        let _ = done_tx.send(res);
    });
    (event_rx, done_rx)
}

/// Load a space's persisted messages (the reopened-space initial load).
pub fn get_space_messages(
    core: Arc<AppCore>,
    space_id: String,
) -> oneshot::Receiver<Result<Vec<SpaceMessage>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.get_space_messages(space_id).await
    })
}

// ---------------------------------------------------------------------------
// The Record — windowed read-only queries (becomes a window-scoped reader
// entity in step 3).
// ---------------------------------------------------------------------------

pub fn list_attestations(
    core: Arc<AppCore>,
    limit: i64,
    offset: i64,
) -> oneshot::Receiver<Result<Vec<AttestationInfo>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.list_attestations(limit, offset).await
    })
}

pub fn attestation_detail(
    core: Arc<AppCore>,
    hash: String,
) -> oneshot::Receiver<Result<Option<AttestationDetail>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.attestation_detail(hash).await
    })
}

pub fn list_requests(
    core: Arc<AppCore>,
    limit: i64,
    offset: i64,
) -> oneshot::Receiver<Result<Vec<RequestInfo>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.list_requests(limit, offset).await
    })
}

pub fn request_detail(
    core: Arc<AppCore>,
    id: String,
) -> oneshot::Receiver<Result<Option<RequestDetail>, AppError>> {
    spawn_oneshot(
        core,
        move |core| async move { core.request_detail(id).await },
    )
}

pub fn spend_trail(
    core: Arc<AppCore>,
    limit: i64,
    offset: i64,
) -> oneshot::Receiver<Result<Vec<SpendTrailEntry>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.spend_trail(limit, offset).await
    })
}

/// Spawn `make_fut` on the core runtime and hand back a oneshot receiver for
/// its result. The shared spine of the Record + space-message bridges, which
/// own their gpui `Task` directly (they await this receiver inside it).
fn spawn_oneshot<MakeFut, Fut, T>(
    core: Arc<AppCore>,
    make_fut: MakeFut,
) -> oneshot::Receiver<Result<T, AppError>>
where
    MakeFut: FnOnce(Arc<AppCore>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T, AppError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    core.runtime().handle().clone().spawn(async move {
        let _ = tx.send(make_fut(core).await);
    });
    rx
}
