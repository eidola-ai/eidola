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
    AppCore, AttestationDetail, AttestationInfo, ChatResult, ChatStreamEvent, IncomingReference,
    NotificationPlan, PostNode, PostResult, ReferenceSpec, RequestDetail, RequestInfo,
    SpendTrailEntry, SubmitResult,
};
use tokio::sync::{mpsc, oneshot};

/// Run a future on `AppCore`'s tokio runtime and await its result from gpui.
///
/// The future is produced by `make_fut` (so it can capture an
/// `Arc<AppCore>` and `.await` core methods) and spawned on the runtime; the
/// returned future resolves on the caller's (gpui) executor when the oneshot
/// fires. Cancelling the gpui `Task` that holds this future drops the
/// receiver — the core-side work runs to completion regardless (see the
/// atomicity rules in `crates/eidola-gui/STATE.md`).
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
/// drained from gpui's main thread by the `Space` entity's submit runner. `reply_to`,
/// when set, branches the new turn off that post (vs the linear tail).
pub fn chat_stream(
    core: Arc<AppCore>,
    prompt: String,
    model: String,
    space_id: Option<String>,
    reply_to: Option<String>,
) -> (
    mpsc::UnboundedReceiver<ChatStreamEvent>,
    oneshot::Receiver<Result<ChatResult, AppError>>,
) {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (done_tx, done_rx) = oneshot::channel();
    core.runtime().handle().clone().spawn(async move {
        let res = core
            .chat_stream_reply(prompt, model, space_id, reply_to, event_tx)
            .await;
        let _ = done_tx.send(res);
    });
    (event_rx, done_rx)
}

/// Streaming turn **as a participant**: request a response to an
/// already-persisted post from a specific space participant (its effective
/// model + system prompt), without posting a new user turn. This is both the
/// fan-out driver (one call per [`eidola_app_core::PlannedTurn`] a submit
/// returned) and the explicit-ask / retry entry point — explicit asks bypass
/// the cascade guard by construction. Same channels as [`chat_stream`].
pub fn respond_stream_as(
    core: Arc<AppCore>,
    space_id: String,
    participant_id: String,
    target_action_id: String,
) -> (
    mpsc::UnboundedReceiver<ChatStreamEvent>,
    oneshot::Receiver<Result<ChatResult, AppError>>,
) {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (done_tx, done_rx) = oneshot::channel();
    core.runtime().handle().clone().spawn(async move {
        let res = core
            .respond_stream_as(space_id, participant_id, target_action_id, event_tx)
            .await;
        let _ = done_tx.send(res);
    });
    (event_rx, done_rx)
}

/// Save a post **and** plan notifications over the space's participants — the
/// composer's Post CTA. Returns the saved post plus the
/// [`NotificationPlan`]; the caller drives one [`respond_stream_as`] per
/// planned turn (and re-plans on each resulting post to continue a cascade
/// until the guard pauses it).
pub fn submit(
    core: Arc<AppCore>,
    text: String,
    space_id: Option<String>,
    reply_to: Option<String>,
    references: Vec<ReferenceSpec>,
) -> oneshot::Receiver<Result<SubmitResult, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.submit_with_references(text, space_id, reply_to, references)
            .await
    })
}

/// Compute the auto-response plan for an already-persisted post (the cascade
/// continuation after each driven turn). A pure read — commits nothing.
pub fn plan_notifications(
    core: Arc<AppCore>,
    space_id: String,
    post_action_id: String,
) -> oneshot::Receiver<Result<NotificationPlan, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.plan_notifications(space_id, post_action_id).await
    })
}

/// Save a post without requesting a response (the save side of save-vs-request:
/// `⌘⇧↩`). Creates the space when `space_id` is `None`; needs no credential.
/// `reply_to`, when set, branches off that post (vs the linear tail).
pub fn post(
    core: Arc<AppCore>,
    prompt: String,
    space_id: Option<String>,
    reply_to: Option<String>,
    references: Vec<ReferenceSpec>,
) -> oneshot::Receiver<Result<PostResult, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.post_with_references(prompt, space_id, reply_to, references)
            .await
    })
}

/// Edit a post in place — append a new human generation of its item (the
/// inline-edit commit). No credential, no model call.
pub fn edit_post(
    core: Arc<AppCore>,
    action_id: String,
    new_prompt: String,
    remove_references: Vec<i64>,
) -> oneshot::Receiver<Result<PostResult, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.edit_post_with_removals(action_id, new_prompt, remove_references)
            .await
    })
}

/// Every current-generation post referencing `action_id` (the reverse index
/// behind source-post highlights). Pure read.
pub fn references_to(
    core: Arc<AppCore>,
    action_id: String,
) -> oneshot::Receiver<Result<Vec<IncomingReference>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.references_to(action_id).await
    })
}

/// The home space of a persisted action (`None` if unknown) — resolves where
/// a quoted post lives before navigating a footnote (cross-space references).
pub fn action_space(
    core: Arc<AppCore>,
    action_id: String,
) -> oneshot::Receiver<Result<Option<String>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.action_space(action_id).await
    })
}

/// Regenerate an inference — append a new agent generation of its item.
pub fn regenerate(
    core: Arc<AppCore>,
    action_id: String,
    model: String,
) -> oneshot::Receiver<Result<ChatResult, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.regenerate(action_id, model).await
    })
}

/// Load a space's threaded-post render tree (the reopened-space initial load
/// and the post-stream reload). The flattened `PostNode` list the transcript
/// renders from — see `AppCore::get_space_tree`.
pub fn get_space_tree(
    core: Arc<AppCore>,
    space_id: String,
) -> oneshot::Receiver<Result<Vec<PostNode>, AppError>> {
    spawn_oneshot(core, move |core| async move {
        core.get_space_tree(space_id).await
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
