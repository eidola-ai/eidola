//! `Space` — the per-conversation domain entity.
//!
//! Per `docs/architecture/state.md` ("Space entities — shared, registried"),
//! a `Space` is a long-lived gpui entity owning *everything* about one
//! conversation: the transcript (`Loadable<Vec<ChatMessageView>>`), the live
//! streaming buffers + reasoning disclosure, the per-space model selection,
//! and the space id (`None` until the first exchange persists and assigns
//! one). It is created and shared through [`crate::stores::SpacesStore`]'s
//! registry, so **two windows on the same space hold the same entity** —
//! a submit/stream in one window appears in the other, structurally (the
//! wave-2 bug-4 fix).
//!
//! Tasks-as-fields, per the doctrine:
//!
//! - `submit_runner` is the **single runner slot** (`Option<Task<()>>`). The
//!   current UX is preserved: a submit while one is in flight is a no-op (the
//!   runner just makes the ordering of the load-vs-submit race structural,
//!   retiring the old `transcript_generation` counter — the entity owns both
//!   the initial transcript load and every submit, so they serialize on
//!   `&mut self` between awaits and can never clobber each other).
//! - `load_task` owns the reopened-space initial transcript load (supersede
//!   slot).
//!
//! No `.detach()`: every async operation lives in an owned field on the
//! entity and dies with it.
//!
//! `Space` is an [`EventEmitter`] of [`SpaceEvent`] so window-local views can
//! react *semantically* (e.g. tail-scroll only on `StreamDelta`) on top of the
//! plain `cx.observe` re-render path.

use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatStreamEvent, PostNode, PostReference, SpaceMessage};
use gpui::{Context, EventEmitter, Task};

use crate::bridge;
use crate::loadable::Loadable;

/// In-flight assistant response. While this is `Some(...)`, the space is
/// streaming — `reasoning` and `content` grow as deltas arrive. On
/// completion the streaming response is dropped; the captured reasoning is
/// moved onto the just-finalized assistant entry in the transcript so the
/// disclosure remains available after the stream ends.
#[derive(Default, Clone)]
pub struct StreamingResponse {
    pub reasoning: String,
    pub content: String,
    /// Whether the reasoning disclosure is open. Independent of whether
    /// reasoning has any content yet.
    pub expanded: bool,
    /// In-stream error: the stream produced something the user should see,
    /// but the request as a whole has not necessarily failed.
    pub error: Option<String>,
}

/// A single rendered chat row: the persisted post plus the byline identity
/// shown in its gutter, and any reasoning captured for it during streaming.
///
/// The post-tree redesign (wave 5.3) feeds this from [`PostNode`]
/// (`AppCore::get_space_tree`): the `byline` is the post's gutter label (the
/// model name for an assistant turn, "You" for the human), and `action_id` /
/// `item_id` carry the stable post identity the 5.4 hover affordances
/// (reply / edit / regenerate) will wire to. `message` (role + concatenated
/// text) is retained for the body render and the test API. Synthetic rows (the
/// optimistic user turn, test fixtures) come through [`Self::new`], which
/// derives the byline from the role and leaves the ids `None`.
///
/// Reasoning is ephemeral session state — the local DB stores only the
/// assistant's final content — so older posts from a re-loaded space carry
/// `reasoning = None`. New assistant posts adopt whatever reasoning was
/// streaming at finalize.
#[derive(Clone)]
pub struct ChatMessageView {
    pub message: SpaceMessage,
    /// The gutter byline ("You" / a model name / "Eidola" / "Error").
    pub byline: String,
    /// The post's current-generation action id, when this row came from the
    /// post tree (`None` for synthetic/optimistic/test rows). The 5.4 hover
    /// affordances key off this.
    pub action_id: Option<String>,
    /// The post's stable item id (see `action_id`).
    pub item_id: Option<String>,
    /// The structural reply antecedent — the action this post replies to
    /// (`None` for a root). The space-tree view relinks the flat transcript
    /// into a navigable tree through this edge.
    pub parent_action_id: Option<String>,
    /// Wall-clock creation time (unix seconds) of the post, for the gutter
    /// time byline. `0` for synthetic rows with no persisted timestamp.
    pub created_at: i64,
    /// Thread depth from the flattener: `0` is the spine, `> 0` an indented
    /// branch. Drives the branch indent + margin rail in the render.
    pub depth: usize,
    /// `true` when this post is a non-first reply to its parent (a branch head).
    pub is_branch: bool,
    /// Total generations of this item (`>= 1`); `> 1` means the post has been
    /// edited/regenerated and a generation switcher applies.
    pub generation_count: i64,
    /// Non-structural antecedent links (`reference` edges) — inline quotes /
    /// backlinks, rendered as `❝ quote ❞ — re: X` chips at the top of the post.
    pub references: Vec<PostReference>,
    pub reasoning: Option<String>,
    pub reasoning_expanded: bool,
}

impl ChatMessageView {
    /// A synthetic row from a role/content message (the optimistic user turn
    /// and test fixtures). The byline is derived from the role; the post ids
    /// are unknown until the row is reloaded from the tree.
    pub fn new(message: SpaceMessage) -> Self {
        let byline = byline_for_role(&message.role).to_string();
        Self {
            message,
            byline,
            action_id: None,
            item_id: None,
            parent_action_id: None,
            created_at: 0,
            depth: 0,
            is_branch: false,
            generation_count: 1,
            references: Vec::new(),
            reasoning: None,
            reasoning_expanded: false,
        }
    }

    /// A row from the persisted post tree: the body is the post's concatenated
    /// text blocks, the byline is its participant identity, and the post ids
    /// are carried for hover wiring.
    pub fn from_post(node: PostNode) -> Self {
        let role = match node.action_type.as_str() {
            "user_input" => "user",
            "inference" => "assistant",
            "error" => "error",
            _ => "assistant",
        }
        .to_string();
        let content = node
            .blocks
            .iter()
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("");
        let byline = byline_for_participant(&node.participant.kind, &node.participant.label);
        Self {
            message: SpaceMessage { role, content },
            byline,
            action_id: Some(node.action_id),
            item_id: Some(node.item_id),
            parent_action_id: node.parent_action_id,
            created_at: node.created_at,
            depth: node.depth,
            is_branch: node.is_branch,
            generation_count: node.generation_count,
            references: node.references,
            reasoning: None,
            reasoning_expanded: false,
        }
    }
}

/// The gutter byline for a synthetic row, from its chat role.
fn byline_for_role(role: &str) -> &'static str {
    match role {
        "user" => "You",
        "assistant" => "Eidola",
        "error" => "Error",
        _ => "—",
    }
}

/// The gutter byline for a post-tree row, from its participant identity. The
/// *local* human (the generic "user" participant) reads as "You"; any other
/// human reads by name (multi-party spaces); an agent's byline is its model
/// label.
fn byline_for_participant(kind: &str, label: &str) -> String {
    match kind {
        "human" if label.is_empty() || label.eq_ignore_ascii_case("user") => "You".to_string(),
        "human" => label.to_string(),
        "agent" if !label.is_empty() => label.to_string(),
        "agent" => "Eidola".to_string(),
        "system" => "System".to_string(),
        _ if !label.is_empty() => label.to_string(),
        _ => "—".to_string(),
    }
}

/// Map a freshly-loaded post tree into render rows.
fn views_from_nodes(nodes: Vec<PostNode>) -> Vec<ChatMessageView> {
    nodes.into_iter().map(ChatMessageView::from_post).collect()
}

/// Semantic events a `Space` emits. `cx.observe` covers plain re-render; these
/// let a view react to *what* happened (tail-scroll only on `StreamDelta`, a
/// failure band on `Failed`, etc.).
#[derive(Clone, Debug)]
pub enum SpaceEvent {
    /// The transcript message list changed (a reload landed, or a submit
    /// appended the user's turn / finalized the assistant's).
    MessagesChanged,
    /// A streaming delta arrived (reasoning or content). The tail-scroll
    /// policy keys off this.
    StreamDelta,
    /// The stream finished (success): the assistant turn is finalized into
    /// the transcript and `streaming` has been cleared.
    StreamEnded,
    /// A submit failed with a typed error. The view routes onboarding-degraded
    /// states (`InsufficientBalance`) off this.
    Failed(AppError),
}

pub struct Space {
    app_core: Option<Arc<AppCore>>,
    /// The persisted space id. `None` for a blank ⌘N space until its first
    /// exchange persists and assigns one (at which point the registry adopts
    /// the entity under that id — see [`crate::stores::SpacesStore`]).
    id: Option<String>,
    /// The conversation transcript.
    transcript: Loadable<Vec<ChatMessageView>>,
    /// In-flight streaming assistant response, or `None` when idle.
    streaming: Option<StreamingResponse>,
    /// The window-independent model choice for this space's sends. `None`
    /// means "follow the config default". A switch mid-stream applies to the
    /// next send — the in-flight request is never hot-swapped (the model is
    /// captured into the runner at submit time).
    selected_model: Option<String>,
    /// The model id handed to the most recent submit (set on every submit,
    /// including stub-core submits, before the backend guard). Behavior tests
    /// assert against this to prove what a real send would use.
    last_submitted_model: Option<String>,
    /// The single submit runner slot. Replace-cancels; while `Some`, a submit
    /// is in flight and a new submit is a no-op (the current UX). The runner
    /// owns the streaming pump and the post-stream transcript reload.
    submit_runner: Option<Task<()>>,
    /// Supersede slot for the reopened-space initial transcript load.
    load_task: Option<Task<()>>,
}

impl EventEmitter<SpaceEvent> for Space {}

impl Space {
    /// Construct a blank space (⌘N): no id, empty transcript, instant. The
    /// registry adopts it once its first exchange assigns an id.
    pub fn blank(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            id: None,
            transcript: Loadable::loaded(Vec::new()),
            streaming: None,
            selected_model: None,
            last_submitted_model: None,
            submit_runner: None,
            load_task: None,
        }
    }

    /// Construct a space bound to an existing id and kick off the initial
    /// transcript load. The load lands via [`Self::apply_loaded_transcript`]
    /// inside the entity, so it serializes against any submit that races it.
    pub fn existing(app_core: Option<Arc<AppCore>>, id: String, cx: &mut Context<Self>) -> Self {
        let mut space = Self {
            app_core: app_core.clone(),
            id: Some(id.clone()),
            transcript: Loadable::NotLoaded,
            streaming: None,
            selected_model: None,
            last_submitted_model: None,
            submit_runner: None,
            load_task: None,
        };
        space.load_transcript(cx);
        space
    }

    /// A stub space with a fixture transcript (tests). No backend, so async
    /// methods early-return after the local mutation.
    pub fn stub(id: Option<String>, messages: Vec<ChatMessageView>) -> Self {
        Self {
            app_core: None,
            id,
            transcript: Loadable::loaded(messages),
            streaming: None,
            selected_model: None,
            last_submitted_model: None,
            submit_runner: None,
            load_task: None,
        }
    }

    // -- Readers -----------------------------------------------------------

    /// The persisted space id, if one has been assigned.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The transcript cell.
    pub fn transcript(&self) -> &Loadable<Vec<ChatMessageView>> {
        &self.transcript
    }

    /// The transcript as a slice (empty if not loaded).
    pub fn messages(&self) -> &[ChatMessageView] {
        self.transcript.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// The in-flight streaming response, if any.
    pub fn streaming(&self) -> Option<&StreamingResponse> {
        self.streaming.as_ref()
    }

    /// Whether a submit is currently in flight (the runner slot is occupied).
    pub fn is_streaming(&self) -> bool {
        self.streaming.is_some()
    }

    /// This space's explicit model selection, if any.
    pub fn selected_model(&self) -> Option<&str> {
        self.selected_model.as_deref()
    }

    /// The model id handed to the most recent submit (see field docs).
    pub fn last_submitted_model(&self) -> Option<&str> {
        self.last_submitted_model.as_deref()
    }

    // -- Model selection ---------------------------------------------------

    /// Choose the model for this space's subsequent sends. A switch while a
    /// response is streaming applies to the *next* send — the in-flight
    /// request is never hot-swapped (the runner captured its model at submit).
    pub fn select_model(&mut self, id: String, cx: &mut Context<Self>) {
        self.selected_model = Some(id);
        cx.notify();
    }

    // -- Streaming disclosure ----------------------------------------------

    /// Toggle the streaming reasoning disclosure.
    pub fn toggle_streaming_reasoning(&mut self, cx: &mut Context<Self>) {
        if let Some(s) = self.streaming.as_mut() {
            s.expanded = !s.expanded;
            cx.notify();
        }
    }

    /// Toggle the reasoning disclosure on a finalized message at `idx`.
    pub fn toggle_message_reasoning(&mut self, idx: usize, cx: &mut Context<Self>) {
        if let Loadable::Loaded { value, .. } = &mut self.transcript
            && let Some(entry) = value.get_mut(idx)
        {
            entry.reasoning_expanded = !entry.reasoning_expanded;
            cx.notify();
        }
    }

    // -- Bus integration ---------------------------------------------------

    /// React to a `Change::Space(id)` for *this* space's id — refresh the
    /// transcript (this is how a CLI write to the same space appears, in
    /// process). A no-op if the id doesn't match or no exchange is in flight
    /// to clobber. Routed through the bus-bridge dispatch in
    /// `stores::dispatch_change`.
    pub fn on_space_changed(&mut self, changed_id: &str, cx: &mut Context<Self>) {
        if self.id.as_deref() != Some(changed_id) {
            return;
        }
        // A submit currently streaming already owns the transcript's truth and
        // will reload on finalize; don't race it with a bus-driven reload.
        if self.submit_runner.is_some() {
            return;
        }
        self.load_transcript(cx);
    }

    // -- Transcript loading ------------------------------------------------

    /// (Re)load the transcript from the DB. Supersede slot. The completion
    /// re-enters the entity via [`Self::apply_loaded_transcript`], so even a
    /// slow load that finishes after a local submit cannot clobber it — the
    /// merge preserves what's already present by position.
    fn load_transcript(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.id.clone() else {
            return;
        };
        let Some(app_core) = self.app_core.clone() else {
            return;
        };
        self.transcript = std::mem::take(&mut self.transcript).to_loading();
        let rx = bridge::get_space_tree(app_core, id);
        self.load_task = Some(cx.spawn(async move |this, cx| {
            let result = rx
                .await
                .unwrap_or_else(|_| {
                    Err(AppError::Internal {
                        message: "fetch space tree task cancelled".into(),
                    })
                })
                .map(views_from_nodes);
            let _ = this.update(cx, |this, cx| {
                let _ = this.apply_loaded_transcript(result, cx);
                this.load_task = None;
            });
        }));
        cx.notify();
    }

    /// Apply a completed transcript load. On success, merge into the existing
    /// transcript (preserving reasoning by position); on failure, retain the
    /// prior snapshot via `Loadable::Failed { prior }`.
    ///
    /// **The load-vs-submit race is serialized here**, which retires the old
    /// `transcript_generation` counter: if a mutation has moved the transcript
    /// ahead since this load started (`streaming` or the runner slot is
    /// occupied), the load result is *stale* and dropped — the mutation's own
    /// post-commit reload is the authoritative truth and would clobber the
    /// just-appended user turn. (Every mutation also cancels the load task via
    /// [`Self::supersede_load_for_mutation`], so this guard is defense in
    /// depth.) Returns whether the load was applied.
    fn apply_loaded_transcript(
        &mut self,
        result: Result<Vec<ChatMessageView>, AppError>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.streaming.is_some() || self.submit_runner.is_some() {
            // A mutation raced ahead of this load; its reload wins.
            return false;
        }
        match result {
            Ok(messages) => {
                self.merge_from_db(messages, None);
                cx.emit(SpaceEvent::MessagesChanged);
            }
            Err(error) => {
                self.transcript = std::mem::take(&mut self.transcript).resolve(Err(error));
            }
        }
        cx.notify();
        true
    }

    /// Merge a fresh post-tree render list into the transcript, preserving any
    /// previously-attached reasoning by index (we only ever append, so
    /// positions are stable) and attaching the just-captured streaming
    /// reasoning to the new last assistant entry if non-empty. The incoming
    /// rows carry the byline + post ids from the tree; only the ephemeral
    /// reasoning state is carried forward from the prior snapshot.
    fn merge_from_db(&mut self, mut next: Vec<ChatMessageView>, new_reasoning: Option<String>) {
        let prior = self.transcript.value().cloned();
        for (idx, entry) in next.iter_mut().enumerate() {
            let prior_entry = prior.as_ref().and_then(|p| p.get(idx));
            let same_position = prior_entry.is_some_and(|p| {
                p.message.role == entry.message.role && p.message.content == entry.message.content
            });
            if same_position {
                entry.reasoning = prior_entry.and_then(|p| p.reasoning.clone());
                entry.reasoning_expanded = prior_entry.is_some_and(|p| p.reasoning_expanded);
            }
        }

        if let Some(reasoning) = new_reasoning
            && !reasoning.is_empty()
            && let Some(entry) = next
                .iter_mut()
                .rev()
                .find(|e| e.message.role == "assistant")
        {
            entry.reasoning = Some(reasoning);
        }

        self.transcript = Loadable::loaded(next);
    }

    // -- Submit ------------------------------------------------------------

    /// Prologue shared by every transcript-mutating operation (submit /
    /// post-only / edit / regenerate), called at the moment the operation is
    /// accepted: cancel any in-flight transcript load by dropping its task.
    /// The mutation's own post-commit reload is now the authoritative truth,
    /// and a superseded load must never land late around it — the same
    /// stale-fetch class as the Record listing race from PR #179; the fix is
    /// the same replace-cancels idiom (`docs/architecture/state.md` →
    /// "Concurrency patterns"). If the cancelled load was an *initial* one
    /// (`Loading`, nothing kept visible), the cell steps back to `NotLoaded`
    /// so no spinner outlives its task ("every spinner maps to a live task");
    /// a re-fetch over data stays `Loaded { stale: true }` until the
    /// mutation's reload resolves it.
    fn supersede_load_for_mutation(&mut self) {
        self.load_task = None;
        if self.transcript.is_loading() {
            self.transcript = Loadable::NotLoaded;
        }
    }

    /// Submit a prompt with an explicitly-resolved model. The model is
    /// resolved by the caller (window selection → config default → fallback)
    /// because the config snapshot lives in `ConfigStore`, which the view
    /// observes — keeping `Space` free of a config dependency.
    ///
    /// Submit-during-streaming is a no-op (the current UX): the runner slot is
    /// the honest "in flight" signal. Returns `true` if the submit was
    /// accepted (a turn was appended), `false` if it was a no-op (empty prompt
    /// or already streaming).
    pub fn submit(
        &mut self,
        prompt: String,
        model: String,
        reply_to: Option<String>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.submit_runner.is_some() || self.streaming.is_some() {
            return false;
        }
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return false;
        }

        // Record the resolved model before the backend guard so stub-core
        // tests observe exactly what a real send would use. This is also the
        // value app-core persists on the action row.
        self.last_submitted_model = Some(model.clone());

        // Cancel any in-flight transcript load: the submit's own post-stream
        // reload is now the authoritative truth (the `apply_loaded_transcript`
        // guard also enforces this, but dropping the task frees the slot
        // eagerly).
        self.supersede_load_for_mutation();

        // Append the user's turn locally and enter the streaming state. This
        // mutation is what a submit-vs-load race must not clobber; since the
        // same entity owns the load, the merge preserves it by position.
        let mut messages = self.transcript.value().cloned().unwrap_or_default();
        messages.push(ChatMessageView::new(SpaceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        }));
        self.transcript = Loadable::loaded(messages);
        self.streaming = Some(StreamingResponse::default());
        cx.emit(SpaceEvent::MessagesChanged);
        cx.notify();

        let Some(app_core) = self.app_core.clone() else {
            // Stub stores (behavior tests): the local state update above has
            // happened; without a backend there is nothing more to drive. We
            // intentionally leave `streaming = Some(...)` (the current UX:
            // tests assert the view entered the streaming state) and do NOT
            // occupy the runner slot — a stub has no task to own.
            return true;
        };
        let space_id = self.id.clone();
        self.spawn_stream(app_core, prompt, model, space_id, reply_to, cx);
        true
    }

    /// Save a post **without requesting a response** — the save side of the
    /// save-vs-request split (`⌘⇧↩`). Appends the user's turn and persists it
    /// via `AppCore::post` (no credential, no model call); on completion the
    /// transcript reloads from the tree and a blank space adopts its new id.
    /// Returns `true` if accepted, `false` if a no-op (empty / busy).
    pub fn post_only(
        &mut self,
        prompt: String,
        reply_to: Option<String>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.submit_runner.is_some() || self.streaming.is_some() {
            return false;
        }
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return false;
        }

        self.supersede_load_for_mutation();

        // Optimistically append the user's turn (no streaming state — this path
        // requests nothing).
        let mut messages = self.transcript.value().cloned().unwrap_or_default();
        messages.push(ChatMessageView::new(SpaceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        }));
        self.transcript = Loadable::loaded(messages);
        cx.emit(SpaceEvent::MessagesChanged);
        cx.notify();

        let Some(app_core) = self.app_core.clone() else {
            // Stub stores (behavior tests): the local append above is the
            // observable effect; no backend to persist to.
            return true;
        };
        let space_id = self.id.clone();
        let rx = bridge::post(app_core.clone(), prompt, space_id, reply_to);
        self.submit_runner = Some(cx.spawn(async move |this, cx| {
            let outcome = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "post task cancelled".into(),
                })
            });
            match outcome {
                Ok(result) => {
                    let msgs_rx = bridge::get_space_tree(app_core, result.space_id.clone());
                    let msgs = msgs_rx
                        .await
                        .unwrap_or_else(|_| {
                            Err(AppError::Internal {
                                message: "fetch space tree task cancelled".into(),
                            })
                        })
                        .map(views_from_nodes);
                    let _ = this.update(cx, |this, cx| {
                        this.id = Some(result.space_id.clone());
                        this.submit_runner = None;
                        match msgs {
                            Ok(messages) => {
                                this.merge_from_db(messages, None);
                                cx.emit(SpaceEvent::MessagesChanged);
                                // StreamEnded drives the registry's blank-space
                                // adoption (it reads id()); a post earns an id
                                // too, so reuse the same signal.
                                cx.emit(SpaceEvent::StreamEnded);
                            }
                            Err(e) => {
                                this.transcript =
                                    std::mem::take(&mut this.transcript).resolve(Err(e.clone()));
                                cx.emit(SpaceEvent::Failed(e));
                            }
                        }
                        cx.notify();
                    });
                }
                Err(e) => {
                    let _ = this.update(cx, |this, cx| {
                        this.submit_runner = None;
                        if this.id.is_none()
                            && let Some(id) = e.chat_space_id()
                        {
                            this.id = Some(id.to_string());
                        }
                        cx.emit(SpaceEvent::Failed(e.root().clone()));
                        cx.notify();
                    });
                }
            }
        }));
        true
    }

    /// Commit an inline edit — append a new human generation of `action_id`'s
    /// item via `AppCore::edit_post`, then reload the tree (the edited post
    /// replaces its prior generation in place). No credential, no model call.
    pub fn edit(&mut self, action_id: String, new_prompt: String, cx: &mut Context<Self>) -> bool {
        if self.submit_runner.is_some() || self.streaming.is_some() {
            return false;
        }
        let new_prompt = new_prompt.trim().to_string();
        if new_prompt.is_empty() {
            return false;
        }
        self.supersede_load_for_mutation();
        let Some(app_core) = self.app_core.clone() else {
            return true; // stub: no backend
        };
        let rx = bridge::edit_post(app_core.clone(), action_id, new_prompt);
        self.submit_runner = Some(cx.spawn(async move |this, cx| {
            let outcome = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "edit task cancelled".into(),
                })
            });
            Self::finish_reload(this, cx, app_core, outcome.map(|r| r.space_id)).await;
        }));
        true
    }

    /// Regenerate an inference — append a new agent generation of `action_id`'s
    /// item via `AppCore::regenerate` (spends credits), then reload the tree.
    pub fn regenerate_post(
        &mut self,
        action_id: String,
        model: String,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.submit_runner.is_some() || self.streaming.is_some() {
            return false;
        }
        self.last_submitted_model = Some(model.clone());
        self.supersede_load_for_mutation();
        let Some(app_core) = self.app_core.clone() else {
            return true; // stub: no backend
        };
        let rx = bridge::regenerate(app_core.clone(), action_id, model);
        self.submit_runner = Some(cx.spawn(async move |this, cx| {
            let outcome = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "regenerate task cancelled".into(),
                })
            });
            Self::finish_reload(this, cx, app_core, outcome.map(|r| r.space_id)).await;
        }));
        true
    }

    /// Shared completion for edit/regenerate: on success reload the tree from
    /// the resulting space id (adopting it if the space was blank) and emit
    /// `StreamEnded`; on failure surface `Failed`. Clears the runner slot.
    async fn finish_reload(
        this: gpui::WeakEntity<Self>,
        cx: &mut gpui::AsyncApp,
        app_core: Arc<AppCore>,
        outcome: Result<String, AppError>,
    ) {
        match outcome {
            Ok(space_id) => {
                let msgs = bridge::get_space_tree(app_core, space_id.clone())
                    .await
                    .unwrap_or_else(|_| {
                        Err(AppError::Internal {
                            message: "fetch space tree task cancelled".into(),
                        })
                    })
                    .map(views_from_nodes);
                let _ = this.update(cx, |this, cx| {
                    this.id = Some(space_id);
                    this.submit_runner = None;
                    match msgs {
                        Ok(messages) => {
                            this.merge_from_db(messages, None);
                            cx.emit(SpaceEvent::MessagesChanged);
                            cx.emit(SpaceEvent::StreamEnded);
                        }
                        Err(e) => {
                            this.transcript =
                                std::mem::take(&mut this.transcript).resolve(Err(e.clone()));
                            cx.emit(SpaceEvent::Failed(e));
                        }
                    }
                    cx.notify();
                });
            }
            Err(e) => {
                let _ = this.update(cx, |this, cx| {
                    this.submit_runner = None;
                    cx.emit(SpaceEvent::Failed(e.root().clone()));
                    cx.notify();
                });
            }
        }
    }

    /// Drive a streaming chat request inside the single runner slot. On
    /// completion the transcript is reloaded from the DB and the captured
    /// (ephemeral) reasoning is attached to the new last assistant entry.
    fn spawn_stream(
        &mut self,
        app_core: Arc<AppCore>,
        prompt: String,
        model: String,
        space_id: Option<String>,
        reply_to: Option<String>,
        cx: &mut Context<Self>,
    ) {
        let (mut event_rx, done_rx) =
            bridge::chat_stream(app_core.clone(), prompt, model, space_id, reply_to);

        self.submit_runner = Some(cx.spawn(async move |this, cx| {
            while let Some(event) = event_rx.recv().await {
                let _ = this.update(cx, |this, cx| {
                    if let Some(s) = this.streaming.as_mut() {
                        match event {
                            ChatStreamEvent::ReasoningDelta(d) => s.reasoning.push_str(&d),
                            ChatStreamEvent::ContentDelta(d) => s.content.push_str(&d),
                        }
                        cx.emit(SpaceEvent::StreamDelta);
                        cx.notify();
                    }
                });
            }

            let outcome = done_rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "chat task cancelled".into(),
                })
            });

            match outcome {
                Ok(result) => {
                    let msgs_rx = bridge::get_space_tree(app_core, result.space_id.clone());
                    let msgs = msgs_rx
                        .await
                        .unwrap_or_else(|_| {
                            Err(AppError::Internal {
                                message: "fetch space tree task cancelled".into(),
                            })
                        })
                        .map(views_from_nodes);
                    let _ = this.update(cx, |this, cx| {
                        let captured_reasoning =
                            this.streaming.as_ref().map(|s| s.reasoning.clone());
                        this.streaming = None;
                        // Assigning the id (a blank space earning its first
                        // persisted id) is what lets the registry adopt this
                        // entity: a `SpacesStore` subscriber reads `id()` on
                        // `StreamEnded` and keys the entity under it, so a
                        // later open of the same id shares this same `Space`.
                        this.id = Some(result.space_id.clone());
                        this.submit_runner = None;
                        match msgs {
                            Ok(messages) => {
                                this.merge_from_db(messages, captured_reasoning);
                                cx.emit(SpaceEvent::MessagesChanged);
                                cx.emit(SpaceEvent::StreamEnded);
                            }
                            Err(e) => {
                                this.transcript =
                                    std::mem::take(&mut this.transcript).resolve(Err(e.clone()));
                                cx.emit(SpaceEvent::Failed(e));
                            }
                        }
                        cx.notify();
                    });
                }
                Err(e) => {
                    let _ = this.update(cx, |this, cx| {
                        this.streaming = None;
                        this.submit_runner = None;
                        // Blank-space id adoption on the FAILURE path: app-core
                        // wraps any post-persist error as `ChatFailed` carrying
                        // the now-persisted space id. If this is the first
                        // exchange of a blank space (id still `None`), learn the
                        // id from the wrapper so the registry can adopt this
                        // entity (mirroring the `StreamEnded` success path) — a
                        // later open of that id then shares this same `Space`.
                        if this.id.is_none()
                            && let Some(id) = e.chat_space_id()
                        {
                            this.id = Some(id.to_string());
                        }
                        // Surface the *unwrapped* source so the view's error
                        // routing (and band copy) sees the real error variant,
                        // not the id-carrying wrapper.
                        cx.emit(SpaceEvent::Failed(e.root().clone()));
                        cx.notify();
                    });
                }
            }
        }));
    }

    // -- Test seams --------------------------------------------------------

    /// Test-only: replace the transcript with a fixture list (snapshot tests).
    #[doc(hidden)]
    pub fn set_messages_for_test(&mut self, messages: Vec<SpaceMessage>, cx: &mut Context<Self>) {
        self.transcript =
            Loadable::loaded(messages.into_iter().map(ChatMessageView::new).collect());
        cx.notify();
    }

    /// Test-only: replace the transcript with a fixture post tree, preserving
    /// each post's depth/branch/generation metadata — so snapshot cases can
    /// render threaded branches and the generation switcher without a backend.
    #[doc(hidden)]
    pub fn set_post_tree_for_test(&mut self, nodes: Vec<PostNode>, cx: &mut Context<Self>) {
        self.transcript = Loadable::loaded(views_from_nodes(nodes));
        cx.notify();
    }

    /// Test-only: attach reasoning to the message at `idx`.
    #[doc(hidden)]
    pub fn set_reasoning_for_test(
        &mut self,
        idx: usize,
        reasoning: String,
        expanded: bool,
        cx: &mut Context<Self>,
    ) {
        if let Loadable::Loaded { value, .. } = &mut self.transcript
            && let Some(entry) = value.get_mut(idx)
        {
            entry.reasoning = Some(reasoning);
            entry.reasoning_expanded = expanded;
        }
        cx.notify();
    }

    /// Test-only: set the streaming response directly (snapshot tests).
    #[doc(hidden)]
    pub fn set_streaming_for_test(
        &mut self,
        streaming: Option<StreamingResponse>,
        cx: &mut Context<Self>,
    ) {
        self.streaming = streaming;
        cx.notify();
    }

    /// Test-only: push a content delta into the live streaming buffer and emit
    /// `StreamDelta`, exactly as the real streaming runner does. Drives the
    /// two-window sync test (both lenses observe the same entity, so both see
    /// the delta) without a backend.
    #[doc(hidden)]
    pub fn push_content_delta_for_test(&mut self, delta: &str, cx: &mut Context<Self>) {
        if let Some(s) = self.streaming.as_mut() {
            s.content.push_str(delta);
            cx.emit(SpaceEvent::StreamDelta);
            cx.notify();
        }
    }

    /// Test-only: arm a never-completing task in the load slot, standing in
    /// for an in-flight transcript load on a stub space (where
    /// `load_transcript` early-returns without a backend). Drives the
    /// mutation-supersedes-load replay tests.
    #[doc(hidden)]
    pub fn arm_load_for_test(&mut self, cx: &mut Context<Self>) {
        self.load_task = Some(cx.spawn(async move |_, _| std::future::pending::<()>().await));
    }

    /// Test-only: whether a transcript load is in flight (the load slot is
    /// occupied).
    #[doc(hidden)]
    pub fn has_pending_load_for_test(&self) -> bool {
        self.load_task.is_some()
    }

    /// Test-only: simulate completion of a transcript load (the race-replay
    /// test). Returns whether the load was applied — `false` when a submit has
    /// raced ahead (`streaming.is_some()`), proving a slow initial load that
    /// finishes after a local submit cannot clobber the submitted prompt.
    #[doc(hidden)]
    pub fn apply_loaded_transcript_for_test(
        &mut self,
        messages: Vec<SpaceMessage>,
        cx: &mut Context<Self>,
    ) -> bool {
        let views = messages.into_iter().map(ChatMessageView::new).collect();
        self.apply_loaded_transcript(Ok(views), cx)
    }

    /// Test-only: drive the chat-failure outcome exactly as `spawn_stream`'s
    /// error arm does — adopt the id from a `ChatFailed` wrapper if the space
    /// is still blank, clear streaming, and emit `Failed` with the unwrapped
    /// source. Drives the blank-space id-adoption-on-failure regression test.
    #[doc(hidden)]
    pub fn apply_chat_failure_for_test(&mut self, error: AppError, cx: &mut Context<Self>) {
        self.streaming = None;
        self.submit_runner = None;
        if self.id.is_none()
            && let Some(id) = error.chat_space_id()
        {
            self.id = Some(id.to_string());
        }
        cx.emit(SpaceEvent::Failed(error.root().clone()));
        cx.notify();
    }
}
