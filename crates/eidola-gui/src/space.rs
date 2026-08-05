//! `Space` — the per-conversation domain entity.
//!
//! Per `crates/eidola-gui/STATE.md` ("Space entities — shared, registried"),
//! a `Space` is a long-lived gpui entity owning *everything* about one
//! conversation: the transcript (`Loadable<Vec<ChatMessageView>>`), the live
//! streaming turns, and the space id (`None` until the first exchange persists
//! and assigns one). It is created and shared through
//! [`crate::stores::SpacesStore`]'s registry, so **two windows on the same
//! space hold the same entity** — a submit/stream in one window appears in the
//! other, structurally (the wave-2 bug-4 fix).
//!
//! Tasks-as-fields, per the doctrine:
//!
//! - `post_runner` is the **exclusive mutation slot** (`Option<Task<()>>`) for
//!   the save-side operations — submit (post + notification plan), post-only,
//!   edit, regenerate. A mutation while one is in flight is a no-op.
//! - `turn_runners` is the **keyed slot map** (`HashMap<u64, Task<()>>`) for
//!   streaming response turns — STATE.md's "independent per-key work" pattern.
//!   A submit's notification plan can fan out to several participants at once
//!   (Participants v1), and each turn owns its own runner + streaming buffers,
//!   so concurrent turns stream side by side and one turn's failure never
//!   disturbs its siblings.
//! - `load_task` owns the reopened-space initial transcript load (supersede
//!   slot).
//!
//! No `.detach()`: every async operation lives in an owned field on the
//! entity and dies with it.
//!
//! `Space` is an [`EventEmitter`] of [`SpaceEvent`] so window-local views can
//! react *semantically* (e.g. tail-scroll only on `StreamDelta`) on top of the
//! plain `cx.observe` re-render path.

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AppCore, ChatResult, ChatStreamEvent, IncomingReference, NotificationPlan, PostNode,
    PostReference, PostTrace, ReferenceSpec, SpaceMessage,
};
use gpui::{Context, EventEmitter, Task};
use tokio::sync::{mpsc, oneshot};

use crate::bridge;
use crate::loadable::Loadable;

/// One in-flight assistant response's buffers. `reasoning` and `content` grow
/// as deltas arrive; on completion the captured reasoning is moved onto the
/// just-finalized assistant entry in the transcript so the disclosure remains
/// available after the stream ends.
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

/// One in-flight **turn** — a streaming response from one participant to one
/// post. Several can run concurrently (a submit's notification fan-out); each
/// renders as its own synthetic streaming leaf attached at its target, in
/// `seq` (start-time) order, so concurrent replies land as timestamp-ordered
/// sibling branches.
#[derive(Clone)]
pub struct StreamingTurn {
    /// Monotonic per-space turn sequence — the stable render key.
    pub seq: u64,
    /// The responding participant (`None` for a stub-mode synthetic turn,
    /// where no plan was computable).
    pub participant_id: Option<String>,
    /// The post being answered (`None` for a stub-mode synthetic turn on a
    /// not-yet-persisted post — the view attaches it at the selected leaf).
    pub target_action_id: Option<String>,
    /// The live buffers.
    pub response: StreamingResponse,
}

/// The turn a failed ask leaves behind — who was asked, about what — so the
/// recovery notice's Retry can re-ask the *same* participant without
/// disturbing any sibling turns still streaming.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedTurn {
    pub participant_id: String,
    pub target_action_id: String,
}

/// One content block's span within a post's concatenated body text — the
/// mapping a selection quote needs. The body a post's editor renders is the
/// blocks' texts joined with no separator, so a selection's editor-buffer
/// byte range maps to (`block_id`, block-relative range) by subtracting the
/// span start; a selection crossing block boundaries names no single block
/// and is not quotable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostBlockSpan {
    /// The content block's id (`PostBlock.id`) — what a `ReferenceSpec` names.
    pub block_id: String,
    /// This block's byte range within the concatenated content.
    pub range: std::ops::Range<usize>,
}

/// A single rendered chat row: the persisted post plus the byline identity
/// shown in its gutter, and any reasoning captured for it during streaming.
///
/// The post-tree redesign (wave 5.3) feeds this from [`PostNode`]
/// (`AppCore::get_space_tree`): the `byline` is the post's gutter label (the
/// responding participant's label for an agent turn, "You" for the human), and
/// `action_id` / `item_id` carry the stable post identity the hover
/// affordances (reply / edit / regenerate) wire to. `message` (role +
/// concatenated text) is retained for the body render and the test API.
/// Synthetic rows (the optimistic user turn, test fixtures) come through
/// [`Self::new`], which derives the byline from the role and leaves the ids
/// `None`.
///
/// Reasoning is **durable**: an inference's captured thinking is persisted as
/// a `thinking` content block beside its `text` block, so a re-loaded space
/// still shows the disclosure. [`Self::from_post`] splits the two — `text`
/// blocks are the readable body (and the only quotable ones), the `thinking`
/// block is the disclosure. The in-flight capture on `StreamingResponse` is
/// still attached at finalize so the disclosure never blinks out in the frame
/// between the stream ending and the reload landing.
#[derive(Clone)]
pub struct ChatMessageView {
    pub message: SpaceMessage,
    /// The gutter byline ("You" / a participant label / "Eidola" / "Error").
    pub byline: String,
    /// The post's current-generation action id, when this row came from the
    /// post tree (`None` for synthetic/optimistic/test rows). The hover
    /// affordances key off this.
    pub action_id: Option<String>,
    /// The post's stable item id (see `action_id`).
    pub item_id: Option<String>,
    /// The structural reply antecedent — the action this post replies to
    /// (`None` for a root). The space-tree view relinks the flat transcript
    /// into a navigable tree through this edge.
    pub parent_action_id: Option<String>,
    /// The model that produced an inference row (`None` for human/synthetic
    /// rows). Regenerate re-uses the post's own recorded model.
    pub model: Option<String>,
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
    /// Non-structural antecedent links (`reference` edges) — the post's quoted
    /// references. The body's `{{ embed N }}` markers materialize from these
    /// (via the embed map) and the footnote rail lists them below the body.
    pub references: Vec<PostReference>,
    /// The content blocks' spans within `message.content` (see
    /// [`PostBlockSpan`]) — what maps a selection to a quotable block range.
    pub blocks: Vec<PostBlockSpan>,
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
            model: None,
            created_at: 0,
            depth: 0,
            is_branch: false,
            generation_count: 1,
            references: Vec::new(),
            blocks: Vec::new(),
            reasoning: None,
            reasoning_expanded: false,
        }
    }

    /// A row from the persisted post tree: the body is the post's concatenated
    /// **text** blocks, the byline is its participant identity, and the post
    /// ids are carried for hover wiring. A `thinking` block (the model's own
    /// reasoning, persisted by `TurnPrep::persist_turn`) is lifted out of the
    /// body into [`Self::reasoning`] — it is a disclosure, not prose: it must
    /// not join the reading column, and it must not be quotable (the block
    /// spans that back a selection→quote carry only text blocks).
    pub fn from_post(node: PostNode) -> Self {
        let role = match node.action_type.as_str() {
            "user_input" => "user",
            "inference" => "assistant",
            "error" => "error",
            _ => "assistant",
        }
        .to_string();
        // Concatenate the text blocks, recording each block's span within the
        // joined content (the selection→quote mapping); collect the thinking
        // blocks separately as the disclosure.
        let mut content = String::new();
        let mut blocks = Vec::new();
        let mut reasoning = String::new();
        for b in &node.blocks {
            let Some(text) = b.text.as_deref() else {
                continue;
            };
            match b.block_type.as_str() {
                "text" => {
                    let start = content.len();
                    content.push_str(text);
                    blocks.push(PostBlockSpan {
                        block_id: b.id.clone(),
                        range: start..content.len(),
                    });
                }
                "thinking" => reasoning.push_str(text),
                // Other typed blocks (tool_use/…) have no v1 render.
                _ => {}
            }
        }
        let byline = byline_for_participant(&node.participant.kind, &node.participant.label);
        Self {
            message: SpaceMessage { role, content },
            byline,
            action_id: Some(node.action_id),
            item_id: Some(node.item_id),
            parent_action_id: node.parent_action_id,
            model: node.model,
            created_at: node.created_at,
            depth: node.depth,
            is_branch: node.is_branch,
            generation_count: node.generation_count,
            references: node.references,
            blocks,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
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
/// human reads by name (multi-party spaces); an agent's byline is its label.
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

/// The optimistic user turn a save appends before its post is durable.
///
/// **It carries `reply_to` as its structural parent.** Without it the row is
/// unparented, and `space_view::model::build_tree` chains an unparented row
/// onto the previous row in the flat list — which is the *tail of the
/// transcript*, not the post being replied to. Posting into a **branch** then
/// momentarily emptied that branch of everything but its first sibling; gpui
/// clamps the parent strip's scroll offset to the single remaining page, and
/// the branch selection was gone by the time the reload restored the second
/// child. Carrying the parent keeps the new post in the very slot the draft
/// occupied, so the selected branch survives the save with no re-selection
/// dance and no visible flash.
fn optimistic_user_turn(prompt: &str, reply_to: Option<&str>) -> ChatMessageView {
    let mut view = ChatMessageView::new(SpaceMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    });
    view.parent_action_id = reply_to.map(str::to_string);
    view
}

/// Semantic events a `Space` emits. `cx.observe` covers plain re-render; these
/// let a view react to *what* happened (tail-scroll only on `StreamDelta`, a
/// failure band on `Failed`, etc.).
#[derive(Clone, Debug)]
pub enum SpaceEvent {
    /// The transcript message list changed (a reload landed, or a submit
    /// appended the user's turn / finalized an assistant's).
    MessagesChanged,
    /// A streaming delta arrived (reasoning or content). The tail-scroll
    /// policy keys off this.
    StreamDelta,
    /// A turn finished (success): its response is finalized into the
    /// transcript and its streaming buffers have been cleared. Also emitted
    /// when a plain post persists — either way the space now has a durable id,
    /// which is what the registry's blank-space adoption keys off.
    StreamEnded,
    /// A **keyed turn** finished successfully, naming itself by the `seq`
    /// [`Space::ask`] handed back and by the post it wrote (`None` for a
    /// decline — the turn produced no post). Emitted beside `StreamEnded`,
    /// which says only *that* an exchange settled.
    ///
    /// It exists because a turn's streaming leaf is an ephemeral render key:
    /// the moment the turn lands, the leaf is gone and the persisted response
    /// takes its place. Anything that deferred work onto the leaf — the view's
    /// branch selection ([`crate::space_view::SpaceView::select_turn_branch`])
    /// — must be able to follow the turn across that swap, which needs both
    /// halves of the identity in one event.
    TurnEnded {
        seq: u64,
        response_action_id: Option<String>,
    },
    /// A mutation or turn failed with a typed error. The view routes
    /// onboarding-degraded states (`InsufficientBalance`) off this; a failed
    /// *turn* additionally records [`Space::failed_turn`] so the notice's
    /// Retry can re-ask the same participant.
    Failed(AppError),
    /// A submit's (or a driven turn's) notification plan hit the space's
    /// cascade limit at `target_action_id` — the resumable paused state. The
    /// view renders a quiet, dismissible "cascade limit reached — ask to
    /// continue" notice whose action is an explicit ask (which bypasses the
    /// guard by construction).
    CascadePaused {
        depth: i64,
        limit: i64,
        target_action_id: String,
    },
}

pub struct Space {
    app_core: Option<Arc<AppCore>>,
    /// The persisted space id. `None` for a blank ⌘N space until its first
    /// exchange persists and assigns one (at which point the registry adopts
    /// the entity under that id — see [`crate::stores::SpacesStore`]).
    id: Option<String>,
    /// The conversation transcript.
    transcript: Loadable<Vec<ChatMessageView>>,
    /// The in-flight streaming turns, in start order (`seq` ascending) — the
    /// timestamp order concurrent sibling replies land in.
    streams: Vec<StreamingTurn>,
    /// Monotonic source for [`StreamingTurn::seq`].
    next_turn_seq: u64,
    /// The model handed to the most recent regenerate (set before the backend
    /// guard). Behavior tests assert against this to prove what a real
    /// regenerate would use.
    last_submitted_model: Option<String>,
    /// The quoted references handed to the most recent accepted post, and the
    /// reference ordinals handed to the most recent accepted edit — both
    /// recorded **before** the backend guard, exactly like
    /// `last_submitted_model`, so behavior tests can assert what a real post
    /// or edit would carry without a live core.
    last_submitted_references: Vec<ReferenceSpec>,
    last_edit_removals: Vec<i64>,
    last_edit_text: String,
    /// The exclusive mutation slot: submit's post phase, post-only, edit,
    /// regenerate. While `Some`, another mutation is a no-op.
    post_runner: Option<Task<()>>,
    /// One runner per in-flight streaming turn, keyed by `seq` — the doctrine's
    /// keyed-slot pattern. Removing an entry cancels that turn only.
    turn_runners: HashMap<u64, Task<()>>,
    /// The turn a failed ask leaves behind (who + what), for Retry.
    failed_turn: Option<FailedTurn>,
    /// Supersede slot for the reopened-space initial transcript load.
    load_task: Option<Task<()>>,
    /// **Incoming references**, keyed by the quoted post's action id: every
    /// current-generation post quoting a range of that post
    /// (`AppCore::references_to`). This is the data behind the source-post
    /// highlights ("this passage was quoted") and their click-to-navigate.
    ///
    /// It lives here, on the shared per-space entity, rather than in a window:
    /// it is durable, queryable, per-space domain data, so two windows on one
    /// space must paint identical highlights and a `Change::Space` must
    /// refresh both (STATE.md — "views never own domain data"). It is loaded
    /// **lazily per post** (`ensure_incoming_references`, keyed fetch slots —
    /// STATE.md's "independent per-key work") rather than eagerly with the
    /// transcript, so the cost is one query per *rendered* post, not one per
    /// transcript row.
    incoming_refs: HashMap<String, Loadable<Vec<IncomingReference>>>,
    /// Per-key fetch slots for `incoming_refs`. Replacing an entry cancels
    /// that post's in-flight fetch and nothing else.
    incoming_ref_tasks: HashMap<String, Task<()>>,
    /// **Trace disclosures**, keyed by the post they hang under: every turn's
    /// tool rounds and decline decisions (`AppCore::space_traces`) — the
    /// actions the post tree deliberately collapses out.
    ///
    /// Space-wide rather than per-post (unlike `incoming_refs`): one query
    /// answers the whole space, and a turn's trace is by construction local to
    /// the space that ran it. It lives on the shared entity for the same
    /// reason the reverse index does — two windows on one space must disclose
    /// the same activity, and `Change::Space` must refresh both.
    traces: Loadable<HashMap<String, Vec<PostTrace>>>,
    /// Supersede slot for the trace fetch.
    traces_task: Option<Task<()>>,
    /// Which disclosures are open, by **turn** (`PostTrace::id`) — several
    /// turns can hang under one post, so the anchor is not an identity. Pure
    /// view state, but held here (as `ChatMessageView::reasoning_expanded` is)
    /// so a reload — or the other window on the same space — can't collapse one
    /// under the reader.
    traces_expanded: std::collections::HashSet<String>,
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
            streams: Vec::new(),
            next_turn_seq: 0,
            last_submitted_model: None,
            last_submitted_references: Vec::new(),
            last_edit_removals: Vec::new(),
            last_edit_text: String::new(),
            post_runner: None,
            turn_runners: HashMap::new(),
            failed_turn: None,
            load_task: None,
            incoming_refs: HashMap::new(),
            incoming_ref_tasks: HashMap::new(),
            traces: Loadable::NotLoaded,
            traces_task: None,
            traces_expanded: std::collections::HashSet::new(),
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
            streams: Vec::new(),
            next_turn_seq: 0,
            last_submitted_model: None,
            last_submitted_references: Vec::new(),
            last_edit_removals: Vec::new(),
            last_edit_text: String::new(),
            post_runner: None,
            turn_runners: HashMap::new(),
            failed_turn: None,
            load_task: None,
            incoming_refs: HashMap::new(),
            incoming_ref_tasks: HashMap::new(),
            traces: Loadable::NotLoaded,
            traces_task: None,
            traces_expanded: std::collections::HashSet::new(),
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
            streams: Vec::new(),
            next_turn_seq: 0,
            last_submitted_model: None,
            last_submitted_references: Vec::new(),
            last_edit_removals: Vec::new(),
            last_edit_text: String::new(),
            post_runner: None,
            turn_runners: HashMap::new(),
            failed_turn: None,
            load_task: None,
            incoming_refs: HashMap::new(),
            incoming_ref_tasks: HashMap::new(),
            traces: Loadable::NotLoaded,
            traces_task: None,
            traces_expanded: std::collections::HashSet::new(),
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

    /// The in-flight streaming turns, in start (`seq`) order.
    pub fn streams(&self) -> &[StreamingTurn] {
        &self.streams
    }

    // -- Incoming references (source highlights) ---------------------------

    /// The posts quoting `action_id`, if that post's reverse index has been
    /// loaded. Empty slice while it hasn't — highlights are decoration, so a
    /// not-yet-loaded index simply paints nothing rather than a spinner.
    pub fn incoming_references(&self, action_id: &str) -> &[IncomingReference] {
        self.incoming_refs
            .get(action_id)
            .and_then(|l| l.value())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Load `action_id`'s reverse index if it hasn't been requested yet — the
    /// lazy, per-post fetch the view calls for each post it actually renders.
    /// Idempotent: a cell that is loading, loaded, or failed is left alone (a
    /// failed decoration must not re-request every frame).
    pub fn ensure_incoming_references(&mut self, action_id: &str, cx: &mut Context<Self>) {
        if self.incoming_refs.contains_key(action_id) {
            return;
        }
        let Some(app_core) = self.app_core.clone() else {
            // Stub mode: record an empty cell so we don't re-enter each frame.
            self.incoming_refs
                .insert(action_id.to_string(), Loadable::loaded(Vec::new()));
            return;
        };
        self.incoming_refs
            .insert(action_id.to_string(), Loadable::Loading);
        let key = action_id.to_string();
        let rx = bridge::references_to(app_core, key.clone());
        let task = cx.spawn(async move |this, cx| {
            let result = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "reference lookup cancelled".into(),
                })
            });
            this.update(cx, |this, cx| {
                let prior = this.incoming_refs.remove(&key).unwrap_or(Loadable::Loading);
                this.incoming_refs
                    .insert(key.clone(), prior.resolve(result));
                this.incoming_ref_tasks.remove(&key);
                cx.notify();
            })
            .ok();
        });
        self.incoming_ref_tasks.insert(action_id.to_string(), task);
    }

    /// Drop every cached reverse index (and cancel its in-flight fetch), so
    /// the next render re-requests the ones it still needs.
    ///
    /// Called for **any** `Change::Space`, not just this space's: a reference
    /// created in space B changes what space A's posts should highlight, and
    /// the bus carries only the *written* space's id. Since the fetch is lazy
    /// per rendered post, a blanket invalidation costs at most one query per
    /// visible post rather than a whole-transcript sweep.
    pub fn invalidate_incoming_references(&mut self, cx: &mut Context<Self>) {
        if self.incoming_refs.is_empty() {
            return;
        }
        self.incoming_refs.clear();
        self.incoming_ref_tasks.clear();
        cx.notify();
    }

    /// Test seam: seed a post's reverse index without a backend, so behavior
    /// tests can drive the highlight surfaces against stub stores.
    #[doc(hidden)]
    pub fn seed_incoming_references_for_test(
        &mut self,
        action_id: impl Into<String>,
        refs: Vec<IncomingReference>,
    ) {
        self.incoming_refs
            .insert(action_id.into(), Loadable::loaded(refs));
    }

    // -- Trace disclosures (what the turn actually did) --------------------

    /// The traces anchored to `action_id`, if the space's trace index has been
    /// loaded. Empty slice while it hasn't — like the source highlights, a
    /// disclosure that isn't there yet simply doesn't render (never a spinner
    /// in the reading column).
    pub fn traces_for(&self, action_id: &str) -> &[PostTrace] {
        self.traces
            .value()
            .and_then(|m| m.get(action_id))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Whether the turn `trace_id`'s ([`PostTrace::id`]) disclosure is open.
    pub fn trace_expanded(&self, trace_id: &str) -> bool {
        self.traces_expanded.contains(trace_id)
    }

    /// Open/close one turn's trace disclosure. Keyed on the turn, not the post
    /// it hangs under: a post can carry several turns' disclosures, and opening
    /// one must not open the rest.
    pub fn toggle_trace(&mut self, trace_id: &str, cx: &mut Context<Self>) {
        if !self.traces_expanded.remove(trace_id) {
            self.traces_expanded.insert(trace_id.to_string());
        }
        cx.notify();
    }

    /// Load the space's trace index if it hasn't been requested yet — the
    /// lazy, idempotent fetch the view calls once per frame. A cell that is
    /// loading, loaded, or failed is left alone (a failed disclosure must not
    /// re-request every frame; the next `Change::Space` invalidates it).
    pub fn ensure_traces(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.traces, Loadable::NotLoaded) {
            return;
        }
        let Some(id) = self.id.clone() else {
            // A blank space has nothing persisted to disclose. Left
            // `NotLoaded`, so the index loads the moment its first exchange
            // assigns an id.
            return;
        };
        let Some(app_core) = self.app_core.clone() else {
            // Stub mode: record an empty index so we don't re-enter each frame
            // (a seeded fixture has already replaced it).
            self.traces = Loadable::loaded(HashMap::new());
            return;
        };
        self.traces = Loadable::Loading;
        let rx = bridge::space_traces(app_core, id);
        self.traces_task = Some(cx.spawn(async move |this, cx| {
            let result = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "trace lookup cancelled".into(),
                })
            });
            this.update(cx, |this, cx| {
                let indexed = result.map(|traces| {
                    let mut index: HashMap<String, Vec<PostTrace>> = HashMap::new();
                    for trace in traces {
                        index
                            .entry(trace.anchor_action_id.clone())
                            .or_default()
                            .push(trace);
                    }
                    index
                });
                this.traces = std::mem::take(&mut this.traces).resolve(indexed);
                this.traces_task = None;
                cx.notify();
            })
            .ok();
        }));
    }

    /// Drop the cached trace index (and cancel its in-flight fetch) so the next
    /// render re-requests it. Unlike the reverse index this is scoped to *this*
    /// space: a turn's rounds are written in the space that ran them.
    pub fn invalidate_traces(&mut self, cx: &mut Context<Self>) {
        if matches!(self.traces, Loadable::NotLoaded) {
            return;
        }
        self.traces = Loadable::NotLoaded;
        self.traces_task = None;
        cx.notify();
    }

    /// Test seam: seed the trace index without a backend, so behavior tests can
    /// drive the disclosure against stub stores.
    #[doc(hidden)]
    pub fn seed_traces_for_test(&mut self, traces: Vec<PostTrace>) {
        let mut index: HashMap<String, Vec<PostTrace>> = HashMap::new();
        for trace in traces {
            index
                .entry(trace.anchor_action_id.clone())
                .or_default()
                .push(trace);
        }
        self.traces = Loadable::loaded(index);
    }

    /// Whether any response turn is currently streaming.
    pub fn is_streaming(&self) -> bool {
        !self.streams.is_empty()
    }

    /// Whether any operation that will re-establish the transcript itself is
    /// in flight (the exclusive mutation, or any turn). While busy, a
    /// completed transcript load is stale by construction — the in-flight
    /// operation's own reload is authoritative — so
    /// [`Self::apply_loaded_transcript`] drops it and
    /// [`Self::on_space_changed`] defers to the operation's reload.
    ///
    /// Public because it is also the honest answer to "has this exchange
    /// settled?", which is what bounds the space view's post-submit tail pin
    /// (`space_view::follow_streaming_tail`): between the save landing and the
    /// response streaming there is no stream to observe, but the document is
    /// still the exchange's to grow.
    pub fn is_busy(&self) -> bool {
        self.post_runner.is_some() || !self.streams.is_empty() || !self.turn_runners.is_empty()
    }

    /// The model id handed to the most recent regenerate (see field docs).
    pub fn last_submitted_model(&self) -> Option<&str> {
        self.last_submitted_model.as_deref()
    }

    /// The quoted references the most recent accepted post carried (see the
    /// field docs).
    pub fn last_submitted_references(&self) -> &[ReferenceSpec] {
        &self.last_submitted_references
    }

    /// The reference ordinals the most recent accepted edit asked to remove.
    pub fn last_edit_removals(&self) -> &[i64] {
        &self.last_edit_removals
    }

    /// The body text the most recent accepted edit submitted. The view strips
    /// a removed reference's `{{ embed N }}` marker out of the submission, so
    /// this is what proves the marker left with its edge.
    pub fn last_edit_text(&self) -> &str {
        &self.last_edit_text
    }

    /// The turn a failed ask left behind, if any (drives the notice's Retry).
    pub fn failed_turn(&self) -> Option<&FailedTurn> {
        self.failed_turn.as_ref()
    }

    // -- Streaming disclosure ----------------------------------------------

    /// Toggle the reasoning disclosure on the in-flight turn `seq`.
    pub fn toggle_streaming_reasoning(&mut self, seq: u64, cx: &mut Context<Self>) {
        if let Some(s) = self.streams.iter_mut().find(|s| s.seq == seq) {
            s.response.expanded = !s.response.expanded;
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
    /// process). A no-op if the id doesn't match or an operation owning the
    /// transcript's truth is in flight (each turn/mutation reloads on its own
    /// completion, so nothing is lost). Routed through the bus-bridge dispatch
    /// in `stores::dispatch_change`.
    pub fn on_space_changed(&mut self, changed_id: &str, cx: &mut Context<Self>) {
        if self.id.as_deref() != Some(changed_id) {
            return;
        }
        if self.is_busy() {
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
    /// **The load-vs-mutation race is serialized here**: if an operation that
    /// owns the transcript's truth is in flight ([`Self::is_busy`] — the
    /// exclusive mutation, or any streaming turn), the load result is *stale*
    /// and dropped — the operation's own post-commit reload is authoritative
    /// and would otherwise be clobbered (e.g. the just-appended optimistic
    /// user turn). (Every mutation also cancels the load task via
    /// [`Self::supersede_load_for_mutation`], so this guard is defense in
    /// depth.) Returns whether the load was applied.
    fn apply_loaded_transcript(
        &mut self,
        result: Result<Vec<ChatMessageView>, AppError>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_busy() {
            // An operation raced ahead of this load; its reload wins.
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

    /// Merge a fresh post-tree render list into the transcript, carrying the
    /// **disclosure state** forward by index (we only ever append, so positions
    /// are stable) and attaching just-captured streaming reasoning to its
    /// finalized post. `new_reasoning` is `(action_id, reasoning)` — attached
    /// to the row with that action id (the turn's persisted response), falling
    /// back to the last assistant entry when the id is unknown.
    ///
    /// The reasoning **text** now comes from the DB (a persisted `thinking`
    /// block), so the prior snapshot is only a *fallback* for it: it fills a
    /// row whose reasoning the tree didn't carry (the frame between a stream
    /// ending and its reload landing, or a build whose upstream emitted no
    /// thinking). It must never overwrite what the tree supplied — that would
    /// blank a reloaded space's disclosure with a stale `None`. `expanded`, by
    /// contrast, is pure view state and is always carried forward, so a reload
    /// doesn't collapse an open disclosure under the reader.
    /// Apply a completed turn's outcome: drop its stream buffers, adopt the
    /// space id, and merge the reloaded tree — attaching the reasoning the
    /// turn streamed to the response it produced.
    ///
    /// A **declined** turn (the agent-side decline checkpoint) produced no
    /// post: `response_action_id` is `None`, and `merge_from_db`'s
    /// `None`-action fallback would attach the captured reasoning to the last
    /// assistant message in the transcript — i.e. put this agent's private
    /// thinking under *another* agent's post. So a decline drops the capture
    /// instead. (Rendering the decision itself is a follow-up; `get_space_tree`
    /// keeps only post-bearing action types, so nothing shows today.)
    fn apply_turn_success(
        &mut self,
        seq: u64,
        result: &ChatResult,
        msgs: Result<Vec<ChatMessageView>, AppError>,
        cx: &mut Context<Self>,
    ) {
        let captured = self
            .streams
            .iter()
            .find(|s| s.seq == seq)
            .map(|s| s.response.reasoning.clone());
        self.streams.retain(|s| s.seq != seq);
        self.id = Some(result.space_id.clone());
        match msgs {
            Ok(messages) => {
                let attach = captured
                    .filter(|r| !r.is_empty())
                    .filter(|_| result.declined.is_none())
                    .map(|r| (result.response_action_id.clone(), r));
                self.merge_from_db(messages, attach);
                cx.emit(SpaceEvent::MessagesChanged);
                cx.emit(SpaceEvent::StreamEnded);
            }
            Err(e) => {
                self.transcript = std::mem::take(&mut self.transcript).resolve(Err(e.clone()));
                cx.emit(SpaceEvent::Failed(e));
            }
        }
        // The turn names itself on the way out: its streaming leaf no longer
        // exists, and anything the view deferred onto that leaf has to follow
        // the turn onto the post it wrote (`SpaceEvent::TurnEnded`). Emitted on
        // both arms — a failed *reload* still ended the turn, and a listener
        // that can't find the post simply drops the request.
        cx.emit(SpaceEvent::TurnEnded {
            seq,
            response_action_id: result.response_action_id.clone(),
        });
        cx.notify();
    }

    fn merge_from_db(
        &mut self,
        mut next: Vec<ChatMessageView>,
        new_reasoning: Option<(Option<String>, String)>,
    ) {
        let prior = self.transcript.value().cloned();
        for (idx, entry) in next.iter_mut().enumerate() {
            let prior_entry = prior.as_ref().and_then(|p| p.get(idx));
            let same_position = prior_entry.is_some_and(|p| {
                p.message.role == entry.message.role && p.message.content == entry.message.content
            });
            if same_position {
                if entry.reasoning.is_none() {
                    entry.reasoning = prior_entry.and_then(|p| p.reasoning.clone());
                }
                entry.reasoning_expanded = prior_entry.is_some_and(|p| p.reasoning_expanded);
            }
        }

        if let Some((action_id, reasoning)) = new_reasoning
            && !reasoning.is_empty()
        {
            let target = match action_id {
                Some(aid) => next
                    .iter_mut()
                    .find(|e| e.action_id.as_deref() == Some(aid.as_str())),
                None => next
                    .iter_mut()
                    .rev()
                    .find(|e| e.message.role == "assistant"),
            };
            if let Some(entry) = target {
                entry.reasoning = Some(reasoning);
            }
        }

        self.transcript = Loadable::loaded(next);
    }

    // -- Mutations ---------------------------------------------------------

    /// Prologue shared by every transcript-mutating operation (submit /
    /// post-only / edit / regenerate / ask), called at the moment the
    /// operation is accepted: cancel any in-flight transcript load by dropping
    /// its task. The operation's own post-commit reload is now the
    /// authoritative truth, and a superseded load must never land late around
    /// it — the same stale-fetch class as the Record listing race from PR
    /// #179; the fix is the same replace-cancels idiom
    /// (`crates/eidola-gui/STATE.md` → "Concurrency patterns"). If the
    /// cancelled load was an *initial* one (`Loading`, nothing kept visible),
    /// the cell steps back to `NotLoaded` so no spinner outlives its task
    /// ("every spinner maps to a live task"); a re-fetch over data stays
    /// `Loaded { stale: true }` until the operation's reload resolves it.
    ///
    /// **Cancelling here creates a debt**: the cancelled load may have carried
    /// another writer's change (a bus-driven refresh), so the operation must
    /// re-establish the durable truth at *every* exit — the success arms
    /// reload inline, and every failure arm goes through
    /// [`Self::fail_mutation`] / [`Self::fail_turn`], which restart the load.
    fn supersede_load_for_mutation(&mut self) {
        self.load_task = None;
        if self.transcript.is_loading() {
            self.transcript = Loadable::NotLoaded;
        }
    }

    /// Shared failure completion for the exclusive mutation runner (submit's
    /// post phase / post-only / edit / regenerate): clear the runner slot,
    /// drop any synthetic streams, adopt the id a `ChatFailed` wrapper carries
    /// (blank-space adoption on failure), **restart the transcript load**, and
    /// emit `Failed` with the unwrapped source so the view's error routing
    /// sees the real variant.
    ///
    /// The reload is load-bearing, not defensive: accepting the mutation
    /// cancelled any in-flight transcript load
    /// ([`Self::supersede_load_for_mutation`]) on the promise that the
    /// mutation's own post-commit reload would re-establish the durable
    /// truth. The success arms keep that promise inline; this keeps it on
    /// failure — otherwise a cancelled bus-driven refresh (another writer's
    /// change) is silently lost, and durable rows committed by the failed
    /// mutation itself (whose `Change::Space` the [`Self::on_space_changed`]
    /// guard dropped while the operation was in flight) never render until
    /// the next unrelated invalidation (the codex finding on PR #206). A
    /// pre-persist failure on a blank space has no id, so `load_transcript`
    /// no-ops — nothing durable exists to reload.
    fn fail_mutation(&mut self, e: AppError, cx: &mut Context<Self>) {
        self.streams.clear();
        self.post_runner = None;
        if self.id.is_none()
            && let Some(id) = e.chat_space_id()
        {
            self.id = Some(id.to_string());
        }
        self.load_transcript(cx);
        cx.emit(SpaceEvent::Failed(e.root().clone()));
        cx.notify();
    }

    /// Failure completion for one streaming **turn**: remove that turn's
    /// buffers + runner (sibling turns keep streaming untouched), record the
    /// [`FailedTurn`] so Retry can re-ask the same participant, adopt a
    /// `ChatFailed` id, restart the transcript load (the same cancelled-load
    /// debt as [`Self::fail_mutation`]; while siblings still stream the
    /// restarted load is dropped as stale and their completions re-establish
    /// the truth instead), and emit `Failed` with the unwrapped source.
    fn fail_turn(
        &mut self,
        seq: u64,
        participant_id: Option<String>,
        target_action_id: Option<String>,
        e: AppError,
        cx: &mut Context<Self>,
    ) {
        self.streams.retain(|s| s.seq != seq);
        self.turn_runners.remove(&seq);
        if let (Some(p), Some(t)) = (participant_id, target_action_id) {
            self.failed_turn = Some(FailedTurn {
                participant_id: p,
                target_action_id: t,
            });
        }
        if self.id.is_none()
            && let Some(id) = e.chat_space_id()
        {
            self.id = Some(id.to_string());
        }
        self.load_transcript(cx);
        cx.emit(SpaceEvent::Failed(e.root().clone()));
        cx.notify();
    }

    /// **Post** — save the prompt and drive its notification plan (the
    /// composer CTA). The post itself needs no credential and no model; the
    /// space's participants decide who responds (notify policies), and one
    /// streaming turn per planned responder is driven concurrently via
    /// [`Self::start_turn`]. When the plan pauses at the cascade limit,
    /// [`SpaceEvent::CascadePaused`] surfaces the resumable state instead.
    ///
    /// A mutation while anything is in flight is a no-op (the current UX).
    /// Returns `true` if accepted (a turn was appended), `false` if a no-op
    /// (empty prompt or busy).
    /// `references` are the draft's pending quoted references (specs in
    /// ordinal order, matching the body's `{{ embed N }}` markers); empty for
    /// a plain post.
    pub fn submit(
        &mut self,
        prompt: String,
        reply_to: Option<String>,
        references: Vec<ReferenceSpec>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_busy() {
            return false;
        }
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return false;
        }

        self.last_submitted_references = references.clone();

        // Cancel any in-flight transcript load: the submit's own post-commit
        // reload is now the authoritative truth (the `apply_loaded_transcript`
        // guard also enforces this, but dropping the task frees the slot
        // eagerly).
        self.supersede_load_for_mutation();

        // Append the user's turn locally, **under its reply antecedent** (see
        // `optimistic_user_turn` — this is what keeps a branch reply on its own
        // branch across the save). This mutation is what a submit-vs-load race
        // must not clobber; since the same entity owns the load, the guard
        // drops the stale result.
        let mut messages = self.transcript.value().cloned().unwrap_or_default();
        messages.push(optimistic_user_turn(&prompt, reply_to.as_deref()));
        self.transcript = Loadable::loaded(messages);
        cx.emit(SpaceEvent::MessagesChanged);
        cx.notify();

        let Some(app_core) = self.app_core.clone() else {
            // Stub stores (behavior tests): the local append above has
            // happened; without a backend no plan is computable. Enter a
            // synthetic streaming turn (the observable "a response was
            // requested" state, and the busy guard the stale-load replay
            // asserts) without occupying any runner slot.
            let seq = self.mint_turn_seq();
            self.streams.push(StreamingTurn {
                seq,
                participant_id: None,
                target_action_id: None,
                response: StreamingResponse::default(),
            });
            return true;
        };

        let space_id = self.id.clone();
        let rx = bridge::submit(app_core.clone(), prompt, space_id, reply_to, references);
        self.post_runner = Some(cx.spawn(async move |this, cx| {
            let outcome = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "submit task cancelled".into(),
                })
            });
            match outcome {
                Ok(result) => {
                    let space_id = result.post.space_id.clone();
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
                        this.post_runner = None;
                        match msgs {
                            Ok(messages) => {
                                this.merge_from_db(messages, None);
                                cx.emit(SpaceEvent::MessagesChanged);
                                // StreamEnded drives the registry's blank-space
                                // adoption (it reads id()); the saved post
                                // earned the id whether or not anyone responds.
                                cx.emit(SpaceEvent::StreamEnded);
                            }
                            Err(e) => {
                                this.transcript =
                                    std::mem::take(&mut this.transcript).resolve(Err(e.clone()));
                                cx.emit(SpaceEvent::Failed(e));
                            }
                        }
                        // Drive the plan: one concurrent turn per planned
                        // responder, or surface the paused cascade.
                        match result.plan {
                            NotificationPlan::Turns(turns) => {
                                for t in turns {
                                    this.start_turn(t.participant_id, t.target_action_id, cx);
                                }
                            }
                            NotificationPlan::Paused { depth, limit } => {
                                cx.emit(SpaceEvent::CascadePaused {
                                    depth,
                                    limit,
                                    target_action_id: result.post.action_id.clone(),
                                });
                            }
                        }
                        cx.notify();
                    });
                }
                Err(e) => {
                    let _ = this.update(cx, |this, cx| this.fail_mutation(e, cx));
                }
            }
        }));
        true
    }

    /// Save a post **quietly** — no notification plan, nobody is asked to
    /// respond (`⌘⇧↩` / the ⌥-revealed "Post quietly"). Appends the user's
    /// turn and persists it via `AppCore::post`; on completion the transcript
    /// reloads from the tree and a blank space adopts its new id. Returns
    /// `true` if accepted, `false` if a no-op (empty / busy).
    /// `references`: see [`Self::submit`].
    pub fn post_only(
        &mut self,
        prompt: String,
        reply_to: Option<String>,
        references: Vec<ReferenceSpec>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_busy() {
            return false;
        }
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return false;
        }

        self.last_submitted_references = references.clone();
        self.supersede_load_for_mutation();

        // Optimistically append the user's turn under its reply antecedent (no
        // streaming state — this path requests nothing).
        let mut messages = self.transcript.value().cloned().unwrap_or_default();
        messages.push(optimistic_user_turn(&prompt, reply_to.as_deref()));
        self.transcript = Loadable::loaded(messages);
        cx.emit(SpaceEvent::MessagesChanged);
        cx.notify();

        let Some(app_core) = self.app_core.clone() else {
            // Stub stores (behavior tests): the local append above is the
            // observable effect; no backend to persist to.
            return true;
        };
        let space_id = self.id.clone();
        let rx = bridge::post(app_core.clone(), prompt, space_id, reply_to, references);
        self.post_runner = Some(cx.spawn(async move |this, cx| {
            let outcome = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "post task cancelled".into(),
                })
            });
            Self::finish_reload(this, cx, app_core, outcome.map(|r| r.space_id)).await;
        }));
        true
    }

    /// Commit an inline edit — append a new human generation of `action_id`'s
    /// item via `AppCore::edit_post`, then reload the tree (the edited post
    /// replaces its prior generation in place). No credential, no model call.
    /// `remove_references` names reference **ordinals** to drop from the new
    /// generation (`edit_post_with_removals`; ordinal 0 — the reply edge — is
    /// never removable and is refused core-side). Empty = a plain edit.
    pub fn edit(
        &mut self,
        action_id: String,
        new_prompt: String,
        remove_references: Vec<i64>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_busy() {
            return false;
        }
        let new_prompt = new_prompt.trim().to_string();
        if new_prompt.is_empty() {
            return false;
        }
        self.last_edit_removals = remove_references.clone();
        self.last_edit_text = new_prompt.clone();
        self.supersede_load_for_mutation();
        let Some(app_core) = self.app_core.clone() else {
            return true; // stub: no backend
        };
        let rx = bridge::edit_post(app_core.clone(), action_id, new_prompt, remove_references);
        self.post_runner = Some(cx.spawn(async move |this, cx| {
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
    /// `model` is resolved by the caller from the post's own recorded model
    /// (regenerating re-asks the model that answered), falling back to the
    /// configured default.
    pub fn regenerate_post(
        &mut self,
        action_id: String,
        model: String,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_busy() {
            return false;
        }
        self.last_submitted_model = Some(model.clone());
        self.supersede_load_for_mutation();
        let Some(app_core) = self.app_core.clone() else {
            return true; // stub: no backend
        };
        let rx = bridge::regenerate(app_core.clone(), action_id, model);
        self.post_runner = Some(cx.spawn(async move |this, cx| {
            let outcome = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "regenerate task cancelled".into(),
                })
            });
            Self::finish_reload(this, cx, app_core, outcome.map(|r| r.space_id)).await;
        }));
        true
    }

    /// Shared completion for post-only/edit/regenerate: on success reload the
    /// tree from the resulting space id (adopting it if the space was blank)
    /// and emit `StreamEnded`; on failure [`Self::fail_mutation`] surfaces
    /// `Failed` and restarts the transcript load. Clears the runner slot
    /// either way.
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
                    this.post_runner = None;
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
                let _ = this.update(cx, |this, cx| this.fail_mutation(e, cx));
            }
        }
    }

    // -- Asks (streaming turns) --------------------------------------------

    /// **Explicit ask** — request a streaming response from `participant_id`
    /// to the persisted post `target_action_id` (a separator's Ask, the
    /// cascade-paused "ask to continue", and the failure notice's Retry all
    /// route here). Explicit asks bypass the cascade guard by construction
    /// (`AppCore::respond_stream_as`). Runs as an independent keyed turn, so
    /// asking is legal while other turns still stream; a duplicate ask (same
    /// participant, same target, already in flight) and an ask during the
    /// exclusive mutation are no-ops. Returns the **seq of the turn that
    /// started** — the render key of the streaming leaf the answer will grow
    /// in, so the view can select *that branch* (a new sibling of whatever the
    /// target already replied with) rather than merely the target's path.
    /// `None` when nothing started: a refusal, or a space with no id yet.
    pub fn ask(
        &mut self,
        participant_id: String,
        target_action_id: String,
        cx: &mut Context<Self>,
    ) -> Option<u64> {
        if self.post_runner.is_some() {
            return None;
        }
        let duplicate = self.streams.iter().any(|s| {
            s.participant_id.as_deref() == Some(participant_id.as_str())
                && s.target_action_id.as_deref() == Some(target_action_id.as_str())
        });
        if duplicate {
            return None;
        }
        // Re-asking the *failed turn itself* (same participant **and** same
        // target) is the Retry; clear the recorded failure so `can_retry` reads
        // honestly while it streams. Matching the participant alone would let an
        // explicit ask of P about a *different* post orphan a failure recorded
        // for P about the post that actually failed — so both must match.
        if self.failed_turn.as_ref().is_some_and(|f| {
            f.participant_id == participant_id && f.target_action_id == target_action_id
        }) {
            self.failed_turn = None;
        }
        self.supersede_load_for_mutation();

        if self.app_core.is_none() {
            // Stub stores (behavior tests): the observable effect is the
            // synthetic streaming turn carrying who was asked about what.
            let seq = self.mint_turn_seq();
            self.streams.push(StreamingTurn {
                seq,
                participant_id: Some(participant_id),
                target_action_id: Some(target_action_id),
                response: StreamingResponse::default(),
            });
            cx.emit(SpaceEvent::MessagesChanged);
            cx.notify();
            return Some(seq);
        }
        let seq = self.start_turn(participant_id, target_action_id, cx);
        cx.emit(SpaceEvent::MessagesChanged);
        cx.notify();
        seq
    }

    fn mint_turn_seq(&mut self) -> u64 {
        self.next_turn_seq += 1;
        self.next_turn_seq
    }

    /// Start one streaming response turn (`respond_stream_as`) inside its own
    /// keyed runner slot. Each turn owns its buffers and its completion:
    /// pump deltas → reload the tree (attaching the captured reasoning to the
    /// persisted response) → **re-plan the cascade** on the fresh post
    /// (`plan_notifications`), driving any follow-on turns or surfacing the
    /// paused state. A turn failure routes through [`Self::fail_turn`],
    /// leaving sibling turns untouched.
    fn start_turn(
        &mut self,
        participant_id: String,
        target_action_id: String,
        cx: &mut Context<Self>,
    ) -> Option<u64> {
        let app_core = self.app_core.clone()?;
        let space_id = self.id.clone()?;
        let seq = self.mint_turn_seq();
        self.streams.push(StreamingTurn {
            seq,
            participant_id: Some(participant_id.clone()),
            target_action_id: Some(target_action_id.clone()),
            response: StreamingResponse::default(),
        });

        let (event_rx, done_rx) = bridge::respond_stream_as(
            app_core.clone(),
            space_id.clone(),
            participant_id.clone(),
            target_action_id.clone(),
        );
        let runner = self.spawn_turn_runner(
            seq,
            participant_id,
            target_action_id,
            app_core,
            space_id,
            event_rx,
            done_rx,
            cx,
        );
        self.turn_runners.insert(seq, runner);
        cx.notify();
        Some(seq)
    }

    /// The keyed turn runner: pump this turn's deltas, then finalize —
    /// success reloads the tree and continues the cascade; failure routes
    /// through [`Self::fail_turn`]. The runner's own map entry is removed only
    /// at the very end (removing it earlier would drop — and cancel — the
    /// running task at its next await).
    #[allow(clippy::too_many_arguments)]
    fn spawn_turn_runner(
        &mut self,
        seq: u64,
        participant_id: String,
        target_action_id: String,
        app_core: Arc<AppCore>,
        space_id: String,
        mut event_rx: mpsc::UnboundedReceiver<ChatStreamEvent>,
        done_rx: oneshot::Receiver<Result<ChatResult, AppError>>,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        cx.spawn(async move |this, cx| {
            while let Some(event) = event_rx.recv().await {
                let _ = this.update(cx, |this, cx| {
                    if let Some(s) = this.streams.iter_mut().find(|s| s.seq == seq) {
                        match event {
                            ChatStreamEvent::ReasoningDelta(d) => s.response.reasoning.push_str(&d),
                            ChatStreamEvent::ContentDelta(d) => s.response.content.push_str(&d),
                        }
                        cx.emit(SpaceEvent::StreamDelta);
                        cx.notify();
                    }
                });
            }

            let outcome = done_rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "turn task cancelled".into(),
                })
            });

            match outcome {
                Ok(result) => {
                    let msgs = bridge::get_space_tree(app_core.clone(), result.space_id.clone())
                        .await
                        .unwrap_or_else(|_| {
                            Err(AppError::Internal {
                                message: "fetch space tree task cancelled".into(),
                            })
                        })
                        .map(views_from_nodes);
                    let _ = this.update(cx, |this, cx| {
                        this.apply_turn_success(seq, &result, msgs, cx)
                    });

                    // Continue the cascade on the fresh **post**: the
                    // responding participant's reply may itself notify others
                    // (`all` policies), bounded by the data-derived cascade
                    // guard and filtered by the space's may-decline router
                    // (`plan_notifications` refines). Best-effort — a failed
                    // *plan* read is not a failed turn. A declined turn wrote
                    // no post, so `response_action_id` is `None` and the
                    // cascade simply ends here.
                    if let Some(response_id) = result.response_action_id.clone() {
                        let plan =
                            bridge::plan_notifications(app_core, space_id, response_id.clone())
                                .await
                                .unwrap_or_else(|_| {
                                    Err(AppError::Internal {
                                        message: "plan task cancelled".into(),
                                    })
                                });
                        let _ = this.update(cx, |this, cx| {
                            match plan {
                                Ok(NotificationPlan::Turns(turns)) => {
                                    for t in turns {
                                        this.start_turn(t.participant_id, t.target_action_id, cx);
                                    }
                                }
                                Ok(NotificationPlan::Paused { depth, limit }) => {
                                    cx.emit(SpaceEvent::CascadePaused {
                                        depth,
                                        limit,
                                        target_action_id: response_id,
                                    });
                                }
                                Err(_) => {}
                            }
                            this.turn_runners.remove(&seq);
                            cx.notify();
                        });
                    } else {
                        let _ = this.update(cx, |this, cx| {
                            this.turn_runners.remove(&seq);
                            cx.notify();
                        });
                    }
                }
                Err(e) => {
                    let _ = this.update(cx, |this, cx| {
                        this.fail_turn(seq, Some(participant_id), Some(target_action_id), e, cx)
                    });
                }
            }
        })
    }

    /// Whether the failed turn can be **re-asked**: a failure is recorded and
    /// nothing exclusive is in flight ([`Self::ask`] itself refuses a
    /// duplicate of a turn already streaming). Sibling turns streaming do
    /// *not* block a retry — per-turn recovery is independent by design.
    pub fn can_retry(&self) -> bool {
        self.failed_turn.is_some() && self.post_runner.is_none()
    }

    /// Forget the recorded failed turn (the recovery notice was **explicitly
    /// dismissed**). The recovery notice's lifetime is owned by this record —
    /// it persists across sibling turns finishing until the turn is retried or
    /// the user dismisses it — so ending the recovery clears the record here,
    /// after which [`Self::can_retry`] reads `false`. The saved user post is
    /// untouched (Edit / a fresh ask remain available).
    pub fn clear_failed_turn(&mut self, cx: &mut Context<Self>) {
        if self.failed_turn.take().is_some() {
            cx.notify();
        }
    }

    /// Re-ask the failed turn's participant about the same post — **without**
    /// re-posting anything (the post is already durable; the ask bypasses the
    /// cascade guard). Returns the retried turn's **seq** (so the view can
    /// select the branch its streaming leaf lands on before it renders — PR
    /// #218 review), or `None` if nothing is retryable.
    pub fn retry(&mut self, cx: &mut Context<Self>) -> Option<u64> {
        let failed = self.failed_turn.clone()?;
        self.ask(
            failed.participant_id.clone(),
            failed.target_action_id.clone(),
            cx,
        )
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

    /// Test-only: replace the streaming state with zero or one synthetic
    /// turn (snapshot tests that render a single in-flight response).
    #[doc(hidden)]
    pub fn set_streaming_for_test(
        &mut self,
        streaming: Option<StreamingResponse>,
        cx: &mut Context<Self>,
    ) {
        self.streams.clear();
        if let Some(response) = streaming {
            let seq = self.mint_turn_seq();
            self.streams.push(StreamingTurn {
                seq,
                participant_id: None,
                target_action_id: None,
                response,
            });
        }
        cx.notify();
    }

    /// Test-only: push one synthetic in-flight turn (multi-stream scenes).
    #[doc(hidden)]
    pub fn push_streaming_turn_for_test(
        &mut self,
        participant_id: Option<String>,
        target_action_id: Option<String>,
        response: StreamingResponse,
        cx: &mut Context<Self>,
    ) -> u64 {
        let seq = self.mint_turn_seq();
        self.streams.push(StreamingTurn {
            seq,
            participant_id,
            target_action_id,
            response,
        });
        cx.notify();
        seq
    }

    /// Test-only: push a content delta into turn `seq`'s live buffer and emit
    /// `StreamDelta`, exactly as the real turn runner does. Drives the
    /// two-window sync test (both lenses observe the same entity, so both see
    /// the delta) without a backend.
    #[doc(hidden)]
    pub fn push_content_delta_for_test(&mut self, seq: u64, delta: &str, cx: &mut Context<Self>) {
        if let Some(s) = self.streams.iter_mut().find(|s| s.seq == seq) {
            s.response.content.push_str(delta);
            cx.emit(SpaceEvent::StreamDelta);
            cx.notify();
        }
    }

    /// Test-only: the first in-flight turn's seq (single-stream tests).
    #[doc(hidden)]
    pub fn first_stream_seq_for_test(&self) -> Option<u64> {
        self.streams.first().map(|s| s.seq)
    }

    /// Test-only: arm a never-completing task in the load slot, standing in
    /// for an in-flight transcript load on a stub space (where
    /// `load_transcript` early-returns without a backend). Drives the
    /// mutation-supersedes-load replay tests.
    #[doc(hidden)]
    pub fn arm_load_for_test(&mut self, cx: &mut Context<Self>) {
        self.load_task = Some(cx.spawn(async move |_, _| std::future::pending::<()>().await));
    }

    /// Test-only: occupy the **exclusive mutation slot** with a never-completing
    /// task, standing in for an in-flight save/plan (`submit`/`post_only`/edit/
    /// regenerate) whose `post_runner` is busy but which is **not yet
    /// streaming**. That is the window a second Post must be rejected in (so its
    /// draft is preserved, not consumed-then-dropped) — the composer's
    /// accept-before-consume gate. While armed, `is_busy` is `true` and
    /// `submit`/`post_only` return `false`.
    #[doc(hidden)]
    pub fn arm_post_runner_for_test(&mut self, cx: &mut Context<Self>) {
        self.post_runner = Some(cx.spawn(async move |_, _| std::future::pending::<()>().await));
    }

    /// Test-only: release the exclusive mutation slot armed above, so the next
    /// mutation is accepted.
    #[doc(hidden)]
    pub fn clear_post_runner_for_test(&mut self, cx: &mut Context<Self>) {
        self.post_runner = None;
        cx.notify();
    }

    /// Test-only: complete one streaming turn **successfully** — drop its stream
    /// entry and emit `MessagesChanged` + `StreamEnded`, as a real turn
    /// runner's success arm does (minus the DB reload a stub can't perform,
    /// and minus `TurnEnded`: there is no reloaded post to name).
    /// Drives the sibling-success-keeps-the-failed-turn-notice regression: a
    /// sibling of a fan-out finishing must not hide a still-recorded failed
    /// turn's recovery notice.
    #[doc(hidden)]
    pub fn finish_streaming_turn_for_test(&mut self, seq: u64, cx: &mut Context<Self>) {
        self.streams.retain(|s| s.seq != seq);
        self.turn_runners.remove(&seq);
        cx.emit(SpaceEvent::MessagesChanged);
        cx.emit(SpaceEvent::StreamEnded);
        cx.notify();
    }

    /// Test-only: drive one streaming turn's **success** completion exactly as
    /// the real runner's success arm does ([`Self::apply_turn_success`]),
    /// with the reloaded tree supplied by the caller. `declined` stands in for
    /// a turn that ended at the agent-side decline checkpoint (no post, so
    /// `response_action_id` is `None`). Drives the regression that a decline's
    /// captured reasoning must not be attached to another agent's post.
    #[doc(hidden)]
    pub fn apply_turn_success_for_test(
        &mut self,
        seq: u64,
        nodes: Vec<PostNode>,
        declined: bool,
        cx: &mut Context<Self>,
    ) {
        let result = ChatResult {
            space_id: self.id.clone().unwrap_or_default(),
            content: String::new(),
            model: String::new(),
            input_tokens: None,
            output_tokens: None,
            credits_charged: 0,
            response_action_id: if declined {
                None
            } else {
                nodes.last().map(|n| n.action_id.clone())
            },
            declined: declined.then(|| eidola_app_core::DeclineOutcome {
                reason: "nothing to add".into(),
                action_id: "decision-1".into(),
            }),
        };
        self.apply_turn_success(seq, &result, Ok(views_from_nodes(nodes)), cx);
    }

    /// Test-only: whether a transcript load is in flight (the load slot is
    /// occupied).
    #[doc(hidden)]
    pub fn has_pending_load_for_test(&self) -> bool {
        self.load_task.is_some()
    }

    /// Test-only: simulate completion of a transcript load (the race-replay
    /// test). Returns whether the load was applied — `false` when a mutation
    /// has raced ahead (a turn is streaming), proving a slow initial load that
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

    /// Test-only: drive the exclusive-mutation failure completion exactly as
    /// the post/edit/regenerate runners' error arms do (they delegate to the
    /// same [`Self::fail_mutation`]): adopt the id from a `ChatFailed` wrapper
    /// if the space is still blank, clear synthetic streams, restart the
    /// transcript load, and emit `Failed` with the unwrapped source. Drives
    /// the blank-space id-adoption and failure-restarts-load regressions.
    #[doc(hidden)]
    pub fn apply_chat_failure_for_test(&mut self, error: AppError, cx: &mut Context<Self>) {
        self.fail_mutation(error, cx);
    }

    /// Test-only: drive one turn's failure completion exactly as a real turn
    /// runner's error arm does ([`Self::fail_turn`]): the failed turn's
    /// buffers are dropped (siblings untouched), the participant + target are
    /// recorded for Retry, and `Failed` is emitted with the unwrapped source.
    /// When a synthetic turn matching the pair is in flight its seq is used;
    /// otherwise a fresh seq stands in for an already-collapsed turn.
    #[doc(hidden)]
    pub fn apply_turn_failure_for_test(
        &mut self,
        participant_id: &str,
        target_action_id: &str,
        error: AppError,
        cx: &mut Context<Self>,
    ) {
        let seq = self
            .streams
            .iter()
            .find(|s| {
                s.participant_id.as_deref() == Some(participant_id)
                    && s.target_action_id.as_deref() == Some(target_action_id)
            })
            .map(|s| s.seq)
            .unwrap_or_else(|| self.mint_turn_seq());
        self.fail_turn(
            seq,
            Some(participant_id.to_string()),
            Some(target_action_id.to_string()),
            error,
            cx,
        );
    }

    /// Test-only: emit the cascade-paused event (stub scenes can't compute a
    /// real plan).
    #[doc(hidden)]
    pub fn emit_cascade_paused_for_test(
        &mut self,
        depth: i64,
        limit: i64,
        target_action_id: String,
        cx: &mut Context<Self>,
    ) {
        cx.emit(SpaceEvent::CascadePaused {
            depth,
            limit,
            target_action_id,
        });
        cx.notify();
    }
}
