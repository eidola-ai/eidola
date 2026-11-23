//! `Conversation` — one space's domain entity (the desktop app's `Space`,
//! kept to what a phone needs; `crates/eidola-gui/STATE.md` → Space entities).
//!
//! It owns the transcript (a [`Loadable`] of the space's posts, the
//! database's truth) and what is not in the database yet, drawn over it: the
//! reader's post while it saves ([`Pending`]) and each answer while it streams
//! ([`Turn`]). Each overlay leaves the screen only when a transcript read that
//! contains what it stood for has landed, so nothing blinks out between a
//! commit and the read that shows it, and nothing is shown that was not
//! asked for.
//!
//! **Tasks are fields** — the save in `post_runner` (one at a time), each turn
//! in its own keyed slot (a fan-out runs several, and one's failure never
//! disturbs another), the read in `load_task` (a newer read supersedes it).
//! Cancellation is drop: app-core runs every call to completion whatever
//! happens to the task awaiting it, so a dropped task loses an answer's
//! delivery, never a write.
//!
//! **Honest states**: a refused save gives the reader's words back to the
//! composer ([`ConversationEvent::SubmitFailed`]) and says why; a failed turn
//! keeps a Retry until it is retried or dismissed, unless nothing could lift
//! it; a failed read keeps what was shown and says so; a space whose creation
//! was refused says that instead of offering a composer over nothing.

use std::collections::HashMap;
use std::rc::Rc;

use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatResult, ChatStreamEvent, NotificationPlan, PostNode};
use gpui::{Context, EventEmitter, SharedString, Task};
use tokio::sync::{mpsc, watch};

use super::errors::is_retryable;
use super::loadable::Loadable;

/// Who wrote a post.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Author {
    /// The person holding the phone.
    Reader,
    /// An agent or another participant, by its label.
    Named(SharedString),
}

/// One persisted post, as the transcript shows it.
#[derive(Clone, Debug)]
pub struct Post {
    pub action_id: String,
    /// The structural parent (a reply edge), resolved by app-core to the
    /// current generation; `None` for an unparented row.
    pub parent: Option<String>,
    pub author: Author,
    pub text: SharedString,
    /// The model's reasoning, when it left one (a `thinking` block).
    pub reasoning: Option<SharedString>,
    pub created_at: i64,
    /// The answer stopped at its length limit.
    pub truncated: bool,
}

impl Post {
    fn from_node(node: &PostNode) -> Self {
        let mut text = String::new();
        let mut reasoning = None;
        for block in &node.blocks {
            match (block.block_type.as_str(), block.text.as_deref()) {
                ("text", Some(t)) => {
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(t);
                }
                ("thinking", Some(t)) if !t.trim().is_empty() => {
                    reasoning = Some(SharedString::from(t.to_string()));
                }
                _ => {}
            }
        }
        let author = if node.participant.kind == "human" {
            Author::Reader
        } else {
            Author::Named(node.participant.label.clone().into())
        };
        Self {
            action_id: node.action_id.clone(),
            parent: node.parent_action_id.clone(),
            author,
            text: text.into(),
            reasoning,
            created_at: node.created_at,
            truncated: node.truncated,
        }
    }
}

/// The reader's post while it saves.
#[derive(Clone, Debug)]
pub struct Pending {
    pub text: SharedString,
    pub reply_to: Option<String>,
    /// Known once the save answered; the overlay stands until a read
    /// carrying this post lands.
    pub action_id: Option<String>,
    pub created_at: i64,
}

/// One answer streaming in.
#[derive(Clone, Debug)]
pub struct Turn {
    pub seq: u64,
    pub participant_id: String,
    /// The post it answers.
    pub target: String,
    pub reasoning: String,
    pub content: String,
    /// The turn ended and this is the post it wrote (`Some(None)`: none — the
    /// agent declined); the overlay stands until a read carries it.
    pub landed: Option<Option<String>>,
}

/// Where an entity's space comes from: what it shows before a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Minted by this app a moment ago: it has no posts, so the page is
    /// answered-and-empty from the first frame and nothing is read.
    Created,
    /// An existing space (from the Library): its transcript is read, once its
    /// row is known to stand — after a pending disposal keeps it, if one is
    /// deciding.
    Listed,
}

/// Whether this space's row is known to exist.
#[derive(Clone, Debug)]
pub enum RowState {
    /// Its creation (or a disposal's verdict) is in flight: saves and reads
    /// wait for it.
    Deciding,
    Committed,
    /// It does not exist: its creation was refused, or it was disposed of.
    Gone(AppError),
}

#[derive(Clone, Debug)]
pub enum ConversationEvent {
    /// The reader's post saved; `reply_to` is what it answers.
    Posted {
        action_id: String,
        reply_to: Option<String>,
    },
    /// The save was refused: the words go back to the composer.
    SubmitFailed {
        text: String,
        reply_to: Option<String>,
    },
    /// A streaming answer grew.
    Delta { seq: u64 },
    /// A streamed answer's post has been read back: the turn's overlay gave
    /// way to it (a view carries what it showed open across).
    Landed { seq: u64, action_id: String },
}

pub struct Conversation {
    core: Rc<AppCore>,
    id: String,
    transcript: Loadable<Vec<Post>>,
    /// The change-bus mark the transcript on screen was read under: a read
    /// that began before it describes less than what is shown, so it is not
    /// applied.
    covers_through: u64,
    load_task: Option<Task<()>>,
    row: watch::Sender<RowState>,
    pending: Option<Pending>,
    post_runner: Option<Task<()>>,
    turns: Vec<Turn>,
    turn_runners: HashMap<u64, Task<()>>,
    plan_tasks: HashMap<u64, Task<()>>,
    next_seq: u64,
    /// The turn a Retry re-asks: `(participant, target)`.
    failed_turn: Option<(String, String)>,
    /// What the recovery notice explains.
    error: Option<AppError>,
    cascade_paused: Option<i64>,
    /// Participants' display labels, read with the transcript: a streaming
    /// answer is attributed before its post exists.
    labels: HashMap<String, SharedString>,
}

impl EventEmitter<ConversationEvent> for Conversation {}

impl Conversation {
    /// A conversation whose row is `row`. A space this app just created
    /// starts answered-and-empty; any other starts its read, which waits for
    /// a deciding row to settle (`load` → `wait_for_row`), so a space reopened
    /// while its disposal is pending is read the moment the disposal keeps it.
    pub fn new(
        core: Rc<AppCore>,
        id: String,
        row: RowState,
        origin: Origin,
        cx: &mut Context<Self>,
    ) -> Self {
        let fresh = origin == Origin::Created;
        let (row, _) = watch::channel(row);
        let mut this = Self {
            core,
            id,
            transcript: Loadable::NotLoaded,
            covers_through: 0,
            load_task: None,
            row,
            pending: None,
            post_runner: None,
            turns: Vec::new(),
            turn_runners: HashMap::new(),
            plan_tasks: HashMap::new(),
            next_seq: 0,
            failed_turn: None,
            error: None,
            cascade_paused: None,
            labels: HashMap::new(),
        };
        if fresh {
            this.transcript = Loadable::Loaded {
                value: Vec::new(),
                stale: false,
            };
        } else {
            this.load(cx);
        }
        this
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn transcript(&self) -> &Loadable<Vec<Post>> {
        &self.transcript
    }

    pub fn pending(&self) -> Option<&Pending> {
        self.pending.as_ref()
    }

    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    pub fn error(&self) -> Option<&AppError> {
        self.error.as_ref()
    }

    pub fn can_retry(&self) -> bool {
        self.failed_turn.is_some()
    }

    pub fn cascade_paused(&self) -> Option<i64> {
        self.cascade_paused
    }

    /// A participant's display label, when the last read knew it.
    pub fn label_of(&self, participant_id: &str) -> Option<SharedString> {
        self.labels.get(participant_id).cloned()
    }

    /// The refusal that means this space does not exist, if it does not.
    pub fn gone(&self) -> Option<AppError> {
        match &*self.row.borrow() {
            RowState::Gone(e) => Some(e.clone()),
            _ => None,
        }
    }

    pub fn is_deciding(&self) -> bool {
        matches!(*self.row.borrow(), RowState::Deciding)
    }

    /// Anything of this conversation's own is still in flight: a save, a
    /// streaming turn, or the cascade planning a finished turn's replies.
    /// While it is, the entity must outlive every view of it
    /// (`SpacesStore::left` holds it).
    pub fn is_busy(&self) -> bool {
        self.post_runner.is_some() || !self.turn_runners.is_empty() || !self.plan_tasks.is_empty()
    }

    /// A save is in flight: the composer holds its next one.
    pub fn is_saving(&self) -> bool {
        self.post_runner.is_some()
    }

    /// The row committed (the creation landed, or a disposal kept it).
    pub fn row_committed(&mut self, cx: &mut Context<Self>) {
        self.row.send_replace(RowState::Committed);
        cx.notify();
    }

    /// The row does not exist: everything waiting on it answers with `error`,
    /// and the transcript says so instead of loading.
    pub fn row_is_gone(&mut self, error: AppError, cx: &mut Context<Self>) {
        self.row.send_replace(RowState::Gone(error.clone()));
        self.load_task = None;
        self.transcript = Loadable::Failed { error, prior: None };
        cx.notify();
    }

    /// Read the transcript. A newer read supersedes one in flight (it began
    /// later, so it covers at least as much).
    pub fn load(&mut self, cx: &mut Context<Self>) {
        if matches!(*self.row.borrow(), RowState::Gone(_)) {
            return;
        }
        self.transcript = std::mem::replace(&mut self.transcript, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        let id = self.id.clone();
        let row = self.row.subscribe();
        self.load_task = Some(cx.spawn(async move |this, cx| {
            if wait_for_row(row).await.is_err() {
                return;
            }
            let mark = core.change_seq();
            let result = core.get_space_tree(id.clone()).await;
            // Labels are decoration: a failed read only leaves them as they were.
            let labels = core.list_space_participants(id).await.ok();
            this.update(cx, |this, cx| {
                if let Some(labels) = labels {
                    this.labels = labels
                        .into_iter()
                        .map(|p| (p.id, SharedString::from(p.label)))
                        .collect();
                }
                this.apply_read(mark, result, cx)
            })
            .ok();
        }));
    }

    fn apply_read(
        &mut self,
        mark: u64,
        result: Result<Vec<PostNode>, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.load_task = None;
        match result {
            // A read that began before the one on screen describes less.
            Ok(_) if mark < self.covers_through => {}
            Ok(nodes) => {
                self.covers_through = mark;
                let posts: Vec<Post> = nodes.iter().map(Post::from_node).collect();
                let has = |id: &str| posts.iter().any(|p| p.action_id == id);
                if self
                    .pending
                    .as_ref()
                    .and_then(|p| p.action_id.as_deref())
                    .is_some_and(has)
                {
                    self.pending = None;
                }
                let mut landed = Vec::new();
                self.turns.retain(|t| match &t.landed {
                    None => true,
                    Some(None) => false,
                    Some(Some(id)) if has(id) => {
                        landed.push((t.seq, id.clone()));
                        false
                    }
                    Some(Some(_)) => true,
                });
                for (seq, action_id) in landed {
                    cx.emit(ConversationEvent::Landed { seq, action_id });
                }
                self.transcript = Loadable::Loaded {
                    value: posts,
                    stale: false,
                };
            }
            Err(e) => {
                log::warn!("reading the conversation failed: {e}");
                self.transcript =
                    std::mem::replace(&mut self.transcript, Loadable::NotLoaded).resolve(Err(e));
            }
        }
        cx.notify();
    }

    /// Another writer (or this one) committed to this space.
    pub fn on_space_changed(&mut self, cx: &mut Context<Self>) {
        self.load(cx);
    }

    /// Save the reader's post and start the answers its plan asks for.
    /// Refused (`false`) while another save is in flight or the space does
    /// not exist; the caller keeps the draft then.
    pub fn submit(
        &mut self,
        text: String,
        reply_to: Option<String>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.post_runner.is_some() || matches!(*self.row.borrow(), RowState::Gone(_)) {
            return false;
        }
        self.pending = Some(Pending {
            text: text.clone().into(),
            reply_to: reply_to.clone(),
            action_id: None,
            created_at: now_ms(),
        });
        self.error = None;
        self.failed_turn = None;
        self.cascade_paused = None;
        let core = self.core.clone();
        let id = self.id.clone();
        let row = self.row.subscribe();
        self.post_runner = Some(cx.spawn(async move |this, cx| {
            let result = match wait_for_row(row).await {
                Ok(()) => core.submit(text.clone(), Some(id), reply_to.clone()).await,
                Err(e) => Err(e),
            };
            this.update(cx, |this, cx| {
                this.post_runner = None;
                match result {
                    Ok(submitted) => {
                        let action_id = submitted.post.action_id.clone();
                        if let Some(pending) = &mut this.pending {
                            pending.action_id = Some(action_id.clone());
                        }
                        cx.emit(ConversationEvent::Posted {
                            action_id,
                            reply_to,
                        });
                        this.load(cx);
                        this.follow_plan(submitted.plan, cx);
                    }
                    Err(e) => {
                        this.pending = None;
                        this.error = Some(e);
                        cx.emit(ConversationEvent::SubmitFailed { text, reply_to });
                        // A refused save may still have written something;
                        // the read says what is true.
                        this.load(cx);
                    }
                }
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
        true
    }

    fn follow_plan(&mut self, plan: NotificationPlan, cx: &mut Context<Self>) {
        match plan {
            NotificationPlan::Turns(turns) => {
                for turn in turns {
                    self.start_turn(turn.participant_id, turn.target_action_id, cx);
                }
            }
            NotificationPlan::Paused { depth, .. } => self.cascade_paused = Some(depth),
        }
    }

    /// Ask `participant` to answer `target`, streaming.
    pub fn start_turn(&mut self, participant_id: String, target: String, cx: &mut Context<Self>) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.turns.push(Turn {
            seq,
            participant_id: participant_id.clone(),
            target: target.clone(),
            reasoning: String::new(),
            content: String::new(),
            landed: None,
        });
        let core = self.core.clone();
        let id = self.id.clone();
        let (events, mut deltas) = mpsc::unbounded_channel();
        let call = async move {
            core.respond_stream_as(id, participant_id, target, events)
                .await
        };
        self.turn_runners.insert(
            seq,
            cx.spawn(async move |this, cx| {
                let drain = async {
                    while let Some(event) = deltas.recv().await {
                        if this
                            .update(cx, |this, cx| this.on_delta(seq, event, cx))
                            .is_err()
                        {
                            break;
                        }
                    }
                };
                let (result, ()) = futures_util::future::join(call, drain).await;
                this.update(cx, |this, cx| this.finish_turn(seq, result, cx))
                    .ok();
            }),
        );
        cx.notify();
    }

    fn on_delta(&mut self, seq: u64, event: ChatStreamEvent, cx: &mut Context<Self>) {
        let Some(turn) = self.turns.iter_mut().find(|t| t.seq == seq) else {
            return;
        };
        match event {
            ChatStreamEvent::ReasoningDelta(d) => turn.reasoning.push_str(&d),
            ChatStreamEvent::ContentDelta(d) => turn.content.push_str(&d),
        }
        cx.emit(ConversationEvent::Delta { seq });
        cx.notify();
    }

    fn finish_turn(
        &mut self,
        seq: u64,
        result: Result<ChatResult, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.turn_runners.remove(&seq);
        match result {
            Ok(done) => {
                if let Some(turn) = self.turns.iter_mut().find(|t| t.seq == seq) {
                    turn.landed = Some(done.response_action_id.clone());
                }
                if let Some(response) = done.response_action_id {
                    self.continue_cascade(seq, response, cx);
                }
            }
            Err(e) => {
                let turn = self
                    .turns
                    .iter()
                    .position(|t| t.seq == seq)
                    .map(|i| self.turns.remove(i));
                // A failure nothing could lift ends any recovery standing.
                self.failed_turn = turn
                    .filter(|_| is_retryable(&e))
                    .map(|t| (t.participant_id, t.target));
                self.error = Some(e);
            }
        }
        self.load(cx);
        cx.notify();
    }

    /// Plan the room's next replies to a fresh answer (best-effort: a failed
    /// plan only means the cascade stops here).
    fn continue_cascade(&mut self, seq: u64, response: String, cx: &mut Context<Self>) {
        let core = self.core.clone();
        let id = self.id.clone();
        self.plan_tasks.insert(
            seq,
            cx.spawn(async move |this, cx| {
                let plan = core.plan_notifications(id, response).await;
                this.update(cx, |this, cx| {
                    this.plan_tasks.remove(&seq);
                    if let Ok(plan) = plan {
                        this.follow_plan(plan, cx);
                    }
                    cx.notify();
                })
                .ok();
            }),
        );
    }

    /// Re-ask the participant whose turn failed.
    pub fn retry(&mut self, cx: &mut Context<Self>) {
        if let Some((participant, target)) = self.failed_turn.take() {
            self.error = None;
            self.start_turn(participant, target, cx);
        }
    }

    /// The reader read the notice: the recovery ends with it.
    pub fn dismiss_error(&mut self, cx: &mut Context<Self>) {
        self.error = None;
        self.failed_turn = None;
        cx.notify();
    }

    pub fn dismiss_cascade_notice(&mut self, cx: &mut Context<Self>) {
        self.cascade_paused = None;
        cx.notify();
    }
}

/// Wait until the row is known: `Ok` once committed, the refusal if gone.
async fn wait_for_row(mut row: watch::Receiver<RowState>) -> Result<(), AppError> {
    loop {
        match &*row.borrow_and_update() {
            RowState::Committed => return Ok(()),
            RowState::Gone(e) => return Err(e.clone()),
            RowState::Deciding => {}
        }
        if row.changed().await.is_err() {
            return Err(AppError::Internal {
                message: "the conversation was released".into(),
            });
        }
    }
}

pub(crate) fn now_ms() -> i64 {
    #[cfg(target_family = "wasm")]
    {
        js_sys::Date::now() as i64
    }
    #[cfg(not(target_family = "wasm"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }
}
