//! The turn driver for agent-spawned sub-spaces.
//!
//! Every other conversation in this app is driven from a window: something
//! posts, the consumer plans the notifications over that post, drives one turn
//! per planned responder, and re-plans on each answer until the plan comes back
//! empty or the cascade guard pauses it. A **sub-space** has no window — it has
//! no human in it at all — so without a driver of its own a delegated room
//! receives its brief and then sits there. This module is that driver, and it
//! lives here rather than in the GUI because a delegation is business logic and
//! because the CLI must get the same behaviour without reimplementing it.
//!
//! ## What it drives, and what it will not
//!
//! **Only sub-spaces**, and only while they are live. An ordinary conversation
//! is still the consumer's to drive, exactly as before. The rule is held at the
//! planning door rather than by convention: `Inner::plan_and_refine` answers no
//! turns for a sub-space unless the asker is this driver
//! (`Planner`), so a consumer cannot double-drive a room even by
//! mistake, and "who drives this" has one answer per room rather than one per
//! consumer.
//!
//! **Archival stops it at every step**, because archival is what stops new work
//! everywhere: the room's liveness is re-read before each hop, planning already
//! yields nothing for an archived space, and `prepare_turn` refuses one outright
//! — so a retirement that archives an owner's rooms mid-cascade ends them
//! without a further turn, and without a report about a room somebody closed.
//!
//! ## Where a delegation's state lives: nowhere
//!
//! There is no status column, and the lifecycle is not stored. Two derived
//! facts carry everything:
//!
//! * **Is there work outstanding?** The room's last post has not yet been
//!   quoted back to the parent by its owner (`db::has_reference_from`). That is
//!   true the moment a brief is written, false again once a report goes out, and
//!   true again the moment anyone posts into the room — which is precisely what
//!   "continuation is just posting" means. It reads the same after a restart as
//!   before one, because it reads rows.
//! * **How much has this delegation spent?** The count of turns taken in the
//!   room (`db::turns_taken_in_space`). An in-memory tally would reset every
//!   time the process came back, so the budget would bound a session rather
//!   than a delegation.
//!
//! ## The report
//!
//! A delegation ends in exactly one way from the outside: the owning agent
//! writes a post in the **parent**, quoting the delegated room's last word. The
//! quote is attached mechanically by this driver at the authoring seam
//! (`AttachedReference`) — the model is shown the passage and writes
//! about it, but never chooses what the edge names — and the annotation on that
//! edge says *how* the room ended. Concluding, pausing at the cascade guard,
//! spending the budget and failing a turn all take that same channel, because a
//! delegation that stopped without saying why is worse than one that failed
//! loudly.

use std::collections::HashMap;
use std::sync::Arc;

use crate::changes::{Change, ChangeOrigin, with_origin};
use crate::error::AppError;
use crate::{
    AttachedReference, AttachmentOrigin, ChatStreamEvent, Inner, NotificationPlan, PlannedTurn,
    Planner, ReferenceSpec, ResponseMode, TurnDirective, TurnSelector, db,
};

/// The per-room driver registry: room id → the arm that arrived while its task
/// was running, if one did (see [`Inner::arm_subspace_driver`]).
///
/// **The pending entry is a value, not a flag.** An arm carries a licence (see
/// [`Arm`]), and a room being driven is exactly when the licence that matters
/// most arrives — a wait's own alarm comes due while an unrelated parent event
/// has the room rechecking. Recorded as "something happened", that licence is
/// gone: the next pass runs on the arm the task captured when it started, the
/// alarm was one-shot and its wait is not fresh, so nothing schedules another
/// and the delegation waits until the process restarts.
pub(crate) type DriverRegistry = HashMap<String, Option<Arm>>;

/// Delegated room → the **item** the turn that opened it answers under,
/// process-wide. See [`Inner::note_spawning_answer_item`].
pub(crate) type SpawningAnswerRegistry = Arc<std::sync::Mutex<HashMap<String, String>>>;

/// One room's record of which turn opened it, **kept only once the spawn has
/// committed** — released on drop for the reason a regeneration's claim is: an
/// early `?`, a guard refusal and a panic all release it, and nothing has to
/// remember to.
///
/// The record has to be written *before* the spawning transaction (the spawn's
/// own emissions can arm the driver, so a record written after the room exists
/// leaves a window in which the driver asks and is told nothing) — which means
/// every way that transaction can fail is a way to leave a record naming a room
/// that was never created. Nothing would ever remove it: the driver's own
/// `forget_spawning_answer_item` runs from a walk, and there is no room to
/// walk. That is not a rounding error either — the live-rooms ceiling is a
/// standing refusal, so an owner that has reached it leaks one entry per
/// delegation it goes on attempting.
///
/// `rooms_spawned_here` needs no such guard and says so: it stops recording
/// once the startup sweep has read it, so it is bounded by construction.
pub(crate) struct SpawningAnswerGuard {
    rooms: SpawningAnswerRegistry,
    space_id: String,
    kept: bool,
}

impl SpawningAnswerGuard {
    pub(crate) fn note(rooms: &SpawningAnswerRegistry, space_id: &str, item_id: &str) -> Self {
        rooms
            .lock()
            .expect("spawning answer record poisoned")
            .insert(space_id.to_string(), item_id.to_string());
        Self {
            rooms: rooms.clone(),
            space_id: space_id.to_string(),
            kept: false,
        }
    }

    /// The room exists, so the record is the driver's to drop when the
    /// delegation ends.
    pub(crate) fn keep(mut self) {
        self.kept = true;
    }
}

impl Drop for SpawningAnswerGuard {
    fn drop(&mut self) {
        if self.kept {
            return;
        }
        if let Ok(mut rooms) = self.rooms.lock() {
            rooms.remove(&self.space_id);
        }
    }
}

/// Why a room is being armed, which decides one thing: whether it may wait for
/// its anchor to be answered (see [`Inner::report_delegation`]).
///
/// **The two are told apart by provenance, not by shape**, and the distinction
/// is load-bearing enough to be worth stating twice: the whole of `Sweep`'s
/// licence to stop waiting is the claim that *no turn that could answer this
/// room's anchor is still running*, and the only turn that could is one **this
/// process** made — a spawn happens inside its owner's turn. Anything a
/// **live** process does — including recovering from a bus it fell behind on —
/// is a `Signal`, because a turn it cannot see may be running right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arm {
    /// Something happened that this room might care about: a write the bus
    /// announced, or a re-ask by a process that is already running (lag
    /// recovery). A waiting room keeps waiting.
    Signal,
    /// **The startup sweep, and only for a room this process did not open.** A
    /// process that has just started cannot be waiting on a turn of its own —
    /// so a room it *finds* was opened by some earlier run, whose turns died
    /// with it, and nothing can still be coming. That is what makes it safe,
    /// and the only thing that makes it safe, to stop waiting for an answer
    /// here. A room this process spawned is the exception the claim has to
    /// name rather than survive: it was opened inside a turn that may be
    /// running at this moment, so the sweep arms it as an ordinary
    /// [`Arm::Signal`] and its wait ends the way every other one does — the
    /// answer, or the grace.
    Sweep,
    /// A wait's own alarm, come due (see [`ANCHOR_WAIT_GRACE`]). The answer has
    /// had longer than any turn takes and has not come; the room stops holding
    /// out for it.
    Grace,
}

impl Arm {
    /// The lattice: `Signal < Grace < Sweep`, **strongest wins** when two arms
    /// meet on one room (see [`DriverRegistry`]).
    ///
    /// The ordering is by licence, not by urgency. `Signal` claims nothing —
    /// something happened, a waiting room keeps waiting. `Grace` claims that
    /// *this* wait has outlived any turn that could answer it. `Sweep` claims
    /// that nothing anywhere can be in flight, which is the broadest of the
    /// three and true only of a process that has just started.
    ///
    /// **A licence, once earned, is not taken back by a later signal**, which
    /// is what makes merging a `max` rather than a replacement: an elapsed
    /// clock does not un-elapse because somebody posted in the parent, so a
    /// `Grace` that arrives mid-walk is honoured on the next pass rather than
    /// flattened into the walk's own arm. The order between the two upper
    /// elements is a naming convention and nothing more: [`Arm`] is read in
    /// exactly one place, where the only question asked of it is whether it is
    /// `Signal`.
    fn strength(self) -> u8 {
        match self {
            Self::Signal => 0,
            Self::Grace => 1,
            Self::Sweep => 2,
        }
    }

    /// The stronger of two arms.
    fn merge(self, other: Self) -> Self {
        if other.strength() > self.strength() {
            other
        } else {
            self
        }
    }
}

/// What a report attempt did.
///
/// The difference matters exactly once: a walk that ended by waiting spent no
/// attempt, because it tried nothing (see [`Inner::drive_subspace`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Report {
    /// The report was delivered, or the room turned out to have nothing to
    /// report to (an empty parent, a room archived under us).
    Settled,
    /// The post this delegation was opened from has not been answered yet, so
    /// the report is holding until it is.
    Waiting,
}

/// What one process has done about one room's last word — the meter that stops
/// a failing delegation being retried forever. See [`Inner::claim_walk`].
#[derive(Clone, Debug)]
pub(crate) struct Walk {
    tail: String,
    attempts: u32,
    /// **An ending this room reached and could not deliver.** A walk decides an
    /// outcome and then reports it, and the report can be held back — the owner
    /// has not answered the post the delegation was opened from yet, so the
    /// report has nowhere to sit. The decision is not re-derivable afterwards
    /// (a failed turn leaves nothing durable saying it failed), so it is kept
    /// here, beside the meter. It stays true of the room for exactly as long
    /// as the room's **current last word is among its leaves** — the memory
    /// twin of `db::has_reference_from`'s premise — which is what recognizes a
    /// walk by its own newest driven post (`tail` cannot: it is the word the
    /// walk *started* from, and the walk's own turns move the room past it)
    /// while a stranger's post falls outside the leaves and gets the walk it
    /// is owed. See [`Inner::remembered_ending`].
    ///
    /// It is what lets every wake while the anchor goes unanswered be delivery
    /// only, and a room whose *work* allowance is spent still deliver what
    /// that work already decided — see [`Inner::drive_subspace`].
    decided: Option<(DelegationEnd, Vec<String>)>,
}

/// The per-room walk ledger: room id → what has been tried against its current
/// last word.
pub(crate) type WalkLedger = HashMap<String, Walk>;

/// How many times one process may walk a delegated room from the same last
/// word.
///
/// It is a *retry* bound, not a work bound — the turn budget is that. What it
/// bounds is the arm-fail-arm circuit: a failure writes no post, so nothing
/// about the room changes and the failure's own change event arms it again. A
/// handful of attempts rides out a blip; anything past that is an outage, and
/// an outage is waited out rather than billed against. A restart, or any post
/// from outside, starts the count over.
pub const MAX_ATTEMPTS_PER_TAIL: u32 = 3;

/// How long a failed walk waits before trying again. Short — the meter is only
/// [`MAX_ATTEMPTS_PER_TAIL`] deep, so this is the difference between riding out
/// a blip and hammering an upstream that is away, not a backoff schedule.
const RETRY_PAUSE: std::time::Duration = std::time::Duration::from_millis(250);

/// How many times the owner may retry a report whose generation was
/// withdrawn at persist because a finding or the attach target moved while
/// the model request was in flight. The cascade has already finished; these
/// are extra report turns, not extra walks. Past this the ending is kept and
/// the room waits to be armed again, so a reader who keeps regenerating
/// cannot bill an unbounded number of reports and cannot force a re-cascade.
const MAX_REPORT_ATTEMPTS: u32 = 3;

/// How long a delegation holds out for the post it was opened from to be
/// answered before reporting against that post instead.
///
/// **This is a clock because there is nothing else to read**, and that was
/// established rather than assumed. What the wait is holding out for is the end
/// of a *turn*, and a turn that fails leaves the database exactly as it found
/// it: the only durable trace is an unattached `request` row, which carries no
/// space, no action and no participant, so nothing can be joined back to the
/// post being waited on; and the tool rounds a spawning turn wrote chain off
/// that post whether it later answers or dies, so they say "a turn started" and
/// never "a turn ended". In-process bookkeeping would be precise, but its
/// release cannot be ordered in front of the terminal emissions without
/// threading a guard through every error arm — and a release that lands after
/// the emission recreates the lost wake-up this wait exists to be ordered
/// against.
///
/// So the end is time: the thing being waited on is bounded *work* — at most
/// [`crate::MAX_TURN_ROUNDS`] model round trips — and a wait for work that ends
/// ends by outliving it.
///
/// **What ten minutes is not is a proof.** Nothing here bounds a turn's
/// wall-clock length: no client in this crate sets a request or read timeout, a
/// stream stays open as long as the upstream holds it, and a local engine may
/// generate for as long as it likes. So this is a *policy* about how long a
/// delegation holds out, not a deduction that no turn can still be running —
/// and the doc says so rather than helping itself to a bound the code does not
/// enforce.
///
/// **What it costs when the policy is wrong is bounded and positional.** A
/// legitimately slow spawning turn whose answer lands after the alarm gets a
/// report attached to the anchor instead of beneath that answer: the report is
/// delivered whole (every finding quoted, the ending recorded, the delegation
/// marked reported), the owner's answer lands whole, and the two are *siblings*
/// under the anchor rather than parent and child — the render's spine follows
/// the first reply, so the report takes the spine and the answer indents beside
/// it. Nothing is lost — no post, no edge, no finding, no second spend — and
/// the one further cost is that the reporting model wrote without sight of an
/// answer still being composed. Getting it wrong costs a reader a worse-placed
/// pair of posts, once; refusing to end the wait at all would cost the report
/// entirely, forever, whenever a spawning turn died.
///
/// **And the alarm never overrides an answer that arrived**: every wake
/// re-asks, this one included, so a turn that finishes inside the grace reports
/// beneath its own answer like any other.
///
/// **The alarm is the wait's own**, scheduled when it begins, so termination
/// depends on no other event: an unrelated post in the parent still never
/// spends the wait, and a parent that goes quiet forever still ends it.
pub const ANCHOR_WAIT_GRACE: std::time::Duration = std::time::Duration::from_secs(600);

/// How many delegated rooms this process may be walking at once.
///
/// **The backlog is the case this exists for.** Arming is per room and the
/// registry only stops a room being walked twice — nothing bounded how many
/// rooms walked at all, so a start that finds a previous run's rooms
/// outstanding, or a lag recovery that re-asks the question of every one, put
/// every walk on the runtime at the same instant. Each walk is a chain of
/// *billed* turns: a hundred rooms is a hundred live requests, a hundred
/// credential steps queued against each other, and — on a local backend — a
/// hundred engine loads competing for one pool's memory budget, none of it
/// asked for by anybody who is looking at the app right now.
///
/// **Small on purpose, and the number is not about throughput.** The paying
/// step is already serialized process-wide (`Inner::spend_gate` holds the
/// acquire → spend-proof → flip for one request at a time), so concurrency past
/// a handful buys nothing where it costs money and multiplies everything
/// around it. What the bound has to leave room for is a room that *stalls*: an
/// upstream that has gone away takes its attempts and its pause, and must not
/// hold the queue. Three lets two rooms make progress while one is stuck, and
/// still means a restart's backlog arrives as a trickle rather than a
/// stampede.
///
/// It bounds **execution**, not arming: a room whose turn has not come keeps
/// its registry entry, so arms still merge into it (see [`Arm::merge`]) and the
/// walk that eventually runs runs once, from the room's newest tail.
pub const MAX_CONCURRENT_WALKS: usize = 3;

/// How many turns one delegation may take, across its whole life.
///
/// The spawn guards bound the *roster* — eight seats, three levels deep, eight
/// live rooms per owner — and none of them bounds the work. Every seat is
/// written notify-all, so each answer wakes every other seat: a full room's
/// scheduled work grows with the square of the roster, and until now the only
/// thing that ever stopped it was a human closing the window it was running in.
/// A driver that gives those rooms headless turns has to carry the bound the
/// window used to be.
///
/// The value is read off the same register the other guards use rather than
/// invented: [`crate::MAX_SUBAGENTS_PER_SPAWN`] (8) turns is one answer from
/// every seat, and [`crate::DEFAULT_CASCADE_LIMIT`] (4) is how many consecutive
/// agent replies the cascade guard admits — so 32 is *one full-roster answer at
/// each level the cascade guard already allows*, and a delegation that wants
/// more than that is one nobody is watching. Whichever guard binds first wins;
/// this one is the ceiling that exists even when the room is small enough that
/// the cascade guard never fires.
pub const MAX_DELEGATION_TURNS: i64 = 32;

/// Why a delegated room's turn could not run — a **bounded category**, never
/// the error's own words.
///
/// A failure message can carry an upstream response body, and an upstream is on
/// the other side of a trust boundary: splicing one into the owning agent's
/// prompt would let whatever answered the room write into another model's
/// context, at whatever length it liked. It can equally carry a local path or a
/// database detail, which is nobody's business one conversation over. So the
/// report says which *kind* of thing went wrong and the detail stays where it
/// is useful and contained — the local warning log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelegationFailure {
    /// The model's endpoint could not be reached, or refused the request.
    Upstream,
    /// The turn could not be paid for.
    Funding,
    /// The room's own configuration stopped it — a model that no longer
    /// resolves, a participant that is gone.
    Configuration,
    /// The turn ran and could not be finished. Everything that is neither of
    /// the above, including a local failure with no bearing on the room.
    Unfinished,
}

impl DelegationFailure {
    /// The category an error falls into. Exhaustive over [`AppError`] on
    /// purpose: a new variant is a compile error here rather than a silent
    /// slide into the catch-all.
    fn of(error: &AppError) -> Self {
        match error {
            AppError::Network { .. }
            | AppError::Attestation { .. }
            | AppError::Server { .. }
            | AppError::Credential { .. } => Self::Upstream,
            AppError::NoAccount
            | AppError::InsufficientBalance { .. }
            | AppError::ProvisioningTimeout { .. }
            | AppError::TermsAcceptanceRequired { .. } => Self::Funding,
            AppError::NotConfigured { .. }
            | AppError::Config { .. }
            | AppError::NotAParticipant { .. }
            | AppError::WrongPostKind { .. }
            | AppError::SpawnRefused { .. }
            | AppError::NotJoined { .. }
            | AppError::DrivenConversation { .. }
            | AppError::SpaceArchived { .. } => Self::Configuration,
            // A truncation is the sharpest case of "ran, produced no answer",
            // and a regeneration collision cannot reach a driven room at all
            // (the driver never revises) — both land in the honest catch-all.
            AppError::ResponseTruncated { .. }
            | AppError::RegenerationInFlight { .. }
            | AppError::ToolLoop { .. }
            | AppError::Internal { .. }
            | AppError::Database { .. }
            | AppError::DatabaseInUse { .. }
            | AppError::LocalModel { .. }
            | AppError::Update { .. } => Self::Unfinished,
        }
    }

    fn token(self) -> &'static str {
        match self {
            Self::Upstream => "upstream",
            Self::Funding => "funding",
            Self::Configuration => "configuration",
            Self::Unfinished => "unfinished",
        }
    }

    fn parse(token: &str) -> Option<Self> {
        Some(match token {
            "upstream" => Self::Upstream,
            "funding" => Self::Funding,
            "configuration" => Self::Configuration,
            "unfinished" => Self::Unfinished,
            _ => return None,
        })
    }

    /// What a model reads. English, like every other model-facing string this
    /// crate builds, and built at render time rather than persisted.
    fn describe(self) -> &'static str {
        match self {
            Self::Upstream => "the model it was talking to could not be reached",
            Self::Funding => "the turn could not be paid for",
            Self::Configuration => "something about that conversation's setup stopped it",
            Self::Unfinished => "the turn could not be finished",
        }
    }
}

/// The reserved prefix of a delegation ending written into a reference edge's
/// `annotation`.
///
/// The column holds either a person's note about the passage they quoted or one
/// of these, and nothing else — which is why the prefix has to be something
/// nobody types. [`crate::PostReference`] splits the two apart so no reader can
/// print one where the other belongs.
pub(crate) const DELEGATION_END_PREFIX: &str = "eidola:delegation/";

/// Whether an annotation claims the reserved namespace — the question every
/// door that accepts an annotation **from a caller** asks before writing it.
///
/// The column can say two things and a reader tells them apart by this prefix
/// alone, so a caller allowed to write one would be writing lifecycle state: a
/// note reading `eidola:delegation/concluded` hides itself from every surface
/// that shows a person's note *and* makes a quote report an ending nothing
/// ended. Reserving it at the write is what keeps the parse total — a stored
/// token is one this crate wrote — and it is the whole rule: only a value
/// *starting* here is refused, so a note that mentions the prefix in passing is
/// an ordinary note.
pub(crate) fn is_reserved_annotation(annotation: &str) -> bool {
    annotation.starts_with(DELEGATION_END_PREFIX)
}

/// How a delegated room stopped.
///
/// Every variant is reported, on the same channel, in the same shape — the
/// failure arm is not a quieter version of the success one.
///
/// **It is persisted as a token, never as a sentence.** The ending rides the
/// report's reference edge, and `annotation` is a column a person's own note
/// lives in, so whatever this crate writes there is read as-is in every
/// language — the same rule that keeps a spawned room's *title* a bare name.
/// So the durable form is [`Self::token`] and the prose is built at read time:
/// English for models (which is what every model-facing string here is), and
/// the presentation layer's own words for people, from
/// [`crate::PostReference::delegation_end`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// **The truncation marker rides every ending a reader might act on as if the
/// room's words were whole.** A turn that stopped at its completion ceiling
/// keeps its partial text and the walk goes on — partial text is real text —
/// but three of these four endings would otherwise let a reader take the room's
/// last word for a finished thought: `Concluded` says it ran out of things to
/// say, `Paused` says it can be resumed by posting there, and `BudgetSpent`
/// says an external cap stopped it. Each invites an action (accept the result,
/// resume, raise the budget) that assumes coherent words to build on.
/// `TurnFailed` invites none of them, which is why it alone carries no marker.
///
/// Durably the marker is an **optional trailing `/truncated`** on the arm's own
/// token, so every ending written before it existed still reads as itself.
pub enum DelegationEnd {
    /// Planning returned no turns: nobody's notify policy fired on the room's
    /// last post, which is what a conversation running out of things to say
    /// looks like from here.
    Concluded {
        /// See the type's own note on the marker.
        truncated: bool,
    },
    /// The room hit its own cascade guard. Resumable by posting into it.
    Paused {
        depth: i64,
        limit: i64,
        /// See the type's own note on the marker.
        truncated: bool,
    },
    /// The per-delegation turn budget is spent.
    BudgetSpent {
        limit: i64,
        /// See the type's own note on the marker.
        truncated: bool,
    },
    /// A turn failed, in the bounded sense of [`DelegationFailure`].
    ///
    /// **Deliberately carries no truncation marker.** The marker exists to stop
    /// an ending claiming the room's words were whole, and this one claims the
    /// opposite already: it says the room did not go well and names why. A
    /// reader acts on the failure either way, so adding "and an earlier answer
    /// was cut off" changes nothing they would do and dilutes the reason that
    /// does. See the type's own note.
    TurnFailed { reason: DelegationFailure },
}

impl DelegationEnd {
    /// The failure arm for an error, categorized. The error itself never
    /// travels — see [`DelegationFailure`].
    fn failed(error: &AppError) -> Self {
        Self::TurnFailed {
            reason: DelegationFailure::of(error),
        }
    }

    /// The durable form: locale-neutral, stable, and unmistakable for a
    /// person's note.
    pub fn token(&self) -> String {
        let body = match self {
            Self::Concluded { .. } => "concluded".to_string(),
            Self::Paused { depth, limit, .. } => format!("paused/{depth}/{limit}"),
            Self::BudgetSpent { limit, .. } => format!("budget/{limit}"),
            Self::TurnFailed { reason } => format!("failed/{}", reason.token()),
        };
        // One tail for every arm that has one, appended after the arm's own
        // fields, so the marker never has to be threaded through their shapes.
        let tail = if self.truncated() { "/truncated" } else { "" };
        format!("{DELEGATION_END_PREFIX}{body}{tail}")
    }

    /// Whether this ending rests on an answer cut off at its completion
    /// ceiling. Always `false` for [`Self::TurnFailed`] — see the type's note.
    pub fn truncated(&self) -> bool {
        match self {
            Self::Concluded { truncated }
            | Self::Paused { truncated, .. }
            | Self::BudgetSpent { truncated, .. } => *truncated,
            Self::TurnFailed { .. } => false,
        }
    }

    /// Read an annotation back as an ending, or `None` when it is not one —
    /// which is every annotation a person wrote, and also any token a future
    /// version writes that this one does not understand. Both degrade to "a
    /// note", which is the safe direction: a reader shows prose it cannot
    /// interpret rather than claiming an ending it guessed at.
    pub fn parse(annotation: &str) -> Option<Self> {
        let rest = annotation.strip_prefix(DELEGATION_END_PREFIX)?;
        // **The marker comes off first, so every arm parses the shape it always
        // did.** It is an optional tail, so a token written before it existed
        // still reads as itself; anything else trailing falls through to the
        // exhaustive match below and degrades to "a note", which is the safe
        // direction.
        let mut parts: Vec<&str> = rest.split('/').collect();
        let truncated = parts.len() > 1 && parts.last() == Some(&"truncated");
        if truncated {
            parts.pop();
        }
        match parts.as_slice() {
            ["concluded"] => Some(Self::Concluded { truncated }),
            ["paused", depth, limit] => Some(Self::Paused {
                depth: depth.parse().ok()?,
                limit: limit.parse().ok()?,
                truncated,
            }),
            ["budget", limit] => Some(Self::BudgetSpent {
                limit: limit.parse().ok()?,
                truncated,
            }),
            // A failure carries no marker, so one on the wire is not a token
            // this version wrote — and not one it should guess at either.
            ["failed", reason] if !truncated => Some(Self::TurnFailed {
                reason: DelegationFailure::parse(reason)?,
            }),
            _ => None,
        }
    }

    /// What a model reads where a person's annotation would go — built here,
    /// never stored.
    pub(crate) fn describe(&self) -> String {
        let base = match self {
            Self::Concluded { .. } => "the delegated conversation ran to a stop".to_string(),
            Self::Paused { depth, limit, .. } => format!(
                "the delegated conversation reached its reply limit ({depth} replies in a row, \
                 limit {limit}) and can be resumed by posting there"
            ),
            Self::BudgetSpent { limit, .. } => format!(
                "the delegated conversation used all {limit} of the turns it is allowed and was \
                 stopped there"
            ),
            Self::TurnFailed { reason } => format!(
                "the delegated conversation stopped because {}",
                reason.describe()
            ),
        };
        // One clause for every arm that carries the marker — this is
        // model-facing English in a single language, so it composes safely
        // where the reader-facing strings deliberately do not (a translator
        // needs the whole sentence; see the GUI's footnote messages).
        if self.truncated() {
            format!("{base}, and an answer in it reached its length limit and stops mid-thought")
        } else {
            base
        }
    }
}

/// What a reference edge's `annotation` says to a **model**: a delegation
/// ending in this crate's own English, or the person's note as they wrote it.
///
/// One function, called wherever a [`crate::ReferenceEntry`] is built, so the
/// two cannot drift into showing a reader a raw token.
pub(crate) fn annotation_for_model(annotation: Option<&str>) -> Option<String> {
    let annotation = annotation?;
    Some(match DelegationEnd::parse(annotation) {
        Some(end) => end.describe(),
        None => annotation.to_string(),
    })
}

impl Inner {
    /// Whether this process drives sub-spaces at all — see
    /// [`crate::AppCore::start_subspace_driver`].
    pub(crate) fn subspace_driver_running(&self) -> bool {
        self.subspace_driver_started
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Record that **this process** is opening `space_id`, so a startup sweep
    /// cannot later mistake it for a room left behind by an earlier run (see
    /// [`Inner::rearm_live_subspaces`]).
    ///
    /// Called **before** the spawning transaction, not after it: a record
    /// written afterwards leaves a window in which the room is visible to the
    /// sweep's enumeration and unrecorded, which is precisely the state the
    /// record exists to rule out. Marking early can only over-mark — a refused
    /// spawn leaves an id naming no room, which nothing ever asks about.
    ///
    /// **It stops recording once the sweep has been**, which is what keeps this
    /// from growing for the life of the process: [`Arm::Sweep`] is issued by
    /// one call, once, and after it has read the record no answer this could
    /// give would ever be consulted again. `None` is that state, and it is the
    /// type saying so.
    pub(crate) fn note_room_spawned_here(&self, space_id: &str) {
        if let Some(rooms) = self
            .rooms_spawned_here
            .lock()
            .expect("spawn record poisoned")
            .as_mut()
        {
            rooms.insert(space_id.to_string());
        }
    }

    /// Record the **item** the turn that is opening `space_id` will write its
    /// answer under — the answer this room's report attaches beneath.
    ///
    /// **Why an item and not an action.** A turn's rounds chain off the post it
    /// answers, never off its own inference, because a capped or budget-stopped
    /// turn writes no inference at all — so at the moment `delegate` runs there
    /// is no answer id to record. What does exist is the item that answer will
    /// be written under: `prepare_turn` mints it before the first request, and
    /// a regeneration reuses the item it revises. That makes it exactly "which
    /// turn asked for this room", which is the question `last_reply_by_participant`
    /// cannot answer: its watermark rules out answers that predate the room,
    /// and the case left over is an answer by the same owner to the same post
    /// that commits *after* the room opened and belongs to a different turn —
    /// a second explicit ask, or a regeneration running beside a reply.
    ///
    /// **In memory, deliberately, and that is not a gap.** The record is only
    /// ever consulted while the turn that made it could still be running, and a
    /// turn cannot outlive its process: after a restart the spawning turn is
    /// gone, no answer of its item will ever appear, and the room's wait ends
    /// the way it already did — at [`Arm::Sweep`], against the anchor. So the
    /// durable rule stays the fallback, and this refines it exactly where a
    /// refinement is meaningful. It is the same lifetime argument the sweep's
    /// own licence rests on.
    /// Kept unconditionally — the caller is a test seam standing in for a
    /// spawn that has already committed. The **spawn door** takes
    /// [`SpawningAnswerGuard::note`] instead, so a refusal cannot leave a
    /// record behind.
    pub(crate) fn note_spawning_answer_item(&self, space_id: &str, item_id: &str) {
        SpawningAnswerGuard::note(&self.spawning_answer_items, space_id, item_id).keep();
    }

    /// The registry itself, for the spawn door's guard.
    pub(crate) fn spawning_answers(&self) -> &SpawningAnswerRegistry {
        &self.spawning_answer_items
    }

    /// The item the turn that opened `space_id` answers under, if this process
    /// opened it — see [`Inner::note_spawning_answer_item`].
    fn spawning_answer_item(&self, space_id: &str) -> Option<String> {
        self.spawning_answer_items
            .lock()
            .expect("spawning answer record poisoned")
            .get(space_id)
            .cloned()
    }

    /// Drop the record for a delegation that is over — from the terminal exits
    /// `drive_subspace` takes, **and from [`Inner::close_rooms`]**, which is
    /// the archived room's only clearing: an archived room is not a live
    /// delegated room, so the supervisor calls it ordinary and never arms it,
    /// and the walk that would have cleared this is the walk that no longer
    /// runs. Between them the map is bounded by the rooms this process has
    /// open rather than by everything it ever opened.
    pub(crate) fn forget_spawning_answer_item(&self, space_id: &str) {
        self.spawning_answer_items
            .lock()
            .expect("spawning answer record poisoned")
            .remove(space_id);
    }

    /// The rooms this process opened, and the end of the record — see
    /// [`Inner::note_room_spawned_here`].
    fn take_rooms_spawned_here(&self) -> std::collections::HashSet<String> {
        self.rooms_spawned_here
            .lock()
            .expect("spawn record poisoned")
            .take()
            .unwrap_or_default()
    }

    /// Give `space_id` a driver if it is a room this driver owns and does not
    /// already have one.
    ///
    /// Idempotent, cheap, and safe to call for any space id — the caller is the
    /// change bus, which raises `Change::Space` for every post in every
    /// conversation, so most calls are about rooms this has nothing to do with.
    ///
    /// **A room already being driven records the arm rather than dropping it,
    /// and records it whole.** The driver walks the posts it planned; a post
    /// that arrives from somewhere else while it is walking — a human asking a
    /// question in a room they are watching — is not on that walk, and
    /// forgetting it would leave the answer unanswered until something else
    /// woke the room. So the second arm is kept for the running task to pick
    /// up, and the walk simply starts again from the room's new tail.
    ///
    /// It is kept as the [`Arm`] it is, merged strongest-wins, because an arm
    /// is a licence rather than a nudge: a wait's own alarm can come due while
    /// the room is being rechecked for an unrelated reason, and that alarm is
    /// one-shot. Flattened into "something happened", it is simply lost — the
    /// next pass waits again on the arm the task started with, the wait is not
    /// fresh so nothing schedules a replacement, and the delegation holds out
    /// for an answer that is never coming until the process restarts.
    pub(crate) fn arm_subspace_driver(self: &Arc<Self>, space_id: &str, arm: Arm) {
        if !self.subspace_driver_running() {
            return;
        }
        // No runtime (a synchronous unit test) ⇒ nothing to spawn onto.
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        {
            let mut running = self.subspace_drivers.lock().expect("driver map poisoned");
            if let Some(pending) = running.get_mut(space_id) {
                *pending = Some(match *pending {
                    Some(waiting) => waiting.merge(arm),
                    None => arm,
                });
                return;
            }
            running.insert(space_id.to_string(), None);
        }
        let inner = self.clone();
        let space_id = space_id.to_string();
        tokio::spawn(async move {
            let mut arm = arm;
            loop {
                // **The queue is here, behind the registry claim and in front
                // of the walk** ([`MAX_CONCURRENT_WALKS`]). Claiming the room
                // first is what keeps this a queue of *rooms* rather than of
                // tasks: a room waiting for its turn still holds its entry, so
                // everything that happens meanwhile merges into its pending
                // arm and the walk that finally runs runs once, from the
                // newest tail. The permit is per pass, so a room retrying
                // after a failure gives it up over its pause rather than
                // holding the queue closed while it waits.
                let permit = inner.walk_permits.clone().acquire_owned().await;
                if permit.is_err() {
                    return; // the semaphore is closed — the process is going away
                }
                // Everything this task commits is unattended by construction:
                // no consumer call is outstanding, so a window that drops the
                // invalidation while it is busy loses it for good.
                let result = with_origin(
                    ChangeOrigin::Unattended,
                    inner.drive_subspace(space_id.clone(), arm),
                )
                .await;
                drop(permit);
                // **A walk that failed arms itself.** Most failures re-arm the
                // room for free, because the turn that failed emitted a change
                // into it — but the ones that matter most do not: a room whose
                // plan was empty from the start, one that paused on its brief,
                // one whose budget was already spent. Those drive nothing, so
                // nothing is written into the room, so nothing announces
                // anything, and a report that failed there would leave the
                // delegation unreported with no event left in the world to pick
                // it up until the process restarted. Retrying here makes every
                // failure take the same path; `claim_walk` is what bounds it,
                // and a failure changes nothing about the room, so the meter it
                // reads is the same one every time round and runs out.
                let failed = result.is_err();
                let waiting = matches!(result, Ok(Report::Waiting));
                if let Err(ref e) = result {
                    eprintln!(
                        "warning: the delegated conversation {space_id} could not be driven: {e}"
                    );
                }
                {
                    let mut running = inner.subspace_drivers.lock().expect("driver map poisoned");
                    match running.get_mut(&space_id) {
                        // Another pass, on the strongest licence in play for
                        // *this* wait: what arrived while this one ran, merged
                        // with the one it ran under (see [`Arm::merge`]). A
                        // retry after a failure keeps its own arm for the same
                        // reason — the failure did not take back what the walk
                        // was allowed to conclude. A **successful delivery**
                        // spends that licence: a Grace queued after the answer
                        // was found would otherwise skip the ten-minute wait
                        // of a later continuation. Pending work still runs,
                        // but as an ordinary signal.
                        Some(pending) if pending.is_some() || failed => {
                            if let Some(next) = pending.take() {
                                arm = if failed || waiting {
                                    arm.merge(next)
                                } else {
                                    Arm::Signal
                                };
                            }
                        }
                        _ => {
                            running.remove(&space_id);
                            return;
                        }
                    }
                }
                if failed {
                    // A pause between attempts, so a room whose upstream is
                    // briefly away is retried rather than hammered. Short,
                    // because the meter is only three deep.
                    tokio::time::sleep(RETRY_PAUSE).await;
                }
            }
        });
    }

    /// Arm every live sub-space. `arm` says on whose authority — see [`Arm`];
    /// only a process that has just started may pass [`Arm::Sweep`].
    ///
    /// A process that comes back mid-delegation has to pick the room up, and
    /// "mid-delegation" is not a thing that was written down anywhere: each
    /// armed driver decides for itself, from the rows, whether there is
    /// anything outstanding (see [`Inner::drive_subspace`]), and retires
    /// immediately when there is not. So this arms broadly and lets the one
    /// definition of outstanding work do the deciding, rather than keeping a
    /// second, subtly different copy of it here.
    ///
    /// **A sweep does not claim a room this process opened.** `Arm::Sweep`'s
    /// whole licence is that no turn which could answer these rooms' anchors is
    /// still running, and the one turn that could is one this process made — a
    /// spawn happens *inside* its owner's turn. The enumeration is a query, and
    /// a spawn committing before it is in its answer, so the licence would
    /// otherwise be claimed for exactly the room it is false about, and the
    /// corrective `Signal` from that spawn's own change event could not take it
    /// back (the merge is strongest-wins, correctly). So this asks what this
    /// process spawned and arms those as ordinary signals.
    ///
    /// **The two reads are ordered, and the order is the whole of it**: the
    /// rooms are enumerated *first* and the spawn record is taken *after*, so a
    /// spawn racing the enumeration is either invisible to it (and armed by its
    /// own change event) or already in the record when the record is read. The
    /// record is written before the spawning transaction rather than after it,
    /// for the same reason a space's stamp precedes the write it describes — a
    /// mark that lands after its row leaves a window in which the row is
    /// visible and unmarked, and here that window is the hazard itself.
    ///
    /// A failed enumeration is retried rather than swallowed: this pass is the
    /// only recovery a pre-existing room has — nothing else will ever raise a
    /// signal for it — so giving up on an error leaves such a room at its
    /// brief for the life of the process. The spawn record is consumed only by
    /// an enumeration that succeeded, so a retry finds it intact and still
    /// growing, and the ordering above holds of the attempt that finally
    /// answers. The caller sits out each pause, which is safe where blocking
    /// usually is not: the supervisor's bus traffic surfaces as lag if it
    /// overflows the wait, and lag recovery re-asks this very question of
    /// every live room.
    pub(crate) async fn rearm_live_subspaces(self: &Arc<Self>, arm: Arm) {
        loop {
            let rooms = async {
                let conn = self.db_conn().await?;
                db::live_subspaces(&conn).await
            }
            .await;
            #[cfg(feature = "test-support")]
            let rooms = match self.enumeration_faults.fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |n| n.checked_sub(1),
            ) {
                Ok(_) => Err(crate::AppError::Config {
                    message: "test: the enumeration is made to fail".into(),
                }),
                Err(_) => rooms,
            };
            match rooms {
                Ok(rooms) => {
                    // Read after an enumeration that **succeeded** — after,
                    // because that order is the sweep's licence (above), and
                    // only on success so a failed attempt cannot consume the
                    // record a retry still needs. Until then it goes on
                    // growing: the record ends when the sweep has been, and a
                    // sweep that could not enumerate has not. Only the sweep
                    // reads it — every other arm already claims nothing.
                    let spawned_here = match arm {
                        Arm::Sweep => self.take_rooms_spawned_here(),
                        _ => std::collections::HashSet::new(),
                    };
                    for room in rooms {
                        let arm = if spawned_here.contains(&room.id) {
                            Arm::Signal
                        } else {
                            arm
                        };
                        self.arm_subspace_driver(&room.id, arm);
                    }
                    return;
                }
                Err(e) => {
                    eprintln!(
                        "warning: delegated conversations could not be enumerated \
                         (will retry): {e}"
                    );
                    tokio::time::sleep(RETRY_PAUSE).await;
                }
            }
        }
    }

    /// Claim the right to walk `space_id` from `tail`, or refuse.
    ///
    /// **This is what stops a driver paying for the same failure forever.** A
    /// driven turn that fails writes no post, so the room's last word is
    /// unchanged and its delegation is still outstanding — and the failure's own
    /// `Change::Space` arms the room again. With a dead upstream both the turn
    /// and the report fail, nothing is ever marked, and the arm-fail-arm circuit
    /// bills a request every time round. Neither the turn budget nor the cascade
    /// guard closes it: both count *posts*, and a failure produces none.
    ///
    /// So the meter here is attempts, keyed on the room's last word: a walk from
    /// a tail this process has already walked [`MAX_ATTEMPTS_PER_TAIL`] times is
    /// refused. An **external** post is a different tail and starts fresh, which
    /// is the whole discrimination the loop needed — a failure changes nothing
    /// and a stranger's post changes exactly this. In-memory rather than durable
    /// on purpose: a new process is a new chance, and the state it is protecting
    /// against is a running one's.
    fn claim_walk(&self, space_id: &str, tail: &str) -> bool {
        let mut walks = self.subspace_walks.lock().expect("walk map poisoned");
        match walks.get_mut(space_id) {
            Some(walk) if walk.tail == tail => {
                if walk.attempts >= MAX_ATTEMPTS_PER_TAIL {
                    return false;
                }
                walk.attempts += 1;
            }
            _ => {
                walks.insert(
                    space_id.to_string(),
                    Walk {
                        tail: tail.to_string(),
                        attempts: 1,
                        decided: None,
                    },
                );
            }
        }
        true
    }

    /// Keep an ending this walk decided but could not deliver, against the last
    /// word it was decided about (see [`Walk::decided`]).
    fn remember_ending(&self, space_id: &str, tail: &str, end: DelegationEnd, leaves: &[String]) {
        let mut walks = self.subspace_walks.lock().expect("walk map poisoned");
        if let Some(walk) = walks.get_mut(space_id)
            && walk.tail == tail
        {
            walk.decided = Some((end, leaves.to_vec()));
        }
    }

    /// The ending this room already reached, if one is waiting to be delivered
    /// and is still true of the room — that is, if `tail` (the room's current
    /// last word) is **among the ending's own leaves**. That containment is
    /// the memory twin of [`db::has_reference_from`]'s premise: the walk's
    /// leaves are every word it accounted for, its own newest driven post
    /// included, so a room nothing has touched since is recognized whether or
    /// not the walk moved it — and a stranger's post is in nobody's leaves,
    /// so it falls through to the walk it is owed.
    fn remembered_ending(
        &self,
        space_id: &str,
        tail: &str,
    ) -> Option<(DelegationEnd, Vec<String>)> {
        let walks = self.subspace_walks.lock().expect("walk map poisoned");
        walks.get(space_id).and_then(|walk| {
            walk.decided
                .as_ref()
                .filter(|(_, leaves)| leaves.iter().any(|leaf| leaf == tail))
                .cloned()
        })
    }

    /// Drop a remembered ending: it was delivered, or the attempt to deliver it
    /// failed. **A failed delivery forgets**, which is what bounds delivery to
    /// the work that earned it: the next pass finds nothing to deliver, and
    /// what it may do instead is bounded by the work meter.
    fn forget_ending(&self, space_id: &str) {
        let mut walks = self.subspace_walks.lock().expect("walk map poisoned");
        if let Some(walk) = walks.get_mut(space_id) {
            walk.decided = None;
        }
    }

    /// Give back the attempt a walk claimed, when it turns out not to have been
    /// one. Only a walk that ended by waiting does this — everything else
    /// either changed the room's last word (a fresh count) or genuinely tried.
    fn release_walk(&self, space_id: &str, tail: &str) {
        let mut walks = self.subspace_walks.lock().expect("walk map poisoned");
        if let Some(walk) = walks.get_mut(space_id)
            && walk.tail == tail
        {
            walk.attempts = walk.attempts.saturating_sub(1);
        }
    }

    /// Drive one delegated room to a stop, then report it to its parent.
    ///
    /// Returns without doing anything at all when the room is not one this
    /// driver owns, is archived, or has already had its last word reported —
    /// which is what makes arming cheap enough to do from a signal every post
    /// raises.
    pub(crate) async fn drive_subspace(
        &self,
        space_id: String,
        arm: Arm,
    ) -> Result<Report, AppError> {
        let conn = self.db_conn().await?;
        // **Taken before the reads it divides, and taken from the rows.** This
        // is the line between "already in front of me" and "arrived while I was
        // working", and the reads below are several `await`s wide — a post
        // committing inside them would otherwise fall between the two: too late
        // for the tail this walk starts from, too early for the refill's
        // window, and its own change event spent on a walk already in progress.
        //
        // It is a `rowid` high-water mark rather than a clock because a clock
        // could not answer it (see [`db::action_watermark`]): every writer here
        // samples `now_ms()` *above* its own transaction, so a post's
        // `created_at` can predate this line while its commit lands after it —
        // and a boundary drawn on that timestamp misses exactly the writes that
        // raced it. Missing one is not a delay but a loss: the walk drives on,
        // a newer answer becomes the room's last word, the report settles the
        // room on that word, and the post nobody served sits in a room that
        // reads as reported.
        let since_row = db::action_watermark(&conn).await?;
        // **A room that is done stops being waited on.** An anchor wait is a
        // standing registration against the parent, and every post there wakes
        // every room registered under it — so a room that can never run again
        // and stays registered makes the parent's every post pay for it, for as
        // long as the process lives. Each terminal exit below clears it.
        let Some(sub) = db::subspace(&conn, &space_id).await? else {
            self.end_anchor_wait(&space_id);
            self.forget_spawning_answer_item(&space_id);
            return Ok(Report::Settled); // not a delegated room
        };
        if sub.archived_at.is_some() {
            self.end_anchor_wait(&space_id);
            self.forget_spawning_answer_item(&space_id);
            return Ok(Report::Settled); // archival stops new work, here as everywhere
        }
        let Some(tail) = db::last_action_in_space(&conn, &space_id).await? else {
            self.end_anchor_wait(&space_id);
            self.forget_spawning_answer_item(&space_id);
            return Ok(Report::Settled); // a room with no posts — unreachable, a brief opens every one
        };
        // The entry window is a caller-controlled pause. A turn must not hold
        // a connection across an await it does not own — the same rule as
        // inference — and a test that writes during the pause needs the lock.
        drop(conn);
        #[cfg(feature = "test-support")]
        self.pause_in_entry_window().await;
        // The whole of "is there anything to do here", asked of the rows. A
        // reported tail is a delegation whose last word the parent already has;
        // anything posted since makes a new tail, and the answer flips back.
        let conn = self.db_conn().await?;
        if db::has_reference_from(
            &conn,
            &sub.parent_space_id,
            &sub.owner_participant_id,
            &tail,
        )
        .await?
        {
            self.end_anchor_wait(&space_id);
            return Ok(Report::Settled);
        }
        drop(conn);
        // **An ending already decided against this last word is delivered, not
        // re-derived.** A walk that ends while its anchor is unanswered keeps
        // what it decided (see [`Walk::decided`]), and a wait hands back the
        // attempt it claimed — so a wake while the answer is pending would
        // otherwise claim afresh and re-run the cascade over an unchanged
        // tail: a router inference per wake where the room routes, a
        // nondeterministic chance of scheduling helpers again, and a
        // re-derived ending whose leaves are just the tail, overwriting the
        // branch tips the walk that did the work collected. The lookup answers
        // only while the room's current last word is among the decided
        // ending's leaves — nothing has landed the walk did not account for,
        // the same premise `has_reference_from` reads durably — so the ending
        // is still true of the room, and the pass is delivery only: no plan,
        // no turn, no re-derivation — the one report the delegation has owed
        // all along.
        //
        // A decided **failure** is the exception, while the meter has room: it
        // is provisional rather than final — a blip has to be able to heal —
        // so it takes the claim path below and the walk is retried. What makes
        // it final is the meter refusing, and the refused walk then delivers
        // it: the ending the spent work decided, which the arm the anchor's
        // answer raises must meet rather than the meter (the notification the
        // meter exists to protect, refused by the meter itself, was the
        // alternative). Delivery cannot loop, because a failed delivery
        // forgets the ending, and whatever the next wake does — retry while
        // the meter has room, nothing once it is spent — is bounded by the
        // meter like all work.
        //
        // The claim is asked *after* the outstanding check, so a settled room
        // costs no attempt and a room that is genuinely stuck cannot spend
        // forever.
        let remembered = self.remembered_ending(&space_id, &tail);
        let decided_and_final = remembered
            .as_ref()
            .is_some_and(|(end, _)| !matches!(end, DelegationEnd::TurnFailed { .. }));
        let claimed = !decided_and_final && self.claim_walk(&space_id, &tail);
        let walked = if decided_and_final {
            remembered.expect("checked just above")
        } else if claimed {
            // Boxed, like every other await on the turn path:
            // `run_turn_stream`'s state machine is the largest in the crate,
            // and stacking this walk's frame on top of it overflows a worker
            // stack.
            match Box::pin(self.cascade_subspace(&space_id, tail.clone(), since_row)).await? {
                Some(walked) => walked,
                None => {
                    self.end_anchor_wait(&space_id);
                    return Ok(Report::Settled); // the room closed under us; nothing to report
                }
            }
        } else if let Some(decided) = remembered {
            decided
        } else {
            eprintln!(
                "warning: the delegated conversation {space_id} has been tried \
                 {MAX_ATTEMPTS_PER_TAIL} times against the same last post and is being \
                 left alone; a post there, or a restart, will pick it up again"
            );
            // **A spent meter with nothing remembered is a terminal exit.**
            // The wait is registered before the anchor is asked, so a lookup
            // that then errors leaves a registration standing and decides
            // nothing. Once the meter refuses, every later arm hits this same
            // refuse — keeping the wait would make every post in the parent
            // pay for a walk that can only no-op, until the grace alarm. A
            // post in the room, or a restart, starts a fresh meter; neither
            // needs this wait. A decided failure still keeps its wait (the
            // branch above): that ending is still owed.
            self.end_anchor_wait(&space_id);
            return Ok(Report::Settled);
        };
        let (end, leaves) = walked;
        // **A wait tried nothing — but a failure before the wait tried
        // something.** The meter bounds retries of work that failed, and a walk
        // that ended by waiting for the post it was delegated from failed at
        // nothing and left the room's last word unchanged, so charging it an
        // attempt would spend the room's whole allowance on three unrelated
        // posts in the parent and then refuse the walk the real answer finally
        // arms. That reasoning is about the *wait*, and it stops being true the
        // moment the walk in front of it broke a turn: a failed turn with an
        // unanswered anchor ends in exactly the same `Waiting`, and giving the
        // attempt back there hands a dead upstream an unbounded retry — the
        // one circuit `claim_walk` exists to close, reopened by the one exit
        // that gives its claim away. So the release is for a wait that follows
        // no failure; a failure keeps its claim, and the meter binds on it.
        let failed_a_turn = matches!(end, DelegationEnd::TurnFailed { .. });
        let outcome = Box::pin(self.report_delegation(&sub, end, leaves.clone(), arm)).await;
        match outcome {
            // Held back: keep the ending, so every wake until the anchor is
            // answered is delivery only — whether or not the meter has room
            // for more work by then. A delivery pass stores nothing: what it
            // delivered is the stored ending, untouched.
            Ok(Report::Waiting) => {
                if claimed {
                    self.remember_ending(&space_id, &tail, end, &leaves);
                    if !failed_a_turn {
                        self.release_walk(&space_id, &tail);
                    }
                }
            }
            // Delivered, or there was nothing to deliver to. Either way this
            // ending is done with.
            // Settled: the report is delivered, or there was nothing to
            // deliver it to. Either way this delegation is over, so the
            // record of which turn opened it can go with the ending.
            Ok(Report::Settled) => {
                self.forget_ending(&space_id);
                self.forget_spawning_answer_item(&space_id);
            }
            // The delivery itself failed. Forgetting is what keeps a decided
            // ending from being retried past the work that earned it.
            Err(_) => self.forget_ending(&space_id),
        }
        outcome
    }

    /// Plan one hop of a walk. A thin name so the cascade can match on the
    /// `Result` without the `?` that used to skip [`Self::stop_walk`].
    async fn plan_walk(&self, space_id: &str, post: &str) -> Result<NotificationPlan, AppError> {
        self.plan_and_refine(space_id, post, Planner::Driver).await
    }

    /// The plan → drive → re-plan walk, ending at the first terminal outcome.
    ///
    /// Returns the outcome **and every branch tip the walk followed** — see
    /// [`finish`]. Those posts, not whatever the room's tail happens to be by
    /// the time the report is written, are what the report quotes: a post
    /// landing in between would otherwise be reported as if the walk had
    /// considered it, and would then read as already-reported when its own walk
    /// came round, so the room would go quiet holding an unanswered post.
    /// Quoting what the walk actually reached leaves such a post unreported,
    /// which is exactly what makes its arrival arm the room again and get it the
    /// walk it is owed.
    ///
    /// **`since_row` is the caller's**, taken before the reads it divides: the
    /// boundary between what was already there and what arrived mid-walk has to
    /// sit in front of every read, or a post committing inside them belongs to
    /// neither. It is a commit-ordered `rowid` mark rather than a timestamp —
    /// see [`db::action_watermark`] for why a clock cannot draw this line.
    ///
    /// `Ok(None)` means the room stopped being drivable while we were in it (an
    /// archival landed), which is deliberately **not** an outcome to report: the
    /// room was closed on purpose, and a report about it would be a message
    /// nobody asked for about work nobody wants continued.
    async fn cascade_subspace(
        &self,
        space_id: &str,
        tail: String,
        since_row: i64,
    ) -> Result<Option<(DelegationEnd, Vec<String>)>, AppError> {
        // **The tips of everything this walk followed, not one of them.** A
        // delegated room fans out: several helpers answer the brief, each
        // branch runs down to a post nothing follows, and every one of those is
        // a finding. Quoting whichever happened to be written last would report
        // one helper and silently drop the rest — and the room would then read
        // as reported, so nothing would ever go back for them.
        let mut leaves: Vec<String> = Vec::new();
        // The first branch to reach the room's reply limit. **A pause is a
        // branch's, not the walk's**: the guard is derived per post, so one
        // thread of the room running out of replies says nothing about its
        // siblings — abandoning them there would drop their findings exactly
        // the way a single witness did.
        let mut paused: Option<(i64, i64)> = None;
        // Whether any turn in this walk stopped at its **completion ceiling**
        // with partial text. The walk goes on — a partial answer is real text
        // and the room may well have more to say about it — but it changes what
        // the ending is allowed to claim: `Concluded` says the room ran out of
        // things to say, and a room resting on an answer that stopped
        // mid-thought did not. Walk-wide rather than per-branch, because the
        // report speaks for the whole room.
        let mut truncated_any = false;
        // Every post this walk has planned off. It is what tells the walk's own
        // work apart from a stranger's, and there is nothing durable that could:
        // a post the driver planned and a post nobody has looked at are the same
        // row, so the only honest record of "I have served this" is the memory
        // of having done it.
        let mut served: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Posts whose replies have not been planned yet. A fan-out puts several
        // answers on it at once; taking the newest first walks each thread of
        // the room down before starting the next, which is the order a reader
        // watching the transcript would expect.
        let mut frontier: Vec<String> = vec![tail.clone()];
        // **Planning is work**, even when it drives nothing. A router that
        // selects nobody bills an inference and writes no row, so `taken`
        // would stay still while a burst of watched-room posts paid for an
        // unbounded number of planning calls. Empty and paused hops in this
        // walk share the ceiling with persisted turns (`taken + planned`);
        // hops that drove a turn are already in `taken`. The empty-hop count
        // is the walk's, because a restart sees only the tail and a reported
        // budget-stop settles the room.
        let mut planned: i64 = 0;
        #[cfg(feature = "test-support")]
        let mut paused_once = false;
        loop {
            let Some(queued) = frontier.pop() else {
                // **Nothing planned is not the same as nothing outstanding.** A
                // post can land in the room while the walk is walking — a person
                // asking a question in a room they are watching — and it is on
                // nobody's frontier: the walk never saw it, and re-deriving "the
                // tail" would not find it either, because a driven turn has very
                // likely written something newer since. Refilling from what
                // actually arrived is what gets that post its turn; without it
                // the report would name the driven tail, the room would read as
                // reported, and the question would sit there answered by nobody.
                let conn = self.db_conn().await?;
                let arrived = db::posts_in_space_since(&conn, space_id, since_row).await?;
                drop(conn);
                // **Pushed newest-first so they come off oldest-first.** The
                // frontier is a stack — taking the newest first is what walks
                // one thread of a fan-out down before starting the next — and
                // a refill is not a fan-out but a stretch of conversation,
                // which reads forwards. Reversed here, the batch is taken up
                // in the order it was written, which is what the read that
                // produced it already promises.
                frontier.extend(arrived.into_iter().rev().filter(|id| !served.contains(id)));
                if frontier.is_empty() {
                    let end = match paused {
                        Some((depth, limit)) => DelegationEnd::Paused {
                            depth,
                            limit,
                            truncated: truncated_any,
                        },
                        None => DelegationEnd::Concluded {
                            truncated: truncated_any,
                        },
                    };
                    let conn = self.db_conn().await?;
                    let quoted = Self::visible_fallback(&conn, space_id, &tail).await?;
                    drop(conn);
                    return Ok(Some((end, finish(leaves, &frontier, quoted.as_deref()))));
                }
                continue;
            };
            let conn = self.db_conn().await?;
            let live = db::is_live_subspace(&conn, space_id).await?;
            let taken = db::turns_taken_in_space(&conn, space_id).await?;
            let post = db::visible_tip_of_action(&conn, &queued).await?;
            drop(conn);
            if !live {
                return Ok(None);
            }
            // **The queued id is a generation, not an item.** An edit or
            // regeneration that lands while a sibling branch is walking leaves
            // the old id on the frontier; planning it would bill replies
            // against wording the transcript has hidden, and the refill would
            // then also return the replacement, so both wordings get a turn.
            // The item's visible tip is the one post; no visible tip is
            // nothing to plan (a failed regeneration's hidden `error`).
            let Some(post) = post else {
                // The queued generation is done with: refill must not hand
                // the same hidden tip back on the next empty-frontier pass.
                served.insert(queued);
                continue;
            };
            if !served.insert(post.clone()) {
                continue;
            }
            // The budget is checked before the plan, not after it: planning a
            // room whose budget is spent may cost a router inference, and the
            // turns it returned could not be driven anyway. The two meters
            // share the ceiling — a room one turn short of it has one hop
            // left, not a fresh 32 empty router calls.
            if taken.saturating_add(planned) >= MAX_DELEGATION_TURNS {
                return self
                    .stop_walk(
                        space_id,
                        DelegationEnd::BudgetSpent {
                            limit: MAX_DELEGATION_TURNS,
                            truncated: truncated_any,
                        },
                        with(leaves, post),
                        &frontier,
                        &tail,
                        since_row,
                        &served,
                    )
                    .await;
            }
            let turns = match self.plan_walk(space_id, &post).await {
                Ok(NotificationPlan::Paused { depth, limit }) => {
                    planned += 1;
                    paused.get_or_insert((depth, limit));
                    leaves.push(post.clone());
                    continue;
                }
                Ok(NotificationPlan::Turns(turns)) => turns,
                Err(AppError::SpaceArchived { .. }) => return Ok(None),
                Err(e) => {
                    // Same door as a driven turn that failed: the error's own
                    // words stop here, the report carries a category, and the
                    // room is not left outstanding with nothing to arm it.
                    eprintln!(
                        "warning: planning in the delegated conversation {space_id} failed: {e}"
                    );
                    return self
                        .stop_walk(
                            space_id,
                            DelegationEnd::failed(&e),
                            with(leaves, post),
                            &frontier,
                            &tail,
                            since_row,
                            &served,
                        )
                        .await;
                }
            };
            // A hop that persists a turn is already in `taken`. Counting it
            // here as well would spend the ceiling twice on the common path.
            if turns.is_empty() {
                planned += 1;
            }
            // **Nothing follows this one, so it is a finding** — decided
            // after the turns run, not from the plan's shape. A post whose
            // plan comes back empty is where a branch of the room stopped
            // having anything to add; a post whose every planned turn
            // *declines* is in exactly the same position — each decline wrote
            // a decision, not a post, so nothing follows it and nothing will
            // re-plan from it — and a plan-shaped test left such a post
            // neither leaf nor parent: the branch the report silently
            // dropped, or, as the room's newest word, the tip whose absence
            // from the quoted set kept the room outstanding for another
            // billed round of the same declines.
            let mut answered = false;
            for turn in turns {
                let conn = self.db_conn().await?;
                let taken = db::turns_taken_in_space(&conn, space_id).await?;
                // **The target is a generation, not an item.** A fan-out
                // shares one `target_action_id` across several responders and
                // drives them in sequence; an edit or regeneration of that
                // post after the first reply would otherwise leave the rest
                // answering wording the transcript has hidden, and the refill
                // would then plan the replacement too.
                let target = db::visible_tip_of_action(&conn, &turn.target_action_id).await?;
                drop(conn);
                if taken.saturating_add(planned) >= MAX_DELEGATION_TURNS {
                    return self
                        .stop_walk(
                            space_id,
                            DelegationEnd::BudgetSpent {
                                limit: MAX_DELEGATION_TURNS,
                                truncated: truncated_any,
                            },
                            with(leaves, post),
                            &frontier,
                            &tail,
                            since_row,
                            &served,
                        )
                        .await;
                }
                let Some(target) = target else {
                    continue;
                };
                let turn = PlannedTurn {
                    target_action_id: target,
                    ..turn
                };
                match Box::pin(self.drive_planned_turn(space_id, &turn)).await {
                    // A turn that wrote a post is a post to re-plan from, and
                    // the room's newest word; a turn that declined wrote a
                    // decision, which is not something anyone replies to.
                    Ok(Some((post_action_id, was_truncated))) => {
                        answered = true;
                        truncated_any |= was_truncated;
                        frontier.push(post_action_id);
                        #[cfg(feature = "test-support")]
                        if !std::mem::replace(&mut paused_once, true) {
                            self.pause_in_cascade_window().await;
                        }
                    }
                    Ok(None) => {}
                    Err(AppError::SpaceArchived { .. }) => return Ok(None),
                    Err(e) => {
                        // The error's own words stop here — the report carries a
                        // category, never a message (see [`DelegationFailure`]).
                        eprintln!(
                            "warning: a turn in the delegated conversation {space_id} failed: {e}"
                        );
                        return self
                            .stop_walk(
                                space_id,
                                DelegationEnd::failed(&e),
                                with(leaves, post),
                                &frontier,
                                &tail,
                                since_row,
                                &served,
                            )
                            .await;
                    }
                }
            }
            if !answered {
                leaves.push(post.clone());
            }
        }
    }

    /// Close a walk that is stopping **while it still had somewhere to go** —
    /// the turn budget spent, or a turn that failed.
    ///
    /// **The refill belongs to every ending, not to the concluding one.** A
    /// post can land in the room while the walk is walking, and the walk that
    /// stops early never reaches the frontier-empty branch where those
    /// arrivals are collected — so it reported the driven tail, the room read
    /// as reported on that word, and the arrival sat there unserved *and*
    /// unquoted, in a room nothing would walk again. The refill is therefore
    /// asked here too, and asked the same way: what has arrived since the walk
    /// opened, minus what the walk served itself.
    ///
    /// **What differs is where the arrivals go, and it follows from the
    /// ending.** Concluding, the room may still take turns, so an arrival goes
    /// on the *frontier* and gets one. Here the room may not: the budget is
    /// spent or a turn just failed, and driving anything more is the thing the
    /// ending exists to stop. So an arrival becomes a **leaf** — quoted into
    /// the report like every other tip the walk reached, unanswered but not
    /// lost. That is the honest reading of a spent budget: *the room may take
    /// no more turns, and the report still carries what arrived*, so the owner
    /// learns of the question and can open a new delegation for it, rather than
    /// the room retiring with it invisible.
    #[allow(clippy::too_many_arguments)]
    async fn stop_walk(
        &self,
        space_id: &str,
        end: DelegationEnd,
        leaves: Vec<String>,
        frontier: &[String],
        tail: &str,
        since_row: i64,
        served: &std::collections::HashSet<String>,
    ) -> Result<Option<(DelegationEnd, Vec<String>)>, AppError> {
        let conn = self.db_conn().await?;
        let arrived = db::posts_in_space_since(&conn, space_id, since_row).await?;
        // Leftover frontier ids are generations, same as a pop: quote the
        // item's visible tip, or skip a hidden one, rather than naming wording
        // the transcript has replaced.
        let mut unwalked: Vec<String> = Vec::new();
        for id in frontier
            .iter()
            .chain(arrived.iter().filter(|id| !served.contains(*id)))
        {
            let Some(tip) = db::visible_tip_of_action(&conn, id).await? else {
                continue;
            };
            if !served.contains(&tip) && !unwalked.contains(&tip) {
                unwalked.push(tip);
            }
        }
        let quoted = Self::visible_fallback(&conn, space_id, tail).await?;
        drop(conn);
        // The frontier first — those were next in line — then what arrived,
        // oldest first, which is the newest end of the room.
        Ok(Some((end, finish(leaves, &unwalked, quoted.as_deref()))))
    }

    /// The starting tail as a reader can still see it, or the room's current
    /// last word if that item is gone. A failed regeneration that lands after
    /// the tail was snapshotted leaves a hidden `error` tip; quoting the
    /// snapshot would name wording the transcript has replaced, and settlement
    /// reads the newest *visible* last word, which is no longer that id.
    async fn visible_fallback(
        conn: &turso::Connection,
        space_id: &str,
        entry_tail: &str,
    ) -> Result<Option<String>, AppError> {
        if let Some(tip) = db::visible_tip_of_action(conn, entry_tail).await? {
            return Ok(Some(tip));
        }
        db::last_action_in_space(conn, space_id).await
    }

    /// Each finding as a reader can still see it. An edit or regeneration that
    /// lands after the walk collected the id is one wording; a hidden tip is
    /// skipped. An empty set falls back the way [`finish`] does.
    async fn visible_quoted_leaves(
        conn: &turso::Connection,
        space_id: &str,
        leaves: &[String],
    ) -> Result<Vec<String>, AppError> {
        let mut quoted: Vec<String> = Vec::new();
        for id in leaves {
            let Some(tip) = db::visible_tip_of_action(conn, id).await? else {
                continue;
            };
            if !quoted.contains(&tip) {
                quoted.push(tip);
            }
        }
        if quoted.is_empty() {
            let fallback = match leaves.first() {
                Some(id) => Self::visible_fallback(conn, space_id, id).await?,
                None => db::last_action_in_space(conn, space_id).await?,
            };
            if let Some(tip) = fallback {
                quoted.push(tip);
            }
        }
        Ok(quoted)
    }

    /// One driven turn — the same door `respond_stream_as` takes, minus a
    /// consumer to stream to.
    ///
    /// The stream still runs: the transports differ only in transport, and
    /// taking the blocking twin here would mean a delegated room's turns went
    /// through a different code path from every other turn in the app. The
    /// events go into a channel nobody reads, and the room's *posts* reach any
    /// window that happens to be open on it through the ordinary `Change::Space`
    /// refresh, which is all a reader watching a delegation needs.
    async fn drive_planned_turn(
        &self,
        space_id: &str,
        turn: &PlannedTurn,
    ) -> Result<Option<(String, bool)>, AppError> {
        // **The receiver is dropped before the turn runs, not held.** Nothing
        // here watches a delegated room stream; keeping the handle alive would
        // buffer every delta of every turn for the length of the walk, for a
        // reader that does not exist. The transport treats a closed channel as
        // "nobody is listening" (`let _ = sender.send(..)` on every event), so
        // dropping it is the honest way to say so.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        drop(rx);
        let result = Box::pin(self.run_turn_stream(
            space_id,
            TurnSelector::Participant(turn.participant_id.clone()),
            &turn.target_action_id,
            ResponseMode::Reply,
            None,
            &TurnDirective::default(),
            tx,
        ))
        .await?;
        // The post **and whether it stopped at the ceiling**. There is no
        // reader here to show a marker to, which is exactly why the fact has to
        // travel: the only surface a delegated room ever gets is its report,
        // and an ending that claims the room ran to a stop is a claim about an
        // answer that stopped mid-thought. (A turn cut off before writing *any*
        // answer never reaches here — that is `AppError::ResponseTruncated`,
        // which stops the walk as a failure like any other.)
        Ok(result.response_action_id.map(|id| (id, result.truncated)))
    }

    /// Deliver the owner's report into the parent conversation.
    ///
    /// A turn for the owning agent, replying **on the branch the delegation was
    /// opened from**, carrying every finding the walk reached as a quoted
    /// reference the driver attaches itself, at ordinals `1..=N`.
    ///
    /// Every refusal the parent can raise is the parent's to raise: an archived
    /// parent refuses the turn at the same gate every other turn meets, an
    /// owner who has left the parent is refused by participant resolution, and
    /// a funding failure fails like any other turn. None of them is special-
    /// cased here, and the room's last word stays unreported, so a later run
    /// tries again.
    async fn report_delegation(
        &self,
        sub: &db::SubspaceRow,
        end: DelegationEnd,
        leaves: Vec<String>,
        arm: Arm,
    ) -> Result<Report, AppError> {
        for _attempt in 0..MAX_REPORT_ATTEMPTS {
            let conn = self.db_conn().await?;
            // **The room is re-read one last time before anything acts.** An
            // archival can land during the walk — a retirement archives every room
            // its agent owned — and a report about a conversation somebody closed
            // is a message nobody asked for about work nobody wants continued. One
            // read, the same shape as every other liveness gate here.
            if !db::is_live_subspace(&conn, &sub.id).await? {
                self.end_anchor_wait(&sub.id);
                return Ok(Report::Settled);
            }
            // The anchor window is a caller-controlled pause. A turn must not hold
            // a connection across an await it does not own, and a test that writes
            // a finding during the pause needs the lock.
            drop(conn);

            // Where it attaches. The anchor is the post in the parent this
            // delegation was opened from, captured at the spawn because only the
            // caller knew it; the report belongs on *that* branch, beneath the
            // owner's own answer there.
            let target = match sub.parent_action_id.as_deref() {
                Some(anchor) => {
                    // **Registered before the question is asked, not after it.**
                    // The answer can commit between the two, and its
                    // `Change::Space` is the only thing that would ever wake this
                    // room again: asking first and registering second leaves a
                    // window in which that change finds nothing registered, and the
                    // registration that follows then waits for a wake-up already
                    // gone by. Registering first closes it by ordering — a commit
                    // is either early enough for the query to see it or late enough
                    // for the entry to be there — at the cost of an entry to take
                    // back out on every path that does not wait, which is what
                    // `end_anchor_wait` is for.
                    self.begin_anchor_wait(sub);
                    #[cfg(feature = "test-support")]
                    self.pause_in_anchor_window().await;
                    #[cfg(feature = "test-support")]
                    if self
                        .anchor_lookup_faults
                        .fetch_update(
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                            |n| n.checked_sub(1),
                        )
                        .is_ok()
                    {
                        return Err(AppError::Config {
                            message: "test: the anchor lookup is made to fail".into(),
                        });
                    }
                    let conn = self.db_conn().await?;
                    // **Only an answer newer than this room.** A spawn that
                    // named an anchor happened inside the owner's turn (the
                    // `space.parent_action_id` column says so: it is written
                    // only by the spawn door, which is reached from inside a
                    // turn), so the answer this report belongs under is the one
                    // that turn has yet to persist. An answer of the same owner
                    // to the same anchor that is *already* there is a different
                    // answer — the generation a regeneration is replacing, or
                    // an earlier reply to the same post — and accepting it ends
                    // the wait against the wrong word while the right one is
                    // still on the wire. The room's brief is that line and
                    // needs no new state to record it (see
                    // [`db::subspace_opened_at_row`]). A room whose brief
                    // somehow cannot be read leaves the line unset, which is
                    // the pre-existing rule rather than a refusal: a delegation
                    // must still be able to report.
                    // **The turn's own answer, when this process knows which
                    // turn it was.** The watermark below rules out answers that
                    // predate the room; what it cannot rule out is an answer by
                    // the same owner to the same post that commits *after* the
                    // room opened and belongs to a different turn — a second
                    // explicit ask, or a regeneration running beside a reply,
                    // neither of which app-core serializes. The item the
                    // spawning turn will answer under is the one thing that
                    // tells them apart (see
                    // [`Inner::note_spawning_answer_item`]), so where it is
                    // known the question is asked of that item alone: no
                    // visible post of it yet means *this* turn has not
                    // answered, whatever else has landed on the anchor.
                    let opened_at = db::subspace_opened_at_row(&conn, &sub.id).await?;
                    let answered = match self.spawning_answer_item(&sub.id) {
                        // The item names the turn; the watermark names the
                        // generation. A regeneration's item is the one it is
                        // revising, whose visible post until the turn lands is
                        // the answer being replaced — so both rules apply.
                        Some(item) => {
                            db::visible_post_of_item(&conn, &sub.parent_space_id, &item, opened_at)
                                .await?
                        }
                        // No turn identity: a direct caller, or a room this
                        // process did not open. The durable rule stands alone.
                        None => {
                            db::last_reply_by_participant(
                                &conn,
                                &sub.parent_space_id,
                                &sub.owner_participant_id,
                                anchor,
                                opened_at,
                            )
                            .await?
                        }
                    };
                    match answered {
                        Some(answer) => {
                            self.end_anchor_wait(&sub.id);
                            Some(answer)
                        }
                        // **The owner has not answered the anchor yet**, which is
                        // not an ordinary absence: a spawn happens inside the
                        // owner's turn, so this state means that turn is still in
                        // flight (or failed). Reporting now would make the report
                        // the first reply to the anchor and the owner's own answer
                        // an indented branch off it — durably the wrong way round.
                        // So the delegation stays outstanding and the room waits for
                        // its parent to change, which the answer (or the failure)
                        // will do.
                        //
                        // **It goes on waiting until the answer is actually
                        // there.** A wake-up is not evidence of one: the parent is
                        // an ordinary conversation and anything at all can move it,
                        // so a wait that spent itself on the first wake would attach
                        // to the anchor because somebody said something unrelated,
                        // and the answer — arriving a moment later — would become
                        // the report's sibling instead of its parent. Every wake
                        // re-asks, and only the answer ends the wait.
                        //
                        // **Termination is a licence, and a signal is not one.**
                        // Waiting costs nothing — no turn, no spend, no row — so an
                        // indefinite one is not a leak, it is a room correctly
                        // declining to guess. What it must not be is permanent, so
                        // two arms end it and both claim something a wake-up does
                        // not: this wait's own alarm (`Arm::Grace`), set when it
                        // began and honoured whenever it arrives — including on a
                        // room that happens to be mid-walk, which is why the
                        // pending arm is a value — and the startup sweep behind it
                        // for a process that died inside the grace. The case where
                        // the answer never comes at all is a spawning
                        // turn that died; that process cannot outlive its own crash,
                        // and the next start's sweep arms with `Arm::Sweep`, which
                        // never waits — at that moment nothing can still be in
                        // flight, so the answer either exists or never will, and the
                        // anchor is then the right attachment because there is
                        // nothing for the report to sit beneath.
                        None if arm == Arm::Signal => return Ok(Report::Waiting),
                        None => {
                            self.end_anchor_wait(&sub.id);
                            Some(anchor.to_string())
                        }
                    }
                }
                // A spawn that named no anchor — a direct API caller with no turn
                // behind it. The owner's own last word is the best available guess
                // and the conversation's tail is the fallback behind that; both are
                // honest, and neither is as good as an anchor.
                None => {
                    let conn = self.db_conn().await?;
                    match db::last_post_by_participant(
                        &conn,
                        &sub.parent_space_id,
                        &sub.owner_participant_id,
                    )
                    .await?
                    {
                        Some(id) => Some(id),
                        None => db::last_action_in_space(&conn, &sub.parent_space_id).await?,
                    }
                }
            };
            let Some(target) = target else {
                self.end_anchor_wait(&sub.id);
                return Ok(Report::Settled); // an empty parent has nothing to reply to
            };

            // **The findings are generations, remapped immediately before the
            // turn.** An edit or regeneration that lands after the walk collected
            // them — including during the anchor wait above — would otherwise
            // quote wording the transcript has hidden, and settlement reads the
            // visible last word, which is no longer that id. Persist does not
            // rewrite the edges: a regen while the model request is in flight
            // would then attach a footnote to wording the report never saw. The
            // generation is withdrawn instead, and this loop retries against the
            // visible tips. Ordinary human quotes are not remapped at persist:
            // those name a concrete generation on purpose.
            let conn = self.db_conn().await?;
            let quoted = Self::visible_quoted_leaves(&conn, &sub.id, &leaves).await?;
            let mut attached: Vec<AttachedReference> = Vec::with_capacity(quoted.len());
            for (i, leaf) in quoted.iter().enumerate() {
                let (content_block_id, range_start, range_end) =
                    match db::first_quotable_block(&conn, leaf).await? {
                        Some((block_id, text)) if !text.is_empty() => {
                            (Some(block_id), Some(0), Some(text.len() as i64))
                        }
                        // No text to quote still gets an edge — a pointer to the
                        // post rather than a quote of it, which is what a
                        // range-less reference means everywhere.
                        _ => (None, None, None),
                    };
                // Named the way every other reference read names an author: the
                // label that post's **own space** gives them, with no liveness
                // filter — so an agent retired between writing the finding and its
                // being reported is still named, rather than degrading to an
                // anonymous post from somewhere else.
                let author_label = db::post_author_label(&conn, leaf).await?;
                attached.push(AttachedReference {
                    ordinal: (i + 1) as i64,
                    origin: AttachmentOrigin::Authored,
                    spec: ReferenceSpec {
                        antecedent_action_id: leaf.clone(),
                        content_block_id,
                        range_start,
                        range_end,
                        // Typed, not a sentence: this is persisted, and a persisted
                        // sentence is read as-is in every language.
                        annotation: Some(end.token()),
                    },
                    author_label,
                });
            }
            drop(conn);

            let directive = TurnDirective {
                attached,
                mechanical: true,
                reporting_on: Some(sub.id.clone()),
            };
            // Dropped before the turn runs, for the reason the driven turns' is.
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
            drop(rx);
            let result = Box::pin(self.run_turn_stream(
                &sub.parent_space_id,
                TurnSelector::Participant(sub.owner_participant_id.clone()),
                &target,
                ResponseMode::Reply,
                None,
                &directive,
                tx,
            ))
            .await?;
            // **A report that wrote no post is not a report.** The reference edge
            // rides the answer, so without one the delegation has said nothing and
            // is still outstanding. A closed room is settled; a finding or attach
            // target that moved is retried against the wording a reader still
            // sees. Anything else postless is the unreachable decline — after
            // the last attempt the room waits rather than being mistaken for a
            // delivery.
            if result.response_action_id.is_none() {
                // A closed room suppresses the generation at persist. A finding
                // or attach target that moved in flight does the same, so this
                // loop can rebuild the attachments and retry.
                let conn = self.db_conn().await?;
                if !db::is_live_subspace(&conn, &sub.id).await? {
                    self.end_anchor_wait(&sub.id);
                    return Ok(Report::Settled);
                }
                continue;
            }
            self.end_anchor_wait(&sub.id);
            return Ok(Report::Settled);
        }
        // The last persist was withdrawn and the room is still live. Keep
        // the ending so the next arm is delivery-only, and wait on the
        // parent so a hidden attach target can become visible again.
        self.begin_anchor_wait(sub);
        Ok(Report::Waiting)
    }

    /// Register `sub` against its parent, so a change there wakes it.
    ///
    /// **Unconditional, and before the question it protects** (see the call
    /// site): the answer can commit between registering and asking, and
    /// registering second would leave that commit's change looking at an empty
    /// map. Idempotent — a room that is already waiting stays waiting, because
    /// a wake-up that found no answer has established nothing.
    fn begin_anchor_wait(&self, sub: &db::SubspaceRow) {
        // One alarm per wait, set when the wait begins. A room woken by
        // unrelated traffic re-registers into the same entry — keeping its
        // generation — and does not stack a second one; a room that stopped
        // waiting and later starts again gets a fresh generation and a fresh
        // alarm, which is right — the clock is per wait, not per room.
        let mut waits = self
            .awaiting_anchor
            .lock()
            .expect("anchor wait map poisoned");
        if let std::collections::hash_map::Entry::Vacant(entry) = waits.entry(sub.id.clone()) {
            let generation = self
                .anchor_wait_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            entry.insert((sub.parent_space_id.clone(), generation));
            drop(waits);
            self.schedule_anchor_grace(&sub.id, generation);
        }
    }

    /// Wake this room once the wait has outlived any turn that could still
    /// answer it (see [`ANCHOR_WAIT_GRACE`]).
    ///
    /// **The alarm answers only for its own wait.** It outlives the wait that
    /// set it — nothing cancels a sleeping task — so when it fires it asks
    /// whether the registration it was set for is still the live one, by
    /// generation. Unmarked, an alarm from an ended wait would fire into a
    /// *later* wait on the same room (an answer arrived, was hidden by a
    /// failed regeneration, and a continuation began a second wait) and
    /// expire it on the remainder of the older clock — a report against the
    /// anchor after a fraction of the grace the new wait was owed. A stale
    /// alarm with no live successor now costs nothing at all instead of a
    /// no-op walk.
    fn schedule_anchor_grace(&self, space_id: &str, generation: u64) {
        // No runtime (a synchronous unit test) ⇒ nothing to spawn onto.
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let Some(inner) = self.self_ref.upgrade() else {
            return;
        };
        let space_id = space_id.to_string();
        let grace = self.anchor_wait_grace();
        tokio::spawn(async move {
            tokio::time::sleep(grace).await;
            let still_this_wait = inner
                .awaiting_anchor
                .lock()
                .expect("anchor wait map poisoned")
                .get(&space_id)
                .is_some_and(|(_, live)| *live == generation);
            if still_this_wait {
                inner.arm_subspace_driver(&space_id, Arm::Grace);
            }
        });
    }

    /// How long a wait holds out. [`ANCHOR_WAIT_GRACE`] unless a test has
    /// shortened it — the behaviour under test is what happens when the alarm
    /// comes due, not how long it takes to.
    fn anchor_wait_grace(&self) -> std::time::Duration {
        match self
            .anchor_wait_grace_ms
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => ANCHOR_WAIT_GRACE,
            ms => std::time::Duration::from_millis(ms),
        }
    }

    /// Hold the walk inside the anchor window while a test commits the write
    /// the window is about (see `Inner::anchor_window`). A no-op — one lock and
    /// a `None` — whenever no test has opened one.
    #[cfg(feature = "test-support")]
    async fn pause_in_anchor_window(&self) {
        pause_in_window(&self.anchor_window).await;
    }

    /// The same, for the window inside a walk (see `Inner::cascade_window`).
    #[cfg(feature = "test-support")]
    async fn pause_in_cascade_window(&self) {
        pause_in_window(&self.cascade_window).await;
    }

    /// The same, for the window between a walk's reads (see
    /// `Inner::entry_window`).
    #[cfg(feature = "test-support")]
    async fn pause_in_entry_window(&self) {
        pause_in_window(&self.entry_window).await;
    }

    /// The same, for the window immediately before a mechanical report
    /// persists (see `Inner::persist_window`).
    #[cfg(feature = "test-support")]
    pub(crate) async fn pause_in_report_persist_window(&self) {
        pause_in_window(&self.persist_window).await;
    }

    /// The same, for the window immediately before a regeneration claims its
    /// item (see `Inner::claim_window`).
    #[cfg(feature = "test-support")]
    pub(crate) async fn pause_in_regeneration_claim_window(&self) {
        pause_in_window(&self.claim_window).await;
    }

    /// **What every door that closes a room owes it**: announce the room, and
    /// end any wait it was holding.
    ///
    /// The two halves answer different readers and neither substitutes for the
    /// other. The **announcement** (`Change::Space`) is what tells a window
    /// open on that room it has stopped, and what wakes any *other* delegation
    /// registered against it as a parent — that one arrives through
    /// `rooms_awaiting`, which is keyed by parent. The **clearing** is the
    /// room's own registration, and nothing on the bus can deliver it: an
    /// archived room is not a live delegated room, so `is_ordinary_space`
    /// answers "ordinary" and the supervisor never arms it — the terminal path
    /// that would have cleared the wait is precisely the path that no longer
    /// runs. Left standing, that registration makes every post in the room's
    /// parent pay for a walk that can only no-op, until the grace alarm comes
    /// due.
    ///
    /// A walk already in flight for one of these rooms may register again after
    /// this; it then finds the room archived at its next hop or at the report
    /// gate and clears it there, which is the same one-walk cost that door
    /// always had. Ending it here is what removes the standing one.
    ///
    /// **Every per-room record this process holds is released here, for that
    /// one reason.** The anchor wait was the first; the record of which turn
    /// opened the room ([`Inner::note_spawning_answer_item`]) is the second,
    /// and it leaks the same way and worse — its own clearing sits at
    /// `drive_subspace`'s terminal exits, which is exactly the path an archived
    /// room no longer takes, and unlike a wait nothing ever expires it. A
    /// long-lived GUI opening and archiving delegations would grow the map
    /// without bound. So this is where a room's in-memory state ends, and
    /// anything added later belongs here too.
    ///
    /// **One door for all three archival paths**: a direct archival, a parent's
    /// (`archive_space_tx` answers with the whole subtree it closed), and a
    /// retirement's or a departure's set. Each hands its archived ids straight
    /// here, so none of them has to know what a room keeps.
    pub(crate) fn close_rooms(&self, space_ids: &[String]) {
        for id in space_ids {
            self.end_anchor_wait(id);
            self.forget_spawning_answer_item(id);
            self.bus.emit(Change::Space(id.clone()));
        }
    }

    /// Stop waiting: the answer arrived, or there is no longer one to wait for.
    /// Called on every path out of the anchor question that is not a wait, so
    /// the registration a wait needed cannot outlive it.
    fn end_anchor_wait(&self, space_id: &str) {
        self.awaiting_anchor
            .lock()
            .expect("anchor wait map poisoned")
            .remove(space_id);
    }

    /// The rooms waiting on a change in `space_id` — see
    /// [`Inner::begin_anchor_wait`]. A pure in-memory lookup, because it is
    /// asked of every `Change::Space` in the process.
    pub(crate) fn rooms_awaiting(&self, space_id: &str) -> Vec<String> {
        self.awaiting_anchor
            .lock()
            .expect("anchor wait map poisoned")
            .iter()
            .filter(|(_, (parent, _))| parent.as_str() == space_id)
            .map(|(room, _)| room.clone())
            .collect()
    }
}

/// The posts a report will quote: every branch tip the walk followed, plus —
/// where the walk stopped short — the post it stopped at and anything still
/// waiting on its frontier, which nothing followed either. Deduped, oldest first.
/// Empty only when nothing the walk reached is still visible — a failed
/// regeneration of the starting tail after it was snapshotted, with no other
/// finding left to show. The fallback is that tail as a reader can still see
/// it, never the snapshot of a generation the transcript has hidden.
///
/// **Every tip, and no cap.** The seat guard bounds a room's roster and never
/// bounded its frontier: every seat is notify-all, so one post's fan-out puts
/// an answer from each of them on it, and a walk stopped by the budget or a
/// failure can be holding many more tips than the room has agents. Dropping the
/// oldest of those was silently the worst outcome available — each is a branch
/// nothing followed, the newest one alone settles `db::has_reference_from`, and
/// a settled room is never walked again, so the findings that were dropped were
/// dropped permanently.
///
/// **The edge and the rendering are different things**, which is what makes
/// keeping them all affordable. An edge's recorded range must describe its
/// quoted text exactly, and every tip's does — the human's footnote rail
/// resolves each one whole, and the room counts as reported because its last
/// word is among them. What is bounded instead is the *prompt*: the report
/// turn's context renders each passage through the app's existing clipping
/// (`crate::ATTACHED_PASSAGE_MAX_BYTES`), exactly as every other model-facing
/// rendering already elides — previews strip markers, chore prompts clip the
/// middle out of a post. The block is then bounded by the walk's own ceiling:
/// at most one tip per driven turn ([`MAX_DELEGATION_TURNS`]) plus whatever
/// arrived while it walked, each within one clipped passage.
fn finish(mut leaves: Vec<String>, unwalked: &[String], entry_tail: Option<&str>) -> Vec<String> {
    // Anything still on the frontier when the walk stopped is a tip nothing
    // followed either — it just never got its turn.
    leaves.extend(unwalked.iter().cloned());
    let mut seen = std::collections::HashSet::new();
    leaves.retain(|id| seen.insert(id.clone()));
    if leaves.is_empty()
        && let Some(tail) = entry_tail
    {
        leaves.push(tail.to_string());
    }
    leaves
}

/// `leaves` with `post` appended — the post a walk stopped at is a tip too.
fn with(mut leaves: Vec<String>, post: String) -> Vec<String> {
    leaves.push(post);
    leaves
}

/// Hold here until whoever opened this window lets go. A no-op — one lock and a
/// `None` — whenever nobody has.
#[cfg(feature = "test-support")]
async fn pause_in_window(
    window: &std::sync::Mutex<
        Option<tokio::sync::mpsc::UnboundedSender<tokio::sync::oneshot::Sender<()>>>,
    >,
) {
    let gate = window.lock().expect("test window lock poisoned").clone();
    if let Some(tx) = gate {
        let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
        if tx.send(resume_tx).is_ok() {
            let _ = resume_rx.await;
        }
    }
}

/// The supervisor loop: arm the rooms a previous run left outstanding, then arm
/// a room every time anything is written into it.
///
/// Listening on the bus rather than calling `arm_subspace_driver` at each write
/// seam is what makes **continuation** work without anyone remembering to wire
/// it: the owner posting through its own tool, a human posting after joining, a
/// test seam writing directly — each of them commits and emits, and the room
/// wakes up. `Change::Space` is raised by every post in every conversation, so
/// the arm has to be cheap for the overwhelming majority that are about
/// ordinary rooms; [`Inner::is_ordinary_space`] is what keeps it to one point
/// read per space rather than one per post.
pub(crate) async fn supervise(
    inner: Arc<Inner>,
    mut bus: tokio::sync::broadcast::Receiver<crate::changes::ChangeEvent>,
) {
    // The one place `Arm::Sweep` is honest: this process has just started, so
    // nothing it finds can still be in flight.
    inner.rearm_live_subspaces(Arm::Sweep).await;
    loop {
        match bus.recv().await {
            Ok(event) => {
                if let Change::Space(space_id) = event.change {
                    // Rooms waiting on this space as their **parent** — the
                    // anchor's answer landing there is what they are for. Asked
                    // first and from memory alone, because the parent of a
                    // delegation is an ordinary conversation and the check
                    // below would have cached it as one and skipped it.
                    for room in inner.rooms_awaiting(&space_id) {
                        inner.arm_subspace_driver(&room, Arm::Signal);
                    }
                    if !inner.is_ordinary_space(&space_id).await {
                        inner.arm_subspace_driver(&space_id, Arm::Signal);
                    }
                }
            }
            // A lagged supervisor missed writes it cannot name, so it re-asks
            // the question of every live room — the same recovery every other
            // lagged subscriber makes. **As signals, not as a sweep**: this
            // process is already running, and an owner's turn may be in flight
            // at this very moment, which is exactly the premise `Arm::Sweep`
            // would be helping itself to. A room waiting for that turn's answer
            // therefore keeps waiting, and the answer's own commit wakes it —
            // the wait re-checks on every parent event, so nothing is lost by
            // declining to guess here.
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                inner.rearm_live_subspaces(Arm::Signal).await;
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
        }
    }
}

impl Inner {
    /// Whether `space_id` is definitely **not** a live delegated room, answered
    /// from a cache after the first time.
    ///
    /// Caching a negative is sound because neither half of the answer can turn
    /// back into a yes: `parent_space_id` is written once, when the space is
    /// created, and there is no door anywhere in the app that un-archives a
    /// conversation. So a space that has once answered "ordinary" answers
    /// "ordinary" for the life of the process, and the driver costs one read per
    /// space instead of one per post on a signal every post raises.
    ///
    /// A database failure answers "ordinary" for *this signal only* — skipping
    /// one event rather than wedging the supervisor — and caches nothing, so
    /// the room's next event asks again. Only a successful "not a delegated
    /// room" answer earns the permanent cache entry.
    async fn is_ordinary_space(&self, space_id: &str) -> bool {
        if self
            .ordinary_spaces
            .lock()
            .expect("ordinary-space cache poisoned")
            .contains(space_id)
        {
            return true;
        }
        let live = async {
            let conn = self.db_conn().await?;
            db::is_live_subspace(&conn, space_id).await
        }
        .await;
        match live {
            Ok(true) => false,
            Ok(false) => {
                self.ordinary_spaces
                    .lock()
                    .expect("ordinary-space cache poisoned")
                    .insert(space_id.to_string());
                true
            }
            Err(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The durable form round-trips, which is the whole of what makes it safe
    /// to store instead of a sentence.
    #[test]
    fn every_ending_survives_being_written_down_and_read_back() {
        for end in [
            DelegationEnd::Concluded { truncated: false },
            // The marker is part of the ending, so it has to survive the round
            // trip too — a conclusion resting on a cut-off answer that read
            // back as an ordinary one would be the lie in durable form.
            DelegationEnd::Concluded { truncated: true },
            DelegationEnd::Paused {
                depth: 4,
                limit: 4,
                truncated: false,
            },
            // The marker rides every arm that carries one, so each round-trips
            // both ways — a resumable room resting on a cut-off answer that
            // read back as an ordinary pause would be the same lie in durable
            // form the conclusion's marker exists to prevent.
            DelegationEnd::Paused {
                depth: 4,
                limit: 4,
                truncated: true,
            },
            DelegationEnd::BudgetSpent {
                limit: 32,
                truncated: false,
            },
            DelegationEnd::BudgetSpent {
                limit: 32,
                truncated: true,
            },
            DelegationEnd::TurnFailed {
                reason: DelegationFailure::Upstream,
            },
            DelegationEnd::TurnFailed {
                reason: DelegationFailure::Funding,
            },
            DelegationEnd::TurnFailed {
                reason: DelegationFailure::Configuration,
            },
            DelegationEnd::TurnFailed {
                reason: DelegationFailure::Unfinished,
            },
        ] {
            let token = end.token();
            assert_eq!(DelegationEnd::parse(&token), Some(end), "{token}");
            assert!(!end.describe().is_empty());
        }
    }

    /// **A person's note is never mistaken for one** — which is what lets the
    /// two share a column. Neither is anything a future version writes that
    /// this one does not understand: both read back as a note, which shows the
    /// reader prose rather than an ending somebody guessed at.
    #[test]
    fn a_note_is_not_an_ending() {
        for note in [
            "the delegated conversation ran to a stop",
            "eidola:delegation",
            "eidola:delegation/",
            "eidola:delegation/concluded/extra",
            "eidola:delegation/paused/4",
            "eidola:delegation/budget/lots",
            "eidola:delegation/failed/moonshot",
            "eidola:delegation/adjourned",
            "",
        ] {
            assert_eq!(DelegationEnd::parse(note), None, "{note:?}");
            assert_eq!(annotation_for_model(Some(note)), Some(note.to_string()));
        }
        assert_eq!(annotation_for_model(None), None);
    }

    /// The failure a report carries is a category, and the category is decided
    /// from the error's *variant* — never from its words.
    #[test]
    fn a_failure_reports_its_kind_and_not_its_message() {
        let secret = "https://internal.example/v1 — token abc123";
        let described = DelegationEnd::failed(&AppError::Server {
            status: 500,
            message: secret.to_string(),
        })
        .describe();
        assert!(!described.contains(secret), "{described}");
        assert_eq!(
            DelegationEnd::failed(&AppError::Server {
                status: 500,
                message: secret.to_string(),
            }),
            DelegationEnd::TurnFailed {
                reason: DelegationFailure::Upstream
            }
        );
        assert_eq!(
            DelegationEnd::failed(&AppError::NoAccount),
            DelegationEnd::TurnFailed {
                reason: DelegationFailure::Funding
            }
        );
    }
}
