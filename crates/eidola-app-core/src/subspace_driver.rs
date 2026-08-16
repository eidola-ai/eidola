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
            | AppError::SpaceArchived { .. } => Self::Configuration,
            AppError::ToolLoop { .. }
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
pub enum DelegationEnd {
    /// Planning returned no turns: nobody's notify policy fired on the room's
    /// last post, which is what a conversation running out of things to say
    /// looks like from here.
    Concluded,
    /// The room hit its own cascade guard. Resumable by posting into it.
    Paused { depth: i64, limit: i64 },
    /// The per-delegation turn budget is spent.
    BudgetSpent { limit: i64 },
    /// A turn failed, in the bounded sense of [`DelegationFailure`].
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
        match self {
            Self::Concluded => format!("{DELEGATION_END_PREFIX}concluded"),
            Self::Paused { depth, limit } => {
                format!("{DELEGATION_END_PREFIX}paused/{depth}/{limit}")
            }
            Self::BudgetSpent { limit } => format!("{DELEGATION_END_PREFIX}budget/{limit}"),
            Self::TurnFailed { reason } => {
                format!("{DELEGATION_END_PREFIX}failed/{}", reason.token())
            }
        }
    }

    /// Read an annotation back as an ending, or `None` when it is not one —
    /// which is every annotation a person wrote, and also any token a future
    /// version writes that this one does not understand. Both degrade to "a
    /// note", which is the safe direction: a reader shows prose it cannot
    /// interpret rather than claiming an ending it guessed at.
    pub fn parse(annotation: &str) -> Option<Self> {
        let rest = annotation.strip_prefix(DELEGATION_END_PREFIX)?;
        let mut parts = rest.split('/');
        let out = match parts.next()? {
            "concluded" => Self::Concluded,
            "paused" => Self::Paused {
                depth: parts.next()?.parse().ok()?,
                limit: parts.next()?.parse().ok()?,
            },
            "budget" => Self::BudgetSpent {
                limit: parts.next()?.parse().ok()?,
            },
            "failed" => Self::TurnFailed {
                reason: DelegationFailure::parse(parts.next()?)?,
            },
            _ => return None,
        };
        parts.next().is_none().then_some(out)
    }

    /// What a model reads where a person's annotation would go — built here,
    /// never stored.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Concluded => "the delegated conversation ran to a stop".to_string(),
            Self::Paused { depth, limit } => format!(
                "the delegated conversation reached its reply limit ({depth} replies in a row, \
                 limit {limit}) and can be resumed by posting there"
            ),
            Self::BudgetSpent { limit } => format!(
                "the delegated conversation used all {limit} of the turns it is allowed and was \
                 stopped there"
            ),
            Self::TurnFailed { reason } => format!(
                "the delegated conversation stopped because {}",
                reason.describe()
            ),
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
                if let Err(e) = result {
                    eprintln!(
                        "warning: the delegated conversation {space_id} could not be driven: {e}"
                    );
                }
                {
                    let mut running = inner.subspace_drivers.lock().expect("driver map poisoned");
                    match running.get_mut(&space_id) {
                        // Another pass, on the strongest licence in play: what
                        // arrived while this one ran, merged with the one it
                        // ran under, so neither is dropped by the other (see
                        // [`Arm::merge`]). A retry after a failure keeps its
                        // own arm for the same reason — the failure did not
                        // take back what the walk was allowed to conclude.
                        Some(pending) if pending.is_some() || failed => {
                            if let Some(next) = pending.take() {
                                arm = arm.merge(next);
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
    /// Failures are warned about and swallowed: picking delegations back up is
    /// housekeeping, and a process that refused to start over it would cost the
    /// reader everything else.
    pub(crate) async fn rearm_live_subspaces(self: &Arc<Self>, arm: Arm) {
        let rooms = async {
            let conn = self.db_conn().await?;
            db::live_subspaces(&conn).await
        }
        .await;
        // Read after the enumeration, and only where it can matter: this is
        // the sweep's licence being checked, and every other arm already
        // claims nothing.
        let spawned_here = match arm {
            Arm::Sweep => self.take_rooms_spawned_here(),
            _ => std::collections::HashSet::new(),
        };
        match rooms {
            Ok(rooms) => {
                for room in rooms {
                    let arm = if spawned_here.contains(&room.id) {
                        Arm::Signal
                    } else {
                        arm
                    };
                    self.arm_subspace_driver(&room.id, arm);
                }
            }
            Err(e) => eprintln!("warning: delegated conversations could not be enumerated: {e}"),
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
                    eprintln!(
                        "warning: the delegated conversation {space_id} has been tried \
                         {MAX_ATTEMPTS_PER_TAIL} times against the same last post and is being \
                         left alone; a post there, or a restart, will pick it up again"
                    );
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
                    },
                );
            }
        }
        true
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
    pub(crate) async fn drive_subspace(&self, space_id: String, arm: Arm) -> Result<(), AppError> {
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
            return Ok(()); // not a delegated room
        };
        if sub.archived_at.is_some() {
            self.end_anchor_wait(&space_id);
            return Ok(()); // archival stops new work, here as everywhere
        }
        let Some(tail) = db::last_action_in_space(&conn, &space_id).await? else {
            self.end_anchor_wait(&space_id);
            return Ok(()); // a room with no posts — unreachable, a brief opens every one
        };
        #[cfg(feature = "test-support")]
        self.pause_in_entry_window().await;
        // The whole of "is there anything to do here", asked of the rows. A
        // reported tail is a delegation whose last word the parent already has;
        // anything posted since makes a new tail, and the answer flips back.
        if db::has_reference_from(
            &conn,
            &sub.parent_space_id,
            &sub.owner_participant_id,
            &tail,
        )
        .await?
        {
            self.end_anchor_wait(&space_id);
            return Ok(());
        }
        drop(conn);
        // Asked *after* the outstanding check, so a settled room costs no
        // attempt and a room that is genuinely stuck cannot spend forever.
        if !self.claim_walk(&space_id, &tail) {
            return Ok(());
        }

        // Boxed, like every other await on the turn path: `run_turn_stream`'s
        // state machine is the largest in the crate, and stacking this walk's
        // frame on top of it overflows a worker stack.
        let Some((end, leaves)) =
            Box::pin(self.cascade_subspace(&space_id, tail.clone(), since_row)).await?
        else {
            self.end_anchor_wait(&space_id);
            return Ok(()); // the room closed under us; nothing to report
        };
        let outcome = Box::pin(self.report_delegation(&sub, end, leaves, arm)).await;
        // **Waiting is not an attempt.** The meter bounds retries of work that
        // failed; a walk that ended by waiting for the post it was delegated
        // from tried nothing and failed at nothing, and the room's last word is
        // unchanged — so counting it would spend the room's whole allowance on
        // three unrelated posts in the parent and then refuse the walk that the
        // real answer finally arms.
        if matches!(outcome, Ok(Report::Waiting)) {
            self.release_walk(&space_id, &tail);
        }
        outcome.map(|_| ())
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
        #[cfg(feature = "test-support")]
        let mut paused_once = false;
        loop {
            let Some(post) = frontier.pop() else {
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
                frontier.extend(arrived.into_iter().filter(|id| !served.contains(id)));
                if frontier.is_empty() {
                    let end = match paused {
                        Some((depth, limit)) => DelegationEnd::Paused { depth, limit },
                        None => DelegationEnd::Concluded,
                    };
                    return Ok(Some((end, finish(leaves, &frontier, &tail))));
                }
                continue;
            };
            served.insert(post.clone());
            let conn = self.db_conn().await?;
            let live = db::is_live_subspace(&conn, space_id).await?;
            let taken = db::turns_taken_in_space(&conn, space_id).await?;
            drop(conn);
            if !live {
                return Ok(None);
            }
            // The budget is checked before the plan, not after it: planning a
            // room whose budget is spent may cost a router inference, and the
            // turns it returned could not be driven anyway.
            if taken >= MAX_DELEGATION_TURNS {
                return Ok(Some((
                    DelegationEnd::BudgetSpent {
                        limit: MAX_DELEGATION_TURNS,
                    },
                    finish(with(leaves, post), &frontier, &tail),
                )));
            }
            let turns = match self
                .plan_and_refine(space_id, &post, Planner::Driver)
                .await?
            {
                NotificationPlan::Paused { depth, limit } => {
                    paused.get_or_insert((depth, limit));
                    leaves.push(post.clone());
                    continue;
                }
                NotificationPlan::Turns(turns) => turns,
            };
            // **Nothing follows this one, so it is a finding.** A post whose
            // plan comes back empty is where a branch of the room stopped
            // having anything to add — which is exactly what the parent is owed
            // a look at.
            if turns.is_empty() {
                leaves.push(post.clone());
            }
            for turn in turns {
                let conn = self.db_conn().await?;
                let taken = db::turns_taken_in_space(&conn, space_id).await?;
                drop(conn);
                if taken >= MAX_DELEGATION_TURNS {
                    return Ok(Some((
                        DelegationEnd::BudgetSpent {
                            limit: MAX_DELEGATION_TURNS,
                        },
                        finish(with(leaves, post), &frontier, &tail),
                    )));
                }
                match Box::pin(self.drive_planned_turn(space_id, &turn)).await {
                    // A turn that wrote a post is a post to re-plan from, and
                    // the room's newest word; a turn that declined wrote a
                    // decision, which is not something anyone replies to.
                    Ok(Some(post_action_id)) => {
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
                        return Ok(Some((
                            DelegationEnd::failed(&e),
                            finish(with(leaves, post), &frontier, &tail),
                        )));
                    }
                }
            }
        }
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
    ) -> Result<Option<String>, AppError> {
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
        Ok(result.response_action_id)
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
        // One attachment per finding, at ordinals `1..=N` in the order the walk
        // found them — every one of them, because each is a branch nothing
        // followed and only the room's own last word among them settles the
        // delegation (see [`finish`]). The **edge** names the whole passage; the
        // report turn's *rendering* of it is what is clipped, at the seam every
        // attached passage renders through.
        let mut attached: Vec<AttachedReference> = Vec::with_capacity(leaves.len());
        for (i, leaf) in leaves.iter().enumerate() {
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
                match db::last_reply_by_participant(
                    &conn,
                    &sub.parent_space_id,
                    &sub.owner_participant_id,
                    anchor,
                )
                .await?
                {
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
            None => match db::last_post_by_participant(
                &conn,
                &sub.parent_space_id,
                &sub.owner_participant_id,
            )
            .await?
            {
                Some(id) => Some(id),
                None => db::last_action_in_space(&conn, &sub.parent_space_id).await?,
            },
        };
        let Some(target) = target else {
            self.end_anchor_wait(&sub.id);
            return Ok(Report::Settled); // an empty parent has nothing to reply to
        };
        drop(conn);

        let directive = TurnDirective {
            attached,
            mechanical: true,
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
        // is still outstanding — which is the truth, and saying so loudly is
        // what keeps it from being mistaken for a delivery. Unreachable while
        // the directive withdraws the decline checkpoint (the only way a turn
        // ends postless), and kept as the thing that would notice if it stopped
        // being.
        if result.response_action_id.is_none() {
            return Err(AppError::Internal {
                message: format!(
                    "the report for delegated conversation {} produced no post",
                    sub.id
                ),
            });
        }
        self.end_anchor_wait(&sub.id);
        Ok(Report::Settled)
    }

    /// Register `sub` against its parent, so a change there wakes it.
    ///
    /// **Unconditional, and before the question it protects** (see the call
    /// site): the answer can commit between registering and asking, and
    /// registering second would leave that commit's change looking at an empty
    /// map. Idempotent — a room that is already waiting stays waiting, because
    /// a wake-up that found no answer has established nothing.
    fn begin_anchor_wait(&self, sub: &db::SubspaceRow) {
        let fresh = self
            .awaiting_anchor
            .lock()
            .expect("anchor wait map poisoned")
            .insert(sub.id.clone(), sub.parent_space_id.clone())
            .is_none();
        // One alarm per wait, set when the wait begins. A room woken by
        // unrelated traffic re-registers into the same entry and does not stack
        // a second one; a room that stopped waiting and later starts again gets
        // a new one, which is right — the clock is per wait, not per room.
        if fresh {
            self.schedule_anchor_grace(&sub.id);
        }
    }

    /// Wake this room once the wait has outlived any turn that could still
    /// answer it (see [`ANCHOR_WAIT_GRACE`]).
    fn schedule_anchor_grace(&self, space_id: &str) {
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
            inner.arm_subspace_driver(&space_id, Arm::Grace);
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
            .filter(|(_, parent)| parent.as_str() == space_id)
            .map(|(room, _)| room.clone())
            .collect()
    }
}

/// The posts a report will quote: every branch tip the walk followed, plus —
/// where the walk stopped short — the post it stopped at and anything still
/// waiting on its frontier, which nothing followed either. Deduped, oldest first, and never empty: a walk that got no
/// further than the room's first post still has that post to show.
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
fn finish(mut leaves: Vec<String>, unwalked: &[String], entry_tail: &str) -> Vec<String> {
    // Anything still on the frontier when the walk stopped is a tip nothing
    // followed either — it just never got its turn.
    leaves.extend(unwalked.iter().cloned());
    let mut seen = std::collections::HashSet::new();
    leaves.retain(|id| seen.insert(id.clone()));
    if leaves.is_empty() {
        leaves.push(entry_tail.to_string());
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
            DelegationEnd::Concluded,
            DelegationEnd::Paused { depth: 4, limit: 4 },
            DelegationEnd::BudgetSpent { limit: 32 },
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
