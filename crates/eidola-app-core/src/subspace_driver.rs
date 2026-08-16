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
    Planner, ReferenceSpec, ResponseMode, TurnDirective, TurnSelector, db, now_ms,
};

/// The per-room driver registry: room id → whether an arm arrived while its
/// task was running (see [`Inner::arm_subspace_driver`]).
pub(crate) type DriverRegistry = HashMap<String, bool>;

/// Why a room is being armed, which decides one thing: whether it may wait for
/// its anchor to be answered (see [`Inner::report_delegation`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arm {
    /// Something was written into the room and the bus said so.
    Signal,
    /// The startup sweep. A process that has just started cannot be waiting on
    /// a turn of its own, so nothing it finds is still in flight.
    Sweep,
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
const DELEGATION_END_PREFIX: &str = "eidola:delegation/";

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

    /// Give `space_id` a driver if it is a room this driver owns and does not
    /// already have one.
    ///
    /// Idempotent, cheap, and safe to call for any space id — the caller is the
    /// change bus, which raises `Change::Space` for every post in every
    /// conversation, so most calls are about rooms this has nothing to do with.
    ///
    /// **A room already being driven records the arm rather than dropping it.**
    /// The driver walks the posts it planned; a post that arrives from
    /// somewhere else while it is walking — a human asking a question in a room
    /// they are watching — is not on that walk, and forgetting it would leave
    /// the answer unanswered until something else woke the room. So the second
    /// arm sets a flag the running task checks before it retires, and the walk
    /// simply starts again from the room's new tail.
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
            if let Some(rearm) = running.get_mut(space_id) {
                *rearm = true;
                return;
            }
            running.insert(space_id.to_string(), false);
        }
        let inner = self.clone();
        let space_id = space_id.to_string();
        tokio::spawn(async move {
            loop {
                // Everything this task commits is unattended by construction:
                // no consumer call is outstanding, so a window that drops the
                // invalidation while it is busy loses it for good.
                let result = with_origin(
                    ChangeOrigin::Unattended,
                    inner.drive_subspace(space_id.clone(), arm),
                )
                .await;
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
                        Some(rearm) if *rearm || failed => *rearm = false,
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

    /// Arm every live sub-space — the startup path.
    ///
    /// A process that comes back mid-delegation has to pick the room up, and
    /// "mid-delegation" is not a thing that was written down anywhere: each
    /// armed driver decides for itself, from the rows, whether there is
    /// anything outstanding (see [`Inner::drive_subspace`]), and retires
    /// immediately when there is not. So this arms broadly and lets the one
    /// definition of outstanding work do the deciding, rather than keeping a
    /// second, subtly different copy of it here.
    ///
    /// Failures are warned about and swallowed: picking delegations back up is
    /// housekeeping, and a process that refused to start over it would cost the
    /// reader everything else.
    pub(crate) async fn rearm_live_subspaces(self: &Arc<Self>) {
        let rooms = async {
            let conn = self.db_conn().await?;
            db::live_subspaces(&conn).await
        }
        .await;
        match rooms {
            Ok(rooms) => {
                for room in rooms {
                    self.arm_subspace_driver(&room.id, Arm::Sweep);
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

    /// Drive one delegated room to a stop, then report it to its parent.
    ///
    /// Returns without doing anything at all when the room is not one this
    /// driver owns, is archived, or has already had its last word reported —
    /// which is what makes arming cheap enough to do from a signal every post
    /// raises.
    pub(crate) async fn drive_subspace(&self, space_id: String, arm: Arm) -> Result<(), AppError> {
        let conn = self.db_conn().await?;
        let Some(sub) = db::subspace(&conn, &space_id).await? else {
            return Ok(()); // not a delegated room
        };
        if sub.archived_at.is_some() {
            return Ok(()); // archival stops new work, here as everywhere
        }
        let Some(tail) = db::last_action_in_space(&conn, &space_id).await? else {
            return Ok(()); // a room with no posts — unreachable, a brief opens every one
        };
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
        let Some((end, witness)) = Box::pin(self.cascade_subspace(&space_id, tail)).await? else {
            return Ok(()); // the room closed under us; nothing to report
        };
        Box::pin(self.report_delegation(&sub, end, witness, arm)).await
    }

    /// The plan → drive → re-plan walk, ending at the first terminal outcome.
    ///
    /// Returns the outcome **and the post the walk ended on** — its witness.
    /// That post, not whatever the room's tail happens to be by the time the
    /// report is written, is what the report quotes: a post landing in between
    /// would otherwise be reported as if the walk had considered it, and would
    /// then read as already-reported when its own walk came round, so the room
    /// would go quiet holding an unanswered post. Reporting the witness leaves
    /// that post unreported, which is exactly what makes its own arrival arm the
    /// room again and get it the walk it is owed.
    ///
    /// `Ok(None)` means the room stopped being drivable while we were in it (an
    /// archival landed), which is deliberately **not** an outcome to report: the
    /// room was closed on purpose, and a report about it would be a message
    /// nobody asked for about work nobody wants continued.
    async fn cascade_subspace(
        &self,
        space_id: &str,
        tail: String,
    ) -> Result<Option<(DelegationEnd, String)>, AppError> {
        // When this walk began. Anything posted from here on that the walk did
        // not write itself came from somewhere else and is owed a hearing —
        // see the refill below.
        let began = now_ms();
        // The room's last word *as this walk saw it* — the entry tail until a
        // driven turn produces something newer.
        let mut witness = tail.clone();
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
        let mut frontier: Vec<String> = vec![tail];
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
                let arrived = db::posts_in_space_since(&conn, space_id, began).await?;
                drop(conn);
                frontier.extend(arrived.into_iter().filter(|id| !served.contains(id)));
                if frontier.is_empty() {
                    return Ok(Some((DelegationEnd::Concluded, witness)));
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
                    witness,
                )));
            }
            let turns = match self
                .plan_and_refine(space_id, &post, Planner::Driver)
                .await?
            {
                NotificationPlan::Paused { depth, limit } => {
                    return Ok(Some((DelegationEnd::Paused { depth, limit }, witness)));
                }
                NotificationPlan::Turns(turns) => turns,
            };
            for turn in turns {
                let conn = self.db_conn().await?;
                let taken = db::turns_taken_in_space(&conn, space_id).await?;
                drop(conn);
                if taken >= MAX_DELEGATION_TURNS {
                    return Ok(Some((
                        DelegationEnd::BudgetSpent {
                            limit: MAX_DELEGATION_TURNS,
                        },
                        witness,
                    )));
                }
                match Box::pin(self.drive_planned_turn(space_id, &turn)).await {
                    // A turn that wrote a post is a post to re-plan from, and
                    // the room's newest word; a turn that declined wrote a
                    // decision, which is not something anyone replies to.
                    Ok(Some(post_action_id)) => {
                        witness = post_action_id.clone();
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
                        return Ok(Some((DelegationEnd::failed(&e), witness)));
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
    /// opened from**, carrying `witness` — the room's last word as the walk saw
    /// it — as a quoted reference the driver attaches itself.
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
        witness: String,
        arm: Arm,
    ) -> Result<(), AppError> {
        let conn = self.db_conn().await?;
        // **The room is re-read one last time before anything acts.** An
        // archival can land during the walk — a retirement archives every room
        // its agent owned — and a report about a conversation somebody closed
        // is a message nobody asked for about work nobody wants continued. One
        // read, the same shape as every other liveness gate here.
        if !db::is_live_subspace(&conn, &sub.id).await? {
            return Ok(());
        }
        // The passage: the post the walk ended on, whole. A delegation's last
        // word is its finding, and clipping it here would report a fragment
        // while the edge claimed the post.
        let (content_block_id, range_start, range_end) =
            match db::first_quotable_block(&conn, &witness).await? {
                Some((block_id, text)) if !text.is_empty() => {
                    (Some(block_id), Some(0), Some(text.len() as i64))
                }
                // No text to quote (a room whose last post carries none) still
                // gets an edge — a pointer to the post rather than a quote of
                // it, which is what a range-less reference means everywhere.
                _ => (None, None, None),
            };
        // Named the way every other reference read names an author: the label
        // that post's **own space** gives them, with no liveness filter — so an
        // agent retired between writing the finding and its being reported is
        // still named, rather than degrading to an anonymous post from
        // somewhere else.
        let author_label = db::post_author_label(&conn, &witness).await?;

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
                    // **Termination is the sweep, and only the sweep.** Waiting
                    // costs nothing — no turn, no spend, no row — so an
                    // indefinite one is not a leak, it is a room correctly
                    // declining to guess. What it must not be is permanent, and
                    // the case where the answer never comes at all is a spawning
                    // turn that died; that process cannot outlive its own crash,
                    // and the next start's sweep arms with `Arm::Sweep`, which
                    // never waits — at that moment nothing can still be in
                    // flight, so the answer either exists or never will, and the
                    // anchor is then the right attachment because there is
                    // nothing for the report to sit beneath.
                    None if arm == Arm::Signal => return Ok(()),
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
            return Ok(()); // an empty parent has nothing to reply to
        };
        drop(conn);

        let directive = TurnDirective {
            attached: vec![AttachedReference {
                // Ordinal 0 is the reply edge's; a report quotes exactly one
                // thing, so the finding is 1.
                ordinal: 1,
                origin: AttachmentOrigin::Authored,
                spec: ReferenceSpec {
                    antecedent_action_id: witness,
                    content_block_id,
                    range_start,
                    range_end,
                    // Typed, not a sentence: this is persisted, and a persisted
                    // sentence is read as-is in every language.
                    annotation: Some(end.token()),
                },
                author_label,
            }],
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
        Ok(())
    }

    /// Register `sub` against its parent, so a change there wakes it.
    ///
    /// **Unconditional, and before the question it protects** (see the call
    /// site): the answer can commit between registering and asking, and
    /// registering second would leave that commit's change looking at an empty
    /// map. Idempotent — a room that is already waiting stays waiting, because
    /// a wake-up that found no answer has established nothing.
    fn begin_anchor_wait(&self, sub: &db::SubspaceRow) {
        self.awaiting_anchor
            .lock()
            .expect("anchor wait map poisoned")
            .insert(sub.id.clone(), sub.parent_space_id.clone());
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
    fn rooms_awaiting(&self, space_id: &str) -> Vec<String> {
        self.awaiting_anchor
            .lock()
            .expect("anchor wait map poisoned")
            .iter()
            .filter(|(_, parent)| parent.as_str() == space_id)
            .map(|(room, _)| room.clone())
            .collect()
    }
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
    inner.rearm_live_subspaces().await;
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
            // lagged subscriber makes.
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                inner.rearm_live_subspaces().await;
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
    /// A database failure answers "ordinary", which skips a room rather than
    /// wedging the supervisor; the next restart's sweep picks it up.
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
        .await
        .unwrap_or(false);
        if !live {
            self.ordinary_spaces
                .lock()
                .expect("ordinary-space cache poisoned")
                .insert(space_id.to_string());
        }
        !live
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
