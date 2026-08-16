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
    AttachedReference, ChatStreamEvent, Inner, NotificationPlan, PlannedTurn, Planner,
    ReferenceSpec, ResponseMode, TurnSelector, db,
};

/// The per-room driver registry: room id → whether an arm arrived while its
/// task was running (see [`Inner::arm_subspace_driver`]).
pub(crate) type DriverRegistry = HashMap<String, bool>;

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

/// How a delegated room stopped.
///
/// Every variant is reported, on the same channel, in the same shape — the
/// failure arm is not a quieter version of the success one. What differs is one
/// sentence, which becomes the annotation on the reference edge the report
/// carries, so the reason is durable and reaches the human's footnote rail as
/// well as the next model to read the post.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DelegationEnd {
    /// Planning returned no turns: nobody's notify policy fired on the room's
    /// last post, which is what a conversation running out of things to say
    /// looks like from here.
    Concluded,
    /// The room hit its own cascade guard. Resumable by posting into it.
    Paused { depth: i64, limit: i64 },
    /// The per-delegation turn budget is spent.
    BudgetSpent { limit: i64 },
    /// A turn failed. The message is the typed error's own words.
    TurnFailed { message: String },
}

impl DelegationEnd {
    /// The sentence that rides the report's reference edge as its annotation.
    ///
    /// Written for whoever reads the report next — the participants of the
    /// parent conversation, model and human alike — so it says what happened to
    /// the room rather than naming an internal state.
    pub(crate) fn annotation(&self) -> String {
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
            Self::TurnFailed { message } => {
                format!("a turn in the delegated conversation failed: {message}")
            }
        }
    }
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
    pub(crate) fn arm_subspace_driver(self: &Arc<Self>, space_id: &str) {
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
                    inner.drive_subspace(space_id.clone()),
                )
                .await;
                if let Err(e) = result {
                    eprintln!(
                        "warning: the delegated conversation {space_id} could not be driven: {e}"
                    );
                }
                let mut running = inner.subspace_drivers.lock().expect("driver map poisoned");
                match running.get_mut(&space_id) {
                    Some(rearm) if *rearm => *rearm = false,
                    _ => {
                        running.remove(&space_id);
                        return;
                    }
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
                    self.arm_subspace_driver(&room.id);
                }
            }
            Err(e) => eprintln!("warning: delegated conversations could not be enumerated: {e}"),
        }
    }

    /// Drive one delegated room to a stop, then report it to its parent.
    ///
    /// Returns without doing anything at all when the room is not one this
    /// driver owns, is archived, or has already had its last word reported —
    /// which is what makes arming cheap enough to do from a signal every post
    /// raises.
    pub(crate) async fn drive_subspace(&self, space_id: String) -> Result<(), AppError> {
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

        // Boxed, like every other await on the turn path: `run_turn_stream`'s
        // state machine is the largest in the crate, and stacking this walk's
        // frame on top of it overflows a worker stack.
        let end = Box::pin(self.cascade_subspace(&space_id, tail)).await?;
        let Some(end) = end else {
            return Ok(()); // the room closed under us; nothing to report
        };
        Box::pin(self.report_delegation(&sub, end)).await
    }

    /// The plan → drive → re-plan walk, ending at the first terminal outcome.
    ///
    /// `Ok(None)` means the room stopped being drivable while we were in it (an
    /// archival landed), which is deliberately **not** an outcome to report: the
    /// room was closed on purpose, and a report about it would be a message
    /// nobody asked for about work nobody wants continued.
    async fn cascade_subspace(
        &self,
        space_id: &str,
        tail: String,
    ) -> Result<Option<DelegationEnd>, AppError> {
        // Posts whose replies have not been planned yet. A fan-out puts several
        // answers on it at once; taking the newest first walks each thread of
        // the room down before starting the next, which is the order a reader
        // watching the transcript would expect.
        let mut frontier: Vec<String> = vec![tail];
        loop {
            let Some(post) = frontier.pop() else {
                return Ok(Some(DelegationEnd::Concluded));
            };
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
                return Ok(Some(DelegationEnd::BudgetSpent {
                    limit: MAX_DELEGATION_TURNS,
                }));
            }
            let turns = match self
                .plan_and_refine(space_id, &post, Planner::Driver)
                .await?
            {
                NotificationPlan::Paused { depth, limit } => {
                    return Ok(Some(DelegationEnd::Paused { depth, limit }));
                }
                NotificationPlan::Turns(turns) => turns,
            };
            for turn in turns {
                let conn = self.db_conn().await?;
                let taken = db::turns_taken_in_space(&conn, space_id).await?;
                drop(conn);
                if taken >= MAX_DELEGATION_TURNS {
                    return Ok(Some(DelegationEnd::BudgetSpent {
                        limit: MAX_DELEGATION_TURNS,
                    }));
                }
                match Box::pin(self.drive_planned_turn(space_id, &turn)).await {
                    // A turn that wrote a post is a post to re-plan from; a
                    // turn that declined wrote a decision, which is not
                    // something anyone replies to.
                    Ok(Some(post_action_id)) => frontier.push(post_action_id),
                    Ok(None) => {}
                    Err(AppError::SpaceArchived { .. }) => return Ok(None),
                    Err(e) => {
                        return Ok(Some(DelegationEnd::TurnFailed {
                            message: e.to_string(),
                        }));
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
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let result = Box::pin(self.run_turn_stream(
            space_id,
            TurnSelector::Participant(turn.participant_id.clone()),
            &turn.target_action_id,
            ResponseMode::Reply,
            None,
            &[],
            tx,
        ))
        .await?;
        Ok(result.response_action_id)
    }

    /// Deliver the owner's report into the parent conversation.
    ///
    /// A turn for the owning agent, replying to **its own last post there** —
    /// the one-reply spine the rest of the thread model depends on, and the
    /// branch the delegation was opened from — carrying the delegated room's
    /// last word as a quoted reference the driver attaches itself.
    ///
    /// Every refusal the parent can raise is the parent's to raise: an archived
    /// parent refuses the turn at the same gate every other turn meets, an
    /// owner who has left the parent is refused by participant resolution, and
    /// a funding failure fails like any other turn. None of them is special-
    /// cased here, and the room's tail stays unreported, so a later run tries
    /// again.
    async fn report_delegation(
        &self,
        sub: &db::SubspaceRow,
        end: DelegationEnd,
    ) -> Result<(), AppError> {
        let conn = self.db_conn().await?;
        let Some(tail) = db::last_action_in_space(&conn, &sub.id).await? else {
            return Ok(());
        };
        // The passage: the room's last post, whole. A delegation's last word is
        // its finding, and clipping it here would report a fragment while the
        // edge claimed the post.
        let (content_block_id, range_start, range_end) =
            match db::first_quotable_block(&conn, &tail).await? {
                Some((block_id, text)) if !text.is_empty() => {
                    (Some(block_id), Some(0), Some(text.len() as i64))
                }
                // No text to quote (a room whose last post carries none) still
                // gets an edge — a pointer to the post rather than a quote of
                // it, which is what a range-less reference means everywhere.
                _ => (None, None, None),
            };
        // Named as the *source* space names them: the parent may never have met
        // this participant, and its own override would be the wrong name.
        let author_label = match db::action_author(&conn, &tail).await? {
            Some((author_id, _, _)) => db::space_participants(&conn, &sub.id)
                .await?
                .into_iter()
                .find(|m| m.participant_id == author_id)
                .map(|m| m.label),
            None => None,
        };
        // Where it attaches: the owner's own last word in the parent, which is
        // the turn that opened this delegation. A parent whose owner has posted
        // nothing at all falls back to the conversation's tail rather than
        // refusing — the report is the point, and an unattached one would be a
        // second thread root.
        let target = match db::last_post_by_participant(
            &conn,
            &sub.parent_space_id,
            &sub.owner_participant_id,
        )
        .await?
        {
            Some(id) => Some(id),
            None => db::last_action_in_space(&conn, &sub.parent_space_id).await?,
        };
        let Some(target) = target else {
            return Ok(()); // an empty parent has nothing to reply to
        };
        drop(conn);

        let attached = [AttachedReference {
            spec: ReferenceSpec {
                antecedent_action_id: tail,
                content_block_id,
                range_start,
                range_end,
                annotation: Some(end.annotation()),
            },
            author_label,
        }];
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        Box::pin(self.run_turn_stream(
            &sub.parent_space_id,
            TurnSelector::Participant(sub.owner_participant_id.clone()),
            &target,
            ResponseMode::Reply,
            None,
            &attached,
            tx,
        ))
        .await?;
        Ok(())
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
                if let Change::Space(space_id) = event.change
                    && !inner.is_ordinary_space(&space_id).await
                {
                    inner.arm_subspace_driver(&space_id);
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
