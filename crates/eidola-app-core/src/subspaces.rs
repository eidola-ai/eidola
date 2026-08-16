//! Agent-spawned sub-spaces: the door an agent delegates through.
//!
//! An agent that needs work done elsewhere spawns a **sub-space** — a new
//! space with **no human participant**, holding the spawning (owning) agent
//! and the sub-agents it delegates to, opened by a *brief* the owner writes.
//! Nothing is copied: the sub-space starts fresh, and the parent hears back
//! through an ordinary cross-space reference rather than a merged transcript.
//!
//! Three things about it are structural rather than conventional, and each is
//! enforced in one place:
//!
//! * **Ownership is `parent_space_id` + the owner's `role = 'owner'`
//!   membership row.** No new column: an owner is definitionally a member, so
//!   the membership row can carry it.
//! * **Capabilities are a snapshot, taken at spawn and never re-evaluated.**
//!   A sub-space holds a subset of what its parent held, copied verbatim from
//!   the parent's rows (see `space_capability` in the schema). A capability
//!   the parent gains later does not reach a sub-space that already exists,
//!   and the remedy for a grant someone regrets is archiving the sub-space.
//!   Because each sub-space's own snapshot is the only source its children can
//!   be granted from, a chain attenuates without anyone walking it.
//! * **Runaway is bounded by three constants**, checked inside the spawning
//!   transaction: how deep the chain may go, how many live sub-spaces one
//!   owner may hold, and how many sub-agents one room may seat.
//! * **The owner membership is structural.** Nothing can end it and nothing
//!   can add a second one (`db::refuse_second_subspace_owner`, and the leave's
//!   own `WHERE` in `db::remove_space_participant_tx`), so the row that
//!   records who is answerable for a delegation cannot be edited into
//!   something else by ordinary roster work.
//!
//! **Billing is unchanged.** The pricing contract inspects only the shape of
//! the message and tool JSON, and holds are minted from the process's single
//! configured account — nothing anywhere assumes a human at the root of a
//! thread. A sub-space turn spends like any turn; what the human authorized by
//! authorizing the owning agent's turn is what gets spent.
//!
//! The writes all live in `db::spawn_subspace_tx`, deliberately: the stamp
//! ledger that keeps the pristine-space disposal honest scans `db.rs`, so a
//! `space` write anywhere else would escape it.

use uuid::Uuid;

use crate::db;
use crate::error::AppError;
use crate::{Change, Inner, derive_space_title, now_ms};

/// How deep the `parent_space_id` chain may go. A space nobody spawned is at
/// depth 0, so this admits great-grandchildren and refuses their children.
///
/// A constant rather than a setting: it is a structural guard against runaway
/// delegation, not a preference, and every surveyed agent harness carries one.
pub const MAX_SPAWN_DEPTH: i64 = 3;

/// How many live (non-archived) sub-spaces one owner may hold at once.
/// Archiving one frees a slot, which is also the stated remedy for a
/// delegation that went wrong.
pub const MAX_LIVE_SUBSPACES_PER_OWNER: i64 = 8;

/// How many sub-agents one spawn may seat beside the owner.
///
/// Every seat is written with `override_notify_policy = 'all'`, so each
/// sub-agent answers every post in the room *and* each of those answers
/// notifies all the others: the work a single spawn schedules grows with the
/// square of the roster, and the cascade guard is the only thing that ever
/// stops it. A panel is the point of seating several, but a panel is small —
/// so this shares the live-room guard's register rather than inventing a
/// second scale.
pub const MAX_SUBAGENTS_PER_SPAWN: i64 = 8;

/// Why a spawn was refused.
///
/// Every variant is something the asking agent can act on — narrow the
/// request, finish a sub-space it already has, ask for a capability it holds —
/// so each says what happened in words a model reads without further
/// translation. None of them names anything the asker did not already supply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnRefusal {
    /// The brief was empty. A sub-space with no brief is a room with no
    /// reason.
    EmptyBrief,
    /// The named parent space does not exist.
    UnknownParent { space_id: String },
    /// The parent conversation has been archived, so it is closed to new work
    /// — and a room opened under a closed one is work nobody asked to
    /// continue. Reachable because a turn already in flight when the archival
    /// landed runs to completion by design.
    ParentArchived { space_id: String },
    /// The spawner is not a live, shared agent taking part in the parent
    /// space. A space-owned participant cannot be referenced into another
    /// space at all, so it cannot own one either.
    SpawnerNotEligible { participant_id: String },
    /// The chain is already as deep as it may go.
    TooDeep { depth: i64, limit: i64 },
    /// The owner already holds the maximum number of live sub-spaces.
    TooManyLiveSubspaces { live: i64, limit: i64 },
    /// More sub-agents were asked for in one room than may be seated in one.
    TooManySubagents { requested: i64, limit: i64 },
    /// A participant the spawn would seat — the owner itself, or one of the
    /// sub-agents — carries no model of its own. The sub-space sees each
    /// participant's **base** configuration (a spawn copies no overrides), and
    /// an agent with no model is skipped by every planner, so seating it would
    /// report a room that schedules nothing.
    NoModelConfigured { label: String },
    /// A requested capability is one the parent space does not hold, so there
    /// is nothing to pass down. This is the attenuation gate.
    CapabilityNotHeld { name: String },
    /// A requested sub-agent is not a live shared agent.
    ParticipantNotEligible { participant_id: String },
    /// The post the delegation says it is being opened from is not a post of
    /// the parent conversation. The report attaches there, so an anchor
    /// pointing somewhere else would answer a conversation nobody asked.
    AnchorNotInParent { action_id: String },
    /// No anchor was named and the parent conversation has nothing to attach a
    /// report to. The room would do its work and then have nowhere to say so.
    NothingToReportTo,
}

impl std::fmt::Display for SpawnRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBrief => write!(
                f,
                "a brief is required: say what the work is, what it covers, and what you \
                 need back — the participants there share none of this conversation"
            ),
            Self::UnknownParent { space_id } => {
                write!(f, "no such conversation: {space_id}")
            }
            Self::ParentArchived { .. } => write!(
                f,
                "this conversation has been archived, so it takes no new work — nothing \
                 delegated from it now would be read"
            ),
            Self::SpawnerNotEligible { participant_id } => write!(
                f,
                "{participant_id} is not a shared agent taking part in this conversation, so \
                 it cannot open one of its own"
            ),
            Self::TooDeep { depth, limit } => write!(
                f,
                "that would be {depth} levels of delegation deep and the limit is {limit} — do \
                 this work here, or report back and let the level above delegate it"
            ),
            Self::TooManyLiveSubspaces { live, limit } => write!(
                f,
                "you already have {live} delegated conversations open and the limit is {limit} \
                 — finish and archive one before opening another"
            ),
            Self::TooManySubagents { requested, limit } => write!(
                f,
                "that would seat {requested} participants and one delegated conversation holds \
                 at most {limit} — send fewer, or split the work across conversations"
            ),
            Self::NoModelConfigured { label } => write!(
                f,
                "{label} has no model of its own, so it would never answer there — give it one, \
                 or delegate to an agent that has one"
            ),
            Self::CapabilityNotHeld { name } => write!(
                f,
                "you cannot grant `{name}` because this conversation does not have it; a \
                 delegated conversation never gets more than the one it came from"
            ),
            Self::ParticipantNotEligible { participant_id } => write!(
                f,
                "{participant_id} is not a shared agent that can be invited into another \
                 conversation"
            ),
            Self::AnchorNotInParent { action_id } => write!(
                f,
                "{action_id} is not a post in this conversation, so a delegation opened from it \
                 would have nowhere here to report back to"
            ),
            Self::NothingToReportTo => write!(
                f,
                "there is nothing in this conversation to reply to, so work delegated from it \
                 could never be reported back — say something here first"
            ),
        }
    }
}

/// One sub-space, as a parent or an owner sees it.
#[derive(Clone, Debug)]
pub struct SubspaceInfo {
    pub id: String,
    pub parent_space_id: String,
    /// The agent that spawned it, and is its `role = 'owner'` member.
    pub owner_participant_id: String,
    /// The post in the parent it was opened from, when the spawn named one —
    /// where its report attaches. `None` for a spawn that named none.
    pub parent_action_id: Option<String>,
    pub title: Option<String>,
    pub created_at: i64,
    pub archived_at: Option<i64>,
}

impl From<db::SubspaceRow> for SubspaceInfo {
    fn from(r: db::SubspaceRow) -> Self {
        Self {
            id: r.id,
            parent_space_id: r.parent_space_id,
            owner_participant_id: r.owner_participant_id,
            parent_action_id: r.parent_action_id,
            title: r.title,
            created_at: r.created_at,
            archived_at: r.archived_at,
        }
    }
}

/// What a spawn wrote — enough for the caller to address the new room and
/// report it honestly, with no read after the commit.
#[derive(Clone, Debug)]
pub struct SpawnedSubspace {
    pub space: SubspaceInfo,
    /// The brief, as the sub-space's first post.
    pub brief_action_id: String,
    /// The sub-agents seated beside the owner, in the order requested.
    pub participant_ids: Vec<String>,
    /// The capabilities carried down, by name (empty in practice today).
    pub capabilities: Vec<String>,
}

/// One capability a space holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpaceCapability {
    pub name: String,
    /// Capability-specific JSON; `{}` for a bare grant.
    pub config: String,
}

impl Inner {
    /// Spawn a sub-space of `parent_space_id`, owned by `owner_participant_id`
    /// and opened by `brief`. See the module docs for the shape and
    /// `db::spawn_subspace_tx` for the guards, all of which are decided inside
    /// the writing transaction.
    ///
    /// The only work done before the transaction is the work that cannot race:
    /// trimming the brief, deduping the requested participants, and deriving a
    /// title when none was given. Every refusal leaves zero durable trace and
    /// emits nothing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn spawn_subspace(
        &self,
        parent_space_id: &str,
        owner_participant_id: &str,
        brief: &str,
        participants: &[String],
        capabilities: &[String],
        title: Option<&str>,
        parent_action_id: Option<&str>,
    ) -> Result<SpawnedSubspace, AppError> {
        let brief = brief.trim();
        if brief.is_empty() {
            return Err(AppError::SpawnRefused {
                refusal: SpawnRefusal::EmptyBrief,
            });
        }

        // The owner is seated by the transaction itself, and asking twice for
        // the same agent is a request for one membership, not two.
        let mut seats: Vec<String> = Vec::new();
        for p in participants {
            let p = p.trim();
            if p.is_empty() || p == owner_participant_id || seats.iter().any(|s| s == p) {
                continue;
            }
            seats.push(p.to_string());
        }

        let mut names: Vec<String> = Vec::new();
        for c in capabilities {
            let c = c.trim();
            if c.is_empty() || names.iter().any(|n| n == c) {
                continue;
            }
            names.push(c.to_string());
        }

        // A sub-space is **always** titled: it appears in the Library beside
        // every other conversation, and the human reading it there has no
        // prompt of their own to recognize it by — nor a snippet, since a
        // brief is not what the listing's fallback text reads. An explicit
        // title wins; otherwise the brief's own opening line names it, the
        // same derivation an untitled space's first post gets. A brief that
        // yields no line at all (pure markdown scaffolding, say) is named
        // after its owner inside the transaction, which is where that agent's
        // label is already read.
        let derived = title
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .or_else(|| derive_space_title(brief));

        let conn = self.db_conn().await?;
        let now = now_ms();
        let space_id = crate::new_space_id();
        let brief_action_id = Uuid::now_v7().to_string();
        let brief_item_id = Uuid::now_v7().to_string();

        let plan = db::SubspacePlan {
            space_id: &space_id,
            parent_space_id,
            owner_participant_id,
            title: derived.as_deref(),
            brief,
            brief_action_id: &brief_action_id,
            brief_item_id: &brief_item_id,
            participant_ids: &seats,
            capabilities: &names,
            parent_action_id,
            now,
        };
        let title = match db::spawn_subspace_tx(&conn, &plan).await? {
            Ok(title) => title,
            Err(refusal) => return Err(AppError::SpawnRefused { refusal }),
        };

        // One emission per thing the spawn wrote, mirroring what an
        // instantiation announces: the Library gained a row, the new space has
        // a transcript to read, and a roster was written. The parent is
        // untouched, so nothing is said about it.
        self.bus.emit(Change::SpaceIndex);
        self.bus.emit(Change::Space(space_id.clone()));
        self.bus.emit(Change::Participants);

        Ok(SpawnedSubspace {
            space: SubspaceInfo {
                id: space_id,
                parent_space_id: parent_space_id.to_string(),
                owner_participant_id: owner_participant_id.to_string(),
                parent_action_id: parent_action_id.map(str::to_string),
                title: Some(title),
                created_at: now,
                archived_at: None,
            },
            brief_action_id,
            participant_ids: seats,
            capabilities: names,
        })
    }

    pub(crate) async fn subspaces_of(
        &self,
        parent_space_id: &str,
    ) -> Result<Vec<SubspaceInfo>, AppError> {
        let conn = self.db_conn().await?;
        Ok(db::subspaces_of(&conn, parent_space_id)
            .await?
            .into_iter()
            .map(SubspaceInfo::from)
            .collect())
    }

    pub(crate) async fn live_subspaces_owned_by(
        &self,
        owner_participant_id: &str,
    ) -> Result<Vec<SubspaceInfo>, AppError> {
        let conn = self.db_conn().await?;
        Ok(db::live_subspaces_owned_by(&conn, owner_participant_id)
            .await?
            .into_iter()
            .map(SubspaceInfo::from)
            .collect())
    }

    pub(crate) async fn subspace(&self, space_id: &str) -> Result<Option<SubspaceInfo>, AppError> {
        let conn = self.db_conn().await?;
        Ok(db::subspace(&conn, space_id).await?.map(SubspaceInfo::from))
    }

    pub(crate) async fn space_capabilities(
        &self,
        space_id: &str,
    ) -> Result<Vec<SpaceCapability>, AppError> {
        let conn = self.db_conn().await?;
        Ok(db::space_capabilities(&conn, space_id)
            .await?
            .into_iter()
            .map(|c| SpaceCapability {
                name: c.name,
                config: c.config,
            })
            .collect())
    }
}
