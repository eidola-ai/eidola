//! Invalidation bus — the narrow seam through which every durable write in
//! app-core announces itself to subscribers (GUI stores, CLI, tests).
//!
//! ## Design
//!
//! A [`tokio::sync::broadcast`] channel lives inside [`crate::AppCore`] (via
//! [`Inner`]) as the v1 implementation.  Every write path in [`crate::Inner`]
//! emits exactly one [`Change`] per affected domain **after** its durable
//! commit succeeds; it never emits on error.  Multi-domain operations
//! (e.g. `account_allocate`, which touches both the wallet and the account
//! balance) emit one message per domain, in order, after the last write.
//!
//! ## Lagged receivers
//!
//! `broadcast` drops messages for receivers that fall behind the channel
//! capacity ([`BUS_CAPACITY`]).  A [`tokio::sync::broadcast::error::RecvError::Lagged`]
//! error on a receiver means "you missed at least one change; treat it as
//! stale and refresh everything you care about."  The capacity is sized
//! generously so a slow consumer only lags under extreme write bursts.
//!
//! ## Who wrote it
//!
//! A subscriber receives a [`ChangeEvent`]: the [`Change`] itself, a
//! [`ChangeOrigin`] saying whether anyone is waiting on the write, and a
//! [`ChangeEvent::seq`] saying where it falls in this process's one stream of
//! durable writes.  A consumer that is mid-operation re-reads at its own exit,
//! so a change *that re-read covers* is already accounted for — while a change
//! emitted by work app-core drives on its own has nobody to read it in, and a
//! consumer that drops it while busy loses it for good.
//!
//! **Attendance and coverage are different questions**, and coverage is the one
//! a busy consumer actually needs answered.  `Caller` says a call is
//! outstanding; it never said the call was *this* consumer's, and a
//! caller-stamped write from somewhere else — another window archiving the
//! conversation this one is streaming in — is covered by nothing this consumer
//! is about to do.  The sequence answers coverage exactly: sample
//! [`ChangeSource::current_seq`] before a read, and every event numbered below
//! it committed before that read began.
//!
//! The origin is **ambient rather than an argument**: [`with_origin`] scopes a
//! future, and every emission made from inside it is stamped.  Every write path
//! reaches the bus through the same `emit` it always did, so there is no
//! per-emission decision to get wrong and no way for a new write inside an
//! unattended chore to be stamped as something a caller is waiting for.
//!
//! ## The `ChangeSource` seam (v2 extension point)
//!
//! The [`ChangeSource`] trait is the documented interface between app-core and
//! its consumers.  The v1 implementation is an in-process broadcast.  The
//! intended v2 implementation is **Turso CDC tailing**
//! (`PRAGMA capture_data_changes_conn` → `turso_cdc`), which extends the same
//! [`Change`] stream across processes — bridging the gap when the CLI writes
//! while the GUI is open.  Swapping the implementation requires only
//! replacing the [`ChangeSource`] implementation stored inside [`crate::AppCore`];
//! all subscribers remain unchanged.

use tokio::sync::broadcast;

/// A domain-level change notification emitted by app-core after every durable
/// write.  Consumers subscribe via [`ChangeSource::subscribe`] and refresh
/// the affected domain(s) on receipt.
///
/// Each variant maps 1:1 to a domain store (see `crates/eidola-gui/STATE.md`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// The on-disk config was updated (any field: base URL, account
    /// credentials, default model, trusted measurements, …).
    Config,
    /// Account-level balance or lifecycle changed (allocation moved credits
    /// from the account into a credential).
    Account,
    /// Wallet-level credential changed (issuance, spend start, refund, recovery).
    Wallet,
    /// The space index changed: a space was created, archived, renamed, or
    /// auto-titled.
    SpaceIndex,
    /// Actions or messages within a specific space changed.
    Space(SpaceId),
    /// The attestation / request / spend-trail record was appended to.
    Record,
    /// The update-check state was written (last result, accepted claims).
    UpdateState,
    /// The local-model domain changed: a download started / progressed /
    /// finished / failed, a model was deleted, or an engine was loaded,
    /// became ready, was unloaded, or exited. Subscribers re-snapshot via
    /// `AppCore::local_models_state`. Progress emissions are throttled in
    /// the transfer task, so bursts stay far below [`BUS_CAPACITY`].
    LocalModels,
    /// The backend registry changed: a backend was added, updated,
    /// enabled/disabled, or removed. Subscribers re-snapshot via
    /// `AppCore::list_backends` (and should treat per-backend model
    /// catalogs as stale — the set of destinations changed).
    Backends,
    /// A space's participant membership/config changed: a participant was
    /// added to, edited in, or removed from a space (the per-space
    /// Participants surface). Subscribers re-snapshot the affected space's
    /// participants via `AppCore::list_space_participants`. Distinct from
    /// [`Change::Templates`] (the reusable blueprints) — the two map to two
    /// GUI surfaces (the per-space Participants view vs. the Space Templates
    /// settings pane), so a template edit never over-invalidates a space's
    /// live participant list and vice versa (the STATE.md 1:1 variant↔store
    /// rule). Carries no id: the per-space participant list is small and the
    /// active surface re-snapshots on receipt.
    Participants,
    /// The space-template registry changed: a template was created, edited,
    /// removed (soft), or set as the default. Subscribers re-snapshot via
    /// `AppCore::list_space_templates`. `Change::Config` is emitted
    /// *additionally* when set-as-default writes the `default_template`
    /// config key.
    Templates,
}

/// Identifies a single conversation space.  String form matches the UUIDs
/// stored in the `space` table.
pub type SpaceId = String;

/// Whether anyone is waiting on the write a [`Change`] announces.
///
/// The distinction exists for consumers that own an operation's truth while it
/// runs: such a consumer wants to know whether a write is one *somebody's* exit
/// re-read will pick up.  **It says attendance, and attendance is not
/// ownership**: `Caller` means a call was outstanding, not that the call was
/// the reading consumer's, so a busy surface that treated every `Caller` as its
/// own dropped another window's write about the same conversation.  Which of
/// the deferred writes a consumer's own read already covered is
/// [`ChangeEvent::seq`]'s question, and it is the one a busy surface should
/// ask; this one still says, cheaply and up front, whether *anyone* is going to
/// read a write in at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChangeOrigin {
    /// Emitted while serving a call somebody made.  Some caller's return —
    /// reload, re-list, re-render — comes after this write, though not
    /// necessarily any particular consumer's.
    Caller,
    /// Emitted by work app-core drives on its own: a background chore, or a
    /// turn driver giving a windowless conversation its turns.  No call is
    /// outstanding, so nothing else is going to read this write in — a
    /// consumer that drops it while busy loses it until something unrelated
    /// invalidates the same surface.
    Unattended,
}

/// One bus message: a [`Change`], the [`ChangeOrigin`] it was emitted under,
/// and the [`ChangeEvent::seq`] that says **when**.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChangeEvent {
    pub change: Change,
    pub origin: ChangeOrigin,
    /// Where this event falls in the process's single stream of durable
    /// writes — assigned at emission, and therefore *after* the commit it
    /// announces.
    ///
    /// **It exists so a consumer can ask whether one of its own reads already
    /// covers a write**, which is a sharper question than who made the write.
    /// A read that sampled [`ChangeSource::current_seq`] before it began sees
    /// every commit whose event carries a lower number, because the event
    /// follows the commit; one numbered at or above the sample may have
    /// committed while the read was running, and only that case needs
    /// re-reading. Consumers that own a surface's truth while an operation runs
    /// used to answer this with [`ChangeOrigin`] alone, which answers "is
    /// anyone waiting on it" and not "is it mine" — let alone "is it covered".
    pub seq: u64,
}

tokio::task_local! {
    /// The origin stamped onto every change emitted from inside this task.
    ///
    /// Ambient rather than threaded through ~30 emission sites and every
    /// function between them: the question "is a caller waiting on this?" is a
    /// property of *why this code is running*, not of the individual write, and
    /// the answer is the same for every write a chore makes.  Task-scoped, so
    /// concurrent turns cannot read each other's answer.
    static ORIGIN: ChangeOrigin;
}

/// Run `f` with every [`Change`] it emits stamped `origin`.
///
/// Scoping is per task: work moved onto a *new* task inside `f` (a `spawn`)
/// starts outside the scope again and falls back to [`ChangeOrigin::Caller`],
/// which is the safe direction — a mis-stamped `Unattended` would make a
/// consumer re-read for nothing, but a mis-stamped `Caller` is a dropped
/// invalidation, and the fallback never produces the second.
pub async fn with_origin<F: std::future::Future>(origin: ChangeOrigin, f: F) -> F::Output {
    ORIGIN.scope(origin, f).await
}

/// The origin in force for the current task, or [`ChangeOrigin::Caller`]
/// outside any scope (which is every ordinary consumer call, and any emission
/// made off a tokio task at all).
fn current_origin() -> ChangeOrigin {
    ORIGIN.try_with(|o| *o).unwrap_or(ChangeOrigin::Caller)
}

/// Broadcast capacity for the invalidation bus.  Slow receivers that fall
/// behind by more than this many messages will receive a
/// [`tokio::sync::broadcast::error::RecvError::Lagged`] error — callers
/// should treat that as "refresh everything".
pub const BUS_CAPACITY: usize = 256;

/// The narrow seam between app-core and its change subscribers.
///
/// The v1 implementation is an in-process [`broadcast`] channel.
/// The documented v2 seam is Turso CDC tailing; see module-level docs.
pub trait ChangeSource {
    /// Returns a new receiver that will see all [`ChangeEvent`] messages
    /// emitted from this point forward.  The receiver is independent of all
    /// other receivers — dropping it does not affect the channel or other
    /// subscribers.
    fn subscribe(&self) -> broadcast::Receiver<ChangeEvent>;

    /// The watermark: **every event emitted so far carries a `seq` strictly
    /// below this**, and the next emission takes this number.  A consumer
    /// samples it *before* issuing a read, so it can later tell which of the
    /// events it deferred that read had already covered.
    ///
    /// Sampling before the read is what makes the claim true rather than
    /// approximately true: an event numbered below the sample was emitted
    /// before the read began, and an emission follows its commit, so the read
    /// saw that write.  Anything numbered at or above it may have committed
    /// mid-read, which is the case that has to be re-read — and the same
    /// answer, conservatively, for anything that merely raced the sample.
    fn current_seq(&self) -> u64;
}

/// In-process broadcast implementation of [`ChangeSource`].
///
/// Owned by [`crate::Inner`] and cloned into [`crate::AppCore`] so that
/// `AppCore` can hand out receivers while `Inner` (running on the tokio
/// runtime) holds the [`broadcast::Sender`] for emission.
#[derive(Clone)]
pub struct BroadcastSource {
    sender: broadcast::Sender<ChangeEvent>,
    /// The number the next emission takes.  Shared with every clone, because
    /// there is one stream of writes per process however many handles hand it
    /// out.
    seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl BroadcastSource {
    /// Create a new broadcast bus with [`BUS_CAPACITY`] slots.
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(BUS_CAPACITY);
        Self {
            sender,
            seq: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Emit a change, stamped with the origin in force for this task (see
    /// [`with_origin`]) and the next sequence number.  Silently succeeds when
    /// there are no active receivers (the `send` error variant means "no
    /// receivers", not a failure worth propagating to the write path).
    ///
    /// **The number is taken even when nobody is listening**, so a consumer
    /// that subscribes later cannot be handed a watermark that a dropped
    /// message has already passed.
    pub fn emit(&self, change: Change) {
        let seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = self.sender.send(ChangeEvent {
            change,
            origin: current_origin(),
            seq,
        });
    }
}

impl ChangeSource for BroadcastSource {
    fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.sender.subscribe()
    }

    fn current_seq(&self) -> u64 {
        self.seq.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Default for BroadcastSource {
    fn default() -> Self {
        Self::new()
    }
}
