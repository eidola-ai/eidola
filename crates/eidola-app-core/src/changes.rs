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
    /// Returns a new receiver that will see all [`Change`] messages emitted
    /// from this point forward.  The receiver is independent of all other
    /// receivers — dropping it does not affect the channel or other
    /// subscribers.
    fn subscribe(&self) -> broadcast::Receiver<Change>;
}

/// In-process broadcast implementation of [`ChangeSource`].
///
/// Owned by [`crate::Inner`] and cloned into [`crate::AppCore`] so that
/// `AppCore` can hand out receivers while `Inner` (running on the tokio
/// runtime) holds the [`broadcast::Sender`] for emission.
#[derive(Clone)]
pub struct BroadcastSource {
    sender: broadcast::Sender<Change>,
}

impl BroadcastSource {
    /// Create a new broadcast bus with [`BUS_CAPACITY`] slots.
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(BUS_CAPACITY);
        Self { sender }
    }

    /// Emit a change.  Silently succeeds when there are no active receivers
    /// (the `send` error variant means "no receivers", not a failure worth
    /// propagating to the write path).
    pub fn emit(&self, change: Change) {
        let _ = self.sender.send(change);
    }
}

impl ChangeSource for BroadcastSource {
    fn subscribe(&self) -> broadcast::Receiver<Change> {
        self.sender.subscribe()
    }
}

impl Default for BroadcastSource {
    fn default() -> Self {
        Self::new()
    }
}
