//! `ProxyStore` — the local inference proxy's configuration **and** the
//! listener that configuration describes.
//!
//! It owns both because they are one question asked twice: the reader's stored
//! intent ("listen on 127.0.0.1:11437, expose these backends") and whether a
//! socket is actually bound. Keeping them apart would let the pane report an
//! address nothing is listening on — which is the one thing a surface that
//! tells a person where to point their tool must never do. So the store
//! reconciles the listener against the settings on every refresh, and the pane
//! reads the **bound** address rather than the stored one.
//!
//! Refreshed at launch and on every `Change::Proxy`. Writes go through
//! app-core; a refusal lands in `op_errors` under the operation it refused
//! (see [`ProxyOp`]), and a bind failure — which is a different kind of
//! failure, since nothing the reader typed was wrong — lands in its own
//! `listen_error` beside them.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::proxy::{
    LocalExposure, MintedProxyKey, ProxyKeyInfo, ProxySettings, ProxySettingsUpdate,
    parse_bind_address,
};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;
use crate::proxy::ProxyHandle;

/// Why nothing is listening where the reader asked for something to be.
///
/// Two different facts, kept apart because they want different sentences: the
/// operating system refused the bind (a typed [`AppError`], usually
/// `ProxyRefusal::CannotListen`), or a listener that was running gave up
/// accepting (`reason` is this crate's own diagnostic of the loop's last
/// failure).
#[derive(Clone, Debug)]
pub enum ListenFailure {
    Refused(AppError),
    StoppedAccepting(String),
}

/// One operation the pane can start — **the key its slot, its refusal and (for
/// a settings write) its rollback are all filed under.**
///
/// A key names a *control*, not the surface (the keyed-slots rule): the pane
/// offers a switch, a binding, an exposure choice, a checkbox per backend and a
/// revoke per key all at once, so each is its own operation and none of them
/// can replace, clear or report for another. Typed rather than a string so the
/// pane asks for a control's refusal by the same value the store filed it
/// under, and a misspelt key is a compile error rather than a band that never
/// paints.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ProxyOp {
    Enabled,
    Binding,
    Exposure,
    Backend(String),
    /// The one slot a key generation ever occupies — the re-entry refusal and
    /// the operation itself have to mean the same slot.
    CreateKey,
    Revoke(String),
}

/// What one settings write assigns — and, read back from a snapshot, what that
/// assignment displaced.
///
/// **Every settings operation is an assignment to columns only its own key
/// writes**: the switch owns `enabled`, the binding owns the address and port,
/// the exposure choice owns `local_exposure`, and a backend's checkbox owns that
/// one id's membership of `exposed_ids`. So the inverse of an edit is the same
/// edit carrying the value it displaced — never a snapshot of the whole row,
/// which would also take back a differently-keyed sibling's pending edit (the
/// `SpacesStore` argument for inverses over snapshots, reached by the other
/// door). The backend arm restores *membership*, not position: an id's place in
/// the list is not something the reader chose, and the resolving read puts the
/// database's order back.
#[derive(Clone, Debug, PartialEq)]
pub enum SettingsEdit {
    Enabled(bool),
    Binding { address: String, port: u16 },
    Exposure(LocalExposure),
    Backend { id: String, exposed: bool },
}

impl SettingsEdit {
    fn apply(&self, settings: &mut ProxySettings) {
        match self {
            SettingsEdit::Enabled(enabled) => settings.enabled = *enabled,
            SettingsEdit::Binding { address, port } => {
                settings.bind_address = address.clone();
                settings.bind_port = *port;
            }
            SettingsEdit::Exposure(exposure) => settings.local_exposure = *exposure,
            SettingsEdit::Backend { id, exposed } => {
                settings.exposed_ids.retain(|b| b != id);
                if *exposed {
                    settings.exposed_ids.push(id.clone());
                }
            }
        }
    }

    /// The edit that puts these same columns back to what `settings` holds.
    fn displaced_in(&self, settings: &ProxySettings) -> Self {
        match self {
            SettingsEdit::Enabled(_) => SettingsEdit::Enabled(settings.enabled),
            SettingsEdit::Binding { .. } => SettingsEdit::Binding {
                address: settings.bind_address.clone(),
                port: settings.bind_port,
            },
            SettingsEdit::Exposure(_) => SettingsEdit::Exposure(settings.local_exposure),
            SettingsEdit::Backend { id, .. } => SettingsEdit::Backend {
                id: id.clone(),
                exposed: settings.exposed_ids.contains(id),
            },
        }
    }
}

/// Whether a landing operation is still the one its slot is waiting for.
///
/// A superseded op is chained ahead of its successor, so it lands first — and
/// it still has one thing to say even though it reports nothing: whether its
/// write reached the database, which is what its successor's rollback has to
/// restore to (see [`ProxyStore::settle_settings`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    Current,
    Superseded,
}

pub struct ProxyStore {
    app_core: Option<Arc<AppCore>>,
    settings: Loadable<ProxySettings>,
    keys: Loadable<Vec<ProxyKeyInfo>>,
    /// The listener, shared with the full-shutdown hook.
    handle: ProxyHandle,
    /// Supersede slots for the two reads.
    settings_task: Option<Task<()>>,
    keys_task: Option<Task<()>>,
    /// Keyed per-operation slots — the pane offers several verbs at once
    /// (a toggle, a per-backend checkbox, a per-key revoke), and a store-wide
    /// slot would drop one of them. Two writes on the **same** key are
    /// **chained**, not replaced; the generation beside each task is what tells
    /// a settling op whether it is still the current one. See
    /// [`Self::start_op`].
    op_tasks: HashMap<ProxyOp, (u64, Task<()>)>,
    next_op_gen: u64,
    /// **Keyed exactly as the slots are**, so two refused controls both stand,
    /// a write to one clears only its own report, and each is dismissed on its
    /// own (`AgentsStore`'s keyed reports). One shared slot let the last
    /// refusal erase the other and any write anywhere clear both.
    ///
    /// **The typed error, never its text** — the pane chooses the words at
    /// render (`proxy_settings::refusal_copy`), so a locale change repaints a
    /// refusal already on screen and a localized pane never prints this
    /// crate's English (the "state holds the value, render chooses the words"
    /// rule).
    op_errors: HashMap<ProxyOp, AppError>,
    /// **What each settings key's columns hold as far as the database is known
    /// to agree** — present exactly while a chain of writes on that key is in
    /// flight. Captured from the cache when the chain's first edit leaves
    /// (nothing else on that key is optimistic then), advanced when a
    /// superseded write in the chain lands accepted, and restored when the
    /// chain's current write is refused. See [`Self::settle_settings`].
    restore: HashMap<ProxyOp, SettingsEdit>,
    /// Why the listener is not running, when the reader asked for it to be.
    /// Separate from `op_errors` because the write succeeded — what failed is
    /// the socket, and the two want different words. Typed for the same
    /// reason `op_errors` is.
    listen_error: Option<AppError>,
    /// A key at the one moment it exists in full. Held until the reader
    /// dismisses it, because there is no second chance to show it.
    minted: Option<MintedProxyKey>,
}

impl ProxyStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            settings: Loadable::NotLoaded,
            keys: Loadable::NotLoaded,
            handle: ProxyHandle::default(),
            settings_task: None,
            keys_task: None,
            op_tasks: HashMap::new(),
            next_op_gen: 0,
            op_errors: HashMap::new(),
            restore: HashMap::new(),
            listen_error: None,
            minted: None,
        }
    }

    /// A stub store with fixture state (tests). It has no core, so nothing it
    /// is told to do ever binds a socket.
    pub fn stub(settings: Option<ProxySettings>, keys: Vec<ProxyKeyInfo>) -> Self {
        Self {
            app_core: None,
            settings: match settings {
                Some(settings) => Loadable::loaded(settings),
                None => Loadable::NotLoaded,
            },
            keys: if keys.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(keys)
            },
            handle: ProxyHandle::default(),
            settings_task: None,
            keys_task: None,
            op_tasks: HashMap::new(),
            next_op_gen: 0,
            op_errors: HashMap::new(),
            restore: HashMap::new(),
            listen_error: None,
            minted: None,
        }
    }

    /// Test-only: install a fixture settings snapshot.
    #[doc(hidden)]
    pub fn set_settings_for_test(&mut self, settings: Loadable<ProxySettings>) {
        self.settings = settings;
    }

    /// Test-only: install a fixture key listing.
    #[doc(hidden)]
    pub fn set_keys_for_test(&mut self, keys: Loadable<Vec<ProxyKeyInfo>>) {
        self.keys = keys;
    }

    /// Test-only: stand a minted key in front of the reader. A stub store has
    /// no core to generate one with, and what the pane does *while* one stands
    /// is exactly what wants pinning.
    #[doc(hidden)]
    pub fn set_minted_for_test(&mut self, minted: MintedProxyKey) {
        self.minted = Some(minted);
    }

    pub fn settings(&self) -> &Loadable<ProxySettings> {
        &self.settings
    }

    pub fn keys(&self) -> &Loadable<Vec<ProxyKeyInfo>> {
        &self.keys
    }

    /// Every key the listing holds (empty unless it answered).
    pub fn key_list(&self) -> &[ProxyKeyInfo] {
        self.keys.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// The handle, for the full-shutdown hook.
    pub fn handle(&self) -> ProxyHandle {
        self.handle.clone()
    }

    /// Where a downstream tool should point, or `None` when nothing is
    /// listening. **The listener's own answer**, never the stored setting.
    pub fn address(&self) -> Option<SocketAddr> {
        self.handle.address()
    }

    pub fn is_running(&self) -> bool {
        self.handle.is_running()
    }

    /// The refusal standing for one control, if its last write was refused.
    pub fn op_error(&self, op: &ProxyOp) -> Option<&AppError> {
        self.op_errors.get(op)
    }

    /// Whether any control's refusal is standing — what a test waiting for a
    /// batch to finish one way or the other asks.
    pub fn any_op_error(&self) -> bool {
        !self.op_errors.is_empty()
    }

    /// Test seam: stand a refusal for one control, as a refused write would.
    #[doc(hidden)]
    pub fn set_op_error_for_test(&mut self, op: ProxyOp, error: AppError, cx: &mut Context<Self>) {
        self.op_errors.insert(op, error);
        cx.notify();
    }

    /// Forget the refusals whose control the pane no longer has.
    ///
    /// A refusal is rendered under the control it was about, so one whose
    /// backend was removed, or whose key the listing no longer carries, has no
    /// place to stand and nothing left to say. **Only a listing that has
    /// answered can say a row is absent**, so each side is asked only when its
    /// caller has one — `None` leaves that side's refusals alone. The pane tells
    /// the store rather than the store deciding for itself, the Agents pane's
    /// shape (`AgentsStore::forget_op_errors_absent_from`).
    pub fn forget_op_errors_absent_from(
        &mut self,
        backends: Option<&[String]>,
        keys: Option<&[String]>,
    ) {
        self.op_errors.retain(|op, _| match op {
            ProxyOp::Backend(id) => backends.is_none_or(|ids| ids.contains(id)),
            ProxyOp::Revoke(id) => keys.is_none_or(|ids| ids.contains(id)),
            _ => true,
        });
    }

    /// Whether any write is still in flight. What the resolving read waits for,
    /// and the one honest thing a test can wait on: the database row moves
    /// before the continuation that adopts it does.
    pub fn writing(&self) -> bool {
        !self.op_tasks.is_empty()
    }

    /// Why nothing is listening, when the reader asked for something to be.
    ///
    /// **Derived, never only cached.** A listener whose accept loop gave up
    /// stops answering an address without any write having failed, so the
    /// handle's own reason outranks the last bind failure recorded here: the
    /// pane's question is "why is nothing listening", and the loop that stopped
    /// is the truest answer available. A bind that failed left no listener at
    /// all, so the two can never both answer.
    pub fn listen_error(&self) -> Option<ListenFailure> {
        self.handle
            .accept_failure()
            .map(ListenFailure::StoppedAccepting)
            .or_else(|| self.listen_error.clone().map(ListenFailure::Refused))
    }

    /// Test seam: stand a listen failure in the slot, as a refused bind would.
    #[doc(hidden)]
    pub fn set_listen_error_for_test(&mut self, error: AppError, cx: &mut Context<Self>) {
        self.listen_error = Some(error);
        cx.notify();
    }

    /// The key just generated, if one is waiting to be read.
    pub fn minted(&self) -> Option<&MintedProxyKey> {
        self.minted.as_ref()
    }

    /// Whether generating a key is something the reader can ask for now.
    ///
    /// **The invariant is "no live row whose secret was never shown."** A key's
    /// value exists for exactly one render — only its digest is stored — so a
    /// second generation while one is pending, or while a minted key still
    /// stands unacknowledged, would insert a live row whose secret nothing ever
    /// displayed: the banner shows one key, and the other is a credential
    /// authenticating requests that nobody holds and nobody can revoke by
    /// recognising it. Both states therefore withhold the verb, and this one
    /// predicate decides the press *and* whether the control is painted at all,
    /// so an accepted press and an offered verb cannot disagree.
    pub fn can_create_key(&self) -> bool {
        self.minted.is_none() && !self.op_tasks.contains_key(&ProxyOp::CreateKey)
    }

    /// Whether a key is being generated right now — the pending half of
    /// [`Self::can_create_key`], which the pane words differently from the
    /// unacknowledged half.
    pub fn create_key_pending(&self) -> bool {
        self.op_tasks.contains_key(&ProxyOp::CreateKey)
    }

    /// Acknowledge the generated key. **The value is gone after this** — only
    /// its digest was ever stored.
    pub fn dismiss_minted(&mut self, cx: &mut Context<Self>) {
        if self.minted.take().is_some() {
            cx.notify();
        }
    }

    /// Acknowledge one control's refusal. The others stand — each is a fact
    /// about a different control.
    pub fn clear_op_error(&mut self, op: &ProxyOp, cx: &mut Context<Self>) {
        if self.op_errors.remove(op).is_some() {
            cx.notify();
        }
    }

    /// Re-read the settings and the key listing, then bring the listener into
    /// line with what the settings say.
    ///
    /// **A refresh defers to a write in flight.** Every write emits its own
    /// `Change::Proxy`, which comes back through the bus as a refresh — so a
    /// read issued while a *second* write is still travelling reads the
    /// database between the two and lands after the second settled, replacing
    /// what the reader last asked for with the value they took back. The
    /// deferral loses nothing, because the last write of a batch always takes
    /// the resolving read on its way out (`start_op`): "applied last" is not
    /// "read last", and issuing the read once `op_tasks` is empty is what makes
    /// the difference a property of *when* it is taken.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        if !self.op_tasks.is_empty() {
            return;
        }
        self.settings = std::mem::take(&mut self.settings).to_loading();
        let settings_core = core.clone();
        self.settings_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(settings_core, |c| async move { c.proxy_settings().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.settings_task = None;
                this.land_settings_read(result, cx);
            });
        }));

        self.keys = std::mem::take(&mut self.keys).to_loading();
        self.keys_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.proxy_keys().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.keys = std::mem::take(&mut this.keys).resolve(result);
                this.keys_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// A settings read's landing: resolve the cell, then reconcile the listener
    /// against whatever the cell now holds.
    ///
    /// **A failed read reconciles against the value it kept** — `Failed {
    /// prior }` is what the pane shows, so it is what the listener follows.
    /// That is honest only because no refused edit is ever still standing in
    /// that value by the time this runs: the resolving read is issued once every
    /// write has landed, and each landed write has either been confirmed or had
    /// its edit taken back ([`Self::settle_settings`]).
    fn land_settings_read(
        &mut self,
        result: Result<ProxySettings, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.settings = std::mem::take(&mut self.settings).resolve(result);
        self.reconcile_listener(cx);
        cx.notify();
    }

    /// Test seam: land a settings read as `refresh`'s continuation would — the
    /// way a *failed* resolving read is staged, since a real core's read does
    /// not fail on demand.
    #[doc(hidden)]
    pub fn land_settings_read_for_test(
        &mut self,
        result: Result<ProxySettings, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.land_settings_read(result, cx);
    }

    /// Bring the listener into line with the settings.
    ///
    /// **Reconciliation, not a command.** The reader's intent lives in the
    /// database and this is the one place that acts on it, so a bus-driven
    /// change made in another window moves this process's socket exactly as a
    /// click in this one does — and a restart is idempotent, because the
    /// question asked is "is what is bound what the settings describe" rather
    /// than "did something just change".
    ///
    /// **A listener that gave up is reconcilable again, and that falls out of
    /// the same question.** The accept loop stops after sixteen consecutive
    /// refused accepts, and a stopped loop answers no address — so the
    /// comparison below misses, and this restarts it. Nothing special-cases the
    /// terminal state; what makes it recoverable is that the handle stops
    /// claiming an address the moment it stops accepting on one.
    fn reconcile_listener(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        // A settings read that has not answered says nothing about what should
        // be listening, so nothing is started or stopped on it.
        let Some(settings) = self.settings.value().cloned() else {
            return;
        };
        if !settings.enabled {
            self.handle.stop();
            self.listen_error = None;
            cx.notify();
            return;
        }
        // **Compared as values, never as text.** `SocketAddr`'s own `Display`
        // brackets an IPv6 host (`[::1]:11437`) and a naive `host:port` join
        // does not, so an IPv6 binding — `::1` is explicitly supported — never
        // matched what was bound: every refresh treated a correct listener as
        // wrong and restarted it, and since a restart closes before it binds,
        // any unrelated invalidation could leave the endpoint stopped when the
        // old socket had not been released yet. An address that does not parse
        // falls through to `start`, which is the one place that reports why.
        let wanted = parse_bind_address(&settings.bind_address)
            .ok()
            .map(|ip| SocketAddr::new(ip, settings.bind_port));
        if wanted.is_some() && self.handle.address() == wanted {
            return;
        }
        match self.handle.start(&core, &settings) {
            Ok(_) => self.listen_error = None,
            Err(e) => {
                // `start` closed the running listener before it tried to bind,
                // so a refused address leaves the proxy **stopped** rather than
                // still answering on the old one: a reader who changed the
                // binding must not be told it moved when it did not.
                self.listen_error = Some(e);
            }
        }
        cx.notify();
    }

    /// **Advance the cached snapshot as the write leaves** (STATE.md's
    /// derived-value rule, the shape `SpaceSettingsStore`'s steppers take) —
    /// and remember what the edit displaced, because the failure exit owes it
    /// back.
    ///
    /// Every control in this pane derives its *next* press from what it
    /// renders — the switch writes `!enabled`, a checkbox writes `!exposed` —
    /// so leaving the snapshot at the stored value until the round trip
    /// settles makes two presses one press: start-then-stop persisted as
    /// start, because both handlers read the same stale `false` and both wrote
    /// `true`. Chaining the writes was never the answer to that: they were
    /// sequenced correctly and carried the same value. What has to move is the
    /// value the second press is derived *from*.
    ///
    /// **An optimistic edit is a promise the failure exit has to keep.** The
    /// resolving read cannot be relied on to take a refused edit back, because
    /// it can fail too — and `Failed { prior }` then preserves the edit, which
    /// here is not just a wrong picture: [`Self::reconcile_listener`] acts on
    /// it, so a refused enable started the listener against the database's old
    /// intent and kept it running until something unrelated re-read. So the
    /// displaced value is captured **once per chain** — when the first edit on
    /// this key leaves, while nothing else on these columns is optimistic — and
    /// a successor's edit leaves it alone, since what it displaces is only its
    /// predecessor's optimism.
    fn begin_settings_edit(&mut self, op: &ProxyOp, edit: &SettingsEdit) {
        if let Some(settings) = self.settings.value_mut() {
            self.restore
                .entry(op.clone())
                .or_insert_with(|| edit.displaced_in(settings));
            edit.apply(settings);
        }
    }

    /// Test seam: make a settings edit leave without a write behind it, so a
    /// test can stage that write's landing itself ([`Self::settle_for_test`]).
    #[doc(hidden)]
    pub fn begin_settings_edit_for_test(&mut self, op: ProxyOp, edit: SettingsEdit) {
        self.begin_settings_edit(&op, &edit);
    }

    /// Turn the proxy on or off.
    pub fn set_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.write_settings(
            ProxyOp::Enabled,
            SettingsEdit::Enabled(enabled),
            ProxySettingsUpdate {
                enabled: Some(enabled),
                ..Default::default()
            },
            cx,
        );
    }

    /// Move the binding. A restart, so a refusal by the OS leaves the proxy
    /// stopped and says why in `listen_error`.
    pub fn set_binding(&mut self, address: String, port: u16, cx: &mut Context<Self>) {
        self.write_settings(
            ProxyOp::Binding,
            SettingsEdit::Binding {
                address: address.clone(),
                port,
            },
            ProxySettingsUpdate {
                bind_address: Some(address),
                bind_port: Some(port),
                ..Default::default()
            },
            cx,
        );
    }

    /// Choose what an engine-backed backend offers.
    pub fn set_local_exposure(&mut self, exposure: LocalExposure, cx: &mut Context<Self>) {
        self.write_settings(
            ProxyOp::Exposure,
            SettingsEdit::Exposure(exposure),
            ProxySettingsUpdate {
                local_exposure: Some(exposure),
                ..Default::default()
            },
            cx,
        );
    }

    /// Expose or withdraw one backend.
    pub fn set_backend_exposed(&mut self, id: String, exposed: bool, cx: &mut Context<Self>) {
        let op = ProxyOp::Backend(id.clone());
        // The checkbox derives its next press from `exposed_ids`, so the same
        // rule the switch takes applies here — see [`Self::begin_settings_edit`].
        let edit = SettingsEdit::Backend {
            id: id.clone(),
            exposed,
        };
        self.begin_settings_edit(&op, &edit);
        let landed = op.clone();
        self.start_op(
            op,
            cx,
            move |core| async move {
                bridge(core, move |c| async move {
                    c.set_proxy_backend_exposed(id, exposed).await
                })
                .await
            },
            move |this, landing, result, cx| {
                this.settle_settings(landed, edit, landing, result, cx)
            },
        );
    }

    /// Generate a key. **Its secret reaches `minted` and nowhere else.**
    ///
    /// Refused while another generation is pending or while a minted key is
    /// still waiting to be read — see [`Self::can_create_key`] for why that is
    /// an invariant about live rows rather than a courtesy to the reader.
    pub fn create_key(&mut self, label: String, cx: &mut Context<Self>) {
        if !self.can_create_key() {
            return;
        }
        self.start_op(
            ProxyOp::CreateKey,
            cx,
            move |core| async move {
                bridge(core, |c| async move { c.create_proxy_key(label).await }).await
            },
            // The listing the new row belongs in is not read here: the
            // batch-end read below covers it, and covers it *after* every write
            // still in flight rather than after this one.
            |this, landing, result, _cx| match (landing, result) {
                (Landing::Superseded, _) => {}
                (Landing::Current, Ok(minted)) => this.minted = Some(minted),
                (Landing::Current, Err(e)) => {
                    this.op_errors.insert(ProxyOp::CreateKey, e);
                }
            },
        );
    }

    /// Revoke a key. Keyed per key, because the pane offers the verb on every
    /// row at once and a store-wide slot would drop one of two presses.
    pub fn revoke_key(&mut self, id: String, cx: &mut Context<Self>) {
        let op = ProxyOp::Revoke(id.clone());
        let landed = op.clone();
        self.start_op(
            op,
            cx,
            move |core| async move {
                bridge(core, |c| async move { c.revoke_proxy_key(id).await }).await
            },
            move |this, landing, result, _cx| {
                if let (Landing::Current, Err(e)) = (landing, result) {
                    this.op_errors.insert(landed, e);
                }
            },
        );
    }

    /// A write through `update_proxy_settings`, which is column-partial: the
    /// update names exactly the columns `edit` assigns, so two different
    /// controls' writes never contend.
    fn write_settings(
        &mut self,
        op: ProxyOp,
        edit: SettingsEdit,
        update: ProxySettingsUpdate,
        cx: &mut Context<Self>,
    ) {
        self.begin_settings_edit(&op, &edit);
        let landed = op.clone();
        self.start_op(
            op,
            cx,
            move |core| async move {
                bridge(
                    core,
                    |c| async move { c.update_proxy_settings(update).await },
                )
                .await
            },
            move |this, landing, result, cx| {
                this.settle_settings(landed, edit, landing, result, cx)
            },
        );
    }

    /// Start a keyed operation, **chained behind any predecessor on that key**.
    ///
    /// Dropping a predecessor's `Task` cancels only its gpui half — `bridge`
    /// leaves the core write running — so a superseded write could reach the
    /// database *after* its successor, or, unpolled, never run at all. Here that
    /// is not hypothetical: two presses of the enable switch, or a binding
    /// change over a switch still settling, are one keyboard's work apart, and
    /// the column each writes is the one the pane then reads back. Owning the
    /// predecessor and awaiting it makes the successor start strictly after that
    /// round trip, so last-wins is true by sequencing rather than by hope
    /// (`AgentsStore::write_then_settle`'s rule, and app-core's column-partial
    /// write is the other half — two *different* columns never contend at all).
    ///
    /// **Only the current generation reports.** A superseded op removes no slot
    /// — the slot is its successor's by then, and dropping it would cancel the
    /// very write it just sequenced — and records no refusal, since its
    /// successor's start already cleared the key's report and the outcome the
    /// reader waits on is the newer one's. It still lands, as
    /// [`Landing::Superseded`], because whether its write reached the database
    /// is what the chain's rollback has to restore to.
    fn start_op<T, Fut>(
        &mut self,
        op_key: ProxyOp,
        cx: &mut Context<Self>,
        op: impl FnOnce(Arc<AppCore>) -> Fut + 'static,
        settle: impl FnOnce(&mut Self, Landing, T, &mut Context<Self>) + 'static,
    ) where
        T: 'static,
        Fut: std::future::Future<Output = T> + 'static,
    {
        let Some(core) = self.begin_op(&op_key) else {
            return;
        };
        // Take over the read for the duration of the write: a refresh already
        // travelling would otherwise land after this settles, carrying a
        // snapshot from before the write. The debt is discharged below, where
        // the last write of the batch issues the resolving read.
        self.settings_task = None;
        self.keys_task = None;
        self.next_op_gen += 1;
        let generation = self.next_op_gen;
        let previous = self.op_tasks.remove(&op_key).map(|(_, task)| task);
        let slot = op_key.clone();
        let task = cx.spawn(async move |this, cx| {
            if let Some(previous) = previous {
                previous.await;
            }
            let result = op(core).await;
            let _ = this.update(cx, |this, cx| {
                if this.op_tasks.get(&slot).map(|(g, _)| *g) != Some(generation) {
                    settle(this, Landing::Superseded, result, cx);
                    return;
                }
                this.op_tasks.remove(&slot);
                settle(this, Landing::Current, result, cx);
                // **The resolving read is taken after the last write.** Every
                // op adopts or reports its own outcome; only once nothing is
                // still writing does the store ask the database what it now
                // says — and `refresh` re-arms the rule, since a write starting
                // during that read defers it again.
                if this.op_tasks.is_empty() {
                    this.refresh(cx);
                }
                cx.notify();
            });
        });
        self.op_tasks.insert(op_key, (generation, task));
        cx.notify();
    }

    /// Test seam: land a settings write as its completion would, so the
    /// decision below can be held while another slot is genuinely occupied —
    /// an interleaving two spawned writes cannot be made to take on demand.
    /// No slot is removed: the staged write never had one.
    #[doc(hidden)]
    pub fn settle_for_test(
        &mut self,
        op: ProxyOp,
        edit: SettingsEdit,
        landing: Landing,
        result: Result<ProxySettings, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.settle_settings(op, edit, landing, result, cx);
    }

    /// A settings write's landing — **one rule for every exit, and the
    /// optimistic edit's promise kept on each**:
    ///
    /// - **Accepted, and the last write of the batch**: adopt what the core
    ///   answered — the resolved settings, not what was asked for — and
    ///   reconcile the listener against it.
    /// - **Accepted while a sibling is still writing**: adopt nothing. An
    ///   answer is a whole snapshot of the database at the moment *that* write
    ///   committed, so adopting it overwrites the sibling's optimistic delta
    ///   with a row that predates it — an exposure checkbox back to unchecked
    ///   while its own write is on its way to making it true. Worse than a
    ///   flicker, because the pane derives its next press from what it
    ///   renders: taking the choice back then reads the reverted checkbox and
    ///   writes `true` a second time. Until the batch-end read, the optimistic
    ///   snapshot is what will be true. The listener waits for the same read.
    /// - **Refused**: take back this key's edit — put its columns back to what
    ///   the database is known to hold — *before* the resolving read is
    ///   issued, and report the refusal under this key. Before, because that
    ///   read can fail, and `Failed { prior }` keeps whatever the cell held:
    ///   left standing, a refused enable was the value the listener then
    ///   started on. With every exit honest, the batch ends with no refused
    ///   optimism anywhere in the cell, so a failed read preserves the
    ///   database's own intent and the reconcile acts on that.
    /// - **Superseded**: report nothing (the successor's outcome is the one the
    ///   reader waits on), but if this write was accepted, it is now what the
    ///   database holds for this key — so it becomes what a refused successor
    ///   restores to. Same-key writes are sequenced, so nothing else can have
    ///   written these columns in between, and the restore is exact rather than
    ///   a guess at the pre-batch value.
    ///
    /// The deferral above and the rollback here are one doctrine read from two
    /// ends: **only the database says what is true, and until it can be asked,
    /// the cell shows what every write still standing will make true** — an
    /// accepted sibling's answer is not adopted because the batch is not over,
    /// and a refused edit is taken back because it will never be true. The slot
    /// is removed before a current write settles, so an empty map means *this*
    /// was the last.
    fn settle_settings(
        &mut self,
        op: ProxyOp,
        edit: SettingsEdit,
        landing: Landing,
        result: Result<ProxySettings, AppError>,
        cx: &mut Context<Self>,
    ) {
        match (landing, result) {
            (Landing::Superseded, Ok(_)) => {
                if let Some(confirmed) = self.restore.get_mut(&op) {
                    *confirmed = edit;
                }
                return;
            }
            (Landing::Superseded, Err(_)) => return,
            (Landing::Current, Ok(settings)) => {
                self.restore.remove(&op);
                if self.op_tasks.is_empty() {
                    self.settings = Loadable::loaded(settings);
                    self.reconcile_listener(cx);
                }
            }
            (Landing::Current, Err(e)) => {
                if let Some(confirmed) = self.restore.remove(&op)
                    && let Some(settings) = self.settings.value_mut()
                {
                    confirmed.apply(settings);
                }
                self.op_errors.insert(op, e);
            }
        }
        cx.notify();
    }

    /// Clear this operation's standing refusal — **its own, and only its own**
    /// — and hand back the core, or `None` on a stub. The opening of every
    /// operation here. A write to the switch says nothing about a refused
    /// revoke, so it must not take that report off the screen.
    fn begin_op(&mut self, op: &ProxyOp) -> Option<Arc<AppCore>> {
        self.op_errors.remove(op);
        self.app_core.clone()
    }
}
