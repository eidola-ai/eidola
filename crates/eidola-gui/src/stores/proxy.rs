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
//! app-core; a refusal lands in `op_error`, and a bind failure — which is a
//! different kind of failure, since nothing the reader typed was wrong — lands
//! in its own `listen_error` beside it.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::proxy::{
    LocalExposure, MintedProxyKey, ProxyKeyInfo, ProxySettings, ProxySettingsUpdate,
};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;
use crate::proxy::ProxyHandle;

/// The one slot a key generation ever occupies. Named, because the re-entry
/// refusal and the operation itself have to mean the same slot.
const CREATE_KEY_SLOT: &str = "create-key";

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
    op_tasks: HashMap<String, (u64, Task<()>)>,
    next_op_gen: u64,
    op_error: Option<String>,
    /// Why the listener is not running, when the reader asked for it to be.
    /// Separate from `op_error` because the write succeeded — what failed is
    /// the socket, and the two want different words.
    listen_error: Option<String>,
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
            op_error: None,
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
            op_error: None,
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

    pub fn op_error(&self) -> Option<&str> {
        self.op_error.as_deref()
    }

    /// Why nothing is listening, when the reader asked for something to be.
    ///
    /// **Derived, never only cached.** A listener whose accept loop gave up
    /// stops answering an address without any write having failed, so the
    /// handle's own reason outranks the last bind failure recorded here: the
    /// pane's question is "why is nothing listening", and the loop that stopped
    /// is the truest answer available. A bind that failed left no listener at
    /// all, so the two can never both answer.
    pub fn listen_error(&self) -> Option<String> {
        self.handle
            .accept_failure()
            .or_else(|| self.listen_error.clone())
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
        self.minted.is_none() && !self.op_tasks.contains_key(CREATE_KEY_SLOT)
    }

    /// Whether a key is being generated right now — the pending half of
    /// [`Self::can_create_key`], which the pane words differently from the
    /// unacknowledged half.
    pub fn create_key_pending(&self) -> bool {
        self.op_tasks.contains_key(CREATE_KEY_SLOT)
    }

    /// Acknowledge the generated key. **The value is gone after this** — only
    /// its digest was ever stored.
    pub fn dismiss_minted(&mut self, cx: &mut Context<Self>) {
        if self.minted.take().is_some() {
            cx.notify();
        }
    }

    pub fn clear_op_error(&mut self, cx: &mut Context<Self>) {
        if self.op_error.take().is_some() {
            cx.notify();
        }
    }

    /// Re-read the settings and the key listing, then bring the listener into
    /// line with what the settings say.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.settings = std::mem::take(&mut self.settings).to_loading();
        let settings_core = core.clone();
        self.settings_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(settings_core, |c| async move { c.proxy_settings().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.settings = std::mem::take(&mut this.settings).resolve(result);
                this.settings_task = None;
                this.reconcile_listener(cx);
                cx.notify();
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
        let wanted = format!("{}:{}", settings.bind_address, settings.bind_port);
        if self.handle.address().map(|a| a.to_string()).as_deref() == Some(wanted.as_str()) {
            return;
        }
        match self.handle.start(&core, &settings) {
            Ok(_) => self.listen_error = None,
            Err(e) => {
                // `start` closed the running listener before it tried to bind,
                // so a refused address leaves the proxy **stopped** rather than
                // still answering on the old one: a reader who changed the
                // binding must not be told it moved when it did not.
                self.listen_error = Some(e.to_string());
            }
        }
        cx.notify();
    }

    /// Turn the proxy on or off.
    pub fn set_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.write_settings(
            "enabled",
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
            "binding",
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
            "exposure",
            ProxySettingsUpdate {
                local_exposure: Some(exposure),
                ..Default::default()
            },
            cx,
        );
    }

    /// Expose or withdraw one backend.
    pub fn set_backend_exposed(&mut self, id: String, exposed: bool, cx: &mut Context<Self>) {
        let slot = format!("backend:{id}");
        self.start_op(
            slot,
            cx,
            move |core| async move {
                bridge(core, move |c| async move {
                    c.set_proxy_backend_exposed(id, exposed).await
                })
                .await
            },
            |this, result, cx| this.settle(result, cx),
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
            CREATE_KEY_SLOT.to_string(),
            cx,
            move |core| async move {
                bridge(core, |c| async move { c.create_proxy_key(label).await }).await
            },
            |this, result, cx| match result {
                Ok(minted) => {
                    this.minted = Some(minted);
                    this.refresh(cx);
                }
                Err(e) => this.op_error = Some(e.to_string()),
            },
        );
    }

    /// Revoke a key. Keyed per key, because the pane offers the verb on every
    /// row at once and a store-wide slot would drop one of two presses.
    pub fn revoke_key(&mut self, id: String, cx: &mut Context<Self>) {
        let slot = format!("revoke:{id}");
        self.start_op(
            slot,
            cx,
            move |core| async move {
                bridge(core, |c| async move { c.revoke_proxy_key(id).await }).await
            },
            |this, result, cx| match result {
                Ok(_) => this.refresh(cx),
                Err(e) => this.op_error = Some(e.to_string()),
            },
        );
    }

    fn write_settings(&mut self, key: &str, update: ProxySettingsUpdate, cx: &mut Context<Self>) {
        self.start_op(
            key.to_string(),
            cx,
            move |core| async move {
                bridge(
                    core,
                    |c| async move { c.update_proxy_settings(update).await },
                )
                .await
            },
            |this, result, cx| this.settle(result, cx),
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
    /// **Only the current generation settles.** A superseded op reports nothing
    /// and removes no slot: the slot is its successor's by then, and dropping it
    /// would cancel the very write it just sequenced.
    fn start_op<T, Fut>(
        &mut self,
        key: String,
        cx: &mut Context<Self>,
        op: impl FnOnce(Arc<AppCore>) -> Fut + 'static,
        settle: impl FnOnce(&mut Self, T, &mut Context<Self>) + 'static,
    ) where
        T: 'static,
        Fut: std::future::Future<Output = T> + 'static,
    {
        let Some(core) = self.begin_op() else { return };
        self.next_op_gen += 1;
        let generation = self.next_op_gen;
        let previous = self.op_tasks.remove(&key).map(|(_, task)| task);
        let slot = key.clone();
        let task = cx.spawn(async move |this, cx| {
            if let Some(previous) = previous {
                previous.await;
            }
            let result = op(core).await;
            let _ = this.update(cx, |this, cx| {
                if this.op_tasks.get(&slot).map(|(g, _)| *g) != Some(generation) {
                    return;
                }
                this.op_tasks.remove(&slot);
                settle(this, result, cx);
                cx.notify();
            });
        });
        self.op_tasks.insert(key, (generation, task));
        cx.notify();
    }

    /// A settings write's landing: adopt what the core answered — which is the
    /// resolved settings, not what was asked for — and reconcile the listener
    /// against it.
    fn settle(
        &mut self,
        result: Result<ProxySettings, eidola_app_core::error::AppError>,
        cx: &mut Context<Self>,
    ) {
        match result {
            Ok(settings) => {
                self.settings = Loadable::loaded(settings);
                self.reconcile_listener(cx);
            }
            Err(e) => {
                self.op_error = Some(e.to_string());
                // A refused write changed nothing, so nothing about the
                // listener has to move — but the snapshot is re-read anyway,
                // because a refusal is also the moment a stale cache is most
                // likely to be showing.
                self.refresh(cx);
            }
        }
        cx.notify();
    }

    /// Clear the standing refusal and hand back the core, or `None` on a stub —
    /// the opening of every operation here.
    fn begin_op(&mut self) -> Option<Arc<AppCore>> {
        self.op_error = None;
        self.app_core.clone()
    }
}
