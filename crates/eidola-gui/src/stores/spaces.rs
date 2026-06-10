//! `SpacesStore` — the Library index (the list of non-archived spaces).
//!
//! Per `docs/architecture/state.md` this store will *also* hold the per-space
//! entity registry (`HashMap<SpaceId, WeakEntity<Space>>`) once the `Space`
//! entity lands — that is **step 3**, out of scope here. A placeholder field +
//! doc comment marks the seam.

use std::sync::Arc;

use eidola_app_core::{AppCore, SpaceInfo};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct SpacesStore {
    app_core: Option<Arc<AppCore>>,
    /// The Library index (archived excluded), newest activity first.
    index: Loadable<Vec<SpaceInfo>>,
    /// Supersede slot for the index refresh.
    task: Option<Task<()>>,
    // STEP 3: the per-space entity registry lands here —
    //   `registry: HashMap<SpaceId, WeakEntity<Space>>`
    // with `open(space_id)` getting-or-creating, so two windows on one space
    // share the same `Space` entity (fixes wave-2 bug 4). Not in scope now.
}

impl SpacesStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            index: Loadable::NotLoaded,
            task: None,
        }
    }

    /// A stub store with a fixture listing (tests).
    pub fn stub(spaces: Vec<SpaceInfo>) -> Self {
        Self {
            app_core: None,
            index: if spaces.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(spaces)
            },
            task: None,
        }
    }

    /// The current Library index.
    pub fn index(&self) -> &Loadable<Vec<SpaceInfo>> {
        &self.index
    }

    /// The listing as a slice.
    pub fn list(&self) -> &[SpaceInfo] {
        self.index.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Refresh the Library index. Fire-and-notify supersede slot.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.index = std::mem::take(&mut self.index).to_loading();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.list_spaces(false).await }).await;
            let _ = this.update(cx, |this, cx| {
                this.index = std::mem::take(&mut this.index).resolve(result);
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Archive a space: drop the cached row immediately (so the Library
    /// responds without a backend round-trip — and so stub tests exercise the
    /// local path), then archive core-side and let `Change::SpaceIndex` drive
    /// the reconciling refresh. The optimistic removal makes this safe to own
    /// in the store: even if the window closes, the core write completes and
    /// the bus reconciles every other window.
    pub fn archive(&mut self, space_id: String, cx: &mut Context<Self>) {
        // Optimistic local removal — operate on whatever value is present.
        if let Some(list) = self.index.value() {
            let next: Vec<SpaceInfo> = list.iter().filter(|s| s.id != space_id).cloned().collect();
            self.index = Loadable::loaded(next);
        }
        cx.notify();

        let Some(core) = self.app_core.clone() else {
            return;
        };
        // Own the core write in the supersede slot; its completion triggers a
        // reconciling refresh from the bus (`Change::SpaceIndex`). We re-list
        // here too so a stub-less local run reconciles even without the bus.
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, move |c| async move {
                c.archive_space(space_id).await?;
                c.list_spaces(false).await
            })
            .await;
            let _ = this.update(cx, |this, cx| {
                if let Ok(spaces) = result {
                    this.index = Loadable::loaded(spaces);
                }
                this.task = None;
                cx.notify();
            });
        }));
    }
}
