//! `RecordStore` — the bus-relay seam for the Record domain.
//!
//! Unlike the other stores this one owns **no data**: Record listings,
//! cursors, and details are *window-scoped reader* state (the doctrine's
//! "Record pattern" — `docs/architecture/state.md`), held by each open
//! `RecordView` together with its own fetch tasks, dying with the window.
//! What a window-scoped reader cannot do on its own is hear the
//! invalidation bus — the app-lifetime bus bridge dispatches `Change`s to
//! stores only. `RecordStore` is that seam: the bridge bumps `epoch` on
//! every `Change::Record`, and each open Record window observes this entity,
//! compares the epoch against the one it last fetched at, and marks itself
//! stale — surfacing the quiet "new entries — refresh" affordance, never
//! mutating rows under the user's scroll position.

use gpui::Context;

pub struct RecordStore {
    /// Monotonic count of `Change::Record` signals seen this session.
    /// Readers snapshot it when they (re)fetch and compare on observe, so a
    /// change that lands mid-refresh still reads as new.
    epoch: u64,
}

impl RecordStore {
    pub fn new() -> Self {
        Self { epoch: 0 }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Bus-driven on `Change::Record` (and on `Lagged`, where the dropped
    /// change may have been a Record write): the local trail grew.
    pub fn notify_changed(&mut self, cx: &mut Context<Self>) {
        self.epoch += 1;
        cx.notify();
    }
}

impl Default for RecordStore {
    fn default() -> Self {
        Self::new()
    }
}
