//! QA probes — one annotation that serves two consumers.
//!
//! `el.probe(name, role, label)` does two things at once:
//!
//! 1. **Accessibility** (always): sets the AccessKit role and label on the
//!    element via gpui's a11y builders, so assistive technology sees a real
//!    node. This requires the element to also carry an [`ElementId`] (call
//!    `.id(…)` before `.probe(…)`) — gpui derives the AccessKit node id from
//!    the `GlobalElementId`, and an id-less element never reaches the tree.
//! 2. **The UI driver** (only when probes are enabled): records the element's
//!    painted bounds — plus the role and label — into a process-global
//!    registry keyed by window, so `examples/driver.rs` can list named,
//!    clickable elements and target them by name instead of guessing
//!    coordinates. Think of the registry as our Playwright selector map.
//!
//! The pairing is deliberate: the accessible name *is* the driver's selector
//! vocabulary, so annotating for AT and annotating for automated QA are the
//! same act, and the two views of the UI can't drift apart.
//!
//! gpui keeps its own per-frame bounds maps (`debug_bounds`, the AccessKit
//! tree) crate-private on real-rendering windows, so the registry is recorded
//! from inside the element tree using the public `canvas` idiom: an absolute,
//! full-size, paint-nothing child whose prepaint callback sees the parent's
//! final bounds. Absolute children don't participate in flex layout and a
//! `canvas` registers no hitbox, so a probe never changes layout or event
//! routing. When probes are disabled (the default — production and ordinary
//! tests), `probe()` only applies the a11y attributes; the canvas child is
//! not constructed at all.
//!
//! Names are slash-scoped lowercase identifiers (`"chat/composer"`,
//! `"library/row/2/archive"`). Dynamic rows interpolate their index so a
//! driver can address "the third row" precisely.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use gpui::{
    Bounds, ParentElement, Pixels, Role, SharedString, StatefulInteractiveElement, Styled, canvas,
};

/// One recorded element: its a11y metadata plus the bounds it painted at.
#[derive(Clone, Debug)]
pub struct ProbeEntry {
    /// The AccessKit role given to the element.
    pub role: Role,
    /// The accessible label given to the element.
    pub label: SharedString,
    /// The element's bounds in window coordinates, as of the last frame in
    /// which it painted.
    pub bounds: Bounds<Pixels>,
}

/// Whether probes record into the registry. Off by default; the driver turns
/// it on at startup (`set_probes_enabled(true)`), and `EIDOLA_PROBES=1` turns
/// it on for ad-hoc runs. The a11y half of [`Probe::probe`] is unconditional.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// window id (`WindowId::as_u64`) → probe name → entry.
static REGISTRY: LazyLock<Mutex<HashMap<u64, HashMap<String, ProbeEntry>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Enable or disable probe recording process-wide.
pub fn set_probes_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether probe recording is currently enabled (via [`set_probes_enabled`]
/// or the `EIDOLA_PROBES=1` environment variable, checked once).
pub fn probes_enabled() -> bool {
    static FROM_ENV: LazyLock<bool> =
        LazyLock::new(|| matches!(std::env::var("EIDOLA_PROBES").as_deref(), Ok("1")));
    ENABLED.load(Ordering::Relaxed) || *FROM_ENV
}

/// Drop every recorded entry for a window. The driver calls this before
/// forcing a redraw so unmounted elements (a dismissed picker, a virtualized
/// row scrolled away) don't linger as stale click targets.
pub fn clear_window(window_id: u64) {
    if let Some(entries) = REGISTRY.lock().unwrap().get_mut(&window_id) {
        entries.clear();
    }
}

/// All entries recorded for a window since its last [`clear_window`], sorted
/// by name for stable output.
pub fn window_entries(window_id: u64) -> Vec<(String, ProbeEntry)> {
    let mut entries: Vec<(String, ProbeEntry)> = REGISTRY
        .lock()
        .unwrap()
        .get(&window_id)
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn record(window_id: u64, name: String, entry: ProbeEntry) {
    REGISTRY
        .lock()
        .unwrap()
        .entry(window_id)
        .or_default()
        .insert(name, entry);
}

/// The annotation entry point — see the module docs.
///
/// Bounded on [`StatefulInteractiveElement`] (where gpui defines the aria
/// builders), which makes "call `.id(…)` before `.probe(…)`" a compile-time
/// requirement rather than a convention — exactly the property the a11y tree
/// needs, since id-less elements never reach it.
pub trait Probe: StatefulInteractiveElement + ParentElement + Sized {
    /// Set the AccessKit `role` and `label` on this element, and (when probes
    /// are enabled) record its painted bounds under `name` for the UI driver.
    fn probe(
        self,
        name: impl Into<SharedString>,
        role: Role,
        label: impl Into<SharedString>,
    ) -> Self {
        let label = label.into();
        let this = self.role(role).aria_label(label.clone());
        if !probes_enabled() {
            return this;
        }
        let name = name.into();
        this.child(
            canvas(
                move |bounds, window, _| {
                    record(
                        window.window_handle().window_id().as_u64(),
                        name.to_string(),
                        ProbeEntry {
                            role,
                            label,
                            bounds,
                        },
                    );
                },
                |_, _, _, _| {},
            )
            .absolute()
            .size_full(),
        )
    }
}

impl<T: StatefulInteractiveElement + ParentElement + Sized> Probe for T {}
