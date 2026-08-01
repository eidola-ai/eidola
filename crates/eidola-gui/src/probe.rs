//! QA probes — one annotation that serves two consumers.
//!
//! `el.probe(name, role, label)` does two things at once:
//!
//! 1. **Accessibility** (always): sets the AccessKit role and label on the
//!    element via gpui's a11y builders, so assistive technology sees a real
//!    node — and, since wave B, the **focus attributes derived from that
//!    role** (see [`crate::focus`]): an interactive role becomes a focusable
//!    tab stop wearing the focus-visible ring, a container role stays a
//!    container. This requires the element to also carry an [`ElementId`]
//!    (call `.id(…)` before `.probe(…)`) — gpui derives the AccessKit node id
//!    from the `GlobalElementId`, and an id-less element never reaches the
//!    tree.
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
//! `probe_value(name, role, label, value)` is the same annotation plus the
//! element's **content** (`aria_value`) — a settled post's text, a balance, an
//! alert's message. It is recorded too, so the content channel is regression-
//! tested at the call site rather than merely compile-checked (the emitted
//! AccessKit tree is unobservable at this pin). Read the value rule in
//! [`Probe::probe_value`] before wiring one to anything that changes often.
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
    /// The accessible **value** (`aria_value`), when the call site set one via
    /// [`Probe::probe_value`]. This is the content channel — a post's text, a
    /// balance, an alert's message — as distinct from the label, which names
    /// the element. `None` for a plain [`Probe::probe`].
    pub value: Option<SharedString>,
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
        self.probe_inner(name, role, label.into(), None, true)
    }

    /// Like [`Probe::probe`], plus the element's accessible **value**
    /// (`aria_value`) — the content channel AT reads on request, and the one
    /// the macOS adapter announces from once a node is a live region.
    ///
    /// Use it wherever the element *has* content distinct from its name: a
    /// settled post's text under a byline label, a balance figure under
    /// "Balance", an alert's message. **Never bind it to text that mutates at
    /// speed** (a streaming reply, a live editor buffer, a download counter):
    /// assistive technology re-reads the whole value on every change of a
    /// focused control, which turns annotation into noise. See the audit's §4
    /// (`work/tasks/12a`).
    fn probe_value(
        self,
        name: impl Into<SharedString>,
        role: Role,
        label: impl Into<SharedString>,
        value: impl Into<SharedString>,
    ) -> Self {
        self.probe_inner(name, role, label.into(), Some(value.into()), true)
    }

    /// Like [`Probe::probe`], for an element that **delegates its focus**: it
    /// carries the role, the label and the bounds, and something else carries
    /// the keyboard. Applies **no focus attributes at all**. Two cases:
    ///
    /// - a **wrapper** around a `gpui-component` widget that owns its own focus
    ///   handle (`Button`, `Checkbox`) — the shrink-wrapped `div` that exists
    ///   only because those widgets carry no a11y annotations at our pin. The
    ///   widget inside is already a tab stop, wears its own ring, and —
    ///   decisively — owns the `on_click` that gpui's Enter/Space activation
    ///   invokes. gpui's keyboard click runs *only the focused element's own
    ///   click listeners* (`div.rs`, gated on that element having any), so a
    ///   focusable wrapper is a tab stop that can never be activated: it would
    ///   ring, swallow a Tab, and do nothing, with the working control one Tab
    ///   further on.
    /// - a **row of a roving-focus list** (the Library's listing), where the
    ///   list is the single tab stop and moves a cursor over its rows. A tab
    ///   stop per row cannot work in a virtualized list at all: only the
    ///   materialized window would be in the tab order.
    ///
    /// See [`crate::focus`] for the rule in full.
    fn probe_delegating(
        self,
        name: impl Into<SharedString>,
        role: Role,
        label: impl Into<SharedString>,
    ) -> Self {
        self.probe_inner(name, role, label.into(), None, false)
    }

    #[doc(hidden)]
    fn probe_inner(
        self,
        name: impl Into<SharedString>,
        role: Role,
        label: SharedString,
        value: Option<SharedString>,
        derive_focus: bool,
    ) -> Self {
        let mut this = self.role(role).aria_label(label.clone());
        if let Some(value) = value.clone() {
            this = this.aria_value(value);
        }
        // Wave B: focus is derived from the role, so one annotation still
        // carries everything. `tab_index(0)` puts the element at index 0 of
        // whatever tab *group* encloses it — the enclosing landmark supplies
        // the region's place in the order (`focus::region`), and equal indices
        // fall back to paint order, so within a region the tab walk follows
        // the page. `focusable()` alone (a post) means "can hold focus, is not
        // a Tab destination". The ring is gpui's own `:focus-visible`: shown
        // when this element is focused *and* the last input was a key.
        //
        // `probe_delegating` opts out: the widget inside owns the focus, the
        // ring and the activation.
        if derive_focus && crate::focus::is_focusable(role) {
            this = this.focusable();
            if crate::focus::is_tab_stop(role) {
                this = this.tab_index(0);
                let shadows = crate::focus::ring_shadows(crate::focus::ring_colors());
                this = this.focus_visible(move |s| s.shadow(shadows));
            }
        }
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
                            value,
                            bounds,
                        },
                    );
                },
                |_, _, _, _| {},
            )
            // Inset-0 is load-bearing: with auto insets an absolute child
            // falls back to its *static position* — after any siblings — so
            // a probe added after the element's content records bounds
            // offset by that content (a click-by-name then misses). Pinning
            // to the parent's origin makes recording independent of whether
            // `.probe()` is called before or after `.child(…)`.
            .absolute()
            .top_0()
            .left_0()
            .size_full(),
        )
    }
}

impl<T: StatefulInteractiveElement + ParentElement + Sized> Probe for T {}
