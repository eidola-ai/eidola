//! Per-window modifier state.
//!
//! `ModifiersChangedEvent` in gpui (at our pinned commit 969a67fc) is
//! dispatched along the **focused element's ancestor path only** — a listener
//! on a sibling branch of the focused node never fires. Placing one listener
//! per window on the **root view** (whose tracked focus handle is always an
//! ancestor of whatever is focused) and mirroring events into a `WindowInput`
//! entity makes the modifier state available to every descendant via
//! `cx.observe(&window_input, …)`, regardless of which leaf holds focus.
//!
//! ## Rules (see `docs/architecture/state.md` — "Input-state sharing")
//!
//! - **One listener per window**, registered on the root view. Descendants
//!   observe the entity; they never register `on_modifiers_changed` themselves.
//! - The root's listener is the only place modifier events are consumed.
//!
//! ## Window-activation / key-status staleness
//!
//! gpui at our pin (969a67fc) does not expose a clean window-activation or
//! "window became/resigned key" observer hook (only `on_window_should_close`
//! and `on_focus_lost` exist, neither of which maps cleanly to "another
//! window just stole the modifier stream"). As a result, `alt` may remain
//! `true` in `WindowInput` after a window loses key status if the user moved
//! focus to another window while holding ⌥. This is **self-healing**: the
//! next `ModifiersChangedEvent` delivered to this window (on regaining focus
//! or on the next modifier transition) will write the current state, and the
//! stale `true` is transient and harmless (the ⌥ reveal is a pure cosmetic
//! affordance with no irreversible action behind it). Revisit once gpui
//! exposes a window-activation observer.

use gpui::{App, AppContext, Context, Entity, Modifiers, ModifiersChangedEvent};

/// Holds the live modifier state for one window.
///
/// Created by each `open_*_window` builder and handed to the window's root
/// view. The root registers the window's single `on_modifiers_changed`
/// listener and calls `WindowInput::update_modifiers` on every event.
/// Descendant views observe this entity instead of registering their own
/// modifier listeners.
pub struct WindowInput {
    modifiers: Modifiers,
}

impl WindowInput {
    /// Create a new `WindowInput` with all modifiers released.
    pub fn new(cx: &mut App) -> Entity<Self> {
        cx.new(|_| WindowInput {
            modifiers: Modifiers::default(),
        })
    }

    /// Mirror a `ModifiersChangedEvent` into the entity. Called by the root
    /// view's `on_modifiers_changed` listener; returns whether the state
    /// actually changed (so the caller can skip a `cx.notify()` on no-ops).
    pub fn update_modifiers(
        &mut self,
        event: &ModifiersChangedEvent,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.modifiers != event.modifiers {
            self.modifiers = event.modifiers;
            cx.notify();
            true
        } else {
            false
        }
    }

    /// The current modifier state.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// Whether ⌥ (alt/option) is currently held.
    pub fn alt_held(&self) -> bool {
        self.modifiers.alt
    }

    /// Test-only: set the alt modifier directly without constructing a full
    /// `ModifiersChangedEvent`. Used by snapshot tests that want to render the
    /// ⌥-revealed state without synthesizing platform events.
    #[doc(hidden)]
    pub fn set_alt_for_test(&mut self, alt: bool, cx: &mut Context<Self>) {
        if self.modifiers.alt != alt {
            self.modifiers.alt = alt;
            cx.notify();
        }
    }
}
