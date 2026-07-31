//! **Overlay containment — what a surface that covers live content does with
//! the mouse.**
//!
//! gpui does not occlude by painting. A hit test reports **every** hitbox under
//! the cursor and a listener fires whenever its own hitbox is hovered, so an
//! element painted on top of another does not, by itself, take the mouse away
//! from it. Containment is an explicit opt-in ([`InteractiveElement::occlude`] /
//! [`InteractiveElement::block_mouse_except_scroll`]), and it suppresses only
//! hitboxes registered **before** the blocking one — so a contained surface
//! must also be painted *after* whatever it covers.
//!
//! Both halves have been forgotten twice, in the same shape: the window's
//! transparent drag band let a press through to the post scrolled beneath it,
//! so moving the window dragged out a text selection (task 32); the floating
//! composer — opaque, and the surface you are *typing into* — let a
//! drag-select through to the post underneath, which both selected that post
//! and drove the page's selection-autoscroll, so selecting your own draft
//! scrolled the window. Two instances of one bug class is the reason this
//! module exists rather than a third local `.occlude()`.
//!
//! ## The doctrine
//!
//! Every surface that visually overlays live content declares an [`Overlay`]
//! class. There are exactly three, and the axis that separates them is **what
//! should happen to the wheel**, because clicks are unambiguous — a click
//! belongs to the thing you can see under the pointer, always.
//!
//! | Class | Clicks | Wheel | Examples |
//! |---|---|---|---|
//! | [`Overlay::Fade`] | captured | **falls through** to the page | the title/drag bands |
//! | [`Overlay::Scrolling`] | captured | reaches the surface's **own** scroll handler | the floating composer |
//! | [`Overlay::Popover`] | captured | captured | menus, pickers, notices |
//!
//! [`Overlay::Fade`] is the translucent-chrome case: a band you can read live
//! content *through*, where a wheel gesture that starts in it is plainly aimed
//! at the page it is showing you. [`Overlay::Scrolling`] and [`Overlay::Fade`]
//! use the same primitive (`block_mouse_except_scroll`); what distinguishes
//! them is that a `Scrolling` surface **registers a scroll handler**, so the
//! wheel is consumed by the surface under the cursor — the macOS convention —
//! instead of continuing to the page. (The composer's handler deliberately
//! *routes* rather than always consuming: a floating composer with nothing of
//! its own to scroll hands the gesture to the page, which moves it toward its
//! dock. That is the surface's own policy, expressed where it belongs, and it
//! is why `Scrolling` is not simply [`Overlay::Popover`].)
//!
//! [`Overlay::Popover`] is the everything-stops-here case. A menu has nothing
//! to scroll, and letting the page scroll out from under an open menu is not
//! what any platform does — the menu would end up anchored to content that has
//! moved. So it blocks the wheel too.
//!
//! ## Using it
//!
//! ```ignore
//! use crate::overlay::{Contain as _, Overlay};
//!
//! div().id("my-popover").contain_mouse(Overlay::Popover)
//! ```
//!
//! and **paint the surface after what it covers** — as a later sibling, or a
//! later root child. A contained surface painted first contains nothing.

use gpui::InteractiveElement;

/// How an overlay surface treats mouse events aimed at the content beneath it.
/// See the [module docs](self) for the doctrine behind the three classes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Overlay {
    /// **Translucent chrome over live content** — the title/drag bands. The
    /// content beneath stays readable *through* it, so a wheel gesture that
    /// starts here is aimed at that content and passes through; a press is the
    /// band's own (the window move).
    Fade,
    /// **An opaque surface that owns its own scrolling** — the floating
    /// composer. Clicks are its own; the wheel reaches its scroll handler,
    /// which decides what the gesture means.
    Scrolling,
    /// **An opaque surface with nothing to scroll** — menus, pickers, notice
    /// cards. Every mouse event stops here, including the wheel: a menu whose
    /// page scrolled away beneath it would be anchored to nothing.
    Popover,
}

/// Fluent containment for overlay surfaces (see the [module docs](self)).
pub(crate) trait Contain: InteractiveElement {
    /// Declare this element's overlay class, containing the mouse accordingly.
    ///
    /// Remember the other half of the contract: a hitbox only suppresses
    /// hitboxes registered **before** it, so the surface must also be painted
    /// after whatever it covers.
    fn contain_mouse(self, kind: Overlay) -> Self {
        match kind {
            // Same primitive; the difference is whether the surface registers
            // a scroll handler to consume what this lets through.
            Overlay::Fade | Overlay::Scrolling => self.block_mouse_except_scroll(),
            Overlay::Popover => self.occlude(),
        }
    }
}

impl<E: InteractiveElement> Contain for E {}
