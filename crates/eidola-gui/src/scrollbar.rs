//! Quiet overlay scroll indicators — the app-wide idiom for scrollable views.
//!
//! Every scrollable surface that isn't the space page (whose minimap *is* its
//! scroll indicator) shares one presentation: a right-edge
//! [`gpui_component::scroll::Scrollbar`] in [`ScrollbarShow::Scrolling`] mode —
//! it appears while scrolling and fades out, the macOS overlay idiom, matching
//! the calm design voice. It never reserves a permanent gutter and never
//! shifts layout: the strip is an `absolute` overlay, so a surface that never
//! overflows simply shows nothing.
//!
//! The onboarding window established the idiom (`onboarding/mod.rs`); this
//! module factors it out so the Library, Record, Settings, Participants,
//! Updates surfaces — and the Participants/Templates model-picker dropdown —
//! render the identical control from one place.
//!
//! **Placement.** Both constructors return the positioned overlay strip; the
//! caller drops it as a sibling of the scroll container inside a `relative`
//! ancestor that spans exactly the scroll viewport (so the indicator tracks the
//! viewport's right edge, not necessarily the window's). It must be a *sibling*
//! of the `overflow_y_scroll` element, never a child of it — a child would
//! scroll away with the content.
//!
//! **Two variants, by whether the viewport meets a window corner:**
//!
//! - [`vertical`] — a **window-edge** strip (a view's full-height scroll body).
//!   Its ends can coincide with the window's rounded corners, so on Linux CSD it
//!   insets both ends by [`crate::chrome::corner_clearance`] (the minimap
//!   precedent) so the square thumb never paints over the corner arc. Zero on
//!   macOS / SSD / tiled edges, so callers apply it unconditionally.
//! - [`vertical_floating`] — a **bounded** overlay (a dropdown/popover that
//!   floats mid-window, never touching a window corner). It takes **no**
//!   clearance — corner clearance keys off the window's decorations, not the
//!   element's position, so applying it here would wrongly inset a mid-window
//!   dropdown's thumb on Linux. Needs no `Window`.
//!
//! **And one horizontal twin**, [`horizontal_floating`], for the rare surface
//! that scrolls on both axes: a bottom-edge strip, same bounded reasoning as
//! its vertical sibling. It **yields the corner** — its right end insets by the
//! strip width — because two overlays meeting there would put two thumbs on one
//! point, and the vertical one is the axis every surface in this app scrolls.
//! Only a surface with real horizontal overflow wants it: gpui does not
//! redirect a vertical-only wheel to the other axis once both are scrollable,
//! so without a thumb an ordinary mouse has no path to what is clipped there.

use gpui::{Div, InteractiveElement, ParentElement, Pixels, Stateful, Styled, Window, div, px};
use gpui_component::scroll::{Scrollbar, ScrollbarHandle, ScrollbarShow};

/// Width of the overlay strip that houses the thumb. Matches the onboarding
/// window's original inline value.
const STRIP_WIDTH: Pixels = px(14.);

/// The shared positioned strip: an `absolute` right-edge overlay carrying the
/// scrolling-mode [`Scrollbar`], inset at both ends by `clearance`.
///
/// Its horizontal twin is [`horizontal_floating`].
fn overlay<H>(id: &'static str, handle: &H, clearance: Pixels) -> Stateful<Div>
where
    H: ScrollbarHandle + Clone,
{
    div()
        .id(id)
        .absolute()
        .top_0()
        .right_0()
        .bottom_0()
        .pt(clearance)
        .pb(clearance)
        .w(STRIP_WIDTH)
        .child(Scrollbar::vertical(handle).scrollbar_show(ScrollbarShow::Scrolling))
}

/// A right-edge vertical scroll indicator for a **window-edge** scroll body,
/// bound to `handle` and shown only while scrolling. Bind it to either a plain
/// [`gpui::ScrollHandle`] (an `overflow_y_scroll` div's `track_scroll`) or a
/// [`gpui::UniformListScrollHandle`] (a `uniform_list`'s `track_scroll`) — both
/// implement [`ScrollbarHandle`]. Applies Linux-CSD corner clearance.
///
/// Drop the returned element as a sibling of the scroll container; see the
/// module docs for placement and corner-duty notes.
pub(crate) fn vertical<H>(id: &'static str, handle: &H, window: &Window) -> Stateful<Div>
where
    H: ScrollbarHandle + Clone,
{
    overlay(id, handle, crate::chrome::corner_clearance(window))
}

/// A right-edge vertical scroll indicator for a **bounded** overlay — a
/// dropdown/popover floating mid-window that never meets a window corner, so it
/// takes no corner clearance (and needs no `Window`). Otherwise identical to
/// [`vertical`]: same overlay-sibling placement rule.
pub(crate) fn vertical_floating<H>(id: &'static str, handle: &H) -> Stateful<Div>
where
    H: ScrollbarHandle + Clone,
{
    overlay(id, handle, px(0.))
}

/// A bottom-edge **horizontal** scroll indicator for a bounded overlay that
/// really does scroll on both axes.
///
/// Same placement rule as the vertical constructors — a sibling of the scroll
/// container inside a `relative` ancestor — and the same bounded reasoning, so
/// no corner clearance. It **yields the bottom-right corner** to the vertical
/// strip (`right(STRIP_WIDTH)`), which is the axis every scrollable surface in
/// this app has: two overlays meeting there would stack two thumbs on one
/// point, and only one of them can win the press.
pub(crate) fn horizontal_floating<H>(id: &'static str, handle: &H) -> Stateful<Div>
where
    H: ScrollbarHandle + Clone,
{
    div()
        .id(id)
        .absolute()
        .left_0()
        .right(STRIP_WIDTH)
        .bottom_0()
        .h(STRIP_WIDTH)
        .child(Scrollbar::horizontal(handle).scrollbar_show(ScrollbarShow::Scrolling))
}
