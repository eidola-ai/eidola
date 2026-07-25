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
//! module factors it out so the Library, Record, Settings, Participants, and
//! Updates surfaces render the identical control from one place.
//!
//! **Placement.** [`vertical`] returns the positioned overlay strip; the caller
//! drops it as a sibling of the scroll container inside a `relative` ancestor
//! that spans exactly the scroll viewport (so the indicator tracks the
//! viewport's right edge, not necessarily the window's). It must be a *sibling*
//! of the `overflow_y_scroll` element, never a child of it — a child would
//! scroll away with the content.
//!
//! **Corner duty (Linux CSD).** A full-height right-edge strip whose ends meet
//! the window's rounded corners insets both ends by
//! [`crate::chrome::corner_clearance`] so the square thumb never paints over the
//! corner arc — the same discipline the minimap applies. The clearance is zero
//! on macOS / SSD / tiled edges, so callers apply [`vertical`] unconditionally.

use gpui::{Div, InteractiveElement, ParentElement, Pixels, Stateful, Styled, Window, div, px};
use gpui_component::scroll::{Scrollbar, ScrollbarHandle, ScrollbarShow};

/// Width of the overlay strip that houses the thumb. Matches the onboarding
/// window's original inline value.
const STRIP_WIDTH: Pixels = px(14.);

/// A right-edge vertical scroll indicator bound to `handle`, shown only while
/// scrolling. Bind it to either a plain [`gpui::ScrollHandle`] (an
/// `overflow_y_scroll` div's `track_scroll`) or a
/// [`gpui::UniformListScrollHandle`] (a `uniform_list`'s `track_scroll`) — both
/// implement [`ScrollbarHandle`].
///
/// Drop the returned element as a sibling of the scroll container; see the
/// module docs for placement and corner-duty notes.
pub(crate) fn vertical<H>(id: &'static str, handle: &H, window: &Window) -> Stateful<Div>
where
    H: ScrollbarHandle + Clone,
{
    let clearance = crate::chrome::corner_clearance(window);
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
