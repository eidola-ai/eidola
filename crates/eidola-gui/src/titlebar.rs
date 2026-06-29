//! Shared window-drag affordance for the transparent-titlebar windows.
//!
//! Every Eidola window uses the edge-to-edge transparent titlebar from
//! `lib.rs::transparent_titlebar`: macOS extends the content view under the
//! traffic lights and paints no separate titlebar background, so the OS no
//! longer provides a draggable strip of its own. `WindowControlArea::Drag` is
//! a no-op on macOS, so window dragging is wired explicitly in the gpui
//! content view (the same approach `space_view`'s title bar already uses):
//!
//! - arm on left mouse-down,
//! - call [`Window::start_window_move`] on the first move while armed (so a
//!   plain click doesn't begin a drag), and
//! - forward a double-click to [`Window::titlebar_double_click`] for
//!   zoom/minimize parity.
//!
//! The armed flag lives in a [`DragArm`] stored on the view (not a per-render
//! cell) so it survives any re-render that lands between the mouse-down and the
//! first move.

use std::cell::Cell;
use std::rc::Rc;

use gpui::{
    Div, InteractiveElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    Stateful, Styled, Window, div,
};
use gpui_component::InteractiveElementExt;

/// Per-view armed flag for the titlebar drag gesture. Construct one with
/// [`drag_arm`], store it on the window's view, and hand a clone to
/// [`make_draggable`] / [`drag_band`] each render.
pub(crate) type DragArm = Rc<Cell<bool>>;

/// A fresh, disarmed [`DragArm`] for a view's constructor.
pub(crate) fn drag_arm() -> DragArm {
    Rc::new(Cell::new(false))
}

/// Attach the arm-on-down / move-on-first-move / double-click-to-zoom window
/// drag gesture to an existing stateful element (e.g. a window's title strip).
pub(crate) fn make_draggable(el: Stateful<Div>, armed: DragArm) -> Stateful<Div> {
    let on_down = armed.clone();
    let on_up = armed.clone();
    el.on_mouse_down(MouseButton::Left, move |_: &MouseDownEvent, _, _| {
        on_down.set(true);
    })
    .on_mouse_up(MouseButton::Left, move |_: &MouseUpEvent, _, _| {
        on_up.set(false);
    })
    .on_mouse_move(move |_: &MouseMoveEvent, window: &mut Window, _| {
        if armed.get() {
            armed.set(false);
            window.start_window_move();
        }
    })
    .on_double_click(|_, window, _| window.titlebar_double_click())
}

/// A full-width, transparent drag band absolutely positioned over a window's
/// top titlebar reserve. Overlay it above the content of windows whose top
/// strip is otherwise just empty traffic-light reserve.
pub(crate) fn drag_band(id: &'static str, height: Pixels, armed: DragArm) -> Stateful<Div> {
    make_draggable(
        div().id(id).absolute().top_0().left_0().right_0().h(height),
        armed,
    )
}
