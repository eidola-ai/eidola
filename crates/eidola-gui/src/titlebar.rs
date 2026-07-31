//! Shared window-drag affordance for the transparent-titlebar windows.
//!
//! Every Eidola window uses the edge-to-edge transparent titlebar from
//! `lib.rs::transparent_titlebar`: macOS extends the content view under the
//! traffic lights and paints no separate titlebar background, so the OS no
//! longer provides a draggable strip of its own. `WindowControlArea::Drag` is
//! a no-op on macOS, so window dragging is wired explicitly in the gpui
//! content view:
//!
//! - arm on left mouse-down,
//! - call [`Window::start_window_move`] on the first move while armed (so a
//!   plain click doesn't begin a drag), and
//! - forward a double-click to [`Window::titlebar_double_click`] for
//!   zoom/minimize parity.
//!
//! The armed flag is **element-owned state** (`window.use_keyed_state`, keyed
//! on the strip's id): it survives any re-render that lands between the
//! mouse-down and the first move (element state persists while the element
//! keeps painting, and the drag strip is unconditional window chrome), and no
//! view field or constructor plumbing is needed. Mutation goes through the
//! `Cell` without an entity update, so arming never triggers a re-render.
//!
//! **The strip captures the mouse** (`block_mouse_except_scroll`). A gpui
//! mouse listener fires whenever its hitbox is *hovered*, and a plain overlay
//! hitbox doesn't hide what it covers — so a press on the transparent header
//! reached the content beneath it too: over a space's transcript the press
//! landed in the post's `MarkdownEditor` and the drag that moved the window
//! also dragged out a text selection (task 32). Blocking is the whole point of
//! a header: everything painted before the strip stops being hovered while the
//! pointer is in it, so the press belongs to the window move alone. `_except_scroll`
//! is deliberate — the space view's band is a *fade* over live content, and a
//! wheel gesture that starts under it must still scroll the page (hitboxes
//! stay in the hit test for scroll, which is what `should_handle_scroll`
//! reads). Children of the strip are painted after it and so are unaffected —
//! the Record's section tabs keep their own clicks.

use std::cell::Cell;

use crate::overlay::{Contain as _, Overlay};

/// **The drag band's height — one constant for every window.**
///
/// The band is a hit target the user cannot see, so its height being uniform is
/// what makes "grab the top of any window" mean the same thing everywhere. It
/// also has to clear the macOS traffic lights (`traffic_light_position` puts
/// them at y=11, ~12px tall) and the Linux CSD control cluster.
///
/// Settings used to reserve 44px here while every other window used 36 — an
/// 8px strip of extra dead space that, once the band started *blocking* the
/// mouse (task 32), swallowed the top of the Backends pane's tab strip and made
/// the tabs unclickable. The panes carry their own heading padding; the band
/// does not fake it for them.
pub const DRAG_BAND_HEIGHT: gpui::Pixels = gpui::px(36.);

use gpui::{
    App, Div, InteractiveElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    Pixels, Stateful, Styled, Window, div, prelude::FluentBuilder,
};
use gpui_component::InteractiveElementExt;

/// Attach the arm-on-down / move-on-first-move / double-click-to-zoom window
/// drag gesture to an existing stateful element (e.g. a window's title strip).
/// `key` scopes the element-owned armed flag — use the strip's element id.
///
/// The same gesture serves both platforms (`start_window_move` is wired on
/// macOS and Wayland alike; the `WindowControlArea` hitbox path is a no-op on
/// both). The double-click and right-click affordances differ:
/// - macOS: double-click → `titlebar_double_click` (respects the user's
///   zoom-vs-minimize preference). No right-click menu.
/// - Linux CSD: double-click → `zoom_window` (maximize toggle, the desktop
///   idiom), right-click → `show_window_menu` (the *compositor's* window menu:
///   move / resize / workspace / always-on-top — the idiomatic window-scoped
///   menu, maintained by the compositor, one call).
pub(crate) fn make_draggable(
    el: Stateful<Div>,
    key: &'static str,
    window: &mut Window,
    cx: &mut App,
) -> Stateful<Div> {
    let armed = window.use_keyed_state(
        gpui::SharedString::from(format!("{key}-drag-arm")),
        cx,
        |_, _| Cell::new(false),
    );
    let on_down = armed.clone();
    let on_up = armed.clone();
    // Translucent chrome over live content — see `crate::overlay`: the press
    // is the band's own (the window move), while a wheel gesture that starts
    // in it is plainly aimed at the page showing through and passes down.
    el.contain_mouse(Overlay::Fade)
        .on_mouse_down(
            MouseButton::Left,
            move |ev: &MouseDownEvent, window: &mut Window, cx: &mut App| {
                // A press at the very window edge belongs to the CSD resize band
                // (which reaches a few px inside the frame); arming a move there
                // would race the resize the chrome backdrop is about to start.
                // No-op off Linux (`in_resize_band` is always false there).
                if crate::chrome::in_resize_band(window, ev.position) {
                    return;
                }
                on_down.read(cx).set(true);
            },
        )
        .on_mouse_up(MouseButton::Left, move |_: &MouseUpEvent, _, cx| {
            on_up.read(cx).set(false);
        })
        .on_mouse_move(move |_: &MouseMoveEvent, window: &mut Window, cx| {
            if armed.read(cx).get() {
                armed.read(cx).set(false);
                window.start_window_move();
            }
        })
        .on_double_click(|_, window, _| {
            if cfg!(target_os = "macos") {
                window.titlebar_double_click();
            } else {
                window.zoom_window();
            }
        })
        // Right-click → the compositor's window menu (move / resize / workspace).
        // Linux-only: `show_window_menu` is a no-op on macOS, so we skip registering
        // a dead right-click hitbox over the drag strip there.
        .when(!cfg!(target_os = "macos"), |el| {
            el.on_mouse_down(
                MouseButton::Right,
                |ev: &MouseDownEvent, window: &mut Window, _| {
                    window.show_window_menu(ev.position);
                },
            )
        })
}

/// A full-width, transparent drag band absolutely positioned over a window's
/// top titlebar reserve. Overlay it above the content of windows whose top
/// strip is otherwise just empty traffic-light reserve.
pub(crate) fn drag_band(
    id: &'static str,
    height: Pixels,
    window: &mut Window,
    cx: &mut App,
) -> Stateful<Div> {
    make_draggable(
        div().id(id).absolute().top_0().left_0().right_0().h(height),
        id,
        window,
        cx,
    )
}
