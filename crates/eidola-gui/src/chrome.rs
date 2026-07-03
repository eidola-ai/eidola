//! Linux client-side window decorations — the window chrome layer.
//!
//! On Wayland there is no server-drawn titlebar unless the compositor offers
//! one via `zxdg_toplevel_decoration_v1`: KDE's KWin does, GNOME's Mutter
//! never does. Every Eidola window therefore *requests*
//! [`WindowDecorations::Client`] and draws its own chrome when the request is
//! granted (`Decorations::Client`), while degrading to no chrome at all when
//! the compositor insists on server-side decorations (`Decorations::Server` —
//! KDE with the user's "force SSD" preference). macOS never reaches this
//! module's rendering paths: [`ChromeRoot::wrap`] is an identity there.
//!
//! The layer has three parts:
//!
//! - [`ChromeRoot`] — a wrapper view installed between `gpui_component::Root`
//!   and each window's real view. It renders the zed-style CSD frame: shadow
//!   padding on untiled edges, a 1px `theme.border` frame with rounded *top*
//!   corners, resize-edge mouse handling (`start_window_resize`), the resize
//!   cursor, and the window-controls overlay.
//! - [`window_controls`] — the ghost-button cluster (minimize / zoom / close)
//!   painted at the window's top-right, in our quiet voice rather than a
//!   fake-native theme. Buttons render only for capabilities the compositor
//!   reports (`window.window_controls()`); close is always present. Deferred
//!   at a high priority so window chrome paints (and hit-tests) above any
//!   view content, including other deferred layers.
//! - Corner helpers — [`round_top_client_corners`] for full-bleed view
//!   surfaces and [`controls_reserve`] for view layouts that keep their own
//!   content clear of the cluster.
//!
//! **Top corners round; bottom corners stay square (settled).** gpui content
//! masks are strictly rectangular, so a rounded window frame cannot *clip*
//! child surfaces — every surface touching a rounded corner must round
//! itself. The top of each window is chrome we own (strips, bands, empty
//! reserve), so the audit is cheap; the bottom edge is live content (docked
//! composer, error band, minimap tail) where the audit would be invasive.
//! Square bottom corners are the same trade Chromium and VS Code ship on
//! Wayland. Any new full-bleed surface that touches the window's top edge
//! must apply [`round_top_client_corners`].

#![cfg_attr(target_os = "macos", allow(unused))]

use gpui::{
    AnyView, App, AppContext, Bounds, Context, CursorStyle, Decorations, Div, Global, Hsla,
    InteractiveElement, IntoElement, MouseButton, ParentElement, Pixels, Point, Render, ResizeEdge,
    Size, Stateful, StatefulInteractiveElement, Styled, Tiling, Window, canvas, div, point,
    prelude::FluentBuilder, px, size,
};
use gpui_component::{ActiveTheme, Icon, IconName, Sizable, StyledExt, h_flex};

use crate::probe::Probe;

/// Shadow / resize-border reach outside the visible frame, on untiled edges.
pub(crate) const SHADOW_SIZE: Pixels = px(12.);

/// Window corner radius when client-decorated and untiled. 12px matches the
/// GNOME (Adwaita) window radius so the frame sits naturally among native
/// windows.
pub(crate) const CORNER_RADIUS: Pixels = px(12.);

/// Height of the window-controls buttons (matches the 36px title strip).
const CONTROL_HEIGHT: Pixels = px(36.);
/// Width of one window-control button.
const CONTROL_WIDTH: Pixels = px(40.);

/// Is this window currently drawing its own decorations? False on macOS, on
/// server-decorated Wayland windows, and in test contexts.
pub(crate) fn is_client_decorated(window: &Window) -> bool {
    cfg!(target_os = "linux") && matches!(window.window_decorations(), Decorations::Client { .. })
}

/// Horizontal room a view's top-right content must leave for the
/// window-controls cluster. Zero when no chrome is drawn (macOS, SSD).
pub(crate) fn controls_reserve(window: &Window) -> Pixels {
    if !is_client_decorated(window) {
        return px(0.);
    }
    let caps = window.window_controls();
    let mut buttons = 1; // close is always shown
    if caps.maximize {
        buttons += 1;
    }
    if caps.minimize {
        buttons += 1;
    }
    CONTROL_WIDTH * buttons as f32
}

/// Round the *top* corners of a full-bleed surface to match the window's
/// client-side frame (no-op on macOS / SSD / tiled edges). Apply to any
/// surface that touches the window's top edge — see the module docs.
pub(crate) fn round_top_client_corners<E: Styled>(el: E, window: &Window) -> E {
    round_tr_client_corner(round_tl_client_corner(el, window), window)
}

/// Round only the top-left window corner — for a surface that owns the
/// window's top-left but not its top-right (e.g. the Settings nav band).
pub(crate) fn round_tl_client_corner<E: Styled>(el: E, window: &Window) -> E {
    let Decorations::Client { tiling } = window.window_decorations() else {
        return el;
    };
    if !cfg!(target_os = "linux") || tiling.top || tiling.left {
        return el;
    }
    el.rounded_tl(CORNER_RADIUS)
}

/// Round only the top-right window corner (see [`round_tl_client_corner`]).
pub(crate) fn round_tr_client_corner<E: Styled>(el: E, window: &Window) -> E {
    let Decorations::Client { tiling } = window.window_decorations() else {
        return el;
    };
    if !cfg!(target_os = "linux") || tiling.top || tiling.right {
        return el;
    }
    el.rounded_tr(CORNER_RADIUS)
}

/// The wrapper view between `gpui_component::Root` and a window's real view.
/// On macOS [`ChromeRoot::wrap`] returns the view unchanged (the render tree
/// is bit-identical to before this module existed); on Linux it interposes
/// the CSD frame and the primary menu.
pub struct ChromeRoot {
    child: AnyView,
    /// Whether the primary-menu popover is open. Lives here (not on views)
    /// because the menu is window chrome — every window gets the same one.
    menu_open: bool,
}

impl ChromeRoot {
    /// Public (not just crate-visible) so integration tests can wrap a
    /// view the way the production window builders do.
    pub fn wrap(child: AnyView, cx: &mut App) -> AnyView {
        if cfg!(target_os = "macos") {
            return child;
        }
        cx.new(|_| ChromeRoot {
            child,
            menu_open: false,
        })
        .into()
    }
}

gpui::actions!(chrome, [TogglePrimaryMenu]);

/// The hovered resize edge, shared between the backdrop's mouse handling and
/// the cursor canvas (zed's `GlobalResizeEdge` pattern).
struct HoveredResizeEdge(ResizeEdge);
impl Global for HoveredResizeEdge {}

impl Render for ChromeRoot {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let decorations = window.window_decorations();

        match decorations {
            Decorations::Server => {
                // A native (server-drawn) titlebar handles controls, drag,
                // and resize — but the primary menu is *app* chrome, not
                // window management, so it renders in both modes.
                window.set_client_inset(px(0.));
                div()
                    .id("window-chrome")
                    .relative()
                    .size_full()
                    .on_action(
                        cx.listener(|this, _: &TogglePrimaryMenu, _, cx| this.toggle_menu(cx)),
                    )
                    .child(self.child.clone())
                    .children(self.menu_layer(window, cx))
                    .into_any_element()
            }
            Decorations::Client { tiling } => {
                window.set_client_inset(SHADOW_SIZE);
                self.render_client_frame(tiling, window, cx)
                    .into_any_element()
            }
        }
    }
}

impl ChromeRoot {
    fn render_client_frame(
        &self,
        tiling: Tiling,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let theme = cx.theme();
        let border = theme.border;
        let background = theme.background;

        div()
            .id("window-backdrop")
            .size_full()
            .map(|el| round_top_client_corners(el, window))
            .when(!tiling.top, |el| el.pt(SHADOW_SIZE))
            .when(!tiling.bottom, |el| el.pb(SHADOW_SIZE))
            .when(!tiling.left, |el| el.pl(SHADOW_SIZE))
            .when(!tiling.right, |el| el.pr(SHADOW_SIZE))
            // Re-render when the hovered resize edge changes so the cursor
            // canvas below sees fresh state; start an interactive resize on
            // mouse-down in the border region.
            .on_mouse_move(cx.listener(move |_, e: &gpui::MouseMoveEvent, window, cx| {
                let size = window.window_bounds().get_bounds().size;
                let new_edge = resize_edge(e.position, SHADOW_SIZE, size, tiling);
                let old_edge = cx.try_global::<HoveredResizeEdge>().map(|e| e.0);
                if new_edge != old_edge {
                    cx.notify();
                }
            }))
            .on_mouse_down(MouseButton::Left, move |e, window, _| {
                let size = window.window_bounds().get_bounds().size;
                if let Some(edge) = resize_edge(e.position, SHADOW_SIZE, size, tiling) {
                    window.start_window_resize(edge);
                }
            })
            .on_action(cx.listener(|this, _: &TogglePrimaryMenu, _, cx| this.toggle_menu(cx)))
            .child(
                // The inner frame. Chrome overlays (controls, primary menu)
                // live *inside* it so their absolute positions resolve
                // against the visible frame, not the shadow-padded backdrop.
                div()
                    .relative()
                    .size_full()
                    .cursor(CursorStyle::Arrow)
                    .map(|el| round_top_client_corners(el, window))
                    .bg(background)
                    .border_color(border)
                    .when(!tiling.top, |el| el.border_t_1())
                    .when(!tiling.bottom, |el| el.border_b_1())
                    .when(!tiling.left, |el| el.border_l_1())
                    .when(!tiling.right, |el| el.border_r_1())
                    .when(!tiling.is_tiled(), |el| {
                        el.shadow(vec![
                            gpui::BoxShadow::new(
                                px(0.),
                                px(2.),
                                Hsla {
                                    h: 0.,
                                    s: 0.,
                                    l: 0.,
                                    a: 0.35,
                                },
                            )
                            .blur_radius(SHADOW_SIZE - px(4.)),
                        ])
                    })
                    // Content mouse-moves are the content's business — don't
                    // let them bubble to the backdrop's edge tracking.
                    .on_mouse_move(|_, _, cx| cx.stop_propagation())
                    .child(self.child.clone())
                    // Window chrome paints (and hit-tests) above everything
                    // the view renders, including its own deferred layers
                    // (the space view's floating composer at priority 0 and
                    // minimap at priority 1).
                    .children(
                        window_controls(window, cx)
                            .map(|controls| gpui::deferred(controls).with_priority(100)),
                    )
                    .children(self.menu_layer(window, cx)),
            )
            // Resize cursor: a paint-phase canvas that reads the hovered edge
            // each frame and sets the cursor over the whole window while the
            // pointer is in the shadow/border region.
            .child(
                canvas(
                    |_, window, _| {
                        window.insert_hitbox(
                            Bounds::new(
                                point(px(0.), px(0.)),
                                window.window_bounds().get_bounds().size,
                            ),
                            gpui::HitboxBehavior::Normal,
                        )
                    },
                    move |_, hitbox, window, cx| {
                        let mouse = window.mouse_position();
                        let size = window.window_bounds().get_bounds().size;
                        let Some(edge) = resize_edge(mouse, SHADOW_SIZE, size, tiling) else {
                            return;
                        };
                        cx.set_global(HoveredResizeEdge(edge));
                        window.set_cursor_style(
                            match edge {
                                ResizeEdge::Top | ResizeEdge::Bottom => CursorStyle::ResizeUpDown,
                                ResizeEdge::Left | ResizeEdge::Right => {
                                    CursorStyle::ResizeLeftRight
                                }
                                ResizeEdge::TopLeft | ResizeEdge::BottomRight => {
                                    CursorStyle::ResizeUpLeftDownRight
                                }
                                ResizeEdge::TopRight | ResizeEdge::BottomLeft => {
                                    CursorStyle::ResizeUpRightDownLeft
                                }
                            },
                            &hitbox,
                        );
                    },
                )
                .size_full()
                .absolute(),
            )
    }
}

impl ChromeRoot {
    fn toggle_menu(&mut self, cx: &mut Context<Self>) {
        self.menu_open = !self.menu_open;
        cx.notify();
    }

    /// The primary menu — the Linux stand-in for the macOS app/File menus
    /// (GNOME's "primary menu" idiom, in our voice): a quiet italic "Eidola"
    /// wordmark in the window's top-left corner that opens a popover of the
    /// app-scoped commands, each with its chord rendered beside it (so the
    /// popover doubles as the keyboard-reference card). Rendered in both
    /// decoration modes — a native titlebar replaces window *management*,
    /// not the app's command surface. Returns the affordance and (when open)
    /// the popover panel, deferred so chrome paints above view content.
    ///
    /// Dismissal: click-out, a second click on the wordmark, F10 (the
    /// desktop-standard primary-menu key, bound in `install_keybindings`),
    /// or selecting an item. Keyboard navigation inside the popover is
    /// deliberately absent in v1 — every item shows its direct chord.
    fn menu_layer(&self, window: &Window, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        if cfg!(target_os = "macos") {
            return Vec::new();
        }
        let theme = cx.theme();

        let wordmark = div()
            .id("chrome-menu")
            .probe("chrome/menu", gpui::Role::Button, "Eidola menu")
            .absolute()
            .top_0()
            .left_0()
            .h(CONTROL_HEIGHT)
            .px_4()
            .flex()
            .items_center()
            .text_sm()
            .italic()
            .cursor_pointer()
            .text_color(if self.menu_open {
                theme.foreground
            } else {
                theme.muted_foreground
            })
            .hover(|s| s.text_color(theme.foreground))
            // Both-phase press swallow so the drag band beneath never arms.
            .on_mouse_down(MouseButton::Left, |_, window, cx| {
                window.prevent_default();
                cx.stop_propagation();
            })
            .on_click(cx.listener(|this, _, _, cx| {
                cx.stop_propagation();
                this.toggle_menu(cx);
            }))
            .child("Eidola");

        let mut layer: Vec<gpui::AnyElement> = vec![
            gpui::deferred(wordmark)
                .with_priority(100)
                .into_any_element(),
        ];

        if self.menu_open {
            layer.push(
                gpui::deferred(self.render_menu_panel(window, cx))
                    .with_priority(101)
                    .into_any_element(),
            );
        }
        layer
    }

    fn render_menu_panel(&self, _window: &Window, cx: &Context<Self>) -> Stateful<Div> {
        use crate::actions::{
            About, CheckForUpdates, NewSpace, OpenLibrary, OpenRecord, OpenSettings, Quit,
            primary_chord, primary_shift_chord,
        };
        let theme = cx.theme();

        let separator = || div().my_1().h(px(1.)).w_full().bg(theme.border);

        gpui_component::v_flex()
            .id("chrome-menu-panel")
            .probe("chrome/menu/panel", gpui::Role::Group, "Eidola menu")
            .occlude()
            .absolute()
            .top(CONTROL_HEIGHT)
            .left(px(8.))
            .w(px(260.))
            .popover_style(cx)
            .py_1()
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.menu_open = false;
                cx.notify();
            }))
            .child(menu_item(
                "new-space",
                "New Space",
                Some(primary_chord("N")),
                |w, cx| w.dispatch_action(Box::new(NewSpace), cx),
                cx,
            ))
            .child(menu_item(
                "library",
                "Library…",
                Some(primary_chord("L")),
                |w, cx| w.dispatch_action(Box::new(OpenLibrary), cx),
                cx,
            ))
            .child(menu_item(
                "record",
                "Record…",
                Some(primary_shift_chord("L")),
                |w, cx| w.dispatch_action(Box::new(OpenRecord), cx),
                cx,
            ))
            .child(separator())
            .child(menu_item(
                "settings",
                "Settings…",
                Some(primary_chord(",")),
                |w, cx| w.dispatch_action(Box::new(OpenSettings), cx),
                cx,
            ))
            .child(menu_item(
                "updates",
                "Check for Updates…",
                None,
                |w, cx| w.dispatch_action(Box::new(CheckForUpdates), cx),
                cx,
            ))
            .child(menu_item(
                "about",
                "About Eidola",
                None,
                |w, cx| w.dispatch_action(Box::new(About), cx),
                cx,
            ))
            .child(separator())
            .child(menu_item(
                "quit",
                "Quit",
                Some(primary_chord("Q")),
                |w, cx| w.dispatch_action(Box::new(Quit), cx),
                cx,
            ))
    }
}

/// One row of the primary-menu popover: label left, muted chord right.
/// Selecting an item dispatches its action from the window (reaching the
/// global handlers in `lib.rs::install_action_handlers`) and closes the menu.
fn menu_item(
    slug: &'static str,
    label: &'static str,
    chord: Option<String>,
    dispatch: fn(&mut Window, &mut App),
    cx: &Context<ChromeRoot>,
) -> Stateful<Div> {
    let theme = cx.theme();
    h_flex()
        .id(slug)
        .probe(format!("chrome/menu/{slug}"), gpui::Role::Button, label)
        .px_3()
        .py_1p5()
        .gap_6()
        .justify_between()
        .cursor_pointer()
        .hover(|s| s.bg(theme.muted.opacity(0.5)))
        .on_click(cx.listener(move |this, _, window, cx| {
            cx.stop_propagation();
            this.menu_open = false;
            cx.notify();
            dispatch(window, cx);
        }))
        .child(div().text_sm().child(label))
        .children(chord.map(|c| div().text_xs().text_color(theme.muted_foreground).child(c)))
}

/// The window-controls cluster: quiet ghost buttons at the window's
/// top-right — minimize / zoom (per compositor capabilities) and close.
/// `None` when this window draws no chrome (macOS, SSD).
fn window_controls(window: &Window, cx: &App) -> Option<Div> {
    if !is_client_decorated(window) {
        return None;
    }
    let caps = window.window_controls();
    let maximized = window.is_maximized();

    Some(
        h_flex()
            .absolute()
            .top_0()
            .right_0()
            .h(CONTROL_HEIGHT)
            .when(caps.minimize, |el| {
                el.child(control_button(
                    "chrome-minimize",
                    "chrome/controls/minimize",
                    "Minimize window",
                    IconName::WindowMinimize,
                    cx,
                    |window, _| window.minimize_window(),
                ))
            })
            .when(caps.maximize, |el| {
                el.child(control_button(
                    "chrome-zoom",
                    "chrome/controls/zoom",
                    if maximized {
                        "Restore window size"
                    } else {
                        "Maximize window"
                    },
                    if maximized {
                        IconName::WindowRestore
                    } else {
                        IconName::WindowMaximize
                    },
                    cx,
                    |window, _| window.zoom_window(),
                ))
            })
            .child(control_button(
                "chrome-close",
                "chrome/controls/close",
                "Close window",
                IconName::WindowClose,
                cx,
                |window, _| window.remove_window(),
            )),
    )
}

fn control_button(
    id: &'static str,
    probe_name: &'static str,
    label: &'static str,
    icon: IconName,
    cx: &App,
    on_click: fn(&mut Window, &mut App),
) -> Stateful<Div> {
    let theme = cx.theme();
    div()
        .id(id)
        .probe(probe_name, gpui::Role::Button, label)
        .flex()
        .items_center()
        .justify_center()
        .w(CONTROL_WIDTH)
        .h_full()
        .text_color(theme.muted_foreground)
        .hover(|s| s.bg(theme.muted.opacity(0.5)).text_color(theme.foreground))
        // Swallow the press so the drag band beneath never arms a window
        // move (same both-phase blocking as the Library's reveal buttons).
        .on_mouse_down(MouseButton::Left, |_, window, cx| {
            window.prevent_default();
            cx.stop_propagation();
        })
        .on_click(move |_, window, cx| {
            cx.stop_propagation();
            on_click(window, cx);
        })
        .child(Icon::new(icon).small())
}

/// Which resize edge (if any) the position falls on, honoring tiled edges.
/// Corner zones reach 1.5× the shadow size, mirroring zed.
fn resize_edge(
    pos: Point<Pixels>,
    shadow_size: Pixels,
    window_size: Size<Pixels>,
    tiling: Tiling,
) -> Option<ResizeEdge> {
    let bounds = Bounds::new(Point::default(), window_size).inset(shadow_size * 1.5);
    if bounds.contains(&pos) {
        return None;
    }

    let corner_size = size(shadow_size * 1.5, shadow_size * 1.5);
    let top_left = Bounds::new(point(px(0.), px(0.)), corner_size);
    if !tiling.top && !tiling.left && top_left.contains(&pos) {
        return Some(ResizeEdge::TopLeft);
    }
    let top_right = Bounds::new(
        point(window_size.width - corner_size.width, px(0.)),
        corner_size,
    );
    if !tiling.top && !tiling.right && top_right.contains(&pos) {
        return Some(ResizeEdge::TopRight);
    }
    let bottom_left = Bounds::new(
        point(px(0.), window_size.height - corner_size.height),
        corner_size,
    );
    if !tiling.bottom && !tiling.left && bottom_left.contains(&pos) {
        return Some(ResizeEdge::BottomLeft);
    }
    let bottom_right = Bounds::new(
        point(
            window_size.width - corner_size.width,
            window_size.height - corner_size.height,
        ),
        corner_size,
    );
    if !tiling.bottom && !tiling.right && bottom_right.contains(&pos) {
        return Some(ResizeEdge::BottomRight);
    }

    if !tiling.top && pos.y < shadow_size {
        Some(ResizeEdge::Top)
    } else if !tiling.bottom && pos.y > window_size.height - shadow_size {
        Some(ResizeEdge::Bottom)
    } else if !tiling.left && pos.x < shadow_size {
        Some(ResizeEdge::Left)
    } else if !tiling.right && pos.x > window_size.width - shadow_size {
        Some(ResizeEdge::Right)
    } else {
        None
    }
}
