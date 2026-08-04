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
//! **This module is the single CSD authority.** `gpui_component::Root` ships
//! its own Linux CSD layer (`window_border()`: 12px shadow padding, a 1px
//! `theme.window_border` frame, resize hit zones, and an opaque
//! `bg(theme.background)` across the whole window). Left active it stacks
//! under ours: its background fills our shadow padding with an opaque cliff
//! and its resize cursors land at the wrong edge. [`themed_root`] therefore
//! constructs every window's `Root` with that layer disabled outright via
//! `Root::bordered(false)` (upstream gpui-component `#2466`: `render` returns
//! the inner content directly, skipping the `window_border()` wrapper
//! entirely) plus a transparent background. Belt-and-braces, the Circadian
//! palettes also pin `window_border` transparent so a vestigial frame could
//! never paint even if the flag regressed.
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
//! **Containment first, corner duty second.** The inner frame clips every
//! descendant to its rect (`overflow_hidden`) — and gpui deferred draws
//! re-apply the content mask captured at their tree position, so even the
//! space view's deferred composer/minimap overlays cannot paint into the
//! shadow band. What the clip *cannot* do is follow the curve: gpui content
//! masks are strictly rectangular, so the rounded corner notches stay each
//! corner-touching surface's own duty. All four corners round: window roots
//! apply [`round_client_corners`]; bands that own part of an edge use the
//! top/bottom/single-corner variants (Settings nav band: `tl` + `bl`; the
//! space composer bar: bottom — its geometry is clamped so its bottom edge
//! always coincides with the window's); full-height edge strips keep
//! [`corner_clearance`] away from the arcs (the minimap). The helpers are
//! no-ops on macOS / SSD / tiled edges, so views apply them unconditionally.

#![cfg_attr(target_os = "macos", allow(unused))]

use gpui::{
    AnyView, App, AppContext, Bounds, Context, CursorStyle, Decorations, Div, Edges, Hsla,
    InteractiveElement, IntoElement, MouseButton, ParentElement, Pixels, Point, Render, ResizeEdge,
    SharedString, Size, Stateful, StatefulInteractiveElement, Styled, Tiling, Window, div, point,
    prelude::FluentBuilder, px, size,
};
use gpui_component::{ActiveTheme, Icon, IconName, Root, Sizable, StyledExt, h_flex};

use crate::probe::Probe;

/// Shadow / resize-border reach outside the visible frame, on untiled edges.
const SHADOW_SIZE: Pixels = px(12.);

/// Width of the visible frame's border (the `.border_*_1()` calls below).
/// Part of [`content_insets`] so view-side geometry math lands exactly on the
/// frame's content box.
const FRAME_BORDER: Pixels = px(1.);

/// How far the resize hit band reaches *inside* the visible frame edge, so
/// the resize cursor is discoverable at the border itself, not only out in
/// the (invisible once the shadow fades) margin.
const RESIZE_INNER_REACH: Pixels = px(4.);

/// Window corner radius when client-decorated and untiled. 12px matches the
/// GNOME (Adwaita) window radius so the frame sits naturally among native
/// windows.
const CORNER_RADIUS: Pixels = px(12.);

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

/// Construct the `gpui_component::Root` for an Eidola window. On Linux,
/// Root's built-in CSD layer is disabled outright — `bordered(false)` makes
/// `Root::render` return its inner content directly instead of wrapping it in
/// `window_border()`, so there is no second ring of shadow padding, no
/// competing resize hit zones at the wrong edge, and no vestigial 1px frame.
/// The background is made transparent (Root's default opaque
/// `bg(theme.background)` would fill [`ChromeRoot`]'s shadow padding with a
/// hard-edged cliff instead of letting the desktop show through). The
/// background moves into the chrome layer: the client frame paints it inside
/// the visible border, and the `Decorations::Server` arm paints it edge to
/// edge. See the module docs.
pub(crate) fn themed_root(view: AnyView, window: &mut Window, cx: &mut Context<Root>) -> Root {
    let root = Root::new(view, window, cx);
    if cfg!(target_os = "linux") {
        root.bordered(false).bg(gpui::transparent_black())
    } else {
        root
    }
}

/// Per-side distance from the window surface's edge to the visible frame's
/// content box: shadow padding + frame border on untiled edges when client-
/// decorated, zero everywhere else (macOS, SSD, tests, tiled edges).
fn content_insets(window: &Window) -> Edges<Pixels> {
    if !cfg!(target_os = "linux") {
        return Edges::default();
    }
    let Decorations::Client { tiling } = window.window_decorations() else {
        return Edges::default();
    };
    let side = SHADOW_SIZE + FRAME_BORDER;
    let inset = |tiled: bool| if tiled { px(0.) } else { side };
    Edges {
        top: inset(tiling.top),
        bottom: inset(tiling.bottom),
        left: inset(tiling.left),
        right: inset(tiling.right),
    }
}

/// The size of the window's *content box* — the viewport minus the chrome's
/// shadow padding and frame border. Any view math that anchors to the window
/// bottom/right (floating overlays, scroll ranges, dock positions) must use
/// this instead of `window.viewport_size()`, or the result lands in the
/// shadow band. Identical to the viewport when no chrome is drawn.
pub(crate) fn content_size(window: &Window) -> Size<Pixels> {
    let viewport = window.viewport_size();
    let insets = content_insets(window);
    size(
        viewport.width - insets.left - insets.right,
        viewport.height - insets.top - insets.bottom,
    )
}

/// The window content-box height — the same value onboarding's `min_h` slide
/// floor is measured against. Test seam for the size-to-content contract.
#[doc(hidden)]
pub fn content_height_for_test(window: &Window) -> f32 {
    content_size(window).height.as_f32()
}

/// Whether a window-coordinate position falls in the resize hit band (the
/// shadow margin plus [`RESIZE_INNER_REACH`] inside the frame). Drag bands
/// check this on mouse-down so a press at the very edge starts a *resize*,
/// never arms a window move.
pub(crate) fn in_resize_band(window: &Window, pos: Point<Pixels>) -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    let Decorations::Client { tiling } = window.window_decorations() else {
        return false;
    };
    let size = window.window_bounds().get_bounds().size;
    resize_edge(pos, SHADOW_SIZE, size, tiling).is_some()
}

/// Round all four corners of a full-bleed surface to match the window's
/// client-side frame (no-op on macOS / SSD / tiled edges). Apply to any
/// surface that spans the whole window (view roots) — see the module docs.
pub(crate) fn round_client_corners<E: Styled>(el: E, window: &Window) -> E {
    round_bottom_client_corners(round_top_client_corners(el, window), window)
}

/// Round the *top* corners of a full-bleed surface to match the window's
/// client-side frame (no-op on macOS / SSD / tiled edges). Apply to any
/// surface that owns the window's top edge but not its bottom (e.g. the
/// space view's gradient title band) — see the module docs.
pub(crate) fn round_top_client_corners<E: Styled>(el: E, window: &Window) -> E {
    round_tr_client_corner(round_tl_client_corner(el, window), window)
}

/// Round the *bottom* corners (see [`round_top_client_corners`]).
pub(crate) fn round_bottom_client_corners<E: Styled>(el: E, window: &Window) -> E {
    round_br_client_corner(round_bl_client_corner(el, window), window)
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

/// Round only the bottom-left window corner — for a surface that owns the
/// window's bottom-left but not its bottom-right (e.g. the Settings nav
/// band, which spans the window's full height on the left).
pub(crate) fn round_bl_client_corner<E: Styled>(el: E, window: &Window) -> E {
    let Decorations::Client { tiling } = window.window_decorations() else {
        return el;
    };
    if !cfg!(target_os = "linux") || tiling.bottom || tiling.left {
        return el;
    }
    el.rounded_bl(CORNER_RADIUS)
}

/// Round only the bottom-right window corner (see [`round_bl_client_corner`]).
pub(crate) fn round_br_client_corner<E: Styled>(el: E, window: &Window) -> E {
    let Decorations::Client { tiling } = window.window_decorations() else {
        return el;
    };
    if !cfg!(target_os = "linux") || tiling.bottom || tiling.right {
        return el;
    }
    el.rounded_br(CORNER_RADIUS)
}

/// Vertical clearance a full-height *right-edge* overlay strip (the space
/// view's minimap) needs at each end to stay out of the window's rounded
/// corner arcs. Zero whenever the right-side corners aren't rounded
/// (macOS, SSD, tests, tiled).
pub(crate) fn corner_clearance(window: &Window) -> Pixels {
    let Decorations::Client { tiling } = window.window_decorations() else {
        return px(0.);
    };
    if !cfg!(target_os = "linux") || tiling.right || (tiling.top && tiling.bottom) {
        return px(0.);
    }
    CORNER_RADIUS
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

impl Render for ChromeRoot {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let decorations = window.window_decorations();

        match decorations {
            Decorations::Server => {
                // A native (server-drawn) titlebar handles controls, drag,
                // and resize — but the primary menu is *app* chrome, not
                // window management, so it renders in both modes. The window
                // background is painted here (edge to edge) because
                // `themed_root` makes Root's own background transparent.
                window.set_client_inset(px(0.));
                div()
                    .id("window-chrome")
                    .relative()
                    .size_full()
                    .bg(cx.theme().background)
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
            .map(|el| round_client_corners(el, window))
            .when(!tiling.top, |el| el.pt(SHADOW_SIZE))
            .when(!tiling.bottom, |el| el.pb(SHADOW_SIZE))
            .when(!tiling.left, |el| el.pl(SHADOW_SIZE))
            .when(!tiling.right, |el| el.pr(SHADOW_SIZE))
            // Start an interactive resize on mouse-down in the hit band.
            .on_mouse_down(MouseButton::Left, move |e, window, cx| {
                let size = window.window_bounds().get_bounds().size;
                if let Some(edge) = resize_edge(e.position, SHADOW_SIZE, size, tiling) {
                    // Stop the bubble so Root's window_border (which keeps a
                    // vestigial resize handler of its own) can't issue a
                    // second, competing resize request.
                    cx.stop_propagation();
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
                    .map(|el| round_client_corners(el, window))
                    // Containment, not convention: clip every descendant —
                    // including deferred overlays, which re-apply the content
                    // mask captured at their tree position — to the frame's
                    // rect, so nothing can ever paint over the shadow band.
                    // The mask is rectangular (gpui masks carry no radii), so
                    // the rounded corner *notches* remain each corner-touching
                    // surface's own duty — see the module docs.
                    .overflow_hidden()
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
                    .child(self.child.clone())
                    // Window chrome paints (and hit-tests) above everything
                    // the view renders: the deferred pass runs after the
                    // whole normal-pass tree (and the space view's layered
                    // composer/minimap stay in the normal pass).
                    .children(
                        window_controls(window, cx)
                            .map(|controls| gpui::deferred(controls).with_priority(100)),
                    )
                    .children(self.menu_layer(window, cx)),
            )
            // Resize cursors: declarative zones over the hit band. Placed
            // after the inner frame so they sit above content at the edges.
            .children(resize_cursor_zones(tiling))
    }
}

/// Transparent, listener-less strips and corner squares covering the resize
/// hit band, each carrying its `.cursor(...)`. gpui resolves the cursor per
/// pointer move against painted hitboxes, so this needs no re-render
/// choreography and cannot go stale the way an imperatively set cursor can
/// (an earlier paint-phase-canvas version requested the cursor with a
/// full-window hitbox, which outlived the pointer's stay in the band until
/// the next repaint — the "stuck resize arrow"). The zones carry no
/// listeners: the actual resize starts from the backdrop's mouse-down via
/// [`resize_edge`], whose geometry these zones mirror (strips: the shadow
/// margin plus [`RESIZE_INNER_REACH`] inside the frame edge; corners: 1.5×
/// the shadow size, pushed after the strips so they win hit-testing).
///
/// Coordinates are relative to the backdrop's padded content box (the
/// frame's outer edge), so the shadow margin is at negative offsets.
fn resize_cursor_zones(tiling: Tiling) -> Vec<gpui::AnyElement> {
    if tiling.top && tiling.bottom && tiling.left && tiling.right {
        return Vec::new();
    }

    // Outer offset (into the shadow margin) and strip thickness.
    let out = px(0.) - SHADOW_SIZE;
    let strip = SHADOW_SIZE + RESIZE_INNER_REACH;
    let corner = SHADOW_SIZE * 1.5;
    let zone = || div().absolute();

    let mut zones: Vec<gpui::AnyElement> = Vec::new();

    if !tiling.top {
        zones.push(
            zone()
                .top(out)
                .left(if tiling.left { px(0.) } else { out })
                .right(if tiling.right { px(0.) } else { out })
                .h(strip)
                .cursor(CursorStyle::ResizeUpDown)
                .into_any_element(),
        );
    }
    if !tiling.bottom {
        zones.push(
            zone()
                .bottom(out)
                .left(if tiling.left { px(0.) } else { out })
                .right(if tiling.right { px(0.) } else { out })
                .h(strip)
                .cursor(CursorStyle::ResizeUpDown)
                .into_any_element(),
        );
    }
    if !tiling.left {
        zones.push(
            zone()
                .left(out)
                .top(if tiling.top { px(0.) } else { out })
                .bottom(if tiling.bottom { px(0.) } else { out })
                .w(strip)
                .cursor(CursorStyle::ResizeLeftRight)
                .into_any_element(),
        );
    }
    if !tiling.right {
        zones.push(
            zone()
                .right(out)
                .top(if tiling.top { px(0.) } else { out })
                .bottom(if tiling.bottom { px(0.) } else { out })
                .w(strip)
                .cursor(CursorStyle::ResizeLeftRight)
                .into_any_element(),
        );
    }

    // Corners after strips so their diagonal cursors win where they overlap.
    if !tiling.top && !tiling.left {
        zones.push(
            zone()
                .top(out)
                .left(out)
                .w(corner)
                .h(corner)
                .cursor(CursorStyle::ResizeUpLeftDownRight)
                .into_any_element(),
        );
    }
    if !tiling.top && !tiling.right {
        zones.push(
            zone()
                .top(out)
                .right(out)
                .w(corner)
                .h(corner)
                .cursor(CursorStyle::ResizeUpRightDownLeft)
                .into_any_element(),
        );
    }
    if !tiling.bottom && !tiling.left {
        zones.push(
            zone()
                .bottom(out)
                .left(out)
                .w(corner)
                .h(corner)
                .cursor(CursorStyle::ResizeUpRightDownLeft)
                .into_any_element(),
        );
    }
    if !tiling.bottom && !tiling.right {
        zones.push(
            zone()
                .bottom(out)
                .right(out)
                .w(corner)
                .h(corner)
                .cursor(CursorStyle::ResizeUpLeftDownRight)
                .into_any_element(),
        );
    }

    zones
}

impl ChromeRoot {
    fn toggle_menu(&mut self, cx: &mut Context<Self>) {
        self.menu_open = !self.menu_open;
        cx.notify();
    }

    /// The primary menu — the Linux stand-in for the macOS app/Space menus
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
            About, ActualSize, CheckForUpdates, NewSpace, NewSpaceFromTemplate, OpenLibrary,
            OpenParticipants, OpenRecord, OpenSettings, Quit, Quote, QuoteInReply, ToggleInspector,
            ZoomIn, ZoomOut, primary_alt_chord, primary_chord, primary_shift_chord,
        };
        let theme = cx.theme();

        let separator = || div().my_1().h(px(1.)).w_full().bg(theme.border);

        // The live template registry (the macOS "New Space from Template ▸"
        // submenu, flattened into the popover's space-scoped group). Read at
        // render time: the popover re-renders on every open, so a template
        // created/renamed/removed in Settings is reflected the next time the
        // menu opens. Empty until the registry loads.
        let template_rows: Vec<Stateful<Div>> = cx
            .try_global::<crate::AppGlobal>()
            .map(|g| g.stores.templates.read(cx).list().to_vec())
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let template_id = t.id.clone();
                menu_item(
                    SharedString::from(format!("template/{i}")),
                    SharedString::from(format!("New Space from “{}”", t.title)),
                    None,
                    move |w, cx| {
                        w.dispatch_action(
                            Box::new(NewSpaceFromTemplate {
                                template_id: template_id.clone(),
                            }),
                            cx,
                        )
                    },
                    cx,
                )
            })
            .collect();

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
            // The space-scoped group (the macOS "Space" menu): New Space, the
            // per-template creators, and Participants… (a no-op without a
            // focused space window, matching the greyed macOS item). The zoom
            // trio is the macOS "View" menu, its own group next. Library/Record
            // are app-level and group with Settings below — mirroring the
            // macOS move of Library/Record up into the Eidola app menu.
            .child(menu_item(
                "new-space",
                "New Space",
                Some(primary_chord("N")),
                |w, cx| w.dispatch_action(Box::new(NewSpace), cx),
                cx,
            ))
            .children(template_rows)
            .child(menu_item(
                "participants",
                "Participants…",
                None,
                |w, cx| w.dispatch_action(Box::new(OpenParticipants), cx),
                cx,
            ))
            // The space inspector's other door (the space itself carries no
            // visual toggle) — a no-op without a focused space window, like
            // Participants… above.
            .child(menu_item(
                "inspector",
                "Show/Hide Inspector",
                Some(primary_alt_chord("I")),
                |w, cx| w.dispatch_action(Box::new(ToggleInspector), cx),
                cx,
            ))
            // The selection-scoped verbs (the macOS "Edit" menu's quote pair).
            // Linux has no Edit menu here — Undo/Cut/Copy/Paste are keyboard
            // only — but Quote has no chord, so the popover is its only
            // pointer route. It groups with the space-scoped items above it
            // because that is what it acts within.
            .child(menu_item(
                "quote",
                "Quote",
                None,
                |w, cx| w.dispatch_action(Box::new(Quote), cx),
                cx,
            ))
            .child(menu_item(
                "quote-in-reply",
                "Quote in Reply",
                None,
                |w, cx| w.dispatch_action(Box::new(QuoteInReply), cx),
                cx,
            ))
            .child(separator())
            .child(menu_item(
                "actual-size",
                "Actual Size",
                Some(primary_chord("0")),
                |w, cx| w.dispatch_action(Box::new(ActualSize), cx),
                cx,
            ))
            .child(menu_item(
                "zoom-in",
                "Zoom In",
                Some(primary_chord("+")),
                |w, cx| w.dispatch_action(Box::new(ZoomIn), cx),
                cx,
            ))
            .child(menu_item(
                "zoom-out",
                "Zoom Out",
                Some(primary_chord("-")),
                |w, cx| w.dispatch_action(Box::new(ZoomOut), cx),
                cx,
            ))
            .child(separator())
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
    slug: impl Into<SharedString>,
    label: impl Into<SharedString>,
    chord: Option<String>,
    dispatch: impl Fn(&mut Window, &mut App) + 'static,
    cx: &Context<ChromeRoot>,
) -> Stateful<Div> {
    let theme = cx.theme();
    let slug = slug.into();
    let label = label.into();
    h_flex()
        .id(slug.clone())
        .probe(
            format!("chrome/menu/{slug}"),
            gpui::Role::Button,
            label.clone(),
        )
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
/// The straight-edge band spans the whole shadow margin plus
/// [`RESIZE_INNER_REACH`] inside the visible frame (so the cursor flips to a
/// resize arrow *at* the border, where the eye looks for it); corner zones
/// reach 1.5× the shadow size, mirroring zed.
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

    let edge_reach = shadow_size + RESIZE_INNER_REACH;

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

    if !tiling.top && pos.y < edge_reach {
        Some(ResizeEdge::Top)
    } else if !tiling.bottom && pos.y > window_size.height - edge_reach {
        Some(ResizeEdge::Bottom)
    } else if !tiling.left && pos.x < edge_reach {
        Some(ResizeEdge::Left)
    } else if !tiling.right && pos.x > window_size.width - edge_reach {
        Some(ResizeEdge::Right)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win() -> Size<Pixels> {
        size(px(800.), px(600.))
    }

    fn edge_at(x: f32, y: f32) -> Option<ResizeEdge> {
        resize_edge(point(px(x), px(y)), SHADOW_SIZE, win(), Tiling::default())
    }

    #[test]
    fn band_spans_margin_and_inner_reach() {
        // Whole shadow margin hits…
        assert_eq!(edge_at(400., 2.), Some(ResizeEdge::Top));
        assert_eq!(edge_at(400., 11.), Some(ResizeEdge::Top));
        // …and so do the first few px inside the visible frame edge (12px).
        assert_eq!(edge_at(400., 15.), Some(ResizeEdge::Top));
        assert_eq!(edge_at(400., 585.), Some(ResizeEdge::Bottom));
        assert_eq!(edge_at(15., 300.), Some(ResizeEdge::Left));
        assert_eq!(edge_at(785., 300.), Some(ResizeEdge::Right));
        // Deeper into content is not a resize.
        assert_eq!(edge_at(400., 20.), None);
        assert_eq!(edge_at(400., 300.), None);
    }

    #[test]
    fn corners_take_precedence() {
        assert_eq!(edge_at(10., 10.), Some(ResizeEdge::TopLeft));
        assert_eq!(edge_at(790., 10.), Some(ResizeEdge::TopRight));
        assert_eq!(edge_at(10., 590.), Some(ResizeEdge::BottomLeft));
        assert_eq!(edge_at(790., 590.), Some(ResizeEdge::BottomRight));
    }

    #[test]
    fn tiled_edges_do_not_resize() {
        let tiling = Tiling {
            top: true,
            ..Tiling::default()
        };
        assert_eq!(
            resize_edge(point(px(400.), px(2.)), SHADOW_SIZE, win(), tiling),
            None
        );
        // The other edges keep working.
        assert_eq!(
            resize_edge(point(px(400.), px(585.)), SHADOW_SIZE, win(), tiling),
            Some(ResizeEdge::Bottom)
        );
    }
}
