//! The topology minimap — a right-edge bar whose rows are the levels of the
//! selected path. Each row's height is the *selected* post's real (cached)
//! height shared by every column in the row (unselected siblings are purely
//! topological), and the band gaps mirror the real inter-post spacing — so the
//! selected path is a true spatial map of the document. The selected branch is
//! drawn dark where it's on-screen and medium where it's scrolled off; siblings
//! (one horizontal gesture away) are drawn light.
//!
//! Unlike the mockup, positions come from the cached layout heights + the live
//! scroll offset, not a per-frame `canvas` sweep of every post — so the minimap
//! costs nothing per frame beyond reading the selection.

use gpui::{
    Animation, AnimationExt, AnyElement, Context, Hsla, InteractiveElement, IntoElement,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, WeakEntity, Window, div, point, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use crate::probe::Probe as _;

use super::model::{NodeSrc, TreeNode};
use super::{
    BAND_HEIGHT, MINIMAP_COL_GAP, MINIMAP_FADE, MINIMAP_HIDE_DELAY, MINIMAP_WIDTH, SpaceView,
};

// ---------------------------------------------------------------------------
// The scrollbar-like minimap interaction — pure mapping math + drag state.
//
// The minimap column is a linear 1:1 image of the scrollable document: with
// `scale = viewport_h / total_h`, a document position `doc` maps to minimap-local
// y `doc * scale`, and inversely a minimap-local y `m` maps to document position
// `m / scale`. A document element with doc-space top `doc_top` paints at screen y
// `doc_top + scroll_y` (scroll sign convention: `scroll_y <= 0`, scrolling down
// makes it more negative). The "handle" (the dark on-screen indicator) occupies
// minimap y-range `[t0, t1]` with `t0 = (-scroll_y) * scale`, `t1 = t0 +
// viewport_h * scale`.
// ---------------------------------------------------------------------------

/// An in-flight scrollbar-style minimap drag. Snapshotted at drag start: the
/// selected branch is locked for the drag's duration, so `scale` and the scroll
/// `floor` stay valid (the page height can't change without a branch switch,
/// which a drag never does).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MinimapDrag {
    /// The grabbed offset within the handle, in minimap-local y (px). On a handle
    /// press this is `m - t0` (no jump); on a track press it's half the handle
    /// height (the handle jumps to center on the cursor).
    pub(crate) grab: f32,
    /// `viewport_h / total_h` at drag start (stable while the branch is fixed).
    pub(crate) scale: f32,
    /// The most-negative valid `page_scroll` y for the locked branch (the floor).
    pub(crate) floor: f32,
}

/// The document position under a press at minimap-local y `m`: `m / scale`.
pub(crate) fn doc_at_minimap_y(m: f32, scale: f32) -> f32 {
    if scale <= 0.0 { 0.0 } else { m / scale }
}

/// The vertical fraction (0..1) within a cell whose document range is
/// `[doc_top, doc_top + height]`, for a press at minimap-local y `m`.
pub(crate) fn cell_fraction(m: f32, scale: f32, doc_top: f32, height: f32) -> f32 {
    if height <= 0.0 {
        return 0.0;
    }
    ((doc_at_minimap_y(m, scale) - doc_top) / height).clamp(0.0, 1.0)
}

/// The `page_scroll` y that lands document point `doc_click` directly under a
/// cursor at minimap-local y `m` (direct manipulation), clamped to `[floor, 0]`.
pub(crate) fn scroll_for_press(m: f32, doc_click: f32, floor: f32) -> f32 {
    (m - doc_click).clamp(floor, 0.0)
}

/// The handle's minimap y-range `[t0, t1]` for a given `scroll_y`.
pub(crate) fn handle_range(scroll_y: f32, viewport_h: f32, scale: f32) -> (f32, f32) {
    let t0 = -scroll_y * scale;
    (t0, t0 + viewport_h * scale)
}

/// The grab offset for a press at minimap-local y `m`: pressing on the handle
/// grabs at the press offset (`m - t0`, no jump); pressing the track outside it
/// jumps the handle center to the cursor (`viewport_h * scale / 2`).
pub(crate) fn drag_grab(m: f32, scroll_y: f32, viewport_h: f32, scale: f32) -> f32 {
    let (t0, t1) = handle_range(scroll_y, viewport_h, scale);
    if m >= t0 && m <= t1 {
        m - t0
    } else {
        viewport_h * scale / 2.0
    }
}

/// The `page_scroll` y during a drag, for a cursor at minimap-local y `m`:
/// `t0' = m - grab`, `scroll_y = -t0' / scale`, clamped to `[floor, 0]`.
pub(crate) fn drag_scroll(m: f32, grab: f32, scale: f32, floor: f32) -> f32 {
    if scale <= 0.0 {
        return 0.0;
    }
    let t0 = m - grab;
    (-t0 / scale).clamp(floor, 0.0)
}

impl SpaceView {
    /// Record a scroll event for the minimap's show/hide. `moved` is whether the
    /// position actually changed. Called for every scroll event, whichever
    /// container handled it.
    pub(crate) fn note_scroll_activity(
        &mut self,
        phase: gpui::TouchPhase,
        moved: bool,
        cx: &mut Context<Self>,
    ) {
        match phase {
            gpui::TouchPhase::Started => self.minimap_gesturing = true,
            gpui::TouchPhase::Ended => self.minimap_gesturing = false,
            gpui::TouchPhase::Moved => {}
        }
        if moved {
            self.minimap_visible = true;
        }
        if moved || matches!(phase, gpui::TouchPhase::Ended) {
            self.arm_minimap_hide(cx);
            cx.notify();
        }
    }

    /// (Re)start the linger timer: after [`MINIMAP_HIDE_DELAY`] of quiet (no
    /// gesture, cursor off the bar), fade the minimap out.
    pub(crate) fn arm_minimap_hide(&mut self, cx: &mut Context<Self>) {
        self.minimap_hide_task = Some(cx.spawn(async move |this: WeakEntity<Self>, cx| {
            cx.background_executor().timer(MINIMAP_HIDE_DELAY).await;
            this.update(cx, |this, cx| {
                if this.minimap_gesturing || this.minimap_hovered {
                    this.arm_minimap_hide(cx);
                } else if this.minimap_visible {
                    this.minimap_visible = false;
                    this.minimap_fade_gen = this.minimap_fade_gen.wrapping_add(1);
                    cx.notify();
                }
            })
            .ok();
        }));
    }

    /// A cheap hash of the layout inputs the minimap reads, so `render`
    /// schedules exactly one catch-up frame when they change.
    pub(crate) fn minimap_signature(
        &self,
        page_width: gpui::Pixels,
        viewport_h: gpui::Pixels,
    ) -> f32 {
        let streaming = false; // the structure (not the live partial) drives the map
        let tree = self.effective_tree(page_width, streaming);
        let selected = self.selected_total_height(&tree, page_width, viewport_h);
        // Use the *clamped* scroll position (see `clamped_scroll_y`) so transient
        // momentum overshoot past the ends doesn't schedule a catch-up frame
        // that would re-render the minimap mid-overshoot.
        let scroll_y = self.clamped_scroll_y();
        viewport_h.as_f32()
            + self.composer_content_h.borrow().as_f32() * 5.0
            + scroll_y * 19.0
            + selected * 3.0
            + self.posts.len() as f32
    }

    /// A screen-reader / table-of-contents label for a map column: the post's
    /// byline plus a short snippet ("You: I keep circling back to…"), or "Draft"
    /// / "Eidola, responding" for the overlays.
    fn node_label(&self, node: &TreeNode) -> String {
        match node.src {
            NodeSrc::Streaming => "Eidola, responding".to_string(),
            NodeSrc::Draft => "Draft".to_string(),
            NodeSrc::Msg(i) => {
                let Some(p) = self.posts.get(i) else {
                    return "Post".to_string();
                };
                let snippet: String = p
                    .content
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .chars()
                    .take(56)
                    .collect();
                if snippet.is_empty() {
                    p.byline.to_string()
                } else {
                    format!("{}: {}", p.byline, snippet)
                }
            }
        }
    }

    /// The topology minimap (see the module docs). Reads the selected path's
    /// cached heights; positions derive from the live page scroll offset.
    pub(crate) fn render_minimap(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        viewport_h: gpui::Pixels,
        window: &Window,
        cx: &Context<Self>,
    ) -> AnyElement {
        let fg = cx.theme().scrollbar_thumb;
        let light = fg.opacity(0.18);
        let medium = cx.theme().scrollbar_thumb.opacity(0.45);
        let dark = cx.theme().scrollbar_thumb_hover.opacity(0.78);
        // Drafts read in the `info` hue (matching the branch dots), at the same
        // on-screen / off-screen / sibling opacity ramp as the scroll colors.
        let info = cx.theme().info;
        // Cells are click-to-navigate only while the map is up (it's a transient
        // overlay; a faded-out 36px strip must not steal clicks).
        let interactive = self.minimap_visible;

        // Keep the strip's colored cells out of the window's rounded corner
        // arcs (the chrome frame cannot clip to the curve — gpui masks are
        // rectangular): inset both ends by the corner radius and scale the
        // map into the reduced run. Zero inset when no corner is rounded.
        let clearance = crate::chrome::corner_clearance(window);
        let mut container = div()
            .id("space-minimap")
            .probe("space/minimap", gpui::Role::Group, "Conversation map")
            .absolute()
            .top_0()
            .bottom_0()
            .right_0()
            .pt(clearance)
            .pb(clearance)
            .w(MINIMAP_WIDTH);

        let levels = self.selected_levels(roots, page_width);
        // The same top headroom the scrollable document uses (zero for an empty
        // notebook), so the map's scale matches the real scroll range exactly —
        // no phantom reserve band above a sole composer.
        let reserve = self.doc_reserve();
        let selected_h = self.selected_total_height(roots, page_width, viewport_h);
        // A floating, off-branch draft pads the page bottom (item 4); fold it
        // into the denominator + a trailing spacer so the scroll indicator maps
        // 1:1 to the real scrollable height on every branch.
        let pad = self.floating_pad(roots, page_width, viewport_h, false);
        let total_h = reserve + selected_h + pad;
        // The indicator reflects the *clamped* scroll position (the page
        // hard-stops at the ends; the raw offset transiently overshoots during
        // momentum), so the visible-window never slides past the end and
        // flickers back — see `clamped_scroll_y`.
        let scroll_y = self.clamped_scroll_y();

        let strip_h = viewport_h - clearance * 2.;
        if total_h > 0.0 && strip_h > px(0.) && !levels.is_empty() {
            let scale = strip_h.as_f32() / total_h;
            let mut col = v_flex().w_full();

            // The reserve scrolls off like content: dark at the very top. Absent
            // entirely for an empty notebook (no headroom → no band).
            if reserve > 0.0 {
                let reserve_top = scroll_y;
                col = col.child(selected_column(
                    Some((reserve_top, reserve)),
                    viewport_h.as_f32(),
                    px(reserve * scale),
                    dark,
                    medium,
                ));
            }

            // Accumulate the document top of each level's selected node.
            let mut doc_y = reserve;
            for (level, (sibs, active)) in levels.iter().enumerate() {
                if level > 0 {
                    col = col.child(div().w_full().h(px(BAND_HEIGHT.as_f32() * scale)));
                    doc_y += BAND_HEIGHT.as_f32();
                }
                let node = sibs[*active];
                let h = self.node_height(node, page_width, viewport_h);
                let row_h = px(h * scale);
                let screen_top = doc_y + scroll_y;

                let mut row = h_flex().w_full().h(row_h).gap(MINIMAP_COL_GAP);
                for (i, sib) in sibs.iter().enumerate() {
                    let is_active = i == *active;
                    // Drafts use `info`; everything else the scroll colors.
                    let (on, off, flat) = if matches!(sib.src, NodeSrc::Draft) {
                        (info.opacity(0.78), info.opacity(0.45), info.opacity(0.22))
                    } else {
                        (dark, medium, light)
                    };
                    let cell = if is_active {
                        selected_column(Some((screen_top, h)), viewport_h.as_f32(), row_h, on, off)
                    } else {
                        div().w_full().h_full().bg(flat)
                    };

                    // Each column is a selectable entry in the map's "table of
                    // contents" — a labelled button that, on mousedown, either
                    // navigates to a different branch (positioning by the click's
                    // vertical offset within the item) or, on the selected branch,
                    // begins a scrollbar-style drag (see `minimap_press`).
                    let sib_id = sib.id.clone();
                    let mut wrap = div()
                        .id(SharedString::from(format!("space-mm-{level}-{i}")))
                        .probe(
                            SharedString::from(format!("space/minimap/cell/{level}/{i}")),
                            gpui::Role::Button,
                            self.node_label(sib),
                        )
                        .aria_selected(is_active)
                        .flex_1()
                        .h_full()
                        .child(cell);
                    if interactive {
                        // Snapshot this cell's rendered geometry (its level's
                        // document top and the row height, both shared across the
                        // row's columns) so the handler can map the press.
                        let cell_doc_top = doc_y;
                        let cell_h = h;
                        wrap = wrap.cursor_pointer().on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, ev: &MouseDownEvent, window, cx| {
                                this.minimap_press(
                                    sib_id.clone(),
                                    is_active,
                                    cell_doc_top,
                                    cell_h,
                                    scale,
                                    ev.position.y.as_f32(),
                                    window,
                                    cx,
                                );
                            }),
                        );
                    }
                    row = row.child(wrap);
                }
                col = col.child(row);
                doc_y += h;
            }
            // Trailing scroll room for a floating off-branch draft.
            if pad > 0.0 {
                col = col.child(div().w_full().h(px(pad * scale)));
            }
            container = container.child(col);
            // An invisible, hitbox-free overlay that (prepaint) records the
            // container's absolute bounds — so a mousedown/drag can convert a
            // window-space y into a minimap-local y — and (paint) registers the
            // window-global move/up listeners a scrollbar-style drag needs to keep
            // tracking after the cursor leaves the 36px strip. `on_mouse_event`
            // listeners are cleared each frame, so they are re-registered every
            // frame (mirroring `gpui_component::Scrollbar`); they no-op unless a
            // drag is actually in flight, so registering unconditionally is cheap
            // and avoids a first-move gap.
            let bounds_cell = self.minimap_bounds.clone();
            let strip_clearance = clearance;
            let weak = cx.entity().downgrade();
            container = container.child(
                gpui::canvas(
                    move |mut bounds, _, _| {
                        // The interactive strip starts `clearance` below the
                        // container's padding-box top (the `.pt(clearance)`
                        // corner inset on Linux CSD). Record the *strip* top, not
                        // the container top, so `minimap_local_y` yields a
                        // strip-relative y — the origin the scrollbar mapping
                        // math (`handle_range`, `drag_grab`, …) is measured from
                        // (0 = first cell). Zero off Linux CSD, so macOS/tests
                        // are unchanged.
                        bounds.origin.y += strip_clearance;
                        bounds_cell.set(Some(bounds));
                    },
                    move |_bounds, _, window, _cx| {
                        let move_weak = weak.clone();
                        window.on_mouse_event(move |ev: &MouseMoveEvent, _phase, _window, cx| {
                            let Some(this) = move_weak.upgrade() else {
                                return;
                            };
                            this.update(cx, |this, cx| {
                                if this.minimap_drag.is_none() {
                                    return;
                                }
                                if !ev.dragging() {
                                    // Button released without a delivered up event.
                                    this.minimap_drag_end(cx);
                                    return;
                                }
                                cx.stop_propagation();
                                this.minimap_drag_move(ev.position.y.as_f32(), cx);
                            });
                        });
                        let up_weak = weak.clone();
                        window.on_mouse_event(move |_ev: &MouseUpEvent, _phase, _window, cx| {
                            let Some(this) = up_weak.upgrade() else {
                                return;
                            };
                            this.update(cx, |this, cx| this.minimap_drag_end(cx));
                        });
                    },
                )
                // Pin to the container's top-left. Without an explicit inset an
                // `absolute` element takes its *static* position — here, after
                // `col` (which fills the container), so its recorded origin.y
                // would be the container's BOTTOM (≈ window height), not its top.
                // A mousedown's window-y minus that bogus origin is negative,
                // which reads as below every handle range → every press became a
                // track press that clamped the scroll to the very top.
                .absolute()
                .top_0()
                .left_0()
                .size_full(),
            );
        }

        if self.minimap_visible {
            container
                .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                    this.minimap_hovered = *hovered;
                    this.arm_minimap_hide(cx);
                    cx.notify();
                }))
                .into_any_element()
        } else if self.minimap_fade_gen == 0 {
            container.opacity(0.0).into_any_element()
        } else {
            container
                .with_animation(
                    ("space-minimap-fade", self.minimap_fade_gen),
                    Animation::new(MINIMAP_FADE),
                    |el, delta| el.opacity(1.0 - delta),
                )
                .into_any_element()
        }
    }

    /// Minimap-local y of a window-space y, using the strip top recorded last
    /// frame (the container's padding-box top plus the CSD corner clearance, so
    /// 0 is the first cell — the origin the scrollbar mapping math expects).
    fn minimap_local_y(&self, window_y: f32) -> f32 {
        let top = self
            .minimap_bounds
            .get()
            .map(|b| b.origin.y.as_f32())
            .unwrap_or(0.0);
        window_y - top
    }

    /// Handle a mousedown on a minimap cell. A press on a **different** branch
    /// (not the selected path) switches to it and positions by the click's
    /// vertical offset within the item (direct manipulation), completing the
    /// interaction — no drag follows, because switching branches changes the page
    /// height. A press on the **selected** branch (or the on-screen handle) begins
    /// a scrollbar-style drag locked to that branch (see [`MinimapDrag`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn minimap_press(
        &mut self,
        node_id: SharedString,
        is_active: bool,
        cell_doc_top: f32,
        cell_height: f32,
        scale: f32,
        window_y: f32,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        let m = self.minimap_local_y(window_y);
        // Content box, not the raw surface — on Linux CSD the surface includes
        // the shadow padding, which must not enter the scroll/branch geometry.
        let viewport = crate::chrome::content_size(window);
        let page_width = viewport.width;
        let window_h = viewport.height;

        if !is_active {
            // Different branch: navigate + complete. Position so the pressed
            // document point lands under the cursor on the new page.
            let fraction = cell_fraction(m, scale, cell_doc_top, cell_height);
            let streaming = self.space.read(cx).is_streaming();
            let tree = self.effective_tree(page_width, streaming);
            if super::model::node_ref(&tree, &node_id).is_none() {
                return;
            }
            // Order matters: switch the branch first, then read the new layout —
            // the doc positions sum heights along the *selected* path.
            self.select_path_to(&tree, &node_id, page_width);
            if let Some(new_top) = self.selected_path_doc_top(&tree, &node_id, page_width, window_h)
            {
                let new_h = super::model::node_ref(&tree, &node_id)
                    .map(|n| self.node_height(n, page_width, window_h))
                    .unwrap_or(cell_height);
                let doc_click = new_top + fraction * new_h;
                // Clamp against the *new* branch's scroll range (its height
                // changed with the switch).
                let total = self.doc_reserve()
                    + self.selected_total_height(&tree, page_width, window_h)
                    + self.floating_pad(&tree, page_width, window_h, streaming);
                let floor = (window_h.as_f32() - total).min(0.0);
                let y = scroll_for_press(m, doc_click, floor);
                let off = self.page_scroll.offset();
                self.page_scroll.set_offset(point(off.x, px(y)));
            }
            self.minimap_visible = true;
            self.arm_minimap_hide(cx);
            cx.notify();
            return;
        }

        // Same branch (or the handle): start a drag. Grab at the press offset on
        // the handle (no jump), or jump the handle center to the cursor on the
        // track — then apply once so the initial press already positions.
        let scroll_y = self.clamped_scroll_y();
        let grab = drag_grab(m, scroll_y, window_h.as_f32(), scale);
        let floor = self.scroll_min_y.get();
        let y = drag_scroll(m, grab, scale, floor);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
        self.minimap_drag = Some(MinimapDrag { grab, scale, floor });
        self.minimap_visible = true;
        self.arm_minimap_hide(cx);
        cx.notify();
    }

    /// One frame of an active minimap drag: move `page_scroll` so the grabbed
    /// handle point tracks the cursor at minimap-local y from `window_y`.
    pub(crate) fn minimap_drag_move(&mut self, window_y: f32, cx: &mut Context<Self>) {
        let Some(drag) = self.minimap_drag else {
            return;
        };
        let m = self.minimap_local_y(window_y);
        let y = drag_scroll(m, drag.grab, drag.scale, drag.floor);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
        self.minimap_visible = true;
        self.arm_minimap_hide(cx);
        cx.notify();
    }

    /// End an active minimap drag (mouse-up or button-release); a no-op if none.
    pub(crate) fn minimap_drag_end(&mut self, cx: &mut Context<Self>) {
        if self.minimap_drag.take().is_some() {
            self.arm_minimap_hide(cx);
            cx.notify();
        }
    }
}

/// One selected-branch minimap column: a full-height column split into medium
/// (scrolled-off) and dark (on-screen) spans, from the block's on-screen
/// `(top, height)` clipped against the visible region `[0, vis_bot]`.
fn selected_column(
    block: Option<(f32, f32)>,
    vis_bot: f32,
    col_h: gpui::Pixels,
    dark: Hsla,
    medium: Hsla,
) -> gpui::Div {
    let Some((top, height)) = block else {
        return div().w_full().h(col_h).bg(medium);
    };
    let height = height.max(1.0);
    let vt = top.max(0.0);
    let vb = (top + height).min(vis_bot);
    if vb <= vt {
        return div().w_full().h(col_h).bg(medium);
    }
    let ch = col_h.as_f32();
    let above = ((vt - top) / height).clamp(0.0, 1.0) * ch;
    let visible = ((vb - vt) / height).clamp(0.0, 1.0) * ch;
    let below = (ch - above - visible).max(0.0);
    v_flex()
        .w_full()
        .h(col_h)
        .child(div().w_full().h(px(above)).bg(medium))
        .child(div().w_full().h(px(visible)).bg(dark))
        .child(div().w_full().h(px(below)).bg(medium))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A scene: viewport 500 tall over a 2000-tall document → scale 0.25. The
    // minimap column is that document scaled by 0.25 (a 500px-tall strip).
    const VIEWPORT_H: f32 = 500.0;
    const TOTAL_H: f32 = 2000.0;
    const SCALE: f32 = VIEWPORT_H / TOTAL_H; // 0.25
    // Scroll floor: window_h - total = -1500 (most-negative valid page y).
    const FLOOR: f32 = VIEWPORT_H - TOTAL_H; // -1500

    #[test]
    fn minimap_y_maps_linearly_to_document() {
        // Minimap-local y m ↔ document m / scale.
        assert!((doc_at_minimap_y(0.0, SCALE) - 0.0).abs() < 1e-4);
        assert!((doc_at_minimap_y(125.0, SCALE) - 500.0).abs() < 1e-4);
        // The whole 500px strip maps onto the whole 2000px document.
        assert!((doc_at_minimap_y(VIEWPORT_H, SCALE) - TOTAL_H).abs() < 1e-4);
        // Degenerate scale is safe.
        assert_eq!(doc_at_minimap_y(100.0, 0.0), 0.0);
    }

    #[test]
    fn cell_fraction_is_relative_position_within_the_item() {
        // A cell spanning document [800, 1000] (height 200) → minimap [200, 250].
        let (doc_top, height) = (800.0, 200.0);
        // Press at the cell's minimap top → fraction 0.
        assert!((cell_fraction(200.0, SCALE, doc_top, height) - 0.0).abs() < 1e-4);
        // Press at the cell's minimap middle (225) → fraction 0.5.
        assert!((cell_fraction(225.0, SCALE, doc_top, height) - 0.5).abs() < 1e-4);
        // Press past the bottom clamps to 1.0; before the top clamps to 0.0.
        assert!((cell_fraction(400.0, SCALE, doc_top, height) - 1.0).abs() < 1e-4);
        assert!((cell_fraction(0.0, SCALE, doc_top, height) - 0.0).abs() < 1e-4);
        // Zero-height cell is safe.
        assert_eq!(cell_fraction(225.0, SCALE, doc_top, 0.0), 0.0);
    }

    #[test]
    fn scroll_for_press_lands_the_pressed_point_under_the_cursor() {
        // Pressing minimap y=225 (document 900) on an item; we want document 900
        // to paint at screen y == m (225). scroll_y = m - doc_click = 225 - 900.
        let y = scroll_for_press(225.0, 900.0, FLOOR);
        assert!((y - (-675.0)).abs() < 1e-4);
        // Verify the invariant: doc_click + scroll_y == m (cursor position).
        assert!((900.0 + y - 225.0).abs() < 1e-4);
        // Clamps to the floor when the requested scroll would exceed it.
        assert_eq!(scroll_for_press(0.0, 5000.0, FLOOR), FLOOR);
        // Never scrolls above the top.
        assert_eq!(scroll_for_press(400.0, 100.0, FLOOR), 0.0);
    }

    #[test]
    fn handle_range_tracks_the_visible_viewport() {
        // At rest (scroll_y 0) the handle sits at the top, height viewport*scale.
        let (t0, t1) = handle_range(0.0, VIEWPORT_H, SCALE);
        assert!((t0 - 0.0).abs() < 1e-4);
        assert!((t1 - 125.0).abs() < 1e-4);
        // Scrolled to the floor (-1500) the handle sits at the bottom of the strip.
        let (t0, t1) = handle_range(FLOOR, VIEWPORT_H, SCALE);
        assert!((t0 - 375.0).abs() < 1e-4); // 1500 * 0.25
        assert!((t1 - 500.0).abs() < 1e-4);
    }

    #[test]
    fn drag_grab_distinguishes_handle_from_track() {
        // At rest the handle is [0, 125]. Pressing inside it (y=40) grabs at the
        // press offset (no jump).
        assert!((drag_grab(40.0, 0.0, VIEWPORT_H, SCALE) - 40.0).abs() < 1e-4);
        // Pressing the track outside the handle (y=300) jumps the handle center
        // to the cursor → grab is half the handle height (125/2).
        assert!((drag_grab(300.0, 0.0, VIEWPORT_H, SCALE) - 62.5).abs() < 1e-4);
    }

    #[test]
    fn drag_scroll_grabbed_point_tracks_cursor_without_jump() {
        // Grab the handle at rest (scroll 0) at its top (grab 0): a no-op press.
        assert!((drag_scroll(0.0, 0.0, SCALE, FLOOR) - 0.0).abs() < 1e-4);
        // Now drag the cursor down to y=125: the handle top follows to 125, so
        // scroll_y = -125/scale = -500 (one viewport down).
        assert!((drag_scroll(125.0, 0.0, SCALE, FLOOR) - (-500.0)).abs() < 1e-4);
        // A handle grab preserves the offset: pressing at y=40 with grab=40
        // (t0 stays 0) yields no movement.
        assert!((drag_scroll(40.0, 40.0, SCALE, FLOOR) - 0.0).abs() < 1e-4);
        // Clamps to the floor at the bottom.
        assert_eq!(drag_scroll(10_000.0, 0.0, SCALE, FLOOR), FLOOR);
        // Degenerate scale is safe.
        assert_eq!(drag_scroll(100.0, 0.0, 0.0, FLOOR), 0.0);
    }

    #[test]
    fn track_press_then_drag_is_continuous() {
        // Press the track at y=300 (below the resting handle). The grab centers
        // the handle on the cursor, and applying the drag at the same y lands the
        // handle centered there — scroll = -(300 - 62.5)/0.25 = -950.
        let grab = drag_grab(300.0, 0.0, VIEWPORT_H, SCALE);
        let y0 = drag_scroll(300.0, grab, SCALE, FLOOR);
        assert!((y0 - (-950.0)).abs() < 1e-4);
        // Dragging further to y=310 moves smoothly by the same 1/scale factor.
        let y1 = drag_scroll(310.0, grab, SCALE, FLOOR);
        assert!((y1 - (y0 - 40.0)).abs() < 1e-4); // 10px * (1/0.25) = 40px
    }
}
