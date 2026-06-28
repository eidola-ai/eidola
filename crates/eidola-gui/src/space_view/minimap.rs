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
    ParentElement, StatefulInteractiveElement, Styled, WeakEntity, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use super::model::TreeNode;
use super::{
    BAND_HEIGHT, MINIMAP_COL_GAP, MINIMAP_FADE, MINIMAP_HIDE_DELAY, MINIMAP_WIDTH, SpaceView,
    TITLE_BAR_RESERVE,
};

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
        let (tree, _) = self.effective_tree(page_width, streaming);
        let selected = match tree.first() {
            Some(root) => self.selected_subtree_height(root, page_width, viewport_h),
            None => 0.0,
        };
        viewport_h.as_f32()
            + self.composer_content_h.borrow().as_f32() * 5.0
            + self.page_scroll.offset().y.as_f32() * 19.0
            + selected * 3.0
            + self.posts.len() as f32
    }

    /// The topology minimap (see the module docs). Reads the selected path's
    /// cached heights; positions derive from the live page scroll offset.
    pub(crate) fn render_minimap(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        viewport_h: gpui::Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        let fg = cx.theme().scrollbar_thumb;
        let light = fg.opacity(0.18);
        let medium = cx.theme().scrollbar_thumb.opacity(0.45);
        let dark = cx.theme().scrollbar_thumb_hover.opacity(0.78);

        let mut container = div()
            .id("space-minimap")
            .absolute()
            .top_0()
            .bottom_0()
            .right_0()
            .w(MINIMAP_WIDTH);

        let levels = self.selected_levels(roots, page_width);
        let reserve = TITLE_BAR_RESERVE.as_f32();
        let selected_h = match roots.first() {
            Some(root) => self.selected_subtree_height(root, page_width, viewport_h),
            None => 0.0,
        };
        let total_h = reserve + selected_h;
        let scroll_y = self.page_scroll.offset().y.as_f32();

        if total_h > 0.0 && viewport_h > px(0.) && !levels.is_empty() {
            let scale = viewport_h.as_f32() / total_h;
            let mut col = v_flex().w_full();

            // The reserve scrolls off like content: dark at the very top.
            let reserve_top = scroll_y;
            col = col.child(selected_column(
                Some((reserve_top, reserve)),
                viewport_h.as_f32(),
                px(reserve * scale),
                dark,
                medium,
            ));

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
                for (i, _sib) in sibs.iter().enumerate() {
                    let cell = if i == *active {
                        selected_column(
                            Some((screen_top, h)),
                            viewport_h.as_f32(),
                            row_h,
                            dark,
                            medium,
                        )
                    } else {
                        div().w_full().h_full().bg(light)
                    };
                    row = row.child(div().flex_1().h_full().child(cell));
                }
                col = col.child(row);
                doc_y += h;
            }
            container = container.child(col);
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
