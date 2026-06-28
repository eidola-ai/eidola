//! The cached layout model — the performance core.
//!
//! The mockup re-measured every post every frame (a `canvas` overlay per post)
//! and rendered the whole tree, so per-frame cost was O(all posts) of text
//! shaping. Here, post heights are **cached** keyed by node id (for the current
//! page width); only posts intersecting the viewport render the real
//! `MarkdownEditor` (and carry a measuring `canvas` that refreshes the cache),
//! while off-screen posts render as sized placeholders from the cached (or
//! estimated) height. So per-frame shaping is bounded to *visible* posts, and
//! the minimap reads cached heights instead of sweeping every post each frame.
//!
//! This file holds the [`Layout`] cache itself; the selection- and
//! scroll-dependent height/position computations live on `SpaceView` (they read
//! its scroll handles and window size) in `impl` blocks alongside the view.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use gpui::SharedString;

/// Per-post measured-height cache, valid for one page width. A width change
/// invalidates every entry (markdown height is a function of wrap width), so
/// posts re-measure as they re-enter the viewport.
///
/// Interior-mutable and cheaply `Clone` (refcounted) so a visible post's
/// paint-time measuring `canvas` can capture a handle and `record` into the
/// same cache the view reads. A post records its height the frame it renders
/// real; off-screen posts read the last recorded value (or an estimate), which
/// is how the document total, scroll range, and minimap scale stay right while
/// only visible posts actually shape.
#[derive(Clone, Default)]
pub struct Layout {
    /// The page width the cached heights were measured at.
    width: Rc<Cell<f32>>,
    /// Measured block height (byline + body, the full post element) by node id.
    heights: Rc<RefCell<HashMap<SharedString, f32>>>,
}

impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point the cache at `width`, clearing measurements if it changed. Call
    /// once per frame before reading heights.
    pub fn ensure_width(&self, width: f32) {
        if (self.width.get() - width).abs() > 0.5 {
            self.width.set(width);
            self.heights.borrow_mut().clear();
        }
    }

    /// Record a freshly-measured height for `id` (called from an on-screen
    /// post's measuring canvas). Returns whether the value changed materially
    /// (so the caller can schedule a catch-up frame for layout to settle).
    pub fn record(&self, id: &SharedString, height: f32) -> bool {
        let mut heights = self.heights.borrow_mut();
        match heights.get(id) {
            Some(prev) if (prev - height).abs() <= 0.5 => false,
            _ => {
                heights.insert(id.clone(), height);
                true
            }
        }
    }

    /// The cached height for `id`, if measured at the current width.
    pub fn measured(&self, id: &str) -> Option<f32> {
        self.heights.borrow().get(id).copied()
    }

    /// Drop cache entries whose ids are no longer live (transcript reshaped).
    pub fn retain(&self, live: &dyn Fn(&str) -> bool) {
        self.heights.borrow_mut().retain(|id, _| live(id));
    }
}

/// A cheap height estimate for a post that hasn't been measured yet, so the
/// document total (and thus the scroll range and minimap scale) is approximately
/// right before the post first scrolls into view and measures for real. Mirrors
/// `MarkdownEditor`'s book typography: characters per line from the body width,
/// lines from the content length, times the line height, plus the post's
/// vertical padding. Deliberately rough — it is replaced by the real measured
/// height the first frame the post is visible.
pub fn estimate_post_height(
    content: &str,
    body_width: f32,
    font_size: f32,
    line_height: f32,
    pad_y: f32,
) -> f32 {
    let char_w = (font_size * 0.5).max(1.0);
    let cols = (body_width / char_w).max(1.0);
    // Sum wrapped lines across hard breaks so multi-paragraph posts estimate
    // taller than a single long run of the same length.
    let mut lines = 0.0_f32;
    for para in content.split('\n') {
        let chars = para.chars().count().max(1) as f32;
        lines += (chars / cols).ceil().max(1.0);
    }
    // A blank post (the empty composer) still reserves one line.
    lines = lines.max(1.0);
    lines * (font_size * line_height) + pad_y
}

// ---------------------------------------------------------------------------
// Selection- and scroll-dependent layout, on `SpaceView` (reads its scroll
// handles, the height cache, and the window size).
// ---------------------------------------------------------------------------

use super::model::{NodeSrc, TreeNode};
use super::{
    BAND_HEIGHT, BODY_MAX_WIDTH, COMPOSER_MAX_FRACTION, GUTTER_GAP, GUTTER_WIDTH, POST_PAD_Y,
    PROSE_FONT_SIZE, PROSE_LINE_HEIGHT, SpaceView, TITLE_BAR_RESERVE,
};
use gpui::Pixels;

/// The reading-column width for a post body at a given page width — the mockup's
/// gutter-aware centered measure, capped at [`BODY_MAX_WIDTH`].
pub(crate) fn body_width(page_width: Pixels) -> f32 {
    (page_width - GUTTER_WIDTH * 1.5 - GUTTER_GAP * 2.)
        .min(BODY_MAX_WIDTH)
        .max(gpui::px(240.))
        .as_f32()
}

impl SpaceView {
    /// Which child page a node's scroller is resting on, from its scroll offset.
    /// Pages are `page_width + BAND_HEIGHT` apart, so the nearest is
    /// `round(scrolled / stride)`, clamped to the branch count.
    pub(crate) fn active_child_index(
        &self,
        node_id: &str,
        page_width: Pixels,
        count: usize,
    ) -> usize {
        if count <= 1 || page_width <= gpui::px(0.) {
            return 0;
        }
        let Some(handle) = self.scrolls.get(node_id) else {
            return 0;
        };
        let stride = (page_width + BAND_HEIGHT).as_f32();
        let scrolled = (-handle.offset().x).as_f32();
        ((scrolled / stride).round() as i64).clamp(0, count as i64 - 1) as usize
    }

    /// Half the document's inter-post spacing — the composer bar's fixed top
    /// chrome.
    pub(crate) fn composer_chrome() -> f32 {
        POST_PAD_Y.as_f32() / 2.0
    }

    /// A branch's trailing runway: at least a window tall (so the docked
    /// composer can stand alone), or as tall as the composer's content if more.
    pub(crate) fn runway_height(&self, window_h: Pixels) -> f32 {
        let content = self.composer_content_h.borrow().as_f32();
        window_h.as_f32().max(Self::composer_chrome() + content)
    }

    /// The in-flow slot height the draft reserves — the runway (the composer
    /// floats/docks into it).
    pub(crate) fn draft_slot_height(
        &self,
        _node: &TreeNode,
        _page_width: Pixels,
        window_h: Pixels,
    ) -> f32 {
        self.runway_height(window_h)
    }

    /// The node's own in-flow block height: a post's measured (or estimated)
    /// height, the streaming reply's measured height, the active draft's runway
    /// slot, or an inactive draft's measured inline height.
    pub(crate) fn node_height(&self, node: &TreeNode, page_width: Pixels, window_h: Pixels) -> f32 {
        match node.src {
            NodeSrc::Draft if self.active_draft.as_deref() == Some(&node.id) => {
                self.draft_slot_height(node, page_width, window_h)
            }
            // An inactive draft renders inline as an editable post, but reserves
            // at least a full window (it's always the end of its branch), so
            // there's perfect continuity with the active draft's runway slot —
            // activating/deactivating it never resizes the layout. `min_h` on
            // the inline frame makes the measured height honour the same floor.
            NodeSrc::Draft => self
                .layout
                .measured(&node.id)
                .unwrap_or(0.0)
                .max(window_h.as_f32()),
            NodeSrc::Streaming => self
                .layout
                .measured(&node.id)
                .unwrap_or_else(|| window_h.as_f32() * 0.3),
            NodeSrc::Msg(i) => self.layout.measured(&node.id).unwrap_or_else(|| {
                let content = self.posts.get(i).map(|p| p.content.as_ref()).unwrap_or("");
                estimate_post_height(
                    content,
                    body_width(page_width),
                    PROSE_FONT_SIZE.as_f32(),
                    PROSE_LINE_HEIGHT,
                    POST_PAD_Y.as_f32() * 2.0,
                )
            }),
        }
    }

    /// The id of the selected leaf — follow the active child at each level
    /// (starting from the active root when there's more than one) until a node
    /// with no replies.
    pub(crate) fn selected_leaf_id(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
    ) -> Option<gpui::SharedString> {
        let mut node = self.active_root(roots, page_width)?;
        while !node.children.is_empty() {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            node = &node.children[active];
        }
        Some(node.id.clone())
    }

    /// The selected root (the active page of the implicit top-level scroller).
    fn active_root<'a>(
        &self,
        roots: &'a [TreeNode],
        page_width: gpui::Pixels,
    ) -> Option<&'a TreeNode> {
        if roots.is_empty() {
            return None;
        }
        let idx = self.active_child_index(super::model::ROOT_SCROLLER_ID, page_width, roots.len());
        roots.get(idx).or_else(|| roots.first())
    }

    /// Rendered height of the *currently selected* path from `node` down to its
    /// leaf — the dynamic-height core: each branch scroller is sized to its
    /// selected child, so the document height tracks the active branch.
    pub(crate) fn selected_subtree_height(
        &self,
        node: &TreeNode,
        page_width: Pixels,
        window_h: Pixels,
    ) -> f32 {
        let h = self.node_height(node, page_width, window_h);
        if node.children.is_empty() {
            // A draft/streaming leaf is the editing/streaming surface (its slot
            // already reserves the runway); a normal leaf ends with a trailing
            // separator band.
            if matches!(node.src, NodeSrc::Draft | NodeSrc::Streaming) {
                return h;
            }
            return h + BAND_HEIGHT.as_f32();
        }
        let active = self.active_child_index(&node.id, page_width, node.children.len());
        h + BAND_HEIGHT.as_f32()
            + self.selected_subtree_height(&node.children[active], page_width, window_h)
    }

    /// The selected path as levels (for the minimap): level 0 is the roots with
    /// the active root index; each subsequent level is the active node's
    /// children with its active index.
    pub(crate) fn selected_levels<'a>(
        &self,
        roots: &'a [TreeNode],
        page_width: Pixels,
    ) -> Vec<(Vec<&'a TreeNode>, usize)> {
        let mut levels = Vec::new();
        if roots.is_empty() {
            return levels;
        }
        let root_active =
            self.active_child_index(super::model::ROOT_SCROLLER_ID, page_width, roots.len());
        let root_active = root_active.min(roots.len() - 1);
        levels.push((roots.iter().collect(), root_active));
        let mut node = &roots[root_active];
        while !node.children.is_empty() {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            levels.push((node.children.iter().collect(), active));
            node = &node.children[active];
        }
        levels
    }

    /// Document-space top of a node that is **on the selected path** (after
    /// `select_path_to`), by accumulating heights down the path. `None` if the
    /// node isn't on the selected path.
    pub(crate) fn selected_path_doc_top(
        &self,
        roots: &[TreeNode],
        node_id: &str,
        page_width: Pixels,
        window_h: Pixels,
    ) -> Option<f32> {
        let mut y = TITLE_BAR_RESERVE.as_f32();
        for (i, (sibs, active)) in self
            .selected_levels(roots, page_width)
            .into_iter()
            .enumerate()
        {
            if i > 0 {
                y += BAND_HEIGHT.as_f32();
            }
            let node = sibs[active];
            if node.id == node_id {
                return Some(y);
            }
            y += self.node_height(node, page_width, window_h);
        }
        None
    }

    /// Document-space top of the selected leaf's draft slot — everything on the
    /// selected path above it.
    pub(crate) fn placeholder_doc_top(
        &self,
        roots: &[TreeNode],
        page_width: Pixels,
        window_h: Pixels,
    ) -> f32 {
        let Some(root) = self.active_root(roots, page_width) else {
            return TITLE_BAR_RESERVE.as_f32();
        };
        let total = self.selected_subtree_height(root, page_width, window_h);
        // The leaf's own (slot) height.
        let leaf_h = self
            .selected_leaf_id(roots, page_width)
            .and_then(|id| super::model::node_ref(roots, &id).map(|n| (n, id)))
            .map(|(n, _)| self.node_height(n, page_width, window_h))
            .unwrap_or(0.0);
        TITLE_BAR_RESERVE.as_f32() + total - leaf_h
    }

    /// The page scroll `y`, clamped to the content's real scrollable range
    /// `[scroll_min_y, 0]` (set once per frame in `render`). Read by everything
    /// that *positions* content from the scroll offset, so transient momentum
    /// overshoot past the ends never moves it (the generalized flicker fix).
    pub(crate) fn clamped_scroll_y(&self) -> f32 {
        self.page_scroll
            .offset()
            .y
            .as_f32()
            .clamp(self.scroll_min_y.get(), 0.0)
    }

    /// Total rendered height of the selected path from the active root.
    pub(crate) fn selected_total_height(
        &self,
        roots: &[TreeNode],
        page_width: Pixels,
        window_h: Pixels,
    ) -> f32 {
        match self.active_root(roots, page_width) {
            Some(root) => self.selected_subtree_height(root, page_width, window_h),
            None => 0.0,
        }
    }

    /// Bottom padding the page needs so the selected branch's tail can scroll
    /// clear of a *floating, off-branch* active draft. On the draft's own branch
    /// its in-flow slot already reserves the room (it docks), so this is zero;
    /// off-branch the floating bar (its content height, capped at half the
    /// window) occludes the bottom, so pad by it. This is what lets the minimap
    /// and the scroll range account for a foreign floating draft (item 4).
    pub(crate) fn floating_pad(
        &self,
        roots: &[TreeNode],
        page_width: Pixels,
        window_h: Pixels,
        _streaming: bool,
    ) -> f32 {
        let Some(active) = self.active_draft.as_deref() else {
            return 0.0;
        };
        if self.selected_leaf_id(roots, page_width).as_deref() == Some(active) {
            return 0.0; // on its own branch — docks, no extra room needed
        }
        let win = window_h.as_f32();
        let content = self.composer_content_h.borrow().as_f32();
        (Self::composer_chrome() + content).min(COMPOSER_MAX_FRACTION * win)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_change_invalidates() {
        let l = Layout::new();
        l.ensure_width(600.0);
        let id = SharedString::from("a1");
        assert!(l.record(&id, 200.0));
        assert_eq!(l.measured("a1"), Some(200.0));
        // Same width, same value → no change reported, value retained.
        assert!(!l.record(&id, 200.3));
        // Width change clears.
        l.ensure_width(480.0);
        assert_eq!(l.measured("a1"), None);
    }

    #[test]
    fn estimate_grows_with_content() {
        let short = estimate_post_height("hi", 600.0, 17.0, 1.65, 80.0);
        let long = estimate_post_height(&"word ".repeat(400), 600.0, 17.0, 1.65, 80.0);
        assert!(long > short * 4.0, "long content estimates much taller");
        // A blank reserves a line, not zero.
        assert!(estimate_post_height("", 600.0, 17.0, 1.65, 80.0) > 80.0);
    }

    #[test]
    fn retain_prunes_dead_ids() {
        let l = Layout::new();
        l.ensure_width(600.0);
        l.record(&SharedString::from("a1"), 100.0);
        l.record(&SharedString::from("a2"), 100.0);
        l.retain(&|id| id == "a1");
        assert_eq!(l.measured("a1"), Some(100.0));
        assert_eq!(l.measured("a2"), None);
    }
}
