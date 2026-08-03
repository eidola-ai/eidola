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
#[derive(Clone)]
pub struct Layout {
    /// The **reading-column** width (`body_width`) the cached heights were
    /// measured at — a post's height depends only on this, not the window width.
    width: Rc<Cell<f32>>,
    /// The type-scale factor the cached heights were measured at. A post's
    /// shaped height scales with the prose font size, so a zoom must invalidate
    /// the cache exactly like a column change (the same estimate→real re-measure
    /// jitter otherwise appears on the off-screen posts).
    scale: Rc<Cell<f32>>,
    /// Measured block height (byline + body, the full post element) by node id.
    heights: Rc<RefCell<HashMap<SharedString, f32>>>,
    /// How many times the cache has been invalidated (a resize that leaves the
    /// reading column unchanged must not bump this). Test-observable.
    clears: Rc<Cell<u32>>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            width: Rc::new(Cell::new(0.0)),
            scale: Rc::new(Cell::new(1.0)),
            heights: Rc::new(RefCell::new(HashMap::new())),
            clears: Rc::new(Cell::new(0)),
        }
    }
}

impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point the cache at reading-column `width` and type-`scale`, clearing
    /// measurements if either changed. Call once per frame before reading
    /// heights. Because the width is the *clamped* `body_width` (not the raw
    /// window width), resizes above the column cap don't invalidate anything —
    /// the measured heights survive, so the document height and scroll offset
    /// stay stable across the resize. A zoom, by contrast, reshapes every post,
    /// so a scale change *does* clear.
    pub fn ensure_width(&self, width: f32, scale: f32) {
        if (self.width.get() - width).abs() > 0.5 || (self.scale.get() - scale).abs() > 1e-3 {
            self.width.set(width);
            self.scale.set(scale);
            self.heights.borrow_mut().clear();
            self.clears.set(self.clears.get().wrapping_add(1));
        }
    }

    /// The type-scale factor the cache is currently keyed at — the height
    /// estimate multiplies the prose font size by it so an unmeasured post is
    /// approximated at the current zoom.
    pub fn scale(&self) -> f32 {
        self.scale.get()
    }

    /// How many times the cache has been invalidated (test seam).
    pub fn clears(&self) -> u32 {
        self.clears.get()
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
/// lines from the content length, times the line height, plus the editor's own
/// inter-block spacing and the post's vertical padding. Deliberately rough — it
/// is replaced by the real measured height the first frame the post is visible.
///
/// `paragraph_gap` is the editor's inter-block spacing as a multiple of the font
/// size (it pads half above / half below every block). The first block's leading
/// half plus the last block's trailing half sum to one full gap of internal
/// padding the editor *always* reserves; omitting it under-counts every post by
/// a near-constant ~`font_size * paragraph_gap`, which made minimap columns grow
/// a touch as each post first measured. The single-paragraph case is then exact.
pub fn estimate_post_height(
    content: &str,
    body_width: f32,
    font_size: f32,
    line_height: f32,
    paragraph_gap: f32,
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
    // The editor's leading `spacing_above` + trailing `spacing_below` (half a
    // `paragraph_gap` each) — a constant the old estimate dropped.
    let editor_pad = font_size * paragraph_gap;
    lines * (font_size * line_height) + editor_pad + pad_y
}

// ---------------------------------------------------------------------------
// Selection- and scroll-dependent layout, on `SpaceView` (reads its scroll
// handles, the height cache, and the window size).
// ---------------------------------------------------------------------------

use super::model::{NodeSrc, TreeNode};
use super::{
    BAND_HEIGHT, BODY_MAX_WIDTH, GUTTER_GAP, GUTTER_WIDTH, POST_PAD_Y, PROSE_FONT_SIZE,
    PROSE_LINE_HEIGHT, PROSE_PARAGRAPH_GAP, SpaceView, TITLE_BAR_RESERVE,
};
use gpui::Pixels;

/// The reading-column width for a post body at a given page width — the
/// centered measure between the two symmetric gutters (byline on the left,
/// actions on the right), capped at [`BODY_MAX_WIDTH`].
pub(crate) fn body_width(page_width: Pixels) -> f32 {
    (page_width - (GUTTER_WIDTH + GUTTER_GAP) * 2.)
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

    /// Half the document's inter-post spacing — the composer bar's **total**
    /// fixed top chrome (bar top edge → editor content). Every height / dock /
    /// runway computation uses this total; the render alone splits it at the
    /// scroll clip — a thin pane-separator band outside
    /// ([`super::composer::COMPOSER_SEPARATOR_H`]) and the remainder as an
    /// in-content spacer ([`super::composer::composer_scroll_gap`]) — so the
    /// split is invisible until the composer scrolls internally.
    pub(crate) fn composer_chrome() -> f32 {
        POST_PAD_Y.as_f32() / 2.0
    }

    /// The document's top reserve: headroom that holds whatever leads the
    /// document clear of the (transparent, overlaid) titlebar. It is
    /// **unconditional** — a composer-only notebook is led by the composer, and
    /// the words typed there have to sit exactly where they will sit once
    /// they're a post, or posting moves them (task 40: the reserve appearing
    /// with the first post shifted the whole document down by
    /// [`TITLE_BAR_RESERVE`] at the submit moment). Every scrollable-document
    /// computation (the scroll range, the forest origin, the minimap, the dock
    /// math) reads this single value, so "what's interactive" and "what's
    /// visible" stay in lockstep. The titlebar's own visual height (the
    /// gradient overlay in `render_title_bar`) is the same constant and
    /// independent of this.
    pub(crate) fn doc_reserve(&self) -> f32 {
        TITLE_BAR_RESERVE.as_f32()
    }

    /// The height a slot that **stands alone** claims: one window *below* the
    /// document's top reserve.
    ///
    /// This is what keeps the reserve free: a lone composer plus the reserve is
    /// then exactly one window, so an empty notebook still has nothing to
    /// scroll (the phantom-scroll invariant the conditional reserve used to buy
    /// by giving the composer no headroom at all), and at the end of a real
    /// conversation the fully-docked composer comes to rest with its slot top
    /// at the reserve — where a post's slot top sits — rather than sliding its
    /// text up under the titlebar.
    pub(crate) fn standalone_slot_h(&self, window_h: Pixels) -> f32 {
        (window_h.as_f32() - self.doc_reserve()).max(0.0)
    }

    /// A branch's trailing runway: at least a standalone slot (so the docked
    /// composer can stand alone below the reserve), or as tall as the
    /// composer's content if more.
    pub(crate) fn runway_height(&self, window_h: Pixels) -> f32 {
        let content = self.composer_content_h.borrow().as_f32();
        self.standalone_slot_h(window_h)
            .max(Self::composer_chrome() + content)
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
            // at least a standalone slot (it's always the end of its branch), so
            // there's perfect continuity with the active draft's runway slot —
            // activating/deactivating it never resizes the layout. `min_h` on
            // the inline frame makes the measured height honour the same floor.
            NodeSrc::Draft => self
                .layout
                .measured(&node.id)
                .unwrap_or(0.0)
                .max(self.standalone_slot_h(window_h)),
            NodeSrc::Streaming(_) => self
                .layout
                .measured(&node.id)
                .unwrap_or_else(|| window_h.as_f32() * 0.3),
            NodeSrc::Msg(i) => self.layout.measured(&node.id).unwrap_or_else(|| {
                let content = self.posts.get(i).map(|p| p.content.as_ref()).unwrap_or("");
                estimate_post_height(
                    content,
                    body_width(page_width),
                    // Match the scaled prose font the post will actually shape
                    // at, so the pre-measure estimate is right at any zoom.
                    PROSE_FONT_SIZE.as_f32() * self.layout.scale(),
                    PROSE_LINE_HEIGHT,
                    PROSE_PARAGRAPH_GAP,
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

    /// The slot height a **trailing draft** claims at the end of the selected
    /// path, or `0.0` when the path ends in a post or a streaming leaf.
    ///
    /// That slot is speculative — an empty composer reserves a whole window of
    /// runway — which is why tail-following stops short of it (see
    /// [`super::SpaceView::follow_streaming_tail`]).
    pub(crate) fn trailing_draft_slot_h(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
    ) -> f32 {
        let mut node = match self.active_root(roots, page_width) {
            Some(n) => n,
            None => return 0.0,
        };
        while !node.children.is_empty() {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            node = &node.children[active];
        }
        match node.src {
            NodeSrc::Draft => self.node_height(node, page_width, window_h),
            _ => 0.0,
        }
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
            if matches!(node.src, NodeSrc::Draft | NodeSrc::Streaming(_)) {
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

    /// Whether any **post** on the selected path lacks a measured height (still
    /// using an estimate). Drives the warm pass (see `render`): off-path siblings
    /// are intentionally ignored — they never render on the path, so they'd never
    /// measure and would keep the warm armed forever.
    pub(crate) fn path_has_unmeasured(&self, roots: &[TreeNode], page_width: Pixels) -> bool {
        self.selected_levels(roots, page_width)
            .into_iter()
            .map(|(sibs, active)| sibs[active])
            .any(|node| {
                matches!(node.src, NodeSrc::Msg(_)) && self.layout.measured(&node.id).is_none()
            })
    }

    /// Whether the **selected path** carries a live streaming overlay — the
    /// honest scope for tail-following (`follow_streaming_tail`).
    ///
    /// "Is some turn streaming?" is a *space*-wide question, and Participants v1
    /// makes concurrent turns on sibling branches ordinary: a fan-out streams
    /// several replies at once, each attached at its own target post. The reader
    /// is on exactly one branch, and only *that* branch's tail is producing for
    /// them. Answering the space-wide question would let a sibling's stream
    /// re-enable following for a selected branch whose own growth is the
    /// composer's runway or a post measuring for the first time — precisely the
    /// non-stream growth the design excludes (and which the composer's own
    /// `caret_into_view` path owns). Like every other selection question here it
    /// is answered by *observation* of the tree the frame actually renders — no
    /// flag, no mode.
    pub(crate) fn selected_path_is_streaming(
        &self,
        roots: &[TreeNode],
        page_width: Pixels,
    ) -> bool {
        self.selected_levels(roots, page_width)
            .into_iter()
            .any(|(sibs, active)| matches!(sibs[active].src, NodeSrc::Streaming(_)))
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
        let mut y = self.doc_reserve();
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
            return self.doc_reserve();
        };
        let total = self.selected_subtree_height(root, page_width, window_h);
        // The leaf's own (slot) height.
        let leaf_h = self
            .selected_leaf_id(roots, page_width)
            .and_then(|id| super::model::node_ref(roots, &id).map(|n| (n, id)))
            .map(|(n, _)| self.node_height(n, page_width, window_h))
            .unwrap_or(0.0);
        self.doc_reserve() + total - leaf_h
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

    /// The floating composer bar's height under the window's live sizing
    /// state: the natural content height (`composer_chrome() + content`)
    /// capped at `composer_fraction · window` (**Max**, the resting behavior),
    /// or pinned to exactly that fraction regardless of content (**Exact**,
    /// entered by the separator-handle resize drag). The one place the
    /// window-local fraction meets geometry — the composer render, the
    /// pre-dock glide, and the off-branch floating pad all read this, so they
    /// can never disagree on the bar. Pure core:
    /// [`super::composer::float_bar_height`], unit-tested.
    pub(crate) fn composer_float_bar_h(&self, window_h: Pixels) -> f32 {
        let natural = Self::composer_chrome() + self.composer_content_h.borrow().as_f32();
        super::composer::float_bar_height(
            natural,
            self.composer_fraction,
            window_h.as_f32(),
            self.composer_sizing,
        )
    }

    /// Bottom padding the page needs so the selected branch's tail can scroll
    /// clear of a *floating, off-branch* active draft. On the draft's own branch
    /// its in-flow slot already reserves the room (it docks), so this is zero;
    /// off-branch the floating bar occludes the bottom, so pad by its height
    /// ([`Self::composer_float_bar_h`] — fraction- and sizing-aware). This is
    /// what lets the minimap and the scroll range account for a foreign
    /// floating draft (item 4).
    pub(crate) fn floating_pad(
        &self,
        roots: &[TreeNode],
        page_width: Pixels,
        window_h: Pixels,
    ) -> f32 {
        let Some(active) = self.active_draft.as_deref() else {
            return 0.0;
        };
        if self.selected_leaf_id(roots, page_width).as_deref() == Some(active) {
            return 0.0; // on its own branch — docks, no extra room needed
        }
        self.composer_float_bar_h(window_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_change_invalidates() {
        let l = Layout::new();
        l.ensure_width(600.0, 1.0);
        let id = SharedString::from("a1");
        assert!(l.record(&id, 200.0));
        assert_eq!(l.measured("a1"), Some(200.0));
        // Same width + scale, same value → no change reported, value retained.
        assert!(!l.record(&id, 200.3));
        // Same width and scale re-asserted → cache survives (no phantom clear).
        l.ensure_width(600.0, 1.0);
        assert_eq!(l.measured("a1"), Some(200.0));
        // Width change clears.
        l.ensure_width(480.0, 1.0);
        assert_eq!(l.measured("a1"), None);
    }

    #[test]
    fn scale_change_invalidates() {
        let l = Layout::new();
        l.ensure_width(600.0, 1.0);
        let id = SharedString::from("a1");
        l.record(&id, 200.0);
        assert_eq!(l.measured("a1"), Some(200.0));
        assert_eq!(l.scale(), 1.0);
        // A zoom reshapes every post, so a scale change clears at the same
        // width and updates the reported scale (which the estimate reads).
        let before = l.clears();
        l.ensure_width(600.0, 1.25);
        assert_eq!(l.measured("a1"), None);
        assert_eq!(l.scale(), 1.25);
        assert_eq!(l.clears(), before + 1);
    }

    #[test]
    fn estimate_grows_with_content() {
        let short = estimate_post_height("hi", 600.0, 17.0, 1.65, 1.5, 80.0);
        let long = estimate_post_height(&"word ".repeat(400), 600.0, 17.0, 1.65, 1.5, 80.0);
        assert!(long > short * 4.0, "long content estimates much taller");
        // A blank reserves a line, not zero.
        assert!(estimate_post_height("", 600.0, 17.0, 1.65, 1.5, 80.0) > 80.0);
    }

    #[test]
    fn estimate_includes_editor_internal_spacing() {
        // A single-line post: one line box + the editor's leading+trailing
        // spacing (a full `paragraph_gap` of the font size) + the post padding.
        // This is the term the old estimate dropped, which made columns grow on
        // first measure.
        let h = estimate_post_height("hi", 600.0, 17.0, 1.65, 1.5, 80.0);
        let expected = 17.0 * 1.65 + 17.0 * 1.5 + 80.0;
        assert!(
            (h - expected).abs() < 0.01,
            "single-line estimate {h} should match {expected} exactly",
        );
    }

    #[test]
    fn retain_prunes_dead_ids() {
        let l = Layout::new();
        l.ensure_width(600.0, 1.0);
        l.record(&SharedString::from("a1"), 100.0);
        l.record(&SharedString::from("a2"), 100.0);
        l.retain(&|id| id == "a1");
        assert_eq!(l.measured("a1"), Some(100.0));
        assert_eq!(l.measured("a2"), None);
    }
}
