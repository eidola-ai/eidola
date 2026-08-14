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
    /// Whether the cached row heights include compact metadata/action lines.
    stacked: Rc<Cell<bool>>,
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
            stacked: Rc::new(Cell::new(false)),
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
    pub(crate) fn ensure_width(&self, width: f32, scale: f32, gutters: GutterPlacement) {
        let stacked = gutters == GutterPlacement::Stacked;
        if (self.width.get() - width).abs() > 0.5
            || (self.scale.get() - scale).abs() > 1e-3
            || self.stacked.get() != stacked
        {
            self.width.set(width);
            self.scale.set(scale);
            self.stacked.set(stacked);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GutterPlacement {
    Sides,
    Stacked,
}

/// The shared horizontal contract for every conversation row. Side gutters
/// stay in place while they leave the prose at its full measure. Once they
/// would squeeze it, metadata and actions stack around the prose instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PageLayout {
    pub(crate) body_width: f32,
    pub(crate) gutters: GutterPlacement,
}

/// The compact composer's vertical occupancy around the editor — both parts
/// exist only in the Stacked scheme, and they live in *different states*:
///
/// - `top` is the **docked byline row** — the "You" line the docked bar shows
///   at a post's metadata position. It is docked-only chrome (the floating bar
///   carries no byline; see the dock-reveal machinery in `composer.rs`), so it
///   counts toward the docked natural height and the runway but **not** toward
///   the floating bar.
/// - `bottom` is the **bottom action bar** — Post and its siblings on their
///   own surface anchored to the window's bottom edge, present (and reserved)
///   whenever the actions are revealed, floating and docked alike.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ComposerGutterHeights {
    pub(crate) top: f32,
    pub(crate) bottom: f32,
}

impl ComposerGutterHeights {
    pub(crate) fn total(self) -> f32 {
        self.top + self.bottom
    }
}

pub(crate) const COMPACT_GUTTER_LINE_REMS: f32 = 1.5;
pub(crate) const COMPACT_GUTTER_GAP_REMS: f32 = 0.5;
/// The bottom action bar's clearance under its verb line — what keeps Post
/// off the very edge of the screen.
pub(crate) const COMPACT_ACTION_BAR_CLEARANCE_REMS: f32 = 1.0;
pub(crate) const COMPACT_PAGE_INSET: Pixels = POST_PAD_Y;

pub(crate) fn compact_gutter_occupancy(rem_size: Pixels) -> f32 {
    rem_size.as_f32() * (COMPACT_GUTTER_LINE_REMS + COMPACT_GUTTER_GAP_REMS)
}

/// The compact bottom action bar's total height — a verb line plus a gutter
/// gap and a full clearance of surrounding room (the line rides vertically
/// centered in it), so the verbs never crowd the window edge.
pub(crate) fn compact_action_bar_h(rem_size: Pixels) -> f32 {
    rem_size.as_f32()
        * (COMPACT_GUTTER_GAP_REMS + COMPACT_GUTTER_LINE_REMS + COMPACT_ACTION_BAR_CLEARANCE_REMS)
}

pub(crate) fn composer_gutter_heights(
    page_layout: PageLayout,
    rem_size: Pixels,
    actions_revealed: bool,
) -> ComposerGutterHeights {
    match page_layout.gutters {
        GutterPlacement::Sides => ComposerGutterHeights::default(),
        GutterPlacement::Stacked => ComposerGutterHeights {
            top: compact_gutter_occupancy(rem_size),
            bottom: if actions_revealed {
                compact_action_bar_h(rem_size)
            } else {
                0.0
            },
        },
    }
}

/// How much page travel past the dock threshold completes the compact
/// byline's reveal (and the matching content slide) — short enough that any
/// deliberately docked position shows the settled, post-parity layout, long
/// enough that the transition reads as a slide rather than a pop.
pub(crate) const DOCK_REVEAL_SPAN: f32 = 56.0;

/// The compact docked byline's reveal progress: `0` at (or above) the float
/// line — the floating bar carries no byline — ramping to `1` over the first
/// [`DOCK_REVEAL_SPAN`] of page travel past the dock threshold. Drives both
/// the byline's opacity and the dead-space inset that slides the editor down
/// to its post-parity offset, so the two can never disagree.
pub(crate) fn dock_reveal_progress(float_top: f32, top_y: f32) -> f32 {
    ((float_top - top_y) / DOCK_REVEAL_SPAN).clamp(0.0, 1.0)
}

pub(crate) fn page_layout(page_width: Pixels) -> PageLayout {
    if page_width >= full_measure_page_width() {
        PageLayout {
            body_width: BODY_MAX_WIDTH.as_f32(),
            gutters: GutterPlacement::Sides,
        }
    } else {
        PageLayout {
            body_width: (page_width.min(compact_full_measure_page_width())
                - COMPACT_PAGE_INSET * 2.0)
                .max(gpui::px(0.))
                .as_f32(),
            gutters: GutterPlacement::Stacked,
        }
    }
}

/// The reading-column width for a post body at a given page width.
pub(crate) fn body_width(page_width: Pixels) -> f32 {
    page_layout(page_width).body_width
}

/// The narrowest page that fits the full reading measure between side gutters.
/// Below it the gutters stack; `lib.rs::writing_surface_size` derives the
/// default space-window width from this so a fresh window opens in side mode.
pub(crate) fn full_measure_page_width() -> Pixels {
    BODY_MAX_WIDTH + (GUTTER_WIDTH + GUTTER_GAP) * 2.
}

/// The narrowest stacked page that preserves the full reading measure.
pub(crate) fn compact_full_measure_page_width() -> Pixels {
    BODY_MAX_WIDTH + COMPACT_PAGE_INSET * 2.0
}

/// The selected branch's two ends as page-scroll `y`s (both ≤ 0). See
/// [`SpaceView::page_end_ys`] — `content` is where a reader comes to rest,
/// `document` is only the scroll floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PageEnds {
    /// The end of the document — the scroll range's floor.
    pub(crate) document: f32,
    /// The end of what has been **written**: the document less a trailing
    /// draft's speculative runway. Never past `document`.
    pub(crate) content: f32,
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

    /// The document's full inter-post pad — the composer bar's **total** fixed
    /// top chrome (bar top edge → editor content). It is the whole
    /// [`POST_PAD_Y`] because the docked bar's top edge sits exactly at its
    /// slot top, where a post row begins: the editor content then starts
    /// `POST_PAD_Y` below the slot top, exactly where a post's body sits under
    /// its own top pad, and the bar's surface abuts the separator band above
    /// it the way every post row does. Every height / dock / runway
    /// computation uses this total; the render alone splits it at the scroll
    /// clip — a thin pane-separator band outside
    /// ([`super::composer::COMPOSER_SEPARATOR_H`]) and the remainder as an
    /// in-content spacer ([`super::composer::composer_scroll_gap`]) — so the
    /// split is invisible until the composer scrolls internally.
    pub(crate) fn composer_chrome() -> f32 {
        POST_PAD_Y.as_f32()
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

    /// The minimum runway for a trailing draft, independent of any draft's
    /// content. Populated conversations claim **exactly one window**, and the
    /// docked bar's top edge is its slot top ([`Self::composer_chrome`]), so
    /// at the document floor the two readings coincide: the slot — the
    /// composer's minimum height, and the runway a reader scrolls through —
    /// fills the window, and the painted surface is flush with the window's
    /// top edge, the previous separator band just cleared above it. A blank
    /// notebook keeps its titlebar-adjusted no-scroll slot.
    pub(crate) fn runway_floor(&self, window_h: Pixels) -> f32 {
        if self.posts.is_empty() {
            self.standalone_slot_h(window_h)
        } else {
            window_h.as_f32()
        }
    }

    /// The active composer's runway combines the shared floor with that
    /// composer's own measured natural height.
    pub(crate) fn runway_height(&self, window_h: Pixels) -> f32 {
        self.runway_floor(window_h)
            .max(self.composer_natural_height())
    }

    /// An inactive draft is an inline row, so its own recorded layout — never
    /// the active composer's natural height — decides whether it exceeds the
    /// same trailing runway floor.
    pub(crate) fn inactive_draft_height(&self, id: &str, window_h: Pixels) -> f32 {
        self.layout
            .measured(id)
            .unwrap_or(0.0)
            .max(self.runway_floor(window_h))
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
            // An inactive draft renders inline as an editable post and claims
            // its own measured height over the shared runway floor. That keeps
            // one draft continuous across activation without letting another
            // active draft inflate it. `min_h` on the inline frame makes the
            // measurement honour the same floor.
            NodeSrc::Draft => self.inactive_draft_height(&node.id, window_h),
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

    /// **The selected branch's two ends**, as page-scroll `y`s — the single
    /// definition every "scroll to the end" reads, so no caller can invent a
    /// second one.
    ///
    /// The distinction is task 46, bug 2: a trailing draft's slot is a whole
    /// window of *speculative* runway ([`Self::trailing_draft_slot_h`]), and
    /// coming to rest past the last written word carries the reply the reader
    /// was reading off the top of the window. So tail-following stops at
    /// `content` ([`super::SpaceView::follow_streaming_tail`]) — and so does
    /// every programmatic settle at a branch's end
    /// ([`super::SpaceView::scroll_to_branch_end`]), which lands on frames
    /// where `sync_tail_drafts` has already docked that composer. The two
    /// coincide whenever no draft trails the path, which is every frame a turn
    /// is actually streaming.
    pub(crate) fn page_end_ys(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
    ) -> PageEnds {
        let total_doc = self.doc_reserve()
            + self.selected_total_height(roots, page_width, window_h)
            + self.floating_pad(roots, page_width, window_h);
        let document = (window_h.as_f32() - total_doc).min(0.0);
        let content = (window_h.as_f32()
            - (total_doc - self.trailing_draft_slot_h(roots, page_width, window_h)))
        .min(0.0)
        .max(document);
        PageEnds { document, content }
    }

    /// The slot height a **trailing draft** claims at the end of the selected
    /// path, or `0.0` when the path ends in a post or a streaming leaf.
    ///
    /// That slot is speculative — an empty composer reserves a whole window of
    /// runway — which is why "the end" has two values (see
    /// [`Self::page_end_ys`]).
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

    /// **Which** turn's streaming leaf the **selected path** carries, if any (a
    /// streaming overlay is always a leaf, so there is at most one) — the
    /// honest scope for tail-following (`follow_streaming_tail`), and the
    /// identity a completed turn is followed by (`follow_completed_turn`).
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
    ///
    /// It names the turn rather than merely reporting one because the second
    /// consumer asks *after the fact*: a completed turn's leaf is gone from the
    /// tree, so "was the reader parked on it?" can only be answered by what the
    /// last frame observed — which `render` records in
    /// [`super::SpaceView::selected_turn`].
    pub(crate) fn selected_turn_seq(&self, roots: &[TreeNode], page_width: Pixels) -> Option<u64> {
        self.selected_levels(roots, page_width)
            .into_iter()
            .find_map(|(sibs, active)| match sibs[active].src {
                NodeSrc::Streaming(seq) => Some(seq),
                _ => None,
            })
    }

    /// The same question asked of a branch switch's **destination**: the
    /// streaming turn at the end of the path that will be selected once
    /// `parent_id`'s strip rests on child `index`.
    ///
    /// A switch knows its destination at click time; the strip's offset does
    /// not say so until the switch lands (and says the *old* child for as long
    /// as an animation is rounding toward the new one). Recording this is what
    /// lets a turn completing across that window carry the reader where they
    /// were going rather than where they were — see
    /// [`super::SpaceView::follow_completed_turn`].
    pub(crate) fn turn_seq_under_child(
        &self,
        roots: &[TreeNode],
        parent_id: &str,
        index: usize,
        page_width: Pixels,
    ) -> Option<u64> {
        let parent = super::model::node_ref(roots, parent_id)?;
        let mut node = parent.children.get(index)?;
        while !node.children.is_empty() {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            node = &node.children[active];
        }
        match node.src {
            NodeSrc::Streaming(seq) => Some(seq),
            _ => None,
        }
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
        // Exact sizing pins the bar to the window fraction regardless of
        // content, and at the minimum fraction in a short window that can
        // dip below the bar's *fixed* surfaces (the top chrome and the
        // compact action bar). Clamp so the chrome always fits — the editor
        // is what compresses, never the fixed surfaces.
        let fixed =
            (Self::composer_chrome() + self.composer_gutters.get().bottom).min(window_h.as_f32());
        super::composer::float_bar_height(
            self.composer_floating_natural_height(),
            self.composer_fraction,
            window_h.as_f32(),
            self.composer_sizing,
        )
        .max(fixed)
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
        l.ensure_width(600.0, 1.0, GutterPlacement::Sides);
        let id = SharedString::from("a1");
        assert!(l.record(&id, 200.0));
        assert_eq!(l.measured("a1"), Some(200.0));
        // Same width + scale, same value → no change reported, value retained.
        assert!(!l.record(&id, 200.3));
        // Same width and scale re-asserted → cache survives (no phantom clear).
        l.ensure_width(600.0, 1.0, GutterPlacement::Sides);
        assert_eq!(l.measured("a1"), Some(200.0));
        // Width change clears.
        l.ensure_width(480.0, 1.0, GutterPlacement::Sides);
        assert_eq!(l.measured("a1"), None);
    }

    #[test]
    fn scale_change_invalidates() {
        let l = Layout::new();
        l.ensure_width(600.0, 1.0, GutterPlacement::Sides);
        let id = SharedString::from("a1");
        l.record(&id, 200.0);
        assert_eq!(l.measured("a1"), Some(200.0));
        assert_eq!(l.scale(), 1.0);
        // A zoom reshapes every post, so a scale change clears at the same
        // width and updates the reported scale (which the estimate reads).
        let before = l.clears();
        l.ensure_width(600.0, 1.25, GutterPlacement::Sides);
        assert_eq!(l.measured("a1"), None);
        assert_eq!(l.scale(), 1.25);
        assert_eq!(l.clears(), before + 1);
    }

    #[test]
    fn gutter_placement_change_invalidates() {
        let l = Layout::new();
        l.ensure_width(600.0, 1.0, GutterPlacement::Sides);
        l.record(&SharedString::from("a1"), 200.0);
        let before = l.clears();
        l.ensure_width(600.0, 1.0, GutterPlacement::Stacked);
        assert_eq!(l.measured("a1"), None);
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
    fn page_layout_preserves_measure_across_gutter_breakpoint() {
        let full = full_measure_page_width();
        assert_eq!(
            page_layout(full),
            PageLayout {
                body_width: BODY_MAX_WIDTH.as_f32(),
                gutters: GutterPlacement::Sides,
            }
        );
        assert_eq!(
            page_layout(full - gpui::px(1.)),
            PageLayout {
                body_width: BODY_MAX_WIDTH.as_f32(),
                gutters: GutterPlacement::Stacked,
            }
        );
        assert_eq!(
            body_width(compact_full_measure_page_width()),
            BODY_MAX_WIDTH.as_f32()
        );
        assert_eq!(
            body_width(compact_full_measure_page_width() - gpui::px(1.)),
            BODY_MAX_WIDTH.as_f32() - 1.0
        );
    }

    #[test]
    fn composer_gutters_share_compact_occupancy() {
        let compact = PageLayout {
            body_width: BODY_MAX_WIDTH.as_f32(),
            gutters: GutterPlacement::Stacked,
        };
        // The docked byline row is unconditional — a docked composer always
        // shows "You", exactly as a post always shows its metadata; only the
        // bottom action bar waits for the actions to reveal.
        let line = compact_gutter_occupancy(gpui::px(16.));
        let bar = compact_action_bar_h(gpui::px(16.));
        assert_eq!(
            composer_gutter_heights(compact, gpui::px(16.), false),
            ComposerGutterHeights {
                top: line,
                bottom: 0.0,
            }
        );
        assert_eq!(
            composer_gutter_heights(compact, gpui::px(16.), true),
            ComposerGutterHeights {
                top: line,
                bottom: bar,
            }
        );
        assert!(
            bar > line,
            "the action bar carries clearance beyond a bare gutter line, \
             keeping Post off the window edge"
        );
        assert_eq!(
            composer_gutter_heights(
                PageLayout {
                    body_width: BODY_MAX_WIDTH.as_f32(),
                    gutters: GutterPlacement::Sides,
                },
                gpui::px(16.),
                true,
            ),
            ComposerGutterHeights::default()
        );
    }

    #[test]
    fn dock_reveal_ramps_over_the_first_travel_past_the_threshold() {
        // Floating (at or above the float line): no byline.
        assert_eq!(dock_reveal_progress(500.0, 500.0), 0.0);
        assert_eq!(dock_reveal_progress(500.0, 520.0), 0.0);
        // Half the span in: half revealed.
        assert_eq!(
            dock_reveal_progress(500.0, 500.0 - DOCK_REVEAL_SPAN / 2.0),
            0.5
        );
        // Anywhere deeper than the span: fully settled, post-parity layout.
        assert_eq!(dock_reveal_progress(500.0, 500.0 - DOCK_REVEAL_SPAN), 1.0);
        assert_eq!(dock_reveal_progress(500.0, 0.0), 1.0);
    }

    #[test]
    fn retain_prunes_dead_ids() {
        let l = Layout::new();
        l.ensure_width(600.0, 1.0, GutterPlacement::Sides);
        l.record(&SharedString::from("a1"), 100.0);
        l.record(&SharedString::from("a2"), 100.0);
        l.retain(&|id| id == "a1");
        assert_eq!(l.measured("a1"), Some(100.0));
        assert_eq!(l.measured("a2"), None);
    }
}
