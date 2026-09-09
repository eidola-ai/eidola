//! **The Find-all overlay** — the surface behind the find bar's "N total ▲"
//! disclosure.
//!
//! It expands down from the bar over the rest of the window and answers a
//! different question from the bar's: not *where am I in this branch*, but
//! *what does this whole conversation hold*. Two halves, side by side:
//!
//! - **Left, the topological map.** A brand-new surface, and deliberately **not
//!   the minimap**. The minimap is the scroll handle: spatially sized to the
//!   branch the reader is on, with the branches one gesture away drawn beside
//!   it. This map is the whole space's shape with no regard to the current
//!   branch and none to post length — one circle per post, `git log --graph`,
//!   every node at the same relative position it occupies in the conversation.
//!   It carries **no counts**: a node either holds matches or it does not, and
//!   what it is *for* is navigating the ordered result list, not orienting the
//!   reader inside a branch. Neither surface duplicates the other.
//! - **Right, the results.** One group per node that holds matches, under the
//!   post's own attribution, and inside it one **fragment** per matching block
//!   — the block still wearing its ancestor containers, because that is what
//!   makes a paragraph inside a blockquote inside a list item read as what it
//!   is. Contiguous blocks are consolidated into one fragment.
//!
//! ## Order
//!
//! **Depth from the root first, then left to right as laid out on the map.**
//! That is deliberately *not* the transcript's own pre-order: the overlay
//! exists to present the same results in a different orientation, for quick
//! discovery across a deeply branched space, so agreeing with the transcript
//! would be the failure mode rather than the goal. [`map_layout`] assigns each
//! node its `(depth, lane)` once and both surfaces read it, so the list and the
//! map cannot disagree about where a post is.
//!
//! ## Where the results come from
//!
//! **Nowhere new.** A fragment is cut from the block ranges and the hits the
//! *projection cache* already holds — `sync_find` fills a node's entry for the
//! visible branch and the whole-space count fills it for every other post, both
//! through `hits_of`. So the overlay introduces no second reading of what
//! matches: it shows what the bar counted, and the exactness doctrine's sums
//! stay the only counts in the feature (the overlay shows none).
//!
//! ## What clicking one spends
//!
//! Find never moves the reader's branch — that property is what makes ⌘F safe
//! in a tree, and every other surface is built to avoid spending it (the
//! off-branch composer is revealed by its *own* viewport for exactly this
//! reason). **The overlay is the one place the feature does change branch, and
//! only because the reader asked**: a click selects the path to the fragment's
//! node, sets the bar's anchor to that match, and hands the reveal to the
//! ordinary two-phase machinery by arming `reveal_when_anchored` — the same
//! debt a new query arms, discharged in `sync_find` where the anchor exists.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::rc::Rc;

use gpui::{
    AnyElement, AppContext, Bounds, Context, Entity, InteractiveElement, IntoElement,
    ParentElement, Pixels, SharedString, StatefulInteractiveElement, Styled, WeakEntity, Window,
    div, prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::{HighlightLayer, MarkdownEditor, MarkdownEditorState};

use super::find::{FIND_BAR_H, MatchAnchor};
use super::model::{NodeSrc, TreeNode};
use super::{SpaceView, TITLE_BAR_RESERVE, prose_style};
use crate::overlay::{Contain as _, Overlay};
use crate::probe::Probe as _;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// The map column's width. Fixed, because the map is a sketch of shape rather
/// than a measured image of the document — nothing about it scales with the
/// conversation except its height.
const MAP_WIDTH: f32 = 168.0;
/// The vertical stride between two depths of the map.
const MAP_ROW_H: f32 = 26.0;
/// The horizontal stride between two lanes, and the floor it may shrink to
/// before a very wide graph simply clips: a dot is [`MAP_DOT`] across, so
/// anything below this would draw two branches on top of each other and say
/// something untrue about the topology.
const MAP_LANE_W: f32 = 20.0;
const MAP_LANE_MIN_W: f32 = 11.0;
/// One node's circle.
const MAP_DOT: f32 = 9.0;
/// The map's own breathing room inside its column.
const MAP_PAD: f32 = 16.0;
/// A result group's attribution header.
const GROUP_HEADER_H: f32 = 30.0;
/// The gap under each fragment card.
const FRAGMENT_GAP: f32 = 10.0;
/// How far outside the results viewport a fragment still renders for real.
///
/// The same shape the transcript's virtualization takes, for the same reason: a
/// fragment renders a whole `MarkdownEditor` — a parse and a render pass — so
/// building one per result would make a one-character query on a long
/// conversation freeze the window. Everything outside the band is a sized
/// placeholder, and its size is the measured height where the fragment has
/// painted once and an estimate before that.
const RESULT_MARGIN: f32 = 600.0;
/// The estimated height of a fragment that has never painted — one prose line
/// per wrapped line of its source, plus the card's own chrome. Replaced by the
/// measurement the frame after it first paints, exactly as a post's estimate is.
const FRAGMENT_CHROME_H: f32 = 24.0;
/// How much of a fragment its accessible name carries. A card's whole text
/// would be read aloud on every cursor move; the opening is what tells a reader
/// which result this is, and Enter takes them to the passage itself.
const FRAGMENT_LABEL_CHARS: usize = 120;
/// The results column's own inset either side of a fragment card.
const RESULTS_PAD: f32 = 16.0;

// ---------------------------------------------------------------------------
// The map's layout — pure
// ---------------------------------------------------------------------------

/// One node's place on the map: how far it is from a root, and which lane it
/// occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MapNode {
    pub(crate) node: SharedString,
    /// Distance from the thread root — the map's row, and the **primary** key
    /// of the result order.
    pub(crate) depth: usize,
    /// The column, `git log --graph`-style: a node continues its parent's lane
    /// when it is the parent's first child and takes a fresh one otherwise.
    /// Lane index *is* the x position, so ordering by it is ordering the way
    /// the map is laid out.
    pub(crate) lane: usize,
    /// The parent's `(depth, lane)`, for the edge drawn back up to it.
    pub(crate) parent: Option<(usize, usize)>,
}

/// Lay the whole space out as a graph: depth from the root down, lanes across.
///
/// `include` decides what is a node of the conversation at all — a post always
/// is; a draft is one once the reader has written in it; a streaming leaf never
/// is, on the same rule that keeps it out of the match set until it finalizes.
/// Every excluded node is a leaf, so excluding one changes nothing about the
/// shape of what is left.
///
/// **Lanes cannot collide**, which is what makes the sort total: a lane is
/// allocated once and then only ever continues downward through first children,
/// so it is a single root-to-leaf chain and meets each depth exactly once.
pub(crate) fn map_layout(roots: &[TreeNode], include: &dyn Fn(&TreeNode) -> bool) -> Vec<MapNode> {
    let mut out = Vec::new();
    let mut next_lane = 0usize;
    let kept: Vec<&TreeNode> = roots.iter().filter(|n| include(n)).collect();
    let lanes = reserve_lanes(kept.len(), None, &mut next_lane);
    for (root, lane) in kept.into_iter().zip(lanes) {
        lay_out_node(root, 0, lane, None, &mut next_lane, include, &mut out);
    }
    // The result order, and the reading order of the map itself.
    out.sort_by_key(|n| (n.depth, n.lane));
    out
}

/// Lanes for one node's children (or for the thread roots, where `own` is
/// `None`): the first continues its parent's, and each of the rest opens a new
/// one.
///
/// **Reserved for the whole sibling set before any of them is descended into**,
/// which is the difference between a map whose forks read as forks and one
/// whose branches are ordered by how deep their neighbours happen to run: with
/// lanes allocated on the way down, a branch off the root lands to the *right*
/// of a branch two levels below it, simply because the earlier subtree opened
/// its lanes first. Siblings are the thing a reader is comparing, so siblings
/// are what stay adjacent.
fn reserve_lanes(count: usize, own: Option<usize>, next_lane: &mut usize) -> Vec<usize> {
    (0..count)
        .map(|i| match (i, own) {
            (0, Some(lane)) => lane,
            _ => {
                let l = *next_lane;
                *next_lane += 1;
                l
            }
        })
        .collect()
}

fn lay_out_node(
    node: &TreeNode,
    depth: usize,
    lane: usize,
    parent: Option<(usize, usize)>,
    next_lane: &mut usize,
    include: &dyn Fn(&TreeNode) -> bool,
    out: &mut Vec<MapNode>,
) {
    out.push(MapNode {
        node: node.id.clone(),
        depth,
        lane,
        parent,
    });
    let kids: Vec<&TreeNode> = node.children.iter().filter(|c| include(c)).collect();
    let lanes = reserve_lanes(kids.len(), Some(lane), next_lane);
    for (child, child_lane) in kids.into_iter().zip(lanes) {
        lay_out_node(
            child,
            depth + 1,
            child_lane,
            Some((depth, lane)),
            next_lane,
            include,
            out,
        );
    }
}

/// Cut one node's hits into **fragments**: consolidated runs of the blocks the
/// render laid the node out in, each carrying the index of the first hit inside
/// it.
///
/// A fragment is the unit the reader sees and clicks, and it is a *block* run
/// rather than a match because a match is a phrase and a block is a thing on
/// the page. Blocks that are contiguous in the render — and any two blocks a
/// single hit spans — consolidate into one fragment, which is what keeps two
/// matches in adjacent list items from being drawn as two disjoint cards with a
/// seam between them.
///
/// Hits that fall in no block are dropped rather than approximated: every hit
/// the projection reports comes out of a block, so one that does not is a
/// disagreement, and showing a fragment for it would be inventing a place.
///
/// **Both inputs are already in document order, so this is one walk rather
/// than a cross product.** A nested loop compared every hit with every block on
/// every frame the overlay drew — a one-character query matching each of ten
/// thousand paragraphs is a hundred million range checks per frame, ahead of
/// the virtualization that exists to bound exactly this, and the window freezes
/// while the reader merely scrolls. The block cursor never rewinds, because a
/// later hit can only start at or after the current one; the inner walk covers
/// the one hit that spans several blocks without moving it.
pub(crate) fn fragment_runs(blocks: &[Range<usize>], hits: &[Range<usize>]) -> Vec<FragmentRun> {
    // block index -> the lowest and highest hit index landing in it.
    let mut touched: Vec<Option<(usize, usize)>> = vec![None; blocks.len()];
    let mut b = 0usize;
    for (h, hit) in hits.iter().enumerate() {
        let end = hit.end.max(hit.start + 1);
        while b < blocks.len() && blocks[b].end <= hit.start {
            b += 1;
        }
        let mut j = b;
        while j < blocks.len() && blocks[j].start < end {
            match &mut touched[j] {
                Some((_, last)) => *last = h,
                slot => *slot = Some((h, h)),
            }
            j += 1;
        }
    }
    let mut out: Vec<FragmentRun> = Vec::new();
    // (first block, last block, first hit, last hit)
    let mut run: Option<(usize, usize, usize, usize)> = None;
    let close = |out: &mut Vec<FragmentRun>,
                 (first, last, h0, h1): (usize, usize, usize, usize)| {
        out.push(FragmentRun {
            range: blocks[first].start..blocks[last].end,
            hits: h0..h1 + 1,
        });
    };
    for (b, slot) in touched.iter().enumerate() {
        let Some((lo, hi)) = *slot else { continue };
        run = match run {
            Some((first, last, h0, h1)) if b == last + 1 => {
                Some((first, b, h0.min(lo), h1.max(hi)))
            }
            Some(open) => {
                close(&mut out, open);
                Some((b, b, lo, hi))
            }
            None => Some((b, b, lo, hi)),
        };
    }
    if let Some(open) = run {
        close(&mut out, open);
    }
    out
}

/// One fragment the cutter found: the consolidated block span it paints, and
/// **the slice of the node's hits that fall inside it** — a range into that
/// node's own ordered hit vector rather than a copy.
///
/// The hits are a range because a node's hits partition across its fragments
/// exactly: they ascend, and two blocks a single hit spans are consolidated
/// into one run by construction, so no hit can straddle two fragments. The
/// range's start is the fragment's **ordinal within the node** — half of the
/// anchor a click hands the bar — and its length is what the card's highlight
/// layer takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FragmentRun {
    /// The consolidated block span, in the node's source bytes.
    pub(crate) range: Range<usize>,
    /// The fragment's hits, as indices into the node's hit vector.
    pub(crate) hits: Range<usize>,
}

/// The horizontal stride between two lanes of the map, for a graph that wide.
///
/// Lanes are squeezed toward [`MAP_LANE_MIN_W`] as the fork count grows and
/// never past it: below that a dot is on top of its neighbour, which says less
/// than a map that runs off its own edge and can be scrolled.
pub(crate) fn map_lane_width(lanes: usize) -> f32 {
    if lanes <= 1 {
        return MAP_LANE_W;
    }
    let avail = (MAP_WIDTH - 2.0 * MAP_PAD - MAP_DOT).max(MAP_LANE_MIN_W);
    (avail / (lanes - 1) as f32).clamp(MAP_LANE_MIN_W, MAP_LANE_W)
}

/// Where a lane's dots sit, in a graph that wide.
///
/// **Lane index *is* the x position** — that is what makes ordering by lane the
/// same as reading the map left to right. Past thirteen lanes the floor above
/// carries the last of them beyond the column's own width, and the excess is
/// *scrolled to* rather than folded back: clamping it stacked distinct branches
/// — and their buttons — on one x, contradicting the topology the map exists to
/// show and putting some nodes out of reach.
pub(crate) fn map_lane_x(lane: usize, lanes: usize) -> f32 {
    lane as f32 * map_lane_width(lanes)
}

/// The scroll offset that brings `[start, start + extent)` inside a viewport of
/// `viewport`, moving as little as possible — one axis of the map's reveal.
///
/// `offset` is gpui's own sign convention (zero at the content's origin, going
/// negative as the content moves up/left under the viewport), so the visible
/// content span is `[-offset, -offset + viewport)`. Already-visible spans are
/// left exactly where the reader put them; the result is clamped into
/// `[-max, 0]`, so a viewport wider than its content never scrolls at all.
pub(crate) fn map_reveal_axis(
    start: f32,
    extent: f32,
    viewport: f32,
    offset: f32,
    max: f32,
) -> f32 {
    if viewport <= 0.0 {
        return offset;
    }
    let wanted = if start < -offset {
        -start
    } else if start + extent > -offset + viewport {
        viewport - (start + extent)
    } else {
        offset
    };
    wanted.clamp(-max.max(0.0), 0.0)
}

/// Every post's node id, resolved **once**.
///
/// Both halves of the overlay ask "which transcript row is this node?" — the
/// results for its attribution, the map for every dot's label — and each answer
/// used to be a linear scan that recomputed each candidate's id on the way
/// past. That is quadratic in the conversation and paid on every frame the
/// overlay draws, whether or not anything matched: a ten-thousand-post space
/// spends a hundred million comparisons just to label the map, outside the
/// virtualization that bounds the cards. One pass builds the index and every
/// lookup is a hash.
pub(crate) fn post_index(posts: &[super::model::PostData]) -> HashMap<SharedString, usize> {
    (0..posts.len())
        .map(|i| (super::model::node_id(posts, i), i))
        .collect()
}

/// The discriminator a fragment's measurement key carries for a node whose text
/// can move under a stable id — see [`ResultFragment::id`]. Zero for every
/// other node, whose id already changes with its text, so nothing is hashed for
/// the conversation at large.
fn content_stamp(result: &super::find::NodeResult<'_>) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    if !result.live_editor {
        return 0;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    result.content.as_ref().hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// The overlay's state
// ---------------------------------------------------------------------------

/// One result the list draws: a block run of one node, and the match a click on
/// it takes the reader to.
#[derive(Clone)]
pub(crate) struct ResultFragment {
    /// `{node}@{stamp}#{start}` — what a measured height and an editor state
    /// belong to.
    ///
    /// **The stamp is there because a node id is not always enough.** A post
    /// mints a new action id whenever its text changes, so its node id already
    /// discriminates; a **draft** and a **post being edited in place** do not —
    /// their text moves under a stable id, and a height measured against the
    /// old text then stood in for the new one as a placeholder, corrupting
    /// every group top below it (and with them the map's scroll targets and the
    /// list's extent) until the card happened back into the viewport band. The
    /// stamp is a hash of the node's own content, taken once per node and only
    /// for the nodes that can move — the ones the projection rendered
    /// cursor-aware, which is exactly the set of live editors.
    pub(crate) id: SharedString,
    pub(crate) node: SharedString,
    pub(crate) item_id: Option<SharedString>,
    /// Who wrote it — the group's attribution, carried so the card's accessible
    /// name can say it. The header beside it is a sibling `Label`, so a reader
    /// hearing the active descendant alone would otherwise get the snippet with
    /// no author and could not tell two similar results apart.
    pub(crate) byline: SharedString,
    /// The consolidated block span this fragment paints.
    pub(crate) range: Range<usize>,
    /// The node's own markdown — the document the fragment is a window onto.
    pub(crate) content: SharedString,
    /// **Only this fragment's hits**, in the node's own order. A node with many
    /// separated matching blocks produces many fragments, and giving each one
    /// the whole node's hit list materialized that vector per fragment — five
    /// thousand isolated matches is twenty-five million ranges built before a
    /// single card is virtualized, and every visible card then rebuilt its
    /// highlight set from the oversized copy. The painted result is the same
    /// set either way: the element lays out no line the others could land on.
    pub(crate) hits: Vec<Range<usize>>,
    /// The first hit inside this fragment, as its **ordinal within the node** —
    /// half of the anchor a click hands the bar.
    pub(crate) ordinal: usize,
}

/// One node's results, under its own attribution.
pub(crate) struct ResultGroup {
    pub(crate) node: SharedString,
    pub(crate) byline: SharedString,
    pub(crate) time: SharedString,
    /// The whole attribution folded onto one line — the group header's
    /// accessible name, exactly as a post's article label is.
    pub(crate) label: SharedString,
    pub(crate) fragments: Vec<ResultFragment>,
}

/// Everything the overlay holds between frames.
///
/// It lives **inside `FindSession`** for the same reason the projection cache
/// does: closing the bar drops it, so nothing here can outlive the search it is
/// about, and the lazy-activation invariant stays structural rather than
/// remembered.
pub(crate) struct FindOverlay {
    /// Whether the surface is expanded.
    pub(crate) open: bool,
    /// The results list's scroll position. **Retained across close and reopen**
    /// while the query stands — the task's own rule — and reset with the query,
    /// because a position in one search's results means nothing in another's.
    pub(crate) scroll: gpui::ScrollHandle,
    /// The map column's scroll position.
    pub(crate) map_scroll: gpui::ScrollHandle,
    /// The results list is **one tab stop** with a roving cursor, so the handle
    /// rides the element carrying the `List` role.
    pub(crate) list_focus: gpui::FocusHandle,
    /// The overlay's own handle — the landmark, and where the keyboard goes
    /// when there is no result to stand on.
    pub(crate) focus: gpui::FocusHandle,
    /// The roving cursor, as an index into the flat fragment list. Clamped on
    /// read, never chased on write: the results move under it whenever the
    /// conversation does.
    pub(crate) cursor: usize,
    /// Per-fragment measured heights, written by each rendered card's measuring
    /// canvas. `Rc` because that canvas outlives the borrow that built it.
    heights: Rc<RefCell<HashMap<SharedString, f32>>>,
    /// The editor state each rendered fragment paints through. Kept for every
    /// fragment the reader has scrolled past — re-minting one costs a parse and
    /// a `set_value` notify, and dropping them on the way out of the band would
    /// pay that on every scroll step.
    bodies: HashMap<SharedString, Entity<MarkdownEditorState>>,
    /// Each group's and each fragment's top offset inside the list, recorded
    /// as the list is laid out — keyed by node id for a group and by fragment
    /// id for a card, which cannot collide (a fragment's id carries a `#`).
    /// What a map press scrolls to, what the roving cursor follows, and what
    /// the scroll↔map sync reads.
    tops: HashMap<SharedString, (f32, f32)>,
    /// The nodes whose group is in the list's viewport this frame — the map's
    /// `aria_selected` set, derived rather than stored as state of its own.
    in_view: HashSet<SharedString>,
    /// One **tracked** focus handle per map dot that is a tab stop, keyed by
    /// the post it represents — the same key its element id takes, and for the
    /// same reason (a background write reshapes the tree, and a slot keyed by
    /// position would hand the reader's focus to another post's dot).
    ///
    /// It exists so the view can ask *which* dot the keyboard is on. A
    /// probe-derived stop rides gpui's **implicit** handle, which this crate
    /// never receives, and the focused element's bounds are crate-private — so
    /// there is no other way to know a dot is off-screen, which is exactly the
    /// gap the "Tab does not reveal an off-screen control" note names. The
    /// handle carries `tab_index(0)`, reproducing what `probe` derives, so the
    /// tab order is unchanged; only the *question* is newly answerable.
    map_slots: HashMap<SharedString, gpui::FocusHandle>,
}

impl FindOverlay {
    pub(crate) fn new(cx: &mut gpui::App) -> Self {
        Self {
            open: false,
            scroll: gpui::ScrollHandle::new(),
            map_scroll: gpui::ScrollHandle::new(),
            // **A real tab stop, like every other roving list in the app.**
            // `Role::List` is deliberately not in the focusable set `probe`
            // derives from, so the element carrying it takes focus only when
            // this handle says so: without it the list could be focused
            // explicitly (opening the overlay does) and then never again, so a
            // reader who tabbed onto a map node could not get back to the
            // results cursor without closing and reopening the surface. It
            // rides the find bar's own region, because the overlay is that
            // bar's surface rather than the conversation's.
            list_focus: cx
                .focus_handle()
                .tab_index(crate::focus::region::FIND)
                .tab_stop(true),
            focus: cx.focus_handle(),
            cursor: 0,
            heights: Rc::new(RefCell::new(HashMap::new())),
            bodies: HashMap::new(),
            tops: HashMap::new(),
            in_view: HashSet::new(),
            map_slots: HashMap::new(),
        }
    }

    /// Forget everything that was an answer about the **previous** query: the
    /// reader's place in a result list that no longer exists, the measured
    /// heights of fragments that are gone, and the editor states behind them.
    /// The retention rule is "while the query is unchanged", so this is where
    /// it ends.
    /// Drop every per-fragment cell whose fragment this frame no longer has.
    ///
    /// The heights and the editor states are keyed by fragment id, and a
    /// fragment cut from a live editor carries a stamp of its own content — so
    /// a reader typing in a draft supersedes that draft's cards on every
    /// keystroke. Without this the maps would grow one entry per edit and hold
    /// an editor entity for each until the query moved.
    pub(crate) fn retain_results(&mut self, live: &HashSet<SharedString>) {
        self.heights.borrow_mut().retain(|id, _| live.contains(id));
        self.bodies.retain(|id, _| live.contains(id));
    }

    pub(crate) fn forget_results(&mut self) {
        self.cursor = 0;
        self.scroll.set_offset(gpui::point(px(0.), px(0.)));
        self.heights.borrow_mut().clear();
        self.bodies.clear();
        self.tops.clear();
        self.in_view.clear();
        self.map_slots.clear();
    }

    /// Hand back the handle for a map dot that is a tab stop, minting one the
    /// first time, and drop the slots of every node this frame's map no longer
    /// has one for. Same shape as [`Self::retain_results`]: a slot map is only
    /// bounded by pruning it against what actually painted.
    fn map_slots_for(
        &mut self,
        live: &HashSet<SharedString>,
        cx: &mut gpui::App,
    ) -> &HashMap<SharedString, gpui::FocusHandle> {
        self.map_slots.retain(|id, _| live.contains(id));
        for id in live {
            self.map_slots
                .entry(id.clone())
                .or_insert_with(|| cx.focus_handle().tab_index(0).tab_stop(true));
        }
        &self.map_slots
    }

    /// Whether the overlay owns the keyboard — its list, or anything else
    /// inside it.
    pub(crate) fn holds_focus(&self, window: &Window, cx: &gpui::App) -> bool {
        self.list_focus.is_focused(window)
            || self.focus.is_focused(window)
            || self.focus.contains_focused(window, cx)
    }
}

// ---------------------------------------------------------------------------
// The view's half
// ---------------------------------------------------------------------------

impl SpaceView {
    /// Whether the Find-all overlay is expanded.
    pub(crate) fn find_overlay_open(&self) -> bool {
        self.find.as_ref().is_some_and(|s| s.overlay.open)
    }

    /// The disclosure's verb: expand the overlay, or collapse it again.
    #[doc(hidden)]
    pub fn toggle_find_overlay(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.find_overlay_open() {
            self.close_find_overlay(window, cx);
            return;
        }
        let Some(session) = self.find.as_mut() else {
            return;
        };
        session.overlay.open = true;
        // **A surface that takes the window takes the keyboard.** The results
        // list is the single tab stop inside it, so that is where a reader
        // lands; with nothing to list, the overlay itself holds the handle,
        // because focusing a list that is not painted is the dead slot this
        // window's focus doctrine is built around.
        let handle = session.overlay.list_focus.clone();
        window.focus(&handle, cx);
        cx.notify();
    }

    /// Collapse the overlay, handing the keyboard back to the bar's own field.
    /// Returns whether there was one open — the Escape rung's answer.
    ///
    /// **The unmount owes the handback** (`RecordView::close_detail`'s rule):
    /// the results list and the map's nodes are real focus destinations, and
    /// they all go at once, so a reader who was standing in one would be left
    /// on a handle nobody paints. Only from an overlay that is actually holding
    /// the keyboard — a pointer press on the disclosure takes nothing from a
    /// reader composing elsewhere.
    #[doc(hidden)]
    pub fn close_find_overlay(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let Some(session) = self.find.as_mut() else {
            return false;
        };
        if !session.overlay.open {
            return false;
        }
        session.overlay.open = false;
        let held = session.overlay.holds_focus(window, cx);
        let field = session.input.clone();
        if held {
            field.update(cx, |s, cx| s.focus(window, cx));
        }
        cx.notify();
        true
    }

    /// The map's membership rule, and therefore the results': every post, plus
    /// a draft the reader has actually written in.
    ///
    /// A streaming leaf is out on the same rule that keeps it out of the match
    /// set — its body grows token by token — and an *empty* draft is not
    /// something the conversation contains: it is the composer the tree mints
    /// at every leaf, and drawing fifty of them would bury the shape the map
    /// exists to show. Both are leaves, so leaving them out changes nothing
    /// about the topology of what remains.
    pub(crate) fn find_map_includes_for_test(&self, node: &TreeNode, cx: &gpui::App) -> bool {
        self.find_map_includes(node, cx)
    }

    /// The results for a map this caller already laid out — the test seam's
    /// half of [`Self::find_results`].
    pub(crate) fn find_results_for_map_for_test(
        &self,
        map: &[MapNode],
        cx: &gpui::App,
    ) -> Vec<ResultGroup> {
        self.find_results(map, &post_index(&self.posts), cx)
    }

    fn find_map_includes(&self, node: &TreeNode, cx: &gpui::App) -> bool {
        match node.src {
            NodeSrc::Msg(_) => true,
            NodeSrc::Streaming(_) => false,
            NodeSrc::Draft => self
                .drafts
                .iter()
                .find(|d| d.id == node.id)
                .is_some_and(|d| !d.editor.read(cx).value().is_empty()),
        }
    }

    /// This frame's results, in **depth-then-lane** order.
    ///
    /// Built from the projection cache and nothing else, so a group can only
    /// ever show what the bar counted. A node whose entry has not answered for
    /// this query yet — the pass is still walking, or a composing draft was
    /// deliberately left unprojected — is simply not a group; the list says it
    /// is still counting rather than presenting a partial set as a whole one.
    fn find_results(
        &self,
        map: &[MapNode],
        posts: &HashMap<SharedString, usize>,
        cx: &gpui::App,
    ) -> Vec<ResultGroup> {
        let Some(session) = self.find.as_ref() else {
            return Vec::new();
        };
        let mut groups = Vec::new();
        for entry in map {
            let post = posts.get(&entry.node).map(|i| &self.posts[*i]);
            // **A post being regenerated is out here for the same reason it is
            // out of the branch's matches and out of the count.** The answer on
            // screen is the pending revision, not the generation the cache
            // still holds a projection of — and `CountKey` carries the revising
            // set precisely because that exclusion moves with no rebuild behind
            // it, so the memo survives while the total drops the post. Read
            // through the memo alone, the overlay went on offering a result the
            // total excluded and the conversation no longer showed.
            if post.is_some_and(|p| {
                p.action_id
                    .as_deref()
                    .is_some_and(|id| self.space.read(cx).revising_seq(id).is_some())
            }) {
                continue;
            }
            let Some(result) = session.node_result(&entry.node) else {
                continue;
            };
            let runs = fragment_runs(result.blocks, result.hits);
            if runs.is_empty() {
                continue;
            }
            let (byline, time, backend) = match post {
                Some(p) => (p.byline.clone(), p.time.clone(), p.byline_backend.clone()),
                // A draft is the reader's own unposted words; it has no byline
                // row of its own, so the overlay names it for what it is.
                None => (
                    crate::i18n::msg::find_result_draft(cx),
                    SharedString::default(),
                    None,
                ),
            };
            let item_id = post.and_then(|p| p.item_id.clone());
            let stamp = content_stamp(&result);
            let fragments = runs
                .into_iter()
                .map(|run| ResultFragment {
                    id: SharedString::from(format!("{}@{stamp:x}#{}", entry.node, run.range.start)),
                    node: entry.node.clone(),
                    item_id: item_id.clone(),
                    byline: byline.clone(),
                    ordinal: run.hits.start,
                    hits: result.hits[run.hits].to_vec(),
                    range: run.range,
                    content: result.content.clone(),
                })
                .collect();
            groups.push(ResultGroup {
                label: super::post::article_label(&byline, backend.as_deref(), &time),
                node: entry.node.clone(),
                byline,
                time,
                fragments,
            });
        }
        groups
    }

    /// **A pointer press opens a result only when nothing is selected in it.**
    ///
    /// A card's editor is read-only *and selectable* — the read-only editor's
    /// own contract, and the I-beam over it says so — so a drag inside one ends
    /// with the pointer released over the card, and the ancestor's click read
    /// that as "open this result": the overlay collapsed and took the passage
    /// away before it could be copied. A plain click has already collapsed the
    /// selection by the time it lands (the press places the caret), so
    /// click-to-navigate is untouched; a **double-click** selects a word and
    /// therefore does not navigate, deliberately — the gesture asked for the
    /// word, and a card whose meaning changed with the click count would be the
    /// worse surprise.
    ///
    /// The **keyboard** path deliberately does not ask: Enter on the roving
    /// cursor is an unambiguous request to open, whatever some card happens to
    /// have selected.
    pub(crate) fn click_find_result(
        &mut self,
        fragment: ResultFragment,
        editor: &Entity<MarkdownEditorState>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !editor.read(cx).selection().is_collapsed() {
            return;
        }
        self.open_find_result(fragment, window, cx);
    }

    /// Press the `index`-th result card the way a pointer would, guard and all
    /// — the seam a test uses, because a card's painted bounds depend on where
    /// the list has been scrolled.
    #[doc(hidden)]
    pub fn press_find_result_for_test(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        let map = map_layout(&tree, &|node| self.find_map_includes(node, cx));
        let Some(fragment) = self
            .find_results(&map, &post_index(&self.posts), cx)
            .into_iter()
            .flat_map(|g| g.fragments)
            .nth(index)
        else {
            return;
        };
        let Some(editor) = self
            .find
            .as_ref()
            .and_then(|s| s.overlay.bodies.get(&fragment.id).cloned())
        else {
            return;
        };
        self.click_find_result(fragment, &editor, window, cx);
    }

    /// Take the reader to the `index`-th result the overlay is showing — the
    /// keyboard's own path (Enter on the roving cursor), reached by index
    /// because a test cannot press a card whose bounds depend on where the list
    /// has been scrolled.
    #[doc(hidden)]
    pub fn open_find_result_for_test(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        let map = map_layout(&tree, &|node| self.find_map_includes(node, cx));
        let posts = post_index(&self.posts);
        let Some(fragment) = self
            .find_results(&map, &posts, cx)
            .into_iter()
            .flat_map(|g| g.fragments)
            .nth(index)
        else {
            return;
        };
        self.open_find_result(fragment, window, cx);
    }

    /// The byline the map speaks for a node — the post's own, or the word for a
    /// draft, which has no byline row of its own.
    fn find_map_byline(
        &self,
        node: &SharedString,
        posts: &HashMap<SharedString, usize>,
        cx: &gpui::App,
    ) -> SharedString {
        posts
            .get(node)
            .map(|i| self.posts[*i].byline.clone())
            .unwrap_or_else(|| crate::i18n::msg::find_result_draft(cx))
    }

    /// The roving cursor, clamped into this frame's results — `None` when there
    /// is nothing to point at. Derived on read, the Library's rule: the results
    /// move under it whenever a background write lands, and a stored index one
    /// past the end is a dead Enter and a ring nobody draws.
    fn find_result_cursor(&self, total: usize) -> Option<usize> {
        let session = self.find.as_ref()?;
        total
            .checked_sub(1)
            .map(|last| session.overlay.cursor.min(last))
    }

    /// The cursor **as a card renders it** — `None` unless the list itself
    /// holds the keyboard. The cursor is the row's focus identity, so a row may
    /// only claim it while the list is the focused element; otherwise a reader
    /// who has tabbed away sees two focus indications for one focus.
    fn find_result_cursor_row(&self, total: usize, window: &Window) -> Option<usize> {
        let session = self.find.as_ref()?;
        session
            .overlay
            .list_focus
            .is_focused(window)
            .then(|| self.find_result_cursor(total))
            .flatten()
    }

    /// Take the reader to a result: collapse the overlay, select the branch the
    /// fragment lives on, and hand the reveal to the bar.
    ///
    /// **This is the one place find spends branch selection**, and it does so
    /// because the reader asked for this result by name. Everything after the
    /// `select_path_to` is the ordinary machinery: the anchor is set by
    /// identity (item where there is one, node id for a draft — the same key
    /// `MatchAnchor` uses everywhere), and `reveal_when_anchored` records the
    /// debt that `sync_find` discharges once the new branch's match list exists.
    /// Nothing here reveals anything itself, which is what keeps the two-phase
    /// reveal the only reveal.
    pub(crate) fn open_find_result(
        &mut self,
        fragment: ResultFragment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.close_find_overlay(window, cx);
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        if super::model::node_ref(&tree, &fragment.node).is_some() {
            self.select_path_to(&tree, &fragment.node, page_width);
        }
        if let Some(session) = self.find.as_mut() {
            session.anchor = Some(MatchAnchor {
                key: fragment
                    .item_id
                    .clone()
                    .unwrap_or_else(|| fragment.node.clone()),
                ordinal: fragment.ordinal,
            });
            session.pending_reveal = None;
            session.reveal_when_anchored = true;
        }
        cx.notify();
    }

    /// The results list's roving key map: ↑/↓ move, Home/End take its ends,
    /// Enter opens the result the cursor sits on.
    ///
    /// **Escape is deliberately not among them** — it means dismiss, and it is
    /// the overlay's own rung of the space root's Escape chain. A cursor that
    /// consumed it would shadow the only way out.
    fn handle_find_results_key(
        &mut self,
        fragments: &[ResultFragment],
        viewport_h: f32,
        ev: &gpui::KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let holds = self
            .find
            .as_ref()
            .is_some_and(|s| s.overlay.list_focus.is_focused(window));
        if !holds || ev.keystroke.modifiers.modified() {
            return false;
        }
        let (Some(last), Some(cursor)) = (
            fragments.len().checked_sub(1),
            self.find_result_cursor(fragments.len()),
        ) else {
            return false;
        };
        let target = match ev.keystroke.key.as_str() {
            "up" => cursor.saturating_sub(1),
            "down" => (cursor + 1).min(last),
            "home" => 0,
            "end" => last,
            "enter" => {
                self.open_find_result(fragments[cursor].clone(), window, cx);
                return true;
            }
            _ => return false,
        };
        self.move_find_result_cursor(target, fragments, viewport_h, cx);
        true
    }

    /// Move the cursor and bring what it lands on into view — the move that
    /// makes one tab stop equivalent to a stop per card, since a fragment
    /// outside the viewport band is a placeholder with nothing to read and
    /// nothing a screen reader could report.
    ///
    /// The offsets are the *last* frame's, which is the lag every estimate in
    /// this list already carries: a card that has not painted is placed from
    /// its estimate and corrected on the frame it does.
    fn move_find_result_cursor(
        &mut self,
        idx: usize,
        fragments: &[ResultFragment],
        viewport_h: f32,
        cx: &mut Context<Self>,
    ) {
        let placed = self.find.as_mut().and_then(|session| {
            session.overlay.cursor = idx;
            fragments
                .get(idx)
                .and_then(|f| session.overlay.tops.get(&f.id).copied())
                .map(|placed| (placed, -session.overlay.scroll.offset().y.as_f32()))
        });
        // Minimal motion, the reveal's own rule: a card already in view is left
        // where the reader put it.
        if let Some(((top, height), at)) = placed {
            let next = if top < at {
                Some(top)
            } else if top + height > at + viewport_h {
                Some(top + height - viewport_h)
            } else {
                None
            };
            if let Some(next) = next {
                self.scroll_find_results_to(next);
            }
        }
        cx.notify();
    }

    /// What a press on a map node does: take the results list to that node's
    /// group, and **take the roving cursor with it**.
    ///
    /// Scrolling alone left the cursor on whatever fragment it was on, so the
    /// list's active descendant then named a card outside the viewport band — a
    /// sized placeholder with nothing for assistive technology to read — and
    /// the reader's next arrow scrolled the viewport back toward it, undoing
    /// the press they had just made. The group's **first** fragment is exactly
    /// where the cursor's own reveal scrolls to (`nth == 0` reveals from the
    /// group's top), so the two agree by construction rather than by a second
    /// scroll here.
    pub(crate) fn reveal_find_group(
        &mut self,
        node: &SharedString,
        first_fragment: Option<usize>,
        cx: &mut Context<Self>,
    ) {
        let Some((top, _)) = self
            .find
            .as_ref()
            .and_then(|s| s.overlay.tops.get(node).copied())
        else {
            return;
        };
        self.scroll_find_results_to(top);
        if let (Some(index), Some(session)) = (first_fragment, self.find.as_mut()) {
            session.overlay.cursor = index;
        }
        cx.notify();
    }

    /// Press the map's `index`-th node the way a pointer would — the seam a
    /// test uses, because a dot's painted bounds depend on where the map has
    /// been scrolled.
    #[doc(hidden)]
    pub fn press_find_map_node_for_test(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        let map = map_layout(&tree, &|node| self.find_map_includes(node, cx));
        let Some(node) = map.get(index).map(|n| n.node.clone()) else {
            return;
        };
        let groups = self.find_results(&map, &post_index(&self.posts), cx);
        let mut at = 0usize;
        let mut first = None;
        for group in &groups {
            if group.node == node {
                first = Some(at);
                break;
            }
            at += group.fragments.len();
        }
        self.reveal_find_group(&node, first, cx);
    }

    /// Put the keyboard on the *i*-th map dot, as Tab would — the seam a test
    /// needs, because a probe-derived stop rides gpui's implicit handle and a
    /// test cannot name one. Answers whether that dot is a stop at all (only
    /// the nodes with matches are). The slot map is filled during the map's
    /// render, so this is asked after a frame.
    #[doc(hidden)]
    pub fn focus_find_map_node_for_test(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        let map = map_layout(&tree, &|node| self.find_map_includes(node, cx));
        let Some(node) = map.get(index).map(|n| n.node.clone()) else {
            return false;
        };
        let Some(handle) = self
            .find
            .as_ref()
            .and_then(|s| s.overlay.map_slots.get(&node))
            .cloned()
        else {
            return false;
        };
        window.focus(&handle, cx);
        true
    }

    /// How many nodes the map draws, in depth-then-lane order.
    #[doc(hidden)]
    pub fn find_map_nodes_for_test(&mut self, window: &Window, cx: &mut Context<Self>) -> usize {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        map_layout(&tree, &|node| self.find_map_includes(node, cx)).len()
    }

    /// The map column's scroll offset — what a reveal moves.
    #[doc(hidden)]
    pub fn find_map_scroll_for_test(&self) -> (f32, f32) {
        self.find
            .as_ref()
            .map(|s| {
                let o = s.overlay.map_scroll.offset();
                (o.x.as_f32(), o.y.as_f32())
            })
            .unwrap_or((0.0, 0.0))
    }

    /// The editor one result card paints through — the seam a test needs to put
    /// a selection in a card the way a drag would.
    #[doc(hidden)]
    pub fn find_result_editor_for_test(
        &self,
        index: usize,
        window: &Window,
        cx: &gpui::App,
    ) -> Option<Entity<MarkdownEditorState>> {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        let map = map_layout(&tree, &|node| self.find_map_includes(node, cx));
        let id = self
            .find_results(&map, &post_index(&self.posts), cx)
            .into_iter()
            .flat_map(|g| g.fragments)
            .nth(index)?
            .id;
        self.find.as_ref()?.overlay.bodies.get(&id).cloned()
    }

    /// Scroll the results list so `top` is at the top of its viewport — what a
    /// map press does, and what following the cursor does.
    pub(crate) fn scroll_find_results_to(&mut self, top: f32) {
        let Some(session) = self.find.as_mut() else {
            return;
        };
        session
            .overlay
            .scroll
            .set_offset(gpui::point(px(0.), px(-top.max(0.0))));
    }

    /// The overlay itself.
    ///
    /// Painted after everything it covers — the composer, the notices, the
    /// minimap — and geometrically clear of the drag band and the find bar
    /// above it, so window dragging and the bar's own controls are untouched.
    /// `Overlay::Scrolling`: it owns its scrolling, and its own handler is what
    /// keeps a wheel gesture over its padding from reaching the conversation
    /// behind it.
    pub(crate) fn render_find_overlay(
        &mut self,
        tree: &[TreeNode],
        window_h: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        if !self.find_overlay_open() {
            return None;
        }
        // The surface doing the covering builds its own controls in the clear
        // ([`crate::focus::Covered`]) — the map's dots and the results list are
        // the tab stops a reader is meant to reach while it stands.
        let _uncovered = crate::focus::Covered::new(false);
        let map = map_layout(tree, &|node| self.find_map_includes(node, cx));
        let posts = post_index(&self.posts);
        let groups = self.find_results(&map, &posts, cx);
        let settled = self
            .find
            .as_ref()
            .is_some_and(|s| s.space.is_settled() && s.space.total.is_some());

        let top = TITLE_BAR_RESERVE.as_f32() + FIND_BAR_H;
        let viewport_h = (window_h.as_f32() - top).max(0.0);
        // **The overlay's ground is the separators' muted paper, not the page's.**
        // A fragment is a sheet lifted out of the conversation, and a sheet on a
        // sheet is invisible: the cards below take `theme.background`, so the
        // surface under them has to be the band's ground for the lift to read at
        // all. Same inversion the compact composer's action bar takes.
        let (bg, border) = {
            let theme = cx.theme();
            (theme.muted, theme.border)
        };

        let list = self.render_find_results(&groups, viewport_h, settled, window, cx);
        let map_column = self.render_find_map(&map, &groups, &posts, window, cx);

        let focus = self.find.as_ref().expect("checked").overlay.focus.clone();
        Some(
            div()
                .id("space-find-overlay")
                .track_focus(&focus)
                // A named landmark: the reader opened a surface over the whole
                // window, and it is the one thing in it.
                .probe(
                    "space/find/overlay",
                    gpui::Role::Region,
                    crate::i18n::msg::find_overlay_label(cx),
                )
                .absolute()
                .top(px(top))
                .left_0()
                .right_0()
                .bottom_0()
                .bg(bg)
                .border_t_1()
                .border_color(border)
                .contain_mouse(Overlay::Scrolling)
                // The surface's own wheel policy: nothing reaches the
                // conversation behind it. Inner scrollers run first (dispatch
                // bubbles inner→outer), so this stops only the gestures that
                // landed on the overlay's own padding.
                .on_scroll_wheel(|_, _, cx| cx.stop_propagation())
                .flex()
                .flex_row()
                .items_stretch()
                .child(map_column)
                .child(list)
                .into_any_element(),
        )
    }

    /// The topological map.
    fn render_find_map(
        &mut self,
        map: &[MapNode],
        groups: &[ResultGroup],
        posts: &HashMap<SharedString, usize>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let (muted, border, wash, accent) = {
            let style = prose_style(cx);
            let theme = cx.theme();
            (
                theme.muted_foreground,
                theme.border,
                style.highlight_overlay_color,
                style.highlight_accent_color,
            )
        };
        let with_matches: HashSet<SharedString> = groups.iter().map(|g| g.node.clone()).collect();
        // Where each group's **first** fragment sits in the flat result list —
        // what a press on that node's dot moves the roving cursor to.
        let first_fragment: HashMap<SharedString, usize> = {
            let mut at = 0usize;
            groups
                .iter()
                .map(|g| {
                    let here = at;
                    at += g.fragments.len();
                    (g.node.clone(), here)
                })
                .collect()
        };
        let in_view = self
            .find
            .as_ref()
            .map(|s| s.overlay.in_view.clone())
            .unwrap_or_default();
        // **The current match's post is distinguished**, in the same colour the
        // conversation gives that match's own wash — so the map says where the
        // reader *is* as well as what the space holds. It is always a node of
        // the visible branch, because find never leaves it: the anchor is
        // resolved against the branch's own match list.
        let current = self
            .find
            .as_ref()
            .and_then(|s| s.current())
            .map(|m| m.node.clone());
        let lanes = map.iter().map(|n| n.lane + 1).max().unwrap_or(1);
        let depths = map.iter().map(|n| n.depth + 1).max().unwrap_or(1);
        let x = |lane: usize| map_lane_x(lane, lanes);
        let y = |depth: usize| depth as f32 * MAP_ROW_H;

        // **A dot the keyboard reaches is a dot the reader can see.** Only the
        // nodes with matches are tab stops, and the column clips on both axes —
        // a deep conversation carries dots below the fold, and past thirteen
        // lanes `map_lane_x` states an x beyond the column's own width rather
        // than folding it back — so Tab could land on a dot with no visible
        // ring and Enter then scroll the list to a group chosen out of sight.
        // Revealed the way the results cursor reveals its card: minimally, so a
        // dot already in view is left exactly where the reader put it.
        //
        // Deliberately **not** a roving cursor like the list beside it. That
        // idiom exists because a virtualized list cannot have a stop per row;
        // every dot here paints, and the map is a *graph* — the minimap, this
        // app's other topology map, likewise gives each node its own stop, and
        // collapsing them onto one would make the map a second linear walk of
        // the sequence the list already is.
        let handle = self
            .find
            .as_ref()
            .expect("checked")
            .overlay
            .map_scroll
            .clone();
        let slots = {
            let overlay = &mut self.find.as_mut().expect("checked").overlay;
            overlay.map_slots_for(&with_matches, cx).clone()
        };
        if let Some(node) = map
            .iter()
            .find(|n| slots.get(&n.node).is_some_and(|h| h.is_focused(window)))
        {
            // The viewport is the scroller's own painted box less its padding;
            // both come from the last paint, which is what "where the reader is
            // looking" means. Before the first one there is nothing to reveal
            // into, and `max_offset` is zero, so the clamp answers "stay".
            let view = handle.bounds().size;
            let (offset, max) = (handle.offset(), handle.max_offset());
            let target = gpui::point(
                px(map_reveal_axis(
                    x(node.lane),
                    MAP_DOT,
                    view.width.as_f32() - 2.0 * MAP_PAD,
                    offset.x.as_f32(),
                    max.x.as_f32(),
                )),
                px(map_reveal_axis(
                    y(node.depth),
                    MAP_DOT,
                    view.height.as_f32() - 2.0 * MAP_PAD,
                    offset.y.as_f32(),
                    max.y.as_f32(),
                )),
            );
            if target != offset {
                handle.set_offset(target);
            }
        }

        // **Excess lanes run off the edge and are scrolled to, never folded
        // onto one x.** Past thirteen lanes the minimum stride carries the last
        // of them beyond the column's own width, and clamping there stacked
        // distinct branches — and their buttons — on top of one another at the
        // right edge: a map contradicting the topology it exists to show, with
        // some nodes unreachable. Lane index *is* the x position, so the canvas
        // states its real width and the column scrolls horizontally.
        let mut canvas = div()
            .relative()
            .w(px(
                (x(lanes.saturating_sub(1)) + MAP_DOT).max(MAP_WIDTH - 2.0 * MAP_PAD)
            ))
            .flex_none()
            .h(px(y(depths.saturating_sub(1)) + MAP_DOT + MAP_ROW_H));

        // Edges first, so a dot always sits on top of the line into it.
        for node in map {
            let Some((pd, pl)) = node.parent else {
                continue;
            };
            let cx0 = x(pl) + MAP_DOT / 2.0;
            let cx1 = x(node.lane) + MAP_DOT / 2.0;
            let cy0 = y(pd) + MAP_DOT / 2.0;
            let cy1 = y(node.depth) + MAP_DOT / 2.0;
            if (cx1 - cx0).abs() > 0.5 {
                canvas = canvas.child(
                    div()
                        .absolute()
                        .top(px(cy0 - 0.5))
                        .left(px(cx0.min(cx1)))
                        .w(px((cx1 - cx0).abs()))
                        .h(px(1.))
                        .bg(border),
                );
            }
            canvas = canvas.child(
                div()
                    .absolute()
                    .top(px(cy0))
                    .left(px(cx1 - 0.5))
                    .w(px(1.))
                    .h(px((cy1 - cy0).max(0.0)))
                    .bg(border),
            );
        }

        for (i, node) in map.iter().enumerate() {
            let has = with_matches.contains(&node.node);
            let showing = in_view.contains(&node.node);
            let is_current = current.as_ref() == Some(&node.node);
            let byline = self.find_map_byline(&node.node, posts, cx);
            // **The role tracks whether a handler attaches** — the bar's own
            // step-arrow rule. A node with matches scrolls the list to its
            // group, so it is a `Button`; one without has nothing to do, and a
            // `Button` with no listener is a control VoiceOver offers,
            // activates, and silently does nothing with.
            let (role, label) = if has {
                (
                    gpui::Role::Button,
                    crate::i18n::msg::find_map_node_matches(cx, byline.to_string()),
                )
            } else {
                (
                    gpui::Role::Label,
                    crate::i18n::msg::find_map_node(cx, byline.to_string()),
                )
            };
            let target = node.node.clone();
            let mut dot = div()
                // **Keyed by the post it represents, not by where it sits.** A
                // dot is a real tab stop, and the sort is depth-then-lane over
                // a tree a background write can reshape — so an id keyed by
                // index leaves the reader's focus on position *i* while the
                // label and the click target under it become another post's.
                // The probe name stays positional: it is the driver's selector
                // for "the i-th dot", which is what a test presses.
                .id(SharedString::from(format!("space-find-map-{}", node.node)))
                .probe(format!("space/find/map/{i}"), role, label)
                .aria_selected(showing)
                .absolute()
                .top(px(y(node.depth)))
                .left(px(x(node.lane)))
                .w(px(MAP_DOT))
                .h(px(MAP_DOT))
                .rounded_full()
                .bg(match (is_current, has) {
                    (true, _) => accent,
                    (false, true) => wash,
                    (false, false) => muted.opacity(0.35),
                });
            if showing {
                dot = dot.border_1().border_color(accent);
            }
            if has {
                let cursor_to = first_fragment.get(&node.node).copied();
                dot = dot
                    // The view's own handle for this dot, so a reveal can ask
                    // where the keyboard is. `probe` has already made the
                    // element a stop at index 0; gpui reads a *tracked*
                    // handle's own flags instead, and the handle carries the
                    // same pair, so the tab order does not move.
                    .track_focus(&slots[&node.node])
                    .cursor_pointer()
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.reveal_find_group(&target, cursor_to, cx);
                    }));
            } else {
                dot = dot.tab_stop(false);
            }
            canvas = canvas.child(dot);
        }

        // The scroller and its indicator are **siblings inside a `relative`
        // ancestor** — the house rule, because an overlay painted as a child of
        // the scrolling element scrolls away with the content. Floating rather
        // than window-edge: this column is a bounded mid-window surface, so the
        // CSD corner clearance would inset it wrongly.
        div()
            .relative()
            .flex_none()
            .w(px(MAP_WIDTH))
            .h_full()
            .border_r_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .id("space-find-map")
                    // A `Group` of the conversation's nodes — a table of
                    // contents for the list beside it, not a way into the page
                    // behind it.
                    .probe(
                        "space/find/map",
                        gpui::Role::Group,
                        crate::i18n::msg::find_map_label(cx),
                    )
                    .size_full()
                    .p(px(MAP_PAD))
                    .overflow_y_scroll()
                    .overflow_x_scroll()
                    .track_scroll(&handle)
                    .child(canvas),
            )
            .child(crate::scrollbar::vertical_floating(
                "space-find-map-scroll",
                &handle,
            ))
            .into_any_element()
    }

    /// The grouped results list.
    fn render_find_results(
        &mut self,
        groups: &[ResultGroup],
        viewport_h: f32,
        settled: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let fragments: Vec<ResultFragment> = groups
            .iter()
            .flat_map(|g| g.fragments.iter().cloned())
            .collect();
        let total = fragments.len();
        // **Forget what belongs to a fragment that no longer exists.** A
        // measurement and an editor state are keyed by fragment id, and a live
        // editor's id carries its content, so every keystroke in a draft
        // supersedes its own cards. Retaining to this frame's ids is what keeps
        // that from accumulating a state per edit for the length of a search.
        if let Some(session) = self.find.as_mut() {
            let live: HashSet<SharedString> = fragments.iter().map(|f| f.id.clone()).collect();
            session.overlay.retain_results(&live);
        }
        let cursor = self.find_result_cursor_row(total, window);
        let keyboard = window.last_input_was_keyboard();
        let (muted, border, card) = {
            let theme = cx.theme();
            (theme.muted_foreground, theme.border, theme.background)
        };

        let scroll = self.find.as_ref().expect("checked").overlay.scroll.clone();
        let heights = self.find.as_ref().expect("checked").overlay.heights.clone();
        let list_focus = self
            .find
            .as_ref()
            .expect("checked")
            .overlay
            .list_focus
            .clone();
        let offset = -scroll.offset().y.as_f32();
        let band = (offset - RESULT_MARGIN)..(offset + viewport_h + RESULT_MARGIN);

        // The reading measure holds here too: a fragment is prose, and prose
        // set across a thousand pixels is not readable because it is a result.
        let mut column = v_flex()
            .w_full()
            .max_w(super::BODY_MAX_WIDTH + px(2.0 * RESULTS_PAD))
            .gap_0();
        let mut y = 0.0_f32;
        let mut tops: HashMap<SharedString, (f32, f32)> = HashMap::new();
        let mut in_view: HashSet<SharedString> = HashSet::new();
        let mut index = 0usize;

        if !settled {
            // The honest in-progress state, in the same words the bar uses: a
            // list that is still filling must not read as a whole one.
            let sentence = crate::i18n::msg::find_total_counting(cx);
            column = column.child(
                div()
                    .id("space-find-overlay-counting")
                    .probe_value(
                        "space/find/overlay/counting",
                        gpui::Role::Label,
                        sentence.clone(),
                        sentence.clone(),
                    )
                    .px_4()
                    .py_2()
                    .text_sm()
                    .text_color(muted)
                    .child(sentence),
            );
            y += GROUP_HEADER_H;
        }

        if groups.is_empty() && settled {
            let sentence = crate::i18n::msg::find_overlay_empty(cx);
            column = column.child(
                div()
                    .id("space-find-overlay-empty")
                    .probe_value(
                        "space/find/overlay/empty",
                        gpui::Role::Label,
                        sentence.clone(),
                        sentence.clone(),
                    )
                    .px_4()
                    .py_4()
                    .text_sm()
                    .text_color(muted)
                    .child(sentence),
            );
        }

        for group in groups {
            let group_top = y;
            column = column.child(
                h_flex()
                    .id(SharedString::from(format!(
                        "space-find-group-{}",
                        group.node
                    )))
                    .probe(
                        format!("space/find/result-group/{}", index),
                        gpui::Role::Label,
                        group.label.clone(),
                    )
                    .h(px(GROUP_HEADER_H))
                    .px_4()
                    .gap_2()
                    .items_center()
                    .text_sm()
                    .child(
                        div()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .child(group.byline.clone()),
                    )
                    .child(div().text_xs().text_color(muted).child(group.time.clone())),
            );
            y += GROUP_HEADER_H;

            for (nth, fragment) in group.fragments.iter().enumerate() {
                let measured = heights.borrow().get(&fragment.id).copied();
                let height = measured.unwrap_or_else(|| self.find_fragment_estimate(fragment));
                let visible = y < band.end && (y + height) > band.start;
                let on_cursor = cursor == Some(index);
                if visible {
                    column = column.child(self.render_find_fragment(
                        fragment,
                        index,
                        total,
                        on_cursor,
                        keyboard,
                        card,
                        border,
                        heights.clone(),
                        window,
                        cx,
                    ));
                } else {
                    column = column.child(
                        div()
                            .id(SharedString::from(format!(
                                "space-find-frag-slot-{}",
                                fragment.id
                            )))
                            .w_full()
                            .h(px(height)),
                    );
                }
                // **A card's reveal starts at its attribution where it has
                // one.** The cursor's whole job is to make one tab stop read
                // like a stop per card, and a card scrolled to its own top with
                // the byline just above the fold is a result whose author the
                // reader cannot see. The first fragment of a group therefore
                // reveals from the group's top; the rest reveal from their own.
                let reveal_top = if nth == 0 { group_top } else { y };
                tops.insert(fragment.id.clone(), (reveal_top, y + height - reveal_top));
                y += height;
                index += 1;
            }
            let group_h = y - group_top;
            tops.insert(group.node.clone(), (group_top, group_h));
            if group_top < offset + viewport_h && group_top + group_h > offset {
                in_view.insert(group.node.clone());
            }
        }

        if let Some(session) = self.find.as_mut() {
            session.overlay.tops = tops;
            session.overlay.in_view = in_view;
        }

        div()
            .relative()
            .flex_1()
            .min_w_0()
            .h_full()
            .child(
                div()
                    .id("space-find-results")
                    // One tab stop with a roving cursor: a card per stop would
                    // describe a tab order that does not contain the results
                    // nobody scrolled to.
                    .probe(
                        "space/find/results",
                        gpui::Role::List,
                        crate::i18n::msg::find_results_label(cx),
                    )
                    .track_focus(&list_focus)
                    .on_key_down(
                        cx.listener(move |this, ev: &gpui::KeyDownEvent, window, cx| {
                            if this.handle_find_results_key(&fragments, viewport_h, ev, window, cx)
                            {
                                cx.stop_propagation();
                            }
                        }),
                    )
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&scroll)
                    .child(column),
            )
            .child(crate::scrollbar::vertical_floating(
                "space-find-results-scroll",
                &scroll,
            ))
            .into_any_element()
    }

    /// The height a fragment that has never painted is assumed to take.
    ///
    /// The transcript's own estimate over the fragment's source, plus the
    /// card's chrome — honest in the same direction, and replaced by the real
    /// measurement the frame after the card first paints.
    fn find_fragment_estimate(&self, fragment: &ResultFragment) -> f32 {
        let text = fragment
            .content
            .get(fragment.range.clone())
            .unwrap_or_default();
        super::layout::estimate_post_height(
            text,
            (super::BODY_MAX_WIDTH.as_f32() - 2.0 * super::POST_PAD_Y.as_f32()).max(1.0),
            super::PROSE_FONT_SIZE.as_f32(),
            super::PROSE_LINE_HEIGHT,
            super::PROSE_PARAGRAPH_GAP,
            0.0,
        ) + FRAGMENT_CHROME_H
            + FRAGMENT_GAP
    }

    /// One fragment card: the matched block still inside its own containers,
    /// with the match washed exactly as it is in the conversation.
    ///
    /// It is the **document's** own render, filtered
    /// ([`MarkdownEditor::fragment`]) rather than re-parsed standalone: the
    /// standalone path carries no source ranges, so a fragment through it would
    /// paint no highlight at all — a results list whose matches are not
    /// highlighted defeats the point of the list.
    #[allow(clippy::too_many_arguments)]
    fn render_find_fragment(
        &mut self,
        fragment: &ResultFragment,
        index: usize,
        total: usize,
        on_cursor: bool,
        keyboard: bool,
        card: gpui::Hsla,
        border: gpui::Hsla,
        heights: Rc<RefCell<HashMap<SharedString, f32>>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let editor = self.find_fragment_editor(fragment, window, cx);
        let style = prose_style(cx);
        let opener = fragment.clone();
        let label = self.find_fragment_label(fragment, cx);
        // **A selection is not a click.** A card's editor is read-only *and
        // selectable* — that is the whole read-only editor contract, and the
        // I-beam over it says so — so a drag inside one ends with the pointer
        // released over the card, which the ancestor handler read as "open this
        // result": the overlay collapsed and took the passage away before it
        // could be copied. The editor's own selection is the discriminator, and
        // a plain click has already collapsed it (a press places the caret), so
        // click-to-navigate is untouched. A **double-click** selects a word and
        // therefore does not navigate — deliberately: the gesture asked for the
        // word, and a card whose meaning changed with the click count would be
        // the worse surprise.
        let selection = editor.clone();
        div()
            // Keyed by the fragment, not by its seat: the results are re-cut on
            // every frame and a background write reorders them, so an
            // index-keyed element would carry one card's per-element state — an
            // armed mouse-down among it — onto whatever result took its place.
            // The probe name stays positional, as the driver's selector.
            .id(SharedString::from(format!(
                "space-find-frag-{}",
                fragment.id
            )))
            // A managed descendant of the list, never a tab stop of its own.
            .probe_delegating(
                format!("space/find/result/{index}"),
                gpui::Role::ListItem,
                label,
            )
            // Set position over the **data rows** — the fragments, which are
            // the only kind of row the count is about.
            .aria_position_in_set(index + 1)
            .aria_size_of_set(total)
            .aria_selected(on_cursor)
            .when(on_cursor, |d| d.aria_active_descendant())
            .relative()
            .w_full()
            .mx_0()
            .px_4()
            .pb(px(FRAGMENT_GAP))
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, window, cx| {
                this.click_find_result(opener.clone(), &selection, window, cx);
            }))
            .child(
                div()
                    .w_full()
                    .rounded_md()
                    .bg(card)
                    .border_1()
                    .border_color(if on_cursor && keyboard { border } else { card })
                    .shadow_xs()
                    .when(on_cursor && keyboard, |d| {
                        d.shadow(crate::focus::ring_shadows(crate::focus::ring_colors()))
                    })
                    .px_3()
                    .py_2()
                    .child(
                        MarkdownEditor::new(&editor)
                            .style(style)
                            .disabled(true)
                            .fragment(fragment.range.clone()),
                    ),
            )
            .child(record_fragment_height(
                fragment.id.clone(),
                heights,
                cx.entity().downgrade(),
            ))
            .into_any_element()
    }

    /// A fragment's accessible name: **who wrote it**, and the opening of what
    /// it says — the minimap cell's shape, over the fragment's own text.
    ///
    /// The attribution is part of the name rather than left to the group header
    /// beside it, because that header is a sibling `Label`: a reader met the
    /// card through the list's active descendant, heard the snippet alone, and
    /// could not tell two similar results by different participants apart.
    /// **The join is a message, not a `format!`.** Its parts are data, but the
    /// punctuation between them and the order they read in are not: the
    /// Chinese locales want a full-width colon, and a hard-coded `": "` is the
    /// concatenation the localization doctrine bans however few English words
    /// it contains.
    fn find_fragment_label(&self, fragment: &ResultFragment, cx: &gpui::App) -> SharedString {
        let text = fragment
            .content
            .get(fragment.range.clone())
            .unwrap_or_default();
        // No references: a fragment's own bytes are what it paints, and an
        // embed marker inside one is hidden there exactly as it is in the post.
        let snippet = super::minimap::spoken_snippet(text, &[], FRAGMENT_LABEL_CHARS);
        crate::i18n::msg::find_result_name(cx, fragment.byline.to_string(), snippet)
    }

    /// The editor state one fragment paints through, minted on first sight.
    ///
    /// Its value is the node's **whole** markdown, because a block's source
    /// range indexes into that document and the filter is what narrows the
    /// paint. Setting it is guarded on a change, so a card that is already
    /// showing the right text notifies nothing and the overlay settles after
    /// one frame rather than repainting forever.
    fn find_fragment_editor(
        &mut self,
        fragment: &ResultFragment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Entity<MarkdownEditorState> {
        let existing = self
            .find
            .as_ref()
            .and_then(|s| s.overlay.bodies.get(&fragment.id).cloned());
        let editor = match existing {
            Some(e) => e,
            None => {
                let e = cx.new(|cx| MarkdownEditorState::new(window, cx));
                if let Some(session) = self.find.as_mut() {
                    session
                        .overlay
                        .bodies
                        .insert(fragment.id.clone(), e.clone());
                }
                e
            }
        };
        if editor.read(cx).value() != fragment.content.as_ref() {
            let content = fragment.content.to_string();
            editor.update(cx, |e, cx| e.set_value(content, cx));
        }
        // The same compare-before-set guard the conversation's own match layers
        // take: without it every frame notifies.
        let ranges: Vec<(Range<usize>, u64)> =
            fragment.hits.iter().cloned().map(|r| (r, 0u64)).collect();
        let stale = *editor.read(cx).highlights_in(HighlightLayer::Overlay)
            != gpui_markdown_editor::HighlightSet::new(ranges.clone());
        if stale {
            editor.update(cx, |e, cx| {
                e.set_highlights_in(HighlightLayer::Overlay, ranges, cx);
            });
        }
        editor
    }
}

/// The measuring canvas one fragment card carries: it records the card's real
/// height so the placeholder that stands in for it once it leaves the band is
/// the right size, and schedules one frame when the number moves.
fn record_fragment_height(
    id: SharedString,
    heights: Rc<RefCell<HashMap<SharedString, f32>>>,
    view: WeakEntity<SpaceView>,
) -> impl IntoElement {
    gpui::canvas(
        |_, _, _| {},
        move |bounds: Bounds<Pixels>, _, window, _cx| {
            let h = bounds.size.height.as_f32();
            let moved = {
                let mut map = heights.borrow_mut();
                let moved = map.get(&id).is_none_or(|prev| (prev - h).abs() > 0.5);
                if moved {
                    map.insert(id.clone(), h);
                }
                moved
            };
            if moved {
                let view = view.clone();
                window.on_next_frame(move |_, cx| {
                    view.update(cx, |_, cx| cx.notify()).ok();
                });
            }
        },
    )
    .absolute()
    .top_0()
    .left_0()
    .size_full()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space_view::model::{NodeSrc, TreeNode};

    fn node(id: &str, children: Vec<TreeNode>) -> TreeNode {
        TreeNode {
            src: NodeSrc::Msg(0),
            id: id.into(),
            children,
        }
    }

    fn all(_: &TreeNode) -> bool {
        true
    }

    #[test]
    fn the_map_lays_a_spine_straight_down_and_fans_its_branches_right() {
        //     a
        //     b ── c is b's second child
        //   /   \
        //  d      c
        let tree = vec![node(
            "a",
            vec![node("b", vec![node("d", vec![]), node("c", vec![])])],
        )];
        let laid = map_layout(&tree, &all);
        let by = |id: &str| laid.iter().find(|n| n.node == id).expect("present");
        // The spine keeps lane 0 all the way down.
        assert_eq!((by("a").depth, by("a").lane), (0, 0));
        assert_eq!((by("b").depth, by("b").lane), (1, 0));
        assert_eq!((by("d").depth, by("d").lane), (2, 0));
        // The second child opens a lane of its own, at the same depth.
        assert_eq!((by("c").depth, by("c").lane), (2, 1));
        assert_eq!(by("c").parent, Some((1, 0)));
    }

    #[test]
    fn the_order_is_depth_first_then_left_to_right() {
        // Deliberately *not* the transcript's pre-order, which would read
        // a, b, d, e, c: the overlay presents the same results in a different
        // orientation, so it reads across each generation before descending.
        let tree = vec![node(
            "a",
            vec![node(
                "b",
                vec![node("d", vec![node("e", vec![])]), node("c", vec![])],
            )],
        )];
        let laid = map_layout(&tree, &all);
        let order: Vec<&str> = laid.iter().map(|n| n.node.as_ref()).collect();
        assert_eq!(order, vec!["a", "b", "d", "c", "e"]);
    }

    #[test]
    fn no_two_nodes_at_one_depth_share_a_lane() {
        // A lane is allocated once and then only continues through first
        // children, so it is one root-to-leaf chain and meets each depth once.
        // The property is what makes the (depth, lane) sort total.
        let tree = vec![
            node(
                "r1",
                vec![
                    node("a", vec![node("a1", vec![]), node("a2", vec![])]),
                    node("b", vec![node("b1", vec![])]),
                ],
            ),
            node("r2", vec![node("c", vec![])]),
        ];
        let laid = map_layout(&tree, &all);
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for n in &laid {
            assert!(
                seen.insert((n.depth, n.lane)),
                "{} reuses ({}, {})",
                n.node,
                n.depth,
                n.lane
            );
        }
        // And siblings stay adjacent: the two thread roots are lanes 0 and 1,
        // and the branches either one opens are to the right of both — which
        // is the reservation rule, not an accident of which subtree is deeper.
        let lane_of = |id: &str| laid.iter().find(|n| n.node == id).expect("present").lane;
        assert_eq!((lane_of("r1"), lane_of("r2")), (0, 1));
        assert!(
            lane_of("a") == 0 && lane_of("b") > 1,
            "r1's own fork opens past both roots (a {}, b {})",
            lane_of("a"),
            lane_of("b")
        );
    }

    #[test]
    fn an_excluded_leaf_leaves_the_shape_alone() {
        let tree = vec![node(
            "a",
            vec![
                node("keep", vec![]),
                node("drop", vec![]),
                node("also", vec![]),
            ],
        )];
        let laid = map_layout(&tree, &|n| n.id != "drop");
        let order: Vec<&str> = laid.iter().map(|n| n.node.as_ref()).collect();
        assert_eq!(order, vec!["a", "keep", "also"]);
        // `also` takes the lane the excluded sibling would have, not the one
        // after it — the map draws no gap for something it is not drawing.
        let also = laid.iter().find(|n| n.node == "also").expect("present");
        assert_eq!(also.lane, 1);
    }

    #[test]
    fn contiguous_matching_blocks_consolidate_and_a_gap_splits() {
        let blocks = vec![0..10, 10..20, 20..30, 30..40];
        // Hits in blocks 0, 1 and 3: the first two are contiguous, the last is
        // its own fragment.
        let hits = vec![2..4, 12..14, 33..35];
        let runs = fragment_runs(&blocks, &hits);
        assert_eq!(cuts(&runs), vec![(0..20, 0..2), (30..40, 2..3)]);
    }

    /// A run's source span and the slice of the node's hits it owns.
    fn cuts(runs: &[FragmentRun]) -> Vec<(Range<usize>, Range<usize>)> {
        runs.iter()
            .map(|r| (r.range.clone(), r.hits.clone()))
            .collect()
    }

    /// What the cross product answered: every hit against every block, the
    /// lowest and highest hit index recorded per block, then consecutive block
    /// indices consolidated. The two-pointer walk replaced this, so the walk is
    /// pinned against it rather than against a hand-written expectation.
    fn cuts_by_scan(
        blocks: &[Range<usize>],
        hits: &[Range<usize>],
    ) -> Vec<(Range<usize>, Range<usize>)> {
        let mut touched: Vec<Option<(usize, usize)>> = vec![None; blocks.len()];
        for (h, hit) in hits.iter().enumerate() {
            for (b, block) in blocks.iter().enumerate() {
                if block.start < hit.end.max(hit.start + 1) && hit.start < block.end {
                    touched[b] = Some(match touched[b] {
                        Some((lo, hi)) => (lo.min(h), hi.max(h)),
                        None => (h, h),
                    });
                }
            }
        }
        let mut out = Vec::new();
        let mut run: Option<(usize, usize, usize, usize)> = None;
        for (b, slot) in touched.iter().enumerate() {
            let Some((lo, hi)) = *slot else { continue };
            run = match run {
                Some((f, l, h0, h1)) if b == l + 1 => Some((f, b, h0.min(lo), h1.max(hi))),
                Some((f, l, h0, h1)) => {
                    out.push((blocks[f].start..blocks[l].end, h0..h1 + 1));
                    Some((b, b, lo, hi))
                }
                None => Some((b, b, lo, hi)),
            };
        }
        if let Some((f, l, h0, h1)) = run {
            out.push((blocks[f].start..blocks[l].end, h0..h1 + 1));
        }
        out
    }

    #[test]
    fn the_run_walk_answers_what_the_cross_product_would() {
        // A layout with every shape the walk has to keep straight: a hit before
        // the first block, a hit that spans three blocks, several hits inside
        // one block, adjacent blocks with a source gap, a block nothing
        // touches, and a hit past the end. The walk's block cursor never
        // rewinds, so a hit landing behind one that spanned forward is exactly
        // where a one-pass version goes wrong.
        let blocks = vec![10..20, 20..30, 30..40, 50..60, 70..80];
        let hits = vec![
            0..2,   // before every block
            12..14, // block 0
            15..16, // block 0 again
            25..35, // spans blocks 1 and 2
            26..27, // back inside block 1, after the spanning hit
            72..73, // block 4, with block 3 untouched between them
            90..92, // past the end
        ];
        assert_eq!(
            cuts(&fragment_runs(&blocks, &hits)),
            cuts_by_scan(&blocks, &hits),
            "the walk selects what the scan selected"
        );
        // And the answer really is the cut the reader sees: three fragments,
        // each owning a contiguous slice of the node's hits, with no hit in two
        // of them and none of the placed hits lost.
        let runs = fragment_runs(&blocks, &hits);
        assert_eq!(
            cuts(&runs),
            vec![(10..40, 1..5), (70..80, 5..6)],
            "the spanning hit consolidates its blocks, and an untouched block splits"
        );
    }

    #[test]
    fn a_fragment_records_the_first_match_inside_it() {
        // Two fragments with an unmatched block between them, and hits that
        // arrive in the *other* order — a projection reports them in source
        // order, but the ordinal a fragment hands the anchor is the hit's own
        // index in that list, never its position among the fragments.
        let blocks = vec![0..10, 10..20, 20..30];
        let hits = vec![3..4, 25..26];
        let runs = fragment_runs(&blocks, &hits);
        assert_eq!(cuts(&runs), vec![(0..10, 0..1), (20..30, 1..2)]);
    }

    #[test]
    fn adjacent_blocks_consolidate_even_across_the_gap_between_them() {
        // "Contiguous" is a fact about the *render*, not about the bytes: two
        // blocks with nothing laid out between them are one thing on the page,
        // and the whitespace separating them in the source is not a third.
        let blocks = vec![0..10, 20..30];
        let hits = vec![3..4, 25..26];
        assert_eq!(cuts(&fragment_runs(&blocks, &hits)), vec![(0..30, 0..2)]);
    }

    #[test]
    fn the_post_index_answers_what_a_scan_would() {
        // The map labels every dot and the results attribute every group, so
        // both ask "which row is this node?" once per node per frame. That was
        // a linear scan recomputing each candidate's id on the way past —
        // quadratic in the conversation, ahead of any virtualization. The index
        // is pinned against the scan it replaced, over rows of both kinds: a
        // persisted one that names itself by its action id, and an optimistic
        // one with none, which `node_id` names by its seat.
        let row = |action: Option<&str>| super::super::model::PostData {
            action_id: action.map(SharedString::from),
            item_id: None,
            parent_action_id: None,
            role: "user".into(),
            byline: "You".into(),
            byline_backend: None,
            time: "".into(),
            content: "".into(),
            model: None,
            generation_count: 1,
            reasoning: None,
            reasoning_expanded: false,
            references: Vec::new(),
            blocks: Vec::new(),
            regenerable: false,
            truncated: false,
        };
        let posts = vec![row(Some("a1")), row(None), row(Some("a3")), row(None)];
        let index = post_index(&posts);
        for i in 0..posts.len() {
            let id = super::super::model::node_id(&posts, i);
            let scanned = (0..posts.len())
                .find(|j| super::super::model::node_id(&posts, *j) == id)
                .expect("the scan finds what it just named");
            assert_eq!(index.get(&id).copied(), Some(scanned), "row {i} ({id})");
        }
        assert_eq!(index.len(), posts.len(), "one entry per row, and no more");
        assert_eq!(index.get(&SharedString::from("nobody")), None);
    }

    #[test]
    fn a_lane_past_the_columns_edge_keeps_its_own_x() {
        // Lanes squeeze to the floor and no further, so a wide graph runs off
        // the column and is scrolled to. Clamping the excess instead stacked
        // every lane past the twelfth on one x — distinct branches, and their
        // buttons, on top of one another.
        assert_eq!(
            map_lane_width(14),
            MAP_LANE_MIN_W,
            "a wide graph is at the lane floor"
        );
        let xs: Vec<f32> = (0..14).map(|lane| map_lane_x(lane, 14)).collect();
        let avail = MAP_WIDTH - 2.0 * MAP_PAD - MAP_DOT;
        assert!(
            xs.last().copied().unwrap_or(0.0) > avail,
            "the last lane really is past the column's own width"
        );
        for pair in xs.windows(2) {
            assert!(pair[1] > pair[0], "no two lanes share an x: {xs:?}");
        }
        // A narrow graph still spreads to the full stride.
        assert_eq!(map_lane_width(1), MAP_LANE_W);
        assert_eq!(map_lane_width(2), MAP_LANE_W);
    }

    #[test]
    fn a_reveal_moves_the_map_as_little_as_it_can() {
        // The dot the keyboard is on has to be inside the column, and nothing
        // else may move: a reader who scrolled the map somewhere keeps that
        // place for every dot already showing.
        let view = MAP_WIDTH - 2.0 * MAP_PAD;
        let max = 60.0;

        // Already in view, at either end of the visible span: unchanged.
        assert_eq!(map_reveal_axis(0.0, MAP_DOT, view, 0.0, max), 0.0);
        assert_eq!(
            map_reveal_axis(view - MAP_DOT, MAP_DOT, view, 0.0, max),
            0.0,
            "a dot ending exactly at the edge is in view"
        );

        // Past the far edge: scrolled just far enough to end at the edge.
        let past = view + 4.0;
        let o = map_reveal_axis(past, MAP_DOT, view, 0.0, max);
        assert_eq!(o, view - (past + MAP_DOT));
        assert!(
            past >= -o && past + MAP_DOT <= -o + view,
            "…and that really does bring the whole dot inside"
        );

        // Behind the near edge: scrolled to sit exactly on it, never past it.
        assert_eq!(map_reveal_axis(10.0, MAP_DOT, view, -40.0, max), -10.0);

        // The clamp is gpui's own range, so a viewport with nothing to scroll
        // answers "stay" rather than inventing an offset.
        assert_eq!(map_reveal_axis(400.0, MAP_DOT, view, 0.0, 0.0), 0.0);
        assert_eq!(
            map_reveal_axis(400.0, MAP_DOT, view, 0.0, max),
            -max,
            "and a dot past the content's own end stops at the end"
        );
        // Before the first paint there is no viewport to reveal into.
        assert_eq!(map_reveal_axis(400.0, MAP_DOT, 0.0, -7.0, max), -7.0);
    }

    #[test]
    fn a_hit_in_no_block_is_dropped_rather_than_placed() {
        let blocks = vec![0..10, 20..30];
        // Two, so the block list and the hit list are both plural — a lone
        // range literal reads to clippy as an attempt to write `(50..52)`.
        let hits = vec![50..52, 60..62];
        assert!(fragment_runs(&blocks, &hits).is_empty());
    }
}
