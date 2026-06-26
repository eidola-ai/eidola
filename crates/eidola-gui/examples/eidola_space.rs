use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};

use eidola_gui::theme;
use gpui::*;
use gpui_component::{ActiveTheme, InteractiveElementExt, Root, h_flex, v_flex};
use gpui_markdown_editor::{
    EditorState, MarkdownEditor, MarkdownEditorEvent, MarkdownEditorState, MarkdownStyle,
};

// ---------------------------------------------------------------------------
// Layout constants. Kept inline (and a little duplicated from the real chat
// view) on purpose — this is a short-lived visual experiment, so we favour a
// self-contained file over reaching into the app's chat module.
// ---------------------------------------------------------------------------

/// Height reserved at the top of the window for the (transparent) titlebar.
/// macOS extends the content view under the traffic-light buttons, so we leave
/// this much room and treat the band as a draggable title-bar surface.
const TITLE_BAR_RESERVE: Pixels = px(36.);

/// Prose typography for the user-/AI-authored narrative content. The body is
/// Newsreader (a serif) at a book-like size and leading — deliberately distinct
/// from the system UI font the theme uses for components/chrome.
const PROSE_FONT_SIZE: Pixels = px(17.);
const PROSE_LINE_HEIGHT: f32 = 1.65;

/// The byline gutter (right-aligned author + time) and the centered reading
/// column it sits beside.
const GUTTER_WIDTH: Pixels = px(120.);
const GUTTER_GAP: Pixels = px(28.);
const BODY_MAX_WIDTH: Pixels = px(600.);

/// Vertical breathing room around each post, plus the faint full-bleed band
/// that separates one depth level (one row of the tree) from the next.
const POST_PAD_Y: Pixels = px(40.);
const BAND_HEIGHT: Pixels = px(48.);

/// Minimum per-event horizontal finger step (px) at release that counts as a
/// directional *flick* — above this, the snap advances/retreats one branch in
/// the flick's direction; below it, the snap goes to the nearest branch.
const SNAP_FLING_THRESHOLD: f32 = 8.0;

/// The composer — a single editable markdown pane pinned over the bottom of the
/// window. Its body aligns with the posts (same gutter/centering/margins). It
/// floats over the bottom, growing with content up to [`COMPOSER_MAX_FRACTION`]
/// of the window, then scrolling internally; near the bottom of a branch it
/// *docks* to the trailing separator and grows into the page (see
/// [`SpaceView::render_active_draft`]).
const COMPOSER_MAX_FRACTION: f32 = 0.5;

/// Width of the topology minimap pinned to the right edge.
const MINIMAP_WIDTH: Pixels = px(36.);
/// Gap between sibling columns within a minimap row.
const MINIMAP_COL_GAP: Pixels = px(4.);
/// How long the minimap lingers after scrolling stops / the cursor leaves it,
/// before it fades (mirrors macOS overlay scrollbars).
const MINIMAP_HIDE_DELAY: Duration = Duration::from_millis(400);
/// Fade-out duration once hiding begins.
const MINIMAP_FADE: Duration = Duration::from_millis(200);

/// One post in the conversation tree: who wrote it, when, its markdown content,
/// and its replies. The space is a tree of these (replies only); the UI follows
/// the tree structure (see [`SpaceView`]).
struct Node {
    /// Stable id, unique across the tree — also the element id for the post's
    /// page (keeps the per-post markdown editors from colliding) and the key
    /// for its editor state in [`SpaceView::bodies`].
    id: &'static str,
    author: &'static str,
    created_at: &'static str,
    /// Formatted as markdown.
    content: &'static str,
    /// Replies, ordered left-to-right by creation time (earliest first).
    children: Vec<Node>,
}

/// Which axis a scroll gesture is locked to. Determined from the first real
/// movement of a gesture and held until the gesture ends, so a mostly-vertical
/// scroll never nudges the branches sideways (and vice versa).
#[derive(Clone, Copy, PartialEq)]
enum ScrollAxis {
    Horizontal,
    Vertical,
}

/// Who owns a vertical scroll *session* — decided by where the gesture starts
/// and held until the next gesture begins (so it spans momentum and direction
/// reversals). A gesture that starts over the conversation, or over a docked
/// composer, scrolls the `Body` (page) and freezes the composer's internal
/// scroll for the whole session; one that starts over a *floating* composer is
/// owned by the `Composer` (internal scroll only — the page never moves, even at
/// the composer's scroll limits).
#[derive(Clone, Copy, PartialEq)]
enum ScrollOwner {
    Body,
    Composer,
}

/// An in-flight "scroll-snap" glide of one branch scroller toward a page
/// boundary. gpui forwards macOS's already-decayed momentum deltas straight into
/// the scroll offset and exposes no momentum-end signal (only the finger lift,
/// `TouchPhase::Ended`), so a CSS-style "let momentum land on a snap point"
/// isn't reachable by cooperating with the OS. Instead we take over at finger
/// lift: capture the release velocity, then drive our own eased glide to the
/// target branch and suppress the OS momentum that would otherwise fight it
/// (see [`SpaceView::start_snap`]).
#[derive(Clone, Copy)]
struct SnapAnim {
    /// The branch scroller (node with children) being animated.
    node_id: &'static str,
    /// Horizontal scroll offset (`ScrollHandle` x, ≤ 0) at the start and end of
    /// the glide.
    from_x: f32,
    to_x: f32,
    /// Wall-clock start and total duration of the glide.
    start: Instant,
    duration: Duration,
}

/// Cubic ease-out: fast departure, gentle arrival — reads as a thrown page
/// floating to rest on its snap point.
fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t;
    1.0 - u * u * u
}

/// The space view renders a conversation *tree* as **recursively nested**
/// scrollers. Each node renders its post, then (if it has replies) a separator
/// band and a horizontal scroller whose pages are its children — and each of
/// those pages is the child's *entire subtree*, rendered the same way. So
/// scrolling a node's children scroller moves between whole branches (every
/// descendant travels with its branch), and the nesting *is* the tree.
///
/// A horizontal scroll is claimed by the innermost branch scroller under the
/// cursor (it stops propagation), so scrolling over a deep post navigates that
/// post's level while scrolling over a shallower region navigates the level
/// that encloses it. Vertical scroll bubbles to the one outer page scroller.
///
/// **Dynamic-height variant.** Each branch scroller is sized to its *selected*
/// child's subtree (not the tallest), so the overall content height — and the
/// minimap scale — track the active branch: switching branches grows or shrinks
/// the document instead of holding one static maximum with slack at the bottom.
///
/// **Editing model.** At rest there is no editor anywhere — every post ends in a
/// separator band carrying a "+" that appends a *draft* reply (a new child of the
/// post above it, see [`SpaceView::create_draft`]). A draft renders like any node
/// but with an editable editor under a "Draft" gutter. Focusing a draft's editor
/// makes it the [`SpaceView::active_draft`]: it adopts the floating/docking
/// "composer" behavior — docked at its in-flow spot when scrolled to, floating
/// pinned to the window bottom when scrolled above it *or* when the user has
/// swiped to a different branch while editing (see
/// [`SpaceView::render_active_draft`]). Escape (or focusing another draft)
/// deactivates it back to a plain inline editor. A draft at the end of the
/// selected branch reproduces the single-composer behavior exactly.
pub struct SpaceView {
    /// The conversation tree (a single root for this experiment). Mutable: the
    /// "+" affordance on a separator appends a *draft* child (see
    /// [`SpaceView::create_draft`]).
    root: Node,
    /// One markdown-editor state per node, keyed by node id. Posts are read-only
    /// (`disabled`); draft nodes (ids in [`SpaceView::drafts`]) are editable.
    bodies: HashMap<&'static str, Entity<MarkdownEditorState>>,
    /// Ids of nodes that are unsent *drafts* — rendered like any node but with an
    /// editable editor and a "Draft" gutter. Created by the separator "+" button.
    drafts: HashSet<&'static str>,
    /// The draft whose editor currently has focus, if any. The active draft
    /// adopts the floating/docking composer behavior (see
    /// [`SpaceView::render_active_draft`]); every other draft is a plain inline
    /// editor. `None` means no editor is floating — the default resting state.
    active_draft: Option<&'static str>,
    /// Focus subscriptions for draft editors, keyed by draft id: a draft's
    /// `MarkdownEditorEvent::Focus` makes it the [`SpaceView::active_draft`].
    /// Held so the subscriptions outlive the closure that created them, and keyed
    /// so a deleted draft's subscription is dropped with it.
    draft_subs: HashMap<&'static str, Subscription>,
    /// Monotonic counter for minting unique (leaked) `&'static str` draft ids.
    next_draft_seq: usize,
    /// Painted bounds of each *inline* (non-active) draft's body, keyed by draft
    /// id — the editor's natural content height (distinct from the post's
    /// window-clamped block in `post_bounds`). Read by [`SpaceView::floating_pad`]
    /// to size the off-branch bottom padding to a tall selected-leaf draft.
    draft_body_bounds: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,
    /// Internal scroll of the *active draft's* overlay, used when its content
    /// exceeds the floating cap; tracking it gives the pane native wheel
    /// scrolling and lets the offset clamp as the pane grows. Shared across
    /// drafts since only one is active at a time.
    composer_scroll: ScrollHandle,
    /// The active draft overlay's internal scroll offset `y` as of the last
    /// render. When the *body* owns the scroll session, we restore the overlay to
    /// this value on every wheel event (the built-in listener scrolls it first)
    /// so its internal scroll stays frozen while the page scrolls. Synced each
    /// frame in `render`.
    composer_prev_off_y: f32,
    /// Owner of the current vertical scroll session (see [`ScrollOwner`]); `None`
    /// between gestures, decided on the first vertical move and reset on the next
    /// gesture's `Started`.
    scroll_owner: Option<ScrollOwner>,
    /// Whether the active draft overlay is currently floating (vs docked), cached
    /// from the last `render_active_draft` so a scroll handler can decide session
    /// ownership.
    composer_overlayed: Cell<bool>,
    /// The active draft's *natural* (unclipped) content height, recorded each
    /// frame by a canvas inside the overlay's scrolled content. Drives both the
    /// floating-height cap and the in-flow placeholder's height.
    composer_content_h: Rc<RefCell<Pixels>>,
    /// Painted bounds of the active draft's in-flow *placeholder* (the empty slot
    /// it reserves in the tree while its editor floats in the overlay), keyed by
    /// draft id. Positions the dock and feeds the minimap. At most one live entry
    /// (the active draft); stale entries from past drafts are unread.
    slot_bounds: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,
    /// One horizontal `ScrollHandle` per node that has children, keyed by node
    /// id. Read back to highlight the active page-indicator dot, and written
    /// directly to undo the built-in scroller's stray horizontal nudge during a
    /// vertical gesture.
    scrolls: HashMap<&'static str, ScrollHandle>,
    /// The axis the current scroll gesture is locked to (see [`ScrollAxis`]);
    /// `None` between gestures.
    scroll_axis: Option<ScrollAxis>,
    /// The most recent non-zero horizontal step of the live gesture, used as the
    /// release velocity for the snap's flick decision. Reset on gesture start.
    last_h_delta: Pixels,
    /// The branch scroller currently gliding to a snap point (frame-driven), if
    /// any (see [`SnapAnim`]). At most one snaps at a time.
    snap: Option<SnapAnim>,
    /// After a glide completes (or a release that was already aligned), the
    /// branch and its resting x are pinned here until the next gesture, so any
    /// trailing OS momentum is absorbed instead of drifting the page off-branch.
    snap_pin: Option<(&'static str, f32)>,
    /// The `page_width` of the previous frame. Branch offsets are absolute
    /// pixels but pages are window-width apart, so a window resize would shift
    /// every offset relative to its branches. Diffing this against the current
    /// width lets us remap offsets by the stride ratio (see
    /// [`SpaceView::remap_for_resize`]), keeping the selected branch invariant.
    last_page_width: Option<Pixels>,
    /// Painted bounds of every post, keyed by node id, recorded each frame by a
    /// `canvas` overlay (see [`record_bounds`]). The topology minimap reads
    /// these (from the previous frame) to size its rows and decide which spans
    /// are on-screen. Off-screen posts are still measured — the transcript is a
    /// plain `overflow_scroll`, not a virtualized list, so every post prepaints.
    post_bounds: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,
    /// Signature of the last minimap inputs (post bounds + viewport), so a
    /// reflow or scroll triggers exactly one follow-up frame to redraw the
    /// minimap, converging when the layout settles.
    minimap_sig: f32,
    /// macOS-style overlay-scrollbar visibility for the minimap.
    /// `visible`: shown now (opacity 1). `gesturing`: fingers are on the
    /// trackpad (between scroll `Started`/`Ended`), so it stays up even with no
    /// movement. `hovered`: cursor over the bar. `fade_gen` bumps on each hide
    /// to restart the fade-out animation. `hide_task` is the 1s linger timer.
    minimap_visible: bool,
    minimap_gesturing: bool,
    minimap_hovered: bool,
    minimap_fade_gen: usize,
    minimap_hide_task: Option<Task<()>>,
    /// Set on mouse-down in the title-bar band, consumed on the first
    /// mouse-move to begin a native window drag (mirrors gpui-component's
    /// `TitleBar`: `start_window_move` wants a drag event, not the down).
    should_move_window: bool,
}

impl SpaceView {
    /// Build the view: seed the conversation tree and a read-only
    /// markdown-editor state per node (each needs `window`/`cx`, so this runs
    /// inside `cx.new`).
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let root = sample_tree();

        let mut bodies = HashMap::new();
        let mut scrolls = HashMap::new();
        build_state(&root, &mut bodies, &mut scrolls, window, cx);

        Self {
            root,
            bodies,
            drafts: HashSet::new(),
            active_draft: None,
            draft_subs: HashMap::new(),
            next_draft_seq: 0,
            draft_body_bounds: Rc::new(RefCell::new(HashMap::new())),
            composer_scroll: ScrollHandle::new(),
            composer_prev_off_y: 0.0,
            scroll_owner: None,
            composer_overlayed: Cell::new(false),
            composer_content_h: Rc::new(RefCell::new(px(0.))),
            slot_bounds: Rc::new(RefCell::new(HashMap::new())),
            scrolls,
            scroll_axis: None,
            last_h_delta: px(0.),
            snap: None,
            snap_pin: None,
            last_page_width: None,
            post_bounds: Rc::new(RefCell::new(HashMap::new())),
            minimap_sig: f32::NAN,
            minimap_visible: false,
            minimap_gesturing: false,
            minimap_hovered: false,
            minimap_fade_gen: 0,
            minimap_hide_task: None,
            should_move_window: false,
        }
    }

    /// Whether `id` names an unsent draft node.
    fn is_draft(&self, id: &str) -> bool {
        self.drafts.contains(id)
    }

    /// Whether the subtree rooted at `node` contains any draft (drafts are always
    /// leaves, but can sit anywhere within a branch). Drives the info-colored
    /// branch indicator for a branch that holds an unsent draft.
    fn subtree_has_draft(&self, node: &Node) -> bool {
        self.is_draft(node.id) || node.children.iter().any(|c| self.subtree_has_draft(c))
    }

    /// Append a new draft reply as a child of `parent_id` (the node whose
    /// separator "+" was clicked), select that branch, and make the draft active
    /// (floating composer). The draft is a fresh editable editor; mints a unique
    /// leaked id (fine for a short-lived experiment — a handful of drafts).
    fn create_draft(
        &mut self,
        parent_id: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let page_width = window.viewport_size().width;
        self.next_draft_seq += 1;
        // Leak a unique 'static id so it slots into the existing &'static-keyed
        // maps and Copy closures without threading a SharedString everywhere.
        let id: &'static str = Box::leak(format!("draft-{}", self.next_draft_seq).into_boxed_str());

        let editor =
            cx.new(|cx| MarkdownEditorState::with_state(EditorState::default(), window, cx));
        // Focus on this draft's editor makes it the active (floating) draft.
        let sub = cx.subscribe(&editor, move |this, _editor, event, cx| {
            if matches!(event, MarkdownEditorEvent::Focus) && this.active_draft != Some(id) {
                this.activate_draft(id, cx);
            }
        });
        self.draft_subs.insert(id, sub);
        self.bodies.insert(id, editor.clone());
        self.drafts.insert(id);

        // Append the draft node to the parent and learn its (last) page index.
        let mut new_index = 0usize;
        if let Some(parent) = node_mut(&mut self.root, parent_id) {
            parent.children.push(Node {
                id,
                author: "You",
                created_at: "now",
                content: "",
                children: vec![],
            });
            new_index = parent.children.len() - 1;
        }
        // The parent now has children, so it needs a horizontal scroller.
        self.scrolls.entry(parent_id).or_default();

        // Select the new branch: jump (and pin) the parent scroller to it.
        let stride = (page_width + BAND_HEIGHT).as_f32();
        let to_x = -(new_index as f32) * stride;
        if let Some(handle) = self.scrolls.get(parent_id) {
            let off = handle.offset();
            handle.set_offset(point(px(to_x), off.y));
        }
        self.cancel_snap();
        self.snap_pin = Some((parent_id, to_x));

        self.activate_draft(id, cx);
        let focus = editor.read(cx).focus_handle.clone();
        window.focus(&focus, cx);
    }

    /// Make `id` the active (floating) draft, resetting the shared overlay scroll
    /// so a previously-scrolled draft's offset doesn't carry over. Switching away
    /// from another draft retires it (deletes it if left blank).
    fn activate_draft(&mut self, id: &'static str, cx: &mut Context<Self>) {
        if self.active_draft != Some(id) {
            self.retire_active_draft(cx);
        }
        self.active_draft = Some(id);
        self.composer_scroll.set_offset(point(px(0.), px(0.)));
        self.composer_prev_off_y = 0.0;
        cx.notify();
    }

    /// Deactivate the active draft (Escape / external request). A draft left with
    /// content stays in the tree as a plain inline editor; only its floating
    /// behavior ends. A blank one is deleted (see [`SpaceView::retire_active_draft`]).
    fn deactivate_active_draft(&mut self, cx: &mut Context<Self>) {
        if self.active_draft.is_some() {
            self.retire_active_draft(cx);
            cx.notify();
        }
    }

    /// Clear the active draft. If its editor was left empty, delete the node too —
    /// an abandoned blank draft (Escape, or switching to another) leaves no trace.
    fn retire_active_draft(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.active_draft.take() else {
            return;
        };
        let empty = self
            .bodies
            .get(id)
            .map(|e| e.read(cx).is_empty())
            .unwrap_or(false);
        if empty {
            self.delete_draft(id, cx);
        }
    }

    /// Remove a draft node from the tree and forget its editor/state. Used by
    /// [`SpaceView::retire_active_draft`] for an abandoned blank draft. Clamps the
    /// parent scroller back onto a still-valid branch (or drops it if the parent
    /// is a leaf again).
    fn delete_draft(&mut self, id: &'static str, cx: &mut Context<Self>) {
        if let Some(parent_id) = parent_of(&self.root, id) {
            let remaining = match node_mut(&mut self.root, parent_id) {
                Some(parent) => {
                    parent.children.retain(|c| c.id != id);
                    parent.children.len()
                }
                None => 0,
            };
            if remaining == 0 {
                self.scrolls.remove(parent_id);
            } else if let Some(pw) = self.last_page_width {
                // The deleted branch may have been the selected one; clamp the
                // scroller onto the nearest still-valid branch and pin it.
                let stride = (pw + BAND_HEIGHT).as_f32();
                if stride > 0.0
                    && let Some(handle) = self.scrolls.get(parent_id).cloned()
                {
                    let off = handle.offset();
                    let idx =
                        ((-off.x.as_f32() / stride).round() as i64).clamp(0, remaining as i64 - 1);
                    let to_x = -(idx as f32) * stride;
                    handle.set_offset(point(px(to_x), off.y));
                    self.cancel_snap();
                    self.snap_pin = Some((parent_id, to_x));
                }
            }
        }
        self.drafts.remove(id);
        self.bodies.remove(id);
        self.draft_subs.remove(id);
        self.draft_body_bounds.borrow_mut().remove(id);
        cx.notify();
    }

    /// Lock the scroll gesture to an axis. `Started`/`Ended` are gesture
    /// boundaries that clear the lock; the first real movement of the next
    /// gesture sets it. gpui has no `scrollend`, but macOS reports trackpad
    /// phases (Began → `Started`, lift → `Ended`); momentum arrives as `Moved`
    /// and simply re-locks to the same axis.
    fn resolve_scroll_axis(&mut self, phase: TouchPhase, delta: Point<Pixels>) -> ScrollAxis {
        if !matches!(phase, TouchPhase::Moved) {
            self.scroll_axis = None;
        }
        if let Some(axis) = self.scroll_axis {
            return axis;
        }
        let axis = if delta.y.as_f32().abs() >= delta.x.as_f32().abs() {
            ScrollAxis::Vertical
        } else {
            ScrollAxis::Horizontal
        };
        if !delta.x.is_zero() || !delta.y.is_zero() {
            self.scroll_axis = Some(axis);
        }
        axis
    }

    /// Record a scroll event for the minimap's show/hide. `moved` is whether
    /// this event actually changed the scroll position (non-zero delta). Called
    /// for every scroll event, whichever container handles it.
    fn note_scroll_activity(&mut self, phase: TouchPhase, moved: bool, cx: &mut Context<Self>) {
        match phase {
            TouchPhase::Started => self.minimap_gesturing = true,
            TouchPhase::Ended => self.minimap_gesturing = false,
            TouchPhase::Moved => {}
        }
        // macOS-style: reveal only once the scroll actually moves, not on mere
        // finger contact (a delta-free `Started`/stationary `Moved`). A contact
        // that never scrolls thus never shows the bar.
        if moved {
            self.minimap_visible = true;
        }
        // Arm the linger timer on real movement, or on lift (so the 1s countdown
        // starts precisely when the gesture ends). A contact-without-scroll
        // arms nothing, leaving no stray polling task behind.
        if moved || matches!(phase, TouchPhase::Ended) {
            self.arm_minimap_hide(cx);
            cx.notify();
        }
    }

    /// (Re)start the linger timer. After `MINIMAP_HIDE_DELAY`, if fingers are
    /// off the trackpad and the cursor isn't over the bar, fade it out;
    /// otherwise re-check after another delay. Each scroll event / hover change
    /// replaces this task, so the timer only elapses once everything has been
    /// quiet for the full delay (covering momentum, which keeps emitting events,
    /// and resting fingers, which keep `gesturing` set).
    fn arm_minimap_hide(&mut self, cx: &mut Context<Self>) {
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

    /// Abort any in-flight snap glide and release its settle-pin. Called when a
    /// fresh gesture begins, handing the branch's X back to the fingers.
    fn cancel_snap(&mut self) {
        self.snap = None;
        self.snap_pin = None;
    }

    /// Begin (or immediately resolve) a snap glide for `node_id` from its current
    /// resting position. Pages are `page_width + BAND_HEIGHT` apart; the target
    /// branch is the nearest one, biased a page forward/back when the release was
    /// a flick (`last_h_delta` past [`SNAP_FLING_THRESHOLD`]) — clamped to the
    /// available branches. If already aligned we just pin; otherwise we start a
    /// distance-scaled eased glide and kick the frame loop. Called once, on the
    /// finger lift (`TouchPhase::Ended`) of a horizontal gesture.
    fn start_snap(
        &mut self,
        node_id: &'static str,
        page_width: Pixels,
        count: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if count <= 1 {
            return;
        }
        let stride = (page_width + BAND_HEIGHT).as_f32();
        if stride <= 0.0 {
            return;
        }
        let off = match self.scrolls.get(node_id) {
            Some(handle) => handle.offset(),
            None => return,
        };
        let from_x = off.x.as_f32();
        // Pages live at x = -index * stride, so the fractional page is -x / stride.
        let cur = (-from_x) / stride;
        let v = self.last_h_delta.as_f32();
        // A forward flick (content dragged left) carries a negative delta.x.
        let raw = if v <= -SNAP_FLING_THRESHOLD {
            cur.floor() as i64 + 1
        } else if v >= SNAP_FLING_THRESHOLD {
            cur.ceil() as i64 - 1
        } else {
            cur.round() as i64
        };
        let target = raw.clamp(0, count as i64 - 1);
        let to_x = -(target as f32) * stride;
        let dist = (to_x - from_x).abs();
        if dist < 0.5 {
            // Already on a boundary: snap exactly and pin (absorbs stray momentum).
            if let Some(handle) = self.scrolls.get(node_id) {
                handle.set_offset(point(px(to_x), off.y));
            }
            self.snap = None;
            self.snap_pin = Some((node_id, to_x));
            cx.notify();
            return;
        }
        // Scale the glide with distance so a single-branch hop is quick and a
        // multi-branch correction still floats rather than lurches.
        let dur = Duration::from_secs_f32((0.18 + (dist / stride) * 0.16).clamp(0.18, 0.42));
        self.snap = Some(SnapAnim {
            node_id,
            from_x,
            to_x,
            start: Instant::now(),
            duration: dur,
        });
        self.drive_snap(window, cx);
    }

    /// Glide the `node_id` branch scroller to page `index` — a click on that
    /// branch's indicator dot. Reuses the snap glide so it floats into place like
    /// a flick, and pins the destination so any trailing input can't drift it.
    fn glide_to_branch(
        &mut self,
        node_id: &'static str,
        index: usize,
        page_width: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let stride = (page_width + BAND_HEIGHT).as_f32();
        if stride <= 0.0 {
            return;
        }
        let (from_x, off_y) = match self.scrolls.get(node_id) {
            Some(h) => {
                let o = h.offset();
                (o.x.as_f32(), o.y)
            }
            None => return,
        };
        let to_x = -(index as f32) * stride;
        self.cancel_snap();
        let dist = (to_x - from_x).abs();
        if dist < 0.5 {
            if let Some(h) = self.scrolls.get(node_id) {
                h.set_offset(point(px(to_x), off_y));
            }
            self.snap_pin = Some((node_id, to_x));
            cx.notify();
            return;
        }
        let dur = Duration::from_secs_f32((0.18 + (dist / stride) * 0.16).clamp(0.18, 0.42));
        self.snap = Some(SnapAnim {
            node_id,
            from_x,
            to_x,
            start: Instant::now(),
            duration: dur,
        });
        self.drive_snap(window, cx);
    }

    /// One frame of the active snap glide: ease the offset toward the target and,
    /// until it arrives, schedule the next frame. On arrival the offset is pinned
    /// (see [`SpaceView::snap_pin`]) so trailing momentum can't undo it.
    fn drive_snap(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(a) = self.snap else { return };
        let t = (a.start.elapsed().as_secs_f32() / a.duration.as_secs_f32()).clamp(0.0, 1.0);
        let x = a.from_x + (a.to_x - a.from_x) * ease_out_cubic(t);
        if let Some(handle) = self.scrolls.get(a.node_id) {
            let off = handle.offset();
            handle.set_offset(point(px(x), off.y));
        }
        if t >= 1.0 {
            self.snap = None;
            self.snap_pin = Some((a.node_id, a.to_x));
        } else {
            let entity = cx.entity();
            window.on_next_frame(move |window, cx| {
                entity.update(cx, |this, cx| this.drive_snap(window, cx));
            });
        }
        cx.notify();
    }

    /// Re-assert the glide-or-pin X for `node_id` over whatever the built-in
    /// scroll listener just applied. Called from the horizontal scroll handler
    /// (which runs *after* the built-in) so OS momentum arriving during/after the
    /// glide is overwritten with the snapped position instead of moving the page.
    /// A no-op when this node isn't snapping.
    fn reassert_horizontal(&self, node_id: &str) {
        let x = if let Some(a) = self.snap.filter(|a| a.node_id == node_id) {
            let t = (a.start.elapsed().as_secs_f32() / a.duration.as_secs_f32()).clamp(0.0, 1.0);
            a.from_x + (a.to_x - a.from_x) * ease_out_cubic(t)
        } else if let Some((_, x)) = self.snap_pin.filter(|(id, _)| *id == node_id) {
            x
        } else {
            return;
        };
        if let Some(handle) = self.scrolls.get(node_id) {
            let off = handle.offset();
            handle.set_offset(point(px(x), off.y));
        }
    }

    /// Rescale every branch scroller's horizontal offset by `ratio` (the new
    /// page stride over the old) after a window resize. Pages sit at
    /// `x = -index * stride`, so scaling the offset by the stride ratio keeps the
    /// *exact* fractional page position — a snapped branch stays snapped on the
    /// same branch (no glide), and the selected path is invariant to width. Any
    /// in-flight glide / settle-pin is rescaled the same way so it stays
    /// consistent if a resize lands mid-animation.
    fn remap_for_resize(&mut self, ratio: f32) {
        for handle in self.scrolls.values() {
            let off = handle.offset();
            handle.set_offset(point(px(off.x.as_f32() * ratio), off.y));
        }
        if let Some(a) = self.snap.as_mut() {
            a.from_x *= ratio;
            a.to_x *= ratio;
        }
        if let Some((_, x)) = self.snap_pin.as_mut() {
            *x *= ratio;
        }
    }

    /// Which child page a node's scroller is resting on, derived from its scroll
    /// offset. Each page is one viewport wide, so the nearest is
    /// `round(scrolled / page_width)`. Cosmetic — it only highlights the active
    /// page-indicator dot.
    fn active_child_index(&self, node_id: &str, page_width: Pixels, count: usize) -> usize {
        if count <= 1 || page_width <= px(0.) {
            return 0;
        }
        let Some(handle) = self.scrolls.get(node_id) else {
            return 0;
        };
        // Pages are separated by a vertical band, so the stride between page
        // origins is the page width plus that separator.
        let stride = (page_width + BAND_HEIGHT).as_f32();
        let scrolled = (-handle.offset().x).as_f32();
        let idx = (scrolled / stride).round() as i64;
        idx.clamp(0, count as i64 - 1) as usize
    }

    /// The currently-viewed path through the tree as a list of levels. Level 0
    /// is the root alone; each subsequent level is the children of the level
    /// above's active node, paired with that active index.
    fn selected_levels(&self, page_width: Pixels) -> Vec<(Vec<&Node>, usize)> {
        let mut levels = vec![(vec![&self.root], 0usize)];
        let mut node = &self.root;
        while !node.children.is_empty() {
            let active = self.active_child_index(node.id, page_width, node.children.len());
            levels.push((node.children.iter().collect(), active));
            node = &node.children[active];
        }
        levels
    }

    /// The selected leaf — follow the active child at each level until a node
    /// with no replies. Its trailing runway is the one the composer docks into.
    fn selected_leaf(&self, page_width: Pixels) -> &Node {
        let mut node = &self.root;
        while !node.children.is_empty() {
            let active = self.active_child_index(node.id, page_width, node.children.len());
            node = &node.children[active];
        }
        node
    }

    /// Fixed vertical chrome of the composer bar: just the *top* margin (half
    /// `POST_PAD_Y`). The matching bottom padding lives *inside* the scrolled
    /// content (a `pb` on the editor body) instead — so it reads as breathing
    /// room when the draft fits, but becomes usable scroll space when it
    /// overflows. The float-height and runway math are unchanged because the
    /// recorded content height grows by that same half (`record_height` measures
    /// the `pb` too). When docked, `render_active_draft` adds the other half as a gap
    /// above the bar so the band-to-body spacing matches the document.
    fn composer_chrome() -> f32 {
        POST_PAD_Y.as_f32() / 2.0
    }

    /// Height of a branch's trailing runway item: at least one window tall (so
    /// the docked composer can stand alone, distraction-free), or as tall as the
    /// composer's full content if that's larger.
    fn runway_height(&self, window_h: Pixels) -> Pixels {
        let content = self.composer_content_h.borrow().as_f32();
        px(window_h.as_f32().max(Self::composer_chrome() + content))
    }

    /// Bottom padding the scrolling page needs so the end of the selected branch
    /// can clear a *floating* (off-branch) active draft. On the selected branch
    /// the draft's in-flow placeholder already reserves this room (it docks), so
    /// none is needed. Off-branch the floating bar (its content height, capped at
    /// half the window) occludes the bottom of the window:
    ///
    /// - A normal selected leaf is flush at the bottom — pad by the whole bar.
    /// - A selected leaf that is *itself* a draft is window-tall (min-height), so
    ///   it already reserves slack below its content. Pad only for the shortfall:
    ///   zero while its content is short, growing as it approaches/exceeds the
    ///   window — never more than what's missing.
    fn floating_pad(&self, page_width: Pixels, window_h: Pixels) -> f32 {
        let Some(draft_id) = self.active_draft else {
            return 0.0;
        };
        let leaf = self.selected_leaf(page_width);
        if leaf.id == draft_id {
            return 0.0;
        }
        let win = window_h.as_f32();
        let float_bar_h = (Self::composer_chrome() + self.composer_content_h.borrow().as_f32())
            .min(COMPOSER_MAX_FRACTION * win);
        if !self.is_draft(leaf.id) {
            return float_bar_h;
        }
        // Slack the min-window draft already leaves below its content: the post's
        // bottom padding plus any min-height fill. `content` is the draft's
        // natural body height (recorded inline), so `slack` shrinks to one
        // `POST_PAD_Y` as the content grows past the window.
        let pad_y = POST_PAD_Y.as_f32();
        let content = self
            .draft_body_bounds
            .borrow()
            .get(leaf.id)
            .map(|b| b.size.height.as_f32())
            .unwrap_or(0.0);
        let slack = (win - pad_y - content).max(pad_y);
        (float_bar_h - slack).max(0.0)
    }

    /// Rendered height of the *currently selected* path from `node` down to its
    /// leaf (post heights from the previous frame's recorded bounds, plus the
    /// inter-level bands and the trailing runway). This is the dynamic variant's
    /// core: the overall content height — and the minimap scale — track the
    /// active branch (each branch scroller is sized to its selected child), so
    /// switching branches grows/shrinks the document instead of holding one
    /// static maximum.
    fn selected_subtree_height(&self, node: &Node, page_width: Pixels, window_h: Pixels) -> f32 {
        // The node's in-flow height: a post records into `post_bounds`; the active
        // draft renders an empty placeholder that records into `slot_bounds`.
        let h = self
            .post_bounds
            .borrow()
            .get(node.id)
            .map(|b| b.size.height.as_f32())
            .or_else(|| {
                self.slot_bounds
                    .borrow()
                    .get(node.id)
                    .map(|b| b.size.height.as_f32())
            })
            .unwrap_or(0.0);
        if node.children.is_empty() {
            // The selected leaf. A draft leaf *is* the editing surface (at least a
            // window tall), with no trailing separator; a normal leaf ends the
            // conversation with a trailing separator band.
            if self.is_draft(node.id) {
                return h.max(window_h.as_f32());
            }
            return h + BAND_HEIGHT.as_f32();
        }
        let active = self.active_child_index(node.id, page_width, node.children.len());
        h + BAND_HEIGHT.as_f32()
            + self.selected_subtree_height(&node.children[active], page_width, window_h)
    }

    /// A cheap hash of the layout inputs the composer/minimap read from the
    /// *previous* frame (post + runway bounds, composer content height, viewport)
    /// so `render` can schedule exactly one catch-up frame when they change.
    fn minimap_signature(&self, viewport_h: Pixels) -> f32 {
        let mut sig = viewport_h.as_f32() + self.composer_content_h.borrow().as_f32() * 5.0;
        for (id, b) in self.post_bounds.borrow().iter() {
            sig += id.len() as f32 + b.origin.y.as_f32() * 2.0 + b.size.height.as_f32() * 3.0;
        }
        for (id, b) in self.slot_bounds.borrow().iter() {
            sig += id.len() as f32 + b.origin.y.as_f32() * 7.0 + b.size.height.as_f32() * 11.0;
        }
        // Inline draft body heights feed `floating_pad` (and thus the minimap
        // scale), so a content change must trigger a catch-up frame too.
        for (id, b) in self.draft_body_bounds.borrow().iter() {
            sig += id.len() as f32 + b.size.height.as_f32() * 13.0;
        }
        sig
    }

    /// The topology minimap: a right-edge bar whose rows are the levels of the
    /// selected path. Each row's height is the *selected* post's real height
    /// (shared by every column in the row, so the unselected siblings are purely
    /// topological), and the band gaps mirror the real inter-post spacing — so
    /// the selected path is a true spatial map of the document.
    ///
    /// The scale is **fixed**: the longest possible root-to-leaf branch fills the
    /// bar exactly (bottom-pinned), and a shorter selected branch ends partway
    /// down with slack below it — just like the real scroll view. Because the
    /// scale doesn't depend on the current selection, switching a branch leaves
    /// every pixel above the change identical.
    ///
    /// The selected branch is drawn dark where it's on-screen and medium where
    /// it's scrolled off; the unselected siblings (reachable with one horizontal
    /// gesture) are drawn light.
    fn render_minimap(
        &self,
        page_width: Pixels,
        viewport_h: Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        let fg = cx.theme().scrollbar_thumb;
        let light = fg.opacity(0.18);
        let medium = cx.theme().scrollbar_thumb.opacity(0.45);
        let dark = cx.theme().scrollbar_thumb_hover.opacity(0.78);

        let mut container = div()
            .id("minimap")
            .absolute()
            .top_0()
            .bottom_0()
            .right_0()
            .w(MINIMAP_WIDTH);

        let bounds = self.post_bounds.borrow();
        let slot = self.slot_bounds.borrow();
        // A node's painted block as `(top, height)` in screen space. Posts record
        // into `post_bounds` (origin inset by `POST_PAD_Y` — undo it); the active
        // draft's empty placeholder records into `slot_bounds` (no padding).
        let block_of = |id: &str| -> Option<(f32, f32)> {
            if let Some(b) = bounds.get(id) {
                Some((
                    b.origin.y.as_f32() - POST_PAD_Y.as_f32(),
                    b.size.height.as_f32(),
                ))
            } else {
                slot.get(id)
                    .map(|b| (b.origin.y.as_f32(), b.size.height.as_f32()))
            }
        };
        let height_of = |id: &str| block_of(id).map(|(_, h)| h).unwrap_or(0.);

        // Dynamic scale: the *selected* path fills the bar exactly, so switching
        // branches re-scales the minimap to the new active branch (rather than
        // the static tallest-branch denominator the other variant uses). The
        // empty title-bar reserve above the root is part of the scrollable
        // content, so include it in the denominator (and as a spacer below) —
        // that keeps the dark visible-window a consistent size as it slides off
        // the top, instead of "growing" through the reserve.
        let reserve = TITLE_BAR_RESERVE.as_f32();
        let selected_h = self.selected_subtree_height(&self.root, page_width, viewport_h);
        // A floating off-branch draft adds bottom scroll padding to the page (see
        // `floating_pad`); fold it into the denominator + a trailing spacer so the
        // scroll indicator still maps 1:1 to the real scrollable height.
        let pad = self.floating_pad(page_width, viewport_h);
        let total_h = reserve + selected_h + pad;
        if total_h > 0.0 && viewport_h > px(0.) {
            let scale = viewport_h.as_f32() / total_h;
            let levels = self.selected_levels(page_width);
            let mut col = v_flex().w_full();
            // The reserve scrolls off like content: dark while visible at the top.
            let reserve_block = bounds
                .get(self.root.id)
                .map(|b| (b.origin.y.as_f32() - POST_PAD_Y.as_f32() - reserve, reserve));
            col = col.child(selected_column(
                reserve_block,
                0.0,
                viewport_h.as_f32(),
                px(reserve * scale),
                dark,
                medium,
            ));
            for (level, (sibs, active)) in levels.iter().enumerate() {
                if level > 0 {
                    // The separator band's share of the scaled height.
                    col = col.child(div().w_full().h(px(BAND_HEIGHT.as_f32() * scale)));
                }
                // Every column in the row takes the *selected* branch's height.
                let row_h = px(height_of(sibs[*active].id) * scale);
                let mut row = h_flex().w_full().h(row_h).gap(MINIMAP_COL_GAP);
                for (i, sib) in sibs.iter().enumerate() {
                    let cell = if i == *active {
                        selected_column(
                            block_of(sib.id),
                            // Conversation is visible from the window top (y = 0,
                            // under the transparent titlebar) to the window bottom.
                            0.0,
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
            }

            // A non-draft selected leaf ends with a trailing separator band — a
            // transparent gap in the map, mirroring the inter-level bands. A draft
            // leaf is the editing surface itself and has no trailing band (its
            // placeholder is already the final level above).
            let leaf_is_draft = levels
                .last()
                .map(|(sibs, active)| self.is_draft(sibs[*active].id))
                .unwrap_or(false);
            if !leaf_is_draft {
                col = col.child(div().w_full().h(px(BAND_HEIGHT.as_f32() * scale)));
            }
            // The floating-draft bottom padding: empty scroll room at the very end.
            if pad > 0.0 {
                col = col.child(div().w_full().h(px(pad * scale)));
            }

            container = container.child(col);
        }

        // macOS-style overlay visibility: shown instantly, faded out after the
        // linger timer. Hovering the bar keeps it up.
        if self.minimap_visible {
            container
                .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                    this.minimap_hovered = *hovered;
                    this.arm_minimap_hide(cx);
                    cx.notify();
                }))
                .into_any_element()
        } else if self.minimap_fade_gen == 0 {
            // Never shown yet — stay invisible without a fade-in flash.
            container.opacity(0.0).into_any_element()
        } else {
            container
                .with_animation(
                    ("minimap-fade", self.minimap_fade_gen),
                    Animation::new(MINIMAP_FADE),
                    |el, delta| el.opacity(1.0 - delta),
                )
                .into_any_element()
        }
    }

    /// Render a node's whole subtree: its post, then (if it has replies) a
    /// separator band and a horizontal scroller whose pages are each child's
    /// *entire subtree* (recursively). The recursion builds the nesting that
    /// carries the tree structure.
    fn render_node(
        &self,
        node: &Node,
        page_width: Pixels,
        window_h: Pixels,
        cx: &Context<Self>,
    ) -> Div {
        let theme = cx.theme();
        let is_draft = self.is_draft(node.id);
        let is_active = self.active_draft == Some(node.id);
        // A leaf on the selected path is the "final" post — the bottom of the
        // visible conversation. For a draft (always a leaf) this also means it is
        // *within* the selected branch (so its overlay docks rather than floats).
        let is_final = node.children.is_empty() && self.selected_leaf(page_width).id == node.id;

        // Definite pixel widths (not `w_full`) throughout the subtree: a page
        // lives inside an `overflow_x_scroll`, where percentage widths resolve
        // against the scroller's (effectively unbounded) content size rather
        // than the page, collapsing everything to content width.
        let mut column = v_flex().w(page_width);

        if is_draft && is_active {
            // The active draft's editor floats in the overlay (see
            // `render_active_draft`); its in-flow slot is an empty placeholder
            // that reserves scroll room and records the dock line + minimap
            // bounds. A *final* draft reserves a full window-tall "runway" so it
            // can stand alone at the bottom; otherwise just its content height.
            let slot_h = if is_final {
                self.runway_height(window_h)
            } else {
                px(Self::composer_chrome() + self.composer_content_h.borrow().as_f32())
            };
            column = column.child(
                div()
                    .w(page_width)
                    .h(slot_h)
                    .flex_none()
                    .child(record_bounds(self.slot_bounds.clone(), node.id)),
            );
        } else {
            column =
                column.child(self.render_post(node, page_width, window_h, is_draft, is_final, cx));
        }

        if node.children.is_empty() {
            // A draft is always a childless leaf you can't reply to, so it simply
            // ends — no trailing separator (and on the selected branch a draft is
            // therefore always the very last node). Every other leaf ends with a
            // separator band whose "+" starts a reply here.
            if is_draft {
                return column;
            }
            return column.child(self.render_band(page_width, node, cx));
        }

        let count = node.children.len();
        let active = self.active_child_index(node.id, page_width, count);
        // The dotted band carries the clickable branch indicators + a "+".
        column = column.child(self.render_band(page_width, node, cx));

        // The branch scroller: each page is one child's full subtree. The
        // innermost scroller under the cursor claims a horizontal scroll (stops
        // propagation) so it doesn't also move the scrollers above it; vertical
        // deltas fall through to the outer page scroller.
        // Dynamic height: size the scroller to the *selected* child's subtree
        // (`items_start` + explicit height), so the document height tracks the
        // active branch instead of the tallest sibling. Taller non-selected
        // branches overflow and are clipped (off-screen horizontally until
        // snapped to); the separators get an explicit `h_full` to match.
        let strip_h = self.selected_subtree_height(&node.children[active], page_width, window_h);
        let node_id = node.id;
        let mut strip = h_flex()
            .id(SharedString::from(format!("{}-children", node.id)))
            .w(page_width)
            .h(px(strip_h))
            .items_start()
            .overflow_x_scroll()
            // The gesture's locked axis decides which container moves. For a
            // horizontal gesture the innermost branch scroller under the cursor
            // claims it (its built-in already applied delta.x) and stops, so no
            // ancestor scroller also moves. For a vertical gesture we *don't*
            // stop: we only undo the stray horizontal step the built-in applied
            // here, and let the event bubble so the one outer scroller does the
            // vertical scroll **natively** — preserving smooth momentum. Every
            // ancestor branch scroller undoes its own step the same way.
            .on_scroll_wheel(cx.listener(move |this, ev: &ScrollWheelEvent, window, cx| {
                let delta = ev.delta.pixel_delta(window.line_height());
                let moved = !delta.x.is_zero() || !delta.y.is_zero();
                this.note_scroll_activity(ev.touch_phase, moved, cx);

                // Snap bookkeeping: a fresh gesture aborts any in-flight glide and
                // returns control to the fingers; track the latest horizontal step
                // as the release velocity for the flick decision.
                match ev.touch_phase {
                    TouchPhase::Started => {
                        this.cancel_snap();
                        this.last_h_delta = px(0.);
                    }
                    TouchPhase::Moved => {
                        if !delta.x.is_zero() {
                            this.last_h_delta = delta.x;
                        }
                    }
                    TouchPhase::Ended => {}
                }
                // Capture the gesture's locked axis before `resolve_scroll_axis`
                // clears it on `Ended`, so we only snap a horizontal gesture.
                let locked = this.scroll_axis;

                match this.resolve_scroll_axis(ev.touch_phase, delta) {
                    ScrollAxis::Horizontal => {
                        cx.stop_propagation();
                        // A glide (or its settle-pin) owns this node's X until the
                        // next gesture: re-assert it over the momentum delta the
                        // built-in listener just applied, so trailing OS momentum
                        // can't drift the page off the snapped branch.
                        this.reassert_horizontal(node_id);
                    }
                    ScrollAxis::Vertical => {
                        if !delta.x.is_zero()
                            && let Some(handle) = this.scrolls.get(node_id)
                        {
                            let off = handle.offset();
                            handle.set_offset(point(off.x - delta.x, off.y));
                        }
                    }
                }

                // On finger lift of a horizontal gesture, glide to the nearest
                // (or flicked-toward) branch.
                if matches!(ev.touch_phase, TouchPhase::Ended)
                    && locked == Some(ScrollAxis::Horizontal)
                {
                    this.start_snap(node_id, page_width, count, window, cx);
                }
            }));
        // Only respond to the dominant axis — never let a pure-vertical delta
        // bleed into horizontal motion via gpui's cross-axis fallback.
        strip.style().restrict_scroll_to_axis = Some(true);
        // Clip a taller non-selected branch's vertical overflow to the selected
        // height — it's off-screen horizontally, so this just keeps it from
        // bleeding past the scroller during a switch.
        strip.style().overflow.y = Some(Overflow::Hidden);
        if let Some(handle) = self.scrolls.get(node.id) {
            strip = strip.track_scroll(handle);
        }
        for (i, child) in node.children.iter().enumerate() {
            // A vertical separator between branches — same thickness and ground
            // as the horizontal band, so a scroll across the seam reads as a
            // real boundary between two branches.
            if i > 0 {
                strip = strip.child(div().w(BAND_HEIGHT).flex_none().h_full().bg(theme.muted));
            }
            // The page wrapper carries the child id so the per-post markdown
            // editors (which all share the element id "markdown-editor") get
            // distinct global ids across branches.
            strip = strip.child(
                div()
                    .id(SharedString::from(child.id))
                    .w(page_width)
                    .flex_none()
                    .child(self.render_node(child, page_width, window_h, cx)),
            );
        }
        column.child(strip)
    }

    /// One post: the right-aligned byline gutter (system font) beside the
    /// centered reading column (Newsreader prose). A normal post is read-only
    /// markdown with an author/time byline; a *draft* post (`is_draft`) is an
    /// editable editor under a "Draft" byline. The active draft never reaches
    /// here — its editor renders in the floating overlay instead (see
    /// [`SpaceView::render_node`]).
    fn render_post(
        &self,
        node: &Node,
        page_width: Pixels,
        window_h: Pixels,
        is_draft: bool,
        is_final: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let id = node.id;

        let body_width = (page_width - GUTTER_WIDTH * 1.5 - GUTTER_GAP * 2)
            .min(BODY_MAX_WIDTH)
            .max(px(240.));

        // Byline — UI/chrome voice (system font). A draft stands in a faint
        // "Draft" where the author would be; a post shows its bold name + time.
        let byline = if is_draft {
            v_flex()
                .w(GUTTER_WIDTH)
                .flex_none()
                .items_end()
                .pt_5()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::BOLD)
                        .text_color(theme.muted_foreground)
                        .child("Draft"),
                )
        } else {
            v_flex()
                .w(GUTTER_WIDTH)
                .flex_none()
                .items_end()
                .pt_5()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::BOLD)
                        .text_color(theme.foreground)
                        .child(SharedString::from(node.author)),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(node.created_at)),
                )
        };

        // Body — narrative voice (Newsreader prose). Read-only for a post,
        // editable for a draft. A definite width makes the markdown's
        // height-for-width measurement correct.
        let mut body = div().w(body_width).child(
            MarkdownEditor::new(&self.bodies[id])
                .style(prose_style(cx))
                .disabled(!is_draft),
        );
        if is_draft {
            // Record the draft's natural content height (the body's own size,
            // not the post's min-window block) for `floating_pad`.
            body = body.child(record_bounds(self.draft_body_bounds.clone(), id));
        }

        let mut post = h_flex()
            .relative()
            .w(page_width)
            .py(POST_PAD_Y)
            .justify_center()
            .items_start()
            .gap(GUTTER_GAP)
            .child(byline)
            .child(body)
            .pr(GUTTER_WIDTH / 2. + GUTTER_GAP)
            // Records this post's painted bounds for the minimap (no layout
            // effect — absolute overlay).
            .child(record_bounds(self.post_bounds.clone(), id));

        if is_draft && is_final {
            // The final draft is the bottom editing surface: at least a window
            // tall so it stands alone (matches the active-draft runway).
            post = post.min_h(window_h);
        }
        if is_draft {
            // Clicking a draft (re)activates it even when it already had focus —
            // e.g. after Escape, where no fresh `Focus` event would fire.
            post = post.on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| this.activate_draft(id, cx)),
            );
        }
        post
    }

    /// The faint full-bleed separator band between a post and what follows it.
    /// When the post has more than one reply the band carries page-indicator dots,
    /// one per branch: each is clickable (glides the scroller to that branch), the
    /// active one is highlighted, and a branch whose subtree holds an unsent draft
    /// is drawn in the theme's `info` color. It always carries a "+" affordance
    /// that appends a draft reply as a new child of `node` (a new sibling branch
    /// for a post with replies, or the first reply to a leaf). A draft is a
    /// childless leaf you can't reply to, so it renders no band at all — there is
    /// never a separator following a draft to suppress.
    fn render_band(&self, page_width: Pixels, node: &Node, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let info = theme.info;
        let active_color = theme.muted_foreground;
        let band_bg = theme.muted;
        let plus_fg = theme.muted_foreground;
        let plus_bg = theme.background.opacity(0.55);
        let plus_bg_hover = theme.background;

        let parent_id = node.id;
        let count = node.children.len();

        let mut row = h_flex().items_center().gap_3();
        if count >= 2 {
            let active = self.active_child_index(node.id, page_width, count);
            let dots = h_flex()
                .gap_1()
                .children(node.children.iter().enumerate().map(|(i, child)| {
                    let base_color = if self.subtree_has_draft(child) {
                        info
                    } else {
                        active_color
                    };

                    let color = if i == active {
                        base_color
                    } else {
                        base_color.alpha(0.5)
                    };

                    div()
                        .id(SharedString::from(format!("dot-{parent_id}-{i}")))
                        .flex_none()
                        .p(px(3.))
                        .cursor_pointer()
                        .child(div().size(px(5.)).rounded_full().bg(color))
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.glide_to_branch(parent_id, i, page_width, window, cx);
                        }))
                }));
            row = row.child(dots);
        }
        row = row.child(
            div()
                .id(SharedString::from(format!("plus-{parent_id}")))
                .size(px(20.))
                .flex_none()
                .rounded_full()
                .items_center()
                .justify_center()
                .text_color(plus_fg)
                .bg(plus_bg)
                .cursor_pointer()
                .hover(move |s| s.bg(plus_bg_hover))
                .child("+")
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.create_draft(parent_id, window, cx);
                })),
        );
        h_flex()
            .w(page_width)
            .h(BAND_HEIGHT)
            .bg(band_bg)
            .items_center()
            .justify_center()
            .child(row)
    }

    /// The active draft's floating editor: the editable markdown pane pinned over
    /// the bottom of the window, positioned and styled like a post (gutter,
    /// centered body, matching margins). Renders nothing when no draft is active.
    ///
    /// **Float vs. dock.** Its body grows with content up to
    /// [`COMPOSER_MAX_FRACTION`] of the window, then scrolls internally — that's
    /// the *floating* bar, bottom-aligned, available while reading. When the
    /// draft is on the selected branch and the user scrolls toward it, its in-flow
    /// placeholder rises; once its top would pass above the floating bar's top,
    /// the bar *docks* to it (`top = min(float_top, slot_top)`) and grows into the
    /// page, so it reads as one continuous scroll ending in the editor. When the
    /// draft is *not* on the selected branch (the user swiped to a sibling while
    /// editing), it always floats. As the bar grows past the content, the internal
    /// scroll offset clamps to zero on its own, revealing the whole draft.
    fn render_active_draft(
        &self,
        page_width: Pixels,
        window_h: Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        let Some(draft_id) = self.active_draft else {
            return div().into_any_element();
        };
        let theme = cx.theme();
        let body_width = (page_width - GUTTER_WIDTH * 1.5 - GUTTER_GAP * 2)
            .min(BODY_MAX_WIDTH)
            .max(px(240.));

        // On the selected branch the overlay docks to its placeholder; off it
        // (swiped to a sibling while editing), it always floats.
        let on_path = self.selected_leaf(page_width).id == draft_id;

        let win = window_h.as_f32();
        let chrome = Self::composer_chrome();
        let content = self.composer_content_h.borrow().as_f32();
        // Half the document's inter-post spacing. It's the bar's top padding
        // (fixed chrome) and — mirrored as a `pb` inside the scrolled content —
        // the bottom padding; the dock adds the *other* half as a gap above the
        // bar so a docked composer reads with the full `POST_PAD_Y` band-to-body
        // spacing while a floating one is only half as thick.
        let half_pad = POST_PAD_Y.as_f32() / 2.0;

        // Floating bar: chrome + content, capped at half the window. Bottom-pinned.
        let float_bar_h = (chrome + content).min(COMPOSER_MAX_FRACTION * win);
        let float_top = win - float_bar_h;
        // Dock: if the active draft's placeholder top (plus the half-spacing gap)
        // has risen above the floating top, follow it up (and grow). One-frame-
        // lagged like the minimap. Off the selected branch we don't dock at all.
        let slot_top = if on_path {
            self.slot_bounds
                .borrow()
                .get(draft_id)
                .map(|b| b.origin.y.as_f32())
        } else {
            None
        };
        let top_y = match slot_top {
            Some(s) => float_top.min(s + half_pad),
            None => float_top,
        };

        // Floating (overlaying the conversation) vs docked (following the page).
        let overlayed = top_y >= float_top - 0.5;
        let docked = !overlayed;
        // Cached for the scroll handlers' session-ownership decision.
        self.composer_overlayed.set(overlayed);

        // Docked height grows with scroll position, interpolated linearly: the
        // floating height at the dock threshold (`top_y == float_top`), up to
        // full height by the time the composer's top reaches the top of the
        // window content. This resolves a partway internal scroll smoothly —
        // as the bar grows the body fits more content, so the (frozen) offset
        // clamps toward the top. Floating keeps its bottom-pinned height.
        let full_h = (content + chrome).max(win);
        let bar_h = if docked {
            let denom = (float_top - TITLE_BAR_RESERVE.as_f32()).max(1.0);
            let progress = ((float_top - top_y) / denom).clamp(0.0, 1.0);
            float_bar_h + progress * (full_h - float_bar_h)
        } else {
            float_bar_h
        };
        let body_h = (bar_h - chrome).max(0.0);

        // Internal scroll position (≤ 0); < 0 means content is hidden above.
        let scrolled_down = self.composer_scroll.offset().y.as_f32() < -0.5;

        // Byline gutter — a faint "Draft" standing in for the author slot, so the
        // body aligns with the posts.
        let byline = v_flex()
            .w(GUTTER_WIDTH)
            .flex_none()
            .items_end()
            .pt_5()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::BOLD)
                    .text_color(theme.muted_foreground)
                    .child("Draft"),
            );

        let mut body = div()
            .id("composer-body")
            .w(body_width)
            .h(px(body_h))
            .overflow_y_scroll()
            .track_scroll(&self.composer_scroll)
            // Session ownership: this handler fires first when the cursor is over
            // the composer, so a gesture that *starts* here claims the session —
            // a floating composer owns it (`Composer`: internal scroll only, the
            // page is locked out even at the limits), a docked one defers to the
            // body (`Body`: page scrolls, internal scroll frozen). The built-in
            // listener already scrolled `composer_scroll`; for a body-owned
            // session we restore it to its last-render value so it stays put.
            .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, window, cx| {
                if matches!(ev.touch_phase, TouchPhase::Started) {
                    this.scroll_owner = None;
                }
                let delta_y = ev.delta.pixel_delta(window.line_height()).y.as_f32();
                if delta_y == 0.0 {
                    return;
                }
                let floating = this.composer_overlayed.get();
                let owner = *this.scroll_owner.get_or_insert(if floating {
                    ScrollOwner::Composer
                } else {
                    ScrollOwner::Body
                });
                match owner {
                    ScrollOwner::Composer => cx.stop_propagation(),
                    ScrollOwner::Body => {
                        let off = this.composer_scroll.offset();
                        this.composer_scroll
                            .set_offset(point(off.x, px(this.composer_prev_off_y)));
                    }
                }
            }))
            .child(
                // Auto-height inner content so the recorder canvas captures the
                // editor's *natural* height (the scroll viewport above clips it).
                // The bottom padding lives here, inside the scrolled content: it
                // shows as breathing room under a short draft, but scrolls away
                // (becoming usable space) once the draft overflows.
                div()
                    .w_full()
                    .pb(px(half_pad))
                    .child(MarkdownEditor::new(&self.bodies[draft_id]).style(prose_style(cx)))
                    .child(record_height(
                        self.composer_content_h.clone(),
                        cx.entity().downgrade(),
                    )),
            );
        body.style().restrict_scroll_to_axis = Some(true);

        let mut composer = div()
            .id("composer")
            .absolute()
            .left_0()
            .right_0()
            .top(px(top_y))
            .h(px(bar_h))
            // Opaque so the conversation behind the bar is occluded.
            .bg(theme.background)
            // Escape while editing deactivates the draft (it stays as a plain
            // inline editor). The focused editor is a descendant, so its unhandled
            // Escape bubbles to here.
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _, cx| {
                if ev.keystroke.key == "escape" {
                    this.deactivate_active_draft(cx);
                }
            }))
            .child(
                h_flex()
                    .w(page_width)
                    .h_full()
                    .pt(px(half_pad))
                    .justify_center()
                    .items_start()
                    .gap(GUTTER_GAP)
                    .pr(GUTTER_WIDTH / 2. + GUTTER_GAP)
                    .child(byline)
                    .child(body),
            );
        // Inner top scroll-shadow: whenever the composer's content is scrolled
        // down (floating *or* docked), a small, subtle shadow at the scroll
        // viewport's top edge signals content above. It spans the *whole pane*
        // (full width, over the gutter too), even though only the editor's
        // contents scroll — and is pinned here (a sibling of the scroll) so it
        // stays put as the content moves under it.
        if scrolled_down {
            composer = composer.child(
                div()
                    .absolute()
                    .top(px(half_pad))
                    .left_0()
                    .right_0()
                    .h(px(8.))
                    .bg(linear_gradient(
                        180.,
                        linear_color_stop(hsla(0., 0., 0., 0.08), 0.),
                        linear_color_stop(hsla(0., 0., 0., 0.), 1.),
                    )),
            );
        }
        if overlayed {
            // Floating: cast a subtle shadow up over the conversation behind it.
            // (Docked, it's part of the page and casts nothing.)
            composer = composer.shadow(vec![
                BoxShadow::new(px(0.), px(-3.), hsla(0., 0., 0., 0.12)).blur_radius(px(18.)),
            ]);
        }
        composer.into_any_element()
    }

    /// A draggable band across the top of the window standing in for the
    /// (now transparent) titlebar. On macOS `WindowControlArea::Drag` is a
    /// no-op, so dragging is wired explicitly: arm on mouse-down, then call
    /// `window.start_window_move()` on the first move while armed.
    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("title-bar")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(TITLE_BAR_RESERVE)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.should_move_window = true),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.should_move_window = false),
            )
            .on_mouse_move(cx.listener(|this, _, window, _| {
                if this.should_move_window {
                    this.should_move_window = false;
                    window.start_window_move();
                }
            }))
            .on_double_click(|_, window, _| window.titlebar_double_click())
    }
}

impl Render for SpaceView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let viewport = window.viewport_size();
        let page_width = viewport.width;

        // Window resize: branch offsets are absolute pixels but pages are a
        // window-width apart, so a width change would slide every offset relative
        // to its branches — leaving them mid-page and even reselecting a branch
        // on a discrete jump (zoom / fullscreen / tiling). Remap every offset by
        // the stride ratio so the selected branch and exact position survive any
        // resize. We have no resize event with before/after, but the previous
        // frame's width is enough — even a single-frame jump diffs cleanly here.
        if let Some(prev) = self.last_page_width
            && prev != page_width
            && prev > px(0.)
            && page_width > px(0.)
        {
            let ratio = (page_width + BAND_HEIGHT).as_f32() / (prev + BAND_HEIGHT).as_f32();
            self.remap_for_resize(ratio);
        }
        self.last_page_width = Some(page_width);

        // The minimap reads bounds recorded during the *previous* paint. When
        // those (or the viewport) change — reflow, scroll, selection — schedule
        // one follow-up frame so the minimap catches up; this converges once the
        // layout is stable, so it's not a per-frame spin.
        let sig = self.minimap_signature(viewport.height);
        if self.minimap_sig.to_bits() != sig.to_bits() {
            self.minimap_sig = sig;
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        }

        // Keep the composer's frozen-scroll baseline in step with its actual
        // internal offset between gestures (it can shift as the pane docks/grows
        // and the offset clamps), so a body-owned session restores to the right
        // value the first time the composer slides under the cursor.
        self.composer_prev_off_y = self.composer_scroll.offset().y.as_f32();

        // The whole tree is one recursively-nested element rooted at A.
        let tree = self.render_node(&self.root, page_width, viewport.height, cx);

        div()
            .relative()
            .size_full()
            .bg(theme.background)
            // Components/chrome render in the system UI font (the theme leaves
            // `font_family` unset); only prose opts into Newsreader.
            .font_family(theme.font_family.clone())
            .text_color(theme.foreground)
            .child(self.render_title_bar(cx))
            .child({
                let mut scroll = div()
                    .id("scroll")
                    .size_full()
                    .overflow_y_scroll()
                    // Catch scrolls that don't hit a branch scroller (e.g. over
                    // the root post) for the minimap's show/hide, and claim the
                    // scroll session for the body: reaching the page scroller
                    // means the composer didn't own this gesture.
                    .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, window, cx| {
                        let delta = ev.delta.pixel_delta(window.line_height());
                        let moved = !delta.x.is_zero() || !delta.y.is_zero();
                        this.note_scroll_activity(ev.touch_phase, moved, cx);
                        if matches!(ev.touch_phase, TouchPhase::Started) {
                            this.scroll_owner = None;
                        }
                        if !delta.y.is_zero() {
                            this.scroll_owner.get_or_insert(ScrollOwner::Body);
                        }
                    }))
                    .child(
                        v_flex()
                            .w_full()
                            .pt(TITLE_BAR_RESERVE)
                            // Room to scroll the selected branch's tail clear of a
                            // floating (off-branch) active draft; zero otherwise.
                            .pb(px(self.floating_pad(page_width, viewport.height)))
                            .child(tree),
                    );
                // Vertical only: a horizontal swipe over a region with no branch
                // scroller (e.g. the root post) must not scroll the page sideways
                // (or, via the fallback, vertically).
                scroll.style().restrict_scroll_to_axis = Some(true);
                scroll
            })
            // The active draft's editor overlays the conversation (below the
            // minimap so the scroll indicator stays visible over it). Renders
            // nothing when no draft is active — the default resting state.
            .child(self.render_active_draft(page_width, viewport.height, cx))
            // Topology minimap, painted last so it sits over everything.
            .child(self.render_minimap(page_width, viewport.height, cx))
    }
}

/// An invisible, zero-layout-impact overlay that records its own painted height
/// into `cell` each frame. Placed inside the composer's (auto-height) content so
/// we capture the editor's natural height even though the scroll viewport above
/// it clips what's shown.
///
/// When the measured height *changes*, it schedules one follow-up frame so the
/// composer resizes the same frame the content settles. Without this, the height
/// is computed from the previous frame's recording, so a single edit (Enter,
/// paste) wouldn't be reflected until the *next* interaction — the recorded
/// value is written during paint, after the height was already computed.
fn record_height(cell: Rc<RefCell<Pixels>>, view: WeakEntity<SpaceView>) -> impl IntoElement {
    canvas(
        move |bounds, window, _| {
            let h = bounds.size.height;
            if (cell.borrow().as_f32() - h.as_f32()).abs() > 0.5 {
                *cell.borrow_mut() = h;
                let view = view.clone();
                window.on_next_frame(move |_, cx| {
                    view.update(cx, |_, cx| cx.notify()).ok();
                });
            }
        },
        |_, _, _, _| {},
    )
    .absolute()
    .size_full()
}

/// An invisible, zero-layout-impact overlay that records its (absolute) painted
/// bounds into `map` under `id` each frame. Placed over a post so the minimap
/// can read the post's height and on-screen position.
fn record_bounds(
    map: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,
    id: &'static str,
) -> impl IntoElement {
    canvas(
        move |bounds, _, _| {
            map.borrow_mut().insert(id, bounds);
        },
        |_, _, _, _| {},
    )
    .absolute()
    .size_full()
}

/// The minimap cell for the *selected* branch at a level: a full-height column
/// split into medium (scrolled-off) and dark (on-screen) spans, derived from
/// the post's painted bounds against the visible region.
///
/// Takes the block's true on-screen `(top, height)` (screen space) and the
/// visible region `[vis_top, vis_bot]` (also screen space) that the dark span is
/// clipped against. Posts apply the `origin.y - POST_PAD_Y` correction before
/// calling (the runway, having no top padding, passes its recorded bounds
/// directly):
///
/// - **`origin.y - POST_PAD_Y`.** The bounds-recorder `canvas`
///   (`.absolute().size_full()`) is laid out at the post's *content-box* origin
///   (inset by the post's vertical padding) yet sized to the full element, so
///   its recorded `origin.y` sits `POST_PAD_Y` too low while its `height` is
///   already the true block height. Subtracting `POST_PAD_Y` recovers the top.
///
/// The visible region `[vis_top, vis_bot]` is supplied by the caller (screen
/// space), not assumed: the conversation is visible from screen `y = 0` (it
/// slides under the *transparent* titlebar — `render_title_bar` is painted
/// behind the content and has no background) to `viewport_h`. An earlier version
/// hard-coded `vis_top = TITLE_BAR_RESERVE`, which pushed the dark span down by
/// that reserve; the empty reserve is instead represented by its own spacer row
/// at the top of the minimap, so the dark window stays a consistent size.
fn selected_column(
    block: Option<(f32, f32)>,
    vis_top: f32,
    vis_bot: f32,
    col_h: Pixels,
    dark: Hsla,
    medium: Hsla,
) -> Div {
    let Some((top, height)) = block else {
        return div().w_full().h(col_h).bg(medium);
    };
    let height = height.max(1.0);
    let vt = top.max(vis_top);
    let vb = (top + height).min(vis_bot);
    if vb <= vt {
        // Fully scrolled off-screen (or fully behind the composer).
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

/// `MarkdownStyle` for prose bodies: Newsreader at a book size/leading with a
/// gentle heading ramp. `from_theme` seeds the system font, so we override the
/// family back to Newsreader for narrative content.
fn prose_style(cx: &App) -> MarkdownStyle {
    let mut style = MarkdownStyle::from_theme(cx)
        .font_size(PROSE_FONT_SIZE)
        .line_height(rems(PROSE_LINE_HEIGHT))
        .paragraph_gap(rems(1.5))
        .heading_base_font_size(PROSE_FONT_SIZE)
        .heading_font_size(|level, base| match level {
            1 => base * 1.5,
            2 => base * 1.25,
            3 => base * 1.125,
            _ => base,
        });
    style.font_family = theme::FONT_FAMILY.into();
    style
}

/// Depth-first search for the id of the node whose direct child is `id` (used to
/// detach a deleted draft from its parent). `None` if `id` is the root or absent.
fn parent_of(node: &Node, id: &str) -> Option<&'static str> {
    for child in &node.children {
        if child.id == id {
            return Some(node.id);
        }
        if let Some(found) = parent_of(child, id) {
            return Some(found);
        }
    }
    None
}

/// Depth-first search for the node with `id`, returning a mutable reference so a
/// new draft child can be appended (see [`SpaceView::create_draft`]).
fn node_mut<'a>(node: &'a mut Node, id: &str) -> Option<&'a mut Node> {
    if node.id == id {
        return Some(node);
    }
    for child in &mut node.children {
        if let Some(found) = node_mut(child, id) {
            return Some(found);
        }
    }
    None
}

/// Walk the tree once, creating a read-only markdown-editor state for every
/// node and a horizontal scroll handle for every node that has replies.
fn build_state(
    node: &Node,
    bodies: &mut HashMap<&'static str, Entity<MarkdownEditorState>>,
    scrolls: &mut HashMap<&'static str, ScrollHandle>,
    window: &mut Window,
    cx: &mut Context<SpaceView>,
) {
    let markdown = node.content.to_string();
    let state = cx.new(|cx| {
        MarkdownEditorState::with_state(
            EditorState {
                markdown,
                ..Default::default()
            },
            window,
            cx,
        )
    });
    bodies.insert(node.id, state);
    if !node.children.is_empty() {
        scrolls.insert(node.id, ScrollHandle::new());
    }
    for child in &node.children {
        build_state(child, bodies, scrolls, window, cx);
    }
}

/// The seed tree for the experiment. A has four direct replies (B, C, G, H) so
/// snapping can be tested across — and *into the middle of* — a row of branches;
/// G carries its own reply (I) so an intermediate branch isn't a dead end.
///
/// ```text
/// 0          A
///        / / | \
/// 1     B  C G  H
///      /|    |
/// 2   D E    I
///       |
/// 3     F
/// ```
fn sample_tree() -> Node {
    Node {
        id: "A",
        author: "Mara Vance",
        created_at: "10:03 AM",
        content: "I've started treating every note I write as the first room of a house I might never finish building. You walk in, set down one true sentence, and leave the door open behind you.\n\n\
            For years I wrote into documents – clean, walled, finished-feeling things. A document *wants* to be done. It pulls the last paragraph toward it like a tide. But the thoughts I care about most aren't done; they're held in a kind of suspension, waiting for someone – a friend, a stranger, some patient machine – to disturb them. A document has no room for that disturbance. It has margins, but no *space*.\n\n\
            So this is the small wager of writing here instead: that the note stays exactly as I wrote it, at the size I wrote it, and the conversation grows around it rather than burying it. The first post is load-bearing. Everything else leans on it.\n\n\
            *A reply should feel less like a comment and more like someone pulling a chair up to the same table.*\n\n\
            What I don't yet know is how deep that table can get before it stops feeling like one table. At some point a thread becomes a forest, and you lose the path back to the clearing where you started. Maybe that's fine. Maybe the clearing should always be one gesture away.",
        children: vec![
            Node {
                id: "B",
                author: "Kimi K2",
                created_at: "10:04 AM",
                content: "There's a quiet radicalism in refusing the document's pull toward done. Most tools treat a note as a draft of something else, a means to a finished end. You're treating it as a place worth standing in.\n\n\
                    The risk you name – that a place can sprawl – is real. But a sprawling place is still a place. A failed essay is a failure; a rambling house is just a house with too many rooms, and you can always close a door. The structure forgives more than the document does.",
                children: vec![
                    Node {
                        id: "D",
                        author: "Mara Vance",
                        created_at: "10:06 AM",
                        content: "Sprawl is exactly the fear, though. A failed essay at least has an ending – you know when you've lost. A place can just keep adding rooms until no one remembers where the front door was.",
                        children: vec![],
                    },
                    Node {
                        id: "E",
                        author: "Mara Vance",
                        created_at: "10:07 AM",
                        content: "Maybe the front door is the whole point. You don't memorize a house. You keep returning to the one room that matters, and the rest stays *available* without ever being *demanded* of you.",
                        children: vec![Node {
                            id: "F",
                            author: "Kimi K2",
                            created_at: "10:08 AM",
                            content: "Right – availability without obligation. The forest is fine as long as there's always a path marked back to the clearing. The branching only hurts when it erases the trunk.",
                            children: vec![],
                        }],
                    },
                ],
            },
            Node {
                id: "C",
                author: "Kimi K2",
                created_at: "10:04 AM",
                content: "The load-bearing-post idea is the whole thing for me. In chat apps the first message is the most disposable – it scrolls into the dark within the hour. Here you're proposing the opposite: the origin stays lit, and everything is measured against it.\n\n\
                    Does that put a lot of pressure on the first sentence, though?",
                children: vec![],
            },
            Node {
                id: "G",
                author: "Kimi K2",
                created_at: "10:05 AM",
                content: "I keep snagging on \"one gesture away.\" That's a spatial promise, not just a metaphor – it says the trunk is never more than a single motion behind you, no matter how far out on a limb you've climbed.\n\n\
                    If that holds, depth stops being scary. You can wander because returning is cheap.",
                children: vec![Node {
                    id: "I",
                    author: "Mara Vance",
                    created_at: "10:09 AM",
                    content: "Exactly – cheap return is what makes the wandering safe. The cost of a branch isn't the branch; it's forgetting how to get home. Keep that free and the tree can grow as wild as it likes.",
                    children: vec![],
                }],
            },
            Node {
                id: "H",
                author: "Mara Vance",
                created_at: "10:05 AM",
                content: "And there's the fourth direction this could go: not deeper and not back, but *sideways* – a reply that belongs to the root as much as the others, sitting beside them rather than beneath any one of them.\n\n\
                    Four chairs at the same table, not a queue.",
                children: vec![],
            },
        ],
    }
}

fn main() {
    let app = gpui_platform::application().with_assets(gpui_component_assets::Assets);

    app.run(move |cx| {
        // This must be called before using any GPUI Component features.
        gpui_component::init(cx);

        // Install the theme.
        theme::install(cx);

        // Initialize the markdown editor key bindings.
        gpui_markdown_editor::init(cx);

        // Open an initial window. `cx` here is `&mut App`, so the window opens
        // synchronously (no `cx.spawn`, which would hand back an `AsyncApp`
        // that `WindowBounds::centered` can't take).
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::centered(size(px(900.), px(680.)), cx)),
                titlebar: Some(TitlebarOptions {
                    appears_transparent: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| {
                theme::observe_window_appearance(window);

                // Root the view in the window.
                let view = cx.new(|cx| SpaceView::new(window, cx));
                cx.new(|cx| Root::new(view, window, cx))
            },
        )
        .expect("Failed to open window.");
    });
}
