use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use eidola_gui::theme;
use gpui::*;
use gpui_component::{ActiveTheme, InteractiveElementExt, Root, h_flex, v_flex};
use gpui_markdown_editor::{EditorState, MarkdownEditor, MarkdownEditorState, MarkdownStyle};

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
/// Because a branch scroller is as tall as its tallest child subtree, a shorter
/// sibling leaves empty space *below* it — the root view is taller than any one
/// branch needs, but that slack is always at the bottom.
pub struct SpaceView {
    /// The conversation tree (a single root for this experiment).
    root: Node,
    /// One read-only markdown-editor state per node, keyed by node id.
    bodies: HashMap<&'static str, Entity<MarkdownEditorState>>,
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

    /// A cheap hash of the minimap's inputs (recorded post bounds + viewport
    /// height) so `render` can tell when a redraw is warranted.
    fn minimap_signature(&self, viewport_h: Pixels) -> f32 {
        let mut sig = viewport_h.as_f32();
        for (id, b) in self.post_bounds.borrow().iter() {
            sig += id.len() as f32 + b.origin.y.as_f32() * 2.0 + b.size.height.as_f32() * 3.0;
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
        let height_of = |id: &str| bounds.get(id).map(|b| b.size.height.as_f32()).unwrap_or(0.);

        // Fixed scale: the tallest possible branch fills the bar, independent of
        // what's currently selected.
        let longest = max_path_height(&self.root, &bounds);
        if longest > 0.0 && viewport_h > px(0.) {
            let scale = viewport_h.as_f32() / longest;
            let levels = self.selected_levels(page_width);
            let mut col = v_flex().w_full();
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
                            bounds.get(sib.id).copied(),
                            viewport_h,
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
    fn render_node(&self, node: &Node, page_width: Pixels, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        // Definite pixel widths (not `w_full`) throughout the subtree: a page
        // lives inside an `overflow_x_scroll`, where percentage widths resolve
        // against the scroller's (effectively unbounded) content size rather
        // than the page, collapsing everything to content width.
        let mut column = v_flex()
            .w(page_width)
            .child(self.render_post(node, page_width, cx));

        if node.children.is_empty() {
            return column;
        }

        let count = node.children.len();
        let active = self.active_child_index(node.id, page_width, count);
        column = column.child(render_band(page_width, count, active, theme));

        // The branch scroller: each page is one child's full subtree. The
        // innermost scroller under the cursor claims a horizontal scroll (stops
        // propagation) so it doesn't also move the scrollers above it; vertical
        // deltas fall through to the outer page scroller.
        // `items_stretch` so the vertical branch separators (and short branch
        // pages) fill the scroller's full height — the slack of a shorter branch
        // ends up at its bottom.
        let node_id = node.id;
        let mut strip = h_flex()
            .id(SharedString::from(format!("{}-children", node.id)))
            .w(page_width)
            .items_stretch()
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
        if let Some(handle) = self.scrolls.get(node.id) {
            strip = strip.track_scroll(handle);
        }
        for (i, child) in node.children.iter().enumerate() {
            // A vertical separator between branches — same thickness and ground
            // as the horizontal band, so a scroll across the seam reads as a
            // real boundary between two branches.
            if i > 0 {
                strip = strip.child(div().w(BAND_HEIGHT).flex_none().bg(theme.muted));
            }
            // The page wrapper carries the child id so the per-post markdown
            // editors (which all share the element id "markdown-editor") get
            // distinct global ids across branches.
            strip = strip.child(
                div()
                    .id(SharedString::from(child.id))
                    .w(page_width)
                    .flex_none()
                    .child(self.render_node(child, page_width, cx)),
            );
        }
        column.child(strip)
    }

    /// One post: the right-aligned byline gutter (system font) beside the
    /// centered reading column (Newsreader prose, read-only markdown).
    fn render_post(&self, node: &Node, page_width: Pixels, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        let body_width = (page_width - GUTTER_WIDTH * 1.5 - GUTTER_GAP * 2)
            .min(BODY_MAX_WIDTH)
            .max(px(240.));

        // Byline — UI/chrome voice (system font): bold name over a muted time.
        let byline = v_flex()
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
            );

        // Body — narrative voice (Newsreader prose), read-only markdown. A
        // definite width makes the markdown's height-for-width measurement
        // correct.
        let body = div().w(body_width).child(
            MarkdownEditor::new(&self.bodies[node.id])
                .style(prose_style(cx))
                .disabled(true),
        );

        h_flex()
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
            .child(record_bounds(self.post_bounds.clone(), node.id))
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

        // The whole tree is one recursively-nested element rooted at A.
        let tree = self.render_node(&self.root, page_width, cx);

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
                    // the root post) for the minimap's show/hide.
                    .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, window, cx| {
                        let delta = ev.delta.pixel_delta(window.line_height());
                        let moved = !delta.x.is_zero() || !delta.y.is_zero();
                        this.note_scroll_activity(ev.touch_phase, moved, cx);
                    }))
                    .child(v_flex().w_full().pt(TITLE_BAR_RESERVE).child(tree));
                // Vertical only: a horizontal swipe over a region with no branch
                // scroller (e.g. the root post) must not scroll the page sideways
                // (or, via the fallback, vertically).
                scroll.style().restrict_scroll_to_axis = Some(true);
                scroll
            })
            // Topology minimap, painted last so it sits over everything.
            .child(self.render_minimap(page_width, viewport.height, cx))
    }
}

/// The tallest root-to-leaf path height (posts + bands) using recorded post
/// heights — the fixed denominator the minimap scales against, so the longest
/// possible branch exactly fills the bar.
fn max_path_height(node: &Node, bounds: &HashMap<&'static str, Bounds<Pixels>>) -> f32 {
    let h = bounds
        .get(node.id)
        .map(|b| b.size.height.as_f32())
        .unwrap_or(0.0);
    if node.children.is_empty() {
        return h;
    }
    let deepest = node
        .children
        .iter()
        .map(|c| max_path_height(c, bounds))
        .fold(0.0_f32, f32::max);
    h + BAND_HEIGHT.as_f32() + deepest
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
/// Two corrections turn the recorded bounds into the post's true on-screen
/// block:
///
/// - **`origin.y - POST_PAD_Y`.** The bounds-recorder `canvas`
///   (`.absolute().size_full()`) is laid out at the post's *content-box* origin
///   (inset by the post's vertical padding) yet sized to the full element, so
///   its recorded `origin.y` sits `POST_PAD_Y` too low while its `height` is
///   already the true block height. Subtracting `POST_PAD_Y` recovers the top.
/// - **visible top = `TITLE_BAR_RESERVE`, not 0.** The content rests
///   `TITLE_BAR_RESERVE` below the window top (the transparent-titlebar reserve)
///   and that band is overlaid, so it's the real visible edge. Using 0 left a
///   dead-zone where the first `TITLE_BAR_RESERVE` px of scrolling didn't move
///   the dark span.
fn selected_column(
    post: Option<Bounds<Pixels>>,
    viewport_h: Pixels,
    col_h: Pixels,
    dark: Hsla,
    medium: Hsla,
) -> Div {
    let Some(pb) = post else {
        return div().w_full().h(col_h).bg(medium);
    };
    let top = pb.origin.y.as_f32() - POST_PAD_Y.as_f32();
    let height = pb.size.height.as_f32().max(1.0);
    let vis_top = top.max(TITLE_BAR_RESERVE.as_f32());
    let vis_bot = (top + height).min(viewport_h.as_f32());
    if vis_bot <= vis_top {
        // Fully scrolled off-screen.
        return div().w_full().h(col_h).bg(medium);
    }
    let ch = col_h.as_f32();
    let above = ((vis_top - top) / height).clamp(0.0, 1.0) * ch;
    let visible = ((vis_bot - vis_top) / height).clamp(0.0, 1.0) * ch;
    let below = (ch - above - visible).max(0.0);
    v_flex()
        .w_full()
        .h(col_h)
        .child(div().w_full().h(px(above)).bg(medium))
        .child(div().w_full().h(px(visible)).bg(dark))
        .child(div().w_full().h(px(below)).bg(medium))
}

/// The faint full-bleed separator band between a post and its replies. When the
/// post has more than one reply, the band carries a row of page-indicator dots
/// (the active branch highlighted); a single reply shows a plain band, so a
/// non-branching conversation reads like a sequential list.
fn render_band(
    page_width: Pixels,
    count: usize,
    active: usize,
    theme: &gpui_component::Theme,
) -> Div {
    let mut band = h_flex()
        .w(page_width)
        .h(BAND_HEIGHT)
        .bg(theme.muted)
        .items_center()
        .justify_center();
    if count >= 2 {
        band = band.child(h_flex().gap_2().children((0..count).map(|i| {
            div().size(px(5.)).rounded_full().bg(if i == active {
                theme.muted_foreground
            } else {
                theme.border
            })
        })));
    }
    band
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
