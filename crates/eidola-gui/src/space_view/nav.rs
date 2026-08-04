//! Horizontal branch navigation: the scroll-gesture bookkeeping and the
//! "scroll-snap" glide physics.
//!
//! macOS forwards its already-decayed momentum deltas straight into the scroll
//! offset and exposes no momentum-end signal (only the finger lift,
//! `TouchPhase::Ended`), so a CSS-style "let momentum land on a snap point"
//! isn't reachable by cooperating with the OS. Instead we take over at finger
//! lift: capture the release velocity, then drive our own eased glide to the
//! target branch and suppress the OS momentum that would otherwise fight it.
//!
//! The pure pieces (the axis/owner enums, the easing, the flick→target-index
//! decision) live here and are unit-tested; the frame-driven `drive_snap` /
//! `start_snap` methods (which need `Window`/`Context`) live on `SpaceView` in
//! [`super::layout`]-adjacent impl blocks.

use std::time::{Duration, Instant};

use gpui::SharedString;

/// Minimum per-event horizontal finger step (px) at release that counts as a
/// directional *flick* — above this, the snap advances/retreats one branch in
/// the flick's direction; below it, the snap goes to the nearest branch.
pub const SNAP_FLING_THRESHOLD: f32 = 8.0;

/// Which axis a scroll gesture is locked to. Determined from the first real
/// movement of a gesture and held until the gesture ends, so a mostly-vertical
/// scroll never nudges the branches sideways (and vice versa).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScrollAxis {
    Horizontal,
    Vertical,
}

/// Who owns a vertical scroll *session* — decided by where the gesture starts
/// and held until the next gesture begins (so it spans momentum and direction
/// reversals). A gesture that starts over the conversation, or over a docked
/// composer, scrolls the page and freezes the composer's internal scroll; one
/// that starts over a *floating* composer is owned by the composer (internal
/// scroll only — the page never moves, even at the composer's scroll limits).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScrollOwner {
    Body,
    Composer,
}

/// An in-flight "scroll-snap" glide of one branch scroller toward a page
/// boundary.
#[derive(Clone, Debug)]
pub struct SnapAnim {
    /// The branch scroller (node with children) being animated.
    pub node_id: SharedString,
    /// Horizontal scroll offset (`ScrollHandle` x, ≤ 0) at the start and end of
    /// the glide.
    pub from_x: f32,
    pub to_x: f32,
    /// Wall-clock start and total duration of the glide.
    pub start: Instant,
    pub duration: Duration,
}

/// An in-flight glide of the **page** (vertical) scroll toward a destination
/// the reader asked to be taken to — a footnote's source, a highlight's
/// referencer, "See in context". The horizontal twin is [`SnapAnim`]; both ride
/// the same easing and the same frame loop shape.
///
/// Navigation is animated because a jump gives the reader no way to tell
/// *where* they were taken from: the glide carries the intervening page past
/// them, which is the whole difference between "this is elsewhere in the same
/// conversation" and "the page changed". Gesture-driven scrolls (the wheel, a
/// minimap drag) are never animated — they are already the reader's own motion.
#[derive(Clone, Copy, Debug)]
pub struct PageGlide {
    /// Page scroll `y` (≤ 0) at the start and end of the glide.
    pub from_y: f32,
    pub to_y: f32,
    pub start: Instant,
    pub duration: Duration,
}

/// The current eased `y` of a page glide at progress fraction `t` (clamped).
pub fn glide_y_at(a: &PageGlide, t: f32) -> f32 {
    a.from_y + (a.to_y - a.from_y) * ease_out_cubic(t.clamp(0.0, 1.0))
}

/// Elapsed progress `t` (0..=1) of a page glide as of now.
pub fn glide_progress(a: &PageGlide) -> f32 {
    (a.start.elapsed().as_secs_f32() / a.duration.as_secs_f32().max(f32::EPSILON)).clamp(0.0, 1.0)
}

/// Cubic ease-out: fast departure, gentle arrival — reads as a thrown page
/// floating to rest on its snap point.
pub fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t;
    1.0 - u * u * u
}

/// The current eased x of a glide given its progress fraction `t` (clamped).
pub fn snap_x_at(a: &SnapAnim, t: f32) -> f32 {
    a.from_x + (a.to_x - a.from_x) * ease_out_cubic(t.clamp(0.0, 1.0))
}

/// Elapsed progress `t` (0..=1) of a glide as of now.
pub fn snap_progress(a: &SnapAnim) -> f32 {
    (a.start.elapsed().as_secs_f32() / a.duration.as_secs_f32().max(f32::EPSILON)).clamp(0.0, 1.0)
}

/// The target page index a release should snap to, given the current offset
/// `from_x` (≤ 0), the page `stride`, the release velocity `v` (the last
/// horizontal step; a forward flick — content dragged left — is negative), and
/// the branch `count`. A flick past [`SNAP_FLING_THRESHOLD`] biases one page in
/// its direction; otherwise it rounds to the nearest. Clamped to `0..count`.
pub fn snap_target_index(from_x: f32, stride: f32, v: f32, count: usize) -> usize {
    if count <= 1 || stride <= 0.0 {
        return 0;
    }
    // Pages live at x = -index * stride, so the fractional page is -x / stride.
    let cur = (-from_x) / stride;
    let raw = if v <= -SNAP_FLING_THRESHOLD {
        cur.floor() as i64 + 1
    } else if v >= SNAP_FLING_THRESHOLD {
        cur.ceil() as i64 - 1
    } else {
        cur.round() as i64
    };
    raw.clamp(0, count as i64 - 1) as usize
}

/// Glide duration scaled by distance so a single-branch hop is quick and a
/// multi-branch correction still floats rather than lurches.
pub fn snap_duration(dist: f32, stride: f32) -> Duration {
    Duration::from_secs_f32((0.18 + (dist / stride.max(1.0)) * 0.16).clamp(0.18, 0.42))
}

/// The *proximity* snap decision for a variable-height vertical page (the
/// onboarding flow's slide stack). Unlike [`snap_target_index`] — which is
/// *mandatory* and always returns a page to land on — this mirrors CSS
/// `scroll-snap-type: proximity`: it returns `Some(index)` only when a released
/// gesture came to rest **near** a slide boundary, and `None` when it ended
/// deep in a slide's content (stay put — never yank a reader off the prose they
/// were scrolling through).
///
/// - `tops` are the content-space y of each slide's top, ascending, with
///   `tops[0] == 0` (measured from the live child bounds, so it copes with
///   slides taller than the window).
/// - `viewport_top` is the content-space y currently at the viewport's top
///   (`-offset.y`).
/// - `v` is the release velocity (the last vertical finger step); a downward
///   flick — revealing later slides — is **negative**, matching
///   [`snap_target_index`]'s convention.
/// - `proximity` is the px band around a boundary within which a rest snaps.
///
/// A flick past [`SNAP_FLING_THRESHOLD`] only redirects when the release was
/// *already* near a boundary (it advances/retreats one slide in the flick's
/// direction); a flick deep in content is fast reading, not navigation, and
/// returns `None`.
pub fn proximity_snap_target(
    tops: &[f32],
    viewport_top: f32,
    v: f32,
    proximity: f32,
) -> Option<usize> {
    let (nearest, dist) = tops
        .iter()
        .enumerate()
        .map(|(i, &t)| (i, (t - viewport_top).abs()))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;

    let flick_down = v <= -SNAP_FLING_THRESHOLD;
    let flick_up = v >= SNAP_FLING_THRESHOLD;
    if flick_down || flick_up {
        // Directional intent, but honored only near a boundary — a flick in the
        // middle of a long slide is content scrolling, so leave it to momentum.
        if dist > proximity {
            return None;
        }
        let target = if flick_down {
            tops.iter()
                .position(|&t| t > viewport_top + 1.0)
                .unwrap_or(tops.len() - 1)
        } else {
            tops.iter()
                .rposition(|&t| t < viewport_top - 1.0)
                .unwrap_or(0)
        };
        return Some(target);
    }

    // Gentle release: snap only if it rests inside the proximity band.
    (dist <= proximity).then_some(nearest)
}

// ---------------------------------------------------------------------------
// The frame-driven glide + gesture bookkeeping, on `SpaceView`.
// ---------------------------------------------------------------------------

use super::model::TreeNode;
use super::{BAND_HEIGHT, SpaceView};

use gpui::{Context, IsZero, Pixels, Point, ScrollHandle, TouchPhase, Window, point, px};

impl SpaceView {
    /// Lock the scroll gesture to an axis. Gesture boundaries (`Started`/`Ended`)
    /// clear the lock; the first real movement of the next gesture sets it.
    pub(crate) fn resolve_scroll_axis(
        &mut self,
        phase: TouchPhase,
        delta: Point<Pixels>,
    ) -> ScrollAxis {
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

    /// Abort any in-flight glide and release its settle-pin.
    pub(crate) fn cancel_snap(&mut self) {
        self.snap = None;
        self.snap_pin = None;
    }

    /// Glide the page to `y` (a page scroll offset, ≤ 0) instead of jumping —
    /// the navigation motion (see [`PageGlide`]). A short hop, an equal
    /// destination, or a destination the page is already at lands immediately.
    pub(crate) fn glide_page_to(&mut self, y: f32, window: &mut Window, cx: &mut Context<Self>) {
        let from_y = self.page_scroll.offset().y.as_f32();
        let dist = (y - from_y).abs();
        // A reader who asked for less motion gets the destination, not the
        // journey. `App::reduce_motion` is gpui's own flag (it also drives every
        // `Animation` element); nothing feeds it from the platform at this pin —
        // see the note in `crates/eidola-gui/AGENTS.md`.
        if dist < 1.0 || cx.reduce_motion() {
            self.set_page_scroll_y(y);
            cx.notify();
            return;
        }
        let window_h = crate::chrome::content_size(window).height.as_f32().max(1.0);
        self.page_glide.set(Some(PageGlide {
            from_y,
            to_y: y,
            start: std::time::Instant::now(),
            duration: snap_duration(dist, window_h),
        }));
        self.drive_page_glide(window, cx);
    }

    /// **The one door for a programmatic page scroll that lands at once.**
    /// Retires any glide in flight, then writes `y` (the horizontal offset is
    /// preserved — branch selection is the strips' business).
    ///
    /// It exists because a glide *owns* `page_scroll` for its whole duration:
    /// [`Self::apply_page_glide`] writes each frame's position from the
    /// trajectory alone, so anything that set the offset between two glide
    /// frames was silently overwritten on the next one — a keyboard move, a
    /// caret reveal, a settling post, all undone until the glide landed. The
    /// gesture paths said this already ([`Self::cancel_page_glide`] from
    /// `note_scroll_activity` / `minimap_press`); the instant family had no
    /// such seam, and "remember to cancel first" is exactly the discipline a
    /// new call site forgets. **Nothing else in `space_view` writes
    /// `page_scroll`'s offset** — only this and the glide's own frame body.
    ///
    /// The reader's own navigation therefore wins by construction: whoever
    /// writes last owns the page, and a glide can never take it back.
    pub(crate) fn set_page_scroll_y(&self, y: f32) {
        self.cancel_page_glide();
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    /// One frame of the page glide: ease the offset toward the target and,
    /// until it arrives, schedule the next frame.
    pub(crate) fn drive_page_glide(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(t) = self.page_glide.get().as_ref().map(glide_progress) else {
            return;
        };
        if !self.apply_page_glide(t) {
            let entity = cx.entity();
            window.on_next_frame(move |window, cx| {
                entity.update(cx, |this, cx| this.drive_page_glide(window, cx));
            });
        }
        cx.notify();
    }

    /// Place the page at the glide's eased position for progress `t`, retiring
    /// the glide once it arrives. Returns whether it arrived. Split from the
    /// frame loop so the landing is testable without a real clock (no test
    /// dispatcher pumps `on_next_frame`).
    /// The glide is the **only** writer that does not go through
    /// [`Self::set_page_scroll_y`] — it is the motion that seam takes the page
    /// away from.
    pub(crate) fn apply_page_glide(&mut self, t: f32) -> bool {
        let Some(a) = self.page_glide.get() else {
            return true;
        };
        let off = self.page_scroll.offset();
        self.page_scroll
            .set_offset(point(off.x, px(glide_y_at(&a, t))));
        if t >= 1.0 {
            self.page_glide.set(None);
            return true;
        }
        false
    }

    /// Drop any in-flight page glide — the reader's own scrolling or a gesture
    /// now owns the offset. Programmatic scrolls get this for free through
    /// [`Self::set_page_scroll_y`]; the direct callers are the gesture paths,
    /// which take the page over without writing an offset of their own (gpui's
    /// built-in scroller does that).
    pub(crate) fn cancel_page_glide(&self) {
        self.page_glide.set(None);
    }

    /// Begin (or immediately resolve) a snap glide for `node_id` from its
    /// current resting position to the nearest/flicked-toward branch.
    pub(crate) fn start_snap(
        &mut self,
        node_id: gpui::SharedString,
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
        let off = match self.scrolls.get(&node_id) {
            Some(handle) => handle.offset(),
            None => return,
        };
        let from_x = off.x.as_f32();
        let target = snap_target_index(from_x, stride, self.last_h_delta.as_f32(), count);
        let to_x = -(target as f32) * stride;
        let dist = (to_x - from_x).abs();
        if dist < 0.5 {
            if let Some(handle) = self.scrolls.get(&node_id) {
                handle.set_offset(point(px(to_x), off.y));
            }
            self.snap = None;
            self.snap_pin = Some((node_id, to_x));
            cx.notify();
            return;
        }
        self.snap = Some(SnapAnim {
            node_id,
            from_x,
            to_x,
            start: std::time::Instant::now(),
            duration: snap_duration(dist, stride),
        });
        self.drive_snap(window, cx);
    }

    /// Glide a branch scroller to page `index` (a click on its indicator dot).
    pub(crate) fn glide_to_branch(
        &mut self,
        node_id: gpui::SharedString,
        index: usize,
        page_width: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let stride = (page_width + BAND_HEIGHT).as_f32();
        if stride <= 0.0 {
            return;
        }
        let (from_x, off_y) = match self.scrolls.get(&node_id) {
            Some(h) => {
                let o = h.offset();
                (o.x.as_f32(), o.y)
            }
            None => return,
        };
        let to_x = -(index as f32) * stride;
        self.cancel_snap();
        let dist = (to_x - from_x).abs();
        // Navigation, so it honors reduce-motion (see `glide_page_to`).
        if dist < 0.5 || cx.reduce_motion() {
            if let Some(h) = self.scrolls.get(&node_id) {
                h.set_offset(point(px(to_x), off_y));
            }
            self.snap_pin = Some((node_id, to_x));
            cx.notify();
            return;
        }
        self.snap = Some(SnapAnim {
            node_id,
            from_x,
            to_x,
            start: std::time::Instant::now(),
            duration: snap_duration(dist, stride),
        });
        self.drive_snap(window, cx);
    }

    /// One frame of the active glide: ease the offset toward the target and,
    /// until it arrives, schedule the next frame; on arrival pin it.
    pub(crate) fn drive_snap(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(a) = self.snap.clone() else { return };
        let t = snap_progress(&a);
        let x = snap_x_at(&a, t);
        if let Some(handle) = self.scrolls.get(&a.node_id) {
            let off = handle.offset();
            handle.set_offset(point(px(x), off.y));
        }
        if t >= 1.0 {
            self.snap = None;
            self.snap_pin = Some((a.node_id.clone(), a.to_x));
        } else {
            let entity = cx.entity();
            window.on_next_frame(move |window, cx| {
                entity.update(cx, |this, cx| this.drive_snap(window, cx));
            });
        }
        cx.notify();
    }

    /// Re-assert the glide-or-pin x for `node_id` over whatever the built-in
    /// scroll listener just applied (so OS momentum can't drift the page off the
    /// snapped branch). A no-op when this node isn't snapping/pinned.
    pub(crate) fn reassert_horizontal(&self, node_id: &str) {
        let x = if let Some(a) = self.snap.as_ref().filter(|a| a.node_id == node_id) {
            snap_x_at(a, snap_progress(a))
        } else if let Some((_, x)) = self.snap_pin.as_ref().filter(|(id, _)| id == node_id) {
            *x
        } else {
            return;
        };
        if let Some(handle) = self.scrolls.get(node_id) {
            let off = handle.offset();
            handle.set_offset(point(px(x), off.y));
        }
    }

    /// Nudge a branch scroller by `-delta_x` (undo the built-in scroller's stray
    /// horizontal step during a vertical gesture).
    pub(crate) fn undo_horizontal_nudge(handle: &ScrollHandle, delta_x: Pixels) {
        let off = handle.offset();
        handle.set_offset(point(off.x - delta_x, off.y));
    }

    /// Rescale every branch offset (and any in-flight glide/pin) by `ratio`
    /// after a resize, keeping the selected branch + exact position invariant.
    pub(crate) fn remap_for_resize(&mut self, ratio: f32) {
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

    /// Select the branch leading to `target` at every level (set each ancestor's
    /// scroller to the child on the path). Used to bring an off-branch composer
    /// onto the selected path.
    pub(crate) fn select_path_to(&mut self, roots: &[TreeNode], target: &str, page_width: Pixels) {
        let Some(path) = super::model::path_ids(roots, target) else {
            return;
        };
        let stride = (page_width + BAND_HEIGHT).as_f32();
        for pair in path.windows(2) {
            let (parent_id, child_id) = (&pair[0], &pair[1]);
            let idx = super::model::node_ref(roots, parent_id)
                .and_then(|p| p.children.iter().position(|c| c.id == *child_id));
            if let Some(idx) = idx
                && let Some(handle) = self.scrolls.get(parent_id)
            {
                let off = handle.offset();
                handle.set_offset(point(px(-(idx as f32) * stride), off.y));
            }
        }
        self.cancel_snap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ease_endpoints() {
        assert!((ease_out_cubic(0.0) - 0.0).abs() < 1e-6);
        assert!((ease_out_cubic(1.0) - 1.0).abs() < 1e-6);
        assert!(
            ease_out_cubic(0.5) > 0.5,
            "ease-out is ahead at the midpoint"
        );
    }

    #[test]
    fn page_glide_eases_between_its_endpoints() {
        let g = PageGlide {
            from_y: -1000.0,
            to_y: 0.0,
            start: Instant::now(),
            duration: Duration::from_millis(300),
        };
        assert!((glide_y_at(&g, 0.0) - g.from_y).abs() < 1e-3);
        assert!((glide_y_at(&g, 1.0) - g.to_y).abs() < 1e-3);
        // Clamped outside 0..=1, and ahead of linear at the midpoint.
        assert!((glide_y_at(&g, 2.0) - g.to_y).abs() < 1e-3);
        assert!(glide_y_at(&g, 0.5) > -500.0);
    }

    #[test]
    fn nearest_snap_rounds() {
        // At x = -110 with stride 100, nearest page is 1.
        assert_eq!(snap_target_index(-110.0, 100.0, 0.0, 4), 1);
        // At x = -140, nearest is 1; -160 rounds to 2.
        assert_eq!(snap_target_index(-140.0, 100.0, 0.0, 4), 1);
        assert_eq!(snap_target_index(-160.0, 100.0, 0.0, 4), 2);
    }

    #[test]
    fn flick_biases_one_page() {
        // Sitting on page 1 (-100), a forward flick (negative v) advances to 2.
        assert_eq!(snap_target_index(-100.0, 100.0, -20.0, 4), 2);
        // A backward flick (positive v) retreats to 0.
        assert_eq!(snap_target_index(-100.0, 100.0, 20.0, 4), 0);
    }

    #[test]
    fn snap_target_clamps_to_branches() {
        assert_eq!(snap_target_index(-1000.0, 100.0, -50.0, 3), 2);
        assert_eq!(snap_target_index(50.0, 100.0, 50.0, 3), 0);
        assert_eq!(snap_target_index(-100.0, 100.0, 0.0, 1), 0);
    }

    #[test]
    fn proximity_snaps_only_near_a_boundary() {
        // Variable-height slides: boundaries at 0, 100, 250, 400.
        let tops = [0.0, 100.0, 250.0, 400.0];
        // Resting near boundary 1 (dist 20 < 30): a gentle release snaps to it.
        assert_eq!(proximity_snap_target(&tops, 120.0, 0.0, 30.0), Some(1));
        // Resting in the dead middle of a long slide (nearest dist 70 > 30):
        // stay put — this is the "reading a long slide" case.
        assert_eq!(proximity_snap_target(&tops, 170.0, 0.0, 30.0), None);
    }

    #[test]
    fn proximity_flick_redirects_only_near_a_boundary() {
        let tops = [0.0, 100.0, 250.0, 400.0];
        // A downward flick just below boundary 1 advances to the next boundary.
        assert_eq!(proximity_snap_target(&tops, 105.0, -20.0, 30.0), Some(2));
        // An upward flick just above boundary 2 (near it, in slide 1's tail)
        // retreats to the previous boundary.
        assert_eq!(proximity_snap_target(&tops, 245.0, 20.0, 30.0), Some(1));
        // A flick deep in a long slide is fast reading, not navigation — no snap.
        assert_eq!(proximity_snap_target(&tops, 170.0, -20.0, 30.0), None);
    }

    #[test]
    fn proximity_empty_is_none() {
        assert_eq!(proximity_snap_target(&[], 0.0, 0.0, 30.0), None);
    }
}
