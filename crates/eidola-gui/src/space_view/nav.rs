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
        if dist < 0.5 {
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

    /// Navigate the page to a node (a minimap column click): select the branch
    /// leading to it, then scroll so its top sits comfortably in view.
    pub(crate) fn navigate_to_node(
        &mut self,
        node_id: gpui::SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let viewport = window.viewport_size();
        let streaming = self.space.read(cx).is_streaming();
        let tree = self.effective_tree(viewport.width, streaming);
        if super::model::node_ref(&tree, &node_id).is_none() {
            return;
        }
        self.select_path_to(&tree, &node_id, viewport.width);
        if let Some(doc_top) =
            self.selected_path_doc_top(&tree, &node_id, viewport.width, viewport.height)
        {
            // Land the node's top ~28% down the window.
            let y = (viewport.height.as_f32() * 0.28 - doc_top).min(0.0);
            let off = self.page_scroll.offset();
            self.page_scroll.set_offset(point(off.x, px(y)));
        }
        cx.notify();
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
}
