//! Drafts + the composer — the unsent-reply model.
//!
//! A band's "+" creates a [`Draft`](super::Draft): a local UI node with its own
//! editor, attached to the persisted tree as a leaf of the post it replies to.
//! Drafts **persist** when deselected (they keep their content and their place
//! in the tree, taking real vertical space and tinting their branch's
//! navigation dots), exactly like the `examples/eidola_space.rs` mockup.
//!
//! One draft is *active* at a time — the focused one — and adopts the
//! floating/docking composer behavior: its editor floats over the bottom, its
//! in-flow slot a placeholder. Focusing another draft (or Escape) retires the
//! current one; a retired **empty** draft is deleted, leaving no trace.
//!
//! `⌘↩` (`Send`) persists the active draft and streams a reply; `⌘⇧↩`
//! (`PostOnly`) persists it without asking. Both consume the draft.

use gpui::{
    AnyElement, AppContext, BoxShadow, Context, Focusable, InteractiveElement, IntoElement,
    KeyDownEvent, ParentElement, ScrollWheelEvent, StatefulInteractiveElement, Styled, TouchPhase,
    Window, div, hsla, linear_color_stop, linear_gradient, point, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::{MarkdownEditor, MarkdownEditorEvent, MarkdownEditorState};

use std::collections::HashSet;

use crate::loadable::Loadable;

use super::layout::body_width;
use super::model::{self, TreeNode};
use super::nav::ScrollOwner;
use super::{
    COMPOSER_MAX_FRACTION, Draft, GUTTER_GAP, GUTTER_WIDTH, POST_PAD_Y, PostOnly, Send, SpaceView,
    TITLE_BAR_RESERVE, prose_style,
};

impl SpaceView {
    // -- Draft lifecycle ---------------------------------------------------

    /// Mint a draft node replying to `parent`: a fresh editor wired to activate
    /// on focus and submit on the Enter chords, pushed into `drafts`. Returns its
    /// id. Shared by the focused-fork path ([`Self::create_draft`]) and the
    /// auto-ensured docked tail drafts ([`Self::sync_tail_drafts`]).
    fn create_draft_node(
        &mut self,
        parent: Option<gpui::SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::SharedString {
        self.next_draft_seq += 1;
        let id = gpui::SharedString::from(format!("draft-{}", self.next_draft_seq));
        let editor = cx.new(|cx| MarkdownEditorState::new(window, cx));

        // The draft's editor drives both activation (focus) and submit
        // (`PressEnter`): focusing it makes it the active draft; the modified
        // Enter chords route to the save-vs-ask gestures.
        let sub_id = id.clone();
        let sub =
            cx.subscribe_in(
                &editor,
                window,
                move |this, _editor, event, window, cx| match event {
                    MarkdownEditorEvent::Focus if this.active_draft.as_ref() != Some(&sub_id) => {
                        this.activate_draft(sub_id.clone(), cx);
                    }
                    MarkdownEditorEvent::PressEnter {
                        secondary: true,
                        shift,
                    } => {
                        if *shift {
                            this.post_only(&PostOnly, window, cx);
                        } else {
                            this.submit(&Send, window, cx);
                        }
                    }
                    _ => {}
                },
            );

        self.drafts.push(Draft {
            id: id.clone(),
            parent,
            editor,
            _sub: sub,
        });
        id
    }

    /// Open a draft replying to `parent` (a band's "+" / ⌘N): make it the active
    /// (floating) composer and focus it. The branch selection + page dock happen
    /// on the next render (`pending_select`), against the real tree.
    pub(crate) fn create_draft(
        &mut self,
        parent: Option<gpui::SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let id = self.create_draft_node(parent, window, cx);
        let focus = self.drafts.last().unwrap().editor.read(cx).focus_handle(cx);
        self.activate_draft(id.clone(), cx);
        self.pending_select = Some(id);
        window.focus(&focus, cx);
    }

    /// Ensure every branch leaf has a **tail draft** — the always-present,
    /// docked end-of-branch composer that replaces the leaf "+": a draft whose
    /// parent is a leaf (a post nothing replies to; the root `None` for a blank
    /// space). Empty tail drafts persist (docked); empty, inactive drafts whose
    /// parent is *not* a leaf (orphans, or a fork whose branch was committed) are
    /// pruned. Skipped while streaming and until the transcript has loaded (so we
    /// know the real leaves). Runs each frame in `render` (cheap + idempotent).
    pub(crate) fn sync_tail_drafts(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.space.read(cx).is_streaming() {
            return;
        }
        if !matches!(self.space.read(cx).transcript(), Loadable::Loaded { .. }) {
            return;
        }

        let leaf_parents = self.tail_parents(); // leaf action ids (Some-parents)
        let want_root = self.posts.is_empty(); // blank space → root (None) tail draft

        // Ensure the blank-space root tail draft.
        if want_root && !self.drafts.iter().any(|d| d.parent.is_none()) {
            self.create_draft_node(None, window, cx);
        }
        // Ensure a docked tail draft for each leaf that lacks one.
        for parent in &leaf_parents {
            let has = self
                .drafts
                .iter()
                .any(|d| d.parent.as_deref() == Some(parent.as_ref()));
            if !has {
                self.create_draft_node(Some(parent.clone()), window, cx);
            }
        }

        // Prune empty, inactive drafts whose parent is no longer a tail parent
        // (an orphan, or a fork whose branch was committed). Compute the tail-set
        // as a local so the filter borrows no `self` method.
        let leaf_set: HashSet<&str> = leaf_parents.iter().map(|s| s.as_ref()).collect();
        let active = self.active_draft.clone();
        let stale: Vec<gpui::SharedString> = self
            .drafts
            .iter()
            .filter(|d| {
                let is_tail = match d.parent.as_deref() {
                    None => want_root,
                    Some(p) => leaf_set.contains(p),
                };
                Some(&d.id) != active.as_ref() && !is_tail && d.editor.read(cx).is_empty()
            })
            .map(|d| d.id.clone())
            .collect();
        for id in stale {
            self.delete_draft(&id);
        }
    }

    /// Make `id` the active (floating) draft, retiring the previous one (which
    /// is deleted if it was an empty *fork*; an empty tail draft just docks).
    /// Resets the shared composer scroll.
    pub(crate) fn activate_draft(&mut self, id: gpui::SharedString, cx: &mut Context<Self>) {
        if self.active_draft.as_ref() != Some(&id) {
            self.retire_active_draft(cx);
        }
        self.active_draft = Some(id);
        self.composer_scroll.set_offset(point(px(0.), px(0.)));
        self.composer_prev_off_y = 0.0;
        cx.notify();
    }

    /// Deactivate the active draft (Escape / external request). An empty **tail**
    /// draft stays as the docked end-of-branch composer; an empty **fork** draft
    /// (a transient new branch) is deleted; a draft with content stays inline.
    pub(crate) fn deactivate_active_draft(&mut self, cx: &mut Context<Self>) {
        if self.active_draft.is_some() {
            self.retire_active_draft(cx);
            cx.notify();
        }
    }

    /// Clear the active draft; delete it only if it was an **empty fork** (an
    /// abandoned new branch). An empty tail draft is kept (docked) — it's the
    /// always-present composer at the end of its branch.
    fn retire_active_draft(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.active_draft.take() else {
            return;
        };
        let Some(draft) = self.drafts.iter().find(|d| d.id == id) else {
            return;
        };
        let empty = draft.editor.read(cx).is_empty();
        let parent = draft.parent.clone();
        if empty && !self.is_tail_parent(parent.as_deref()) {
            self.delete_draft(&id);
        }
    }

    /// Remove a draft from the tree and forget its editor/state. The parent
    /// scroller re-clamps onto a still-valid branch at render time
    /// (`active_child_index` clamps to the new child count).
    fn delete_draft(&mut self, id: &str) {
        self.drafts.retain(|d| d.id != id);
        self.layout.retain(&|live| live != id);
    }

    // -- Submit ------------------------------------------------------------

    /// Route the composer's outward events when dispatched as actions (tests /
    /// menu). The editor's own subscription (see `create_draft`) is the
    /// production path; this keeps `Send`/`PostOnly` dispatchable.
    pub(crate) fn submit(&mut self, _: &Send, window: &mut Window, cx: &mut Context<Self>) {
        self.send_active_draft(false, window, cx);
    }

    pub(crate) fn post_only(&mut self, _: &PostOnly, window: &mut Window, cx: &mut Context<Self>) {
        self.send_active_draft(true, window, cx);
    }

    /// Persist the active draft (and stream a reply unless `post_only`),
    /// consuming the draft. A no-op while streaming, with no active draft, or on
    /// an empty draft.
    fn send_active_draft(&mut self, post_only: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.space.read(cx).is_streaming() {
            return;
        }
        let Some(active) = self.active_draft.clone() else {
            return;
        };
        let Some(draft) = self.drafts.iter().find(|d| d.id == active) else {
            return;
        };
        let editor = draft.editor.clone();
        let parent = draft.parent.clone();
        let prompt = editor.read(cx).value().trim().to_string();
        if prompt.is_empty() {
            return;
        }

        // The reply antecedent is the draft's parent, but only a real persisted
        // action id is valid (a `None`/synthetic parent → app-core uses the
        // tail).
        let reply_to = parent.and_then(|p| {
            self.posts
                .iter()
                .find(|post| post.action_id.as_deref() == Some(p.as_ref()))
                .and_then(|post| post.action_id.clone())
                .map(|s| s.to_string())
        });

        // Consume the draft (it's becoming a persisted post). Clearing the
        // active slot before `Space::submit` avoids briefly showing the draft
        // beside the optimistic post.
        self.active_draft = None;
        self.delete_draft(&active);
        self.error = None;

        if post_only {
            self.space.update(cx, |s, cx| {
                s.post_only(prompt, reply_to, cx);
            });
        } else {
            let model = self.current_model(cx);
            self.space.update(cx, |s, cx| {
                s.submit(prompt, model, reply_to, cx);
            });
        }
        self.scroll_to_tail(window, cx);
        cx.notify();
    }

    // -- Scrolling ---------------------------------------------------------

    /// Scroll the page so the bottom of the selected branch (the composer /
    /// streaming leaf) sits at the window bottom.
    fn scroll_to_tail(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let viewport = window.viewport_size();
        let streaming = self.space.read(cx).is_streaming();
        let tree = self.effective_tree(viewport.width, streaming);
        let total = self.selected_total_height(&tree, viewport.width, viewport.height);
        let doc = TITLE_BAR_RESERVE.as_f32() + total;
        let y = (viewport.height.as_f32() - doc).min(0.0);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    /// Dock the active draft at its "home": its slot top around 40% of the
    /// window, computed from the (already-selected) tree.
    pub(crate) fn dock_active_draft(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
    ) {
        let doc_top = self.placeholder_doc_top(roots, page_width, window_h);
        let target = window_h.as_f32() * 0.4;
        let y = (target - doc_top).min(0.0);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    /// "See in context": dock the active draft back at its place in the branch.
    pub(crate) fn go_home(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let viewport = window.viewport_size();
        let streaming = self.space.read(cx).is_streaming();
        let tree = self.effective_tree(viewport.width, streaming);
        if let Some(active) = self.active_draft.clone()
            && model::node_ref(&tree, &active).is_some()
        {
            self.select_path_to(&tree, &active, viewport.width);
            self.dock_active_draft(&tree, viewport.width, viewport.height);
        }
        cx.notify();
    }

    // -- The floating composer ---------------------------------------------

    /// The active draft's editor, pinned over the bottom and styled like a post.
    /// Floats (bottom-aligned, overlaying the conversation) when its in-flow slot
    /// is below the fold or when the user has swiped to another branch while
    /// editing; docks to the slot when scrolled to on its own branch. Renders
    /// nothing when no draft is active.
    pub(crate) fn render_active_draft(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        let Some(active) = self.active_draft.clone() else {
            return div().into_any_element();
        };
        let Some(editor) = self
            .drafts
            .iter()
            .find(|d| d.id == active)
            .map(|d| d.editor.clone())
        else {
            return div().into_any_element();
        };
        let theme = cx.theme();
        let bw = px(body_width(page_width));

        // On its own branch the overlay docks to its placeholder; off it (swiped
        // to a sibling while editing) it always floats.
        let on_path = self.selected_leaf_id(roots, page_width).as_ref() == Some(&active);

        let win = window_h.as_f32();
        let chrome = Self::composer_chrome();
        let content = self.composer_content_h.borrow().as_f32();
        let half_pad = POST_PAD_Y.as_f32() / 2.0;

        let float_bar_h = (chrome + content).min(COMPOSER_MAX_FRACTION * win);
        let float_top = win - float_bar_h;
        let slot_top = if on_path {
            Some(self.placeholder_doc_top(roots, page_width, window_h) + self.clamped_scroll_y())
        } else {
            None
        };
        let top_y = match slot_top {
            Some(s) => float_top.min(s + half_pad),
            None => float_top,
        };

        let overlayed = top_y >= float_top - 0.5;
        let docked = !overlayed;
        self.composer_overlayed.set(overlayed);

        let full_h = (content + chrome).max(win);
        let bar_h = if docked {
            let denom = (float_top - TITLE_BAR_RESERVE.as_f32()).max(1.0);
            let progress = ((float_top - top_y) / denom).clamp(0.0, 1.0);
            float_bar_h + progress * (full_h - float_bar_h)
        } else {
            float_bar_h
        };
        let body_h = (bar_h - chrome).max(0.0);
        let scrolled_down = self.composer_scroll.offset().y.as_f32() < -0.5;

        let mut byline = v_flex()
            .w(GUTTER_WIDTH)
            .flex_none()
            .items_end()
            .pt_5()
            .child(
                div()
                    .text_sm()
                    .font_weight(gpui::FontWeight::BOLD)
                    .opacity(0.85)
                    .text_color(theme.info)
                    .child("You"),
            );
        if overlayed {
            let home_fg = theme.muted_foreground;
            let home_fg_hover = theme.foreground;
            let home_bg_hover = theme.muted;
            byline = byline.child(
                div()
                    .id("space-draft-home")
                    .mt_1()
                    .px_1()
                    .rounded_md()
                    .text_sm()
                    .text_color(home_fg)
                    .cursor_pointer()
                    .hover(move |s| s.text_color(home_fg_hover).bg(home_bg_hover))
                    .child("See in context")
                    .on_click(cx.listener(|this, _, window, cx| this.go_home(window, cx))),
            );
        }

        let mut body = div()
            .id("space-composer-body")
            .w(bw)
            .h(px(body_h))
            .overflow_y_scroll()
            .track_scroll(&self.composer_scroll)
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
                div()
                    .w_full()
                    .pb(px(half_pad))
                    .child(MarkdownEditor::new(&editor).style(prose_style(cx)))
                    .child(record_height(
                        self.composer_content_h.clone(),
                        cx.entity().downgrade(),
                    )),
            );
        body.style().restrict_scroll_to_axis = Some(true);

        let mut composer = div()
            .id("space-composer")
            .probe("space/composer", gpui::Role::TextInput, "Message composer")
            .absolute()
            .left_0()
            .right_0()
            .top(px(top_y))
            .h(px(bar_h))
            .bg(theme.background)
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, window, cx| {
                if ev.keystroke.key == "escape" {
                    // Deactivate (deleting it if empty) and move focus off the
                    // editor to the view root, so a kept draft reads as exited
                    // (no stray cursor) until it's clicked back into.
                    this.deactivate_active_draft(cx);
                    window.focus(&this.focus_handle, cx);
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
            composer = composer.shadow(vec![
                BoxShadow::new(px(0.), px(-3.), hsla(0., 0., 0., 0.12)).blur_radius(px(18.)),
            ]);
        }
        composer.into_any_element()
    }
}

use crate::probe::Probe as _;
use std::cell::RefCell;
use std::rc::Rc;

/// Record the composer's natural (unclipped) content height each frame,
/// scheduling a catch-up frame when it changes so the bar resizes the same
/// frame the content settles.
fn record_height(
    cell: Rc<RefCell<gpui::Pixels>>,
    view: gpui::WeakEntity<SpaceView>,
) -> impl IntoElement {
    gpui::canvas(
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
