//! The composer — the single live, editable `MarkdownEditor`, pinned over the
//! bottom of the window. It aligns with the posts (same gutter/centering),
//! grows with content up to [`COMPOSER_MAX_FRACTION`] of the window then scrolls
//! internally, and *docks* into the page near the bottom of the selected branch.
//!
//! `⌘↩` persists the human post (reply-targeted) **and** streams an assistant
//! reply as the new selected leaf; `⌘⇧↩` posts the human turn only. A band's
//! "+" sets [`SpaceView::reply_to`] to that post, branching the thread there.

use gpui::{
    AnyElement, BoxShadow, Context, Focusable, InteractiveElement, IntoElement, KeyDownEvent,
    ParentElement, ScrollWheelEvent, StatefulInteractiveElement, Styled, TouchPhase, Window, div,
    hsla, linear_color_stop, linear_gradient, point, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::{MarkdownEditor, MarkdownEditorEvent, MarkdownEditorState};

use super::layout::body_width;
use super::model::{self, TreeNode};
use super::nav::ScrollOwner;
use super::{
    BAND_HEIGHT, COMPOSER_MAX_FRACTION, GUTTER_GAP, GUTTER_WIDTH, POST_PAD_Y, PostOnly, Send,
    SpaceView, TITLE_BAR_RESERVE, prose_style,
};

impl SpaceView {
    /// Route the composer's outward events. `PressEnter` with the primary
    /// modifier maps to the save-vs-ask gestures (`⌘↩` post & ask, `⌘⇧↩` post
    /// only); plain Enter never reaches here (the editor inserts a newline).
    pub(crate) fn on_editor_event(
        &mut self,
        _: &gpui::Entity<MarkdownEditorState>,
        event: &MarkdownEditorEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let MarkdownEditorEvent::PressEnter {
            secondary: true,
            shift,
        } = event
        {
            if *shift {
                self.post_only(&PostOnly, window, cx);
            } else {
                self.submit(&Send, window, cx);
            }
        }
    }

    /// `⌘↩` — persist the composer's markdown and stream a reply.
    pub(crate) fn submit(&mut self, _: &Send, window: &mut Window, cx: &mut Context<Self>) {
        if self.space.read(cx).is_streaming() {
            return;
        }
        let prompt = self.composer.read(cx).value().trim().to_string();
        if prompt.is_empty() {
            return;
        }
        let model = self.current_model(cx);
        let reply_to = self.resolve_reply_to(window.viewport_size().width, cx);
        self.composer.update(cx, |e, cx| e.clear(cx));
        self.error = None;
        self.reply_to = None;
        self.space.update(cx, |s, cx| {
            s.submit(prompt, model, reply_to, cx);
        });
        self.scroll_to_tail(window, cx);
        cx.notify();
    }

    /// `⌘⇧↩` — persist the composer's markdown without requesting a reply.
    pub(crate) fn post_only(&mut self, _: &PostOnly, window: &mut Window, cx: &mut Context<Self>) {
        if self.space.read(cx).is_streaming() {
            return;
        }
        let prompt = self.composer.read(cx).value().trim().to_string();
        if prompt.is_empty() {
            return;
        }
        let reply_to = self.resolve_reply_to(window.viewport_size().width, cx);
        self.composer.update(cx, |e, cx| e.clear(cx));
        self.error = None;
        self.reply_to = None;
        self.space.update(cx, |s, cx| {
            s.post_only(prompt, reply_to, cx);
        });
        self.scroll_to_tail(window, cx);
        cx.notify();
    }

    /// The antecedent action id the new post should reply to: the explicit
    /// `reply_to` (a branch) or the selected leaf (continue where the reader is
    /// looking). `None` for a blank space's first post. Only a real persisted
    /// action id is a valid antecedent.
    fn resolve_reply_to(&self, page_width: gpui::Pixels, cx: &gpui::App) -> Option<String> {
        let _ = cx;
        let base = model::build_tree(&self.posts);
        let target = self.reply_target_id(&base, page_width)?;
        self.posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(target.as_ref()))
            .and_then(|p| p.action_id.clone())
            .map(|s| s.to_string())
    }

    /// Start a reply to `target` (a band's "+"): branch the composer there.
    /// Selects the path to `target`, points its scroller at the (to-be-appended)
    /// draft page, and focuses the composer.
    pub(crate) fn start_reply(
        &mut self,
        target: gpui::SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let page_width = window.viewport_size().width;
        self.reply_to = Some(target.clone());
        let base = model::build_tree(&self.posts);
        self.select_path_to(&base, &target, page_width);
        if let Some(node) = model::node_ref(&base, &target) {
            // The draft is appended after the persisted children.
            let draft_idx = node.children.len();
            let stride = (page_width + BAND_HEIGHT).as_f32();
            let to_x = -(draft_idx as f32) * stride;
            let handle = self.scrolls.entry(target.clone()).or_default();
            let off = handle.offset();
            handle.set_offset(point(px(to_x), off.y));
            self.cancel_snap();
            self.snap_pin = Some((target.clone(), to_x));
        }
        let focus = self.composer.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
        self.scroll_to_tail(window, cx);
        cx.notify();
    }

    /// Scroll the page so the bottom of the selected branch (the composer /
    /// streaming leaf) sits at the window bottom.
    fn scroll_to_tail(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let viewport = window.viewport_size();
        let streaming = self.space.read(cx).is_streaming();
        let (tree, _) = self.effective_tree(viewport.width, streaming);
        let total = match tree.len() {
            0 => 0.0,
            1 => self.selected_subtree_height(&tree[0], viewport.width, viewport.height),
            _ => {
                let active =
                    self.active_child_index(model::ROOT_SCROLLER_ID, viewport.width, tree.len());
                self.selected_subtree_height(
                    &tree[active.min(tree.len() - 1)],
                    viewport.width,
                    viewport.height,
                )
            }
        };
        let doc = TITLE_BAR_RESERVE.as_f32() + total;
        let y = (viewport.height.as_f32() - doc).min(0.0);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    /// "See in context": dock the composer back at its place in the branch.
    pub(crate) fn go_home(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.scroll_to_tail(window, cx);
        cx.notify();
    }

    /// The composer overlay: the editable markdown pane pinned over the bottom,
    /// styled like a post. Floats (bottom-aligned, overlaying the conversation)
    /// when its in-flow slot is below the fold, and docks to the slot when
    /// scrolled to. Renders nothing while streaming (no compose surface then).
    pub(crate) fn render_active_draft(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
        streaming: bool,
        cx: &Context<Self>,
    ) -> AnyElement {
        if streaming {
            return div().into_any_element();
        }
        let theme = cx.theme();
        let bw = px(body_width(page_width));

        let win = window_h.as_f32();
        let chrome = Self::composer_chrome();
        let content = self.composer_content_h.borrow().as_f32();
        let half_pad = POST_PAD_Y.as_f32() / 2.0;

        // Floating bar: chrome + content, capped at half the window, bottom-pinned.
        let float_bar_h = (chrome + content).min(COMPOSER_MAX_FRACTION * win);
        let float_top = win - float_bar_h;
        // Dock: if the draft slot's top (plus the half-spacing gap) has risen
        // above the floating top, follow it up. Computed from the live scroll
        // offset so the threshold tracks the scroll with no one-frame lag.
        let slot_top = self.placeholder_doc_top(roots, page_width, window_h)
            + self.page_scroll.offset().y.as_f32();
        let top_y = float_top.min(slot_top + half_pad);

        let overlayed = top_y >= float_top - 0.5;
        let docked = !overlayed;
        self.composer_overlayed.set(overlayed);

        // Docked height grows with scroll position; floating keeps its
        // bottom-pinned height.
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

        // Byline gutter — a faint "Draft", plus a "See in context" affordance
        // while floating.
        let mut byline = v_flex()
            .w(GUTTER_WIDTH)
            .flex_none()
            .items_end()
            .pt_5()
            .child(
                div()
                    .text_sm()
                    .font_weight(gpui::FontWeight::BOLD)
                    .text_color(theme.muted_foreground)
                    .child("Draft"),
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
                    .child(MarkdownEditor::new(&self.composer).style(prose_style(cx)))
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
                    // Esc clears a branch reply target, returning the composer to
                    // the tail.
                    if this.reply_to.take().is_some() {
                        this.go_home(window, cx);
                    }
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
