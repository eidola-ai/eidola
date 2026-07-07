//! Rendering one post — the byline gutter beside the reading column — and the
//! separator band that follows it (branch indicators + the "+" reply
//! affordance). Off-screen posts render as sized placeholders (the
//! virtualization), so only visible posts build the real `MarkdownEditor` and
//! shape their text.

use gpui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, WeakEntity, Window, canvas, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::MarkdownEditor;

use crate::probe::Probe as _;

use super::layout::body_width;
use super::model::{NodeSrc, TreeNode};
use super::{
    BAND_HEIGHT, GUTTER_GAP, GUTTER_WIDTH, POST_PAD_Y, SpaceView, VIRT_MARGIN, layout::Layout,
    prose_style,
};

impl SpaceView {
    /// Render `node`'s in-flow block: the real post when it intersects the
    /// viewport (± a margin), or a sized placeholder of the cached/estimated
    /// height otherwise. This is the per-frame virtualization — only visible
    /// posts shape their markdown.
    pub(crate) fn render_post_or_placeholder(
        &self,
        node: &TreeNode,
        doc_y: f32,
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        let h = self.node_height(node, page_width, window_h);
        let screen_top = doc_y + self.clamped_scroll_y();
        // While warming (see `warm_remaining`), render every post real so it
        // measures into the cache up front, even off-screen.
        let visible = self.warm_remaining.get() > 0
            || (screen_top + h > -VIRT_MARGIN && screen_top < window_h.as_f32() + VIRT_MARGIN);
        if visible {
            self.render_post(node, page_width, cx).into_any_element()
        } else {
            div().w(page_width).h(px(h)).flex_none().into_any_element()
        }
    }

    /// One post: the right-aligned byline gutter (UI font) beside the centered
    /// reading column (Newsreader prose, rendered through a `disabled`
    /// `MarkdownEditor` so it's pixel-identical to the composer) and the
    /// action gutter on the right (hover-revealed Edit / Regenerate). A
    /// measuring `canvas` records the block height into the layout cache.
    fn render_post(
        &self,
        node: &TreeNode,
        page_width: gpui::Pixels,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let bw = px(body_width(page_width));
        let editing_this = self.editing.as_ref().map(|e| &e.node_id) == Some(&node.id);

        let (byline, time, body): (SharedString, SharedString, AnyElement) = match node.src {
            NodeSrc::Msg(i) => {
                let post = &self.posts[i];
                (
                    post.byline.clone(),
                    post.time.clone(),
                    self.render_post_body(i, node, bw, editing_this, cx),
                )
            }
            NodeSrc::Streaming => {
                let byline = self
                    .space
                    .read(cx)
                    .last_submitted_model()
                    .map(SharedString::from)
                    .unwrap_or_else(|| SharedString::from("Eidola"));
                (
                    byline,
                    SharedString::from("now"),
                    self.render_streaming_body(bw, cx),
                )
            }
            // Draft never reaches here (it renders an in-flow slot placeholder).
            NodeSrc::Draft => (
                SharedString::default(),
                SharedString::default(),
                div().into_any_element(),
            ),
        };

        let byline_el = byline_gutter(byline, theme.foreground, None).child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(time),
        );

        let hover_id = node.id.clone();
        let mut row = h_flex()
            .id(SharedString::from(format!("space-post-{}", node.id)))
            .relative()
            .w(page_width)
            .py(POST_PAD_Y)
            .justify_center()
            .items_start()
            .gap(GUTTER_GAP)
            .on_hover(cx.listener(move |this, hovering: &bool, _, cx| {
                this.set_post_hover(&hover_id, *hovering, cx);
            }))
            .child(byline_el)
            .child(body)
            .child(self.render_post_actions(node, cx))
            .child(record_height(
                self.layout.clone(),
                node.id.clone(),
                cx.entity().downgrade(),
            ));
        if editing_this {
            // Escape restores the pre-edit text and exits the session. The
            // row (an ancestor of the focused editor) sees the key first.
            row = row.on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, window, cx| {
                if ev.keystroke.key == "escape" {
                    this.cancel_edit(window, cx);
                }
            }));
        }
        row
    }

    /// A finalized post's reading column: an optional reasoning disclosure
    /// (the streaming "Thinking…" pattern, preserved after finalize —
    /// reasoning is ephemeral, re-attached by position on reload) above the
    /// prose body. The body renders read-only, or **editable in place** while
    /// this post is the active edit session.
    fn render_post_body(
        &self,
        i: usize,
        node: &TreeNode,
        bw: gpui::Pixels,
        editing: bool,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let post = &self.posts[i];

        let mut col = v_flex().w(bw).gap_2();
        if let Some(reasoning) = post.reasoning.clone() {
            let label = if post.reasoning_expanded {
                "Hide thinking"
            } else {
                "Thinking…"
            };
            col = col.child(
                div()
                    .id(SharedString::from(format!(
                        "space-post-reasoning-{}",
                        node.id
                    )))
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .cursor_pointer()
                    .child(label)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.space
                            .update(cx, |s, cx| s.toggle_message_reasoning(i, cx));
                    })),
            );
            if post.reasoning_expanded {
                col = col.child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(reasoning),
                );
            }
        }
        if let Some(editor) = self.bodies.get(&node.id) {
            col = col.child(
                MarkdownEditor::new(editor)
                    .style(prose_style(cx))
                    .disabled(!editing),
            );
        }
        col.into_any_element()
    }

    /// The post's action gutter: reserved empty space at rest; on hover it
    /// reveals the per-post verbs — **Edit** (your own posts; an inline edit
    /// of the post's editor, committed as a new generation) and **Regenerate**
    /// (the assistant's; a new agent generation via the current model). While
    /// this post is being edited it shows the session's Save/Cancel verbs
    /// instead. Hidden entirely while streaming (the entity refuses mutations
    /// mid-stream, so dead verbs would lie).
    fn render_post_actions(&self, node: &TreeNode, cx: &Context<Self>) -> gpui::Div {
        let col = action_gutter().gap_0p5();
        let NodeSrc::Msg(i) = node.src else {
            return col;
        };
        let post = &self.posts[i];
        let Some(action_id) = post.action_id.clone() else {
            return col; // optimistic/synthetic rows aren't actionable yet
        };
        if self.space.read(cx).is_streaming() {
            return col;
        }

        let theme = cx.theme();
        let fg = theme.muted_foreground;
        let fg_hover = theme.foreground;
        let bg_hover = theme.muted;
        let verb = |id: SharedString, probe: String, label: &'static str, aria: SharedString| {
            h_flex()
                .id(id)
                .probe(probe, gpui::Role::Button, aria)
                .px_1()
                .ml_neg_1()
                .rounded_md()
                .cursor_pointer()
                .text_sm()
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                .child(label)
        };

        if self.editing.as_ref().map(|e| &e.node_id) == Some(&node.id) {
            return col
                .child(
                    verb(
                        SharedString::from(format!("space-edit-save-{}", node.id)),
                        format!("space/post/{i}/save"),
                        "Save",
                        "Save edit".into(),
                    )
                    .on_click(cx.listener(|this, _, window, cx| this.commit_edit(window, cx))),
                )
                .child(
                    verb(
                        SharedString::from(format!("space-edit-cancel-{}", node.id)),
                        format!("space/post/{i}/cancel"),
                        "Cancel",
                        "Cancel edit".into(),
                    )
                    .on_click(cx.listener(|this, _, window, cx| this.cancel_edit(window, cx))),
                );
        }

        if self.hovered_post.as_ref() != Some(&node.id) || self.editing.is_some() {
            return col;
        }

        match post.role.as_ref() {
            "user" => {
                let id = node.id.clone();
                col.child(
                    verb(
                        SharedString::from(format!("space-edit-{}", node.id)),
                        format!("space/post/{i}/edit"),
                        "Edit",
                        "Edit this post".into(),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.begin_edit(id.clone(), window, cx)
                    })),
                )
            }
            "assistant" => {
                let id = action_id.clone();
                col.child(
                    verb(
                        SharedString::from(format!("space-regenerate-{}", node.id)),
                        format!("space/post/{i}/regenerate"),
                        "Regenerate",
                        "Regenerate this response".into(),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| this.regenerate(&id, cx))),
                )
            }
            _ => col,
        }
    }

    // -- Post affordance state ----------------------------------------------

    /// Track which post's gutter shows its hover affordances. Clearing is
    /// guarded on the id still being the hovered one — moving the cursor up
    /// the page, the row being left can fire hover-false *after* the row
    /// being entered fired hover-true (the Library's out-of-order-leave
    /// lesson).
    pub(crate) fn set_post_hover(
        &mut self,
        id: &SharedString,
        hovering: bool,
        cx: &mut Context<Self>,
    ) {
        if hovering {
            if self.hovered_post.as_ref() != Some(id) {
                self.hovered_post = Some(id.clone());
                cx.notify();
            }
        } else if self.hovered_post.as_ref() == Some(id) {
            self.hovered_post = None;
            cx.notify();
        }
    }

    /// Begin editing a persisted user post in place: retire any active draft,
    /// enable the post's own body editor, and route its `⌘↩` to
    /// [`Self::commit_edit`]. The editor already holds the post's content
    /// (`sync_bodies`), which becomes the edit buffer; the pre-edit text is
    /// stashed for Escape.
    pub fn begin_edit(
        &mut self,
        node_id: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.space.read(cx).is_streaming() {
            return;
        }
        let Some(post) = self
            .posts
            .iter()
            .position(|p| p.action_id.as_deref() == Some(node_id.as_ref()))
            .map(|i| &self.posts[i])
        else {
            return;
        };
        if post.role.as_ref() != "user" {
            return;
        }
        let Some(action_id) = post.action_id.clone() else {
            return;
        };
        let Some(editor) = self.bodies.get(&node_id).cloned() else {
            return;
        };

        self.deactivate_active_draft(cx);
        self.close_request_panel(cx);

        let original = editor.read(cx).value().to_string();
        let sub = cx.subscribe_in(&editor, window, |this, _, event, window, cx| {
            if let gpui_markdown_editor::MarkdownEditorEvent::PressEnter {
                secondary: true, ..
            } = event
            {
                this.commit_edit(window, cx);
            }
        });
        self.editing = Some(super::EditingPost {
            action_id,
            node_id,
            original,
            _sub: sub,
        });
        let focus = editor.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
        cx.notify();
    }

    /// Commit the in-progress edit as a new generation via [`Space::edit`]
    /// (`item_current` resolves to it; the prior generation is preserved).
    /// A no-op on an empty buffer (nothing to commit — the session stays
    /// open) or when the entity refuses (mid-stream).
    pub fn commit_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ed) = self.editing.as_ref() else {
            return;
        };
        let Some(editor) = self.bodies.get(&ed.node_id) else {
            return;
        };
        let value = editor.read(cx).value().trim().to_string();
        if value.is_empty() {
            return;
        }
        let action_id = ed.action_id.to_string();
        let accepted = self.space.update(cx, |s, cx| s.edit(action_id, value, cx));
        if accepted {
            self.editing = None;
            window.focus(&self.focus_handle, cx);
            cx.notify();
        }
    }

    /// Abandon the edit: restore the pre-edit text into the post's editor and
    /// return focus to the view root.
    pub fn cancel_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ed) = self.editing.take() else {
            return;
        };
        if let Some(editor) = self.bodies.get(&ed.node_id) {
            let original = ed.original.clone();
            editor.update(cx, |e, cx| e.set_value(original, cx));
        }
        window.focus(&self.focus_handle, cx);
        cx.notify();
    }

    /// Regenerate an assistant post — a new agent generation of its item via
    /// the current model ([`Space::regenerate_post`]; refuses mid-stream).
    pub fn regenerate(&mut self, action_id: &SharedString, cx: &mut Context<Self>) {
        let model = self.current_model(cx);
        let id = action_id.to_string();
        self.space.update(cx, |s, cx| {
            s.regenerate_post(id, model, cx);
        });
        cx.notify();
    }

    /// An *inactive* draft rendered inline: a "Draft" byline beside its editable
    /// body, taking real vertical space in the tree. Clicking it focuses the
    /// editor, which re-activates the draft (floating composer). Off-screen it
    /// renders as a sized placeholder (virtualization), like a post.
    pub(crate) fn render_inactive_draft(
        &self,
        node: &TreeNode,
        doc_y: f32,
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        let h = self.node_height(node, page_width, window_h);
        let screen_top = doc_y + self.clamped_scroll_y();
        let visible = self.warm_remaining.get() > 0
            || (screen_top + h > -VIRT_MARGIN && screen_top < window_h.as_f32() + VIRT_MARGIN);
        let editor = self
            .drafts
            .iter()
            .find(|d| d.id == node.id)
            .map(|d| d.editor.clone());
        let Some(editor) = editor else {
            return div().w(page_width).h(px(h)).flex_none().into_any_element();
        };
        if !visible {
            return div().w(page_width).h(px(h)).flex_none().into_any_element();
        }

        let theme = cx.theme();
        let bw = px(body_width(page_width));
        let focus = editor.read(cx).focus_handle(cx);
        let id = node.id.clone();
        // The editor fills the inline runway (the frame reserves a full window,
        // minus its top/bottom padding), so a click in the blank space below the
        // text lands inside the editor and resolves to document end — the same
        // notes-editor affordance as the active composer, owned by the editor.
        let editor_fill = px((window_h.as_f32() - 2.0 * POST_PAD_Y.as_f32()).max(0.0));

        let byline_el = byline_gutter("You", theme.info, Some(DRAFT_BYLINE_OPACITY));

        h_flex()
            .relative()
            .w(page_width)
            // A draft is always the end of its branch, so reserve at least a
            // full window — the same `max(natural, window)` runway the active
            // composer docks into — so activating/deactivating it never shifts
            // the layout (`node_height` reports the same height).
            .min_h(window_h)
            .py(POST_PAD_Y)
            .justify_center()
            .items_start()
            .gap(GUTTER_GAP)
            .id(SharedString::from(format!(
                "space-draft-inactive-{}",
                node.id
            )))
            // Re-activate on click (byline/gutter clicks that miss the editor);
            // `on_click` (mouse-up) so the focus sticks — focusing during
            // mouse-down is undone by that same event's focus pass. Clicking the
            // editor itself activates via its `Focus` event; the editor also
            // owns caret placement (text → clicked position, blank tail → end),
            // so there's no cursor-resetting overlay here.
            .on_click(cx.listener(move |this, _, window, cx| {
                this.activate_draft(id.clone(), cx);
                window.focus(&focus, cx);
            }))
            .child(byline_el)
            .child(
                div().w(bw).child(
                    MarkdownEditor::new(&editor)
                        .style(prose_style(cx))
                        .min_height(editor_fill),
                ),
            )
            .child(action_gutter())
            .child(record_height(
                self.layout.clone(),
                node.id.clone(),
                cx.entity().downgrade(),
            ))
            .into_any_element()
    }

    /// The streaming reply body: a reasoning disclosure (clickable "Thinking…"
    /// header + the reasoning text when open) above the partial answer, rendered
    /// through the `streaming_body` editor (synced to the live content each
    /// frame in `render`).
    fn render_streaming_body(&self, bw: gpui::Pixels, cx: &Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let streaming = self.space.read(cx).streaming().cloned().unwrap_or_default();

        let mut col = v_flex().w(bw).gap_2();
        if !streaming.reasoning.is_empty() {
            let label = if streaming.expanded {
                "Hide thinking"
            } else {
                "Thinking…"
            };
            col = col.child(
                div()
                    .id("space-reasoning-toggle")
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .cursor_pointer()
                    .child(label)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.space
                            .update(cx, |s, cx| s.toggle_streaming_reasoning(cx));
                    })),
            );
            if streaming.expanded {
                col = col.child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(streaming.reasoning.clone())),
                );
            }
        }
        col.child(
            MarkdownEditor::new(&self.streaming_body)
                .style(prose_style(cx))
                .disabled(true),
        )
        .into_any_element()
    }

    /// Whether `node`'s subtree contains a draft **with content** — drives the
    /// info-colored branch dots. An empty (just-docked) tail draft doesn't tint;
    /// only once the user types does its branch read as in-progress.
    fn subtree_has_draft_content(&self, node: &TreeNode, cx: &gpui::App) -> bool {
        if matches!(node.src, NodeSrc::Draft)
            && let Some(d) = self.drafts.iter().find(|d| d.id == node.id)
            && !d.editor.read(cx).is_empty()
        {
            return true;
        }
        node.children
            .iter()
            .any(|c| self.subtree_has_draft_content(c, cx))
    }

    /// The faint full-bleed separator band between a post and what follows it.
    /// With more than one branch it carries clickable branch-indicator dots
    /// (glide to that branch; active one highlighted; a branch with an
    /// in-progress draft tinted `info`). The "+" (start a new branch) shows only
    /// on a post that already has a committed reply — a leaf's tail draft is its
    /// own reply affordance.
    pub(crate) fn render_band(
        &self,
        node: &TreeNode,
        page_width: gpui::Pixels,
        cx: &Context<Self>,
    ) -> gpui::Div {
        let theme = cx.theme();
        let band_bg = theme.muted;
        let active_color = theme.muted_foreground;
        let plus_fg = theme.muted_foreground;
        let plus_bg = theme.background.opacity(0.55);
        let plus_bg_hover = theme.background;

        let parent_id = node.id.clone();
        let info = theme.info;
        // Dots are the navigable branches — persisted siblings *and* drafts
        // (a draft is a branch you can navigate to). The streaming overlay isn't
        // navigable, so it's excluded.
        let dot_children: Vec<&TreeNode> = node
            .children
            .iter()
            .filter(|c| !matches!(c.src, NodeSrc::Streaming))
            .collect();
        let count = dot_children.len();

        let mut row = h_flex().items_center().gap_3();
        if count >= 2 {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            let dots = h_flex()
                .gap_1()
                .children(dot_children.iter().enumerate().map(|(i, child)| {
                    // A branch with an in-progress (non-empty) draft is tinted
                    // `info`; the active one is full-strength, the rest dimmed.
                    let base = if self.subtree_has_draft_content(child, cx) {
                        info
                    } else {
                        active_color
                    };
                    // The branch scroller indexes *all* children, so glide to and
                    // highlight this dot by its position among them.
                    let all_idx = node
                        .children
                        .iter()
                        .position(|c| c.id == child.id)
                        .unwrap_or(i);
                    let color = if all_idx == active {
                        base
                    } else {
                        base.alpha(0.5)
                    };
                    let pid = parent_id.clone();
                    div()
                        .id(SharedString::from(format!("space-dot-{parent_id}-{i}")))
                        .probe(
                            "space/branch-dot",
                            gpui::Role::Button,
                            SharedString::from(format!("Branch {}", i + 1)),
                        )
                        .flex_none()
                        .p(px(3.))
                        .cursor_pointer()
                        .child(div().size(px(5.)).rounded_full().bg(color))
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.glide_to_branch(pid.clone(), all_idx, page_width, window, cx);
                        }))
                }));
            row = row.child(dots);
        }

        // The "+" forks a new branch — shown only on a post that already has a
        // committed (persisted) reply. A leaf's only child is its tail draft (the
        // docked composer), which is itself the reply affordance, so no "+".
        let has_committed_reply = node
            .children
            .iter()
            .any(|c| matches!(c.src, NodeSrc::Msg(_)));
        if has_committed_reply {
            let add_id = parent_id.clone();
            row = row.child(
                div()
                    .id(SharedString::from(format!("space-add-{parent_id}")))
                    .probe("space/band/add", gpui::Role::Button, "Reply here")
                    .size(px(20.))
                    .flex_none()
                    .rounded_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(plus_fg)
                    .bg(plus_bg)
                    .cursor_pointer()
                    .hover(move |s| s.bg(plus_bg_hover))
                    .child("+")
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.create_draft(Some(add_id.clone()), window, cx);
                    })),
            );
        }

        h_flex()
            .w(page_width)
            .h(BAND_HEIGHT)
            .bg(band_bg)
            .items_center()
            .justify_center()
            .child(row)
    }
}

/// The label opacity for a draft's "You" byline — softened relative to a
/// committed post's byline so an unsent draft reads as tentative.
pub(crate) const DRAFT_BYLINE_OPACITY: f32 = 0.85;

/// The right-hand **action gutter** column — the symmetric mirror of the
/// byline gutter, left-aligned toward the reading column it acts on. Posts
/// fill it on hover with their per-post verbs (Edit / Regenerate — see
/// [`SpaceView::render_post_actions`]); the active composer fills it with the
/// request actions (see `request.rs`); otherwise it's reserved empty space
/// that keeps the reading column centered.
pub(crate) fn action_gutter() -> gpui::Div {
    v_flex().w(GUTTER_WIDTH).flex_none().items_start().pt_4()
}

/// The right-aligned byline gutter column that sits beside a reading column:
/// the author label in small bold `color`, geometry shared by posts, inline
/// drafts, and the floating composer — so the gutter (width, alignment, top
/// padding) can't drift between them and activating a draft never shifts its
/// byline. Callers append extras (the time line, "See in context") as further
/// children.
pub(crate) fn byline_gutter(
    label: impl Into<SharedString>,
    color: gpui::Hsla,
    label_opacity: Option<f32>,
) -> gpui::Div {
    use gpui::prelude::FluentBuilder as _;
    v_flex()
        .w(GUTTER_WIDTH)
        .flex_none()
        .items_end()
        .pt_4()
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::BOLD)
                .when_some(label_opacity, |d, o| d.opacity(o))
                .text_color(color)
                .child(label.into()),
        )
}

/// An invisible, zero-layout-impact overlay that records its own painted height
/// into the layout cache under `id`, scheduling one catch-up frame when the
/// height changes (so off-screen placeholders + the minimap settle the same
/// frame the content does).
fn record_height(
    layout: Layout,
    id: SharedString,
    view: WeakEntity<SpaceView>,
) -> impl IntoElement {
    canvas(
        move |bounds, window, _| {
            if layout.record(&id, bounds.size.height.as_f32()) {
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
