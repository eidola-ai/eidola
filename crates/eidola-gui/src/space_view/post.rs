//! Rendering one post — the byline gutter beside the reading column — and the
//! separator band that follows it (branch indicators + the "+" reply
//! affordance). Off-screen posts render as sized placeholders (the
//! virtualization), so only visible posts build the real `MarkdownEditor` and
//! shape their text.

use gpui::prelude::FluentBuilder as _;
use gpui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, WeakEntity, Window, canvas, div, px, rems,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::MarkdownEditor;

use crate::probe::Probe as _;

use crate::overlay::{Contain as _, Overlay};

use super::context_menu::ContextTarget;
use super::layout::{
    COMPACT_GUTTER_GAP_REMS, COMPACT_GUTTER_LINE_REMS, GutterPlacement, PageLayout, page_layout,
};
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
            // A post that actually renders wants its incoming-reference index
            // (the source-highlight data); the request is made from the next
            // frame's `sync_references`, where the `&mut` borrow exists.
            if matches!(node.src, NodeSrc::Msg(_)) {
                self.want_incoming_refs(&node.id);
            }
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
        let page_layout = page_layout(page_width);
        let bw = px(page_layout.body_width);
        let editing_this = self.editing.as_ref().map(|e| &e.node_id) == Some(&node.id);

        // A **settled** post carries its byline/backend/time as the article's
        // accessible name and its whole text as the article's value. A
        // streaming one deliberately carries neither — see `article` below.
        let mut article: Option<(usize, SharedString, SharedString)> = None;

        let (byline, byline_backend, time, body): (
            SharedString,
            Option<SharedString>,
            SharedString,
            AnyElement,
        ) = match node.src {
            NodeSrc::Msg(i) => {
                let post = &self.posts[i];
                article = Some((
                    i,
                    article_label(&post.byline, post.byline_backend.as_deref(), &post.time),
                    SharedString::from(super::minimap::spoken_text(
                        &post.content,
                        &post.references,
                    )),
                ));
                (
                    post.byline.clone(),
                    post.byline_backend.clone(),
                    post.time.clone(),
                    self.render_post_body(i, node, bw, editing_this, cx),
                )
            }
            NodeSrc::Streaming(seq) => {
                // The in-flight turn's byline resolves live (the snapshot only
                // carries persisted rows): the responding participant's label.
                let participant_id = self
                    .space
                    .read(cx)
                    .streams()
                    .iter()
                    .find(|t| t.seq == seq)
                    .and_then(|t| t.participant_id.clone());
                let byline = self.participant_label(participant_id.as_deref(), cx);
                (
                    byline,
                    None,
                    SharedString::from("now"),
                    self.render_streaming_body(seq, bw, cx),
                )
            }
            // Draft never reaches here (it renders an in-flow slot placeholder).
            NodeSrc::Draft => (
                SharedString::default(),
                None,
                SharedString::default(),
                div().into_any_element(),
            ),
        };

        // The gutter stack: author (bold), the serving backend (quiet,
        // assistant rows only — suppressed when it would just repeat the
        // author line), then the time.
        let probe_base = format!("space/post/{}/metadata", node.id);
        let mut byline_el = byline_gutter_frame(page_layout)
            .id(SharedString::from(format!(
                "space-post-{}-metadata",
                node.id
            )))
            .probe_bounds(probe_base.clone(), gpui::Role::Label, "Post metadata row")
            .child(
                byline_label(page_layout, byline.clone(), theme.foreground, None, true)
                    .id(SharedString::from(format!(
                        "space-post-{}-metadata-author",
                        node.id
                    )))
                    .probe_bounds(
                        format!("{probe_base}/author"),
                        gpui::Role::Label,
                        "Post author metadata",
                    ),
            );
        if let Some(backend) = byline_backend.filter(|b| *b != byline) {
            byline_el = byline_el.child(
                div()
                    .id(SharedString::from(format!(
                        "space-post-{}-metadata-backend",
                        node.id
                    )))
                    .probe_bounds(
                        format!("{probe_base}/backend"),
                        gpui::Role::Label,
                        "Post backend metadata",
                    )
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .when(page_layout.gutters == GutterPlacement::Stacked, |d| {
                        d.flex_shrink_1().min_w_0().truncate()
                    })
                    .child(backend),
            );
        }
        let byline_el = byline_el.child(
            div()
                .id(SharedString::from(format!(
                    "space-post-{}-metadata-time",
                    node.id
                )))
                .probe_bounds(
                    format!("{probe_base}/time"),
                    gpui::Role::Label,
                    "Post time metadata",
                )
                .text_xs()
                .text_color(theme.muted_foreground)
                .when(page_layout.gutters == GutterPlacement::Stacked, |d| {
                    d.flex_shrink_0().truncate()
                })
                .child(time),
        );

        let hover_id = node.id.clone();
        let mut row = match page_layout.gutters {
            GutterPlacement::Sides => h_flex().justify_center().items_start().gap(GUTTER_GAP),
            GutterPlacement::Stacked => v_flex().items_center(),
        }
        .id(SharedString::from(format!("space-post-{}", node.id)))
        // The conversation itself, in the tree at last: each settled post
        // is an `Article` (`AXGroup` + `AXDocumentArticle`) named for its
        // author and time, carrying its whole text as the value. Only
        // settled posts — a streaming reply's text mutates every token, and
        // AT re-reads a changed value in full, so binding one there would
        // make the app *less* usable than silence (audit §4). The row
        // becomes a node the moment the stream finalizes into a `Msg`.
        .when_some(article, |d, (i, label, value)| {
            d.probe_value(format!("space/post/{i}"), gpui::Role::Article, label, value)
        })
        // Wave B: the *focused* post row tracks the view's single post
        // focus handle. `Role::Article` already made the row focusable and
        // gave it the `:focus-visible` ring; tracking the handle is what
        // lets the arrow keys *move* focus here, and what tells gpui to
        // report this node as focused in the AccessKit tree. One handle
        // moved between rows — see `SpaceView::post_focus`.
        // Only at the *post* level: once focus enters the affordance row
        // the verb tracks its own handle, and two elements tracking one
        // handle would claim focus twice in a frame (gpui asserts on it).
        .when(self.post_row_holds_focus(&node.id), |d| {
            d.track_focus(&self.post_focus)
        })
        .relative()
        .w(page_width)
        .py(POST_PAD_Y)
        .on_hover(cx.listener(move |this, hovering: &bool, _, cx| {
            this.set_post_hover(&hover_id, *hovering, cx);
        }))
        .child(byline_el)
        .child(body)
        .child(self.render_post_actions(node, page_layout, cx))
        .child(record_height(
            self.layout.clone(),
            node.id.clone(),
            cx.entity().downgrade(),
        ));
        if editing_this {
            // Escape restores the pre-edit text and exits the session. The
            // row (an ancestor of the focused editor) sees the key first —
            // which is exactly why it must yield to an open context menu
            // (`context_menu_absorbs_escape`): a right-click inside the
            // session then Escape-to-dismiss would otherwise throw the
            // unsaved edit away on the same press that closed the menu. The
            // root closes it; the next Escape reaches this handler.
            row = row.on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, window, cx| {
                if ev.keystroke.key == "escape" && !this.context_menu_absorbs_escape() {
                    this.cancel_edit(window, cx);
                }
            }));
        }
        row
    }

    /// A finalized post's reading column: an optional reasoning disclosure
    /// above the prose body. The reasoning is **durable** (a persisted
    /// `thinking` content block), so the disclosure is present whenever
    /// thinking exists — on a freshly-finalized turn *and* on a space reopened
    /// weeks later. Its label reads "Show thinking" / "Hide thinking" here:
    /// the stream is over, so "Thinking…" would be a lie (that state belongs
    /// to [`Self::render_streaming_body`]). The body renders read-only, or
    /// **editable in place** while this post is the active edit session.
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
        // Wave B's focus ring, drawn here rather than by `probe`: the post
        // *row* is full-bleed, so a ring on its own bounds is two window-wide
        // rules with its sides off-screen. The reading column is what the
        // reader perceives as the post, so that is what the ring frames — same
        // hairline, same accent, a wider gap so it reads as a frame around
        // prose rather than a rule under its first line. No `:focus-visible`
        // guard is needed: tree focus is only ever set by the keyboard model.
        if self.post_row_holds_focus(&node.id) {
            let shadows = crate::focus::ring_shadows_at(
                crate::focus::ring_colors(),
                crate::focus::POST_RING_OFFSET,
            );
            col = col.rounded_sm().shadow(shadows);
        }
        if let Some(reasoning) = post.reasoning.clone() {
            let (label, aria) = thinking_labels(false, post.reasoning_expanded);
            col = col.child(
                verb_button(
                    SharedString::from(format!("space-post-reasoning-{}", node.id)),
                    format!("space/post/{i}/reasoning"),
                    label,
                    aria,
                    cx,
                )
                .aria_expanded(post.reasoning_expanded)
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
            // Source highlights: the passages other posts have quoted, painted
            // by the editor's opaque highlight plugin. The keys are indexes
            // into this post's incoming-reference list — the editor never
            // learns what they mean, and hands them back verbatim on a click.
            let node_id = node.id.clone();
            // The context menu: read-only while the post is settled (Select
            // All / Copy / the quote pair), the editable set during an Edit
            // session. The target carries the node id so Quote can resolve the
            // selection to this post's generation + block.
            let menu_node = node.id.clone();
            let menu_editor = editor.clone();
            col = col.child(
                MarkdownEditor::new(editor)
                    .style(prose_style(cx))
                    .disabled(!editing)
                    .on_highlight_click(cx.listener(move |this, keys: &[u64], window, cx| {
                        this.on_highlight_click(node_id.clone(), keys, window, cx);
                    }))
                    .on_context_menu(cx.listener(
                        move |this, at: &gpui::Point<gpui::Pixels>, _, cx| {
                            let target = if editing {
                                ContextTarget::Editable
                            } else {
                                ContextTarget::Post {
                                    node_id: Some(menu_node.clone()),
                                }
                            };
                            this.open_context_menu(*at, menu_editor.clone(), target, cx);
                        },
                    )),
            );
        }
        // The footnote rail — the post's references, rendered *outside* the
        // markdown (the editor never learns what a reference is).
        if let Some(rail) = self.render_post_footnotes(i, node, editing, cx) {
            col = col.child(rail);
        }
        // The trace disclosure — what the turn actually did. Last, and last on
        // purpose: references belong to the post's own story, activity is
        // operational detail subordinate to all of it.
        if let Some(traces) = self.render_post_traces(i, node, cx) {
            col = col.child(traces);
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
    fn render_post_actions(
        &self,
        node: &TreeNode,
        page_layout: PageLayout,
        cx: &Context<Self>,
    ) -> gpui::Div {
        let empty = || action_gutter(page_layout, false).gap_0p5();
        let NodeSrc::Msg(i) = node.src else {
            return empty();
        };
        let post = &self.posts[i];
        let Some(action_id) = post.action_id.clone() else {
            return empty(); // optimistic/synthetic rows aren't actionable yet
        };
        if !matches!(post.role.as_ref(), "user" | "assistant") {
            return empty();
        }
        if self.space.read(cx).is_streaming() {
            return empty();
        }

        // The shared quiet-verb look (see `verb_button`) — the same family the
        // reading column's thinking disclosure uses. `slot` is the verb's index
        // in this post's affordance row, and while the keyboard has entered
        // *this* post every verb tracks that slot's own handle
        // (`SpaceView::affordance_slots`, pre-grown in `render`). A handle per
        // slot rather than one for "the focused verb" is what lets
        // `sync_tree_focus` see which verb a Tab landed on and resync the
        // level's index to it. Only the level's own post tracks from the pool —
        // two elements claiming one handle report focus twice in a frame.
        let holds_level = self.post_holds_affordance_level(&node.id);
        let verb = |slot: usize,
                    id: SharedString,
                    probe: String,
                    label: &'static str,
                    aria: SharedString| {
            let b = verb_button(id, probe, label, aria, cx);
            match self.affordance_slots.get(slot).filter(|_| holds_level) {
                Some(handle) => b.track_focus(handle),
                None => b,
            }
        };

        if self.editing.as_ref().map(|e| &e.node_id) == Some(&node.id) {
            return action_gutter(page_layout, true)
                .gap_0p5()
                .child(
                    verb(
                        0,
                        SharedString::from(format!("space-edit-save-{}", node.id)),
                        format!("space/post/{i}/save"),
                        "Save",
                        "Save edit".into(),
                    )
                    .on_click(cx.listener(|this, _, window, cx| this.commit_edit(window, cx))),
                )
                .child(
                    verb(
                        1,
                        SharedString::from(format!("space-edit-cancel-{}", node.id)),
                        format!("space/post/{i}/cancel"),
                        "Cancel",
                        "Cancel edit".into(),
                    )
                    .on_click(cx.listener(|this, _, window, cx| this.cancel_edit(window, cx))),
                );
        }

        // Revealed by hover **or by keyboard focus** (wave B / audit S7): gpui
        // suppresses hover entirely while the input modality is keyboard, so a
        // hover-only gate would hide these verbs from exactly the user the
        // keyboard model exists for.
        if !self.post_affordances_revealed(&node.id) || self.editing.is_some() {
            return action_gutter(page_layout, true).gap_0p5();
        }

        let col = action_gutter(page_layout, true).gap_0p5();
        match post.role.as_ref() {
            "user" => {
                let id = node.id.clone();
                col.child(
                    verb(
                        0,
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
                        0,
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
        self.band_menu = None;

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
            removed_references: Vec::new(),
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
        // Reference ordinals the session's footnote chips marked for removal.
        // `edit_post_with_removals` drops them from the **new** generation
        // only; the prior generation keeps them (history is append-only), and
        // ordinal 0 — the reply edge — can never be here.
        let removals = ed.removed_references.clone();
        // A removed footnote takes its marker with it: the edge is gone, so a
        // surviving `{{ embed N }}` would render as literal wire syntax on
        // reload and go upstream literally. Stripped from the *submitted*
        // string only — the buffer is untouched, so Cancel still restores the
        // original and a rejected (busy) submit loses nothing.
        let value = super::references::strip_removed_markers(&value, &removals);
        if value.is_empty() {
            return;
        }
        let accepted = self
            .space
            .update(cx, |s, cx| s.edit(action_id, value, removals, cx));
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
    /// the post's **own recorded model** (regenerating re-asks the model that
    /// answered; the config default is the fallback for rows that carry none).
    /// [`Space::regenerate_post`] refuses mid-stream.
    pub fn regenerate(&mut self, action_id: &SharedString, cx: &mut Context<Self>) {
        let model = self
            .posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(action_id.as_ref()))
            .and_then(|p| p.model.clone())
            .map(|m| m.to_string())
            .unwrap_or_else(|| self.stores.config.read(cx).default_model());
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
        let page_layout = page_layout(page_width);
        let bw = px(page_layout.body_width);
        let focus = editor.read(cx).focus_handle(cx);
        let id = node.id.clone();
        // The editor fills the inline runway (the frame reserves a standalone
        // slot, minus its top/bottom padding), so a click in the blank space
        // below the text lands inside the editor and resolves to document end —
        // the same notes-editor affordance as the active composer, owned by the
        // editor.
        let slot_h = self.runway_height(window_h);
        let compact_top = match page_layout.gutters {
            GutterPlacement::Sides => 0.0,
            GutterPlacement::Stacked => super::layout::compact_gutter_occupancy(px(
                crate::theme::UI_FONT_SIZE * self.layout.scale(),
            )),
        };
        let editor_fill = px((slot_h - 2.0 * POST_PAD_Y.as_f32() - compact_top).max(0.0));

        let byline_el = byline_gutter(page_layout, "You", theme.info, Some(DRAFT_BYLINE_OPACITY));

        match page_layout.gutters {
            GutterPlacement::Sides => h_flex().justify_center().items_start().gap(GUTTER_GAP),
            GutterPlacement::Stacked => v_flex().items_center(),
        }
        .relative()
        .w(page_width)
        // A draft is always the end of its branch, so reserve at least a
        // standalone slot — the same `max(natural, standalone)` runway the
        // active composer docks into — so activating/deactivating it never
        // shifts the layout (`node_height` reports the same height).
        .min_h(px(slot_h))
        .py(POST_PAD_Y)
        .id(SharedString::from(format!(
            "space-draft-inactive-{}",
            node.id
        )))
        .probe(
            "space/draft/inactive",
            gpui::Role::Button,
            "Draft — open to edit",
        )
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
            div()
                .w(bw)
                .child(
                    MarkdownEditor::new(&editor)
                        .style(prose_style(cx))
                        .min_height(editor_fill),
                )
                // The draft's pending quotes, as footnotes — the same rail
                // a posted exchange carries, so composing looks like what
                // it will become.
                .children(self.render_draft_footnotes(&node.id, false, cx)),
        )
        .child(action_gutter(page_layout, false))
        .child(record_height(
            self.layout.clone(),
            node.id.clone(),
            cx.entity().downgrade(),
        ))
        .into_any_element()
    }

    /// One streaming turn's reply body: a reasoning disclosure (clickable
    /// "Thinking…" header + the reasoning text when open) above the partial
    /// answer, rendered through that turn's editor in `streaming_bodies`
    /// (synced to the live content each frame in `render`). Several turns can
    /// render at once, each with its own editor and disclosure.
    ///
    /// A turn whose engine is still warming leads with **"Loading model…"** —
    /// the same quiet line in the same slot as "Thinking…", because it answers
    /// the same question (what is this silence?) with the honest reason. It is
    /// a readout, not a control: no click, `Role::Label`.
    fn render_streaming_body(&self, seq: u64, bw: gpui::Pixels, cx: &Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let turn = self
            .space
            .read(cx)
            .streams()
            .iter()
            .find(|t| t.seq == seq)
            .cloned();
        let streaming = turn
            .as_ref()
            .map(|t| t.response.clone())
            .unwrap_or_default();
        let warming = self
            .turn_engine_is_warming(turn.as_ref().and_then(|t| t.participant_id.as_deref()), cx);

        let mut col = v_flex().w(bw).gap_2();
        if warming {
            col = col.child(
                h_flex()
                    .id(SharedString::from(format!("space-loading-model-{seq}")))
                    .probe(
                        format!("space/streaming/{seq}/loading"),
                        gpui::Role::Label,
                        "Loading model…",
                    )
                    .self_start()
                    .px_1()
                    .ml_neg_1()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("Loading model…"),
            );
        }
        if !streaming.reasoning.is_empty() {
            let (label, aria) = thinking_labels(true, streaming.expanded);
            col = col.child(
                verb_button(
                    SharedString::from(format!("space-reasoning-toggle-{seq}")),
                    format!("space/reasoning/toggle/{seq}"),
                    label,
                    aria,
                    cx,
                )
                .aria_expanded(streaming.expanded)
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.space
                        .update(cx, |s, cx| s.toggle_streaming_reasoning(seq, cx));
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
        if let Some(editor) = self.streaming_bodies.get(&seq) {
            // A streaming reply is read-only content with no persisted post
            // behind it yet, so its menu offers Select All / Copy and never
            // the quote pair (`node_id: None`).
            let menu_editor = editor.clone();
            col = col.child(
                MarkdownEditor::new(editor)
                    .style(prose_style(cx))
                    .disabled(true)
                    .on_context_menu(cx.listener(
                        move |this, at: &gpui::Point<gpui::Pixels>, _, cx| {
                            this.open_context_menu(
                                *at,
                                menu_editor.clone(),
                                ContextTarget::Post { node_id: None },
                                cx,
                            );
                        },
                    )),
            );
        }
        col.into_any_element()
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
    /// in-progress draft tinted `info`). The "+" opens the band's quiet
    /// **Reply-or-Ask menu** inline: **Reply** (start a new branch here — only
    /// on a post that already has a committed reply; a leaf's tail draft is
    /// its own reply affordance) and **Ask <agent>** per agent participant (an
    /// explicit ask targeting this band's post). A tail band therefore offers
    /// Ask only, and shows no "+" at all when the space has no agents.
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
        // Dots are the navigable branches — persisted siblings, drafts (a
        // draft is a branch you can navigate to), *and* in-flight streaming
        // turns: a fan-out lands concurrent replies as sibling branches, and
        // without a dot the second stream would be invisible until it
        // finalized. A streaming dot is tinted `info` (something is happening
        // on that branch), same as a content-bearing draft.
        let dot_children: Vec<&TreeNode> = node.children.iter().collect();
        let count = dot_children.len();

        let mut row = h_flex().items_center().gap_3();
        if count >= 2 {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            let dots = h_flex()
                .gap_1()
                .children(dot_children.iter().enumerate().map(|(i, child)| {
                    // A branch with an in-progress (non-empty) draft or a
                    // live streaming turn is tinted `info`; the active one is
                    // full-strength, the rest dimmed.
                    let base = if matches!(child.src, NodeSrc::Streaming(_))
                        || self.subtree_has_draft_content(child, cx)
                    {
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
                    let selector = format!("space-dot-{parent_id}-{i}");
                    div()
                        .id(SharedString::from(selector.clone()))
                        .debug_selector(move || selector.clone())
                        .probe(
                            format!("space/band/dot/{i}"),
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

        // Only a persisted post can be replied to / asked about; overlay
        // nodes (draft/streaming) carry no band affordances.
        let is_persisted = matches!(node.src, NodeSrc::Msg(_))
            && !parent_id.starts_with("idx-")
            && self
                .posts
                .iter()
                .any(|p| p.action_id.as_deref() == Some(parent_id.as_ref()));
        // "Reply" (a new fork) is offered only on a post that already has a
        // committed reply — a leaf's tail draft is its own reply affordance.
        let has_committed_reply = node
            .children
            .iter()
            .any(|c| matches!(c.src, NodeSrc::Msg(_)));
        let agents = if is_persisted {
            self.space_agents(cx)
        } else {
            Vec::new()
        };
        let offer_reply = is_persisted && has_committed_reply;
        let offer_ask = !agents.is_empty();
        let menu_open = self.band_menu.as_ref() == Some(&parent_id);

        if menu_open {
            // The inline menu, in the band's own quiet voice: Reply, then one
            // Ask per agent. Click-out (or Escape via the composer's handler,
            // or a choice) dismisses.
            let mut menu = h_flex()
                .id(SharedString::from(format!("space-band-menu-{parent_id}")))
                .probe(
                    "space/band/menu",
                    gpui::Role::Group,
                    "Reply or ask a participant",
                )
                .contain_mouse(Overlay::Popover)
                .items_center()
                .gap_1()
                .px_1()
                .rounded_md()
                .bg(theme.background.opacity(0.7))
                .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                    this.band_menu = None;
                    cx.notify();
                }));
            let chip_fg = theme.muted_foreground;
            let chip_fg_hover = theme.foreground;
            let chip_bg_hover = theme.background;
            if offer_reply {
                let add_id = parent_id.clone();
                menu = menu.child(
                    h_flex()
                        .id(SharedString::from(format!("space-band-reply-{parent_id}")))
                        .probe("space/band/menu/reply", gpui::Role::Button, "Reply here")
                        .px_1p5()
                        .py_0p5()
                        .rounded_md()
                        .cursor_pointer()
                        .text_sm()
                        .text_color(chip_fg)
                        .hover(move |s| s.text_color(chip_fg_hover).bg(chip_bg_hover))
                        .child("Reply")
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.band_menu = None;
                            this.create_draft(Some(add_id.clone()), window, cx);
                        })),
                );
            }
            for (i, (pid, label)) in agents.iter().enumerate() {
                let pid = pid.clone();
                let target = parent_id.clone();
                menu = menu.child(
                    h_flex()
                        .id(SharedString::from(format!(
                            "space-band-ask-{parent_id}-{i}"
                        )))
                        .probe(
                            format!("space/band/menu/ask/{i}"),
                            gpui::Role::Button,
                            SharedString::from(format!("Ask {label} to respond here")),
                        )
                        .px_1p5()
                        .py_0p5()
                        .rounded_md()
                        .cursor_pointer()
                        .text_sm()
                        .text_color(chip_fg)
                        .hover(move |s| s.text_color(chip_fg_hover).bg(chip_bg_hover))
                        .child(SharedString::from(format!("Ask {label}")))
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.ask_participant(pid.clone(), target.to_string(), window, cx);
                        })),
                );
            }
            row = row.child(menu);
        } else if offer_reply || offer_ask {
            let add_id = parent_id.clone();
            row = row.child(
                div()
                    .id(SharedString::from(format!("space-add-{parent_id}")))
                    .probe(
                        "space/band/add",
                        gpui::Role::Button,
                        if offer_reply { "Reply or ask" } else { "Ask" },
                    )
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
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.band_menu = Some(add_id.clone());
                        cx.notify();
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

/// A settled post's accessible name: everything its byline gutter shows, on
/// one line — author, the serving backend when it adds something (the same
/// suppression the gutter applies), then the time. The gutter's three stacked
/// lines are node-less text, so without this the model/backend line and the
/// timestamp reach nobody.
pub(crate) fn article_label(byline: &str, backend: Option<&str>, time: &str) -> SharedString {
    let mut label = byline.to_string();
    if let Some(backend) = backend.filter(|b| *b != byline && !b.is_empty()) {
        label.push_str(" · ");
        label.push_str(backend);
    }
    if !time.is_empty() {
        label.push_str(" · ");
        label.push_str(time);
    }
    SharedString::from(label)
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
pub(crate) fn action_gutter(layout: PageLayout, occupied: bool) -> gpui::Div {
    match layout.gutters {
        GutterPlacement::Sides => v_flex().w(GUTTER_WIDTH).flex_none().items_start().pt_4(),
        GutterPlacement::Stacked => h_flex()
            .w(px(layout.body_width))
            .flex_none()
            .items_center()
            .when(occupied, |d| {
                d.h(rems(COMPACT_GUTTER_LINE_REMS))
                    .mt(rems(COMPACT_GUTTER_GAP_REMS))
            }),
    }
}

/// A **quiet verb** — the shared look of every per-post affordance: small,
/// muted at rest, lifting to full foreground on a `muted` chip when hovered,
/// hung a hair left (`ml_neg_1` cancels its own `px_1`) so its text aligns with
/// the column it sits in.
///
/// `self_start` is load-bearing for the reading-column host: a `v_flex`
/// stretches its children across the cross axis, which blew the hover chip out
/// to the full 600px measure — a wash behind the word "Show thinking" rather
/// than a chip on it. (The action gutter already sets `items_start`, so this is
/// a no-op there; carrying it on the verb makes the look independent of what
/// the host happens to do.)
///
/// Shared so the verbs can't drift: the action gutter's Edit / Regenerate /
/// Save / Cancel and the reading column's thinking disclosure are one visual
/// family, which is exactly what the disclosure previously wasn't (it was bare
/// muted text with no hover, reading as a caption rather than a control).
pub(crate) fn verb_button(
    id: impl Into<SharedString>,
    probe: String,
    label: impl Into<SharedString>,
    aria: impl Into<SharedString>,
    cx: &gpui::App,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    let fg = theme.muted_foreground;
    let fg_hover = theme.foreground;
    let bg_hover = theme.muted;
    h_flex()
        .id(id.into())
        .probe(probe, gpui::Role::Button, aria.into())
        .self_start()
        .px_1()
        .ml_neg_1()
        .rounded_md()
        .cursor_pointer()
        .text_sm()
        .text_color(fg)
        .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
        .child(label.into())
}

/// The thinking disclosure's label + accessible name, for the three states the
/// control has: **live** (reasoning still arriving — "Thinking…", the state
/// that says *something is happening*), **collapsed** (finished, hidden), and
/// **expanded**. The accessible name always says what the click *does*, even
/// while the visible label narrates the stream.
pub(crate) fn thinking_labels(streaming: bool, expanded: bool) -> (&'static str, &'static str) {
    match (streaming, expanded) {
        (_, true) => ("Hide thinking", "Hide thinking"),
        (true, false) => ("Thinking…", "Show thinking"),
        (false, false) => ("Show thinking", "Show thinking"),
    }
}

/// The right-aligned byline gutter column that sits beside a reading column:
/// the author label in small bold `color`, geometry shared by posts, inline
/// drafts, and the floating composer — so the gutter (width, alignment, top
/// padding) can't drift between them and activating a draft never shifts its
/// byline. Callers append extras (the time line, "See in context") as further
/// children.
pub(crate) fn byline_gutter(
    layout: PageLayout,
    label: impl Into<SharedString>,
    color: gpui::Hsla,
    label_opacity: Option<f32>,
) -> gpui::Div {
    byline_gutter_frame(layout).child(byline_label(layout, label, color, label_opacity, false))
}

fn byline_gutter_frame(layout: PageLayout) -> gpui::Div {
    match layout.gutters {
        GutterPlacement::Sides => v_flex().w(GUTTER_WIDTH).items_end().pt_4(),
        GutterPlacement::Stacked => h_flex()
            .w(px(layout.body_width))
            .min_w_0()
            .overflow_hidden()
            .items_center()
            .gap_2()
            .h(rems(COMPACT_GUTTER_LINE_REMS))
            .mb(rems(COMPACT_GUTTER_GAP_REMS)),
    }
    .flex_none()
}

fn byline_label(
    layout: PageLayout,
    label: impl Into<SharedString>,
    color: gpui::Hsla,
    label_opacity: Option<f32>,
    compact_flexible: bool,
) -> gpui::Div {
    use gpui::prelude::FluentBuilder as _;
    div()
        .text_sm()
        .font_weight(FontWeight::BOLD)
        .when_some(label_opacity, |d, o| d.opacity(o))
        .text_color(color)
        .when(
            compact_flexible && layout.gutters == GutterPlacement::Stacked,
            |d| d.flex_shrink_1().min_w_0().truncate(),
        )
        .child(label.into())
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

#[cfg(test)]
mod tests {
    use super::article_label;

    #[test]
    fn article_label_folds_the_whole_byline_gutter_onto_one_line() {
        // Author alone when there is nothing else (a synthetic row carries no
        // timestamp, and `fmt_clock` renders one as empty).
        assert_eq!(article_label("You", None, ""), "You");
        assert_eq!(article_label("You", None, "9:05 AM"), "You · 9:05 AM");
        // An assistant row's second gutter line is the serving backend — the
        // "model name" a screen reader otherwise never hears.
        assert_eq!(
            article_label("Gemma 4 E2B", Some("Local"), "9:05 AM"),
            "Gemma 4 E2B · Local · 9:05 AM"
        );
        // Suppressed exactly where the gutter suppresses it: when it would
        // only repeat the author, or when it is empty.
        assert_eq!(article_label("Eidola", Some("Eidola"), ""), "Eidola");
        assert_eq!(article_label("Eidola", Some(""), ""), "Eidola");
    }
}
