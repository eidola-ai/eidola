//! Rendering one post — the byline gutter beside the reading column — and the
//! separator band that follows it (branch indicators + the "+" reply
//! affordance). Off-screen posts render as sized placeholders (the
//! virtualization), so only visible posts build the real `MarkdownEditor` and
//! shape their text.

use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, WeakEntity, canvas, div, px,
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
        let screen_top = doc_y + self.page_scroll.offset().y.as_f32();
        let visible = screen_top + h > -VIRT_MARGIN && screen_top < window_h.as_f32() + VIRT_MARGIN;
        if visible {
            self.render_post(node, page_width, cx).into_any_element()
        } else {
            div().w(page_width).h(px(h)).flex_none().into_any_element()
        }
    }

    /// One post: the right-aligned byline gutter (UI font) beside the centered
    /// reading column (Newsreader prose, rendered through a `disabled`
    /// `MarkdownEditor` so it's pixel-identical to the composer). A measuring
    /// `canvas` records the block height into the layout cache.
    fn render_post(
        &self,
        node: &TreeNode,
        page_width: gpui::Pixels,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let bw = px(body_width(page_width));

        let (byline, time, body): (SharedString, SharedString, AnyElement) = match node.src {
            NodeSrc::Msg(i) => {
                let post = &self.posts[i];
                let body = self
                    .bodies
                    .get(&node.id)
                    .map(|editor| {
                        div()
                            .w(bw)
                            .child(
                                MarkdownEditor::new(editor)
                                    .style(prose_style(cx))
                                    .disabled(true),
                            )
                            .into_any_element()
                    })
                    .unwrap_or_else(|| div().w(bw).into_any_element());
                (post.byline.clone(), post.time.clone(), body)
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

        let byline_el = v_flex()
            .w(GUTTER_WIDTH)
            .flex_none()
            .items_end()
            .pt_5()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::BOLD)
                    .text_color(theme.foreground)
                    .child(byline),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(time),
            );

        h_flex()
            .relative()
            .w(page_width)
            .py(POST_PAD_Y)
            .justify_center()
            .items_start()
            .gap(GUTTER_GAP)
            .pr(GUTTER_WIDTH / 2. + GUTTER_GAP)
            .child(byline_el)
            .child(body)
            .child(record_height(
                self.layout.clone(),
                node.id.clone(),
                cx.entity().downgrade(),
            ))
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

    /// The faint full-bleed separator band between a post and what follows it.
    /// With more than one reply it carries clickable branch-indicator dots (glide
    /// to that branch; active one highlighted); it always carries a "+" that
    /// starts a reply to `node` (a new branch for a post with replies, the first
    /// reply for a leaf).
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
        // The draft/streaming overlay child is not a real branch the user can
        // navigate to; count only persisted siblings for the dots.
        let real_children: Vec<&TreeNode> = node
            .children
            .iter()
            .filter(|c| matches!(c.src, NodeSrc::Msg(_)))
            .collect();
        let count = real_children.len();

        let mut row = h_flex().items_center().gap_3();
        if count >= 2 {
            let active = self.active_child_index(&node.id, page_width, node.children.len());
            let dots = h_flex()
                .gap_1()
                .children(real_children.iter().enumerate().map(|(i, child)| {
                    let color = if i == active {
                        active_color
                    } else {
                        active_color.alpha(0.5)
                    };
                    let pid = parent_id.clone();
                    // The branch scroller indexes *all* children (including the
                    // overlay), so glide to this dot's position among them.
                    let all_idx = node
                        .children
                        .iter()
                        .position(|c| c.id == child.id)
                        .unwrap_or(i);
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
                    this.start_reply(add_id.clone(), window, cx);
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
