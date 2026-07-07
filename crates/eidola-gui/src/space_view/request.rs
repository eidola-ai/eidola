//! The request affordances — how a draft becomes a request.
//!
//! The composer's **action gutter** (the right-hand mirror of the byline
//! gutter) is where submission becomes discoverable: **Ask** posts the draft
//! and requests a response (⌘↩, the common gesture); holding ⌥ reveals
//! **Post** (⌘⇧↩, save without asking) plus the keyboard hints — the
//! "Option reveals power" pattern. Below the verbs sits the *addressee*: the
//! model the ask goes to, which opens the **request panel**.
//!
//! The affordances appear only once they are actionable — a draft with
//! content (or ⌥ held / the panel open). The blank page stays sacred.
//!
//! The request panel is the home of per-request configuration. Today that is
//! model selection (honest per-model info from `/models`, current + default
//! markers, a quiet set-as-default footer); when app-core plumbs sampling
//! parameters (temperature, thinking/no-thinking) through `run_turn`, they
//! join this panel as rows below the model list — same anchor, same gesture.

use gpui::{
    AnyElement, Context, Entity, FontWeight, InteractiveElement, IntoElement, ParentElement,
    Pixels, SharedString, StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder,
    px,
};
use gpui_component::{ActiveTheme, StyledExt, h_flex, v_flex};
use gpui_markdown_editor::MarkdownEditorState;

use crate::chat::model_info_line;
use crate::probe::Probe as _;

use super::layout::body_width;
use super::{
    GUTTER_GAP, POST_PAD_Y, PostOnly, Send, SpaceView, TITLE_BAR_RESERVE, ToggleModelPicker,
};

/// Width of the request panel.
const PANEL_WIDTH: Pixels = px(340.);
/// Cap on the panel height — a long model list scrolls internally.
const PANEL_MAX_HEIGHT: f32 = 420.;
/// Vertical drop from the composer bar's top to a downward-opening panel:
/// clears the gutter's top padding plus the verb + model rows.
const PANEL_DROP: f32 = 76.;

impl SpaceView {
    // -- Panel state ---------------------------------------------------------

    /// Toggle the request panel (the model chip's click, or ⌥⌘M).
    pub fn toggle_request_panel(&mut self, cx: &mut Context<Self>) {
        self.request_panel_open = !self.request_panel_open;
        cx.notify();
    }

    /// Close the request panel (Esc, click-out, selection).
    pub fn close_request_panel(&mut self, cx: &mut Context<Self>) {
        if self.request_panel_open {
            self.request_panel_open = false;
            cx.notify();
        }
    }

    /// Whether the request panel is open.
    pub fn request_panel_open(&self) -> bool {
        self.request_panel_open
    }

    /// ⌥⌘M — meaningful only while a composer is open (the panel anchors to
    /// its action gutter).
    pub(crate) fn toggle_request_panel_action(
        &mut self,
        _: &ToggleModelPicker,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.active_draft.is_some() {
            self.toggle_request_panel(cx);
        }
    }

    /// Choose the model for this space's subsequent sends and close the
    /// panel. The selection lives on the shared `Space` (per-space, not
    /// per-window); a switch while a response is streaming applies to the
    /// *next* send.
    pub fn select_model(&mut self, id: String, cx: &mut Context<Self>) {
        self.space.update(cx, |s, cx| s.select_model(id, cx));
        self.request_panel_open = false;
        cx.notify();
    }

    /// Persist the current model as the config default — the panel's quiet
    /// footer affordance. The panel stays open so the moved "default" marker
    /// is the visible confirmation.
    fn set_current_model_as_default(&mut self, cx: &mut Context<Self>) {
        let model = self.current_model(cx);
        self.stores
            .config
            .update(cx, |c, cx| c.set_default_model(model, cx));
        cx.notify();
    }

    // -- The action gutter ---------------------------------------------------

    /// The active composer's action gutter: Ask (⌘↩), the ⌥-revealed Post
    /// (⌘⇧↩), and the model chip that opens the request panel. Renders as an
    /// empty (reserved) gutter until the draft has content, ⌥ is held, or the
    /// panel is open — the affordance appears the moment it's actionable.
    pub(crate) fn render_composer_actions(
        &self,
        editor: &Entity<MarkdownEditorState>,
        cx: &Context<Self>,
    ) -> gpui::Div {
        let theme = cx.theme();
        let alt = self.window_input.read(cx).alt_held();
        let revealed = !editor.read(cx).is_empty() || alt || self.request_panel_open;

        let mut col = super::post::action_gutter().gap_0p5();
        if !revealed {
            return col;
        }

        let fg = theme.muted_foreground;
        let fg_hover = theme.foreground;
        let bg_hover = theme.muted;
        let hint_fg = theme.muted_foreground.opacity(0.7);

        // Ask — post the draft *and* request a response (the common gesture).
        col = col.child(
            h_flex()
                .id("space-ask")
                .probe(
                    "space/composer/ask",
                    gpui::Role::Button,
                    "Ask — post and request a response",
                )
                .px_1()
                .ml_neg_1()
                .rounded_md()
                .cursor_pointer()
                .items_baseline()
                .gap_1p5()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                .child("Ask")
                .when(alt, |d| d.child(kbd_hint("⌘↩", hint_fg)))
                .on_click(cx.listener(|this, _, window, cx| this.submit(&Send, window, cx))),
        );

        // Post — save without asking. ⌥-revealed: present when reached for,
        // invisible in the common case.
        if alt {
            col = col.child(
                h_flex()
                    .id("space-post-only")
                    .probe(
                        "space/composer/post",
                        gpui::Role::Button,
                        "Post without asking",
                    )
                    .px_1()
                    .ml_neg_1()
                    .rounded_md()
                    .cursor_pointer()
                    .items_baseline()
                    .gap_1p5()
                    .text_sm()
                    .text_color(fg)
                    .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                    .child("Post")
                    .child(kbd_hint("⇧⌘↩", hint_fg))
                    .on_click(
                        cx.listener(|this, _, window, cx| this.post_only(&PostOnly, window, cx)),
                    ),
            );
        }

        // The addressee — who the ask goes to. Mirrors the time line under the
        // byline (text_xs, muted); opens the request panel.
        let model = self.current_model(cx);
        col = col.child(
            h_flex()
                .id("space-model-chip")
                .probe(
                    "space/composer/model",
                    gpui::Role::Button,
                    SharedString::from(format!("Model: {model}")),
                )
                .px_1()
                .ml_neg_1()
                .mt_0p5()
                .rounded_md()
                .cursor_pointer()
                .items_baseline()
                .gap_1()
                .text_xs()
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                .child(SharedString::from(model))
                .child(div().text_color(hint_fg).child(if self.request_panel_open {
                    "⌃"
                } else {
                    "⌄"
                }))
                .on_click(cx.listener(|this, _, _, cx| this.toggle_request_panel(cx))),
        );

        col
    }

    // -- The request panel ---------------------------------------------------

    /// The request panel, anchored to the composer's action gutter: the model
    /// list with honest per-model info. Opens downward when the composer sits
    /// high on the page (a docked blank notebook), upward when it floats near
    /// the bottom. Renders nothing when closed or without an active composer.
    pub(crate) fn render_request_panel(
        &self,
        page_width: Pixels,
        window_h: Pixels,
        cx: &Context<Self>,
    ) -> AnyElement {
        if !self.request_panel_open || self.active_draft.is_none() {
            return div().into_any_element();
        }
        let theme = cx.theme();
        let bw = body_width(page_width);
        let win = window_h.as_f32();

        // Anchor x: the action gutter's left edge, clamped into the window.
        let gutter_x = page_width.as_f32() / 2.0 + bw / 2.0 + GUTTER_GAP.as_f32();
        let left = gutter_x
            .min(page_width.as_f32() - PANEL_WIDTH.as_f32() - 12.0)
            .max(12.0);

        // Anchor y: the composer bar's top, recorded this frame.
        let anchor_top = self.composer_anchor_top.get();
        let half_pad = POST_PAD_Y.as_f32() / 2.0;
        let open_down = anchor_top < win * 0.5;
        let max_h = if open_down {
            (win - (anchor_top + half_pad + PANEL_DROP) - 12.0).min(PANEL_MAX_HEIGHT)
        } else {
            (anchor_top - TITLE_BAR_RESERVE.as_f32() - 12.0).min(PANEL_MAX_HEIGHT)
        }
        .max(120.0);

        let current = self.current_model(cx);
        let default_model = self
            .stores
            .config
            .read(cx)
            .state()
            .map(|s| s.default_model.clone())
            .unwrap_or_else(|| eidola_app_core::config::DEFAULT_MODEL.to_string());
        let models = self.stores.models.read(cx).list().to_vec();

        let mut panel = v_flex()
            .id("space-request-panel")
            .probe("space/request-panel", gpui::Role::ListBox, "Models")
            .occlude()
            .absolute()
            .left(px(left))
            .w(PANEL_WIDTH)
            .max_h(px(max_h))
            .overflow_y_scroll()
            .popover_style(cx)
            .py_1()
            .on_mouse_down_out(cx.listener(|this, _, _, cx| this.close_request_panel(cx)));
        panel = if open_down {
            panel.top(px(anchor_top + half_pad + PANEL_DROP))
        } else {
            panel.bottom(px(win - anchor_top + 8.0))
        };

        if models.is_empty() {
            // Honest empty state: the model list hasn't loaded (or the fetch
            // failed) — say what a send will actually use.
            return panel
                .child(
                    div()
                        .px_3()
                        .py_2()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(format!(
                            "Model list unavailable — asks use {current}."
                        ))),
                )
                .into_any_element();
        }

        for (idx, model) in models.iter().enumerate() {
            let is_current = model.id == current;
            let is_default = model.id == default_model;

            let mut markers: Vec<&str> = Vec::new();
            if is_current {
                markers.push("current");
            }
            if is_default {
                markers.push("default");
            }

            let mut name = div().text_sm().child(SharedString::from(model.id.clone()));
            if is_current {
                name = name.font_weight(FontWeight::SEMIBOLD);
            }
            let mut name_row = h_flex()
                .w_full()
                .justify_between()
                .items_baseline()
                .child(name);
            if !markers.is_empty() {
                name_row = name_row.child(
                    div()
                        .text_xs()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(markers.join(" · "))),
                );
            }

            let id = model.id.clone();
            panel = panel.child(
                v_flex()
                    .id(("space-model-row", idx))
                    .probe(
                        format!("space/request-panel/row/{idx}"),
                        gpui::Role::ListBoxOption,
                        model.id.clone(),
                    )
                    .aria_selected(is_current)
                    .w_full()
                    .px_3()
                    .py_2()
                    .gap_0p5()
                    .when(idx > 0, |d| d.border_t_1().border_color(theme.border))
                    .cursor_pointer()
                    .hover(|s| s.bg(cx.theme().muted.opacity(0.5)))
                    .on_click(cx.listener(move |this, _, _, cx| this.select_model(id.clone(), cx)))
                    .child(name_row)
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(model_info_line(model))),
                    ),
            );
        }

        if current != default_model {
            // Quiet, secondary: persist this space's model as the config
            // default. Only offered when it would change anything.
            let label = format!("Set {current} as default");
            panel = panel.child(
                div()
                    .id("space-set-default-model")
                    .probe(
                        "space/request-panel/set-default",
                        gpui::Role::Button,
                        label.clone(),
                    )
                    .w_full()
                    .px_3()
                    .py_2()
                    .border_t_1()
                    .border_color(theme.border)
                    .text_xs()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .cursor_pointer()
                    .hover(|s| s.text_color(cx.theme().foreground))
                    .on_click(cx.listener(|this, _, _, cx| this.set_current_model_as_default(cx)))
                    .child(SharedString::from(label)),
            );
        }

        panel.into_any_element()
    }
}

/// A small muted keyboard hint beside a verb (shown while ⌥ is held).
fn kbd_hint(text: &'static str, color: gpui::Hsla) -> gpui::Div {
    div().text_xs().text_color(color).child(text)
}
