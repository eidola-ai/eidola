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

use crate::plans::format_credits;
use crate::probe::Probe as _;
use eidola_app_core::ModelInfo;

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

    /// Re-fetch one backend's model catalog — the panel's per-backend retry
    /// (after a failed fetch) and refresh (over a good one) affordance. The
    /// store owns the task slot; the panel stays open so the refreshing →
    /// result transition is visible in place, and the other backends'
    /// catalogs are untouched.
    fn refresh_backend_models(&mut self, backend_id: String, cx: &mut Context<Self>) {
        self.stores
            .models
            .update(cx, |s, cx| s.refresh_backend(backend_id, cx));
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

        // The addressee — who the ask goes to. Mirrors the byline's display
        // pair: the model's human name over its backend's name (text_xs,
        // muted); opens the request panel.
        let model = self.current_model(cx);
        let (model_name, backend_name) = self.model_display(&model, cx);
        col = col.child(
            h_flex()
                .id("space-model-chip")
                .probe(
                    "space/composer/model",
                    gpui::Role::Button,
                    SharedString::from(format!("Model: {model_name}, via {backend_name}")),
                )
                .px_1()
                .ml_neg_1()
                .mt_0p5()
                .rounded_md()
                .cursor_pointer()
                .items_start()
                .gap_1()
                .text_xs()
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                .child(
                    v_flex()
                        .items_start()
                        .child(model_name.clone())
                        // Suppressed when it would just repeat the name
                        // (mirrors the gutter's rule).
                        .when(backend_name != model_name, |d| {
                            d.child(div().text_color(hint_fg).child(backend_name))
                        }),
                )
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

        // One group per backend. Engine-backed groups lead — models on this
        // machine outrank any catalog for immediacy: the managed local
        // store, then each enabled llamacpp backend, listing **every**
        // on-disk model (a request against an unloaded one loads its engine
        // on demand; the per-row info line says which are loaded). The
        // fetch-based catalogs (eidola, then openai backends) follow, each
        // carrying its own health so a dead server degrades to *its own*
        // retry footer while every other group stays selectable — the panel
        // is never a dead end.
        let local_store = self.stores.local_models.read(cx);
        let backends_store = self.stores.backends.read(cx);
        let mut engine_groups: Vec<(String, String, Vec<eidola_app_core::LocalModelInfo>)> =
            Vec::new();
        if backends_store.is_enabled(eidola_app_core::LOCAL_BACKEND_ID) {
            let selectable = local_store.selectable_models();
            if !selectable.is_empty() {
                let header = backends_store
                    .get(eidola_app_core::LOCAL_BACKEND_ID)
                    .map(|b| b.display_name.clone())
                    .unwrap_or_else(|| "Local".into());
                engine_groups.push(("local".into(), header, selectable));
            }
        }
        for ext in local_store.external() {
            if !ext.enabled {
                continue;
            }
            let selectable = local_store.external_selectable_models(&ext.backend_id);
            if !selectable.is_empty() {
                engine_groups.push((ext.backend_id.clone(), ext.display_name.clone(), selectable));
            }
        }
        let catalogs: Vec<crate::stores::BackendCatalog> =
            self.stores.models.read(cx).catalogs().to_vec();

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

        if engine_groups.is_empty() && catalogs.is_empty() {
            // Honest note when nothing is selectable yet — say what a send
            // will actually use. Rendered inline (no early return) so any
            // catalog footers below stay actionable.
            let (name, backend) = self.model_display(&current, cx);
            panel = panel.child(
                div()
                    .px_3()
                    .py_2()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(format!(
                        "Model list unavailable — asks use {name} (via {backend})."
                    ))),
            );
        }

        let mut first_group = true;

        // Engine-backed groups: every on-disk model, its load state told
        // honestly in the info line.
        for (backend_id, header, models) in &engine_groups {
            panel = panel.child(group_header(header.clone(), !first_group, cx));
            first_group = false;
            for (idx, model) in models.iter().enumerate() {
                let probe_name = format!("space/request-panel/engine/{backend_id}/{idx}");
                let info = match model.status {
                    eidola_app_core::LocalModelStatus::Loaded {
                        context_tokens,
                        pinned,
                        ..
                    } => format!(
                        "loaded{} · {}-token context · no charge",
                        if pinned { " · pinned" } else { "" },
                        format_credits(context_tokens as i64)
                    ),
                    eidola_app_core::LocalModelStatus::Loading => {
                        "starting engine… · no charge".to_string()
                    }
                    _ => "loads on request · no charge".to_string(),
                };
                panel = panel.child(self.panel_model_row(
                    probe_name,
                    model.display_name.clone(),
                    model.id.clone(),
                    info,
                    &current,
                    &default_model,
                    idx > 0,
                    cx,
                ));
            }
        }

        // Fetch-based catalogs, each with its own status footer.
        for catalog in &catalogs {
            let backend_id = catalog.backend.id.clone();
            let header = if catalog.backend.kind == eidola_app_core::BackendKind::Eidola {
                "Via Eidola".to_string()
            } else {
                catalog.backend.display_name.clone()
            };
            panel = panel.child(group_header(header, !first_group, cx));
            first_group = false;

            let models = catalog.models.value().map(|v| v.as_slice()).unwrap_or(&[]);
            for (idx, model) in models.iter().enumerate() {
                let probe_name = format!("space/request-panel/remote/{backend_id}/{idx}");
                panel = panel.child(self.panel_model_row(
                    probe_name,
                    model.id.clone(),
                    model.id.clone(),
                    model_info_line(model),
                    &current,
                    &default_model,
                    idx > 0,
                    cx,
                ));
            }

            // This backend's status + retry/refresh: refreshing → a quiet
            // in-flight note; a failed/offline fetch → "Retry" quoting the
            // error; a good list → "Refresh" (offered even over success).
            let refreshing = catalog.models.is_loading() || catalog.models.is_stale();
            let error = catalog.models.error().map(|e| e.to_string());
            if refreshing {
                panel = panel.child(
                    div()
                        .w_full()
                        .px_3()
                        .py_2()
                        .text_xs()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child("Refreshing models…"),
                );
            } else if let Some(err) = error {
                let id_retry = backend_id.clone();
                panel = panel.child(
                    v_flex()
                        .id(SharedString::from(format!(
                            "space-request-panel-retry-{backend_id}"
                        )))
                        .probe(
                            format!("space/request-panel/{backend_id}/retry"),
                            gpui::Role::Button,
                            format!("Retry loading {}'s models", catalog.backend.display_name),
                        )
                        .w_full()
                        .px_3()
                        .py_2()
                        .gap_0p5()
                        .cursor_pointer()
                        .hover(|s| s.bg(cx.theme().muted.opacity(0.5)))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.refresh_backend_models(id_retry.clone(), cx)
                        }))
                        .child(div().text_xs().text_color(theme.muted_foreground).child(
                            SharedString::from(format!("Couldn't load the model list — {err}")),
                        ))
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(theme.foreground)
                                .child("Retry"),
                        ),
                );
            } else {
                if models.is_empty() {
                    panel = panel.child(
                        div()
                            .px_3()
                            .py_1()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child("No models listed."),
                    );
                }
                let id_refresh = backend_id.clone();
                panel = panel.child(
                    div()
                        .id(SharedString::from(format!(
                            "space-request-panel-refresh-{backend_id}"
                        )))
                        .probe(
                            format!("space/request-panel/{backend_id}/refresh"),
                            gpui::Role::Button,
                            format!("Refresh {}'s models", catalog.backend.display_name),
                        )
                        .w_full()
                        .px_3()
                        .py_1p5()
                        .text_xs()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .cursor_pointer()
                        .hover(|s| s.text_color(cx.theme().foreground))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.refresh_backend_models(id_refresh.clone(), cx)
                        }))
                        .child("Refresh models"),
                );
            }
        }

        if current != default_model {
            // Quiet, secondary: persist this space's model as the config
            // default. Only offered when it would change anything. Displays
            // the human name; the persisted value stays the selection id.
            let (name, _) = self.model_display(&current, cx);
            let label = format!("Set {name} as default");
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

    /// One selectable model row of the request panel: the name line with
    /// current/default markers, and an honest info line. Shared by the
    /// engine-backed and fetch-based groups (they differ only in what the
    /// info line says).
    #[allow(clippy::too_many_arguments)]
    fn panel_model_row(
        &self,
        probe_name: String,
        display: String,
        model_id: String,
        info: String,
        current: &str,
        default_model: &str,
        bordered: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let is_current = model_id == current;
        let is_default = model_id == default_model;

        let mut markers: Vec<&str> = Vec::new();
        if is_current {
            markers.push("current");
        }
        if is_default {
            markers.push("default");
        }

        let mut name = div().text_sm().child(SharedString::from(display));
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

        let id = model_id.clone();
        v_flex()
            .id(SharedString::from(format!("row-{probe_name}")))
            .probe(probe_name, gpui::Role::ListBoxOption, model_id)
            .aria_selected(is_current)
            .w_full()
            .px_3()
            .py_2()
            .gap_0p5()
            .when(bordered, |d| d.border_t_1().border_color(theme.border))
            .cursor_pointer()
            .hover(|s| s.bg(cx.theme().muted.opacity(0.5)))
            .on_click(cx.listener(move |this, _, _, cx| this.select_model(id.clone(), cx)))
            .child(name_row)
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(info)),
            )
    }
}

/// A small muted keyboard hint beside a verb (shown while ⌥ is held).
fn kbd_hint(text: &'static str, color: gpui::Hsla) -> gpui::Div {
    div().text_xs().text_color(color).child(text)
}

/// A quiet group header inside the request panel — one per backend group.
fn group_header(label: String, bordered: bool, cx: &gpui::App) -> gpui::Div {
    let theme = cx.theme();
    let mut header = div()
        .px_3()
        .pt_2()
        .pb_0p5()
        .text_xs()
        .italic()
        .text_color(theme.muted_foreground)
        .child(SharedString::from(label));
    if bordered {
        header = header.border_t_1().border_color(theme.border);
    }
    header
}

/// One honest line of per-model info for the panel, from the `/models`
/// payload: context length plus the credit rates that will actually be
/// charged. Per-request models show their flat rate; if the payload carried
/// no pricing at all, only the context length is shown — we don't invent
/// numbers.
fn model_info_line(model: &ModelInfo) -> String {
    // Per-request models (e.g. transcription) report no meaningful context
    // length; showing "0-token context" would be noise, not honesty.
    let ctx = (model.context_length > 0).then(|| {
        format!(
            "{}-token context",
            format_credits(model.context_length as i64)
        )
    });
    let price = if let Some(request) = model.request_credits {
        Some(format!("{} credits per request", format_rate(request)))
    } else if model.prompt_credits_per_token > 0.0 || model.completion_credits_per_token > 0.0 {
        Some(format!(
            "{} in / {} out credits per token",
            format_rate(model.prompt_credits_per_token),
            format_rate(model.completion_credits_per_token)
        ))
    } else {
        None
    };
    match (ctx, price) {
        (Some(ctx), Some(price)) => format!("{ctx} · {price}"),
        (Some(ctx), None) => ctx,
        (None, Some(price)) => price,
        (None, None) => "no published details".to_string(),
    }
}

/// Format a credit rate: whole thousands get separators ("9,000"),
/// everything else shows up to three decimals with trailing zeros trimmed
/// ("1.500" → "1.5", "0.530" → "0.53", "3.000" → "3").
fn format_rate(rate: f64) -> String {
    if rate >= 1000.0 && rate.fract() == 0.0 {
        return format_credits(rate as i64);
    }
    let s = format!("{rate:.3}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}
