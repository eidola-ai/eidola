//! Models settings pane — on-device inference via llama.cpp.
//!
//! Three quiet sections in the settings voice (hairline rows, no cards):
//! the **engine** line (resolved `llama-server` path, or an install hint),
//! the **installed** models (status + per-state verbs: load / unload /
//! cancel / delete, with download progress and honest inline errors), and
//! the curated **Gemma 4 catalog** plus a paste-a-URL row for anything
//! else (a direct `.gguf` link or a Hugging Face file page).
//!
//! All state lives in `LocalModelsStore` (refreshed by
//! `Change::LocalModels`); this view is a lens. Operations route through
//! store methods, which write through app-core — the downloads and engine
//! processes themselves are core-owned, so closing this window interrupts
//! nothing.

use eidola_app_core::{LOCAL_MODEL_CATALOG, LocalModelInfo, LocalModelStatus};
use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Subscription, Window, div,
    prelude::FluentBuilder, px,
};
use gpui_component::{
    ActiveTheme, StyledExt, h_flex,
    input::{Input, InputEvent, InputState},
    label::Label,
    v_flex,
};

use crate::probe::Probe as _;
use crate::stores::{LocalModelsStore, Stores};

pub struct ModelsSettingsView {
    local_models: Entity<LocalModelsStore>,
    /// The paste-a-URL input.
    url_state: Entity<InputState>,
    _subscriptions: Vec<Subscription>,
}

impl ModelsSettingsView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let local_models = stores.local_models.clone();
        let url_state = cx.new(|cx| {
            InputState::new(window, cx).placeholder("https://huggingface.co/…/model.gguf")
        });

        let _subscriptions = vec![
            cx.observe(&local_models, |_, _, cx| cx.notify()),
            // Enter in the URL field submits, mirroring the button.
            cx.subscribe_in(
                &url_state,
                window,
                |this, _, event: &InputEvent, window, cx| {
                    if matches!(event, InputEvent::PressEnter { .. }) {
                        this.submit_url(window, cx);
                    }
                },
            ),
        ];

        Self {
            local_models,
            url_state,
            _subscriptions,
        }
    }

    /// Start downloading whatever URL is in the paste field. Public so
    /// behavior tests share the button/Enter path.
    pub fn submit_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let url = self.url_state.read(cx).value().trim().to_string();
        if url.is_empty() {
            return;
        }
        self.local_models.update(cx, |s, cx| s.download(url, cx));
        let url_state = self.url_state.clone();
        url_state.update(cx, |s, cx| s.set_value("", window, cx));
    }

    /// Start a curated-catalog download. Public for behavior tests.
    pub fn download_catalog(&mut self, url: &str, cx: &mut Context<Self>) {
        self.local_models
            .update(cx, |s, cx| s.download(url.to_string(), cx));
    }

    fn installed_row(
        &self,
        idx: usize,
        model: &LocalModelInfo,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let id = model.id.clone();

        // Status text + the verbs this state affords.
        let (status_text, progress): (String, Option<f32>) = match &model.status {
            LocalModelStatus::Downloading { received, total } => match total {
                Some(t) if *t > 0 => (
                    format!("downloading — {} of {}", fmt_size(*received), fmt_size(*t)),
                    Some((*received as f64 / *t as f64) as f32),
                ),
                _ => (format!("downloading — {}", fmt_size(*received)), None),
            },
            LocalModelStatus::Available => ("downloaded".into(), None),
            LocalModelStatus::Loading => ("starting engine…".into(), None),
            LocalModelStatus::Loaded { port, .. } => {
                (format!("loaded — serving on 127.0.0.1:{port}"), None)
            }
        };

        let mut verbs = h_flex().gap_3().flex_none();
        match &model.status {
            LocalModelStatus::Downloading { .. } => {
                let id_c = id.clone();
                verbs = verbs.child(
                    quiet_verb(("model-cancel", idx), "Cancel", cx)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.local_models
                                .update(cx, |s, cx| s.cancel_download(id_c.clone(), cx));
                        }))
                        .probe(
                            format!("settings/models/installed/{idx}/cancel"),
                            gpui::Role::Button,
                            format!("Cancel download of {}", model.display_name),
                        ),
                );
            }
            LocalModelStatus::Available => {
                let id_l = id.clone();
                let id_d = id.clone();
                verbs = verbs
                    .child(
                        quiet_verb(("model-load", idx), "Load", cx)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.local_models
                                    .update(cx, |s, cx| s.load(id_l.clone(), cx));
                            }))
                            .probe(
                                format!("settings/models/installed/{idx}/load"),
                                gpui::Role::Button,
                                format!("Load {}", model.display_name),
                            ),
                    )
                    .child(
                        quiet_verb(("model-delete", idx), "Delete", cx)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.local_models
                                    .update(cx, |s, cx| s.delete(id_d.clone(), cx));
                            }))
                            .probe(
                                format!("settings/models/installed/{idx}/delete"),
                                gpui::Role::Button,
                                format!("Delete {}", model.display_name),
                            ),
                    );
            }
            LocalModelStatus::Loading => {}
            LocalModelStatus::Loaded { .. } => {
                let id_u = id.clone();
                verbs = verbs.child(
                    quiet_verb(("model-unload", idx), "Unload", cx)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.local_models
                                .update(cx, |s, cx| s.unload(id_u.clone(), cx));
                        }))
                        .probe(
                            format!("settings/models/installed/{idx}/unload"),
                            gpui::Role::Button,
                            format!("Unload {}", model.display_name),
                        ),
                );
            }
        }

        let loaded = matches!(model.status, LocalModelStatus::Loaded { .. });
        let mut row = v_flex()
            .w_full()
            .py_2()
            .gap_1()
            .when(idx > 0, |r| r.border_t_1().border_color(theme.border))
            .child(
                h_flex()
                    .w_full()
                    .gap_4()
                    .items_baseline()
                    .child(
                        h_flex()
                            .gap_2()
                            .items_baseline()
                            .flex_1()
                            .min_w_0()
                            .child(div().child(SharedString::from(model.display_name.clone())))
                            .child(div().text_sm().text_color(theme.muted_foreground).child(
                                SharedString::from(
                                    model.size_bytes.map(fmt_size).unwrap_or_default(),
                                ),
                            )),
                    )
                    .child(verbs),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(if loaded {
                        theme.link
                    } else {
                        theme.muted_foreground
                    })
                    .child(SharedString::from(status_text)),
            );

        // A thin determinate progress bar while downloading.
        if let Some(fraction) = progress {
            row = row.child(
                div()
                    .w_full()
                    .h(px(3.))
                    .rounded_full()
                    .bg(theme.muted)
                    .child(
                        div()
                            .h_full()
                            .rounded_full()
                            .bg(theme.link)
                            .w(gpui::relative(fraction.clamp(0.0, 1.0))),
                    ),
            );
        }

        if let Some(err) = &model.last_error {
            let first_line = err.lines().next().unwrap_or(err).to_string();
            row = row.child(
                div()
                    .id(("model-error", idx))
                    .probe(
                        format!("settings/models/installed/{idx}/error"),
                        gpui::Role::Alert,
                        first_line.clone(),
                    )
                    .text_sm()
                    .text_color(theme.danger)
                    .child(SharedString::from(first_line)),
            );
        }

        row
    }
}

impl Render for ModelsSettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let store = self.local_models.read(cx);
        let engine_path = store.engine_path();
        let models: Vec<LocalModelInfo> = store.models().to_vec();
        let op_error = store.op_error().map(|s| s.to_string());

        let mut col = v_flex().id("models-pane").px_6().py_5().gap_4().w_full();

        // --- Engine ------------------------------------------------------
        col = col.child(section_header("Local inference", cx));
        col = col.child(match &engine_path {
            Some(p) => div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(format!("Engine: llama-server at {p}"))),
            None => div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(
                    "llama-server not found. Install llama.cpp (e.g. `brew install llama.cpp`) \
                     to load models on this device; downloads work either way.",
                )),
        });

        if let Some(err) = op_error {
            col = col.child(
                div()
                    .id("models-op-error")
                    .probe("settings/models/error", gpui::Role::Alert, err.clone())
                    .child(error_banner(&err, cx)),
            );
        }

        // --- Installed models --------------------------------------------
        if !models.is_empty() {
            col = col.child(section_header("On this device", cx));
            let mut list = v_flex().w_full();
            for (idx, model) in models.iter().enumerate() {
                list = list.child(self.installed_row(idx, model, cx));
            }
            col = col.child(list);
        }

        // --- Catalog -------------------------------------------------------
        col = col.child(section_header("Gemma 4 — official releases", cx));
        let mut catalog = v_flex().w_full();
        for (idx, entry) in LOCAL_MODEL_CATALOG.iter().enumerate() {
            let installed = models.iter().any(|m| m.file_name == entry.file_name);
            let mut row = h_flex()
                .w_full()
                .py_2()
                .gap_4()
                .items_baseline()
                .when(idx > 0, |r| r.border_t_1().border_color(theme.border))
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .gap_0p5()
                        .child(
                            h_flex()
                                .gap_2()
                                .items_baseline()
                                .child(div().child(entry.display_name))
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(theme.muted_foreground)
                                        .child(SharedString::from(fmt_size(entry.size_bytes))),
                                ),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.muted_foreground)
                                .child(entry.description),
                        ),
                );
            if installed {
                row = row.child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .flex_none()
                        .child("Installed"),
                );
            } else {
                let url = entry.url;
                row = row.child(
                    quiet_verb(("catalog-download", idx), "Download", cx)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.download_catalog(url, cx);
                        }))
                        .probe(
                            format!("settings/models/catalog/{idx}/download"),
                            gpui::Role::Button,
                            format!("Download {}", entry.display_name),
                        ),
                );
            }
            catalog = catalog.child(row);
        }
        col = col.child(catalog);

        // --- From a URL ----------------------------------------------------
        col = col.child(section_header("From a URL", cx));
        col = col.child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child("Paste a direct .gguf link or a Hugging Face file page."),
        );
        col = col.child(
            h_flex()
                .w_full()
                .gap_2()
                .child(
                    div()
                        .id("models-url-wrap")
                        .probe("settings/models/url", gpui::Role::TextInput, "Model URL")
                        .flex_1()
                        .min_w_0()
                        .child(Input::new(&self.url_state)),
                )
                .child(
                    quiet_verb("models-url-download", "Download", cx)
                        .on_click(cx.listener(|this, _, window, cx| this.submit_url(window, cx)))
                        .probe(
                            "settings/models/url/download",
                            gpui::Role::Button,
                            "Download from URL",
                        ),
                ),
        );

        col
    }
}

// --- Local helpers (the settings voice — mirrors general.rs) --------------

fn section_header(label: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_color(theme.muted_foreground)
        .text_sm()
        .font_medium()
        .child(SharedString::from(label.to_string()))
}

fn error_banner(message: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .gap_2()
        .px_3()
        .py_2()
        .rounded_md()
        .bg(theme.danger.opacity(0.08))
        .text_color(theme.danger)
        .child(Label::new(SharedString::from(message.to_string())))
}

/// A quiet text verb (the settings surface's button voice).
fn quiet_verb(
    id: impl Into<gpui::ElementId>,
    label: &'static str,
    cx: &gpui::App,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    div()
        .id(id)
        .cursor_pointer()
        .flex_none()
        .text_sm()
        .text_color(theme.link)
        .hover(|s| s.text_color(theme.link_hover))
        .child(label)
}

/// Human-readable byte size for model rows.
fn fmt_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1e6)
    } else {
        format!("{:.0} KB", (bytes as f64 / 1e3).max(1.0))
    }
}
