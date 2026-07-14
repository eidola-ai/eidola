//! Backends settings pane — *where an ask can be routed*.
//!
//! One quiet section per configured backend, in the settings voice
//! (hairline rows, no cards):
//!
//! - **Eidola** — the confidential service. Enable/disable only (disabling
//!   is the "no account, on-device only" configuration).
//! - **On this device** — the managed local store: the **engine** line
//!   (resolved `llama-server` path, or an install hint), the installed
//!   models (status + per-state verbs: load / unload / cancel / delete,
//!   with download progress and honest inline errors), the curated
//!   **Gemma 4 catalog**, and a paste-a-URL row.
//! - **Each external backend** — an `openai` server shows its URL /
//!   key-state / pinned models; a `llamacpp` backend shows its user-owned
//!   directory and the scanned models with load/unload verbs (never
//!   download/delete — the files are the user's).
//! - **Add a backend** — two quiet affordances that expand an inline form
//!   (OpenAI-compatible server: id + URL + optional key; llama.cpp
//!   directory: id + path).
//!
//! State lives in `BackendsStore` (refreshed by `Change::Backends`) and
//! `LocalModelsStore` (refreshed by `Change::LocalModels`); this view is a
//! lens. Operations route through store methods, which write through
//! app-core — downloads and engine processes are core-owned, so closing
//! this window interrupts nothing.

use eidola_app_core::{
    BackendInfo, BackendKind, LOCAL_MODEL_CATALOG, LocalModelInfo, LocalModelStatus, NewBackend,
};
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
use crate::stores::{BackendsStore, LocalModelsStore, Stores};

/// Which kind the inline add-backend form is collecting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddKind {
    OpenAi,
    LlamaCpp,
}

/// The inline add-backend form's inputs.
struct AddForm {
    kind: AddKind,
    id_state: Entity<InputState>,
    /// Base URL (openai) — unused for llamacpp.
    url_state: Entity<InputState>,
    /// API key (openai, optional) — unused for llamacpp.
    key_state: Entity<InputState>,
    /// Models directory (llamacpp) — unused for openai.
    dir_state: Entity<InputState>,
}

pub struct BackendsSettingsView {
    backends: Entity<BackendsStore>,
    local_models: Entity<LocalModelsStore>,
    /// The paste-a-URL input (the local section's download row).
    url_state: Entity<InputState>,
    /// The inline add-backend form, while one is open.
    add_form: Option<AddForm>,
    _subscriptions: Vec<Subscription>,
}

impl BackendsSettingsView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let backends = stores.backends.clone();
        let local_models = stores.local_models.clone();
        let url_state = cx.new(|cx| {
            InputState::new(window, cx).placeholder("https://huggingface.co/…/model.gguf")
        });

        let _subscriptions = vec![
            cx.observe(&backends, |_, _, cx| cx.notify()),
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
            backends,
            local_models,
            url_state,
            add_form: None,
            _subscriptions,
        }
    }

    // -- Operations (public so behavior tests share the click paths) --------

    /// Start downloading whatever URL is in the paste field.
    pub fn submit_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let url = self.url_state.read(cx).value().trim().to_string();
        if url.is_empty() {
            return;
        }
        self.local_models.update(cx, |s, cx| s.download(url, cx));
        let url_state = self.url_state.clone();
        url_state.update(cx, |s, cx| s.set_value("", window, cx));
    }

    /// Start a curated-catalog download.
    pub fn download_catalog(&mut self, url: &str, cx: &mut Context<Self>) {
        self.local_models
            .update(cx, |s, cx| s.download(url.to_string(), cx));
    }

    /// Flip a backend's enabled state.
    pub fn toggle_backend(&mut self, id: String, enabled: bool, cx: &mut Context<Self>) {
        self.backends
            .update(cx, |s, cx| s.set_enabled(id, enabled, cx));
    }

    /// Remove an external backend.
    pub fn remove_backend(&mut self, id: String, cx: &mut Context<Self>) {
        self.backends.update(cx, |s, cx| s.remove(id, cx));
    }

    /// Open the inline add form for `kind` (idempotent; switching kinds
    /// resets the inputs).
    pub fn begin_add(&mut self, kind: AddKind, window: &mut Window, cx: &mut Context<Self>) {
        if self.add_form.as_ref().is_some_and(|f| f.kind == kind) {
            return;
        }
        let id_state = cx.new(|cx| InputState::new(window, cx).placeholder("my-server"));
        let url_state =
            cx.new(|cx| InputState::new(window, cx).placeholder("http://192.168.1.20:8000"));
        let key_state = cx.new(|cx| InputState::new(window, cx).placeholder("optional API key"));
        let dir_state = cx.new(|cx| InputState::new(window, cx).placeholder("/path/to/models"));
        self.add_form = Some(AddForm {
            kind,
            id_state,
            url_state,
            key_state,
            dir_state,
        });
        cx.notify();
    }

    /// Close the add form without adding.
    pub fn cancel_add(&mut self, cx: &mut Context<Self>) {
        self.add_form = None;
        cx.notify();
    }

    /// Submit the add form. Validation lives in app-core; a refusal
    /// surfaces in the store's `op_error` banner. The form closes
    /// optimistically — a failed add leaves the error visible and the
    /// affordance one click away.
    pub fn submit_add(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.add_form.take() else {
            return;
        };
        let id = form.id_state.read(cx).value().trim().to_string();
        let new = match form.kind {
            AddKind::OpenAi => {
                let url = form.url_state.read(cx).value().trim().to_string();
                let key = form.key_state.read(cx).value().trim().to_string();
                NewBackend {
                    id,
                    kind: BackendKind::OpenAi,
                    display_name: String::new(),
                    base_url: Some(url),
                    api_key: (!key.is_empty()).then_some(key),
                    models_dir: None,
                    model_overrides: None,
                }
            }
            AddKind::LlamaCpp => {
                let dir = form.dir_state.read(cx).value().trim().to_string();
                NewBackend {
                    id,
                    kind: BackendKind::LlamaCpp,
                    display_name: String::new(),
                    base_url: None,
                    api_key: None,
                    models_dir: Some(dir),
                    model_overrides: None,
                }
            }
        };
        self.backends.update(cx, |s, cx| s.add(new, cx));
        cx.notify();
    }

    /// The open add form's kind, if any (test accessor).
    pub fn adding(&self) -> Option<AddKind> {
        self.add_form.as_ref().map(|f| f.kind)
    }

    // -- Sections ------------------------------------------------------------

    /// A backend's heading row: name + a one-line subtitle on the left, the
    /// enable/disable (and for externals, remove) verbs on the right.
    fn backend_header(
        &self,
        backend: &BackendInfo,
        subtitle: String,
        removable: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let id = backend.id.clone();
        let enabled = backend.enabled;

        let mut verbs = h_flex().gap_3().flex_none();
        {
            let id_t = id.clone();
            verbs = verbs.child(
                quiet_verb(
                    SharedString::from(format!("backend-toggle-{}", backend.id)),
                    if enabled { "Disable" } else { "Enable" },
                    cx,
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.toggle_backend(id_t.clone(), !enabled, cx);
                }))
                .probe(
                    format!("settings/backends/{}/toggle", backend.id),
                    gpui::Role::Button,
                    format!(
                        "{} {}",
                        if enabled { "Disable" } else { "Enable" },
                        backend.display_name
                    ),
                ),
            );
        }
        if removable {
            let id_r = id.clone();
            verbs = verbs.child(
                quiet_verb(
                    SharedString::from(format!("backend-remove-{}", backend.id)),
                    "Remove",
                    cx,
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.remove_backend(id_r.clone(), cx);
                }))
                .probe(
                    format!("settings/backends/{}/remove", backend.id),
                    gpui::Role::Button,
                    format!("Remove {}", backend.display_name),
                ),
            );
        }

        v_flex()
            .w_full()
            .gap_0p5()
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
                            .child(section_header(&backend.display_name, cx))
                            .when(!enabled, |r| {
                                r.child(
                                    div()
                                        .text_xs()
                                        .italic()
                                        .text_color(theme.muted_foreground)
                                        .child("disabled"),
                                )
                            }),
                    )
                    .child(verbs),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(subtitle)),
            )
    }

    /// One engine-served model row: status line, per-state verbs, an
    /// optional progress bar and inline error. `managed` gates the verbs
    /// that mutate files (cancel/delete) — a llamacpp backend's files are
    /// the user's, so only load/unload apply.
    fn model_row(
        &self,
        probe_prefix: &str,
        idx: usize,
        model: &LocalModelInfo,
        managed: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let id = model.id.clone();

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
            LocalModelStatus::Loaded { port, pinned, .. } => (
                format!(
                    "loaded — serving on 127.0.0.1:{port}{}",
                    if *pinned { " · pinned" } else { "" }
                ),
                None,
            ),
        };

        let mut verbs = h_flex().gap_3().flex_none();
        match &model.status {
            LocalModelStatus::Downloading { .. } => {
                if managed {
                    let id_c = id.clone();
                    verbs = verbs.child(
                        quiet_verb(
                            SharedString::from(format!("model-cancel-{id}")),
                            "Cancel",
                            cx,
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.local_models
                                .update(cx, |s, cx| s.cancel_download(id_c.clone(), cx));
                        }))
                        .probe(
                            format!("{probe_prefix}/{idx}/cancel"),
                            gpui::Role::Button,
                            format!("Cancel download of {}", model.display_name),
                        ),
                    );
                }
            }
            LocalModelStatus::Available => {
                let id_l = id.clone();
                verbs = verbs.child(
                    quiet_verb(SharedString::from(format!("model-load-{id}")), "Load", cx)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.local_models
                                .update(cx, |s, cx| s.load(id_l.clone(), cx));
                        }))
                        .probe(
                            format!("{probe_prefix}/{idx}/load"),
                            gpui::Role::Button,
                            format!("Load {}", model.display_name),
                        ),
                );
                if managed {
                    let id_d = id.clone();
                    verbs = verbs.child(
                        quiet_verb(
                            SharedString::from(format!("model-delete-{id}")),
                            "Delete",
                            cx,
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.local_models
                                .update(cx, |s, cx| s.delete(id_d.clone(), cx));
                        }))
                        .probe(
                            format!("{probe_prefix}/{idx}/delete"),
                            gpui::Role::Button,
                            format!("Delete {}", model.display_name),
                        ),
                    );
                }
            }
            LocalModelStatus::Loading => {}
            LocalModelStatus::Loaded { pinned, .. } => {
                // Pin protects the engine from *automatic* (LRU) unloading
                // when another model needs the memory; the manual Unload
                // verb stays available either way.
                let pinned = *pinned;
                let id_p = id.clone();
                verbs = verbs.child(
                    quiet_verb(
                        SharedString::from(format!("model-pin-{id}")),
                        if pinned { "Unpin" } else { "Pin" },
                        cx,
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.local_models
                            .update(cx, |s, cx| s.set_pinned(id_p.clone(), !pinned, cx));
                    }))
                    .probe(
                        format!("{probe_prefix}/{idx}/pin"),
                        gpui::Role::Button,
                        format!(
                            "{} {}",
                            if pinned { "Unpin" } else { "Pin" },
                            model.display_name
                        ),
                    ),
                );
                let id_u = id.clone();
                verbs = verbs.child(
                    quiet_verb(
                        SharedString::from(format!("model-unload-{id}")),
                        "Unload",
                        cx,
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.local_models
                            .update(cx, |s, cx| s.unload(id_u.clone(), cx));
                    }))
                    .probe(
                        format!("{probe_prefix}/{idx}/unload"),
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
                    .id(SharedString::from(format!("model-error-{id}")))
                    .probe(
                        format!("{probe_prefix}/{idx}/error"),
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

    /// The managed local section: engine line, installed models, catalog,
    /// paste-a-URL.
    fn local_section(&self, backend: &BackendInfo, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let theme = cx.theme();
        let store = self.local_models.read(cx);
        let engine_path = store.engine_path();
        let models: Vec<LocalModelInfo> = store.models().to_vec();

        let mut out: Vec<gpui::AnyElement> = Vec::new();
        out.push(
            self.backend_header(
                backend,
                "Models on this machine, run by a managed llama.cpp engine — no charge, \
                 no account."
                    .into(),
                false,
                cx,
            )
            .into_any_element(),
        );
        out.push(
            match &engine_path {
                Some(p) => div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(format!("Engine: llama-server at {p}"))),
                None => {
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(
                            "llama-server not found. Install llama.cpp (e.g. `brew install \
                         llama.cpp`) to load models on this device; downloads work either way.",
                        ))
                }
            }
            .into_any_element(),
        );

        if !models.is_empty() {
            let mut list = v_flex().w_full();
            for (idx, model) in models.iter().enumerate() {
                list = list.child(self.model_row(
                    "settings/backends/local/installed",
                    idx,
                    model,
                    true,
                    cx,
                ));
            }
            out.push(list.into_any_element());
        }

        // Curated catalog.
        out.push(subsection_header("Gemma 4 — official releases", cx).into_any_element());
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
                            format!("settings/backends/local/catalog/{idx}/download"),
                            gpui::Role::Button,
                            format!("Download {}", entry.display_name),
                        ),
                );
            }
            catalog = catalog.child(row);
        }
        out.push(catalog.into_any_element());

        // From a URL.
        out.push(subsection_header("From a URL", cx).into_any_element());
        out.push(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child("Paste a direct .gguf link or a Hugging Face file page.")
                .into_any_element(),
        );
        out.push(
            h_flex()
                .w_full()
                .gap_2()
                .child(
                    div()
                        .id("models-url-wrap")
                        .probe(
                            "settings/backends/local/url",
                            gpui::Role::TextInput,
                            "Model URL",
                        )
                        .flex_1()
                        .min_w_0()
                        .child(Input::new(&self.url_state)),
                )
                .child(
                    quiet_verb("models-url-download", "Download", cx)
                        .on_click(cx.listener(|this, _, window, cx| this.submit_url(window, cx)))
                        .probe(
                            "settings/backends/local/url/download",
                            gpui::Role::Button,
                            "Download from URL",
                        ),
                )
                .into_any_element(),
        );
        out
    }

    /// One external backend's section.
    fn external_section(&self, backend: &BackendInfo, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let theme = cx.theme();
        let mut out: Vec<gpui::AnyElement> = Vec::new();

        let subtitle = match backend.kind {
            BackendKind::OpenAi => {
                let key = if backend.has_api_key {
                    " · API key set"
                } else {
                    ""
                };
                format!(
                    "OpenAI-compatible server at {}{key} — models address as `<model>@{}`.",
                    backend.base_url.as_deref().unwrap_or("<no URL>"),
                    backend.id,
                )
            }
            BackendKind::LlamaCpp => format!(
                "Your llama.cpp models in {} — Eidola starts and stops engines, never \
                 touches the files.",
                backend.models_dir.as_deref().unwrap_or("<no directory>"),
            ),
            _ => String::new(),
        };
        out.push(
            self.backend_header(backend, subtitle, true, cx)
                .into_any_element(),
        );

        match backend.kind {
            BackendKind::OpenAi => {
                if let Some(pinned) = &backend.model_overrides {
                    out.push(
                        div()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(format!(
                                "Pinned models: {}",
                                pinned.join(", ")
                            )))
                            .into_any_element(),
                    );
                }
            }
            BackendKind::LlamaCpp => {
                // The scanned directory, with load/unload verbs.
                let external = self.local_models.read(cx).external_models(&backend.id);
                if external.is_empty() {
                    out.push(
                        div()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .child("No .gguf files found in the directory.")
                            .into_any_element(),
                    );
                } else {
                    let prefix = format!("settings/backends/{}/model", backend.id);
                    let mut list = v_flex().w_full();
                    for (idx, model) in external.iter().enumerate() {
                        list = list.child(self.model_row(&prefix, idx, model, false, cx));
                    }
                    out.push(list.into_any_element());
                }
            }
            _ => {}
        }
        out
    }

    /// The add-a-backend affordances and (when open) the inline form.
    fn add_section(&self, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let theme = cx.theme();
        let mut out: Vec<gpui::AnyElement> = Vec::new();
        out.push(section_header("Add a backend", cx).into_any_element());
        out.push(
            h_flex()
                .w_full()
                .gap_4()
                .child(
                    quiet_verb("add-openai", "OpenAI-compatible server…", cx)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.begin_add(AddKind::OpenAi, window, cx)
                        }))
                        .probe(
                            "settings/backends/add/openai",
                            gpui::Role::Button,
                            "Add an OpenAI-compatible server",
                        ),
                )
                .child(
                    quiet_verb("add-llamacpp", "llama.cpp directory…", cx)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.begin_add(AddKind::LlamaCpp, window, cx)
                        }))
                        .probe(
                            "settings/backends/add/llamacpp",
                            gpui::Role::Button,
                            "Add a llama.cpp models directory",
                        ),
                )
                .into_any_element(),
        );

        let Some(form) = &self.add_form else {
            return out;
        };

        let mut col = v_flex().w_full().gap_2().pt_1();
        col = col.child(labeled_input(
            "Name",
            "settings/backends/add/id",
            "lowercase letters, digits, hyphens — models address as <model>@<name>",
            &form.id_state,
            cx,
        ));
        match form.kind {
            AddKind::OpenAi => {
                col = col
                    .child(labeled_input(
                        "Base URL",
                        "settings/backends/add/url",
                        "",
                        &form.url_state,
                        cx,
                    ))
                    .child(labeled_input(
                        "API key",
                        "settings/backends/add/key",
                        "optional; sent as a Bearer token",
                        &form.key_state,
                        cx,
                    ));
            }
            AddKind::LlamaCpp => {
                col = col.child(labeled_input(
                    "Models directory",
                    "settings/backends/add/dir",
                    "a folder of .gguf files you manage",
                    &form.dir_state,
                    cx,
                ));
            }
        }
        col = col.child(
            h_flex()
                .gap_4()
                .pt_1()
                .child(
                    quiet_verb("add-submit", "Add", cx)
                        .on_click(cx.listener(|this, _, _, cx| this.submit_add(cx)))
                        .probe(
                            "settings/backends/add/submit",
                            gpui::Role::Button,
                            "Add backend",
                        ),
                )
                .child(
                    div()
                        .id("add-cancel")
                        .cursor_pointer()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .hover(|s| s.text_color(cx.theme().foreground))
                        .child("Cancel")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel_add(cx)))
                        .probe(
                            "settings/backends/add/cancel",
                            gpui::Role::Button,
                            "Cancel adding a backend",
                        ),
                ),
        );
        out.push(col.into_any_element());
        out
    }
}

impl Render for BackendsSettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let backends: Vec<BackendInfo> = self.backends.read(cx).list().to_vec();
        let backends_op_error = self.backends.read(cx).op_error().map(|s| s.to_string());
        let local_op_error = self.local_models.read(cx).op_error().map(|s| s.to_string());

        let mut col = v_flex().id("backends-pane").px_6().py_5().gap_4().w_full();

        // Honest operation errors, pinned at the top of the pane.
        for (probe_name, err) in [
            ("settings/backends/error", backends_op_error),
            ("settings/backends/local/error", local_op_error),
        ] {
            if let Some(err) = err {
                col = col.child(
                    div()
                        .id(SharedString::from(probe_name.to_string()))
                        .probe(probe_name, gpui::Role::Alert, err.clone())
                        .child(error_banner(&err, cx)),
                );
            }
        }

        if backends.is_empty() {
            // The registry hasn't loaded yet (or a fixture-less stub scene):
            // say so rather than rendering a blank pane.
            col = col.child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("Loading backends…"),
            );
        }

        let mut first = true;
        for backend in &backends {
            if !first {
                col = col.child(div().w_full().border_t_1().border_color(theme.border));
            }
            first = false;
            let children: Vec<gpui::AnyElement> = match backend.kind {
                BackendKind::Eidola => {
                    vec![
                        self.backend_header(
                            backend,
                            "Confidential inference via the Eidola service — attested \
                             hardware, anonymous credits. Disable to run with no account, \
                             on-device only."
                                .into(),
                            false,
                            cx,
                        )
                        .into_any_element(),
                    ]
                }
                BackendKind::Local => self.local_section(backend, cx),
                BackendKind::OpenAi | BackendKind::LlamaCpp => self.external_section(backend, cx),
            };
            let mut section = v_flex().w_full().gap_3();
            for child in children {
                section = section.child(child);
            }
            col = col.child(section);
        }

        if !backends.is_empty() {
            col = col.child(div().w_full().border_t_1().border_color(theme.border));
        }
        for child in self.add_section(cx) {
            col = col.child(child);
        }

        col
    }
}

// --- Local helpers (the settings voice — mirrors general.rs) --------------

fn section_header(label: &str, cx: &gpui::App) -> gpui::Div {
    let theme = cx.theme();
    div()
        .text_color(theme.foreground)
        .font_medium()
        .child(SharedString::from(label.to_string()))
}

fn subsection_header(label: &str, cx: &gpui::App) -> impl IntoElement {
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

/// A labelled input row for the add-backend form.
fn labeled_input(
    label: &'static str,
    probe_name: &str,
    hint: &'static str,
    state: &Entity<InputState>,
    cx: &gpui::App,
) -> impl IntoElement {
    let theme = cx.theme();
    let mut row = v_flex().w_full().gap_0p5().child(
        h_flex()
            .w_full()
            .gap_3()
            .items_center()
            .child(
                div()
                    .w(px(120.))
                    .flex_none()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(label),
            )
            .child(
                div()
                    .id(SharedString::from(probe_name.to_string()))
                    .probe(probe_name.to_string(), gpui::Role::TextInput, label)
                    .flex_1()
                    .min_w_0()
                    .child(Input::new(state)),
            ),
    );
    if !hint.is_empty() {
        row = row.child(
            div()
                .pl(px(132.))
                .text_xs()
                .text_color(theme.muted_foreground.opacity(0.8))
                .child(hint),
        );
    }
    row
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
