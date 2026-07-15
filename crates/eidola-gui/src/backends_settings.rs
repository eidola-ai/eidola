//! Backends settings pane — *where an ask can be routed*.
//!
//! An internal tab strip (Eidola · Local · External) splits the registry
//! into the three mental buckets; the selected tab renders in the settings
//! voice (hairline rows, no cards):
//!
//! - **Eidola** — the confidential service's connection + trust surface.
//!   Enable/disable the singleton; when enabled, the **base-URL override**
//!   editor (edit/save/cancel/change/revert-to-pin — honest about pin vs
//!   override), the **trusted-measurements** override state (+ revert), and
//!   the **hardware CA** override state (two quiet lines; editing stays
//!   CLI-only). When disabled, a short "no account, on-device only"
//!   explanation. (Disabling `eidola` *is* the on-device-only configuration.)
//!   The account surface is a top-level Settings pane (`AccountView`), shown
//!   only while this backend is enabled.
//! - **Local** — the managed local store: enable/disable, the **engine**
//!   line (resolved bundled `llama-server`, or an honest dev-build hint), the
//!   installed models (status + per-state verbs: load / unload / cancel /
//!   delete, with download progress and honest inline errors), the curated
//!   **Gemma 4 catalog**, and a paste-a-URL row.
//! - **External** — everything the user owns. An `openai` server shows its
//!   URL / key-state / pinned models; a `llamacpp` backend shows its
//!   user-owned directory, its resolved engine line, an **auto-start**
//!   toggle (whether a request may spawn an engine), and the scanned models
//!   with load/unload verbs (never download/delete — the files are the
//!   user's). Plus the **add a backend** inline form (OpenAI-compatible
//!   server, or System llama.cpp with an optional engine path + auto-start).
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
    ActiveTheme, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputEvent, InputState},
    label::Label,
    v_flex,
};

use crate::probe::Probe as _;
use crate::stores::{BackendsStore, ConfigStore, LocalModelsStore, Stores};

/// Which internal tab of the Backends pane is showing. View-local state — the
/// selection isn't persisted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendsTab {
    Eidola,
    Local,
    External,
}

impl BackendsTab {
    fn label(self) -> &'static str {
        match self {
            BackendsTab::Eidola => "Eidola",
            BackendsTab::Local => "Local",
            BackendsTab::External => "External",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            BackendsTab::Eidola => "eidola",
            BackendsTab::Local => "local",
            BackendsTab::External => "external",
        }
    }
}

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
    /// Explicit `llama-server` path (llamacpp, optional) — unused for openai.
    engine_state: Entity<InputState>,
    /// Whether a request may auto-start an engine (llamacpp). Default `true`.
    auto_start: bool,
}

pub struct BackendsSettingsView {
    backends: Entity<BackendsStore>,
    local_models: Entity<LocalModelsStore>,
    /// The config store — the Eidola tab's base-URL + trust rows read the
    /// `EidolaTrust` snapshot from here (it lives on the eidola backend row;
    /// the store keeps a cached copy refreshed on `Change::Backends`) and
    /// write via its async setters.
    config: Entity<ConfigStore>,
    /// Which internal tab is showing (view-local; not persisted).
    tab: BackendsTab,
    /// The paste-a-URL input (the local section's download row).
    url_state: Entity<InputState>,
    /// The Eidola tab's base-URL editor state.
    base_url_state: Entity<InputState>,
    /// Whether the base-URL row is in its edit state (input + save/cancel).
    editing_base_url: bool,
    /// The inline add-backend form, while one is open.
    add_form: Option<AddForm>,
    _subscriptions: Vec<Subscription>,
}

impl BackendsSettingsView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let backends = stores.backends.clone();
        let local_models = stores.local_models.clone();
        let config = stores.config.clone();
        let url_state = cx.new(|cx| {
            InputState::new(window, cx).placeholder("https://huggingface.co/…/model.gguf")
        });
        let base_url_state = cx.new(|cx| InputState::new(window, cx).placeholder("https://…"));

        let _subscriptions = vec![
            cx.observe(&backends, |_, _, cx| cx.notify()),
            cx.observe(&local_models, |_, _, cx| cx.notify()),
            // The base-URL + trust rows read the config store's cached
            // `EidolaTrust`; re-render when it refreshes (bus `Change::Backends`).
            cx.observe(&config, |_, _, cx| cx.notify()),
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
            config,
            tab: BackendsTab::Eidola,
            url_state,
            base_url_state,
            editing_base_url: false,
            add_form: None,
            _subscriptions,
        }
    }

    /// Whether the Eidola tab's base-URL row is in its edit state (test
    /// accessor).
    pub fn editing_base_url(&self) -> bool {
        self.editing_base_url
    }

    /// Enter the base-URL edit state, seeding the input with the resolved
    /// value.
    pub fn begin_edit_base_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let current = self
            .config
            .read(cx)
            .eidola_trust()
            .map(|t| t.base_url.clone())
            .unwrap_or_default();
        self.base_url_state.update(cx, |s, cx| {
            s.set_value(&current, window, cx);
        });
        self.editing_base_url = true;
        cx.notify();
    }

    pub fn cancel_edit_base_url(&mut self, cx: &mut Context<Self>) {
        self.editing_base_url = false;
        cx.notify();
    }

    /// Save the edited value as an override. Saving the pin itself is
    /// treated as a revert — the row stays honest about its source.
    pub fn save_base_url(&mut self, cx: &mut Context<Self>) {
        let value = self.base_url_state.read(cx).value().trim().to_string();
        if value.is_empty() {
            return;
        }
        let pin = self
            .config
            .read(cx)
            .eidola_trust()
            .map(|t| t.base_url_pin.clone());
        self.config.update(cx, |c, cx| {
            if pin.as_deref() == Some(value.as_str()) {
                c.clear_base_url_override(cx);
            } else {
                c.set_base_url(value, cx);
            }
        });
        self.editing_base_url = false;
        cx.notify();
    }

    /// One-click revert from a base-URL override back to the built-in pin.
    pub fn revert_base_url(&mut self, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.clear_base_url_override(cx));
        self.editing_base_url = false;
        cx.notify();
    }

    /// One-click revert of a trusted-measurements override back to the
    /// built-in pin.
    pub fn revert_measurements(&mut self, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.revert_trusted_measurements(cx));
        cx.notify();
    }

    /// The selected internal tab (test accessor).
    pub fn tab(&self) -> BackendsTab {
        self.tab
    }

    /// Switch internal tabs. Public so the tab strip and behavior tests share
    /// one path.
    pub fn select_tab(&mut self, tab: BackendsTab, cx: &mut Context<Self>) {
        if self.tab != tab {
            self.tab = tab;
            cx.notify();
        }
    }

    /// One tab in the internal strip (quiet text, active = underline).
    fn tab_item(&self, tab: BackendsTab, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let active = self.tab == tab;
        let mut item = div()
            .id(tab.slug())
            .probe(
                format!("settings/backends/tab/{}", tab.slug()),
                gpui::Role::Tab,
                tab.label(),
            )
            .aria_selected(active)
            .cursor_pointer()
            .pb_1()
            .border_b_2()
            .on_click(cx.listener(move |this, _, _, cx| this.select_tab(tab, cx)))
            .child(tab.label());
        if active {
            item = item
                .border_color(theme.primary)
                .text_color(theme.foreground);
        } else {
            item = item
                .border_color(gpui::transparent_black())
                .text_color(theme.muted_foreground)
                .hover(|s| s.text_color(theme.foreground));
        }
        item
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

    /// Flip a `llamacpp` backend's request-triggered auto-start.
    pub fn set_auto_start(&mut self, id: String, auto_start: bool, cx: &mut Context<Self>) {
        self.backends
            .update(cx, |s, cx| s.set_auto_start(id, auto_start, cx));
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
        let engine_state = cx.new(|cx| {
            InputState::new(window, cx).placeholder("optional — discovered if left blank")
        });
        self.add_form = Some(AddForm {
            kind,
            id_state,
            url_state,
            key_state,
            dir_state,
            engine_state,
            auto_start: true,
        });
        cx.notify();
    }

    /// Close the add form without adding.
    pub fn cancel_add(&mut self, cx: &mut Context<Self>) {
        self.add_form = None;
        cx.notify();
    }

    /// Flip the pending llama.cpp add-form's auto-start checkbox.
    pub fn toggle_add_auto_start(&mut self, cx: &mut Context<Self>) {
        if let Some(form) = self.add_form.as_mut() {
            form.auto_start = !form.auto_start;
            cx.notify();
        }
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
                    engine_path: None,
                    auto_start: true,
                }
            }
            AddKind::LlamaCpp => {
                let dir = form.dir_state.read(cx).value().trim().to_string();
                let engine = form.engine_state.read(cx).value().trim().to_string();
                NewBackend {
                    id,
                    kind: BackendKind::LlamaCpp,
                    display_name: String::new(),
                    base_url: None,
                    api_key: None,
                    models_dir: Some(dir),
                    model_overrides: None,
                    engine_path: (!engine.is_empty()).then_some(engine),
                    auto_start: form.auto_start,
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
                            "No bundled engine in this build — loading models on this device \
                             is unavailable. (Dev builds: run `just engine`, or point the \
                             `llama_server_path` config at a llama-server.) Downloads work \
                             either way.",
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
                // The resolved engine for this backend (its pinned
                // `engine_path`, else discovery), stated plainly.
                let resolved_engine = self
                    .local_models
                    .read(cx)
                    .external()
                    .iter()
                    .find(|b| b.backend_id == backend.id)
                    .and_then(|b| b.engine_path.clone())
                    .or_else(|| backend.engine_path.clone());
                out.push(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(match &resolved_engine {
                            Some(p) => format!("Engine: llama-server at {p}"),
                            None => "Engine: llama-server not found — set a path or install it \
                                     on this machine."
                                .to_string(),
                        }))
                        .into_any_element(),
                );

                // Auto-start toggle: whether a request may spawn an engine.
                out.push(
                    self.auto_start_toggle(&backend.id, backend.auto_start, cx)
                        .into_any_element(),
                );

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

    /// A llamacpp backend's auto-start toggle: a checkbox-style row wired to
    /// `update_backend`. Auto-start gates *request-triggered* engine loads;
    /// an explicit Load ignores it.
    fn auto_start_toggle(
        &self,
        backend_id: &str,
        auto_start: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let id = backend_id.to_string();
        let id_click = id.clone();
        h_flex()
            .id(SharedString::from(format!("autostart-{id}")))
            .probe(
                format!("settings/backends/{id}/autostart"),
                gpui::Role::CheckBox,
                "Start an engine automatically on request",
            )
            .aria_selected(auto_start)
            .cursor_pointer()
            .gap_2()
            .items_center()
            .on_click(cx.listener(move |this, _, _, cx| {
                this.set_auto_start(id_click.clone(), !auto_start, cx);
            }))
            .child(
                div()
                    .size(px(14.))
                    .flex_none()
                    .rounded(px(3.))
                    .border_1()
                    .border_color(theme.border)
                    .when(auto_start, |d| {
                        d.bg(theme.primary).border_color(theme.primary)
                    }),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("Start an engine automatically when a request needs one"),
            )
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
                    quiet_verb("add-llamacpp", "System llama.cpp…", cx)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.begin_add(AddKind::LlamaCpp, window, cx)
                        }))
                        .probe(
                            "settings/backends/add/llamacpp",
                            gpui::Role::Button,
                            "Add a system llama.cpp install",
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
                col = col
                    .child(labeled_input(
                        "Models directory",
                        "settings/backends/add/dir",
                        "a folder of .gguf files you manage",
                        &form.dir_state,
                        cx,
                    ))
                    .child(labeled_input(
                        "Engine path",
                        "settings/backends/add/engine-path",
                        "optional path to llama-server; discovered on PATH if left blank",
                        &form.engine_state,
                        cx,
                    ))
                    .child(
                        h_flex().pl(px(132.)).child(
                            h_flex()
                                .id("add-autostart")
                                .probe(
                                    "settings/backends/add/autostart",
                                    gpui::Role::CheckBox,
                                    "Start an engine automatically on request",
                                )
                                .aria_selected(form.auto_start)
                                .cursor_pointer()
                                .gap_2()
                                .items_center()
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.toggle_add_auto_start(cx)),
                                )
                                .child(
                                    div()
                                        .size(px(14.))
                                        .flex_none()
                                        .rounded(px(3.))
                                        .border_1()
                                        .border_color(theme.border)
                                        .when(form.auto_start, |d| {
                                            d.bg(theme.primary).border_color(theme.primary)
                                        }),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(theme.muted_foreground)
                                        .child("Start an engine automatically on request"),
                                ),
                        ),
                    );
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
        // Copy the colors we need up front — `tab_item` takes `&mut Context`
        // (for its click listener), so we can't hold a `cx.theme()` borrow
        // across the tab-strip build.
        let border = cx.theme().border;
        let muted = cx.theme().muted_foreground;
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

        // The internal tab strip: Eidola · Local · External.
        col = col.child(
            h_flex()
                .w_full()
                .gap_5()
                .pb_2()
                .border_b_1()
                .border_color(border)
                .child(self.tab_item(BackendsTab::Eidola, cx))
                .child(self.tab_item(BackendsTab::Local, cx))
                .child(self.tab_item(BackendsTab::External, cx)),
        );

        if backends.is_empty() {
            // The registry hasn't loaded yet (or a fixture-less stub scene):
            // say so rather than rendering a blank pane.
            col = col.child(div().text_sm().text_color(muted).child("Loading backends…"));
            return col;
        }

        // Per-tab content. Each tab pulls the backend(s) it owns out of the
        // registry snapshot.
        let children: Vec<gpui::AnyElement> = match self.tab {
            BackendsTab::Eidola => self.eidola_tab(&backends, cx),
            BackendsTab::Local => backends
                .iter()
                .find(|b| b.kind == BackendKind::Local)
                .map(|b| self.local_section(b, cx))
                .unwrap_or_default(),
            BackendsTab::External => self.external_tab(&backends, cx),
        };
        let mut section = v_flex().w_full().gap_3();
        for child in children {
            section = section.child(child);
        }
        col = col.child(section);

        col
    }
}

impl BackendsSettingsView {
    /// The Eidola tab: the singleton's enable/disable header, and — when
    /// enabled — the connection + trust surface (base-URL override editor,
    /// trusted-measurements override state, hardware CA state). The account
    /// itself is a top-level Settings pane, shown only while this backend is
    /// enabled.
    fn eidola_tab(&self, backends: &[BackendInfo], cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let theme = cx.theme();
        let mut out: Vec<gpui::AnyElement> = Vec::new();
        let Some(backend) = backends.iter().find(|b| b.kind == BackendKind::Eidola) else {
            return out;
        };
        out.push(
            self.backend_header(
                backend,
                "Confidential inference via the Eidola service — attested hardware, \
                 anonymous credits. Disable to run with no account, on-device only."
                    .into(),
                false,
                cx,
            )
            .into_any_element(),
        );
        if backend.enabled {
            out.extend(self.eidola_trust_surface(cx));
        } else {
            out.push(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(
                        "No account, on-device only. Enable Eidola to create or link an \
                         account and buy credits for confidential inference.",
                    )
                    .into_any_element(),
            );
        }
        out
    }

    /// The connection + trust rows: base-URL override editor, then the
    /// measurement + hardware-CA override state. Reads the config store's
    /// cached `EidolaTrust` (the eidola backend row's bundle, NULL = the
    /// embedded pin) and writes via the async setters. Full measurement /
    /// CA editing stays CLI-only (`eidola configure`); this surface shows the
    /// security state and offers revert-to-pin where an override exists.
    fn eidola_trust_surface(&self, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let theme = cx.theme();
        let mut out: Vec<gpui::AnyElement> = Vec::new();
        let trust = self.config.read(cx).eidola_trust().cloned();
        let Some(trust) = trust else {
            out.push(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("Loading connection settings…")
                    .into_any_element(),
            );
            return out;
        };

        // --- Base URL: honest about override vs pin --------------------
        let mut base_value = v_flex().flex_1().gap_1();
        if self.editing_base_url {
            base_value = base_value
                .child(
                    // Probed wrapper for the a11y role/label — probe the
                    // wrapping div, not the gpui-component Input.
                    div()
                        .id("eidola-base-url-input-wrap")
                        .probe(
                            "settings/backends/eidola/url/base-url",
                            gpui::Role::TextInput,
                            "Base URL",
                        )
                        .w_full()
                        .flex()
                        .child(Input::new(&self.base_url_state).flex_1()),
                )
                .child(
                    h_flex()
                        .gap_2()
                        .pt_1()
                        .child(
                            div()
                                .id("eidola-save-base-url-wrap")
                                .probe(
                                    "settings/backends/eidola/url/save",
                                    gpui::Role::Button,
                                    "Save",
                                )
                                .child(
                                    Button::new("eidola-save-base-url")
                                        .primary()
                                        .small()
                                        .label("Save")
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.save_base_url(cx)),
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .id("eidola-cancel-base-url-wrap")
                                .probe(
                                    "settings/backends/eidola/url/cancel",
                                    gpui::Role::Button,
                                    "Cancel",
                                )
                                .child(
                                    Button::new("eidola-cancel-base-url")
                                        .ghost()
                                        .small()
                                        .label("Cancel")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.cancel_edit_base_url(cx)
                                        })),
                                ),
                        ),
                );
        } else {
            base_value = base_value.child(
                div()
                    .text_sm()
                    .font_family("Menlo")
                    .child(SharedString::from(trust.base_url.clone())),
            );
            let status: String = if trust.base_url_is_override {
                format!("Override — the built-in pin is {}.", trust.base_url_pin)
            } else {
                "Built-in pin — verified against this build's trust root.".into()
            };
            base_value = base_value.child(
                div()
                    .w_full()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(status)),
            );
            let mut links = h_flex().gap_3().text_xs();
            if trust.base_url_is_override {
                links = links.child(
                    quiet_verb("eidola-revert-base-url", "Revert to pin", cx)
                        .probe(
                            "settings/backends/eidola/url/revert-to-pin",
                            gpui::Role::Button,
                            "Revert to pin",
                        )
                        .on_click(cx.listener(|this, _, _, cx| this.revert_base_url(cx))),
                );
            }
            links = links.child(
                quiet_verb("eidola-edit-base-url", "Change…", cx)
                    .probe(
                        "settings/backends/eidola/url/change",
                        gpui::Role::Button,
                        "Change base URL",
                    )
                    .on_click(
                        cx.listener(|this, _, window, cx| this.begin_edit_base_url(window, cx)),
                    ),
            );
            base_value = base_value.child(links);
        }
        out.push(trust_row("Base URL", cx, base_value).into_any_element());

        // --- Trusted measurements: pin vs override + revert ------------
        let measurements_len = trust.trusted_measurements.len();
        let summary = if trust.trusted_measurements_are_override {
            format!(
                "{} user-trusted measurement{} — overriding the pin.",
                measurements_len,
                if measurements_len == 1 { "" } else { "s" }
            )
        } else {
            "1 measurement — pinned at build.".to_string()
        };
        let mut measurements_value = v_flex()
            .flex_1()
            .gap_1()
            .child(muted_text(summary, cx))
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child("Editing the trusted set is a CLI operation (`eidola configure`)."),
            );
        if trust.trusted_measurements_are_override {
            measurements_value = measurements_value.child(
                h_flex().text_xs().child(
                    quiet_verb("eidola-revert-measurements", "Revert to pin", cx)
                        .probe(
                            "settings/backends/eidola/measurements/revert",
                            gpui::Role::Button,
                            "Revert trusted measurements to pin",
                        )
                        .on_click(cx.listener(|this, _, _, cx| this.revert_measurements(cx))),
                ),
            );
        }
        out.push(trust_row("Trusted measurements", cx, measurements_value).into_any_element());

        // --- Hardware CAs: quiet override-state lines (CLI-only) -------
        out.push(
            trust_row(
                "Hardware root CA",
                cx,
                muted_text(
                    if trust.has_hardware_root_ca {
                        "Custom certificate set (override)."
                    } else {
                        "Built-in AMD/Intel vendor chain (pin)."
                    },
                    cx,
                ),
            )
            .into_any_element(),
        );
        out.push(
            trust_row(
                "Intermediate CA",
                cx,
                muted_text(
                    if trust.has_hardware_intermediate_ca {
                        "Custom certificate set (override)."
                    } else {
                        "Built-in AMD/Intel vendor chain (pin)."
                    },
                    cx,
                ),
            )
            .into_any_element(),
        );

        out
    }

    /// The External tab: the openai/llamacpp backends, plus the add-a-backend
    /// form.
    fn external_tab(&self, backends: &[BackendInfo], cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        let theme = cx.theme();
        let mut out: Vec<gpui::AnyElement> = Vec::new();
        let mut first = true;
        for backend in backends
            .iter()
            .filter(|b| matches!(b.kind, BackendKind::OpenAi | BackendKind::LlamaCpp))
        {
            if !first {
                out.push(
                    div()
                        .w_full()
                        .border_t_1()
                        .border_color(theme.border)
                        .into_any_element(),
                );
            }
            first = false;
            let mut section = v_flex().w_full().gap_3();
            for child in self.external_section(backend, cx) {
                section = section.child(child);
            }
            out.push(section.into_any_element());
        }

        if !first {
            out.push(
                div()
                    .w_full()
                    .border_t_1()
                    .border_color(theme.border)
                    .into_any_element(),
            );
        }
        let mut add = v_flex().w_full().gap_3();
        for child in self.add_section(cx) {
            add = add.child(child);
        }
        out.push(add.into_any_element());
        out
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

/// A two-column label/value field row (mirrors general.rs's `field_row`),
/// used by the Eidola tab's connection + trust surface.
fn trust_row<C: IntoElement>(label: &str, cx: &gpui::App, child: C) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .w_full()
        .gap_4()
        .py_1()
        .items_start()
        .child(
            div()
                .w(px(144.))
                .flex_none()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(label.to_string())),
        )
        .child(div().flex_1().min_w_0().child(child))
}

/// A muted value cell for the trust rows.
fn muted_text(text: impl Into<String>, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    let text = text.into();
    div()
        .text_color(theme.muted_foreground)
        .child(SharedString::from(text))
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
