//! Backends settings pane — *where an ask can be routed*.
//!
//! An internal tab strip (Eidola · Local · External) splits the registry
//! into the three mental buckets; the selected tab renders in the settings
//! voice (hairline rows, no cards):
//!
//! - **Eidola** — the confidential service's connection + trust surface.
//!   Enable/disable the singleton; when enabled, an always-visible danger
//!   warning band whenever any override is active, then **one row per trust
//!   setting, each row both display and editor** (the macOS settings idiom:
//!   the current value is always visible, and a quiet verb swaps in the
//!   input on demand — the base-URL row's edit-in-place pattern,
//!   generalized). Base URL (change/save/cancel/revert-to-pin), trusted
//!   measurements (compact per-measurement lines with Copy — full triple —
//!   and Untrust/Trust verbs, a reveal-on-demand add input, revert, and a
//!   link to the Record for the attestation evidence), and the two hardware
//!   CAs (status + Copy/Replace…/Clear when overridden, "Set custom
//!   certificate…" revealing a paste-PEM textarea otherwise). Nothing is
//!   duplicated and nothing hides behind a disclosure — sovereignty means
//!   the trust surface is always visible; the danger states carry the
//!   warning band and one-click reverts. Read-only connection details
//!   (attestation URL, domain separator) close the tab. When disabled, a
//!   short "no account, on-device only" explanation. (Disabling `eidola`
//!   *is* the on-device-only configuration.) The account surface is a
//!   top-level Settings pane (`AccountView`), shown only while this backend
//!   is enabled.
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
    BackendInfo, BackendKind, EidolaTrust, LOCAL_MODEL_CATALOG, LocalModelInfo, LocalModelStatus,
    MeasurementInfo, NewBackend,
};
use gpui::{
    App, AppContext, ClipboardItem, Context, Entity, Focusable as _, InteractiveElement,
    IntoElement, ParentElement, Render, SharedString, StatefulInteractiveElement, Styled,
    Subscription, Window, div, prelude::FluentBuilder, px,
};
use gpui_component::{
    ActiveTheme, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputEvent, InputState},
    label::Label,
    switch::Switch,
    tab::{Tab, TabBar},
    v_flex,
};

use crate::actions::OpenRecord;
use crate::focus::TabRegion as _;
use crate::probe::Probe as _;
use crate::stores::{BackendsStore, ConfigStore, LocalModelsStore, Stores};

/// Which hardware CA a trust-editor row targets. The two are the same
/// component parameterized; the core has a separate setter/clearer per kind.
/// Public so behavior tests can drive `submit_ca` / `clear_ca`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CaKind {
    Root,
    Intermediate,
}

/// How a measurement line in the trusted-measurements row should present —
/// which verbs it carries and how it's tagged. See `measurement_line`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MeasurementRole {
    /// The build pin, and it is the trusted set (no override). Read-only.
    Pinned,
    /// A user override entry at this index in `trusted_measurements`.
    Override(usize),
    /// The build pin while an override set has replaced it — auditable, and
    /// re-addable via a Trust verb.
    PinNotTrusted,
}

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
    /// Whether the trusted-measurements row is showing its add input
    /// ("Trust a measurement…" revealed it). The same edit-in-place shape as
    /// the base-URL row: the input appears on demand, the row's resting state
    /// stays a value display.
    adding_measurement: bool,
    /// Which hardware-CA row is showing its paste-PEM textarea, if any
    /// ("Set custom certificate…" / "Replace…" revealed it). One at a time —
    /// the two rows share the edit-in-place shape with the base-URL row.
    editing_ca: Option<CaKind>,
    /// The Eidola tab's add-a-measurement input (`<snp>:<rtmr1>:<rtmr2>`).
    add_measurement_state: Entity<InputState>,
    /// The Eidola tab's hardware root-CA paste-PEM input.
    root_ca_state: Entity<InputState>,
    /// The Eidola tab's hardware intermediate-CA paste-PEM input.
    intermediate_ca_state: Entity<InputState>,
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
        let add_measurement_state =
            cx.new(|cx| InputState::new(window, cx).placeholder("<snp>:<rtmr1>:<rtmr2>"));
        // Hardware-CA overrides are full PEM blocks — multi-line textareas
        // that grow with the pasted certificate rather than a single-line
        // field that hides all but a sliver of it.
        let root_ca_state = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(4, 16)
                .placeholder("-----BEGIN CERTIFICATE-----\n…\n-----END CERTIFICATE-----")
        });
        let intermediate_ca_state = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(4, 16)
                .placeholder("-----BEGIN CERTIFICATE-----\n…\n-----END CERTIFICATE-----")
        });

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
            // Enter in the add-measurement field submits, mirroring the verb.
            cx.subscribe_in(
                &add_measurement_state,
                window,
                |this, _, event: &InputEvent, window, cx| {
                    if matches!(event, InputEvent::PressEnter { .. }) {
                        this.submit_add_measurement(window, cx);
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
            adding_measurement: false,
            editing_ca: None,
            add_measurement_state,
            root_ca_state,
            intermediate_ca_state,
            add_form: None,
            _subscriptions,
        }
    }

    /// Whether the trusted-measurements row is showing its add input (test
    /// accessor).
    pub fn adding_measurement(&self) -> bool {
        self.adding_measurement
    }

    /// Reveal the add-a-measurement input ("Trust a measurement…") and focus
    /// it.
    pub fn begin_add_measurement(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.adding_measurement = true;
        self.add_measurement_state
            .update(cx, |s, cx| s.focus(window, cx));
        cx.notify();
    }

    /// Hide the add-a-measurement input without adding.
    pub fn cancel_add_measurement(&mut self, cx: &mut Context<Self>) {
        self.adding_measurement = false;
        cx.notify();
    }

    /// Which hardware-CA row is in its edit state, if any (test accessor).
    pub fn editing_ca(&self) -> Option<CaKind> {
        self.editing_ca
    }

    /// Reveal a hardware-CA row's paste-PEM textarea ("Set custom
    /// certificate…" / "Replace…") and focus it.
    pub fn begin_edit_ca(&mut self, kind: CaKind, window: &mut Window, cx: &mut Context<Self>) {
        self.editing_ca = Some(kind);
        let state = match kind {
            CaKind::Root => &self.root_ca_state,
            CaKind::Intermediate => &self.intermediate_ca_state,
        };
        state.update(cx, |s, cx| s.focus(window, cx));
        cx.notify();
    }

    /// Hide the CA textarea without setting anything.
    pub fn cancel_edit_ca(&mut self, cx: &mut Context<Self>) {
        self.editing_ca = None;
        cx.notify();
    }

    /// The add-a-measurement input entity (test seam — behavior tests seed a
    /// triple before calling `submit_add_measurement`).
    #[doc(hidden)]
    pub fn add_measurement_input(&self) -> Entity<InputState> {
        self.add_measurement_state.clone()
    }

    /// A hardware-CA paste-PEM input entity (test seam — behavior tests seed a
    /// PEM before calling `submit_ca`).
    #[doc(hidden)]
    pub fn ca_input(&self, kind: CaKind) -> Entity<InputState> {
        match kind {
            CaKind::Root => self.root_ca_state.clone(),
            CaKind::Intermediate => self.intermediate_ca_state.clone(),
        }
    }

    /// Whether the Eidola tab's base-URL row is in its edit state (test
    /// accessor).
    /// Test seam: whether the base-URL editor's input holds the window's
    /// focus — what a keyboard-activated reveal must leave behind.
    #[doc(hidden)]
    pub fn base_url_input_is_focused(&self, window: &Window, cx: &App) -> bool {
        self.base_url_state
            .read(cx)
            .focus_handle(cx)
            .is_focused(window)
    }

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
            // **A reveal focuses what it revealed.** The affordance that opened
            // this row unmounts as the row becomes the form, so a keyboard
            // reader who pressed Enter on it is left with focus on nothing at
            // all — the window keeps a handle whose element is gone, the
            // dispatch tree has no node for it, and Tab restarts from the top
            // of the window. Its siblings here (`begin_add_measurement`,
            // `begin_edit_ca`) already did this; this row did not.
            s.focus(window, cx);
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

    /// Add the trusted measurement in the add field (the CLI's
    /// `<snp>:<rtmr1>:<rtmr2>` triple). A parse/validation failure surfaces in
    /// the config store's op-error banner and the input is kept so the user
    /// can fix it; a successful add clears the field.
    pub fn submit_add_measurement(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let spec = self
            .add_measurement_state
            .read(cx)
            .value()
            .trim()
            .to_string();
        if spec.is_empty() {
            return;
        }
        self.config
            .update(cx, |c, cx| c.trust_measurement(spec, cx));
        if self.config.read(cx).error().is_none() {
            self.add_measurement_state
                .update(cx, |s, cx| s.set_value("", window, cx));
            self.adding_measurement = false;
        }
        cx.notify();
    }

    /// Untrust a single measurement (by its SNP key). Removing the last one
    /// reverts to the pin.
    pub fn untrust_measurement(&mut self, snp: String, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.untrust_measurement(snp, cx));
        cx.notify();
    }

    /// Trust a measurement from an explicit `<snp>:<rtmr1>:<rtmr2>` triple
    /// (not the add-field). Backs the pin card's "Trust" verb — re-adding the
    /// build pin to a trusted set that overrode it. Failures land in the
    /// config store's op-error banner.
    pub fn trust_measurement_spec(&mut self, spec: String, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.trust_measurement(spec, cx));
        cx.notify();
    }

    /// Set a hardware CA override from the pasted PEM. Validation lives in
    /// app-core; a failure lands in the op-error banner and the input is kept.
    pub fn submit_ca(&mut self, kind: CaKind, window: &mut Window, cx: &mut Context<Self>) {
        let state = match kind {
            CaKind::Root => &self.root_ca_state,
            CaKind::Intermediate => &self.intermediate_ca_state,
        }
        .clone();
        let pem = state.read(cx).value().trim().to_string();
        if pem.is_empty() {
            return;
        }
        self.config.update(cx, |c, cx| match kind {
            CaKind::Root => c.set_hardware_root_ca(pem, cx),
            CaKind::Intermediate => c.set_hardware_intermediate_ca(pem, cx),
        });
        if self.config.read(cx).error().is_none() {
            state.update(cx, |s, cx| s.set_value("", window, cx));
            self.editing_ca = None;
        }
        cx.notify();
    }

    /// Clear a hardware CA override (row column back to NULL — reverts toward
    /// the built-in AMD/Intel vendor chain).
    pub fn clear_ca(&mut self, kind: CaKind, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| match kind {
            CaKind::Root => c.clear_hardware_root_ca(cx),
            CaKind::Intermediate => c.clear_hardware_intermediate_ca(cx),
        });
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

    /// One tab in the internal strip.
    ///
    /// The strip is gpui-component's `TabBar` in its default styling rather
    /// than a hand-rolled row — one fewer bespoke control to keep in step with
    /// the theme. `Tab` implements `StatefulInteractiveElement + ParentElement`,
    /// so it carries our `.probe(..)` annotation directly and the probe names
    /// (`settings/backends/tab/{slug}`) are unchanged; the element id has to be
    /// written through `Interactivity` because `InteractiveElement::id` returns
    /// a `Stateful<Tab>`, which `TabBar::child` (an `Into<Tab>`) cannot take —
    /// and without an id gpui builds no AccessKit node for the role we set.
    fn tab_item(&self, tab: BackendsTab, cx: &mut Context<Self>) -> Tab {
        let active = self.tab == tab;
        let mut item = Tab::new().label(tab.label());
        item.interactivity().element_id = Some(gpui::ElementId::from(tab.slug()));
        item.probe(
            format!("settings/backends/tab/{}", tab.slug()),
            gpui::Role::Tab,
            tab.label(),
        )
        .aria_selected(active)
        .on_click(cx.listener(move |this, _, _, cx| this.select_tab(tab, cx)))
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
        // The reveal focuses its first field — see `begin_edit_base_url`.
        id_state.update(cx, |s, cx| s.focus(window, cx));
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
                        .child(Input::new(&self.url_state).a11y_labelled_by_ancestor()),
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

    /// A llamacpp backend's auto-start toggle: a gpui-component `Switch` wired
    /// to `update_backend`. Auto-start gates *request-triggered* engine loads;
    /// an explicit Load ignores it. The probed wrapper carries the a11y
    /// role/label + selected state for the QA driver; the `Switch` owns the
    /// interaction and its own focus/keyboard handling.
    fn auto_start_toggle(
        &self,
        backend_id: &str,
        auto_start: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let id = backend_id.to_string();
        let id_click = id.clone();
        let id_key = id.clone();
        div()
            .id(SharedString::from(format!("autostart-{id}")))
            .probe(
                format!("settings/backends/{id}/autostart"),
                gpui::Role::CheckBox,
                "Start an engine automatically on request",
            )
            // A checkbox's state is `toggled`, not `selected`:
            // `accesskit_macos` reads `accessibilityValue` from `toggled()`
            // first and only falls through to `is_selected()` for `Role::Tab`,
            // so a `CheckBox` carrying only `aria_selected` reports **no
            // value at all** — VoiceOver announces the control and not whether
            // it is on.
            .aria_toggled(auto_start.into())
            // The wrapper owns the **keyboard** activation. Unlike `Button` /
            // `Checkbox`, `gpui_component::Switch` tracks no focus handle at
            // our pin, so there is nothing inside for Tab to reach — see
            // `probe_delegating`'s doc for the other half of the rule. This does
            // not double-fire on a pointer click: `Switch` handles the press
            // in `on_mouse_down` and calls `stop_propagation`, so the wrapper
            // never arms a click of its own (gpui bubbles mouse listeners
            // innermost-first).
            .on_click(cx.listener(move |this, _, _, cx| {
                this.set_auto_start(id_key.clone(), !auto_start, cx);
            }))
            .child(
                // Switch sets no AccessKit role/label at our gpui-component
                // rev, so the probed wrapper is the only node; if Switch gains
                // self-annotation upstream, this site must join the
                // `.a11y_labelled_by_ancestor()` opt-out.
                Switch::new(SharedString::from(format!("autostart-switch-{id}")))
                    .small()
                    .checked(auto_start)
                    .label("Start an engine automatically when a request needs one")
                    .on_click(cx.listener(move |this, checked: &bool, _, cx| {
                        this.set_auto_start(id_click.clone(), *checked, cx);
                    })),
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
                            div()
                                .id("add-autostart")
                                .probe(
                                    "settings/backends/add/autostart",
                                    gpui::Role::CheckBox,
                                    "Start an engine automatically on request",
                                )
                                // See `autostart_row`: a checkbox's state is
                                // `toggled`, which is what the adapter reads.
                                .aria_toggled(form.auto_start.into())
                                // The wrapper owns the keyboard activation —
                                // see `autostart_row` for why.
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.toggle_add_auto_start(cx)),
                                )
                                .child(
                                    // Switch sets no AccessKit role/label at
                                    // our gpui-component rev (wrapper is the
                                    // only node); if Switch gains
                                    // self-annotation upstream, join the
                                    // `.a11y_labelled_by_ancestor()` opt-out.
                                    Switch::new("add-autostart-switch")
                                        .small()
                                        .checked(form.auto_start)
                                        .label("Start an engine automatically on request")
                                        .on_click(cx.listener(|this, _checked: &bool, _, cx| {
                                            this.toggle_add_auto_start(cx)
                                        })),
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
        let muted = cx.theme().muted_foreground;
        let backends: Vec<BackendInfo> = self.backends.read(cx).list().to_vec();
        let backends_op_error = self.backends.read(cx).op_error().map(|s| s.to_string());
        let local_op_error = self.local_models.read(cx).op_error().map(|s| s.to_string());

        let mut col = v_flex()
            .id("backends-pane")
            .px_6()
            .py_5()
            .gap_4()
            .w_full()
            // The pane's own title, in the voice Wallet ("Credentials") and
            // Templates ("Space Templates") already use. It is also what keeps
            // the tab strip clear of the window's drag band: the band used to
            // be 44px here purely to fake this padding, and once it started
            // blocking the mouse that made the tabs unclickable.
            .child(
                div()
                    .text_color(muted)
                    .text_sm()
                    .font_medium()
                    .child("Backends"),
            );

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

        // The internal tab strip: Eidola · Local · External — gpui-component's
        // `TabBar` in its **segmented** variant. `selected_index` drives the
        // visual selection (and the segmented indicator that slides to it);
        // each `Tab` keeps its own click handler (TabBar only overrides those
        // when the bar itself carries an `on_click`).
        let selected = [
            BackendsTab::Eidola,
            BackendsTab::Local,
            BackendsTab::External,
        ]
        .iter()
        .position(|t| *t == self.tab)
        .unwrap_or(0);
        col = col.child(
            div()
                .id("backends-tabs")
                .probe(
                    "settings/backends/tabs",
                    gpui::Role::TabList,
                    "Backend kinds",
                )
                .tab_region(crate::focus::region::NAV)
                .w_full()
                .child(
                    TabBar::new("backends-tab-bar")
                        .segmented()
                        .selected_index(selected)
                        .child(self.tab_item(BackendsTab::Eidola, cx))
                        .child(self.tab_item(BackendsTab::Local, cx))
                        .child(self.tab_item(BackendsTab::External, cx)),
                ),
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

    /// The connection + trust rows: an always-visible override warning band,
    /// then one row per setting, each both display and editor — base URL,
    /// trusted measurements, and the two hardware CAs — followed by the
    /// read-only connection details (attestation URL, domain separator).
    /// Reads the config store's cached `EidolaTrust` (the eidola backend
    /// row's bundle, NULL = the embedded pin) and writes via the async
    /// setters. Nothing hides behind a disclosure: the resting state of each
    /// row is its current value, and the inputs appear in place on demand.
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

        // --- Warning band: always visible when any override is active ---
        let mut overridden: Vec<&str> = Vec::new();
        if trust.base_url_is_override {
            overridden.push("base URL");
        }
        if trust.trusted_measurements_are_override {
            overridden.push("trusted measurements");
        }
        if trust.has_hardware_root_ca {
            overridden.push("hardware root CA");
        }
        if trust.has_hardware_intermediate_ca {
            overridden.push("hardware intermediate CA");
        }
        if !overridden.is_empty() {
            out.push(trust_warning_band(&overridden, cx).into_any_element());
        }

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
                        .child(
                            Input::new(&self.base_url_state)
                                .a11y_labelled_by_ancestor()
                                .flex_1(),
                        ),
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
                                .on_click(cx.listener(|this, _, _, cx| this.save_base_url(cx)))
                                .child(
                                    Button::new("eidola-save-base-url")
                                        .a11y_labelled_by_ancestor()
                                        .primary()
                                        .small()
                                        .label("Save")
                                        .tab_stop(false),
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
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.cancel_edit_base_url(cx)),
                                )
                                .child(
                                    Button::new("eidola-cancel-base-url")
                                        .a11y_labelled_by_ancestor()
                                        .ghost()
                                        .small()
                                        .label("Cancel")
                                        .tab_stop(false),
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
                    .text_color(if trust.base_url_is_override {
                        theme.danger
                    } else {
                        theme.muted_foreground
                    })
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

        // --- Trusted measurements: one row, display + editor ------------
        out.push(self.measurements_row(&trust, cx));

        // --- Hardware CAs: one row each, display + editor ---------------
        out.push(self.ca_row(
            CaKind::Root,
            "Hardware root CA",
            "root",
            trust.hardware_root_ca_pem.as_deref(),
            cx,
        ));
        out.push(self.ca_row(
            CaKind::Intermediate,
            "Intermediate CA",
            "intermediate",
            trust.hardware_intermediate_ca_pem.as_deref(),
            cx,
        ));

        // --- Connection details: read-only facts about the service ------
        // (Moved here from the General pane — everything about the Eidola
        // connection lives on the Eidola backend.)
        if let Some(state) = self.config.read(cx).state() {
            out.push(
                div()
                    .w_full()
                    .border_t_1()
                    .border_color(theme.border)
                    .into_any_element(),
            );
            out.push(
                trust_row(
                    "Attestation URL",
                    cx,
                    div()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(
                            state
                                .attestation_url
                                .clone()
                                .unwrap_or_else(|| "Default (Tinfoil ATC)".into()),
                        )),
                )
                .into_any_element(),
            );
            // The domain separator is one long unbreakable token, so it gets
            // a stacked row (value under label, full width) rather than the
            // two-column layout.
            out.push(
                v_flex()
                    .w_full()
                    .py_1()
                    .gap_1()
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .child("Domain separator"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .font_family("Menlo")
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(state.domain_separator.clone())),
                    )
                    .into_any_element(),
            );
        }

        out
    }

    /// The trusted-measurements row — display and editor in one place. The
    /// status line, one compact line per measurement (`measurement_line`),
    /// the verbs (revert toward the pin, "Trust a measurement…"), the
    /// reveal-on-demand add input, and a link to the Record — the raw
    /// attestation evidence's home. Full values never render inline; they
    /// travel via the Copy verbs and a11y labels.
    fn measurements_row(&self, trust: &EidolaTrust, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let is_override = trust.trusted_measurements_are_override;
        let count = trust.trusted_measurements.len();

        let mut value = v_flex().flex_1().gap_1();

        // Status line — danger when overriding the pin.
        let summary = if is_override {
            format!(
                "{count} user-trusted measurement{} — overriding the pin.",
                if count == 1 { "" } else { "s" }
            )
        } else {
            "1 measurement — pinned at build.".to_string()
        };
        value = value.child(
            div()
                .text_color(if is_override {
                    theme.danger
                } else {
                    theme.muted_foreground
                })
                .child(SharedString::from(summary)),
        );

        // The measurement lines.
        if is_override {
            for (idx, m) in trust.trusted_measurements.iter().enumerate() {
                value = value.child(self.measurement_line(m, MeasurementRole::Override(idx), cx));
            }
            // The build pin, if the override set dropped it (an override
            // *replaces* the pin): auditable + re-addable — the "trust
            // official + my canary" flow.
            let pin = &trust.pinned_measurement;
            let pin_trusted = trust
                .trusted_measurements
                .iter()
                .any(|m| m.snp.eq_ignore_ascii_case(&pin.snp));
            if !pin_trusted {
                value = value.child(self.measurement_line(pin, MeasurementRole::PinNotTrusted, cx));
            }
        } else {
            value = value.child(self.measurement_line(
                &trust.pinned_measurement,
                MeasurementRole::Pinned,
                cx,
            ));
        }

        // Verbs: revert toward the pin (the safe direction, only meaningful
        // on an override) and the add-input reveal.
        let mut verbs = h_flex().gap_3().pt_0p5().text_xs();
        if is_override {
            verbs = verbs.child(
                quiet_verb("eidola-revert-measurements", "Revert to pin", cx)
                    .text_xs()
                    .probe(
                        "settings/backends/eidola/measurements/revert",
                        gpui::Role::Button,
                        "Revert trusted measurements to pin",
                    )
                    .on_click(cx.listener(|this, _, _, cx| this.revert_measurements(cx))),
            );
        }
        if !self.adding_measurement {
            verbs = verbs.child(
                quiet_verb("eidola-begin-add-measurement", "Trust a measurement…", cx)
                    .text_xs()
                    .probe(
                        "settings/backends/eidola/measurements/trust-new",
                        gpui::Role::Button,
                        "Trust a new measurement",
                    )
                    .on_click(
                        cx.listener(|this, _, window, cx| this.begin_add_measurement(window, cx)),
                    ),
            );
        }
        value = value.child(verbs);

        // The add input, revealed on demand (edit-in-place, like the
        // base-URL row). Enter submits, mirroring the Add verb.
        if self.adding_measurement {
            value = value
                .child(
                    div()
                        .id("eidola-add-measurement-wrap")
                        .probe(
                            "settings/backends/eidola/measurements/add",
                            gpui::Role::TextInput,
                            "Add a trusted measurement",
                        )
                        .w_full()
                        .flex()
                        .child(
                            Input::new(&self.add_measurement_state)
                                .a11y_labelled_by_ancestor()
                                .flex_1(),
                        ),
                )
                .child(
                    h_flex()
                        .gap_3()
                        .text_xs()
                        .child(
                            quiet_verb("eidola-add-measurement", "Add", cx)
                                .text_xs()
                                .probe(
                                    "settings/backends/eidola/measurements/add/submit",
                                    gpui::Role::Button,
                                    "Add trusted measurement",
                                )
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.submit_add_measurement(window, cx)
                                })),
                        )
                        .child(
                            quiet_verb("eidola-add-measurement-cancel", "Cancel", cx)
                                .text_xs()
                                .probe(
                                    "settings/backends/eidola/measurements/add/cancel",
                                    gpui::Role::Button,
                                    "Cancel adding a measurement",
                                )
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.cancel_add_measurement(cx)),
                                ),
                        ),
                );
        }

        // The raw attestation evidence lives in the Record.
        value = value.child(
            h_flex().pt_0p5().text_xs().child(
                quiet_verb(
                    "eidola-open-record",
                    format!(
                        "Inspect attestation evidence in the Record ({})",
                        crate::actions::primary_shift_chord("L")
                    ),
                    cx,
                )
                .text_xs()
                .probe(
                    "settings/backends/eidola/open-record",
                    gpui::Role::Link,
                    "Inspect attestation evidence in the Record",
                )
                .on_click(|_, window, cx| {
                    window.dispatch_action(Box::new(OpenRecord), cx);
                }),
            ),
        );

        trust_row("Trusted measurements", cx, value).into_any_element()
    }

    /// One compact measurement line: the middle-truncated SNP launch digest
    /// (mono — the full `<snp>:<rtmr1>:<rtmr2>` triple travels via the Copy
    /// verb and the a11y labels), a role tag, and the role's verbs. Copy is
    /// always offered; Untrust only on user-override entries (the build pin
    /// is never untrustable — `untrust_measurement` edits the *override*
    /// list, so an Untrust on the pin was a no-op bug); Trust re-adds a
    /// dropped pin.
    fn measurement_line(
        &self,
        m: &MeasurementInfo,
        role: MeasurementRole,
        cx: &Context<Self>,
    ) -> gpui::AnyElement {
        let theme = cx.theme();
        let triple = measurement_triple(m);
        let slug = match role {
            MeasurementRole::Override(idx) => format!("{idx}"),
            MeasurementRole::Pinned | MeasurementRole::PinNotTrusted => "pin".to_string(),
        };

        // Only the dropped pin needs an inline marker — the row's status line
        // already says "pinned at build" in the no-override state, and
        // override entries are what the danger-colored status line counts.
        let tag = match role {
            MeasurementRole::PinNotTrusted => Some("pin — not trusted"),
            MeasurementRole::Pinned | MeasurementRole::Override(_) => None,
        };

        let mut verbs = h_flex().gap_3().flex_none().text_xs().child(copy_verb(
            SharedString::from(format!("eidola-measurement-{slug}-copy")),
            format!("settings/backends/eidola/measurements/{slug}/copy"),
            format!("Copy measurement {}", hex_summary(&m.snp)),
            triple.clone(),
            cx,
        ));
        match role {
            MeasurementRole::Pinned => {}
            MeasurementRole::Override(idx) => {
                let snp = m.snp.clone();
                // Short in the name, whole in the value: a 64-character hex
                // string as an accessible *name* is read out character by
                // character every time the element is focused.
                let aria = format!("Untrust measurement {}", hex_summary(&snp));
                verbs =
                    verbs.child(
                        quiet_verb(
                            SharedString::from(format!("eidola-untrust-{idx}")),
                            "Untrust",
                            cx,
                        )
                        .text_xs()
                        .probe(
                            format!("settings/backends/eidola/measurements/{idx}/untrust"),
                            gpui::Role::Button,
                            aria,
                        )
                        .aria_value(m.snp.clone())
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.untrust_measurement(snp.clone(), cx)
                        })),
                    );
            }
            MeasurementRole::PinNotTrusted => {
                verbs = verbs.child(
                    quiet_verb("eidola-trust-pin", "Trust", cx)
                        .text_xs()
                        .probe(
                            "settings/backends/eidola/measurements/pin/trust",
                            gpui::Role::Button,
                            "Trust the build pin measurement",
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.trust_measurement_spec(triple.clone(), cx)
                        })),
                );
            }
        }

        h_flex()
            .w_full()
            .gap_2()
            .items_baseline()
            .child(
                div()
                    .flex_none()
                    .text_xs()
                    .font_family("Menlo")
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(hex_summary(&m.snp))),
            )
            .when_some(tag, |line, tag| {
                line.child(
                    div()
                        .min_w_0()
                        .truncate()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .child(tag),
                )
            })
            .child(div().flex_1())
            .child(verbs)
            .into_any_element()
    }

    /// One hardware-CA row (root or intermediate) — display and editor in one
    /// place. The status line (danger when overridden), then either the
    /// resting verbs (Copy / Replace… / Clear on an override; "Set custom
    /// certificate…" on the vendor-chain pin) or, while editing, the
    /// paste-PEM textarea with Set/Cancel. PEM validation lives in app-core;
    /// a failure lands in the op-error banner and the input is kept.
    fn ca_row(
        &self,
        kind: CaKind,
        label: &'static str,
        slug: &'static str,
        current_pem: Option<&str>,
        cx: &Context<Self>,
    ) -> gpui::AnyElement {
        let theme = cx.theme();
        let is_override = current_pem.is_some();
        let editing = self.editing_ca == Some(kind);

        let mut value = v_flex().flex_1().gap_1().child(
            div()
                .text_color(if is_override {
                    theme.danger
                } else {
                    theme.muted_foreground
                })
                .child(if is_override {
                    "Custom certificate — overriding the AMD/Intel vendor chain."
                } else {
                    "Built-in AMD/Intel vendor chain."
                }),
        );

        if editing {
            let state = match kind {
                CaKind::Root => &self.root_ca_state,
                CaKind::Intermediate => &self.intermediate_ca_state,
            };
            value = value
                .child(
                    div()
                        .id(SharedString::from(format!("eidola-ca-{slug}-input-wrap")))
                        .probe(
                            format!("settings/backends/eidola/ca/{slug}/input"),
                            gpui::Role::TextInput,
                            format!("Paste {label} PEM"),
                        )
                        .w_full()
                        .flex()
                        .child(Input::new(state).a11y_labelled_by_ancestor().flex_1()),
                )
                .child(
                    h_flex()
                        .gap_3()
                        .text_xs()
                        .child(
                            quiet_verb(
                                SharedString::from(format!("eidola-ca-{slug}-set")),
                                "Set",
                                cx,
                            )
                            .text_xs()
                            .probe(
                                format!("settings/backends/eidola/ca/{slug}/set"),
                                gpui::Role::Button,
                                format!("Set {label}"),
                            )
                            .on_click(cx.listener(
                                move |this, _, window, cx| this.submit_ca(kind, window, cx),
                            )),
                        )
                        .child(
                            quiet_verb(
                                SharedString::from(format!("eidola-ca-{slug}-cancel")),
                                "Cancel",
                                cx,
                            )
                            .text_xs()
                            .probe(
                                format!("settings/backends/eidola/ca/{slug}/cancel"),
                                gpui::Role::Button,
                                format!("Cancel editing {label}"),
                            )
                            .on_click(cx.listener(|this, _, _, cx| this.cancel_edit_ca(cx))),
                        ),
                );
        } else {
            let mut verbs = h_flex().gap_3().text_xs();
            if is_override {
                verbs = verbs
                    .child(copy_verb(
                        SharedString::from(format!("eidola-ca-{slug}-copy")),
                        format!("settings/backends/eidola/ca/{slug}/copy"),
                        format!("Copy {label}"),
                        current_pem.unwrap_or_default().to_string(),
                        cx,
                    ))
                    .child(
                        quiet_verb(
                            SharedString::from(format!("eidola-ca-{slug}-change")),
                            "Replace…",
                            cx,
                        )
                        .text_xs()
                        .probe(
                            format!("settings/backends/eidola/ca/{slug}/change"),
                            gpui::Role::Button,
                            format!("Replace {label}"),
                        )
                        .on_click(cx.listener(
                            move |this, _, window, cx| this.begin_edit_ca(kind, window, cx),
                        )),
                    )
                    .child(
                        quiet_verb(
                            SharedString::from(format!("eidola-ca-{slug}-clear")),
                            "Clear",
                            cx,
                        )
                        .text_xs()
                        .probe(
                            format!("settings/backends/eidola/ca/{slug}/clear"),
                            gpui::Role::Button,
                            format!("Clear {label}"),
                        )
                        .on_click(cx.listener(move |this, _, _, cx| this.clear_ca(kind, cx))),
                    );
            } else {
                verbs =
                    verbs.child(
                        quiet_verb(
                            SharedString::from(format!("eidola-ca-{slug}-change")),
                            "Set custom certificate…",
                            cx,
                        )
                        .text_xs()
                        .probe(
                            format!("settings/backends/eidola/ca/{slug}/change"),
                            gpui::Role::Button,
                            format!("Set a custom {label}"),
                        )
                        .on_click(cx.listener(
                            move |this, _, window, cx| this.begin_edit_ca(kind, window, cx),
                        )),
                    );
            }
            value = value.child(verbs);
        }

        trust_row(label, cx, value).into_any_element()
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

/// Middle-truncated hex for compact display (`ab12cd…90abcd`). Full values
/// never render inline in Settings — they travel via the Copy verbs and a11y
/// labels, and the Record window holds the raw evidence.
fn hex_summary(hex: &str) -> String {
    if hex.len() <= 16 {
        hex.to_string()
    } else {
        format!("{}…{}", &hex[..6], &hex[hex.len() - 6..])
    }
}

/// The re-addable `<snp>:<rtmr1>:<rtmr2>` triple form of a measurement — the
/// format the add field / CLI accept, and what the Copy verbs place on the
/// clipboard.
fn measurement_triple(m: &MeasurementInfo) -> String {
    format!("{}:{}:{}", m.snp, m.tdx_rtmr1, m.tdx_rtmr2)
}

/// A quiet "Copy" text verb that places `value` on the clipboard. Mirrors the
/// Record window's copy affordance (`write_to_clipboard`); the click touches
/// no view state, so it takes a plain handler rather than `cx.listener`.
fn copy_verb(
    id: impl Into<gpui::ElementId>,
    probe_name: String,
    aria: String,
    value: String,
    cx: &gpui::App,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    div()
        .id(id)
        .probe(probe_name, gpui::Role::Button, aria)
        // The name says *what* is copied, short enough to hear; the value
        // carries the payload itself for AT that reads it on request.
        .aria_value(value.clone())
        .flex_none()
        .text_xs()
        .cursor_pointer()
        .text_color(theme.link)
        .hover(|s| s.text_color(theme.link_hover))
        .child("Copy")
        .on_click(move |_, _, cx| {
            cx.write_to_clipboard(ClipboardItem::new_string(value.clone()));
        })
}

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

/// The prominent danger-tinted band shown at the top of the Eidola trust
/// surface whenever any part of the bundle is overridden — mirrors the
/// Updates window's Unverifiable security band. Names exactly which values
/// are overridden and states the consequence plainly. `role=Alert`.
fn trust_warning_band(overridden: &[&str], cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    let list = overridden.join(", ");
    let summary = format!("Overridden: {list}.");
    v_flex()
        .id("eidola-trust-warning")
        .probe(
            "settings/backends/eidola/trust-warning",
            gpui::Role::Alert,
            format!(
                "{summary} This client no longer verifies against the trust root built into \
                 this binary."
            ),
        )
        .w_full()
        .px_3()
        .py_3()
        .gap_2()
        .rounded_md()
        .bg(theme.danger.opacity(0.08))
        .child(
            div()
                .font_semibold()
                .text_color(theme.danger)
                .child("Trust overrides active"),
        )
        .child(
            div()
                .text_sm()
                .text_color(theme.danger)
                .child(SharedString::from(summary)),
        )
        .child(div().text_xs().text_color(theme.danger).child(
            "This client no longer verifies against the trust root built into this \
                 binary. Revert to the pin unless you know exactly why these are set.",
        ))
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
    label: impl Into<SharedString>,
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
        .child(label.into())
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
                    .child(Input::new(state).a11y_labelled_by_ancestor()),
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
