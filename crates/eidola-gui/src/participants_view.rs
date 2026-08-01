//! The per-space **Participants** view — who is in a conversation and how each
//! one responds.
//!
//! Opened via `Space > Participants…` (registered per-`SpaceView`, so the menu
//! item targets the focused space and macOS greys it when no space window is
//! open). A window-local lens over the shared per-space participant membership
//! owned by [`crate::stores::ParticipantsStore`] (keyed by space id, refreshed
//! on `Change::Participants`). The view never touches `AppCore` directly.
//!
//! **The edit-everywhere-vs-override-here fork** (spec §4): a participant is
//! either **owned** by this space (`source == "owned"` — editing edits this
//! space only) or a **referenced global** (`source == "referenced"` — the
//! shared library "You" today, agents later). A referenced global's editor
//! carries a mode toggle: **Everyone** writes the shared global's own config
//! (via `ParticipantsStore::update_everywhere`); **This space only** writes the
//! per-membership override (via `ParticipantsStore::set_override`; a field left
//! blank/reset inherits). Owned participants have no fork — just the one editor.

use eidola_app_core::{
    NewParticipant, ParticipantInfo, ParticipantOverride, ParticipantReference, ParticipantUpdate,
};
use gpui::{
    AppContext, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement,
    Render, ScrollHandle, SharedString, StatefulInteractiveElement, Styled, Subscription, Window,
    div, prelude::FluentBuilder, px,
};
use gpui_component::{
    ActiveTheme, StyledExt, h_flex,
    input::{Input, InputState},
    label::Label,
    v_flex,
};

use crate::actions::CloseWindow;
use crate::probe::Probe as _;
use crate::stores::{ParticipantsStore, Stores};

const TITLE_BAR_RESERVE: gpui::Pixels = crate::titlebar::DRAG_BAND_HEIGHT;

/// The three notify-policy values with their human labels. The stored value is
/// the schema enum (`explicit`/`human`/`all`); the label is what a person reads
/// ("Responds: …").
pub(crate) const NOTIFY_POLICIES: [(&str, &str); 3] = [
    ("explicit", "when asked"),
    ("human", "to people"),
    ("all", "to everything"),
];

/// The system prompt a newly created agent participant starts from — a short,
/// general-purpose charter rather than a persona, so the field arrives filled
/// with something honest that a person can edit down or replace. Shared so
/// every participant-creating surface offers the same starting point.
pub const DEFAULT_AGENT_SYSTEM_PROMPT: &str = "You are a participant in a shared conversation. Answer plainly, say when \
     you are unsure, and keep replies as short as the question allows.";

pub(crate) fn notify_label(policy: &str) -> &'static str {
    NOTIFY_POLICIES
        .iter()
        .find(|(v, _)| *v == policy)
        .map(|(_, l)| *l)
        .unwrap_or("when asked")
}

/// Which editor writes a referenced global's fields.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditMode {
    /// Edit the shared global's own config (edits everywhere).
    Everywhere,
    /// Write the per-membership override (this space only).
    OverrideHere,
}

/// The in-progress editor for one participant. Created on `begin_edit`, dropped
/// on save/cancel. Holds the field input entities + the working notify policy /
/// model ref / mode. For a referenced global it carries the reference detail so
/// the two modes can seed the right values.
struct EditState {
    participant_id: String,
    kind: String,
    is_referenced: bool,
    reference: Option<ParticipantReference>,
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    /// The working model selection (`None` = no model set).
    model_ref: Option<String>,
    notify_policy: String,
    mode: EditMode,
}

/// The in-progress add-a-participant form (agents only).
struct AddState {
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    model_ref: Option<String>,
    notify_policy: String,
}

/// The in-progress "Save as template…" input.
struct TemplateState {
    title: Entity<InputState>,
}

/// Where an open model picker writes its selection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PickerTarget {
    Edit,
    Add,
}

pub struct ParticipantsView {
    stores: Stores,
    participants: Entity<ParticipantsStore>,
    space_id: String,
    space_title: Option<String>,
    focus_handle: FocusHandle,
    editing: Option<EditState>,
    adding: Option<AddState>,
    template_form: Option<TemplateState>,
    /// When `Some`, the model picker dropdown is open, targeting the edit or add
    /// form.
    picker: Option<PickerTarget>,
    /// Tracks the roster body scroll so the right-edge overlay indicator can
    /// bind to it (shown only while scrolling).
    body_scroll: ScrollHandle,
    /// Tracks the open model-picker dropdown's own (nested) scroll, so its
    /// overlay indicator binds independently of the roster body. One handle
    /// suffices — at most one picker is open at a time; it's reset to the top
    /// on each open.
    picker_scroll: ScrollHandle,
    _subscriptions: Vec<Subscription>,
}

impl ParticipantsView {
    pub fn new(
        stores: Stores,
        space_id: String,
        space_title: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let participants = stores.participants.clone();
        // Ensure this space's membership is loaded (lazy per-space fetch).
        participants.update(cx, |s, cx| s.ensure(space_id.clone(), cx));
        let mut subs = vec![cx.observe(&participants, |_, _, cx| cx.notify())];
        // The model field renders backend/model display names, so re-render when
        // those stores change (a backend enabled, an engine loaded).
        subs.push(cx.observe(&stores.backends, |_, _, cx| cx.notify()));
        subs.push(cx.observe(&stores.models, |_, _, cx| cx.notify()));
        subs.push(cx.observe(&stores.local_models, |_, _, cx| cx.notify()));

        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        Self {
            stores,
            participants,
            space_id,
            space_title,
            focus_handle,
            editing: None,
            adding: None,
            template_form: None,
            picker: None,
            body_scroll: ScrollHandle::new(),
            picker_scroll: ScrollHandle::new(),
            _subscriptions: subs,
        }
    }

    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    /// Wrap the roster scroll container in a `relative` column that also holds
    /// the right-edge overlay scroll indicator as a sibling (never a child of
    /// the scrolling element, or it scrolls away with the content).
    fn wrap_body(&self, body: impl IntoElement, window: &Window) -> impl IntoElement {
        v_flex()
            .relative()
            .flex_1()
            .w_full()
            .min_h_0()
            .child(body)
            .child(crate::scrollbar::vertical(
                "participants-scrollbar",
                &self.body_scroll,
                window,
            ))
    }

    fn list(&self, cx: &gpui::App) -> Vec<ParticipantInfo> {
        self.participants.read(cx).list(&self.space_id).to_vec()
    }

    /// Re-fetch this space's participants (the Retry affordance on a failed
    /// load — `ensure` declines once a `Failed` cell exists, so retry is the
    /// only path back).
    pub fn retry_load(&mut self, cx: &mut Context<Self>) {
        let space_id = self.space_id.clone();
        self.participants
            .update(cx, |s, cx| s.refresh(space_id, cx));
        cx.notify();
    }

    // --- Test seams ------------------------------------------------------

    #[doc(hidden)]
    pub fn editing_participant_id(&self) -> Option<&str> {
        self.editing.as_ref().map(|e| e.participant_id.as_str())
    }

    #[doc(hidden)]
    pub fn editing_mode(&self) -> Option<EditMode> {
        self.editing.as_ref().map(|e| e.mode)
    }

    #[doc(hidden)]
    pub fn editing_label_state(&self) -> Option<Entity<InputState>> {
        self.editing.as_ref().map(|e| e.label.clone())
    }

    #[doc(hidden)]
    pub fn is_adding(&self) -> bool {
        self.adding.is_some()
    }

    #[doc(hidden)]
    pub fn adding_label_state(&self) -> Option<Entity<InputState>> {
        self.adding.as_ref().map(|a| a.label.clone())
    }

    #[doc(hidden)]
    pub fn is_saving_template(&self) -> bool {
        self.template_form.is_some()
    }

    // --- Editing ---------------------------------------------------------

    /// Begin editing a participant. Seeds the field inputs from its effective
    /// config; for a referenced global, defaults to the "override here" mode
    /// (the safe, this-space-only choice).
    pub fn begin_edit(
        &mut self,
        participant_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(p) = self.list(cx).into_iter().find(|p| p.id == participant_id) else {
            return;
        };
        self.adding = None;
        self.template_form = None;
        self.picker = None;
        let is_referenced = p.source == "referenced";
        let label = cx.new(|cx| InputState::new(window, cx).default_value(&p.label));
        let system_prompt = cx.new(|cx| {
            let s = InputState::new(window, cx)
                .auto_grow(2, 8)
                .placeholder("A short instruction for how this participant behaves.");
            s.default_value(p.system_prompt.clone().unwrap_or_default())
        });
        self.editing = Some(EditState {
            participant_id: p.id.clone(),
            kind: p.kind.clone(),
            is_referenced,
            reference: p.reference.clone(),
            label,
            system_prompt,
            model_ref: p.model_ref.clone(),
            notify_policy: p.notify_policy.clone(),
            mode: if is_referenced {
                EditMode::OverrideHere
            } else {
                EditMode::Everywhere
            },
        });
        cx.notify();
    }

    /// Switch a referenced global's editor between "edit everywhere" and
    /// "override here", re-seeding the visible field values from the mode's
    /// source (the shared base, or the effective/override).
    pub fn set_edit_mode(&mut self, mode: EditMode, window: &mut Window, cx: &mut Context<Self>) {
        // Snapshot the handles + reference so the `self.editing` borrow doesn't
        // straddle the `cx.update` calls below.
        let (label_entity, prompt_entity, reference, already) = {
            let Some(edit) = self.editing.as_ref() else {
                return;
            };
            (
                edit.label.clone(),
                edit.system_prompt.clone(),
                edit.reference.clone(),
                edit.mode == mode,
            )
        };
        if already {
            return;
        }
        // Compute the field values this mode surfaces.
        let (label, model_ref, notify_policy, prompt) = match (&reference, mode) {
            (Some(r), EditMode::Everywhere) => (
                r.base_label.clone(),
                r.base_model_ref.clone(),
                r.base_notify_policy.clone(),
                r.base_system_prompt.clone().unwrap_or_default(),
            ),
            (Some(r), EditMode::OverrideHere) => (
                r.override_label
                    .clone()
                    .unwrap_or_else(|| r.base_label.clone()),
                r.override_model_ref
                    .clone()
                    .or_else(|| r.base_model_ref.clone()),
                r.override_notify_policy
                    .clone()
                    .unwrap_or_else(|| r.base_notify_policy.clone()),
                r.override_system_prompt
                    .clone()
                    .or_else(|| r.base_system_prompt.clone())
                    .unwrap_or_default(),
            ),
            // Owned participants never switch modes; nothing to re-seed.
            (None, _) => {
                if let Some(edit) = self.editing.as_mut() {
                    edit.mode = mode;
                }
                cx.notify();
                return;
            }
        };
        label_entity.update(cx, |s, cx| s.set_value(&label, window, cx));
        prompt_entity.update(cx, |s, cx| s.set_value(&prompt, window, cx));
        if let Some(edit) = self.editing.as_mut() {
            edit.mode = mode;
            edit.model_ref = model_ref;
            edit.notify_policy = notify_policy;
        }
        cx.notify();
    }

    pub fn set_edit_notify_policy(&mut self, policy: &str, cx: &mut Context<Self>) {
        if let Some(edit) = self.editing.as_mut() {
            edit.notify_policy = policy.to_string();
            cx.notify();
        }
    }

    pub fn cancel_edit(&mut self, cx: &mut Context<Self>) {
        self.editing = None;
        self.picker = None;
        cx.notify();
    }

    /// Commit the editor. Routes to "edit everywhere" or "override here" per the
    /// current mode; an owned participant is always "edit everywhere" (of its
    /// own config).
    pub fn save_edit(&mut self, cx: &mut Context<Self>) {
        let Some(edit) = self.editing.take() else {
            return;
        };
        let label = edit.label.read(cx).value().trim().to_string();
        let system_prompt = edit.system_prompt.read(cx).value().trim().to_string();
        let is_agent = edit.kind == "agent";
        let space_id = self.space_id.clone();
        let pid = edit.participant_id.clone();

        if edit.is_referenced && edit.mode == EditMode::OverrideHere {
            // Write per-membership overrides. An empty field on a referenced
            // agent's model/prompt reverts to inherited (None); label always
            // has a value (it's required), so a label equal to the base still
            // writes an (equal) override — acceptable and honest.
            let mut ov = ParticipantOverride {
                label: Some(Some(label)),
                notify_policy: Some(Some(edit.notify_policy.clone())),
                ..Default::default()
            };
            if is_agent {
                ov.model_ref = Some(edit.model_ref.clone().filter(|s| !s.is_empty()));
                ov.system_prompt = Some(if system_prompt.is_empty() {
                    None
                } else {
                    Some(system_prompt)
                });
            }
            self.participants
                .update(cx, |s, cx| s.set_override(space_id, pid, ov, cx));
        } else {
            // Edit the participant's own config (edit everywhere / owned edit).
            let mut update = ParticipantUpdate {
                label: Some(label),
                notify_policy: Some(edit.notify_policy.clone()),
                ..Default::default()
            };
            if is_agent {
                update.model_ref = Some(edit.model_ref.clone().filter(|s| !s.is_empty()));
                update.system_prompt = Some(if system_prompt.is_empty() {
                    None
                } else {
                    Some(system_prompt)
                });
            }
            self.participants
                .update(cx, |s, cx| s.update_everywhere(space_id, pid, update, cx));
        }
        self.picker = None;
        cx.notify();
    }

    pub fn remove(&mut self, participant_id: &str, cx: &mut Context<Self>) {
        let space_id = self.space_id.clone();
        let pid = participant_id.to_string();
        self.participants
            .update(cx, |s, cx| s.remove(space_id, pid, cx));
        if self.editing.as_ref().map(|e| e.participant_id.as_str()) == Some(participant_id) {
            self.editing = None;
        }
        cx.notify();
    }

    // --- Adding ----------------------------------------------------------

    pub fn begin_add(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.editing = None;
        self.template_form = None;
        self.picker = None;
        let label = cx.new(|cx| InputState::new(window, cx).placeholder("Participant name"));
        let system_prompt = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(2, 8)
                .placeholder("A short instruction for how this participant behaves.")
        });
        self.adding = Some(AddState {
            label,
            system_prompt,
            model_ref: Some(self.stores.config.read(cx).default_model()),
            notify_policy: "human".to_string(),
        });
        cx.notify();
    }

    pub fn set_add_notify_policy(&mut self, policy: &str, cx: &mut Context<Self>) {
        if let Some(add) = self.adding.as_mut() {
            add.notify_policy = policy.to_string();
            cx.notify();
        }
    }

    pub fn cancel_add(&mut self, cx: &mut Context<Self>) {
        self.adding = None;
        self.picker = None;
        cx.notify();
    }

    pub fn save_add(&mut self, cx: &mut Context<Self>) {
        let Some(add) = self.adding.take() else {
            return;
        };
        let label = add.label.read(cx).value().trim().to_string();
        if label.is_empty() {
            // Keep the form open; app-core would reject an empty label anyway,
            // but not sending saves a round-trip.
            self.adding = Some(add);
            return;
        }
        let system_prompt = add.system_prompt.read(cx).value().trim().to_string();
        let participant = NewParticipant {
            label,
            model_ref: add.model_ref.clone().filter(|s| !s.is_empty()),
            system_prompt: if system_prompt.is_empty() {
                None
            } else {
                Some(system_prompt)
            },
            notify_policy: add.notify_policy.clone(),
        };
        let space_id = self.space_id.clone();
        self.participants
            .update(cx, |s, cx| s.add(space_id, participant, cx));
        self.picker = None;
        cx.notify();
    }

    // --- Save as template ------------------------------------------------

    pub fn begin_template(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.editing = None;
        self.adding = None;
        let default_title = self
            .space_title
            .clone()
            .unwrap_or_else(|| "My template".to_string());
        let title = cx.new(|cx| InputState::new(window, cx).default_value(&default_title));
        self.template_form = Some(TemplateState { title });
        cx.notify();
    }

    pub fn cancel_template(&mut self, cx: &mut Context<Self>) {
        self.template_form = None;
        cx.notify();
    }

    pub fn save_template(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.template_form.take() else {
            return;
        };
        let title = form.title.read(cx).value().trim().to_string();
        if title.is_empty() {
            self.template_form = Some(form);
            return;
        }
        let space_id = self.space_id.clone();
        self.stores
            .templates
            .update(cx, |s, cx| s.create_from_space(space_id, title, cx));
        cx.notify();
    }

    // --- Model picker ----------------------------------------------------

    fn toggle_picker(&mut self, target: PickerTarget, cx: &mut Context<Self>) {
        self.picker = if self.picker == Some(target) {
            None
        } else {
            // A freshly opened picker starts at the top.
            self.picker_scroll = ScrollHandle::new();
            Some(target)
        };
        cx.notify();
    }

    /// Test seam: open the add form's model picker (the nested dropdown), so a
    /// render-smoke can draw the picker with its overlay indicator bound.
    #[doc(hidden)]
    pub fn open_add_picker_for_test(&mut self, cx: &mut Context<Self>) {
        if self.picker != Some(PickerTarget::Add) {
            self.toggle_picker(PickerTarget::Add, cx);
        }
    }

    /// Test seam: select a model into the active form (edit or add).
    pub fn select_model(&mut self, model_id: &str, cx: &mut Context<Self>) {
        match self.picker.take().or(if self.editing.is_some() {
            Some(PickerTarget::Edit)
        } else if self.adding.is_some() {
            Some(PickerTarget::Add)
        } else {
            None
        }) {
            Some(PickerTarget::Edit) => {
                if let Some(edit) = self.editing.as_mut() {
                    edit.model_ref = Some(model_id.to_string());
                }
            }
            Some(PickerTarget::Add) => {
                if let Some(add) = self.adding.as_mut() {
                    add.model_ref = Some(model_id.to_string());
                }
            }
            None => {}
        }
        cx.notify();
    }

    /// The human display for a model selection id: `(model name, backend name)`.
    fn model_display(&self, selection: &str, cx: &gpui::App) -> (SharedString, SharedString) {
        model_display(&self.stores, selection, cx)
    }
}

/// The human display for a model selection id: `(model name, backend name)`.
/// Shared by the Participants view and the Space Templates pane.
pub(crate) fn model_display(
    stores: &Stores,
    selection: &str,
    cx: &gpui::App,
) -> (SharedString, SharedString) {
    let mref = eidola_app_core::parse_model_ref(selection);
    let backend_name = stores
        .backends
        .read(cx)
        .get(&mref.backend_id)
        .map(|b| b.display_name.clone())
        .unwrap_or_else(|| match mref.backend_id.as_str() {
            eidola_app_core::EIDOLA_BACKEND_ID => "Eidola".to_string(),
            eidola_app_core::LOCAL_BACKEND_ID => "Local".to_string(),
            other => other.to_string(),
        });
    let local = stores.local_models.read(cx);
    let model_name = local
        .models()
        .iter()
        .chain(local.external().iter().flat_map(|b| b.models.iter()))
        .find(|m| m.id == selection)
        .map(|m| m.display_name.clone())
        .unwrap_or_else(|| mref.model.clone());
    (model_name.into(), backend_name.into())
}

/// The grouped list of selectable models — engine groups first, then the
/// fetch-based catalogs. Mirrors the request panel's data sources so a
/// participant's model field offers exactly what an ask can route to. Shared by
/// the Participants view and the Space Templates pane.
pub(crate) fn model_groups(
    stores: &Stores,
    cx: &gpui::App,
) -> Vec<(String, Vec<(String, String)>)> {
    let mut groups: Vec<(String, Vec<(String, String)>)> = Vec::new();
    let local = stores.local_models.read(cx);
    let backends = stores.backends.read(cx);
    if backends.is_enabled(eidola_app_core::LOCAL_BACKEND_ID) {
        let selectable = local.selectable_models();
        if !selectable.is_empty() {
            let header = backends
                .get(eidola_app_core::LOCAL_BACKEND_ID)
                .map(|b| b.display_name.clone())
                .unwrap_or_else(|| "Local".into());
            groups.push((
                header,
                selectable
                    .into_iter()
                    .map(|m| (m.id, m.display_name))
                    .collect(),
            ));
        }
    }
    for ext in local.external() {
        if !ext.enabled {
            continue;
        }
        let selectable = local.external_selectable_models(&ext.backend_id);
        if !selectable.is_empty() {
            groups.push((
                ext.display_name.clone(),
                selectable
                    .into_iter()
                    .map(|m| (m.id, m.display_name))
                    .collect(),
            ));
        }
    }
    for catalog in stores.models.read(cx).catalogs() {
        let header = if catalog.backend.kind == eidola_app_core::BackendKind::Eidola {
            "Via Eidola".to_string()
        } else {
            catalog.backend.display_name.clone()
        };
        if let Some(models) = catalog.models.value()
            && !models.is_empty()
        {
            groups.push((
                header,
                models
                    .iter()
                    .map(|m| (m.id.clone(), m.id.clone()))
                    .collect(),
            ));
        }
    }
    groups
}

/// A model-picker dropdown field shared by the Participants view and Templates
/// pane: a button showing the current model, plus (when `open`) a grouped list
/// of selectable models. `on_pick` receives the chosen model id.
// Distinct, non-groupable params (data, open-state, the picker's scroll handle,
// two callbacks) — a config struct would obscure more than it'd tidy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn model_field<V: 'static>(
    stores: &Stores,
    current: Option<&str>,
    open: bool,
    probe_prefix: SharedString,
    picker_scroll: &ScrollHandle,
    cx: &Context<V>,
    on_toggle: impl Fn(&mut V, &gpui::ClickEvent, &mut Window, &mut Context<V>) + 'static,
    on_pick: impl Fn(&str, &mut V, &mut Context<V>) + Clone + 'static,
) -> gpui::AnyElement {
    let theme = cx.theme();
    let (name, backend) = match current {
        Some(sel) if !sel.is_empty() => {
            let (n, b) = model_display(stores, sel, cx);
            (n, Some(b))
        }
        _ => ("Choose a model…".into(), None),
    };
    let button = h_flex()
        .id(SharedString::from(format!("{probe_prefix}-button")))
        .probe(probe_prefix.clone(), gpui::Role::Button, "Model")
        .w_full()
        .px_2()
        .py_1()
        .gap_2()
        .items_center()
        .justify_between()
        .rounded_md()
        .border_1()
        .border_color(theme.border)
        .cursor_pointer()
        .hover(|s| s.bg(theme.secondary.opacity(0.5)))
        .child(
            h_flex()
                .gap_1p5()
                .items_baseline()
                .child(div().text_sm().child(name))
                .when_some(backend, |el, b| {
                    el.child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(format!("· {b}"))),
                    )
                }),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child("▾"),
        )
        .on_click(cx.listener(on_toggle));

    let mut col = v_flex().w_full().gap_1().child(button);
    if open {
        let groups = model_groups(stores, cx);
        let mut menu = v_flex()
            .id(SharedString::from(format!("{probe_prefix}-menu")))
            .probe(
                format!("{probe_prefix}/menu"),
                gpui::Role::ListBox,
                "Models",
            )
            .w_full()
            .max_h(px(220.))
            .overflow_y_scroll()
            .track_scroll(picker_scroll)
            .py_1()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.background);
        if groups.is_empty() {
            menu = menu.child(
                div()
                    .px_3()
                    .py_2()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("No models available."),
            );
        }
        for (gi, (header, models)) in groups.into_iter().enumerate() {
            menu = menu.child(
                div()
                    .px_3()
                    .pt_2()
                    .pb_1()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(header)),
            );
            for (mi, (id, display)) in models.into_iter().enumerate() {
                let selected = current == Some(id.as_str());
                let pick_id = id.clone();
                let on_pick = on_pick.clone();
                menu = menu.child(
                    div()
                        .id(SharedString::from(format!("{probe_prefix}-opt-{gi}-{mi}")))
                        .probe(
                            format!("{probe_prefix}/option/{gi}/{mi}"),
                            gpui::Role::Button,
                            display.clone(),
                        )
                        .px_3()
                        .py_1()
                        .cursor_pointer()
                        .text_sm()
                        .hover(|s| s.bg(theme.secondary.opacity(0.6)))
                        .when(selected, |el| el.text_color(theme.link))
                        .child(SharedString::from(display))
                        .on_click(cx.listener(move |this, _, _, cx| on_pick(&pick_id, this, cx))),
                );
            }
        }
        // The dropdown overflows its max-height with enough models, scrolling
        // independently of the roster body — so it carries its own overlay
        // indicator, a sibling of the scroll container inside a `relative`
        // wrapper (a bounded popover, so no window-corner clearance).
        col = col.child(div().relative().w_full().child(menu).child(
            crate::scrollbar::vertical_floating("model-picker-scrollbar", picker_scroll),
        ));
    }
    col.into_any_element()
}

impl Render for ParticipantsView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let drag_band =
            crate::titlebar::drag_band("participants-titlebar", TITLE_BAR_RESERVE, window, cx);
        let theme = cx.theme();
        let participants = self.list(cx);
        let (loading, load_error, has_value) = {
            let cell = self.participants.read(cx).participants(&self.space_id);
            (
                cell.is_loading(),
                cell.error().map(|e| e.to_string()),
                cell.has_value(),
            )
        };
        // A failed *initial* load (no prior data) must not read as an empty
        // membership with live Add/Save controls — render the error + a Retry
        // instead. A failed *refresh* over existing data keeps the list (stale)
        // and surfaces a quiet retry alongside it.
        let load_failed_blank = load_error.is_some() && !has_value;
        let op_error = self
            .participants
            .read(cx)
            .op_error(&self.space_id)
            .map(str::to_string);

        let mut root = crate::chrome::round_client_corners(v_flex(), window)
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .relative()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .pt(TITLE_BAR_RESERVE);

        // Chapter-style heading: "Participants" + the space title, between
        // hairline rules — the same book voice as the Library.
        let subtitle = self
            .space_title
            .clone()
            .unwrap_or_else(|| "Untitled space".to_string());
        root = root.child(
            v_flex()
                .w_full()
                .px_10()
                .pt_4()
                .pb_2()
                .gap_0p5()
                .child(
                    h_flex()
                        .w_full()
                        .items_center()
                        .gap_4()
                        .child(div().h(px(1.)).flex_1().bg(theme.border))
                        .child(
                            div()
                                .italic()
                                .text_color(theme.muted_foreground)
                                .child("Participants"),
                        )
                        .child(div().h(px(1.)).flex_1().bg(theme.border)),
                )
                .child(
                    div()
                        .w_full()
                        .text_center()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .truncate()
                        .child(SharedString::from(subtitle)),
                ),
        );

        let mut body = v_flex()
            .id("participants-body")
            .w_full()
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.body_scroll)
            .px_10()
            .py_4()
            .gap_2();

        // A failed initial load owns the whole body: the honest error + Retry,
        // no phantom-empty roster with live controls.
        if load_failed_blank {
            let err = load_error.clone().unwrap_or_default();
            body = body.child(load_error_panel(
                "participants/retry",
                "Couldn't load this space's participants.",
                &err,
                cx,
                cx.listener(|this, _, _, cx| this.retry_load(cx)),
            ));
            return root.child(self.wrap_body(body, window));
        }

        if participants.is_empty() && loading {
            body = body.child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("Loading…"),
            );
        }

        for p in &participants {
            let editing_this = self
                .editing
                .as_ref()
                .map(|e| e.participant_id == p.id)
                .unwrap_or(false);
            if editing_this {
                body = body.child(self.render_editor(p, cx));
            } else {
                body = body.child(self.render_row(p, cx));
            }
        }

        // Add-a-participant.
        if self.adding.is_some() {
            body = body.child(self.render_add_form(cx));
        } else {
            body = body.child(
                div()
                    .id("participants-add")
                    .probe("participants/add", gpui::Role::Button, "Add participant")
                    .mt_2()
                    .cursor_pointer()
                    .text_sm()
                    .text_color(theme.link)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("Add participant…")
                    .on_click(cx.listener(|this, _, window, cx| this.begin_add(window, cx))),
            );
        }

        // Save-as-template + op error.
        if let Some(form) = self.template_form.as_ref() {
            body = body.child(self.render_template_form(&form.title, cx));
        } else {
            body = body.child(
                div()
                    .id("participants-save-template")
                    .probe(
                        "participants/save-template",
                        gpui::Role::Button,
                        "Save as template",
                    )
                    .mt_1()
                    .cursor_pointer()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("Save these participants as a template…")
                    .on_click(cx.listener(|this, _, window, cx| this.begin_template(window, cx))),
            );
        }

        if let Some(err) = op_error {
            body = body.child(
                div()
                    .id("participants-error")
                    .probe("participants/error", gpui::Role::Alert, err.clone())
                    .mt_2()
                    .child(error_banner(&err, cx)),
            );
        }

        // A refresh that failed *over* existing data: keep the (stale) list but
        // offer a quiet retry so the staleness isn't silent.
        if load_error.is_some() && has_value {
            body = body.child(
                div()
                    .id("participants-retry")
                    .probe("participants/retry", gpui::Role::Button, "Retry")
                    .mt_1()
                    .cursor_pointer()
                    .text_xs()
                    .text_color(theme.link)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("Couldn't refresh — retry")
                    .on_click(cx.listener(|this, _, _, cx| this.retry_load(cx))),
            );
        }

        // The drag band goes on **last** — a blocking hitbox only suppresses
        // hitboxes registered before it (see `crate::overlay`).
        root.child(self.wrap_body(body, window)).child(drag_band)
    }
}

impl ParticipantsView {
    /// A resting participant row: identity + who-responds line + Edit/Remove.
    fn render_row(&self, p: &ParticipantInfo, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let is_human = p.kind == "human";
        let is_referenced = p.source == "referenced";
        let can_remove = p.id != eidola_app_core::HUMAN_PARTICIPANT_ID;
        let pid = p.id.clone();
        let pid2 = p.id.clone();
        let label = p.label.clone();
        let verb_subject = p.label.clone();

        // Secondary line: for an agent, "model · backend"; for the human, the
        // model line is meaningless (people don't have a model), so it is
        // suppressed. A referenced global also shows a quiet "shared" tag.
        let detail: Option<SharedString> = if is_human {
            None
        } else if let Some(model) = p.model_ref.as_deref() {
            let (name, backend) = self.model_display(model, cx);
            Some(format!("{name} · {backend}").into())
        } else {
            Some("no model set".into())
        };

        let responds = format!("Responds {}", notify_label(&p.notify_policy));

        h_flex()
            .id(SharedString::from(format!("participant-row-{pid}")))
            .w_full()
            .py_2()
            .gap_3()
            .items_start()
            .border_b_1()
            .border_color(theme.border.opacity(0.5))
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .gap_0p5()
                    .child(
                        h_flex()
                            .items_center()
                            .gap_2()
                            .child(div().font_medium().child(SharedString::from(label)))
                            .when(is_referenced, |el| {
                                el.child(
                                    div()
                                        .px_1p5()
                                        .rounded_sm()
                                        .bg(theme.muted.opacity(0.5))
                                        .text_xs()
                                        .text_color(theme.muted_foreground)
                                        .child("shared"),
                                )
                            }),
                    )
                    .when_some(detail, |el, detail| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .child(detail),
                        )
                    })
                    .when(!is_human, |el| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground.opacity(0.8))
                                .child(SharedString::from(responds)),
                        )
                    }),
            )
            .child(
                h_flex()
                    .gap_1()
                    .child(ghost_button_labeled(
                        SharedString::from(format!("edit-{pid}")),
                        SharedString::from(format!("participants/{pid}/edit")),
                        "Edit",
                        format!("Edit {verb_subject}"),
                        false,
                        cx,
                        cx.listener(move |this, _, window, cx| this.begin_edit(&pid, window, cx)),
                    ))
                    .when(can_remove, |el| {
                        el.child(ghost_button_labeled(
                            SharedString::from(format!("remove-{pid2}")),
                            SharedString::from(format!("participants/{pid2}/remove")),
                            "Remove",
                            format!("Remove {verb_subject}"),
                            false,
                            cx,
                            cx.listener(move |this, _, _, cx| this.remove(&pid2, cx)),
                        ))
                    }),
            )
    }

    /// The inline editor card for a participant.
    fn render_editor(&self, p: &ParticipantInfo, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let Some(edit) = self.editing.as_ref() else {
            return div().into_any_element();
        };
        let is_agent = edit.kind == "agent";

        let mut card = v_flex()
            .id(SharedString::from(format!("participant-editor-{}", p.id)))
            .w_full()
            .p_3()
            .gap_3()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3));

        // Referenced globals: the edit-everywhere-vs-override-here fork.
        if edit.is_referenced {
            let mode = edit.mode;
            card = card.child(
                v_flex()
                    .gap_1()
                    .child(
                        h_flex()
                            .gap_2()
                            .child(mode_chip(
                                "edit-everywhere".into(),
                                "participants/editor/mode/everywhere".into(),
                                "Everyone".into(),
                                mode == EditMode::Everywhere,
                                cx,
                                cx.listener(|this, _, window, cx| {
                                    this.set_edit_mode(EditMode::Everywhere, window, cx)
                                }),
                            ))
                            .child(mode_chip(
                                "override-here".into(),
                                "participants/editor/mode/override".into(),
                                "This space only".into(),
                                mode == EditMode::OverrideHere,
                                cx,
                                cx.listener(|this, _, window, cx| {
                                    this.set_edit_mode(EditMode::OverrideHere, window, cx)
                                }),
                            )),
                    )
                    .child(div().text_xs().text_color(theme.muted_foreground.opacity(0.8)).child(
                        match mode {
                            EditMode::Everywhere => {
                                "Editing the shared participant — changes apply everywhere it is used."
                            }
                            EditMode::OverrideHere => {
                                "Overriding just this space — the shared participant is unchanged."
                            }
                        },
                    )),
            );
        }

        // Label.
        card = card.child(field_label("Name", cx));
        card = card.child(
            div()
                .id("editor-label-wrap")
                .probe("participants/editor/label", gpui::Role::TextInput, "Name")
                .child(Input::new(&edit.label)),
        );

        if is_agent {
            // Model field (button + dropdown).
            card = card.child(field_label("Model", cx));
            card = card.child(self.render_model_field(
                edit.model_ref.as_deref(),
                PickerTarget::Edit,
                cx,
            ));

            // System prompt.
            card = card.child(field_label("System prompt", cx));
            card = card.child(
                div()
                    .id("editor-prompt-wrap")
                    .probe(
                        "participants/editor/system-prompt",
                        gpui::Role::TextInput,
                        "System prompt",
                    )
                    .child(Input::new(&edit.system_prompt)),
            );

            // Notify policy.
            card = card.child(field_label("Responds", cx));
            card = card.child(self.notify_row(&edit.notify_policy, "editor", true, cx));
        }

        card.child(
            h_flex()
                .gap_2()
                .justify_end()
                .child(ghost_button(
                    "editor-cancel".into(),
                    "participants/editor/cancel".into(),
                    "Cancel",
                    false,
                    cx,
                    cx.listener(|this, _, _, cx| this.cancel_edit(cx)),
                ))
                .child(ghost_button(
                    "editor-save".into(),
                    "participants/editor/save".into(),
                    "Save",
                    true,
                    cx,
                    cx.listener(|this, _, _, cx| this.save_edit(cx)),
                )),
        )
        .into_any_element()
    }

    fn render_add_form(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let Some(add) = self.adding.as_ref() else {
            return div().into_any_element();
        };
        v_flex()
            .id("participants-add-form")
            .w_full()
            .p_3()
            .gap_3()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(field_label("Name", cx))
            .child(
                div()
                    .id("add-label-wrap")
                    .probe("participants/add/name", gpui::Role::TextInput, "Name")
                    .child(Input::new(&add.label)),
            )
            .child(field_label("Model", cx))
            .child(self.render_model_field(add.model_ref.as_deref(), PickerTarget::Add, cx))
            .child(field_label("System prompt", cx))
            .child(
                div()
                    .id("add-prompt-wrap")
                    .probe(
                        "participants/add/system-prompt",
                        gpui::Role::TextInput,
                        "System prompt",
                    )
                    .child(Input::new(&add.system_prompt)),
            )
            .child(field_label("Responds", cx))
            .child(self.notify_row(&add.notify_policy, "add", false, cx))
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button(
                        "add-cancel".into(),
                        "participants/add/cancel".into(),
                        "Cancel",
                        false,
                        cx,
                        cx.listener(|this, _, _, cx| this.cancel_add(cx)),
                    ))
                    .child(ghost_button(
                        "add-save".into(),
                        "participants/add/submit".into(),
                        "Add",
                        true,
                        cx,
                        cx.listener(|this, _, _, cx| this.save_add(cx)),
                    )),
            )
            .into_any_element()
    }

    fn render_template_form(
        &self,
        title: &Entity<InputState>,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        v_flex()
            .id("participants-template-form")
            .w_full()
            .mt_2()
            .p_3()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(field_label("Template name", cx))
            .child(
                div()
                    .id("template-title-wrap")
                    .probe(
                        "participants/template/title",
                        gpui::Role::TextInput,
                        "Template name",
                    )
                    .child(Input::new(title)),
            )
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button(
                        "template-cancel".into(),
                        "participants/template/cancel".into(),
                        "Cancel",
                        false,
                        cx,
                        cx.listener(|this, _, _, cx| this.cancel_template(cx)),
                    ))
                    .child(ghost_button(
                        "template-save".into(),
                        "participants/template/save".into(),
                        "Save template",
                        true,
                        cx,
                        cx.listener(|this, _, _, cx| this.save_template(cx)),
                    )),
            )
    }

    /// The model field, delegating to the shared [`model_field`] widget.
    fn render_model_field(
        &self,
        current: Option<&str>,
        target: PickerTarget,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let open = self.picker == Some(target);
        let probe_prefix: SharedString = match target {
            PickerTarget::Edit => "participants/editor/model".into(),
            PickerTarget::Add => "participants/add/model".into(),
        };
        model_field(
            &self.stores,
            current,
            open,
            probe_prefix,
            &self.picker_scroll,
            cx,
            move |this, _, _, cx| this.toggle_picker(target, cx),
            |id, this, cx| this.select_model(id, cx),
        )
    }

    /// A three-option notify-policy row ("Responds: when asked / to people / to
    /// everything").
    fn notify_row(
        &self,
        current: &str,
        scope: &'static str,
        is_edit: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let mut row = h_flex().gap_2();
        for (value, label) in NOTIFY_POLICIES {
            let active = current == value;
            row = row.child(mode_chip(
                SharedString::from(format!("notify-{scope}-{value}")),
                SharedString::from(format!("participants/{scope}/notify/{value}")),
                SharedString::from(label),
                active,
                cx,
                cx.listener(move |this, _, _, cx| {
                    if is_edit {
                        this.set_edit_notify_policy(value, cx);
                    } else {
                        this.set_add_notify_policy(value, cx);
                    }
                }),
            ));
        }
        row
    }
}

pub(crate) fn field_label(label: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_xs()
        .text_color(theme.muted_foreground)
        .child(SharedString::from(label.to_string()))
}

/// A small selectable chip (mode toggle / notify option), styled like the
/// General pane's `choice_chip` — active gets the sidebar-accent pill.
pub(crate) fn mode_chip(
    id: SharedString,
    probe_name: SharedString,
    label: SharedString,
    active: bool,
    cx: &gpui::App,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    let el = div()
        .id(id)
        .probe(probe_name, gpui::Role::Button, label.clone())
        .aria_selected(active)
        .cursor_pointer()
        .px_2()
        .py_0p5()
        .rounded_md()
        .text_sm();
    let el = if active {
        el.bg(theme.sidebar_accent)
            .text_color(theme.sidebar_accent_foreground)
    } else {
        el.text_color(theme.muted_foreground)
            .hover(|s| s.text_color(theme.foreground))
    };
    el.child(label).on_click(on_click)
}

/// A quiet ghost-style action button (Edit / Remove / Cancel / Save …) rendered
/// as a probed styled `div` — `.probe` doesn't apply to gpui-component's
/// `Button`, and this keeps the calm, book-like voice. `primary` fills it with
/// the accent (Save / Add); otherwise it's a quiet ghost.
pub(crate) fn ghost_button(
    id: SharedString,
    probe_name: SharedString,
    label: &'static str,
    primary: bool,
    cx: &gpui::App,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    ghost_button_labeled(id, probe_name, label, label, primary, cx, on_click)
}

/// [`ghost_button`] with the accessible name spelled separately from the
/// visible one — for the repeated verbs (Edit / Remove / …) whose row supplies
/// the subject the label is otherwise missing. Sighted readers get the subject
/// from the row; a screen reader hears five identical "Edit"s without it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ghost_button_labeled(
    id: SharedString,
    probe_name: SharedString,
    label: &'static str,
    aria: impl Into<SharedString>,
    primary: bool,
    cx: &gpui::App,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    let el = div()
        .id(id)
        .probe(probe_name, gpui::Role::Button, aria)
        .cursor_pointer()
        .px_2p5()
        .py_1()
        .rounded_md()
        .text_sm();
    let el = if primary {
        el.bg(theme.primary)
            .text_color(theme.primary_foreground)
            .hover(|s| s.opacity(0.9))
    } else {
        el.text_color(theme.muted_foreground).hover(|s| {
            s.bg(theme.secondary.opacity(0.6))
                .text_color(theme.foreground)
        })
    };
    el.child(label).on_click(on_click)
}

/// A centered "couldn't load — retry" panel for a failed *initial* store load
/// (vs `error_banner`, the inline write-error strip). Shared by the Participants
/// view and the Templates pane so a `Loadable::Failed` never renders as a
/// plausible-empty surface. `retry_probe` is the button's probe name.
pub(crate) fn load_error_panel(
    retry_probe: &'static str,
    headline: &'static str,
    detail: &str,
    cx: &gpui::App,
    on_retry: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let theme = cx.theme();
    v_flex()
        .w_full()
        .py_8()
        .gap_2()
        .items_center()
        .child(
            div()
                .text_sm()
                .text_color(theme.foreground)
                .child(SharedString::from(headline.to_string())),
        )
        .child(
            div()
                .max_w(px(360.))
                .text_center()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(detail.to_string())),
        )
        .child(ghost_button(
            "load-retry".into(),
            SharedString::from(retry_probe),
            "Retry",
            true,
            cx,
            on_retry,
        ))
}

pub(crate) fn error_banner(message: &str, cx: &gpui::App) -> impl IntoElement {
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
