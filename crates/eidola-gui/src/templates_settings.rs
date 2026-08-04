//! The **Space Templates** settings pane — the registry of templates a new
//! space is born from.
//!
//! A window-local lens over [`crate::stores::TemplatesStore`] (the registry,
//! refreshed on `Change::Templates`). Lists templates; creates / edits (title,
//! cascade limit, and the owned agent participants — the same field set as the
//! per-space Participants view, via the shared `participants_view` helpers);
//! soft-removes; and sets the default (the template `New Space` / ⌘N
//! instantiates, written through `ConfigStore::set_default_template`).
//!
//! Template participant editing works on a **working copy** (`Vec<ParticipantDraft>`)
//! and saves the whole set through `AppCore::update_template` /
//! `create_template` (which replace the template's owned **agent** rows
//! atomically) — templates own those rows outright, so there is no
//! override/referenced fork here (that is a per-space concept). A template's
//! *referenced* globals (`SpaceTemplateInfo::referenced` — the shared "You" a
//! space→template projection carries) are listed **read-only**: they belong to
//! another surface, and no write path here touches them.
//!
//! The **router model** (task 22's may-decline router) is a template setting
//! copied into every space the template instantiates. Its default — `None`,
//! rendered "Off" — is a normal choice, not a degraded one; and because a
//! remote (`eidola`) router bills an inference on *every* post in those spaces,
//! the cost is stated inline under the row whenever a remote reference is
//! selected. It saves through the dedicated
//! `AppCore::set_template_router_model` (composed into `TemplatesStore::create`
//! / `update`), not through the create/update signatures.

use eidola_app_core::{NewTemplateParticipant, SpaceTemplateInfo};
use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement, Render,
    ScrollHandle, SharedString, StatefulInteractiveElement, Styled, Subscription, Window, div,
    prelude::FluentBuilder,
};
use gpui_component::{
    ActiveTheme, StyledExt, h_flex,
    input::{Input, InputState},
    v_flex,
};

use crate::participants_view::{
    DEFAULT_AGENT_SYSTEM_PROMPT, NOTIFY_POLICIES, RouterField, error_banner, field_label,
    ghost_button, ghost_button_labeled, load_error_panel, mode_chip, model_display, model_field,
    notify_label, picker_value, router_field,
};
use crate::probe::Probe as _;
use crate::stores::{ConfigStore, Stores, TemplatesStore};

/// The mandatory cost copy under a router row holding a **remote** reference.
/// A remote router bills an inference per post in every space this template
/// makes; an engine-served one is genuinely free (the zero-spend path). Always
/// visible, never a tooltip.
pub const ROUTER_REMOTE_COST_NOTE: &str = "Every post in spaces from this template is routed through this model, \
     billed per call. Local models route free.";

/// What the router picker does, said once under the row.
const ROUTER_HELP: &str = "A small model that decides which participants a post is worth waking. When off, \
     notifications simply follow each participant's notify setting.";

/// One agent participant being edited inside a template draft.
struct ParticipantDraft {
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    model_ref: Option<String>,
    notify_policy: String,
}

/// Which model picker is open — at most one at a time (they share
/// `picker_scroll`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenPicker {
    Participant(usize),
    Router,
}

/// The in-progress template editor (create or edit).
struct TemplateDraft {
    /// `None` while creating a brand-new template.
    id: Option<String>,
    title: Entity<InputState>,
    cascade_limit: i64,
    /// The may-decline router reference; `None` = Off (the default).
    router_model: Option<String>,
    /// What the router was when the draft opened — so a save writes the setting
    /// only when it actually moved (its setter is separate from
    /// `update_template` and emits its own `Change::Templates`).
    router_original: Option<String>,
    participants: Vec<ParticipantDraft>,
    /// The open model picker, if any.
    picker: Option<OpenPicker>,
}

pub struct TemplatesSettingsView {
    stores: Stores,
    templates_store: Entity<TemplatesStore>,
    config: Entity<ConfigStore>,
    draft: Option<TemplateDraft>,
    /// Tracks the open model-picker dropdown's own scroll (see the identical
    /// field on `ParticipantsView`) — one handle, reset to the top on each
    /// open, since at most one participant's picker is open at a time.
    picker_scroll: ScrollHandle,
    _subscriptions: Vec<Subscription>,
}

impl TemplatesSettingsView {
    pub fn new(stores: Stores, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let templates_store = stores.templates.clone();
        let config = stores.config.clone();
        templates_store.update(cx, |s, cx| s.refresh(cx));
        let mut subs = vec![
            cx.observe(&templates_store, |_, _, cx| cx.notify()),
            cx.observe(&config, |_, _, cx| cx.notify()),
        ];
        subs.push(cx.observe(&stores.backends, |_, _, cx| cx.notify()));
        subs.push(cx.observe(&stores.models, |_, _, cx| cx.notify()));
        subs.push(cx.observe(&stores.local_models, |_, _, cx| cx.notify()));
        Self {
            stores,
            templates_store,
            config,
            draft: None,
            picker_scroll: ScrollHandle::new(),
            _subscriptions: subs,
        }
    }

    fn templates(&self, cx: &gpui::App) -> Vec<SpaceTemplateInfo> {
        self.templates_store.read(cx).list().to_vec()
    }

    fn default_template_id(&self, cx: &gpui::App) -> Option<String> {
        self.config.read(cx).default_template()
    }

    /// Re-fetch the template registry (the Retry affordance on a failed load).
    pub fn retry_load(&mut self, cx: &mut Context<Self>) {
        self.templates_store.update(cx, |s, cx| s.refresh(cx));
        cx.notify();
    }

    // --- Test seams ------------------------------------------------------

    #[doc(hidden)]
    pub fn is_editing(&self) -> bool {
        self.draft.is_some()
    }

    #[doc(hidden)]
    pub fn editing_template_id(&self) -> Option<Option<String>> {
        self.draft.as_ref().map(|d| d.id.clone())
    }

    #[doc(hidden)]
    pub fn draft_title_state(&self) -> Option<Entity<InputState>> {
        self.draft.as_ref().map(|d| d.title.clone())
    }

    #[doc(hidden)]
    pub fn draft_cascade(&self) -> Option<i64> {
        self.draft.as_ref().map(|d| d.cascade_limit)
    }

    #[doc(hidden)]
    pub fn draft_participant_count(&self) -> Option<usize> {
        self.draft.as_ref().map(|d| d.participants.len())
    }

    /// The draft's router reference — outer `None` = not editing, inner `None`
    /// = Off.
    #[doc(hidden)]
    pub fn draft_router_model(&self) -> Option<Option<String>> {
        self.draft.as_ref().map(|d| d.router_model.clone())
    }

    #[doc(hidden)]
    pub fn draft_participant_prompt_state(&self, idx: usize) -> Option<Entity<InputState>> {
        self.draft
            .as_ref()?
            .participants
            .get(idx)
            .map(|p| p.system_prompt.clone())
    }

    #[doc(hidden)]
    pub fn draft_participant_prompt(&self, idx: usize, cx: &gpui::App) -> Option<String> {
        self.draft
            .as_ref()?
            .participants
            .get(idx)
            .map(|p| p.system_prompt.read(cx).value().to_string())
    }

    // --- Editing ---------------------------------------------------------

    fn new_participant_draft(
        &self,
        label: &str,
        model_ref: Option<String>,
        system_prompt: Option<String>,
        notify_policy: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> ParticipantDraft {
        let label_state = cx.new(|cx| InputState::new(window, cx).default_value(label));
        let prompt_state = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(2, 8)
                .placeholder("A short instruction for how this participant behaves.")
                .default_value(system_prompt.unwrap_or_default())
        });
        ParticipantDraft {
            label: label_state,
            system_prompt: prompt_state,
            model_ref,
            notify_policy: notify_policy.to_string(),
        }
    }

    /// A brand-new agent draft: the resolved default model plus the shared
    /// default system prompt, so the field arrives filled rather than blank.
    fn new_agent_draft(&self, window: &mut Window, cx: &mut Context<Self>) -> ParticipantDraft {
        let default_model = self.stores.config.read(cx).default_model();
        self.new_participant_draft(
            "Assistant",
            Some(default_model),
            Some(DEFAULT_AGENT_SYSTEM_PROMPT.to_string()),
            "human",
            window,
            cx,
        )
    }

    pub fn begin_create(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let title = cx.new(|cx| InputState::new(window, cx).placeholder("Template name"));
        // A reveal focuses what it revealed — see `backends_settings`'s
        // `begin_edit_base_url` for why an unfocused reveal strands the reader.
        title.update(cx, |s, cx| s.focus(window, cx));
        let agent = self.new_agent_draft(window, cx);
        self.draft = Some(TemplateDraft {
            id: None,
            title,
            cascade_limit: 4,
            router_model: None,
            router_original: None,
            participants: vec![agent],
            picker: None,
        });
        cx.notify();
    }

    pub fn begin_edit(&mut self, template_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        let Some(t) = self.templates(cx).into_iter().find(|t| t.id == template_id) else {
            return;
        };
        let title = cx.new(|cx| InputState::new(window, cx).default_value(&t.title));
        title.update(cx, |s, cx| s.focus(window, cx));
        let participants = t
            .participants
            .iter()
            .map(|p| {
                self.new_participant_draft(
                    &p.label,
                    p.model_ref.clone(),
                    p.system_prompt.clone(),
                    &p.notify_policy,
                    window,
                    cx,
                )
            })
            .collect();
        self.draft = Some(TemplateDraft {
            id: Some(t.id.clone()),
            title,
            cascade_limit: t.cascade_limit,
            router_model: t.router_model.clone(),
            router_original: t.router_model.clone(),
            participants,
            picker: None,
        });
        cx.notify();
    }

    pub fn cancel_edit(&mut self, cx: &mut Context<Self>) {
        self.draft = None;
        cx.notify();
    }

    pub fn add_participant(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let agent = self.new_agent_draft(window, cx);
        if let Some(draft) = self.draft.as_mut() {
            draft.participants.push(agent);
        }
        cx.notify();
    }

    pub fn remove_participant(&mut self, idx: usize, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut()
            && idx < draft.participants.len()
        {
            draft.participants.remove(idx);
            draft.picker = None;
        }
        cx.notify();
    }

    pub fn set_participant_notify(&mut self, idx: usize, policy: &str, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut()
            && let Some(p) = draft.participants.get_mut(idx)
        {
            p.notify_policy = policy.to_string();
        }
        cx.notify();
    }

    fn toggle_picker(&mut self, target: OpenPicker, cx: &mut Context<Self>) {
        let mut opened = false;
        if let Some(draft) = self.draft.as_mut() {
            draft.picker = if draft.picker == Some(target) {
                None
            } else {
                opened = true;
                Some(target)
            };
        }
        if opened {
            // A freshly opened picker starts at the top.
            self.picker_scroll = ScrollHandle::new();
        }
        cx.notify();
    }

    pub fn toggle_participant_picker(&mut self, idx: usize, cx: &mut Context<Self>) {
        self.toggle_picker(OpenPicker::Participant(idx), cx);
    }

    pub fn toggle_router_picker(&mut self, cx: &mut Context<Self>) {
        self.toggle_picker(OpenPicker::Router, cx);
    }

    pub fn set_participant_model(&mut self, idx: usize, model_id: &str, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            if let Some(p) = draft.participants.get_mut(idx) {
                p.model_ref = Some(model_id.to_string());
            }
            draft.picker = None;
        }
        cx.notify();
    }

    /// Choose the draft's router reference — `None` is **Off**, the default and
    /// an ordinary choice.
    pub fn set_router_model(&mut self, model_id: Option<&str>, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.router_model = model_id.map(str::to_string);
            draft.picker = None;
        }
        cx.notify();
    }

    pub fn cascade_inc(&mut self, delta: i64, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.cascade_limit = (draft.cascade_limit + delta).clamp(1, 99);
        }
        cx.notify();
    }

    pub fn save(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = self.draft.take() else {
            return;
        };
        let title = draft.title.read(cx).value().trim().to_string();
        if title.is_empty() {
            self.draft = Some(draft);
            return;
        }
        let participants: Vec<NewTemplateParticipant> = draft
            .participants
            .iter()
            .map(|p| {
                let label = p.label.read(cx).value().trim().to_string();
                let system_prompt = p.system_prompt.read(cx).value().trim().to_string();
                NewTemplateParticipant {
                    label,
                    model_ref: p.model_ref.clone().filter(|s| !s.is_empty()),
                    system_prompt: if system_prompt.is_empty() {
                        None
                    } else {
                        Some(system_prompt)
                    },
                    notify_policy: p.notify_policy.clone(),
                }
            })
            .collect();
        let cascade_limit = draft.cascade_limit;
        let router = draft.router_model.clone();
        match draft.id {
            Some(id) => {
                // The router has its own setter, so write it only when it moved.
                let router_change = (router != draft.router_original).then_some(router);
                self.templates_store.update(cx, |s, cx| {
                    s.update(
                        id,
                        Some(title),
                        Some(cascade_limit),
                        Some(participants),
                        router_change,
                        cx,
                    )
                })
            }
            None => self.templates_store.update(cx, |s, cx| {
                s.create(title, cascade_limit, participants, router, cx)
            }),
        }
        cx.notify();
    }

    pub fn remove_template(&mut self, id: &str, cx: &mut Context<Self>) {
        self.templates_store
            .update(cx, |s, cx| s.remove(id.to_string(), cx));
        cx.notify();
    }

    pub fn set_default(&mut self, id: &str, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_default_template(id.to_string(), cx));
        cx.notify();
    }
}

impl Render for TemplatesSettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let templates = self.templates(cx);
        let default_id = self.default_template_id(cx);
        let op_error = self.templates_store.read(cx).op_error().map(str::to_string);
        // A failed initial registry load must not read as a plausible-empty
        // registry (Default "missing") — `op_error` only covers writes.
        let (load_error, has_value) = {
            let cell = self.templates_store.read(cx).templates();
            (cell.error().map(|e| e.to_string()), cell.has_value())
        };
        let load_failed_blank = load_error.is_some() && !has_value && self.draft.is_none();

        let mut col = v_flex()
            .id("templates-pane")
            .px_6()
            .py_5()
            .gap_3()
            .w_full()
            .child(
                div()
                    .text_color(theme.muted_foreground)
                    .text_sm()
                    .font_medium()
                    .child("Space Templates"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child(
                        "Templates decide who is in a new space. New Space uses the default; \
                         the Space menu lists the rest.",
                    ),
            );

        if load_failed_blank {
            let err = load_error.clone().unwrap_or_default();
            col = col.child(load_error_panel(
                "settings/templates/retry",
                "Couldn't load your space templates.",
                &err,
                cx,
                cx.listener(|this, _, _, cx| this.retry_load(cx)),
            ));
            // A refused write and a failed re-list are two different truths and
            // can hold at once (the write's error wins `op_error`, the read's
            // resolves the cell — see `stores::settle_mutation`). This branch
            // owns the whole body, so it must carry the write's error too or the
            // refusal would be silent exactly when the listing went blank.
            if let Some(err) = op_error {
                col = col.child(self.render_op_error(&err, cx));
            }
            return col;
        }

        if self.draft.is_some() {
            col = col.child(self.render_editor(cx));
        } else {
            for t in &templates {
                col = col.child(self.render_row(t, default_id.as_deref(), cx));
            }
            col = col.child(
                div()
                    .id("templates-new")
                    .probe("settings/templates/new", gpui::Role::Button, "New template")
                    .mt_1()
                    .cursor_pointer()
                    .text_sm()
                    .text_color(theme.link)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("New template…")
                    .on_click(cx.listener(|this, _, window, cx| this.begin_create(window, cx))),
            );
        }

        if let Some(err) = op_error {
            col = col.child(self.render_op_error(&err, cx));
        }

        // A refresh failure over existing data: keep the list, offer a quiet retry.
        if load_error.is_some() && has_value {
            col = col.child(
                div()
                    .id("templates-retry")
                    .probe("settings/templates/retry", gpui::Role::Button, "Retry")
                    .mt_1()
                    .cursor_pointer()
                    .text_xs()
                    .text_color(theme.link)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("Couldn't refresh — retry")
                    .on_click(cx.listener(|this, _, _, cx| this.retry_load(cx))),
            );
        }

        col
    }
}

impl TemplatesSettingsView {
    /// The write-error banner. One helper because it renders in two places —
    /// under the listing, and under the failed-load panel that owns the body.
    fn render_op_error(&self, err: &str, cx: &Context<Self>) -> impl IntoElement {
        div()
            .id("templates-error")
            .probe(
                "settings/templates/error",
                gpui::Role::Alert,
                err.to_string(),
            )
            .child(error_banner(err, cx))
    }

    fn render_row(
        &self,
        t: &SpaceTemplateInfo,
        default_id: Option<&str>,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let is_default = default_id == Some(t.id.as_str());
        let is_builtin = t.id == eidola_app_core::DEFAULT_TEMPLATE_ID;
        let id = t.id.clone();
        let id_edit = t.id.clone();
        let id_default = t.id.clone();
        let id_remove = t.id.clone();
        let subject = t.title.clone();
        let mut names: Vec<String> = t.referenced.iter().map(|r| r.label.clone()).collect();
        names.extend(t.participants.iter().map(|p| p.label.clone()));
        // A router that bills per post shouldn't be hidden a click deep, so the
        // resting row names it — **model and backend**, in the same
        // `<model> · <backend>` shape the picker uses (one helper, so the two
        // can't drift). The backend is the whole point of naming it at rest:
        // `gemma4-31b@eidola` bills an inference on every post where a
        // same-named on-device model is free, and the model name alone cannot
        // tell those apart. Off says nothing — it is the default.
        let router = t.router_model.as_deref().map(|r| {
            let (name, backend) = model_display(&self.stores, r, cx);
            format!("router {} · ", picker_value(&name, Some(&backend)))
        });
        let summary = SharedString::from(format!(
            "cascade {} · {}{}",
            t.cascade_limit,
            router.unwrap_or_default(),
            if names.is_empty() {
                "no agents".to_string()
            } else {
                names.join(", ")
            }
        ));
        // A settled row: the title names it (and says when it is the one ⌘N
        // uses), the summary line is its content — which is where the router
        // and its backend live, so a screen reader hears the billing
        // difference the sighted reader can see.
        let spoken_name = if is_default {
            SharedString::from(format!("{subject} — the default template"))
        } else {
            SharedString::from(subject.clone())
        };

        h_flex()
            .id(SharedString::from(format!("template-row-{id}")))
            .probe_value(
                format!("settings/templates/{id}"),
                gpui::Role::ListItem,
                spoken_name,
                summary.clone(),
            )
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
                            .child(
                                div()
                                    .font_medium()
                                    .child(SharedString::from(t.title.clone())),
                            )
                            .when(is_default, |el| {
                                el.child(
                                    div()
                                        .px_1p5()
                                        .rounded_sm()
                                        .bg(theme.sidebar_accent)
                                        .text_xs()
                                        .text_color(theme.sidebar_accent_foreground)
                                        .child("default"),
                                )
                            }),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(summary),
                    ),
            )
            .child(
                h_flex()
                    .gap_1()
                    .flex_wrap()
                    .justify_end()
                    .when(!is_default, |el| {
                        el.child(ghost_button_labeled(
                            SharedString::from(format!("tmpl-default-{id_default}")),
                            SharedString::from(format!(
                                "settings/templates/{id_default}/set-default"
                            )),
                            "Set as default",
                            format!("Set {subject} as the default template"),
                            false,
                            cx,
                            cx.listener(move |this, _, _, cx| this.set_default(&id_default, cx)),
                        ))
                    })
                    .child(ghost_button_labeled(
                        SharedString::from(format!("tmpl-edit-{id_edit}")),
                        SharedString::from(format!("settings/templates/{id_edit}/edit")),
                        "Edit",
                        format!("Edit {subject}"),
                        false,
                        cx,
                        cx.listener(move |this, _, window, cx| {
                            this.begin_edit(&id_edit, window, cx)
                        }),
                    ))
                    .when(!is_builtin, |el| {
                        el.child(ghost_button_labeled(
                            SharedString::from(format!("tmpl-remove-{id_remove}")),
                            SharedString::from(format!("settings/templates/{id_remove}/remove")),
                            "Remove",
                            format!("Remove {subject}"),
                            false,
                            cx,
                            cx.listener(move |this, _, _, cx| this.remove_template(&id_remove, cx)),
                        ))
                    }),
            )
    }

    fn render_editor(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let Some(draft) = self.draft.as_ref() else {
            return div().into_any_element();
        };

        let mut card = v_flex()
            .id("template-editor")
            .w_full()
            .p_3()
            .gap_3()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(field_label("Template name", cx))
            .child(
                div()
                    .id("template-title-wrap")
                    .probe_bounds(
                        "settings/templates/editor/title",
                        gpui::Role::TextInput,
                        "Template name",
                    )
                    .child(Input::new(&draft.title).aria_label("Template name")),
            );

        // Cascade limit — a small +/- stepper.
        card = card.child(field_label("Cascade limit", cx)).child(
            h_flex()
                .gap_2()
                .items_center()
                .child(ghost_button_labeled(
                    "cascade-dec".into(),
                    "settings/templates/editor/cascade/dec".into(),
                    "−",
                    "Decrease cascade limit",
                    false,
                    cx,
                    cx.listener(|this, _, _, cx| this.cascade_inc(-1, cx)),
                ))
                .child(
                    div()
                        .min_w(gpui::px(24.))
                        .text_center()
                        .text_sm()
                        .child(SharedString::from(draft.cascade_limit.to_string())),
                )
                .child(ghost_button_labeled(
                    "cascade-inc".into(),
                    "settings/templates/editor/cascade/inc".into(),
                    "+",
                    "Increase cascade limit",
                    false,
                    cx,
                    cx.listener(|this, _, _, cx| this.cascade_inc(1, cx)),
                ))
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .child("How many agent replies in a row before a space pauses."),
                ),
        );

        // Router model — Off by default, with the per-call cost stated inline
        // whenever a remote reference is chosen.
        card = card
            .child(field_label("Router model", cx))
            .child(self.render_router_field(draft, cx));

        // Participants. The referenced globals render from the **live store
        // snapshot** (see `editor_referenced`), the owned agents from the draft.
        card = card.child(field_label("Participants", cx));
        for (idx, r) in self.editor_referenced(draft, cx).iter().enumerate() {
            card = card.child(self.render_referenced(idx, r, cx));
        }
        for (idx, p) in draft.participants.iter().enumerate() {
            card = card.child(self.render_participant(
                idx,
                p,
                draft.picker == Some(OpenPicker::Participant(idx)),
                cx,
            ));
        }
        card = card.child(
            div()
                .id("template-add-participant")
                .probe(
                    "settings/templates/editor/add-participant",
                    gpui::Role::Button,
                    "Add participant",
                )
                .cursor_pointer()
                .text_sm()
                .text_color(theme.link)
                .hover(|s| s.text_color(theme.foreground))
                .child("Add participant…")
                .on_click(cx.listener(|this, _, window, cx| this.add_participant(window, cx))),
        );

        card.child(
            h_flex()
                .gap_2()
                .justify_end()
                .child(ghost_button(
                    "template-editor-cancel".into(),
                    "settings/templates/editor/cancel".into(),
                    "Cancel",
                    false,
                    cx,
                    cx.listener(|this, _, _, cx| this.cancel_edit(cx)),
                ))
                .child(ghost_button(
                    "template-editor-save".into(),
                    "settings/templates/editor/save".into(),
                    "Save template",
                    true,
                    cx,
                    cx.listener(|this, _, _, cx| this.save(cx)),
                )),
        )
        .into_any_element()
    }

    /// The router-model row — the shared [`router_field`] with this surface's
    /// ids, probes, and cost wording (a template's router bills in *every*
    /// space it makes).
    fn render_router_field(&self, draft: &TemplateDraft, cx: &Context<Self>) -> gpui::AnyElement {
        router_field(
            &self.stores,
            RouterField {
                id_prefix: "template-router",
                probe_prefix: "settings/templates/editor/router",
                selection: draft.router_model.as_deref(),
                open: draft.picker == Some(OpenPicker::Router),
                cost_note: ROUTER_REMOTE_COST_NOTE,
                help: ROUTER_HELP,
                picker_scroll: &self.picker_scroll,
                scrollbar_id: "router-picker-scrollbar",
            },
            cx,
            |this, _, _, cx| this.toggle_router_picker(cx),
            |id, this: &mut Self, cx| this.set_router_model(id, cx),
        )
    }

    /// The referenced globals the open editor lists, resolved against the
    /// **live registry** rather than carried in the draft.
    ///
    /// A draft holds only what the editor edits, and these rows are read-only
    /// display of config that lives elsewhere and is shared everywhere it is
    /// referenced — so an "edit everywhere" from the Participants window (which
    /// emits `Change::Participants`, routed to this store) must be visible under
    /// an editor that is already open. A clone taken at `begin_edit` would keep
    /// showing the old label and prompt for as long as the editor stayed open.
    /// A create draft has no template id and therefore no referenced rows.
    fn editor_referenced(
        &self,
        draft: &TemplateDraft,
        cx: &gpui::App,
    ) -> Vec<eidola_app_core::TemplateReferencedParticipant> {
        let Some(id) = draft.id.as_deref() else {
            return Vec::new();
        };
        self.templates_store
            .read(cx)
            .list()
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.referenced.clone())
            .unwrap_or_default()
    }

    /// A global this template references — listed so it isn't invisible, and
    /// read-only because it is another surface's row (shared everywhere it is
    /// referenced; no write path here touches it).
    fn render_referenced(
        &self,
        idx: usize,
        r: &eidola_app_core::TemplateReferencedParticipant,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let detail = match r.model_ref.as_deref() {
            Some(m) => {
                let (name, backend) = model_display(&self.stores, m, cx);
                format!(
                    "{name} · {backend} · responds {}",
                    notify_label(&r.notify_policy)
                )
            }
            None => format!("Responds {}", notify_label(&r.notify_policy)),
        };
        // Its effective system prompt is real config a template can carry (a
        // space→template projection preserves a per-membership override), so it
        // is shown rather than silently dropped — read-only like the rest of the
        // row, and part of the spoken content.
        let prompt = r
            .system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let spoken = match &prompt {
            Some(p) => format!("{detail} · {p}"),
            None => detail.clone(),
        };
        h_flex()
            .id(SharedString::from(format!("template-referenced-{idx}")))
            // A settled read-only row: the name labels it, the lines under it
            // are its content.
            .probe_value(
                format!("settings/templates/editor/referenced/{idx}"),
                gpui::Role::Label,
                format!("{} — shared participant", r.label),
                spoken,
            )
            .w_full()
            .p_2p5()
            .gap_2()
            .items_start()
            .justify_between()
            .rounded_md()
            .border_1()
            .border_color(theme.border.opacity(0.4))
            .child(
                v_flex()
                    .gap_0p5()
                    .min_w_0()
                    .child(div().text_sm().child(SharedString::from(r.label.clone())))
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(detail)),
                    )
                    .when_some(prompt, |el, p| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground.opacity(0.8))
                                .child(SharedString::from(p)),
                        )
                    }),
            )
            .child(
                div()
                    .px_1p5()
                    .rounded_sm()
                    .bg(theme.secondary)
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child("shared"),
            )
    }

    fn render_participant(
        &self,
        idx: usize,
        p: &ParticipantDraft,
        picker_open: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        v_flex()
            .id(SharedString::from(format!("template-participant-{idx}")))
            .w_full()
            .p_2p5()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border.opacity(0.6))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .id(SharedString::from(format!("tp-label-wrap-{idx}")))
                            .flex_1()
                            .probe_bounds(
                                format!("settings/templates/participant/{idx}/name"),
                                gpui::Role::TextInput,
                                "Name",
                            )
                            .child(Input::new(&p.label).aria_label("Name")),
                    )
                    .child(ghost_button_labeled(
                        SharedString::from(format!("tp-remove-{idx}")),
                        SharedString::from(format!("settings/templates/participant/{idx}/remove")),
                        "Remove",
                        format!("Remove participant {}", idx + 1),
                        false,
                        cx,
                        cx.listener(move |this, _, _, cx| this.remove_participant(idx, cx)),
                    )),
            )
            .child(model_field(
                &self.stores,
                p.model_ref.as_deref(),
                picker_open,
                SharedString::from(format!("settings/templates/participant/{idx}/model")),
                &self.picker_scroll,
                cx,
                move |this, _, _, cx| this.toggle_participant_picker(idx, cx),
                move |id, this, cx| this.set_participant_model(idx, id, cx),
            ))
            .child(
                div()
                    .id(SharedString::from(format!("tp-prompt-wrap-{idx}")))
                    .probe_bounds(
                        format!("settings/templates/participant/{idx}/system-prompt"),
                        gpui::Role::TextInput,
                        "System prompt",
                    )
                    .child(Input::new(&p.system_prompt).aria_label("System prompt")),
            )
            .child({
                let mut row = h_flex().gap_2().items_center().child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child("Responds"),
                );
                for (value, label) in NOTIFY_POLICIES {
                    let active = p.notify_policy == value;
                    row = row.child(mode_chip(
                        SharedString::from(format!("tp-notify-{idx}-{value}")),
                        SharedString::from(format!(
                            "settings/templates/participant/{idx}/notify/{value}"
                        )),
                        SharedString::from(label),
                        active,
                        cx,
                        cx.listener(move |this, _, _, cx| {
                            this.set_participant_notify(idx, value, cx)
                        }),
                    ));
                }
                row
            })
    }
}
