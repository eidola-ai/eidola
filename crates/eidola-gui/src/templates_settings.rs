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
//! `create_template` (which replace the template's owned agents atomically) —
//! templates own their participant rows outright, so there is no
//! override/referenced fork here (that is a per-space concept).

use eidola_app_core::{NewTemplateParticipant, SpaceTemplateInfo};
use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Subscription, Window, div,
    prelude::FluentBuilder,
};
use gpui_component::{
    ActiveTheme, StyledExt, h_flex,
    input::{Input, InputState},
    v_flex,
};

use crate::participants_view::{
    NOTIFY_POLICIES, error_banner, field_label, ghost_button, load_error_panel, mode_chip,
    model_field,
};
use crate::probe::Probe as _;
use crate::stores::{ConfigStore, Stores, TemplatesStore};

/// One agent participant being edited inside a template draft.
struct ParticipantDraft {
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    model_ref: Option<String>,
    notify_policy: String,
}

/// The in-progress template editor (create or edit).
struct TemplateDraft {
    /// `None` while creating a brand-new template.
    id: Option<String>,
    title: Entity<InputState>,
    cascade_limit: i64,
    participants: Vec<ParticipantDraft>,
    /// The participant index whose model picker is open, if any.
    picker: Option<usize>,
}

pub struct TemplatesSettingsView {
    stores: Stores,
    templates_store: Entity<TemplatesStore>,
    config: Entity<ConfigStore>,
    draft: Option<TemplateDraft>,
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

    pub fn begin_create(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let title = cx.new(|cx| InputState::new(window, cx).placeholder("Template name"));
        let default_model = self.stores.config.read(cx).default_model();
        let agent =
            self.new_participant_draft("Assistant", Some(default_model), None, "human", window, cx);
        self.draft = Some(TemplateDraft {
            id: None,
            title,
            cascade_limit: 4,
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
        let default_model = self.stores.config.read(cx).default_model();
        let agent =
            self.new_participant_draft("Assistant", Some(default_model), None, "human", window, cx);
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

    pub fn toggle_participant_picker(&mut self, idx: usize, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.picker = if draft.picker == Some(idx) {
                None
            } else {
                Some(idx)
            };
        }
        cx.notify();
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
        match draft.id {
            Some(id) => self.templates_store.update(cx, |s, cx| {
                s.update(id, Some(title), Some(cascade_limit), Some(participants), cx)
            }),
            None => self
                .templates_store
                .update(cx, |s, cx| s.create(title, cascade_limit, participants, cx)),
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
            col = col.child(
                div()
                    .id("templates-error")
                    .probe("settings/templates/error", gpui::Role::Alert, err.clone())
                    .child(error_banner(&err, cx)),
            );
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
        let names: Vec<String> = t.participants.iter().map(|p| p.label.clone()).collect();
        let summary = format!(
            "cascade {} · {}",
            t.cascade_limit,
            if names.is_empty() {
                "no agents".to_string()
            } else {
                names.join(", ")
            }
        );

        h_flex()
            .id(SharedString::from(format!("template-row-{id}")))
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
                            .child(SharedString::from(summary)),
                    ),
            )
            .child(
                h_flex()
                    .gap_1()
                    .flex_wrap()
                    .justify_end()
                    .when(!is_default, |el| {
                        el.child(ghost_button(
                            SharedString::from(format!("tmpl-default-{id_default}")),
                            SharedString::from(format!(
                                "settings/templates/{id_default}/set-default"
                            )),
                            "Set as default",
                            false,
                            cx,
                            cx.listener(move |this, _, _, cx| this.set_default(&id_default, cx)),
                        ))
                    })
                    .child(ghost_button(
                        SharedString::from(format!("tmpl-edit-{id_edit}")),
                        SharedString::from(format!("settings/templates/{id_edit}/edit")),
                        "Edit",
                        false,
                        cx,
                        cx.listener(move |this, _, window, cx| {
                            this.begin_edit(&id_edit, window, cx)
                        }),
                    ))
                    .when(!is_builtin, |el| {
                        el.child(ghost_button(
                            SharedString::from(format!("tmpl-remove-{id_remove}")),
                            SharedString::from(format!("settings/templates/{id_remove}/remove")),
                            "Remove",
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
                    .probe(
                        "settings/templates/editor/title",
                        gpui::Role::TextInput,
                        "Template name",
                    )
                    .child(Input::new(&draft.title)),
            );

        // Cascade limit — a small +/- stepper.
        card = card.child(field_label("Cascade limit", cx)).child(
            h_flex()
                .gap_2()
                .items_center()
                .child(ghost_button(
                    "cascade-dec".into(),
                    "settings/templates/editor/cascade/dec".into(),
                    "−",
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
                .child(ghost_button(
                    "cascade-inc".into(),
                    "settings/templates/editor/cascade/inc".into(),
                    "+",
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

        // Participants.
        card = card.child(field_label("Participants", cx));
        for (idx, p) in draft.participants.iter().enumerate() {
            card = card.child(self.render_participant(idx, p, draft.picker == Some(idx), cx));
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
                            .probe(
                                format!("settings/templates/participant/{idx}/name"),
                                gpui::Role::TextInput,
                                "Name",
                            )
                            .child(Input::new(&p.label)),
                    )
                    .child(ghost_button(
                        SharedString::from(format!("tp-remove-{idx}")),
                        SharedString::from(format!("settings/templates/participant/{idx}/remove")),
                        "Remove",
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
                cx,
                move |this, _, _, cx| this.toggle_participant_picker(idx, cx),
                move |id, this, cx| this.set_participant_model(idx, id, cx),
            ))
            .child(
                div()
                    .id(SharedString::from(format!("tp-prompt-wrap-{idx}")))
                    .probe(
                        format!("settings/templates/participant/{idx}/system-prompt"),
                        gpui::Role::TextInput,
                        "System prompt",
                    )
                    .child(Input::new(&p.system_prompt)),
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
