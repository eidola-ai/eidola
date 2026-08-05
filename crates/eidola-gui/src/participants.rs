//! **Participant field helpers** — the shared vocabulary every surface that
//! edits participants speaks.
//!
//! There are three of them: the space inspector's Participants section
//! (`space_view::inspector_participants`, which is where the standalone
//! Participants window went in wave 26.3), the Space Templates settings pane,
//! and the inspector's Space section (for the router picker alone). They share
//! one voice and one set of controls so a model picked in one place reads the
//! same as a model picked in another — and so the two router pickers cannot
//! drift on the thing they must not get wrong (the per-call cost of a remote
//! router).
//!
//! What lives here: the notify-policy vocabulary, the default agent charter,
//! the model/router picker fields (with their grouped catalogs and accessible
//! names), the chips and ghost buttons, and the two error surfaces
//! (`load_error_panel` for a failed read, `error_banner` for a refused write).

use gpui::{
    Context, InteractiveElement, IntoElement, ParentElement, ScrollHandle, SharedString,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder, px,
};
use gpui_component::{ActiveTheme, h_flex, label::Label, v_flex};

use crate::probe::Probe as _;
use crate::stores::Stores;

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

/// The human display for a model selection id: `(model name, backend name)`.
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
/// participant's model field offers exactly what an ask can route to.
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

/// What a model-picker button says it holds, as one spoken line — the same
/// `name · backend` the row renders, or the bare name when there is no backend
/// to qualify (an unset field, the router's "Off").
///
/// This is the picker's **content**, distinct from the label that names it: a
/// picker labelled only "Model" tells a screen reader nothing about which model
/// is selected. It is settled by construction — it moves only when the user
/// picks — which is what makes it safe to ride `aria_value` (see
/// [`crate::probe::Probe::probe_value`]). Shared so the participant field and
/// the Templates pane's router picker cannot drift apart.
pub(crate) fn picker_value(name: &SharedString, backend: Option<&SharedString>) -> SharedString {
    match backend {
        Some(b) => SharedString::from(format!("{name} · {b}")),
        None => name.clone(),
    }
}

/// What one option in a model dropdown is *called*, as distinct from what it
/// shows: the visible row is just the model name because the group header above
/// it supplies the backend — but that header is a role-less `div`, so it never
/// reaches the accessibility tree, and two enabled backends serving the same
/// model name produce two rows a screen reader cannot tell apart. Folding the
/// backend into the name is the `ghost_button_labeled` rule (a repeated verb
/// names its subject), in the [`picker_value`] shape the picker buttons already
/// speak — so choosing a model sounds like what the button then reads back.
pub(crate) fn option_label(display: &SharedString, backend: &SharedString) -> SharedString {
    picker_value(display, Some(backend))
}

/// A model-picker dropdown field shared by the inspector's Participants section
/// and the Templates pane: a button showing the current model, plus (when `open`) a grouped list
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
    let value = picker_value(&name, backend.as_ref());
    let button = h_flex()
        .id(SharedString::from(format!("{probe_prefix}-button")))
        .probe_value(probe_prefix.clone(), gpui::Role::Button, "Model", value)
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
            // The group header is a node-less `div`, so it exists only for the
            // eye — an option's accessible name has to carry the backend itself
            // (see `option_label`).
            let group = SharedString::from(header);
            menu = menu.child(
                div()
                    .px_3()
                    .pt_2()
                    .pb_1()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(group.clone()),
            );
            for (mi, (id, display)) in models.into_iter().enumerate() {
                let selected = current == Some(id.as_str());
                let pick_id = id.clone();
                let on_pick = on_pick.clone();
                let display = SharedString::from(display);
                menu = menu.child(
                    div()
                        .id(SharedString::from(format!("{probe_prefix}-opt-{gi}-{mi}")))
                        .probe(
                            format!("{probe_prefix}/option/{gi}/{mi}"),
                            gpui::Role::Button,
                            option_label(&display, &group),
                        )
                        .px_3()
                        .py_1()
                        .cursor_pointer()
                        .text_sm()
                        .hover(|s| s.bg(theme.secondary.opacity(0.6)))
                        .when(selected, |el| el.text_color(theme.link))
                        .child(display)
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

/// True when a model reference routes to the **remote** eidola backend — the
/// condition (and only condition) for a router row's per-call cost note. The
/// registry answers it; an unloaded registry falls back to the reference's own
/// backend id, which `parse_model_ref` canonicalizes (a bare name is eidola).
pub(crate) fn is_remote_ref(stores: &Stores, selection: &str, cx: &gpui::App) -> bool {
    let backend_id = eidola_app_core::parse_model_ref(selection).backend_id;
    match stores.backends.read(cx).get(&backend_id) {
        Some(b) => b.kind == eidola_app_core::BackendKind::Eidola,
        None => backend_id == eidola_app_core::EIDOLA_BACKEND_ID,
    }
}

/// Where a [`router_field`] renders: the element ids and probe names it mints,
/// the copy it carries, and the scroll handle its dropdown tracks. The two
/// router surfaces (a template's, a space's) differ only in these.
pub(crate) struct RouterField<'a> {
    /// Prefix for element ids (`{id_prefix}-button`, `-menu`, `-opt-*`, `-cost`).
    pub id_prefix: &'static str,
    /// Prefix for probe names (`{probe_prefix}`, `/menu`, `/option/off`,
    /// `/option/{g}/{m}`, `/cost`).
    pub probe_prefix: &'static str,
    /// The selected reference; `None` is **Off** — the default and an ordinary
    /// choice, never a degraded one.
    pub selection: Option<&'a str>,
    pub open: bool,
    /// The mandatory per-call cost copy for a **remote** selection. Its wording
    /// is the surface's (a template's spaces vs this space), the rule is not:
    /// it is always visible whenever a remote reference is chosen.
    pub cost_note: &'static str,
    /// One quiet line saying what the router does, under the row.
    pub help: &'static str,
    pub picker_scroll: &'a ScrollHandle,
    /// Element id for the dropdown's overlay scroll indicator.
    pub scrollbar_id: &'static str,
}

/// The **router-model** picker — the same qualified references the chat model
/// picker offers, with **Off** leading the list (its resting label too), and the
/// per-call cost stated inline under the row whenever a remote reference is
/// selected.
///
/// Shared by the Space Templates pane and the space inspector, because the thing
/// it must not get wrong is the same in both: a remote router bills an inference
/// on every post, and that is disclosed in the row rather than on hover. Only
/// the ids, the probe names, and the wording of the cost line differ (see
/// [`RouterField`]).
pub(crate) fn router_field<V: 'static>(
    stores: &Stores,
    spec: RouterField<'_>,
    cx: &Context<V>,
    on_toggle: impl Fn(&mut V, &gpui::ClickEvent, &mut Window, &mut Context<V>) + 'static,
    on_pick: impl Fn(Option<&str>, &mut V, &mut Context<V>) + Clone + 'static,
) -> gpui::AnyElement {
    let theme = cx.theme();
    let RouterField {
        id_prefix,
        probe_prefix,
        selection,
        open,
        cost_note,
        help,
        picker_scroll,
        scrollbar_id,
    } = spec;
    let (name, backend) = match selection {
        Some(sel) => {
            let (n, b) = model_display(stores, sel, cx);
            (n, Some(b))
        }
        None => ("Off".into(), None),
    };
    // The picker's *content* is which router is chosen — settled (it moves only
    // on a click) and otherwise unreachable to a screen reader, which would hear
    // "Router model" whether the space bills per post or not. Same shape as the
    // participant model field, deliberately.
    let value = picker_value(&name, backend.as_ref());

    let button = h_flex()
        .id(SharedString::from(format!("{id_prefix}-button")))
        .probe_value(probe_prefix, gpui::Role::Button, "Router model", value)
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
        let off_pick = on_pick.clone();
        let mut menu = v_flex()
            .id(SharedString::from(format!("{id_prefix}-menu")))
            .probe(
                format!("{probe_prefix}/menu"),
                gpui::Role::ListBox,
                "Router models",
            )
            .w_full()
            .max_h(px(220.))
            .overflow_y_scroll()
            .track_scroll(picker_scroll)
            .py_1()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.background)
            // Off leads the list: it is the default, and it is a choice.
            .child(
                div()
                    .id(SharedString::from(format!("{id_prefix}-opt-off")))
                    .probe(
                        format!("{probe_prefix}/option/off"),
                        gpui::Role::Button,
                        "Off",
                    )
                    .px_3()
                    .py_1()
                    .cursor_pointer()
                    .text_sm()
                    .hover(|s| s.bg(theme.secondary.opacity(0.6)))
                    .when(selection.is_none(), |el| el.text_color(theme.link))
                    .child("Off")
                    .on_click(cx.listener(move |this, _, _, cx| off_pick(None, this, cx))),
            );
        for (gi, (header, models)) in model_groups(stores, cx).into_iter().enumerate() {
            // The header is node-less chrome; the backend has to ride each
            // option's own name (see `option_label`) — and here it is also the
            // billing difference.
            let group = SharedString::from(header);
            menu = menu.child(
                div()
                    .px_3()
                    .pt_2()
                    .pb_1()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(group.clone()),
            );
            for (mi, (id, display)) in models.into_iter().enumerate() {
                let selected = selection == Some(id.as_str());
                let pick_id = id.clone();
                let on_pick = on_pick.clone();
                let display = SharedString::from(display);
                menu = menu.child(
                    div()
                        .id(SharedString::from(format!("{id_prefix}-opt-{gi}-{mi}")))
                        .probe(
                            format!("{probe_prefix}/option/{gi}/{mi}"),
                            gpui::Role::Button,
                            option_label(&display, &group),
                        )
                        .px_3()
                        .py_1()
                        .cursor_pointer()
                        .text_sm()
                        .hover(|s| s.bg(theme.secondary.opacity(0.6)))
                        .when(selected, |el| el.text_color(theme.link))
                        .child(display)
                        .on_click(
                            cx.listener(move |this, _, _, cx| on_pick(Some(&pick_id), this, cx)),
                        ),
                );
            }
        }
        col = col.child(div().relative().w_full().child(menu).child(
            crate::scrollbar::vertical_floating(scrollbar_id, picker_scroll),
        ));
    }

    // The cost note is exactly remote-conditional, and always visible when it
    // applies (a cost a person only discovers on hover is not disclosed).
    if selection.is_some_and(|sel| is_remote_ref(stores, sel, cx)) {
        col = col.child(
            div()
                .id(SharedString::from(format!("{id_prefix}-cost")))
                .probe(format!("{probe_prefix}/cost"), gpui::Role::Label, cost_note)
                .text_xs()
                .text_color(theme.warning)
                .child(cost_note),
        );
    }

    col.child(
        div()
            .text_xs()
            .text_color(theme.muted_foreground.opacity(0.8))
            .child(help),
    )
    .into_any_element()
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
/// (vs `error_banner`, the inline write-error strip), so a `Loadable::Failed`
/// never renders as a plausible-empty surface. `retry_probe` is the button's probe name.
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
