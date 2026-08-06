//! The **Agents** settings pane — the shared agent library (task 36).
//!
//! A window-local lens over [`crate::stores::AgentsStore`] (global agents,
//! refreshed on `Change::Participants`). It is app-level configuration rather
//! than a space's: a shared agent is one identity across every conversation it
//! joins, so its name, model, charter and notify default belong beside
//! Templates, not inside any one space. What a *space* decides about it — the
//! per-membership override — stays in that space's inspector, which is the same
//! division the two write paths already make (`update_space_participant` here,
//! `set_space_participant_override` there).
//!
//! **The roster is read-only until you open a row.** Editing works on a working
//! copy (`AgentDraft`) saved whole through `AgentsStore::update_agent` — the
//! Templates pane's idiom, with the same shared field helpers
//! (`crate::participants`), so a model picked here reads exactly as one picked
//! in a space.
//!
//! Two verbs exist only here, because only the library can offer them:
//!
//! * **Open notebook** — the agent's private space, where its core memory
//!   blocks live. A real space, so it opens through the ordinary
//!   `open_space_window`; it is simply not in the Library listing.
//! * **Retire** — the soft-remove, behind a deliberate confirm because it also
//!   archives that notebook. It is **not** a demotion (there is none): the
//!   agent's posts, authorship and memory all stand, and nothing here says
//!   otherwise.

use eidola_app_core::{GlobalAgentInfo, ParticipantUpdate};
use gpui::{
    AppContext, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement,
    Render, ScrollHandle, SharedString, StatefulInteractiveElement, Styled, Subscription, Window,
    div, prelude::FluentBuilder,
};
use gpui_component::{
    ActiveTheme, StyledExt as _, h_flex,
    input::{Input, InputState},
    v_flex,
};

use crate::participants::{
    NOTIFY_POLICIES, error_banner, field_label, ghost_button, ghost_button_labeled,
    load_error_panel, mode_chip, model_display, model_field, notify_label,
};
use crate::probe::Probe as _;
use crate::stores::{AgentsStore, Stores};

/// The placeholder the charter field carries, so an empty one reads as a choice
/// rather than a missing value. (Same wording as the inspector's.)
const PROMPT_PLACEHOLDER: &str = "A short instruction for how this participant behaves.";

/// What retirement does, said in full before it is done — it reaches past the
/// library into the agent's notebook, and there is no un-retire.
const RETIRE_NOTE: &str = "Retiring takes this agent out of the library and archives its notebook. \
     What it has written stays where it is, and this can't be undone.";

/// The in-progress edit of one shared agent — a working copy, saved whole.
struct AgentDraft {
    participant_id: String,
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    model_ref: Option<String>,
    notify_policy: String,
    /// The editor subtree's own focus handle — what the handback question is
    /// asked of. **Containment, not an enumeration of fields**: the editor holds
    /// more than its two inputs, and a predicate that lists them answers "not
    /// held" for anything else focused inside it, so the keyboard is dropped on
    /// the floor by exactly the controls a future edit adds (Codex review, PR
    /// #279).
    focus: gpui::FocusHandle,
    /// Whether the model dropdown is open. It is painted **inside this draft**,
    /// so it dies with the draft — the inspector's derive-don't-clear rule, made
    /// structural by ownership rather than by a reader.
    picker_open: bool,
}

pub struct AgentsSettingsView {
    stores: Stores,
    agents_store: Entity<AgentsStore>,
    draft: Option<AgentDraft>,
    /// The agent whose retirement is armed, if any. Like the draft, it names a
    /// participant rather than an index: the roster can move under it (another
    /// window retires something, an edit lands), and an armed irreversible verb
    /// must never re-aim at whoever took that row's place.
    retiring: Option<String>,
    /// The open dropdown's own scroll (reset to the top on each open).
    picker_scroll: ScrollHandle,
    /// The retire confirmation's subtree focus handle. One handle, because one
    /// confirmation renders at a time — and its two buttons are real tab stops
    /// (`probe(Role::Button)` derives `focusable()` + `tab_index(0)`), so
    /// unmounting them without a handback strands the keyboard exactly as an
    /// editor would.
    retire_focus: FocusHandle,
    /// Where the keyboard goes when an editor holding it closes — the pane's
    /// own root, tracked on the element below, so Tab resumes from the roster
    /// rather than from the window.
    focus_handle: FocusHandle,
    /// How many notebook windows this pane has asked for — the behavior tests'
    /// seam, since the open is deferred and stub tests have no window to
    /// inspect (the Library's `open_space_requests` counter, same reason).
    notebooks_opened: usize,
    _subscriptions: Vec<Subscription>,
}

impl AgentsSettingsView {
    pub fn new(stores: Stores, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let agents_store = stores.agents.clone();
        agents_store.update(cx, |s, cx| s.refresh(cx));
        // The roster's own store, plus the three the model picker reads
        // (`model_groups` / `model_display`) — each lands asynchronously and
        // alone, and an unobserved one leaves a row naming a bare slug until
        // something unrelated repaints the pane.
        let subscriptions = vec![
            cx.observe(&agents_store, |_, _, cx| cx.notify()),
            cx.observe(&stores.backends, |_, _, cx| cx.notify()),
            cx.observe(&stores.models, |_, _, cx| cx.notify()),
            cx.observe(&stores.local_models, |_, _, cx| cx.notify()),
        ];
        Self {
            stores,
            agents_store,
            draft: None,
            retiring: None,
            picker_scroll: ScrollHandle::new(),
            focus_handle: cx.focus_handle(),
            retire_focus: cx.focus_handle(),
            notebooks_opened: 0,
            _subscriptions: subscriptions,
        }
    }

    fn agents(&self, cx: &gpui::App) -> Vec<GlobalAgentInfo> {
        self.agents_store.read(cx).list().to_vec()
    }

    // -- Test seams --------------------------------------------------------

    /// The pane's own focus target — what an editor hands the keyboard back to.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    #[doc(hidden)]
    pub fn editing_agent(&self) -> Option<&str> {
        self.draft.as_ref().map(|d| d.participant_id.as_str())
    }

    /// The armed confirmation's subtree focus handle (tests).
    #[doc(hidden)]
    pub fn retire_focus_handle(&self) -> FocusHandle {
        self.retire_focus.clone()
    }

    #[doc(hidden)]
    pub fn retiring_agent(&self) -> Option<&str> {
        self.retiring.as_deref()
    }

    /// The open editor's subtree focus handle (tests: focus something inside
    /// the editor that is not one of its two inputs).
    #[doc(hidden)]
    pub fn editing_focus_handle(&self) -> Option<FocusHandle> {
        self.draft.as_ref().map(|d| d.focus.clone())
    }

    #[doc(hidden)]
    pub fn editing_label_state(&self) -> Option<Entity<InputState>> {
        self.draft.as_ref().map(|d| d.label.clone())
    }

    #[doc(hidden)]
    pub fn editing_prompt_state(&self) -> Option<Entity<InputState>> {
        self.draft.as_ref().map(|d| d.system_prompt.clone())
    }

    #[doc(hidden)]
    pub fn picker_open_for_test(&self) -> bool {
        self.draft.as_ref().is_some_and(|d| d.picker_open)
    }

    #[doc(hidden)]
    pub fn notebooks_opened_for_test(&self) -> usize {
        self.notebooks_opened
    }

    // -- Editing -----------------------------------------------------------

    /// Open (or close) one agent's editor, seeded from the shared row.
    pub fn toggle_edit(
        &mut self,
        participant_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.editing_agent() == Some(participant_id) {
            self.cancel_edit(window, cx);
            return;
        }
        let Some(agent) = self.agents(cx).into_iter().find(|a| a.id == participant_id) else {
            return;
        };
        self.retiring = None;
        let label = cx.new(|cx| InputState::new(window, cx).default_value(&agent.label));
        // A reveal focuses what it revealed (the Settings idiom's rule).
        label.update(cx, |s, cx| s.focus(window, cx));
        let system_prompt = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(2, 8)
                .placeholder(PROMPT_PLACEHOLDER)
                .default_value(agent.system_prompt.clone().unwrap_or_default())
        });
        self.draft = Some(AgentDraft {
            participant_id: agent.id.clone(),
            label,
            system_prompt,
            focus: cx.focus_handle(),
            model_ref: agent.model_ref.clone(),
            notify_policy: agent.notify_policy.clone(),
            picker_open: false,
        });
        cx.notify();
    }

    /// Hand the keyboard back to the pane's root when a form that **was**
    /// holding it has gone and left nothing focused.
    ///
    /// `held` is an observation taken *before* the form was dropped, and it has
    /// to be: the question is asked of the form's own `InputState`s, which die
    /// with it. Restoring only from a lender still holding it is
    /// `set_inspector_open`'s rule — a form that never had the keyboard must not
    /// take it from whatever does.
    fn hand_back_focus(&mut self, held: bool, window: &mut Window, cx: &mut Context<Self>) {
        if held && !self.draft_field_focused(window, cx) {
            window.focus(&self.focus_handle, cx);
        }
    }

    /// Whether the keyboard is anywhere **inside** one of the pane's two open
    /// forms — the editor or the retire confirmation. The question the handback
    /// has to ask, since each is a subtree and not a list of inputs
    /// (`contains_focused` is the same idiom the Library's reveal uses).
    fn draft_field_focused(&self, window: &Window, cx: &gpui::App) -> bool {
        self.draft
            .as_ref()
            .is_some_and(|d| d.focus.contains_focused(window, cx))
            || (self.retiring.is_some() && self.retire_focus.contains_focused(window, cx))
    }

    pub fn cancel_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.draft_field_focused(window, cx);
        self.draft = None;
        self.hand_back_focus(held, window, cx);
        cx.notify();
    }

    pub fn set_draft_notify(&mut self, policy: &str, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.notify_policy = policy.to_string();
            cx.notify();
        }
    }

    pub fn toggle_picker(&mut self, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.picker_open = !draft.picker_open;
            if draft.picker_open {
                // A freshly opened picker starts at the top.
                self.picker_scroll = ScrollHandle::new();
            }
            cx.notify();
        }
    }

    pub fn select_model(&mut self, model_id: &str, cx: &mut Context<Self>) {
        if let Some(draft) = self.draft.as_mut() {
            draft.model_ref = Some(model_id.to_string());
            draft.picker_open = false;
            cx.notify();
        }
    }

    /// Commit the open editor — an **edit everywhere**: this is the agent's own
    /// config, which every space that has not overridden the field follows.
    pub fn save_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.draft_field_focused(window, cx);
        let Some(draft) = self.draft.take() else {
            return;
        };
        let label = draft.label.read(cx).value().trim().to_string();
        if label.is_empty() {
            // Keep the editor open; app-core would refuse an empty label, and
            // not sending saves the round trip.
            self.draft = Some(draft);
            return;
        }
        let system_prompt = draft.system_prompt.read(cx).value().trim().to_string();
        let update = ParticipantUpdate {
            label: Some(label),
            model_ref: Some(draft.model_ref.clone().filter(|s| !s.is_empty())),
            system_prompt: Some((!system_prompt.is_empty()).then_some(system_prompt)),
            notify_policy: Some(draft.notify_policy.clone()),
        };
        self.agents_store
            .update(cx, |s, cx| s.update_agent(draft.participant_id, update, cx));
        self.hand_back_focus(held, window, cx);
        cx.notify();
    }

    // -- Retirement --------------------------------------------------------

    pub fn arm_retire(
        &mut self,
        participant_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let held = self.draft_field_focused(window, cx);
        self.draft = None;
        self.retiring = Some(participant_id.to_string());
        self.hand_back_focus(held, window, cx);
        cx.notify();
    }

    /// "Keep" — and the button that was pressed unmounts with the
    /// confirmation, so the keyboard comes back the same way an editor's does.
    pub fn cancel_retire(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.draft_field_focused(window, cx);
        self.retiring = None;
        self.hand_back_focus(held, window, cx);
        cx.notify();
    }

    pub fn confirm_retire(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.draft_field_focused(window, cx);
        let Some(id) = self.retiring.take() else {
            return;
        };
        self.agents_store.update(cx, |s, cx| s.retire(id, cx));
        self.hand_back_focus(held, window, cx);
        cx.notify();
    }

    // -- The notebook door -------------------------------------------------

    /// Open an agent's notebook. It is an ordinary space — hidden from the
    /// Library, reached from here — so it goes through the same
    /// `open_space_window` every other space does, deferred like the Library's
    /// row-open so the window opens after this update cycle.
    pub fn open_notebook(&mut self, space_id: String, cx: &mut Context<Self>) {
        self.notebooks_opened += 1;
        let stores = self.stores.clone();
        cx.defer(move |cx: &mut gpui::App| {
            crate::open_space_window(cx, stores, space_id);
        });
    }

    pub fn retry_load(&mut self, cx: &mut Context<Self>) {
        self.agents_store.update(cx, |s, cx| s.refresh(cx));
        cx.notify();
    }
}

impl Render for AgentsSettingsView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // The roster can move under an open editor or an armed retirement
        // (another window shares or retires an agent; a bus refresh lands), and
        // neither may go on describing a row that has left. Reconciled at the
        // head of `render` for the same reason the inspector's editor is: the
        // unmount is not a verb anyone pressed.
        self.sync_open_forms(window, cx);

        let theme = cx.theme();
        let agents = self.agents(cx);
        // Refusals are keyed per agent and render under their own rows — except
        // in the failed-load branch below, which owns the whole body and would
        // otherwise swallow them.
        let standing_errors: Vec<(GlobalAgentInfo, String)> = {
            let store = self.agents_store.read(cx);
            agents
                .iter()
                .filter_map(|a| store.op_error(&a.id).map(|e| (a.clone(), e.to_string())))
                .collect()
        };
        let (load_error, has_value) = {
            let cell = self.agents_store.read(cx).agents();
            (cell.error().map(|e| e.to_string()), cell.has_value())
        };

        let mut col = v_flex()
            .id("agents-pane")
            .track_focus(&self.focus_handle)
            .px_6()
            .py_5()
            .gap_3()
            .w_full()
            .child(
                div()
                    .text_color(theme.muted_foreground)
                    .text_sm()
                    .font_medium()
                    .child("Agents"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child(
                        "Agents you have shared are one colleague across every space they join. \
                         What you set here they carry everywhere; a space can still override it \
                         for itself.",
                    ),
            );

        // "Failed is not empty": a failed *initial* read must not render as an
        // empty library with a plausible "share one from a space" invitation —
        // and `refresh` is the only way back once a `Failed` cell exists.
        if load_error.is_some() && !has_value {
            let err = load_error.clone().unwrap_or_default();
            col = col.child(load_error_panel(
                "settings/agents/retry",
                "Couldn't load your shared agents.",
                &err,
                cx,
                cx.listener(|this, _, _, cx| this.retry_load(cx)),
            ));
            // A refused write and a failed re-list are two truths that can hold
            // at once; this branch owns the body, so it carries both.
            for (agent, err) in &standing_errors {
                col = col.child(self.render_op_error(agent, err, cx));
            }
            return col;
        }

        // **A listing that has not answered is not an empty library.** `Loadable`'s
        // rule (`crates/eidola-gui/STATE.md`): `NotLoaded`/`Loading` with no
        // value says nothing, and rendering the "share one from a space"
        // invitation over it tells a cold-opening reader their library is empty
        // when it is merely unread (Codex review, PR #279). The quiet one-line
        // readout is the Participants section's idiom, matched here.
        if !has_value {
            return col.child(
                div()
                    .id("agents-loading")
                    .probe("settings/agents/loading", gpui::Role::Label, "Loading…")
                    .py_4()
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child("Loading…"),
            );
        }

        if agents.is_empty() {
            col = col.child(
                div()
                    .id("agents-empty")
                    .probe(
                        "settings/agents/empty",
                        gpui::Role::Label,
                        "No shared agents yet",
                    )
                    .py_4()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    // The invitation names the actual door, since there is no
                    // "new agent" here: an agent becomes shared by being shared
                    // *from a conversation it already works in*.
                    .child(
                        "No shared agents yet. Open a space's inspector and choose \
                         “Share this agent…” to give one a life beyond that conversation.",
                    ),
            );
        }

        for agent in &agents {
            col = col.child(self.render_row(agent, cx));
        }

        // A refresh failure over rows we still hold: keep them, say so quietly.
        if load_error.is_some() && has_value {
            col = col.child(
                div()
                    .id("agents-retry")
                    .probe("settings/agents/retry", gpui::Role::Button, "Retry")
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

impl AgentsSettingsView {
    /// Retire an open editor or an armed retirement whose agent has left the
    /// roster. Only a listing that has *answered* can say a row is gone — a load
    /// in flight, or a failed one with nothing prior, knows nothing.
    ///
    /// **Here presence is the whole question**, where the space inspector's twin
    /// must also ask whether the row still *is* what its editor was seeded from
    /// (see `sync_inspector_participant_edit`). The difference is structural, not
    /// an oversight: this roster is `list_global_agents`, and "a live shared
    /// agent, edited everywhere" is both the only shape this editor assumes and
    /// the roster's own membership predicate — so leaving is the only way to stop
    /// answering to it. Nothing shape-like is cached either: `AgentDraft` holds
    /// values, and the row's one conditional affordance (the Notebook door) is
    /// re-derived from the live row every frame rather than snapshotted.
    fn sync_open_forms(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // The roster is read out before anything is retired, so the store's
        // borrow ends here rather than straddling the focus handback below.
        let Some(present_ids) = self
            .agents_store
            .read(cx)
            .agents()
            .value()
            .map(|list| list.iter().map(|a| a.id.clone()).collect::<Vec<String>>())
        else {
            return;
        };
        let present = |id: &str| present_ids.iter().any(|p| p == id);
        if self
            .draft
            .as_ref()
            .is_some_and(|d| !present(&d.participant_id))
        {
            // The unmount no verb pressed still owes the keyboard back — this
            // is the path the rule was written for.
            let held = self.draft_field_focused(window, cx);
            self.draft = None;
            self.hand_back_focus(held, window, cx);
        }
        if self.retiring.as_ref().is_some_and(|id| !present(id)) {
            self.retiring = None;
        }
        // A refusal about an agent the roster no longer carries has no row to
        // render under, and nothing left to say — the agent is gone. The store
        // is told here rather than deciding for itself, because only a listing
        // that has answered can say a row is absent.
        self.agents_store
            .update(cx, |s, _| s.forget_op_errors_absent_from(&present_ids));
    }

    /// A refused write, rendered **under the row it was about**.
    ///
    /// The store keys refusals per agent, so the surface has to as well: two can
    /// stand at once, and a single store-wide band could not tell the reader
    /// which of their two actions was refused — nor could it show the second
    /// without discarding the first. The row is the context, so the band needs
    /// no subject in its visible text; its **accessible name carries the name**,
    /// because a screen reader meets the alert without the row above it. Its
    /// probes are keyed for the same reason element ids must be unique per
    /// painted element.
    ///
    /// Dismissible: nothing else clears a refusal until the next write to that
    /// same agent, so the × is how a reader acknowledges one — it never implies
    /// the write succeeded.
    fn render_op_error(
        &self,
        agent: &GlobalAgentInfo,
        err: &str,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let id = agent.id.clone();
        let dismiss_id = agent.id.clone();
        let subject = agent.label.clone();
        h_flex()
            .id(SharedString::from(format!("agents-error-{id}")))
            .probe(
                SharedString::from(format!("settings/agents/{id}/error")),
                gpui::Role::Alert,
                SharedString::from(format!("{subject}: {err}")),
            )
            .w_full()
            .gap_2()
            .items_start()
            .justify_between()
            .child(div().flex_1().min_w_0().child(error_banner(err, cx)))
            .child(
                div()
                    .id(SharedString::from(format!("agents-error-dismiss-{id}")))
                    .probe(
                        SharedString::from(format!("settings/agents/{id}/error/dismiss")),
                        gpui::Role::Button,
                        SharedString::from(format!("Dismiss the message about {subject}")),
                    )
                    .cursor_pointer()
                    .text_color(theme.muted_foreground)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("×")
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let id = dismiss_id.clone();
                        this.agents_store
                            .update(cx, |s, cx| s.clear_op_error(&id, cx));
                    })),
            )
    }

    /// One agent: the resting row, plus whichever of its two forms is open.
    fn render_row(&self, agent: &GlobalAgentInfo, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let id = agent.id.clone();
        let subject = agent.label.clone();
        let editing = self.editing_agent() == Some(id.as_str());
        let retiring = self.retiring.as_deref() == Some(id.as_str());

        // The summary line is the row's content: what it answers with, and when
        // it answers. `model · backend` comes from the shared helper, so the
        // backend — the billing difference — is named at rest here exactly as it
        // is in a space.
        let model = match agent.model_ref.as_deref() {
            Some(m) => {
                let (name, backend) = model_display(&self.stores, m, cx);
                format!("{name} · {backend}")
            }
            None => "no model set".to_string(),
        };
        let summary = SharedString::from(format!(
            "{model} · responds {}",
            notify_label(&agent.notify_policy)
        ));

        let mut wrap = v_flex()
            .w_full()
            .py_2()
            .gap_2()
            .border_b_1()
            .border_color(theme.border.opacity(0.5))
            .child(
                h_flex()
                    .id(SharedString::from(format!("agent-row-{id}")))
                    .probe_value(
                        format!("settings/agents/{id}"),
                        gpui::Role::ListItem,
                        SharedString::from(subject.clone()),
                        summary.clone(),
                    )
                    .w_full()
                    .gap_3()
                    .items_start()
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .gap_0p5()
                            .child(
                                div()
                                    .font_medium()
                                    .child(SharedString::from(agent.label.clone())),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(theme.muted_foreground)
                                    .child(summary),
                            ),
                    )
                    .child(self.render_row_verbs(agent, editing, cx)),
            );

        if editing {
            wrap = wrap.child(self.render_editor(agent, cx));
        }
        if retiring {
            wrap = wrap.child(self.render_retire_confirm(agent, cx));
        }
        // This agent's own refused write, under this agent's row.
        if let Some(err) = self
            .agents_store
            .read(cx)
            .op_error(&agent.id)
            .map(str::to_string)
        {
            wrap = wrap.child(self.render_op_error(agent, &err, cx));
        }
        wrap
    }

    /// The row's verbs. **Notebook first** — it is the one that says what a
    /// shared agent now has that a space-owned one didn't.
    fn render_row_verbs(
        &self,
        agent: &GlobalAgentInfo,
        editing: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let id_edit = agent.id.clone();
        let id_retire = agent.id.clone();
        let subject = agent.label.clone();
        let subject_retire = agent.label.clone();
        let subject_notebook = agent.label.clone();
        let notebook = agent.notebook_space_id.clone();
        let notebook_probe = agent.id.clone();

        h_flex()
            .gap_1()
            .flex_wrap()
            .justify_end()
            // A notebook is offered only where there is one to open — an
            // affordance that could only fail is not an affordance.
            .when_some(notebook, |el, space_id| {
                el.child(ghost_button_labeled(
                    SharedString::from(format!("agent-notebook-{notebook_probe}")),
                    SharedString::from(format!("settings/agents/{notebook_probe}/notebook")),
                    "Notebook",
                    format!("Open {subject_notebook}'s notebook"),
                    false,
                    cx,
                    cx.listener(move |this, _, _, cx| this.open_notebook(space_id.clone(), cx)),
                ))
            })
            .child(ghost_button_labeled(
                SharedString::from(format!("agent-edit-{id_edit}")),
                SharedString::from(format!("settings/agents/{id_edit}/edit")),
                if editing { "Close" } else { "Edit" },
                if editing {
                    format!("Close {subject}")
                } else {
                    format!("Edit {subject}")
                },
                false,
                cx,
                cx.listener(move |this, _, window, cx| this.toggle_edit(&id_edit, window, cx)),
            ))
            .child(ghost_button_labeled(
                SharedString::from(format!("agent-retire-{id_retire}")),
                SharedString::from(format!("settings/agents/{id_retire}/retire")),
                "Retire",
                format!("Retire {subject_retire}"),
                false,
                cx,
                cx.listener(move |this, _, window, cx| this.arm_retire(&id_retire, window, cx)),
            ))
    }

    fn render_editor(&self, agent: &GlobalAgentInfo, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let Some(draft) = self.draft.as_ref() else {
            return div().into_any_element();
        };
        v_flex()
            .id("agent-editor")
            .track_focus(&draft.focus)
            .w_full()
            .p_3()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    // The one thing an editor here has to say that a space's
                    // does not: this reaches every space.
                    .child(
                        "These are the agent's own settings — every space it is in follows them, \
                         except where that space set its own.",
                    ),
            )
            .child(field_label("Name", cx))
            .child(
                div()
                    .id("agent-name-wrap")
                    .probe_bounds("settings/agents/editor/name", gpui::Role::TextInput, "Name")
                    .child(Input::new(&draft.label).aria_label("Name")),
            )
            .child(field_label("Model", cx))
            .child(model_field(
                &self.stores,
                draft.model_ref.as_deref(),
                draft.picker_open,
                "settings/agents/editor/model".into(),
                &self.picker_scroll,
                cx,
                |this: &mut Self, _, _, cx| this.toggle_picker(cx),
                |id, this: &mut Self, cx| this.select_model(id, cx),
            ))
            .child(field_label("Charter", cx))
            .child(
                div()
                    .id("agent-prompt-wrap")
                    .probe_bounds(
                        "settings/agents/editor/system-prompt",
                        gpui::Role::TextInput,
                        "Charter",
                    )
                    .child(Input::new(&draft.system_prompt).aria_label("Charter")),
            )
            .child(field_label("Responds", cx))
            .child(self.render_notify(&draft.notify_policy, cx))
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button(
                        "agent-editor-cancel".into(),
                        "settings/agents/editor/cancel".into(),
                        "Cancel",
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| this.cancel_edit(window, cx)),
                    ))
                    .child(ghost_button_labeled(
                        "agent-editor-save".into(),
                        "settings/agents/editor/save".into(),
                        "Save",
                        format!("Save {}", agent.label),
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| this.save_edit(window, cx)),
                    )),
            )
            .into_any_element()
    }

    fn render_notify(&self, current: &str, cx: &Context<Self>) -> impl IntoElement {
        let mut row = h_flex().gap_1().flex_wrap();
        for (value, label) in NOTIFY_POLICIES {
            row = row.child(mode_chip(
                SharedString::from(format!("agent-notify-{value}")),
                SharedString::from(format!("settings/agents/editor/notify/{value}")),
                SharedString::from(label),
                current == value,
                cx,
                cx.listener(move |this, _, _, cx| this.set_draft_notify(value, cx)),
            ));
        }
        row
    }

    /// The two-step retire confirm (the Account pane's shape — an armed
    /// warning, never a modal). The copy names both consequences, because one of
    /// them is a *different object*: the notebook.
    fn render_retire_confirm(
        &self,
        agent: &GlobalAgentInfo,
        cx: &Context<Self>,
    ) -> gpui::AnyElement {
        let theme = cx.theme();
        let subject = agent.label.clone();
        v_flex()
            .id("agent-retire-confirm")
            .track_focus(&self.retire_focus)
            .w_full()
            .p_3()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(
                div()
                    .id("agent-retire-note")
                    .probe(
                        format!("settings/agents/{}/retire/note", agent.id),
                        gpui::Role::Label,
                        RETIRE_NOTE,
                    )
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(RETIRE_NOTE),
            )
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button_labeled(
                        "agent-retire-cancel".into(),
                        SharedString::from(format!("settings/agents/{}/retire/cancel", agent.id)),
                        "Keep",
                        format!("Keep {subject}"),
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| this.cancel_retire(window, cx)),
                    ))
                    .child(ghost_button_labeled(
                        "agent-retire-confirm-button".into(),
                        SharedString::from(format!("settings/agents/{}/retire/confirm", agent.id)),
                        "Retire",
                        format!("Retire {}", agent.label),
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| this.confirm_retire(window, cx)),
                    )),
            )
            .into_any_element()
    }
}
