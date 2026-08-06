//! The inspector's **Participants** section (task 26, wave 26.3) — who is in
//! this conversation, and how each of them answers.
//!
//! This is where the standalone Participants window went. Per-space
//! configuration belongs visually inside the space (task 26's decision), and
//! participants are the per-space setting people actually change, so the roster
//! sits under [`super::inspector`]'s Space section in the same Settings voice:
//! quiet section header, ruled rows, label-left/control-right.
//!
//! **A row is a disclosure, and the disclosure is the editor.** The panel is
//! 320px wide, so a resting row says only what identifies a member (name, the
//! "shared" tag, `model · backend`); opening it grows the full editor — system
//! prompt, model, notify policy, and, for a referenced global, the
//! edit-everywhere-vs-override-here fork. One is open at a time: the editor owns
//! live `InputState` entities and an explicit Save, and a panel this narrow with
//! several open at once is a scroll, not a surface.
//!
//! **The fork** (unchanged from the window it replaces): a participant is either
//! **owned** by this space (`source == "owned"` — one editor, writing its own
//! config) or a **referenced global** (`source == "referenced"`), whose editor
//! carries a mode toggle — **Everyone** writes the shared global's own config
//! (`ParticipantsStore::update_everywhere`), **This space only** writes the
//! per-membership override (`ParticipantsStore::set_override`; a cleared field
//! inherits). Switching modes re-seeds the fields from that mode's source.
//!
//! **Where the data lives.** The membership is
//! [`crate::stores::ParticipantsStore`]'s, keyed per space and refreshed on
//! `Change::Participants` — so two windows on one space agree and neither owns a
//! private copy. Only the transient editing bits (which row is open, its input
//! buffers, which picker is open) are view fields.

use eidola_app_core::{
    NewParticipant, ParticipantInfo, ParticipantOverride, ParticipantReference, ParticipantUpdate,
};
use gpui::{
    AnyElement, AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder as _, px,
};
use gpui_component::{
    ActiveTheme, StyledExt as _, h_flex,
    input::{Input, InputState},
    v_flex,
};

use crate::participants::{
    DEFAULT_AGENT_SYSTEM_PROMPT, EditMode, NOTIFY_POLICIES, error_banner, field_label,
    ghost_button, ghost_button_labeled, load_error_panel, mode_chip, model_display, model_field,
};
use crate::probe::Probe as _;

use super::SpaceView;

/// The placeholder every system-prompt field carries, so an empty one reads as
/// a choice rather than a missing value.
const PROMPT_PLACEHOLDER: &str = "A short instruction for how this participant behaves.";

/// The open editor for one participant — the row's disclosure. Holds the field
/// inputs plus the working model/notify/mode, and (for a referenced global) the
/// reference detail the two modes seed from.
pub(crate) struct ParticipantEdit {
    /// The form subtree's focus handle — the handback asks **containment**, not
    /// an enumeration of this form's inputs (see
    /// [`SpaceView::inspector_participant_field_focused`]).
    focus: gpui::FocusHandle,
    pub(crate) participant_id: String,
    kind: String,
    is_referenced: bool,
    reference: Option<ParticipantReference>,
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    /// The working model selection (`None` = no model set).
    model_ref: Option<String>,
    notify_policy: String,
    mode: EditMode,
    /// Whether "Share this agent…" has been pressed and the editor is showing
    /// its confirmation. It lives **inside the editor** rather than beside it
    /// because sharing is one-way: a flag with its own lifetime would have to be
    /// cleared at every transition that closes this editor (the roster dropping
    /// the row, Remove, the add form, another disclosure), and the one that
    /// forgot would leave an armed irreversible verb over a different
    /// participant. Owned by the thing it describes, it cannot outlive it.
    promote_confirm: bool,
}

impl ParticipantEdit {
    /// The visible fields as an **"edit everywhere"** update — the participant's
    /// own config.
    ///
    /// Shared by the two verbs that write it: Save (for an owned participant, or
    /// a referenced global in Everyone mode) and Share, which saves the draft
    /// before promoting. One reading of "what the fields mean" for both, so the
    /// share can never write a different persona than the save beside it would.
    fn as_everywhere_update(&self, cx: &gpui::App) -> ParticipantUpdate {
        let label = self.label.read(cx).value().trim().to_string();
        let system_prompt = self.system_prompt.read(cx).value().trim().to_string();
        let mut update = ParticipantUpdate {
            label: Some(label),
            notify_policy: Some(self.notify_policy.clone()),
            ..Default::default()
        };
        if self.kind == "agent" {
            update.model_ref = Some(self.model_ref.clone().filter(|s| !s.is_empty()));
            update.system_prompt = Some((!system_prompt.is_empty()).then_some(system_prompt));
        }
        update
    }
}

/// The open add-a-participant form (agents only).
pub(crate) struct ParticipantAdd {
    focus: gpui::FocusHandle,
    label: Entity<InputState>,
    system_prompt: Entity<InputState>,
    model_ref: Option<String>,
    notify_policy: String,
}

/// **The sentence a grant has to say** (task 37), for each of the two shapes a
/// candidate comes in.
///
/// A shared agent joins by ordinary membership, and what the reader needs to
/// know is what membership *is* here: it can read this conversation. A
/// **space-owned** one cannot join at all without being shared first, so the
/// sentence says that too — including the part that cannot be taken back, and
/// the reassurance promotion actually earns (its persona where it works today
/// is preserved byte-for-byte, because promotion moves ownership, not config).
pub(crate) fn grant_statement(
    label: &str,
    shared: bool,
    home_space_title: Option<&str>,
) -> SharedString {
    if shared {
        return SharedString::from(format!(
            "{label} will be able to read this conversation, including everything already in it."
        ));
    }
    let home = match home_space_title {
        Some(home) => format!(" It keeps its persona in {home} exactly as it is."),
        None => String::new(),
    };
    SharedString::from(format!(
        "{label} works in one conversation today, so adding it here shares it across spaces.\
{home} It will be able to read this conversation, including everything already in it. \
Sharing can't be undone."
    ))
}

/// The open **"Invite an agent…"** form (task 37's grant): the candidates this
/// space could give membership to, and — once one is chosen — the sentence that
/// says what granting it means.
///
/// Its data is a **view-owned read**, not a store cell: the list is a transient
/// picker's material, only ever looked at while this form is open, and killing
/// its fetch when the window closes strands nothing (STATE.md's owner = blast
/// radius). The *write* it leads to is the store's, like every other membership
/// change.
pub(crate) struct InviteForm {
    pub(crate) focus: gpui::FocusHandle,
    /// `None` while the read is in flight — the form says "Loading…" rather
    /// than reading as an empty library.
    pub(crate) candidates: Option<Vec<eidola_app_core::GrantableAgent>>,
    /// A failed read, said out loud rather than shown as "nobody to invite".
    pub(crate) error: Option<SharedString>,
    /// The chosen candidate: `(id, label, shared, home space title)`.
    pub(crate) confirming: Option<(String, SharedString, bool, Option<SharedString>)>,
}

/// The open "Save these participants as a template…" form.
pub(crate) struct TemplateForm {
    focus: gpui::FocusHandle,
    title: Entity<InputState>,
}

/// Which model picker is open. At most one at a time — they share one scroll
/// handle, reset to the top on each open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ParticipantPicker {
    Editor,
    Add,
}

impl SpaceView {
    // -- Reads -------------------------------------------------------------

    fn space_id(&self, cx: &gpui::App) -> Option<String> {
        self.space.read(cx).id().map(str::to_string)
    }

    /// Whether this space is `participant_id`'s own notebook — the one
    /// membership no roster may offer to end.
    fn inspector_participant_owns_this_notebook(
        &self,
        participant_id: &str,
        cx: &gpui::App,
    ) -> bool {
        let Some(space_id) = self.space_id(cx) else {
            return false;
        };
        self.stores
            .space_settings
            .read(cx)
            .settings(&space_id)
            .value()
            .and_then(|s| s.notebook_participant_id.as_deref())
            == Some(participant_id)
    }

    /// Whether this space is some agent's private **notebook**.
    ///
    /// **Flagged, not decided** (task 37): whether an agent may be granted
    /// observer membership of another agent's notebook is one of the two
    /// questions the spec leaves to Mike, so this surface takes the
    /// conservative reading and offers no *new* door here — app-core's rules
    /// are unchanged either way (`add_global_participant` has never refused a
    /// notebook), so nothing here forecloses the answer. If notebooks turn out
    /// to be grantable, deleting this predicate's one use is the whole change.
    fn inspector_space_is_a_notebook(&self, cx: &gpui::App) -> bool {
        let Some(space_id) = self.space_id(cx) else {
            return false;
        };
        self.stores
            .space_settings
            .read(cx)
            .settings(&space_id)
            .value()
            .is_some_and(|s| s.notebook_participant_id.is_some())
    }

    /// This space's membership as the store holds it.
    fn inspector_participants(&self, cx: &gpui::App) -> Vec<ParticipantInfo> {
        match self.space_id(cx) {
            Some(id) => self.stores.participants.read(cx).list(&id).to_vec(),
            None => Vec::new(),
        }
    }

    /// Whether any of the section's text fields holds the window's focus — read
    /// by [`SpaceView::inspector_field_focused`], which gates type-to-compose
    /// and the close-time focus handoff.
    /// Whether the keyboard is anywhere **inside** one of this section's three
    /// forms.
    ///
    /// **Containment, not an enumeration of inputs.** A form is a subtree — its
    /// fields, its mode chips, its verbs, its model dropdown — and a predicate
    /// that lists the text inputs answers "not held" for everything else in it.
    /// Since this is what the focus handback consults (`hand_back_inspector_focus`),
    /// an enumerated answer drops the keyboard on the floor for exactly the
    /// controls a future edit adds, and re-breaks each time (Codex review, PR
    /// #279). `contains_focused` is the idiom the Library's reveal already uses.
    pub(crate) fn inspector_participant_field_focused(
        &self,
        window: &Window,
        cx: &gpui::App,
    ) -> bool {
        let holds = |focus: &gpui::FocusHandle| focus.contains_focused(window, cx);
        self.inspector_participant_edit
            .as_ref()
            .is_some_and(|e| holds(&e.focus))
            || self
                .inspector_participant_add
                .as_ref()
                .is_some_and(|a| holds(&a.focus))
            || self
                .inspector_template_form
                .as_ref()
                .is_some_and(|t| holds(&t.focus))
            // The invite form is the section's fourth, and it is a form in
            // exactly this sense: its rows and verbs are tab stops, so a
            // keystroke inside it is a keystroke inside the panel.
            || self
                .inspector_invite
                .as_ref()
                .is_some_and(|i| holds(&i.focus))
    }

    // -- Test seams --------------------------------------------------------

    /// Which participant's disclosure is open, if any.
    #[doc(hidden)]
    pub fn inspector_editing_participant(&self) -> Option<&str> {
        self.inspector_participant_edit
            .as_ref()
            .map(|e| e.participant_id.as_str())
    }

    #[doc(hidden)]
    pub fn inspector_editing_mode(&self) -> Option<EditMode> {
        self.inspector_participant_edit.as_ref().map(|e| e.mode)
    }

    /// The open disclosure's subtree focus handle (tests).
    #[doc(hidden)]
    pub fn inspector_editing_focus_handle(&self) -> Option<gpui::FocusHandle> {
        self.inspector_participant_edit
            .as_ref()
            .map(|e| e.focus.clone())
    }

    #[doc(hidden)]
    pub fn inspector_editing_label_state(&self) -> Option<Entity<InputState>> {
        self.inspector_participant_edit
            .as_ref()
            .map(|e| e.label.clone())
    }

    #[doc(hidden)]
    pub fn inspector_editing_prompt_state(&self) -> Option<Entity<InputState>> {
        self.inspector_participant_edit
            .as_ref()
            .map(|e| e.system_prompt.clone())
    }

    #[doc(hidden)]
    pub fn inspector_adding_participant(&self) -> bool {
        self.inspector_participant_add.is_some()
    }

    #[doc(hidden)]
    pub fn inspector_adding_label_state(&self) -> Option<Entity<InputState>> {
        self.inspector_participant_add
            .as_ref()
            .map(|a| a.label.clone())
    }

    #[doc(hidden)]
    pub fn inspector_adding_prompt(&self, cx: &gpui::App) -> Option<String> {
        self.inspector_participant_add
            .as_ref()
            .map(|a| a.system_prompt.read(cx).value().to_string())
    }

    #[doc(hidden)]
    pub fn inspector_saving_template(&self) -> bool {
        self.inspector_template_form.is_some()
    }

    #[doc(hidden)]
    pub fn inspector_participant_picker_open_for_test(&self) -> bool {
        self.open_inspector_participant_picker().is_some()
    }

    /// Open the add form's model dropdown without a click (tests, driver).
    #[doc(hidden)]
    pub fn inspector_open_add_picker_for_test(&mut self, cx: &mut Context<Self>) {
        if self.inspector_participant_picker != Some(ParticipantPicker::Add) {
            self.inspector_toggle_participant_picker(ParticipantPicker::Add, cx);
        }
    }

    /// The same, for the open disclosure's model dropdown.
    #[doc(hidden)]
    pub fn inspector_open_editor_picker_for_test(&mut self, cx: &mut Context<Self>) {
        if self.inspector_participant_picker != Some(ParticipantPicker::Editor) {
            self.inspector_toggle_participant_picker(ParticipantPicker::Editor, cx);
        }
    }

    // -- Editing -----------------------------------------------------------

    /// Open (or close) a participant's disclosure. Opening seeds the fields from
    /// its effective config; a referenced global starts in the safe
    /// this-space-only mode.
    pub fn inspector_toggle_participant(
        &mut self,
        participant_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.inspector_editing_participant() == Some(participant_id) {
            self.inspector_cancel_participant_edit(window, cx);
            return;
        }
        let Some(p) = self
            .inspector_participants(cx)
            .into_iter()
            .find(|p| p.id == participant_id)
        else {
            return;
        };
        self.inspector_participant_add = None;
        self.inspector_template_form = None;
        self.inspector_invite = None;
        self.inspector_participant_picker = None;
        let is_referenced = p.source == "referenced";
        let label = cx.new(|cx| InputState::new(window, cx).default_value(&p.label));
        // A reveal focuses what it revealed (the Settings idiom's rule).
        label.update(cx, |s, cx| s.focus(window, cx));
        let system_prompt = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(2, 8)
                .placeholder(PROMPT_PLACEHOLDER)
                .default_value(p.system_prompt.clone().unwrap_or_default())
        });
        self.inspector_participant_edit = Some(ParticipantEdit {
            focus: cx.focus_handle(),
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
            promote_confirm: false,
        });
        cx.notify();
    }

    /// Switch a referenced global's editor between "edit everywhere" and
    /// "override here", re-seeding the visible fields from the mode's source
    /// (the shared base, or the effective/override) — otherwise the toggle would
    /// silently retarget the values already on screen.
    pub fn inspector_set_edit_mode(
        &mut self,
        mode: EditMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Snapshot the handles so the `self` borrow doesn't straddle the
        // `update` calls below.
        let (label_entity, prompt_entity, reference, already) = {
            let Some(edit) = self.inspector_participant_edit.as_ref() else {
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
                if let Some(edit) = self.inspector_participant_edit.as_mut() {
                    edit.mode = mode;
                }
                cx.notify();
                return;
            }
        };
        label_entity.update(cx, |s, cx| s.set_value(&label, window, cx));
        prompt_entity.update(cx, |s, cx| s.set_value(&prompt, window, cx));
        if let Some(edit) = self.inspector_participant_edit.as_mut() {
            edit.mode = mode;
            edit.model_ref = model_ref;
            edit.notify_policy = notify_policy;
        }
        cx.notify();
    }

    pub fn inspector_set_edit_notify(&mut self, policy: &str, cx: &mut Context<Self>) {
        if let Some(edit) = self.inspector_participant_edit.as_mut() {
            edit.notify_policy = policy.to_string();
            cx.notify();
        }
    }

    /// Hand the keyboard back to the view root when a form that **was** holding
    /// it has gone and left nothing focused.
    ///
    /// `held` is an observation taken *before* the form was dropped, and it has
    /// to be: the question is asked of the form's own `InputState`s, which die
    /// with it. Every path that closes one of this section's three forms goes
    /// through here — the verbs a reader presses as well as the roster-driven
    /// retire, which is where the rule started (Codex review, PR #279). And the
    /// restore is `set_inspector_open`'s: only from a lender still holding it,
    /// so a form that never had the keyboard cannot take it from whatever does.
    pub(crate) fn hand_back_inspector_focus(
        &mut self,
        held: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if held && !self.inspector_field_focused(window, cx) {
            window.focus(&self.focus_handle, cx);
        }
    }

    pub fn inspector_cancel_participant_edit(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let held = self.inspector_field_focused(window, cx);
        self.inspector_participant_edit = None;
        self.inspector_participant_picker = None;
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    /// Commit the open editor — "edit everywhere" or "override here" per its
    /// mode; an owned participant is always an edit of its own config.
    pub fn inspector_save_participant_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        let held = self.inspector_field_focused(window, cx);
        let Some(edit) = self.inspector_participant_edit.take() else {
            return;
        };
        let label = edit.label.read(cx).value().trim().to_string();
        let system_prompt = edit.system_prompt.read(cx).value().trim().to_string();
        let is_agent = edit.kind == "agent";
        let pid = edit.participant_id.clone();

        if edit.is_referenced && edit.mode == EditMode::OverrideHere {
            // Per-membership overrides. An empty model/prompt on a referenced
            // agent reverts to inherited (`None`); the label is required, so a
            // label equal to the base still writes an (equal) override —
            // acceptable and honest.
            let mut ov = ParticipantOverride {
                label: Some(Some(label)),
                notify_policy: Some(Some(edit.notify_policy.clone())),
                ..Default::default()
            };
            if is_agent {
                ov.model_ref = Some(edit.model_ref.clone().filter(|s| !s.is_empty()));
                ov.system_prompt = Some((!system_prompt.is_empty()).then_some(system_prompt));
            }
            self.stores
                .participants
                .update(cx, |s, cx| s.set_override(space_id, pid, ov, cx));
        } else {
            let update = edit.as_everywhere_update(cx);
            // The premise the editor was seeded on: an owned row belongs to this
            // space, a referenced one is a shared identity. Either can move
            // under an open editor (promotion), and the write is where that is
            // caught.
            let expected = match edit.is_referenced {
                true => eidola_app_core::ExpectedScope::Global,
                false => eidola_app_core::ExpectedScope::SpaceOwned {
                    space_id: space_id.clone(),
                },
            };
            self.stores.participants.update(cx, |s, cx| {
                s.update_everywhere(space_id, pid, update, expected, cx)
            });
        }
        self.inspector_participant_picker = None;
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    // -- Sharing (task 36's in-place promotion) ----------------------------

    /// Arm the "Share this agent…" confirmation on the open disclosure.
    ///
    /// Sharing is **one-way** (app-core has no demotion — it would strand
    /// memberships and memory), so it asks first. The confirmation is also where
    /// the reassurance is spoken: promotion moves the row's ownership, never its
    /// configuration, so the agent answers in this space exactly as it does now.
    /// Arm or stand down the share confirmation.
    ///
    /// **Either direction unmounts the control that was pressed** — arming
    /// replaces "Share this agent…" with the confirmation, standing down does
    /// the reverse — and these are real tab stops (`probe(Role::Button)` derives
    /// `focusable()` + `tab_index(0)`), so a keyboard reader is left holding a
    /// handle to something nobody paints. The form itself survives, so the
    /// keyboard goes **to the form**, not out to the view root: the reader is
    /// still editing this participant (Codex review, PR #279).
    fn set_inspector_promote_confirm(
        &mut self,
        armed: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(edit) = self.inspector_participant_edit.as_mut() else {
            return;
        };
        edit.promote_confirm = armed;
        let form = edit.focus.clone();
        if form.contains_focused(window, cx) {
            window.focus(&form, cx);
        }
        cx.notify();
    }

    pub fn inspector_begin_promote(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.set_inspector_promote_confirm(true, window, cx);
    }

    pub fn inspector_cancel_promote(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.set_inspector_promote_confirm(false, window, cx);
    }

    /// Confirm the share: `ParticipantsStore::promote` — the open editor's
    /// fields saved, then `AppCore::promote_participant`, in one store op.
    ///
    /// **The draft is what is being shared.** The confirmation renders *inside*
    /// the editor, with name, model, charter and notify policy still on screen
    /// above it, and it promises the agent "keeps this space's persona exactly as
    /// it is" — a sentence a reader can only read against those visible values.
    /// Consuming the draft and promoting the stored row would make it false about
    /// everything they had just typed (Codex review, PR #279). The participant is
    /// space-**owned** at this moment (that is what `can_share` means), so the
    /// save is an ordinary edit of its own config, and promotion's byte-for-byte
    /// guarantee then applies to the values just written. Both core calls travel
    /// in one `bridge` closure — see [`crate::stores::ParticipantsStore::promote`].
    ///
    /// A blank name is refused **before** the first write, the way the add form
    /// refuses one: app-core would reject it anyway, and returning the draft
    /// keeps the rest of the reader's typing (with the confirmation still armed,
    /// over the empty field that is the reason).
    ///
    /// The editor closes, because what it was editing has changed shape: the row
    /// comes back from the re-list as a **referenced global**, whose editor
    /// carries the everywhere-vs-here fork this one was seeded without. Nothing
    /// else is view work — the roster's "shared" tag falls out of the same
    /// re-list.
    pub fn inspector_confirm_promote(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        let held = self.inspector_field_focused(window, cx);
        let Some(edit) = self.inspector_participant_edit.take() else {
            return;
        };
        let update = edit.as_everywhere_update(cx);
        if update.label.as_deref().is_some_and(str::is_empty) {
            self.inspector_participant_edit = Some(edit);
            return;
        }
        self.inspector_participant_picker = None;
        self.stores.participants.update(cx, |s, cx| {
            s.promote(space_id, edit.participant_id, Some(update), cx)
        });
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    /// Whether the open disclosure is showing its share confirmation (tests).
    #[doc(hidden)]
    pub fn inspector_promote_confirming(&self) -> bool {
        self.inspector_participant_edit
            .as_ref()
            .is_some_and(|e| e.promote_confirm)
    }

    pub fn inspector_remove_participant(
        &mut self,
        participant_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        let pid = participant_id.to_string();
        self.stores
            .participants
            .update(cx, |s, cx| s.remove(space_id, pid, cx));
        if self.inspector_editing_participant() == Some(participant_id) {
            let held = self.inspector_field_focused(window, cx);
            self.inspector_participant_edit = None;
            self.hand_back_inspector_focus(held, window, cx);
        }
        cx.notify();
    }

    // -- Adding ------------------------------------------------------------

    pub fn inspector_begin_add_participant(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.inspector_participant_edit = None;
        self.inspector_template_form = None;
        self.inspector_invite = None;
        self.inspector_participant_picker = None;
        let label = cx.new(|cx| InputState::new(window, cx).placeholder("Participant name"));
        label.update(cx, |s, cx| s.focus(window, cx));
        // A new agent starts from the shared default charter — the same
        // starting point the Templates pane offers, so the two surfaces don't
        // contradict each other on what "a new participant" begins as.
        let system_prompt = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(2, 8)
                .placeholder(PROMPT_PLACEHOLDER)
                .default_value(DEFAULT_AGENT_SYSTEM_PROMPT)
        });
        self.inspector_participant_add = Some(ParticipantAdd {
            focus: cx.focus_handle(),
            label,
            system_prompt,
            model_ref: Some(self.stores.config.read(cx).default_model()),
            notify_policy: "human".to_string(),
        });
        cx.notify();
    }

    pub fn inspector_set_add_notify(&mut self, policy: &str, cx: &mut Context<Self>) {
        if let Some(add) = self.inspector_participant_add.as_mut() {
            add.notify_policy = policy.to_string();
            cx.notify();
        }
    }

    pub fn inspector_cancel_add_participant(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let held = self.inspector_field_focused(window, cx);
        self.inspector_participant_add = None;
        self.inspector_participant_picker = None;
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    pub fn inspector_save_add_participant(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        let held = self.inspector_field_focused(window, cx);
        let Some(add) = self.inspector_participant_add.take() else {
            return;
        };
        let label = add.label.read(cx).value().trim().to_string();
        if label.is_empty() {
            // Keep the form open; app-core would refuse an empty label anyway,
            // and not sending saves the round trip.
            self.inspector_participant_add = Some(add);
            return;
        }
        let system_prompt = add.system_prompt.read(cx).value().trim().to_string();
        let participant = NewParticipant {
            label,
            model_ref: add.model_ref.clone().filter(|s| !s.is_empty()),
            system_prompt: (!system_prompt.is_empty()).then_some(system_prompt),
            notify_policy: add.notify_policy.clone(),
        };
        self.stores
            .participants
            .update(cx, |s, cx| s.add(space_id, participant, cx));
        self.inspector_participant_picker = None;
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    // -- Inviting an agent (task 37's grant) -------------------------------

    /// Open the invite form and start its read. The candidates come from
    /// app-core's viewer-scoped listing, so this surface can only ever offer
    /// agents this reader already knows about.
    ///
    /// **The reveal focuses what it revealed** — the Settings idiom every other
    /// form in this section follows (the disclosure, Add and the template form
    /// each focus their first field). It is the mount-side half of the handback
    /// rule, and it is owed here for the same reason: the form *replaces* the
    /// "Invite an agent…" door, which is a real tab stop (`probe(Role::Button)`
    /// derives `focusable()` + `tab_index(0)`), so opening it otherwise leaves
    /// the window holding a handle to something nobody paints — keystrokes
    /// reach nothing and Tab restarts from the window root (Codex review, PR
    /// #280). This form has no text field to focus, so the keyboard goes to the
    /// form itself, which is what `contains_focused` asks about anyway.
    pub fn inspector_begin_invite(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.inspector_participant_edit = None;
        self.inspector_participant_add = None;
        self.inspector_template_form = None;
        self.inspector_participant_picker = None;
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        self.inspector_invite = Some(InviteForm {
            focus,
            candidates: None,
            error: None,
            confirming: None,
        });
        let (Some(space_id), Some(core)) = (self.space_id(cx), self.stores.app_core()) else {
            // No backend (stub mode) or no space yet: the form stands empty and
            // honest rather than spinning forever.
            if let Some(form) = self.inspector_invite.as_mut() {
                form.candidates = Some(Vec::new());
            }
            cx.notify();
            return;
        };
        let rx = crate::bridge::list_grantable_agents(core, space_id);
        self.inspector_invite_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = rx.await;
            this.update(cx, |this, cx| {
                if let Some(form) = this.inspector_invite.as_mut() {
                    match result {
                        Ok(Ok(candidates)) => form.candidates = Some(candidates),
                        Ok(Err(err)) => {
                            form.candidates = Some(Vec::new());
                            form.error = Some(SharedString::from(err.to_string()));
                        }
                        // A dropped receiver is a closing window; say nothing.
                        Err(_) => {}
                    }
                }
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    /// Arm the confirmation for one candidate — where the consequence is said.
    ///
    /// The candidate rows are tab stops, and the confirmation **replaces** them
    /// inside the same form, so the pressed row unmounts: the keyboard goes to
    /// the form, exactly as it does when the share confirmation replaces a
    /// verb ([`Self::set_inspector_promote_confirm`]) — and, as there, only
    /// from a form that was holding it, so a pointer press from elsewhere in
    /// the window takes nothing.
    pub fn inspector_arm_invite(
        &mut self,
        participant_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(form) = self.inspector_invite.as_mut() else {
            return;
        };
        let Some(candidate) = form
            .candidates
            .as_ref()
            .and_then(|c| c.iter().find(|c| c.id == participant_id))
        else {
            return;
        };
        form.confirming = Some((
            candidate.id.clone(),
            SharedString::from(candidate.label.clone()),
            candidate.shared,
            candidate.home_space_title.clone().map(SharedString::from),
        ));
        let focus = form.focus.clone();
        if focus.contains_focused(window, cx) {
            window.focus(&focus, cx);
        }
        cx.notify();
    }

    /// The invite form's subtree focus handle (tests).
    #[doc(hidden)]
    pub fn inspector_invite_focus_handle(&self) -> Option<gpui::FocusHandle> {
        self.inspector_invite.as_ref().map(|f| f.focus.clone())
    }

    pub fn inspector_cancel_invite(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.inspector_field_focused(window, cx);
        self.inspector_invite = None;
        self.inspector_invite_task = None;
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    /// Grant the armed candidate membership of this space, as an observer.
    pub fn inspector_confirm_invite(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        let Some((participant_id, _, shared, _)) = self
            .inspector_invite
            .as_ref()
            .and_then(|f| f.confirming.clone())
        else {
            return;
        };
        self.stores.participants.update(cx, |s, cx| {
            s.grant_membership(space_id, participant_id, shared, cx)
        });
        self.inspector_cancel_invite(window, cx);
    }

    /// Whether the roster offers the grant door here (the notebook question,
    /// asked of the surface rather than of a painted frame).
    #[doc(hidden)]
    pub fn inspector_offers_grant_door_for_test(&self, cx: &gpui::App) -> bool {
        !self.inspector_space_is_a_notebook(cx)
    }

    /// Seed the invite form's candidates without a backend — the driver and
    /// the visual harness run on stub stores, where there is nothing to list.
    #[doc(hidden)]
    pub fn seed_invite_candidates_for_test(
        &mut self,
        candidates: Vec<eidola_app_core::GrantableAgent>,
        cx: &mut Context<Self>,
    ) {
        if let Some(form) = self.inspector_invite.as_mut() {
            form.candidates = Some(candidates);
        }
        cx.notify();
    }

    /// What the invite form is showing (behavior tests read the transition
    /// here rather than through the painted panel): the candidate labels, and
    /// the sentence an armed candidate is showing.
    #[doc(hidden)]
    pub fn inspector_invite_for_test(&self) -> Option<(Vec<String>, Option<SharedString>)> {
        let form = self.inspector_invite.as_ref()?;
        Some((
            form.candidates
                .as_ref()
                .map(|c| c.iter().map(|c| c.label.clone()).collect())
                .unwrap_or_default(),
            form.confirming
                .as_ref()
                .map(|(_, label, shared, home)| grant_statement(label, *shared, home.as_deref())),
        ))
    }

    // -- Save as template --------------------------------------------------

    pub fn inspector_begin_template(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.inspector_participant_edit = None;
        self.inspector_participant_add = None;
        self.inspector_invite = None;
        let default_title = self
            .space_title(cx)
            .map(|t| t.to_string())
            .unwrap_or_else(|| "My template".to_string());
        let title = cx.new(|cx| InputState::new(window, cx).default_value(&default_title));
        title.update(cx, |s, cx| s.focus(window, cx));
        self.inspector_template_form = Some(TemplateForm {
            focus: cx.focus_handle(),
            title,
        });
        cx.notify();
    }

    pub fn inspector_cancel_template(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.inspector_field_focused(window, cx);
        self.inspector_template_form = None;
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    pub fn inspector_save_template(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        let held = self.inspector_field_focused(window, cx);
        let Some(form) = self.inspector_template_form.take() else {
            return;
        };
        let title = form.title.read(cx).value().trim().to_string();
        if title.is_empty() {
            self.inspector_template_form = Some(form);
            return;
        }
        self.stores
            .templates
            .update(cx, |s, cx| s.create_from_space(space_id, title, cx));
        self.hand_back_inspector_focus(held, window, cx);
        cx.notify();
    }

    // -- Pickers & errors --------------------------------------------------

    /// The dropdown that is **actually on screen**, if any — the one door every
    /// reader of the flag goes through.
    ///
    /// The field records a click; this answers the question its readers actually
    /// have. A model dropdown is painted *inside* a form — the open disclosure's
    /// editor, or the add form — so every transition that unmounts that form
    /// takes the dropdown off the screen with it, while the flag stands. A
    /// standing flag then tells [`SpaceView::transient_overlay_open`] that an
    /// overlay owns the keyboard, and the window goes on yielding every arrow,
    /// Escape and printable to something nobody can see (Codex review, PR #278:
    /// "Save these participants as a template…" drops the form the dropdown
    /// hangs in; **Remove** does the same to the editor; so does a re-list that
    /// drops the edited row — see
    /// [`SpaceView::sync_inspector_participant_edit`]).
    ///
    /// Deriving the answer from the owning form is what makes the class
    /// unrepresentable, rather than a clear-the-flag at each transition — the
    /// next of which would forget. A stale flag can never be *revived* either:
    /// the only two things that mount a form ([`Self::inspector_toggle_participant`],
    /// [`Self::inspector_begin_add_participant`]) both reset it. The panel close
    /// keeps its own explicit clear for the other reason — the half-written
    /// editor deliberately survives a mis-hit ⌥⌘I, and the dropdown deliberately
    /// does not.
    pub(crate) fn open_inspector_participant_picker(&self) -> Option<ParticipantPicker> {
        let target = self.inspector_participant_picker?;
        let mounted = match target {
            ParticipantPicker::Editor => self.inspector_participant_edit.is_some(),
            ParticipantPicker::Add => self.inspector_participant_add.is_some(),
        };
        mounted.then_some(target)
    }

    /// Retire an open disclosure whose participant **no longer answers to the
    /// shape it was seeded from** — the roster is what decides whether the
    /// editor paints, so the roster is what has to retire it. Two ways a row
    /// stops answering, and the same reconcile covers both:
    ///
    /// * **It left.** Another window removed it, and the store's re-list is the
    ///   only news this view gets. Nothing paints the editor once its row is
    ///   gone, so every window-local thing hanging off it becomes a claim about
    ///   a surface that is not there: its dropdown (above), and — worse — the
    ///   field focus, which would go on reporting through
    ///   `inspector_field_focused` that a **dead** input holds the keyboard,
    ///   leaving type-to-compose inert until a click revived it.
    /// * **It became something else.** Promotion keeps the participant's id, so
    ///   the re-list carries the same row with `source` flipped to
    ///   `referenced` — invisible to an id check, and the editor is seeded on
    ///   exactly that distinction. Left standing it paints the *owned* form over
    ///   a shared agent: no Everyone/This-space fork, a live "Share this agent…"
    ///   over something that already is one, and a **Save** routed to
    ///   `update_everywhere` — publishing the draft to every space the agent
    ///   joins without the reader ever being shown the choice (Codex review, PR
    ///   #279).
    ///
    /// **Retire rather than re-seed**, because the fork is a decision only the
    /// reader can make: adopting a mode for them would silently retarget the
    /// values on screen to a different destination, which is the very thing
    /// [`Self::inspector_set_edit_mode`] re-seeds to prevent. Re-opening the row
    /// gives the referenced editor, its chips, and its safe default. This is the
    /// same rule [`Self::inspector_confirm_promote`] already applies when *this*
    /// window does the promoting ("what it was editing has changed shape"); all
    /// that is new is that another window's promotion arrives as a re-list
    /// rather than as a verb, and a re-list is not a thing anyone can clean up
    /// after by hand. The armed share confirmation needs no separate handling —
    /// it lives **inside** `ParticipantEdit`, so it cannot outlive it.
    ///
    /// The mirror case cannot arise: promotion is one-way (app-core offers no
    /// demotion), so a `referenced` row never becomes `owned` under an open
    /// editor. `scope` moves with `source` and adds nothing to ask about.
    ///
    /// **Focus comes back from a panel that is holding it** — `set_inspector_open`'s
    /// rule, restored only to a lender who still has nothing.
    pub(crate) fn sync_inspector_participant_edit(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((pid, was_referenced)) = self
            .inspector_participant_edit
            .as_ref()
            .map(|e| (e.participant_id.clone(), e.is_referenced))
        else {
            return;
        };
        let Some(space_id) = self.space_id(cx) else {
            return;
        };
        // Only a listing that has answered can say anything about a row: a first
        // load in flight, or a failed one with nothing prior, knows nothing.
        let stale = {
            let store = self.stores.participants.read(cx);
            match store.participants(&space_id).value() {
                Some(list) => match list.iter().find(|p| p.id == pid) {
                    None => true,
                    Some(p) => (p.source == "referenced") != was_referenced,
                },
                None => false,
            }
        };
        if !stale {
            return;
        }
        let held = self.inspector_field_focused(window, cx);
        self.inspector_participant_edit = None;
        self.inspector_participant_picker = None;
        self.hand_back_inspector_focus(held, window, cx);
    }

    pub(crate) fn inspector_toggle_participant_picker(
        &mut self,
        target: ParticipantPicker,
        cx: &mut Context<Self>,
    ) {
        self.inspector_participant_picker =
            if self.open_inspector_participant_picker() == Some(target) {
                None
            } else {
                // A freshly opened picker starts at the top.
                self.inspector_participant_picker_scroll = gpui::ScrollHandle::new();
                Some(target)
            };
        cx.notify();
    }

    /// Close an open participant model picker, reporting whether it was open —
    /// the Escape rung the view root owns, beside the router picker's. Only a
    /// dropdown the reader can *see* consumes the press; a flag standing behind
    /// an unmounted form is tidied away and the press falls to the next rung.
    pub(crate) fn close_inspector_participant_picker(&mut self, cx: &mut Context<Self>) -> bool {
        let was_visible = self.open_inspector_participant_picker().is_some();
        if self.inspector_participant_picker.take().is_none() {
            return false;
        }
        cx.notify();
        was_visible
    }

    /// Select a model into whichever form is open (the picker's target, or the
    /// open form when driven from a test).
    pub fn inspector_select_participant_model(&mut self, model_id: &str, cx: &mut Context<Self>) {
        let target = self.open_inspector_participant_picker().or({
            if self.inspector_participant_edit.is_some() {
                Some(ParticipantPicker::Editor)
            } else if self.inspector_participant_add.is_some() {
                Some(ParticipantPicker::Add)
            } else {
                None
            }
        });
        // The choice closes the dropdown either way.
        self.inspector_participant_picker = None;
        match target {
            Some(ParticipantPicker::Editor) => {
                if let Some(edit) = self.inspector_participant_edit.as_mut() {
                    edit.model_ref = Some(model_id.to_string());
                }
            }
            Some(ParticipantPicker::Add) => {
                if let Some(add) = self.inspector_participant_add.as_mut() {
                    add.model_ref = Some(model_id.to_string());
                }
            }
            None => {}
        }
        cx.notify();
    }

    /// Re-fetch this space's membership — the Retry on a failed load (`ensure`
    /// declines once a `Failed` cell exists, so this is the only path back).
    pub fn inspector_retry_participants(&mut self, cx: &mut Context<Self>) {
        if let Some(id) = self.space_id(cx) {
            self.stores
                .participants
                .update(cx, |s, cx| s.refresh(id, cx));
        }
        cx.notify();
    }

    // -- Render ------------------------------------------------------------

    /// Section 2 — **Participants**. `None` for a blank ⌘N space: there is no
    /// membership until the space is saved, and the Space section above already
    /// says so once.
    pub(crate) fn render_inspector_participants_section(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let space_id = self.space_id(cx)?;
        let (muted, border, link, fg) = {
            let theme = cx.theme();
            (
                theme.muted_foreground,
                theme.border,
                theme.link,
                theme.foreground,
            )
        };
        let (load_error, has_value, loading) = {
            let store = self.stores.participants.read(cx);
            let cell = store.participants(&space_id);
            (
                cell.error().map(|e| e.to_string()),
                cell.has_value(),
                cell.is_loading(),
            )
        };
        let op_errors = self.stores.participants.read(cx).op_errors_for(&space_id);

        let mut col = v_flex()
            .w_full()
            .gap_2()
            // The ruled group: the section reads as its own block beside the
            // Space rows above it.
            .child(div().w_full().h(px(1.)).bg(border))
            .child(
                div()
                    .pt_1()
                    .text_sm()
                    .font_medium()
                    .text_color(muted)
                    .child("Participants"),
            );

        // "Failed is not empty": a failed *initial* load must not read as an
        // empty roster with a live Add — and `ensure` declines to re-fetch once
        // a `Failed` cell exists, so the Retry is the only way back.
        if load_error.is_some() && !has_value {
            col = col.child(load_error_panel(
                "space/inspector/participants/retry",
                "Couldn't load this space's participants.",
                load_error.as_deref().unwrap_or_default(),
                cx,
                cx.listener(|this, _, _, cx| this.inspector_retry_participants(cx)),
            ));
            // A refused write and a failed re-list can stand at once (see
            // `stores::settle_mutation`); this branch owns the section, so it
            // carries the write's refusal rather than swallowing it.
            if !op_errors.is_empty() {
                col =
                    col.child(self.render_inspector_participants_error(&op_errors, &space_id, cx));
            }
            return Some(col.into_any_element());
        }

        let participants = self.inspector_participants(cx);
        if participants.is_empty() && loading {
            return Some(
                col.child(
                    div()
                        .text_xs()
                        .text_color(muted.opacity(0.8))
                        .child("Loading…"),
                )
                .into_any_element(),
            );
        }

        for p in &participants {
            col = col.child(self.render_inspector_participant(p, cx));
        }

        col = match self.inspector_participant_add.is_some() {
            true => col.child(self.render_inspector_add_form(cx)),
            false => col.child(
                div()
                    .id("space-inspector-participants-add")
                    .probe(
                        "space/inspector/participants/add",
                        gpui::Role::Button,
                        "Add participant",
                    )
                    .mt_1()
                    .cursor_pointer()
                    .text_sm()
                    .text_color(link)
                    .hover(move |s| s.text_color(fg))
                    .child("Add participant…")
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.inspector_begin_add_participant(window, cx)
                    })),
            ),
        };

        col = match (
            self.inspector_invite.is_some(),
            self.inspector_space_is_a_notebook(cx),
        ) {
            (true, _) => col.child(self.render_inspector_invite_form(cx)),
            // A notebook withholds the grant door — see
            // `inspector_space_is_a_notebook`.
            (false, true) => col,
            (false, false) => col.child(
                div()
                    .id("space-inspector-participants-invite")
                    .probe(
                        "space/inspector/participants/invite",
                        gpui::Role::Button,
                        "Invite an agent",
                    )
                    .cursor_pointer()
                    .text_sm()
                    .text_color(link)
                    .hover(move |s| s.text_color(fg))
                    .child("Invite an agent…")
                    .on_click(
                        cx.listener(|this, _, window, cx| this.inspector_begin_invite(window, cx)),
                    ),
            ),
        };

        col =
            match self.inspector_template_form.is_some() {
                true => col.child(self.render_inspector_template_form(cx)),
                false => col.child(
                    div()
                        .id("space-inspector-participants-template")
                        .probe(
                            "space/inspector/participants/template",
                            gpui::Role::Button,
                            "Save as template",
                        )
                        .cursor_pointer()
                        .text_xs()
                        .text_color(muted)
                        .hover(move |s| s.text_color(fg))
                        .child("Save these participants as a template…")
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.inspector_begin_template(window, cx)
                        })),
                ),
            };

        if !op_errors.is_empty() {
            col = col.child(self.render_inspector_participants_error(&op_errors, &space_id, cx));
        }

        // A failed *refresh* over rows we still hold: keep them, and say the
        // staleness out loud with a quiet retry.
        if load_error.is_some() {
            col = col.child(
                div()
                    .id("space-inspector-participants-retry")
                    .probe(
                        "space/inspector/participants/retry",
                        gpui::Role::Button,
                        "Retry",
                    )
                    .cursor_pointer()
                    .text_xs()
                    .text_color(link)
                    .hover(move |s| s.text_color(fg))
                    .child("Couldn't refresh — retry")
                    .on_click(cx.listener(|this, _, _, cx| this.inspector_retry_participants(cx))),
            );
        }

        Some(col.into_any_element())
    }

    /// One member: the disclosure row, plus its editor when open.
    fn render_inspector_participant(
        &self,
        p: &ParticipantInfo,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let (muted, border) = (theme.muted_foreground, theme.border);
        let expanded = self.inspector_editing_participant() == Some(p.id.as_str());
        let is_human = p.kind == "human";
        let is_referenced = p.source == "referenced";
        let pid = p.id.clone();

        // The identifying line under the name: for an agent, `model · backend`;
        // for the human it would be meaningless (people have no model), and its
        // absence is what says "person" on a row this compact. An open row drops
        // it — the editor's own Model field is two lines below, and one of them
        // would be repeating the other.
        let detail: Option<SharedString> = if is_human || expanded {
            None
        } else if let Some(model) = p.model_ref.as_deref() {
            let (name, backend) = model_display(&self.stores, model, cx);
            Some(format!("{name} · {backend}").into())
        } else {
            Some("no model set".into())
        };

        let row = h_flex()
            .id(SharedString::from(format!("space-inspector-p-{pid}")))
            .probe(
                SharedString::from(format!("space/inspector/participants/{pid}")),
                gpui::Role::Button,
                SharedString::from(p.label.clone()),
            )
            .aria_expanded(expanded)
            .w_full()
            .py_1p5()
            .gap_2()
            .items_start()
            .cursor_pointer()
            .hover(move |s| s.bg(theme.secondary.opacity(0.4)))
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
                                    .text_sm()
                                    .font_medium()
                                    .child(SharedString::from(p.label.clone())),
                            )
                            .when(is_referenced, |el| {
                                el.child(
                                    div()
                                        .px_1p5()
                                        .rounded_sm()
                                        .bg(theme.muted.opacity(0.5))
                                        .text_xs()
                                        .text_color(muted)
                                        .child("shared"),
                                )
                            }),
                    )
                    .when_some(detail, |el, detail| {
                        el.child(div().text_xs().text_color(muted).truncate().child(detail))
                    }),
            )
            // The chevron says the row opens; its state is on the row's own node
            // (`aria_expanded`), so this is decoration.
            .child(
                div()
                    .text_xs()
                    .text_color(muted)
                    .child(if expanded { "▾" } else { "▸" }),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.inspector_toggle_participant(&pid, window, cx)
            }));

        let mut wrap = v_flex()
            .w_full()
            .border_b_1()
            .border_color(border.opacity(0.5))
            .child(row);
        if expanded {
            wrap = wrap.child(self.render_inspector_participant_editor(p, cx));
        }
        wrap.into_any_element()
    }

    /// The open disclosure's body — the fork (when referenced), name, model,
    /// system prompt, notify policy, and the row's own verbs.
    fn render_inspector_participant_editor(
        &self,
        p: &ParticipantInfo,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        let Some(edit) = self.inspector_participant_edit.as_ref() else {
            return div().into_any_element();
        };
        let is_agent = edit.kind == "agent";
        // Two members are never removable: the shared "You", and — when this
        // space is an agent's **notebook** — that agent. The notebook exists
        // only for it and is where its `core` memory lives, so the membership is
        // structural and app-core refuses to end it. Withholding the affordance
        // is the courtesy; the refusal is the guarantee (task 36; Codex review,
        // PR #279). Until the space's settings have landed the space cannot be
        // known to be a notebook, and Remove is offered as it is anywhere else —
        // the honest degradation, since pressing it is refused rather than
        // obeyed.
        let can_remove = p.id != eidola_app_core::HUMAN_PARTICIPANT_ID
            && !self.inspector_participant_owns_this_notebook(&p.id, cx);
        let remove_id = p.id.clone();
        let subject = p.label.clone();
        // Only a **space-owned** agent can be shared: a referenced global
        // already is one, and promotion is one-way, so there is deliberately
        // nothing here that reads as "unshare".
        let can_share = is_agent && !edit.is_referenced;

        let mut card = v_flex().track_focus(&edit.focus).w_full().pb_3().gap_2();

        if edit.is_referenced {
            let mode = edit.mode;
            card = card
                .child(
                    h_flex()
                        .gap_2()
                        .child(mode_chip(
                            "space-inspector-p-everywhere".into(),
                            "space/inspector/participants/editor/mode/everywhere".into(),
                            "Everyone".into(),
                            mode == EditMode::Everywhere,
                            cx,
                            cx.listener(|this, _, window, cx| {
                                this.inspector_set_edit_mode(EditMode::Everywhere, window, cx)
                            }),
                        ))
                        .child(mode_chip(
                            "space-inspector-p-override".into(),
                            "space/inspector/participants/editor/mode/override".into(),
                            "This space only".into(),
                            mode == EditMode::OverrideHere,
                            cx,
                            cx.listener(|this, _, window, cx| {
                                this.inspector_set_edit_mode(EditMode::OverrideHere, window, cx)
                            }),
                        )),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(muted.opacity(0.8))
                        .child(match mode {
                            EditMode::Everywhere => {
                                "Editing the shared participant — changes apply everywhere it is used."
                            }
                            EditMode::OverrideHere => {
                                "Overriding just this space — the shared participant is unchanged."
                            }
                        }),
                );
        }

        card = card.child(field_label("Name", cx)).child(
            div()
                .id("space-inspector-p-name-wrap")
                .probe_bounds(
                    "space/inspector/participants/editor/name",
                    gpui::Role::TextInput,
                    "Name",
                )
                .child(Input::new(&edit.label).aria_label("Name")),
        );

        if is_agent {
            card = card
                .child(field_label("Model", cx))
                .child(self.render_inspector_participant_model(
                    edit.model_ref.as_deref(),
                    ParticipantPicker::Editor,
                    cx,
                ))
                .child(field_label("System prompt", cx))
                .child(
                    div()
                        .id("space-inspector-p-prompt-wrap")
                        .probe_bounds(
                            "space/inspector/participants/editor/system-prompt",
                            gpui::Role::TextInput,
                            "System prompt",
                        )
                        .child(Input::new(&edit.system_prompt).aria_label("System prompt")),
                )
                .child(field_label("Responds", cx))
                .child(self.render_inspector_notify(&edit.notify_policy, "editor", true, cx));
        }

        // Sharing gets its own line rather than a fourth seat in the verb row:
        // the panel is 320px, and a row of four verbs pushed Save off the edge.
        // It is also not a peer of Cancel/Save — those settle this edit, while
        // this one changes what the participant *is*.
        if can_share {
            card = card.child(match edit.promote_confirm {
                true => self.render_inspector_share_confirm(p, cx),
                false => h_flex()
                    .w_full()
                    .child(ghost_button_labeled(
                        "space-inspector-p-share".into(),
                        SharedString::from(format!("space/inspector/participants/{}/share", p.id)),
                        "Share this agent…",
                        format!("Share {subject} across spaces"),
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| this.inspector_begin_promote(window, cx)),
                    ))
                    .into_any_element(),
            });
        }

        card.child(
            h_flex()
                .w_full()
                .gap_2()
                .items_center()
                .justify_between()
                .child(div().when(can_remove, |el| {
                    el.child(ghost_button_labeled(
                        "space-inspector-p-remove".into(),
                        SharedString::from(format!(
                            "space/inspector/participants/{remove_id}/remove"
                        )),
                        "Remove",
                        format!("Remove {subject}"),
                        false,
                        cx,
                        cx.listener(move |this, _, window, cx| {
                            this.inspector_remove_participant(&remove_id, window, cx)
                        }),
                    ))
                }))
                .child(
                    h_flex()
                        .gap_2()
                        .child(ghost_button(
                            "space-inspector-p-cancel".into(),
                            "space/inspector/participants/editor/cancel".into(),
                            "Cancel",
                            false,
                            cx,
                            cx.listener(|this, _, window, cx| {
                                this.inspector_cancel_participant_edit(window, cx)
                            }),
                        ))
                        .child(ghost_button(
                            "space-inspector-p-save".into(),
                            "space/inspector/participants/editor/save".into(),
                            "Save",
                            true,
                            cx,
                            cx.listener(|this, _, window, cx| {
                                this.inspector_save_participant_edit(window, cx)
                            }),
                        )),
                ),
        )
        .into_any_element()
    }

    /// The share confirmation — what "Share this agent…" reveals in place.
    ///
    /// It says the two things a reader needs before an irreversible verb: that
    /// **nothing about this space changes** (promotion moves the row's
    /// ownership, not its configuration — the space keeps the agent as a member
    /// with empty overrides, so its persona here is preserved byte for byte),
    /// and that it **cannot be undone** (there is no demotion; retirement is the
    /// soft-remove).
    fn render_inspector_share_confirm(
        &self,
        p: &ParticipantInfo,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        let name = p.label.clone();
        let reassurance = format!(
            "{name} keeps this space's persona exactly as it is, and can then join other spaces. \
             You'll manage it in Settings → Agents. Sharing can't be undone."
        );
        v_flex()
            .id("space-inspector-p-share-confirm")
            .w_full()
            .p_2()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(
                div()
                    .id("space-inspector-p-share-note")
                    .probe(
                        SharedString::from(format!(
                            "space/inspector/participants/{}/share/note",
                            p.id
                        )),
                        gpui::Role::Label,
                        SharedString::from(reassurance.clone()),
                    )
                    .text_xs()
                    .text_color(muted)
                    .child(SharedString::from(reassurance)),
            )
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button_labeled(
                        "space-inspector-p-share-cancel".into(),
                        SharedString::from(format!(
                            "space/inspector/participants/{}/share/cancel",
                            p.id
                        )),
                        "Not now",
                        format!("Keep {name} in this space"),
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| {
                            this.inspector_cancel_promote(window, cx)
                        }),
                    ))
                    .child(ghost_button_labeled(
                        "space-inspector-p-share-confirm-button".into(),
                        SharedString::from(format!(
                            "space/inspector/participants/{}/share/confirm",
                            p.id
                        )),
                        "Share",
                        format!("Share {name} across spaces"),
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| {
                            this.inspector_confirm_promote(window, cx)
                        }),
                    )),
            )
            .into_any_element()
    }

    fn render_inspector_add_form(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let Some(add) = self.inspector_participant_add.as_ref() else {
            return div().into_any_element();
        };
        v_flex()
            .id("space-inspector-participants-add-form")
            .track_focus(&add.focus)
            .w_full()
            .p_2()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(field_label("Name", cx))
            .child(
                div()
                    .id("space-inspector-add-name-wrap")
                    .probe_bounds(
                        "space/inspector/participants/add/name",
                        gpui::Role::TextInput,
                        "Name",
                    )
                    .child(Input::new(&add.label).aria_label("Name")),
            )
            .child(field_label("Model", cx))
            .child(self.render_inspector_participant_model(
                add.model_ref.as_deref(),
                ParticipantPicker::Add,
                cx,
            ))
            .child(field_label("System prompt", cx))
            .child(
                div()
                    .id("space-inspector-add-prompt-wrap")
                    .probe_bounds(
                        "space/inspector/participants/add/system-prompt",
                        gpui::Role::TextInput,
                        "System prompt",
                    )
                    .child(Input::new(&add.system_prompt).aria_label("System prompt")),
            )
            .child(field_label("Responds", cx))
            .child(self.render_inspector_notify(&add.notify_policy, "add", false, cx))
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button(
                        "space-inspector-add-cancel".into(),
                        "space/inspector/participants/add/cancel".into(),
                        "Cancel",
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| {
                            this.inspector_cancel_add_participant(window, cx)
                        }),
                    ))
                    .child(ghost_button(
                        "space-inspector-add-submit".into(),
                        "space/inspector/participants/add/submit".into(),
                        "Add",
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| {
                            this.inspector_save_add_participant(window, cx)
                        }),
                    )),
            )
            .into_any_element()
    }

    /// The invite form: who could join, and — once one is chosen — what
    /// letting them in means.
    fn render_inspector_invite_form(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let Some(form) = self.inspector_invite.as_ref() else {
            return div().into_any_element();
        };
        let muted = theme.muted_foreground;
        let fg = theme.foreground;
        let mut col = v_flex()
            .id("space-inspector-invite-form")
            .track_focus(&form.focus)
            .probe(
                "space/inspector/participants/invite/form",
                gpui::Role::Group,
                "Invite an agent",
            )
            .w_full()
            .p_2()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3));

        if let Some((id, label, shared, home)) = form.confirming.clone() {
            let statement = grant_statement(&label, shared, home.as_deref());
            let verb = if shared {
                "Add as observer"
            } else {
                "Share and add"
            };
            let _ = id;
            return col
                .child(
                    div()
                        .id("space-inspector-invite-note")
                        .probe_value(
                            "space/inspector/participants/invite/note",
                            gpui::Role::Label,
                            "What this grant does",
                            statement.clone(),
                        )
                        .text_xs()
                        .text_color(muted)
                        .child(statement),
                )
                .child(
                    h_flex()
                        .gap_2()
                        .justify_end()
                        .child(ghost_button(
                            "space-inspector-invite-cancel".into(),
                            "space/inspector/participants/invite/cancel".into(),
                            "Not now",
                            false,
                            cx,
                            cx.listener(|this, _, window, cx| {
                                this.inspector_cancel_invite(window, cx)
                            }),
                        ))
                        .child(ghost_button(
                            "space-inspector-invite-confirm".into(),
                            "space/inspector/participants/invite/confirm".into(),
                            verb,
                            true,
                            cx,
                            cx.listener(|this, _, window, cx| {
                                this.inspector_confirm_invite(window, cx)
                            }),
                        )),
                )
                .into_any_element();
        }

        col = col.child(field_label("Invite an agent", cx));
        match form.candidates.as_ref() {
            // A read in flight knows nothing — and an unanswered list is not an
            // empty one (the Agents pane's rule).
            None => {
                col = col.child(
                    div()
                        .text_xs()
                        .text_color(muted.opacity(0.8))
                        .child("Loading…"),
                );
            }
            Some(candidates) if candidates.is_empty() => {
                let line = match &form.error {
                    Some(err) => err.clone(),
                    None => {
                        SharedString::from("Every agent you could invite already takes part here.")
                    }
                };
                col = col.child(div().text_xs().text_color(muted).child(line));
            }
            Some(candidates) => {
                for (i, candidate) in candidates.iter().enumerate() {
                    let line = match (&candidate.home_space_title, candidate.shared) {
                        (_, true) => SharedString::from(candidate.label.clone()),
                        (Some(home), false) => {
                            SharedString::from(format!("{} — from {home}", candidate.label))
                        }
                        (None, false) => SharedString::from(candidate.label.clone()),
                    };
                    let id = candidate.id.clone();
                    col = col.child(
                        div()
                            .id(SharedString::from(format!("space-inspector-invite-{i}")))
                            .probe(
                                format!("space/inspector/participants/invite/{i}"),
                                gpui::Role::Button,
                                line.clone(),
                            )
                            .w_full()
                            .px_1()
                            .py_0p5()
                            .rounded_sm()
                            .text_xs()
                            .truncate()
                            .cursor_pointer()
                            .hover(move |s| s.bg(theme.muted).text_color(fg))
                            .child(line)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.inspector_arm_invite(&id, window, cx)
                            })),
                    );
                }
            }
        }
        // The way out is offered in **every** state — empty and failed
        // included: a form whose only exit is a successful listing is a dead
        // end in exactly the state a reader most needs to leave it.
        col = col.child(h_flex().justify_end().child(ghost_button(
            "space-inspector-invite-close".into(),
            "space/inspector/participants/invite/cancel".into(),
            "Cancel",
            false,
            cx,
            cx.listener(|this, _, window, cx| this.inspector_cancel_invite(window, cx)),
        )));
        col.into_any_element()
    }

    fn render_inspector_template_form(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let Some(form) = self.inspector_template_form.as_ref() else {
            return div().into_any_element();
        };
        v_flex()
            .id("space-inspector-template-form")
            .track_focus(&form.focus)
            .w_full()
            .p_2()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .child(field_label("Template name", cx))
            .child(
                div()
                    .id("space-inspector-template-title-wrap")
                    .probe_bounds(
                        "space/inspector/participants/template/title",
                        gpui::Role::TextInput,
                        "Template name",
                    )
                    .child(Input::new(&form.title).aria_label("Template name")),
            )
            .child(
                h_flex()
                    .gap_2()
                    .justify_end()
                    .child(ghost_button(
                        "space-inspector-template-cancel".into(),
                        "space/inspector/participants/template/cancel".into(),
                        "Cancel",
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| {
                            this.inspector_cancel_template(window, cx)
                        }),
                    ))
                    .child(ghost_button(
                        "space-inspector-template-save".into(),
                        "space/inspector/participants/template/save".into(),
                        "Save template",
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| this.inspector_save_template(window, cx)),
                    )),
            )
            .into_any_element()
    }

    /// The model field, delegating to the shared picker widget.
    fn render_inspector_participant_model(
        &self,
        current: Option<&str>,
        target: ParticipantPicker,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let open = self.open_inspector_participant_picker() == Some(target);
        let probe_prefix: SharedString = match target {
            ParticipantPicker::Editor => "space/inspector/participants/editor/model".into(),
            ParticipantPicker::Add => "space/inspector/participants/add/model".into(),
        };
        model_field(
            &self.stores,
            current,
            open,
            probe_prefix,
            &self.inspector_participant_picker_scroll,
            cx,
            move |this, _, _, cx| this.inspector_toggle_participant_picker(target, cx),
            |id, this: &mut Self, cx| this.inspector_select_participant_model(id, cx),
        )
    }

    /// The three-state notify control ("Responds: when asked / to people / to
    /// everything").
    fn render_inspector_notify(
        &self,
        current: &str,
        scope: &'static str,
        is_edit: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let mut row = h_flex().gap_1().flex_wrap();
        for (value, label) in NOTIFY_POLICIES {
            row = row.child(mode_chip(
                SharedString::from(format!("space-inspector-notify-{scope}-{value}")),
                SharedString::from(format!(
                    "space/inspector/participants/{scope}/notify/{value}"
                )),
                SharedString::from(label),
                current == value,
                cx,
                cx.listener(move |this, _, _, cx| {
                    if is_edit {
                        this.inspector_set_edit_notify(value, cx);
                    } else {
                        this.inspector_set_add_notify(value, cx);
                    }
                }),
            ));
        }
        row.into_any_element()
    }

    /// The membership's write-refusal band — **one per space, listing every
    /// refusal that stands**.
    ///
    /// The store keys refusals per `(space, participant)`, so two independent
    /// writes can both be refused; a band that showed one would be the very
    /// defect keyed reports exist to prevent. But a band under each disclosure
    /// row is not available here the way it is in the Agents pane — this roster
    /// is a compact list of closed rows in a 320px panel — so the one band lists
    /// them, each line **naming its subject**, since the band no longer sits
    /// under the row it is about. A subject the roster cannot name (a refused
    /// *add* has no row yet) speaks for itself and the line stands bare.
    ///
    /// Dismissible for the same reason the panel's other two are: nothing else
    /// clears a refusal until the next write to that same row, so an
    /// unacknowledged one stands indefinitely. The × acknowledges what the band
    /// shows — all of it, matching the one band it dismisses — and never implies
    /// a write succeeded.
    fn render_inspector_participants_error(
        &self,
        errors: &[(String, String)],
        space_id: &str,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let id = space_id.to_string();
        // Name each subject from the roster the reader is looking at; a refused
        // *add* has no row yet, so its message stands bare.
        let roster = self.inspector_participants(cx);
        let lines: Vec<String> = errors
            .iter()
            .map(
                |(pid, message)| match roster.iter().find(|p| &p.id == pid) {
                    Some(p) => format!("{}: {message}", p.label),
                    None => message.clone(),
                },
            )
            .collect();
        let mut column = v_flex().flex_1().min_w_0().gap_1();
        for line in &lines {
            column = column.child(error_banner(line, cx));
        }
        h_flex()
            .id("space-inspector-participants-error")
            .probe(
                "space/inspector/participants/error",
                gpui::Role::Alert,
                lines.join(" · "),
            )
            .w_full()
            .gap_2()
            .items_start()
            .justify_between()
            .child(column)
            .child(
                div()
                    .id("space-inspector-participants-error-dismiss")
                    .probe(
                        "space/inspector/participants/error/dismiss",
                        gpui::Role::Button,
                        "Dismiss",
                    )
                    .cursor_pointer()
                    .text_color(theme.muted_foreground)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("×")
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let id = id.clone();
                        this.stores
                            .participants
                            .update(cx, |s, cx| s.clear_op_error(&id, cx));
                    })),
            )
            .into_any_element()
    }
}
