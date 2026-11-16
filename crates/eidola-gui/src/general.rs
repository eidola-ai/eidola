//! General settings pane — how the app looks, when it starts, and how a
//! terminal reaches it.
//!
//! Mostly the circadian **Appearance** axes (day/night, time of day, and the
//! fixed light character while the sun is off), then the one **Startup** row,
//! "Open at login", opt-in and never default, and on macOS the **Command
//! line** row, which links an `eidola` command on `PATH` to this bundle's CLI.
//! Those two rows are the only state in this pane that does not come from a
//! store — the system owns both (see [`crate::login_item`] and
//! [`crate::cli_link`]), so each is read at construction and re-read after
//! every write; the command-line link is also re-read whenever the pane is
//! shown, because the app moving is what leaves it dangling and no `Change`
//! announces that. Everything about the Eidola *connection* lives in Settings
//! → Backends → Eidola.
//!
//! Every string the pane draws is Fluent (`locales/*/general.ftl`), except its
//! *nav* label, which is an English literal beside the other six because the
//! nav row's probe name derives from it.

use eidola_app_core::config::{AppearanceSetting, LightCharacter, TimeOfDayTint};
use gpui::{
    AsyncApp, ClipboardItem, Context, Entity, FocusHandle, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription, Task,
    WeakEntity, Window, div, prelude::FluentBuilder,
};
use gpui_component::{
    ActiveTheme, Disableable, Sizable, StyledExt, h_flex, label::Label, switch::Switch, v_flex,
};
use std::path::{Path, PathBuf};

use crate::cli_link::{self, Blocker, Decision, InstallFailure, LinkState, Snapshot, Unavailable};
use crate::i18n::msg;
use crate::login_item::{self, LoginItemRefusal, LoginItemState};
use crate::participants::ghost_button_labeled;
use crate::probe::Probe as _;
use crate::stores::ConfigStore;

/// How the last command-line install ended, when it left something to say.
/// The value, not the words: the render chooses them, so a locale change
/// re-words it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliOutcome {
    /// The administrator prompt was dismissed; nothing changed.
    Declined,
    /// The install failed, in the words of whatever failed.
    Failed(String),
}

/// The command-line row's state. Present only where the row is shown (macOS,
/// or a test that stages one).
struct CliRow {
    /// The executable whose bundle is linked to, and the directory the link
    /// goes in — what every re-read reads.
    exe: PathBuf,
    dir: PathBuf,
    snapshot: Snapshot,
    /// An install in flight; `Some(admin)` while it runs.
    pending: Option<bool>,
    outcome: Option<CliOutcome>,
    /// The verb row, which the press unmounts (the verb becomes the pending
    /// line) — so the press asks it whether it held the keyboard.
    verb_focus: FocusHandle,
    task: Option<Task<()>>,
}

impl CliRow {
    fn new(exe: PathBuf, dir: PathBuf, cx: &gpui::App) -> Self {
        let snapshot = cli_link::snapshot_for(&exe, &dir);
        Self {
            exe,
            dir,
            snapshot,
            pending: None,
            outcome: None,
            verb_focus: cx.focus_handle(),
            task: None,
        }
    }

    fn reread(&mut self) {
        self.snapshot = cli_link::snapshot_for(&self.exe, &self.dir);
    }
}

pub struct GeneralView {
    config: Entity<ConfigStore>,
    /// The system's answer about our login item. Not store-backed — macOS
    /// owns this state and a second copy would be a second answer (see
    /// `crate::login_item`). Read once here and again after every write.
    login_item: LoginItemState,
    /// Why the last register/unregister was refused, until the next attempt.
    login_item_error: Option<LoginItemRefusal>,
    /// The pane's own handle: where the keyboard goes when a verb it was on
    /// unmounts. Carries a `Region` role, because the adapter reports focus
    /// only on a node the a11y tree has.
    focus_handle: FocusHandle,
    cli: Option<CliRow>,
    _subscriptions: Vec<Subscription>,
}

impl GeneralView {
    pub fn new(config: Entity<ConfigStore>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let _subscriptions = vec![cx.observe(&config, |_, _, cx| cx.notify())];

        // The command line needs a way onto `PATH` only on macOS: the `.deb`
        // installs `/usr/bin/eidola` and a Nix profile carries `bin/eidola`.
        let cli = if cfg!(target_os = "macos") {
            std::env::current_exe()
                .ok()
                .map(|exe| CliRow::new(exe, PathBuf::from(cli_link::LINK_DIR), cx))
        } else {
            None
        };

        Self {
            config,
            login_item: login_item::state(),
            login_item_error: None,
            focus_handle: cx.focus_handle(),
            cli,
            _subscriptions,
        }
    }

    /// The login item as the system last reported it (test accessor).
    pub fn login_item(&self) -> LoginItemState {
        self.login_item
    }

    pub fn login_item_error(&self) -> Option<&LoginItemRefusal> {
        self.login_item_error.as_ref()
    }

    /// The command-line link as last read, where the row is shown.
    pub fn cli_link(&self) -> Option<&Snapshot> {
        self.cli.as_ref().map(|row| &row.snapshot)
    }

    pub fn cli_outcome(&self) -> Option<&CliOutcome> {
        self.cli.as_ref().and_then(|row| row.outcome.as_ref())
    }

    pub fn cli_install_pending(&self) -> bool {
        self.cli.as_ref().is_some_and(|row| row.pending.is_some())
    }

    /// Point the command-line row at another executable and link directory —
    /// a temporary bundle and directory, so a test exercises the real row and
    /// the real write without touching `/usr/local/bin`.
    pub fn stage_cli_link_for_test(&mut self, exe: &Path, dir: &Path, cx: &mut Context<Self>) {
        self.cli = Some(CliRow::new(exe.to_path_buf(), dir.to_path_buf(), cx));
        cx.notify();
    }

    /// Land an install's result as its own task does. Public so tests can
    /// stage an outcome (a declined prompt) that only a human can cause.
    pub fn finish_cli_install(
        &mut self,
        result: Result<(), InstallFailure>,
        cx: &mut Context<Self>,
    ) {
        let Some(row) = self.cli.as_mut() else {
            return;
        };
        row.pending = None;
        row.task = None;
        row.outcome = match result {
            Ok(()) => None,
            Err(InstallFailure::Declined) => Some(CliOutcome::Declined),
            Err(InstallFailure::Failed(reason)) => Some(CliOutcome::Failed(reason)),
            // Something appeared at the path that is not ours; the re-read
            // names it, which is the whole of what there is to say.
            Err(InstallFailure::Refused) => None,
        };
        // **The write's result is never assumed** — the link on disk is the
        // answer.
        row.reread();
        cx.notify();
    }

    /// The pane has been shown. The link is re-read because the event that
    /// leaves it dangling — the app moving — is announced by nothing.
    pub fn pane_activated(&mut self, cx: &mut Context<Self>) {
        if let Some(row) = self.cli.as_mut()
            && row.pending.is_none()
        {
            row.reread();
            cx.notify();
        }
    }

    /// Install or update the `eidola` link. Runs the write on a thread of its
    /// own — the privileged route waits on an administrator's password for as
    /// long as the reader takes — and lands the answer in this view's task
    /// slot, so it dies with the window (the write itself completes either
    /// way, and the next read sees it).
    pub fn install_cli(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(row) = self.cli.as_mut() else {
            return;
        };
        if row.pending.is_some() {
            return;
        }
        let Decision::Install { admin, .. } = row.snapshot.decision() else {
            return;
        };
        let Some(source) = row.snapshot.source.clone() else {
            return;
        };
        // The verb is replaced by the pending line, so the press hands the
        // keyboard back from the row it unmounts.
        if row.verb_focus.contains_focused(window, cx) {
            self.focus_handle.focus(window, cx);
        }
        let row = self.cli.as_mut().expect("checked above");
        row.pending = Some(admin);
        row.outcome = None;
        let dir = row.dir.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            let _ = tx.send(cli_link::install(
                &source,
                &dir,
                cli_link::Route::for_admin(admin),
            ));
        });
        row.task = Some(
            cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                let result = rx.await.unwrap_or_else(|_| {
                    Err(InstallFailure::Failed("the install was interrupted".into()))
                });
                let _ = this.update(cx, |this, cx| this.finish_cli_install(result, cx));
            }),
        );
        cx.notify();
    }

    /// Turn "Open at login" on or off. **The write's result is never
    /// assumed** — a refusal (unsigned bundle, a user-level denial) leaves
    /// the system exactly as it was, so the row re-reads the system rather
    /// than flipping to what was asked for.
    pub fn set_open_at_login(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.login_item_error = login_item::set(enabled).err();
        self.login_item = login_item::state();
        cx.notify();
    }

    /// Circadian day/night axis. Writes through the store; the theme
    /// re-applies via its config observation (`theme::wire_config`).
    pub fn set_appearance(&mut self, appearance: AppearanceSetting, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_appearance(appearance, cx));
        cx.notify();
    }

    /// Circadian time-of-day axis.
    pub fn set_time_of_day_tint(&mut self, tint: TimeOfDayTint, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_time_of_day_tint(tint, cx));
        cx.notify();
    }

    /// Circadian fixed light character (shown only while the time-of-day
    /// axis is off).
    pub fn set_light_character(&mut self, character: LightCharacter, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_light_character(character, cx));
        cx.notify();
    }

    /// Text size — the same type-scale ladder as View → Zoom In / Zoom Out /
    /// Actual Size, surfaced here as a visible control (the menu shortcuts stay
    /// the fast path, but a settings row is where an older user looks first).
    pub fn zoom_in(&mut self, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| c.zoom_in(cx));
        cx.notify();
    }

    pub fn zoom_out(&mut self, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| c.zoom_out(cx));
        cx.notify();
    }

    pub fn reset_zoom(&mut self, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| c.reset_zoom(cx));
        cx.notify();
    }
}

/// The muted line under the login-item switch.
pub fn login_item_copy(state: LoginItemState, cx: &gpui::App) -> SharedString {
    match state {
        LoginItemState::Off => msg::general_login_item_off(cx),
        LoginItemState::On => msg::general_login_item_on(cx),
        LoginItemState::NeedsApproval => msg::general_login_item_needs_approval(cx),
        LoginItemState::Unsupported if cfg!(target_os = "macos") => {
            msg::general_login_item_unsupported(cx)
        }
        LoginItemState::Unsupported => msg::general_login_item_unsupported_linux(cx),
    }
}

/// Why a login-item write was refused. macOS's own words are already in the
/// reader's language and are quoted, never re-worded.
pub fn login_item_refusal_copy(refusal: &LoginItemRefusal, cx: &gpui::App) -> SharedString {
    match refusal {
        LoginItemRefusal::TooOld => msg::general_login_item_too_old(cx),
        LoginItemRefusal::NotHere => msg::general_login_item_not_here(cx),
        LoginItemRefusal::System(words) => SharedString::from(words.clone()),
    }
}

/// What the command-line row says about the link.
pub fn cli_state_copy(snapshot: &Snapshot, cx: &gpui::App) -> SharedString {
    let path = snapshot.link().display().to_string();
    let dir = snapshot.dir.display().to_string();
    match &snapshot.state {
        LinkState::NotInstalled => msg::general_cli_not_installed(cx, dir),
        LinkState::Installed => msg::general_cli_installed(cx, path),
        LinkState::Stale {
            target,
            dangling: true,
        } => msg::general_cli_dangling(cx, path, target.display().to_string()),
        LinkState::Stale {
            target,
            dangling: false,
        } => msg::general_cli_other_copy(cx, path, target.display().to_string()),
        LinkState::Occupied(Blocker::File) => msg::general_cli_occupied_file(cx, path),
        LinkState::Occupied(Blocker::Directory) => msg::general_cli_occupied_directory(cx, path),
        LinkState::Occupied(Blocker::Link(target)) => {
            msg::general_cli_occupied_link(cx, path, target.display().to_string())
        }
        LinkState::Occupied(Blocker::Other) => msg::general_cli_occupied_other(cx, path),
        LinkState::Unreadable(reason) => msg::general_cli_unreadable(cx, path, reason.clone()),
        LinkState::Unavailable(Unavailable::NotInBundle) => msg::general_cli_not_bundled(cx),
        LinkState::Unavailable(Unavailable::Translocated) => msg::general_cli_translocated(cx),
        LinkState::Unavailable(Unavailable::CliMissing) => msg::general_cli_missing(cx),
    }
}

impl Render for GeneralView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let store = self.config.read(cx);
        let state = store.state().cloned();
        let error = store.error().map(|e| e.to_string());

        let mut col = v_flex()
            .id("general-pane")
            .track_focus(&self.focus_handle)
            // A handback target must carry a role, or the adapter has no node
            // to report focus on (`settings/backends/pane`'s rule).
            .probe("settings/general/pane", gpui::Role::Region, "General")
            .px_6()
            .py_5()
            .gap_4()
            .w_full();

        // --- Appearance: the circadian theme's two axes -----------------
        col = col.child(section_header(msg::general_appearance(cx), cx));

        if let Some(s) = state.as_ref() {
            let mut day_night = h_flex().gap_2();
            for (value, id, probe_name, label) in [
                (
                    AppearanceSetting::Auto,
                    "appearance-auto",
                    "settings/general/appearance/auto",
                    msg::general_day_night_auto(cx),
                ),
                (
                    AppearanceSetting::System,
                    "appearance-system",
                    "settings/general/appearance/system",
                    msg::general_day_night_system(cx),
                ),
                (
                    AppearanceSetting::Day,
                    "appearance-day",
                    "settings/general/appearance/day",
                    msg::general_day_night_day(cx),
                ),
                (
                    AppearanceSetting::Night,
                    "appearance-night",
                    "settings/general/appearance/night",
                    msg::general_day_night_night(cx),
                ),
            ] {
                let active = s.appearance == value;
                day_night = day_night.child(
                    choice_chip(id, label.clone(), active, cx)
                        // The chips read "Auto"/"System"/"Day"/"Night"; the
                        // group label that gives them meaning is a plain,
                        // node-less `div`, so the name has to carry it.
                        .probe(
                            probe_name,
                            gpui::Role::Button,
                            msg::general_day_night_choice(cx, label.to_string()),
                        )
                        .aria_selected(active)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.set_appearance(value, cx);
                        })),
                );
            }
            col = col.child(field_row(
                msg::general_day_night(cx),
                cx,
                v_flex()
                    .gap_1()
                    .child(day_night)
                    .child(note(msg::general_day_night_note(cx), cx)),
            ));

            let mut tint_row = h_flex().gap_2();
            for (value, id, probe_name, label) in [
                (
                    TimeOfDayTint::On,
                    "time-of-day-on",
                    "settings/general/time-of-day/on",
                    msg::general_on(cx),
                ),
                (
                    TimeOfDayTint::Off,
                    "time-of-day-off",
                    "settings/general/time-of-day/off",
                    msg::general_off(cx),
                ),
            ] {
                let active = s.time_of_day_tint == value;
                tint_row = tint_row.child(
                    choice_chip(id, label.clone(), active, cx)
                        .probe(
                            probe_name,
                            gpui::Role::Button,
                            msg::general_time_of_day_choice(cx, label.to_string()),
                        )
                        .aria_selected(active)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.set_time_of_day_tint(value, cx);
                        })),
                );
            }
            col = col.child(field_row(
                msg::general_time_of_day(cx),
                cx,
                v_flex()
                    .gap_1()
                    .child(tint_row)
                    .child(note(msg::general_time_of_day_note(cx), cx)),
            ));

            // With the sun turned off, the character becomes a fixed choice.
            if s.time_of_day_tint == TimeOfDayTint::Off {
                let mut light_row = h_flex().gap_2();
                for (value, id, probe_name, label) in [
                    (
                        LightCharacter::Cool,
                        "light-character-cool",
                        "settings/general/light-character/cool",
                        msg::general_light_cool(cx),
                    ),
                    (
                        LightCharacter::Neutral,
                        "light-character-neutral",
                        "settings/general/light-character/neutral",
                        msg::general_light_neutral(cx),
                    ),
                    (
                        LightCharacter::Warm,
                        "light-character-warm",
                        "settings/general/light-character/warm",
                        msg::general_light_warm(cx),
                    ),
                ] {
                    let active = s.light_character == value;
                    light_row = light_row.child(
                        choice_chip(id, label.clone(), active, cx)
                            .probe(
                                probe_name,
                                gpui::Role::Button,
                                msg::general_light_choice(cx, label.to_string()),
                            )
                            .aria_selected(active)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.set_light_character(value, cx);
                            })),
                    );
                }
                col = col.child(field_row(
                    msg::general_light(cx),
                    cx,
                    v_flex()
                        .gap_1()
                        .child(light_row)
                        .child(note(msg::general_light_note(cx), cx)),
                ));
            }

            // --- Text size: the type-scale ladder (also View → Zoom …) -------
            let at_min = s.font_scale <= eidola_app_core::config::FONT_SCALE_MIN + 1e-3;
            let at_max = s.font_scale >= eidola_app_core::config::FONT_SCALE_MAX - 1e-3;
            let percent = format!("{}%", (s.font_scale * 100.0).round() as i32);
            let size_row = h_flex()
                .gap_2()
                .items_center()
                .child(
                    // "A−" / "A+" are glyphs, not words: the same in every
                    // locale. Their accessible names say what they do.
                    choice_chip("text-size-smaller", "A−".into(), false, cx)
                        // The current scale renders as a bare `div` with no
                        // node of its own, so the chips carry it — otherwise
                        // "smaller" is announced with nothing to be smaller
                        // than.
                        .probe(
                            "settings/general/text-size/smaller",
                            gpui::Role::Button,
                            msg::general_text_size_smaller_name(cx, percent.clone()),
                        )
                        .when(at_min, |el| el.opacity(0.4))
                        .on_click(cx.listener(|this, _, _, cx| this.zoom_out(cx))),
                )
                .child(
                    choice_chip("text-size-larger", "A+".into(), false, cx)
                        .probe(
                            "settings/general/text-size/larger",
                            gpui::Role::Button,
                            msg::general_text_size_larger_name(cx, percent.clone()),
                        )
                        .when(at_max, |el| el.opacity(0.4))
                        .on_click(cx.listener(|this, _, _, cx| this.zoom_in(cx))),
                )
                .child(
                    choice_chip(
                        "text-size-reset",
                        msg::general_text_size_reset(cx),
                        false,
                        cx,
                    )
                    .probe(
                        "settings/general/text-size/reset",
                        gpui::Role::Button,
                        msg::general_text_size_reset_name(cx),
                    )
                    .on_click(cx.listener(|this, _, _, cx| this.reset_zoom(cx))),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(percent)),
                );
            col = col.child(field_row(
                msg::general_text_size(cx),
                cx,
                v_flex()
                    .gap_1()
                    .child(size_row)
                    .child(note(msg::general_text_size_note(cx), cx)),
            ));
        }

        // --- Startup: the opt-in login item -----------------------------
        col = col.child(section_header(msg::general_startup(cx), cx));
        col = col.child(field_row(
            msg::general_open_at_login(cx),
            cx,
            v_flex()
                .gap_1()
                .child(self.login_item_toggle(cx))
                .child(note(login_item_copy(self.login_item, cx), cx))
                .when_some(self.login_item_error.clone(), |el, refusal| {
                    let words = login_item_refusal_copy(&refusal, cx);
                    el.child(
                        div()
                            .id("login-item-error")
                            .probe(
                                "settings/general/login-item/error",
                                gpui::Role::Alert,
                                words.clone(),
                            )
                            .child(error_banner(words, cx)),
                    )
                }),
        ));

        // --- Command line: the `eidola` link on PATH (macOS) -------------
        if let Some(row) = self.render_cli(cx) {
            col = col
                .child(section_header(msg::general_command_line(cx), cx))
                .child(row);
        }

        if let Some(err) = error {
            col = col.child(
                div()
                    .id("general-error")
                    .probe("settings/general/error", gpui::Role::Alert, err.clone())
                    .child(error_banner(err.into(), cx)),
            );
        }

        col
    }
}

impl GeneralView {
    /// The command-line row: the link's state, the verb that would change it
    /// (or the reason there is none), and — after a declined prompt or a
    /// failure — the Terminal command that does the same thing.
    fn render_cli(&self, cx: &Context<Self>) -> Option<gpui::AnyElement> {
        let row = self.cli.as_ref()?;
        let theme = cx.theme();
        let snapshot = &row.snapshot;
        let decision = snapshot.decision();
        let status = cli_state_copy(snapshot, cx);

        let mut body = v_flex().gap_1().child(
            div()
                .id("cli-status")
                .probe(
                    "settings/general/cli/status",
                    gpui::Role::Label,
                    status.clone(),
                )
                .text_sm()
                .text_color(theme.foreground)
                .child(status),
        );

        let link_path = snapshot.link().display().to_string();
        let dir = snapshot.dir.display().to_string();
        let verb = match (&decision, row.pending) {
            // **A control over a settled decision is not a control**: while
            // the install runs, the verb is replaced by what is happening — no
            // id, no probe, so no tab stop and nothing to press twice.
            (_, Some(admin)) => Some(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(if admin {
                        msg::general_cli_waiting(cx)
                    } else {
                        msg::general_cli_installing(cx)
                    })
                    .into_any_element(),
            ),
            (Decision::Install { replace, admin }, None) => {
                let (probe_name, label, name) = match (replace, admin) {
                    (false, false) => (
                        "settings/general/cli/install",
                        msg::general_cli_install(cx),
                        msg::general_cli_install_name(cx, dir.clone()),
                    ),
                    (false, true) => (
                        "settings/general/cli/install",
                        msg::general_cli_install_admin(cx),
                        msg::general_cli_install_name(cx, dir.clone()),
                    ),
                    (true, false) => (
                        "settings/general/cli/update",
                        msg::general_cli_update(cx),
                        msg::general_cli_update_name(cx, link_path.clone()),
                    ),
                    (true, true) => (
                        "settings/general/cli/update",
                        msg::general_cli_update_admin(cx),
                        msg::general_cli_update_name(cx, link_path.clone()),
                    ),
                };
                Some(
                    ghost_button_labeled(
                        SharedString::from("cli-install"),
                        SharedString::from(probe_name),
                        label,
                        name,
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| this.install_cli(window, cx)),
                    )
                    .into_any_element(),
                )
            }
            (Decision::Nothing | Decision::Refuse(_), None) => None,
        };
        if let Some(verb) = verb {
            if matches!(decision, Decision::Install { admin: true, .. }) && row.pending.is_none() {
                body = body.child(note(msg::general_cli_needs_admin(cx), cx));
            }
            body = body.child(
                h_flex()
                    .id("cli-verb-row")
                    .track_focus(&row.verb_focus)
                    .gap_2()
                    .child(verb),
            );
        }

        // What to say after an install that did not happen — only while the
        // verb is still the way forward, so the Terminal command and the state
        // it would change cannot disagree.
        if let (Some(outcome), Some(command)) = (&row.outcome, snapshot.manual_command()) {
            let lead = match outcome {
                CliOutcome::Declined => msg::general_cli_declined(cx),
                CliOutcome::Failed(reason) => {
                    let failed = msg::general_cli_failed(cx, reason.clone());
                    body = body.child(
                        div()
                            .id("cli-failed")
                            .probe(
                                "settings/general/cli/error",
                                gpui::Role::Alert,
                                failed.clone(),
                            )
                            .child(error_banner(failed, cx)),
                    );
                    msg::general_cli_manual(cx)
                }
            };
            let copy = command.clone();
            body = body.child(
                v_flex()
                    .id("cli-manual")
                    .probe_value(
                        "settings/general/cli/manual",
                        gpui::Role::Label,
                        lead.clone(),
                        command.clone(),
                    )
                    .gap_1()
                    .px_3()
                    .py_2()
                    .rounded_md()
                    .bg(theme.secondary.opacity(0.4))
                    .child(Label::new(lead).text_xs())
                    .child(
                        div()
                            .font_family("Menlo")
                            .text_xs()
                            .text_color(theme.foreground)
                            .child(command),
                    )
                    .child(h_flex().child(ghost_button_labeled(
                        SharedString::from("cli-manual-copy"),
                        SharedString::from("settings/general/cli/manual/copy"),
                        msg::general_cli_copy(cx),
                        msg::general_cli_copy_name(cx),
                        false,
                        cx,
                        move |_, _, cx: &mut gpui::App| {
                            cx.write_to_clipboard(ClipboardItem::new_string(copy.clone()));
                        },
                    ))),
            );
        }

        Some(field_row(msg::general_cli_field(cx), cx, body).into_any_element())
    }

    /// The "Open at login" switch. `Switch` is a focus-bearing control: the
    /// widget is the accessible node (role, toggled state, its own focus
    /// handle and activation), named here because the "Open at login" field
    /// label is a node-less `div`. The wrapper is a bounds-only probe. An
    /// unsettable state disables the widget, which also takes it out of the
    /// tab order.
    fn login_item_toggle(&self, cx: &Context<Self>) -> impl IntoElement + use<> {
        let label = msg::general_login_item_switch(cx);
        let name = msg::general_login_item_name(cx);
        let state = self.login_item;
        let settable = state.is_settable();
        let switch = Switch::new("login-item-switch")
            .accessibility_label(name.clone())
            .small()
            .checked(state.is_on())
            .disabled(!settable);
        div()
            .id("login-item")
            .probe_bounds("settings/general/login-item", gpui::Role::Switch, name)
            .child(if settable {
                switch
                    .label(label)
                    .on_click(cx.listener(move |this, checked: &bool, _, cx| {
                        this.set_open_at_login(*checked, cx);
                    }))
                    .into_any_element()
            } else {
                // A disabled widget fades its track but not its label, so the
                // label is laid out beside it (the widget's own label row:
                // 8px gap, small text on the small track's 16px line) and
                // faded here.
                h_flex()
                    .gap_2()
                    .items_start()
                    .child(switch)
                    .child(
                        div()
                            .line_height(gpui::px(16.))
                            .text_sm()
                            .text_color(cx.theme().foreground.opacity(0.5))
                            .child(label),
                    )
                    .into_any_element()
            })
    }
}

fn section_header(label: SharedString, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_color(theme.muted_foreground)
        .text_sm()
        .font_medium()
        .child(label)
}

/// The muted explanatory line under a control.
fn note(text: SharedString, cx: &gpui::App) -> impl IntoElement {
    div()
        .text_xs()
        .text_color(cx.theme().muted_foreground.opacity(0.8))
        .child(text)
}

fn field_row<C: IntoElement>(label: SharedString, cx: &gpui::App, child: C) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .w_full()
        .gap_4()
        .py_1()
        .items_start()
        .child(
            div()
                .w(gpui::px(144.))
                .flex_none()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(label),
        )
        .child(div().flex_1().min_w_0().child(child))
}

/// One selectable option in a small chip row (the appearance settings).
/// The active chip gets the sidebar-accent pill, matching the settings nav.
fn choice_chip(
    id: &'static str,
    label: SharedString,
    active: bool,
    cx: &gpui::App,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    let el = div()
        .id(id)
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
    el.child(label)
}

fn error_banner(message: SharedString, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .gap_2()
        .px_3()
        .py_2()
        .rounded_md()
        .bg(theme.danger.opacity(0.08))
        .text_color(theme.danger)
        .child(Label::new(message))
}
