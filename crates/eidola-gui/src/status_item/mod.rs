//! The macOS status item and the background layer behind it (task 17,
//! waves 3 and 3b).
//!
//! Wave 2 made the process outlive its windows. Wave 3 gave that process a
//! face — a menu-bar item that is always there. Wave 3b decided **what the
//! face is for**, and that decision is the whole of this module's policy.
//!
//! ## The model: LM Studio / 1Password, not "menu-bar app"
//!
//! Wave 3 flipped `Regular ⇄ Accessory` on the window count: the Dock icon
//! left when the last window closed. Living with it showed that to be the
//! wrong call — closing the last window dropped the app out of the Dock *and*
//! the menu bar, so the reflexive ⌘N landed in whatever app was now
//! frontmost. So (Mike, 2026-08-03):
//!
//! - **While the app is open it is an ordinary macOS app** — `Regular` even
//!   with zero windows. Closing the last window changes nothing: Dock icon,
//!   menu bar, ⌘N. There is no window-count-driven flip left.
//! - **⌘Q retires to the background**: close every window, go `Accessory`
//!   (the Dock indicator goes), and keep the process, the status item, the
//!   stores, the bus bridge, and — the point of task 17 — the loaded engines
//!   running. [`retire_to_background`].
//! - **Full shutdown is the status menu's "Quit Eidola"**, which also carries
//!   the ⌘Q key equivalent on its own `NSMenuItem`. That is wave 2's
//!   teardown path, engines included; it is the only thing that calls
//!   `cx.quit()`.
//!
//! ## The safety gate, and the one decision that encodes it
//!
//! An `Accessory` app has no Dock icon and no menu bar of its own. With zero
//! windows and no status item it is invisible and unquittable except through
//! Activity Monitor — so **retiring is gated on the status item existing**.
//! [`quit_intent`] is that whole decision, and its second input is why ⌘Q is
//! safe from either side:
//!
//! - **no status item ⇒ full shutdown.** There is no background layer to
//!   retire into, so ⌘Q keeps the meaning it has always had rather than
//!   leaving an invisible process.
//! - **already retired ⇒ full shutdown.** The only way ⌘Q reaches an
//!   `Accessory` app is while its status menu is open, and there the answer
//!   must be "quit" whichever menu AppKit resolves the key equivalent
//!   against. Encoding it here means we do not depend on AppKit's ordering
//!   between an open `NSStatusItem` menu and the app's (unshown) main menu:
//!   both routes land on a full shutdown.
//!
//! ## One choke point for the way back
//!
//! [`window_will_open`] is called from `lib.rs::base_window_options` — the
//! one place every `cx.open_window` passes through — and asserts `Regular`.
//! Putting it there rather than in each `open_*_window` makes "open a window
//! while `Accessory`" unrepresentable, which matters because such a window
//! would sit under some other app's menu bar with no ⌘N. It no longer
//! consults a window count; it is now simply *the* door out of the
//! background state, shared by the status menu's Open / New Space, Spotlight,
//! `open -a`, and `Application::on_reopen`.
//!
//! ## The menu's moment of truth
//!
//! The rows are a pure projection of the `LocalModelsStore` snapshot
//! ([`menu_rows`]), mirrored into the platform object by a store observer and
//! rebuilt into real `NSMenuItem`s in `menuNeedsUpdate:` — AppKit's
//! "the user is about to see this" callback. Neither end captures authority:
//! the mirror is refreshed by the store's own notify (never sampled once at
//! install), and the menu is materialised at open (never rebuilt on every
//! download-progress tick, which arrives several times a second). See
//! `macos::Target` for why the mirror exists rather than a live read from
//! inside the AppKit callback.

#[cfg(target_os = "macos")]
mod macos;

use eidola_app_core::{LocalModelStatus, LocalModelsState};
use gpui::App;

use crate::lifecycle::LaunchOptions;
use crate::stores::Stores;

/// How the app presents itself to the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationPolicy {
    /// Dock icon, menu bar — an ordinary app.
    Regular,
    /// Menu-bar-only: no Dock icon, no app menu bar. Survivable *only*
    /// because the status item is there.
    Accessory,
}

/// What ⌘Q — the app's own Quit action — should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuitIntent {
    /// Close every window and drop to `Accessory`, keeping the process, the
    /// status item, the stores and the loaded engines.
    Retire,
    /// The wave-2 teardown: engines drained, `cx.quit()`, process gone.
    FullShutdown,
}

/// The one quit decision, as a pure function. See the module docs for why
/// both inputs mean "full shutdown".
pub fn quit_intent(status_item_present: bool, already_retired: bool) -> QuitIntent {
    if status_item_present && !already_retired {
        QuitIntent::Retire
    } else {
        QuitIntent::FullShutdown
    }
}

/// What a status-menu command does. Each maps onto an existing app path —
/// the status menu opens no doors of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCommand {
    /// The same reactivation a Dock click performs: focus a window, or open
    /// the Library when there is none.
    Open,
    /// ⌘N.
    NewSpace,
    /// The **full** shutdown — engines included (`crate::actions::QuitApp`),
    /// as against the app's ⌘Q, which now retires to the background. This is
    /// the only door that ends the process from the UI.
    Quit,
}

impl StatusCommand {
    pub fn title(self) -> &'static str {
        match self {
            Self::Open => "Open Eidola",
            Self::NewSpace => "New Space",
            Self::Quit => "Quit Eidola",
        }
    }

    /// The item's `keyEquivalent` (⌘ implied — AppKit's default modifier
    /// mask, which we set explicitly anyway).
    ///
    /// **Quit carries ⌘Q on the status menu itself**, which is what makes
    /// "⌘Q while the toolbar app has focus" a full shutdown: an
    /// `NSStatusItem` menu is not part of the main menu, so this equivalent
    /// is reachable only while that menu is open — precisely the moment the
    /// user means the background layer rather than a window.
    pub fn key_equivalent(self) -> &'static str {
        match self {
            Self::Quit => "q",
            Self::Open | Self::NewSpace => "",
        }
    }

    /// The `NSMenuItem` tag carrying this command back to the click handler.
    pub fn tag(self) -> isize {
        match self {
            Self::Open => 0,
            Self::NewSpace => 1,
            Self::Quit => 2,
        }
    }

    pub fn from_tag(tag: isize) -> Option<Self> {
        match tag {
            0 => Some(Self::Open),
            1 => Some(Self::NewSpace),
            2 => Some(Self::Quit),
            _ => None,
        }
    }
}

/// One row of the status menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusRow {
    Command(StatusCommand),
    /// A read-only readout (the engine lines). Rendered disabled — it is a
    /// label, not an affordance.
    Info(String),
    Separator,
}

/// The whole status menu, from the local-models snapshot.
///
/// `engines` is the `LocalModelsStore`'s loaded value — `None` while the
/// first refresh is still in flight (or when there is no backend at all, as
/// in the stub stores), which the engine section says honestly rather than
/// claiming nothing is running.
pub fn menu_rows(engines: Option<&LocalModelsState>) -> Vec<StatusRow> {
    let mut rows = vec![
        StatusRow::Command(StatusCommand::Open),
        StatusRow::Command(StatusCommand::NewSpace),
        StatusRow::Separator,
    ];
    rows.extend(engine_lines(engines).into_iter().map(StatusRow::Info));
    rows.push(StatusRow::Separator);
    rows.push(StatusRow::Command(StatusCommand::Quit));
    rows
}

/// The engine section: one line per engine that is up or coming up, across
/// the managed store *and* every configured `llamacpp` backend (an engine is
/// an engine — the menu is not a backend registry).
pub fn engine_lines(engines: Option<&LocalModelsState>) -> Vec<String> {
    let Some(state) = engines else {
        return vec!["Checking on-device engines…".to_string()];
    };
    let all = state
        .models
        .iter()
        .chain(state.external.iter().flat_map(|b| b.models.iter()));
    let lines: Vec<String> = all
        .filter_map(|m| match m.status {
            LocalModelStatus::Loaded { .. } => Some(format!("{} — running", m.display_name)),
            LocalModelStatus::Loading => Some(format!("{} — loading…", m.display_name)),
            _ => None,
        })
        .collect();
    if lines.is_empty() {
        vec!["No models running".to_string()]
    } else {
        lines
    }
}

/// Create the status item. Called once at launch, after the action handlers
/// exist (the menu dispatches them) and before the first window opens.
///
/// A no-op off macOS: Linux tray support is deliberately deferred (see the
/// task-17 discussion — StatusNotifierItem is fragmented and never
/// load-bearing there; the long-lived Linux shape is `--windowless` plus the
/// systemd user unit that wave 2 shipped).
pub fn install(stores: &Stores, opts: LaunchOptions, cx: &mut App) {
    #[cfg(target_os = "macos")]
    macos::install(stores, opts, cx);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (stores, opts, cx);
    }
}

/// Declare that a window is about to be opened — the app must be `Regular`
/// before it appears, or it gets no menu bar. The one door out of the
/// background state; see the module docs for why it lives at a choke point.
pub fn window_will_open(cx: &mut App) {
    #[cfg(target_os = "macos")]
    macos::window_will_open(cx);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cx;
    }
}

/// The app's ⌘Q: retire to the background where there is a background to
/// retire into, and otherwise the full shutdown ⌘Q has always been.
///
/// The decision is [`quit_intent`]; this is the half that touches the world.
/// **Off macOS it is always the full shutdown** — Linux's background layer is
/// the systemd user service (`--windowless`), not a tray, so a windowed Linux
/// app quitting on Ctrl+Q is exactly right and nothing here changes it.
pub fn quit_or_retire(cx: &mut App) {
    #[cfg(target_os = "macos")]
    macos::quit_or_retire(cx);
    #[cfg(not(target_os = "macos"))]
    cx.quit();
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_app_core::{ExternalEngineBackend, LocalModelInfo};

    fn model(name: &str, status: LocalModelStatus) -> LocalModelInfo {
        LocalModelInfo {
            id: format!("{name}@local"),
            slug: name.to_string(),
            display_name: name.to_string(),
            file_name: format!("{name}.gguf"),
            size_bytes: None,
            source_url: None,
            status,
            last_error: None,
        }
    }

    fn state(models: Vec<LocalModelInfo>, external: Vec<LocalModelInfo>) -> LocalModelsState {
        LocalModelsState {
            engine_path: Some("/bin/llama-server".into()),
            models,
            external: vec![ExternalEngineBackend {
                backend_id: "mine".into(),
                display_name: "Mine".into(),
                enabled: true,
                models_dir: "/models".into(),
                engine_path: None,
                auto_start: false,
                models: external,
            }],
        }
    }

    #[test]
    fn retiring_needs_a_status_item_to_retire_into() {
        // The ordinary case: a door in the menu bar, so ⌘Q parks the app
        // behind it instead of killing the engines.
        assert_eq!(quit_intent(true, false), QuitIntent::Retire);
        // No door: retiring would leave an invisible, unquittable process,
        // so ⌘Q keeps the meaning it has always had.
        assert_eq!(quit_intent(false, false), QuitIntent::FullShutdown);
    }

    #[test]
    fn a_second_quit_from_the_background_ends_the_process() {
        // ⌘Q can only reach an already-retired app through its status menu,
        // and there it means quit — whichever menu AppKit resolves the key
        // equivalent against, both routes land here.
        assert_eq!(quit_intent(true, true), QuitIntent::FullShutdown);
        assert_eq!(quit_intent(false, true), QuitIntent::FullShutdown);
    }

    #[test]
    fn only_quit_claims_a_key_equivalent() {
        assert_eq!(StatusCommand::Quit.key_equivalent(), "q");
        assert_eq!(StatusCommand::Open.key_equivalent(), "");
        assert_eq!(StatusCommand::NewSpace.key_equivalent(), "");
    }

    #[test]
    fn engine_lines_name_running_and_warming_engines_across_backends() {
        let s = state(
            vec![
                model(
                    "gemma",
                    LocalModelStatus::Loaded {
                        port: 8081,
                        context_tokens: 4096,
                        pinned: false,
                    },
                ),
                model("idle", LocalModelStatus::Available),
                model("coming", LocalModelStatus::Loading),
                model(
                    "fetching",
                    LocalModelStatus::Downloading {
                        received: 1,
                        total: None,
                    },
                ),
            ],
            vec![model(
                "mistral",
                LocalModelStatus::Loaded {
                    port: 8082,
                    context_tokens: 8192,
                    pinned: true,
                },
            )],
        );
        assert_eq!(
            engine_lines(Some(&s)),
            vec![
                "gemma — running".to_string(),
                "coming — loading…".to_string(),
                "mistral — running".to_string(),
            ]
        );
    }

    #[test]
    fn an_idle_or_unknown_engine_section_says_so_rather_than_going_blank() {
        assert_eq!(engine_lines(None), vec!["Checking on-device engines…"]);
        assert_eq!(
            engine_lines(Some(&state(
                vec![model("idle", LocalModelStatus::Available)],
                vec![]
            ))),
            vec!["No models running"]
        );
    }

    #[test]
    fn the_menu_brackets_the_engine_section_with_separators() {
        let rows = menu_rows(None);
        assert_eq!(
            rows,
            vec![
                StatusRow::Command(StatusCommand::Open),
                StatusRow::Command(StatusCommand::NewSpace),
                StatusRow::Separator,
                StatusRow::Info("Checking on-device engines…".into()),
                StatusRow::Separator,
                StatusRow::Command(StatusCommand::Quit),
            ]
        );
    }

    #[test]
    fn every_command_round_trips_through_its_menu_tag() {
        for cmd in [
            StatusCommand::Open,
            StatusCommand::NewSpace,
            StatusCommand::Quit,
        ] {
            assert_eq!(StatusCommand::from_tag(cmd.tag()), Some(cmd));
        }
        assert_eq!(StatusCommand::from_tag(99), None);
    }
}
