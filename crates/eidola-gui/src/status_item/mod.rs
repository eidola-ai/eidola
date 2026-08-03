//! The macOS status item and the Regular ⇄ Accessory activation flip
//! (task 17, wave 3).
//!
//! Wave 2 made the process outlive its windows. This wave gives that process
//! a face: a menu-bar item that is always there, and — because the face
//! exists — permission to drop the Dock icon when no window is open.
//!
//! ## The two halves, and why they ship together
//!
//! An `Accessory` app has no Dock icon and no menu bar of its own. With zero
//! windows and no status item it is invisible and unquittable except through
//! Activity Monitor. So the flip is **gated on the status item existing**:
//! [`policy_for`] returns `Regular` whenever no status item stands, however
//! many windows are open. A status bar too full to accept another item (it
//! happens) leaves the app in the wave-2 shape — Dock icon, reachable — which
//! is a perfectly good app, not a broken one.
//!
//! ## Two verbs, not one
//!
//! The policy is driven by the window count, and the two transitions are
//! observed at different moments:
//!
//! - [`window_will_open`] runs *before* `cx.open_window` (from
//!   `lib.rs::base_window_options`, the one choke point every window passes
//!   through), because at that instant `cx.windows()` does not yet contain
//!   the window being opened. It always resolves to `Regular` — a window
//!   without a menu bar cannot dispatch ⌘N.
//! - [`window_did_close`] runs from `App::on_window_closed`, which gpui fires
//!   *after* removing the window from its registry — so `cx.windows().len()`
//!   there is already the post-close count, and zero means the last one went.
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

/// The one policy decision, as a pure function.
///
/// `status_item_present` is not decoration: without a status item an
/// `Accessory` app with no windows cannot be reached or quit, so a failed
/// status-item install degrades to the wave-2 behaviour (Dock icon, always)
/// rather than to an invisible process.
pub fn policy_for(window_count: usize, status_item_present: bool) -> ActivationPolicy {
    if window_count > 0 || !status_item_present {
        ActivationPolicy::Regular
    } else {
        ActivationPolicy::Accessory
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
    /// ⌘Q — a full shutdown, engines included (decided).
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

/// Create the status item and wire the activation-policy flip. Called once at
/// launch, after the action handlers exist (the menu dispatches them) and
/// before the first window opens.
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
/// before it appears, or it gets no menu bar. See the module docs for why
/// this is a separate verb from [`window_did_close`].
pub fn window_will_open(cx: &mut App) {
    #[cfg(target_os = "macos")]
    macos::window_will_open(cx);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cx;
    }
}

/// Re-decide the policy after a window closed.
pub fn window_did_close(cx: &mut App) {
    #[cfg(target_os = "macos")]
    macos::window_did_close(cx);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cx;
    }
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
    fn the_flip_needs_a_status_item_to_be_survivable() {
        // With a status item: no windows means menu-bar-only.
        assert_eq!(policy_for(0, true), ActivationPolicy::Accessory);
        assert_eq!(policy_for(1, true), ActivationPolicy::Regular);
        // Without one, Accessory would make the app unreachable — so the
        // Dock icon stays whatever the window count says.
        assert_eq!(policy_for(0, false), ActivationPolicy::Regular);
        assert_eq!(policy_for(3, false), ActivationPolicy::Regular);
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
