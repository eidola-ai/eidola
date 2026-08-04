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
//! The rows are a pure projection ([`menu_rows`]) of two inputs — the **live
//! engine registry** and the `LocalModelsStore` snapshot — mirrored into the
//! platform object by a store observer and rebuilt into real `NSMenuItem`s in
//! `menuNeedsUpdate:`, AppKit's "the user is about to see this" callback.
//! Neither end captures authority: the mirror is refreshed by the store's own
//! notify (never sampled once at install), and the menu is materialised at
//! open (never rebuilt on every download-progress tick, which arrives several
//! times a second). See `macos::Target` for why the mirror exists rather than
//! a live read from inside the AppKit callback.
//!
//! Which of the two inputs is authoritative is the whole of [`engine_lines`]'
//! doc comment, and it is not a detail: the snapshot is a *directory scan*,
//! so it cannot see an engine whose backing file has gone.

#[cfg(target_os = "macos")]
mod macos;

use eidola_app_core::{LocalModelStatus, LocalModelsState, RunningEngine};
use gpui::App;

use crate::lifecycle::LaunchOptions;
use crate::loadable::Loadable;
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

/// The whole status menu.
///
/// Both engine inputs are passed down to [`engine_lines`]; see there for why
/// there are two of them.
pub fn menu_rows(
    running: Option<&[RunningEngine]>,
    installed: &Loadable<LocalModelsState>,
) -> Vec<StatusRow> {
    let mut rows = vec![
        StatusRow::Command(StatusCommand::Open),
        StatusRow::Command(StatusCommand::NewSpace),
        StatusRow::Separator,
    ];
    rows.extend(
        engine_lines(running, installed)
            .into_iter()
            .map(StatusRow::Info),
    );
    rows.push(StatusRow::Separator);
    rows.push(StatusRow::Command(StatusCommand::Quit));
    rows
}

/// One engine of the readout, after the two sources have been reconciled.
struct EngineLine {
    name: String,
    ready: bool,
    /// The live registry holds this engine, but the installed-model listing
    /// does not know it — its backing `.gguf` has gone while the engine runs.
    orphaned: bool,
}

/// The engine section: one line per engine that is up or coming up, across
/// the managed store *and* every configured `llamacpp` backend (an engine is
/// an engine — the menu is not a backend registry).
///
/// **The live registry leads and the listing only decorates it.**
/// `running` is [`eidola_app_core::AppCore::running_engines`] — synchronous,
/// infallible, the in-process truth. `installed` is the `LocalModelsStore`
/// snapshot, which app-core reconstructs by *scanning* the model directories:
/// an engine whose `.gguf` was renamed or deleted mid-session (or whose
/// backend row was removed, or whose directory has gone unreadable) is simply
/// absent from it while the subprocess holds gigabytes. Reading only the
/// listing therefore said "No models running" over a live engine — the same
/// scan-versus-registry defect wave 2 fixed on the quit path. So every
/// registry entry produces a line, named from the listing when the listing
/// knows it and from the engine's own slug when it does not; an engine the
/// listing has lost says so, because "why is this model not in Settings
/// while my memory is gone?" is the question that moment raises.
///
/// **`installed` is the whole `Loadable`, not its value.** `value()` folds
/// `Failed { prior: None }` into the same `None` as an in-flight first load,
/// so a failed initial refresh left the menu reading "Checking on-device
/// engines…" forever — a spinner for a request that already failed and (the
/// store retries only on a bus `Change`) may never be retried. The states are
/// now distinguished: a failure says so, and does not suppress the registry's
/// own answer.
///
/// **There is no Retry row, deliberately.** The failure costs display names
/// and the installed listing, never the running truth; the store re-refreshes
/// on any `Change::LocalModels` / `Change::Backends`; and Settings → Backends
/// → Local is the surface with room for the error text and the retry. A
/// fourth verb in a three-verb menu would buy less than the line it displaced.
pub fn engine_lines(
    running: Option<&[RunningEngine]>,
    installed: &Loadable<LocalModelsState>,
) -> Vec<String> {
    let mut lines: Vec<String> = reconcile(running, installed)
        .into_iter()
        .map(|e| {
            let state = if e.ready { "running" } else { "loading…" };
            let lost = if e.orphaned { " (file missing)" } else { "" };
            format!("{} — {state}{lost}", e.name)
        })
        .collect();

    if lines.is_empty() {
        // Nothing is up — but only say so if something could actually have
        // told us. The registry always can; the listing can when it has a
        // value. With neither, the honest word is that we do not know yet.
        if running.is_some() || installed.has_value() {
            lines.push("No models running".to_string());
        } else if installed.error().is_none() {
            lines.push("Checking on-device engines…".to_string());
        }
    }
    if installed.error().is_some() {
        // Quiet, and always — even beside a full engine list, because a
        // failed scan is why those names may be bare slugs.
        lines.push("Couldn't list installed models".to_string());
    }
    lines
}

/// Join the two sources on the model id: the listing's own loaded/loading
/// rows first (they carry display names), then every registry engine the
/// listing did not account for.
fn reconcile(
    running: Option<&[RunningEngine]>,
    installed: &Loadable<LocalModelsState>,
) -> Vec<EngineLine> {
    let state = installed.value();
    let listed = state.into_iter().flat_map(|s| {
        s.models
            .iter()
            .chain(s.external.iter().flat_map(|b| b.models.iter()))
    });

    let mut seen: Vec<&str> = Vec::new();
    let mut out: Vec<EngineLine> = Vec::new();
    for m in listed {
        let ready = match m.status {
            LocalModelStatus::Loaded { .. } => true,
            LocalModelStatus::Loading => false,
            _ => continue,
        };
        seen.push(m.id.as_str());
        out.push(EngineLine {
            name: m.display_name.clone(),
            ready,
            orphaned: false,
        });
    }

    for engine in running.unwrap_or(&[]) {
        if seen.contains(&engine.id.as_str()) {
            continue;
        }
        out.push(EngineLine {
            // The registry has no display name — that lives in a sidecar
            // beside the file, which is exactly what may have gone. The slug
            // is the truest name left.
            name: engine.slug.clone(),
            ready: engine.ready,
            // Only claim the file is missing when the listing actually
            // answered. A listing that is still loading (or failed) is not
            // evidence of anything.
            orphaned: state.is_some(),
        });
    }
    out
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
    use eidola_app_core::error::AppError;
    use eidola_app_core::{ExternalEngineBackend, LocalModelInfo};

    fn model(name: &str, status: LocalModelStatus) -> LocalModelInfo {
        model_in("local", name, status)
    }

    /// The external backend's scan. **The id carries the backend**, exactly
    /// as `engine_model_id` builds it — the join key against the registry, so
    /// a fixture that got it wrong would fake a duplicate engine.
    fn external_model(name: &str, status: LocalModelStatus) -> LocalModelInfo {
        model_in("mine", name, status)
    }

    fn model_in(backend: &str, name: &str, status: LocalModelStatus) -> LocalModelInfo {
        LocalModelInfo {
            id: format!("{name}@{backend}"),
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

    fn loaded(port: u16) -> LocalModelStatus {
        LocalModelStatus::Loaded {
            port,
            context_tokens: 4096,
            pinned: false,
        }
    }

    /// A live registry entry. `slug` is all the registry has for a name —
    /// see `RunningEngine`.
    fn engine(slug: &str, backend: &str, ready: bool) -> RunningEngine {
        RunningEngine {
            id: format!("{slug}@{backend}"),
            backend_id: backend.to_string(),
            slug: slug.to_string(),
            port: 9000,
            context_tokens: 4096,
            ready,
            pinned: false,
        }
    }

    #[test]
    fn engine_lines_name_running_and_warming_engines_across_backends() {
        let s = state(
            vec![
                model("gemma", loaded(8081)),
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
            vec![external_model("mistral", loaded(8082))],
        );
        // The listing and the registry agree, which is the ordinary case:
        // every engine is named by its display name, once.
        let running = [
            engine("gemma", "local", true),
            engine("coming", "local", false),
            engine("mistral", "mine", true),
        ];
        assert_eq!(
            engine_lines(Some(&running), &Loadable::loaded(s)),
            vec![
                "gemma — running".to_string(),
                "coming — loading…".to_string(),
                "mistral — running".to_string(),
            ]
        );
    }

    #[test]
    fn an_engine_whose_file_vanished_still_shows_as_running() {
        // The defect this shape exists for: `local_models_state` is a
        // *directory scan*, so an engine whose backing `.gguf` was renamed or
        // deleted mid-session drops out of the listing while its subprocess
        // keeps eating memory. Reading only the listing said "No models
        // running" over it.
        let listing = Loadable::loaded(state(vec![], vec![]));
        let running = [engine("ghost", "local", true)];
        assert_eq!(
            engine_lines(Some(&running), &listing),
            vec!["ghost — running (file missing)"],
            "named from the slug, and honest about why it is not in Settings"
        );
    }

    #[test]
    fn a_listing_that_has_not_answered_does_not_accuse_an_engine_of_losing_its_file() {
        // Same registry entry, but the listing is in flight / failed with no
        // prior — which is not evidence that anything is missing.
        let running = [engine("ghost", "local", true)];
        for listing in [
            Loadable::Loading,
            Loadable::Failed {
                error: AppError::Internal {
                    message: "nope".into(),
                },
                prior: None,
            },
        ] {
            assert!(
                engine_lines(Some(&running), &listing).contains(&"ghost — running".to_string()),
                "no (file missing) claim without a listing to contradict it"
            );
        }
    }

    #[test]
    fn a_failed_listing_says_so_instead_of_spinning_forever() {
        // `Loadable::value()` folds `Failed { prior: None }` into the same
        // `None` as an in-flight first load, which left the menu reading
        // "Checking on-device engines…" for a request that had already failed
        // and (the store only retries on a bus Change) might never be retried.
        let failed: Loadable<LocalModelsState> = Loadable::Failed {
            error: AppError::Internal {
                message: "unreadable".into(),
            },
            prior: None,
        };
        // With the registry present, the running truth is still answered —
        // the failure only costs the installed listing.
        assert_eq!(
            engine_lines(Some(&[]), &failed),
            vec!["No models running", "Couldn't list installed models"]
        );
        // With neither source, the failure is the whole answer, and it never
        // reads as a spinner.
        let lines = engine_lines(None, &failed);
        assert_eq!(lines, vec!["Couldn't list installed models"]);
        assert!(!lines.iter().any(|l| l.contains("Checking")));
    }

    #[test]
    fn an_idle_or_unknown_engine_section_says_so_rather_than_going_blank() {
        // Nothing can answer yet: honest, and it resolves on the next notify.
        assert_eq!(
            engine_lines(None, &Loadable::NotLoaded),
            vec!["Checking on-device engines…"]
        );
        // The registry can always answer, even with no listing at all.
        assert_eq!(
            engine_lines(Some(&[]), &Loadable::NotLoaded),
            vec!["No models running"]
        );
        assert_eq!(
            engine_lines(
                None,
                &Loadable::loaded(state(
                    vec![model("idle", LocalModelStatus::Available)],
                    vec![]
                ))
            ),
            vec!["No models running"]
        );
    }

    #[test]
    fn the_menu_brackets_the_engine_section_with_separators() {
        let rows = menu_rows(None, &Loadable::NotLoaded);
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
