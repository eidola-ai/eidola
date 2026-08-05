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

use eidola_app_core::{LocalModelInfo, LocalModelStatus, LocalModelsState, RunningEngine};
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
/// **When the registry answers, it is the running set — in both directions;
/// the listing only decorates.** `running` is
/// [`eidola_app_core::AppCore::running_engines`] — synchronous, infallible,
/// the in-process truth. `installed` is the `LocalModelsStore` snapshot, which
/// app-core reconstructs by *scanning* the model directories, and which
/// survives a failed rescan as a preserved `prior`. Both failure modes follow
/// from that, and they are mirror images:
///
/// - **The listing misses a live engine.** Its `.gguf` was renamed or deleted
///   mid-session, or its backend row was removed, or its directory went
///   unreadable — so the scan cannot see it while the subprocess holds
///   gigabytes, and the menu said "No models running" over it. Every registry
///   entry now produces a line; one the listing has lost is named from the
///   engine's own slug and says `(file missing)`, because "why is this model
///   not in Settings while my memory is gone?" is the question that moment
///   raises.
/// - **The listing claims a dead one.** An engine exits or is unloaded, the
///   rescan fails, and `Failed { prior }` keeps the old snapshot standing
///   indefinitely — so a `Loaded` row outlives its engine. The registry
///   already knows it is gone, so that row is vetoed rather than printed.
///
/// Both are the same scan-versus-registry defect wave 2 fixed on the quit
/// path, and one rule closes both: membership comes from the registry, names
/// come from the listing. See [`reconcile`], which also documents why leading
/// with the registry loses no warming engine.
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

/// Resolve the two sources into the readout.
///
/// **When the registry answered it is the running set in both directions** —
/// it adds engines the listing lost, and it *vetoes* engines the listing still
/// claims. The veto is the half that is easy to miss: a listing is a snapshot
/// preserved across a failed rescan (`Failed { prior }` keeps the old value so
/// the UI is never blanked), so after an engine exits or is unloaded, a rescan
/// that fails leaves a stale `Loaded` row standing indefinitely. Emitting it
/// would have the menu insist a stopped engine is running while `running`
/// already knew better. The listing therefore only *decorates* — it supplies
/// display names for entries the registry vouches for, nothing more.
///
/// **The registry owns the warming phase, so nothing is lost by leading with
/// it.** `reserve_engine` inserts the entry with `ready: false` *before* the
/// subprocess is spawned, and `scan_engine_dir` derives the listing's
/// `LocalModelStatus::Loading` from exactly that (`Some(e) if !e.ready`) — so
/// the listing can never report a warming engine the registry does not already
/// hold, and the registry's own `ready` bit is the fresher copy of the same
/// fact. (Before the reservation there is no engine at all: the load is still
/// doing backend lookup, port picking and `fs::metadata`, and the listing
/// would say `Available` too.)
fn reconcile(
    running: Option<&[RunningEngine]>,
    installed: &Loadable<LocalModelsState>,
) -> Vec<EngineLine> {
    let state = installed.value();
    let Some(running) = running else {
        // Nothing to ask (no core — the stub stores). The listing is all
        // there is, so its own rows stand for whatever they are worth.
        return state.map(listing_lines).unwrap_or_default();
    };

    let mut out: Vec<EngineLine> = running
        .iter()
        .map(|engine| {
            let listed = state.and_then(|s| find_listed(s, &engine.id));
            EngineLine {
                // The registry has no display name — that lives in a sidecar
                // beside the file, which is exactly what may have gone. The
                // slug is the truest name left.
                name: listed
                    .map(|m| m.display_name.clone())
                    .unwrap_or_else(|| engine.slug.clone()),
                // The registry's `ready`, not the listing's copy of it: same
                // fact, read later.
                ready: engine.ready,
                // Only claim the file is missing when the listing actually
                // answered. A listing that is still loading (or failed with
                // nothing kept) is not evidence of anything.
                orphaned: state.is_some() && listed.is_none(),
            }
        })
        .collect();
    // `running_engines` is ordered by id; the reader sees names. Sorting the
    // way app-core sorts the listing keeps the menu and Settings agreeing on
    // order — and because this sort is stable over an id-ordered input, two
    // engines sharing a name across backends still come out in a fixed order.
    out.sort_by_key(|e| e.name.to_lowercase());
    out
}

/// Every model in the listing, managed store and llamacpp backends alike — an
/// engine is an engine, and the menu is not a backend registry.
fn listed_models(state: &LocalModelsState) -> impl Iterator<Item = &LocalModelInfo> {
    state
        .models
        .iter()
        .chain(state.external.iter().flat_map(|b| b.models.iter()))
}

fn find_listed<'a>(state: &'a LocalModelsState, id: &str) -> Option<&'a LocalModelInfo> {
    listed_models(state).find(|m| m.id == id)
}

/// The listing's own view of what is up. Reachable only with no registry to
/// ask, which off the test stubs means no `AppCore` at all.
fn listing_lines(state: &LocalModelsState) -> Vec<EngineLine> {
    listed_models(state)
        .filter_map(|m| {
            let ready = match m.status {
                LocalModelStatus::Loaded { .. } => true,
                LocalModelStatus::Loading => false,
                _ => return None,
            };
            Some(EngineLine {
                name: m.display_name.clone(),
                ready,
                orphaned: false,
            })
        })
        .collect()
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

    /// **The display name is deliberately not the slug** — it is what the
    /// sidecar beside the `.gguf` carries, and the whole question in these
    /// tests is which source named a row. A fixture where the two matched
    /// could not tell a decorated line from a bare registry one.
    fn model_in(backend: &str, name: &str, status: LocalModelStatus) -> LocalModelInfo {
        let mut display = name.to_string();
        display[..1].make_ascii_uppercase();
        LocalModelInfo {
            id: format!("{name}@{backend}"),
            slug: name.to_string(),
            display_name: display,
            file_name: format!("{name}.gguf"),
            size_bytes: None,
            source_url: None,
            status,
            last_error: None,
            on_disk: true,
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
        // Named by the listing (capitalised), ordered by that name.
        assert_eq!(
            engine_lines(Some(&running), &Loadable::loaded(s)),
            vec![
                "Coming — loading…".to_string(),
                "Gemma — running".to_string(),
                "Mistral — running".to_string(),
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
    fn a_stale_listing_cannot_claim_a_stopped_engine_is_running() {
        // The mirror of the orphan case, and the reason the registry is
        // authoritative in *both* directions. An engine exits or is unloaded;
        // the rescan that would drop its row fails; `Failed { prior }` keeps
        // the old snapshot standing so the UI is never blanked — and that
        // preserved row says `Loaded` indefinitely. The registry already knows
        // it is gone, so the row is vetoed rather than printed.
        let stale = state(vec![model("ghost", loaded(8081))], vec![]);
        let listing = Loadable::Failed {
            error: AppError::Internal {
                message: "rescan failed".into(),
            },
            prior: Some(stale.clone()),
        };
        assert_eq!(
            engine_lines(Some(&[]), &listing),
            vec!["No models running", "Couldn't list installed models"],
            "the registry vetoes a row its own emptiness contradicts"
        );

        // Same veto without any failure in play: a merely stale `Loaded`
        // snapshot beside an empty registry.
        assert_eq!(
            engine_lines(Some(&[]), &Loadable::loaded(stale)),
            vec!["No models running"]
        );
    }

    #[test]
    fn a_warming_engine_survives_the_registry_leading() {
        // The registry owns the warming phase: `reserve_engine` inserts the
        // entry with `ready: false` *before* spawning, and the listing derives
        // its `Loading` status from exactly that bit — so leading with the
        // registry can never lose a warming row. Both shapes are pinned: with
        // the listing agreeing, and with the listing not yet rescanned (the
        // window between the reservation and the scan that would notice it).
        let running = [engine("warming", "local", false)];

        let agreeing = Loadable::loaded(state(
            vec![model("warming", LocalModelStatus::Loading)],
            vec![],
        ));
        assert_eq!(
            engine_lines(Some(&running), &agreeing),
            vec!["Warming — loading…"],
            "named by the listing, status from the registry"
        );

        // The registry alone still shows it, and does not call it lost — the
        // listing here simply has nothing to say about that id yet.
        let behind = Loadable::loaded(state(vec![], vec![]));
        assert_eq!(
            engine_lines(Some(&running), &behind),
            vec!["warming — loading… (file missing)"]
        );
    }

    #[test]
    fn the_readout_is_ordered_by_the_name_the_reader_sees() {
        // `running_engines` is ordered by id, but the reader sees names —
        // sorted the way app-core sorts the listing, so the menu and Settings
        // agree on order.
        let listing = Loadable::loaded(state(
            vec![
                model("aardvark", loaded(1)),
                model("zebra", loaded(2)),
                model("moose", loaded(3)),
            ],
            vec![],
        ));
        let running = [
            engine("zebra", "local", true),
            engine("aardvark", "local", true),
            engine("moose", "local", true),
        ];
        assert_eq!(
            engine_lines(Some(&running), &listing),
            vec![
                "Aardvark — running".to_string(),
                "Moose — running".to_string(),
                "Zebra — running".to_string(),
            ]
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
