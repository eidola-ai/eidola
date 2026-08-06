//! Eidola GUI library — exposes views and state used by the binary entry
//! point in `main.rs` and by snapshot tests in `tests/visual.rs`.

pub mod about;
pub mod account;
pub mod actions;
pub mod agents_settings;
pub mod backends_settings;
pub mod bridge;
pub mod chrome;
pub mod focus;
pub mod general;
pub mod library;
pub mod lifecycle;
pub mod loadable;
pub mod login_item;
pub mod onboarding;
pub mod overlay;
pub mod participants;
mod plans;
pub mod probe;
pub mod record;
pub mod scrollbar;
pub mod settings;
pub mod solar;
pub mod space;
pub mod space_view;
pub mod status_item;
pub mod stores;
pub mod templates_settings;
pub mod theme;
pub mod titlebar;
pub mod updates;
pub mod wallet;
pub mod window_input;

use gpui::{
    App, AppContext, Bounds, KeyBinding, Menu, MenuItem, OsAction, TitlebarOptions, WindowBounds,
    WindowHandle, WindowKind, WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_component_assets::Assets;

use crate::about::AboutView;
use crate::actions::{
    About, ActualSize, CheckForUpdates, CloseWindow, GetStarted, Hide, HideOthers, Minimize,
    NewSpace, NewSpaceFromTemplate, OpenLibrary, OpenRecord, OpenSettings, Quit, QuitApp, Quote,
    QuoteElsewhere, QuoteInReply, ShowAll, ToggleElementInspector, ToggleInspector, Zoom, ZoomIn,
    ZoomOut,
};
use crate::library::LibraryView;
use crate::lifecycle::LaunchOptions;
use crate::onboarding::OnboardingView;
use crate::record::RecordView;
use crate::settings::SettingsView;
use crate::space_view::SpaceView;
use crate::stores::Stores;
use crate::updates::UpdatesView;
use crate::window_input::WindowInput;

/// Application-scoped state. Stored as a gpui global so action handlers
/// (which only get `&mut App`) can reach it.
struct AppGlobal {
    stores: Stores,
    /// The single About window, if open. Same singleton discipline as
    /// `settings_window`.
    about_window: Option<WindowHandle<Root>>,
    /// The single Settings window, if it's currently open. Used to enforce
    /// the macOS-typical singleton: re-invoking `OpenSettings` raises the
    /// existing window instead of opening another. We don't actively clear
    /// this on close — `try_focus_existing_singleton` checks the cached id
    /// against `cx.windows()` each time and self-heals a stale handle.
    settings_window: Option<WindowHandle<Root>>,
    /// The single Library window, if open. Same singleton discipline as
    /// `settings_window`.
    library_window: Option<WindowHandle<Root>>,
    /// The single Updates window, if open. Same singleton discipline as
    /// `settings_window`.
    updates_window: Option<WindowHandle<Root>>,
    /// The single Record window, if open. Same singleton discipline as
    /// `settings_window`.
    record_window: Option<WindowHandle<Root>>,
    /// The open Record window's view, weakly. Held beside the handle because
    /// every window's root is a `gpui_component::Root` (see
    /// [`try_focus_existing_singleton`]), so the view can't be recovered by
    /// downcasting the handle — and a deep link into a *specific* raw exchange
    /// (a space's trace row, task 34) has to reach the view, not just the
    /// window.
    record_view: Option<gpui::WeakEntity<RecordView>>,
    /// The single onboarding ("Get Started") window, if open. Same singleton
    /// discipline as `settings_window`.
    onboarding_window: Option<WindowHandle<Root>>,
}

impl gpui::Global for AppGlobal {}

/// Run the GUI application with default (windowed) launch options.
pub fn run() {
    run_with(LaunchOptions::default());
}

/// Run the GUI application. The binary's `fn main()` is a thin shim around
/// this; tests do not call this — they use `tests/visual.rs` instead.
///
/// **The process outlives its windows** (task 17 wave 2) — see
/// [`crate::lifecycle`] for the seams and the per-platform rules. On macOS
/// ⌘Q retires it to the background behind the status item rather than ending
/// it (wave 3b, [`crate::status_item`]); the full shutdown is the status
/// menu's own Quit.
pub fn run_with(opts: LaunchOptions) {
    let application = gpui_platform::application()
        .with_assets(Assets)
        // Named explicitly rather than inherited from `QuitMode::Default`'s
        // `cfg!`: on Linux this is a launch-mode decision (a `--windowless`
        // service must survive a visiting window closing), which no upstream
        // default can make for us.
        .with_quit_mode(lifecycle::quit_mode(opts));

    // macOS reactivation (Dock icon, Spotlight relaunch of the running app,
    // a second `open -a Eidola`): focus a window, or open the Library when
    // there is none. Without this, closing the last window leaves the app
    // running but unreachable. `on_reopen` is on the `Application` builder
    // (registered before launch), not on `App`, and returns `&Self` rather
    // than `Self` so we can't chain it before `run()` (which consumes by
    // value). On Linux gpui stores the callback and nothing ever fires it —
    // there is no platform mechanism; that door is the wave-4 socket.
    application.on_reopen(|cx: &mut App| reactivate_app(cx));

    application.run(move |cx: &mut App| {
        gpui_component::init(cx);
        theme::install(cx);

        let stores = Stores::new(cx);

        // The single app-lifetime bus bridge: forwards every app-core
        // `Change` into a gpui main-thread loop that dispatches to the
        // stores (the only place tokio receivers touch gpui). Install it
        // before the startup refreshes so nothing committed during them is
        // missed.
        let bus_bridge = stores::install_bus_bridge(&stores, cx);

        // Point the Circadian theme at the persisted settings (day/night
        // axis + time-of-day tint), re-applying on config changes and on
        // the clock's ~4h slot boundaries. `theme::install` above applied
        // the neutral defaults; this turns the circadian machinery on.
        theme::wire_config(&stores.config, cx);

        // Startup refreshes — each in its own store task slot, no shared
        // busy flag, so none can starve another (the wave-2 launch-order
        // bug is fixed structurally: the model list refresh cannot be
        // dropped by an in-flight wallet recovery).
        stores.backends.update(cx, |s, cx| s.refresh(cx));
        stores.models.update(cx, |s, cx| s.refresh(cx));
        stores.local_models.update(cx, |s, cx| s.refresh(cx));
        stores.spaces.update(cx, |s, cx| s.refresh(cx));
        stores.templates.update(cx, |s, cx| s.refresh(cx));

        // Best-effort recovery of any in-flight credentials left over from a
        // previous run that crashed mid-spend. Owned by the WalletStore; the
        // result surfaces on the wallet view next time the user opens it.
        stores.wallet.update(cx, |s, cx| {
            s.refresh(cx);
            s.recover(cx, |_, _, _| {});
        });

        cx.set_global(AppGlobal {
            stores: stores.clone(),
            about_window: None,
            settings_window: None,
            library_window: None,
            updates_window: None,
            record_window: None,
            record_view: None,
            onboarding_window: None,
        });

        // Verified update-notification polling: one check at launch, then
        // every ~6h while running (tokio task on the core's runtime). A
        // result that lands while no Updates window is open is reflected
        // the next time one opens — no banners in chat windows.
        stores.update.read(cx).start_polling();

        // The *full* shutdown drains the engines — and on macOS this hook
        // is the only thing that delivers it. ⌘Q no longer reaches it (it
        // retires the app instead, keeping the engines up, which is the
        // point); the status menu's Quit and a windowless SIGTERM do. See
        // `lifecycle::install_engine_shutdown`.
        lifecycle::install_engine_shutdown(&stores, bus_bridge, cx);

        // Order matters: `cx.set_menus` snapshots the keymap when it builds
        // NSMenuItems and attaches each item's `keyEquivalent` from
        // `keymap.bindings_for_action(action)`. If we set menus before
        // binding keys, the keymap is empty at lookup time, no keystroke is
        // attached, and macOS can't intercept the shortcut at the menu
        // level — which then breaks ⌘N / ⌘Q etc. when no window has key
        // focus (the only path that *requires* the menu-level intercept;
        // with a window focused, gpui's per-window binding dispatch
        // handles it independently). Diagnostic signal: items appear in
        // the menu without their shortcut text on the right side.
        install_keybindings(cx);
        install_menus(cx);
        install_action_handlers(cx);

        // Rebuild the menus whenever the template registry changes, so the
        // Space → "New Space from Template ▸" submenu tracks templates
        // created / renamed / removed in Settings. App-lifetime observation
        // (the sanctioned detach); `install_menus` is idempotent.
        cx.observe(&stores.templates, |_, cx| install_menus(cx))
            .detach();

        // The menu-bar face — and, because it exists, the thing ⌘Q retires
        // *into* (task 17 waves 3/3b). After the action handlers, because
        // the status menu dispatches them; before any window opens, because
        // `base_window_options` asks it to assert `Regular`. A windowless
        // macOS launch gets one too — that is exactly the toolbar-app shape.
        status_item::install(&stores, opts, cx);

        // Windowless: the process *is* the app — no window at launch, no
        // foreground activation, and (per `lifecycle::quit_mode`) nothing
        // that closes later can stop it. Everything above still runs: the
        // stores, the bus bridge, update polling and any engine loaded
        // through them belong to the process, which is the whole point.
        // The only way out is an explicit quit, which a service manager
        // delivers as a signal.
        if opts.windowless {
            #[cfg(unix)]
            lifecycle::install_signal_quit(&stores, cx).detach();
            return;
        }

        // Bring the app to the foreground at launch. Mirrors Zed; ensures
        // macOS treats us as the active app from frame 0 so the menu bar
        // / key-equivalent dispatch is fully wired before the user
        // interacts with anything.
        cx.activate(true);

        // First-run onboarding: with no account configured, open the "Get
        // Started" window *instead of* a blank space — onboarding is the
        // door, and a blank space behind it is both premature (there's
        // nothing to ask yet) and noise to close. A configured account — or
        // a deliberately *disabled* eidola backend (the "no account,
        // on-device only" choice, recorded in the DB) — opens straight into
        // a space (onboarding is then only reachable via the Eidola menu).
        // Leaving onboarding opens the space it stood in for; see
        // `OnboardingView::leave`.
        //
        // The account bit reads synchronously from the ConfigStore snapshot;
        // the backend bit needs a DB read, so in that case the decision —
        // and with it the first window — is one spawned read behind launch.
        let needs_account = stores
            .config
            .read(cx)
            .state()
            .map(|s| !s.has_account || !s.has_account_secret)
            .unwrap_or(false);
        match stores.app_core().filter(|_| needs_account) {
            Some(core) => {
                // A cold start can be slow enough to ⌘Q through; see
                // `lifecycle::OpenIntent`.
                let intent = lifecycle::intend_to_open(cx);
                let task: gpui::Task<()> = cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                    let backends =
                        crate::bridge::bridge(core, |c| async move { c.list_backends().await })
                            .await;
                    let eidola_enabled = backends
                        .ok()
                        .and_then(|list| list.iter().find(|b| b.id == "eidola").map(|b| b.enabled))
                        // On a read failure, err toward showing onboarding —
                        // the window is dismissible; a silent skip is not.
                        .unwrap_or(true);
                    cx.update(|cx| {
                        if !intent.still_wanted(cx) {
                            return;
                        }
                        if eidola_enabled {
                            open_onboarding_window(cx);
                        } else {
                            open_main_window(cx);
                        }
                    });
                });
                // Startup-scoped one-shot with nothing to own it; the
                // sanctioned app-lifetime detach pattern (see
                // stores::install_bus_bridge).
                task.detach();
            }
            None => open_main_window(cx),
        }
    });
}

/// The app's door back in when it has been running without a face: focus a
/// window, or open the Library when there is none.
///
/// Shared by `Application::on_reopen` (Dock click, Spotlight, a second
/// `open -a Eidola`) and the status menu's "Open Eidola" — one behaviour, so
/// the two can never drift.
pub(crate) fn reactivate_app(cx: &mut App) {
    lifecycle::reactivate(cx, |cx| {
        // A reopen can in principle arrive before launch finishes.
        if cx.has_global::<AppGlobal>() {
            open_library_window(cx);
        }
    });
}

fn install_menus(cx: &mut App) {
    cx.set_menus(vec![
        Menu {
            name: "Eidola".into(),
            items: vec![
                MenuItem::action("About Eidola", About),
                MenuItem::action("Check for Updates…", CheckForUpdates),
                MenuItem::Separator,
                MenuItem::action("Get Started", GetStarted),
                MenuItem::action("Settings…", OpenSettings),
                MenuItem::Separator,
                MenuItem::action("Library…", OpenLibrary),
                MenuItem::action("Record…", OpenRecord),
                MenuItem::Separator,
                MenuItem::action("Hide Eidola", Hide),
                MenuItem::action("Hide Others", HideOthers),
                MenuItem::action("Show All", ShowAll),
                MenuItem::Separator,
                MenuItem::action("Quit", Quit),
            ],
            disabled: false,
        },
        // "Space" is the space-centric replacement for the conventional
        // "File" menu — the space (conversation) is our unit of work. "New
        // Space" keeps the File-menu New idiom; Library/Record moved up into
        // the Eidola app menu. Future space-scoped items (space settings,
        // export, …) land here.
        Menu {
            name: "Space".into(),
            items: vec![
                MenuItem::action("New Space", NewSpace),
                new_space_from_template_submenu(cx),
                MenuItem::Separator,
                // The inspector's only doors are this item and its ⌥⌘I
                // equivalent — the space carries no visual toggle (Mike,
                // 2026-08-01). The label states both directions because
                // `cx.set_menus` builds a static bar: gpui rebuilds it only
                // when we ask it to, and the answer would differ per window.
                MenuItem::action("Show/Hide Inspector", ToggleInspector),
                MenuItem::Separator,
                MenuItem::action("Close Window", CloseWindow),
            ],
            disabled: false,
        },
        // `os_action` ties Edit-menu items to the standard macOS selectors
        // (cut:, copy:, paste:, selectAll:), so the OS routes them through
        // the responder chain to whatever has focus — including system
        // textfields in save panels and the like. Undo/Redo are kept on
        // `handleGPUIMenuItem:` because gpui-macos disables the OS undo:/redo:
        // selectors when there's no NSTextView/NSTextField responder.
        Menu {
            name: "Edit".into(),
            items: vec![
                MenuItem::action("Undo", gpui_component::input::Undo),
                MenuItem::action("Redo", gpui_component::input::Redo),
                MenuItem::Separator,
                MenuItem::os_action("Cut", gpui_component::input::Cut, OsAction::Cut),
                MenuItem::os_action("Copy", gpui_component::input::Copy, OsAction::Copy),
                MenuItem::os_action("Paste", gpui_component::input::Paste, OsAction::Paste),
                MenuItem::Separator,
                MenuItem::os_action(
                    "Select All",
                    gpui_component::input::SelectAll,
                    OsAction::SelectAll,
                ),
                MenuItem::Separator,
                // Quoting is an Edit-menu verb because it acts on the
                // *selection*, like Cut/Copy. Both handlers are registered
                // per-`SpaceView` and only while a quotable post selection
                // exists, so macOS greys them the rest of the time.
                MenuItem::action("Quote", Quote),
                MenuItem::action("Quote in Reply", QuoteInReply),
                MenuItem::action("Quote in Another Conversation…", QuoteElsewhere),
            ],
            disabled: false,
        },
        // The View menu carries the standard macOS type-size trio (Actual
        // Size / Zoom In / Zoom Out). Kept deliberately minimal and scoped to
        // just these items so it doesn't collide with the concurrent File/app
        // menu restructuring — this file only ever adds a View menu.
        Menu {
            name: "View".into(),
            items: vec![
                MenuItem::action("Actual Size", ActualSize),
                MenuItem::action("Zoom In", ZoomIn),
                MenuItem::action("Zoom Out", ZoomOut),
            ],
            disabled: false,
        },
        // Naming this menu "Window" causes gpui_macos to call
        // `app.setWindowsMenu_(menu)`, which tells AppKit "this is the
        // canonical macOS Window menu". AppKit auto-populates it with the
        // open-window switcher and treats the app as fully wired up — which
        // matters for keystroke / menu-equivalent dispatch in edge cases
        // like ⌘Tab back to the app or having no window key.
        Menu {
            name: "Window".into(),
            items: vec![
                MenuItem::action("Minimize", Minimize),
                MenuItem::action("Zoom", Zoom),
            ],
            disabled: false,
        },
    ]);

    // Standard macOS dock menu: right-click the dock icon → "New Space" /
    // "Library…".
    cx.set_dock_menu(vec![
        MenuItem::action("New Space", NewSpace),
        MenuItem::action("Library…", OpenLibrary),
    ]);
}

/// The "New Space from Template ▸" submenu, built from the live template
/// registry (`TemplatesStore` snapshot). `New Space` (⌘N) already instantiates
/// the *default* template; this submenu lets a user pick a specific one. It is
/// rebuilt whenever `Change::Templates` fires (see the observer in `run()`), so
/// a template created/renamed/removed in Settings is reflected here. Empty until
/// the registry loads (it re-lists at launch), at which point the seeded
/// "Default" always appears.
fn new_space_from_template_submenu(cx: &App) -> MenuItem {
    let templates = cx
        .try_global::<AppGlobal>()
        .map(|g| g.stores.templates.read(cx).list().to_vec())
        .unwrap_or_default();
    let items: Vec<MenuItem> = templates
        .into_iter()
        .map(|t| {
            MenuItem::action(
                t.title.clone(),
                NewSpaceFromTemplate {
                    template_id: t.id.clone(),
                },
            )
        })
        .collect();
    MenuItem::Submenu(Menu {
        name: "New Space from Template".into(),
        items,
        disabled: false,
    })
}

/// Public because the UI driver (`examples/driver.rs`) installs the same
/// keymap as the real app so simulated keystrokes resolve identically.
pub fn install_keybindings(cx: &mut App) {
    // Chord-style commands bind with gpui's `secondary-` alias (⌘ on macOS,
    // Ctrl on Linux/Windows) so one table serves both platforms: Ctrl+N /
    // Ctrl+Q / Ctrl+, etc. are the Linux idiom. Window-management chords
    // (Hide/Minimize) are macOS concepts — on Wayland the compositor owns
    // window management (Super+H etc.), so they are bound only on macOS.
    cx.bind_keys([
        KeyBinding::new("secondary-,", OpenSettings, None),
        KeyBinding::new("secondary-n", NewSpace, None),
        KeyBinding::new("secondary-l", OpenLibrary, None),
        KeyBinding::new("secondary-shift-l", OpenRecord, None),
        KeyBinding::new("secondary-w", CloseWindow, None),
        KeyBinding::new("secondary-q", Quit, None),
        // ⌥⌘I shows/hides the space window's inspector (Xcode/Finder); gpui's
        // element inspector — a development overlay — took ⌥⇧⌘I.
        KeyBinding::new("secondary-alt-i", ToggleInspector, None),
        KeyBinding::new("secondary-alt-shift-i", ToggleElementInspector, None),
        // View → type size. ⌘0 / Ctrl+0 resets; ⌘=/⌘+ zooms in (both the bare
        // `=` and the shifted `+` that shares the key, so either keypress
        // works); ⌘-/Ctrl+- zooms out. Global (no context) so they fire even
        // with the composer focused — the editor keymap binds none of these.
        KeyBinding::new("secondary-0", ActualSize, None),
        KeyBinding::new("secondary-=", ZoomIn, None),
        KeyBinding::new("secondary-+", ZoomIn, None),
        KeyBinding::new("secondary--", ZoomOut, None),
        // ⌘↩ (Post) and ⌘⇧↩ (post quietly) are *not* bound here. The
        // composer owns those chords — `gpui_markdown_editor::init` binds them
        // in the `MarkdownEditor` context to `Enter { secondary: true, .. }`,
        // whose handler emits `PressEnter`; the draft's subscription (see
        // `space_view::composer::create_draft_node`) routes that to
        // `Send`/`PostOnly`. Binding them here too would shadow the editor
        // (the composer is the inner focus) and break the inversion.
        // The former ⌥⌘M (`ToggleModelPicker`) is gone with the request
        // panel: the composer no longer carries a model choice — who answers
        // (and with what model) is Participants configuration (the
        // inspector's Participants section), and explicit asks live on the
        // separator bands. Esc
        // routes through the composer's own key handler (band menu first, then
        // draft deactivation), so no Esc binding here.
    ]);

    // macOS window/app management — no Linux analogue (hide is an AppKit
    // concept; minimize belongs to the compositor on Wayland).
    #[cfg(target_os = "macos")]
    cx.bind_keys([
        KeyBinding::new("cmd-h", Hide, None),
        KeyBinding::new("alt-cmd-h", HideOthers, None),
        KeyBinding::new("cmd-m", Minimize, None),
    ]);

    // Linux: F10 toggles the primary menu (the desktop-standard key for the
    // header-bar menu). Handled by `chrome::ChromeRoot`, an ancestor of
    // every focused element, so a global binding always reaches it.
    #[cfg(not(target_os = "macos"))]
    cx.bind_keys([KeyBinding::new(
        "f10",
        crate::chrome::TogglePrimaryMenu,
        None,
    )]);

    // The composer's own keymap (motion, editing, clipboard, and the submit
    // chords) is self-contained in the editor crate, scoped to the
    // `MarkdownEditor` context — installed here like `gpui_component::init`
    // installs the `Input` keymap.
    gpui_markdown_editor::init(cx);
}

fn install_action_handlers(cx: &mut App) {
    // ⌘Q is now two-tier (task 17 wave 3b). Where a background layer exists —
    // macOS with a status item standing — it retires into it: windows close,
    // the Dock indicator goes, the process and its loaded engines stay. Where
    // one does not (no status item; every non-macOS build, whose background
    // layer is the systemd user service, not a tray), it is the full shutdown
    // it has always been. Ending the process deliberately is `QuitApp`, which
    // the status menu's "Quit Eidola" raises.
    cx.on_action(|_: &Quit, cx: &mut App| status_item::quit_or_retire(cx));

    cx.on_action(|_: &QuitApp, cx: &mut App| {
        cx.quit();
    });

    cx.on_action(|_: &About, cx: &mut App| {
        // Singleton: raise the existing About window if alive.
        if try_focus_existing_about(cx) {
            return;
        }
        open_about_window(cx);
    });

    cx.on_action(|_: &OpenSettings, cx: &mut App| {
        // Singleton: raise the existing window if it's still alive,
        // otherwise open a fresh one. We do this *synchronously* so the
        // handle is stored before the action handler returns — earlier
        // we used `cx.spawn` for this and a fast second click could fire
        // the next handler before the spawned task had stored the handle,
        // producing two windows.
        if try_focus_existing_settings(cx) {
            return;
        }
        open_settings_window(cx);
    });

    cx.on_action(|_: &CheckForUpdates, cx: &mut App| {
        // Singleton, like Settings. Opening (or raising) the window is
        // the manual-check gesture: the view triggers a fresh check on
        // construction, and raising an existing window re-checks here so
        // the user always gets a live answer.
        if try_focus_existing_updates(cx) {
            let update = cx.global::<AppGlobal>().stores.update.clone();
            update.update(cx, |s, cx| s.check_now(cx));
            return;
        }
        open_updates_window(cx);
    });

    cx.on_action(|_: &NewSpace, cx: &mut App| {
        open_main_window(cx);
    });

    // New Space from a specific template (Space menu submenu). Routed through
    // `SpacesStore` so the create-and-open op is **owned** in an entity task
    // slot (STATE.md — never `.detach()` domain work): the store keys each
    // activation independently so a committed space always gets its window, and
    // surfaces a failure in the store's `op_error` (Library banner) instead of
    // silently discarding it.
    cx.on_action(|action: &NewSpaceFromTemplate, cx: &mut App| {
        let stores = cx.global::<AppGlobal>().stores.clone();
        let template_id = action.template_id.clone();
        stores.spaces.clone().update(cx, |s, cx| {
            s.create_from_template(template_id, stores.clone(), cx);
        });
    });

    cx.on_action(|_: &GetStarted, cx: &mut App| {
        // Singleton, like Settings: raise the existing onboarding window if
        // it's still alive, otherwise open a fresh one.
        if try_focus_existing_onboarding(cx) {
            return;
        }
        open_onboarding_window(cx);
    });

    cx.on_action(|_: &OpenLibrary, cx: &mut App| {
        // The listing may be stale (exchanges happen while the singleton
        // stays open), so refresh it on every invocation — whether we're
        // raising the existing window or opening a fresh one (which also
        // fetches on construction; the refresh is idempotent).
        let spaces = cx.global::<AppGlobal>().stores.spaces.clone();
        spaces.update(cx, |s, cx| s.refresh(cx));

        if try_focus_existing_library(cx) {
            return;
        }
        open_library_window(cx);
    });

    cx.on_action(|_: &OpenRecord, cx: &mut App| {
        // Singleton like Settings/Library. The view re-queries the local
        // database on construction and exposes its own Refresh affordance,
        // so raising the existing window needs no extra fetch here.
        if try_focus_existing_record(cx) {
            return;
        }
        open_record_window(cx);
    });

    // View → type size. Global handlers that write the `font_scale` config
    // override through the shared `ConfigStore`; the write emits
    // `Change::Config`, which `theme::wire_config`'s observer turns into a
    // re-apply on every window. Reading the current scale off the store's
    // snapshot means the ladder works from any window (or none focused).
    cx.on_action(|_: &ActualSize, cx: &mut App| {
        let config = cx.global::<AppGlobal>().stores.config.clone();
        config.update(cx, |s, cx| s.reset_zoom(cx));
    });
    cx.on_action(|_: &ZoomIn, cx: &mut App| {
        let config = cx.global::<AppGlobal>().stores.config.clone();
        config.update(cx, |s, cx| s.zoom_in(cx));
    });
    cx.on_action(|_: &ZoomOut, cx: &mut App| {
        let config = cx.global::<AppGlobal>().stores.config.clone();
        config.update(cx, |s, cx| s.zoom_out(cx));
    });

    // macOS standard App-menu actions. Without these registered, AppKit
    // may treat the app menu as incomplete in the no-window-focused state
    // and skip menu-equivalent dispatch.
    cx.on_action(|_: &Hide, cx: &mut App| cx.hide());
    cx.on_action(|_: &HideOthers, cx: &mut App| cx.hide_other_apps());
    cx.on_action(|_: &ShowAll, cx: &mut App| cx.unhide_other_apps());

    // Window menu standards. Both need a focused window — `cx.defer` so we
    // run after the current dispatch's window-update completes; without
    // it, `handle.update(cx, ...)` on the same window we were dispatched
    // inside fails (slot is taken) and `.ok()` silently swallows the Err.
    cx.on_action(|_: &Minimize, cx: &mut App| {
        let Some(handle) = cx.active_window() else {
            return;
        };
        cx.defer(move |cx| {
            handle
                .update(cx, |_, window, _| window.minimize_window())
                .ok();
        });
    });
    cx.on_action(|_: &Zoom, cx: &mut App| {
        let Some(handle) = cx.active_window() else {
            return;
        };
        cx.defer(move |cx| {
            handle.update(cx, |_, window, _| window.zoom_window()).ok();
        });
    });

    // Toggle gpui's *element* inspector (development) on the active window.
    // The product's space inspector is `ToggleInspector`, registered per-view.
    // Same `cx.defer`
    // pattern as `Minimize`/`Zoom` — `Window::toggle_inspector` requires
    // `&mut Window`, and dispatching directly on the same window we were
    // invoked from would fail (slot already taken). `gpui-component`'s
    // inspector::init also binds the same action under its own
    // `inspector::ToggleInspector` namespace; ours coexists because the
    // action types are distinct, and gives us an explicit binding in our
    // own keymap regardless of whether gpui-component's inspector is
    // initialized in this build.
    cx.on_action(|_: &ToggleElementInspector, cx: &mut App| {
        let Some(handle) = cx.active_window() else {
            return;
        };
        cx.defer(move |cx| {
            handle
                .update(cx, |_, window, cx| window.toggle_inspector(cx))
                .ok();
        });
    });

    // `CloseWindow` is intentionally NOT registered as a global handler.
    // Each view registers its own listener via `.on_action(cx.listener(…))`
    // (see `space_view::SpaceView` and `settings::SettingsView`). With per-view
    // registration, `is_action_available` returns true only when a window
    // with the listener is alive — so macOS auto-greys "Close Window" in
    // the menu when no window is open, which is the correct behavior.
}

/// Try to bring an existing singleton window (Settings, Library) forward.
/// Returns `true` if a live window was raised, `false` otherwise.
///
/// The liveness check matches the cached id against `cx.windows()` — the
/// authoritative list of live windows. (We can't use the cleaner Zed-style
/// `cx.windows().find_map(downcast::<SettingsView>)` because all of our
/// windows wrap their views in `gpui_component::Root`, which is required by
/// `Root::read` calls inside the `Input` widget — so they're not
/// distinguishable by root view type.) A stale id self-heals here: if the
/// cached window was closed, the containment check fails and we clear the
/// cache.
fn try_focus_existing_singleton(
    cx: &mut App,
    slot: impl Fn(&mut AppGlobal) -> &mut Option<WindowHandle<Root>>,
) -> bool {
    let Some(handle) = *slot(cx.global_mut::<AppGlobal>()) else {
        return false;
    };
    let alive = cx
        .windows()
        .iter()
        .any(|w| w.window_id() == handle.window_id());
    if !alive {
        *slot(cx.global_mut::<AppGlobal>()) = None;
        return false;
    }
    handle
        .update(cx, |_, window, _| window.activate_window())
        .ok();
    true
}

fn try_focus_existing_about(cx: &mut App) -> bool {
    try_focus_existing_singleton(cx, |g| &mut g.about_window)
}

fn try_focus_existing_settings(cx: &mut App) -> bool {
    try_focus_existing_singleton(cx, |g| &mut g.settings_window)
}

fn try_focus_existing_library(cx: &mut App) -> bool {
    try_focus_existing_singleton(cx, |g| &mut g.library_window)
}

fn try_focus_existing_updates(cx: &mut App) -> bool {
    try_focus_existing_singleton(cx, |g| &mut g.updates_window)
}

fn try_focus_existing_record(cx: &mut App) -> bool {
    try_focus_existing_singleton(cx, |g| &mut g.record_window)
}

fn try_focus_existing_onboarding(cx: &mut App) -> bool {
    try_focus_existing_singleton(cx, |g| &mut g.onboarding_window)
}

/// Edge-to-edge titlebar: macOS extends the content view under the
/// traffic-light buttons and stops painting a separate titlebar background.
/// Each view is responsible for leaving room at the top so the lights don't
/// land on real UI — see `space_view::TITLE_BAR_RESERVE` (vertical reserve + fade
/// gradient), `settings::NAV_TOP_RESERVE` (the lights sit over the nav
/// band), and `record::STRIP_LEFT_PAD` (the section strip doubles as the
/// title bar).
///
/// `title: None` here only means "no title at creation" — every
/// `open_*_window` calls `Window::set_window_title` anyway, which names the
/// window for the macOS Window menu, the window switcher, and (the reason it
/// was added) VoiceOver's window chooser, *and* labels the accessibility
/// tree's otherwise-anonymous root node. It paints nothing: gpui_macos pairs
/// `titlebarAppearsTransparent` with `NSWindowTitleHidden`, so the string
/// never reaches the title bar.
fn transparent_titlebar() -> TitlebarOptions {
    TitlebarOptions {
        title: None,
        appears_transparent: true,
        // Vertically centered in the 36px title-bar reserve, tuned by eye to
        // match macOS-native lift (centers the ~12px buttons around y≈17).
        traffic_light_position: Some(point(px(14.), px(11.))),
    }
}

fn centered_window_bounds(cx: &mut App, w: f32, h: f32) -> Option<WindowBounds> {
    let display = cx.primary_display()?;
    let center = display.bounds().center();
    Some(WindowBounds::Windowed(Bounds::centered_at(
        center,
        size(px(w), px(h)),
    )))
}

/// The shared window options for every Eidola window: the transparent
/// titlebar (macOS), and on Linux the client-side-decoration request plus a
/// transparent surface (so the CSD shadow/rounded-corner padding drawn by
/// `chrome::ChromeRoot` is actually see-through) and the Wayland `app_id`
/// (matching the shipped `.desktop` file so the shell can associate windows
/// with the app's identity and icon).
///
/// **It is also the one choke point every window open passes through**, and
/// therefore the single door out of the retired-to-the-background state
/// (`status_item::window_will_open`, which asserts `Regular`) — an
/// `Accessory` app has no menu bar, so a window opened without it would
/// appear under someone else's. Taking `cx` for a side effect is deliberate:
/// putting the call here rather than in each `open_*_window` makes "open a
/// window while `Accessory`" unrepresentable, and it covers every reopen path
/// at once (the status menu, Spotlight, `open -a`, `on_reopen`).
fn base_window_options(
    cx: &mut App,
    bounds: Option<WindowBounds>,
    min_w: f32,
    min_h: f32,
) -> WindowOptions {
    status_item::window_will_open(cx);
    WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(min_w), px(min_h))),
        #[cfg(not(target_os = "macos"))]
        window_decorations: Some(gpui::WindowDecorations::Client),
        #[cfg(not(target_os = "macos"))]
        window_background: gpui::WindowBackgroundAppearance::Transparent,
        #[cfg(not(target_os = "macos"))]
        app_id: Some(APP_ID.into()),
        ..Default::default()
    }
}

/// Wayland application id — must equal the basename of the shipped
/// `.desktop` file (`releases/linux/tech.m6i.Eidola.desktop`) for the shell
/// to resolve the window to its launcher entry (name, icon, pinning).
#[cfg(not(target_os = "macos"))]
const APP_ID: &str = "tech.m6i.Eidola";

/// Open the About window — a small singleton (~360×420). Shows the wordmark,
/// version, a quiet purpose copy (echoing the welcome page's voice), the
/// license note, and a "View on GitHub" link.
fn open_about_window(cx: &mut App) {
    let bounds = centered_window_bounds(cx, 360., 420.);
    let opts = base_window_options(cx, bounds, 300., 340.);

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        window.set_window_title("About Eidola");
        let view = cx.new(|cx| AboutView::new(window, cx));
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().about_window = Some(handle);
    }
    cx.activate(true);
}

fn open_main_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    open_blank_space_window(cx, stores);
}

/// Open a chat window onto an existing space. Public-to-the-crate entry
/// used by the Library; takes the stores explicitly (instead of reading
/// `AppGlobal`) so view code and tests can call it without the global
/// installed.
pub fn open_space_window(cx: &mut App, stores: Stores, space_id: String) {
    open_chat_window(cx, stores, Some(space_id));
}

/// Open a chat window onto a fresh blank space (⌘N). Takes the stores
/// explicitly for the same reason [`open_space_window`] does: onboarding
/// opens one on its way out, and its stub-store tests run without
/// `AppGlobal` installed.
pub fn open_blank_space_window(cx: &mut App, stores: Stores) {
    open_chat_window(cx, stores, None);
}

/// The side length of the square "writing surface" windows — the space (chat)
/// window and the onboarding window, which is sized to match. 90% of the
/// smaller display dimension, capped at 840px so the prose column isn't lost in
/// the middle of a 4K display; falls back to the cap with no primary display
/// (rare; offscreen render contexts).
fn writing_surface_side(cx: &mut App) -> f32 {
    match cx.primary_display() {
        Some(d) => {
            let s = d.bounds().size;
            let smaller = f32::min(s.width.as_f32(), s.height.as_f32());
            (smaller * 0.9).min(840.0)
        }
        None => 820.0,
    }
}

fn open_chat_window(cx: &mut App, stores: Stores, space_id: Option<String>) {
    // Square chat window — a sheet of paper, not a wide chat pane.
    let side = writing_surface_side(cx);
    let bounds = centered_window_bounds(cx, side, side);
    let opts = base_window_options(cx, bounds, 480., 360.);

    let _ = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let wi = WindowInput::new(cx);
        let view = cx.new(|cx| SpaceView::new(stores.clone(), space_id.clone(), wi, window, cx));
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    // Bring the app forward so the new window comes to the front, even when
    // the action originated from another app's context (e.g. the dock
    // right-click menu while a different app is foreground). `focus: true`
    // in WindowOptions makes the window key within our app, but doesn't
    // by itself activate the app vs other apps.
    cx.activate(true);
}

/// Open the Library window — a singleton like Settings. Sized as a tall,
/// narrow page: a table of contents, not a browser.
fn open_library_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let bounds = centered_window_bounds(cx, 520., 620.);
    let opts = base_window_options(cx, bounds, 380., 320.);

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        window.set_window_title("Library");
        let view = cx.new(|cx| LibraryView::new(stores.clone(), window, cx));
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().library_window = Some(handle);
    }
    cx.activate(true);
}

/// Open the Updates window — a small singleton (Eidola menu → "Check for
/// Updates…", standard macOS placement under "About Eidola").
fn open_updates_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let bounds = centered_window_bounds(cx, 480., 360.);
    let opts = base_window_options(cx, bounds, 420., 300.);

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        window.set_window_title("Updates");
        let view = cx.new(|cx| UpdatesView::new(stores.clone(), window, cx));
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().updates_window = Some(handle);
    }
    cx.activate(true);
}

/// Open the Record window — the trust door's raw local trail (attestations,
/// requests, spending). Singleton like Settings/Library; sized wide because
/// its rows are mono request lines, not prose.
fn open_record_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let bounds = centered_window_bounds(cx, 860., 640.);
    let opts = base_window_options(cx, bounds, 560., 400.);

    // The view is minted inside the window builder; capture it on the way out
    // so a deep link can reach it later (see `AppGlobal::record_view`).
    let captured: std::rc::Rc<std::cell::RefCell<Option<gpui::WeakEntity<RecordView>>>> =
        Default::default();
    let sink = captured.clone();
    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        window.set_window_title("The Record");
        let view = cx.new(|cx| RecordView::new(stores.clone(), window, cx));
        *sink.borrow_mut() = Some(view.downgrade());
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    if let Ok(handle) = handle {
        let global = cx.global_mut::<AppGlobal>();
        global.record_window = Some(handle);
        global.record_view = captured.borrow().clone();
    }
    cx.activate(true);
}

/// Open the Record on one specific raw exchange — the deep link behind a
/// space's trace rows (task 34). Raises the existing Record window when there
/// is one (the singleton discipline) and opens it otherwise, then drives the
/// view straight to that request's detail.
///
/// A no-op without `AppGlobal` (stub-store tests, the driver), which is why
/// the caller records the request id before calling.
pub fn open_record_request(cx: &mut App, request_id: String) {
    if !cx.has_global::<AppGlobal>() {
        return;
    }
    if !try_focus_existing_record(cx) {
        open_record_window(cx);
    }
    let view = cx
        .global::<AppGlobal>()
        .record_view
        .clone()
        .and_then(|w| w.upgrade());
    if let Some(view) = view {
        view.update(cx, |view, cx| view.show_request(request_id, cx));
    }
}

/// Open the onboarding window — the from-scratch "Get Started" flow, a
/// singleton like Settings. Sized to match a new space window (the same square
/// writing surface) so onboarding feels like the same page it leads into.
fn open_onboarding_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let side = writing_surface_side(cx);
    let bounds = centered_window_bounds(cx, side, side);
    let opts = base_window_options(cx, bounds, 480., 360.);

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        window.set_window_title("Get Started");
        let view = cx.new(|cx| OnboardingView::new(stores.clone(), window, cx));
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().onboarding_window = Some(handle);
    }
    cx.activate(true);
}

fn open_settings_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let bounds = centered_window_bounds(cx, 620., 520.);
    let opts = base_window_options(cx, bounds, 420., 320.);

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        window.set_window_title("Settings");
        let view = cx.new(|cx| SettingsView::new(stores.clone(), window, cx));
        let view = chrome::ChromeRoot::wrap(view.into(), cx);
        cx.new(|cx| chrome::themed_root(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().settings_window = Some(handle);
    }
}
