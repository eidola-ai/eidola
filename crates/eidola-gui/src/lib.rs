//! Eidola GUI library — exposes views and state used by the binary entry
//! point in `main.rs` and by snapshot tests in `tests/visual.rs`.

pub mod about;
pub mod account;
pub mod actions;
pub mod bridge;
pub mod general;
pub mod library;
pub mod loadable;
pub mod onboarding;
mod plans;
pub mod probe;
pub mod record;
pub mod settings;
pub mod space;
pub mod space_view;
pub mod stores;
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
    About, CheckForUpdates, CloseWindow, GetStarted, Hide, HideOthers, Minimize, NewSpace,
    OpenLibrary, OpenRecord, OpenSettings, Quit, ShowAll, ToggleInspector, Zoom,
};
use crate::library::LibraryView;
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
    /// The single onboarding ("Get Started") window, if open. Same singleton
    /// discipline as `settings_window`.
    onboarding_window: Option<WindowHandle<Root>>,
}

impl gpui::Global for AppGlobal {}

/// Run the GUI application. The binary's `fn main()` is a thin shim around
/// this; tests do not call this — they use `tests/visual.rs` instead.
pub fn run() {
    let application = gpui_platform::application().with_assets(Assets);

    // Standard macOS: clicking the dock icon when the app has no open
    // windows should create one. Without this, closing the last window
    // leaves the app running but unreachable. `on_reopen` is on the
    // `Application` builder (registered before launch), not on `App`, and
    // returns `&Self` rather than `Self` so we can't chain it before
    // `run()` (which consumes by value).
    application.on_reopen(|cx: &mut App| {
        if cx.windows().is_empty() {
            open_main_window(cx);
        }
    });

    application.run(move |cx: &mut App| {
        gpui_component::init(cx);
        theme::install(cx);

        let stores = Stores::new(cx);

        // The single app-lifetime bus bridge: forwards every app-core
        // `Change` into a gpui main-thread loop that dispatches to the
        // stores (the only place tokio receivers touch gpui). Install it
        // before the startup refreshes so nothing committed during them is
        // missed.
        stores::install_bus_bridge(&stores, cx);

        // Startup refreshes — each in its own store task slot, no shared
        // busy flag, so none can starve another (the wave-2 launch-order
        // bug is fixed structurally: the model list refresh cannot be
        // dropped by an in-flight wallet recovery).
        stores.models.update(cx, |s, cx| s.refresh(cx));
        stores.spaces.update(cx, |s, cx| s.refresh(cx));

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
            onboarding_window: None,
        });

        // Verified update-notification polling: one check at launch, then
        // every ~6h while running (tokio task on the core's runtime). A
        // result that lands while no Updates window is open is reflected
        // the next time one opens — no banners in chat windows.
        stores.update.read(cx).start_polling();

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

        // Bring the app to the foreground at launch. Mirrors Zed; ensures
        // macOS treats us as the active app from frame 0 so the menu bar
        // / key-equivalent dispatch is fully wired before the user
        // interacts with anything.
        cx.activate(true);

        open_main_window(cx);

        // First-run onboarding: with no account configured, open the "Get
        // Started" window on top of the main window. A configured account skips
        // straight to the main window (onboarding is then only reachable via the
        // Eidola menu). Read through the ConfigStore snapshot seeded at startup.
        let needs_onboarding = stores
            .config
            .read(cx)
            .state()
            .map(|s| !s.has_account || !s.has_account_secret)
            .unwrap_or(false);
        if needs_onboarding {
            open_onboarding_window(cx);
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
                MenuItem::action("Hide Eidola", Hide),
                MenuItem::action("Hide Others", HideOthers),
                MenuItem::action("Show All", ShowAll),
                MenuItem::Separator,
                MenuItem::action("Quit", Quit),
            ],
            disabled: false,
        },
        Menu {
            name: "File".into(),
            items: vec![
                MenuItem::action("New Space", NewSpace),
                MenuItem::action("Library…", OpenLibrary),
                MenuItem::action("Record…", OpenRecord),
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

/// Public because the UI driver (`examples/driver.rs`) installs the same
/// keymap as the real app so simulated keystrokes resolve identically.
pub fn install_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("cmd-,", OpenSettings, None),
        KeyBinding::new("cmd-n", NewSpace, None),
        KeyBinding::new("cmd-l", OpenLibrary, None),
        KeyBinding::new("cmd-shift-l", OpenRecord, None),
        KeyBinding::new("cmd-w", CloseWindow, None),
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("cmd-h", Hide, None),
        KeyBinding::new("alt-cmd-h", HideOthers, None),
        KeyBinding::new("cmd-m", Minimize, None),
        KeyBinding::new("cmd-alt-i", ToggleInspector, None),
        // ⌘↩ (post & ask) and ⌘⇧↩ (post only) are *not* bound here. The
        // composer owns those chords — `gpui_markdown_editor::init` binds them
        // in the `MarkdownEditor` context to `Enter { secondary: true, .. }`,
        // whose handler emits `PressEnter`; the draft's subscription (see
        // `space_view::composer::create_draft_node`) routes that to
        // `Send`/`PostOnly`. Binding them here too would shadow the editor
        // (the composer is the inner focus) and break the inversion.
        // ⌥⌘M — the space view's request panel (the handler no-ops without a
        // composer to anchor to). Scoped to SpaceView so the distinct
        // keystroke never competes with the global ⌘M (Minimize). Esc routes
        // through the composer's own key handler (panel first, then draft
        // deactivation), so no Esc binding here.
        KeyBinding::new(
            "cmd-alt-m",
            crate::actions::ToggleModelPicker,
            Some("SpaceView"),
        ),
    ]);

    // The composer's own keymap (motion, editing, clipboard, and the submit
    // chords) is self-contained in the editor crate, scoped to the
    // `MarkdownEditor` context — installed here like `gpui_component::init`
    // installs the `Input` keymap.
    gpui_markdown_editor::init(cx);
}

fn install_action_handlers(cx: &mut App) {
    cx.on_action(|_: &Quit, cx: &mut App| {
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

    // Toggle gpui's element inspector on the active window. Same `cx.defer`
    // pattern as `Minimize`/`Zoom` — `Window::toggle_inspector` requires
    // `&mut Window`, and dispatching directly on the same window we were
    // invoked from would fail (slot already taken). `gpui-component`'s
    // inspector::init also binds the same action under its own
    // `inspector::ToggleInspector` namespace; ours coexists because the
    // action types are distinct, and gives us an explicit binding in our
    // own keymap regardless of whether gpui-component's inspector is
    // initialized in this build.
    cx.on_action(|_: &ToggleInspector, cx: &mut App| {
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

/// Open the About window — a small singleton (~360×420). Shows the wordmark,
/// version, a quiet purpose copy (echoing the welcome page's voice), the
/// license note, and a "View on GitHub" link.
fn open_about_window(cx: &mut App) {
    let bounds = centered_window_bounds(cx, 360., 420.);

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(300.), px(340.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| AboutView::new(window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().about_window = Some(handle);
    }
    cx.activate(true);
}

fn open_main_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    open_chat_window(cx, stores, None);
}

/// Open a chat window onto an existing space. Public-to-the-crate entry
/// used by the Library; takes the stores explicitly (instead of reading
/// `AppGlobal`) so view code and tests can call it without the global
/// installed.
pub fn open_space_window(cx: &mut App, stores: Stores, space_id: String) {
    open_chat_window(cx, stores, Some(space_id));
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

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(480.), px(360.))),
        ..Default::default()
    };

    let _ = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let wi = WindowInput::new(cx);
        let view = cx.new(|cx| SpaceView::new(stores.clone(), space_id.clone(), wi, window, cx));
        cx.new(|cx| Root::new(view, window, cx))
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

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(380.), px(320.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| LibraryView::new(stores.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
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

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(420.), px(300.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| UpdatesView::new(stores.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
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

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(560.), px(400.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| RecordView::new(stores.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().record_window = Some(handle);
    }
    cx.activate(true);
}

/// Open the onboarding window — the from-scratch "Get Started" flow, a
/// singleton like Settings. Sized to match a new space window (the same square
/// writing surface) so onboarding feels like the same page it leads into.
fn open_onboarding_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let side = writing_surface_side(cx);
    let bounds = centered_window_bounds(cx, side, side);

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(480.), px(360.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| OnboardingView::new(stores.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().onboarding_window = Some(handle);
    }
    cx.activate(true);
}

fn open_settings_window(cx: &mut App) {
    let stores = cx.global::<AppGlobal>().stores.clone();
    let bounds = centered_window_bounds(cx, 620., 520.);

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(420.), px(320.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let wi = WindowInput::new(cx);
        let view = cx.new(|cx| SettingsView::new(stores.clone(), wi, window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().settings_window = Some(handle);
    }
}
