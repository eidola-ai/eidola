//! Eidola GUI library — exposes views and state used by the binary entry
//! point in `main.rs` and by snapshot tests in `tests/visual.rs`.

pub mod account;
pub mod actions;
pub mod chat;
pub mod core;
pub mod general;
pub mod settings;
pub mod theme;
pub mod wallet;

use gpui::{
    App, AppContext, Bounds, Entity, KeyBinding, Menu, MenuItem, OsAction, TitlebarOptions,
    WindowBounds, WindowHandle, WindowKind, WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_component_assets::Assets;

use crate::actions::{
    About, CloseWindow, Hide, HideOthers, Minimize, NewSpace, OpenSettings, Quit, ShowAll,
    ToggleInspector, Zoom,
};
use crate::chat::ChatView;
use crate::core::Core;
use crate::settings::SettingsView;

/// Application-scoped state. Stored as a gpui global so action handlers
/// (which only get `&mut App`) can reach it.
struct AppGlobal {
    core: Entity<Core>,
    /// The single Settings window, if it's currently open. Used to enforce
    /// the macOS-typical singleton: re-invoking `OpenSettings` raises the
    /// existing window instead of opening another. We don't actively clear
    /// this on close — `try_focus_existing_settings` checks the cached id
    /// against `cx.windows()` each time and self-heals a stale handle.
    settings_window: Option<WindowHandle<Root>>,
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

        let core = Core::new(cx);

        // Best-effort recovery of any in-flight credentials left over from a
        // previous run that crashed mid-spend. Fires-and-forgets — the result
        // surfaces on the wallet view next time the user opens it. Mirrors
        // the SwiftUI app's startup .task on EidolaApp.
        core.update(cx, |c, cx| {
            c.recover_spending_credentials(cx, |_, _, _| {});
        });

        cx.set_global(AppGlobal {
            core,
            settings_window: None,
        });

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
    });
}

fn install_menus(cx: &mut App) {
    cx.set_menus(vec![
        Menu {
            name: "Eidola".into(),
            items: vec![
                MenuItem::action("About Eidola", About),
                MenuItem::Separator,
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

    // Standard macOS dock menu: right-click the dock icon → "New Space".
    cx.set_dock_menu(vec![MenuItem::action("New Space", NewSpace)]);
}

fn install_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("cmd-,", OpenSettings, None),
        KeyBinding::new("cmd-n", NewSpace, None),
        KeyBinding::new("cmd-w", CloseWindow, None),
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("cmd-h", Hide, None),
        KeyBinding::new("alt-cmd-h", HideOthers, None),
        KeyBinding::new("cmd-m", Minimize, None),
        KeyBinding::new("cmd-alt-i", ToggleInspector, None),
        KeyBinding::new("cmd-enter", crate::chat::Send, Some("ChatView")),
    ]);

    install_markdown_editor_keybindings(cx);
}

/// Keybindings for the WYSIWYG markdown composer used in the chat view.
///
/// All bindings are scoped to the `MarkdownEditor` key context (the
/// `key_context` set by `gpui_markdown_editor::MarkdownEditor::render`) so
/// they only fire when the editor — or a descendant — is in the focus
/// chain. That keeps them from competing with `gpui_component::Input`'s
/// own `Input`-context bindings used by settings/account fields, and lets
/// the ChatView-context `cmd-enter → Send` binding still win for submit
/// because the editor itself does not bind `cmd-enter` to anything.
///
/// Mirrors the macOS defaults documented in `apps/gui/Cargo.toml`-adjacent
/// `bin/demo.rs` in the editor crate, minus the global `cmd-up` /
/// `cmd-down` shortcuts that the chat reserves for future scroll-to-end
/// navigation.
fn install_markdown_editor_keybindings(cx: &mut App) {
    use gpui_markdown_editor::{
        Backspace, Copy, Cut, Delete, DeleteToLineEnd, DeleteToLineStart, DeleteWordBackward,
        DeleteWordForward, DocumentEnd, DocumentStart, Down, End, Enter, Home, Left, Paste,
        PastePlain, Right, SelectAll, ShiftDocumentEnd, ShiftDocumentStart, ShiftDown, ShiftEnd,
        ShiftEnter, ShiftHome, ShiftLeft, ShiftRight, ShiftTab, ShiftUp, ShiftWordLeft,
        ShiftWordRight, Tab, Up, WordLeft, WordRight,
    };

    let ctx = Some("MarkdownEditor");
    cx.bind_keys([
        // Editing
        KeyBinding::new("backspace", Backspace, ctx),
        KeyBinding::new("delete", Delete, ctx),
        KeyBinding::new("enter", Enter, ctx),
        KeyBinding::new("shift-enter", ShiftEnter, ctx),
        KeyBinding::new("tab", Tab, ctx),
        KeyBinding::new("shift-tab", ShiftTab, ctx),
        // Word / line delete (macOS standard: Option for word, Cmd for line).
        KeyBinding::new("alt-backspace", DeleteWordBackward, ctx),
        KeyBinding::new("alt-delete", DeleteWordForward, ctx),
        KeyBinding::new("cmd-backspace", DeleteToLineStart, ctx),
        KeyBinding::new("cmd-delete", DeleteToLineEnd, ctx),
        // Caret motion
        KeyBinding::new("left", Left, ctx),
        KeyBinding::new("right", Right, ctx),
        KeyBinding::new("up", Up, ctx),
        KeyBinding::new("down", Down, ctx),
        KeyBinding::new("shift-left", ShiftLeft, ctx),
        KeyBinding::new("shift-right", ShiftRight, ctx),
        KeyBinding::new("shift-up", ShiftUp, ctx),
        KeyBinding::new("shift-down", ShiftDown, ctx),
        KeyBinding::new("home", Home, ctx),
        KeyBinding::new("end", End, ctx),
        KeyBinding::new("cmd-left", Home, ctx),
        KeyBinding::new("cmd-right", End, ctx),
        KeyBinding::new("shift-home", ShiftHome, ctx),
        KeyBinding::new("shift-end", ShiftEnd, ctx),
        KeyBinding::new("cmd-shift-left", ShiftHome, ctx),
        KeyBinding::new("cmd-shift-right", ShiftEnd, ctx),
        KeyBinding::new("cmd-up", DocumentStart, ctx),
        KeyBinding::new("cmd-down", DocumentEnd, ctx),
        KeyBinding::new("cmd-shift-up", ShiftDocumentStart, ctx),
        KeyBinding::new("cmd-shift-down", ShiftDocumentEnd, ctx),
        // Word-granular motion (macOS standard: Option+arrows).
        KeyBinding::new("alt-left", WordLeft, ctx),
        KeyBinding::new("alt-right", WordRight, ctx),
        KeyBinding::new("alt-shift-left", ShiftWordLeft, ctx),
        KeyBinding::new("alt-shift-right", ShiftWordRight, ctx),
        // Clipboard — scoped to the editor context so they coexist with
        // `gpui_component::Input`'s `Input`-context bindings used by the
        // settings/account fields. The Edit menu items remain wired to
        // `gpui_component::input::*` actions for those inputs; menu-driven
        // copy/cut from the composer is a known gap pending wiring of
        // editor actions into the menu.
        KeyBinding::new("cmd-a", SelectAll, ctx),
        KeyBinding::new("cmd-c", Copy, ctx),
        KeyBinding::new("cmd-x", Cut, ctx),
        KeyBinding::new("cmd-v", Paste, ctx),
        KeyBinding::new("cmd-shift-v", PastePlain, ctx),
    ]);
}

fn install_action_handlers(cx: &mut App) {
    cx.on_action(|_: &Quit, cx: &mut App| {
        cx.quit();
    });

    cx.on_action(|_: &About, _cx: &mut App| {
        // Future: show an about panel.
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

    cx.on_action(|_: &NewSpace, cx: &mut App| {
        open_main_window(cx);
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
    // (see `chat::ChatView` and `settings::SettingsView`). With per-view
    // registration, `is_action_available` returns true only when a window
    // with the listener is alive — so macOS auto-greys "Close Window" in
    // the menu when no window is open, which is the correct behavior.
}

/// Try to bring the existing Settings window forward. Returns `true` if a
/// live Settings window was raised, `false` otherwise.
///
/// The liveness check matches the cached id against `cx.windows()` — the
/// authoritative list of live windows. (We can't use the cleaner Zed-style
/// `cx.windows().find_map(downcast::<SettingsView>)` because both our
/// chat and settings windows wrap their views in `gpui_component::Root`,
/// which is required by `Root::read` calls inside the `Input` widget — so
/// they're not distinguishable by root view type.) A stale id self-heals
/// here: if the cached window was closed, the containment check fails and
/// we clear the cache.
fn try_focus_existing_settings(cx: &mut App) -> bool {
    let Some(handle) = cx.global::<AppGlobal>().settings_window else {
        return false;
    };
    let alive = cx
        .windows()
        .iter()
        .any(|w| w.window_id() == handle.window_id());
    if !alive {
        cx.global_mut::<AppGlobal>().settings_window = None;
        return false;
    }
    handle
        .update(cx, |_, window, _| window.activate_window())
        .ok();
    true
}

/// Edge-to-edge titlebar: macOS extends the content view under the
/// traffic-light buttons and stops painting a separate titlebar background.
/// Each view is responsible for leaving room at the top so the lights don't
/// land on real UI — see `chat::TITLE_BAR_RESERVE` (vertical reserve + fade
/// gradient) and `settings::TAB_STRIP_LEFT_PAD` (horizontal pad — the tab
/// row doubles as the title bar).
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

fn open_main_window(cx: &mut App) {
    let core = cx.global::<AppGlobal>().core.clone();

    // Square chat window. Side = 90% of the smaller display dimension,
    // capped at 800px. A square frames the chat as a writing surface —
    // a sheet of paper, not a wide chat pane — and the cap keeps the
    // prose column from feeling lost in the middle of a 4K display. If
    // there's no primary display (rare; offscreen render contexts), we
    // fall back to the cap.
    let side = match cx.primary_display() {
        Some(d) => {
            let s = d.bounds().size;
            let smaller = f32::min(s.width.as_f32(), s.height.as_f32());
            (smaller * 0.9).min(705.0)
        }
        None => 820.0,
    };
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
        let view = cx.new(|cx| ChatView::new(core.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });

    // Bring the app forward so the new window comes to the front, even when
    // the action originated from another app's context (e.g. the dock
    // right-click menu while a different app is foreground). `focus: true`
    // in WindowOptions makes the window key within our app, but doesn't
    // by itself activate the app vs other apps.
    cx.activate(true);
}

fn open_settings_window(cx: &mut App) {
    let core = cx.global::<AppGlobal>().core.clone();
    let bounds = centered_window_bounds(cx, 560., 480.);

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(420.), px(320.))),
        ..Default::default()
    };

    let handle = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| SettingsView::new(core.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });

    if let Ok(handle) = handle {
        cx.global_mut::<AppGlobal>().settings_window = Some(handle);
    }
}
