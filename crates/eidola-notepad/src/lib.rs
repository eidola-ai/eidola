//! Eidola Notepad — a plain markdown file editor on Eidola's writing surface.
//!
//! One window per document, each the composer's editor (`gpui-markdown-editor`)
//! in Eidola's own theme, prose typography and window chrome — taken from
//! `eidola-gui` itself rather than copied, so the two cannot drift. Around it,
//! what a Mac document app is expected to do: the File menu (New, Open, Open
//! Recent, Close, Save, Save As, Revert), the standard Edit menu, files dropped
//! on the Dock icon or into a window, the unsaved-changes sheet on close and
//! at quit, and the edited dot in the close button.
//!
//! The palette follows the reader's Eidola appearance settings (`config.toml`,
//! read only), including the time-of-day tint; the View menu's zoom is this
//! app's own and lasts the session.

pub mod about;
pub mod actions;
pub mod document;
pub mod file;
pub mod recent;

use std::path::{Path, PathBuf};

use futures::StreamExt;
use gpui::{
    App, AppContext, Bounds, EntityId, Global, KeyBinding, Menu, MenuItem, OsAction,
    PathPromptOptions, Pixels, PromptButton, PromptLevel, WeakEntity, WindowBounds, WindowHandle,
    WindowKind, WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_kit_assets::Assets;

use crate::actions::*;
use crate::document::{Contents, DocumentView};
use crate::recent::RecentDocuments;

/// How far each new window steps down and right from the one in front.
const CASCADE: f32 = 24.;

struct Notepad {
    recent: RecentDocuments,
    /// The ordinal the next untitled document takes. Counts up for the whole
    /// session, so two windows never share a name.
    next_untitled: usize,
    documents: Vec<(WindowHandle<Root>, WeakEntity<DocumentView>)>,
    about: Option<WindowHandle<Root>>,
    /// The front document window's frame, for cascading the next one.
    front_bounds: Option<Bounds<Pixels>>,
    /// View ▸ Zoom, once the reader has used it. Until then the type scale
    /// follows Eidola's own.
    zoom: Option<f32>,
}

impl Global for Notepad {}

/// Run the app. `main` is a thin shim around this.
pub fn run() {
    // Files arrive from AppKit — dropped on the Dock icon, opened from
    // Finder — through a callback that has no `App`, and at launch they can
    // arrive before the run callback. A channel carries them across both.
    let (url_tx, mut url_rx) = futures::channel::mpsc::unbounded::<Vec<String>>();
    let launch_paths = paths_from_args(std::env::args().skip(1));

    let application = gpui_platform::application().with_assets(Assets);
    application.on_open_urls(move |urls| {
        let _ = url_tx.unbounded_send(urls);
    });
    // A Dock click with no window open starts a new document.
    application.on_reopen(|cx: &mut App| {
        if cx.has_global::<Notepad>() && live_documents(cx).is_empty() {
            new_document(cx);
        }
    });

    application.run(move |cx: &mut App| {
        install(RecentDocuments::load_default(), cx);
        eidola_gui::theme::start_clock(cx);
        cx.activate(true);

        let mut initial = launch_paths;
        while let Ok(urls) = url_rx.try_recv() {
            initial.extend(paths_from_urls(urls));
        }

        // The launch documents open from a task rather than inline: this
        // callback is not an `update`, so effects queued here (the deferred
        // open, a refusal's sheet) would wait for some unrelated event to be
        // flushed, and a launch with a file would show nothing until then.
        // Then the same task becomes the app-lifetime listener for files
        // handed over later.
        cx.spawn(async move |cx: &mut gpui::AsyncApp| {
            cx.update(|cx| {
                if initial.is_empty() {
                    new_document(cx);
                } else {
                    open_paths_now(initial, cx);
                }
            });
            while let Some(urls) = url_rx.next().await {
                let paths = paths_from_urls(urls);
                cx.update(|cx| open_paths(paths, cx));
            }
        })
        .detach();
    });
}

/// Everything the app sets up before its first window: components, the
/// theme, the app state, keys, menus and handlers. Public so tests can stand
/// the app up without a platform run loop.
pub fn install(recent: RecentDocuments, cx: &mut App) {
    gpui_component::init(cx);
    gpui_component::set_window_border_enabled(false, cx);
    eidola_gui::theme::install(cx);

    cx.set_global(Notepad {
        recent,
        next_untitled: 1,
        documents: Vec::new(),
        about: None,
        front_bounds: None,
        zoom: None,
    });
    follow_eidola_appearance(cx);

    // Keys before menus: `set_menus` reads each item's key equivalent out of
    // the keymap as it builds the menu bar.
    install_keybindings(cx);
    install_menus(cx);
    install_action_handlers(cx);
}

/// The open document windows' views, front to back.
pub fn documents(cx: &mut App) -> Vec<gpui::Entity<DocumentView>> {
    live_documents(cx)
        .into_iter()
        .filter_map(|(_, view)| view.upgrade())
        .collect()
}

// ---------------------------------------------------------------------------
// Appearance
// ---------------------------------------------------------------------------

/// Wear the reader's Eidola appearance: day/night, the time-of-day tint, the
/// fixed light character, and — until View ▸ Zoom is used here — the type
/// scale. Read again whenever a document window comes forward, so a change
/// made in Eidola reaches an open notepad.
pub fn follow_eidola_appearance(cx: &mut App) {
    use eidola_app_core::config::{Config, default_config_path};
    let config = default_config_path()
        .map(|path| Config::load_from(&path))
        .unwrap_or_default();
    let zoom = cx.try_global::<Notepad>().and_then(|n| n.zoom);
    eidola_gui::theme::set_settings(
        eidola_gui::theme::ThemeSettings {
            appearance: config.appearance(),
            tint: config.time_of_day_tint(),
            character: config.light_character(),
            font_scale: zoom.unwrap_or_else(|| config.font_scale()),
        },
        cx,
    );
}

fn set_zoom(scale: f32, cx: &mut App) {
    cx.global_mut::<Notepad>().zoom = Some(scale);
    let settings = eidola_gui::theme::settings(cx);
    eidola_gui::theme::set_settings(
        eidola_gui::theme::ThemeSettings {
            font_scale: scale,
            ..settings
        },
        cx,
    );
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

/// The open document windows, front to back where the platform knows the
/// order.
fn live_documents(cx: &mut App) -> Vec<(WindowHandle<Root>, WeakEntity<DocumentView>)> {
    let live: Vec<_> = cx.windows().iter().map(|w| w.window_id()).collect();
    let notepad = cx.global_mut::<Notepad>();
    notepad
        .documents
        .retain(|(handle, view)| live.contains(&handle.window_id()) && view.upgrade().is_some());
    let mut documents = notepad.documents.clone();
    if let Some(stack) = cx.window_stack() {
        documents.sort_by_key(|(handle, _)| {
            stack
                .iter()
                .position(|w| w.window_id() == handle.window_id())
                .unwrap_or(usize::MAX)
        });
    }
    documents
}

pub fn new_document(cx: &mut App) {
    let notepad = cx.global_mut::<Notepad>();
    let n = notepad.next_untitled;
    notepad.next_untitled += 1;
    open_document_window(Contents::Untitled(n), cx);
}

fn open_document_window(contents: Contents, cx: &mut App) -> Option<WindowHandle<Root>> {
    let bounds = next_window_bounds(cx);
    let options = WindowOptions {
        window_bounds: Some(bounds),
        titlebar: Some(eidola_gui::transparent_titlebar()),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(360.), px(280.))),
        #[cfg(not(target_os = "macos"))]
        app_id: Some("ai.eidola.notepad".into()),
        ..Default::default()
    };
    let mut created = None;
    let handle = cx
        .open_window(options, |window, cx| {
            eidola_gui::theme::observe_window_appearance(window);
            let view = cx.new(|cx| DocumentView::new(contents, window, cx));
            let focus = view.read(cx).editor().read(cx).focus_handle.clone();
            window.focus(&focus, cx);
            created = Some(view.downgrade());
            cx.new(|cx| eidola_gui::chrome::themed_root(view.into(), window, cx))
        })
        .ok()?;
    if let Some(view) = created {
        let notepad = cx.global_mut::<Notepad>();
        notepad.documents.push((handle, view));
        notepad.front_bounds = Some(match bounds {
            WindowBounds::Windowed(b)
            | WindowBounds::Maximized(b)
            | WindowBounds::Fullscreen(b) => b,
        });
    }
    cx.activate(true);
    Some(handle)
}

/// A new window opens at Eidola's writing-surface size, stepped down and
/// right from the window in front of it (centred when it is the first).
fn next_window_bounds(cx: &mut App) -> WindowBounds {
    if let Some(front) = cx.global::<Notepad>().front_bounds
        && !live_documents(cx).is_empty()
    {
        let mut origin = front.origin + point(px(CASCADE), px(CASCADE));
        if let Some(display) = cx.primary_display() {
            let screen = display.bounds();
            // Walk back to the top-left once a cascade would leave the screen.
            if origin.x + front.size.width > screen.right()
                || origin.y + front.size.height > screen.bottom()
            {
                origin = screen.origin + point(px(CASCADE * 2.), px(CASCADE * 2.));
            }
        }
        return WindowBounds::Windowed(Bounds::new(origin, front.size));
    }
    let (w, h) = eidola_gui::writing_surface_size(cx);
    match cx.primary_display() {
        Some(display) => WindowBounds::Windowed(Bounds::centered_at(
            display.bounds().center(),
            size(px(w), px(h)),
        )),
        None => WindowBounds::Windowed(Bounds::new(point(px(80.), px(80.)), size(px(w), px(h)))),
    }
}

/// A document window reports its frame as it moves or comes forward.
pub fn note_front_bounds(bounds: Bounds<Pixels>, cx: &mut App) {
    if cx.has_global::<Notepad>() {
        cx.global_mut::<Notepad>().front_bounds = Some(bounds);
    }
}

/// Open each file — raising the window it is already in, taking the place of
/// a front untitled document nobody has typed in, or in a new window.
/// Deferred, because the request often arrives from inside a window's own
/// update (a drop, a menu item) and opening may need to update that window.
pub fn open_paths(paths: Vec<PathBuf>, cx: &mut App) {
    cx.defer(move |cx| open_paths_now(paths, cx));
}

fn open_paths_now(paths: Vec<PathBuf>, cx: &mut App) {
    let mut failures = Vec::new();
    for path in paths {
        if let Some(handle) = window_showing(&path, cx) {
            handle
                .update(cx, |_, window, _| window.activate_window())
                .ok();
            continue;
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let read = if path.is_dir() {
            Err(format!("“{name}” is a folder."))
        } else {
            match std::fs::read(&path) {
                Ok(bytes) => file::decode(&bytes).map_err(|_| {
                    format!("“{name}” isn’t a UTF-8 text file, so it can’t be opened here.")
                }),
                Err(error) => Err(format!("“{name}” couldn’t be opened. {error}.")),
            }
        };
        let (text, format) = match read {
            Ok(contents) => contents,
            Err(message) => {
                failures.push(message);
                continue;
            }
        };
        note_recent(&path, cx);
        if let Some((handle, view)) = pristine_front_document(cx) {
            let loaded = handle.update(cx, |_, window, cx| {
                view.update(cx, |view, cx| {
                    view.load(path.clone(), text.clone(), format, window, cx)
                })
            });
            if loaded.is_ok() {
                handle
                    .update(cx, |_, window, _| window.activate_window())
                    .ok();
                continue;
            }
        }
        open_document_window(Contents::File { path, text, format }, cx);
    }
    if !failures.is_empty() {
        report_failures(failures, cx);
    }
}

fn window_showing(path: &Path, cx: &mut App) -> Option<WindowHandle<Root>> {
    let wanted = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    live_documents(cx).into_iter().find_map(|(handle, view)| {
        let open = view.upgrade()?.read(cx).path()?.to_path_buf();
        let open = std::fs::canonicalize(&open).unwrap_or(open);
        (open == wanted).then_some(handle)
    })
}

fn pristine_front_document(
    cx: &mut App,
) -> Option<(WindowHandle<Root>, gpui::Entity<DocumentView>)> {
    let (handle, view) = live_documents(cx).into_iter().next()?;
    let view = view.upgrade()?;
    view.read(cx).is_pristine(cx).then_some((handle, view))
}

/// Say why files did not open, on the front document window — opening an
/// untitled one to say it on when nothing else is open.
fn report_failures(failures: Vec<String>, cx: &mut App) {
    if live_documents(cx).is_empty() {
        new_document(cx);
    }
    let Some((handle, _)) = live_documents(cx).into_iter().next() else {
        for failure in &failures {
            eprintln!("eidola-notepad: {failure}");
        }
        return;
    };
    let (message, detail) = if failures.len() == 1 {
        (failures[0].clone(), None)
    } else {
        (
            format!("{} documents couldn’t be opened.", failures.len()),
            Some(failures.join("\n")),
        )
    };
    cx.defer(move |cx| {
        handle
            .update(cx, |_, window, cx| {
                document::alert(
                    window,
                    cx,
                    PromptLevel::Warning,
                    &message,
                    detail.as_deref(),
                );
            })
            .ok();
    });
}

/// Remember a document in File ▸ Open Recent and the Dock's recents.
pub fn note_recent(path: &Path, cx: &mut App) {
    cx.add_recent_document(path);
    if cx.has_global::<Notepad>() {
        cx.global_mut::<Notepad>().recent.note(path);
        install_menus(cx);
    }
}

fn prompt_open(cx: &mut App) {
    let chosen = cx.prompt_for_paths(PathPromptOptions {
        files: true,
        directories: false,
        multiple: true,
        prompt: Some("Open".into()),
    });
    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
        if let Ok(Ok(Some(paths))) = chosen.await {
            cx.update(|cx| open_paths(paths, cx));
        }
    })
    .detach();
}

/// ⌘Q: every edited document is settled first — saved, or discarded on the
/// reader's word — and one Cancel keeps the app open.
///
/// Windows stay editable while a sheet is up, so the list is taken again
/// after each round: a document edited since it was settled, or since the
/// round began, is reviewed again before the app quits.
pub fn quit(cx: &mut App) {
    let dirty = unsettled_documents(&[], cx);
    if dirty.is_empty() {
        cx.quit();
        return;
    }
    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
        if dirty.len() > 1 {
            let (handle, _) = &dirty[0];
            let message = format!(
                "You have {} documents with unsaved changes. Do you want to review these changes before quitting?",
                dirty.len()
            );
            let Ok(answer) = handle.update(cx, |_, window, cx| {
                window.activate_window();
                window.prompt(
                    PromptLevel::Warning,
                    &message,
                    Some("If you don’t review your documents, all your changes will be lost."),
                    &[
                        PromptButton::ok("Review Changes…"),
                        PromptButton::new("Discard Changes"),
                        PromptButton::cancel("Cancel"),
                    ],
                    cx,
                )
            }) else {
                return;
            };
            match answer.await {
                Ok(0) => {}
                Ok(1) => {
                    cx.update(|cx| cx.quit());
                    return;
                }
                _ => return,
            }
        }
        // What each document held when the reader settled it.
        let mut settled: Vec<(EntityId, String)> = Vec::new();
        let mut pending = dirty;
        while !pending.is_empty() {
            for (handle, view) in pending {
                let review = handle.update(cx, |_, window, cx| {
                    view.update(cx, |view, cx| view.review_unsaved(window, cx))
                });
                let Ok(Ok(review)) = review else { continue };
                if !review.await {
                    return;
                }
                if let Some(view) = view.upgrade() {
                    let text = cx.update(|cx| view.read(cx).editor().read(cx).value().to_string());
                    settled.retain(|(id, _)| *id != view.entity_id());
                    settled.push((view.entity_id(), text));
                }
            }
            pending = cx.update(|cx| unsettled_documents(&settled, cx));
        }
        cx.update(|cx| cx.quit());
    })
    .detach();
}

/// Edited documents the quit review still has to ask about: any not yet
/// settled, and any whose text moved on since it was.
fn unsettled_documents(
    settled: &[(EntityId, String)],
    cx: &mut App,
) -> Vec<(WindowHandle<Root>, WeakEntity<DocumentView>)> {
    live_documents(cx)
        .into_iter()
        .filter(|(_, view)| {
            let Some(view) = view.upgrade() else {
                return false;
            };
            let document = view.read(cx);
            let text = document.editor().read(cx).value();
            document.is_dirty()
                && !settled
                    .iter()
                    .any(|(id, settled)| *id == view.entity_id() && settled.as_str() == text)
        })
        .collect()
}

fn open_about(cx: &mut App) {
    if let Some(handle) = cx.global::<Notepad>().about
        && handle
            .update(cx, |_, window, _| window.activate_window())
            .is_ok()
    {
        return;
    }
    let bounds = cx.primary_display().map(|d| {
        WindowBounds::Windowed(Bounds::centered_at(
            d.bounds().center(),
            size(px(340.), px(300.)),
        ))
    });
    let options = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(eidola_gui::transparent_titlebar()),
        kind: WindowKind::Normal,
        is_resizable: false,
        window_min_size: Some(size(px(300.), px(260.))),
        ..Default::default()
    };
    let handle = cx.open_window(options, |window, cx| {
        eidola_gui::theme::observe_window_appearance(window);
        window.set_window_title("About Eidola Notepad");
        let view = cx.new(|cx| about::AboutView::new(window, cx));
        cx.new(|cx| eidola_gui::chrome::themed_root(view.into(), window, cx))
    });
    cx.global_mut::<Notepad>().about = handle.ok();
    cx.activate(true);
}

// ---------------------------------------------------------------------------
// Menus, keys and handlers
// ---------------------------------------------------------------------------

fn install_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("secondary-n", NewDocument, None),
        KeyBinding::new("secondary-o", Open, None),
        KeyBinding::new("secondary-w", CloseWindow, None),
        KeyBinding::new("secondary-s", Save, None),
        KeyBinding::new("secondary-shift-s", SaveAs, None),
        KeyBinding::new("secondary-q", Quit, None),
        KeyBinding::new("secondary-0", ActualSize, None),
        KeyBinding::new("secondary-=", ZoomIn, None),
        KeyBinding::new("secondary-+", ZoomIn, None),
        KeyBinding::new("secondary--", ZoomOut, None),
    ]);
    #[cfg(target_os = "macos")]
    cx.bind_keys([
        KeyBinding::new("cmd-h", Hide, None),
        KeyBinding::new("alt-cmd-h", HideOthers, None),
        KeyBinding::new("cmd-m", Minimize, None),
        KeyBinding::new("ctrl-cmd-f", ToggleFullScreen, None),
    ]);
    // Motion, editing, clipboard, undo and the bold/italic toggles — the
    // editor's own keymap, exactly as Eidola installs it.
    gpui_markdown_editor::init(cx);
}

fn install_menus(cx: &mut App) {
    let recent: Vec<PathBuf> = cx
        .global::<Notepad>()
        .recent
        .paths()
        .map(Path::to_path_buf)
        .collect();
    let mut recent_items: Vec<MenuItem> = recent
        .iter()
        .map(|path| {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            MenuItem::action(name, OpenRecent { path: path.clone() })
        })
        .collect();
    if !recent_items.is_empty() {
        recent_items.push(MenuItem::separator());
    }
    recent_items.push(MenuItem::action("Clear Menu", ClearRecentDocuments));

    cx.set_menus(vec![
        Menu {
            name: "Eidola Notepad".into(),
            items: vec![
                MenuItem::action("About Eidola Notepad", About),
                MenuItem::separator(),
                MenuItem::action("Hide Eidola Notepad", Hide),
                MenuItem::action("Hide Others", HideOthers),
                MenuItem::action("Show All", ShowAll),
                MenuItem::separator(),
                MenuItem::action("Quit Eidola Notepad", Quit),
            ],
            disabled: false,
        },
        Menu {
            name: "File".into(),
            items: vec![
                MenuItem::action("New", NewDocument),
                MenuItem::action("Open…", Open),
                MenuItem::Submenu(Menu {
                    name: "Open Recent".into(),
                    items: recent_items,
                    disabled: false,
                }),
                MenuItem::separator(),
                // The document-scoped commands are registered by each
                // document window, so macOS greys them with none in front —
                // and Revert only while there is something to revert.
                MenuItem::action("Close", CloseWindow),
                MenuItem::action("Save", Save),
                MenuItem::action("Save As…", SaveAs),
                MenuItem::action("Revert to Saved", RevertToSaved),
                MenuItem::separator(),
                MenuItem::action("Show in Finder", ShowInFinder),
            ],
            disabled: false,
        },
        // `os_action` ties the clipboard verbs to the standard selectors so
        // the responder chain reaches system text fields (the save panel's
        // name field) too. Undo/Redo stay on gpui's own dispatch, as in
        // Eidola: gpui disables the OS undo:/redo: selectors when no native
        // text view holds focus.
        Menu {
            name: "Edit".into(),
            items: vec![
                MenuItem::action("Undo", gpui_component::input::Undo),
                MenuItem::action("Redo", gpui_component::input::Redo),
                MenuItem::separator(),
                MenuItem::os_action("Cut", gpui_component::input::Cut, OsAction::Cut),
                MenuItem::os_action("Copy", gpui_component::input::Copy, OsAction::Copy),
                MenuItem::os_action("Paste", gpui_component::input::Paste, OsAction::Paste),
                MenuItem::action("Paste and Match Style", gpui_markdown_editor::PastePlain),
                MenuItem::os_action(
                    "Select All",
                    gpui_component::input::SelectAll,
                    OsAction::SelectAll,
                ),
            ],
            disabled: false,
        },
        Menu {
            name: "Format".into(),
            items: vec![
                MenuItem::action("Bold", gpui_markdown_editor::ToggleBold),
                MenuItem::action("Italic", gpui_markdown_editor::ToggleItalic),
            ],
            disabled: false,
        },
        Menu {
            name: "View".into(),
            items: vec![
                MenuItem::action("Actual Size", ActualSize),
                MenuItem::action("Zoom In", ZoomIn),
                MenuItem::action("Zoom Out", ZoomOut),
                MenuItem::separator(),
                MenuItem::action("Enter Full Screen", ToggleFullScreen),
            ],
            disabled: false,
        },
        // Named "Window" so AppKit adopts it as the canonical Window menu and
        // lists the open documents in it.
        Menu {
            name: "Window".into(),
            items: vec![
                MenuItem::action("Minimize", Minimize),
                MenuItem::action("Zoom", Zoom),
            ],
            disabled: false,
        },
    ]);

    cx.set_dock_menu(vec![MenuItem::action("New Document", NewDocument)]);
}

fn install_action_handlers(cx: &mut App) {
    cx.on_action(|_: &Quit, cx: &mut App| quit(cx));
    cx.on_action(|_: &About, cx: &mut App| open_about(cx));
    cx.on_action(|_: &Hide, cx: &mut App| cx.hide());
    cx.on_action(|_: &HideOthers, cx: &mut App| cx.hide_other_apps());
    cx.on_action(|_: &ShowAll, cx: &mut App| cx.unhide_other_apps());
    cx.on_action(|_: &NewDocument, cx: &mut App| new_document(cx));
    cx.on_action(|_: &Open, cx: &mut App| prompt_open(cx));
    cx.on_action(|action: &OpenRecent, cx: &mut App| open_paths(vec![action.path.clone()], cx));
    cx.on_action(|_: &ClearRecentDocuments, cx: &mut App| {
        cx.global_mut::<Notepad>().recent.clear();
        install_menus(cx);
    });

    cx.on_action(|_: &ActualSize, cx: &mut App| {
        set_zoom(eidola_app_core::config::FONT_SCALE_DEFAULT, cx)
    });
    cx.on_action(|_: &ZoomIn, cx: &mut App| {
        let next = eidola_app_core::config::font_scale_step_up(eidola_gui::theme::font_scale(cx));
        set_zoom(next, cx);
    });
    cx.on_action(|_: &ZoomOut, cx: &mut App| {
        let next = eidola_app_core::config::font_scale_step_down(eidola_gui::theme::font_scale(cx));
        set_zoom(next, cx);
    });

    // The window commands act on the window the action came through, which
    // is mid-update while the handler runs — so the call is deferred until
    // it has unwound (the same reason Eidola defers them).
    cx.on_action(|_: &Minimize, cx: &mut App| {
        with_active_window(cx, |window| window.minimize_window())
    });
    cx.on_action(|_: &Zoom, cx: &mut App| with_active_window(cx, |window| window.zoom_window()));
    cx.on_action(|_: &ToggleFullScreen, cx: &mut App| {
        with_active_window(cx, |window| window.toggle_fullscreen())
    });
}

fn with_active_window(cx: &mut App, f: impl FnOnce(&mut gpui::Window) + 'static) {
    let Some(handle) = cx.active_window() else {
        return;
    };
    cx.defer(move |cx| {
        handle.update(cx, |_, window, _| f(window)).ok();
    });
}

// ---------------------------------------------------------------------------
// Launch arguments and URLs
// ---------------------------------------------------------------------------

/// Paths named on the command line. LaunchServices passes a `-psn_…`
/// argument and Xcode its own `-NS…`/`-Apple…` defaults, each followed by its
/// value; none of those is a document.
fn paths_from_args(args: impl Iterator<Item = String>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        if arg.starts_with("-NS") || arg.starts_with("-Apple") {
            args.next_if(|value| !value.starts_with('-'));
        } else if !arg.starts_with('-') {
            let path = PathBuf::from(arg);
            paths.push(std::path::absolute(&path).unwrap_or(path));
        }
    }
    paths
}

/// The `file://` URLs AppKit hands over, as paths. Anything else — a
/// custom scheme — is not a document and is ignored.
fn paths_from_urls(urls: Vec<String>) -> Vec<PathBuf> {
    urls.iter()
        .filter_map(|url| url.strip_prefix("file://"))
        .map(|rest| {
            // `file://localhost/…` names the same file as `file:///…`.
            let rest = rest.strip_prefix("localhost").unwrap_or(rest);
            PathBuf::from(percent_decode(rest))
        })
        .collect()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            )
        {
            out.push(((h << 4) | l) as u8);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dock_drops_arrive_as_file_urls() {
        let paths = paths_from_urls(vec![
            "file:///Users/me/Notes/My%20Idea.md".into(),
            "file://localhost/tmp/a.md".into(),
            "notepad://ignored".into(),
        ]);
        assert_eq!(
            paths,
            [
                PathBuf::from("/Users/me/Notes/My Idea.md"),
                PathBuf::from("/tmp/a.md")
            ]
        );
    }

    #[test]
    fn launch_flags_are_not_documents() {
        let paths = paths_from_args(
            [
                "-psn_0_12345",
                "/tmp/a.md",
                "-NSDocumentRevisionsDebugMode",
                "YES",
                "/tmp/b.md",
            ]
            .into_iter()
            .map(String::from),
        );
        assert_eq!(
            paths,
            [PathBuf::from("/tmp/a.md"), PathBuf::from("/tmp/b.md")]
        );
    }
}
