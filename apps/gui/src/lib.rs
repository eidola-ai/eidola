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
    App, AppContext, AsyncApp, Bounds, Entity, KeyBinding, Menu, MenuItem, TitlebarOptions,
    WindowBounds, WindowKind, WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_component_assets::Assets;

use crate::actions::{About, OpenSettings, Quit};
use crate::chat::ChatView;
use crate::core::Core;
use crate::settings::SettingsView;

/// Application-scoped state — currently just the shared `Core`. Stored as a
/// gpui global so action handlers (which only get `&mut App`) can reach it.
struct AppGlobal {
    core: Entity<Core>,
}

impl gpui::Global for AppGlobal {}

/// Run the GUI application. The binary's `fn main()` is a thin shim around
/// this; tests do not call this — they use `tests/visual.rs` instead.
pub fn run() {
    gpui_platform::application()
        .with_assets(Assets)
        .run(move |cx: &mut App| {
            gpui_component::init(cx);
            theme::install(cx);

            let core = Core::new(cx);
            cx.set_global(AppGlobal { core: core.clone() });

            install_menus(cx);
            install_keybindings(cx);
            install_action_handlers(cx);

            cx.spawn(async move |cx: &mut AsyncApp| {
                open_main_window(core, cx).await;
            })
            .detach();
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
                MenuItem::action("Quit", Quit),
            ],
            disabled: false,
        },
        Menu {
            name: "Edit".into(),
            items: vec![
                MenuItem::action("Undo", gpui_component::input::Undo),
                MenuItem::action("Redo", gpui_component::input::Redo),
                MenuItem::Separator,
                MenuItem::action("Cut", gpui_component::input::Cut),
                MenuItem::action("Copy", gpui_component::input::Copy),
                MenuItem::action("Paste", gpui_component::input::Paste),
                MenuItem::Separator,
                MenuItem::action("Select All", gpui_component::input::SelectAll),
            ],
            disabled: false,
        },
    ]);
}

fn install_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("cmd-,", OpenSettings, None),
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("cmd-enter", crate::chat::Send, Some("ChatView")),
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
        let core = cx.global::<AppGlobal>().core.clone();
        cx.spawn(async move |cx: &mut AsyncApp| {
            open_settings_window(core, cx).await;
        })
        .detach();
    });
}

fn centered_window_bounds(cx: &mut App, w: f32, h: f32) -> Option<WindowBounds> {
    let display = cx.primary_display()?;
    let center = display.bounds().center();
    Some(WindowBounds::Windowed(Bounds::centered_at(
        center,
        size(px(w), px(h)),
    )))
}

async fn open_main_window(core: Entity<Core>, cx: &mut AsyncApp) {
    let bounds = cx.update(|cx| centered_window_bounds(cx, 900., 640.));

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(TitlebarOptions {
            title: Some("Eidola".into()),
            appears_transparent: false,
            traffic_light_position: Some(point(px(10.), px(10.))),
        }),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(480.), px(360.))),
        ..Default::default()
    };

    let _ = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| ChatView::new(core.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });
}

async fn open_settings_window(core: Entity<Core>, cx: &mut AsyncApp) {
    let bounds = cx.update(|cx| centered_window_bounds(cx, 560., 480.));

    let opts = WindowOptions {
        window_bounds: bounds,
        titlebar: Some(TitlebarOptions {
            title: Some("Eidola Settings".into()),
            appears_transparent: false,
            traffic_light_position: Some(point(px(10.), px(10.))),
        }),
        kind: WindowKind::Normal,
        window_min_size: Some(size(px(420.), px(320.))),
        ..Default::default()
    };

    let _ = cx.open_window(opts, |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| SettingsView::new(core.clone(), window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    });
}
