//! Behavior tests for the QA probe registry (`eidola_gui::probe`) — the
//! annotation layer that feeds both the AccessKit tree and the UI driver
//! (`examples/driver.rs`).
//!
//! Probes record during prepaint into a process-global registry, and the
//! enabled flag is process-global too, so these tests serialize on a local
//! mutex: parallel libtest threads would otherwise collide on window ids
//! (each `TestAppContext` numbers its windows from zero) and on the flag.
//! This file is its own test binary, so the lock never blocks other suites.

use std::sync::{Mutex, MutexGuard};

use eidola_app_core::{BalancesResult, ConfigState, ModelInfo};
use eidola_gui::chat::{ChatView, ToggleModelPicker};
use eidola_gui::library::LibraryView;
use eidola_gui::probe;
use eidola_gui::stores::{Stores, StoresStub};
use eidola_gui::window_input::WindowInput;
use gpui::{AnyWindowHandle, AppContext, Entity, TestAppContext, WindowOptions};
use gpui_component::Root;

static LOCK: Mutex<()> = Mutex::new(());

/// Serialize the test and leave probes enabled for its duration.
fn probes_on() -> MutexGuard<'static, ()> {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    probe::set_probes_enabled(true);
    guard
}

#[gpui::test]
fn chat_probes_record_names_roles_and_bounds(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"chat/composer"),
        "composer probe missing; recorded: {names:?}"
    );
    assert!(
        names.contains(&"chat/transcript"),
        "transcript probe missing; recorded: {names:?}"
    );

    let composer = &entries
        .iter()
        .find(|(n, _)| n == "chat/composer")
        .unwrap()
        .1;
    assert_eq!(format!("{:?}", composer.role), "TextInput");
    assert_eq!(composer.label.as_ref(), "Message composer");
    assert!(
        composer.bounds.size.width.as_f32() > 100.0 && composer.bounds.size.height.as_f32() > 10.0,
        "composer bounds should be a real painted area, got {:?}",
        composer.bounds
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn picker_probes_appear_on_open_and_clear_on_dismiss(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    let id = window.window_id().as_u64();

    // Closed picker: no listbox probes.
    draw(cx, window);
    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("chat/model-picker")),
        "picker probes before opening: {names:?}"
    );

    // Open via the real action dispatch path.
    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"chat/model-picker".to_string()),
        "picker panel probe missing after open: {names:?}"
    );
    assert!(
        names.contains(&"chat/model-picker/row/0".to_string())
            && names.contains(&"chat/model-picker/row/2".to_string()),
        "per-model row probes missing: {names:?}"
    );

    // Dismiss: the clear-then-redraw dance must drop the unmounted picker —
    // stale entries would be ghost click targets for the driver.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("chat/model-picker")),
        "picker probes must clear after dismiss: {names:?}"
    );
    assert!(
        names.contains(&"chat/composer".to_string()),
        "still-mounted probes must survive the refresh: {names:?}"
    );

    let _ = id;
    probe::set_probes_enabled(false);
}

#[gpui::test]
fn library_rows_probe_with_indexed_names(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker, closures, and lifetimes")),
        ];
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let row0 = entries.iter().find(|(n, _)| n == "library/row/0");
    assert!(
        row0.is_some(),
        "library row probe missing; recorded: {:?}",
        entries.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    assert_eq!(row0.unwrap().1.label.as_ref(), "Tides and the moon");

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn disabled_probes_record_nothing(cx: &mut TestAppContext) {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    probe::set_probes_enabled(false);

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    // Window ids restart per TestAppContext, so an earlier (enabled) test in
    // this process may have recorded under the same id — clear first, then
    // prove a disabled draw records nothing.
    probe::clear_window(window.window_id().as_u64());
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    assert!(
        entries.is_empty(),
        "probes disabled must record nothing, got {:?}",
        entries.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Helpers (mirroring tests/behavior.rs)
// ---------------------------------------------------------------------------

/// Clear → redraw → read: the same staleness dance the driver's `elements`
/// command performs.
fn fresh_names(cx: &mut TestAppContext, window: AnyWindowHandle) -> Vec<String> {
    probe::clear_window(window.window_id().as_u64());
    draw(cx, window);
    probe::window_entries(window.window_id().as_u64())
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

/// Force a frame on a test window. `window.refresh()` marks it dirty; the
/// parked dispatcher then runs the scheduled draw.
fn draw(cx: &mut TestAppContext, window: AnyWindowHandle) {
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
}

fn stub_stores(cx: &mut TestAppContext, setup: impl FnOnce(&mut StoresStub)) -> Stores {
    cx.update(|cx| {
        let mut fixture = StoresStub::default();
        setup(&mut fixture);
        Stores::stub_with(fixture, cx)
    })
}

fn ready_stores(cx: &mut TestAppContext) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(ConfigState {
            base_url: "https://eidola.example/v1".into(),
            default_model: "gemma4-31b".into(),
            base_url_pin: "https://eidola.example/v1".into(),
            base_url_is_override: false,
            has_account: true,
            has_account_secret: true,
            domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
            trusted_measurements: Vec::new(),
            trusted_measurements_are_override: false,
            has_hardware_root_ca: false,
            has_hardware_intermediate_ca: false,
            attestation_url: None,
        });
        s.balances = Some(BalancesResult {
            available: 4_200_000,
            pools: Vec::new(),
        });
        s.models = vec![
            ModelInfo {
                id: "gemma4-31b".into(),
                context_length: 131_072,
                prompt_credits_per_token: 0.53,
                completion_credits_per_token: 1.5,
                request_credits: None,
            },
            ModelInfo {
                id: "kimi-k2-6".into(),
                context_length: 262_144,
                prompt_credits_per_token: 3.0,
                completion_credits_per_token: 9.0,
                request_credits: None,
            },
            ModelInfo {
                id: "qwen3-coder-watt".into(),
                context_length: 131_072,
                prompt_credits_per_token: 1.05,
                completion_credits_per_token: 5.25,
                request_credits: None,
            },
        ];
    })
}

fn space_info(id: &str, title: Option<&str>) -> eidola_app_core::SpaceInfo {
    let ts = eidola_app_core::now_ms();
    eidola_app_core::SpaceInfo {
        id: id.into(),
        title: title.map(String::from),
        snippet: None,
        created_at: ts,
        last_activity_at: ts,
        message_count: 4,
        archived_at: None,
    }
}

fn open_view<V: gpui::Render + 'static>(
    cx: &mut TestAppContext,
    build: impl FnOnce(&mut gpui::Window, &mut gpui::App) -> Entity<V>,
) -> (AnyWindowHandle, Entity<V>) {
    cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);

        let mut inner: Option<Entity<V>> = None;
        let window = cx
            .open_window(WindowOptions::default(), |window, cx| {
                let view = build(window, cx);
                inner = Some(view.clone());
                cx.new(|cx| Root::new(view, window, cx))
            })
            .expect("open test window");
        (window.into(), inner.expect("build closure produced a view"))
    })
}
