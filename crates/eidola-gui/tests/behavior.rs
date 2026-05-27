//! Behavior tests for the gpui app — uses `gpui::TestAppContext` (mocked
//! rendering, deterministic dispatcher) so the test runs on libtest's worker
//! thread without touching AppKit. These are the regression gate; the visual
//! snapshot harness in `tests/visual.rs` is a local-only debug aid.
//!
//! Pattern:
//! 1. Construct a `Core::stub()` entity with whatever fixture state you need.
//! 2. Open a window with the view under test (via `cx.open_window`).
//! 3. Drive interactions through the view's `focus_handle()` — the same path
//!    keystrokes take in production.
//! 4. Assert against the view/core's public state with `read_with`.

use eidola_app_core::{BalancesResult, ConfigState, CredentialInfo, SpaceMessage};
use eidola_gui::chat::{ChatView, Send};
use eidola_gui::core::Core;
use eidola_gui::wallet::WalletView;
use gpui::{AnyWindowHandle, AppContext, Entity, TestAppContext, WindowOptions};
use gpui_component::{Root, Theme};
use gpui_markdown_editor::EditorState;

// ---------------------------------------------------------------------------
// Core fixture
// ---------------------------------------------------------------------------

#[gpui::test]
fn core_stub_starts_empty(cx: &mut TestAppContext) {
    let core = cx.update(|cx| cx.new(|_| Core::stub()));

    core.read_with(cx, |c, _| {
        assert!(c.config_state.is_none());
        assert!(c.balances.is_none());
        assert!(c.prices.is_empty());
        assert!(c.credentials.is_empty());
        assert!(c.models.is_empty());
        assert!(c.error_message.is_none());
        assert!(!c.busy);
    });
}

#[gpui::test]
fn core_stub_app_core_is_none(cx: &mut TestAppContext) {
    let core = cx.update(|cx| cx.new(|_| Core::stub()));
    core.read_with(cx, |c, _| {
        assert!(
            c.app_core().is_none(),
            "stub core must report no backend so views skip async work"
        );
    });
}

#[gpui::test]
fn core_stub_async_methods_are_noops(cx: &mut TestAppContext) {
    let core = cx.update(|cx| cx.new(|_| Core::stub()));

    core.update(cx, |c, cx| {
        c.fetch_balances(cx);
        c.fetch_prices(cx);
        c.fetch_credentials(cx);
        c.fetch_models(cx);
        c.create_account(cx);
        c.allocate_credits(100, cx);
    });
    cx.run_until_parked();

    // None of those should have toggled busy or stored state, because the
    // backend is missing.
    core.read_with(cx, |c, _| {
        assert!(!c.busy);
        assert!(c.balances.is_none());
        assert!(c.prices.is_empty());
        assert!(c.credentials.is_empty());
    });
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

#[gpui::test]
fn circadian_themes_install(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);

        let theme = Theme::global(cx);
        assert_eq!(theme.light_theme.name.as_ref(), "Circadian Day");
        assert_eq!(theme.dark_theme.name.as_ref(), "Circadian Night");
    });
}

// ---------------------------------------------------------------------------
// Wallet view
// ---------------------------------------------------------------------------

#[gpui::test]
fn wallet_view_constructs_against_stub_core(cx: &mut TestAppContext) {
    let core = cx.update(|cx| {
        cx.new(|_| {
            let mut c = Core::stub();
            c.credentials = vec![CredentialInfo {
                nonce: "abc123".into(),
                credits: 1_000,
                generation: 0,
            }];
            c
        })
    });

    let (_window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| WalletView::new(core.clone(), window, cx))
    });

    // Construction calls `core.fetch_credentials(cx)` which is a no-op on a
    // stub. The view should sit there harmlessly.
    cx.run_until_parked();

    core.read_with(cx, |c, _| {
        assert_eq!(
            c.credentials.len(),
            1,
            "stub credentials must survive view construction"
        );
        assert!(!c.busy);
    });
}

// ---------------------------------------------------------------------------
// Chat view — action dispatch
// ---------------------------------------------------------------------------

#[gpui::test]
fn chat_submit_with_empty_prompt_is_noop(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), window, cx))
    });

    view.read_with(cx, |v, _| {
        assert!(v.messages.is_empty());
        assert!(v.streaming.is_none());
    });

    dispatch_send(&view, window, cx);

    view.read_with(cx, |v, _| {
        assert!(
            v.messages.is_empty(),
            "submit with empty prompt must not append a message"
        );
        assert!(
            v.streaming.is_none(),
            "submit with empty prompt must not start a streaming response"
        );
    });
}

#[gpui::test]
fn chat_submit_with_prompt_appends_user_message(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), window, cx))
    });

    // Populate the prompt editor the same way a user would, then dispatch
    // the Send action through the focus handle — exercising `submit`'s
    // real path up to the `Some(app_core) else { return }` guard. The
    // stub core has no backend, so submit early-returns after the local
    // state mutations, leaving `messages` and `streaming` populated.
    //
    // We write `EditorState` directly rather than driving the IME path
    // because behavior tests don't have a real text-input chain; this is
    // the same shortcut snapshot tests use to set up populated states.
    let prompt_editor = view.read_with(cx, |v, _| v.prompt_editor_for_test());
    cx.update_window(window, |_, _, cx| {
        prompt_editor.update(cx, |editor, cx| {
            editor.state = EditorState::with_markdown("hi there");
            cx.notify();
        });
    })
    .unwrap();

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&Send, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(v.messages.len(), 1, "submit should append the user message");
        assert_eq!(v.messages[0].message.role, "user");
        assert_eq!(v.messages[0].message.content, "hi there");
        assert!(
            v.streaming.is_some(),
            "submit should enter streaming state with an empty StreamingResponse"
        );
        let s = v.streaming.as_ref().unwrap();
        assert!(s.reasoning.is_empty());
        assert!(s.content.is_empty());
        assert!(!s.expanded);
    });
}

#[gpui::test]
fn chat_renders_markdown_messages_without_panicking(cx: &mut TestAppContext) {
    // Markdown bodies (headings, lists, fenced code) flow through
    // `TextView::markdown` rather than a plain `SharedString`. This guards
    // against the markdown plumbing breaking the per-message invariants —
    // each `SpaceMessage` is still exactly one row in the chat, regardless
    // of how many block elements its content parses into.
    let core = stub_core_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), window, cx))
    });

    view.update(cx, |v, _cx| {
        v.set_messages_for_test(vec![
            SpaceMessage {
                role: "user".into(),
                content: "What does this code do?".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: "# Heading\n\n- one\n- two\n\n```rust\nfn main() {}\n```".into(),
            },
        ]);
    });
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.messages.len(),
            2,
            "markdown content must not multiply messages"
        );
        assert_eq!(v.messages[1].message.role, "assistant");
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn stub_core_with_config(cx: &mut TestAppContext) -> Entity<Core> {
    cx.update(|cx| {
        cx.new(|_| {
            let mut c = Core::stub();
            c.config_state = Some(ConfigState {
                base_url: "https://eidola.example/v1".into(),
                has_account: true,
                has_account_secret: true,
                domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
                trusted_measurements: Vec::new(),
                has_hardware_root_ca: false,
                has_hardware_intermediate_ca: false,
                attestation_url: None,
            });
            c.balances = Some(BalancesResult {
                available: 0,
                pools: Vec::new(),
            });
            c
        })
    })
}

/// Open a window whose root is `gpui_component::Root` wrapping the inner
/// view, the same way production does (`lib.rs::open_main_window`). The
/// `Root` wrapper is required by `gpui_component::Input`: a focused input's
/// `on_blur` calls `Root::update`, which panics if the window root isn't a
/// `Root`. ChatView focuses its input on construction, so opening it
/// without `Root` would panic the moment the test process closes the
/// window. Returns both the `AnyWindowHandle` (for action dispatch /
/// window updates) and the inner `Entity<V>` (for state assertions).
fn open_view<V: gpui::Render + 'static>(
    cx: &mut TestAppContext,
    build: impl FnOnce(&mut gpui::Window, &mut gpui::App) -> Entity<V>,
) -> (AnyWindowHandle, Entity<V>) {
    cx.update(|cx| {
        // Idempotent — gpui-component installs its `Theme` and other globals
        // here. View construction reads them via `cx.theme()`, so the init
        // must happen before `cx.open_window`. Circadian goes on top so any
        // colour-bearing assertions match production.
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

fn dispatch_send(view: &Entity<ChatView>, window: AnyWindowHandle, cx: &mut TestAppContext) {
    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&Send, window, cx);
    })
    .unwrap();
    cx.run_until_parked();
}
