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

use eidola_app_core::error::AppError;
use eidola_app_core::updates::{
    Claim, ClaimDelta, ClaimsComparison, UpdateCheckResult, UpdateCheckSnapshot, VerifiedRelease,
};
use eidola_app_core::{
    BalancesResult, ConfigState, CredentialInfo, ModelInfo, PriceInfo, SpaceInfo, SpaceMessage,
};
use eidola_gui::chat::{ChatView, OnboardingStage, Send, ToggleModelPicker};
use eidola_gui::core::Core;
use eidola_gui::library::LibraryView;
use eidola_gui::updates::{UpdatesDisplay, UpdatesView, relative_time};
use eidola_gui::wallet::WalletView;
use gpui::{
    AnyWindowHandle, AppContext, Entity, Modifiers, TestAppContext, VisualTestContext,
    WindowOptions,
};
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
        cx.new(|cx| ChatView::new(core.clone(), None, window, cx))
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
        cx.new(|cx| ChatView::new(core.clone(), None, window, cx))
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
        cx.new(|cx| ChatView::new(core.clone(), None, window, cx))
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

#[gpui::test]
fn chat_view_records_existing_space_id(cx: &mut TestAppContext) {
    // Opening a space from the Library constructs the ChatView with the
    // existing space_id. With a stub core there's no backend to load
    // messages from, so the transcript starts empty (tests preload via
    // `set_messages_for_test`) — but the space binding must be in place so
    // the next submit continues the space instead of creating a new one.
    let core = stub_core_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), Some("space-123".into()), window, cx))
    });
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(v.space_id(), Some("space-123"));
        assert!(v.messages.is_empty());
    });
}

#[gpui::test]
fn stale_initial_space_load_does_not_replace_submitted_prompt(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), Some("space-123".into()), window, cx))
    });

    let prompt_editor = view.read_with(cx, |v, _| v.prompt_editor_for_test());
    cx.update_window(window, |_, _, cx| {
        prompt_editor.update(cx, |editor, cx| {
            editor.state = EditorState::with_markdown("new prompt");
            cx.notify();
        });
    })
    .unwrap();
    dispatch_send(&view, window, cx);

    view.update(cx, |v, _| {
        let applied = v.merge_initial_messages_for_test(
            0,
            vec![SpaceMessage {
                role: "user".into(),
                content: "old prompt".into(),
            }],
        );
        assert!(!applied, "stale initial load should be ignored");
    });

    view.read_with(cx, |v, _| {
        assert_eq!(v.messages.len(), 1);
        assert_eq!(v.messages[0].message.role, "user");
        assert_eq!(v.messages[0].message.content, "new prompt");
        assert!(v.streaming.is_some());
    });
}

#[gpui::test]
fn chat_view_renders_preloaded_messages(cx: &mut TestAppContext) {
    // A reopened space renders its persisted history. The stub core can't
    // drive the async load, so this exercises the same state the loader
    // produces: messages installed after construction.
    let core = stub_core_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), Some("space-123".into()), window, cx))
    });

    view.update(cx, |v, _| {
        v.set_messages_for_test(vec![
            SpaceMessage {
                role: "user".into(),
                content: "Earlier question".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: "Earlier answer".into(),
            },
        ]);
    });
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(v.messages.len(), 2);
        assert_eq!(v.messages[0].message.role, "user");
        assert_eq!(v.messages[1].message.role, "assistant");
        assert_eq!(v.space_id(), Some("space-123"));
    });
}

// ---------------------------------------------------------------------------
// Chat view — model picker
// ---------------------------------------------------------------------------

#[gpui::test]
fn alt_reveals_model_label(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, _| {
        assert!(
            !v.model_revealed(),
            "resting state is invisible — no model chrome until ⌥"
        );
    });

    // Drive the real modifiers-changed dispatch path (platform event →
    // window → focus dispatch chain → the root element's
    // `on_modifiers_changed` listener).
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_modifiers_change(Modifiers {
        alt: true,
        ..Modifiers::default()
    });
    view.read_with(&vcx, |v, _| {
        assert!(v.model_revealed(), "holding ⌥ must reveal the model label");
    });

    vcx.simulate_modifiers_change(Modifiers::default());
    view.read_with(&vcx, |v, _| {
        assert!(
            !v.model_revealed(),
            "releasing ⌥ must return the page to its resting state"
        );
    });
}

#[gpui::test]
fn picker_stays_open_after_alt_release(cx: &mut TestAppContext) {
    // ⌥⌘M opens the picker; releasing ⌥ afterwards must not yank the
    // panel (or its anchor label) away mid-interaction.
    let core = stub_core_with_config(cx);
    let (window, view) = open_chat(cx, &core);

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_modifiers_change(Modifiers::default());
    view.read_with(&vcx, |v, _| {
        assert!(v.model_picker_open());
        assert!(
            v.model_revealed(),
            "the open picker keeps its anchor label revealed"
        );
    });
}

#[gpui::test]
fn toggle_model_picker_action_round_trips(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, _| assert!(!v.model_picker_open()));

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(v.model_picker_open(), "⌥⌘M must open the picker")
    });

    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(!v.model_picker_open(), "⌥⌘M again must close the picker")
    });
}

#[gpui::test]
fn submit_uses_config_default_model_when_nothing_selected(cx: &mut TestAppContext) {
    // New windows start from the user's default: with no per-window
    // selection, a send resolves the model from `ConfigState::default_model`.
    let core = stub_core(cx, |c| {
        let mut state = config_state(true);
        state.default_model = "custom-default".into();
        c.config_state = Some(state);
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    });
    let (window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.current_model(cx), "custom-default");
        assert_eq!(v.selected_model(), None);
    });

    set_composer_text(&view, window, cx, "hello");
    dispatch_send(&view, window, cx);

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.last_submitted_model(),
            Some("custom-default"),
            "an unselected window must send with the config default"
        );
    });
}

#[gpui::test]
fn selecting_a_model_changes_what_submit_sends(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
        c.models = stub_models();
    });
    let (window, view) = open_chat(cx, &core);

    // Selecting from the picker closes it and pins this window's model.
    view.update(cx, |v, cx| {
        v.set_model_picker_open_for_test(true);
        v.select_model("kimi-k2-6".into(), cx);
    });
    view.read_with(cx, |v, cx| {
        assert_eq!(v.selected_model(), Some("kimi-k2-6"));
        assert_eq!(v.current_model(cx), "kimi-k2-6");
        assert!(!v.model_picker_open(), "selection must close the picker");
    });

    set_composer_text(&view, window, cx, "hi");
    dispatch_send(&view, window, cx);

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.last_submitted_model(),
            Some("kimi-k2-6"),
            "submit must use the window's selected model"
        );
    });
}

#[gpui::test]
fn selection_during_streaming_applies_to_next_send(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (window, view) = open_chat(cx, &core);

    // First send (stub core: streaming state sticks).
    set_composer_text(&view, window, cx, "first");
    dispatch_send(&view, window, cx);
    view.read_with(cx, |v, _| {
        assert!(v.streaming.is_some());
        assert_eq!(v.last_submitted_model(), Some("gemma4-31b"));
    });

    // Switching mid-stream must not touch the in-flight request — only
    // the window's selection for the *next* send.
    view.update(cx, |v, cx| v.select_model("kimi-k2-6".into(), cx));
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.last_submitted_model(),
            Some("gemma4-31b"),
            "an in-flight stream is never hot-swapped"
        );
        assert_eq!(v.selected_model(), Some("kimi-k2-6"));
    });
}

fn stub_models() -> Vec<ModelInfo> {
    vec![
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
    ]
}

fn set_composer_text(
    view: &Entity<ChatView>,
    window: AnyWindowHandle,
    cx: &mut TestAppContext,
    text: &str,
) {
    let prompt_editor = view.read_with(cx, |v, _| v.prompt_editor_for_test());
    let text = text.to_string();
    cx.update_window(window, |_, _, cx| {
        prompt_editor.update(cx, |editor, cx| {
            editor.state = EditorState::with_markdown(text.as_str());
            cx.notify();
        });
    })
    .unwrap();
}

// ---------------------------------------------------------------------------
// Library view
// ---------------------------------------------------------------------------

fn stub_space(id: &str, title: Option<&str>, snippet: Option<&str>, ts: i64) -> SpaceInfo {
    SpaceInfo {
        id: id.into(),
        title: title.map(String::from),
        snippet: snippet.map(String::from),
        created_at: ts,
        last_activity_at: ts,
        message_count: 2,
        archived_at: None,
    }
}

#[gpui::test]
fn library_view_renders_stubbed_spaces(cx: &mut TestAppContext) {
    let core = cx.update(|cx| {
        cx.new(|_| {
            let mut c = Core::stub();
            c.spaces = vec![
                stub_space("s1", Some("Tides and the moon"), None, 1_000),
                stub_space("s2", None, Some("what is a monad?"), 2_000),
            ];
            c
        })
    });

    let (_window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(core.clone(), window, cx))
    });
    cx.run_until_parked();

    // Construction calls `core.fetch_spaces(cx)` — a no-op on a stub — so
    // the stubbed listing must survive render.
    core.read_with(cx, |c, _| {
        assert_eq!(c.spaces.len(), 2);
        assert!(!c.busy);
    });
}

#[gpui::test]
fn library_archive_removes_row(cx: &mut TestAppContext) {
    let core = cx.update(|cx| {
        cx.new(|_| {
            let mut c = Core::stub();
            c.spaces = vec![
                stub_space("s1", Some("Keep me"), None, 1_000),
                stub_space("s2", Some("Archive me"), None, 2_000),
            ];
            c
        })
    });

    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(core.clone(), window, cx))
    });

    // The hover-revealed × calls `LibraryView::archive` with the row's
    // space id; drive the same method directly (behavior tests don't
    // synthesize mouse events).
    view.update(cx, |v, cx| v.archive("s2".into(), cx));
    cx.run_until_parked();

    core.read_with(cx, |c, _| {
        assert_eq!(
            c.spaces.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            vec!["s1"],
            "archiving must remove the row from the cached listing"
        );
    });
}

// ---------------------------------------------------------------------------
// Onboarding state machine
// ---------------------------------------------------------------------------

#[gpui::test]
fn chat_without_account_is_welcome_stage(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(false));
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Welcome,
            "no account → the empty page is the welcome page"
        );
    });
}

#[gpui::test]
fn welcome_begin_enters_account_creation(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(false));
    });
    let (_window, view) = open_chat(cx, &core);

    // Click "Begin" (the button's on_click calls this handler). With a
    // stub core the request can't actually start, so the observable state
    // machine transition — `creating_account` — is the assertion target,
    // mirroring how `chat_submit_with_prompt_appends_user_message` tests
    // submit up to the backend guard.
    view.update(cx, |v, cx| v.begin_onboarding(cx));
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert!(
            v.onboarding().creating_account,
            "Begin must enter the in-flight account-creation state"
        );
        assert!(v.onboarding().create_error.is_none());
    });

    // A second click while in flight is a no-op (idempotent guard).
    view.update(cx, |v, cx| v.begin_onboarding(cx));
    view.read_with(cx, |v, _| assert!(v.onboarding().creating_account));
}

#[gpui::test]
fn account_with_zero_balance_is_plans_stage(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Plans,
            "account + known-zero balance + empty wallet → plans page"
        );
    });
}

#[gpui::test]
fn unknown_balance_is_ready_stage(cx: &mut TestAppContext) {
    // Balances not yet fetched (None) must NOT claim the user is unfunded —
    // the page stays the normal blank page until the snapshot is known.
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Ready
        );
    });
}

#[gpui::test]
fn wallet_credentials_bypass_plans_stage(cx: &mut TestAppContext) {
    // Zero account balance but a spendable wallet credential → chat works,
    // so the plans page must not appear.
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
        c.credentials = vec![CredentialInfo {
            nonce: "abc".into(),
            credits: 50_000,
            generation: 0,
        }];
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Ready
        );
    });
}

#[gpui::test]
fn positive_balance_is_ready_stage(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Ready
        );
    });
}

#[gpui::test]
fn composer_text_overrides_plans_stage(cx: &mut TestAppContext) {
    // If the user has started typing, the onboarding pages must not
    // replace the page out from under them.
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), false),
            OnboardingStage::Ready
        );
    });
}

#[gpui::test]
fn plan_click_enters_checkout_pending(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
        c.prices = vec![PriceInfo {
            id: "price_basic".into(),
            product_name: "Basic".into(),
            product_description: None,
            amount_display: "10.00 USD".into(),
            recurrence: "/month".into(),
            credits: 10_000_000,
        }];
    });
    let (_window, view) = open_chat(cx, &core);

    view.update(cx, |v, cx| v.begin_checkout("price_basic".into(), cx));
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.onboarding().checkout_pending.as_deref(),
            Some("price_basic"),
            "clicking a plan must enter the in-flight checkout state"
        );
    });
}

#[gpui::test]
fn dismiss_returns_to_blank_page(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Plans
        );
    });

    view.update(cx, |v, cx| v.dismiss_onboarding(cx));
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Ready,
            "\"I'll do this later\" must fall through to the normal blank page"
        );
        assert!(!v.onboarding().awaiting_checkout);
    });
}

#[gpui::test]
fn insufficient_balance_failure_surfaces_plans_below_transcript(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (_window, view) = open_chat(cx, &core);

    view.update(cx, |v, cx| {
        v.set_messages_for_test(vec![SpaceMessage {
            role: "user".into(),
            content: "hello".into(),
        }]);
        v.apply_chat_failure(
            AppError::InsufficientBalance {
                available: 100,
                required: 6_200,
            },
            cx,
        );
    });
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert!(
            v.show_plans_after_error,
            "InsufficientBalance must surface the plans below the transcript"
        );
        assert!(v.streaming.is_none());
        // Typed routing: the transcript stays (Ready stage), no page swap.
        assert_eq!(
            v.onboarding_stage(core.read(cx), true),
            OnboardingStage::Ready
        );
    });
}

#[gpui::test]
fn non_balance_failure_does_not_surface_plans(cx: &mut TestAppContext) {
    let core = stub_core_with_config(cx);
    let (_window, view) = open_chat(cx, &core);

    view.update(cx, |v, cx| {
        v.apply_chat_failure(
            AppError::Network {
                message: "request failed: connection refused".into(),
            },
            cx,
        );
    });

    view.read_with(cx, |v, _| {
        assert!(!v.show_plans_after_error);
    });
}

// ---------------------------------------------------------------------------
// Updates window — display-state derivation for every matrix row
// ---------------------------------------------------------------------------

fn verified_release(claims_accepted: bool) -> VerifiedRelease {
    VerifiedRelease {
        version: "0.2.0".into(),
        tag: "v0.2.0".into(),
        release_url: Some("https://github.com/eidola-ai/eidola/releases/tag/v0.2.0".into()),
        published_at: Some("2026-06-01T12:00:00Z".into()),
        ci_identity:
            "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v0.2.0"
                .into(),
        rekor_log_index: 123_456_789,
        manifest_sha256: "ab".repeat(32),
        claims_accepted,
    }
}

fn claims_comparison() -> ClaimsComparison {
    ClaimsComparison {
        expected: vec![
            Claim {
                key: "manifest.schema_version".into(),
                value: "1".into(),
            },
            Claim {
                key: "enclave.snp_measurement".into(),
                value: "SEV-SNP launch measurement (48-byte hex)".into(),
            },
        ],
        attested: vec![Claim {
            key: "manifest.schema_version".into(),
            value: "2".into(),
        }],
        deltas: vec![
            ClaimDelta {
                key: "manifest.schema_version".into(),
                expected: Some("1".into()),
                attested: Some("2".into()),
            },
            ClaimDelta {
                key: "enclave.snp_measurement".into(),
                expected: Some("SEV-SNP launch measurement (48-byte hex)".into()),
                attested: None,
            },
        ],
    }
}

fn snapshot(result: UpdateCheckResult) -> UpdateCheckSnapshot {
    UpdateCheckSnapshot {
        checked_at_ms: eidola_app_core::now_ms() - 5 * 60 * 1000,
        result,
    }
}

fn open_updates(
    cx: &mut TestAppContext,
    core: &Entity<Core>,
) -> (AnyWindowHandle, Entity<UpdatesView>) {
    let core = core.clone();
    open_view(cx, |window, cx| {
        cx.new(|cx| UpdatesView::new(core.clone(), window, cx))
    })
}

#[gpui::test]
fn updates_view_none_yet_on_fresh_stub(cx: &mut TestAppContext) {
    // Stub core: the constructor's load/check calls are no-ops, so the
    // view sits honestly on "no check has completed yet".
    let core = stub_core(cx, |_| {});
    let (_window, view) = open_updates(cx, &core);
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert_eq!(v.display(cx), UpdatesDisplay::NoneYet);
    });
    core.read_with(cx, |c, _| {
        assert!(!c.update_checking, "stub check must not set in-flight");
    });
}

#[gpui::test]
fn updates_view_checking_state(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| c.update_checking = true);
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.display(cx), UpdatesDisplay::Checking);
    });
}

#[gpui::test]
fn updates_view_up_to_date_state(cx: &mut TestAppContext) {
    // Matrix row: no newer `latest` release. Also covers "background-check
    // result is reflected when the window opens": the snapshot is in the
    // core *before* the view is constructed.
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::UpToDate {
            latest_version: Some("0.1.0".into()),
        }));
    });
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        let UpdatesDisplay::UpToDate {
            latest_version,
            checked_at_ms,
        } = v.display(cx)
        else {
            panic!("expected UpToDate display");
        };
        assert_eq!(latest_version.as_deref(), Some("0.1.0"));
        assert!(checked_at_ms > 0);
    });
}

#[gpui::test]
fn updates_view_update_available_state(cx: &mut TestAppContext) {
    // Matrix row: verified update — one action, open the release page.
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::UpdateAvailable {
            release: verified_release(false),
        }));
    });
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        let UpdatesDisplay::UpdateAvailable { release } = v.display(cx) else {
            panic!("expected UpdateAvailable display");
        };
        assert_eq!(release.version, "0.2.0");
        assert!(!release.claims_accepted);
    });
}

#[gpui::test]
fn updates_view_unverifiable_state(cx: &mut TestAppContext) {
    // Matrix row: hard visible security state — the display carries the
    // exact failure reason and no release link.
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::Unverifiable {
            version: "0.2.0".into(),
            tag: "v0.2.0".into(),
            reason: "signature is not from the pinned release identity".into(),
        }));
    });
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        let UpdatesDisplay::Unverifiable {
            version, reason, ..
        } = v.display(cx)
        else {
            panic!("expected Unverifiable display");
        };
        assert_eq!(version, "0.2.0");
        assert!(reason.contains("pinned release identity"));
    });
}

#[gpui::test]
fn updates_view_claims_changed_state(cx: &mut TestAppContext) {
    // Matrix row: authentic but claims changed — side-by-side material is
    // present and the release is NOT framed as an update.
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::ClaimsChanged {
            release: verified_release(false),
            comparison: claims_comparison(),
        }));
    });
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        let UpdatesDisplay::ClaimsChanged {
            release,
            comparison,
        } = v.display(cx)
        else {
            panic!("expected ClaimsChanged display");
        };
        assert!(!release.claims_accepted);
        assert_eq!(comparison.deltas.len(), 2);
        assert_eq!(comparison.expected.len(), 2);
    });
}

#[gpui::test]
fn updates_view_check_failed_state(cx: &mut TestAppContext) {
    // Matrix row: network failure — quiet, carries the message + time.
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::CheckFailed {
            message: "GET …: connection refused".into(),
        }));
    });
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        let UpdatesDisplay::CheckFailed { message, .. } = v.display(cx) else {
            panic!("expected CheckFailed display");
        };
        assert!(message.contains("connection refused"));
    });
}

#[gpui::test]
fn updates_view_rechecking_keeps_standing_result(cx: &mut TestAppContext) {
    // While a re-check runs, the standing result stays up (the footer
    // shows the in-flight hint) — Checking only masks an empty page.
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::UpToDate {
            latest_version: None,
        }));
        c.update_checking = true;
    });
    let (_window, view) = open_updates(cx, &core);

    view.read_with(cx, |v, cx| {
        assert!(
            matches!(v.display(cx), UpdatesDisplay::UpToDate { .. }),
            "standing result must not be masked by a re-check"
        );
    });
}

#[gpui::test]
fn updates_view_actions_are_noops_on_stub(cx: &mut TestAppContext) {
    let core = stub_core(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::ClaimsChanged {
            release: verified_release(false),
            comparison: claims_comparison(),
        }));
    });
    let (_window, view) = open_updates(cx, &core);

    view.update(cx, |v, cx| {
        v.check_now(cx);
        v.accept_claims(cx);
    });
    cx.run_until_parked();

    // No backend: neither flag flips, the standing state is untouched.
    core.read_with(cx, |c, _| {
        assert!(!c.update_checking);
        assert!(matches!(
            c.update_check.as_ref().map(|s| &s.result),
            Some(UpdateCheckResult::ClaimsChanged { .. })
        ));
    });
}

#[gpui::test]
fn relative_time_buckets(cx: &mut TestAppContext) {
    let _ = cx;
    let now = 1_000_000_000_000;
    assert_eq!(relative_time(now - 10_000, now), "just now");
    assert_eq!(relative_time(now - 5 * 60_000, now), "5m ago");
    assert_eq!(relative_time(now - 3 * 3_600_000, now), "3h ago");
    assert_eq!(relative_time(now - 49 * 3_600_000, now), "2d ago");
    // Clock skew (then > now) clamps to "just now", never negative.
    assert_eq!(relative_time(now + 60_000, now), "just now");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn config_state(has_account: bool) -> ConfigState {
    ConfigState {
        base_url: "https://eidola.example/v1".into(),
        default_model: "gemma4-31b".into(),
        has_account,
        has_account_secret: has_account,
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        trusted_measurements: Vec::new(),
        has_hardware_root_ca: false,
        has_hardware_intermediate_ca: false,
        attestation_url: None,
    }
}

fn stub_core(cx: &mut TestAppContext, setup: impl FnOnce(&mut Core)) -> Entity<Core> {
    cx.update(|cx| {
        cx.new(|_| {
            let mut c = Core::stub();
            setup(&mut c);
            c
        })
    })
}

/// A stub core representing a funded, ready account — the fixture the
/// plain chat tests use.
fn stub_core_with_config(cx: &mut TestAppContext) -> Entity<Core> {
    stub_core(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    })
}

fn open_chat(cx: &mut TestAppContext, core: &Entity<Core>) -> (AnyWindowHandle, Entity<ChatView>) {
    let core = core.clone();
    open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(core.clone(), None, window, cx))
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
