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
    AttestationDetail, AttestationInfo, BalancesResult, ConfigState, CredentialInfo,
    CredentialLifecycleInfo, ModelInfo, PriceInfo, RequestDetail, RequestInfo, SpaceInfo,
    SpaceMessage,
};
use eidola_gui::about::AboutView;
use eidola_gui::account::AccountView;
use eidola_gui::chat::{
    ChatView, DismissModelPicker, OnboardingStage, ParticipantIndicator, PickerConfirm, PickerDown,
    PickerUp, Send, ToggleModelPicker,
};
use eidola_gui::library::LibraryView;
use eidola_gui::record::{RecordDetail, RecordSection, RecordView};
use eidola_gui::settings::{SettingsPane, SettingsView};
use eidola_gui::stores::{Stores, StoresStub};
use eidola_gui::updates::{UpdatesDisplay, UpdatesView, relative_time};
use eidola_gui::wallet::WalletView;
use eidola_gui::window_input::WindowInput;
use gpui::{
    AnyWindowHandle, AppContext, Entity, Modifiers, TestAppContext, VisualTestContext,
    WindowOptions,
};
use gpui_component::{Root, Theme};
use gpui_markdown_editor::EditorState;

// ---------------------------------------------------------------------------
// Stores fixture
// ---------------------------------------------------------------------------

#[gpui::test]
fn stub_stores_start_empty(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});

    stores
        .config
        .read_with(cx, |c, _| assert!(c.state().is_none()));
    stores.account.read_with(cx, |a, _| {
        assert!(a.balances().value().is_none());
        assert!(a.prices().value().is_none());
    });
    stores
        .wallet
        .read_with(cx, |w, _| assert!(w.credentials().is_empty()));
    stores
        .models
        .read_with(cx, |m, _| assert!(m.list().is_empty()));
}

#[gpui::test]
fn stub_stores_have_no_backend(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    assert!(
        stores.app_core().is_none(),
        "stub stores must report no backend so views skip async work"
    );
}

#[gpui::test]
fn stub_store_refreshes_are_noops(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});

    stores.account.update(cx, |a, cx| {
        a.refresh_balances(cx);
        a.refresh_prices(cx);
    });
    stores.models.update(cx, |m, cx| m.refresh(cx));
    stores.wallet.update(cx, |w, cx| w.refresh(cx));
    cx.run_until_parked();

    // No backend: every cell stays NotLoaded (a refresh with no `app_core`
    // returns before touching the cell — no spurious Loading spinner).
    stores.account.read_with(cx, |a, _| {
        assert!(a.balances().value().is_none());
        assert!(!a.balances().is_loading());
        assert!(a.prices().value().is_none());
    });
    stores
        .models
        .read_with(cx, |m, _| assert!(m.list().is_empty()));
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
fn wallet_view_constructs_against_stub_stores(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.credential_lifecycle = vec![CredentialLifecycleInfo {
            nonce: "abc123".into(),
            credits: 1_000,
            generation: 0,
            created_at: 1_000,
            state: "active".into(),
            spend_amount: None,
        }];
    });

    let (_window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| WalletView::new(stores.clone(), window, cx))
    });

    // Construction calls `WalletStore::refresh`, a no-op on a stub. The view
    // should sit there harmlessly with the fixture listing intact.
    cx.run_until_parked();

    stores.wallet.read_with(cx, |w, _| {
        assert_eq!(
            w.lifecycle_rows().len(),
            1,
            "stub credential listing must survive view construction"
        );
        assert!(!w.is_loading());
    });
}

// ---------------------------------------------------------------------------
// Chat view — action dispatch
// ---------------------------------------------------------------------------

#[gpui::test]
fn chat_submit_with_empty_prompt_is_noop(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores.clone(), None, WindowInput::new(cx), window, cx))
    });

    view.read_with(cx, |v, cx| {
        assert!(v.messages(cx).is_empty());
        assert!(v.streaming(cx).is_none());
    });

    dispatch_send(&view, window, cx);

    view.read_with(cx, |v, cx| {
        assert!(
            v.messages(cx).is_empty(),
            "submit with empty prompt must not append a message"
        );
        assert!(
            v.streaming(cx).is_none(),
            "submit with empty prompt must not start a streaming response"
        );
    });
}

#[gpui::test]
fn chat_submit_with_prompt_appends_user_message(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores.clone(), None, WindowInput::new(cx), window, cx))
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

    view.read_with(cx, |v, cx| {
        let messages = v.messages(cx);
        assert_eq!(messages.len(), 1, "submit should append the user message");
        assert_eq!(messages[0].message.role, "user");
        assert_eq!(messages[0].message.content, "hi there");
        let streaming = v.streaming(cx);
        assert!(
            streaming.is_some(),
            "submit should enter streaming state with an empty StreamingResponse"
        );
        let s = streaming.as_ref().unwrap();
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
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores.clone(), None, WindowInput::new(cx), window, cx))
    });

    view.update(cx, |v, cx| {
        v.set_messages_for_test(
            vec![
                SpaceMessage {
                    role: "user".into(),
                    content: "What does this code do?".into(),
                },
                SpaceMessage {
                    role: "assistant".into(),
                    content: "# Heading\n\n- one\n- two\n\n```rust\nfn main() {}\n```".into(),
                },
            ],
            cx,
        );
    });
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        let messages = v.messages(cx);
        assert_eq!(
            messages.len(),
            2,
            "markdown content must not multiply messages"
        );
        assert_eq!(messages[1].message.role, "assistant");
    });
}

#[gpui::test]
fn chat_view_records_existing_space_id(cx: &mut TestAppContext) {
    // Opening a space from the Library constructs the ChatView with the
    // existing space_id. With a stub core there's no backend to load
    // messages from, so the transcript starts empty (tests preload via
    // `set_messages_for_test`) — but the space binding must be in place so
    // the next submit continues the space instead of creating a new one.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ChatView::new(
                stores.clone(),
                Some("space-123".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert_eq!(v.space_id(cx).as_deref(), Some("space-123"));
        assert!(v.messages(cx).is_empty());
    });
}

#[gpui::test]
fn stale_initial_space_load_does_not_replace_submitted_prompt(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ChatView::new(
                stores.clone(),
                Some("space-123".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
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

    // Simulate the reopened-space initial load completing *after* the local
    // submit. The race is serialized inside the `Space` entity (which owns
    // both the load and the submit): a stale load that finishes once
    // streaming has begun is dropped, so it cannot clobber the just-submitted
    // prompt. This replaces the old `transcript_generation` counter.
    let space = view.read_with(cx, |v, _| v.space().clone());
    space.update(cx, |s, cx| {
        let applied = s.apply_loaded_transcript_for_test(
            vec![SpaceMessage {
                role: "user".into(),
                content: "old prompt".into(),
            }],
            cx,
        );
        assert!(
            !applied,
            "a stale initial load racing a submit must be dropped"
        );
    });

    view.read_with(cx, |v, cx| {
        let messages = v.messages(cx);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message.role, "user");
        assert_eq!(messages[0].message.content, "new prompt");
        assert!(v.streaming(cx).is_some());
    });
}

#[gpui::test]
fn chat_view_renders_preloaded_messages(cx: &mut TestAppContext) {
    // A reopened space renders its persisted history. The stub core can't
    // drive the async load, so this exercises the same state the loader
    // produces: messages installed after construction.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ChatView::new(
                stores.clone(),
                Some("space-123".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });

    view.update(cx, |v, cx| {
        v.set_messages_for_test(
            vec![
                SpaceMessage {
                    role: "user".into(),
                    content: "Earlier question".into(),
                },
                SpaceMessage {
                    role: "assistant".into(),
                    content: "Earlier answer".into(),
                },
            ],
            cx,
        );
    });
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        let messages = v.messages(cx);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message.role, "user");
        assert_eq!(messages[1].message.role, "assistant");
        assert_eq!(v.space_id(cx).as_deref(), Some("space-123"));
    });
}

#[gpui::test]
fn two_windows_on_one_space_share_state(cx: &mut TestAppContext) {
    // Wave-2 bug 4: two windows opened on the same space hold the *same*
    // `Space` entity (via the `SpacesStore` registry), so a submit + stream
    // driven through one window appears in the other live — structurally, not
    // by any cross-window plumbing. Both `ChatView`s are lenses over one
    // shared transcript + streaming buffer.
    let stores = stub_stores_with_config(cx);

    let (window_a, view_a) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ChatView::new(
                stores.clone(),
                Some("space-shared".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    let (_window_b, view_b) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ChatView::new(
                stores.clone(),
                Some("space-shared".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    cx.run_until_parked();

    // The registry joined both opens onto one entity.
    let space_a = view_a.read_with(cx, |v, _| v.space().clone());
    let space_b = view_b.read_with(cx, |v, _| v.space().clone());
    assert_eq!(
        space_a.entity_id(),
        space_b.entity_id(),
        "two windows on one space must share one Space entity"
    );

    // Submit through window A.
    set_composer_text(&view_a, window_a, cx, "shared question");
    dispatch_send(&view_a, window_a, cx);

    // Window B sees the appended user turn and the streaming state, because it
    // renders from the same entity.
    let agree = |cx: &mut TestAppContext| {
        let a = view_a.read_with(cx, |v, cx| {
            (
                v.messages(cx)
                    .iter()
                    .map(|m| (m.message.role.clone(), m.message.content.clone()))
                    .collect::<Vec<_>>(),
                v.streaming(cx)
                    .map(|s| (s.reasoning.clone(), s.content.clone())),
            )
        });
        let b = view_b.read_with(cx, |v, cx| {
            (
                v.messages(cx)
                    .iter()
                    .map(|m| (m.message.role.clone(), m.message.content.clone()))
                    .collect::<Vec<_>>(),
                v.streaming(cx)
                    .map(|s| (s.reasoning.clone(), s.content.clone())),
            )
        });
        assert_eq!(a, b, "both windows must agree on transcript + streaming");
        a
    };

    let after_submit = agree(cx);
    assert_eq!(
        after_submit.0,
        vec![("user".to_string(), "shared question".to_string())],
    );
    assert!(after_submit.1.is_some(), "both windows are streaming");

    // Drive a stream delta on the shared space — both lenses observe it.
    space_a.update(cx, |s, cx| {
        s.push_content_delta_for_test("partial answer", cx)
    });
    cx.run_until_parked();

    let after_delta = agree(cx);
    assert_eq!(
        after_delta.1.unwrap().1,
        "partial answer",
        "the streamed content appears in both windows"
    );
}

#[gpui::test]
fn transcript_render_work_is_constant_in_message_count(cx: &mut TestAppContext) {
    // Part-1 perf guard (the wave-3 pattern extended to the variable-height
    // transcript): with `list()`, the per-frame work — what the render closure
    // does, render exactly the visible window — must be O(visible), not
    // O(messages). Load a short transcript, then a 500-message one, and assert
    // the visible-window render produces the same fixed row count both times
    // (and far below the total). Also a coarse timing comparison: 500 messages
    // must not cost meaningfully more per frame than 4.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_chat(cx, &stores);

    // A fixed visible window of message rows (what a ~560px-tall viewport
    // shows at the prose line-height); the list only ever renders this many
    // items per frame. We measure message rows specifically — the composer is
    // always exactly one trailing item, O(1), and its editor entity is not
    // what scales — so a message-only window isolates the O(visible) claim.
    let visible = 0..12usize;

    let synth = |n: usize| -> Vec<SpaceMessage> {
        (0..n)
            .map(|i| SpaceMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
                content: format!(
                    "Synthetic turn {i} — a sentence or two of body text so each row has a \
                     realistic measured height under the prose typography."
                ),
            })
            .collect()
    };

    // Set the transcript and let the `observe(&space)` reconcile the item
    // model (it fires on park), then render the visible window `iters` times.
    // Returns (rows-per-frame, total-item-count, elapsed).
    let measure = |cx: &mut TestAppContext, msgs: Vec<SpaceMessage>| {
        view.update(cx, |v, cx| v.set_messages_for_test(msgs, cx));
        cx.run_until_parked();
        cx.update_window(window, |_, win, cx| {
            view.update(cx, |v, cx| {
                let total = v.transcript_item_count_for_test();
                let start = std::time::Instant::now();
                let mut n = 0;
                for _ in 0..200 {
                    n = v.render_transcript_window_for_test(visible.clone(), win, cx);
                }
                (n, total, start.elapsed())
            })
        })
        .unwrap()
    };

    // Short transcript (20 messages → 21 items incl. composer).
    let (short_window, short_total, short_dur) = measure(cx, synth(20));
    // Long transcript (500 messages → 501 items incl. composer).
    let (long_window, long_total, long_dur) = measure(cx, synth(500));

    // The item model grew 25× …
    assert_eq!(short_total, 21, "20 messages + composer");
    assert_eq!(long_total, 501, "500 messages + composer");
    // … but the per-frame render rendered the same fixed 12-item window in
    // both cases — 12 ≪ 501, and independent of the 500 messages behind it.
    // That cap is the O(visible) guarantee: `list()` only asks the closure for
    // the visible range, never the whole transcript.
    assert_eq!(short_window, 12);
    assert_eq!(
        long_window, 12,
        "per-frame row count is capped at the visible window, not the message count"
    );

    // Coarse timing: per-frame visible-window cost must not scale with the
    // message count. Generous slack absorbs scheduler noise — we're catching
    // an O(messages) regression (which would be ~25×), not microbenching.
    eprintln!("transcript per-frame (200 renders): 20 msgs {short_dur:?} vs 500 msgs {long_dur:?}");
    assert!(
        long_dur.as_secs_f64() < short_dur.as_secs_f64() * 5.0 + 0.1,
        "frame work scaled with message count: 20 msgs {short_dur:?} vs 500 msgs {long_dur:?}"
    );
}

#[gpui::test]
fn participant_indicator_visibility_derivation(cx: &mut TestAppContext) {
    // Part-2: the persistent participant indicator is a pure function of the
    // `list()` scroll position + the transcript item model. It is hidden at
    // the page top and while a chapter delim is on screen, and surfaces the
    // turn's voice once the delim has scrolled off. Drive the derivation
    // directly from synthetic scroll positions (no real layout needed).
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_chat(cx, &stores);

    view.update(cx, |v, cx| {
        v.set_messages_for_test(
            vec![
                SpaceMessage {
                    role: "user".into(),
                    content: "User opening question.".into(),
                },
                SpaceMessage {
                    role: "assistant".into(),
                    content: "Assistant answer body.".into(),
                },
            ],
            cx,
        );
    });
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        // Item 0 is the user's opening turn (no leading delim); item 1 the
        // assistant turn (leading "Eidola" delim); item 2 the composer.
        assert_eq!(v.transcript_item_count_for_test(), 3);

        // At the very page top (item 0, offset 0): the first line is visible,
        // no cue needed → hidden.
        assert_eq!(v.participant_indicator_at_for_test(0, 0.0, cx), None);

        // Scrolled into the delim-less first message: its opening line has
        // gone up, so the page-local cue is missing → show "You".
        assert_eq!(
            v.participant_indicator_at_for_test(0, 60.0, cx),
            Some(ParticipantIndicator {
                label: "You",
                is_error: false,
            })
        );

        // Top of item 1, within its leading-delim band: the "Eidola" delim is
        // on screen → hidden (no competition with the real delim).
        assert_eq!(v.participant_indicator_at_for_test(1, 8.0, cx), None);

        // Scrolled past item 1's delim band, into the assistant body: the
        // delim is gone → surface "Eidola".
        assert_eq!(
            v.participant_indicator_at_for_test(1, 200.0, cx),
            Some(ParticipantIndicator {
                label: "Eidola",
                is_error: false,
            })
        );

        // Over the composer item (the writing zone): no voice to surface.
        assert_eq!(v.participant_indicator_at_for_test(2, 200.0, cx), None);
    });
}

#[gpui::test]
fn participant_indicator_error_keeps_danger_voice(cx: &mut TestAppContext) {
    // An error turn's indicator keeps the danger color (is_error = true), so
    // the persistent cue signals the error role the same way the chapter
    // delim's "Error" label does.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_chat(cx, &stores);

    view.update(cx, |v, cx| {
        v.set_messages_for_test(
            vec![
                SpaceMessage {
                    role: "user".into(),
                    content: "Trigger.".into(),
                },
                SpaceMessage {
                    role: "error".into(),
                    content: "Something went wrong.".into(),
                },
            ],
            cx,
        );
    });
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        // Scrolled into the error turn past its delim band.
        assert_eq!(
            v.participant_indicator_at_for_test(1, 200.0, cx),
            Some(ParticipantIndicator {
                label: "Error",
                is_error: true,
            })
        );
    });
}

#[gpui::test]
fn blank_space_adopts_id_on_wrapped_failure(cx: &mut TestAppContext) {
    use eidola_gui::space::Space;

    // A blank ⌘N space (id=None) whose FIRST exchange FAILS after the space was
    // persisted must still learn its id — app-core wraps the post-persist error
    // as `ChatFailed { space_id }`. The registry adopts the now-id'd entity on
    // `Failed` exactly as it does on `StreamEnded`, so a later open of that id
    // shares the SAME entity (no fork).
    let stores = stub_stores_with_config(cx);

    // Mint a blank space through the registry (this installs the adoption
    // subscription on the SpacesStore).
    let blank: Entity<Space> = stores.spaces.update(cx, |store, cx| store.blank(cx));
    cx.run_until_parked();
    blank.read_with(cx, |s, _| assert!(s.id().is_none(), "blank starts id-less"));

    // Drive the wrapped-failure path: the same logic as `spawn_stream`'s error
    // arm (adopt id from wrapper, emit `Failed` with the unwrapped source).
    let wrapped = AppError::ChatFailed {
        space_id: "space-adopted".into(),
        source: Box::new(AppError::Server {
            status: 500,
            message: "upstream blew up".into(),
        }),
    };
    blank.update(cx, |s, cx| s.apply_chat_failure_for_test(wrapped, cx));
    cx.run_until_parked();

    // The entity learned its id from the wrapper…
    blank.read_with(cx, |s, _| {
        assert_eq!(s.id(), Some("space-adopted"), "id adopted on failure");
    });

    // …and the registry adopted it: opening that id returns the SAME entity.
    let reopened = stores
        .spaces
        .update(cx, |store, cx| store.open("space-adopted".into(), cx));
    assert_eq!(
        reopened.entity_id(),
        blank.entity_id(),
        "registry must adopt the blank on Failed — open(id) returns the same entity, no fork"
    );
}

// ---------------------------------------------------------------------------
// Account view — lifecycle failure surfacing
// ---------------------------------------------------------------------------

#[gpui::test]
fn account_op_error_surfaces_and_clears(cx: &mut TestAppContext) {
    // `AccountStore::create_account` must store its `Err` (honest-states rule:
    // the Settings button can't silently do nothing). The banner renders from
    // the stored error; the next attempt clears it.
    let stores = stub_stores(cx, |s| {
        // No account yet — the Account pane shows the "Create account" button.
        s.config_state = Some(config_state(false));
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores.clone(), window, cx))
    });
    cx.run_until_parked();

    // No error at rest.
    stores.account.read_with(cx, |s, _| {
        assert!(s.account_op_error().is_none(), "no error at rest");
    });

    // Stub a failing op by setting the field directly (no failing backend in
    // the stub harness).
    stores.account.update(cx, |s, cx| {
        s.set_account_op_error_for_test(
            Some(AppError::Network {
                message: "create failed".into(),
            }),
            cx,
        );
    });
    cx.run_until_parked();
    stores.account.read_with(cx, |s, _| {
        assert_eq!(
            s.account_op_error().map(|e| e.to_string()),
            Some("network error: create failed".to_string()),
            "the failure is stored, not dropped",
        );
    });
    // The view renders without panicking with the error present (the banner).
    view.update(cx, |_, cx| cx.notify());
    cx.run_until_parked();

    // A retry clears the error at the start of the attempt. On a stub there is
    // no backend, so `create_account` clears the field and early-returns.
    stores.account.update(cx, |s, cx| s.create_account(cx));
    cx.run_until_parked();
    stores.account.read_with(cx, |s, _| {
        assert!(
            s.account_op_error().is_none(),
            "the next attempt clears the prior error",
        );
    });
}

// ---------------------------------------------------------------------------
// Chat view — model picker
// ---------------------------------------------------------------------------

#[gpui::test]
fn alt_reveals_model_label(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert!(
            !v.model_revealed(cx),
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
    view.read_with(&vcx, |v, cx| {
        assert!(
            v.model_revealed(cx),
            "holding ⌥ must reveal the model label"
        );
    });

    vcx.simulate_modifiers_change(Modifiers::default());
    view.read_with(&vcx, |v, cx| {
        assert!(
            !v.model_revealed(cx),
            "releasing ⌥ must return the page to its resting state"
        );
    });
}

#[gpui::test]
fn picker_stays_open_after_alt_release(cx: &mut TestAppContext) {
    // ⌥⌘M opens the picker; releasing ⌥ afterwards must not yank the
    // panel (or its anchor label) away mid-interaction.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_chat(cx, &stores);

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_modifiers_change(Modifiers::default());
    view.read_with(&vcx, |v, cx| {
        assert!(v.model_picker_open());
        assert!(
            v.model_revealed(cx),
            "the open picker keeps its anchor label revealed"
        );
    });
}

#[gpui::test]
fn toggle_model_picker_action_round_trips(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_chat(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        let mut state = config_state(true);
        state.default_model = "custom-default".into();
        c.config_state = Some(state);
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    });
    let (window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.current_model(cx), "custom-default");
        assert_eq!(v.selected_model(cx), None);
    });

    set_composer_text(&view, window, cx, "hello");
    dispatch_send(&view, window, cx);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.last_submitted_model(cx).as_deref(),
            Some("custom-default"),
            "an unselected window must send with the config default"
        );
    });
}

#[gpui::test]
fn selecting_a_model_changes_what_submit_sends(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
        c.models = stub_models();
    });
    let (window, view) = open_chat(cx, &stores);

    // Selecting from the picker closes it and pins this window's model.
    view.update(cx, |v, cx| {
        v.set_model_picker_open_for_test(true);
        v.select_model("kimi-k2-6".into(), cx);
    });
    view.read_with(cx, |v, cx| {
        assert_eq!(v.selected_model(cx).as_deref(), Some("kimi-k2-6"));
        assert_eq!(v.current_model(cx), "kimi-k2-6");
        assert!(!v.model_picker_open(), "selection must close the picker");
    });

    set_composer_text(&view, window, cx, "hi");
    dispatch_send(&view, window, cx);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.last_submitted_model(cx).as_deref(),
            Some("kimi-k2-6"),
            "submit must use the window's selected model"
        );
    });
}

#[gpui::test]
fn selection_during_streaming_applies_to_next_send(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_chat(cx, &stores);

    // First send (stub core: streaming state sticks).
    set_composer_text(&view, window, cx, "first");
    dispatch_send(&view, window, cx);
    view.read_with(cx, |v, cx| {
        assert!(v.streaming(cx).is_some());
        assert_eq!(v.last_submitted_model(cx).as_deref(), Some("gemma4-31b"));
    });

    // Switching mid-stream must not touch the in-flight request — only
    // the space's selection for the *next* send.
    view.update(cx, |v, cx| v.select_model("kimi-k2-6".into(), cx));
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.last_submitted_model(cx).as_deref(),
            Some("gemma4-31b"),
            "an in-flight stream is never hot-swapped"
        );
        assert_eq!(v.selected_model(cx).as_deref(), Some("kimi-k2-6"));
    });
}

#[gpui::test]
fn escape_dismisses_picker(cx: &mut TestAppContext) {
    // Esc must close the picker when it's open, but is a no-op when closed.
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.models = stub_models();
    });
    let (window, view) = open_chat(cx, &stores);
    let focus = view.read_with(cx, |v, _| v.focus_handle());

    // Open the picker first.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| assert!(v.model_picker_open()));

    // Escape dismisses it.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&DismissModelPicker, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(!v.model_picker_open(), "Esc must close the picker");
        assert_eq!(v.picker_highlighted(), None, "highlight clears on dismiss");
    });
}

#[gpui::test]
fn picker_up_down_moves_highlight(cx: &mut TestAppContext) {
    // Down increments the highlight; Up decrements it; both clamp at the ends.
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.models = stub_models(); // 2 models: index 0 and 1
    });
    let (window, view) = open_chat(cx, &stores);
    let focus = view.read_with(cx, |v, _| v.focus_handle());

    // Open the picker.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();

    // Down from None → 0.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerDown, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.picker_highlighted(),
            Some(0),
            "first Down highlights row 0"
        );
    });

    // Down again → 1 (last row).
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerDown, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.picker_highlighted(), Some(1));
    });

    // Down past end → stays at 1.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerDown, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.picker_highlighted(), Some(1), "Down at end clamps");
    });

    // Up → 0.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerUp, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.picker_highlighted(), Some(0));
    });

    // Up past start → stays at 0.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerUp, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.picker_highlighted(), Some(0), "Up at start clamps");
    });
}

#[gpui::test]
fn picker_enter_selects_highlighted_model(cx: &mut TestAppContext) {
    // Enter selects the highlighted row and closes the picker.
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.models = stub_models(); // index 0 = "gemma4-31b", index 1 = "kimi-k2-6"
    });
    let (window, view) = open_chat(cx, &stores);
    let focus = view.read_with(cx, |v, _| v.focus_handle());

    // Open picker and navigate to row 1.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
        focus.dispatch_action(&PickerDown, window, cx);
        focus.dispatch_action(&PickerDown, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.picker_highlighted(), Some(1));
    });

    // Enter selects the highlighted model.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerConfirm, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(!v.model_picker_open(), "Enter must close the picker");
        assert_eq!(
            v.selected_model(cx).as_deref(),
            Some("kimi-k2-6"),
            "Enter selects the highlighted model"
        );
        assert_eq!(
            v.picker_highlighted(),
            None,
            "highlight clears after confirm"
        );
    });
}

#[gpui::test]
fn picker_enter_with_no_highlight_is_noop(cx: &mut TestAppContext) {
    // Enter with nothing highlighted must not close the picker or change
    // the selection — there is nothing to confirm.
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.models = stub_models();
    });
    let (window, view) = open_chat(cx, &stores);
    let focus = view.read_with(cx, |v, _| v.focus_handle());

    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| assert_eq!(v.picker_highlighted(), None));

    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&PickerConfirm, window, cx);
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(v.model_picker_open(), "picker stays open with no highlight");
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
// About view
// ---------------------------------------------------------------------------

#[gpui::test]
fn about_view_constructs_without_panic(cx: &mut TestAppContext) {
    // The About view has no stores and no async work — constructing and
    // rendering it must not panic.
    let (_window, view) = open_view(cx, |window, cx| cx.new(|cx| AboutView::new(window, cx)));

    view.read_with(cx, |v, _| {
        // Just assert the focus handle is valid (construction succeeded).
        let _ = v.focus_handle();
    });
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
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            stub_space("s1", Some("Tides and the moon"), None, 1_000),
            stub_space("s2", None, Some("what is a monad?"), 2_000),
        ];
    });

    let (_window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });
    cx.run_until_parked();

    // Construction calls `SpacesStore::refresh` — a no-op on a stub — so the
    // stubbed listing must survive render.
    stores.spaces.read_with(cx, |s, _| {
        assert_eq!(s.list().len(), 2);
    });
}

#[gpui::test]
fn library_archive_removes_row(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            stub_space("s1", Some("Keep me"), None, 1_000),
            stub_space("s2", Some("Archive me"), None, 2_000),
        ];
    });

    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });

    // The hover-revealed × calls `LibraryView::archive` with the row's
    // space id; drive the same method directly (behavior tests don't
    // synthesize mouse events).
    view.update(cx, |v, cx| v.archive("s2".into(), cx));
    cx.run_until_parked();

    stores.spaces.read_with(cx, |s, _| {
        assert_eq!(
            s.list().iter().map(|sp| sp.id.as_str()).collect::<Vec<_>>(),
            vec!["s1"],
            "archiving must remove the row from the cached listing (optimistic)"
        );
    });
}

#[gpui::test]
fn library_rename_updates_cached_title(cx: &mut TestAppContext) {
    // Calling `SpacesStore::rename` must update the cached title immediately
    // (optimistic local update) so the Library responds without a round-trip.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![stub_space("s1", Some("Old title"), None, 1_000)];
    });

    let (_window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });

    stores
        .spaces
        .update(cx, |s, cx| s.rename("s1".into(), "New title".into(), cx));
    cx.run_until_parked();

    stores.spaces.read_with(cx, |s, _| {
        let title = s.list().first().and_then(|sp| sp.title.as_deref());
        assert_eq!(
            title,
            Some("New title"),
            "rename must update the cached row"
        );
    });
}

#[gpui::test]
fn library_begin_rename_tracks_space(cx: &mut TestAppContext) {
    // `begin_rename` must set the renaming state so the view knows which row
    // is being renamed; `cancel_rename` must clear it.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 1_000)];
    });

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });

    // No rename in progress at construction.
    view.read_with(cx, |v, _| {
        assert_eq!(v.renaming_space_id(), None);
    });

    // Begin rename for s1.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.begin_rename("s1".into(), Some("Tides".into()), window, cx);
        });
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.renaming_space_id(), Some("s1"));
    });

    // Cancel clears the state.
    view.update(cx, |v, cx| v.cancel_rename(cx));
    view.read_with(cx, |v, _| {
        assert_eq!(v.renaming_space_id(), None);
    });
}

// ---------------------------------------------------------------------------
// Onboarding state machine
// ---------------------------------------------------------------------------

#[gpui::test]
fn chat_without_account_is_welcome_stage(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(false));
    });
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(cx, true),
            OnboardingStage::Welcome,
            "no account → the empty page is the welcome page"
        );
    });
}

#[gpui::test]
fn welcome_begin_enters_account_creation(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(false));
    });
    let (_window, view) = open_chat(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(cx, true),
            OnboardingStage::Plans,
            "account + known-zero balance + empty wallet → plans page"
        );
    });
}

#[gpui::test]
fn unknown_balance_is_ready_stage(cx: &mut TestAppContext) {
    // Balances not yet fetched (None) must NOT claim the user is unfunded —
    // the page stays the normal blank page until the snapshot is known.
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
    });
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.onboarding_stage(cx, true), OnboardingStage::Ready);
    });
}

#[gpui::test]
fn wallet_credentials_bypass_plans_stage(cx: &mut TestAppContext) {
    // Zero account balance but a spendable wallet credential → chat works,
    // so the plans page must not appear.
    let stores = stub_stores(cx, |c| {
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
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.onboarding_stage(cx, true), OnboardingStage::Ready);
    });
}

#[gpui::test]
fn positive_balance_is_ready_stage(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.onboarding_stage(cx, true), OnboardingStage::Ready);
    });
}

#[gpui::test]
fn composer_text_overrides_plans_stage(cx: &mut TestAppContext) {
    // If the user has started typing, the onboarding pages must not
    // replace the page out from under them.
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.onboarding_stage(cx, false), OnboardingStage::Ready);
    });
}

#[gpui::test]
fn plan_click_enters_checkout_pending(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| {
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
    let (_window, view) = open_chat(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.config_state = Some(config_state(true));
        c.balances = Some(BalancesResult {
            available: 0,
            pools: Vec::new(),
        });
    });
    let (_window, view) = open_chat(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.onboarding_stage(cx, true), OnboardingStage::Plans);
    });

    view.update(cx, |v, cx| v.dismiss_onboarding(cx));
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.onboarding_stage(cx, true),
            OnboardingStage::Ready,
            "\"I'll do this later\" must fall through to the normal blank page"
        );
        assert!(!v.onboarding().awaiting_checkout);
    });
}

#[gpui::test]
fn insufficient_balance_failure_surfaces_plans_below_transcript(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_chat(cx, &stores);

    view.update(cx, |v, cx| {
        v.set_messages_for_test(
            vec![SpaceMessage {
                role: "user".into(),
                content: "hello".into(),
            }],
            cx,
        );
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
        assert!(v.streaming(cx).is_none());
        // Typed routing: the transcript stays (Ready stage), no page swap.
        assert_eq!(v.onboarding_stage(cx, true), OnboardingStage::Ready);
    });
}

#[gpui::test]
fn non_balance_failure_does_not_surface_plans(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_chat(cx, &stores);

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
    stores: &Stores,
) -> (AnyWindowHandle, Entity<UpdatesView>) {
    let stores = stores.clone();
    open_view(cx, |window, cx| {
        cx.new(|cx| UpdatesView::new(stores.clone(), window, cx))
    })
}

#[gpui::test]
fn updates_view_none_yet_on_fresh_stub(cx: &mut TestAppContext) {
    // Stub stores: the constructor's load/check calls are no-ops, so the
    // view sits honestly on "no check has completed yet".
    let stores = stub_stores(cx, |_| {});
    let (_window, view) = open_updates(cx, &stores);
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert_eq!(v.display(cx), UpdatesDisplay::NoneYet);
    });
    stores.update.read_with(cx, |u, _| {
        assert!(!u.checking(), "stub check must not set in-flight");
    });
}

#[gpui::test]
fn updates_view_checking_state(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| c.update_checking = true);
    let (_window, view) = open_updates(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.display(cx), UpdatesDisplay::Checking);
    });
}

#[gpui::test]
fn updates_view_up_to_date_state(cx: &mut TestAppContext) {
    // Matrix row: no newer `latest` release. Also covers "background-check
    // result is reflected when the window opens": the snapshot is in the
    // core *before* the view is constructed.
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::UpToDate {
            latest_version: Some("0.1.0".into()),
        }));
    });
    let (_window, view) = open_updates(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::UpdateAvailable {
            release: verified_release(false),
        }));
    });
    let (_window, view) = open_updates(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::Unverifiable {
            version: "0.2.0".into(),
            tag: "v0.2.0".into(),
            reason: "signature is not from the pinned release identity".into(),
        }));
    });
    let (_window, view) = open_updates(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::ClaimsChanged {
            release: verified_release(false),
            comparison: claims_comparison(),
        }));
    });
    let (_window, view) = open_updates(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::CheckFailed {
            message: "GET …: connection refused".into(),
        }));
    });
    let (_window, view) = open_updates(cx, &stores);

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
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::UpToDate {
            latest_version: None,
        }));
        c.update_checking = true;
    });
    let (_window, view) = open_updates(cx, &stores);

    view.read_with(cx, |v, cx| {
        assert!(
            matches!(v.display(cx), UpdatesDisplay::UpToDate { .. }),
            "standing result must not be masked by a re-check"
        );
    });
}

#[gpui::test]
fn updates_view_actions_are_noops_on_stub(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |c| {
        c.update_check = Some(snapshot(UpdateCheckResult::ClaimsChanged {
            release: verified_release(false),
            comparison: claims_comparison(),
        }));
    });
    let (_window, view) = open_updates(cx, &stores);

    view.update(cx, |v, cx| {
        v.check_now(cx);
        v.accept_claims(cx);
    });
    cx.run_until_parked();

    // No backend: neither flag flips, the standing state is untouched.
    stores.update.read_with(cx, |u, _| {
        assert!(!u.checking());
        assert!(matches!(
            u.snapshot().map(|s| &s.result),
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
// Settings — two-pane nav, option reveal, reset confirm
// ---------------------------------------------------------------------------

#[gpui::test]
fn settings_nav_switches_panes(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), WindowInput::new(cx), window, cx))
    });

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.selected(),
            SettingsPane::General,
            "General is the resting pane"
        );
    });

    view.update(cx, |v, cx| v.select(SettingsPane::Wallet, cx));
    view.read_with(cx, |v, _| assert_eq!(v.selected(), SettingsPane::Wallet));

    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
    view.read_with(cx, |v, _| assert_eq!(v.selected(), SettingsPane::Account));
}

#[gpui::test]
fn general_option_reveal_tracks_modifier_state(cx: &mut TestAppContext) {
    // The advanced rows appear only while ⌥ is held. The pane's root
    // registers `on_modifiers_changed`, which calls `set_advanced` with the
    // live alt state — this drives the same method.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), WindowInput::new(cx), window, cx))
    });

    let general = view.read_with(cx, |v, _| v.general());
    general.read_with(cx, |g, _| {
        assert!(!g.advanced(), "advanced section is hidden at rest");
    });

    general.update(cx, |g, cx| g.set_advanced(true, cx));
    general.read_with(cx, |g, _| assert!(g.advanced()));

    // Releasing ⌥ hides it again.
    general.update(cx, |g, cx| g.set_advanced(false, cx));
    general.read_with(cx, |g, _| assert!(!g.advanced()));
}

/// Bug replay: wave-2 bug 2 — the Settings > General ⌥ reveal was dead
/// because `ModifiersChangedEvent` dispatches along the focused element's
/// ancestor path only. A `GeneralView`-local listener never fired while a
/// text input in a sibling pane (or the Base URL input inside General itself)
/// had focus. The fix: one listener on the `SettingsView` root (always an
/// ancestor of whatever is focused) that mirrors events into `WindowInput`;
/// `GeneralView` observes that entity.
///
/// This test replays the dispatch through `VisualTestContext::simulate_modifiers_change`
/// (the same platform-event path as production — no mock shortcuts).
#[gpui::test]
fn settings_general_option_reveal_works_via_root_listener(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), WindowInput::new(cx), window, cx))
    });

    let general = view.read_with(cx, |v, _| v.general());
    general.read_with(cx, |g, _| {
        assert!(!g.advanced(), "advanced section hidden at rest");
    });

    // Drive the real modifier-changed dispatch path: platform event →
    // window → gpui focus dispatch chain → `SettingsView` root listener →
    // `WindowInput::update_modifiers` → `GeneralView` observer →
    // `GeneralView::set_advanced(true)`.
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_modifiers_change(Modifiers {
        alt: true,
        ..Modifiers::default()
    });
    general.read_with(&vcx, |g, _| {
        assert!(
            g.advanced(),
            "⌥ held must reveal advanced rows via the root listener"
        );
    });

    vcx.simulate_modifiers_change(Modifiers::default());
    general.read_with(&vcx, |g, _| {
        assert!(!g.advanced(), "releasing ⌥ must hide advanced rows");
    });
}

#[gpui::test]
fn account_reset_requires_two_steps(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores.clone(), window, cx))
    });
    cx.run_until_parked();

    view.read_with(cx, |v, _| assert!(!v.reset_armed()));

    // First click arms; nothing is reset yet.
    view.update(cx, |v, cx| v.request_reset(cx));
    view.read_with(cx, |v, _| assert!(v.reset_armed()));
    stores.config.read_with(cx, |c, _| {
        assert!(
            c.state().unwrap().has_account,
            "arming must not reset anything"
        );
    });

    // Cancel disarms.
    view.update(cx, |v, cx| v.cancel_reset(cx));
    view.read_with(cx, |v, _| assert!(!v.reset_armed()));

    // Confirm without arming is a no-op guard; arm + confirm goes through
    // (stub core: `reset_account` early-returns after the local mutation).
    view.update(cx, |v, cx| v.confirm_reset(cx));
    view.read_with(cx, |v, _| assert!(!v.reset_armed()));
    view.update(cx, |v, cx| {
        v.request_reset(cx);
        v.confirm_reset(cx);
    });
    view.read_with(cx, |v, _| assert!(!v.reset_armed()));
}

// ---------------------------------------------------------------------------
// Record window
// ---------------------------------------------------------------------------

fn stub_attestation(hash: &str, ts: i64) -> AttestationInfo {
    AttestationInfo {
        hash: hash.into(),
        pcr_digest: Some("pcr-abc".into()),
        created_at: ts,
        doc_bytes: 2_048,
        connection_count: 3,
    }
}

fn stub_request(id: &str, ts: i64) -> RequestInfo {
    RequestInfo {
        id: id.into(),
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        response_status: Some(200),
        duration_ms: Some(742),
        request_at: ts,
        error: None,
        attempt_number: 1,
        credential_nonce: Some("nonce-1".into()),
        transport: Some("clearnet".into()),
        base_url: Some("https://eidola.example".into()),
        attestation_hash: Some("att-1".into()),
    }
}

#[gpui::test]
fn record_section_switching(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.section(),
            RecordSection::Attestations,
            "attestations first"
        );
        assert!(v.detail().is_none());
    });

    view.update(cx, |v, cx| v.select_section(RecordSection::Requests, cx));
    view.read_with(cx, |v, _| assert_eq!(v.section(), RecordSection::Requests));

    view.update(cx, |v, cx| v.select_section(RecordSection::Spending, cx));
    view.read_with(cx, |v, _| assert_eq!(v.section(), RecordSection::Spending));
}

#[gpui::test]
fn record_detail_open_and_close(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });

    view.update(cx, |v, _| {
        v.set_requests_for_test(vec![stub_request("req-1", 1_000)], false);
    });

    // Clicking a row starts the detail fetch. With a stub core there is no
    // backend, so the observable transition is the pending marker — the
    // same up-to-the-backend-guard pattern the chat submit tests use.
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Requests, cx);
        v.open_request("req-1".into(), cx);
    });
    view.read_with(cx, |v, _| {
        assert_eq!(v.detail_pending(), Some("req-1"));
        assert!(v.detail().is_none());
    });

    // The fetch landing installs the detail (simulated via the test setter).
    view.update(cx, |v, _| {
        v.set_detail_for_test(Some(RecordDetail::Attestation(AttestationDetail {
            hash: "att-1".into(),
            pcr_digest: None,
            created_at: 1_000,
            doc: b"{\"v\":1}".to_vec(),
        })));
    });
    view.read_with(cx, |v, _| {
        assert!(v.detail().is_some());
        assert!(v.detail_pending().is_none());
    });

    // Back returns to the listing; switching sections also closes detail.
    view.update(cx, |v, cx| v.close_detail(cx));
    view.read_with(cx, |v, _| assert!(v.detail().is_none()));
}

#[gpui::test]
fn record_renders_stubbed_rows_without_backend(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });

    view.update(cx, |v, cx| {
        v.set_attestations_for_test(
            vec![stub_attestation("a1", 2_000), stub_attestation("a2", 1_000)],
            true,
        );
        cx.notify();
    });
    cx.run_until_parked();

    // Rows installed by the setter must survive render (construction's
    // fetch is a no-op on a stub core).
    view.read_with(cx, |v, _| {
        assert_eq!(v.section(), RecordSection::Attestations);
        assert!(v.detail().is_none());
    });
}

#[gpui::test]
fn record_request_detail_exposes_space_link(cx: &mut TestAppContext) {
    // A RequestDetail with space_id set must make that id accessible so the
    // rendering path can present a "Space" link row.  This test verifies the
    // data plumbing (the detail struct carries the id); the actual
    // `open_space_window` dispatch is deferred and requires a real AppGlobal,
    // so we test the reachable data layer only.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });

    let detail = RequestDetail {
        id: "req-linked".into(),
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        request_headers: None,
        request_body: None,
        response_status: Some(200),
        response_headers: None,
        response_body: None,
        request_at: 1_000,
        response_at: None,
        duration_ms: None,
        error: None,
        retry_of_id: None,
        attempt_number: 1,
        credential_nonce: None,
        action_id: Some("act-1".into()),
        transport: None,
        base_url: None,
        attestation_hash: None,
        space_id: Some("space-abc".into()),
        space_title: Some("The quantum eraser experiment".into()),
    };

    view.update(cx, |v, _| {
        v.set_detail_for_test(Some(RecordDetail::Request(Box::new(detail))));
    });

    view.read_with(cx, |v, _| match v.detail() {
        Some(RecordDetail::Request(d)) => {
            assert_eq!(d.space_id.as_deref(), Some("space-abc"));
            assert_eq!(
                d.space_title.as_deref(),
                Some("The quantum eraser experiment")
            );
        }
        _ => panic!("expected Request detail"),
    });
}

#[gpui::test]
fn record_frame_work_is_constant_in_loaded_rows(cx: &mut TestAppContext) {
    // The wave-2 bug-3 fix: with virtualization, the per-frame work (what the
    // `uniform_list` closure does — render exactly the visible window) must be
    // O(visible), not O(loaded). Load one page, then ten pages, and assert the
    // visible-window render produces the same fixed number of rows in both
    // cases (and far fewer than the total) — the structural guarantee. Also a
    // coarse timing comparison: ten pages must not cost meaningfully more per
    // frame than one.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });

    let one_page: Vec<_> = (0..51).map(|i| stub_request(&format!("r{i}"), i)).collect();
    let ten_pages: Vec<_> = (0..510)
        .map(|i| stub_request(&format!("r{i}"), i))
        .collect();

    // A fixed visible window (what a ~640px-tall viewport shows at ROW_H).
    let visible = 0..12usize;

    // One page loaded.
    let (one_window, one_total, one_dur) = view.update(cx, |v, cx| {
        v.set_requests_for_test(one_page.clone(), true);
        v.select_section(RecordSection::Requests, cx);
        let start = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..200 {
            n = v.render_visible_window_for_test(visible.clone(), cx);
        }
        (n, v.display_len_for_test(), start.elapsed())
    });

    // Ten pages loaded.
    let (ten_window, ten_total, ten_dur) = view.update(cx, |v, cx| {
        v.set_requests_for_test(ten_pages.clone(), true);
        let start = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..200 {
            n = v.render_visible_window_for_test(visible.clone(), cx);
        }
        (n, v.display_len_for_test(), start.elapsed())
    });

    // The display model grew 10× …
    assert_eq!(one_total, 52, "one page = 51 rows + load-more");
    assert_eq!(ten_total, 511, "ten pages = 510 rows + load-more");
    // … but the visible window rendered the same fixed count both times,
    // far below the total — O(visible), not O(loaded).
    assert_eq!(one_window, 12);
    assert_eq!(
        ten_window, 12,
        "per-frame row count must not grow with loaded rows"
    );

    // Coarse timing: per-frame visible-window cost must not scale with the
    // loaded-row count. Generous slack absorbs scheduler noise — we're
    // catching O(loaded) regressions (which would be ~10×), not microbenching.
    assert!(
        ten_dur.as_secs_f64() < one_dur.as_secs_f64() * 4.0 + 0.05,
        "frame work scaled with loaded rows: 1 page {one_dur:?} vs 10 pages {ten_dur:?}"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn config_state(has_account: bool) -> ConfigState {
    ConfigState {
        base_url: "https://eidola.example/v1".into(),
        default_model: "gemma4-31b".into(),
        base_url_pin: "https://eidola.example/v1".into(),
        base_url_is_override: false,
        has_account,
        has_account_secret: has_account,
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        trusted_measurements: Vec::new(),
        trusted_measurements_are_override: false,
        has_hardware_root_ca: false,
        has_hardware_intermediate_ca: false,
        attestation_url: None,
    }
}

/// Build stub stores from a declaratively-described scene — the replacement
/// for the old `Core::stub()` field-poking.
fn stub_stores(cx: &mut TestAppContext, setup: impl FnOnce(&mut StoresStub)) -> Stores {
    cx.update(|cx| {
        let mut fixture = StoresStub::default();
        setup(&mut fixture);
        Stores::stub_with(fixture, cx)
    })
}

/// Stub stores representing a funded, ready account — the fixture the plain
/// chat tests use.
fn stub_stores_with_config(cx: &mut TestAppContext) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    })
}

fn open_chat(cx: &mut TestAppContext, stores: &Stores) -> (AnyWindowHandle, Entity<ChatView>) {
    let stores = stores.clone();
    open_view(cx, |window, cx| {
        cx.new(|cx| ChatView::new(stores.clone(), None, WindowInput::new(cx), window, cx))
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
