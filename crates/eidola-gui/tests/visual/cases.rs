//! Snapshot test cases. Each case constructs a `Core` in a known state, then
//! renders one of the views from `eidola-gui` to a PNG.
//!
//! When you add a new view state, add it here as another `s.add(…)` call. The
//! first run will write the golden image; subsequent runs verify against it.

use eidola_app_core::{
    BalancePoolInfo, BalancesResult, ConfigState, CredentialInfo, InFlightCredentialInfo,
    MeasurementInfo, PriceInfo, SpaceInfo, SpaceMessage,
};
use eidola_gui::account::AccountView;
use eidola_gui::chat::{ChatView, StreamingResponse};
use eidola_gui::core::Core;
use eidola_gui::general::GeneralView;
use eidola_gui::library::LibraryView;
use eidola_gui::settings::SettingsView;
use eidola_gui::wallet::WalletView;
use gpui::{App, AppContext, Entity, px, size};
use gpui_markdown_editor::{EditorState, Selection};

use super::harness::Snapshots;

pub fn register(s: &mut Snapshots) {
    register_chat(s);
    register_onboarding(s);
    register_library(s);
    register_account(s);
    register_wallet(s);
    register_general(s);
    register_settings(s);
}

// ---------------------------------------------------------------------------
// Onboarding (chat window empty states)
// ---------------------------------------------------------------------------

fn register_onboarding(s: &mut Snapshots) {
    // No account → the empty page is the welcome page.
    s.add(
        "onboarding_welcome",
        size(px(705.), px(705.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(false));
                c
            });
            cx.new(|cx| ChatView::new(core, None, window, cx))
        },
    );

    // Account just created, balance known-zero → the plans page.
    s.add(
        "onboarding_plans",
        size(px(705.), px(705.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(true));
                c.balances = Some(BalancesResult {
                    available: 0,
                    pools: Vec::new(),
                });
                c.prices = stub_prices();
                c
            });
            cx.new(|cx| ChatView::new(core, None, window, cx))
        },
    );

    // Checkout URL opened — the balance poll is running.
    s.add(
        "onboarding_plans_waiting",
        size(px(705.), px(705.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(true));
                c.balances = Some(BalancesResult {
                    available: 0,
                    pools: Vec::new(),
                });
                c.prices = stub_prices();
                c
            });
            cx.new(|cx| {
                let mut view = ChatView::new(core, None, window, cx);
                view.onboarding_mut_for_test().awaiting_checkout = true;
                view
            })
        },
    );

    // A later submit failed with InsufficientBalance: the plans surface
    // below the transcript via the error band — not a modal.
    s.add(
        "chat_insufficient_balance_plans",
        size(px(705.), px(705.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(true));
                c.balances = Some(BalancesResult {
                    available: 100,
                    pools: Vec::new(),
                });
                c.prices = stub_prices();
                c
            });
            cx.new(|cx| {
                let mut view = ChatView::new(core, None, window, cx);
                view.set_messages_for_test(vec![SpaceMessage {
                    role: "user".into(),
                    content: "Can you summarize the attached design doc?".into(),
                }]);
                view.set_error_for_test(Some(
                    "insufficient balance: 6200 credits required, 100 available".into(),
                ));
                view.show_plans_after_error = true;
                view
            })
        },
    );
}

fn stub_prices() -> Vec<PriceInfo> {
    vec![
        PriceInfo {
            id: "price_starter".into(),
            product_name: "Starter".into(),
            product_description: Some("A month of casual questions".into()),
            amount_display: "5.00 USD".into(),
            recurrence: "/month".into(),
            credits: 5_000_000,
        },
        PriceInfo {
            id: "price_standard".into(),
            product_name: "Standard".into(),
            product_description: Some("Daily thinking, long documents".into()),
            amount_display: "20.00 USD".into(),
            recurrence: "/month".into(),
            credits: 20_000_000,
        },
        PriceInfo {
            id: "price_topup".into(),
            product_name: "Top-up".into(),
            product_description: None,
            amount_display: "10.00 USD".into(),
            recurrence: "".into(),
            credits: 10_000_000,
        },
    ]
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

fn register_chat(s: &mut Snapshots) {
    s.add("chat_empty", size(px(900.), px(640.)), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| ChatView::new(core, None, window, cx))
    });

    // Narrow window — guards that the chapter delimiter tracks the prose
    // body's width edge-for-edge. Earlier the delim sized itself
    // independently and rendered small + left-aligned when the window was
    // narrower than the prose max-width cap. Snapshot here is wider than
    // the rule's hairline so a regression would be obvious.
    s.add(
        "chat_with_messages_narrow",
        size(px(480.), px(520.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                view_with_messages(
                    view,
                    vec![
                        SpaceMessage {
                            role: "user".into(),
                            content: "Quick check?".into(),
                        },
                        SpaceMessage {
                            role: "assistant".into(),
                            content: "Yes — everything looks fine on the latest deploy.".into(),
                        },
                    ],
                )
            })
        },
    );

    // "Mid" width — wider than the prose max-width cap (640px) but well
    // short of the original 900px reference. This is the size where the
    // earlier flex-1-rules implementation collapsed: prose's max-w bound,
    // but the inner h_flex's flex-1 rules had no definite parent width to
    // grow into and rendered as a left-aligned label with no rules.
    // 680px logical width — exactly the size where the user observed
    // the delim outer container collapsing to content width in the live
    // app (1360 physical at 2x DPR). The bug shows the outer `RED` debug
    // border shrunk to wrap the absolute rule + label rather than
    // stretching to the row width. Earlier we used a plain `div()` here
    // and the offscreen renderer happened not to reproduce; switching
    // the outer to `v_flex()` (matching the message row) made the live
    // app and the test both stretch correctly.
    s.add(
        "chat_with_messages_breakpoint",
        size(px(680.), px(640.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                view_with_messages(
                    view,
                    vec![
                        SpaceMessage {
                            role: "user".into(),
                            content: "Breakpoint check.".into(),
                        },
                        SpaceMessage {
                            role: "assistant".into(),
                            content: "Delim outer should stretch to full row width regardless \
                                of the prose column being narrower."
                                .into(),
                        },
                    ],
                )
            })
        },
    );

    // Live-app width — mirrors the size where the user reported the
    // delim breaking with rules visible only on the left side. If this
    // reproduces locally, the bug is in our layout code; if it doesn't,
    // there's something the offscreen renderer does differently from
    // the live app harness.
    s.add(
        "chat_with_messages_live",
        size(px(1400.), px(1000.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                view_with_messages(
                    view,
                    vec![
                        SpaceMessage {
                            role: "user".into(),
                            content: "Live width check.".into(),
                        },
                        SpaceMessage {
                            role: "assistant".into(),
                            content: "If you can see this rule extending past the label on both \
                                left and right, the layout is correct at the user's reported \
                                window width."
                                .into(),
                        },
                    ],
                )
            })
        },
    );

    s.add(
        "chat_with_messages_mid",
        size(px(820.), px(640.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                view_with_messages(
                    view,
                    vec![
                        SpaceMessage {
                            role: "user".into(),
                            content: "Mid-width check.".into(),
                        },
                        SpaceMessage {
                            role: "assistant".into(),
                            content:
                                "Hairline rule should span the full prose column width with the \
                            label centered and masking the line behind it."
                                    .into(),
                        },
                    ],
                )
            })
        },
    );

    s.add(
        "chat_with_messages",
        size(px(900.), px(640.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(true));
                c
            });
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                // Push a few messages directly into the view's state so we can
                // render the populated chat without driving any async work.
                view_with_messages(
                    view,
                    vec![
                        SpaceMessage {
                            role: "user".into(),
                            content: "What's the deployment status?".into(),
                        },
                        SpaceMessage {
                            role: "assistant".into(),
                            content: "Last release v0.0.93 was deployed at 14:02 UTC. The Tinfoil \
                                  enclave verifier reports a fresh attestation chain."
                                .into(),
                        },
                        SpaceMessage {
                            role: "user".into(),
                            content: "Any pending work?".into(),
                        },
                    ],
                )
            })
        },
    );

    s.add(
        "chat_with_markdown",
        size(px(900.), px(640.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(true));
                c
            });
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                view_with_messages(
                    view,
                    vec![
                        SpaceMessage {
                            role: "user".into(),
                            content: "Show me how to register a tokio runtime in a small Rust \
                                program, with a heading, a list, and a code fence."
                                .into(),
                        },
                        SpaceMessage {
                            role: "assistant".into(),
                            content: "## Registering a runtime\n\nYou have two convenient \
                                options:\n\n1. **Macro** — `#[tokio::main]` rewrites `main` for \
                                you.\n2. **Manual** — build a `Runtime` and call `block_on`.\n\n\
                                Manual setup, for when you need fine control:\n\n```rust\n\
                                use tokio::runtime::Runtime;\n\nfn main() {\n    let rt = \
                                Runtime::new().expect(\"build runtime\");\n    rt.block_on(async \
                                {\n        println!(\"hello from tokio\");\n    });\n}\n```\n\n\
                                The macro form is shorter, but the manual form makes the \
                                runtime's *lifetime* explicit — useful when you want to share \
                                one runtime across an FFI boundary."
                                .into(),
                        },
                    ],
                )
            })
        },
    );

    // Composer (WYSIWYG markdown editor) populated with the constructs
    // whose typography has to track the transcript: inline code inside a
    // paragraph and inside list items, a tight list, and a code fence.
    // This is the surface where inline-code sizing and list-item spacing
    // are judged against the prose body under the real Circadian theme +
    // Newsreader pairing (the editor crate's own snapshots render with
    // the default system fonts, so they can't catch pairing problems).
    s.add(
        "chat_composer_markdown",
        size(px(900.), px(640.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            let view = cx.new(|cx| {
                let view = ChatView::new(core, window, cx);
                view_with_messages(
                    view,
                    vec![SpaceMessage {
                        role: "user".into(),
                        content: "Transcript turn above the composer, with `inline code` \
                            for comparison."
                            .into(),
                    }],
                )
            });
            let editor = view.read(cx).prompt_editor_for_test();
            editor.update(cx, |editor, cx| {
                let markdown = "Drafting a reply that mixes `Runtime::new()` and plain \
                    prose in one paragraph.\n\
                    \n\
                    - the macro rewrites `main` for you\n\
                    - manual setup hands you a runtime to `block_on`\n\
                    - both drive the same scheduler\n\
                    \n\
                    And a fence for comparison:\n\
                    \n\
                    ```rust\n\
                    let rt = Runtime::new()?;\n\
                    ```\n";
                editor.state = EditorState {
                    markdown: markdown.into(),
                    selection: Selection::Cursor(0),
                };
                cx.notify();
            });
            view
        },
    );

    s.add("chat_thinking", size(px(900.), px(640.)), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let view = ChatView::new(core, None, window, cx);
            // Empty streaming response — renders the collapsed "Thinking…"
            // header with no body yet, the moment after the user submits.
            view_streaming(view, StreamingResponse::default())
        })
    });

    s.add(
        "chat_finalized_with_thinking",
        size(px(900.), px(640.)),
        |window, cx| {
            // Reasoning persists past the stream end: a finalized
            // assistant message exposes a "Thinking" disclosure that
            // the user can re-open. Rendered here in the expanded
            // state to verify the layout when the thinking body is
            // visible alongside the answer.
            let core = stub_core_with_config(cx);
            cx.new(|cx| {
                let mut view = ChatView::new(core, None, window, cx);
                view.set_messages_for_test(vec![
                    SpaceMessage {
                        role: "user".into(),
                        content: "What's a Hilbert space, in one paragraph?".into(),
                    },
                    SpaceMessage {
                        role: "assistant".into(),
                        content: "A **Hilbert space** is a complete inner-product space — a \
                            vector space equipped with an inner product whose induced norm \
                            makes it a Banach space. The completeness lets you reason about \
                            limits of Cauchy sequences (essential for things like Fourier \
                            analysis), and the inner product gives you geometry: angles, \
                            orthogonality, projections."
                            .into(),
                    },
                ]);
                view.set_reasoning_for_test(
                    1,
                    "The user wants a one-paragraph definition. I should hit: vector space \
                        + inner product, the induced norm, and completeness. Mention Fourier \
                        as an application motivator. Skip the formal axioms — they're not \
                        what 'in one paragraph' is asking for."
                        .into(),
                    true,
                );
                view
            })
        },
    );

    s.add(
        "chat_streaming_partial",
        size(px(900.), px(640.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            cx.new(|cx| {
                let view = ChatView::new(core, None, window, cx);
                let view = view_with_messages(
                    view,
                    vec![SpaceMessage {
                        role: "user".into(),
                        content: "Why is the sky blue?".into(),
                    }],
                );
                view_streaming(
                    view,
                    StreamingResponse {
                        reasoning: "Let me think about Rayleigh scattering. Short \
                            wavelengths interact more strongly with air molecules \
                            than long wavelengths, so blue light gets scattered in \
                            all directions while red passes through more directly.\n\n\
                            I should mention the Sun's spectrum and human vision \
                            response too — but keep it tight."
                            .into(),
                        content: "The sky looks blue because of **Rayleigh scattering**. \
                            Sunlight is white, but as it passes through Earth's \
                            atmosphere, shorter (blue) wavelengths scatter more strongly \
                            than longer (red) ones, so blue light reaches your"
                            .into(),
                        expanded: true,
                        error: None,
                    },
                )
            })
        },
    );
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

fn library_space(id: &str, title: Option<&str>, snippet: Option<&str>, days_ago: i64) -> SpaceInfo {
    let ts = eidola_app_core::now_ms() - days_ago * 24 * 60 * 60 * 1000;
    SpaceInfo {
        id: id.into(),
        title: title.map(String::from),
        snippet: snippet.map(String::from),
        created_at: ts,
        last_activity_at: ts,
        message_count: 4,
        archived_at: None,
    }
}

fn library_core(cx: &mut App) -> Entity<Core> {
    cx.new(|_| {
        let mut c = Core::stub();
        c.spaces = vec![
            library_space("s1", Some("Tides and the moon"), None, 0),
            library_space(
                "s2",
                Some("Borrow checker, closures, and lifetimes"),
                None,
                1,
            ),
            library_space(
                "s3",
                None,
                Some(
                    "what is a monad, really? I keep reading the burrito \
                     explanations and they don't land for me",
                ),
                3,
            ),
            library_space("s4", Some("Reading list for distributed systems"), None, 12),
            library_space(
                "s5",
                Some(
                    "A very long space title that should truncate with an ellipsis \
                      rather than wrap onto a second line",
                ),
                None,
                45,
            ),
            library_space("s6", None, None, 400),
        ];
        c
    })
}

fn register_library(s: &mut Snapshots) {
    s.add("library_empty", size(px(520.), px(620.)), |window, cx| {
        let core = cx.new(|_| Core::stub());
        cx.new(|cx| LibraryView::new(core, window, cx))
    });

    s.add(
        "library_with_spaces",
        size(px(520.), px(620.)),
        |window, cx| {
            let core = library_core(cx);
            cx.new(|cx| LibraryView::new(core, window, cx))
        },
    );

    // Hover state: the archive × is revealed on the hovered row.
    s.add("library_hovered", size(px(520.), px(620.)), |window, cx| {
        let core = library_core(cx);
        cx.new(|cx| {
            let mut view = LibraryView::new(core, window, cx);
            view.set_hovered_for_test(Some(1));
            view
        })
    });
}

// ---------------------------------------------------------------------------
// Account
// ---------------------------------------------------------------------------

fn register_account(s: &mut Snapshots) {
    s.add(
        "account_no_account",
        size(px(560.), px(720.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(false));
                c
            });
            cx.new(|cx| AccountView::new(core, window, cx))
        },
    );

    s.add(
        "account_with_balances",
        size(px(560.), px(720.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.config_state = Some(stub_config_state(true));
                c.balances = Some(BalancesResult {
                    available: 4_200,
                    pools: vec![
                        BalancePoolInfo {
                            amount: 3_000,
                            source: "subscription".into(),
                            expires_at: Some(1_780_000_000_000),
                        },
                        BalancePoolInfo {
                            amount: 1_200,
                            source: "topup".into(),
                            expires_at: None,
                        },
                    ],
                });
                c.prices = vec![
                    PriceInfo {
                        id: "price_basic".into(),
                        product_name: "Basic".into(),
                        product_description: Some("1,000 credits per month".into()),
                        amount_display: "10.00 USD".into(),
                        recurrence: "/month".into(),
                        credits: 1_000,
                    },
                    PriceInfo {
                        id: "price_pro".into(),
                        product_name: "Pro".into(),
                        product_description: Some("5,000 credits per month".into()),
                        amount_display: "40.00 USD".into(),
                        recurrence: "/month".into(),
                        credits: 5_000,
                    },
                ];
                c
            });
            cx.new(|cx| AccountView::new(core, window, cx))
        },
    );
}

// ---------------------------------------------------------------------------
// Wallet
// ---------------------------------------------------------------------------

fn register_wallet(s: &mut Snapshots) {
    s.add("wallet_empty", size(px(560.), px(480.)), |window, cx| {
        let core = cx.new(|_| Core::stub());
        cx.new(|cx| WalletView::new(core, window, cx))
    });

    s.add(
        "wallet_with_credentials",
        size(px(560.), px(480.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.credentials = vec![
                    CredentialInfo {
                        nonce: "a1b2c3d4e5f60718293a4b5c6d7e8f90".into(),
                        credits: 1_500,
                        generation: 0,
                    },
                    CredentialInfo {
                        nonce: "ff1122334455667788990011223344aa".into(),
                        credits: 2_700,
                        generation: 2,
                    },
                ];
                c
            });
            cx.new(|cx| WalletView::new(core, window, cx))
        },
    );

    s.add(
        "wallet_with_in_flight",
        size(px(560.), px(480.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                c.spending_credentials = vec![InFlightCredentialInfo {
                    nonce: "deadbeefcafef00d0123456789abcdef".into(),
                    credits: 800,
                    generation: 1,
                    spend_amount: 700,
                }];
                c.credentials = vec![CredentialInfo {
                    nonce: "a1b2c3d4e5f60718293a4b5c6d7e8f90".into(),
                    credits: 1_500,
                    generation: 0,
                }];
                c
            });
            cx.new(|cx| WalletView::new(core, window, cx))
        },
    );
}

// ---------------------------------------------------------------------------
// General settings
// ---------------------------------------------------------------------------

fn register_general(s: &mut Snapshots) {
    s.add("general_default", size(px(560.), px(720.)), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| GeneralView::new(core, window, cx))
    });

    s.add(
        "general_with_attestation",
        size(px(560.), px(720.)),
        |window, cx| {
            let core = cx.new(|_| {
                let mut c = Core::stub();
                let mut state = stub_config_state(true);
                state.attestation_url = Some("https://atc.tinfoil.sh/v1/attest".into());
                state.has_hardware_root_ca = true;
                state.has_hardware_intermediate_ca = true;
                state.trusted_measurements = vec![MeasurementInfo {
                    snp: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
                    tdx_rtmr1: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                        .into(),
                    tdx_rtmr2: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
                        .into(),
                }];
                c.config_state = Some(state);
                c
            });
            cx.new(|cx| GeneralView::new(core, window, cx))
        },
    );
}

// ---------------------------------------------------------------------------
// Settings (full window with tab strip)
// ---------------------------------------------------------------------------

fn register_settings(s: &mut Snapshots) {
    s.add(
        "settings_window_general_tab",
        size(px(560.), px(480.)),
        |window, cx| {
            let core = stub_core_with_config(cx);
            cx.new(|cx| SettingsView::new(core, window, cx))
        },
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn stub_core_with_config(cx: &mut App) -> Entity<Core> {
    cx.new(|_| {
        let mut c = Core::stub();
        c.config_state = Some(stub_config_state(true));
        c
    })
}

fn stub_config_state(has_account: bool) -> ConfigState {
    ConfigState {
        base_url: "https://eidola.example/v1".into(),
        has_account,
        has_account_secret: has_account,
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        trusted_measurements: Vec::new(),
        has_hardware_root_ca: false,
        has_hardware_intermediate_ca: false,
        attestation_url: None,
    }
}

fn view_with_messages(view: ChatView, messages: Vec<SpaceMessage>) -> ChatView {
    let mut view = view;
    view.set_messages_for_test(messages);
    view
}

fn view_streaming(view: ChatView, streaming: StreamingResponse) -> ChatView {
    let mut view = view;
    view.set_streaming_for_test(Some(streaming));
    view
}
