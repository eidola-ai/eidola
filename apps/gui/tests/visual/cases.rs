//! Snapshot test cases. Each case constructs a `Core` in a known state, then
//! renders one of the views from `eidola-gui` to a PNG.
//!
//! When you add a new view state, add it here as another `s.add(…)` call. The
//! first run will write the golden image; subsequent runs verify against it.

use eidola_app_core::{
    BalancePoolInfo, BalancesResult, ConfigState, CredentialInfo, MeasurementInfo, PriceInfo,
    SpaceMessage,
};
use eidola_gui::account::AccountView;
use eidola_gui::chat::{ChatView, StreamingResponse};
use eidola_gui::core::Core;
use eidola_gui::general::GeneralView;
use eidola_gui::settings::SettingsView;
use eidola_gui::wallet::WalletView;
use gpui::{App, AppContext, Entity, px, size};

use super::harness::Snapshots;

pub fn register(s: &mut Snapshots) {
    register_chat(s);
    register_account(s);
    register_wallet(s);
    register_general(s);
    register_settings(s);
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

fn register_chat(s: &mut Snapshots) {
    s.add("chat_empty", size(px(900.), px(640.)), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| ChatView::new(core, window, cx))
    });

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
                let view = ChatView::new(core, window, cx);
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
                let view = ChatView::new(core, window, cx);
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

    s.add("chat_thinking", size(px(900.), px(640.)), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let view = ChatView::new(core, window, cx);
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
                let mut view = ChatView::new(core, window, cx);
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
                let view = ChatView::new(core, window, cx);
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
                            expires_at: Some("2026-06-01T00:00:00Z".into()),
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
        base_url: Some("https://eidola.example/v1".into()),
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
