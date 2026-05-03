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
use eidola_gui::chat::ChatView;
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

    s.add("chat_thinking", size(px(900.), px(640.)), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let view = ChatView::new(core, window, cx);
            view_thinking(view)
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

fn view_thinking(view: ChatView) -> ChatView {
    let mut view = view;
    view.set_thinking_for_test(true);
    view
}
