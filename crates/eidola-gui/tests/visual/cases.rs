//! Snapshot test cases. Each case constructs a `Core` in a known state, then
//! renders one of the views from `eidola-gui` to a PNG.
//!
//! When you add a new view state, add it here as another `s.add(…)` call. The
//! first run will write the golden image; subsequent runs verify against it.

use eidola_app_core::updates::{
    Claim, ClaimDelta, ClaimsComparison, UpdateCheckResult, UpdateCheckSnapshot, VerifiedRelease,
};
use eidola_app_core::{
    AttestationDetail, AttestationInfo, BalancePoolInfo, BalancesResult, ConfigState,
    CredentialInfo, CredentialLifecycleInfo, InFlightCredentialInfo, MeasurementInfo, ModelInfo,
    PriceInfo, RequestDetail, RequestInfo, SpaceInfo, SpaceMessage, SpendTrailEntry,
};
use eidola_gui::chat::{ChatView, StreamingResponse};
use eidola_gui::core::Core;
use eidola_gui::library::LibraryView;
use eidola_gui::record::{RecordDetail, RecordSection, RecordView};
use eidola_gui::settings::{SettingsPane, SettingsView};
use eidola_gui::updates::UpdatesView;
use eidola_gui::wallet::WalletView;
use gpui::{App, AppContext, Entity, px, size};
use gpui_markdown_editor::{EditorState, Selection};

use super::harness::Snapshots;

pub fn register(s: &mut Snapshots) {
    register_chat(s);
    register_onboarding(s);
    register_library(s);
    register_settings(s);
    register_updates(s);
    register_record(s);
}

// ---------------------------------------------------------------------------
// Updates window — one case per display state, at the window's real size
// ---------------------------------------------------------------------------

fn register_updates(s: &mut Snapshots) {
    fn updates_core(cx: &mut App, setup: impl FnOnce(&mut Core)) -> Entity<Core> {
        cx.new(|_| {
            let mut c = Core::stub();
            setup(&mut c);
            c
        })
    }

    fn snapshot(result: UpdateCheckResult) -> UpdateCheckSnapshot {
        UpdateCheckSnapshot {
            checked_at_ms: eidola_app_core::now_ms() - 23 * 60 * 1000,
            result,
        }
    }

    fn release(claims_accepted: bool) -> VerifiedRelease {
        VerifiedRelease {
            version: "0.2.0".into(),
            tag: "v0.2.0".into(),
            release_url: Some("https://github.com/eidola-ai/eidola/releases/tag/v0.2.0".into()),
            published_at: Some("2026-06-01T12:00:00Z".into()),
            ci_identity:
                "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v0.2.0"
                    .into(),
            rekor_log_index: 168_338_903,
            manifest_sha256: "ab".repeat(32),
            claims_accepted,
        }
    }

    let sz = size(px(480.), px(360.));

    s.add("updates_checking", sz, |window, cx| {
        let core = updates_core(cx, |c| c.update_checking = true);
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_up_to_date", sz, move |window, cx| {
        let core = updates_core(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::UpToDate {
                latest_version: Some("0.1.0".into()),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_check_failed", sz, move |window, cx| {
        let core = updates_core(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::CheckFailed {
                message: "GET https://api.github.com/...: connection timed out".into(),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_available", sz, move |window, cx| {
        let core = updates_core(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::UpdateAvailable {
                release: release(false),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_unverifiable", sz, move |window, cx| {
        let core = updates_core(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::Unverifiable {
                version: "0.2.0".into(),
                tag: "v0.2.0".into(),
                reason: "signature is not from the pinned release identity: leaf cert SAN URI \
                         does not match the expected workflow pattern"
                    .into(),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add(
        "updates_claims_changed",
        size(px(480.), px(440.)),
        move |window, cx| {
            let core = updates_core(cx, |c| {
                c.update_check = Some(snapshot(UpdateCheckResult::ClaimsChanged {
                    release: release(false),
                    comparison: ClaimsComparison {
                        expected: vec![
                            Claim {
                                key: "manifest.schema_version".into(),
                                value: "1".into(),
                            },
                            Claim {
                                key: "enclave.snp_measurement".into(),
                                value: "SEV-SNP launch measurement (48-byte hex)".into(),
                            },
                            Claim {
                                key: "enclave.cmdline".into(),
                                value: "kernel command line (non-empty)".into(),
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
                    },
                }));
            });
            cx.new(|cx| UpdatesView::new(core, window, cx))
        },
    );
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
                let view = ChatView::new(core, None, window, cx);
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

    // ⌥ held — the model label reveals right-aligned in the title-bar band,
    // text_sm muted italic, matching the chapter-delim voice. The page
    // content underneath must be identical to the resting state (the band
    // is absolute chrome; the reveal cannot shift layout).
    s.add(
        "chat_model_reveal",
        size(px(705.), px(705.)),
        |window, cx| {
            let core = model_core(cx);
            cx.new(|cx| {
                let mut view = ChatView::new(core, None, window, cx);
                view.set_messages_for_test(vec![
                    SpaceMessage {
                        role: "user".into(),
                        content: "What's the tide schedule for tomorrow?".into(),
                    },
                    SpaceMessage {
                        role: "assistant".into(),
                        content: "High tide lands at 06:41 and 19:12; lows at 00:55 and 13:03. \
                        The morning high is the stronger of the two."
                            .into(),
                    },
                ]);
                view.set_alt_held_for_test(true);
                view
            })
        },
    );

    // Picker open (⌥⌘M or clicking the revealed label) — a quiet panel
    // under the band's right edge listing Core.models with honest
    // per-model info; current selection and config default marked, and
    // the secondary "set as default" affordance in the footer.
    s.add(
        "chat_model_picker",
        size(px(705.), px(705.)),
        |window, cx| {
            let core = model_core(cx);
            cx.new(|cx| {
                let mut view = ChatView::new(core, None, window, cx);
                view.set_messages_for_test(vec![SpaceMessage {
                    role: "user".into(),
                    content: "Comparing models for a long document review.".into(),
                }]);
                view.select_model("kimi-k2-6".into(), cx);
                view.set_model_picker_open_for_test(true);
                view.set_alt_held_for_test(true);
                view
            })
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
// Settings (two-pane window: nav band + pane)
// ---------------------------------------------------------------------------

/// A funded account fixture with pools and plans, shared by the settings
/// cases.
fn settings_core(cx: &mut App) -> Entity<Core> {
    cx.new(|_| {
        let mut c = Core::stub();
        c.config_state = Some(stub_config_state(true));
        c.balances = Some(BalancesResult {
            available: 4_200_000,
            pools: vec![
                BalancePoolInfo {
                    amount: 3_000_000,
                    source: "subscription".into(),
                    expires_at: Some(eidola_app_core::now_ms() + 23 * 24 * 60 * 60 * 1000),
                },
                BalancePoolInfo {
                    amount: 1_200_000,
                    source: "topup".into(),
                    expires_at: None,
                },
            ],
        });
        c.prices = vec![
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
        ];
        c.credential_lifecycle = vec![
            CredentialLifecycleInfo {
                nonce: "a1b2c3d4e5f60718293a4b5c6d7e8f90".into(),
                credits: 985_400,
                generation: 0,
                created_at: 4_000,
                state: "active".into(),
                spend_amount: None,
            },
            CredentialLifecycleInfo {
                nonce: "deadbeefcafef00d0123456789abcdef".into(),
                credits: 812_000,
                generation: 1,
                created_at: 3_000,
                state: "spending".into(),
                spend_amount: Some(6_200),
            },
            CredentialLifecycleInfo {
                nonce: "ff1122334455667788990011223344aa".into(),
                credits: 1_000_000,
                generation: 0,
                created_at: 2_000,
                state: "spent".into(),
                spend_amount: Some(14_600),
            },
            CredentialLifecycleInfo {
                nonce: "0099aabbccddeeff0011223344556677".into(),
                credits: 52_000,
                generation: 3,
                created_at: 1_000,
                state: "expired".into(),
                spend_amount: None,
            },
        ];
        c
    })
}

fn register_settings(s: &mut Snapshots) {
    let settings_size = size(px(620.), px(520.));

    // General at rest: base URL pin, advanced rows hidden behind ⌥.
    s.add("settings_general", settings_size, |window, cx| {
        let core = settings_core(cx);
        cx.new(|cx| SettingsView::new(core, window, cx))
    });

    // ⌥ held: advanced rows visible, with an overridden base URL and a
    // user-trusted measurement so the honest "override" annotations show.
    s.add("settings_general_advanced", settings_size, |window, cx| {
        let core = cx.new(|_| {
            let mut c = Core::stub();
            let mut state = stub_config_state(true);
            state.base_url = "https://staging.eidola.example/v1".into();
            state.base_url_is_override = true;
            state.attestation_url = Some("https://atc.tinfoil.sh/v1/attest".into());
            state.has_hardware_root_ca = true;
            state.trusted_measurements = vec![MeasurementInfo {
                snp: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
                tdx_rtmr1: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .into(),
                tdx_rtmr2: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
                    .into(),
            }];
            state.trusted_measurements_are_override = true;
            c.config_state = Some(state);
            c
        });
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        let general = view.read(cx).general();
        general.update(cx, |g, cx| g.set_advanced(true, cx));
        view
    });

    // Account pane: balance, pools with humanized expiry, plans.
    s.add("settings_account", settings_size, |window, cx| {
        let core = settings_core(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
        view
    });

    // Account pane with the reset confirm armed (step two of two).
    s.add(
        "settings_account_reset_confirm",
        settings_size,
        |window, cx| {
            let core = settings_core(cx);
            let view = cx.new(|cx| SettingsView::new(core, window, cx));
            view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
            let account = view.read(cx).account();
            account.update(cx, |a, cx| a.request_reset(cx));
            view
        },
    );

    // Wallet pane: the four lifecycle states in one honest listing.
    s.add("settings_wallet", settings_size, |window, cx| {
        let core = settings_core(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Wallet, cx));
        view
    });
}

// ---------------------------------------------------------------------------
// The Record
// ---------------------------------------------------------------------------

fn record_size() -> gpui::Size<gpui::Pixels> {
    size(px(860.), px(640.))
}

fn now_minus(mins: i64) -> i64 {
    1_781_013_753_000 - mins * 60_000 // anchored so timestamps are stable
}

fn record_attestations() -> Vec<AttestationInfo> {
    vec![
        AttestationInfo {
            hash: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
            pcr_digest: Some(
                "77aa00cc190c107d4ec428b54df0b242b4e0fc4e8f2f2a35ee98b8ddfb2dca10".into(),
            ),
            created_at: now_minus(12),
            doc_bytes: 5_882,
            connection_count: 4,
        },
        AttestationInfo {
            hash: "1f00aa45be21b268536059930c717abb7004279e860cbbb8f88be8a48d250d97".into(),
            pcr_digest: None,
            created_at: now_minus(60 * 26),
            doc_bytes: 5_874,
            connection_count: 1,
        },
    ]
}

fn record_requests() -> Vec<RequestInfo> {
    vec![
        RequestInfo {
            id: "req-1".into(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            response_status: Some(200),
            duration_ms: Some(2_741),
            request_at: now_minus(3),
            error: None,
            attempt_number: 1,
            credential_nonce: Some("a1b2c3d4e5f60718293a4b5c6d7e8f90".into()),
            transport: Some("clearnet".into()),
            base_url: Some("https://eidola.example".into()),
            attestation_hash: Some(
                "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
            ),
        },
        RequestInfo {
            id: "req-2".into(),
            method: "POST".into(),
            path: "/v1/credentials/refund".into(),
            response_status: Some(200),
            duration_ms: Some(204),
            request_at: now_minus(9),
            error: None,
            attempt_number: 2,
            credential_nonce: Some("deadbeefcafef00d0123456789abcdef".into()),
            transport: Some("clearnet".into()),
            base_url: Some("https://eidola.example".into()),
            attestation_hash: Some(
                "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
            ),
        },
        RequestInfo {
            id: "req-3".into(),
            method: "GET".into(),
            path: "/v1/models".into(),
            response_status: None,
            duration_ms: None,
            request_at: now_minus(60 * 5),
            error: Some("connection refused".into()),
            attempt_number: 1,
            credential_nonce: None,
            transport: None,
            base_url: None,
            attestation_hash: None,
        },
        RequestInfo {
            id: "req-4".into(),
            method: "GET".into(),
            path: "/v1/account/balances".into(),
            response_status: Some(401),
            duration_ms: Some(96),
            request_at: now_minus(60 * 30),
            error: None,
            attempt_number: 1,
            credential_nonce: None,
            transport: Some("clearnet".into()),
            base_url: Some("https://eidola.example".into()),
            attestation_hash: None,
        },
    ]
}

fn record_spending() -> Vec<SpendTrailEntry> {
    vec![
        SpendTrailEntry {
            credential_nonce: "a1b2c3d4e5f60718293a4b5c6d7e8f90".into(),
            spend_amount: Some(6_200),
            credential_state: "spending".into(),
            request_id: "req-1".into(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            request_at: now_minus(3),
            duration_ms: Some(2_741),
            attempt_number: 1,
            action_id: Some("act-1".into()),
            action_type: Some("inference".into()),
            model: Some("gemma4-31b".into()),
            credits_consumed: Some(6_200),
            intent: None,
            space_id: Some("space-1".into()),
            space_title: Some("Tides and the moon".into()),
            linkability: Some("unlinked".into()),
        },
        SpendTrailEntry {
            credential_nonce: "ff1122334455667788990011223344aa".into(),
            spend_amount: Some(14_600),
            credential_state: "spent".into(),
            request_id: "req-5".into(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            request_at: now_minus(60 * 24),
            duration_ms: Some(5_120),
            attempt_number: 1,
            action_id: Some("act-2".into()),
            action_type: Some("inference".into()),
            model: Some("kimi-k2-6".into()),
            credits_consumed: Some(9_400),
            intent: None,
            space_id: Some("space-2".into()),
            space_title: None,
            linkability: Some("unlinked".into()),
        },
        SpendTrailEntry {
            credential_nonce: "ff1122334455667788990011223344aa".into(),
            spend_amount: Some(14_600),
            credential_state: "spent".into(),
            request_id: "req-6".into(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            request_at: now_minus(60 * 25),
            duration_ms: Some(3_300),
            attempt_number: 1,
            action_id: Some("act-3".into()),
            action_type: Some("inference".into()),
            model: Some("kimi-k2-6".into()),
            credits_consumed: Some(5_200),
            intent: None,
            space_id: Some("space-1".into()),
            space_title: Some("Tides and the moon".into()),
            linkability: Some("unlinked".into()),
        },
    ]
}

fn register_record(s: &mut Snapshots) {
    s.add("record_attestations", record_size(), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_attestations_for_test(record_attestations(), false);
            view
        })
    });

    s.add("record_requests", record_size(), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_requests_for_test(record_requests(), true);
            view.select_section(RecordSection::Requests, cx);
            view
        })
    });

    s.add("record_spending", record_size(), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_spending_for_test(record_spending(), false);
            view.select_section(RecordSection::Spending, cx);
            view
        })
    });

    s.add("record_empty", record_size(), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_requests_for_test(Vec::new(), false);
            view.select_section(RecordSection::Requests, cx);
            view
        })
    });

    s.add("record_request_detail", record_size(), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_requests_for_test(record_requests(), false);
            view.select_section(RecordSection::Requests, cx);
            view.set_detail_for_test(Some(RecordDetail::Request(Box::new(RequestDetail {
                id: "req-1".into(),
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                request_headers: Some(
                    "content-type: application/json\nauthorization: PrivateToken token=\"…\""
                        .into(),
                ),
                request_body: Some(
                    br#"{"model":"gemma4-31b","stream":true,"messages":[{"role":"user","content":"Why is the sky blue?"}]}"#
                        .to_vec(),
                ),
                response_status: Some(200),
                response_headers: Some(
                    "content-type: text/event-stream\nx-credits-charged: 6200".into(),
                ),
                response_body: Some(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"Rayleigh\"}}]}\n\ndata: [DONE]"
                        .to_vec(),
                ),
                request_at: now_minus(3),
                response_at: Some(now_minus(3) + 2_741),
                duration_ms: Some(2_741),
                error: None,
                retry_of_id: None,
                attempt_number: 1,
                credential_nonce: Some("a1b2c3d4e5f60718293a4b5c6d7e8f90".into()),
                action_id: Some("act-1".into()),
                transport: Some("clearnet".into()),
                base_url: Some("https://eidola.example".into()),
                attestation_hash: Some(
                    "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
                ),
            }))));
            view
        })
    });

    s.add("record_attestation_detail", record_size(), |window, cx| {
        let core = stub_core_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_attestations_for_test(record_attestations(), false);
            view.set_detail_for_test(Some(RecordDetail::Attestation(AttestationDetail {
                hash: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
                pcr_digest: Some(
                    "77aa00cc190c107d4ec428b54df0b242b4e0fc4e8f2f2a35ee98b8ddfb2dca10".into(),
                ),
                created_at: now_minus(12),
                doc: br#"{"format":"https://tinfoil.sh/predicate/sev-snp-guest/v1","body":"pZWA2x0aGUgcmVwb3J0IGJvZHkgaXMgYSBsb25nIGJhc2U2NCBibG9i","tls_public_key_fp":"8c41af","nonce":"f00d"}"#
                    .to_vec(),
            })));
            view
        })
    });
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

/// A stub core with a model catalog, for the model-picker cases. Rates are
/// representative of the real catalog (credits are micro-USD-denominated,
/// so credits/token reads as $/M tokens).
fn model_core(cx: &mut App) -> Entity<Core> {
    cx.new(|_| {
        let mut c = Core::stub();
        c.config_state = Some(stub_config_state(true));
        c.models = vec![
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
            ModelInfo {
                id: "whisper-large-v3".into(),
                context_length: 0,
                prompt_credits_per_token: 0.0,
                completion_credits_per_token: 0.0,
                request_credits: Some(9_000.0),
            },
        ];
        c
    })
}

fn stub_config_state(has_account: bool) -> ConfigState {
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
