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
    CredentialLifecycleInfo, MeasurementInfo, ModelInfo, PriceInfo, RequestDetail, RequestInfo,
    SpaceInfo, SpendTrailEntry,
};
use eidola_gui::about::AboutView;
use eidola_gui::backends_settings::BackendsTab;
use eidola_gui::library::LibraryView;
use eidola_gui::onboarding::OnboardingView;
use eidola_gui::record::{RecordDetail, RecordSection, RecordView};
use eidola_gui::settings::{SettingsPane, SettingsView};
use eidola_gui::space_view::SpaceView;
use eidola_gui::stores::{Stores, StoresStub};
use eidola_gui::updates::UpdatesView;
use eidola_gui::window_input::WindowInput;
use gpui::{App, AppContext, px, size};

use super::fixtures::{fixture_post, kitchen_sink_posts};
use super::harness::Snapshots;

pub fn register(s: &mut Snapshots) {
    register_space(s);
    register_onboarding_window(s);
    register_library(s);
    register_settings(s);
    register_updates(s);
    register_record(s);
    register_about(s);
}

// ---------------------------------------------------------------------------
// Space view — the tree-navigation conversation surface (wave-6)
// ---------------------------------------------------------------------------

fn register_space(s: &mut Snapshots) {
    // An *existing* space (Some id → no composer at rest): the kitchen-sink post
    // tree rendered through the new SpaceView — byline gutter, Newsreader prose
    // posts (read-only `MarkdownEditor`, published — no delimiters), and
    // separator bands with the "+" reply affordance. You click "+" to start a
    // draft.
    s.add("space_branches", size(px(900.), px(720.)), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let view = SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
            view.space().update(cx, |sp, cx| {
                sp.set_post_tree_for_test(kitchen_sink_posts(), cx)
            });
            view
        })
    });

    // The populated branch settled all the way onto its active tail composer.
    // Unlike the blank notebook below, this exercises the composer's full
    // docked runway rather than its titlebar-adjusted standalone slot.
    s.add(
        "space_docked_composer",
        size(px(900.), px(720.)),
        |window, cx| {
            let core = stub_stores_with_config(cx);
            cx.new(|cx| {
                let mut view =
                    SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
                view.space().update(cx, |sp, cx| {
                    sp.set_post_tree_for_test(kitchen_sink_posts(), cx)
                });
                view.seed_draft_quote_for_test(
                    Some("a9"),
                    "I want to preserve that distinction in the final paragraph.",
                    vec![],
                    window,
                    cx,
                );
                view.set_page_scroll_for_test(-100_000.0);
                view
            })
        },
    );

    // The space inspector (tasks 26.2 + 26.3): the per-space settings panel
    // splitting the window — chrome type beside the paper, with a remote router
    // selected so the mandatory per-call cost copy is in frame, and the
    // Participants section resting below it.
    s.add(
        "space_inspector",
        size(px(1040.), px(760.)),
        |window, cx| {
            let stores = stub_stores(cx, |s| {
                s.config_state = Some(stub_config_state(true));
                s.participants = Some(inspector_participants());
                s.spaces = vec![SpaceInfo {
                    id: "demo".into(),
                    title: Some("Tides and the moon".into()),
                    snippet: None,
                    created_at: 0,
                    last_activity_at: 0,
                    message_count: 4,
                    archived_at: None,
                }];
                s.space_settings = Some((
                    "demo".into(),
                    eidola_app_core::SpaceSettings {
                        cascade_limit: 4,
                        router_model: Some("gemma4-31b@eidola".into()),
                        ..Default::default()
                    },
                ));
            });
            cx.new(|cx| {
                let mut view = SpaceView::new(
                    stores,
                    Some("demo".into()),
                    WindowInput::new(cx),
                    window,
                    cx,
                );
                view.space().update(cx, |sp, cx| {
                    sp.set_post_tree_for_test(kitchen_sink_posts(), cx)
                });
                view.set_inspector_open_for_test(true, window, cx);
                view
            })
        },
    );

    // The Participants section with a member's disclosure open — the editor the
    // standalone window used to hold, now inside the panel.
    s.add(
        "space_inspector_participants",
        size(px(1040.), px(760.)),
        |window, cx| {
            let stores = stub_stores(cx, |s| {
                s.config_state = Some(stub_config_state(true));
                s.participants = Some(inspector_participants());
                s.space_settings = Some((
                    "demo".into(),
                    eidola_app_core::SpaceSettings {
                        cascade_limit: 4,
                        router_model: None,
                        ..Default::default()
                    },
                ));
            });
            cx.new(|cx| {
                let mut view = SpaceView::new(
                    stores,
                    Some("demo".into()),
                    WindowInput::new(cx),
                    window,
                    cx,
                );
                view.space().update(cx, |sp, cx| {
                    sp.set_post_tree_for_test(kitchen_sink_posts(), cx)
                });
                view.set_inspector_open_for_test(true, window, cx);
                view.inspector_toggle_participant("agent-b", window, cx);
                view
            })
        },
    );

    // Quoted references: a source post whose quoted passages carry the warm
    // highlight wash (one plain, one overlapping pair), the replies whose
    // bodies render the quote as an embed block, and the footnote rails
    // beneath them.
    s.add("space_quotes", size(px(860.), px(760.)), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let view = SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
            let (posts, incoming) = super::fixtures::quoted_reference_posts();
            view.space().update(cx, |sp, cx| {
                sp.set_post_tree_for_test(posts, cx);
                for (action_id, refs) in incoming {
                    sp.seed_incoming_references_for_test(
                        action_id,
                        refs.iter()
                            .map(|r| eidola_app_core::IncomingReference {
                                action_id: r.action_id.clone(),
                                space_id: "demo".into(),
                                ordinal: 1,
                                content_block_id: Some(r.block_id.clone()),
                                range_start: Some(r.range.0),
                                range_end: Some(r.range.1),
                                annotation: None,
                                created_at: 0,
                            })
                            .collect(),
                    );
                }
            });
            view
        })
    });

    // A persisted post quoting across conversations: the footnote rail naming
    // an author this window holds, one only the edge can name, and one nobody
    // can — three rows, three sources.
    s.add(
        "space_cross_space_quote",
        size(px(860.), px(760.)),
        |window, cx| {
            let core = stub_stores_with_config(cx);
            cx.new(|cx| {
                let view =
                    SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
                view.space().update(cx, |sp, cx| {
                    sp.set_post_tree_for_test(super::fixtures::cross_space_reference_posts(), cx)
                });
                view
            })
        },
    );

    // Trace visibility: an answered turn's tool rounds expanded under its own
    // reply, and three declines stacked in the gap under the post they
    // answered — one quiet line per turn, each naming its own agent.
    s.add("space_traces", size(px(860.), px(760.)), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let view = SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
            let (posts, traces) = super::fixtures::trace_posts();
            view.space().update(cx, |sp, cx| {
                sp.set_post_tree_for_test(posts, cx);
                sp.seed_traces_for_test(traces);
                sp.toggle_trace("turn-gemma", cx);
                sp.toggle_trace("turn-mara-2", cx);
            });
            view
        })
    });

    // Composing with a quote: the pending reference rendered as an embed block
    // inside the active draft, with its footnote (and remove affordance)
    // below.
    s.add(
        "space_quote_draft",
        size(px(860.), px(760.)),
        |window, cx| {
            let core = stub_stores_with_config(cx);
            cx.new(|cx| {
                let mut view =
                    SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
                let (posts, _) = super::fixtures::quoted_reference_posts();
                view.space().update(cx, |sp, cx| {
                    sp.set_post_tree_for_test(posts.into_iter().take(1).collect(), cx)
                });
                let quoted = super::fixtures::quoted_reference_selection();
                let source = super::fixtures::quoted_reference_source();
                view.seed_draft_quote_for_test(
                    Some("q1"),
                    "That's the sentence I keep snagging on:\n\n{{ embed 1 }}\n\nIf the care is \
                 real, doesn't the shepherd analogy quietly concede Socrates' point?",
                    vec![(1, "kimi-k2", &source[quoted])],
                    window,
                    cx,
                );
                view
            })
        },
    );

    // A brand-new blank space: the composer open at the top of an empty page
    // (the cursor in a fresh notebook).
    s.add("space_blank", size(px(760.), px(680.)), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| SpaceView::new(core, None, WindowInput::new(cx), window, cx))
    });

    // A failed ask: the dismissible recovery notice attached to the bottom of
    // the exchange (Retry / Copy / ×), with the saved user post above it.
    s.add("space_error", size(px(760.), px(680.)), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let view = SpaceView::new(core, Some("s".into()), WindowInput::new(cx), window, cx);
            view.space().update(cx, |sp, cx| {
                sp.set_post_tree_for_test(
                    vec![fixture_post(
                        "a1",
                        "human",
                        "user",
                        "user_input",
                        "Hello, what is your name?",
                        0,
                        false,
                        1,
                    )],
                    cx,
                );
                sp.apply_chat_failure_for_test(
                    eidola_app_core::error::AppError::ChatFailed {
                        space_id: "s".into(),
                        source: Box::new(eidola_app_core::error::AppError::Network {
                            message: "request failed: error sending request for url \
                                 (https://gateway.eidola.containers.tinfoil.sh/v1/models): \
                                 client error (Connect): dns error: failed to look up address \
                                 information: nodename nor servname provided, or not known"
                                .into(),
                        }),
                    },
                    cx,
                );
            });
            view
        })
    });

    // A draft with content: the action gutter reveals the composer's one
    // CTA — **Post** (the model picker + request panel are gone; who answers
    // is Participants configuration).
    s.add(
        "space_composer_actions",
        size(px(900.), px(680.)),
        |window, cx| {
            let core = stub_stores_with_config(cx);
            cx.new(|cx| {
                let view = SpaceView::new(core, None, WindowInput::new(cx), window, cx);
                if let Some(editor) = view.composer_state_for_test() {
                    editor.update(cx, |e, cx| {
                        e.set_value(
                            "What did Thrasymachus actually claim about justice?".to_string(),
                            cx,
                        )
                    });
                }
                view
            })
        },
    );

    // ⌥ held: the quiet verb (Post quietly — notify no one) and the keyboard
    // hints join the action gutter — the "Option reveals power" expansion.
    s.add(
        "space_composer_alt",
        size(px(900.), px(680.)),
        |window, cx| {
            let core = stub_stores_with_config(cx);
            let wi = WindowInput::new(cx);
            wi.update(cx, |w, cx| w.set_alt_for_test(true, cx));
            cx.new(|cx| {
                let view = SpaceView::new(core, None, wi, window, cx);
                if let Some(editor) = view.composer_state_for_test() {
                    editor.update(cx, |e, cx| {
                        e.set_value(
                            "What did Thrasymachus actually claim about justice?".to_string(),
                            cx,
                        )
                    });
                }
                view
            })
        },
    );

    // A separator band's Reply-or-Ask menu open: Reply (this post has a
    // committed reply) plus one quiet "Ask <agent>" chip per agent
    // participant.
    s.add("space_band_menu", size(px(900.), px(680.)), |window, cx| {
        let core = participant_stores(cx, "demo");
        cx.new(|cx| {
            let mut view =
                SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
            view.space().update(cx, |sp, cx| {
                let q = fixture_post(
                    "a1",
                    "human",
                    "user",
                    "user_input",
                    "What did Thrasymachus actually claim about justice?",
                    0,
                    false,
                    1,
                );
                let mut a = fixture_post(
                    "a2",
                    "agent",
                    "Ida",
                    "inference",
                    "His claim in Republic I is narrower than its reputation.",
                    0,
                    false,
                    1,
                );
                a.parent_action_id = Some("a1".into());
                sp.set_post_tree_for_test(vec![q, a], cx);
            });
            view.set_band_menu_for_test(Some("a1"), cx);
            view
        })
    });

    // Two participants answering the same post at once — a Post's notification
    // fan-out: each in-flight turn streams as its own timestamp-ordered
    // sibling branch, bylined with its participant.
    s.add(
        "space_concurrent_streams",
        size(px(900.), px(720.)),
        |window, cx| {
            let core = participant_stores(cx, "demo");
            cx.new(|cx| {
                let view =
                    SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
                view.space().update(cx, |sp, cx| {
                    sp.set_post_tree_for_test(
                        vec![fixture_post(
                            "a1",
                            "human",
                            "user",
                            "user_input",
                            "What did Thrasymachus actually claim about justice?",
                            0,
                            false,
                            1,
                        )],
                        cx,
                    );
                    let seq_b = sp.push_streaming_turn_for_test(
                        Some("agent-b".into()),
                        Some("a1".into()),
                        Default::default(),
                        cx,
                    );
                    sp.push_content_delta_for_test(
                        seq_b,
                        "The definition he offers at 338c — justice is the advantage of the \
                         stronger — is less a theory than a provocation…",
                        cx,
                    );
                    let seq_c = sp.push_streaming_turn_for_test(
                        Some("agent-c".into()),
                        Some("a1".into()),
                        Default::default(),
                        cx,
                    );
                    sp.push_content_delta_for_test(
                        seq_c,
                        "Start from the shepherd argument at 343b instead:",
                        cx,
                    );
                });
                view
            })
        },
    );

    // The cascade-paused notice: the conversation reached its cascade limit;
    // quiet and dismissible, with an explicit "Ask <agent>" per agent as the
    // way onward. Muted, not danger — nothing failed; the guard did its job.
    s.add(
        "space_cascade_paused",
        size(px(900.), px(680.)),
        |window, cx| {
            let core = participant_stores(cx, "demo");
            cx.new(|cx| {
                let view =
                    SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
                view.space().update(cx, |sp, cx| {
                    let q = fixture_post(
                        "a1",
                        "human",
                        "user",
                        "user_input",
                        "What did Thrasymachus actually claim about justice?",
                        0,
                        false,
                        1,
                    );
                    let mut a = fixture_post(
                        "a2",
                        "agent",
                        "Ida",
                        "inference",
                        "His claim in Republic I is narrower than its reputation.",
                        0,
                        false,
                        1,
                    );
                    a.parent_action_id = Some("a1".into());
                    sp.set_post_tree_for_test(vec![q, a], cx);
                    sp.emit_cascade_paused_for_test(4, 4, "a2".into(), cx);
                });
                view
            })
        },
    );

    // Per-post affordances: hovering the assistant reply reveals Regenerate in
    // its action gutter, and the finalized post keeps its reasoning disclosure
    // ("Thinking…") above the body.
    s.add("space_post_actions", size(px(900.), px(680.)), |window, cx| {
        let core = model_stores(cx);
        cx.new(|cx| {
            let mut view =
                SpaceView::new(core, Some("demo".into()), WindowInput::new(cx), window, cx);
            view.space().update(cx, |sp, cx| {
                let q = fixture_post(
                    "a1",
                    "human",
                    "user",
                    "user_input",
                    "What did Thrasymachus actually claim about justice?",
                    0,
                    false,
                    1,
                );
                let mut a = fixture_post(
                    "a2",
                    "agent",
                    "kimi-k2-6",
                    "inference",
                    "Three threads worth pulling apart: the position, the shepherd, and what the exchange is really about.",
                    0,
                    false,
                    1,
                );
                a.parent_action_id = Some("a1".into());
                sp.set_post_tree_for_test(vec![q, a], cx);
                sp.set_reasoning_for_test(
                    1,
                    "The user is asking about Republic I. Anchor on 338c and 343b.".into(),
                    false,
                    cx,
                );
            });
            view.set_post_hover_for_test("a2", true, cx);
            view
        })
    });
}

// ---------------------------------------------------------------------------
// About window
// ---------------------------------------------------------------------------

fn register_about(s: &mut Snapshots) {
    s.add("about", size(px(360.), px(420.)), |window, cx| {
        cx.new(|cx| AboutView::new(window, cx))
    });
}

// ---------------------------------------------------------------------------
// Updates window — one case per display state, at the window's real size
// ---------------------------------------------------------------------------

fn register_updates(s: &mut Snapshots) {
    fn updates_stores(cx: &mut App, setup: impl FnOnce(&mut StoresStub)) -> Stores {
        stub_stores(cx, setup)
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
        let core = updates_stores(cx, |c| c.update_checking = true);
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_up_to_date", sz, move |window, cx| {
        let core = updates_stores(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::UpToDate {
                latest_version: Some("0.1.0".into()),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_check_failed", sz, move |window, cx| {
        let core = updates_stores(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::CheckFailed {
                message: "GET https://api.github.com/...: connection timed out".into(),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_available", sz, move |window, cx| {
        let core = updates_stores(cx, |c| {
            c.update_check = Some(snapshot(UpdateCheckResult::UpdateAvailable {
                release: release(false),
            }));
        });
        cx.new(|cx| UpdatesView::new(core, window, cx))
    });

    s.add("updates_unverifiable", sz, move |window, cx| {
        let core = updates_stores(cx, |c| {
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
            let core = updates_stores(cx, |c| {
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

// The standalone onboarding window (wave-6): full-window slides. The first
// slide (present at rest) shows the prose heading + intro vertically centered in
// the reading column with the ghost-button CTA at the bottom.
fn register_onboarding_window(s: &mut Snapshots) {
    s.add(
        "onboarding_window",
        size(px(760.), px(760.)),
        |window, cx| {
            let stores = stub_stores(cx, |s| {
                s.config_state = Some(stub_config_state(false));
                s.prices = stub_prices();
            });
            cx.new(|cx| OnboardingView::new(stores, window, cx))
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

fn library_stores(cx: &mut App) -> Stores {
    stub_stores(cx, |s| {
        s.spaces = vec![
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
    })
}

fn register_library(s: &mut Snapshots) {
    s.add("library_empty", size(px(520.), px(620.)), |window, cx| {
        let stores = stub_stores(cx, |_| {});
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    s.add(
        "library_with_spaces",
        size(px(520.), px(620.)),
        |window, cx| {
            let core = library_stores(cx);
            cx.new(|cx| LibraryView::new(core, window, cx))
        },
    );

    // Hover state: the archive × is revealed on the hovered row.
    s.add("library_hovered", size(px(520.), px(620.)), |window, cx| {
        let core = library_stores(cx);
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
fn settings_stores(cx: &mut App) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(stub_config_state(true));
        s.eidola_trust = Some(stub_eidola_trust());
        s.balances = Some(BalancesResult {
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
        s.prices = vec![
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
        s.credential_lifecycle = vec![
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
    })
}

/// `settings_stores` plus a populated backend registry and local-model
/// state, so the Backends pane's three tabs render with content.
fn settings_backends_stores(cx: &mut App) -> Stores {
    use eidola_app_core::{
        BackendInfo, BackendKind, ExternalEngineBackend, LocalModelInfo, LocalModelStatus,
        LocalModelsState,
    };
    stub_stores(cx, |s| {
        s.config_state = Some(stub_config_state(true));
        s.eidola_trust = Some(stub_eidola_trust());
        s.balances = Some(BalancesResult {
            available: 4_200_000,
            pools: vec![BalancePoolInfo {
                amount: 4_200_000,
                source: "subscription".into(),
                expires_at: Some(eidola_app_core::now_ms() + 23 * 24 * 60 * 60 * 1000),
            }],
        });
        s.prices = vec![
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
        s.backends = vec![
            BackendInfo {
                id: "eidola".into(),
                kind: BackendKind::Eidola,
                display_name: "Eidola".into(),
                enabled: true,
                base_url: None,
                has_api_key: false,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
                created_at: 0,
            },
            BackendInfo {
                id: "local".into(),
                kind: BackendKind::Local,
                display_name: "Local".into(),
                enabled: true,
                base_url: None,
                has_api_key: false,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
                created_at: 0,
            },
            BackendInfo {
                id: "my-box".into(),
                kind: BackendKind::LlamaCpp,
                display_name: "My box".into(),
                enabled: true,
                base_url: None,
                has_api_key: false,
                models_dir: Some("/Users/me/models".into()),
                model_overrides: None,
                engine_path: None,
                auto_start: false,
                created_at: 1,
            },
        ];
        s.local_models = Some(LocalModelsState {
            engine_path: Some("/Applications/Eidola.app/Contents/MacOS/llama-server".into()),
            external: vec![ExternalEngineBackend {
                backend_id: "my-box".into(),
                display_name: "My box".into(),
                enabled: true,
                models_dir: "/Users/me/models".into(),
                engine_path: Some("/opt/homebrew/bin/llama-server".into()),
                auto_start: false,
                models: vec![LocalModelInfo {
                    id: "qwen3-8b@my-box".into(),
                    slug: "qwen3-8b".into(),
                    display_name: "Qwen3 8B".into(),
                    file_name: "qwen3-8b.gguf".into(),
                    size_bytes: Some(5_200_000_000),
                    source_url: None,
                    status: LocalModelStatus::Available,
                    last_error: None,
                    on_disk: true,
                }],
            }],
            models: vec![
                LocalModelInfo {
                    id: "gemma-4-E2B_q4_0-it@local".into(),
                    slug: "gemma-4-E2B_q4_0-it".into(),
                    display_name: "Gemma 4 E2B".into(),
                    file_name: "gemma-4-E2B_q4_0-it.gguf".into(),
                    size_bytes: Some(3_349_514_112),
                    source_url: None,
                    status: LocalModelStatus::Available,
                    last_error: None,
                    on_disk: true,
                },
                LocalModelInfo {
                    id: "gemma-4-E4B_q4_0-it@local".into(),
                    slug: "gemma-4-E4B_q4_0-it".into(),
                    display_name: "Gemma 4 E4B".into(),
                    file_name: "gemma-4-E4B_q4_0-it.gguf".into(),
                    size_bytes: Some(5_000_000_000),
                    source_url: None,
                    status: LocalModelStatus::Loaded {
                        port: 4242,
                        context_tokens: 8192,
                        pinned: true,
                    },
                    last_error: None,
                    on_disk: true,
                },
            ],
        });
    })
}

fn register_settings(s: &mut Snapshots) {
    let settings_size = size(px(620.), px(520.));

    // General: the appearance chips — the whole pane (the trust/connection
    // surface lives in Backends → Eidola).
    s.add("settings_general", settings_size, |window, cx| {
        let core = settings_stores(cx);
        cx.new(|cx| SettingsView::new(core, window, cx))
    });

    // Backends → Eidola: the singleton toggle plus the connection + trust
    // surface (base-URL pin/override, trusted-measurements state, hardware CA
    // lines).
    s.add("settings_backends_eidola", settings_size, |window, cx| {
        let core = settings_backends_stores(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        let pane = view.read(cx).backends_pane();
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Eidola, cx));
        view
    });

    // Backends → Eidola with the whole trust bundle overridden: the danger
    // warning band up top, danger-tinted override status lines, the compact
    // measurement lines (override + dropped-pin) with their Copy/Untrust/
    // Trust verbs, and the overridden root CA's Copy/Replace…/Clear verbs —
    // everything visible at rest, no disclosure.
    s.add(
        "settings_backends_eidola_overridden",
        settings_size,
        |window, cx| {
            use eidola_app_core::{BackendInfo, BackendKind};
            let core = stub_stores(cx, |s| {
                s.config_state = Some(stub_config_state(true));
                let mut trust = stub_eidola_trust();
                trust.base_url = "https://staging.eidola.example/v1".into();
                trust.base_url_is_override = true;
                trust.trusted_measurements = vec![MeasurementInfo {
                    snp: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
                    tdx_rtmr1: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                        .into(),
                    tdx_rtmr2: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
                        .into(),
                }];
                trust.trusted_measurements_are_override = true;
                trust.has_hardware_root_ca = true;
                trust.hardware_root_ca_pem = Some(
                    "-----BEGIN CERTIFICATE-----\n\
                     MIIByDCCAW6gAwIBAgIUExampleCustomRootCAForSnapshotsOnly0\n\
                     -----END CERTIFICATE-----"
                        .into(),
                );
                s.eidola_trust = Some(trust);
                s.backends = vec![
                    BackendInfo {
                        id: "eidola".into(),
                        kind: BackendKind::Eidola,
                        display_name: "Eidola".into(),
                        enabled: true,
                        base_url: None,
                        has_api_key: false,
                        models_dir: None,
                        model_overrides: None,
                        engine_path: None,
                        auto_start: true,
                        created_at: 0,
                    },
                    BackendInfo {
                        id: "local".into(),
                        kind: BackendKind::Local,
                        display_name: "Local".into(),
                        enabled: true,
                        base_url: None,
                        has_api_key: false,
                        models_dir: None,
                        model_overrides: None,
                        engine_path: None,
                        auto_start: true,
                        created_at: 0,
                    },
                ];
            });
            let view = cx.new(|cx| SettingsView::new(core, window, cx));
            view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
            let pane = view.read(cx).backends_pane();
            pane.update(cx, |p, cx| p.select_tab(BackendsTab::Eidola, cx));
            view
        },
    );

    // Backends → Eidola with the in-place editors revealed: the add-a-
    // measurement input and the root-CA paste-PEM textarea open at once
    // (each revealed by its own quiet verb; Cancel closes them).
    s.add(
        "settings_backends_eidola_editing",
        settings_size,
        |window, cx| {
            use eidola_gui::backends_settings::CaKind;
            let core = settings_backends_stores(cx);
            let view = cx.new(|cx| SettingsView::new(core, window, cx));
            view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
            let pane = view.read(cx).backends_pane();
            pane.update(cx, |p, cx| {
                p.select_tab(BackendsTab::Eidola, cx);
                p.begin_add_measurement(window, cx);
                p.begin_edit_ca(CaKind::Root, window, cx);
            });
            view
        },
    );

    // Backends → Local: the managed store — engine line, installed models,
    // catalog, paste-a-URL.
    s.add("settings_backends_local", settings_size, |window, cx| {
        let core = settings_backends_stores(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        let pane = view.read(cx).backends_pane();
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
        view
    });

    // Backends → External: a llamacpp backend (engine line + auto-start +
    // scanned models) and the add-a-backend affordances.
    s.add("settings_backends_external", settings_size, |window, cx| {
        let core = settings_backends_stores(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        let pane = view.read(cx).backends_pane();
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::External, cx));
        view
    });

    // Account pane (top-level again): balance + pools + plans, shown while
    // the eidola backend is enabled.
    s.add("settings_account", settings_size, |window, cx| {
        let core = settings_stores(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
        view
    });

    // Wallet pane: the four lifecycle states in one honest listing.
    s.add("settings_wallet", settings_size, |window, cx| {
        let core = settings_stores(cx);
        let view = cx.new(|cx| SettingsView::new(core, window, cx));
        view.update(cx, |v, cx| v.select(SettingsPane::Wallet, cx));
        view
    });

    // Nav gating: with the eidola backend disabled, only General and Backends
    // show (Account/Wallet are hidden — "on-device only").
    s.add("settings_nav_no_eidola", settings_size, |window, cx| {
        use eidola_app_core::{BackendInfo, BackendKind};
        let core = stub_stores(cx, |s| {
            s.config_state = Some(stub_config_state(true));
            s.eidola_trust = Some(stub_eidola_trust());
            s.backends = vec![
                BackendInfo {
                    id: "eidola".into(),
                    kind: BackendKind::Eidola,
                    display_name: "Eidola".into(),
                    enabled: false,
                    base_url: None,
                    has_api_key: false,
                    models_dir: None,
                    model_overrides: None,
                    engine_path: None,
                    auto_start: true,
                    created_at: 0,
                },
                BackendInfo {
                    id: "local".into(),
                    kind: BackendKind::Local,
                    display_name: "Local".into(),
                    enabled: true,
                    base_url: None,
                    has_api_key: false,
                    models_dir: None,
                    model_overrides: None,
                    engine_path: None,
                    auto_start: true,
                    created_at: 0,
                },
            ];
        });
        cx.new(|cx| SettingsView::new(core, window, cx))
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
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_attestations_for_test(record_attestations(), false);
            view
        })
    });

    s.add("record_requests", record_size(), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_requests_for_test(record_requests(), true);
            view.select_section(RecordSection::Requests, cx);
            view
        })
    });

    s.add("record_spending", record_size(), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_spending_for_test(record_spending(), false);
            view.select_section(RecordSection::Spending, cx);
            view
        })
    });

    s.add("record_empty", record_size(), |window, cx| {
        let core = stub_stores_with_config(cx);
        cx.new(|cx| {
            let mut view = RecordView::new(core, window, cx);
            view.set_requests_for_test(Vec::new(), false);
            view.select_section(RecordSection::Requests, cx);
            view
        })
    });

    s.add("record_request_detail", record_size(), |window, cx| {
        let core = stub_stores_with_config(cx);
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
                space_id: None,
                space_title: None,
                backend_id: Some("eidola".into()),
                backend_display_name: Some("Eidola".into()),
            }))));
            view
        })
    });

    s.add("record_attestation_detail", record_size(), |window, cx| {
        let core = stub_stores_with_config(cx);
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

/// Build stub stores from a declaratively-described scene — the visual-case
/// equivalent of the old `Core::stub()` field-poking.
fn stub_stores(cx: &mut App, setup: impl FnOnce(&mut StoresStub)) -> Stores {
    let mut fixture = StoresStub::default();
    setup(&mut fixture);
    Stores::stub_with(fixture, cx)
}

fn stub_stores_with_config(cx: &mut App) -> Stores {
    stub_stores(cx, |s| s.config_state = Some(stub_config_state(true)))
}

/// A stub space-owned agent participant (the separator Ask menus, streaming
/// bylines, and the cascade notice read the space's agent set).
fn agent_participant(id: &str, label: &str) -> eidola_app_core::ParticipantInfo {
    eidola_app_core::ParticipantInfo {
        id: id.into(),
        scope: "space".into(),
        source: "owned".into(),
        kind: "agent".into(),
        label: label.into(),
        model_ref: Some("kimi-k2-6".into()),
        system_prompt: None,
        notify_policy: "human".into(),
        role: "member".into(),
        reference: None,
    }
}

/// The membership the inspector's Participants section renders: the shared
/// human (so the "shared" tag and the fork are in frame) plus two owned agents.
fn inspector_participants() -> (String, Vec<eidola_app_core::ParticipantInfo>) {
    let you = eidola_app_core::ParticipantInfo {
        id: eidola_app_core::HUMAN_PARTICIPANT_ID.into(),
        scope: "global".into(),
        source: "referenced".into(),
        kind: "human".into(),
        // The **wire** label (task 64); the row shows "You".
        label: "User".into(),
        model_ref: None,
        system_prompt: None,
        notify_policy: "explicit".into(),
        role: "member".into(),
        reference: None,
    };
    let mut ida = agent_participant("agent-b", "Ida");
    ida.system_prompt = Some("Keep the thread honest; challenge weak claims.".into());
    (
        "demo".into(),
        vec![you, ida, agent_participant("agent-c", "Sage")],
    )
}

/// Stub stores with two agent participants seeded for `space_id`.
fn participant_stores(cx: &mut App, space_id: &str) -> Stores {
    let sid = space_id.to_string();
    stub_stores(cx, move |s| {
        s.config_state = Some(stub_config_state(true));
        s.participants = Some((
            sid,
            vec![
                agent_participant("agent-b", "Ida"),
                agent_participant("agent-c", "Sage"),
            ],
        ));
    })
}

/// Stub stores with a model catalog, for the model-picker cases. Rates are
/// representative of the real catalog (credits are micro-USD-denominated,
/// so credits/token reads as $/M tokens).
fn model_stores(cx: &mut App) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(stub_config_state(true));
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
            ModelInfo {
                id: "whisper-large-v3".into(),
                context_length: 0,
                prompt_credits_per_token: 0.0,
                completion_credits_per_token: 0.0,
                request_credits: Some(9_000.0),
            },
        ];
    })
}

fn stub_config_state(has_account: bool) -> ConfigState {
    ConfigState {
        default_template: "00000000-0000-7000-8000-000000000010".into(),
        has_account,
        has_account_secret: has_account,
        account_id: has_account.then(|| "00000000-0000-7000-8000-000000000111".into()),
        account_secret: has_account.then(|| "visual-account-secret".into()),
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        appearance: eidola_app_core::config::AppearanceSetting::System,
        time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
        light_character: eidola_app_core::config::LightCharacter::Neutral,
        font_scale: 1.0,
        language: None,
    }
}

fn stub_eidola_trust() -> eidola_app_core::EidolaTrust {
    eidola_app_core::EidolaTrust {
        base_url: "https://eidola.example/v1".into(),
        base_url_pin: "https://eidola.example/v1".into(),
        base_url_is_override: false,
        trusted_measurements: Vec::new(),
        trusted_measurements_are_override: false,
        pinned_measurement: eidola_app_core::MeasurementInfo {
            snp: "1122334455667788112233445566778811223344556677881122334455667788".into(),
            tdx_rtmr1: "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899".into(),
            tdx_rtmr2: "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa".into(),
        },
        has_hardware_root_ca: false,
        hardware_root_ca_pem: None,
        has_hardware_intermediate_ca: false,
        hardware_intermediate_ca_pem: None,
    }
}
